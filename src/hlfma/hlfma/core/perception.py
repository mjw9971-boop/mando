"""
RawPacket + LaneGraph + route → WorldState  (SPEC §3.3)

Perception 은 **판단하지 않는다.** "무엇이 어디에 있다"까지만 만든다.
속도 제한 해석·양보 판단 등은 전부 Planner 의 일이다.
"""
from __future__ import annotations

import math

from .comm import OBJ_COUNT as OBJECT_SLOTS
from .lanegraph import LaneGraph
from .types import EgoState, LaneKey, RawPacket, TrackedObject, WorldState


class Perception:
    """
    현재는 **최소 주행 루프**용으로 1~4 단계만 구현돼 있다.
    객체는 분류만 하고 TTC/will_enter_lane 은 아직 TODO (SPEC §3.3 의 6~7 단계).
    """

    def __init__(self, lg: LaneGraph, route: dict | None, cfg: dict) -> None:
        self.lg = lg
        self.route = route
        self.cfg = cfg

        self._prev_xy: tuple[float, float] | None = None
        self._prev_t: float | None = None
        self._speed = 0.0
        self._accel = 0.0
        self._route_idx = 0          # 경로 진행 인덱스 (단조 증가 힌트)
        self._prev_route_s: float | None = None
        self.reset_count = 0         # VTD 리셋(리스폰) 누적 = 실제 이탈 횟수
        self.stall_count = 0         # 파이프라인 멈춤 누적 (리셋과 섞지 않는다)
        self.reset_log: list = []    # (t, dt, 점프거리, 사유)
        self.stall_log: list = []

        # 경로 차로 -> 경로 인덱스 (객체의 경로상 위치 계산용)
        self._lane_idx: dict = {}
        if route:
            for i, k in enumerate(route['lanes']):
                self._lane_idx.setdefault(k, i)
        # 차선변경 목표 차로. 변경 중에는 여기 붙어 있어야 매칭이 안 흔들린다.
        # planner 상태를 되돌려받을 수 없으므로(단방향 파이프라인) 경로 이벤트에서
        # 미리 뽑아 prefer 에 상시 포함한다.
        self._lc_lanes: list = []
        if route:
            for e in route.get('events', []):
                if e['kind'].startswith('lane_change') and e.get('to_lane'):
                    self._lc_lanes.append(tuple(e['to_lane']))
        self._prefer: list | None = None
        if route:
            self._prefer = list(route['lanes']) + [
                k for k in self._lc_lanes if k not in set(route['lanes'])]
        self._carry_limit: float | None = None
        self._carry_school = False
        # 객체 id -> (TrackedObject, 마지막 수신 시각, 마지막 거리)
        self._tracks: dict[int, tuple] = {}

    def update(self, pkt: RawPacket) -> WorldState:
        """한 패킷을 WorldState 로. 순서는 SPEC §3.3 의 1~8 단계."""
        x, y, z, yaw, pitch, roll = pkt.ego[0], pkt.ego[1], pkt.ego[2], pkt.ego[3], pkt.ego[4], pkt.ego[5]
        flags: dict = {}

        # 0) VTD 리셋(courseRespawn) 감지 — 속도 추정보다 먼저 봐야 한다
        self._detect_reset(x, y, pkt.t_recv, flags)

        # 1) 속도/가속도 추정 (패킷에 속도 필드가 없다)
        speed, accel = self._estimate_motion(x, y, yaw, pkt.t_recv, flags)

        # 2) 차로 매칭
        prefer = self._prefer
        m = self.lg.locate(x, y, yaw, prefer=prefer)

        valid = m is not None
        if not valid:
            flags['locate_failed'] = True
            ego = EgoState(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll,
                           speed=speed, accel=accel, lane=None, s=0.0,
                           route_s=0.0, t_off=0.0, heading_err=0.0)
            return WorldState(t=pkt.t_recv, ego=ego, objects=[], light=None,
                              ahead=[], summ={}, speed_limit=self._default_limit(),
                              school_zone=False, left_solid=False, right_solid=False,
                              left_is_center=False, valid=False, flags=flags)

        idx = self._route_index(m.lane, flags)
        route_s = (float(self.route['cum_s'][idx]) + m.s) if (self.route and idx is not None) else m.s

        # route_s 가 크게 뒤로 가는 것도 리셋 신호다 (위치 점프가 작아도 잡힌다).
        # 단 **경로 위에 있을 때만** 의미가 있다. 경로를 벗어났거나 완주한 뒤에는
        # route_s 가 다른 구간에 다시 붙으면서 크게 요동쳐 오탐이 된다
        # (실측: 완주 직후 10건이 전부 이 경우였다).
        drop_thr = float(self.cfg['percep']['reset_route_s_drop_m'])
        on_route = not flags.get('off_route')
        if (on_route and self._prev_route_s is not None and not flags.get('reset')
                and self._prev_route_s - route_s > drop_thr):
            self._mark_reset(flags, route_s_drop=self._prev_route_s - route_s)
        self._prev_route_s = route_s if on_route else None

        ego = EgoState(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll,
                       speed=speed, accel=accel, lane=m.lane, s=m.s,
                       route_s=route_s, t_off=m.t, heading_err=m.heading_err)

        # 3) 전방 프로파일 (이번 단계에서는 담아만 두고 쓰지 않는다)
        ahead, summ = [], {}
        if self.route is not None and idx is not None:
            horizon = float(self.cfg['percep']['horizon_m'])
            ahead = self.lg.lookahead(self.route, idx, m.s, horizon)
            summ = self.lg.summarize(ahead)

        # 4) 제한속도 carry + 노면표시
        speed_limit, school = self._resolve_speed_limit(m.lane, ahead)
        left_typ, _lc, left_ok = self.lg.mark_at(m.lane, m.s, 'left')
        right_typ, _rc, right_ok = self.lg.mark_at(m.lane, m.s, 'right')

        # 4b) 9910 light_id ↔ 정지선 controller 대조 (검증용, 주행에는 영향 없음)
        self._check_light_controller(pkt, ahead, flags)

        # 5) 객체: 분류까지만
        objects = self._track_objects(pkt, ego, flags)

        return WorldState(
            t=pkt.t_recv, ego=ego, objects=objects,
            light=pkt.lights[0] if pkt.lights else None,
            ahead=ahead, summ=summ,
            speed_limit=speed_limit, school_zone=school,
            left_solid=not left_ok, right_solid=not right_ok,
            left_is_center=bool(self.lg.lanes[m.lane].get('left_is_center', False)),
            valid=True, flags=flags,
        )

    # ── 리셋(리스폰) 감지 ─────────────────────────────────────────────────
    def _detect_reset(self, x: float, y: float, t: float, flags: dict) -> None:
        """
        VTD 리스폰(courseRespawn) 감지 — **스톨과 구분해서**.

        hl_vtd_config.json: respawnEnabled=true, 도로이탈 0.3 s / 경로이탈 0.5 s,
        respawnPoseToleranceM=1.5 → 실제 리스폰 이동량은 3 m 남짓이다.

        거리만 보면 안 된다. 파이프라인이 2 초 멈췄다 재개되면 차는 정상 속도로
        11 m 를 가 있고, 그건 리스폰이 아니라 우리 쪽 지연이다.
        (실측: 12건 중 11건이 이 오탐이었다 — 점프 벡터가 전부 진행방향이고
         환산속도가 5~6 m/s 로 그 시점 실제 속도와 같았다)

        판정:
          dt > stall_dt_s                                  -> 스톨 (리셋 아님)
          점프 > jump_m  AND  점프/dt > max(v*factor, abs) -> 리스폰
          route_s 역행                                      -> 리스폰 (update 에서 별도 검사)
        """
        if self._prev_xy is None or self._prev_t is None:
            return
        p = self.cfg['percep']
        d = math.hypot(x - self._prev_xy[0], y - self._prev_xy[1])
        dt = t - self._prev_t
        if dt <= 1e-6:
            return

        if dt > float(p['stall_dt_s']):
            # 우리 쪽이 멈춘 것. 위치 차분은 여전히 유효하므로 속도추정을 건드리지 않는다.
            self.stall_count += 1
            self.stall_log.append((t, dt, d))
            flags['stall'] = True
            flags['stall_dt_s'] = round(dt, 3)
            flags['stall_jump_m'] = round(d, 2)
            flags['stall_count'] = self.stall_count
            return

        if d <= float(p['jump_m']):
            return
        v_implied = d / dt
        floor = max(self._speed * float(p['reset_speed_factor']),
                    float(p['reset_abs_speed']))
        if v_implied <= floor:
            return                      # 그 속도면 실제로 갈 수 있는 거리다
        self._mark_reset(flags, jump_m=d, dt=dt, v_implied=v_implied)

    def _mark_reset(self, flags: dict, jump_m: float = None,
                    route_s_drop: float = None, dt: float = None,
                    v_implied: float = None) -> None:
        """리셋 처리: 카운트 + 플래그 + 추정 상태 초기화."""
        self.reset_count += 1
        flags['reset'] = True                 # control 이 이 값을 보고 적분항을 턴다
        flags['reset_count'] = self.reset_count
        if jump_m is not None:
            flags['reset_jump_m'] = round(jump_m, 2)
        if dt is not None:
            flags['reset_dt_s'] = round(dt, 3)
        if v_implied is not None:
            flags['reset_implied_mps'] = round(v_implied, 1)
        if route_s_drop is not None:
            flags['reset_route_s_drop'] = round(route_s_drop, 2)
        self.reset_log.append((dt, jump_m, route_s_drop, v_implied))

        # 속도/가속도 추정 초기화 — 순간이동을 주행으로 세면 안 된다
        self._speed = 0.0
        self._accel = 0.0
        self._prev_xy = None
        self._prev_t = None
        # 리셋 후에는 경로상 위치가 뒤로 갈 수 있다. 직전 인덱스 근처만 보면
        # 엉뚱한 데 붙으므로 처음부터 다시 찾게 한다.
        self._route_idx = 0
        self._prev_route_s = None
        # 객체 트랙도 무의미해진다 (자차가 순간이동했으므로 상대량이 전부 어긋남)
        self._tracks.clear()

    # ── 경로 인덱스 ───────────────────────────────────────────────────────
    def _route_index(self, lane: LaneKey, flags: dict) -> int | None:
        """
        route['lanes'] 안에서 현재 차로의 위치. 같은 차로가 여러 번 나올 수 있어
        직전 인덱스에 가장 가까운 것을 고른다(뒤로 튀지 않게).
        """
        if not self.route:
            return None
        lanes = self.route['lanes']
        hits = [i for i, k in enumerate(lanes) if k == lane]
        if not hits:
            flags['off_route'] = True
            return self._route_idx
        self._route_idx = min(hits, key=lambda i: abs(i - self._route_idx))
        return self._route_idx

    def _default_limit(self) -> float:
        return float(self.cfg.get('default_speed_kph', 50)) / 3.6

    # ── 1) 자차 운동 ──────────────────────────────────────────────────────
    def _estimate_motion(self, x: float, y: float, yaw: float, t: float,
                         flags: dict) -> tuple[float, float]:
        """
        위치 미분 + 저역통과로 (speed, accel) 추정.

        SPEC §1.1: 패킷에 ego 속도 필드가 없다. dt 는 t_recv 차분,
        LPF 계수는 config `percep.speed_lpf`.
        변위를 헤딩에 투영해 전진/후진 부호를 살린다.
        좌표 점프(> `percep.jump_m`)는 리스폰으로 보고 추정치를 리셋한다.
        """
        p = self.cfg['percep']
        alpha = float(p['speed_lpf'])
        jump_m = float(p['jump_m'])

        if self._prev_xy is None or self._prev_t is None:
            self._prev_xy, self._prev_t = (x, y), t
            return 0.0, 0.0

        dt = t - self._prev_t
        dx, dy = x - self._prev_xy[0], y - self._prev_xy[1]
        dist = math.hypot(dx, dy)

        if dt <= 1e-4:
            flags['bad_dt'] = dt
            return self._speed, self._accel
        if dist > jump_m:
            # _detect_reset 이 이미 처리했어야 하는 경우 (방어적)
            flags.setdefault('jump_m', round(dist, 2))
            self._prev_xy, self._prev_t = (x, y), t
            self._speed, self._accel = 0.0, 0.0
            return 0.0, 0.0

        v_raw = (dx * math.cos(yaw) + dy * math.sin(yaw)) / dt
        v = alpha * v_raw + (1.0 - alpha) * self._speed
        self._accel = (v - self._speed) / dt
        self._speed = v
        self._prev_xy, self._prev_t = (x, y), t
        return self._speed, self._accel

    # ── 4) 제한속도 ───────────────────────────────────────────────────────
    def _resolve_speed_limit(self, lane: LaneKey | None, ahead: list) -> tuple[float, bool]:
        """
        현재 유효 제한속도 [m/s] 와 스쿨존 여부.

        SPEC §1.5: xodr 에 표준 <speed> 가 없어 노면표시로 부여했고, 표시 없는
        도로는 None 이다 → **직전 값 유지(carry)**. carry 할 값도 없으면
        config `default_speed_kph` (SPEC §7-4, 주최 문의 중).
        """
        limit_kph, school = (None, False)
        if lane is not None:
            limit_kph, school = self.lg.speed_limit_at(lane)

        if limit_kph is not None:
            self._carry_limit = float(limit_kph)
            self._carry_school = bool(school)

        kph = self._carry_limit if self._carry_limit is not None \
            else float(self.cfg.get('default_speed_kph', 50))
        return kph / 3.6, (bool(school) or self._carry_school if limit_kph is None else bool(school))

    # ── 4b) 신호등 id 대조 ────────────────────────────────────────────────
    def _check_light_controller(self, pkt: RawPacket, ahead: list, flags: dict) -> None:
        """
        9910 의 light_id 가 전방 정지선의 controller 목록에 들어있는지 확인한다.

        9910 light_id 는 **xodr <controller> id** 다 (개별 signal id 가 아니다).
        실측: 정지선 signal_ids=[101..106] 인 곳에서 9910 이 id=27 을 줬고,
        ctrl027 이 제어하는 신호가 101/102/105 였다. 즉 계층이 다를 뿐 정상이다.

        **판단에는 쓰지 않는다.** 주행 로직은 "가장 가까운 전방 정지선 + 현재 state"
        를 그대로 쓰고, 여기서는 규약이 맞는지 관측만 해서 flags 에 남긴다.
        불일치해도 주행은 계속된다.
        """
        if not pkt.lights:
            return
        light_id = int(pkt.lights[0][0])
        flags['light_id'] = light_id

        item = next((a for a in ahead if a.kind == 'stop_line'), None)
        if item is None:
            flags['light_ctrl_match'] = None      # 대조할 전방 정지선이 없다
            return

        cids = self._stop_line_controllers(item)
        flags['stop_ctrl_ids'] = cids
        if not cids:
            # 신호 없는 정지선(일단정지/양보)이거나 controller 매핑이 없는 경우
            flags['light_ctrl_match'] = None
            return

        ok = light_id in cids
        flags['light_ctrl_match'] = ok
        if not ok:
            flags['light_ctrl_mismatch'] = f'{light_id} not in {cids}'

    def _stop_line_controllers(self, item) -> list:
        """lookahead 의 stop_line 항목 → 그 정지선의 controller_ids."""
        rec = self.lg.lanes.get(item.lane)
        if rec is None:
            return []
        best, best_d = None, None
        for sl in rec.get('stop_lines', []):
            d = abs(float(sl['s']) - float(item.s_in_lane))
            if best_d is None or d < best_d:
                best, best_d = sl, d
        if best is None or best_d is None or best_d > 1.0:
            return []
        return list(best.get('controller_ids') or [])

    # ── 5) 객체 ───────────────────────────────────────────────────────────
    @staticmethod
    def classify(length: float, width: float, height: float, speed: float) -> str:
        """
        크기로 객체 종류 판정 (SPEC §1.1 — 타입 필드가 없다).
          보행자: width 0.5–0.8, height 1.5–2.0, length < 1.0
          차량  : length > 3.0
          그 외 + speed≈0: 정적 장애물
        """
        if length > 3.0:
            return 'vehicle'
        if 0.5 <= width <= 0.8 and 1.5 <= height <= 2.0 and length < 1.0:
            return 'pedestrian'
        if abs(speed) < 0.2:
            return 'obstacle'
        return 'unknown'

    def _track_objects(self, pkt: RawPacket, ego: EgoState, flags: dict) -> list[TrackedObject]:
        """
        분류 → 차로 매칭 → 경로 기준 상대량 계산.

        객체 id 는 프레임 간 유지되는 고정 정수라 트래킹이 쉽다(SPEC §1.1).
        `s_rel`(+ = 앞), `lat_off`, `v_rel`(+ = 접근), `ttc` 를 채운다.
        보행자는 `percep.ped_extrapolate_s` 만큼 등속 외삽해 `will_enter_lane` 판정.

        [거리/개수 컷오프 — 공식 확인됨]
        객체는 수평거리 `percep.gt_range_m`(80 m) 이내만, 가까운 순 최대 30개만 온다.
        따라서 목록에서 빠진 것이 곧 소멸이 아니다. 위 규칙으로 구분해 처리한다.
        """
        p = self.cfg['percep']
        gt_range = float(p['gt_range_m'])
        margin = float(p['range_margin_m'])
        coast_s = float(p['coast_s'])
        t = pkt.t_recv

        out: list[TrackedObject] = []
        seen: set[int] = set()
        max_dist = 0.0

        for (oid, ox, oy, _oz, ohead, ospeed, olen, owid, ohei) in pkt.objects:
            oid = int(oid)
            seen.add(oid)
            dist = math.hypot(ox - ego.x, oy - ego.y)
            max_dist = max(max_dist, dist)
            obj = TrackedObject(
                id=oid, x=ox, y=oy, heading=ohead, speed=ospeed,
                length=olen, width=owid, height=ohei,
                cls=self.classify(olen, owid, ohei, ospeed),
                **self._locate_object(ox, oy, ohead, ego),
                # TODO: v_rel / ttc / will_enter_lane (SPEC §3.3 6단계)
                v_rel=0.0, ttc=float('inf'), will_enter_lane=False,
                age=0.0, coasting=False,
            )
            out.append(obj)
            self._tracks[oid] = (obj, t, dist)

        # ── 이번 틱에 안 온 객체 ────────────────────────────────────────
        # 공식 확인: 객체는 수평거리 80 m 이내만, 가까운 순 최대 30개만 온다.
        # 즉 **목록에서 사라진 것이 소멸을 뜻하지 않는다.**
        #   · 마지막 거리가 80 m 근처였다  -> 그냥 범위 밖으로 나간 것. 외삽할 이유 없다.
        #   · 목록이 30개로 꽉 차 있었다    -> 더 가까운 객체에 밀려난 것. 아직 거기 있다.
        #   · 둘 다 아니다                  -> 진짜로 사라짐(가림 등). coast_s 동안 유지한다.
        list_full = len(pkt.objects) >= OBJECT_SLOTS
        dropped_far = dropped_lost = coasted = 0

        for oid, (obj, last_t, last_d) in list(self._tracks.items()):
            if oid in seen:
                continue
            age = t - last_t
            near_range_edge = last_d >= gt_range - margin
            if age > coast_s or (near_range_edge and not list_full):
                del self._tracks[oid]
                if near_range_edge:
                    dropped_far += 1
                else:
                    dropped_lost += 1
                continue
            # 외삽 유지 (등속). 위치를 굴려두면 다음 단계가 그대로 쓸 수 있다.
            obj.x += obj.speed * math.cos(obj.heading) * (t - last_t)
            obj.y += obj.speed * math.sin(obj.heading) * (t - last_t)
            obj.age = age
            obj.coasting = True
            self._tracks[oid] = (obj, last_t, last_d)
            out.append(obj)
            coasted += 1

        flags['obj_n'] = len(pkt.objects)
        flags['obj_max_dist'] = round(max_dist, 1)
        if list_full:
            # 30칸이 꽉 차면 더 먼 객체는 잘려서 안 온다 (거리 컷오프가 아니라 개수 컷오프)
            flags['obj_list_full'] = True
        if max_dist > gt_range + margin:
            # 공식 사양(80 m)보다 먼 객체가 왔다 -> 사양 재확인 필요
            flags['obj_beyond_gt_range'] = round(max_dist, 1)
        if coasted:
            flags['obj_coasting'] = coasted
        if dropped_far:
            flags['obj_left_range'] = dropped_far
        if dropped_lost:
            flags['obj_lost'] = dropped_lost
        return out

    def _locate_object(self, ox: float, oy: float, ohead: float,
                       ego: EgoState) -> dict:
        """
        객체의 차로 / 경로상 종방향 상대거리 / 횡오프셋.

        차선변경 안전 확인이 "목표 차로에 뒤 30 m ~ 앞 50 m 가 비었나" 를 묻기 때문에
        객체가 **어느 차로에 있고 경로상 몇 m 앞인지** 를 알아야 한다.

        s_rel 은 경로 누적거리 차이로 낸다(+ = 앞). 객체가 경로 밖 차로에 있으면
        경로 기준 거리가 없으므로 자차 진행방향 투영으로 대신한다.
        """
        m = self.lg.locate(ox, oy, ohead)
        if m is None:
            dx, dy = ox - ego.x, oy - ego.y
            return {'lane': None, 'on_route': False,
                    's_rel': dx * math.cos(ego.yaw) + dy * math.sin(ego.yaw),
                    'lat_off': -dx * math.sin(ego.yaw) + dy * math.cos(ego.yaw)}

        idx = self._lane_idx.get(m.lane)
        if idx is not None and self.route is not None:
            obj_route_s = float(self.route['cum_s'][idx]) + m.s
            s_rel = obj_route_s - ego.route_s
            on_route = True
        else:
            dx, dy = ox - ego.x, oy - ego.y
            s_rel = dx * math.cos(ego.yaw) + dy * math.sin(ego.yaw)
            on_route = False
        return {'lane': m.lane, 'on_route': on_route, 's_rel': s_rel, 'lat_off': m.t}

    # ── 8) 유효성 ─────────────────────────────────────────────────────────
    def _validate(self, pkt: RawPacket, ego: EgoState) -> tuple[bool, dict]:
        """
        dt 이상 / 좌표 점프 / lane 매칭 실패.

        현재는 update() 안에서 flags 로 직접 처리한다. 신선도·객체 이상 등
        판정을 늘릴 때 이쪽으로 모을 것.
        """
        # TODO: 판정 항목이 늘어나면 여기로 통합
        return True, {}
