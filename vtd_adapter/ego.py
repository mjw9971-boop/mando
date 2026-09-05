"""
자차 상태 추정 — 기존 perception.py 에서 검증된 부분만 이식 (phase1).

  EgoSpeedEstimator  속도/가속도 추정 + courseRespawn 리셋·스톨 감지
  EgoTracker         + 차로 매칭 / route_s / lookahead / 제한속도 carry → WorldState

판단(정지선·선행차·보행자 등)은 여기 없다 — PDM-Lite(team_code)가 한다.
객체 트래킹은 phase2 에서 world.py(VtdWorld)로 이식한다.

검증 이력 (원본 perception.py 의 실측 근거를 그대로 승계):
  · 속도: 0.4 s 슬라이딩 창 Σ변위/Σ벽시계 dt — 40/80 ms 불규칙 간격에서 편향 0.0 %
  · 리셋: 거리 단독이 아니라 환산속도·t_off 불연속·route_s 역행의 3중 판정
  · 스톨(우리 쪽 지연)은 리셋과 구분한다 — 위치 차분은 유효한 주행 변위다
"""
from __future__ import annotations

import math

from .lanegraph import LaneGraph
from .route import check_light_controller
from .types import EgoState, LaneKey, RawPacket, WorldState


class EgoSpeedEstimator:
    """9910 에 자차 속도 필드가 없다 — 위치 미분으로 추정하고 리셋을 감지한다."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self._prev_xy: tuple[float, float] | None = None
        self._prev_t: float | None = None
        self._speed = 0.0
        self._accel = 0.0
        # 속도 추정 창: (진행방향 변위 [m], 벽시계 dt [s]) 의 최근 항목들.
        self._win: list[tuple[float, float]] = []
        self.reset_count = 0         # VTD 리셋(리스폰) 누적 = 실제 이탈 횟수
        self.stall_count = 0         # 파이프라인 멈춤 누적 (리셋과 섞지 않는다)
        self.reset_log: list = []    # (dt, 점프거리, route_s_drop, 환산속도)
        self.stall_log: list = []

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
          route_s 역행                                      -> 리스폰 (EgoTracker 에서 별도 검사)
        """
        if self._prev_xy is None or self._prev_t is None:
            return
        p = self.cfg['percep']
        d = math.hypot(x - self._prev_xy[0], y - self._prev_xy[1])
        dt = t - self._prev_t
        if dt <= 1e-6:
            return

        if dt > float(p['stall_dt_s']):
            # 긴 dt 라도 이동량이 물리적으로 불가능하면 스톨이 아니라 텔레포트다.
            # (2026-08-21: 25 s 갭 + 590 m 순간이동이 스톨로 분류돼 속도가 -1.34 로
            #  오염되고 트랙/적분항이 리셋되지 않았다.)
            if d > float(p['stall_teleport_m']):
                self._mark_reset(flags, jump_m=d, dt=dt, v_implied=d / dt)
                return
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
                    v_implied: float = None, toff_jump: float = None) -> None:
        """리셋 처리: 카운트 + 플래그 + 추정 상태 초기화."""
        self.reset_count += 1
        flags['reset'] = True                 # 하류(제어기 상태 초기화)가 이 값을 본다
        flags['reset_count'] = self.reset_count
        if jump_m is not None:
            flags['reset_jump_m'] = round(jump_m, 2)
        if dt is not None:
            flags['reset_dt_s'] = round(dt, 3)
        if v_implied is not None:
            flags['reset_implied_mps'] = round(v_implied, 1)
        if route_s_drop is not None:
            flags['reset_route_s_drop'] = round(route_s_drop, 2)
        if toff_jump is not None:
            flags['reset_toff_jump'] = round(toff_jump, 2)
        self.reset_log.append((dt, jump_m, route_s_drop, v_implied))

        # 속도/가속도 추정 초기화 — 순간이동을 주행으로 세면 안 된다
        self._speed = 0.0
        self._accel = 0.0
        self._win.clear()
        self._prev_xy = None
        self._prev_t = None

    # ── 속도/가속도 ───────────────────────────────────────────────────────
    def _estimate_motion(self, x: float, y: float, yaw: float, t: float,
                         flags: dict) -> tuple[float, float]:
        """
        **슬라이딩 창** 속도 추정: 최근 `percep.speed_win_s`(0.4 s) 동안의
        Σ(진행방향 변위) / Σ(벽시계 dt).

        SPEC §1.1: 패킷에 ego 속도 필드가 없다. dt 는 t_recv(벽시계) 차분이다.

        왜 창인가 — 9910 송신 간격은 40/80 ms 로 불규칙하고(평균 50 ms) 변위는
        **벽시계에 정확히 비례**한다 (2026-08-23 실측: 40 ms 틱 0.591 m, 80 ms 틱
        1.181 m, 둘 다 53 km/h). 예전 코드는 dt 하한을 공칭 50 ms 로 잡아 40 ms
        틱(68 %)에서 속도를 20 % 낮게 냈고 LPF 결과가 −15 % 편향됐다 →
        v_target 45 km/h 에 실속도 53 km/h 로 제한속도(S1.1.01)를 186틱 넘겼다.

        창은 틱 하나의 dt 오차(스톨 직후 10 ms 간격으로 몰려 오는 프레임 등)를
        희석한다. 창은 **항상 speed_win_s 이상**을 유지한다 — 가장 오래된 항목을
        빼도 창이 speed_win_s 이상 남을 때만 뺀다.

        변위를 헤딩에 투영해 전진/후진 부호를 살린다. 리셋(순간이동)은
        _detect_reset/_mark_reset 이 창을 비운다; 스톨은 실제 주행 변위이므로
        그대로 창에 넣는다.
        """
        p = self.cfg['percep']
        alpha = float(p['speed_lpf'])          # 가속도(속도 미분)에만 쓴다
        jump_m = float(p['jump_m'])
        win_s = float(p.get('speed_win_s', 0.4))

        if self._prev_xy is None or self._prev_t is None:
            self._prev_xy, self._prev_t = (x, y), t
            return self._speed, self._accel

        dt = t - self._prev_t
        dx, dy = x - self._prev_xy[0], y - self._prev_xy[1]
        dist = math.hypot(dx, dy)

        if dt <= 1e-4:
            flags['bad_dt'] = dt
            return self._speed, self._accel

        # 리스폰 판정과 같은 척도: 이 dt 동안 현재 속도의 reset_speed_factor 배로
        # 가야 나오는 거리보다 멀어야 점프다. dt 가 길면 허용치도 늘어난다.
        allow = max(jump_m, abs(self._speed) * dt * float(p['reset_speed_factor']))
        # _detect_reset 이 이 틱을 '스톨'(우리 쪽 지연)로 분류했으면 위치 차분은
        # 여전히 유효한 주행 변위다 — 점프 리셋을 걸지 않는다.
        if dist > allow and not flags.get('stall'):
            # _detect_reset 이 이미 처리했어야 하는 경우 (방어적)
            flags.setdefault('jump_m', round(dist, 2))
            self._prev_xy, self._prev_t = (x, y), t
            self._speed, self._accel = 0.0, 0.0
            self._win.clear()
            return 0.0, 0.0

        d_fwd = dx * math.cos(yaw) + dy * math.sin(yaw)
        self._win.append((d_fwd, dt))
        tot = sum(w[1] for w in self._win)
        while len(self._win) > 1 and tot - self._win[0][1] >= win_s:
            tot -= self._win[0][1]
            self._win.pop(0)
        v = sum(w[0] for w in self._win) / tot

        a_raw = (v - self._speed) / dt
        self._accel = alpha * a_raw + (1.0 - alpha) * self._accel
        self._speed = v
        self._prev_xy, self._prev_t = (x, y), t
        return self._speed, self._accel


class EgoTracker(EgoSpeedEstimator):
    """
    EgoSpeedEstimator + 차로 매칭 / route_s / 전방 프로파일 → WorldState.

    로그 스키마(ego.lane/s/route_s/t_off, world.ahead/summ/…)를 채우는 데 필요한
    상태 계산만 있다. objects 는 phase1 에서는 빈 리스트다 (phase2: VtdWorld).
    """

    def __init__(self, lg: LaneGraph, route: dict | None, cfg: dict) -> None:
        super().__init__(cfg)
        self.lg = lg
        self.route = route
        self._route_idx = 0          # 경로 진행 인덱스 (단조 증가 힌트)
        self._prev_route_s: float | None = None
        self._prev_lane: LaneKey | None = None   # t_off 불연속 검사용 (같은 차로끼리만)
        self._prev_toff: float | None = None
        self._carry_limit: float | None = None
        self._carry_school = False
        # 차선변경 목표 차로 — 변경 중에도 매칭이 흔들리지 않게 prefer 에 상시 포함
        self._lc_lanes: list = []
        if route:
            for e in route.get('events', []):
                if e['kind'].startswith('lane_change') and e.get('to_lane'):
                    self._lc_lanes.append(tuple(e['to_lane']))
        self._prefer: list | None = None
        if route:
            self._prefer = list(route['lanes']) + [
                k for k in self._lc_lanes if k not in set(route['lanes'])]

    def _mark_reset(self, flags: dict, **kw) -> None:
        super()._mark_reset(flags, **kw)
        # 리셋 후에는 경로상 위치가 뒤로 갈 수 있다. 직전 인덱스 근처만 보면
        # 엉뚱한 데 붙으므로 처음부터 다시 찾게 한다.
        self._route_idx = 0
        self._prev_route_s = None
        self._prev_lane = self._prev_toff = None

    def update(self, pkt: RawPacket) -> WorldState:
        """한 패킷을 WorldState 로 (objects 제외 — phase2 에서 VtdWorld 가 채운다)."""
        x, y, z, yaw, pitch, roll = pkt.ego[0], pkt.ego[1], pkt.ego[2], pkt.ego[3], pkt.ego[4], pkt.ego[5]
        flags: dict = {}

        # 0) VTD 리셋(courseRespawn) 감지 — 속도 추정보다 먼저 봐야 한다
        self._detect_reset(x, y, pkt.t_recv, flags)

        # 1) 속도/가속도 추정 (패킷에 속도 필드가 없다)
        speed, accel = self._estimate_motion(x, y, yaw, pkt.t_recv, flags)

        # 2) 차로 매칭
        m = self.lg.locate(x, y, yaw, prefer=self._prefer)

        valid = m is not None
        if not valid:
            flags['locate_failed'] = True
            self._prev_lane = self._prev_toff = None   # 복귀 틱에서 낡은 t_off 와 비교 금지
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
        # 단 **경로 위에 있을 때만** 의미가 있다 (완주/이탈 후에는 오탐).
        drop_thr = float(self.cfg['percep']['reset_route_s_drop_m'])
        on_route = not flags.get('off_route')
        if (on_route and self._prev_route_s is not None and not flags.get('reset')
                and self._prev_route_s - route_s > drop_thr):
            self._mark_reset(flags, route_s_drop=self._prev_route_s - route_s)
        self._prev_route_s = route_s if on_route else None

        # 횡 오프셋 불연속도 리셋 신호다. 고속 리스폰은 환산속도 문턱을 빠져나간다.
        # 조건 세 개 전부: 같은 차로 t_off 점프 + 점프 후 중심 근처 + 차로 정렬
        # (2·3 이 리스폰 시그니처 — respawn 은 차를 중심선에 정렬해 놓는다)
        toff_thr = float(self.cfg['percep']['reset_toff_jump_m'])
        if (not flags.get('reset') and self._prev_lane == m.lane
                and self._prev_toff is not None
                and abs(m.t - self._prev_toff) > toff_thr
                and abs(m.t) < 0.7 and abs(m.heading_err) < 0.15):
            self._mark_reset(flags, toff_jump=abs(m.t - self._prev_toff))
        self._prev_lane, self._prev_toff = m.lane, m.t

        ego = EgoState(x=x, y=y, z=z, yaw=yaw, pitch=pitch, roll=roll,
                       speed=speed, accel=accel, lane=m.lane, s=m.s,
                       route_s=route_s, t_off=m.t, heading_err=m.heading_err)

        # 3) 전방 프로파일 (로그 스키마 world.ahead/summ 용)
        ahead, summ = [], {}
        if self.route is not None and idx is not None:
            horizon = float(self.cfg['percep']['horizon_m'])
            ahead = self.lg.lookahead(self.route, idx, m.s, horizon)
            summ = self.lg.summarize(ahead)

        # 4) 제한속도 carry + 노면표시 (붉은 구간은 s 로 판정 — m.s 를 넘긴다)
        speed_limit, school = self._resolve_speed_limit(m.lane, m.s)
        left_typ, _lc, left_ok = self.lg.mark_at(m.lane, m.s, 'left')
        right_typ, _rc, right_ok = self.lg.mark_at(m.lane, m.s, 'right')

        # 4b) 9910 light_id ↔ 정지선 controller 대조 (검증용, 주행에는 영향 없음)
        check_light_controller(self.lg, pkt, ahead, flags)

        return WorldState(
            t=pkt.t_recv, ego=ego, objects=[],
            light=pkt.lights[0] if pkt.lights else None,
            ahead=ahead, summ=summ,
            speed_limit=speed_limit, school_zone=school,
            left_solid=not left_ok, right_solid=not right_ok,
            left_is_center=bool(self.lg.lanes[m.lane].get('left_is_center', False)),
            valid=True, flags=flags,
        )

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
            # 정차 차량 추월로 **경로 차로 바로 옆**에 있을 수 있다. 좌/우 이웃이
            # 경로 차로면 그 인덱스를 그대로 쓴다 — 나란한 차로는 s 매개화가 같아
            # route_s 가 이어진다.
            for side in ('left', 'right'):
                nb = self.lg.neighbor(lane, side)
                if nb is None:
                    continue
                nb_hits = [i for i, k in enumerate(lanes) if k == nb]
                if nb_hits:
                    self._route_idx = min(nb_hits, key=lambda i: abs(i - self._route_idx))
                    flags['beside_route'] = True
                    return self._route_idx
            flags['off_route'] = True
            return self._route_idx
        self._route_idx = min(hits, key=lambda i: abs(i - self._route_idx))
        return self._route_idx

    def _default_limit(self) -> float:
        return float(self.cfg.get('default_speed_kph', 50)) / 3.6

    # ── 제한속도 ──────────────────────────────────────────────────────────
    def _resolve_speed_limit(self, lane: LaneKey | None,
                             s: float | None = None) -> tuple[float, bool]:
        """
        현재 유효 제한속도 [m/s] 와 보호구역(붉은 노면) 여부.

        SPEC §1.5: xodr 에 표준 <speed> 가 없어 노면표시로 부여했고, 표시 없는
        도로는 None 이다 → **직전 값 유지(carry)**. carry 할 값도 없으면
        config `default_speed_kph`.

        `s` 를 주면 `speed_limit_at` 이 붉은 구간(red_spans)을 s 로 판정한다.
        구간 안이면 `red_zone.limit_kph`, 붉은 차로인데 구간 밖이면 None 이 와서
        **carry 규칙이 그대로 앞 값을 물고 간다** — carry 는 손대지 않았다.
        """
        limit_kph, school = (None, False)
        if lane is not None:
            limit_kph, school = self.lg.speed_limit_at(lane, s)

        if limit_kph is not None:
            self._carry_limit = float(limit_kph)
            self._carry_school = bool(school)

        kph = self._carry_limit if self._carry_limit is not None \
            else float(self.cfg.get('default_speed_kph', 50))
        return kph / 3.6, (bool(school) or self._carry_school if limit_kph is None else bool(school))
