"""
한국 대회 규칙 계층 — PDM-Lite 판단 결과를 받아 규칙을 덮어쓴다.

PDM-Lite(autopilot.py) 원문은 건드리지 않는다. autopilot._get_control 맨 끝의
한 줄(`self.kr_rules.apply(...)`)이 유일한 접점이고, 여기서는 PDM 의 min()
중재에 **후보를 덧대는** 형태로만 개입한다 — 새 감속 프로파일을 만들지 않고
PDM 의 _compute_target_speed_idm / 종방향 컨트롤러를 그대로 재사용한다.

phase4 현재: route_end 정지 / 정지선 유지 홀드 / 방향지시등. (RTOR·황색 딜레마는 이후 단계.)

route_end — 경로 종점 정지:
  CARLA 리더보드는 결승선 통과로 시나리오가 끝나 PDM 에 종점 정지 개념이
  없다. 실기(2026-08-26 완주속도_01_기본): 종점 도달 후 v_target 6.9 로 계속
  주행 → 경로 밖 이탈 → courseRespawn 9회.

  구현: "종점에 정지해 있는 길이 0 유령 선행차" 를 IDM 에 넣는다. 유효거리를
  d_end − 앞범퍼, s0 를 speed.stop_gap_route_end_m 으로 주면 앞범퍼가 기준점 −
  stop_gap 에 선다. batch 완주 임계(total − end_margin, end_margin = stop_gap +
  앞범퍼 + end_slack)보다 end_slack_m 만큼 안쪽이라 완주 판정과 자동 정합한다
  (tests/test_route_end). ※ 정지선(적신호) 정지는 이 관례와 무관 — PDM
  red-light IDM 소관이고 run_agent.build_pdm_config 가 stop_gap_stopline_m 로
  주입한다. 여기서는 그 정지의 **0.5 s 유지**만 홀드로 보강한다 (아래).

stopline hold — 정지선 정지 유지:
  대회 7번: 정지선 앞 정지는 0.5 s 이상 유지해야 정상. 실측(2026-08-27,
  보행자집중_06) 0.4 s 만에 재출발한 사례가 감점 대상이라, 적신호 정지선
  근처(stopline_hold_near_m)에서 저속(latch_v — 기존 래치 관례 재사용)이 되면
  stopline_hold_s 동안 목표 0 을 유지한다. 홀드 중에는 신호가 녹색으로 바뀌어도
  잔여 시간을 채운다.

  래치: 종점 근처(latch_m)에서 저속(latch_v)이 되면 래치 — 재출발하지 않는다.
  d_end 가 unlatch_m 이상으로 다시 커지면(courseRespawn 으로 뒤로 간 경우)
  해제해 고착을 막는다.

  정지 목표 기준점(stop_s): 대회 규칙은 "뒷축이 종료 지점 통과 = 시험 종료"라
  route_end.target_mode='finish' 면 scoring.finish_xy 를 경로에 투영한 종료선
  (finish_s)을 뒷축이 finish_clearance_m 만큼 넘어 정지하도록 기준점을 잡는다
  (plan_stop_s — 채점 score.py 와 공용, 단일 출처). d_eff/s0 관례·래치·active_m
  판정 거리는 전부 stop_s 기준으로 그대로 동작한다.

turn signal — 방향지시등:
  채점 동적항목("방향지시등 n초 전"). PDM 은 CARLA 리더보드용이라 지시등 개념이
  없어 9910 turnSignal 이 계속 0 이었다.

  경로가 정적이므로 **점등 구간을 시작 시 1회 계산**한다 — route['events'] 의
  turn_left/right(연결로 시작 s, 끝은 같은 junction 차로가 이어지는 데까지)와
  lane_change_left/right(window_s0 ~ 블렌드 끝). 매 틱은 route_s 로 고르기만
  하므로 재선택 깜빡임(실사고 §6-9)이 구조적으로 생기지 않는다.

  점등 조건: 남은거리 ≤ max(v · lead_s, lead_min_m). 시간 기준만 쓰면 적신호
  대기(v→0)에서 선행거리가 0 이 돼 회전 지시등이 안 켜지므로 거리 하한을 둔다.
  겹치면 SPEC §3.3 대로 **남은거리가 짧은 쪽 우선, 동률이면 회전 우선**.

  결과는 last_turn_signal/last_sig_src/last_sig_lead_s 로 노출하고 run_agent 가
  Command.turn_signal 과 로그에 싣는다 (기존 last_candidate/last_target 관례).
"""
from __future__ import annotations

import math as _math

from scipy.spatial import cKDTree as _cKDTree

from vtd_adapter import frame


def plan_stop_s(cfg: dict, total: float, finish_s: float | None) -> tuple[float, bool]:
    """정지 목표 기준점 stop_s [route_s] 와 클립 여부. 제어·채점 공용 (단일 출처).

    finish_s 있으면 stop_s = min(finish_s + finish_clearance_m + stop_gap + 앞범퍼,
    total − end_slack) — 유령차 기준점에서 앞범퍼가 stop_s − stop_gap, 뒷축이
    stop_s − stop_gap − 앞범퍼 = finish_s + clearance 에 서므로 뒷축이 종료선을
    여유를 두고 넘는다. stop_gap 을 빼먹으면 뒷축이 finish_s − 2.0 에 서서 여전히
    미달한다 (2026-08-27 검토에서 잡은 결함). 클립되면(경로 꼬리 부족) True 와
    함께 total − end_slack 을 돌려준다. finish_s 없으면 기존과 동일하게 total.
    """
    if finish_s is None:
        return float(total), False
    sp, vh = cfg['speed'], cfg['vehicle']
    want = (float(finish_s) + float(cfg['scoring']['finish_clearance_m'])
            + float(sp['stop_gap_route_end_m'])
            + float(vh['wheelbase']) + float(vh['front_overhang_m']))
    cap = float(total) - float(cfg.get('batch', {}).get('end_slack_m', 1.0))
    return (min(want, cap), want > cap)


SIG_OFF, SIG_LEFT, SIG_RIGHT = 0, 1, 2        # 9910 turnSignal (SPEC §1.2)


def _turn_end_s(lg, lanes, cum, lens, ev) -> float:
    """회전 이벤트의 소등 지점 [route_s] — 같은 junction 차로가 이어지는 끝까지.

    build_route 의 turn 이벤트는 시작 s 만 준다(연결로가 여러 개 이어질 수 있어
    끝은 경로에서 되짚어야 한다). lg 나 lanes 가 없으면(목 플래너) 시작점 반환.
    """
    s0 = float(ev['s'])
    if lg is None or not lanes or not cum:
        return s0
    i = min(range(len(cum)), key=lambda j: abs(float(cum[j]) - s0))
    rec = lg.lanes.get(tuple(lanes[i]))
    if rec is None:
        return s0
    end_of = lambda j: float(cum[j]) + (float(lens[j]) if j < len(lens) else 0.0)
    jid = rec.get('junction', -1)
    if jid == -1:
        return end_of(i)
    j = i
    while j + 1 < len(lanes):
        nxt = lg.lanes.get(tuple(lanes[j + 1]))
        if nxt is None or nxt.get('junction') != jid:
            break
        j += 1
    return end_of(j)


def signal_intervals(planner) -> list[dict]:
    """route['events'] → 지시등 점등 구간 [{sig, src, ev_s, end_s}]. 시작 시 1회.

    turn : ev_s = 연결로 시작, end_s = 연결로 끝
    lc   : ev_s = 창 시작(window_s0), end_s = 블렌드 끝(창 끝과 전이길이 중 짧은 쪽
           — 창 끝까지 켜 두면 이미 옮겨탄 뒤에도 점등이 남는다)
    """
    route = getattr(planner, 'route', None) or {}
    lg = getattr(planner, 'lg', None)
    lanes = route.get('lanes') or []
    cum = route.get('cum_s') or []
    lens = route.get('lengths') or []
    lc_span = float(getattr(planner, 'LC_TRANSITION_M', 25.0))

    out: list[dict] = []
    for ev in route.get('events') or []:
        kind = str(ev.get('kind', ''))
        if kind.startswith('turn_'):
            out.append({'sig': SIG_LEFT if kind.endswith('left') else SIG_RIGHT,
                        'src': 'turn', 'ev_s': float(ev['s']),
                        'end_s': _turn_end_s(lg, lanes, cum, lens, ev)})
        elif kind.startswith('lane_change_'):
            s0 = float(ev.get('window_s0', ev['s']))
            s1 = float(ev.get('window_s1', s0))
            out.append({'sig': SIG_LEFT if kind.endswith('left') else SIG_RIGHT,
                        'src': 'lc', 'ev_s': s0,
                        'end_s': min(s1, s0 + lc_span) if s1 > s0 else s0 + lc_span})
    out.sort(key=lambda d: d['ev_s'])
    return out


def _project_route_s(lg, route: dict, x: float, y: float) -> float | None:
    """좌표 → 경로 누적거리 (score.project_route_s 와 같은 정의 — 경로 차로 투영)."""
    best = None
    for i, k in enumerate(route.get('lanes') or []):
        try:
            s_p, _t, d_p, _ = lg.project(tuple(k), x, y)
        except KeyError:
            continue
        if best is None or d_p < best[0]:
            best = (d_p, float(route['cum_s'][i]) + float(s_p))
    return best[1] if best else None


class KrRules:
    def __init__(self, cfg: dict) -> None:
        re_cfg = cfg['route_end']
        sp, vh = cfg['speed'], cfg['vehicle']
        self.cfg = cfg
        self.stop_gap = float(sp['stop_gap_route_end_m'])
        self.front = float(vh['wheelbase']) + float(vh['front_overhang_m'])
        self.T = float(re_cfg['idm_time_headway'])
        self.active_m = float(re_cfg['active_m'])
        self.latch_v = float(re_cfg['latch_v'])
        self.latch_m = float(re_cfg['latch_m'])
        self.unlatch_m = float(re_cfg['unlatch_m'])
        self.target_mode = str(re_cfg['target_mode'])
        self.finish_xy = (cfg['scoring'] or {}).get('finish_xy')
        # 정지선 0.5 s 유지 홀드 (규정 + 여유는 params 가 단일 출처).
        # 틱 카운트로 잰다 — wall clock 은 리플레이/시뮬에서 흐름이 다르다.
        self.sl_hold_ticks = int(round(float(sp['stopline_hold_s'])
                                       * float(cfg['comm']['send_hz'])))
        self.sl_near_m = float(sp['stopline_hold_near_m'])
        # 정적 장애물 회피 시프트 (SPEC §3.4 회피 — PDM 원문은 stub)
        ot = cfg['overtake']
        self.ot_enabled = bool(ot['enabled'])
        self.ot_v_max = float(ot['blocker_speed_max'])
        self.ot_d_max = float(ot['blocker_dist_max'])
        self.ot_ticks = int(round(float(ot['trigger_s']) * float(cfg['comm']['send_hz'])))
        self.ot_min_corridor = float(ot['min_corridor_m'])
        self.ot_clear_r = float(ot['clear_radius_m'])
        self.ot_trans_m = float(ot['transition_m'])
        self.ot_before_m = float(ot['extra_before_m'])
        self.ot_after_m = float(ot['extra_after_m'])
        # 방향지시등 (SPEC §3.3). lc_lead_s 는 규정 미확정 가정값 (§7-2).
        sig = cfg['signal']
        self.turn_lead_s = float(sig['turn_lead_s'])
        self.lc_lead_s = float(sig['lc_lead_s'])
        self.sig_lead_min_m = float(sig['lead_min_m'])

        self.latched = False
        self.stop_s: float | None = None           # 시작 시 1회 계산 캐시 (매 틱 투영 금지)
        self.sl_hold_left = 0                      # 정지선 홀드 잔여 틱
        self.last_candidate: float | None = None   # 이번 틱 route_end 후보 (로그용)
        self.last_target: float | None = None      # 이번 틱 최종 목표속도 (로그용)
        self.last_d_end: float | None = None
        self.ot_blocked_ticks = 0                  # 막힌 채 정지한 틱
        self.ot_span: tuple | None = None          # 시프트한 인덱스 구간
        self.last_overtake: str | None = None      # 로그용 ('left'|'right'|사유)
        self.sig_plan: list[dict] | None = None    # 시작 시 1회 계산 (매 틱 재구성 금지)
        self.last_turn_signal: int = SIG_OFF       # 이번 틱 지시등 (run_agent 가 읽는다)
        self.last_sig_src: str | None = None       # 'turn' | 'lc'
        self.last_sig_lead_s: float | None = None  # 이벤트까지 남은 시간 [s]

    def _resolve_stop_s(self, planner) -> float:
        """정지 목표 기준점 1회 산출. finish 모드 실패 시 경고 후 total 폴백."""
        total = float(planner.route['total_length'])
        if self.target_mode != 'finish':
            return total
        if not self.finish_xy:
            print('[kr_rules] scoring.finish_xy 미설정 — route_total 기준으로 정지 (기존 동작)',
                  flush=True)
            return total
        lg = getattr(planner, 'lg', None)
        finish_s = (_project_route_s(lg, planner.route,
                                     float(self.finish_xy[0]), float(self.finish_xy[1]))
                    if lg is not None else None)
        if finish_s is None:
            print('[kr_rules] finish_xy 를 경로에 투영하지 못함 — route_total 기준으로 정지',
                  flush=True)
            return total
        stop_s, clipped = plan_stop_s(self.cfg, total, finish_s)
        if clipped:
            print(f'[kr_rules] ⚠ 계획 정지점이 종료선을 못 넘는다 — finish_s {finish_s:.1f} '
                  f'+ 여유가 경로 종점을 초과 (경로 꼬리 부족). 종점까지 주행한다', flush=True)
        return stop_s

    def _turn_signal(self, planner, route_s: float, ego_speed: float) -> tuple:
        """점등 구간 중 지금 켤 것을 고른다 → (sig, src, lead_s).

        점등 조건은 남은거리 ≤ max(v·lead_s, lead_min_m). 겹치면 SPEC §3.3 대로
        남은거리가 짧은 쪽, 동률이면 회전 우선. 지난 구간(route_s > end_s)은
        후보에서 빠지므로 재점등이 없다.
        """
        if self.sig_plan is None:
            self.sig_plan = signal_intervals(planner)

        best = None
        for iv in self.sig_plan:
            if route_s > iv['end_s']:
                continue
            lead_s = self.turn_lead_s if iv['src'] == 'turn' else self.lc_lead_s
            remain = iv['ev_s'] - route_s
            if remain > max(ego_speed * lead_s, self.sig_lead_min_m):
                continue
            key = (max(0.0, remain), 0 if iv['src'] == 'turn' else 1)
            if best is None or key < best[0]:
                best = (key, iv, remain)

        if best is None:
            return SIG_OFF, None, None
        _key, iv, remain = best
        # 남은 시간은 참고용 로그 — 저속에서는 발산하므로 남기지 않는다
        lead = max(0.0, remain) / ego_speed if ego_speed > 0.1 else None
        return iv['sig'], iv['src'], lead

    # ── 정적 장애물 회피 시프트 ──────────────────────────────────────────
    def _blocker(self, ap, planner):
        """앞을 막고 선 정적 장애물 → VtdActor. 없으면 None.

        PDM 은 타입 필드가 없는 9910 객체를 전부 vehicle 로 감싸므로(actor.py),
        정차 차량·공사 표지·파손 차량이 모두 여기 걸린다 — 대응이 같으니 무방하다.
        """
        try:
            vehicles = list(ap._world.get_actors().filter('*vehicle*'))
        except Exception:                                  # noqa: BLE001
            return None
        ego = ap._vehicle
        ids = set(planner.compute_leading_vehicles(vehicles, ego.id))
        if not ids:
            return None
        best, best_d = None, None
        ex, ey = ego.get_location().x, ego.get_location().y
        for a in vehicles:
            if a.id not in ids or float(getattr(a, 'speed', 0.0)) > self.ot_v_max:
                continue
            loc = a.get_location()
            d = _math.hypot(loc.x - ex, loc.y - ey)
            if d > self.ot_d_max:
                continue
            if best_d is None or d < best_d:
                best, best_d = a, d
        return best

    def _side_is_clear(self, lg, planner, ap, target) -> bool:
        """목표 차로에 차가 없는가 (lc_clear 대용 — 아직 후방 추종차는 안 본다)."""
        ego = ap._vehicle
        ex, ey = ego.get_location().x, ego.get_location().y
        try:
            actors = list(ap._world.get_actors())
        except Exception:                                  # noqa: BLE001
            return False
        near = {target}
        near |= set(lg.successors(target)) | set(lg.predecessors(target))
        for a in actors:
            if a.id == ego.id:
                continue
            loc = a.get_location()
            if _math.hypot(loc.x - ex, loc.y - ey) > self.ot_clear_r:
                continue
            vx, vy = frame.from_carla_xy(loc.x, loc.y)
            try:
                m = lg.locate(vx, vy)
            except Exception:                              # noqa: BLE001
                continue
            if m is not None and m.lane in near:
                return False
        return True

    def _try_overtake(self, ap, planner, ego_speed: float) -> None:
        """막힌 채 서 있으면 경로를 옆 차로로 밀어 비켜간다 (1회, 게이트 통과 시).

        게이트: 목표 차로 존재 · 교차로 아님 · 점선 회랑 충분(S2.2.05) · 측방 비어 있음.
        시프트는 나갔다 돌아오는 프로파일이라(양 끝 전이계수 0) 복귀는 자동이고,
        지나가면 경로를 원상 복구해 다음 장애물에 다시 쓸 수 있게 한다.
        """
        # 지나갔으면 원복 (다음 장애물용)
        if self.ot_span is not None and planner.route_index > self.ot_span[1]:
            a, b = self.ot_span
            planner.route_points[a:b] = planner.original_route_points[a:b]
            planner.commands[a:b] = planner.commands_orig[a:b]
            planner._kd = _cKDTree(planner.route_points[:, :2])
            self.ot_span = None
            self.last_overtake = 'restored'
            return
        if not self.ot_enabled or self.ot_span is not None:
            return

        blocked = ego_speed < self.latch_v and self._blocker(ap, planner) is not None
        self.ot_blocked_ticks = self.ot_blocked_ticks + 1 if blocked else 0
        if self.ot_blocked_ticks < self.ot_ticks:
            return

        actor = self._blocker(ap, planner)
        if actor is None:
            return
        lg = getattr(planner, 'lg', None)
        ego_lane = getattr(ap, '_kr_ego_lane', None) or self._ego_lane(lg, ap)
        if lg is None or ego_lane is None:
            self.last_overtake = 'no_lane'
            return
        if lg.lanes[ego_lane]['junction'] != -1:
            self.last_overtake = 'junction'
            return

        for side in ('left', 'right'):                     # 좌측 추월 우선
            target = lg.neighbor(ego_lane, side)
            if target is None:
                continue
            if lg.dashed_corridor_m(ego_lane, side) < self.ot_min_corridor:
                self.last_overtake = f'{side}:solid'
                continue
            if not self._side_is_clear(lg, planner, ap, target):
                self.last_overtake = f'{side}:occupied'
                continue
            ppm = float(getattr(planner, 'points_per_meter', 10))
            span = planner.shift_route_around_actors(
                actor,
                obstacle_direction='right' if side == 'left' else 'left',
                transition_length=self.ot_trans_m * ppm,
                extra_length_before=self.ot_before_m * ppm,
                extra_length_after=self.ot_after_m * ppm)
            planner._kd = _cKDTree(planner.route_points[:, :2])
            self.ot_span = span
            self.ot_blocked_ticks = 0
            self.last_overtake = side
            print(f'[kr_rules] 정적 장애물 회피 — {side} 로 경로 시프트 '
                  f'(id={actor.id}, 구간 {span[0]}~{span[1]})', flush=True)
            return
        if self.last_overtake is None:
            self.last_overtake = 'no_neighbor'

    @staticmethod
    def _ego_lane(lg, ap):
        if lg is None:
            return None
        loc = ap._vehicle.get_location()
        vx, vy = frame.from_carla_xy(loc.x, loc.y)
        try:
            m = lg.locate(vx, vy)
        except Exception:                                  # noqa: BLE001
            return None
        return m.lane if m is not None else None

    def _stopline_hold(self, planner, ego_speed: float) -> float | None:
        """적신호 정지선 정지의 최소 유지 (speed.stopline_hold_s) — 목표 0 후보.

        다음 신호 정지선이 적색이고 앞범퍼가 stopline_hold_near_m 안에서
        저속(latch_v — 기존 래치 관례 재사용)이 되면 홀드 시작. 홀드 중에는
        신호가 녹색으로 바뀌어도 잔여 틱을 채운다 (규정 "0.5 s 이상 정지" —
        실측 0.4 s 재출발이 감점 대상). 신호 정보가 없는 환경(목 플래너 등)
        에서는 개입하지 않는다.
        """
        if self.sl_hold_left > 0:
            self.sl_hold_left -= 1
            return 0.0
        dists = getattr(planner, 'distances_to_next_traffic_lights', None)
        tls = getattr(planner, 'next_traffic_lights', None)
        if dists is None or tls is None:
            return None
        tl = tls[planner.route_index]
        if tl is None or getattr(getattr(tl, 'state', None), 'name', None) != 'Red':
            return None
        d_front = float(dists[planner.route_index]) - self.front
        if d_front < self.sl_near_m and ego_speed < self.latch_v:
            self.sl_hold_left = max(0, self.sl_hold_ticks - 1)   # 이번 틱 포함
            return 0.0
        return None

    def apply(self, control, target_speed: float, ap):
        """(control, target_speed) → 규칙 반영 후 (control, target_speed).

        ap 는 AutoPilot 인스턴스 (판단 컨텍스트: _waypoint_planner /
        _compute_target_speed_idm / _longitudinal_controller / _vehicle).
        d_end 는 정지 기준점 stop_s 까지 남은 planner route_s — ego.route_s 와
        같은 축이고, courseRespawn 후 reset_index() 재탐색을 그대로 따라간다.
        래치(latch_m/unlatch_m)·active_m 판정도 이 d_end(stop_s 기준)를 쓴다.
        """
        planner = ap._waypoint_planner
        if self.stop_s is None:
            self.stop_s = self._resolve_stop_s(planner)
        route_s = float(planner.route_s[planner.route_index])
        d_end = self.stop_s - route_s
        ego_speed = ap._vehicle.get_velocity().length()
        self.last_candidate = None
        self.last_d_end = d_end

        # 정적 장애물 회피 — 경로를 옆 차로로 밀면 PDM 의 선행차 판정에서 빠져
        # 다시 달린다. 충돌 판단은 PDM 의 OBB forecast 가 그대로 한다.
        self._try_overtake(ap, planner, ego_speed)

        # 방향지시등 — 속도 중재와 독립이다 (켜는 것이 감속을 만들지 않는다)
        (self.last_turn_signal, self.last_sig_src,
         self.last_sig_lead_s) = self._turn_signal(planner, route_s, ego_speed)

        # 래치 해제: 종점에서 다시 멀어졌다 = 리셋으로 뒤로 갔다 (고착 방지)
        if self.latched and d_end > self.unlatch_m:
            self.latched = False

        # 래치 진입: 종점 근처에서 사실상 정지 (latch_v 는 batch 완주 판정과 동일)
        if not self.latched and d_end <= self.latch_m and ego_speed < self.latch_v:
            self.latched = True

        candidate = None
        if self.latched:
            candidate = 0.0
        elif d_end <= self.active_m and target_speed > 0.1:
            # 종점의 유령 선행차 (정지, 길이 0). 유효거리는 앞범퍼 기준 —
            # IDM 이 net gap ≈ s0(stop_gap)에서 서므로 앞범퍼가 종점 − stop_gap.
            d_eff = max(0.1, d_end - self.front)
            candidate = float(ap._compute_target_speed_idm(
                desired_speed=target_speed,
                leading_actor_length=0.0,
                ego_speed=ego_speed,
                leading_actor_speed=0.0,
                distance_to_leading_actor=d_eff,
                s0=self.stop_gap,
                T=self.T,
            ))

        # 정지선 0.5 s 유지 홀드 — route_end 후보와 min 으로 합류
        hold = self._stopline_hold(planner, ego_speed)
        if hold is not None and (candidate is None or hold < candidate):
            candidate = hold

        if candidate is not None:
            self.last_candidate = candidate
            if candidate < target_speed:
                target_speed = candidate
                # 종방향 재계산 — 본류가 이번 틱 이미 호출했으므로 되감고 다시
                # (되감지 않으면 두 호출이 jerk 창을 나눠 갖는 핑퐁 — rewind_last 참고)
                hazard = target_speed < 1e-5
                ap._longitudinal_controller.rewind_last()
                accel, brake = ap._longitudinal_controller.get_throttle_and_brake(
                    hazard, target_speed, ego_speed)
                control.accel = accel
                control.throttle = accel
                control.brake = float(brake)

        self.last_target = float(target_speed)
        return control, target_speed
