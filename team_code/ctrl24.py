"""
ctrl24 — won_24 제어기 (2026-09-08).

kr_rules 를 **대체**하는 한국 대회 규칙 계층. 접점은 kr_rules 와 같다:
  · autopilot._get_control 끝의 `apply(control, target_speed, ap)` — min() 후보 덧대기
  · autopilot 적신호 IDM 의 `signal_release(ap, d)` — 유일한 오버라이드 훅
  · run_agent 가 읽는 last_* (turn_signal / sig_src / sig_lead_s / kr / avoid / ped / yellow / signal)
  + `pre_pass(ap, route_np, vehicles, target_speed, ego_speed)` — PDM 후보 계산 **앞**에서
    정적 장애물 회피 시프트를 만드는 훅 (커밋 3).

kr_rules 와의 차이 (설계 결정 2026-09-08, 사용자 승인):
  유지  P1 leading / P2 vehicle / P3 red_light (PDM 원문) · K1 stop_profile · K2 stop_hold ·
        K3 ped_intent(walkin·multi hold·release) · K4 emergency · K5 crosswalk(기본 off) ·
        황색 원샷 STOP/GO · 교차로 통과 가드 · 보행자 래치 · timeout GO · RTOR · 지시등
  제거  PDM bicycle·pedestrian 후보(run_agent 가 forecast_walkers 를 비운다) · route_end ·
        BREAKOUT · standoff(프로파일·크립·delay) · gap_v_req ·
        큐 판정·SUPPRESS · WAIT/WAIT_EXPIRED/REACTIVE · 정지 관찰 시계 전부 · span_extend ·
        preempt_latch · side 게이트 8개(no_neighbor/center_line/geom/zone/solid/occupied/
        kappa·lc_overlap/entry_block) · side_pick · gap_fit · shift_entry
  복원  K6 red_zone 붉은 구간 진입 전 감속 (2026-09-09 Q3 번복 — 근거는 _red_approach_profile)
        K7 shift_cap 시프트 전이 횡가속 상한 (2026-09-09 — geom 게이트 삭제의 대가를
        기각이 아니라 속도로 갚는다. 근거는 _shift_speed_cap)
  추가  K9 escape 탈출 바닥 (2026-09-10, 기본 off) — **무한정지 방지 규칙 하나**.
        경로 진행이 멈추고 정당한 정지 원인이 없으면 정당하지 않은 후보에만 바닥을 깐다.
        정당한 원인(신호·보행자·정지선 홀드)은 바닥을 안 받아 구조적으로 못 뚫는다.
        근거는 _escape_tick / _escape_floor.
  추가  K8 span_v_req 이웃 연속 창 (2026-09-09, B-30) — 시프트 목표 이웃이 span 중간에서
        끊기면 route.py 폴백이 경로에 계단을 만든다. **기각하지 않는다**: 장애물 지점에
        목표가 없으면 반대편 side 를 보고, 뒤가 끊기면 span 을 자르는 대신 창에 전이
        2회가 들어가는 속도를 min() 후보로 낸다 (근거는 _span_speed_req).
        side 게이트 부활이 아니다 — no_neighbor 는 그 side 를 **기각**했고, no_target 은
        반대편으로 **넘긴다**.
  변경  정적 장애물은 첫 틱에 PREEMPT (관찰·대기·예산 없음) · side 게이트는 span_too_far 하나 ·
        중첩 시프트 (활성 span 위에 같은 방향으로 한 칸 더, 횟수 제한 없음 — 요동 방지는
        "같은 (객체, 방향) 은 **그 span 동안 1회**", 2026-09-09 사양 확정) ·
        종점 정지 없이 계속 주행 (route.py 의 종점 패드) · P2 정지 객체는 pre_pass 시프트로 넘긴다

상수는 config/params.yaml 의 `ctrl24:` 섹션이 단일 출처다. 차량 제원·종방향 한계
(vehicle.* / control.a_dec_max / speed.a_emergency)·회랑 여유(percep.obstacle_clearance_m —
P1 의 정지 객체 폭 판정과 같은 값이어야 한다)·K6 의 speed.*·red_zone.*·K7 의
overtake.a_lat_max·shift_cap_min_v·shift_latest_m 은 그 섹션을 읽는다.
"""
from __future__ import annotations

import collections
import math as _math

import numpy as np

from vtd_adapter import frame

SIG_OFF, SIG_LEFT, SIG_RIGHT = 0, 1, 2        # 9910 turnSignal (SPEC §1.2)


def _turn_end_s(lg, lanes, cum, lens, ev) -> float:
    """회전 이벤트의 소등 지점 [route_s] — 같은 junction 차로가 이어지는 끝까지 (kr_rules 원문)."""
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


def turn_intervals(planner) -> list[dict]:
    """route['events'] 의 회전만 → 점등 구간 [{sig, src, ev_s, end_s}]. 시작 시 1회."""
    route = getattr(planner, 'route', None) or {}
    lg = getattr(planner, 'lg', None)
    lanes = route.get('lanes') or []
    cum = route.get('cum_s') or []
    lens = route.get('lengths') or []
    out: list[dict] = []
    for ev in route.get('events') or []:
        kind = str(ev.get('kind', ''))
        if kind.startswith('turn_'):
            out.append({'sig': SIG_LEFT if kind.endswith('left') else SIG_RIGHT,
                        'src': 'turn', 'ev_s': float(ev['s']),
                        'end_s': _turn_end_s(lg, lanes, cum, lens, ev)})
    out.sort(key=lambda d: d['ev_s'])
    return out


class Ctrl24:
    def __init__(self, cfg: dict) -> None:
        c = cfg['ctrl24']
        vh = cfg['vehicle']
        self.cfg = cfg
        self.c = c
        self.hz = float(cfg['comm']['send_hz'])
        self.front = float(vh['wheelbase']) + float(vh['front_overhang_m'])
        self.half_ego = float(vh['width']) / 2.0
        # ── K1·K2 정지선 (kr_rules ④′·B-1 확정 스펙 그대로) ────────────────
        self.stop_profile_a = float(c['stop_profile_a'])
        self.sl_hold_ticks = int(round(float(c['stopline_hold_s']) * self.hz))
        self.sl_min_ticks = int(round(float(c['stopline_hold_min_s']) * self.hz))
        self.sl_near_m = float(c['stopline_hold_near_m'])
        # ap.config 를 못 읽는 환경(목)에서만 쓰는 폴백 — 정상 경로는 PDM 주입값
        self.stop_gap_sl_fallback = float(cfg['speed']['stop_gap_stopline_m']) + self.front
        # 황색 원샷 판정·교차로 통과 가드
        self.a_yellow = float(c['a_yellow'])
        self.y_guard_max_m = float(c['yellow_guard_max_m'])
        # ── K3·K4·K5 보행자 ────────────────────────────────────────────────
        self.ped_intent_v = float(c['ped_intent_v'])
        self.ped_emg_ratio = float(c['ped_emergency_ratio'])
        self.ped_release_lat = float(c['ped_release_lat_m'])
        self.ped_release_ticks = int(round(float(c['ped_release_s']) * self.hz))
        self.ped_stop_v = float(c['ped_stop_v'])
        self.ped_offroad_lat = float(c['ped_offroad_lat_m'])
        self.ped_backstop_ticks = int(round(float(c['ped_backstop_s']) * self.hz))
        self.walkin_v = float(c['ped_walkin_v'])
        # 0.0 = 첫 틱 래치 (정지 관찰 시계가 없어 옛 ped_static 경로가 사라진 만큼 되돌린다).
        self.walkin_ticks = int(round(float(c['ped_walkin_s']) * self.hz))
        self.walkin_lat = float(c['ped_walkin_lat_m'])
        self.ped_multi = bool(c['ped_multi_enable'])
        self.ped_coast_ticks = int(round(float(c['ped_hold_coast_s']) * self.hz))
        self.cw_enable = bool(c['ped_crosswalk_creep_enable'])
        self.cw_zone_m = float(c['ped_crosswalk_zone_m'])
        self.cw_lat_m = float(c['ped_crosswalk_lat_m'])
        self.cw_wait_ticks = int(round(float(c['ped_crosswalk_wait_s']) * self.hz))
        self.cw_creep_v = float(c['ped_crosswalk_creep_v'])
        # ── K6 붉은 구간(보호구역) 진입 전 감속 — 2026-09-09 복원 (Q3 번복) ──
        # 상수는 **기존 speed.* / red_zone.* 를 그대로 읽는다** (값을 두 곳에 적지 않는다):
        #   speed.approach_decel_mps2 · speed.red_lookahead_m · speed.red_zone_target_kph,
        #   구간 이탈 여유는 lanegraph 의 red_span_cfg (red_zone.exit_margin_m).
        # ctrl24 는 켜고 끄는 스위치 하나(ctrl24.red_zone_enable)만 자기 섹션에 둔다.
        sp = cfg['speed']
        self.red_approach = bool(c['red_zone_enable'])
        self.red_a = float(sp['approach_decel_mps2'])
        self.red_look_m = float(sp['red_lookahead_m'])
        self.red_v_zone = float(sp['red_zone_target_kph']) / 3.6
        self.a_emergency = float(cfg['speed']['a_emergency'])
        self.a_dec_max = abs(float(cfg['control']['a_dec_max']))
        self.detect_max_m = float(c['detect_max_m'])
        self.v_static = float(c['blocker_speed_max'])
        self.clr = float(cfg['percep']['obstacle_clearance_m'])   # 회랑 여유 = P1 과 같은 값
        # ── 정적 장애물 회피 (A·B·C) ───────────────────────────────────────
        self.ot_enabled = bool(c['avoid_enable'])
        self.trans_min_m = float(c['trans_min_m'])
        self.trans_k = float(c['trans_k'])
        self.shift_ahead_m = float(c['shift_ahead_m'])
        self.ot_before_m = float(c['extra_before_m'])
        self.ot_after_m = float(c['extra_after_m'])
        self.chain_gap_m = float(c['chain_gap_m'])
        self.span_gate_max_m = float(c['span_gate_max_m'])
        self.noop_disp_m = float(c['noop_disp_m'])
        self.prepass_obb = bool(c['prepass_obb_enable'])
        # OBB 예측 후보 사전 필터 — 2 s 예측이 닿을 수 있는 종거리·횡거리 안의 정지 차량만.
        self.obb_reach_k = float(c['prepass_obb_reach_k'])
        self.obb_reach_extra_m = float(c['prepass_obb_reach_extra_m'])
        self.obb_lat_m = float(c['prepass_obb_lat_m'])
        # ── K7 시프트 전이 횡가속 상한 ─────────────────────────────────────
        # 상수는 overtake.* 를 그대로 읽는다 (K6 이 speed.* 를 읽는 것과 같은 규칙).
        # ctrl24 는 켜고 끄는 스위치 하나(ctrl24.shift_cap_enable)만 자기 섹션에 둔다.
        ot = cfg.get('overtake') or {}
        self.shift_cap_on = bool(c['shift_cap_enable'])
        self.a_lat_max = float(ot.get('a_lat_max', 0.0))
        self.shift_cap_min_v = float(ot.get('shift_cap_min_v', 1.0))
        # 미리보기 창의 바닥 [m] — 전이에 **닿기 전에** 감속이 시작돼야 한다.
        # 창의 속도항은 ctrl24 자신의 전이 계수(trans_k)를 쓴다 (전이 길이와 같은 축).
        self.shift_cap_look_m = float(ot.get('shift_latest_m', 25.0))
        # ── K8 span 이웃 연속성 (B-30) ─────────────────────────────────────
        self.span_cont = bool(c['span_v_req_enable'])
        self.span_v_min = float(c['span_v_req_min'])
        # 교차로 관통 연장 — 상수는 overtake.* 를 그대로 읽는다 (kr_rules 와 같은 값).
        self.span_jx = bool(c['span_junction_extend_enable'])
        self.span_jump_m = float(c['span_target_jump_m'])
        # ── K9 탈출 바닥 (무한정지 방지) ───────────────────────────────────
        self.esc_on_cfg = bool(c['escape_enable'])
        self.esc_win = max(1, int(round(float(c['escape_stuck_s']) * self.hz)))
        self.esc_prog_m = float(c['escape_progress_m'])
        self.esc_v = float(c['escape_v'])
        self.esc_release_m = float(c['escape_release_m'])
        self.esc_clear_m = float(c['escape_clear_m'])
        self.esc_rearm = bool(c['escape_rearm_shift_enable'])
        self.esc_virt = bool(c['escape_virtual_shift_enable'])
        self.esc_virt_extra = float(c['escape_virtual_extra_m'])
        self.esc_virt_v = float(c['escape_virtual_v'])
        self.esc_onc_m = float(c['escape_oncoming_clear_m'])
        self.jx_exit_m = float(ot.get('zone_exit_margin_m', 5.0))
        self.jx_gap_m = float(ot.get('zone_junction_gap_m', 5.0))
        self.jx_max_m = float(ot.get('zone_extend_max_m', 120.0))
        # ── 지시등 ─────────────────────────────────────────────────────────
        self.turn_lead_s = float(c['turn_lead_s'])
        self.lc_lead_s = float(c['lc_lead_s'])
        self.sig_lead_min_m = float(c['sig_lead_min_m'])
        self.lat_on_m = float(c['lat_shift_on_m'])
        self.sig_min_on_ticks = int(round(float(c['sig_min_on_s']) * self.hz))
        self.sig_off_delay_ticks = int(round(float(c['sig_off_delay_s']) * self.hz))
        # ── 미보고 신호 · timeout GO ────────────────────────────────────────
        self.sig_stale_ticks = int(round(float(c['signal_stale_s']) * self.hz))
        self.sig_timeout_go = bool(c['signal_timeout_go_enable'])
        self.sig_timeout_ticks = int(round(float(c['signal_unknown_timeout_s']) * self.hz))
        self.sig_timeout_clear_m = float(c['signal_timeout_clear_m'])
        # ── RTOR ───────────────────────────────────────────────────────────
        self.rtor_enable = bool(c['rtor_enable'])
        self.rtor_allow_stale = bool(c['rtor_allow_stale_red'])
        self.rtor_exclude = {int(x) for x in (c.get('rtor_exclude_tl_ids') or [])}
        self.rtor_turn_win_m = float(c['rtor_turn_event_window_m'])
        self.rtor_stop_v = float(c['rtor_stop_v_max'])
        self.rtor_zone_m = float(c['rtor_stop_zone_m'])
        self.rtor_hold_ticks = int(round(float(c['rtor_stop_hold_s']) * self.hz))
        self.rtor_ped_guard_m = float(c['rtor_ped_guard_m'])
        self.rtor_cross_gap_m = float(c['rtor_cross_gap_m'])
        self.rtor_cross_ttc_s = float(c['rtor_cross_ttc_s'])
        self.rtor_go_v = float(c['rtor_go_speed_kph']) / 3.6
        self.rtor_release_m = float(c['rtor_release_dist_m'])

        # ── 상태 ──────────────────────────────────────────────────────────
        self._ap = None
        self.last_target: float | None = None
        self.last_kr: dict = {}                       # 이번 틱 kr 후보 (reasons.kr)
        self.last_kr_winner: str | None = None        # kr 후보 중 최저 (run_agent 가 winner 로)
        self.last_stop_profile: float | None = None
        self.last_avoid: dict | None = None
        self.last_ped: dict | None = None
        self.last_yellow: dict | None = None
        self.last_signal: dict | None = None
        self.red_ivals: list | None = None            # 경로당 1회 캐시 [(진입 s, 이탈 s)]
        self.last_red_zone: dict | None = None        # K6 진단 (진입점·남은거리)
        self.ped_emergency = False
        self.last_turn_signal: int = SIG_OFF
        self.last_sig_src: str | None = None
        self.last_sig_lead_s: float | None = None
        # K2
        self.sl_hold_left = 0
        self.sl_stopped = False
        self.sl_stop_ticks = 0
        # 황색 래치 / 가드
        self.y_decision: str | None = None
        self.y_ctrl: int | None = None
        self.y_v_allow: float | None = None
        self.cross_guard = False
        self.cross_s: float | None = None
        self.cross_junction_seen = False
        # 보행자 (P4)
        self.ped_lat: dict = {}
        self.ped_intent: set = set()
        self.ped_clear: dict = {}
        self.ped_hold: dict = {}
        self.ped_diag: dict = {}
        self.ped_released: dict = {}
        self.ped_hold_ids: set = set()
        self.ped_miss: dict = {}
        self.ped_last: dict = {}
        self.ped_all: dict = {}
        self.ped_walkin: dict = {}
        self.cw_wait: dict = {}
        self._cw_zones: list | None = None
        self._sl_all: list | None = None
        # 훅 호환 — BREAKOUT 은 없다. autopilot 의 `if self.kr_rules.breakout_creep()` 은
        # kr_rules 와 파일을 공유하므로 남겨 두고 여기서는 항상 거짓이다 (오버라이드 아님).
        self._sig_go = False                          # 커밋 2 (timeout GO)
        self._rtor_go = False                         # 커밋 2 (RTOR)
        self.ot_span = None                           # 활성 시프트 인덱스 구간 (합집합)
        self.ot_side: str | None = None
        self.nested = 0                               # 활성 span 에 겹쳐 만든 시프트 수
        self._shifted_for: set = set()                # 이미 시프트를 만든 객체 id (요동 방지)
        self.last_span_plan: tuple | None = None
        self.span_v_req: float | None = None      # K8 후보 (감속하면 담기는 속도)
        # K9 — 진행 시계 하나와 래치 하나. 그 이상은 두지 않는다.
        self._esc_hist: collections.deque = collections.deque(maxlen=self.esc_win + 1)
        self._esc_engaged = False
        self._esc_mark: float | None = None       # 걸린 시점 주행거리 (래치 해제 기준)
        self._esc_odo = 0.0                       # 자차 이동거리 누적 [m]
        self._esc_xy: tuple | None = None         # 직전 틱 위치 (누적용)
        self.last_escape: dict | None = None      # 진단 (reasons.escape)
        self._esc_rearmed: list = []              # 이번 걸림에서 재무장한 객체 id
        self._virt_span: tuple | None = None      # 가상 시프트 구간 (속도 상한용)
        self._virt_wait = 0                       # 대향 대기 틱 (진단)
        self._nb_cache: dict = {}                 # (side, n_steps) → 이웃 연속성 bool 배열
        self.last_overtake: str | None = None
        self._prepass_done = False
        self.last_prepass_ms: float | None = None     # pre_pass 실행 시간 (틱 비용 보고용)
        self._obb_cache = None                        # (후보 id·route_index·span, 교차 id) — 정지 중 재사용
        self.last_obb_cached = False
        # 자차 OBB 예측 재사용 — pre_pass 와 PDM 이 **같은 입력**으로 두 번 돌지 않게.
        # 시프트를 적용하면 경로가 달라지므로 _shift_seq 를 올려 캐시를 무효화한다.
        self._fc_cache: tuple | None = None
        self._shift_seq = 0
        self.fc_reused = 0
        self._corridor: list = []                     # 이번 틱 회랑 안 정지 객체 (커밋 3 이 채운다)
        self._tick_lg = None
        self._tick_ego_lane = None
        # 지시등
        self.sig_plan: list | None = None
        self.sig_on_ticks = 0
        self.sig_off_left = 0
        self.sig_held: int = SIG_OFF
        # 미보고 신호 / timeout GO
        self._obs_tick = 0
        self._light_seen: dict = {}
        self._sig_wait_ticks = 0
        self._sig_go_tl = None
        # RTOR
        self._rtor_go_tl = None
        self._rtor_stop_s = None
        self._rtor_junction_seen = False
        self._rtor_hold_cnt = 0

    # ── 훅 호환 ───────────────────────────────────────────────────────────
    def breakout_creep(self) -> bool:
        return False

    # ── 공통 ──────────────────────────────────────────────────────────────
    def _s0(self, ap) -> float:
        """계획 정지점의 뒷축 gap — PDM 주입값이 단일 출처."""
        return float(getattr(getattr(ap, 'config', None), 'idm_red_light_minimum_distance',
                             self.stop_gap_sl_fallback))

    def _next_stopline(self, planner):
        """전방 신호 정지선 → (뒷축거리 [m], 상태명, 신호 id). 없으면 None."""
        dists = getattr(planner, 'distances_to_next_traffic_lights', None)
        tls = getattr(planner, 'next_traffic_lights', None)
        if dists is None or tls is None:
            return None
        tl = tls[planner.route_index]
        if tl is None:
            return None
        return (float(dists[planner.route_index]),
                getattr(getattr(tl, 'state', None), 'name', None),
                getattr(tl, 'id', None))

    def _project(self, planner, x_carla, y_carla):
        """CARLA 좌표 → (route_s 상대거리, 횡오프셋). 전방 창에서만 찾는다.
        경로점은 **현재** route_points 다 — 시프트가 얹히면 밀린 경로 기준이 된다."""
        pts = planner.route_points
        i0 = planner.route_index
        ppm = int(getattr(planner, 'points_per_meter', 10))
        hi = min(len(pts), i0 + int(self.detect_max_m * ppm))
        seg = pts[i0:hi, :2]
        if len(seg) < 2:
            return None
        q = np.array([x_carla, y_carla])
        j = int(np.argmin(np.linalg.norm(seg - q, axis=1)))
        tan = seg[min(j + 1, len(seg) - 1)] - seg[j]
        n = float(np.linalg.norm(tan))
        if n < 1e-9:
            return None
        tan = tan / n
        dv = q - seg[j]
        lat = float(tan[0] * dv[1] - tan[1] * dv[0])
        return (float(planner.route_s[i0 + j]) - float(planner.route_s[i0]), lat)

    def _all_stopline_s(self, planner) -> list:
        """경로상 모든 정지선 route_s (신호 유무 무관). 시작 시 1회. K5 의 폴백 구간."""
        if self._sl_all is not None:
            return self._sl_all
        out = []
        lg = getattr(planner, 'lg', None)
        route = getattr(planner, 'route', None) or {}
        if lg is not None:
            for i, k in enumerate(route.get('lanes') or []):
                rec = lg.lanes.get(tuple(k))
                if not rec:
                    continue
                for sl in rec.get('stop_lines', []):
                    out.append(float(route['cum_s'][i]) + float(sl['s']))
        self._sl_all = sorted(out)
        return self._sl_all

    # ── 황색 원샷 판정 / 교차로 통과 가드 / 해제 훅 (kr_rules 원문) ──────────
    def _yellow_latch(self, planner, ego_speed: float, ap) -> None:
        """접근당 1회 STOP/GO. v ≤ √(2·a_yellow·(d − s0)) → STOP, 그 외 GO. 번복 없음."""
        nxt = self._next_stopline(planner)
        if nxt is None:
            self._yellow_reset()
            return
        d_line, state, tl_id = nxt
        if self.y_ctrl is not None and tl_id != self.y_ctrl:
            self._yellow_reset()
        if state == 'Green':
            self._yellow_reset()
            return
        if self.a_yellow <= 0.0 or self.y_decision is not None or state != 'Yellow':
            return
        v_allow = _math.sqrt(2.0 * self.a_yellow * max(0.0, d_line - self._s0(ap)))
        self.y_decision = 'stop' if ego_speed <= v_allow else 'go'
        self.y_ctrl = tl_id
        self.y_v_allow = v_allow
        self.last_yellow = {'decision': self.y_decision, 'ctrl': tl_id,
                            'v': round(float(ego_speed), 2),
                            'v_allow': round(float(v_allow), 2),
                            'd_line': round(float(d_line), 2),
                            'a_judge': self.a_yellow}

    def _yellow_reset(self) -> None:
        self.y_decision = None
        self.y_ctrl = None
        self.y_v_allow = None

    def _cross_guard(self, planner, ap, d_line) -> bool:
        """앞범퍼가 정지선을 넘은 뒤 교차로를 벗어날 때까지 신호 정지 후보를 만들지 않는다.
        진입 미관측이면 yellow_guard_max_m 상한으로 푼다 (고착 방지)."""
        route_s = float(planner.route_s[planner.route_index])
        if not self.cross_guard and d_line is not None and (d_line - self.front) <= 0.0:
            self.cross_guard = True
            self.cross_s = route_s
            self.cross_junction_seen = False
        if not self.cross_guard:
            return False
        in_j = bool(getattr(ap, 'junction', False))
        if in_j:
            self.cross_junction_seen = True
        elif self.cross_junction_seen or (
                self.cross_s is not None
                and route_s - self.cross_s > self.y_guard_max_m):
            self.cross_guard = False
            self.cross_s = None
            self._yellow_reset()
            return False
        return True

    def signal_release(self, ap, _distance_to_traffic_light=None) -> bool:
        """PDM 적신호 IDM 을 이번 틱 건너뛸 것인가 — **유일한 오버라이드 훅**.
        황색 GO 래치 / 교차로 통과 가드 / timeout GO / RTOR 래치(붙은 신호 한정)."""
        return bool(self.y_decision == 'go' or self.cross_guard or self._sig_go
                    or self._rtor_active(getattr(ap, '_waypoint_planner', None)))


    # ── K1·K2 ─────────────────────────────────────────────────────────────
    def _stop_target(self, planner, ap) -> tuple | None:
        """정지 후보 대상 — timeout GO / RTOR 래치가 살아 있으면 None."""
        if self._sig_go or self._rtor_active(planner):
            return None
        return self._stop_target_raw(planner, ap)

    def _stop_target_raw(self, planner, ap) -> tuple | None:
        """색 해석의 단일 출처. 적색 / 황색+STOP → (뒷축거리, a). 가드 중이면 None."""
        nxt = self._next_stopline(planner)
        d_line = nxt[0] if nxt else None
        if self._cross_guard(planner, ap, d_line):
            return None
        if nxt is None:
            return None
        d_line, state, _tl_id = nxt
        if state == 'Red' and self.y_decision != 'go':
            return (d_line, self.stop_profile_a)
        if state == 'Yellow' and self.y_decision == 'stop':
            return (d_line, self.stop_profile_a)
        return None

    def _stopline_profile(self, planner, ap) -> float | None:
        """K1: v_allow = √(2·a·(d − s0)) — 단조 감소 상한. min() 후보."""
        tgt = self._stop_target(planner, ap)
        if tgt is None:
            return None
        d_line, a_eff = tgt
        if a_eff <= 0.0:
            return None
        return _math.sqrt(2.0 * a_eff * max(0.0, d_line - self._s0(ap)))

    def _stopline_hold(self, planner, ego_speed: float) -> float | None:
        """K2: 정지선 6 m 안 정지 시 1회 무장 stopline_hold_s. 녹색 전환·래치 해제 시
        최소 시간(stopline_hold_min_s) 미충족분만 남긴다 (B-1 확정 스펙 그대로)."""
        tgt = self._stop_target(planner, self._ap)
        stopped = ego_speed < 0.5
        near = tgt is not None and (tgt[0] - self.front) < self.sl_near_m
        if stopped:
            if near and not self.sl_stopped:
                self.sl_stopped = True
                self.sl_stop_ticks = 0
                self.sl_hold_left = self.sl_hold_ticks
        else:
            self.sl_stopped = False
            self.sl_stop_ticks = 0
        if tgt is None and self.sl_hold_left > 0:
            self.sl_hold_left = min(self.sl_hold_left,
                                    max(0, self.sl_min_ticks - self.sl_stop_ticks))
        if stopped:
            self.sl_stop_ticks += 1
        if self.sl_hold_left > 0:
            self.sl_hold_left -= 1
            return 0.0
        return None

    # ── K3·K5 보행자 (kr_rules 원문 — ped_static 경로만 없다) ──────────────
    @staticmethod
    def _walkers(ap) -> list:
        try:
            return list(ap._world.get_actors().filter('*walker*'))
        except Exception:                                  # noqa: BLE001
            return []

    def _ped_intent(self, planner, ap, ego_speed: float):
        """경로 쪽으로 걸어나오는 보행자의 정지 후보 → (v_allow, a_req, id) | None.

        정지 관찰 시계(obj_static_s)가 없으므로 래치 경로는 walkin 하나다 —
        연속 ped_walkin_s(기본 0.0 = 첫 틱) 동안 보행자 속도 ≥ ped_intent_v ∧ 경로 쪽
        횡속도 ≥ ped_walkin_v ∧ |lat| < ped_walkin_lat_m. 회랑(|lat| < ped_release_lat_m)
        안은 홀드 래치(v_allow 0). 해제 clear/backstop/coast 는 A-1·P4-M 그대로.
        """
        if self.ped_intent_v <= 0.0 or self.stop_profile_a <= 0.0:
            return None
        walkers = self._walkers(ap)
        live = set()
        best = None
        self.ped_diag = {}
        self.ped_released = {}
        self.ped_all = {}
        multi = self.ped_multi and self.ped_release_lat > 0.0
        s0 = float(getattr(getattr(ap, 'config', None),
                           'idm_pedestrian_minimum_distance', 4.0))
        for w in walkers:
            wid = getattr(w, 'id', None)
            if wid is None:
                continue
            live.add(wid)
            loc = w.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is None:
                self.ped_lat.pop(wid, None)
                self.ped_clear.pop(wid, None)
                if multi and wid in self.ped_hold_ids and wid in self.ped_last:
                    self.ped_miss[wid] = self.ped_miss.get(wid, 0) + 1
                    if self.ped_miss[wid] <= self.ped_coast_ticks:
                        v_allow, a_req = self.ped_last[wid]
                        self.ped_all[wid] = {'v_allow': round(float(v_allow), 2),
                                             'latched': wid in self.ped_intent,
                                             'hold': True, 'coast': self.ped_miss[wid],
                                             'unprojectable': True}
                        if best is None or v_allow < best[0]:
                            best = (v_allow, a_req, wid)
                    else:
                        self._ped_unlatch(wid)
                continue
            s_rel, lat = pr
            prev = self.ped_lat.get(wid)
            self.ped_lat[wid] = lat
            v_toward = None if prev is None else (abs(prev) - abs(lat)) * self.hz
            if not (0.0 < s_rel <= self.detect_max_m):
                self._ped_unlatch(wid)
                continue
            self.ped_miss[wid] = 0
            if (multi and abs(lat) < self.ped_release_lat
                    and wid not in self.ped_hold_ids):
                self.ped_hold_ids.add(wid)
                self.ped_clear.setdefault(wid, 0)
                self.ped_hold.setdefault(wid, 0)
            if wid not in self.ped_intent:
                latch_ok = v_toward is not None and self._ped_walkin(
                    wid, float(getattr(w, 'speed', 0.0)), v_toward, lat)
                if latch_ok:
                    self.ped_intent.add(wid)
                    self.ped_clear[wid] = 0
                    self.ped_hold[wid] = 0
                    self.ped_walkin.pop(wid, None)
                elif wid not in self.ped_hold_ids:
                    if multi:
                        self.ped_all[wid] = {'s_rel': round(float(s_rel), 2),
                                             'lat': round(float(lat), 2),
                                             'latched': False, 'hold': False}
                    continue
            why = self._ped_release_tick(wid, w, lat, v_toward)
            self.ped_diag[wid] = {
                'lat': round(float(lat), 2),
                'v_toward': None if v_toward is None else round(float(v_toward), 2),
                'clear_s': round(self.ped_clear.get(wid, 0) / self.hz, 1),
                'hold_s': round(self.ped_hold.get(wid, 0) / self.hz, 1)}
            if why is not None:
                self._ped_unlatch(wid)
                self.ped_released[wid] = why
                continue
            d_eff = s_rel - self.front - s0
            hold = multi and wid in self.ped_hold_ids
            v_allow = 0.0 if hold else _math.sqrt(2.0 * self.stop_profile_a * max(0.0, d_eff))
            a_req = ego_speed * ego_speed / (2.0 * max(d_eff, 0.5))
            if multi:
                self.ped_last[wid] = (v_allow, a_req)
                self.ped_all[wid] = {'s_rel': round(float(s_rel), 2),
                                     'lat': round(float(lat), 2),
                                     'v_allow': round(float(v_allow), 2),
                                     'latched': wid in self.ped_intent, 'hold': bool(hold)}
            if best is None or v_allow < best[0]:
                best = (v_allow, a_req, wid)
        for wid in list(self.ped_intent | self.ped_hold_ids):
            if wid in live:
                continue
            if multi and self.ped_coast_ticks > 0 and wid in self.ped_last:
                self.ped_miss[wid] = self.ped_miss.get(wid, 0) + 1
                if self.ped_miss[wid] <= self.ped_coast_ticks:
                    v_allow, a_req = self.ped_last[wid]
                    self.ped_all[wid] = {'v_allow': round(float(v_allow), 2),
                                         'latched': wid in self.ped_intent,
                                         'hold': wid in self.ped_hold_ids,
                                         'coast': self.ped_miss[wid]}
                    if best is None or v_allow < best[0]:
                        best = (v_allow, a_req, wid)
                    continue
            self._ped_unlatch(wid)
        for d in (self.ped_lat, self.ped_walkin):
            for wid in list(d):
                if wid not in live:
                    d.pop(wid, None)
        return best

    def _ped_walkin(self, wid, w_speed: float, v_toward: float, lat: float) -> bool:
        """래치 조건 (A-4 walkin). ped_walkin_s 0 이면 조건 성립 첫 틱에 래치.
        한 틱이라도 깨지면 처음부터."""
        ok = (w_speed >= self.ped_intent_v and v_toward >= self.walkin_v
              and abs(lat) < self.walkin_lat)
        if not ok:
            self.ped_walkin[wid] = 0
            return False
        self.ped_walkin[wid] = self.ped_walkin.get(wid, 0) + 1
        return self.ped_walkin[wid] >= max(1, self.walkin_ticks)

    def _ped_unlatch(self, wid) -> None:
        self.ped_intent.discard(wid)
        self.ped_hold_ids.discard(wid)
        self.ped_clear.pop(wid, None)
        self.ped_hold.pop(wid, None)
        self.ped_miss.pop(wid, None)
        self.ped_last.pop(wid, None)

    def _ped_release_tick(self, wid, w, lat: float, v_toward) -> str | None:
        """A-1 위치 기반 해제: clear(회랑 밖 ∧ 멀어짐 ∧ 멈춤/차도 밖 연속) / backstop."""
        if self.ped_release_lat <= 0.0:
            return None
        outside = abs(lat) > self.ped_release_lat
        self.ped_hold[wid] = self.ped_hold.get(wid, 0) + 1
        away = v_toward is not None and v_toward <= 0.0
        settled = (float(getattr(w, 'speed', 0.0)) < self.ped_stop_v
                   or abs(lat) > self.ped_offroad_lat)
        if outside and away and settled:
            self.ped_clear[wid] = self.ped_clear.get(wid, 0) + 1
        else:
            self.ped_clear[wid] = 0
        if self.ped_clear[wid] >= self.ped_release_ticks:
            return 'clear'
        if outside and self.ped_hold[wid] >= self.ped_backstop_ticks:
            return 'backstop'
        return None

    def _crosswalk_zones(self, planner) -> list:
        if self._cw_zones is not None:
            return self._cw_zones
        out = []
        lg = getattr(planner, 'lg', None)
        route = getattr(planner, 'route', None) or {}
        if lg is not None:
            for i, k in enumerate(route.get('lanes') or []):
                rec = lg.lanes.get(tuple(k))
                if not rec:
                    continue
                base = float(route['cum_s'][i])
                for a, b, _kind in rec.get('crosswalks', []):
                    out.append((base + float(a), base + float(b)))
        if not out:
            out = [(s, s) for s in self._all_stopline_s(planner)]
        self._cw_zones = sorted(out)
        return self._cw_zones

    def _ped_crosswalk(self, planner, ap, ego_speed: float):
        """K5 (A-3, 기본 off): 횡단보도 앞에 서 있는 보행자 — 정지 프로파일 → 대기 → creep."""
        if not self.cw_enable or self.stop_profile_a <= 0.0:
            return None
        walkers = self._walkers(ap)
        zones = self._crosswalk_zones(planner)
        if not walkers or not zones:
            self.cw_wait.clear()
            return None
        route_s = float(planner.route_s[planner.route_index])
        s0 = float(getattr(getattr(ap, 'config', None),
                           'idm_pedestrian_minimum_distance', 4.0))
        best, live = None, set()
        for w in walkers:
            wid = getattr(w, 'id', None)
            if wid is None or wid in self.ped_intent:
                continue
            if float(getattr(w, 'speed', 0.0)) >= self.ped_stop_v:
                continue
            loc = w.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is None:
                continue
            s_rel, lat = pr
            if not (0.0 < s_rel <= self.detect_max_m):
                continue
            if abs(lat) < self.ped_release_lat or abs(lat) >= self.cw_lat_m:
                continue
            ps = route_s + s_rel
            if not any(a - self.cw_zone_m <= ps <= b + self.cw_zone_m for a, b in zones):
                continue
            live.add(wid)
            d_eff = s_rel - self.front - s0
            v_prof = _math.sqrt(2.0 * self.stop_profile_a * max(0.0, d_eff))
            ticks = self.cw_wait.get(wid, 0)
            if (ticks < self.cw_wait_ticks and ego_speed < self.ped_stop_v
                    and d_eff <= self.cw_zone_m):
                ticks += 1
            self.cw_wait[wid] = ticks
            if ticks >= self.cw_wait_ticks:
                v_allow, phase = self.cw_creep_v, 'creep'
            else:
                v_allow, phase = v_prof, 'wait'
            if best is None or v_allow < best[0]:
                best = (v_allow, {'id': int(wid), 'phase': phase,
                                  'wait_s': round(ticks / self.hz, 1),
                                  'lat': round(float(lat), 2), 's_rel': round(float(s_rel), 1),
                                  'v_allow': round(float(v_allow), 2)})
        for wid in list(self.cw_wait):
            if wid not in live:
                self.cw_wait.pop(wid, None)
        return best

    # ── 자차 차로 ─────────────────────────────────────────────────────────
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

    def _in_junction_lane(self, ap) -> bool:
        lg, lane = self._tick_lg, self._tick_ego_lane
        if lg is not None and lane is not None and lane in lg.lanes:
            return lg.lanes[lane]['junction'] != -1
        return bool(getattr(ap, 'junction', False))

    # ── 회랑 (정지 객체, 관찰 시계 없음) ──────────────────────────────────
    @staticmethod
    def _is_walker(actor) -> bool:
        return 'walker' in str(getattr(actor, 'type_id', ''))

    def _corridor_blockers(self, ap, planner, include_moving: bool = False) -> list:
        """전방 detect_max_m 안에서 주행 회랑을 침범한 객체 [(s_rel, lat, half_w, actor)].

        정지 = GT 속도 < blocker_speed_max, 이번 틱 값 그대로 (정지 지속 시간 조건 없음).
        침범 = |lat| < 자차반폭 + 객체반폭 + percep.obstacle_clearance_m (P1 과 같은 축).
        경로점은 현재 route_points 라 시프트 활성 중에는 밀린 경로 회랑이다.
        보행자(walker)는 제외 — 그건 K3 의 정지 대상이지 비켜갈 대상이 아니다
        (항목 10: 횡단 완료 전 통과 = 중대). include_moving 은 timeout GO 의 이동 차량 검사용.
        """
        try:
            actors = list(ap._world.get_actors())
        except Exception:                                  # noqa: BLE001
            return []
        ego_id = ap._vehicle.id
        out = []
        for a in actors:
            if a.id == ego_id or self._is_walker(a):
                continue
            if not include_moving and float(getattr(a, 'speed', 0.0)) >= self.v_static:
                continue
            loc = a.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is None:
                continue
            s_rel, lat = pr
            if not (0.5 < s_rel <= self.detect_max_m):
                continue
            bb = getattr(a, 'bounding_box', None)
            hw = float(bb.extent.y) if bb is not None else 0.9
            if abs(lat) < self.half_ego + hw + self.clr:
                out.append((s_rel, lat, hw, a))
        out.sort(key=lambda z: z[0])
        return out

    # ── 미보고 신호 · timeout GO ───────────────────────────────────────────
    def observe_lights(self, lights) -> None:
        """9910 lights [(id, state)] — 보고 시각만 센다 (state 는 플래너가 갱신)."""
        self._obs_tick += 1
        for lid, _state in lights or []:
            self._light_seen[int(lid)] = self._obs_tick

    def _signal_stale(self, planner) -> dict | None:
        """다음 정지선 controller 의 미보고 판정 → 진단 dict. 관측이 없거나 대상 없음 → None."""
        if not (self.sig_timeout_go or self.rtor_enable) or self._obs_tick <= 0:
            return None
        tls = getattr(planner, 'next_traffic_lights', None)
        tl = tls[planner.route_index] if tls is not None else None
        if tl is None:
            return None
        ids = [int(i) for i in (getattr(tl, 'controller_ids', None) or [getattr(tl, 'id', -1)])]
        seen = max((self._light_seen.get(i, 0) for i in ids), default=0)
        age = self._obs_tick - seen
        return {'signal_stale': age >= self.sig_stale_ticks,
                'signal_stale_s': round(age / self.hz, 1),
                'signal_last_state': getattr(getattr(tl, 'state', None), 'name', None),
                'signal_ctrl': ids}

    def _signal_timeout_tick(self, ap, planner, ego_speed: float) -> None:
        """B-3(b) 시한 출발 — stale ∧ 정지 후보 있음 ∧ 정지 중 ∧ 앞차 없음(회랑 정지 객체 0 ∧
        clear_m 안 이동 차량 0) ∧ 보행자 래치 없음 이 signal_unknown_timeout_s 이상.
        PDM walker 플래그는 없다 (PDM 보행자 후보 제거) — 회랑 홀드·의도 래치가 그 축이다."""
        if not self.sig_timeout_go:
            return
        sig = self.last_signal
        nxt = self._next_stopline(planner)
        tl_id = nxt[2] if nxt else None
        stale = bool(sig and sig['signal_stale'])
        if self._sig_go and (not stale or tl_id != self._sig_go_tl):
            self._sig_go = False
            self._sig_go_tl = None
        ok = stale and self._stop_target_raw(planner, ap) is not None \
            and ego_speed < 0.5 and not self._corridor \
            and not (self.ped_intent or self.ped_hold_ids)
        if ok:
            moving = self._corridor_blockers(ap, planner, include_moving=True)
            ok = not any(b[0] <= self.sig_timeout_clear_m for b in moving)
        self._sig_wait_ticks = self._sig_wait_ticks + 1 if ok else 0
        if ok and self._sig_wait_ticks >= self.sig_timeout_ticks and not self._sig_go:
            self._sig_go = True
            self._sig_go_tl = tl_id
        if sig is not None:
            sig['timeout_s'] = round(self._sig_wait_ticks / self.hz, 1)
            sig['timeout_go'] = self._sig_go

    # ── RTOR (kr_rules 원문 — 정지 유지 rtor_stop_hold_s 0.5) ─────────────
    def _rtor_reset(self) -> None:
        self._rtor_go = False
        self._rtor_go_tl = None
        self._rtor_stop_s = None
        self._rtor_junction_seen = False
        self._rtor_hold_cnt = 0

    def _rtor_active(self, planner) -> bool:
        """래치가 지금 전방 신호에 붙어 있는가 — _stop_target·signal_release 의 게이트."""
        if not self._rtor_go or planner is None:
            return False
        nxt = self._next_stopline(planner)
        return nxt is not None and nxt[2] == self._rtor_go_tl

    def _rtor_sig(self, planner) -> tuple:
        tls = getattr(planner, 'next_traffic_lights', None)
        tl = tls[planner.route_index] if tls is not None else None
        if tl is None:
            return None, []
        ids = [int(i) for i in (getattr(tl, 'controller_ids', None) or [getattr(tl, 'id', -1)])]
        if self._obs_tick <= 0:
            return 'fresh', ids
        seen = max((self._light_seen.get(i, 0) for i in ids), default=0)
        return ('stale' if self._obs_tick - seen >= self.sig_stale_ticks else 'fresh'), ids

    def _rtor_lead(self, planner, ap, d_line: float):
        """정지선 앞 앞차 → 사유, 없으면 None. 회랑 정지 객체 또는 PDM 선행차(이동 포함)."""
        for b in (self._corridor or []):
            if b[0] < d_line:
                return f'corridor:{int(getattr(b[3], "id", -1))}'
        try:
            vehicles = list(ap._world.get_actors().filter('*vehicle*'))
            ids = set(planner.compute_leading_vehicles(vehicles, ap._vehicle.id))
        except Exception:                                  # noqa: BLE001 — 목 플래너
            return None
        for a in vehicles:
            if a.id not in ids:
                continue
            loc = a.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is not None and 0.0 < pr[0] < d_line:
                return f'lead:{int(a.id)}'
        return None

    def _rtor_ped_block(self, planner, ap, route_s: float, stop_s: float, end_s: float):
        """보행자 차단 사유 — 회랑 홀드·의도 래치, 또는 정지선~회전 종료 주변 guard_m 안 보행자."""
        if self.ped_hold_ids or self.ped_intent:
            return 'latch'
        g = self.rtor_ped_guard_m
        s_lo, s_hi = stop_s - route_s - g, end_s - route_s + g
        for w in self._walkers(ap):
            loc = w.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is not None and s_lo <= pr[0] <= s_hi and abs(pr[1]) <= g:
                return f'near:{int(getattr(w, "id", -1))}'
        return None

    def _rtor_cross_block(self, ap, lg, ego_lane, jid):
        """교차 차량 차단 (kr_rules 원문). 정지 차량 제외, 좌측·전방 접근만."""
        try:
            vehicles = list(ap._world.get_actors().filter('*vehicle*'))
        except Exception:                                  # noqa: BLE001
            return None
        ego = ap._vehicle
        eloc = ego.get_location()
        ex, ey = frame.from_carla_xy(eloc.x, eloc.y)
        eyaw = frame.from_carla_yaw_deg(ego.get_transform().rotation.yaw)
        ce, se = _math.cos(eyaw), _math.sin(eyaw)
        for a in vehicles:
            if a.id == ego.id:
                continue
            v = float(getattr(a, 'speed', 0.0))
            if v < self.ped_stop_v:
                continue
            loc = a.get_location()
            ax, ay = frame.from_carla_xy(loc.x, loc.y)
            ayaw = frame.from_carla_yaw_deg(a.get_transform().rotation.yaw)
            cand = None
            if lg is not None and jid is not None:
                try:
                    m = lg.locate(ax, ay, yaw=ayaw)
                except Exception:                          # noqa: BLE001
                    m = None
                if m is not None and m.lane in lg.lanes:
                    key = m.lane
                    in_j = (lg.lanes[key]['junction'] == jid
                            or any(lg.lanes[s]['junction'] == jid
                                   for s in lg.successors(key) if s in lg.lanes))
                    same_road = ego_lane is not None and key[0] == ego_lane[0]
                    cand = in_j and not same_road
            if cand is False:
                continue
            dx, dy = ax - ex, ay - ey
            lon, lat = dx * ce + dy * se, -dx * se + dy * ce
            if not (lat > 0.0 or lon > 0.0):
                continue
            if (-dx) * _math.cos(ayaw) + (-dy) * _math.sin(ayaw) <= 0.0:
                continue
            dist = _math.hypot(dx, dy)
            if dist < self.rtor_cross_gap_m or dist / max(v, 0.1) < self.rtor_cross_ttc_s:
                return f'{int(a.id)}'
        return None

    def _rtor_log(self, diag: dict) -> None:
        if self.last_signal is None:
            self.last_signal = {}
        self.last_signal['rtor'] = diag

    def _rtor_tick(self, ap, planner, ego_speed: float) -> None:
        """off → hold(정지 구역 안 정지 누적 rtor_stop_hold_s) → wait(보행자·교차) → go(래치).
        리셋 = fresh 녹색 / 교차로 차로 이탈 / 정지선 + rtor_release_dist_m."""
        if not self.rtor_enable:
            return
        route_s = float(planner.route_s[planner.route_index])
        lg, ego_lane = self._tick_lg, self._tick_ego_lane
        nxt = self._next_stopline(planner)
        sig, ids = self._rtor_sig(planner)
        diag = {'state': 'off', 'sig': sig, 'lead': None, 'turn_right': None,
                'hold_s': 0.0, 'ped_block': None, 'cross_block': None, 'reason': None}
        in_j = self._in_junction_lane(ap)
        if self._rtor_go:
            if in_j:
                self._rtor_junction_seen = True
            why = None
            if (nxt is not None and nxt[2] == self._rtor_go_tl
                    and nxt[1] == 'Green' and sig == 'fresh'):
                why = 'green'
            elif self._rtor_junction_seen and not in_j:
                why = 'junction_exit'
            elif route_s >= float(self._rtor_stop_s) + self.rtor_release_m:
                why = 'release_dist'
            if why is not None:
                self._rtor_reset()
                diag['reason'] = 'reset:' + why
            else:
                diag['state'] = 'go'
                diag['hold_s'] = round(self._rtor_hold_cnt / self.hz, 1)
            self._rtor_log(diag)
            return
        tl_id = nxt[2] if nxt else None
        state = nxt[1] if nxt else None
        reason = None
        if nxt is None:
            reason = 'no_signal'
        elif tl_id in self.rtor_exclude or any(i in self.rtor_exclude for i in ids):
            reason = 'excluded'
        else:
            tgt = self._stop_target_raw(planner, ap)
            red = (tgt is not None and state == 'Red'
                   and (sig == 'fresh' or (self.rtor_allow_stale and sig == 'stale')))
            if not red:
                reason = 'not_red'
        d_line = float(nxt[0]) if nxt else None
        stop_s = route_s + d_line if d_line is not None else None
        turn = None
        if reason is None:
            if self.sig_plan is None:
                self.sig_plan = turn_intervals(planner)
            turn = next((iv for iv in self.sig_plan if iv['sig'] == SIG_RIGHT
                         and stop_s - 0.5 <= iv['ev_s'] <= stop_s + self.rtor_turn_win_m), None)
            diag['turn_right'] = turn is not None
            if turn is None:
                reason = 'no_turn_right'
        if reason is None:
            lead = self._rtor_lead(planner, ap, d_line)
            diag['lead'] = lead
            if lead is not None:
                reason = 'lead'
        if reason is not None:
            self._rtor_hold_cnt = 0
            diag['reason'] = reason
            self._rtor_log(diag)
            return
        zone = (d_line - self.front) <= self.rtor_zone_m and ego_speed <= self.rtor_stop_v
        self._rtor_hold_cnt = self._rtor_hold_cnt + 1 if zone else 0
        diag['hold_s'] = round(self._rtor_hold_cnt / self.hz, 1)
        route = getattr(planner, 'route', None) or {}
        jid = next((ev.get('junction') for ev in (route.get('events') or [])
                    if str(ev.get('kind', '')) == 'turn_right'
                    and abs(float(ev.get('s', -1e9)) - turn['ev_s']) < 1e-6), None)
        ped = self._rtor_ped_block(planner, ap, route_s, stop_s, float(turn['end_s']))
        cross = self._rtor_cross_block(ap, lg, ego_lane, jid)
        diag['ped_block'] = ped
        diag['cross_block'] = cross
        if self._rtor_hold_cnt < self.rtor_hold_ticks:
            diag['state'] = 'hold'
            diag['reason'] = 'zone' if zone else 'out_of_zone'
        elif ped is not None or cross is not None:
            diag['state'] = 'wait'
            diag['reason'] = 'ped' if ped is not None else 'cross'
        else:
            self._rtor_go = True
            self._rtor_go_tl = tl_id
            self._rtor_stop_s = stop_s
            self._rtor_junction_seen = False
            diag['state'] = 'go'
            diag['reason'] = 'latch'
        self._rtor_log(diag)

    # ── 지시등 (kr_rules 원문) ─────────────────────────────────────────────
    def _lane_shift(self, planner, ego_speed: float):
        """앞 창에서 경로가 차로 중심 기준으로 옆으로 갈 예정인가 → (sig, 남은거리).
        planner.lat_shift 는 계획 차선변경 + 런타임 회피 시프트를 함께 담는다."""
        lat = getattr(planner, 'lat_shift', None)
        if lat is None or len(lat) == 0:
            return None
        i = int(getattr(planner, 'route_index', 0))
        if i >= len(lat):
            return None
        ppm = float(getattr(planner, 'points_per_meter', 10))
        look = max(ego_speed * self.lc_lead_s, self.sig_lead_min_m)
        j = min(len(lat), i + int(look * ppm) + 1)
        seg = lat[i:j] - lat[i]
        if seg.size == 0:
            return None
        k = int(np.argmax(np.abs(seg)))
        if abs(float(seg[k])) < self.lat_on_m:
            return None
        over = np.nonzero(np.abs(seg) >= self.lat_on_m)[0]
        remain = float(over[0]) / ppm if over.size else 0.0
        return (SIG_LEFT if seg[k] > 0 else SIG_RIGHT), remain

    def _turn_signal(self, planner, route_s: float, ego_speed: float) -> tuple:
        """이번 틱 지시등 → (sig, src, lead_s). 회전(이벤트) 대 차로 이동(기하) —
        남은거리 짧은 쪽, 동률 회전 우선. 유지는 min_on / off_delay 둘뿐 (상한 있음)."""
        if self.sig_plan is None:
            self.sig_plan = turn_intervals(planner)
        best = None
        for iv in self.sig_plan:
            if route_s > iv['end_s']:
                continue
            remain = iv['ev_s'] - route_s
            if remain > max(ego_speed * self.turn_lead_s, self.sig_lead_min_m):
                continue
            key = (max(0.0, remain), 0)
            if best is None or key < best[0]:
                best = (key, iv['sig'], 'turn', remain)
        shift = self._lane_shift(planner, ego_speed)
        if shift is not None:
            sig, remain = shift
            key = (max(0.0, remain), 1)
            if best is None or key < best[0]:
                best = (key, sig, 'lc', remain)
        if best is None:
            sig, src, remain = SIG_OFF, None, None
        else:
            _k, sig, src, remain = best
        if sig != SIG_OFF:
            if self.sig_held != sig:
                self.sig_on_ticks = 0
            self.sig_held = sig
            self.sig_on_ticks += 1
            self.sig_off_left = self.sig_off_delay_ticks
        elif self.sig_held != SIG_OFF and (self.sig_off_left > 0
                                           or self.sig_on_ticks < self.sig_min_on_ticks):
            self.sig_off_left = max(0, self.sig_off_left - 1)
            self.sig_on_ticks += 1
            sig, src = self.sig_held, 'hold'
        else:
            self.sig_held = SIG_OFF
            self.sig_on_ticks = 0
        lead = (max(0.0, remain) / ego_speed
                if remain is not None and ego_speed > 0.1 else None)
        return sig, src, lead

    # ── 정적 장애물 회피 — A(첫 틱 PREEMPT) · B(게이트 1개) · C(중첩) ─────────
    def _trans_m(self, ego_speed: float) -> float:
        """전이 길이 = max(trans_min_m, trans_k·v)."""
        return max(self.trans_min_m, self.trans_k * max(float(ego_speed), 0.0))

    def _chain(self, corridor, actor) -> dict:
        """actor 에서 시작하는 연쇄 (chain_gap_m 안에 이어지는 정지 객체) — kr_rules B-9."""
        out = {'first': actor, 'last': actor, 'ids': [actor.id], 'extent_m': 0.0}
        if self.chain_gap_m <= 0.0 or not corridor:
            return out
        idx = next((i for i, c in enumerate(corridor) if c[3].id == actor.id), None)
        if idx is None:
            return out
        s_first = s_prev = corridor[idx][0]
        for s_rel, _lat, _hw, a in corridor[idx + 1:]:
            if s_rel - s_prev > self.chain_gap_m:
                break
            out['ids'].append(a.id)
            out['last'] = a
            s_prev = s_rel
        out['extent_m'] = s_prev - s_first
        return out

    def _restore_span(self, planner) -> None:
        """지나간 시프트 span 원복 — 원 경로로. 요동 방지 집합도 비운다."""
        a, b = self.ot_span
        planner.route_points[a:b] = planner.original_route_points[a:b]
        if getattr(planner, 'commands_orig', None) is not None:
            planner.commands[a:b] = planner.commands_orig[a:b]
        if getattr(planner, '_lat_build', None) is not None:
            planner.lat_shift[a:b] = planner._lat_build[a:b]
        self._rebuild_kd(planner)
        self._shift_seq += 1                           # 원복도 경로 변경이다
        self.ot_span = None
        self.ot_side = None
        self.nested = 0
        self._shifted_for.clear()
        self.last_overtake = 'restored'

    @staticmethod
    def _rebuild_kd(planner) -> None:
        if getattr(planner, '_kd', None) is not None:
            from scipy.spatial import cKDTree
            planner._kd = cKDTree(planner.route_points[:, :2])

    def pre_pass(self, ap, route_np, vehicles, target_speed: float, ego_speed: float) -> None:
        """PDM 후보 계산 **앞**의 훅 (autopilot._get_control). 트리거 = 회랑 정지 객체 ∪
        원 경로 기준 OBB 2 s 예측이 교차하는 정지 객체. 시프트가 성립하면 route_points 가
        밀려 P1/P2 가 밀린 경로 기준으로 계산된다 — P2 본문은 무수정, 불성립이면 P2 가
        기존대로 0 을 낸다 (정지 객체인데 못 비켰으면 선다)."""
        import time
        t0 = time.perf_counter()
        planner = ap._waypoint_planner
        obb_ids = None
        self.last_obb_cached = False
        # 이번 틱 회피 진단은 여기서 **한 번만** 비운다. apply 의 _tick_head 는 훅이 돈
        # 틱에는 건드리지 않는다 — 두 곳에서 비우면 훅이 만든 진단이 지워지고(7310077),
        # 아무 데서도 안 비우면 직전 틱 진단이 남아 시프트가 매 틱 재기록된다
        # (2026-09-09 리플레이: 1,410틱 로그에 shift 473 — 실제 생성은 그보다 훨씬 적다).
        self.last_avoid = None
        self.last_overtake = None
        if self.ot_enabled and self.prepass_obb:
            obb_ids = self._obb_static_ids(ap, route_np, vehicles, target_speed, ego_speed)
        self._corridor = self._corridor_blockers(ap, planner)
        self._avoid_tick(ap, planner, ego_speed, obb_ids)
        self._prepass_done = True
        self.last_prepass_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    def fc_key(self, ap, ego_speed, target_speed, n) -> tuple:
        """자차 OBB 예측의 입력 동일성 키. 하나라도 다르면 캐시 미스(= 다시 계산)다.
        경로 동일성은 (route_index, _shift_seq) 로 본다 — 시프트·원복이 seq 를 올린다."""
        return (int(n), repr(float(ego_speed)), repr(float(target_speed)),
                int(ap._waypoint_planner.route_index), self._shift_seq)

    def _obb_static_ids(self, ap, route_np, vehicles, target_speed, ego_speed) -> set:
        """원 경로 기준 자차 OBB 예측(2 s, 차선변경 근처 1.1 s)과 교차하는 **정지** 차량 id.
        PDM 의 forecast_ego_agent / predict_other_actors_bounding_boxes / check_obb_intersection
        을 그대로 부른다 — 정지 객체만 넘기므로 forecast 는 등속(0) 상자다."""
        try:
            cfg = ap.config
            ego = ap._vehicle
            loc = ego.get_location()
            # 예측 도달 범위 밖의 정지 차량은 뺀다 — 2 s 등속 예측이 닿을 수 없는 객체는
            # 교차할 수 없으므로 판정은 같고 비용만 준다. 실측 2026-09-08 교통류_04: 정지
            # 차량이 50 m 안에 늘 있어 매 틱 forecast → prepass 중앙값 34 ms / p95 355 ms.
            reach = (max(float(ego_speed), 1.0) * float(cfg.default_forecast_length)
                     * self.obb_reach_k + 2.0 * float(self.cfg['vehicle']['length'])
                     + self.obb_reach_extra_m)
            stopped = []
            for v in vehicles:
                if v.id == ego.id or float(getattr(v, 'speed', 0.0)) >= self.v_static:
                    continue
                vl = v.get_location()
                if vl.distance(loc) >= cfg.detection_radius:
                    continue
                pr = self._project(ap._waypoint_planner, vl.x, vl.y)
                if pr is None or not (-10.0 < pr[0] < reach) or abs(pr[1]) > self.obb_lat_m:
                    continue
                stopped.append(v)
            if not stopped:
                self._obb_cache = None
                return set()
            # 자차가 서 있고(v < blocker_speed_max) 후보 집합·경로가 그대로면 예측 결과는
            # 바뀔 수 없다 — 직전 결과를 재사용한다 (적신호 대기 중 매 틱 forecast 방지).
            key = (tuple(sorted(v.id for v in stopped)), int(ap._waypoint_planner.route_index),
                   self.ot_span)
            if float(ego_speed) < self.v_static and self._obb_cache is not None \
                    and self._obb_cache[0] == key:
                self.last_obb_cached = True
                return set(self._obb_cache[1])
            near_lc = ap.is_near_lane_change(ego_speed, route_np)
            n = int(cfg.bicycle_frame_rate * (cfg.forecast_length_lane_change if near_lc
                                              else cfg.default_forecast_length))
            ego_bbs = ap.forecast_ego_agent(ego.get_transform(), ego_speed, n,
                                            target_speed, route_np)
            # PDM 이 이번 틱 뒤에 같은 입력으로 다시 부른다 — 시프트를 안 만들었으면
            # 그 호출이 이 결과를 그대로 쓴다 (Ctrl24AutoPilot.forecast_ego_agent).
            self._fc_cache = (self.fc_key(ap, ego_speed, target_speed, n), ego_bbs)
            pred = ap.predict_other_actors_bounding_boxes(False, stopped, loc, n, near_lc)
        except Exception:                                  # noqa: BLE001 — 목 조립
            return set()
        out = set()
        for vid, bbs in pred.items():
            for i, ebb in enumerate(ego_bbs):
                if i < len(bbs) and ap.check_obb_intersection(ebb, bbs[i]):
                    out.add(vid)
                    break
        self._obb_cache = (key, frozenset(out))
        return out

    # ── K8 span 이웃 연속성 (B-30, 2026-09-09) ────────────────────────────
    def _nb_ok(self, planner, left: bool, n_steps: int):
        """경로 전 구간의 "그 side 시프트 목표가 **이어지는가**" bool 배열.

        `route.py::_shift_target_wp` 와 같은 목표다 (거기서 None 이면 폴백이 원 차로
        중심으로 스냅해 계단이 된다). 두 가지를 함께 본다:
          · 목표가 없다 (이웃 차로 없음) — 무한 점프
          · 목표가 있는데 **건너뛴다** — 경로가 계획 차선변경으로 차로를 바꾸면
            목표도 한 칸 바깥으로 옮겨간다 (실측 2026-09-10 15_직진20: 3.07 m 점프,
            κ 4.87). span_target_jump_m 을 넘는 점프는 끊김으로 센다.
        시프트는 route_points 만 바꾸고 route_waypoints 는 건드리지 않으므로
        (side, n_steps) 당 한 번만 계산하면 된다. 차로 키가 바뀌는 지점에서만
        재조회한다 — 한 차로가 수백 점이라 그만큼 싸다.
        """
        key = ('left' if left else 'right', int(n_steps))
        hit = self._nb_cache.get(key)
        wps = getattr(planner, 'route_waypoints', None)
        if wps is None:
            return None
        if hit is not None and len(hit) == len(wps):
            return hit
        f = getattr(planner, '_shift_target_wp', None)
        if f is None:
            return None

        def tgt_xy(i):
            try:
                t = f(i, left, n_steps)
            except Exception:                              # noqa: BLE001 — 목 플래너
                return 'err'
            if t is None:
                return None
            loc = getattr(getattr(t, 'transform', None), 'location', None)
            if loc is None:
                return (0.0, 0.0)                          # 좌표가 없는 목 — 점프는 안 본다
            return (float(loc.x), float(loc.y))

        out = np.zeros(len(wps), dtype=bool)
        prev_key = object()
        val = False
        for i in range(len(wps)):
            k = getattr(wps[i], 'key', None)
            jump = False
            if k != prev_key:
                xy = tgt_xy(i)
                if xy == 'err':
                    return None
                val = xy is not None
                # 경로 차로가 바뀌는 지점 — 직전 점의 목표와 이어지는지 본다.
                # 끊기는 것은 **그 한 점**이다 (뒤는 다시 이어진다).
                if val and i > 0 and self.span_jump_m > 0.0:
                    pxy = tgt_xy(i - 1)
                    if pxy is None or pxy == 'err':
                        jump = True                        # 없던 목표가 생겼다 = 계단
                    else:
                        step = _math.hypot(xy[0] - pxy[0], xy[1] - pxy[1])
                        # 경로 간격(1/ppm)만큼은 정상 전진이다 — 그만큼 빼고 잰다.
                        if step - 1.0 / float(getattr(planner, 'points_per_meter', 10)) \
                                > self.span_jump_m:
                            jump = True
                prev_key = k
            out[i] = val and not jump
        self._nb_cache[key] = out
        return out

    @staticmethod
    def _cont_window(ok, i_lo: int, i_hi: int):
        """[i_lo, i_hi] 를 모두 담는 최대 연속 True 창 (a, b) — 못 담으면 None."""
        n = len(ok)
        i_lo = max(0, min(int(i_lo), n - 1))
        i_hi = max(0, min(int(i_hi), n - 1))
        if not ok[i_lo] or not ok[i_hi]:
            return None
        a = i_lo
        while a > 0 and ok[a - 1]:
            a -= 1
        b = i_hi
        while b + 1 < n and ok[b + 1]:
            b += 1
        if not bool(ok[i_lo:i_hi + 1].all()):
            return None
        return a, b + 1

    def _route_zones(self, planner, route_s: float) -> list:
        """route_s 이후 정지선 route_s 목록 (kr_rules._route_zones 원문)."""
        out = set()
        try:
            d = float(planner.distances_to_next_traffic_lights[planner.route_index])
            if d < float('inf'):
                out.add(round(route_s + d, 3))
        except Exception:                                  # noqa: BLE001
            pass
        for x in self._all_stopline_s(planner):
            if x >= route_s:
                out.add(round(float(x), 3))
        return sorted(out)

    def _junction_extend(self, planner, side: str, route_s: float, span_end: float):
        """교차로 관통 판정 + span 끝 연장 → (새 끝 route_s, 기각 사유 | None, 진단).

        kr_rules._zone_extension 원문 이식. 판정은 "옆 차로가 그 교차로를 경로와 같은
        방향으로 관통하는가" 다 — 구간의 경로 차로마다 side 이웃이 있고, 이웃끼리
        successor 로 이어져야 한다(`zone_no_through_lane`). 회전 이벤트가 끼면
        `zone_turn` 으로 기각한다 (돌면서 옆 차로로 나가는 것은 관통이 아니다).
        """
        lg = getattr(planner, 'lg', None)
        route = getattr(planner, 'route', None) or {}
        lanes = [tuple(k) for k in (route.get('lanes') or [])]
        cum = [float(x) for x in (route.get('cum_s') or [])]
        lens = [float(x) for x in (route.get('lengths') or [])]
        if (lg is None or not lanes or len(cum) != len(lanes) or len(lens) != len(lanes)):
            return span_end, 'zone_no_route', {}
        new_end = float(span_end)
        zones = self._route_zones(planner, route_s)
        info: dict = {'zones': []}
        for _ in range(4):
            ahead = [z for z in zones if route_s < z < new_end]
            if not ahead:
                break
            z = ahead[0]
            k0 = next((i for i in range(len(lanes))
                       if lg.lanes.get(lanes[i], {}).get('junction', -1) != -1
                       and cum[i] <= z + self.jx_gap_m and cum[i] + lens[i] > z), None)
            if k0 is None:                                 # 정지선만 있다 (횡단보도)
                info['zones'].append({'s': round(z, 1), 'junction': None})
                zones = [q for q in zones if q > z]
                continue
            jid = lg.lanes[lanes[k0]]['junction']
            k = k0
            while (k + 1 < len(lanes)
                   and lg.lanes.get(lanes[k + 1], {}).get('junction', -1) == jid):
                k += 1
            if k + 1 >= len(lanes):
                return span_end, 'zone_no_exit', info
            j_out = cum[k] + lens[k]
            new_end = max(new_end, j_out + self.jx_exit_m)
            info['zones'].append({'s': round(z, 1), 'junction': int(jid),
                                  'out': round(j_out, 1)})
            zones = [q for q in zones if q > j_out]
        info['new_end'] = round(new_end, 1)
        if new_end <= span_end + 1e-9:
            return span_end, 'zone_no_junction', info
        if new_end - span_end > self.jx_max_m:
            return span_end, 'zone_extend_max', info
        for ev in route.get('events') or []:
            kind = str(ev.get('kind', ''))
            if kind.startswith('turn_'):
                if route_s <= float(ev['s']) <= new_end:
                    return span_end, 'zone_turn', info
            elif kind.startswith('lane_change'):
                a0 = float(ev.get('window_s0', ev.get('s', 0.0)))
                b0 = float(ev.get('window_s1', ev.get('s', 0.0)))
                if a0 <= new_end and b0 >= route_s:
                    return span_end, 'zone_lane_change', info
        prev_nb, prev_cum = None, None
        for i, key in enumerate(lanes):
            if cum[i] + lens[i] < route_s or cum[i] > new_end:
                continue
            nb = lg.neighbor(key, side) if key in lg.lanes else None
            if nb is None:
                info['break_lane'] = list(key)
                return span_end, 'zone_no_through_lane', info
            if (prev_nb is not None and nb != prev_nb and prev_cum is not None
                    and cum[i] > prev_cum + 1e-6):
                try:
                    if nb not in lg.successors(prev_nb):
                        info['break_lane'] = list(key)
                        return span_end, 'zone_no_through_lane', info
                except Exception:                          # noqa: BLE001
                    pass
            prev_nb, prev_cum = nb, cum[i]
        return new_end, None, info

    def _span_fit(self, planner, left: bool, steps: int, a: int, b: int,
                  obs_i: int, last_i: int, trans: float, back: float,
                  ext_m: float, route_s: float):
        """이 span 을 계단 없이 놓을 수 있나 → (판정, v_req, 새 b, 진단).

        판정  'ok'        span 전 구간이 연속 — 그대로 만든다
              'extend'    뒤가 끊긴 곳이 교차로이고 옆 차로가 관통한다 → 끝을 늘려 만든다
              'no_target' 장애물 지점에 그 side 목표가 없다 → 반대편 side 를 본다
              'wait'      감속하면 창 안에 놓인다 → v_req 로 감속 (K8 후보)
              'no_room'   감속해도 못 놓는다 → 그 side 는 담지 않는다 (NOOP)

        **계단은 어느 경우에도 남기지 않는다** (2026-09-10 결정): 만드는 것은
        'ok'/'extend' 뿐이고 둘 다 span 전 구간의 연속을 확인한 뒤다.

        v_req 는 창의 총 길이가 아니라 **장애물 앞뒤 여유**로 낸다. 총 길이를 쓰면
        창이 뒤로 길고 앞으로 짧을 때 복귀 전이가 창 밖으로 나가 계단이 남는다
        (실측 2026-09-09, 6건이 그랬다 — docs/BACKLOG.md B-30):
            v ≤ (뒤여유 − extra_before_m − 객체 반길이) / trans_k
            v ≤ (앞여유 − extra_after_m  − 객체 반길이) / trans_k
        """
        ok = self._nb_ok(planner, left, steps)
        if ok is None or len(ok) == 0:
            return 'ok', None, b, {}
        b = min(int(b), len(ok))
        a = max(0, int(a))
        if b <= a:
            return 'ok', None, b, {}
        if bool(ok[a:b].all()):
            return 'ok', None, b, {}
        win = self._cont_window(ok, obs_i, last_i)
        if win is None:
            return 'no_target', None, b, {}
        w0, w1 = win
        rs = getattr(planner, 'route_s', None)

        def s_at(i):
            if rs is None:
                return float(i) / float(getattr(planner, 'points_per_meter', 10))
            return float(rs[max(0, min(int(i), len(rs) - 1))])

        info = {'win': [int(w0), int(w1)]}
        # 뒤가 끊겼다 — 그 자리가 교차로이고 옆 차로가 관통하면 끝을 늘린다.
        # 실측 2026-09-10 (리플레이 40건의 해당 7건): 끊김은 **연결로에 side 이웃이
        # 없어서** 생기고, 그것이 곧 관통 불가다 — 두 사실이 같아서 연장이 서는 경우가
        # 관찰되지 않았다. 옆 차로의 successor 를 따라가 보면 같은 교차로 연결로는
        # 있으나 전부 **다른 도로로** 나간다 (경로를 벗어난다). 그래서 이 가지는
        # 판정 근거(info['jx'])를 남기는 역할을 하고, 성립하면 늘린다.
        # 판정을 지도가 바꾸면(양방향 연결로) 그때 살아난다 — docs/BACKLOG.md B-30.
        if b > w1 and self.span_jx:
            new_end, why, jx = self._junction_extend(planner, 'left' if left else 'right',
                                                     route_s, s_at(b))
            info['jx'] = why or 'ok'
            if jx:
                info['jx_info'] = jx
            if why is None:
                try:
                    b2 = int(np.searchsorted(rs, new_end)) if rs is not None else b
                except Exception:                          # noqa: BLE001
                    b2 = b
                b2 = min(max(b2, b), len(ok))
                if b2 > b and bool(ok[a:b2].all()):
                    info['extended_to'] = round(float(new_end), 1)
                    return 'extend', None, b2, info
                info['jx'] = 'zone_no_through_lane'        # 연장해도 이웃이 끊긴다
        behind = s_at(obs_i) - s_at(w0)
        front = s_at(w1) - s_at(last_i)
        v_back = (behind - self.ot_before_m - ext_m) / self.trans_k
        v_front = (front - self.ot_after_m - ext_m) / self.trans_k
        v_req = min(v_back, v_front)
        info.update({'behind_m': round(behind, 1), 'front_m': round(front, 1),
                     'v_req': round(v_req, 2)})
        if v_req < self.span_v_min:
            return 'no_room', round(v_req, 2), b, info
        return 'wait', round(v_req, 2), b, info

    def _avoid_tick(self, ap, planner, ego_speed: float, obb_ids) -> None:
        """틱당 1회. 원복 → 트리거 수집 → 아직 시프트를 안 만든 첫 객체에 PREEMPT."""
        self.span_v_req = None                       # K8 — 이번 틱 감속 요구 (틱마다 새로)
        i0 = int(planner.route_index)
        if self.ot_span is not None and i0 > self.ot_span[1]:
            self._restore_span(planner)
            self._corridor = self._corridor_blockers(ap, planner)   # 원 경로 기준으로 다시
            self.last_avoid = {'state': 'RESTORED'}
        if not self.ot_enabled:
            return
        corridor = list(self._corridor)
        trig = {c[3].id: 'corridor' for c in corridor}
        for oid in (obb_ids or ()):
            if oid in trig:
                trig[oid] = 'both'
                continue
            act = next((a for a in ap._world.get_actors() if a.id == oid), None)
            if act is None or self._is_walker(act):
                continue
            loc = act.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is None or not (0.5 < pr[0] <= self.detect_max_m):
                continue
            bb = getattr(act, 'bounding_box', None)
            hw = float(bb.extent.y) if bb is not None else 0.9
            corridor.append((pr[0], pr[1], hw, act))
            trig[oid] = 'obb'
        corridor.sort(key=lambda z: z[0])
        if not corridor:
            if self.ot_span is not None and self.last_avoid is None:
                self.last_avoid = {'state': 'SHIFT_ACTIVE', 'span': list(self.ot_span),
                                   'nested': self.nested}
            return
        target = next((c for c in corridor if c[3].id not in self._shifted_for), None)
        if target is None:                               # 전부 이미 시프트한 객체 — 요동 방지
            self.last_avoid = {'state': 'HANDLED', 'blocker': corridor[0][3].id,
                               's_rel': round(corridor[0][0], 1),
                               'span': list(self.ot_span) if self.ot_span else None,
                               'nested': self.nested}
            return
        s_rel, lat, _hw, actor = target
        self.last_avoid = {'state': 'PREEMPT' if self.ot_span is None else 'SHIFT_NESTED',
                           'blocker': actor.id, 's_rel': round(s_rel, 1), 'lat': round(lat, 2),
                           'trigger': trig.get(actor.id, 'corridor'), 'nested': self.nested}
        chain = self._chain(corridor, actor)
        if self._try_shift(planner, ego_speed, chain):
            return
        # 정상 시프트가 불가능하다고 확정됐다 — 고착 래치가 서 있으면 가상 차로로.
        # 탈출 규칙의 한 갈래다 (새 상태기가 아니다).
        if self._esc_engaged and self.esc_virt:
            self._try_virtual_shift(planner, ap, ego_speed)

    def _try_shift(self, planner, ego_speed: float, chain: dict) -> bool:
        """좌측 우선, 좌측이 변위 0(이웃 없음 — route.py 폴백)이면 우측. 게이트는 span_too_far
        하나. 변위 < noop_disp_m 인 시프트는 성공이 아니다(NOOP): 되돌리고 span 을 세우지 않는다."""
        actor, last = chain['first'], chain['last']
        chain_last = None if last is actor else last
        ppm = float(getattr(planner, 'points_per_meter', 10))
        i0 = int(planner.route_index)
        trans = self._trans_m(ego_speed)
        rejects = []
        v_reqs: list = []
        # 객체 반길이 — span 은 장애물 앞뒤로 이만큼 더 뻗는다 (plan_shift_span).
        bb = getattr(actor, 'bounding_box', None)
        ext_m = float(getattr(getattr(bb, 'extent', None), 'x', 0.0) or 0.0)
        try:
            route_s_now = float(planner.route_s[i0])
        except Exception:                                  # noqa: BLE001 — 목 플래너
            route_s_now = 0.0
        # K8 이 쓸 장애물 인덱스 — 투영이 안 되면 연속성 판정을 건너뛴다 (이전 동작).
        obs_i = last_i = None
        try:
            pa = self._project(planner, actor.get_location().x, actor.get_location().y)
            pb = pa if chain_last is None else self._project(
                planner, last.get_location().x, last.get_location().y)
            if pa is not None and pb is not None:
                obs_i = i0 + int(round(pa[0] * ppm))
                last_i = i0 + int(round(pb[0] * ppm))
        except Exception:                                  # noqa: BLE001 — 목 조립
            obs_i = last_i = None
        for side in ('left', 'right'):
            left = side == 'left'
            try:
                a, b, _l = planner.plan_shift_span(
                    actor, chain_last, obstacle_direction='right' if left else 'left',
                    transition_length=trans * ppm,
                    extra_length_before=self.ot_before_m * ppm,
                    extra_length_after=self.ot_after_m * ppm,
                    min_start_ahead=self.shift_ahead_m * ppm)
            except Exception:                              # noqa: BLE001 — 목 플래너
                rejects.append(f'{side}:no_plan')
                continue
            a, b = int(a), int(b)
            self.last_span_plan = (a, b, left)
            span_off = (a - i0) / ppm
            if span_off >= self.span_gate_max_m:          # 유일한 게이트
                rejects.append(f'{side}:span_too_far')
                self.last_avoid['span_off_m'] = round(span_off, 1)
                break                                      # 기하가 side 와 무관 — 반대쪽도 같다
            if b <= a + 1 or b > len(planner.route_points):
                rejects.append(f'{side}:no_plan')
                continue
            ref = (a + b) // 2                            # 중첩 시프트의 밀림은 플래토에서 잰다
            steps = self._target_steps(planner, left, ref)
            # K8 — 계단 없이 이 span 을 놓을 수 있나 (B-30).
            # 게이트가 아니다: 'no_target' 은 반대편 side 로 넘기고, 'wait' 는
            # 생성을 미루는 대신 감속 후보를 낸다. 뒤가 끊긴 곳이 교차로면 옆 차로가
            # 관통하는지 보고 끝을 늘린다. 'no_room' 만 담지 않는다.
            back = trans * _math.sqrt(max(1, steps)) if (
                self.ot_span is not None and b > self.ot_span[1]) else trans
            jx_diag = None
            if self.span_cont and obs_i is not None:
                verdict, v_req, b_fit, info = self._span_fit(
                    planner, left, steps, a, b, obs_i, last_i, trans, back, ext_m, route_s_now)
                jx_diag = info or None
                if verdict == 'no_target':
                    rejects.append(f'{side}:no_target')
                    self.last_avoid[f'{side}_fit'] = info or None
                    continue
                if verdict == 'no_room':
                    rejects.append(f'{side}:span_no_room')
                    self.last_avoid[f'{side}_fit'] = info
                    continue
                if verdict == 'wait':
                    if float(ego_speed) > v_req + 1e-9:
                        rejects.append(f'{side}:span_v_req')
                        self.last_avoid[f'{side}_fit'] = info
                        v_reqs.append(v_req)
                        continue
                    # v ≤ v_req 인데도 span 이 창을 벗어나면(있으면 안 되는 경우 —
                    # v_req 는 객체 반길이까지 빼고 낸다) 만들지 않는다. 계단 금지가 먼저다.
                    rejects.append(f'{side}:span_no_fit')
                    self.last_avoid[f'{side}_fit'] = info
                    continue
                if verdict == 'extend':
                    b = int(b_fit)
                    back = trans * _math.sqrt(max(1, steps)) if (
                        self.ot_span is not None and b > self.ot_span[1]) else trans
            try:
                d = np.asarray(planner.planned_lateral_offsets(a, b, left, step_pts=int(ppm),
                                                               ref_index=ref), dtype=float)
            except TypeError:
                d = np.asarray(planner.planned_lateral_offsets(a, b, left, step_pts=int(ppm)),
                               dtype=float)
            except Exception:                              # noqa: BLE001
                d = None
            if d is not None and (d.size == 0 or float(np.abs(d).max()) < self.noop_disp_m):
                rejects.append(f'{side}:noop')             # 이웃 없음 → 원 경로 유지
                continue
            snap = (planner.route_points[a:b].copy(),
                    planner.commands[a:b].copy() if getattr(planner, 'commands', None) is not None else None,
                    planner.lat_shift[a:b].copy() if getattr(planner, 'lat_shift', None) is not None else None)
            try:
                planner.shift_route_smoothly(a, b, left, transition_length=trans * ppm,
                                             transition_length_back=back * ppm, ref_index=ref)
            except TypeError:
                planner.shift_route_smoothly(a, b, left, transition_length=trans * ppm)
            disp = float(np.abs(planner.route_points[a:b, :2] - snap[0][:, :2]).max())
            if disp < self.noop_disp_m:                    # 폴백이 원 경로를 그대로 뒀다
                planner.route_points[a:b] = snap[0]
                if snap[1] is not None:
                    planner.commands[a:b] = snap[1]
                if snap[2] is not None:
                    planner.lat_shift[a:b] = snap[2]
                rejects.append(f'{side}:noop')
                continue
            self._rebuild_kd(planner)
            self._shift_seq += 1                       # 경로가 바뀌었다 — 예측 캐시 무효
            self.span_v_req = None                     # 만들었으니 감속 요구는 없다
            if self.ot_span is None:
                self.ot_span = (a, b)
            else:
                self.nested += 1
                self.ot_span = (min(self.ot_span[0], a), max(self.ot_span[1], b))
            self.ot_side = side
            self._shifted_for.add(actor.id)
            self.last_overtake = side
            self.last_avoid.update({
                'shift': side, 'span': list(self.ot_span), 'span_new': [a, b],
                'trans_m': round(trans, 1), 'back_m': round(back, 1), 'steps': int(steps),
                'ahead_m': round(self.shift_ahead_m, 1), 'chain': list(chain['ids']),
                'disp_m': round(disp, 2), 'nested': self.nested,
                'fit': jx_diag, 'rejects': rejects or None})
            print(f'[ctrl24] 정적 장애물 회피 — {side} 로 경로 시프트 '
                  f'(id={chain["ids"]}, 구간 {a}~{b}, 전이 {trans:.1f}/{back:.1f} m, '
                  f'{steps}칸, 중첩 {self.nested})', flush=True)
            return True
        # 감속하면 담기는 side 가 있으면 그 v_req 를 후보로 낸다 (가장 높은 쪽 —
        # 거기까지만 줄이면 한쪽은 담긴다). 생성은 다음 틱 이하에서 자연히 일어난다.
        self.span_v_req = max(v_reqs) if v_reqs else None
        noop = bool(rejects) and all(
            r.endswith((':noop', ':no_target', ':span_no_room', ':span_no_fit'))
            for r in rejects)
        state = ('SPAN_WAIT_V' if v_reqs else
                 ('NOOP' if noop else self.last_avoid['state']))
        self.last_avoid.update({'state': state,
                                'reject': rejects[-1] if rejects else None, 'rejects': rejects})
        if self.span_v_req is not None:
            self.last_avoid['span_v_req'] = self.span_v_req
        self.last_overtake = rejects[-1] if rejects else 'no_plan'
        return False

    @staticmethod
    def _target_steps(planner, left: bool, ref: int) -> int:
        f = getattr(planner, '_shift_target_steps', None)
        if f is None:
            return 1
        try:
            return int(f(left, ref_index=ref))
        except TypeError:
            return int(f(left))

    # ── K6 붉은 구간 진입 전 감속 (kr_rules 원문 이식, 2026-09-09 복원) ────────
    def _red_intervals(self, planner) -> list:
        """경로 위 붉은 구간 [(진입 route_s, 이탈 route_s)] — 경로당 1회.

        재료는 route_waypoints[i].key/.s 와 lane_graph 의 red_spans 다 — 제한속도
        배열로 역추정하지 않는다 (그 배열은 carry 가 섞여 "구간에서 물고 나온 30" 과
        "구간 안" 이 구분되지 않는다). 이탈 쪽은 red_zone.exit_margin_m 을 얹은
        지점이라 speed_limit_at 이 캡을 푸는 지점과 같은 축이다.
        """
        lg = getattr(planner, 'lg', None)
        wps = getattr(planner, 'route_waypoints', None)
        rs = getattr(planner, 'route_s', None)
        if lg is None or not wps or rs is None:
            return []
        try:
            from vtd_adapter.lanegraph import red_span_cfg
            _on, _kph, exit_m = getattr(lg, '_red_cfg', None) or red_span_cfg()
        except Exception:                                  # noqa: BLE001 — 목 플래너
            exit_m = 0.0
        out: list = []
        inside = False
        for i in range(min(len(wps), len(rs))):
            wp = wps[i]
            try:
                spans = lg.lanes[wp.key].get('red_spans') or []
            except Exception:                              # noqa: BLE001
                spans = []
            hit = any(s0 <= wp.s <= s1 + exit_m for s0, s1 in spans)
            if hit and not inside:
                out.append([float(rs[i]), float(rs[i])])
                inside = True
            elif hit:
                out[-1][1] = float(rs[i])
            else:
                inside = False
        return [(a, b) for a, b in out]

    def _red_approach_profile(self, planner) -> float | None:
        """K6: 붉은 구간 **진입 전** 감속 상한 — min() 후보. K1 과 같은 형태다.

            v_ceiling = √(v_zone² + 2·a·d)      d = 진입점까지 남은 경로거리

        구간 **안**에서는 후보를 내지 않는다 — 거기서는 제한속도(red_zone.limit_kph −
        speed.margin_kph)가 이미 상한이고, 두 상한을 겹치면 어느 쪽이 묶는지 로그에서
        갈리지 않는다.

        2026-09-09 복원 근거 (Q3 번복). 이 후보를 뺀 채로 돌린 mock 폐루프 32경로에서
        보호구역 속도 초과가 **13경로에서 5 km/h(항목 2 중대 기준) 초과**, 최대
        15.8 km/h 였다. 같은 조건의 kr_rules(이 후보 있음)는 0경로·최대 0.5 km/h.
        50 도로(45 km/h)에서 27 km/h 로 줄이려면 a=2.0 에 25 m 가 드는데 붉은 구간
        124개 중 43개가 18 m 미만이라, 진입 뒤에 줄이기 시작하면 구간 안에서 초과가 난다.
        """
        if not self.red_approach or self.red_a <= 0.0:
            return None
        if self.red_ivals is None:
            self.red_ivals = self._red_intervals(planner)
        if not self.red_ivals:
            return None
        try:
            route_s = float(planner.route_s[planner.route_index])
        except Exception:                                  # noqa: BLE001
            return None
        nxt = None
        for a, b in self.red_ivals:
            if a <= route_s <= b:
                return None                                # 구간 안 — 제한속도 소관
            if a > route_s:
                nxt = a
                break
        if nxt is None:
            return None
        d = nxt - route_s
        if d > self.red_look_m:
            return None
        v = _math.sqrt(self.red_v_zone * self.red_v_zone + 2.0 * self.red_a * max(0.0, d))
        self.last_red_zone = {'d': round(d, 1), 'v_allow': round(v, 2),
                              'entry_s': round(nxt, 1)}
        return v

    # ── K7 시프트 전이 횡가속 상한 (kr_rules._shift_speed_cap 원문 이식) ──────
    def _shift_speed_cap(self, planner, ego_speed: float) -> float | None:
        """진행 중인 회피 시프트의 전이 곡률에서 나오는 **속도 상한** — min() 후보.

        전이 길이 `trans = max(trans_min_m, trans_k·v)` 는 시프트를 **만든 시점**의
        속도로 굳는다. 정지 중 생성되면 8 m 인데 그대로 6~7 m/s 로 지나면 요구
        횡가속이 a_lat_max 를 크게 넘어 조향이 풀락으로 포화한다.

        ctrl24 는 geom 게이트(전이 + shift_ahead + 여유 ≤ 장애물까지 거리)를 삭제해
        공간이 부족한 시프트를 **기각하지 않고 만든다** — 그 대가를 여기서 속도로
        갚는다. 실측 2026-09-09 리플레이 40건: 시프트 활성 41구간 중 5구간(319틱)이
        생성 시 geom 여유 −5.3 ~ −20.7 m 였고 그 구간에서 |steer| ≥ 0.47 였다.

            κ = |d²(lat_shift)/ds²|            (전이의 횡곡률)
            a_lat = κ·v²  ≤  a_lat_max   →   v ≤ √(a_lat_max / κ)

        · **경로를 다시 밀지 않는다** — 진행 중인 시프트를 재생성하면 현재 위치의
          경로가 옆으로 튀어 급조향이 된다. 대신 같은 기하를 통과 가능한 속도로 만든다.
        · `lat_shift − _lat_build` 를 미분한다 — **회피 시프트 성분만** 본다.
          `lat_shift` 자체에는 계획 차선변경 블렌드와 테이퍼 보정이 함께 실려 있어
          그대로 미분하면 이미 검증된 계획 기하까지 세서 평지에서도 하한까지 내려간다.
        · 평지(plateau)와 span 밖에서는 κ = 0 이라 스스로 비활성이다 — 시프트가 없는
          틱에는 후보를 내지 않는다 (`ot_span is None` → None).
        · 0.5 m 스텐실 — `lat_shift` 는 블렌드로 만든 해석적 배열이라 잡음이 없고,
          폭을 넓히면 전이 경계(κ 최대 지점)에서 평지를 섞어 **과소평가**한다.
        · 하한 `shift_cap_min_v` — 전이 한복판에서 완전히 서면 빠져나올 수 없다.
        · **게이트가 아니다.** 시프트 생성·중첩·NOOP 판정 어디에도 관여하지 않는다.
        """
        if not self.shift_cap_on or self.a_lat_max <= 0.0 or self.ot_span is None:
            return None
        lat = getattr(planner, 'lat_shift', None)
        if lat is None or len(lat) == 0:
            return None
        arr = np.asarray(lat, dtype=float)
        base = getattr(planner, '_lat_build', None)
        if base is not None and len(base) == len(arr):
            arr = arr - np.asarray(base, dtype=float)       # 회피 시프트 성분만
        i = int(getattr(planner, 'route_index', 0))
        ppm = float(getattr(planner, 'points_per_meter', 10))
        look = max(self.shift_cap_look_m, self.trans_k * max(ego_speed, 0.1))
        h = max(1, int(round(0.5 * ppm)))                   # 0.5 m 스텐실 (위 참조)
        j0 = max(i, int(self.ot_span[0]), h)
        j1 = min(len(arr) - 1 - h, int(self.ot_span[1]), i + int(look * ppm))
        if j1 <= j0:
            return None
        hs = h / ppm
        d2 = np.abs(arr[j0 + h:j1 + h + 1] - 2.0 * arr[j0:j1 + 1] + arr[j0 - h:j1 - h + 1])
        kappa = float(d2.max()) / (hs * hs)
        if kappa <= 1e-6:
            return None
        return max(self.shift_cap_min_v, _math.sqrt(self.a_lat_max / kappa))

    # ── 리셋 ──────────────────────────────────────────────────────────────
    def on_reset(self) -> None:
        """courseRespawn — 순간이동 전 래치는 전부 무효 (run_agent 가 부른다)."""
        self._yellow_reset()
        self.cross_guard = False
        self.cross_s = None
        self.cross_junction_seen = False
        self.sl_hold_left = 0
        self.sl_stopped = False
        self.sl_stop_ticks = 0
        self.ped_lat.clear()
        self.ped_intent.clear()
        self.ped_clear.clear()
        self.ped_hold.clear()
        self.cw_wait.clear()
        self.ped_walkin.clear()
        self.ped_hold_ids.clear()
        self.ped_miss.clear()
        self.ped_last.clear()
        self._sig_go = False                          # timeout GO 래치
        self._sig_go_tl = None
        self._sig_wait_ticks = 0
        self._rtor_reset()                            # RTOR 래치·정지 누적 (순간이동 = 새 접근)
        self._esc_hist.clear()                        # K9 — 순간이동은 진행이 아니다
        self._esc_engaged = False
        self._esc_mark = None
        self._esc_xy = None                           # 순간이동 변위를 누적하지 않는다
        self._esc_rearmed = []

    # ── 틱 ────────────────────────────────────────────────────────────────
    def _tick_head(self, ap, planner, ego_speed: float) -> None:
        """apply 머리 — 틱당 1회 상태·진단 리셋과 래치 갱신. 커밋 2·3 이 확장한다."""
        self._ap = ap
        self.last_stop_profile = None
        self.last_yellow = None
        self.last_ped = None
        self.last_kr = {}
        self.last_kr_winner = None
        self.last_red_zone = None
        self.ped_emergency = False
        if not self._prepass_done:
            # pre_pass 가 이미 이 틱의 회피 진단을 만들었으면 지우지 않는다 (2026-09-08
            # 리플레이 41건에서 avoid 가 전부 null 로 남았던 버그).
            self.last_avoid = None
            self.last_overtake = None
        self._yellow_latch(planner, ego_speed, ap)
        # 회랑 안 정지 객체 (관찰 시계 없음) — timeout GO·RTOR 의 앞차 판정 입력.
        # pre_pass 가 경로를 밀었으면 밀린 경로 기준으로 다시 잰 값이다.
        self._corridor = self._corridor_blockers(ap, planner)

    def _tick_signals(self, ap, planner, ego_speed: float, route_s: float) -> None:
        """timeout GO · RTOR · 지시등. 회랑(self._corridor)은 회피 틱이 먼저 채운다."""
        self._tick_lg = getattr(planner, 'lg', None)
        self._tick_ego_lane = (getattr(ap, '_kr_ego_lane', None)
                               or self._ego_lane(self._tick_lg, ap))
        self.last_signal = self._signal_stale(planner)
        self._signal_timeout_tick(ap, planner, ego_speed)
        self._rtor_tick(ap, planner, ego_speed)
        (self.last_turn_signal, self.last_sig_src,
         self.last_sig_lead_s) = self._turn_signal(planner, route_s, ego_speed)

    def _candidates(self, ap, planner, ego_speed: float, target_speed: float):
        """kr 후보 → (candidate, ped_bind, ped). 전부 min() 후보다."""
        kr: dict = {}
        candidate = None

        def add(name, v):
            nonlocal candidate
            if v is None:
                kr[name] = None
                return
            kr[name] = round(float(v), 3)
            if candidate is None or v < candidate:
                candidate = v

        prof = self._stopline_profile(planner, ap)
        self.last_stop_profile = prof
        add('stop_profile', prof)
        add('red_zone', self._red_approach_profile(planner))
        # K7 — 진행 중인 회피 시프트에서만 산다. 게이트가 아니라 상한이라,
        # 신호·보행자가 더 낮으면 그쪽이 이긴다.
        cap = self._shift_speed_cap(planner, ego_speed)
        add('shift_cap', cap)
        # K8 — 이웃 연속 창에 전이 2회가 들어가는 속도. 시프트를 **만들기 위한** 감속이라
        # 생성 전에만 산다 (만든 틱에 None 이 된다). 상한이지 오버라이드가 아니다.
        add('span_v_req', self.span_v_req)
        # 가상 차로 구간 상한 — 중앙선·갓길을 지나는 동안 느리게 간다.
        add('virtual_cap', self._virtual_cap(planner))
        if cap is not None:
            self.last_avoid = dict(self.last_avoid or {'state': 'SHIFT_ACTIVE'},
                                   shift_cap=round(float(cap), 2))
        add('stop_hold', self._stopline_hold(planner, ego_speed))
        add('rtor_cap', self._rtor_cap())
        ped = self._ped_intent(planner, ap, ego_speed)
        ped_bind = False
        if ped is not None:
            v_allow, a_req, wid = ped
            ped_bind = candidate is None or v_allow <= candidate + 1e-9
            add('ped_intent', v_allow)
            self.last_ped = {'id': int(wid), 'v_allow': round(float(v_allow), 2),
                             'a_req': round(float(a_req), 2), 'wins': bool(ped_bind),
                             **self.ped_diag.get(wid, {})}
            if self.ped_released:
                self.last_ped['released'] = {int(k): v for k, v in self.ped_released.items()}
        else:
            add('ped_intent', None)
            if self.ped_released:
                wid, why = next(iter(self.ped_released.items()))
                self.last_ped = {'id': int(wid), 'wins': False, 'release': why,
                                 **self.ped_diag.get(wid, {})}
        if self.ped_multi and (self.ped_all or self.ped_hold_ids):
            if self.last_ped is None:
                self.last_ped = {'id': None, 'wins': False}
            self.last_ped['all'] = {int(k): v for k, v in self.ped_all.items()}
            self.last_ped['hold'] = sorted(int(k) for k in self.ped_hold_ids)
        cw = self._ped_crosswalk(planner, ap, ego_speed)
        if cw is not None:
            v_cw, info = cw
            add('crosswalk', v_cw)
            cw_wins = bool(v_cw <= min(target_speed, candidate) + 1e-9)
            if self.last_ped is None:
                self.last_ped = {'id': int(info['id']), 'wins': cw_wins, 'crosswalk': info}
            else:
                self.last_ped['crosswalk'] = info
        else:
            add('crosswalk', None)
        self.last_kr = kr
        if candidate is not None:
            self.last_kr_winner = min((k for k, v in kr.items() if v is not None),
                                      key=lambda k: kr[k])
        return candidate, ped_bind, ped

    def _rtor_cap(self):
        """RTOR 래치 중 속도 상한 — min() 후보."""
        if self._rtor_go and self.rtor_go_v > 0.0:
            return self.rtor_go_v
        return None

    # ── K9 탈출 바닥 (무한정지 방지, 2026-09-10) ──────────────────────────
    #
    # 규칙 하나:
    #   경로 진행이 escape_stuck_s 동안 escape_progress_m 미만이고 정당한 정지 원인이
    #   없으면, 전방 간격이 escape_clear_m 을 넘는 한 정당하지 않은 모든 min() 후보에
    #   바닥 escape_v 를 깔고 escape_release_m 진행할 때까지 유지한다.
    #
    # 유한 시간 논증 — 걸린 뒤 매 틱 목표가 escape_v 이상이다(간격 조건이 서는 한).
    # 고착의 정의상 정당한 원인이 없으므로 그것이 결과를 0 으로 되돌릴 수 없다.
    # 래치는 escape_release_m 진행까지 유지되므로 매 걸림이 거리를 벌거나 계속 명령한다.
    # 정당한 정지 불가침 논증 — 정당한 후보는 바닥을 **안 받는다**. 최종값은
    # min(정당한 후보들, max(escape_v, 정당하지 않은 후보들)) 이라, 정당한 후보가
    # escape_v 보다 낮으면 그대로 이긴다. 신호·보행자를 뚫는 경로가 구조적으로 없다.

    LEGIT_KR = ('stop_profile', 'stop_hold', 'ped_intent', 'crosswalk')

    def _pdm_legit(self, ap) -> bool:
        """PDM 목표를 정한 것이 정당한 원인인가 (적신호·보행자·정지표지).

        autopilot 원문이 그 셋 중 하나가 min 을 이겼을 때만 세우는 플래그다 —
        후보값을 따로 받아올 필요가 없고, 조립이 무엇이든 있다.
        """
        return bool(getattr(ap, 'traffic_light_hazard', False)
                    or getattr(ap, 'walker_hazard', False)
                    or getattr(ap, 'stop_sign_hazard', False))

    def _front_gap_m(self) -> float | None:
        """회랑 전방 최근접 객체까지 **범퍼 간격** [m]. 없으면 None (막힌 것이 없다).

        s_rel 은 자차 뒷축 → 객체 중심이므로 앞범퍼(front)와 객체 반길이를 뺀다.
        """
        best = None
        for s_rel, _lat, _hw, act in (self._corridor or []):
            bb = getattr(act, 'bounding_box', None)
            ext = float(getattr(getattr(bb, 'extent', None), 'x', 0.0) or 0.0)
            g = float(s_rel) - self.front - ext
            if best is None or g < best:
                best = g
        return best

    def _escape_odo(self, ap) -> float:
        """자차 **이동거리** 누적 [m] — 틱 간 위치 변위의 합.

        route_s 로 재지 않는 이유 (2026-09-10): 종점 패드에서는 route_s 가 더 늘지
        않고, courseRespawn 은 route_s 를 튀게 한다. 둘 다 "진행이 없다" 로 오독된다.
        리스폰은 on_reset 이 직전 위치를 지워 순간이동 거리가 안 섞인다.
        """
        try:
            loc = ap._vehicle.get_location()
            xy = (float(loc.x), float(loc.y))
        except Exception:                                  # noqa: BLE001 — 목 조립
            return self._esc_odo
        if self._esc_xy is not None:
            d = _math.hypot(xy[0] - self._esc_xy[0], xy[1] - self._esc_xy[1])
            if d < 10.0:                                   # 한 틱 10 m 이상은 순간이동
                self._esc_odo += d
        self._esc_xy = xy
        return self._esc_odo

    def _escape_tick(self, ap, legit_min: float | None) -> None:
        """진행 시계 갱신 + 래치 판정. 틱당 1회, 후보 계산 뒤에 부른다."""
        self.last_escape = None
        odo = self._escape_odo(ap)
        if not self.esc_on_cfg:
            self._esc_hist.clear()
            self._esc_engaged = False
            self._esc_mark = None
            return
        self._esc_hist.append(float(odo))
        # 정당한 원인이 바닥보다 낮게 잡고 있으면 그것은 정당한 정지다 — 고착이 아니다.
        legit_alive = legit_min is not None and legit_min < self.esc_v
        if self._esc_engaged:
            if legit_alive or (self._esc_mark is not None
                               and odo - self._esc_mark >= self.esc_release_m):
                self._esc_engaged = False
                self._esc_mark = None
                self._esc_rearmed = []
                self._esc_hist.clear()
                return
            # 걸려 있는 **동안 매 틱** 재무장한다 (2026-09-10). 래치가 서는 틱 1회로는
            # 못 잡는다 — 실측: 18_연속교차로11 은 래치 29.8 s, 시프트 42.8 s, 그 뒤
            # HANDLED 고착이라 걸릴 때는 요동 방지 집합이 비어 있었다.
            got = self._escape_rearm()
            if got:
                self._esc_rearmed = sorted(set(self._esc_rearmed) | set(got))
            return
        if legit_alive:
            return
        if len(self._esc_hist) < self._esc_hist.maxlen:
            return
        if odo - self._esc_hist[0] >= self.esc_prog_m:
            return
        self._esc_engaged = True
        self._esc_mark = float(odo)

    def _escape_floor(self, ap) -> float | None:
        """이번 틱에 깔 바닥 [m/s]. 안 걸렸거나 간격이 모자라면 None.

        간격 조건은 **바닥의 정의**다 (정당한 정지 집합이 아니다): 회랑 전방 최근접
        객체까지 범퍼 간격이 escape_clear_m 이하면 바닥이 0 이라 원래 후보가 그대로다.
        붙어서 밀지 않는다.
        """
        if not self._esc_engaged:
            return None
        gap = self._front_gap_m()
        if gap is not None and gap <= self.esc_clear_m:
            return None
        return self.esc_v

    # ── 가상 차로 시프트 (B-33) ────────────────────────────────────────────
    def _virtual_dir(self, planner, ap, obj_half_w: float):
        """가상 차로를 어느 쪽으로 몇 미터 밀지 → (side, D, 사유).

        방향 우선순위는 사양 그대로다:
          (1) 좌 — 같은 (도로, 섹션) 에 반대 방향 차로가 있으면 (중앙선 침범)
          (2) 우 — sidewalk_right_m 이 D 이상이면 (갓길·보도 앞)
          (3) 둘 다 아니면 None — 벽·건물일 수 있다

        자차의 왼쪽은 항상 도로 중심선(lane 0) 쪽이다: dir=+1 이면 lane_id<0,
        dir=−1 이면 lane_id>0 이므로 두 경우 모두 중심이 왼쪽이다.
        """
        lg = self._tick_lg or getattr(planner, 'lg', None)
        key = self._tick_ego_lane or self._ego_lane(lg, ap)
        if lg is None or key is None or key not in getattr(lg, 'lanes', {}):
            return None, 0.0, 'no_lane'
        r = lg.lanes[key]
        half = float(self.half_ego)
        d = half + float(obj_half_w) + self.esc_virt_extra + self.clr
        if r.get('junction', -1) != -1:
            return None, d, 'junction'                 # 교차로 연결로 — 옆이 무엇인지 모른다
        # (1) 반대 방향 차로가 같은 섹션에 있나
        try:
            opp = any(lg.lanes[q]['road'] == r['road'] and lg.lanes[q]['sec'] == r['sec']
                      and lg.lanes[q]['dir'] != r['dir'] for q in lg.lanes)
        except Exception:                              # noqa: BLE001 — 목 그래프
            opp = False
        if opp:
            return 'left', d, 'oncoming_lane'
        # (2) 우측 보도 안쪽 경계까지의 여유
        sw = r.get('sidewalk_right_m')
        try:
            room = float(np.min(np.asarray(sw, dtype=float))) if sw is not None else None
        except Exception:                              # noqa: BLE001
            room = None
        if room is not None and room >= d:
            return 'right', d, f'shoulder({room:.1f}m)'
        return None, d, 'no_room' if room is not None else 'no_side'

    def _oncoming_clear(self, ap, planner) -> tuple:
        """대향 방향 이동 차량이 escape_oncoming_clear_m 안에 있나 → (clear, 최근접 m)."""
        best = None
        try:
            actors = list(ap._world.get_actors())
        except Exception:                              # noqa: BLE001
            return True, None
        ego_id = ap._vehicle.id
        for a in actors:
            if a.id == ego_id or self._is_walker(a):
                continue
            if float(getattr(a, 'speed', 0.0)) < self.v_static:
                continue                               # 정지 객체는 대향 위험이 아니다
            loc = a.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is None:
                continue
            s_rel, _lat = pr
            if 0.0 < s_rel <= self.esc_onc_m and (best is None or s_rel < best):
                best = s_rel
        return (best is None), best

    def _try_virtual_shift(self, planner, ap, ego_speed: float) -> bool:
        """정상 시프트가 불가능하다고 확정된 뒤의 마지막 수단. 성공하면 True.

        이웃 차로를 조회하지 않고 원 경로를 옆으로 D 만큼 민 폴리라인을 목표로 준다
        (route.py::shift_route_smoothly 의 offset_m). 계단은 정의상 생기지 않는다 —
        목표가 연속 폴리라인이라 끊기는 지점이 없다.
        """
        if not (self.esc_virt and self._esc_engaged) or self.ot_span is not None:
            return False
        if not self._corridor:
            return False
        s_rel, _lat, hw, actor = self._corridor[0]
        side, d, why = self._virtual_dir(planner, ap, hw)
        diag = {'state': 'ESCAPE_VIRTUAL', 'blocker': int(actor.id),
                's_rel': round(float(s_rel), 1), 'D': round(float(d), 2),
                'side': side, 'why': why}
        if side is None:
            diag['state'] = 'ESCAPE_VIRTUAL_NONE'
            self.last_avoid = dict(self.last_avoid or {}, **diag)
            return False
        if side == 'left':
            clear, near = self._oncoming_clear(ap, planner)
            if not clear:
                self._virt_wait += 1
                diag.update({'state': 'ESCAPE_VIRTUAL_WAIT', 'oncoming_m': round(near, 1),
                             'wait_ticks': self._virt_wait})
                self.last_avoid = dict(self.last_avoid or {}, **diag)
                return False
        ppm = float(getattr(planner, 'points_per_meter', 10))
        trans = self._trans_m(min(float(ego_speed), self.esc_virt_v))
        chain = self._chain(list(self._corridor), actor)
        last = chain['last']
        try:
            a, b, _l = planner.plan_shift_span(
                actor, None if last is actor else last,
                obstacle_direction='right' if side == 'left' else 'left',
                transition_length=trans * ppm,
                extra_length_before=self.ot_before_m * ppm,
                extra_length_after=self.ot_after_m * ppm,
                min_start_ahead=self.shift_ahead_m * ppm)
        except Exception:                              # noqa: BLE001 — 목 플래너
            return False
        a, b = int(a), int(b)
        if b <= a + 1 or b > len(planner.route_points):
            return False
        snap = (planner.route_points[a:b].copy(),
                planner.commands[a:b].copy() if getattr(planner, 'commands', None) is not None else None,
                planner.lat_shift[a:b].copy() if getattr(planner, 'lat_shift', None) is not None else None)
        try:
            planner.shift_route_smoothly(a, b, side == 'left', transition_length=trans * ppm,
                                         transition_length_back=trans * ppm,
                                         ref_index=(a + b) // 2, offset_m=d)
        except TypeError:                              # 구형 플래너 — 가상 시프트 없음
            return False
        disp = float(np.abs(planner.route_points[a:b, :2] - snap[0][:, :2]).max())
        if disp < self.noop_disp_m:
            planner.route_points[a:b] = snap[0]
            if snap[1] is not None:
                planner.commands[a:b] = snap[1]
            if snap[2] is not None:
                planner.lat_shift[a:b] = snap[2]
            return False
        self._rebuild_kd(planner)
        self._shift_seq += 1
        self.ot_span = (a, b)
        self.ot_side = side
        self._virt_span = (a, b)
        self._shifted_for.add(actor.id)
        self.last_overtake = f'virtual:{side}'
        diag.update({'span': [a, b], 'trans_m': round(trans, 1),
                     'disp_m': round(disp, 2), 'wait_ticks': self._virt_wait})
        self.last_avoid = dict(self.last_avoid or {}, **diag)
        print(f'[ctrl24] 가상 차로 시프트 — {side} {d:.2f} m '
              f'(id={actor.id}, 구간 {a}~{b}, 전이 {trans:.1f} m, 사유 {why})', flush=True)
        return True

    def _virtual_cap(self, planner):
        """가상 시프트 span 안 속도 상한 — min() 후보."""
        if self._virt_span is None:
            return None
        a, b = self._virt_span
        i = int(getattr(planner, 'route_index', 0))
        if i > b or self.ot_span is None:
            self._virt_span = None
            self._virt_wait = 0
            return None
        return self.esc_virt_v if i >= a - int(self.shift_ahead_m * 10) else None

    def _escape_rearm(self) -> list:
        """래치가 걸려 있는 동안 회랑 전방 객체를 요동 방지 집합에서 뺀다 (별도 스위치).

        매 틱 본다 — 고착의 원인이 되는 HANDLED(이미 시프트한 객체라 재시도 안 함)는
        래치가 선 **뒤에** 생기는 경우가 있다 (실측 2026-09-10, 4런 중 3런).
        """
        if not self.esc_rearm or not self._shifted_for:
            return []
        ids = [int(a.id) for _s, _l, _h, a in (self._corridor or [])
               if int(a.id) in self._shifted_for]
        for i in ids:
            self._shifted_for.discard(i)
        return ids

    def apply(self, control, target_speed: float, ap):
        """(control, target_speed) → 규칙 반영 후 (control, target_speed).
        속도는 전부 min() 후보다. 후보가 PDM 목표보다 낮으면 종방향을 되감아 재계산."""
        planner = ap._waypoint_planner
        route_s = float(planner.route_s[planner.route_index])
        ego_speed = ap._vehicle.get_velocity().length()
        self._tick_head(ap, planner, ego_speed)
        if not self._prepass_done:                    # 훅이 없는 조립(테스트·구형 autopilot)
            self._avoid_tick(ap, planner, ego_speed, None)
        self._prepass_done = False
        self._tick_signals(ap, planner, ego_speed, route_s)

        candidate, ped_bind, ped = self._candidates(ap, planner, ego_speed, target_speed)

        # ── K9 탈출 바닥 ──────────────────────────────────────────────────
        # 후보를 정당/정당하지 않음으로 가르고, **정당하지 않은 후보만** c → max(c, 바닥)
        # 으로 바꾼 뒤 평소대로 min() 한다. 중재는 그대로 min() 이다.
        # PDM 목표가 어느 쪽인지는 autopilot 원문의 hazard 플래그가 정한다.
        kr = self.last_kr or {}
        pdm_legit = self._pdm_legit(ap)
        legit_vals = [float(v) for k, v in kr.items()
                      if k in self.LEGIT_KR and v is not None]
        other_vals = [float(v) for k, v in kr.items()
                      if k not in self.LEGIT_KR and v is not None]
        (legit_vals if pdm_legit else other_vals).append(float(target_speed))
        legit_min = min(legit_vals) if legit_vals else None
        other_min = min(other_vals) if other_vals else None
        base_t = target_speed if candidate is None else min(target_speed, candidate)
        self._escape_tick(ap, legit_min)
        floor = self._escape_floor(ap)
        esc_t = None
        gap = self._front_gap_m() if self._esc_engaged else None
        if floor is not None:
            raised = floor if other_min is None else max(other_min, floor)
            esc_t = raised if legit_min is None else min(legit_min, raised)
            self.last_escape = {'state': 'FLOOR', 'v': round(float(floor), 2),
                                'gap_m': None if gap is None else round(float(gap), 1),
                                'before': round(float(base_t), 2),
                                'after': round(float(esc_t), 2),
                                'raised': bool(esc_t > base_t + 1e-9),
                                'held_m': (None if self._esc_mark is None
                                           else round(self._esc_odo - self._esc_mark, 1))}
            if self._esc_rearmed:
                self.last_escape['rearmed'] = list(self._esc_rearmed)
        elif self._esc_engaged:
            # 걸렸지만 전방 간격이 모자란다 — 바닥이 0 이다 (원래 후보 그대로).
            self.last_escape = {'state': 'BLOCKED_GAP',
                                'gap_m': None if gap is None else round(float(gap), 1),
                                'before': round(float(base_t), 2),
                                'raised': False,
                                'held_m': (None if self._esc_mark is None
                                           else round(self._esc_odo - self._esc_mark, 1))}

        # K4 보행자 비상 우회 — 보행자 후보가 최종 목표를 구속하는 틱 한정.
        final_t = base_t if esc_t is None else esc_t
        emg = bool(ped is not None and ped_bind and self.ped_emg_ratio > 0.0
                   and ped[1] > self.ped_emg_ratio * self.a_dec_max
                   and final_t <= ped[0] + 1e-9)
        if emg:
            self.ped_emergency = True
            if self.last_ped is not None:
                self.last_ped['emergency'] = True

        # K4 비상은 바닥보다 먼저다 — 보행자 후보는 정당한 원인이라 바닥이 뚫지 못하고,
        # 그 위에서 비상 감속까지 필요한 틱이면 비상이 이긴다.
        if emg:
            target_speed = float(final_t)
            ap._longitudinal_controller.rewind_last()
            accel, brake = ap._longitudinal_controller.emergency()
            control.accel = accel
            control.throttle = accel
            control.brake = float(brake)
        elif esc_t is not None and esc_t > base_t + 1e-9:
            # 바닥이 올렸다 — 종방향을 되감아 새 목표로 다시 계산한다.
            target_speed = float(esc_t)
            ap._longitudinal_controller.rewind_last()
            accel, brake = ap._longitudinal_controller.get_throttle_and_brake(
                False, target_speed, ego_speed)
            control.accel = accel
            control.throttle = accel
            control.brake = float(brake)
        elif candidate is not None and candidate < target_speed:
            target_speed = candidate
            hazard = target_speed < 1e-5
            ap._longitudinal_controller.rewind_last()
            accel, brake = ap._longitudinal_controller.get_throttle_and_brake(
                hazard, target_speed, ego_speed)
            control.accel = accel
            control.throttle = accel
            control.brake = float(brake)
        self.last_target = float(target_speed)
        return control, target_speed
