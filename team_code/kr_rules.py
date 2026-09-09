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

import numpy as np
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


def turn_intervals(planner) -> list[dict]:
    """route['events'] 의 **회전만** → 점등 구간 [{sig, src, ev_s, end_s}]. 시작 시 1회.

    회전은 연결로 중심선을 따라가므로 "차로 중심에서 벗어남" 이 0 이라 기하로는
    잡히지 않는다 — 이벤트 목록이 유일한 근거다. 반대로 **차로를 옮기는 움직임**
    (계획된 차선변경 · 런타임 회피 시프트)은 목록이 아니라 경로 기하로 본다
    (_lane_shift). 목록에 없는 런타임 시프트도 자동으로 잡히게 하기 위해서다.
    """
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
        self.sl_min_ticks = int(round(float(sp.get('stopline_hold_min_s', 0.0))
                                      * float(cfg['comm']['send_hz'])))
        # 정지선 정지 프로파일 (④′). 0 이면 완전 비활성 — 되돌리는 스위치다.
        self.stop_profile_a = float(sp.get('stop_profile_a', 0.0))
        # 붉은 구간(보호구역) 진입 전 감속 (작업 2b-B). ④′ 와 같은 형태의 min() 후보.
        self.red_approach = bool(sp.get('red_approach_enable', True))
        self.red_a = float(sp.get('approach_decel_mps2', 2.0))
        self.red_look_m = float(sp.get('red_lookahead_m', 60.0))
        self.red_v_zone = float(sp.get('red_zone_target_kph', 27.0)) / 3.6
        self.red_ivals: list | None = None            # 경로당 1회 캐시 [(진입 s, 이탈 s)]
        self.last_red_zone = None                     # 진단
        # 황색 딜레마 원샷 판정 (C). 0 이면 비활성 = 황색을 PDM 원문에만 맡긴다.
        self.a_yellow = float(sp.get('a_yellow', 0.0))
        # 황색 판정 **거리 한계** [m] (2026-09-09 (D)). 0 = 무제한(이전 동작).
        # 황색 딜레마는 정지선이 가까울 때만 존재한다. 지금은 거리 조건이 없어
        # **300 m 밖 황색에도 STOP 을 래치**하고, 그 래치(y_decision)가
        # `_obstacle_cause` 를 거짓으로 만들어 **크립·BREAKOUT·never_stall 을
        # 전부 죽인다**. 그래서 20 m 앞 정지 차량에 막혀도 탈출구가 하나도 안 열린다.
        # 실측 20260909_093415:
        #   11_직진10  rs 477.7  황색 STOP 래치 d_line **150.8 m** (장애물 20.1 m)
        #   18_연속…   rs 2590.5 황색 STOP 래치 d_line **317.3 m** (장애물 22.0 m)
        #   → 둘 다 30 s 무전진으로 blocked 종료.
        # 기본은 speed.red_lookahead_m 를 그대로 읽는다 ("적색이 의미를 갖는 거리"
        # 의 단일 출처). 그 밖의 황색은 도착 전에 적색→녹색으로 한 바퀴 돈다.
        self.y_gate_m = float(sp.get('yellow_decide_max_m', 0.0))
        # PDM 의 traffic_light_hazard 를 **장애물이 정지선보다 훨씬 앞일 때** 무시할지.
        # (2026-09-09 (D)) false = 이전 동작.
        self.tl_hazard_far = bool(sp.get('tl_hazard_far_blocker_enable', False))
        # 적신호 접근에서 **가속 금지** (2026-09-09 (C)).
        # PDM 의 red-light IDM 은 차간모형이라 남은 거리가 s* 보다 조금만 커도
        # **가속을 요구한다**(_stopline_profile 주석). ④′ 는 √ 프로파일이라
        # 아직 v 위에 있어 min() 에서 지고, 그 사이 차가 붙었다 급제동한다 —
        # 이것이 "정지선 앞 찔끔찔끔" 의 정체다.
        # 실측 20260909_093415/실경로_02: 적색 접근 667틱 중 **290틱(43.5 %)**
        # 이 목표 > 현재속도였고 최대 **+4.17 m/s** 였다 (slf −14.7 에서 v 5.44
        # 인데 목표 6.15 → accel +0.57 로 적신호를 향해 가속).
        # 상한형이므로 cap 축이다 (err/dt 금지 — CLAUDE.md 확정 사실).
        # false = 이전 동작.
        self.no_accel_red = bool(sp.get('no_accel_toward_red_enable', False))
        # 정지 후 **미세 전진** 방지 래치 (2026-09-09 (C)).
        # 실측: 02 정지 3회 중 2회, 07 5회 중 2회에서 정지 뒤 v 가 0.1 을 넘어
        # 0.2 m 씩 기어갔다. false = 이전 동작.
        self.stop_creep_latch = bool(sp.get('stopline_creep_latch_enable', False))
        # ④′ 정지 프로파일·황색 판정의 **jerk 램프 보정** (2026-09-09 [2]).
        # 둘 다 "지금부터 a 를 즉시 낼 수 있다" 를 전제로 √(2·a·d) 를 쓰는데,
        # 종방향에는 jerk 제한이 있어 a 에 도달하는 데 t = a / jerk_rate 가 걸리고
        # 그동안 평균 감속이 절반이라 **v·t/2 만큼 더 간다**.
        # 실측 20260909_093415: 정지선 침범 9건, 최대 +3.28 m. 침범 지점의 접근
        # 속도가 전부 43~49 km/h(12~13.6 m/s)고, a 3.0 / jerk 6.0 → t 0.5 s 라
        # 보정량이 3.0~3.4 m — 실측 침범량과 같은 자릿수다.
        # 저속 접근은 v 에 비례해 보정이 작아진다 (상수를 낮추는 것과 다른 점).
        # false = 이전 동작.
        self.stop_jerk_comp = bool(sp.get('stop_jerk_compensate_enable', False))
        # 감속측 jerk 한계 [m/s³]. **control 이 실제로 쓰는 두 상수를 그대로 읽는다**
        # (VtdLongitudinalController: prev − jerk_dec_mult·jerk_max·dt).
        # 새 상수를 만들면 제동 램프의 정의가 두 벌이 된다.
        self.jerk_rate = float(cfg['control']['jerk_dec_mult']) * float(sp.get('jerk_max', 2.0))
        # 보행자 의도 감지 (P4). 0 이면 비활성 = PDM forecast_walkers 에만 맡긴다.
        self.ped_intent_v = float(sp.get('ped_intent_v', 0.0))
        self.ped_emg_ratio = float(sp.get('ped_emergency_ratio', 0.0))
        # 보행자 래치 **위치 기반 해제** (A-1). 0 이면 비활성 = 지나감·관측 끊김으로만
        # 해제하는 이전 동작. 실측 2026-09-02 실전주행_교통류_02_좌회전8 id4: 횡단을
        # 마치고 |lat| 8.8 m 에 서 있는데도 래치가 남아 로그 끝(118 s)까지 정지했다.
        _hz = float(cfg['comm']['send_hz'])
        self.ped_release_lat = float(sp.get('ped_release_lat_m', 0.0))
        self.ped_release_ticks = int(round(float(sp.get('ped_release_s', 1.5)) * _hz))
        self.ped_stop_v = float(sp.get('ped_stop_v', 0.2))
        self.ped_offroad_lat = float(sp.get('ped_offroad_lat_m', 6.0))
        self.ped_backstop_ticks = int(round(float(sp.get('ped_backstop_s', 30.0)) * _hz))
        # 걷는 채로 등장한 보행자 래치 (A-4). false 면 정지 관찰(ped_static) 전제 그대로.
        self.walkin_enable = bool(sp.get('ped_walkin_enable', False))
        self.walkin_v = float(sp.get('ped_walkin_v', 0.5))
        self.walkin_ticks = int(round(float(sp.get('ped_walkin_s', 0.5)) * _hz))
        self.walkin_lat = float(sp.get('ped_walkin_lat_m', 8.0))
        # 다중 보행자 회랑 홀드 (P4-M, 2026-09-04 실주행 101405). false = 이전 동작:
        # 회랑 안 보행자에게도 정지 프로파일 √(2·a·d_eff) 만 주어 12 m 앞에서 5.1 m/s
        # 를 허용했고, 적신호 IDM 후보(1.2→2.3)가 더 낮아 그쪽이 이기며 재가속했다.
        # 회랑(|lat| < ped_release_lat_m — A-1 해제와 같은 축) 안 보행자는 홀드 래치로
        # v_allow = 0 을 강제한다. 해제는 A-1 규칙(_ped_release_tick) 그대로.
        self.ped_multi = bool(sp.get('ped_multi_enable', False))
        # 관측 드롭아웃 coast — 이 시간 안의 미관측은 '이탈' 이 아니다 (직전 기여 유지).
        self.ped_coast_ticks = int(round(float(sp.get('ped_hold_coast_s', 0.5)) * _hz))
        # 횡단보도 앞 서행 (A-3). false 면 비활성 = 서 있는 보행자는 PDM forecast 에만.
        self.cw_enable = bool(sp.get('ped_crosswalk_creep_enable', False))
        self.cw_zone_m = float(sp.get('ped_crosswalk_zone_m', 10.0))
        self.cw_lat_m = float(sp.get('ped_crosswalk_lat_m', 4.0))
        self.cw_wait_ticks = int(round(float(sp.get('ped_crosswalk_wait_s', 3.0)) * _hz))
        self.cw_creep_v = float(sp.get('ped_crosswalk_creep_v', 2.0))
        self.a_emergency = float(sp.get('a_emergency', -8.0))
        self.a_dec_max = abs(float(cfg['control']['a_dec_max']))
        self.y_guard_max_m = float(sp.get('yellow_guard_max_m', 60.0))
        # ap.config 를 못 읽는 환경(목)에서만 쓰는 폴백 — 정상 경로는 PDM 값 사용
        self.stop_gap_sl_fallback = float(sp.get('stop_gap_stopline_m', 1.5)) + self.front
        # 방향지시등 (SPEC §3.3). lc_lead_s 는 규정 미확정 가정값 (§7-2).
        sig = cfg['signal']
        self.turn_lead_s = float(sig['turn_lead_s'])
        self.lc_lead_s = float(sig['lc_lead_s'])
        # 회전 지시등 **꼬리**가 반대 방향 차로 이동에 양보할지 (2026-09-09).
        # 회전 구간은 연결로 끝까지 유지되는데(_turn_end_s), 그 직후 차로 인계가
        # 오면 선행 점등 시간이 남지 않는다. 실측 20260909_093415/실경로_02:
        # 우회전 연결로가 rs 638.1 에 끝나고 인계 경계가 **647.2** 라 좌측 점등이
        # 9.1 m(1.5 s)뿐 — scoring.signal_lead_s(3.0) 미달로 항목 13 이 났다.
        # false = 이전 동작(회전이 언제나 우선).
        self.sig_tail_yield = bool(sig.get('signal_turn_tail_yield_enable', False))
        self.sig_lead_min_m = float(sig['lead_min_m'])
        self.lat_on_m = float(sig['lat_shift_on_m'])
        self.sig_min_on_ticks = int(round(float(sig['min_on_s'])
                                          * float(cfg['comm']['send_hz'])))
        self.sig_off_delay_ticks = int(round(float(sig['off_delay_s'])
                                             * float(cfg['comm']['send_hz'])))
        # 정적 장애물 회피 시프트 (SPEC §3.4 회피 — PDM 원문은 stub)
        ot = cfg['overtake']
        self.hz = float(cfg['comm']['send_hz'])
        # 규칙 1 — 신호 구역 억제
        self.sup_m = float(ot.get('stopline_suppress_m', 30.0))
        # span_into_zone 게이트의 정지선 앞 여유 (B-1). _signal_zone 의 sup_m 과는
        # 값을 공유만 했지 결합이 없었다 — 분리해 따로 튜닝한다. 0 = 정지선 자체.
        self.zone_gate_margin = float(ot.get('zone_gate_margin_m', 0.0))
        self.queue_gap_min_m = float(ot.get('queue_gap_min_m', 3.0))
        self.queue_lat_max_m = float(ot.get('queue_lat_max_m', 1.5))
        # 규칙 3 — 선제 회피
        self.detect_max_m = float(ot.get('detect_max_m', 80.0))
        self.shift_latest_m = float(ot.get('shift_latest_m', 10.0))
        # standoff 프로파일 전용 바닥 — shift_latest_m 과 소비처를 나눈다.
        # 그 값은 _shift_speed_cap 의 미리보기 창(look)과 _try_overtake 의
        # PREEMPT 시간 예산에도 쓰이므로, standoff 정지 거리만 조정하려면
        # 여기를 쓴다. 미지정이면 shift_latest_m 을 그대로 따른다(무변화).
        self.standoff_floor_m = float(ot.get('standoff_floor_m', self.shift_latest_m))
        # 기준선 안쪽 크립 바닥 (_standoff_creep). 꺼지면 이전 동작(0 고정).
        self.standoff_creep = bool(ot.get('standoff_creep_enable', False))
        self.standoff_creep_v = float(ot.get('standoff_creep_v', 0.8))
        self.standoff_creep_gap_m = float(ot.get('standoff_creep_gap_m', 1.0))
        # 크립 지연 게이트 (_creep_gate). 0 = 지연 없음(즉시 크립) = 이전 동작.
        # 시간 축은 ot_blocked_ticks 를 그대로 쓴다 — 신규 시계를 만들지 않는다.
        self.creep_delay_ticks = int(round(
            float(ot.get('standoff_creep_delay_s', 0.0)) * self.hz))
        # 교차로 안 자기잠금 해제 (A1). 꺼지면 이전 동작 — 교차로 lane 에서
        # 크립도 사다리도 없다. 무장 조건은 _junction_release 참조.
        self.j_release = bool(ot.get('junction_creep_release_enable', False))
        self.j_release_ticks = int(round(
            float(ot.get('junction_release_s', 8.0)) * self.hz))
        # km/h 로 받는 이유: 이 둘은 "교차로를 기어서 지난다" 는 주행 감각의
        # 값이라 사람이 읽는 단위가 낫다. 내부는 전부 m/s 이므로 여기서 한 번만
        # 바꾼다 (standoff_creep_v 는 제동거리 계산에서 온 값이라 m/s 그대로다).
        self.j_creep_floor = float(ot.get('junction_creep_floor_kph', 3.0)) / 3.6
        self.j_creep_cap = float(ot.get('junction_creep_kph', 5.0)) / 3.6
        # 절대 안 멈춤 (A2). 꺼지면 이전 동작 — 시계도 안 돈다.
        self.ns_enable = bool(ot.get('never_stall_enable', False))
        self.ns_max_ticks = int(round(float(ot.get('deadlock_max_s', 20.0)) * self.hz))
        self.ns_step_ticks = int(round(float(ot.get('never_stall_step_s', 5.0)) * self.hz))
        self.ns_creep_v = float(ot.get('never_stall_creep_kph', 3.0)) / 3.6
        self.ns_force_v = float(ot.get('never_stall_force_v', 1.0))
        self.ns_turn_hold_ticks = int(round(
            float(ot.get('never_stall_turn_hold_s', 6.0)) * self.hz))
        self.ns_turn_margin_m = float(ot.get('never_stall_turn_margin_m', 10.0))
        # never_stall 의 종점 배제 폭 — route_end 래치와 같은 축(unlatch_m).
        # active_m(150)은 후보 생성 창이라 여기 쓰면 마지막 150 m 가 통째로 죽는다.
        self.ns_end_m = float(cfg['route_end']['unlatch_m'])
        # 종료 구간 정적 장애물 무시 (실주행 1차 [1](b)). 꺼지면 이전 동작.
        self.fg_ignore = bool(sp.get('finish_gate_ignore_enable', False))
        self.fg_m = float(sp.get('finish_gate_m', 10.0))
        # 게이트를 **자차 위치**가 아니라 **객체 위치**로 판정할지 (2026-09-09 [1]).
        # 왜: 자차 기준이면 "자차가 종료 구간에 들어왔을 때"만 무시가 시작되는데,
        # 회피 시프트는 그보다 훨씬 앞에서 결정된다. 실측 20260909_093415/실경로_02 —
        # 종료선 콘 2개의 **객체 route_s 가 749.2 로 종료선과 같은데**, 자차 rs 700
        # (s_rel 49.3) 에서 이미 시프트가 생성돼 rs 723~751 을 0.9 m/s 로 25 초
        # 왕복했다. 자차 기준 게이트를 아무리 넓혀도(30 → 49.3 초과 필요) 그
        # 순간을 못 덮는다 — 판정 축 자체가 틀렸다.
        # 객체 기준이면 그 콘은 **감지 즉시**(rs 192.5, 556 m 앞) 빠진다.
        # false = 이전 동작(자차 route_s 기준).
        self.fg_by_obj = bool(sp.get('finish_gate_by_object_enable', False))
        # A3: 미보고 신호 시한이 **정지 앞차가 있어도** 돌게 한다. off = 이전 동작.
        self.sig_lead_ok = bool(ot.get('signal_timeout_with_lead_enable', False))
        # 연결로 곡률 감속 (B1). 꺼지면 후보를 만들지 않는다 = 이전 동작.
        self.curv_cap = bool(sp.get('curvature_cap_enable', False))
        self.curv_look_m = float(sp.get('curvature_lookahead_m', 15.0))
        # 속성명이 self.a_lat_max 이면 **안 된다** — 그 이름은 _shift_speed_cap 이
        # overtake.a_lat_max(1.5)로 쓰고 있고, 그 대입이 여기보다 뒤라 조용히
        # 덮어쓴다 (2026-09-08 실제로 그렇게 짜서 v_cap 이 4.03 대신 3.12 로
        # 나왔다 — √(1.5·6.5)). params 키도 같은 이유로 curvature_ 접두어다.
        self.curv_a_lat = float(sp.get('curvature_a_lat_max', 2.5))
        self.curv_min_k = float(sp.get('curvature_min_kappa', 0.005))
        # 차선변경 구간 속도 상한 ([3](a)). 꺼지면 후보를 만들지 않는다.
        self.lc_cap = bool(sp.get('lc_speed_cap_enable', False))
        self.lc_cap_v = float(sp.get('lc_speed_cap_kph', 20.0)) / 3.6
        self.lc_chain_sep_m = float(sp.get('lc_hop_chain_sep_m', 25.0))
        self.lc_look_m = float(sp.get('lc_speed_cap_look_m', 60.0))
        # 시프트가 **대상을 잃으면** 즉시 원복할지 (2026-09-09 [1]).
        # 지금은 `route_index > span[1]` 에서만 원복하는데, span 끝이 경로 끝
        # 근처면 그 지점을 못 밟고 시프트가 영원히 남는다 — 실측
        # 20260909_093415/실경로_02: span [7274,7760] = rs 776.0 인데 종료선이
        # 749.2 다. rs 722.6 부터 blocker 가 None 인데도 시프트를 쥔 채
        # 종료 차로(-1) 대신 -2 로 끝났다(횡오프셋 −3.07 m, 차로유지 −3).
        # 콘과 무관한 일반 버그라 종료 구간에 한정하지 않는다. 다만 조건은
        # **좁게** 둔다 — "회랑이 비었다" 는 시프트 중에 항상 참이라(밀린 경로
        # 기준 회랑에는 피한 물체가 안 들어온다) 그것만 보면 모든 시프트가
        # 즉시 원복돼 왕복이 된다. 그래서 **시프트를 만든 그 id 들이 전부
        # 사라졌을 때**만 푼다 (게이트로 빠졌거나 월드에서 없어졌거나).
        # false = 이전 동작.
        self.span_lost_restore = bool(ot.get('span_lost_restore_enable', False))
        self.ot_ids: list = []                     # 이 시프트를 만든 객체 id
        self.lm_hop_n = 0                          # 이번 시프트의 칸 수 (커밋 B)
        self.shift_k_s = float(ot.get('shift_k_s', 3.0))
        self.shift_ahead_m = float(ot.get('shift_ahead_m', 5.0))
        self.obj_static_ticks = int(round(float(ot.get('obj_static_s', 3.0)) * self.hz))
        # standoff 대상 정지 카운터 (B-5) — 신호와 무관하게 센다. 0 이면 이전 동작
        # (standoff 대상 = 정지 관찰을 마친 회랑 객체).
        self.standoff_stop_ticks = int(round(float(ot.get('standoff_stop_s', 1.5)) * self.hz))
        self.obj_grace = int(ot.get('obj_grace_ticks', 10))
        # 대기열 판별기 (3중 교정)
        self.q_clear_m = float(ot.get('queue_min_clear_m', 0.3))
        self.q_head_m = float(ot.get('queue_head_max_m', 25.0))
        self.q_hold_ticks = int(round(float(ot.get('queue_hold_s', 15.0)) * self.hz))
        # 억제 단일화 (C). 'queue_only' = 억제는 _is_queue 하나, 적색·정지선·교차로는
        # 일시정지/게이트 입력. 'legacy' = 이전 3중 억제(_red_ahead·_signal_zone·_is_queue)
        # 를 바이트 동일하게 유지.
        self.suppress_mode = str(ot.get('suppress_mode', 'queue_only'))
        self.q_green_release_ticks = int(round(float(ot.get('q_green_release_s', 3.0)) * self.hz))
        self.q_nosig_release_ticks = int(round(float(ot.get('q_nosignal_release_s', 10.0)) * self.hz))
        # 신호 미보고 큐 판정 (2026-09-06, 020738/13 근거) — kill switch 기본 off.
        # 9910 은 교차로 연결로 위에서 신호를 보내지 않는다 (실측 16런: in_junction
        # 틱 보고율 0.0~2.8 %). 못 받은 신호는 플래너 기본값 Green 으로 남아, 교차로
        # 출구 정지선의 적신호 대기열이 큐(green_expired 로 해제)가 아니라 장애물로
        # 보였다 → 회피 → junction 기각 → 고착. 다음 정지선 controller 가
        # signal_stale_s 이상 미보고면 UNKNOWN 으로 보고 큐 판정에서 Red 와 같이
        # 다룬다. 정지 후보·SHIFT_HOLD 는 건드리지 않는다 — UNKNOWN 은 Green 을
        # 만들지도 Red 를 풀지도 않는다. 보고 시각은 observe_lights (run_agent) 가
        # 준다 — 신호 state 갱신은 그대로 플래너(route.update_lights) 몫이다.
        self.sig_stale_queue = bool(ot.get('signal_stale_queue_enable', False))
        self.sig_stale_ticks = int(round(float(ot.get('signal_stale_s', 1.0)) * self.hz))
        self._obs_tick = 0                          # observe_lights 호출 수 (= 9910 프레임)
        self._light_seen: dict = {}                 # controller id → 마지막 보고 _obs_tick
        self.last_signal: dict | None = None        # 진단 — 스위치 on 일 때만 채운다
        # B-3(b) 미보고 적신호 시한 출발 (2026-09-06 승인) — kill switch 기본 off.
        # 앞차 없음 ∧ controller 미보고 ∧ 정지 중이 signal_unknown_timeout_s 이상이면
        # 신호 정지 후보(_stop_target·PDM 적신호 IDM)를 놓고 min() 이 다른 후보로
        # 속도를 정하게 둔다. **적신호 통과를 만들 수 있는 유일한 경로다**: 71 s
        # 주기·적색 60 s 인 controller(167/168)에서는 시한이 얼마든 85 % 확률로
        # 적색 통과다. 회랑 안 signal_timeout_clear_m 이동 차량·보행자 래치가 있으면
        # 보류. 020738 16런에서 성립 틱 0 (Red 로 선 뒤 끊긴 사례 없음).
        self.sig_timeout_go = bool(ot.get('signal_timeout_go_enable', False))
        self.sig_timeout_ticks = int(round(float(ot.get('signal_unknown_timeout_s', 10.0)) * self.hz))
        self.sig_timeout_clear_m = float(ot.get('signal_timeout_clear_m', 30.0))
        self._sig_wait_ticks = 0                    # 조건 성립 연속 틱
        self._sig_go = False                        # 시한 만료 → 신호 정지 후보 해제 (래치)
        self._sig_go_tl = None                      # 래치가 붙은 신호 id
        # RTOR 적색 신호 우회전 (2026-09-06 승인) — kill switch 기본 off.
        # 정지선 앞 정지(항목 7)를 마친 뒤 선두 ∧ 다음 기동 우회전 ∧ 보행자·교차
        # 차량 없음이면 _rtor_go 래치. 소비처는 B-3 와 같은 두 곳(_stop_target ·
        # signal_release)뿐이고, 래치는 (신호 id, 정지선 s) 에 붙어 다음 신호에는
        # 적용되지 않는다. off 면 _rtor_go 가 영원히 False 라 이전과 틱 단위 동일.
        self.rtor_enable = bool(sp.get('rtor_enable', False))
        self.rtor_allow_stale = bool(sp.get('rtor_allow_stale_red', True))
        self.rtor_exclude = {int(x) for x in (sp.get('rtor_exclude_tl_ids') or [])}
        self.rtor_turn_win_m = float(sp.get('rtor_turn_event_window_m', 5.0))
        self.rtor_stop_v = float(sp.get('rtor_stop_v_max', 0.3))
        self.rtor_zone_m = float(sp.get('rtor_stop_zone_m', 2.0))
        self.rtor_hold_ticks = int(round(float(sp.get('rtor_stop_hold_s', 1.0)) * _hz))
        self.rtor_ped_guard_m = float(sp.get('rtor_ped_guard_m', 8.0))
        self.rtor_cross_gap_m = float(sp.get('rtor_cross_gap_m', 40.0))
        self.rtor_cross_ttc_s = float(sp.get('rtor_cross_ttc_s', 6.0))
        self.rtor_go_v = float(sp.get('rtor_go_speed_kph', 15.0)) / 3.6
        self.rtor_release_m = float(sp.get('rtor_release_dist_m', 30.0))
        self._rtor_go = False                       # 래치 (B-3 _sig_go 와 별도)
        self._rtor_go_tl = None                     # 래치가 붙은 신호 id
        self._rtor_stop_s = None                    # 래치 시점 정지선 route_s
        self._rtor_junction_seen = False            # 래치 후 교차로 차로 진입 관측
        self._rtor_hold_cnt = 0                     # 정지 구역 안 정지 누적 틱
        # WAIT — 앞차 출발 기회
        self.wait_s = float(ot.get('wait_before_shift_s', 6.0))
        self.ot_dash_slack_m = float(ot.get('dash_slack_m', 2.0))
        # 시프트 전이 횡가속 상한 (P1). 0 이면 비활성.
        self.a_lat_max = float(ot.get('a_lat_max', 0.0))
        self.shift_cap_min_v = float(ot.get('shift_cap_min_v', 1.0))
        # 시프트 기하 계단 검사 (B-7 임시 가드). 0 이면 계측만, 기각 없음.
        self.shift_k_reject = float(ot.get('shift_kappa_reject', 0.0))
        self.shift_k_step_m = float(ot.get('shift_kappa_step_m', 1.0))
        # 계획 LC 중첩 검사 (B-11). 0 이면 계측만, 기각 없음.
        self.lc_overlap_m = float(ot.get('shift_lc_overlap_m', 0.0))
        # 시프트 기하 완성 게이트 (B-12). false 면 완전 비활성 = 이전 동작.
        self.geom_gate = bool(ot.get('shift_geom_gate_enable', False))
        self.geom_margin_m = float(ot.get('shift_geom_margin_m', 0.0))
        # 게이트 ↔ BREAKOUT 단계 연동 (B-2). 단계가 이 값 이상이면 그 게이트를
        # 완화한다. 99 같은 큰 값이면 어느 단계에서도 완화하지 않는다 (= 이전 동작).
        self.zone_relax_lvl = int(ot.get('zone_gate_relax_level', 2))
        self.geom_relax_lvl = int(ot.get('geom_relax_level', 3))
        self.shift_ahead_l3_m = float(ot.get('shift_ahead_l3_m', 1.0))
        # solid 두 바퀴 (B-3). false 면 1바퀴만 = 이전 동작 (실선은 절대 안 넘는다).
        self.solid_second_pass = bool(ot.get('solid_second_pass_enable', True))
        # 규칙 2 — 데드락 해제 (BREAKOUT)
        self.BO_CREEP = 4                                  # 크립이 켜지는 단계
        # E-8 ②: L2 zone 완화가 푸는 사유. 회전·차선변경·통과 차로 없음은 여기 없다.
        self.ZONE_RELAXABLE = ('span_into_zone', 'zone_no_exit', 'zone_extend_max')
        self.bo_enabled = bool(ot.get('breakout_enabled', False))
        self.bo_eps = float(ot.get('stuck_eps', 0.2))
        self.bo_hard_ticks = int(round(float(ot.get('stuck_hard_s', 10.0)) * self.hz))
        self.bo_esc_ticks = int(round(float(ot.get('escalate_s', 2.0)) * self.hz))
        self.bo_fail_ticks = int(round(float(ot.get('creep_fail_s', 6.0)) * self.hz))
        self.bo_creep_v = float(ot.get('creep_v', 1.0))
        self.bo_progress_m = float(ot.get('progress_m', 2.0))
        self.bo_creep_eps_m = float(ot.get('creep_progress_eps_m', 0.3))
        self.ot_enabled = bool(ot['enabled'])
        self.ot_v_max = float(ot['blocker_speed_max'])
        self.ot_d_max = float(ot['blocker_dist_max'])
        self.ot_ticks = int(round(float(ot['trigger_s']) * float(cfg['comm']['send_hz'])))
        self.ot_min_corridor = float(ot['min_corridor_m'])
        self.ot_clear_r = float(ot['clear_radius_m'])
        self.ot_trans_m = float(ot['transition_m'])
        self.ot_before_m = float(ot['extra_before_m'])
        self.ot_after_m = float(ot['extra_after_m'])
        # 연쇄 장애물 병합 (B-9). 회랑에서 다음 정지 객체가 앞 객체 + 이 거리 안이면
        # 한 span 으로 묶는다. 기본 = 뒤여유 + 전이 = 22: 그보다 가까우면 복귀 전이가
        # 다음 객체 위에 떨어진다. 0 = 비활성 (단일 객체 span = 이전 동작).
        self.chain_gap_m = float(ot.get('chain_gap_m', self.ot_after_m + self.ot_trans_m))
        # span 활성 중에도 회랑·standoff·막힘 회계를 돌린다 (B-9 (5)). false = 이전 동작.
        self.span_active_standoff = bool(ot.get('span_active_standoff_enable', True))
        # ── E: 정적 장애물 반응성 (2026-09-03). 각각 false/0 이면 이전 동작. ──
        # E-1 장애물 클래스(world.classify 의 cls=='obstacle') fast path — 관찰 없이
        # 즉시 정적, 큐 판정 대상 제외.
        self.obs_fastpath = bool(ot.get('obstacle_class_fastpath_enable', False))
        # E-2 span_into_zone 연장 — 정지선 뒤 교차로 출구 + 여유까지 extra_after 확장.
        self.zone_extend = bool(ot.get('zone_extend_enable', False))
        self.zone_extend_max_m = float(ot.get('zone_extend_max_m', 120.0))
        self.zone_exit_margin_m = float(ot.get('zone_exit_margin_m', 5.0))
        self.zone_junction_gap_m = float(ot.get('zone_junction_gap_m', 5.0))
        # E-3 BREAKOUT 시계를 첫 기각부터 (주행 중 포함).
        self.bo_reject_clock = bool(ot.get('breakout_reject_clock_enable', False))
        # E-6 SHIFT_HOLD: 원복 검사 우선 + 홀드 중 standoff·회계.
        self.hold_restore = bool(ot.get('shift_hold_restore_enable', False))
        # E-7 적색 일시정지 거리 상한 [m] (관찰 pause·큐 B·BREAKOUT pause). 0 = 무제한.
        self.red_pause_max_m = float(ot.get('red_pause_max_m', 0.0))
        # E-4 예산 소진 래치 — 한 번 PREEMPT/WAIT_EXPIRED 가 된 차단물은 기각돼도
        # 다음 틱 WAIT 로 되돌아가지 않는다. t_left 는 속도에 따라 출렁여(standoff 감속
        # 중 v↓ → t_left↑) 예산 3 s 에서는 한 틱 기각 뒤 WAIT 로 튀어 E-3 시계가 끊겼다
        # (replay 020439/01 t=38.0 PREEMPT → 38.05 WAIT → 39.85 WAIT_EXPIRED).
        self.preempt_latch = bool(ot.get('preempt_latch_enable', False))
        self.preempt_latch_id = None
        # E-8 ① 마킹 'none' 구간은 점선과 같이 넘을 수 있다 — 선이 없으면 위반이 아니다.
        # 실측 020439/02: 정지선 앞 none 구간(lane-local 35.9~57.7)이 커버리지에 안 잡혀
        # 1바퀴 solid 기각 → 2바퀴(실선 생략)로 생성됐다. false = 이전 동작(점선만).
        self.none_crossable = bool(ot.get('none_marking_crossable', False))
        # E-8 ② L2 zone 완화 범위 한정 — zone_no_exit / zone_extend_max (와 평가 불가
        # span_into_zone) 만 해제. zone_turn / zone_lane_change / zone_no_through_lane 은
        # 경로가 그 교차로에서 회전·차선변경·통과를 요구하므로 전 단계·전 바퀴 유지.
        # false = 이전 동작 (L2 부터 zone 게이트 전체 생략).
        self.zone_relax_limited = bool(ot.get('zone_relax_limited', False))
        # span 국소성 게이트 (2026-09-03 실주행 100310/100458). `plan_shift_span` 은
        # `original_route_points[route_index:]` 전체에 cKDTree 를 세워 차단물에 가장
        # 가까운 경로점을 잡는다 (PDM 원문 privileged_route_planner 447~453 과 동일).
        # 순환 코스에서 경로가 같은 자리를 두 번 지나면 4.5 km 앞 경로점이 잡혀
        # (실측 13건 4550~4595 m) 자차 앞이 아니라 한 바퀴 뒤 구간이 밀리고, 자차는
        # SHIFT_ACTIVE 로 원복만 기다리며 서 있었다 (100310 158틱). 시프트 시작점이
        # 자차 route_index 보다 이 거리 이상 앞이면 그 span 을 기각한다. 정상 시프트
        # 실측은 5.0~34.7 m (104648/104807/배치). route.py 의 근본 수정은 범위 밖.
        # false = 이전 동작 (거리 무관).
        self.span_gate = bool(ot.get('span_gate_enable', False))
        self.span_gate_max_m = float(ot.get('span_gate_max_m', 100.0))
        self.last_span_plan: tuple | None = None   # _planned_shift_geom 의 (a, b, left)
        # 전이 배치 (2026-09-03 실주행 100310/100458). 전이 시작은 자차 +shift_ahead_m
        # 고정이라 중간 객체를 스친다 — 100310: 우측 차로 11 m 의 id3 를 OBB 기준 −0.44 m
        # (중심선 기준 0.08 m). 복귀 끝도 표적 +extra_after_m 고정이라 100458 은 복귀
        # 전이가 우측 차로 41 m 의 id5 를 −0.33 m 로 친다. 시프트 폭 D(s)·cos 전이로
        # 자차 궤적을 미리 그려 진입 지연(delay)·복귀 단축(after)을 골라 최소 이격을
        # 최대화한다. 기본 배치가 이미 임계(percep.obstacle_clearance_m) 이상이면 현행
        # 그대로. 전이 길이는 줄이지 않는다. false = 이전 동작.
        self.shift_entry = bool(ot.get('shift_entry_enable', False))
        # gap_fit — 옆 차로 빈 구간에 맞춰 (전이, 앞뒤 여유) 를 다시 고른다.
        self.gap_fit = bool(ot.get('gap_fit_enable', False))
        self.gap_fit_clear_m = float(ot.get('gap_fit_clear_m', 0.6))
        self.gap_fit_step_m = float(ot.get('gap_fit_trans_step_m', 2.0))
        # gap_fit 이 고른 전이를 실제로 만들 수 있는 속도까지 먼저 줄인다.
        self.gap_fit_speed = bool(ot.get('gap_fit_speed_enable', False))
        self.gap_v_req: float | None = None        # 이번 틱 전이 길이 기반 속도 상한
        self.gap_hold_ticks = 0                    # span 생성 보류 누적 (진단용)
        self.entry_max_delay_m = float(ot.get('shift_entry_max_delay_m', 15.0))
        self.entry_step_m = float(ot.get('shift_entry_step_m', 0.5))
        self.exit_min_after_m = float(ot.get('shift_exit_min_after_m', 0.0))
        # side 선택 (2026-09-04 실주행 104648/104807) — _side_pass 의 좌측 고정 순서를
        # "양측을 재고 고르기" 로 바꾼다. **shift_entry 가 켜져 있어야 의미가 있다** —
        # 1차 키인 entry_plateau_ids 가 _shift_placement 산출물이라, 꺼져 있으면
        # 기준 자체가 없어 현행(좌측 우선)으로 폴백한다. 자세한 근거는 _pick_side.
        self.side_pick = bool(ot.get('side_pick_enable', False))
        # span 연장 (2026-09-03 배치 연쇄장애물_01). 시프트 생성 시 detect_max_m 밖이던
        # 연쇄 뒷부분(id6·id7)이 전진하며 보이는데, span 은 연쇄 한가운데(608.3)서 끝나
        # 복귀 직후 id7(618.9)을 만났다. 활성 span 은 새 시프트를 막고(요동 방지, 유지)
        # 원복은 span 끝을 지나야 성립하는데 standoff(25 m)가 속도를 0 으로 잡아 12.7 m 를
        # 못 갔다 — 32 s 고착·미완주. 원복은 반증됐다(자차가 span 안이라 발밑 경로 2.7 m
        # 횡점프, 원복해도 id6 5.3 m 가 geom 미달). 대신 **span 끝 뒤쪽만** 연장한다:
        # shift_route_smoothly 는 현재 경로점을 기준으로 f·loc + (1−f)·cur 로 섞고 f 가
        # 시작점에서 0 이므로 자차 인덱스에서 다시 적용하면 발밑은 그대로(실측 변화
        # 0.000000 m)이고 앞쪽만 밀린다. 네 조건(활성·차단물이 span 밖·정지 지속·
        # **연장 후 예상 이격 ≥ 임계**)을 다 만족할 때만. 네 번째가 없으면 02_우회전처럼
        # 엇갈린 배치에서 뒤쪽 객체(id5 0.86→0.22)를 끌어들인다. false = 이전 동작.
        self.span_extend = bool(ot.get('span_extend_enable', False))
        self.ext_stuck_ticks = int(round(float(ot.get('span_extend_stuck_s', 4.0)) * self.hz))
        self.ext_max_n = int(ot.get('span_extend_max_n', 3))
        self.ot_side: str | None = None            # 활성 span 의 방향 (연장에 같은 방향 사용)
        self.span_extend_n = 0                     # 현재 span 의 연장 횟수

        self.latched = False
        self.stop_s: float | None = None           # 시작 시 1회 계산 캐시 (매 틱 투영 금지)
        self.finish_s: float | None = None         # 종료선 route_s (종료 구간 게이트용)
        self.sl_hold_left = 0                      # 정지선 홀드 잔여 틱
        self.sl_stopped = False                    # 정지 연속성 (B-1 재무장 판정)
        self.sl_stop_ticks = 0                     # 현재 정지의 지속 틱
        self.last_candidate: float | None = None   # 이번 틱 route_end 후보 (로그용)
        self.last_target: float | None = None      # 이번 틱 최종 목표속도 (로그용)
        self.last_d_end: float | None = None
        self.last_stop_profile: float | None = None   # 이번 틱 정지 프로파일 상한 (로그용)
        # 이번 틱 최종 목표를 정상상태 상한(곡률·LC)이 정했나 (실주행 2차 [1], 로그용)
        self.last_cap_binds: bool = False
        # 순수 제한속도(아무도 안 줄인 틱)도 상한으로 실행할지 (실주행 2차 [1]).
        # false = 이전 동작 (곡률·LC·붉은구간 접근만 상한 축).
        self.cap_limit = bool(cfg['control'].get('cap_limit_target_enable', False))
        # ── 차로 지도 (커밋 A, 읽기 전용) ────────────────────────────────
        _lm = cfg.get('avoid_map') or {}
        self.lane_map_on = bool(_lm.get('lane_map_avoid_enable', False))
        self.lane_map_hops = int(_lm.get('lane_map_max_hops', 2))
        self.lane_map_ahead_m = float(_lm.get('lane_map_ahead_m', 80.0))
        self.lane_map_min_w = float(_lm.get('lane_map_min_width_m', 2.0))
        # 커밋 B — 후보 선택·시점·감속.
        self.lm_decide_m = float(_lm.get('lane_map_decide_m', 50.0))
        self.lm_avoid_v = float(_lm.get('lane_map_avoid_speed_kph', 20.0)) / 3.6
        # 커밋 C — 복귀 없음. 복귀 전이를 장애물 직후가 아니라 **데드라인**에 둔다.
        self.lm_no_return = bool(_lm.get('lane_map_no_return_enable', False))
        self.last_lane_plan: dict | None = None    # 이번 틱 결정 (진단·후보)
        # "정지 객체" 임계는 회피 계층과 같은 출처를 읽는다 (상수 복제 금지).
        self.lane_map_static_v = self.ot_v_max
        self.last_lane_map: dict | None = None
        # ── 적색 점멸 일시정지 (B3, 항목 9) ──────────────────────────────
        self.flash_stop = bool(sp.get('signal_flash_stop_enable', False))
        # 유지시간은 규정(0.5 s)보다 길게 잡는다 — 채점기가 0.5 s 를 **넘겨야**
        # 인정하므로 딱 맞추면 틱 경계에서 놓친다. speed.flash_hold_s 가 정본.
        self.flash_hold_ticks = int(round(float(sp.get('flash_hold_s', 0.8)) * self.hz))
        # 판정 임계는 **채점기와 같은 출처**를 읽는다 (scoring.*). 여기서 새
        # 상수를 만들면 같은 주행이 제어기 기준과 채점 기준으로 갈린다.
        # (임계는 scoring.*, 정지 속도는 score.stop_speed_mps — 채점기가 읽는
        #  자리가 둘로 갈려 있어 여기서도 그대로 따라간다)
        _sc = cfg.get('scoring') or {}
        self.flash_ok_m = float(_sc.get('stop_ok_m', 2.0))
        self.flash_stop_v = float((cfg.get('score') or {}).get('stop_speed_mps', 0.5))
        self.flash_latch_tl: int | None = None      # 통과 허가가 난 신호 id
        self.flash_hold_n = 0                       # 그 신호 앞 정지 유지 틱
        self.flash_tl = None                        # 지금 세고 있는 신호 id
        self.last_flash: dict | None = None         # 진단 (로그용)
        # 황색 원샷 판정 래치 (접근당 1회, 번복 금지)
        self.y_decision: str | None = None            # None | 'stop' | 'go'
        self.y_ctrl: int | None = None                # 래치가 걸린 신호 id
        self.y_v_allow: float | None = None           # 판정 시 v_allow (로그용)
        self.last_yellow: dict | None = None          # 판정 순간 1틱만 채운다 (로그용)
        # 교차로 통과 가드
        self.cross_guard = False
        self.cross_s: float | None = None
        self.cross_junction_seen = False
        self._ap = None                               # 홀드가 _stop_target 을 부르려면 필요
        self.sig_plan: list[dict] | None = None    # 회전 구간 (시작 시 1회)
        self.sig_on_ticks = 0                      # 켜진 뒤 지난 틱 (min_on 기준)
        self.sig_off_left = 0                      # 소등 지연 잔여 틱 (상한 있음)
        self.sig_held: int = SIG_OFF               # 유지 중인 값
        self.ot_blocked_ticks = 0                  # 막힌 채 정지한 틱
        self.ot_span: tuple | None = None          # 시프트한 인덱스 구간
        self.last_overtake: str | None = None      # 로그용 ('left'|'right'|사유@p{n})
        self.ot_pass_solid = False                 # 1바퀴에 solid 기각이 있었나 (B-3)
        self.last_avoid: dict | None = None        # 회피 진단 (reasons.avoid)
        self.obj_ticks: dict = {}                  # 객체별 정지 지속 틱 (신호 중 일시정지)
        self.obj_stop_ticks: dict = {}             # 객체별 정지 지속 틱 (신호 무관, B-5)
        self.obj_miss: dict = {}                   # 객체별 미관측 틱 (grace)
        self.standoff_id = None                    # 이번 틱 standoff 대상 id (진단)
        # 보행자 의도 감지 상태 (P4)
        self.ped_static: set = set()               # '정지 관찰' 을 마친 보행자 id
        self.ped_lat: dict = {}                    # id → 직전 틱의 경로 횡거리
        self.ped_intent: set = set()               # 의도 래치가 걸린 id
        self.ped_clear: dict = {}                  # id → 해제 조건 연속 틱 (A-1)
        self.ped_hold: dict = {}                   # id → 래치 유지 틱 (A-1 backstop)
        self.ped_diag: dict = {}                   # id → 이번 틱 진단 (lat/v_toward/…)
        self.ped_released: dict = {}               # id → 이번 틱 해제 사유
        self.last_ped: dict | None = None          # 이번 틱 진단 (reasons.ped)
        self.ped_emergency = False                 # 이번 틱 비상 제동 우회 여부
        # 다중 보행자 회랑 홀드 (P4-M) — ped_multi_enable 일 때만 채워진다
        self.ped_hold_ids: set = set()             # 회랑 홀드 래치가 걸린 id (v_allow = 0)
        self.ped_miss: dict = {}                   # id → 연속 미관측 틱 (coast)
        self.ped_last: dict = {}                   # id → 직전 관측 기여 (v_allow, a_req)
        self.ped_all: dict = {}                    # id → 이번 틱 전 보행자 평가 (reasons.ped.all)
        self._sl_all: list | None = None           # 경로상 전 정지선 route_s (1회)
        # BREAKOUT 상태
        self.bo_state: str | None = None           # None | 'BREAKOUT' | 'CREEP_FAIL'
        self.bo_level = 0
        self.bo_stuck_ticks = 0
        self.bo_stop_ticks = 0                     # 순수 정지(v<eps) 지속 틱 (E-3 안전 가드)
        self.ot_reject_ticks = 0                   # 회피 시도가 양쪽 다 기각된 연속 틱 (E-3)
        # reject='junction' **만** 센 연속 틱 (A1). ot_reject_ticks 로는 못 쓴다 —
        # 그건 사유를 안 가려서 right:no_neighbor 등에도 오르므로, 교차로가
        # 아닌 곳의 기각이 교차로 해제를 무장시킨다.
        self.j_reject_ticks = 0
        # never_stall (A2) — 원인 무관 **하나의** 무진전 시계. 신호·보행자·종점은
        # _ns_cause 게이트에서 빠지므로 여기 쌓이지 않는다.
        self.ns_ticks = 0
        self.ns_ref_s: float | None = None         # 마지막 진전 route_s
        self.ns_level = 0                          # 0 정상 / 1 크립 / 2 사다리 재진입 / 3 최후
        self.ns_turn_ticks = 0                     # (c) 회전 차로 복귀 보류 누적
        self.ns_info: dict | None = None           # 진단 (로그용)
        self.fg_dropped = 0                        # 종료 구간에서 회랑에서 뺀 정지 객체 수
        self.last_curv: float | None = None        # 이번 틱 곡률 상한 (로그·중재용)
        self.last_curv_info: dict | None = None    # 곡률 진단
        self.last_lc_cap: float | None = None       # 차선변경 구간 상한 (진단)
        self._lc_windows: list | None = None        # 병합된 LC 창 (경로당 1회)
        self.bo_lvl_ticks = 0
        self.bo_stall_ticks = 0
        self.bo_entry_s: float | None = None       # 진입 시 route_s (복귀 판정)
        self.bo_ref_s: float | None = None         # 마지막 진전 route_s (무진전 판정)
        self.bo_exit: str | None = None
        self.q_ticks = 0                           # 대기열 판정 지속 틱 (시한 철회)
        self.q_reject: str | None = None           # 대기열 기각 사유 (진단)
        self.q_info: dict | None = None            # 큐 판정 진단 (queue_only, C-3)
        self._tick_corridor: list = []             # 틱당 1회 캐시 — standoff 축 회랑 (C-2)
        self._tick_queue = False                   # 틱당 1회 캐시 — 큐 판정 (q_ticks 증가처)
        self._tick_lg = None
        self._tick_ego_lane = None
        self.green_since_ticks = 0                 # 다음 신호가 녹색인 연속 틱 (C-3 해제)
        self.green_tl_id = None
        self.ped_vt: dict = {}                     # 보행자 id → 직전 틱 v_toward (C-3 가드)
        self.ped_walkin: dict = {}                 # 보행자 id → 걷는 채 접근 연속 틱 (A-4)
        self.cw_wait: dict = {}                    # 보행자 id → 횡단보도 대기 정지 틱 (A-3)
        self._cw_zones: list | None = None         # 경로상 횡단보도 [route_s 구간] (1회)
        self.wait_target_d: float | None = None    # 관찰 감속 목표 (장애물까지 거리)
        # standoff 대상의 반길이 [m] — 크립의 '진짜 정지 거리' 가 객체 크기에
        # 비례해야 한다 (d 가 뒷축→객체 중심 축이라 앞범퍼+반길이를 더한다).
        self.standoff_half_len: float | None = None
        self._creep_diag: dict | None = None       # 크립 진단 (로그용, 판단 무관)
        # 크립 지연 시계 — **크립이 실제로 열릴 수 있는 상태에서 보류한 틱만** 센다.
        # ot_blocked_ticks 를 쓰면 안 된다: 그건 적신호 대기도 세므로 신호가
        # 녹색으로 바뀌는 순간 이미 만료돼 지연이 무의미해진다 (실측 2026-09-05
        # 155049 02_직진3: 녹색 복귀 시 hold_s 22.6 → 즉시 open_why=delay).
        self._creep_hold_ticks = 0
        self._creep_hold_id = None                 # 대상이 바뀌면 시계를 새로
        # 'delay' 로 연 크립의 래치 (2026-09-06, 020738/13 t=30.8·44.2 근거). 지연
        # 만료로 열면 그 틱에 _creep_hold_ticks 를 0 으로 되돌렸고, 다음 틱은 ③ 이
        # 다시 미달이라 닫혔다 — 1틱 개방, 이동 0 m, 10 s 뒤 반복. need·breakout
        # 은 조건 자체가 지속되므로 리셋이 무해했지만 delay 는 시계가 곧 조건이다.
        # 해제 = standoff 대상 변경 / 시프트 성립(ot_span 변화) / 배제(cause·
        # ped_hold — 기존 시계도 거기서 0 이 된다). stop_gap 은 크립 완료라 유지.
        self._creep_open_latched = False
        self._creep_latch_span = None              # 래치 시점의 ot_span
        self.bo_paused = False                     # 적색·황색STOP 중 일시정지
        self.last_turn_signal: int = SIG_OFF       # 이번 틱 지시등 (run_agent 가 읽는다)
        self.last_sig_src: str | None = None       # 'turn' | 'lc'
        self.last_sig_lead_s: float | None = None  # 이벤트까지 남은 시간 [s]

    def _resolve_stop_s(self, planner) -> float:
        """정지 목표 기준점 1회 산출. finish 모드 실패 시 경고 후 total 폴백."""
        total = float(planner.route['total_length'])
        if self.target_mode != 'finish':
            return total
        # params 우선, null 이면 build_route 가 pkl 에 넣은 CSV 마지막 행 자동
        # (route_end.finish_xy_from_route_enable=false 면 route 값 무시 = 기존 동작)
        fxy = self.finish_xy or (
            planner.route.get('finish_xy')
            if self.cfg['route_end'].get('finish_xy_from_route_enable', True) else None)
        if not fxy:
            print('[kr_rules] scoring.finish_xy 미설정 (route.pkl 에도 없음) — '
                  'route_total 기준으로 정지 (기존 동작)', flush=True)
            return total
        print('[kr_rules] finish_xy 출처: '
              + ('params(scoring.finish_xy)' if self.finish_xy
                 else 'route.pkl(CSV 마지막 행)'), flush=True)
        lg = getattr(planner, 'lg', None)
        finish_s = (_project_route_s(lg, planner.route,
                                     float(fxy[0]), float(fxy[1]))
                    if lg is not None else None)
        if finish_s is None:
            print('[kr_rules] finish_xy 를 경로에 투영하지 못함 — route_total 기준으로 정지',
                  flush=True)
            return total
        self.finish_s = finish_s                   # [1](b) 종료 구간 게이트가 읽는다
        stop_s, clipped = plan_stop_s(self.cfg, total, finish_s)
        if clipped:
            print(f'[kr_rules] ⚠ 계획 정지점이 종료선을 못 넘는다 — finish_s {finish_s:.1f} '
                  f'+ 여유가 경로 종점을 초과 (경로 꼬리 부족). 종점까지 주행한다', flush=True)
        return stop_s

    def _lane_shift(self, planner, ego_speed: float):
        """앞 창에서 경로가 차로 중심 기준으로 옆으로 갈 예정인가 → (sig, 남은거리).

        planner.lat_shift 는 경로점마다 "기준 차로 중심에서 밀린 양"(+좌/−우)이다.
        계획된 차선변경 블렌드와 **런타임 회피 시프트**가 둘 다 여기에 반영되므로,
        경로를 옆으로 미는 어떤 동작이든 지시등이 따라온다. 테이퍼(소멸 차로 기하
        보정)는 차로를 옮기는 게 아니라 제외돼 있다.

        한 점만 비교하지 않고 **창 전체를 훑는다** — 창 끝점이 이미 이동을 마친
        뒤라면 차이가 0 으로 나와 놓친다.
        """
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
        # 남은거리 = 임계를 처음 넘는 지점까지 (우선순위 비교용)
        over = np.nonzero(np.abs(seg) >= self.lat_on_m)[0]
        remain = float(over[0]) / ppm if over.size else 0.0
        return (SIG_LEFT if seg[k] > 0 else SIG_RIGHT), remain

    def _turn_signal(self, planner, route_s: float, ego_speed: float) -> tuple:
        """이번 틱 지시등 → (sig, src, lead_s).

        후보는 둘 — 회전(이벤트 구간)과 차로 이동(경로 기하). 겹치면 SPEC §3.3
        대로 남은거리가 짧은 쪽, 동률이면 회전 우선.

        깜빡임 방지는 **최소 점등 시간뿐**이다. 끄는 임계도 래치도 두지 않는다 —
        조건이 거짓이 되면 그대로 꺼진다. 유지 구간에 상한이 있으니 고착되지 않는다.
        """
        if self.sig_plan is None:
            self.sig_plan = turn_intervals(planner)

        best = None                                  # (정렬키, sig, src, remain)
        tail = False                                 # 승자가 회전의 **꼬리**인가
        for iv in self.sig_plan:
            if route_s > iv['end_s']:
                continue
            remain = iv['ev_s'] - route_s
            if remain > max(ego_speed * self.turn_lead_s, self.sig_lead_min_m):
                continue
            key = (max(0.0, remain), 0)              # 0 = 회전 우선
            if best is None or key < best[0]:
                best = (key, iv['sig'], 'turn', remain)
                # 회전 지점을 이미 지났으면 남은 구간은 연결로를 도는 **꼬리**다.
                # 회전 전 선행 점등이라는 목적은 이미 달성됐다.
                tail = route_s > iv['ev_s']

        shift = self._lane_shift(planner, ego_speed)
        if shift is not None:
            sig, remain = shift
            key = (max(0.0, remain), 1)
            if best is None or key < best[0]:
                best = (key, sig, 'lc', remain)
            elif (self.sig_tail_yield and tail and best[2] == 'turn'
                  and sig != best[1]):
                # **꼬리는 반대 방향 차로 이동에 양보한다.**
                # 회전 이후 remain 은 max(0, 음수) = 0 이라 회전이 무조건 이긴다 —
                # 그 사이 반대쪽 지시등이 못 켜져 선행 시간이 통째로 깎인다.
                # 채점 항목 13 은 **차로 변경** 지시등만 본다(회전 꼬리는 안 본다)
                # 이므로, 양보로 잃는 것이 없고 얻는 것이 선행 점등이다.
                # 같은 방향이면 양보할 이유가 없다(그대로 유지되면 되므로).
                best = (key, sig, 'lc', remain)

        if best is None:
            sig, src, remain = SIG_OFF, None, None
        else:
            _k, sig, src, remain = best

        # 유지 장치 둘. 기준 시점이 다르다 — 둘 다 상한이 있어 고착되지 않는다.
        #   min_on   : **켜진 시점부터** 최소 점등 시간. 켜자마자 꺼지는 깜빡임 방지
        #              (조건이 계속 참인 동안 갱신하지 않는다 — 갱신하면 이게 곧
        #               소등 지연이 돼 off_delay 가 무의미해진다)
        #   off_delay: **조건이 끝난 시점부터** 소등까지. 다 돌기 전에 꺼지는 것 방지
        #              (실차 자동소등 관례)
        if sig != SIG_OFF:
            if self.sig_held != sig:
                self.sig_on_ticks = 0                  # 새로 켜짐 / 방향 전환
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

    # ── 정적 장애물 회피 시프트 ──────────────────────────────────────────
    # ── 규칙 1: 신호 구역 억제 ───────────────────────────────────────────
    def _all_stopline_s(self, planner) -> list:
        """경로상 **모든** 정지선의 route_s (신호 유무 무관). 시작 시 1회.

        planner.traffic_lights 는 신호 매핑된 정지선만 담는다 (route.py
        collect_stops 게이트) — 지도 576개 중 245개가 미매핑이라 그것만 보면
        무신호 정지선 앞 대기열을 놓친다. 여기서는 lanegraph 원본을 직접 읽는다.
        """
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

    def _red_ahead(self, planner) -> float | None:
        """다음 신호가 **Red 또는 황색 STOP 래치**면 그 정지선까지 거리, 아니면 None.

        **우선순위 불변식: 신호·정지선 준수 > 회피.**
        이 상태에서는 정지선과 자차 사이의 정지 객체를 **거리 무관** 회피 대상에서
        뺀다 — 그 객체들은 십중팔구 같은 신호에 서 있는 대기열이고, 비켜가면
        신호 위반이 된다. 30 m 억제창(stopline_suppress_m)과 달리 **거리 조건이
        없다**: 적신호면 100 m 밖에서도 회피하지 않는다.

        회피가 다시 열리는 조건은 하나뿐이다 — **녹색 전환 후** 그 객체가
        obj_static_s 이상 계속 정지해 있을 것 (물체별 타이머가 그대로 판정한다).

        실측 근거 (2026-08-30 실전주행_교통류_01, route_s 1589.8): 정지 객체
        2대 뒤에 섰는데 신호가 녹→황→적→녹으로 순환했다. 적색 구간에 회피가
        열려 있었다면 대기열을 비켜 신호를 위반했을 것이다.
        """
        nxt = self._next_stopline(planner)
        if nxt is None:
            return None
        d_line, state, _tl_id = nxt
        if state == 'Red':
            return d_line
        if state == 'Yellow' and self.y_decision == 'stop':
            return d_line
        return None

    def _next_stopzone_s(self, planner) -> float | None:
        """시프트 span 이 넘으면 안 되는 route_s (= 다음 정지선 − zone_gate_margin_m).

        시프트 span 이 여기를 넘으면 시작하지 않는다 — 시프트 도중 신호가 바뀌어
        억제 구역에 걸리면 되돌릴 방법이 없다(진행 중 급조향은 금지).
        예전에는 _signal_zone 의 stopline_suppress_m(30) 을 그대로 뺐다 (B-1 전).
        두 함수는 상수를 공유만 했을 뿐 결합이 없어, 정지선 30 m 앞에서 span 이
        끝나야 하는 과한 조건이 됐다 — 정지 위치 25 m + span 39 m 면 정지선 64 m
        앞부터 기각이다. 여유는 zone_gate_margin_m 이 따로 정한다.
        """
        route_s = float(planner.route_s[planner.route_index])
        cands = []
        try:
            d = float(planner.distances_to_next_traffic_lights[planner.route_index])
            if d < float('inf'):
                cands.append(route_s + d)
        except Exception:                                   # noqa: BLE001
            pass
        cands += [s for s in self._all_stopline_s(planner) if s >= route_s]
        return (min(cands) - self.zone_gate_margin) if cands else None

    def _signal_zone(self, planner, ap) -> tuple | None:
        """신호 구역이면 (사유, 거리), 아니면 None — 회피 계열 전면 억제 게이트.

        **대기열은 신호 앞에만 선다**는 것이 근거다. 신호 대기 차량을 정적
        장애물로 오인해 비켜가면 신호 위반이 된다.

        1차 키는 planner.distances_to_next_traffic_lights + 교차로 판정이다.
        보조로 무신호 정지선까지 본다. 1차 키를 dist_stop_line(world.summ) 대신
        쓰는 이유 — 실측 2026-08-30: dist_stop_line 은 27% 가 null 이고 실제
        장애물 정지 지점에서도 null 이었다. 반면 이 배열은 inf 비율 1~2% 이고
        같은 지점에서 214.8 m 를 정상으로 준다.
        """
        if bool(getattr(ap, 'junction', False)):
            return ('junction', None)
        try:
            d = float(planner.distances_to_next_traffic_lights[planner.route_index])
        except Exception:                                  # noqa: BLE001
            d = float('inf')
        if d < self.sup_m:
            return ('signal', round(d, 1))
        route_s = float(planner.route_s[planner.route_index])
        for s_sl in self._all_stopline_s(planner):         # 보조: 무신호 정지선
            if 0.0 <= s_sl - route_s < self.sup_m:
                return ('stopline', round(s_sl - route_s, 1))
        return None

    # ── 물체별 정지 관찰 타이머 ──────────────────────────────────────────
    def _update_obj_timers(self, ap, paused: bool = False) -> None:
        """객체별 '정지 상태 지속 틱'. 매 틱 1회.

        자차 상태 카운터(ot_blocked_ticks)와 달리 **물체마다** 센다 — 자차가
        달리는 동안에도 관찰이 쌓여야 규칙 3(선제 회피)이 성립한다.
        id 가 잠깐 빠져도 obj_grace_ticks 동안은 타이머를 유지한다 (GT 는
        '가까운 순 30개' 제한이 있어 혼잡 시 밀릴 수 있다). 실측 2026-08-30
        두 로그에서는 드롭아웃이 0회였지만 방어로 둔다.
        """
        try:
            actors = list(ap._world.get_actors())
        except Exception:                                  # noqa: BLE001
            return
        ego_id = ap._vehicle.id
        seen = set()
        for a in actors:
            if a.id == ego_id:
                continue
            seen.add(a.id)
            self.obj_miss[a.id] = 0
            moving = float(getattr(a, 'speed', 0.0)) >= self.ot_v_max
            # 신호 무관 정지 카운터 (B-5) — standoff 대상 선정용. obj_ticks 는
            # 적색 중 멈추므로 녹색 직후 앞에 선 차량이 '정적' 이 되기까지 3 s
            # 가 더 걸리고 그동안 standoff 감속이 없다. 이 카운터는 pause 와
            # 무관하게 세되 PREEMPT/WAIT 판정에는 쓰지 않는다 (그건 obj_ticks).
            self.obj_stop_ticks[a.id] = 0 if moving else self.obj_stop_ticks.get(a.id, 0) + 1
            if moving:
                self.obj_ticks[a.id] = 0                   # 움직이면 즉시 리셋 (철회)
            elif not paused:
                self.obj_ticks[a.id] = self.obj_ticks.get(a.id, 0) + 1
            # paused(적색·황색STOP) 이면 **증가도 리셋도 안 한다** — 신호 대기
            # 시간이 '정지 관찰' 로 쌓이면 녹색이 되자마자 회피가 터진다.
            # 움직임 감지(철회)만은 신호와 무관하므로 위에서 먼저 본다.
        for oid in list(self.obj_ticks):
            if oid in seen:
                continue
            self.obj_miss[oid] = self.obj_miss.get(oid, 0) + 1
            if self.obj_miss[oid] > self.obj_grace:
                self.obj_ticks.pop(oid, None)
                self.obj_stop_ticks.pop(oid, None)
                self.obj_miss.pop(oid, None)

    def _is_obstacle(self, actor) -> bool:
        """장애물 클래스인가 (E-1). vtd_adapter/world.classify 가 9910 크기로 나눈
        cls 를 읽는다 — 9910 에는 타입 필드가 없어(SPEC §1.1) 길이 ≤ 3 m 정지 물체가
        'obstacle', 길이 > 3 m 가 'vehicle' 이다. 박스·라바콘·자재는 출발할 앞차가
        아니므로 정지 관찰도 큐 판정도 의미가 없다. 스위치가 꺼지면 항상 거짓."""
        return self.obs_fastpath and getattr(actor, 'cls', None) == 'obstacle'

    def _static_ok(self, actor) -> bool:
        if self._is_obstacle(actor):
            return True                                    # E-1: 감지 즉시 정적
        return self.obj_ticks.get(getattr(actor, 'id', None), 0) >= self.obj_static_ticks

    def _stop_ok(self, actor) -> bool:
        """standoff 대상 조건 (B-5) — 신호 무관 정지가 standoff_stop_s 이상.
        standoff_stop_s 0 = 킬 스위치: 정적 관찰(_static_ok) 축으로 되돌린다."""
        if self._is_obstacle(actor):
            return True                                    # E-1
        if self.standoff_stop_ticks <= 0:
            return self._static_ok(actor)
        return self.obj_stop_ticks.get(getattr(actor, 'id', None), 0) >= self.standoff_stop_ticks

    def _red_pause(self, planner) -> float | None:
        """회피 계층이 '적색' 으로 보는 거리 (E-7) — _red_ahead 에 거리 상한을 둔 것.

        _red_ahead 는 거리 무관이다 (legacy 억제의 설계). queue_only 에서 적색은
        억제가 아니라 일시정지 입력인데, 456 m 앞 적신호가 관찰 pause·큐 B·BREAKOUT
        pause 를 걸어 정지 차량 1대를 그 신호의 대기열로 오판했다 (실측 2026-09-03
        020439/03 12 s, 02 25 s 정지). 진짜 대기열은 정지선 12~52 m 에서 잡혔고 큐 B
        단독은 전부 137 m 이상이었다. red_pause_max_m 0 = 상한 없음 (이전 동작).
        SHIFT_HOLD 는 이 함수를 쓰지 않는다 — 홀드는 E-6 으로 무해하다.
        """
        d = self._red_ahead(planner)
        if d is None or self.red_pause_max_m <= 0.0 or d <= self.red_pause_max_m:
            return d
        return None

    def _blocker_before_stopline(self, ap, planner) -> bool:
        """막고 선 장애물이 정지선보다 **훨씬 앞**인가 — 그 신호의 대기열이 아니다.

        PDM 의 `traffic_light_hazard` 는 적신호가 앞에 있으면 선다. 그걸 그대로
        "신호가 정지 원인" 으로 보면, 정지선이 **150 m** 밖인데 20 m 앞 정지
        차량에 막힌 경우까지 신호 대기로 분류돼 크립·BREAKOUT·never_stall 이
        전부 안 열린다 (실측 20260909_093415: 11_직진10 정지 1635틱 중 **1167틱**
        이 이 분기, 18_연속교차로14 도 같은 꼴 — 둘 다 30 s 무전진으로 blocked).

        기준은 새로 만들지 않는다 — 대기열 판정이 쓰는 `overtake.queue_head_max_m`
        를 그대로 읽는다 (`_head_near_stopline` 과 같은 잣대: "대기열 선두는
        정지선을 향해 선다"). 장애물이 정지선보다 그만큼 이상 앞이면 대기열이 아니다.

        스위치가 꺼져 있으면 항상 거짓 = 이전 동작.
        """
        if not self.tl_hazard_far:
            return False
        b = self._blocker(ap, planner)
        if b is None:
            return False
        d_line = self._stopline_d(planner)
        if d_line is None:
            return True                                # 정지선이 없다 = 신호 원인 아님
        pr = self._project(planner, b.get_location().x, b.get_location().y)
        if pr is None:
            return False
        return (float(d_line) - float(pr[0])) > self.q_head_m

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

    def _project(self, planner, x_carla, y_carla):
        """CARLA 좌표 → (route_s, 횡오프셋). 전방 창에서만 찾는다."""
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

    def _corridor_blockers(self, ap, planner, static_ok=None, lat_band=None):
        """전방 detect_max_m 안에서 **주행 회랑을 침범한 정지 객체** 목록.

        반환: [(s_rel, lat, half_w, actor)] — s_rel 은 자차 기준 전방거리.
        침범 판정은 선행차 판정([route.py] compute_leading_vehicles) 과 같은 축:
        |lat| < 자차반폭 + 객체반폭 + obstacle_clearance_m.
        static_ok: '정지' 판정 함수. 기본 _static_ok(정지 관찰 3 s, 신호 중 일시정지).
        standoff 대상은 _stop_ok(신호 무관 1.5 s) 로 따로 뽑는다 (B-5).
        lat_band: (lo, hi) 면 회랑 중심을 0 이 아니라 그 횡구간으로 본다 — 시프트가
        휩쓰는 띠(0 ~ 목표 오프셋) 안의 객체. None = 현행(중심 0 의 회랑).
        """
        try:
            actors = list(ap._world.get_actors())
        except Exception:                                  # noqa: BLE001
            return []
        if static_ok is None:
            static_ok = self._static_ok
        ego_id = ap._vehicle.id
        half_ego = float(self.cfg['vehicle']['width']) / 2.0
        clr = float(self.cfg['percep'].get('obstacle_clearance_m', 0.3))
        # [1](b) 종료 구간에서는 **정지 객체**를 회랑에서 뺀다 (종료선 콘).
        # 이 함수가 회피·standoff·큐 판정의 단일 관문이라 여기 한 곳만 막으면
        # 셋이 같이 빠진다. 움직이는 객체는 그대로 본다 — 속도 임계는
        # blocker_speed_max("회피가 정지로 보는 속도")를 그대로 읽는다.
        # PDM 의 IDM 추종·보행자 정지는 여기와 무관하게 계속 산다.
        gate_on = self._finish_gate_s() is not None
        out = []
        for a in actors:
            if a.id == ego_id or not static_ok(a):
                continue
            if (gate_on and float(getattr(a, 'speed', 0.0)) < self.ot_v_max
                    and self._fg_drop_static(planner, a)):
                self.fg_dropped += 1
                continue
            loc = a.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is None:
                continue
            s_rel, lat = pr
            if not (0.5 < s_rel <= self.detect_max_m):
                continue
            hw = float(getattr(getattr(a, 'bounding_box', None), 'extent', None).y) \
                if getattr(a, 'bounding_box', None) is not None else 0.9
            reach = half_ego + hw + clr
            if lat_band is None:
                hit = abs(lat) < reach
            else:
                hit = (lat_band[0] - reach) < lat < (lat_band[1] + reach)
            if hit:
                out.append((s_rel, lat, hw, a))
        out.sort(key=lambda z: z[0])
        return out

    def _finish_gate_s(self) -> float | None:
        """종료 구간이 **시작되는 route_s**. 꺼져 있거나 종료선을 모르면 None.

        게이트 임계를 두 곳(회랑·PDM 접합)에 적지 않기 위한 단일 출처다.

        왜 종료 구간만 따로 막나 — 종료 지점은 주최측이 라바콘 2개로 표시하고,
        좁은 연결로에서는 그 콘이 주행 회랑 안에 들어온다. 실측
        실경로_01_PathShape03: 콘(cls obstacle, 0.3 m)을 blocker 로 잡아 종료선
        17 m 앞에서 영구 정지했다(미완주). 생성기 쪽은 콘을 도로 끝으로 옮겼지만
        **주최측 콘 위치는 우리가 못 정한다** — 대회에서 같은 일이 그대로 난다.
        하필 이 구간은 다른 안전망이 전부 죽어 있다: route_end.active_m(150 m)
        안이라 `_obstacle_cause` 가 거짓이고, BREAKOUT·크립·never_stall 이
        하나도 안 선다.
        """
        if not self.fg_ignore or self.finish_s is None:
            return None
        return self.finish_s - self.fg_m

    def _actor_route_s(self, planner, actor) -> float | None:
        """객체의 **경로 위 절대 s** [m]. 투영 실패면 None.

        전후 범퍼까지 **세 점**을 투영해 그중 가장 **가까운**(작은) s 를 쓴다.
        커밋 A 의 `_actor_lanes` 는 차로 소속을 보므로 좌우 폭 끝점을 투영하지만,
        여기는 **종거리 게이트**라 의미 있는 퍼짐이 종방향이다. 가장 가까운 점을
        쓰는 것이 보수적이다 — 게이트에 걸쳐 선 물체는 "아직 종료 구간 밖" 으로
        보고 계속 장애물로 취급한다.

        `_project` 가 전방 detect_max_m 창에서만 찾으므로, 그 밖의 객체는 창
        끝으로 접혀 s 가 작게 나온다 = 무시되지 않는다(안전한 쪽). 회피·standoff
        결정은 전부 이 창 안에서 일어나므로 실질적으로 "감지 즉시" 와 같다.
        """
        try:
            base_s = float(planner.route_s[planner.route_index])
            loc = actor.get_location()
            yaw = _math.radians(float(getattr(actor, 'yaw_deg', 0.0)))
            half_l = float(getattr(actor, 'length', 0.0)) / 2.0
        except Exception:                                  # noqa: BLE001
            return None
        dx, dy = _math.cos(yaw) * half_l, _math.sin(yaw) * half_l
        best = None
        for x, y in ((loc.x, loc.y), (loc.x + dx, loc.y + dy), (loc.x - dx, loc.y - dy)):
            pr = self._project(planner, x, y)
            if pr is None:
                continue
            s = base_s + pr[0]
            best = s if best is None else min(best, s)
        return best

    def _fg_drop_static(self, planner, actor) -> bool:
        """이 **정지 객체**가 종료 구간 안에 있나 (속도·static 판정은 호출부 몫).

        `finish_gate_drop` 과 `_actors_in_corridor` 가 같은 식을 두 벌 적지 않게
        가운데로 뺀 것이다.
        """
        gate = self._finish_gate_s()
        if gate is None:
            return False
        if not self.fg_by_obj:
            return self._in_finish_gate(planner)           # 이전 동작 — 자차 기준
        s = self._actor_route_s(planner, actor)
        return s is not None and s >= gate

    def _in_finish_gate(self, planner) -> bool:
        """**자차**가 종료 구간 안인가 — `finish_gate_by_object_enable` 이 꺼졌을
        때만 쓰는 이전 판정이다. 켜져 있으면 아무도 부르지 않는다.

        이 축이 왜 틀렸는지는 스위치 주석(`fg_by_obj`) 참조 — 회피 결정이
        자차가 구간에 들기 훨씬 전에 끝나므로 판정 시점이 늦다.
        """
        gate = self._finish_gate_s()
        if gate is None:
            return False
        try:
            return float(planner.route_s[planner.route_index]) >= gate
        except Exception:                                   # noqa: BLE001
            return False

    def finish_gate_drop(self, planner, actor) -> bool:
        """이 객체를 **종료 구간의 정지 장애물**로 보고 빼야 하나 (실주행 2차 [3]).

        `_actors_in_corridor` 안의 같은 판정을 **밖에서도 쓸 수 있게** 꺼낸 것이다
        (조건을 두 벌 적지 않는다). autopilot 의 `# VTD:` 접합부가 PDM 의 actor
        목록을 거를 때 이 함수 하나만 부른다.

        왜 PDM 입력까지 손대나 — kr 후보만 빼는 것으로는 안 멈춘다:
        실측 20260908_222954/실경로_02 rs 745.5(종료선 749.2, **3.7 m 앞**)에서
        `finish_gate.dropped=4` 로 kr 은 콘을 이미 뺐는데 PDM 의
        `compute_target_speeds_wrt_all_actors` 가 콘을 보고 vehicle 후보 0.0 을
        내 483틱(로그 끝까지) 정지했다 — **미완주**다. 종료선 콘은 주최측이
        놓는 것이라 위치를 우리가 못 정한다.

        범위는 좁다: `finish_gate_ignore_enable` on ∧ 종료선 앞 finish_gate_m
        (기본 10 m)부터 ∧ **정지 객체만**(속도 < blocker_speed_max). 움직이는
        것은 그대로 보이고, 보행자 정지·IDM 추종도 이 구간 밖에서는 전부 그대로다.
        """
        if not self._static_ok(actor):
            return False
        if float(getattr(actor, 'speed', 0.0)) >= self.ot_v_max:
            return False
        return self._fg_drop_static(planner, actor)

    def _crossable_runs(self, lg, key, side) -> list:
        """side 로 **넘을 수 있는** 구간 [(s0, s1) …] — 점선 조각 + (E-8 ①) 마킹
        'none' 조각. 선이 없는 구간은 넘어도 실선 차로 변경(항목 6)이 아니다.
        마크 데이터가 없는 목 레인그래프는 점선만 (이전 동작)."""
        runs = list(lg.dashed_runs(key, side) or [])
        if not self.none_crossable:
            return runs
        try:
            marks = lg.lanes[key]['left_mark' if side == 'left' else 'right_mark']
        except Exception:                                   # noqa: BLE001
            return runs
        runs += [(float(a), float(b)) for a, b, typ, _c, _ok in marks if typ == 'none']
        runs.sort()
        out: list = []
        for a, b in runs:                                   # 겹침·맞닿음 병합
            if out and a <= out[-1][1] + 1e-6:
                out[-1] = (out[-1][0], max(out[-1][1], b))
            else:
                out.append((a, b))
        return out

    def _dashed_ahead_m(self, lg, ego_lane, side, ego_local_s, span_m) -> float:
        """자차 앞 span_m 구간 중 **점선인 길이** [m]. 차로 끝을 넘으면 successor 로 잇는다.

        lg.dashed_corridor_m 은 "점선 조각의 길이" 를 주지 실제로 **내 앞에 남은**
        점선을 주지 않는다. 실측 2026-08-30 실전주행_교통류_01: 점선 구간이 차로
        로컬 0.0~76.4 인데 자차가 71.4 에 있어 앞쪽 점선은 5 m 뿐인데도 76.4 가
        반환돼 게이트를 통과했고, 시프트 span 84.1 m 가 **전 구간 실선** 위에 얹혔다.
        나가는 전이와 복귀 전이가 모두 점선 안에서 끝나야 차선변경이 합법이다.
        """
        if lg is None or ego_lane is None or span_m <= 0.0:
            return 0.0
        cover, key, s0, left = 0.0, ego_lane, float(ego_local_s), float(span_m)
        for _ in range(6):                                  # successor 최대 6칸
            try:
                L = lg.length(key)
                runs = self._crossable_runs(lg, key, side)
            except Exception:                               # noqa: BLE001
                break
            hi = min(L, s0 + left)
            for a, b in runs:
                cover += max(0.0, min(hi, b) - max(s0, a))
            left -= max(0.0, hi - s0)
            if left <= 1e-6:
                break
            nxt = [k for k in lg.successors(key) if lg.neighbor(k, side) is not None]
            if not nxt:
                break
            key, s0 = nxt[0], 0.0
        return cover

    def _is_center_mark(self, lg, ego_lane, side, s) -> bool:
        """side 쪽 차선 표식이 **중앙선**인가 — 색으로 본다 (황색 = 중앙선).

        `_dashed_ahead_m` 은 마크의 **종류**(실선/점선)만 보고 색을 읽지 않는다.
        그래서 BREAKOUT L2 의 '실선 허용' 완화가 황색 중앙선까지 통과시킨다
        (2026-09-01 코드 확인). 중앙선 침범은 항목4 **중대**이고, 대향 차로
        진입은 L5(대향차 TTC 게이트) 소관인데 **미구현**이다 — 그러므로 지금은
        단계와 무관하게 **절대 불가**로 자른다. 여기서 걸러야 `lvl >= 2` 가
        dashed 게이트를 건너뛰어도 중앙선을 넘지 않는다.
        """
        if lg is None or ego_lane is None:
            return False
        try:
            _typ, col, _ok = lg.mark_at(ego_lane, float(s), side)
        except Exception:                                  # noqa: BLE001
            return False
        return str(col) == 'yellow'

    def _planned_shift_geom(self, planner, actor, side, trans_m, ahead_m=None,
                            last_actor=None, after_m=None):
        """적용 **전** 시프트 기하 검사 → `(κ, lc_var)`. 못 재면 None.

        ahead_m: 전이 시작 여유 [m]. None 이면 shift_ahead_m (B-2 의 L3 완화가
        실제 시프트와 같은 값을 쓰도록 side 루프가 넘긴다).
        after_m: 뒤여유 [m]. None 이면 extra_after_m (E-2 zone 연장이 실제 시프트와
        같은 값을 쓰도록 side 루프가 넘긴다).

        `plan_shift_span` 이 cKDTree 를 세우므로(≈4.6 ms) **한 번만 부르고**
        두 지표를 같이 낸다.

        · κ  — 목표 이웃 차로 오프셋의 횡곡률 최대값 [1/m]. 계단 검출용 (B-7).
        · lc_var — span 안에서 **빌드 시점 계획 횡오프셋**(`_lat_build`)이 변하는
          폭 [m]. 계획 차선변경이 회피 구간과 겹치는지 본다 (B-11). 겹치면
          `lat_shift = _lat_build ± d` 라 두 횡이동이 **합산**되어, 지시등 한 번에
          차로 경계를 두 번 넘는 일이 생긴다 (실측 6.25 m = 차로폭의 2.08 배).

        찾는 것은 곡률이 아니라 **계단 불연속**이다 (B-7). 실측 2026-09-01
        실전주행_교통류_01 의 시프트 span [6033,6465] 은 경로점 6362 에서
        목표 이웃 차로가 2.854 m 튄다 — laneSection 경계에서 get_right_lane()
        이 가리키는 차로가 바뀌기 때문으로 보인다. 자차가 그 계단을 추종하려다
        조향이 양방향 풀락으로 포화했고, 그것이 황색 중앙선 0.94 m 침범의
        더 깊은 뿌리다.

        전이 계수를 곱하지 않은 **날 오프셋**을 본다 — 계수는 코사인이라
        매끄러워서 빼면 정상 전이의 곡률이 함께 빠지고 판별이 깨끗해진다.
        실측 (간격 1.0 m): 정상 4건 0.0002~0.0095, 계단 1건 2.977 → 313 배.
        """
        self.last_span_plan = None
        try:
            ppm = float(getattr(planner, 'points_per_meter', 10))
            a, b, left = planner.plan_shift_span(
                actor, last_actor, obstacle_direction='right' if side == 'left' else 'left',
                transition_length=trans_m * ppm,
                extra_length_before=self.ot_before_m * ppm,
                extra_length_after=(self.ot_after_m if after_m is None else after_m) * ppm,
                min_start_ahead=(self.shift_ahead_m if ahead_m is None else ahead_m) * ppm)
            # span 국소성 게이트가 읽는다. 실제 시프트(shift_route_around_actors)는
            # 같은 인자로 plan_shift_span 을 다시 불러 같은 span 을 얻으므로, 여기서
            # 본 span 이 곧 적용될 span 이다. 반환 형태는 그대로 (κ, lc_var) / None.
            self.last_span_plan = (int(a), int(b), bool(left))
            step = max(1, int(round(self.shift_k_step_m * ppm)))
            d = planner.planned_lateral_offsets(a, b, left, step_pts=step)
        except Exception:                                  # noqa: BLE001
            return None
        if len(d) < 3:
            return None
        h = step / ppm
        kap = float(np.abs(d[2:] - 2.0 * d[1:-1] + d[:-2]).max()) / (h * h)
        base = getattr(planner, '_lat_build', None)
        lc = 0.0
        if base is not None and b > a:
            seg = np.asarray(base[int(a):int(b)], dtype=float)
            if seg.size:
                lc = float(seg.max() - seg.min())
        return kap, lc

    def _shift_speed_cap(self, planner, ego_speed: float) -> float | None:
        """진행 중인 회피 시프트의 전이 곡률에서 나오는 **속도 상한** — min() 후보.

        `transition_m` 은 시프트를 **만든 시점**의 속도로 정해진다
        (`trans_m = max(transition_m, shift_k_s·v)`). 정지 중 생성되면 12 m 로
        굳는데, 적신호 SHIFT_HOLD 로 21 s 를 서 있다 6.5 m/s 로 통과하면
        요구 횡가속이 4.34 m/s² 가 된다 — 실측 2026-09-01 t=112.4: 조향이 양방향
        풀락(±0.480)으로 포화하고 경로 대비 1.45 m 오버슛, 황색 중앙선을 0.94 m
        물었다 (0.60 s 지속 = 항목4 중대 임계).

        **경로를 다시 밀지 않는다** — 진행 중인 시프트를 재생성하면 현재 위치의
        경로가 옆으로 튀어 급조향이 된다 (`shift_route_around_actors` 의
        `min_start_ahead` 주석과 같은 사고). 대신 속도를 낮춰 같은 기하를 통과
        가능하게 만든다:

            κ = |d²(lat_shift)/ds²|            (전이의 횡곡률)
            a_lat = κ·v²  ≤  a_lat_max   →   v ≤ √(a_lat_max / κ)

        · `lat_shift − _lat_build` 를 미분한다 — **회피 시프트 성분만** 본다.
          `lat_shift` 자체에는 계획 차선변경 블렌드와 테이퍼 보정이 함께 실려
          있어, 그대로 미분하면 이미 검증된 계획 기하의 곡률까지 세서 평지
          구간에서도 상한이 하한(shift_cap_min_v)까지 내려간다 (실측: replay
          t=108.5~110.5 에서 cap 1.00). 전이 길이·형상은 가정하지 않는다.
        · 평지(plateau)와 span 밖에서는 κ = 0 이라 스스로 비활성이다.
        · 0.5 m 스텐실 — `lat_shift` 는 블렌드로 만든 해석적 배열이라 잡음이
          없고, 폭을 넓히면 전이 경계(κ 가 최대인 지점)에서 평지 쪽을 섞어
          **과소평가**한다. 실측 비교(코사인 Δ=3.0): 2 m 폭은 L=12 에서 상한을
          7 %, L=8 에서 14 % 느슨하게 냈고 0.5 m 폭은 각각 0.5 % / 1.1 % 다.
        · 미리보기는 standoff 와 같은 축(`max(shift_latest_m, shift_k_s·v)`)이다 —
          전이에 **닿기 전에** 감속이 시작돼야 한다.
        · 하한 `shift_cap_min_v` — 전이 한복판에서 완전히 서면 빠져나올 수 없다.
        """
        if self.a_lat_max <= 0.0 or self.ot_span is None:
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
        look = max(self.shift_latest_m, self.shift_k_s * max(ego_speed, 0.1))
        h = max(1, int(round(0.5 * ppm)))                  # 0.5 m 스텐실 (위 참조)
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

    def _ego_local_s(self, lg, ap) -> float:
        loc = ap._vehicle.get_location()
        vx, vy = frame.from_carla_xy(loc.x, loc.y)
        try:
            m = lg.locate(vx, vy)
        except Exception:                                   # noqa: BLE001
            return 0.0
        return float(m.s) if m else 0.0

    def _lane_width(self, lg, ego_lane, ap) -> float:
        if lg is None or ego_lane is None:
            return 3.0
        loc = ap._vehicle.get_location()
        vx, vy = frame.from_carla_xy(loc.x, loc.y)
        try:
            m = lg.locate(vx, vy)
            return float(lg.width_at(ego_lane, m.s if m else 0.0))
        except Exception:                                   # noqa: BLE001
            return 3.0

    def _corridor_passable(self, blockers, lg, ego_lane, ap) -> bool:
        """장애물 **옆으로 지나갈 폭**이 차로 안에 남아 있는가.

        대기열/장애물 판별기의 핵심이다 — 대기열은 비켜갈 폭이 없고, 길에 선
        장애물은 있을 수 있다. 폭이 있으면 '대기열' 이 아니라 회피 대상이다.
        """
        if not blockers:
            return True
        W = self._lane_width(lg, ego_lane, ap)
        need = float(self.cfg['vehicle']['width']) + self.q_clear_m
        for _s, lat, hw, _a in blockers:
            free_l = (W / 2.0) - (lat + hw)
            free_r = (lat - hw) + (W / 2.0)
            if max(free_l, free_r) < need:
                return False                                # 하나라도 못 지나가면 막힌 것
        return True

    def _head_near_stopline(self, planner, head_s_rel: float) -> bool:
        """대기열 선두 앞에 정지선·신호가 queue_head_max_m 안에 있는가.

        대기열은 **선두가 정지선을 향해** 선다. 길 한복판에 선 장애물은 그렇지 않다.
        """
        route_s = float(planner.route_s[planner.route_index])
        head_abs = route_s + head_s_rel
        try:
            d_tl = float(planner.distances_to_next_traffic_lights[planner.route_index])
        except Exception:                                   # noqa: BLE001
            d_tl = float('inf')
        if 0.0 <= (route_s + d_tl) - head_abs < self.q_head_m:
            return True
        for s_sl in self._all_stopline_s(planner):
            if 0.0 <= s_sl - head_abs < self.q_head_m:
                return True
        return False

    def _is_queue(self, blockers, planner=None, ap=None, lg=None, ego_lane=None) -> bool:
        """대기열 판정 — 유일한 회피 억제 (C). 모드에 따라 legacy / v2."""
        if self.suppress_mode == 'legacy':
            return self._is_queue_legacy(blockers, planner, ap, lg, ego_lane)
        return self._is_queue_v2(blockers, planner, ap, lg, ego_lane)

    def _tick_cache(self, ap, planner) -> None:
        """틱당 1회 캐시 (C-2): standoff 축 회랑 + 큐 판정. apply 가 부른다.
        직접 _try_overtake 를 부르는 테스트도 먼저 이걸 불러야 한다."""
        if self.suppress_mode != 'legacy':
            self._tick_lg = getattr(planner, 'lg', None)
            self._tick_ego_lane = (getattr(ap, '_kr_ego_lane', None)
                                   or self._ego_lane(self._tick_lg, ap))
            self._tick_corridor = self._corridor_blockers(ap, planner, static_ok=self._stop_ok)
            self._tick_queue = self._is_queue(self._tick_corridor, planner, ap,
                                              self._tick_lg, self._tick_ego_lane)
        else:
            self._tick_corridor, self._tick_queue = [], False

    # ── 차로 지도 (커밋 A — 읽기 전용, 동작 변화 0) ──────────────────────
    def _lane_hops(self, lg, ego_lane, max_hops: int) -> dict:
        """자차 차로 기준 ±max_hops 이웃 → {차로: hop 수}. 자기 자신은 0.

        `lg.neighbor` 만 쓴다 — 이웃 정의를 새로 만들지 않는다. 한쪽으로 가다
        끊기면 그쪽은 거기서 멈춘다.
        """
        out = {ego_lane: 0}
        for side, sign in (('left', -1), ('right', +1)):
            k = ego_lane
            for h in range(1, int(max_hops) + 1):
                k = lg.neighbor(k, side)
                if k is None or k in out:
                    break
                out[k] = sign * h
        return out

    def _lane_width_at(self, lg, key, s: float) -> float:
        """그 지점의 차로 폭 [m]. 폭 배열은 s 격자라 **평균이 아니라 보간**이다 —
        소멸 차로는 끝에서 0 이 되므로 평균으로 보면 통과 가능해 보인다."""
        r = lg.lanes[key]
        try:
            return float(np.interp(float(s), r['s'], r['width']))
        except Exception:                                  # noqa: BLE001
            return 0.0

    def _actor_lanes(self, lg, actor, hops: dict) -> set:
        """객체가 **걸친** 차로 집합 (hops 안의 것만).

        객체 lane 은 9910 에 없다(로그 `lane: null` 은 설계다). 중심 + 좌우 폭
        끝점 **세 점**을 각각 투영해 합집합을 취한다 — 중심만 보면 두 차로에
        걸쳐 선 차가 한 차로만 막은 것으로 보인다 (지시안 증상 3).
        """
        loc = actor.get_location()
        # **좌표계 주의**: get_location 은 CARLA 프레임인데 lg.locate 는 VTD
        # 프레임을 받는다 (_ego_lane 과 같은 규약). 커밋 A 는 이 변환을 빠뜨려
        # 어떤 객체도 차로에 안 잡혔다 — free_run 이 항상 ahead_m 이었다
        # (2026-09-09 avoid_sim 으로 발견: blocked_by 가 늘 비어 있었다).
        vx, vy = frame.from_carla_xy(loc.x, loc.y)
        yaw = _math.radians(float(getattr(actor, 'yaw_deg', 0.0)))
        half_w = float(getattr(actor, 'width', 1.8)) / 2.0
        # 차체 횡방향 단위벡터 (진행방향 +90°). yaw 도 CARLA 라 미러 프레임에서
        # 좌우가 뒤집히지만, **세 점의 합집합**을 쓰므로 결과는 같다.
        nx, ny = -_math.sin(yaw), _math.cos(yaw)
        pts = [(vx, vy),
               (vx + nx * half_w, vy + ny * half_w),
               (vx - nx * half_w, vy - ny * half_w)]
        out = set()
        for x, y in pts:
            m = lg.locate(x, y, prefer=list(hops.keys()))
            # LaneMatch 의 차로 필드는 `.lane` 이다 (`.key` 가 아니다 — _ego_lane
            # 과 같은 규약). 커밋 A 는 `.key` 를 읽어 매번 예외였다.
            if m is not None and m.lane in hops:
                out.add(m.lane)
        return out

    def lane_map(self, ap, planner) -> dict | None:
        """자차 앞 `lane_map_ahead_m` 의 **차로별 여유거리·통행가능** 지도.

        **커밋 A 는 읽기 전용이다** — 아무도 이 값을 읽지 않으므로 동작이 변하지
        않는다 (`reasons.lane_map` 진단으로만 나간다). 후보 선택에 쓰는 것은
        커밋 B 다.

            free_run[k]  그 차로의 첫 **정지 객체**까지 거리 [m]. 없으면 ahead_m.
            passable[k]  존재 ∧ 같은 방향 ∧ 폭 ≥ min_width ∧ 교차로 아님

        큐 제외: `_tick_queue` 가 참인 틱에는 `_tick_corridor` 에 든 id 를 뺀다 —
        신호 대기 줄을 추월 후보로 삼지 않기 위해서다. 객체 단위 큐 판정을 새로
        만들지 않고 **기존 규약을 그대로** 쓴다 (억제 기준이 두 벌이 되면 안 된다).
        """
        if not self.lane_map_on:
            return None
        lg = getattr(planner, 'lg', None)
        ego_lane = self._tick_ego_lane or self._ego_lane(lg, ap)
        if lg is None or ego_lane is None or ego_lane not in lg.lanes:
            return None
        hops = self._lane_hops(lg, ego_lane, self.lane_map_hops)
        ahead = self.lane_map_ahead_m
        ego_s = self._ego_local_s(lg, ap)
        free = {k: ahead for k in hops}
        passable = {}
        for k in hops:
            r = lg.lanes[k]
            w = self._lane_width_at(lg, k, ego_s if k == ego_lane else ego_s)
            passable[k] = bool(r['junction'] == -1
                               and r['dir'] == lg.lanes[ego_lane]['dir']
                               and w >= self.lane_map_min_w)
        # 큐로 판정된 객체는 지도에서 뺀다
        drop = set()
        if self._tick_queue:
            drop = {b[3].id for b in self._tick_corridor}
        blocked: dict = {}
        try:
            actors = list(ap._world.get_actors())
        except Exception:                                  # noqa: BLE001
            actors = []
        ego_id = ap._vehicle.id
        for a in actors:
            if a.id == ego_id or a.id in drop:
                continue
            if float(getattr(a, 'speed', 0.0)) >= self.lane_map_static_v:
                continue                                   # 움직이는 것은 여유를 안 깎는다
            pr = self._project(planner, a.get_location().x, a.get_location().y)
            if pr is None or not (0.0 < pr[0] <= ahead):
                continue
            for k in self._actor_lanes(lg, a, hops):
                if pr[0] < free.get(k, ahead):
                    free[k] = pr[0]
                    blocked[k] = int(a.id)
        return {'ego_lane': list(ego_lane), 'hops': {str(list(k)): h for k, h in hops.items()},
                'free_run': {str(list(k)): round(v, 1) for k, v in free.items()},
                'passable': {str(list(k)): bool(v) for k, v in passable.items()},
                'blocked_by': {str(list(k)): v for k, v in blocked.items()},
                'queue_dropped': len(drop)}

    # ── 차로 지도 커밋 B — 후보 선택·시점·감속 ───────────────────────────
    def _ramp_len_m(self, hops: int, lane_w: float, v: float) -> float:
        """`hops` 칸을 속도 v 로 옮기는 데 필요한 **횡전이 길이** [m].

        `shift_route_smoothly` 가 쓰는 전이 형상 그대로 유도한다:

            y(x) = Y·(1 − cos(π x / L)) / 2      Y = hops × 차로폭
            y''  = Y·π² / (2 L²) · cos(…)  → |y''|max = Y π² / (2 L²)
            a_lat = v²·|y''|max ≤ a_lat_max
            ⇒ L ≥ v·π·√(Y / (2·a_lat_max))

        `a_lat_max` 는 `overtake.a_lat_max` 를 그대로 읽는다 —
        `_shift_speed_cap` 이 진행 중인 시프트를 재는 것과 **같은 축**이라야
        "만들 때는 되는데 지나갈 땐 상한에 걸린다" 가 안 생긴다.
        """
        Y = max(0.0, float(hops)) * max(0.1, float(lane_w))
        a = self.a_lat_max
        if a <= 0.0 or Y <= 0.0:
            return 0.0
        return max(0.0, float(v)) * _math.pi * _math.sqrt(Y / (2.0 * a))

    def lane_plan(self, ap, planner) -> dict | None:
        """차로 지도로 **어느 차로로 언제 옮길지** 정한다 (커밋 B).

        커밋 A 의 `lane_map()` 이 재료고 여기서 고른다. 개입은 두 가지뿐이다:
          · 속도 상한 후보 (min() 에 덧대는 **상한형** — `cap_binds` 축)
          · 시프트 방향 힌트 (기존 게이트 7개는 그대로 통과해야 한다)

        규칙 (지시안 그대로):
          트리거   내 차로 free_run < decide_m. **다른 차로가 막힌 것은 트리거가
                   아니다** — 케이스 10 이 그 반례다.
          후보     passable ∧ free_run > 램프 + shift_ahead_m
          선택     free_run 최대 → 동률이면 hop 이 적은 쪽(가까운 쪽)
          램프     길이 = `_ramp_len_m(hops, 차로폭, 상한속도)`
          시작 s   목표 차로 첫 장애물 s − shift_ahead_m − 램프 길이.
                   내 차로 장애물보다 뒤일 수는 없으므로 그쪽도 같이 본다.
          늦었으면 더 감속해 램프를 줄인다. 그래도 안 되면 후보 없음(standoff).

        반환 None = 개입 없음 (스위치 off · 지도 없음 · 트리거 아님).
        """
        if not self.lane_map_on:
            return None
        lm = self.lane_map(ap, planner)
        if lm is None:
            return None
        lg = getattr(planner, 'lg', None)
        ego_lane = self._tick_ego_lane or self._ego_lane(lg, ap)
        if lg is None or ego_lane is None:
            return None
        key = str(list(ego_lane))
        free = lm['free_run']
        mine = float(free.get(key, self.lane_map_ahead_m))
        if mine >= self.lm_decide_m:
            return None                                    # 트리거 아님
        hops = {k: h for k, h in lm['hops'].items()}
        lane_w = self._lane_width_at(lg, ego_lane, self._ego_local_s(lg, ap))
        v_cap = self.lm_avoid_v
        # 후보 — 통행 가능하고, 램프 + 여유만큼 뚫려 있는 차로
        cands = []
        for k, h in hops.items():
            if k == key or not lm['passable'].get(k):
                continue
            n = abs(int(h))
            L = self._ramp_len_m(n, lane_w, v_cap)
            need = L + self.shift_ahead_m
            if float(free.get(k, 0.0)) > need:
                cands.append((float(free[k]), -n, k, n, L, need))
        plan = {'trigger_free_m': round(mine, 1), 'ego_lane': lm['ego_lane'],
                'lane_w': round(lane_w, 2), 'v_cap': round(v_cap, 2),
                'cands': {c[2]: {'free': round(c[0], 1), 'hops': c[3],
                                 'ramp_m': round(c[4], 1)} for c in cands}}
        if not cands:
            plan['pick'] = None
            plan['why'] = 'no_candidate'                   # → standoff → never_stall
            self.last_lane_plan = plan
            return plan
        cands.sort(reverse=True)                           # free 최대 → hop 적은 쪽
        _f, _nh, pick, n_hops, ramp_m, need = cands[0]
        # 램프 시작 s — 목표 차로 첫 장애물과 내 차로 장애물 중 **먼저 걸리는 쪽**
        first = min(float(free.get(pick, self.lane_map_ahead_m)), mine)
        start = first - self.shift_ahead_m - ramp_m
        late = start < 0.0
        if late:
            # 늦었다 — 남은 거리로 램프를 되풀어 필요한 속도를 낸다.
            #   L_avail = first − shift_ahead_m,  L = v·π·√(Y/2a) → v = L / (π√(Y/2a))
            avail = max(0.0, first - self.shift_ahead_m)
            unit = self._ramp_len_m(n_hops, lane_w, 1.0)   # 1 m/s 당 램프 길이
            v_need = avail / unit if unit > 0.0 else 0.0
            if v_need < self.bo_creep_v:                   # 크립보다 느려야 한다 = 불가
                plan.update({'pick': None, 'why': 'ramp_too_late',
                             'avail_m': round(avail, 1), 'need_m': round(ramp_m, 1)})
                self.last_lane_plan = plan
                return plan
            v_cap = min(v_cap, v_need)
            ramp_m = avail
            start = 0.0
        plan.update({'pick': pick, 'hops': n_hops, 'ramp_m': round(ramp_m, 1),
                     'start_s_rel': round(start, 1), 'late': late,
                     'v_cap': round(v_cap, 2),
                     'side': 'left' if hops[pick] < 0 else 'right',
                     'armed': start <= 0.0})
        self.last_lane_plan = plan
        return plan

    def _stopline_d(self, planner) -> float | None:
        """다음 정지선까지 뒷축 거리 — 신호 정지선 우선, 없으면 무신호 정지선 최근접."""
        nxt = self._next_stopline(planner)
        if nxt is not None and nxt[0] < float('inf'):
            return float(nxt[0])
        route_s = float(planner.route_s[planner.route_index])
        ahead = [s - route_s for s in self._all_stopline_s(planner) if s - route_s >= 0.0]
        return min(ahead) if ahead else None

    def _ped_guard(self, ap, planner) -> bool:
        """무신호 큐 해제를 막는 보행자 — 회랑 안(|lat| < ped_release_lat_m) 이거나
        경로 쪽으로 오는 중(직전 틱 v_toward > 0). 서 있는 인도 보행자는 아니다."""
        try:
            walkers = list(ap._world.get_actors().filter('*walker*'))
        except Exception:                                  # noqa: BLE001
            return False
        for w in walkers:
            loc = w.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is None or not (0.0 < pr[0] <= self.detect_max_m):
                continue
            if abs(pr[1]) < self.ped_release_lat:
                return True
            vt = self.ped_vt.get(getattr(w, 'id', None))
            if vt is not None and vt > 0.0:
                return True
        return False

    def observe_lights(self, lights) -> None:
        """9910 lights [(id, state)] 를 틱마다 받는다 (run_agent). state 는 플래너가
        갱신하고 여기서는 **보고 시각만** 센다 — `_signal_stale` 의 입력."""
        self._obs_tick += 1
        for lid, _state in lights or []:
            self._light_seen[int(lid)] = self._obs_tick

    def _signal_stale(self, planner) -> dict | None:
        """다음 정지선 신호의 미보고 판정 → 진단 dict, 대상 없음/스위치 off 면 None.

        stale = controller 전부가 signal_stale_s 이상 안 보임 (한 번도 안 보인 것
        포함 — 그때 나이는 첫 관측부터 센다). 관측이 한 번도 없으면(테스트·
        observe_lights 미호출) 판정하지 않는다 — off 와 같다.
        """
        if not (self.sig_stale_queue or self.sig_timeout_go) or self._obs_tick <= 0:
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

    def _is_queue_v2(self, blockers, planner, ap, lg, ego_lane) -> bool:
        """큐 (queue_only, C-3): 정지 객체 ≥ 1 ∧ (A 선두가 정지선 25 m 안
        ∨ B 신호 Red/Yellow ∧ 정지 객체 전부가 자차~정지선 사이).

        blockers 는 standoff 축 회랑(_stop_ok, 신호 무관 정지 ≥ standoff_stop_s).
        2대 이상이면 옛 형태 판정(종으로 벌어짐·횡 비슷)과 통과 폭 판정을 그대로
        두고, 1대에는 적용하지 않는다 — 이 코스의 차로폭(실측 2.75~3.12 m)에서는
        _corridor_passable 이 단일 차량을 절대 통과시키지 못한다 (C 작업1 4번).
        해제: 신호 있음 → 녹색 q_green_release_s 경과 (선두가 떠나면 blockers 에서
        빠져 그 틱에 저절로 해소). 무신호 → q_nosignal_release_s 경과 ∧ 보행자
        가드 아님. 옛 15 s hold 는 없다 — 적색 38 s 에서 큐를 철회해 버렸다.
        """
        self.q_info = None
        # 신호 미보고 → UNKNOWN (스위치 off 면 None 이라 아래 unknown 은 항상 거짓)
        self.last_signal = self._signal_stale(planner) if planner is not None else None
        unknown = bool(self.sig_stale_queue and self.last_signal
                       and self.last_signal.get('signal_stale'))
        if blockers and self.obs_fastpath:
            # E-1: 큐는 차량만이다. 박스가 선두든 사이에 끼었든 큐 형태에서 뺀다.
            veh = [b for b in blockers if not self._is_obstacle(b[3])]
            if not veh:
                self.q_ticks = 0
                self.q_reject = 'obstacle_class'
                return False
            blockers = veh
        if planner is None or not blockers:
            self.q_ticks = 0
            self.q_reject = None
            return False
        if len(blockers) >= 2:
            shape = any((s2 - s1) > self.queue_gap_min_m
                        and abs(l2 - l1) < self.queue_lat_max_m
                        for (s1, l1, _h1, _a1), (s2, l2, _h2, _a2)
                        in zip(blockers, blockers[1:]))
            if not shape:
                self.q_ticks = 0
                self.q_reject = 'shape'
                return False
            if ap is not None and self._corridor_passable(blockers, lg, ego_lane, ap):
                self.q_ticks = 0
                self.q_reject = 'passable'
                return False
        head = blockers[-1]
        nxt = self._next_stopline(planner)
        state = nxt[1] if nxt else None
        d_sl = self._stopline_d(planner)
        cond_a = self._head_near_stopline(planner, head[0])
        # B 는 녹색 첫 틱에 사라지지 않는다 — 직전 틱까지 큐였다면(q_ticks > 0) 녹색
        # q_green_release_s 까지 유지한다. 그래야 "녹색 3 s 경과 ∧ 선두 정지 → 해제"
        # 가 성립한다 (실측 003759/05 t=103.9: 녹색 첫 틱에 PREEMPT → standoff 급정지).
        # UNKNOWN 은 큐 조건 B 에서 Red 와 같다 — 못 본 신호는 적색으로 가정한다.
        sig_hold = (unknown or state in ('Red', 'Yellow')
                    or (state == 'Green' and self.q_ticks > 0
                        and self.green_since_ticks < self.q_green_release_ticks))
        # E-7: 정지선이 red_pause_max_m 보다 멀면 그 신호의 대기열일 수 없다.
        red_near = (d_sl is not None
                    and (self.red_pause_max_m <= 0.0 or d_sl <= self.red_pause_max_m))
        cond_b = (sig_hold and red_near
                  and all(b[0] < d_sl for b in blockers))
        if not (cond_a or cond_b):
            # 직전까지 큐였는데 녹색 유지 시한이 끝나 B 가 사라진 것이면 해제 사유를
            # 'green_expired' 로 남긴다 (진단·테스트가 head_far 와 구분한다).
            expired = (self.q_ticks > 0 and state == 'Green'
                       and self.green_since_ticks >= self.q_green_release_ticks)
            self.q_ticks = 0
            self.q_reject = 'green_expired' if expired else 'head_far'
            return False
        self.q_ticks += 1
        signaled = state in ('Red', 'Yellow', 'Green')
        guard = None
        cond = ('A' if cond_a else '') + ('B' if cond_b else '')
        info = {'cond': cond, 'head_id': int(head[3].id), 'n': len(blockers),
                'q_s': round(self.q_ticks / self.hz, 1),
                'green_s': round(self.green_since_ticks / self.hz, 1), 'ped_guard': None,
                # E-7 진단 — 정지선까지(자차·선두) 거리. 큐 B 오판 사후 판정 근거.
                'd_sl': None if d_sl is None else round(d_sl, 1),
                'head_sl': None if d_sl is None else round(d_sl - head[0], 1)}
        if unknown:
            # 해제 시한이 없다 — 녹색(green_expired)도 무신호(hold_expired)도 적용하지
            # 않는다. 선두가 떠나면 blockers 에서 빠져 그 틱에 저절로 풀린다:
            # "앞차가 서면 서고, 가면 따라간다".
            info['queue_by_unknown'] = True
            self.q_reject = None
            self.q_info = info
            return True
        if signaled:
            # 선두 '여전히 정지' 는 blockers 자체가 보장한다 (정지 객체만 들어온다).
            if self.green_since_ticks >= self.q_green_release_ticks:
                self.q_reject = 'green_expired'
                self.q_info = info
                return False
        else:
            if self.q_ticks >= self.q_nosig_release_ticks:
                guard = self._ped_guard(ap, planner) if ap is not None else False
                info['ped_guard'] = guard
                if not guard:
                    self.q_reject = 'hold_expired'
                    self.q_info = info
                    return False
        self.q_reject = None
        self.q_info = info
        return True

    def _is_queue_legacy(self, blockers, planner=None, ap=None, lg=None, ego_lane=None) -> bool:
        """정지 객체가 **종방향으로** 2대 이상 줄지어 있으면 대기열로 본다.

        신호 구역 판정(_signal_zone)의 사각 보완이다 — 정지선 데이터가 없는
        도로에서도 "줄 서 있으면 신호 대기"로 걸러낸다. 케이스 B(스태거드)와
        구분되는 점: 대기열은 **횡 위치가 비슷하고 종방향으로 벌어져** 있다.
        스태거드는 종방향으로 붙어 있고 횡으로 갈린다.
        """
        # 이 판별기는 **2선**이다 — 적신호·황색 STOP 은 위의 절대 규칙(_red_ahead)이
        # 이미 걸렀으므로, 여기 오는 것은 녹색이거나 신호가 없는 상황뿐이다.
        if planner is not None and self._red_ahead(planner) is not None:
            self.q_ticks = 0
            return False
        if len(blockers) < 2:
            self.q_ticks = 0
            return False
        # 형태 (기존): 종으로 벌어지고 횡이 비슷
        shape = any((s2 - s1) > self.queue_gap_min_m
                    and abs(l2 - l1) < self.queue_lat_max_m
                    for (s1, l1, _h1, _a1), (s2, l2, _h2, _a2)
                    in zip(blockers, blockers[1:]))
        if not shape:
            self.q_ticks = 0
            return False
        # 판별기 ①: 옆으로 지나갈 폭이 있으면 대기열이 아니라 회피 대상이다.
        if ap is not None and self._corridor_passable(blockers, lg, ego_lane, ap):
            self.q_ticks = 0
            self.q_reject = 'passable'
            return False
        # 판별기 ②: 대기열은 선두가 정지선을 향한다.
        if planner is not None and not self._head_near_stopline(planner, blockers[-1][0]):
            self.q_ticks = 0
            self.q_reject = 'head_far'
            return False
        # 1급 안전망: 시한 없는 억제 금지. 대기열은 신호 주기로 풀린다 —
        # queue_hold_s 넘게 안 풀리면 판정을 스스로 철회한다 (무한 정지 방지).
        self.q_ticks += 1
        if self.q_hold_ticks and self.q_ticks > self.q_hold_ticks:
            self.q_reject = 'hold_expired'
            return False
        self.q_reject = None
        return True

    # ── 규칙 2: 데드락 해제 (BREAKOUT) ──────────────────────────────────
    def _obstacle_cause(self, planner, ap, ignore_queue: bool = False,
                        end_m: float | None = None,
                        pdm_hazard_ok: bool = False) -> bool:
        """지금 정지 원인이 **장애물 계열**인가. 하나라도 아니면 거짓.

        BREAKOUT 은 제약을 풀고 전진을 강제하므로, 원인이 신호·보행자·종점
        이면 **절대 발동하면 안 된다**. PDM 이 매 틱 세우는 hazard 플래그와
        kr_rules 자신의 래치를 모두 본다.
        RTOR 래치 중에는 무조건 거짓 — 신호 정지 후보를 놓은 상태라 hazard 가
        서지 않으므로, 교차로 안 정지 차량이 크립·BREAKOUT 을 열지 못하게 막는다
        (그 차량은 IDM 추종·standoff 만).
        """
        if self._rtor_go:
            return False
        if getattr(ap, 'traffic_light_hazard', False):
            if not self._blocker_before_stopline(ap, planner):
                return False
        if getattr(ap, 'walker_hazard', False) or getattr(ap, 'walker_close', False):
            return False
        if getattr(ap, 'stop_sign_hazard', False):
            return False
        if self.latched or self.sl_hold_left > 0:          # 종점 래치 / 정지선 홀드
            return False
        if self.y_decision is not None or self.cross_guard:  # 황색 래치 / 통과 가드
            return False
        end_m = self.active_m if end_m is None else end_m
        if self.last_d_end is not None and self.last_d_end <= end_m:
            return False                                   # route_end 유령차 사정권
        if self.suppress_mode == 'legacy':
            if self._red_ahead(planner) is not None:           # 절대 규칙 (신호 > 회피)
                return False
            if self._signal_zone(planner, ap) is not None:     # 규칙 1
                return False
        elif self._tick_queue and not ignore_queue:        # 큐 뒤에 선 것은 데드락이 아니다 (C-5)
            # ignore_queue 는 never_stall(A2) 전용이다 — UNKNOWN 큐만 그 경로로
            # 들어온다 (_ns_cause). BREAKOUT 은 기본값 False 로 옛 동작 그대로.
            return False
        if self._blocker(ap, planner) is not None:         # 실제로 앞이 막혀 있을 것
            return True
        # PDM 이 차량·OBB 로 세우고 있으면 그것도 장애물 정지다 (never_stall 전용).
        # kr 의 회랑만 보면 놓친다: 시프트 span 이 활성이면 회랑은 **밀린 경로**
        # 기준이라 비어 있고(_blocker None), _try_overtake_inner 도 조기 반환해
        # 기각 시계가 안 돈다. 실측 replay(정적회피집중_01_좌회전2): 목표 0 인
        # 489틱 **전부** vehicle_hazard=True · blocker=False · reject_pending=False
        # 라 이 함수가 거짓이었고, 24 s 정지에 아무 안전망도 안 걸렸다.
        # 위의 배제(신호·보행자·정지표지·종점·큐)를 이미 통과한 뒤라 안전하다.
        if pdm_hazard_ok and getattr(ap, 'vehicle_hazard', False):
            return True
        # E-3: 회랑 후보가 있는데 양쪽 다 기각된 채면 30 m 밖이라도 장애물 원인이다
        return self._reject_pending()

    def _reject_pending(self) -> bool:
        """E-3 시계 입력 — 직전 틱 회피 시도가 양쪽 다 기각됐나 (스위치 꺼지면 거짓)."""
        return self.bo_reject_clock and self.ot_reject_ticks > 0

    def _breakout_reset(self, why=None) -> None:
        if why and self.bo_state == 'BREAKOUT':
            self.bo_exit = why
        self.bo_state = None
        self.bo_level = 0
        self.bo_lvl_ticks = 0
        self.bo_stall_ticks = 0
        self.bo_entry_s = None
        self.bo_ref_s = None

    def breakout_creep(self) -> bool:
        """크립 훅 — autopilot 이 선행차·OBB 후보를 무효화할지 묻는다.

        참이 되는 경우는 **BREAKOUT 최종 단계(L4) 단독**이다. 그 외 어떤
        상태에서도 거짓이어야 한다 — 열리면 앞차·장애물을 그대로 들이받는다.

        A1 이후 **교차로 안에서는 L4 여도 거짓**이다. 교차로 해제
        (junction_creep_release_enable)가 사다리를 돌리므로 L4 도달이 가능해졌는데,
        여기까지 열면 연결로 한복판에서 선행차·OBB 를 지운 채 전진한다. 해제가
        푸는 것은 크립 **상한**뿐이고 PDM 의 IDM·OBB 는 최후 안전망으로 남긴다.
        """
        # A2 3단계 — **최후 수단**. 회랑(선행차·OBB)을 무시하고 전진한다. 교차로
        # 차단(A1)보다도 우선한다: 여기까지 왔다는 것은 다른 모든 완화가 실패해
        # 정지가 deadlock_max_s + 2·step 을 넘겼다는 뜻이고, 그 상태로 계속 서
        # 있는 것(=미완주 확정)보다 1 m/s 접촉 위험이 낫다는 판단이다.
        if self.ns_level >= 3:
            return True
        if (self.j_release and self.bo_level >= self.BO_CREEP
                and self._ap is not None and self._in_junction_lane(self._ap)):
            return False
        return bool(self.bo_state == 'BREAKOUT' and self.bo_level >= self.BO_CREEP
                    and not self.bo_paused)          # 적색 중에는 행동 금지

    def _breakout_tick(self, planner, ap, ego_speed: float) -> None:
        """데드락 상태기계. apply() 가 매 틱 부른다.

        NORMAL ──장애물 원인 정지 stuck_hard_s──> BREAKOUT L1
          L1 제약 완화(1회 제한·회랑 하한·측방 반경)
          L2 실선 허용            ← reasons 에 단계·사유 기록
          L3 여유폭 축소
          L4 크립 강제 (훅)
        진전(route_s +progress_m) 감지 시 NORMAL 복귀.
        L4 에서 무진전이 creep_fail_s 지속되면 CREEP_FAIL — 정지 유지, 기록만.
        **접촉은 실패 조건이 아니다**: 진전이 있는 한 계속한다.
        """
        route_s = float(planner.route_s[planner.route_index])
        if self.bo_state == 'CREEP_FAIL':
            if not self._obstacle_cause(planner, ap):
                self._breakout_reset('cause_gone')
                self.bo_state = None
                return
            # A2 2단계 — CREEP_FAIL 은 원래 탈출구가 'cause_gone' 하나뿐이라
            # 장애물이 안 치워지면 영구다 (코드 주석: "정지 유지 … 미구현").
            # 여기서 L4 로 되돌려 사다리를 다시 돌린다. 무진전 시계만 새로 하고
            # 단계는 L4 그대로다 — 아래에서부터 다시 오를 이유가 없다.
            if self.ns_level >= 2:
                self.bo_state = 'BREAKOUT'
                self.bo_stall_ticks = 0
                self.bo_ref_s = route_s
                print('[kr_rules] never_stall — CREEP_FAIL 해제, BREAKOUT L%d 재개'
                      % self.bo_level, flush=True)
            return

        # 적색·황색 STOP 중에는 **일시정지**한다 — 카운터·단계를 그대로 두고
        # 행동(진입·상승·크립)만 멈춘다. 리셋하면 녹색이 짧은 교차로에서
        # BREAKOUT 이 영영 안 서고, 앞차가 고장 나 있어도 탈출이 안 걸린다.
        # 거리 상한은 E-7 (_red_pause) — 456 m 앞 적신호로 멈추지 않는다.
        if self._red_pause(planner) is not None:
            self.bo_paused = True
            return
        # 교차로 안도 **일시정지** (C-6, queue_only). side 루프가 교차로 lane 에서
        # 시프트를 금지하므로 여기서 크립까지 가면 접촉뿐이다. 리셋이 아니라
        # pause 라 카운터·단계는 보존된다. legacy 는 _signal_zone 이 원인 판정에서
        # 걸러 리셋한다 (옛 동작 그대로).
        # A1: 다만 **상한이 없으면 안 된다**. 연결로 안에 정지 차량이 있으면
        # 시프트는 reject='junction' 으로 영구 기각이고 사다리는 여기서 영구
        # 정지라, 탈출 경로가 하나도 남지 않는다 (실측 02_직진11 31.2 s).
        # junction_release_s 를 넘기면 사다리를 다시 돌린다. 실제 전진은 크립
        # 게이트 ⓪ 가 만들고, 여기서 얻는 것은 **상태 유지**다: 얼어 있으면
        # 카운터가 진입 시점 값에 묶여 연결로를 벗어난 뒤에도 그 값부터 다시
        # 시작하고, 진전 감지(bo_entry_s)가 안 돌아 NORMAL 복귀도 못 한다.
        # 단계가 올라도 위험은 없다 — 시프트는 위 reject='junction' 으로 여전히
        # 금지고, L4 의 크립 훅은 breakout_creep() 이 교차로에서 따로 닫는다.
        if (self.suppress_mode != 'legacy' and self._in_junction_lane(ap)
                and not self._junction_release()):
            self.bo_paused = True
            return
        self.bo_paused = False

        # 순수 정지 틱 — E-3 의 안전 가드(occupied 완화·크립은 정지 stuck_hard_s
        # 경과를 요구) 전용. 원인·상태와 무관하게 자차 속도만 본다.
        self.bo_stop_ticks = 0 if ego_speed >= self.bo_eps else self.bo_stop_ticks + 1

        if not self._obstacle_cause(planner, ap):
            self.bo_stuck_ticks = 0
            if self.bo_state is not None:
                self._breakout_reset('cause_gone')
            return

        # E-3: '막힘' = 정지 **또는** 직전 틱 회피 시도 양쪽 기각 (주행 중 포함).
        # 스위치가 꺼지면 reject_pending 은 항상 거짓 = 정지만 센다 (이전 동작).
        rejected = self._reject_pending()
        moving = ego_speed >= self.bo_eps and not rejected
        self.bo_stuck_ticks = 0 if moving else self.bo_stuck_ticks + 1

        if self.bo_state is None:
            if self.bo_stuck_ticks >= self.bo_hard_ticks:
                self.bo_state = 'BREAKOUT'
                self.bo_level = 1
                self.bo_lvl_ticks = 0
                self.bo_stall_ticks = 0
                self.bo_entry_s = route_s
                self.bo_ref_s = route_s
                self.ot_span = None                        # L1: 1회 제한 해제
                print('[kr_rules] 데드락 해제 진입 — BREAKOUT L1 '
                      f'({"기각" if rejected else "정지"} {self.bo_stuck_ticks / self.hz:.1f} s)',
                      flush=True)
            return

        # 진전 감지 → 정상 복귀. 기준은 **진입 시점**이다 — 무진전 판정의
        # bo_ref_s 와 겹쳐 쓰면 조금씩 계속 나아갈 때 기준이 따라 올라가
        # 복귀가 영영 안 걸린다 (테스트가 잡은 결함).
        # E-3: 기각이 이어지는 동안은 주행 진전을 복귀로 보지 않는다 — 주행 중
        # 시계로 들어온 BREAKOUT 이 다음 틱 2 m 진전으로 바로 풀리면 사다리가
        # 영영 못 오른다. 시프트에 성공하면 기각이 끊겨 그때 진전으로 복귀한다.
        if (not rejected and self.bo_entry_s is not None
                and route_s - self.bo_entry_s >= self.bo_progress_m):
            print('[kr_rules] 데드락 해제 — 진전 %.1f m, 정상 복귀'
                  % (route_s - self.bo_entry_s), flush=True)
            self._breakout_reset('progress')
            self.bo_stuck_ticks = 0
            return

        # 무진전 누적 (진전이 조금이라도 있으면 리셋 — 접촉 여부는 보지 않는다)
        if route_s - (self.bo_ref_s or route_s) > self.bo_creep_eps_m:
            self.bo_stall_ticks = 0
            self.bo_ref_s = route_s
        else:
            self.bo_stall_ticks += 1

        if self.bo_level >= self.BO_CREEP:
            if self.bo_stall_ticks >= self.bo_fail_ticks:
                self.bo_state = 'CREEP_FAIL'
                print('[kr_rules] ⚠ 크립 실패 — %.1f s 무진전. 정지 유지 '
                      '(리스폰 대기·유도는 미구현)' % (self.bo_stall_ticks / self.hz),
                      flush=True)
            return

        self.bo_lvl_ticks += 1
        if self.bo_lvl_ticks >= self.bo_esc_ticks:
            # E-3 안전 가드: 크립(L4)은 IDM·OBB 를 끄므로 주행 중 시계로는 열지
            # 않는다 — 정지 stuck_hard_s 가 지나야 오른다 (L3 에서 대기).
            if (self.bo_level + 1 >= self.BO_CREEP and self.bo_reject_clock
                    and self.bo_stop_ticks < self.bo_hard_ticks):
                self.bo_lvl_ticks = self.bo_esc_ticks       # 포화 — 정지가 차면 즉시
                return
            self.bo_level += 1
            self.bo_lvl_ticks = 0
            self.ot_span = None                            # 각 단계에서 재시도 허용
            if self.bo_level >= self.BO_CREEP:
                # 크립 실패 창은 **크립 중** 무진전을 재야 한다. L1~L3 동안 쌓인
                # 정지 시간을 그대로 쓰면 L4 진입 즉시 실패로 떨어진다.
                self.bo_stall_ticks = 0
            print('[kr_rules] BREAKOUT 단계 상승 → L%d' % self.bo_level, flush=True)

    def _relax_label(self) -> str | None:
        """현재 BREAKOUT 단계가 푸는 게이트 목록 (로그용). L2 미만은 None."""
        if self.bo_level < 2:
            return None
        out = ['side']
        if self.bo_level >= self.zone_relax_lvl:
            out.append('zone_gate')
        if self.bo_level >= self.geom_relax_lvl:
            out.append('shift_ahead')
        if self.bo_level >= self.BO_CREEP:
            out.append('creep')
        return '+'.join(out)

    def _in_junction_lane(self, ap) -> bool:
        """자차 lane 이 교차로 연결로인가 (캐시된 lane-graph 판정, 없으면 ap.junction —
        둘은 같은 함수 VtdMap.is_junction 을 본다)."""
        lg, lane = self._tick_lg, self._tick_ego_lane
        if lg is not None and lane is not None and lane in lg.lanes:
            return lg.lanes[lane]['junction'] != -1
        return bool(getattr(ap, 'junction', False))

    def _junction_release(self) -> bool:
        """교차로 안 자기잠금 해제(A1)가 이번 틱 무장됐나.

        조건: 스위치 on ∧ reject='junction' 연속 junction_release_s ∧ 지금도
        교차로 lane. 마지막 조건을 빼면 연결로를 막 벗어난 틱에도 무장이 남는다
        — 시계는 _try_overtake 가 다음 기각 판정에서야 0 이 되기 때문이다.

        푸는 것은 **종방향 두 개뿐**이다: 크립 게이트(_creep_gate ⓪)와
        _breakout_tick 의 교차로 일시정지. 차로 시프트 금지
        (_try_overtake_inner 의 reject='junction')는 어떤 단계에서도 그대로고,
        breakout_creep() 훅(선행차·OBB 무효화)도 교차로 안에서는 열지 않는다 —
        PDM 의 IDM·OBB 가 최후 안전망으로 남아야 한다.
        """
        if not self.j_release or self.j_reject_ticks < self.j_release_ticks:
            return False
        ap = self._ap
        return ap is not None and self._in_junction_lane(ap)

    # ── 절대 안 멈춤 (A2) ────────────────────────────────────────────────
    def _ns_cause(self, planner, ap) -> bool:
        """never_stall 시계가 도는 조건 — "풀어도 되는 정지" 인가.

        기본은 `_obstacle_cause` 와 **같다**. 신호·보행자·정지표지·종점·황색
        래치·RTOR 는 절대 제외다 — 새 배제 목록을 만들지 않고 그 함수를 그대로
        쓴다 (배제가 두 곳에 적히면 어긋난다).

        딱 하나 넓힌다: **UNKNOWN 큐**. `_is_queue_v2` 는 미보고 신호 큐에
        해제 시한을 두지 않고("해제 시한이 없다"), 같은 상황의 다른 안전망
        `_signal_timeout_tick` 은 `not _tick_corridor` 를 요구해 큐(=회랑 객체가
        있다)에서는 시계가 안 돈다 — 둘이 동시에 죽어 무한 정지가 된다.
        여기서만 시한을 준다.

        **실제로 보고된** 적색·황색 큐는 그대로 제외다 (신호 준수 > 정체 해소).

        종점 배제 폭도 좁힌다. `_obstacle_cause` 는 `route_end.active_m`(150 m)
        안이면 거짓인데, 그 값은 유령차 **후보를 만드는 창**이지 "종점 때문에
        서 있다" 는 뜻이 아니다 — 150 m 앞에서 선 것은 종점이 세운 게 아니다.
        그대로 쓰면 **경로 마지막 150 m 에서 never_stall 이 영영 무장하지 못한다**
        (실측: 정적회피집중 계열은 경로가 짧아 정지 구간 대부분이 이 창 안이다).
        실제로 자차를 세우는 것은 래치(`route_end.latch_m` 12 m)이므로, 그 축의
        해제 거리 `unlatch_m`(30 m)까지만 제외한다. 종점 정지는 그대로 지킨다.
        """
        end_m = self.ns_end_m
        if self._obstacle_cause(planner, ap, end_m=end_m, pdm_hazard_ok=True):
            return True
        if not self._tick_queue or not (self.q_info or {}).get('queue_by_unknown'):
            return False
        return self._obstacle_cause(planner, ap, ignore_queue=True, end_m=end_m,
                                    pdm_hazard_ok=True)

    def _ns_turn_lane_pending(self, planner, ap, ego_speed: float):
        """(c) 지금은 **전진보다 회전 차로 복귀가 급한가** → 진단 dict 또는 None.

        재료는 route.pkl 의 `valid_entry_lanes` (build_route 산출, params
        route.valid_entry_lanes_enable). 그 필드의 목적이 문서상 바로 이것이다 —
        "회피 시프트로 옆 차로에 나간 뒤 원래 차로로 복귀해야 하는가".

        조건 (전부):
          · 지금 세그먼트의 항목이 target='pair' 이고 유효 차로 집합이 **비어
            있지 않다** (빈 집합은 "유효 차로 없음" 이라 복귀할 곳이 없다.
            target='finish' 는 "제약 없음" 이라 대상이 아니다 — 둘을 target 으로
            구분하는 것이 그 필드의 규약이다)
          · 자차 차로가 그 집합에 **없다**
          · 다음 교차로까지 남은 거리 ≤ 복귀 전이거리 + never_stall_turn_margin_m
            (전이거리는 시프트와 같은 식 max(transition_m, shift_k_s·v) — 상수를
            복제하지 않는다)

        왜 전진보다 우선인가: 크립으로 앞으로 나가면 남은 거리가 그만큼 줄어
        복귀 전이가 못 들어간다. 회전 차로에 못 붙은 채 교차로에 들어가는 것은
        정지보다 나쁘다 (경로 이탈 + 회전 불가).
        """
        route = getattr(planner, 'route', None) or {}
        vel = route.get('valid_entry_lanes')
        wps = route.get('waypoint_s')
        if not vel or not wps:
            return None
        route_s = float(planner.route_s[planner.route_index])
        lane = self._tick_ego_lane
        if lane is None:
            return None
        for e in vel:
            if e.get('target') != 'pair' or not e.get('lanes'):
                continue
            seg = int(e['seg'])
            if seg + 1 >= len(wps):
                continue
            s0, s1 = float(wps[seg]), float(wps[seg + 1])
            if not (s0 <= route_s < s1):
                continue
            if tuple(lane) in {tuple(k) for k in e['lanes']}:
                return None                        # 이미 유효 차로에 있다
            need = max(self.ot_trans_m, self.shift_k_s * max(ego_speed, 0.1)) \
                + self.ns_turn_margin_m
            left = s1 - route_s
            if left > need:
                return None                        # 아직 복귀할 거리가 넉넉하다
            return {'seg': seg, 'turn': e.get('turn'), 'left_m': round(left, 1),
                    'need_m': round(need, 1), 'lane': list(lane),
                    'want': [list(k) for k in e['lanes']]}
        return None

    def _never_stall_tick(self, planner, ap, ego_speed: float) -> None:
        """A2 시계 — apply 가 _tick_cache 뒤에 틱당 1회 부른다.

        단계 (`ns_level`):
          0  정상
          1  deadlock_max_s 경과 — 크립 바닥을 깐다. `stop_gap`·`no_size` 블록과
             크립 지연 게이트를 풀고, 기준선 **밖**(모드 A)도 크립을 거치게 한다.
          2  +step — CREEP_FAIL 을 풀어 사다리를 다시 돌린다.
          3  +step — 최후. breakout_creep() 훅을 강제로 열어 선행차·OBB 후보를
             무효화하고 never_stall_force_v 로 전진한다. **접촉을 감수하는
             단계다** — 그래서 마지막이고 속도가 1 m/s 다.

        시계는 `_ns_cause` 가 참이고 진전이 progress_m 미만인 틱만 센다. 진전이
        있으면 기준을 옮기고 0 으로 되돌린다 — 조금씩이라도 가고 있으면 stall 이
        아니다.
        """
        if not self.ns_enable:
            self.ns_level = 0
            self.ns_info = None
            return
        route_s = float(planner.route_s[planner.route_index])
        if not self._ns_cause(planner, ap):
            self.ns_ticks = 0
            self.ns_ref_s = route_s
            self.ns_level = 0
            self.ns_turn_ticks = 0
            self.ns_info = None
            return
        if self.ns_ref_s is None:
            self.ns_ref_s = route_s                 # 첫 틱 — 기준만 잡고 세기 시작
        elif route_s - self.ns_ref_s >= self.bo_progress_m:
            self.ns_ref_s = route_s                 # 진전 — 시계를 새로
            self.ns_ticks = 0
            self.ns_level = 0
            self.ns_turn_ticks = 0
            self.ns_info = None
            return
        self.ns_ticks += 1
        if self.ns_ticks < self.ns_max_ticks:
            self.ns_level = 0
            # 무장 전에도 시계를 남긴다 — "왜 안 걸렸나" 를 로그로 답할 수 있어야
            # 한다 (실측 13로그에서 단계 0 만 보고 원인을 못 좁혔다).
            self.ns_info = {'state': 'ARMING',
                            'ns_s': round(self.ns_ticks / self.hz, 1)}
            return

        # (c) 회전 차로 복귀 우선 — 단계 상승을 보류하고 시프트를 원복한다.
        # 보류에도 시한이 있다: never_stall 이 새 무한 대기를 만들면 안 된다.
        turn = self._ns_turn_lane_pending(planner, ap, ego_speed)
        if turn is not None and self.ns_turn_ticks < self.ns_turn_hold_ticks:
            self.ns_turn_ticks += 1
            if self.ot_span is not None:
                self._restore_span(planner)         # 계획 경로가 회전 차로로 데려간다
            self.ns_level = 0
            self.ns_info = dict(turn, state='TURN_LANE_RETURN',
                                hold_s=round(self.ns_turn_ticks / self.hz, 1),
                                ns_s=round(self.ns_ticks / self.hz, 1))
            return

        over = self.ns_ticks - self.ns_max_ticks
        lvl = 1 + (over // self.ns_step_ticks if self.ns_step_ticks > 0 else 2)
        new_level = int(min(3, lvl))
        if new_level != self.ns_level:
            print('[kr_rules] ⚠ never_stall 단계 %d — 무진전 %.1f s'
                  % (new_level, self.ns_ticks / self.hz), flush=True)
        self.ns_level = new_level
        self.ns_info = {'state': 'NEVER_STALL', 'level': self.ns_level,
                        'ns_s': round(self.ns_ticks / self.hz, 1),
                        'turn_hold_s': round(self.ns_turn_ticks / self.hz, 1),
                        'v': round(self.ns_force_v if self.ns_level >= 3
                                   else self.ns_creep_v, 2)}

    def _ns_creep_v(self) -> float:
        """단계별 크립 속도 [m/s]. 3단계만 회랑 무시 속도를 쓴다."""
        return self.ns_force_v if self.ns_level >= 3 else self.ns_creep_v

    @staticmethod
    def _shift_profile(s, s0, end, L):
        """route.py shift_route_smoothly 의 전이 계수를 s 격자로 — 원문 조건 그대로:
        시작 L 안이고 시작에 더 가까우면 나가는 전이, 끝 L 안이면 복귀 전이, 사이는 1."""
        f = np.zeros_like(s)
        inside = (s >= s0) & (s < end)
        out = inside & (s <= s0 + L) & ((s - s0) < (end - s))
        back = inside & ~out & (s >= end - L)
        f[out] = -np.cos(np.clip((s[out] - s0) / L, 0.0, 1.0) * np.pi) / 2.0 + 0.5
        f[back] = -np.cos(np.clip((end - s[back]) / L, 0.0, 1.0) * np.pi) / 2.0 + 0.5
        f[inside & ~out & ~back] = 1.0
        return f

    def _span_clear_model(self, planner, ap, b, left, trans_m):
        """시프트 이격 계측 모델 — _shift_placement(진입·복귀 배치)와 _extend_span(연장)이
        같은 식을 쓴다. 자차 인덱스부터 b 까지의 목표 오프셋 D(s)(현재 경로 대비로 보정)와
        띠 안 객체를 모아, clear_at(s0, end) 가 cos 전이 프로파일의 최소 이격을 돌려준다.
        못 재면 None (호출자가 관례대로 처리)."""
        ppm = float(getattr(planner, 'points_per_meter', 10))
        i0 = int(planner.route_index)
        try:
            d = np.asarray(planner.planned_lateral_offsets(i0, int(b), left, step_pts=int(ppm)),
                           dtype=float)
        except Exception:                                  # noqa: BLE001
            return None
        if len(d) < 3:
            return None
        # D 는 **원 경로점** 기준 목표 오프셋인데 객체 lat(_project)은 **현재 경로점**
        # 기준이다. 앞선 span 이 살아 있으면 둘이 다르다 (104648: 원 경로 +1 차로 위에
        # 서 있음). 실제 블렌드는 new = f·loc + (1−f)·cur 이므로 현재 경로 대비 이동은
        # f·(D − t_cur). t_cur 은 같은 외적 규약으로 원 경로 접선에 대해 잰다.
        try:
            orig = np.asarray(planner.original_route_points)[:, :2]
            cur = np.asarray(planner.route_points)[:, :2]
            idx = np.arange(i0, min(int(b), len(orig) - 1), int(ppm))[:len(d)]
            tan = orig[idx + 1] - orig[idx]
            tan /= (np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9)
            dv = cur[idx] - orig[idx]
            t_cur = tan[:, 0] * dv[:, 1] - tan[:, 1] * dv[:, 0]
            d = d[:len(t_cur)] - t_cur
        except Exception:                                  # noqa: BLE001
            pass                                           # 목 플래너 등: 원 경로 = 현재 경로
        hw_e = float(self.cfg['vehicle']['width']) / 2.0
        hl_e = float(self.cfg['vehicle']['length']) / 2.0
        b_rel = (int(b) - i0) / ppm
        L = float(trans_m)
        band = (min(0.0, float(d.min())), max(0.0, float(d.max())))
        objs = []
        pts = planner.route_points
        for s_rel, lat, hw_o, act in self._corridor_blockers(ap, planner, lat_band=band):
            j = min(len(pts) - 2, i0 + int(round(s_rel * ppm)))
            tan = pts[j + 1, :2] - pts[j, :2]
            ang = float(np.arctan2(tan[1], tan[0])) if np.linalg.norm(tan) > 1e-9 else 0.0
            yaw = getattr(act, 'yaw_deg', None)
            if yaw is None:
                try:
                    yaw = float(act.get_transform().rotation.yaw)
                except Exception:                          # noqa: BLE001
                    yaw = _math.degrees(ang)
            dl = _math.radians(float(yaw)) - ang
            hl_o = float(getattr(getattr(getattr(act, 'bounding_box', None), 'extent', None),
                                 'x', 0.0) or 0.0)
            ht = hl_o * abs(_math.sin(dl)) + hw_o * abs(_math.cos(dl))
            hs = hl_o * abs(_math.cos(dl)) + hw_o * abs(_math.sin(dl))
            objs.append((float(s_rel), float(lat), ht, hs, act.id))
        s = np.arange(0.0, b_rel + 0.25, 0.25)
        D = np.interp(s, np.arange(len(d), dtype=float), d)
        # 헤딩·곡률은 cos 전이 자체의 최대값으로 막는다 — 목표 차로가 laneSection
        # 경계에서 튀면(B-7 계단, get_left_lane None 경계) D 에 단차가 생겨 수치 미분이
        # 폭주한다 (104648 좌측: 사지타 188 m → id8 이격 −20.85). 차체는 단차를 따를
        # 수 없으므로 물리 상한이 맞다: 기울기 D·π/2L, 곡률 D·π²/2L².
        d_max = float(np.abs(D).max()) if len(D) else 0.0
        slope_cap = d_max * np.pi / (2.0 * L)
        kappa_cap = d_max * np.pi * np.pi / (2.0 * L * L)

        def clear_at(s0, end):
            """전이 시작 s0·끝 end(자차 기준 m)의 최소 이격 → (min, {id: 이격})."""
            if end - s0 < 1.0:
                return -1e9, {}
            t_e = D * self._shift_profile(s, s0, end, L)
            slope = np.clip(np.gradient(t_e, s), -slope_cap, slope_cap)
            psi = np.arctan(slope)
            sag = np.minimum(np.abs(np.gradient(slope, s)), kappa_cap) * hl_e * hl_e / 2.0
            w_e = hw_e / np.cos(psi) + sag
            per = {}
            for s_j, lat_j, ht, hs, oid in objs:
                m = np.abs(s - s_j) <= hs
                if not m.any():
                    continue
                per[oid] = float((np.abs(t_e[m] - lat_j) - w_e[m] - ht).min())
            return (min(per.values()) if per else float('inf')), per
        return {'clear_at': clear_at, 'objs': objs, 'b_rel': b_rel}

    def _gap_free_span(self, objs, s_b, hs_b):
        """목표 차로 객체 사이의 **빈 구간** 중 차단물(s_b)을 덮는 것 → (lo, hi).

        objs 는 `_span_clear_model` 이 이미 모아 둔 (s_rel, lat, ht, hs, id) 다 —
        `_corridor_blockers(lat_band=...)` 로 목표 오프셋까지 휩쓰는 띠 안에서
        걸러진 것이라 새 탐색을 만들지 않는다. 여유는 자차 반길이 + 회랑과 같은
        임계(obstacle_clearance_m). 못 찾으면 None.

        이 값은 **판정에 쓰지 않는다** — 진단·탐색 범위 축소용이다. span 을 빈
        구간 안에 넣는 규칙은 너무 엄격하다(202508: 빈 구간 21.3 m → 최대 전이
        6.0 < transition_m 12.0 이라 기각돼 버린다). 전이 시작·끝에서는 횡변위가
        0 이라 옆 차로 객체와 무관하기 때문이다.
        """
        pad = float(self.cfg['vehicle']['length']) / 2.0 + \
            float(self.cfg['percep'].get('obstacle_clearance_m', 0.3))
        occ = sorted((s - hs - pad, s + hs + pad) for s, _lat, _ht, hs, _oid in objs
                     if abs(s - s_b) > hs_b + hs)      # 차단물 자신은 뺀다
        lo = 0.0
        for o0, o1 in occ:
            if o0 > s_b:
                return (round(lo, 1), round(o0, 1))
            lo = max(lo, o1)
        return (round(lo, 1), None)

    @staticmethod
    def _gap_fit_place(pr0, ext_x, extent, ahead_m, trans_m, eb, ea):
        """(전이, 앞여유, 뒤여유) → 자차 기준 (전이 시작 s0, span 끝) [m].

        zone 게이트의 `start_est` 와 **같은 식**이다 (span 시작 = 첫 객체 − 반길이 −
        앞여유 − 전이, 자차 앞 ahead 로 클램프). 끝은 그 대칭 — 마지막 객체(첫 객체
        + 연쇄 extent) + 반길이 + 뒤여유 + 전이.

        `_side_pass` 의 `span_m`(= 2·trans + before + after + extent)을 쓰면 안 된다.
        그건 점선 커버리지 게이트용 근사라 **객체 반길이 2·ext_x 가 빠져 있다** —
        202508 에서 실제 87.7 대신 83.3 이 나와 gap_fit 이 최적 배치를 놓쳤다
        (2026-09-05 실측).
        """
        s0 = max(pr0 - ext_x - trans_m - eb, ahead_m)
        return s0, pr0 + extent + ext_x + ea + trans_m

    def _gap_fit_search(self, planner, ap, actor, chain, trans_m, ahead_m, extra_after,
                        span_plan=None):
        """옆 차로 빈 구간에 맞춰 (전이, 앞여유, 뒤여유) 를 다시 고른다 → (사용값, 진단).

        왜: span 은 차단물 하나를 중심으로 대칭으로 그려지고 전이 길이가 **결정
        시점 속도**에 비례한다(trans = max(transition_m, shift_k_s·v)). 빠를수록
        span 이 길어져 목표 차로에 정차한 차를 삼킨다 — 실주행 3건이 이 형태로
        고착했다 (2026-09-05 202508/211215/212101: entry_base_clear −1.16~−1.81,
        entry_plateau_ids [2], chain [3] → 전부 CREEP_FAIL. 계속 주행한 3건은
        +1.12~+1.13, 플래토 없음).

        판정은 기존 `_span_clear_model.clear_at` 하나로 한다. 202508 실측:
            현행 trans 21.2, span 45.1~106.9 → 이격 0.00 (id2·id4 두 대)
            최적 trans 12.0, eb 0, ea 0, span 59.3~87.7 → 이격 0.86
            전이 21.2 로는 (a,b) 를 어떻게 잡아도 통과 조합이 하나도 없다.

        **기각하지 않는다.** 더 나은 배치를 못 찾으면 현행 값을 그대로 돌려준다 —
        clear_at 이 실제보다 보수적인 경우가 있어(211630/211718 2번째 시프트:
        clear_at 0.08 인데 실제 통과 이격 1.09/1.06) 기각으로 쓰면 잘 가던 케이스를
        막는다. 전이를 줄이는 방향이므로 geom need(trans+ahead+margin)도 함께
        줄어 이미 통과한 게이트를 다시 깨지 않는다.
        """
        base = (trans_m, self.ot_before_m, extra_after)
        plan = span_plan if span_plan is not None else self.last_span_plan
        if not self.gap_fit or plan is None:
            return base, None
        pr = self._project(planner, actor.get_location().x, actor.get_location().y)
        if pr is None:
            return base, None                              # 못 재면 통과 (관례)
        bb = getattr(actor, 'bounding_box', None)
        ext_x = float(getattr(getattr(bb, 'extent', None), 'x', 0.0) or 0.0)
        pr0, extent = float(pr[0]), float(chain.get('extent_m', 0.0))
        b0, left = int(plan[1]), bool(plan[2])
        m0 = self._span_clear_model(planner, ap, b0, left, trans_m)
        if m0 is None:
            return base, None                              # 못 재면 통과 (관례)
        clr = self.gap_fit_clear_m

        def score(model, cand):
            s0, end = self._gap_fit_place(pr0, ext_x, extent, ahead_m, *cand)
            v, _per = model['clear_at'](s0, min(end, model['b_rel']))
            return v

        cur = score(m0, base)
        diag = {'gap_fit_on': True,
                'gap_clear_base': round(cur, 2) if abs(cur) < 1e8 else None,
                'gap_free_span': self._gap_free_span(m0['objs'], pr0, ext_x)}
        if cur >= clr:
            return base, dict(diag, gap_fallback='base_ok')
        best = (cur, base)
        t = trans_m
        while t > self.ot_trans_m + 1e-6:
            t = max(self.ot_trans_m, t - self.gap_fit_step_m)
            m = self._span_clear_model(planner, ap, b0, left, t)
            if m is None:
                continue
            for cand in ((t, 0.0, 0.0), (t, self.ot_before_m, extra_after)):
                v = score(m, cand)
                if v > best[0] + 1e-9:
                    best = (v, cand)
            if best[0] >= clr:
                break
        if best[1] == base:
            return base, dict(diag, gap_fallback='no_better')
        t, eb, ea = best[1]
        return best[1], dict(diag, gap_T_sel=round(t, 1), gap_eb=round(eb, 1),
                             gap_ea=round(ea, 1), gap_clear=round(best[0], 2))

    def _shift_placement(self, planner, ap, trans_m, ahead_m, extra_after, after_locked):
        """전이 배치 탐색 → (delay, after, 진단). 막히면 delay None.

        last_span_plan(기본 배치의 a, b, left) 위에서 시프트 폭 D(s)(planned_lateral_
        offsets, 1 m 격자)와 cos 전이로 자차 중심 궤적 t_e(s) 를 그린다. 자차는 경로를
        따르므로 각 종방향 지점 s 에서 차체 단면은 t_e(s) 를 중심으로 폭 hw/cosψ
        (ψ = atan(dt_e/ds), 비스듬한 단면) + 사지타(κ·hl²/2, 직선 차체 대 곡선 경로)
        의 띠다. 객체는 yaw 를 경로 접선에 대해 풀어 유효 반폭(ht)·종반폭(hs) 을 쓰고,
        객체 종구간 |s − s_j| ≤ hs 안에서 이격 = |t_e − lat| − w_e − ht 의 최소를 본다.
        OBB 대조(재구성 경로 100310): 기본 배치 −0.39 대 OBB −0.44, delay 5.5 에서
        1.4 대 1.44. 임계는 percep.obstacle_clearance_m (회랑과 같은 축).
        기본 배치(delay 0, after 현행)가 전부 임계 이상이면 그대로. 아니면
        delay ∈ [0, max] × after ∈ [min, 현행] 격자에서 최소 이격 최대화(동률이면
        기본에 가까운 쪽). after_locked(E-2 zone 연장)면 after 는 줄이지 않는다 —
        연장은 교차로 출구 뒤로 끝을 미룬 것이라 줄이면 교차로 안에서 복귀한다.
        전이 길이는 바꾸지 않는다 (_shift_speed_cap·geom need 와 같은 값).
        """
        a, b, left = self.last_span_plan
        ppm = float(getattr(planner, 'points_per_meter', 10))
        i0 = int(planner.route_index)
        model = self._span_clear_model(planner, ap, b, left, trans_m)
        if model is None:
            return None, None, {}                          # 못 재면 통과 (관례)
        clr = float(self.cfg['percep'].get('obstacle_clearance_m', 0.3))
        a_rel = (int(a) - i0) / ppm
        b_rel = model['b_rel']
        clear_at = model['clear_at']

        def clear_for(delay, after):
            return clear_at(max(a_rel, ahead_m + delay), b_rel + (after - extra_after))

        base_min, base_per = clear_for(0.0, extra_after)
        info = {'entry_base_clear': round(base_min, 2) if base_min < 1e8 else None,
                'entry_block_ids': [o for o, c in base_per.items() if c < clr]}
        if base_min >= clr:
            info.update({'entry_delay_m': 0.0, 'entry_start_rel_m': round(a_rel, 1),
                         'exit_after_m': round(extra_after, 1), 'entry_clear': info['entry_base_clear']})
            return 0.0, extra_after, info
        n_d = int(round(self.entry_max_delay_m / max(self.entry_step_m, 0.1)))
        delays = [k * self.entry_step_m for k in range(n_d + 1)]
        lo_after = extra_after if after_locked else min(self.exit_min_after_m, extra_after)
        afters = [extra_after - k for k in range(int(_math.floor(extra_after - lo_after)) + 1)]
        grid = [(0.0, extra_after, base_per)]
        for delay in delays:
            for after in afters:
                grid.append((delay, after, clear_for(delay, after)[1]))
        # 배치로 고칠 수 있는 객체만 판정한다 — 격자 어느 (delay, after) 에서도 임계를
        # 못 넘는 객체(플래토 위 목표 차로 객체, 예: 104807 연쇄 끝 id5 옆 id4)는
        # 진입·복귀를 옮겨도 못 지난다. 그것은 시프트 뒤 standoff·IDM 이 다루는
        # **다음 차단물**이지 배치 문제가 아니므로 이전처럼 시프트를 만든다
        # (기각하면 이전보다 못 간다). 진단에 entry_plateau_ids 로 남긴다.
        ids = set().union(*[set(per) for _, _, per in grid])
        best_of = {o: max(per.get(o, float('-inf')) for _, _, per in grid) for o in ids}
        relevant = [o for o, hi in best_of.items() if hi >= clr]
        info['entry_plateau_ids'] = sorted(o for o in ids if o not in relevant)
        info['entry_block_ids'] = [o for o in info['entry_block_ids'] if o in relevant]

        def score(per):
            vals = [per[o] for o in relevant if o in per]
            return min(vals) if vals else float('inf')

        best = None
        for delay, after, per in grid:                 # 동률이면 기본(첫 원소)에 가까운 쪽
            sc = score(per)
            if best is None or sc > best[0] + 1e-6:
                best = (sc, delay, after, per)
        c, delay, after, per = best
        info.update({'entry_clear': round(c, 2) if abs(c) < 1e8 else None,
                     'entry_delay_m': round(delay, 1),
                     'entry_start_rel_m': round(max(a_rel, ahead_m + delay), 1),
                     'exit_after_m': round(after, 1)})
        if c < clr:
            return None, None, info
        return delay, after, info

    @staticmethod
    def _nth_neighbor(lg, key, side: str, n: int):
        """side 로 n 칸 떨어진 차로. 도중에 끊기면 None."""
        for _ in range(max(1, int(n))):
            key = lg.neighbor(key, side)
            if key is None:
                return None
        return key

    def _mid_lanes_clear(self, lg, planner, ap, ego_lane, side, n_hops) -> bool:
        """두 칸 이상 갈 때 **중간 차로**가 램프 구간에서 비어 있나.

        중간 차로는 목적지가 아니다 — 거기 서서 기다릴 것이 아니라 지나간다.
        그래서 `_side_is_clear`(차로 전체 점유)를 걸면 안 되고, **램프가 훑는
        s 구간**에 객체가 있는지만 보면 된다. 범위는 차로 지도가 계산한 램프
        길이 + `shift_ahead_m` 이다 (같은 값을 두 곳에서 다시 만들지 않는다).

        스위치는 따로 없다 — `_lm_hops` 가 None 이면 호출되지 않는다.
        """
        lp = self.last_lane_plan or {}
        reach = float(lp.get('ramp_m') or 0.0) + self.shift_ahead_m
        if reach <= 0.0:
            return True
        mids = set()
        k = ego_lane
        for _ in range(int(n_hops) - 1):
            k = lg.neighbor(k, side)
            if k is None:
                return False
            mids.add(k)
        if not mids:
            return True
        try:
            actors = list(ap._world.get_actors())
        except Exception:                                  # noqa: BLE001
            return False
        ego_id = ap._vehicle.id
        hops = {m: 0 for m in mids}
        for a in actors:
            if a.id == ego_id:
                continue
            if float(getattr(a, 'speed', 0.0)) >= self.ot_v_max:
                continue                                   # 움직이는 것은 지나간다
            pr = self._project(planner, a.get_location().x, a.get_location().y)
            if pr is None or not (0.0 < pr[0] <= reach):
                continue                                   # 램프 구간 밖
            if self._actor_lanes(lg, a, hops):
                return False
        return True

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

        E-3 회계: 본문(_try_overtake_inner)이 "후보가 있었는데 양쪽 다 기각" 이면
        참을 돌려주고, 여기서 연속 기각 틱을 센다. BREAKOUT 시계가 이 값을 읽는다.
        """
        rejected = self._try_overtake_inner(ap, planner, ego_speed)
        self.ot_reject_ticks = self.ot_reject_ticks + 1 if rejected else 0
        # A1: 교차로 **전용** 시계. 사유가 'junction' 인 기각만 센다 — 시프트가
        # 살아 있는 다른 기각(occupied·geom·no_neighbor)은 정상적으로 풀릴 수
        # 있으므로 교차로 해제를 무장시키면 안 된다. 조기 반환(SHIFT_ACTIVE·
        # 큐 억제 등)은 rejected=False 라 여기서 0 이 되고, 그게 맞다.
        self.j_reject_ticks = (self.j_reject_ticks + 1
                               if rejected and self.last_overtake == 'junction' else 0)
        if rejected and self.last_avoid is not None:
            self.last_avoid['reject_s'] = round(self.ot_reject_ticks / self.hz, 1)
            if self.j_release and self.j_reject_ticks:
                # 스위치가 꺼져 있으면 키 자체를 남기지 않는다 — off 는 로그까지
                # 이전과 동일해야 회귀 비교(54 지문)가 성립한다.
                self.last_avoid['j_reject_s'] = round(self.j_reject_ticks / self.hz, 1)
                self.last_avoid['j_release'] = self._junction_release()

    def _try_overtake_inner(self, ap, planner, ego_speed: float) -> bool:
        """_try_overtake 본문. 반환 = 이번 틱 회피 시도가 전부 기각됐나 (E-3)."""
        red_hold = self.ot_span is not None and self._red_ahead(planner) is not None
        # 지나갔으면 원복 (다음 장애물용). E-6: 적색이어도 **원복이 먼저다** —
        # 이미 통과한 span 을 쥔 채 SHIFT_HOLD 로 반환하면 다음 장애물의 회랑·
        # standoff·회계가 전부 멈춘다 (실측 2026-09-03 020439/01·03 접촉).
        # 스위치가 꺼지면 옛 순서(홀드 먼저) 그대로.
        if (self.ot_span is not None and planner.route_index > self.ot_span[1]
                and (self.hold_restore or not red_hold)):
            self._restore_span(planner)
            return False

        # 시프트 진행 중 억제 구역에 걸렸다면 — **원복하지 않는다**.
        # 횡위치를 유지하고 종방향은 정지 후보(④′ 프로파일·홀드)에 맡긴다.
        # 급조향으로 차로 중앙에 복귀하려 들면 정지선 앞에서 조향이 튄다.
        # 차로 중앙 복귀는 span 끝(위 원복)에서만 일어난다 — 녹색으로 바뀌어
        # 다시 달리기 시작하면 자연히 그 지점을 통과하며 복귀한다.
        if red_hold:
            self.last_avoid = {'state': 'SHIFT_HOLD', 'suppress': 'red_ahead',
                               'span': list(self.ot_span)}
            if self.hold_restore and self.span_active_standoff:
                # E-6: 홀드 중에도 span 활성과 같이 본다 — 복귀 전이 위 다음
                # 장애물에 standoff 가 걸려야 한다. 생성만 건너뛴다.
                corridor = self._corridor_blockers(ap, planner)
                self._standoff_target(ap, planner, corridor)
                self._blocked_account(ap, planner, ego_speed)
                self.last_avoid.update(
                    {'blocker': corridor[0][3].id if corridor else None,
                     's_rel': round(corridor[0][0], 1) if corridor else None})
            return False

        if self.ot_span is not None and planner.route_index > self.ot_span[1]:
            self._restore_span(planner)                    # (E-6 꺼짐 + 적색 아님)
            return False
        if not self.ot_enabled:
            return False
        if self.ot_span is not None:
            # 시프트 활성 중 (B-9 (5)) — **생성**만 건너뛴다. 회랑(밀린 경로 기준)·
            # standoff·막힘 회계는 계속 돈다: 복귀 전이 위에 다음 장애물이 있으면
            # standoff 가 25 m 앞에 세운다. 적색이면 위 SHIFT_HOLD 가 먼저 반환한다.
            # false 면 이전 동작(아무것도 안 봄).
            if self.span_lost_restore and self._span_targets_lost(ap, planner):
                self._restore_span(planner)
                self.last_avoid = {'state': 'RESTORE', 'why': 'targets_lost'}
                return False
            if self.span_active_standoff:
                corridor = self._corridor_blockers(ap, planner)
                self._standoff_target(ap, planner, corridor)
                self._blocked_account(ap, planner, ego_speed)
                self.last_avoid = {'state': 'SHIFT_ACTIVE', 'span': list(self.ot_span),
                                   'blocker': corridor[0][3].id if corridor else None,
                                   's_rel': round(corridor[0][0], 1) if corridor else None}
                if self.span_extend:
                    self._extend_span(ap, planner, corridor, ego_speed)
            return False

        lg = getattr(planner, 'lg', None)
        ego_lane = getattr(ap, '_kr_ego_lane', None) or self._ego_lane(lg, ap)
        if self.suppress_mode == 'legacy':
            # ── 절대 규칙: 적신호·황색 STOP 앞에서는 회피 자체가 없다 ─────
            # 우선순위 불변식(신호 준수 > 회피). 거리 무관이며 PREEMPT/WAIT/
            # REACTIVE/BREAKOUT 전부 미발동이다. (legacy 전용 — queue_only 는
            # 적색을 일시정지로만 쓴다: obj_ticks pause·bo_paused·SHIFT_HOLD)
            d_red = self._red_ahead(planner)
            if d_red is not None:
                self.ot_blocked_ticks = 0
                self.q_ticks = 0                               # 대기열 타이머도 리셋
                self.wait_target_d = None
                self.last_avoid = {'state': 'SUPPRESS', 'suppress': 'red_ahead',
                                   'sup_d': round(d_red, 1)}
                return False

            # ── 규칙 1: 신호 구역 억제 (전 상태 공통 게이트) ──────────────
            zone = self._signal_zone(planner, ap)
            if zone is not None:
                self.ot_blocked_ticks = 0
                self.last_avoid = {'state': 'SUPPRESS', 'suppress': zone[0],
                                   'sup_d': zone[1]}
                return False

        corridor = self._corridor_blockers(ap, planner)
        if self.suppress_mode == 'legacy':
            if self._is_queue(corridor, planner, ap, lg, ego_lane):
                self.ot_blocked_ticks = 0
                self.last_avoid = {'state': 'SUPPRESS', 'suppress': 'queue',
                                   'n': len(corridor), 'q_s': round(self.q_ticks / self.hz, 1)}
                return False
        else:
            # queue_only (C-4): standoff 는 큐와 무관하게 **항상** 산출한다 — 큐 뒤에도
            # 25 m 앞에 서는 것이 설계다. 억제는 캐시된 큐 판정 하나뿐이다.
            self._standoff_target(ap, planner, corridor)
            if self._tick_queue:
                self.ot_blocked_ticks = 0
                self.last_avoid = {'state': 'SUPPRESS', 'suppress': 'queue',
                                   'queue': self.q_info, 'n': len(self._tick_corridor),
                                   'q_s': round(self.q_ticks / self.hz, 1)}
                return False

        # ── 규칙 3 + WAIT: 관찰하며 접근, 시간 예산이 다하면 시프트 ────────
        # 일률 6 s 관찰은 못 쓴다 — 10.6 m/s 에서 잔여 16 m 인데 전이가 42 m 면
        # 이미 늦는다. **시간 예산 규칙**: 관찰 중에도 standoff 속도 상한을 걸어
        # 감속시키고(관찰 감속), obj_s ≥ obj_static_s 이면서 남은 여유시간
        #   (d − standoff)/v  <  (wait_before_shift_s − obj_s)
        # 이면 더 못 기다리므로 즉시 시프트한다. 아니면 최대 wait 까지 관찰.
        actor = None
        preempt = False
        if self.suppress_mode == 'legacy':
            self._standoff_target(ap, planner, corridor)
        if corridor:
            s_rel, lat, _hw, cand = corridor[0]
            obj_s = self.obj_ticks.get(cand.id, 0) / self.hz
            standoff = max(self.shift_latest_m, self.shift_k_s * max(ego_speed, 0.1))
            t_left = (s_rel - standoff) / max(ego_speed, 0.1)
            budget = self.wait_s - obj_s
            base = {'blocker': cand.id, 's_rel': round(s_rel, 1), 'lat': round(lat, 2),
                    'obj_s': round(obj_s, 1), 'need_m': round(standoff, 1),
                    't_left': round(t_left, 1), 'budget': round(budget, 1)}
            # 정적 조건은 _static_ok — 장애물 클래스(E-1)는 관찰 없이 참이라
            # 예산 규칙이 t_left < budget 하나로 줄어든다.
            if self.preempt_latch_id is not None and self.preempt_latch_id != cand.id:
                self.preempt_latch_id = None                # 차단물이 바뀌었다 — 새로
            latched = self.preempt_latch and self.preempt_latch_id == cand.id
            if (self._static_ok(cand) and t_left < budget) or latched:
                actor, preempt = cand, True
                self.last_avoid = dict(base, state='PREEMPT', latched=latched)
            elif obj_s >= self.wait_s:
                actor, preempt = cand, True                 # 대기 만료 — 그래도 안 감
                self.last_avoid = dict(base, state='WAIT_EXPIRED')
            else:
                self.last_avoid = dict(base, state='WAIT')  # 앞차 출발 기회를 준다
            if actor is not None and self.preempt_latch:
                self.preempt_latch_id = cand.id             # E-4 래치 (기각돼도 유지)
        else:
            self.preempt_latch_id = None

        # ── '막힌 채 정지' 회계 (B-10) ───────────────────────────────────
        # **자차 상태만으로** 센다. 예전에는 이 회계가 아래 `if actor is None:`
        # 안에 있어서, 회랑 후보가 있어 PREEMPT/WAIT_EXPIRED 로 들어오면
        # (actor 가 None 이 아니므로) 카운터가 아예 증가하지 않았다. 그래서
        # side 루프 게이트(no_neighbor / span_into_zone / solid / occupied /
        # center_line / kappa)가 기각하면 매 틱 같은 일을 반복하고 **REACTIVE 가
        # 영원히 무장되지 않았다** (replay 실측: 기각 10틱 동안 ot_blocked_ticks 0,
        # REACTIVE 0, BREAKOUT 0).
        # 시프트에 성공하면 아래에서 0 으로 리셋하는 기존 관례는 그대로다.
        self._blocked_account(ap, planner, ego_speed)

        # ── REACTIVE: 막힌 채 정지가 지속되면 (기존 경로) ─────────────────
        # 회랑 후보(PREEMPT/WAIT_EXPIRED)가 있어도 **무장한다**. 회계만 밖으로
        # 빼면 상태 배정이 여전히 `actor is None` 에 갇혀 REACTIVE 가 서지
        # 않는다 — ot_ticks 를 넘겼다는 건 "그 후보로는 못 빠져나갔다" 는 뜻이므로
        # 대상을 **가장 가까운 차단물**로 바꾼다. _blocker 는 blocker_dist_max
        # (20 m) 안만 보므로 span 이 짧아져 게이트를 통과할 여지가 생긴다.
        # (시프트에 성공하면 아래에서 ot_blocked_ticks 를 0 으로 리셋한다.)
        if self.ot_blocked_ticks >= self.ot_ticks:
            stuck = self._blocker(ap, planner)
            if stuck is not None:
                actor, preempt = stuck, False
                self.last_avoid = {'state': 'REACTIVE', 'blocker': actor.id}
        if actor is None:
            if corridor and self.last_avoid is None:
                self.last_avoid = {'state': 'WATCH', 'blocker': corridor[0][3].id,
                                   's_rel': round(corridor[0][0], 1)}
            return False
        if lg is None or ego_lane is None:
            self.last_overtake = 'no_lane'
            (self.last_avoid or {}).update({'reject': 'no_lane'})
            return True
        if lg.lanes[ego_lane]['junction'] != -1:
            self.last_overtake = 'junction'
            (self.last_avoid or {}).update({'reject': 'junction'})
            return True

        local_s = self._ego_local_s(lg, ap)
        # 연쇄 장애물 병합 (B-9) — 회랑에서 actor 뒤로 chain_gap_m 안에 이어지는
        # 정지 객체를 한 span 으로 묶는다. 실측 2026-09-03 정적회피집중_01 t=74.6:
        # id3(47.2)·id4(65.3) 18 m 간격인데 id3 만 보고 span 을 만들어 복귀 전이가
        # id4 위에 떨어졌고, 밀린 경로 기준으로 id4 는 선행차 판정에서 빠져(on_route
        # False) OBB 후보가 5.7 m 에서야 서 범퍼 −2.2 m 접촉.
        chain = self._chain(corridor, actor)
        # ── side 루프 두 바퀴 (B-3) ────────────────────────────────────────
        # 1바퀴: 점선(solid) 게이트를 BREAKOUT 단계와 무관하게 **강제**한다 —
        #        점선 회랑이 있으면 여기서 끝난다 (실선을 넘을 이유가 없다).
        # 2바퀴: 1바퀴에서 한쪽이라도 solid 로 기각됐고 양쪽 다 실패했을 때만,
        #        solid 게이트만 건너뛰고 다시 돈다. center_line·geom·zone·
        #        occupied·kappa·lc_overlap 은 2바퀴에서도 그대로다.
        #        1바퀴 기각이 전부 solid 이외(geom/zone/…)면 2바퀴는 결과가
        #        같으므로 돌지 않는다 — cKDTree(≈4.6 ms) 재호출을 아낀다.
        #        solid_second_pass_enable=false 면 1바퀴만 = 이전 동작.
        self.ot_pass_solid = False
        if self._side_pass(ap, planner, ego_speed, chain, preempt,
                           lg, ego_lane, local_s, 1):
            return False
        if self.solid_second_pass and self.ot_pass_solid:
            if self._side_pass(ap, planner, ego_speed, chain, preempt,
                               lg, ego_lane, local_s, 2):
                return False
        if self.last_overtake is None:
            self.last_overtake = 'no_neighbor'
        return True

    def _extend_span(self, ap, planner, corridor, ego_speed: float) -> None:
        """활성 span 의 **끝만** 뒤로 연장 (2026-09-03 연쇄장애물_01). 4번 분기 전용.

        네 조건을 모두 만족할 때만:
          ① ot_span 활성 (호출처가 보장)
          ② standoff 차단물이 span 끝 너머 — route_index + wait_target_d·ppm > ot_span[1]
          ③ 막힌 채 정지 ot_blocked_ticks ≥ span_extend_stuck_s (BREAKOUT L1 과 같은 축)
          ④ 연장 후 예상 최소 이격 ≥ percep.obstacle_clearance_m — _span_clear_model 로
             새 span 전 구간의 모든 객체를 잰다 (새로 묶인 객체만이 아니다). 02_우회전:
             엇갈린 id6/id7 에 연장하면 뒤쪽 id5 가 0.86→0.22 로 끌려든다 → 여기서 걸러 연장 안 함.
        연장은 원복·재생성이 아니다. shift_route_smoothly(route_index, 새 끝) 를 같은 방향으로
        다시 부르면 블렌드가 현재 경로점 기준(f·loc + (1−f)·cur)이고 f(시작)=0 이라 자차
        발밑은 그대로, 앞쪽만 목표 차로로 다시 오른다(실측 발밑 변화 0.000000 m, 이음매
        곡률 0.0605→0.0985 1/m, 스파이크 없음). 억제 규칙(활성 중 새 시프트 금지)은 그대로다.
        새 끝은 plan_shift_span(무수정)으로 얻고, 경로 끝·정지 구역·횟수 상한을 넘으면 안 한다.
        ④ 미달·가드 미달은 기각이 아니라 '연장 포기' 로 진단만 남기고 이전 동작(조기 반환)을
        유지한다. route.py 는 수정하지 않는다 — 목 플래너는 getattr 가드로 건너뛴다.
        """
        if self.ot_span is None or self.wait_target_d is None or not corridor:
            return
        ppm = float(getattr(planner, 'points_per_meter', 10))
        i0 = int(planner.route_index)
        a, b = int(self.ot_span[0]), int(self.ot_span[1])
        if i0 + self.wait_target_d * ppm <= b:              # ② 차단물이 span 안 = 정상 standoff
            return
        if self.ot_blocked_ticks < self.ext_stuck_ticks:  # ③
            return
        diag = {'span_before': [a, b], 'n': self.span_extend_n}
        if self.span_extend_n >= self.ext_max_n:
            self.last_avoid['extend_skip'] = dict(diag, reason='max_n')
            return
        shift = getattr(planner, 'shift_route_smoothly', None)
        if shift is None or self.ot_side not in ('left', 'right'):
            self.last_avoid['extend_skip'] = dict(diag, reason='no_planner')
            return
        # 새 끝: standoff 대상(회랑에서 id 일치, 없으면 첫 객체)과 그 연쇄로 plan_shift_span
        actor = next((c[3] for c in corridor if c[3].id == self.standoff_id), corridor[0][3])
        chain = self._chain(corridor, actor)
        last = None if chain['last'] is actor else chain['last']
        trans_m = max(self.ot_trans_m, self.shift_k_s * max(ego_speed, 0.1))
        side = self.ot_side
        try:
            _a2, b2, left = planner.plan_shift_span(
                actor, last, obstacle_direction='right' if side == 'left' else 'left',
                transition_length=trans_m * ppm,
                extra_length_before=self.ot_before_m * ppm,
                extra_length_after=self.ot_after_m * ppm,
                min_start_ahead=self.shift_ahead_m * ppm)
        except Exception:                                  # noqa: BLE001
            self.last_avoid['extend_skip'] = dict(diag, reason='no_plan')
            return
        b2 = int(b2)
        diag.update({'chain': list(chain['ids']), 'span_end_try': b2})
        n_pts = len(planner.route_points)
        if b2 <= b:
            self.last_avoid['extend_skip'] = dict(diag, reason='not_longer')
            return
        if b2 >= n_pts - 1:                                # 경로 끝 가드
            self.last_avoid['extend_skip'] = dict(diag, reason='route_end')
            return
        # 정지 구역 가드 — 생성 시(E-2)와 같은 규칙. 새 끝이 다음 정지선을 넘으면
        # _zone_extension 이 판정한다: 교차로 없는 정지선(횡단보도)은 그대로 넘고, 교차로면
        # 출구+여유까지 끝을 미루며, 불가 사유(zone_turn 등)면 연장하지 않는다. 실측 01:
        # 615.8 은 신호·교차로 없는 추정 횡단보도라 새 끝 640.9 가 그대로 허용된다.
        # E-2 스위치가 꺼져 있으면 생성 시와 같이 정지선을 넘지 않는다 (span_into_zone).
        zone_lo = self._next_stopzone_s(planner)
        end_s = float(planner.route_s[b2])
        if zone_lo is not None and end_s > zone_lo:
            lg = getattr(planner, 'lg', None)
            if self.zone_extend and lg is not None:
                route_s_now = float(planner.route_s[i0])
                new_end, why, zinfo = self._zone_extension(planner, lg, side, route_s_now, end_s)
                if why == 'zone_no_route':
                    why = 'span_into_zone'
            else:
                new_end, why, zinfo = end_s, 'span_into_zone', {}
            if why is not None:
                self.last_avoid['extend_skip'] = dict(diag, reason=why, zone_lo=round(zone_lo, 1), **zinfo)
                return
            if new_end > end_s + 1e-6:                     # 교차로 출구 뒤로 끝을 미룸
                b2 = int(np.searchsorted(planner.route_s, new_end))
                diag['span_end_try'] = b2
                if b2 >= n_pts - 1:
                    self.last_avoid['extend_skip'] = dict(diag, reason='route_end')
                    return
            diag['zone'] = {'zone_lo': round(zone_lo, 1), **zinfo}
        # ④ 사전 계측 — 자차(f=0)에서 시작해 b2 에서 끝나는 프로파일, 띠 안 모든 객체
        model = self._span_clear_model(planner, ap, b2, left, trans_m)
        if model is None:
            self.last_avoid['extend_skip'] = dict(diag, reason='no_model')
            return
        c, per = model['clear_at'](0.0, model['b_rel'])
        clr = float(self.cfg['percep'].get('obstacle_clearance_m', 0.3))
        worst = min(per, key=per.get) if per else None
        diag.update({'clear': round(c, 2) if abs(c) < 1e8 else None, 'worst_id': worst,
                     'per': {k: round(v, 2) for k, v in per.items()}})
        if c < clr:
            self.last_avoid['extend_skip'] = dict(diag, reason='clearance')
            return
        shift(i0, b2, left, transition_length=trans_m * ppm)
        planner._kd = _cKDTree(planner.route_points[:, :2])
        self.ot_span = (a, b2)
        self.span_extend_n += 1
        self.ot_ids = sorted(set(self.ot_ids) | set(chain['ids']))
        self.last_avoid['span'] = [a, b2]
        self.last_avoid['extend'] = dict(diag, span_after=[a, b2], n=self.span_extend_n)
        print(f'[kr_rules] 회피 span 연장 — {side} 끝 {b}→{b2} (id={chain["ids"]}, '
              f'예상 최소 이격 {c:.2f} m, {self.span_extend_n}회)', flush=True)

    def _lm_deadline_s(self, planner, ap, ego_speed: float) -> float | None:
        """**언제까지** 유효 차로로 돌아와 있어야 하나 — 절대 route_s. 없으면 None.

        재료는 `_ns_turn_lane_pending` 과 **같은 것 하나**다: route.pkl 의
        `valid_entry_lanes`. 그 필드의 목적이 문서상 바로 이것이다.

        · 지금 세그먼트가 `target='pair'` 이고 유효 차로 집합이 비어 있지 않을 때만
          데드라인이 있다 (`'finish'` 는 "제약 없음").
        · 데드라인 = 세그먼트 끝 − 복귀 전이거리 − `never_stall_turn_margin_m`.
          전이거리 식은 시프트와 같다 (상수를 복제하지 않는다).
        · 제약이 없으면 None → 호출자가 "세그먼트 끝까지" 로 해석한다.
        """
        route = getattr(planner, 'route', None) or {}
        vel = route.get('valid_entry_lanes')
        wps = route.get('waypoint_s')
        if not vel or not wps:
            return None
        route_s = float(planner.route_s[planner.route_index])
        for e in vel:
            if e.get('target') != 'pair' or not e.get('lanes'):
                continue
            seg = int(e['seg'])
            if seg + 1 >= len(wps):
                continue
            s0, s1 = float(wps[seg]), float(wps[seg + 1])
            if not (s0 <= route_s < s1):
                continue
            need = (max(self.ot_trans_m, self.shift_k_s * max(ego_speed, 0.1))
                    + self.ns_turn_margin_m)
            return max(route_s, float(s1) - need)
        return None

    def _lm_no_return_m(self, planner, ap, ego_speed: float,
                        chain: dict) -> float | None:
        """복귀 전이를 **데드라인까지 미루는** extra_after [m]. 안 미루면 None.

        커밋 C "복귀 없음" 을 **새 기계 없이** 실현한다. 지금은 시프트 span 이
        마지막 장애물 + `extra_after_m`(10) 에서 끝나 곧바로 원래 차로로 돌아온다.
        원래 차로에 돌아갈 이유가 없다면(다음 회전이 그 차로를 요구하지 않는다면)
        그 복귀는 **불필요한 차로변경 두 번**이고, 그 사이 또 막히면 처음부터다.

        그래서 span 끝을 **데드라인**(유효 차로로 돌아와 있어야 하는 지점)까지
        민다. 데드라인이 없으면(제약 없는 세그먼트) 그대로 둔다 — 경로 끝까지
        미는 것은 span 이 영영 안 풀리는 [1] 의 버그를 다시 만드는 짓이다.

        `_ns_turn_lane_pending`(never_stall (c))과 **같은 재료·같은 식**을 쓴다.
        """
        if not self.lm_no_return:
            return None
        lp = self.last_lane_plan or {}
        if not lp.get('pick'):
            return None
        dl = self._lm_deadline_s(planner, ap, ego_speed)
        if dl is None:
            return None
        route_s = float(planner.route_s[planner.route_index])
        last = chain.get('last') or chain.get('first')
        pr = self._project(planner, last.get_location().x, last.get_location().y)
        if pr is None:
            return None
        end_rel = float(pr[0])                             # 마지막 장애물까지
        want = (dl - route_s) - end_rel                    # 그 뒤로 더 끌 거리
        return want if want > self.ot_after_m else None

    def _lm_hops(self, side: str) -> int | None:
        """이번 틱 차로 지도가 이 side 로 **몇 칸** 가라고 하는가. 아니면 None.

        한 칸씩 두 번 가는 방식은 못 쓴다 — 옆 차로도 막혀 있으면 **첫 전이가
        끝나기 전에 standoff 가 세워** 두 번째 칸으로 갈 기회가 영영 안 온다
        (2026-09-09 avoid_sim 케이스 2·4: 23~25 초 정지). 그래서 처음부터
        목표 칸 수로 연다.
        """
        if not self.lane_map_on:
            return None
        lp = self.last_lane_plan or {}
        if not lp.get('pick') or lp.get('side') != side:
            return None
        n = int(lp.get('hops') or 1)
        return n if n > 1 else None

    def _span_targets_lost(self, ap, planner) -> bool:
        """이 시프트를 만든 객체가 **전부 사라졌나** ([1] span 상실 원복).

        사라짐 = 월드에 없거나, 종료 게이트로 빠졌거나, 더 이상 정지 객체가
        아니다(스스로 움직여 갔다). 하나라도 남아 있으면 거짓 — 시프트 중에는
        밀린 경로 기준 회랑이 비어 보이므로 "회랑이 비었다" 로는 판정할 수 없다.

        id 를 모르는 옛 span(연장·리플레이 중 상태 유실)은 거짓을 돌려
        이전 동작(`route_index > span[1]` 원복)에 맡긴다.
        """
        if not self.ot_ids:
            return False
        try:
            actors = {a.id: a for a in ap._world.get_actors()}
        except Exception:                                  # noqa: BLE001
            return False
        for i in self.ot_ids:
            a = actors.get(i)
            if a is None:
                continue                                   # 월드에서 사라졌다
            if float(getattr(a, 'speed', 0.0)) >= self.ot_v_max:
                continue                                   # 움직여 갔다
            if self._fg_drop_static(planner, a):
                continue                                   # 종료 게이트로 빠졌다
            return False
        return True

    def _restore_span(self, planner) -> None:
        """지나간 시프트 span 원복 (다음 장애물용) — E-6 으로 호출처가 둘이 됐다."""
        a, b = self.ot_span
        planner.route_points[a:b] = planner.original_route_points[a:b]
        planner.commands[a:b] = planner.commands_orig[a:b]
        planner.lat_shift[a:b] = planner._lat_build[a:b]
        planner._kd = _cKDTree(planner.route_points[:, :2])
        self.ot_span = None
        self.ot_side = None
        self.ot_ids = []
        self.lm_hop_n = 0
        self.span_extend_n = 0
        self.last_overtake = 'restored'

    def _standoff_target(self, ap, planner, corridor) -> None:
        """standoff(관찰 감속) 대상 선정 (B-5) — 매 틱 apply 머리에서 None 으로
        리셋된 뒤 여기서만 채운다 (B-8).

        회랑 조건은 corridor 와 같되 '정지' 는 신호 무관 카운터(_stop_ok)로 본다.
        가장 가까운 대상의 s_rel 이 standoff 상한의 기준이다. PREEMPT/WAIT 판정은
        corridor(_static_ok) 그대로. standoff_stop_s 0 이면 이전 동작(corridor[0]).
        """
        if self.suppress_mode != 'legacy':
            # 캐시된 standoff 축 회랑에서, **정지선 너머** 객체는 뺀다 (C-4) —
            # 적색 정지선 건너편에 선 차는 이쪽 정지 위치와 무관하다.
            d_sl = self._stopline_d(planner)
            objs = [c for c in self._tick_corridor if d_sl is None or c[0] < d_sl]
            if objs:
                self.wait_target_d = objs[0][0]
                self.standoff_id = objs[0][3].id
                self.standoff_half_len = self._half_len(objs[0][3])
            return
        if self.standoff_stop_ticks > 0:
            so_objs = self._corridor_blockers(ap, planner, static_ok=self._stop_ok)
            if so_objs:
                self.wait_target_d = so_objs[0][0]
                self.standoff_id = so_objs[0][3].id
                self.standoff_half_len = self._half_len(so_objs[0][3])
        elif corridor:
            self.wait_target_d = corridor[0][0]
            self.standoff_half_len = self._half_len(corridor[0][3])

    @staticmethod
    def _half_len(actor) -> float | None:
        """액터 반길이 [m]. 못 읽으면 None — 그때는 크립을 아예 걸지 않는다
        (크기를 모르면 안전한 정지 거리를 정할 수 없다)."""
        bb = getattr(actor, 'bounding_box', None)
        ex = getattr(bb, 'extent', None) if bb is not None else None
        x = getattr(ex, 'x', None) if ex is not None else None
        return float(x) if x is not None else None

    def _blocked_account(self, ap, planner, ego_speed: float) -> None:
        """'막힌 채 정지' 회계 (B-10) — 자차 상태만으로 센다. span 활성 중에도 돈다."""
        blocked = ego_speed < self.latch_v and self._blocker(ap, planner) is not None
        self.ot_blocked_ticks = self.ot_blocked_ticks + 1 if blocked else 0

    def _chain(self, corridor, actor) -> dict:
        """actor 에서 시작하는 연쇄 장애물 (B-9). corridor 는 s_rel 오름차순.

        반환 {'first', 'last', 'ids', 'extent_m'} — extent_m 은 첫 객체와 마지막
        객체의 s_rel 차 (단일이면 0). span 은 first 앞 ~ last 뒤 하나로 만든다
        (PDM 원문 plan_shift_span 의 first/last_actor). geom need 는 first 기준,
        zone·solid 는 extent_m 만큼 늘어난 span 기준. chain_gap_m 0 = 비활성.
        """
        out = {'first': actor, 'last': actor, 'ids': [actor.id], 'extent_m': 0.0}
        if self.chain_gap_m <= 0.0 or not corridor:
            return out
        idx = next((i for i, c in enumerate(corridor) if c[3].id == actor.id), None)
        if idx is None:                                    # REACTIVE 의 _blocker 가 회랑 밖
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

    # ── E-2: span_into_zone 연장 ────────────────────────────────────────
    def _route_zones(self, planner, route_s: float) -> list:
        """route_s 이후 정지선 route_s 목록 (신호 정지선 + 무신호 정지선, 오름차순).
        _next_stopzone_s 와 같은 출처 — 첫 원소 − zone_gate_margin 이 zone_lo 다."""
        out = set()
        try:
            d = float(planner.distances_to_next_traffic_lights[planner.route_index])
            if d < float('inf'):
                out.add(round(route_s + d, 3))
        except Exception:                                   # noqa: BLE001
            pass
        for s in self._all_stopline_s(planner):
            if s >= route_s:
                out.add(round(float(s), 3))
        return sorted(out)

    def _zone_extension(self, planner, lg, side, route_s: float, span_end: float):
        """E-2. span 끝이 정지선을 넘을 때 → (새 span 끝, 기각 사유 | None, 진단).

        정지선마다 본다:
          · 정지선 뒤 zone_junction_gap_m 안에 교차로 진입이 없으면(횡단보도 정지선)
            그대로 넘는다 — 옆 차로도 같은 도로로 이어진다.
          · 교차로가 있으면 경로 차로가 그 교차로를 빠져나오는 출구 + zone_exit_margin_m
            까지 span 을 늘린다. 새 끝이 다음 정지선을 또 넘으면 반복(최대 4).
        연장이 서는 조건 (하나라도 깨지면 기각):
          · zone_extend_max_m 이내                              → 'zone_extend_max'
          · 교차로 출구 뒤 경로 차로가 있다                        → 'zone_no_exit'
          · [route_s, 새 끝] 에 회전(turn_*) 이벤트 없음             → 'zone_turn'
          · 차선변경 창(window_s0~s1)과 겹치지 않음                 → 'zone_lane_change'
          · 구간의 경로 차로마다 side 이웃이 있고, 이웃끼리 successor 로 이어진다
            (옆 차로가 교차로를 직진 관통)                          → 'zone_no_through_lane'
        경로 차로 정보가 없으면(목 플래너) 'zone_no_route' — 호출자가 옛 라벨로 기각.
        """
        route = getattr(planner, 'route', None) or {}
        lanes = [tuple(k) for k in (route.get('lanes') or [])]
        cum = [float(c) for c in (route.get('cum_s') or [])]
        lens = [float(x) for x in (route.get('lengths') or [])]
        if (lg is None or not lanes or len(cum) != len(lanes)
                or len(lens) != len(lanes)):
            return span_end, 'zone_no_route', {}
        new_end = float(span_end)
        zones = self._route_zones(planner, route_s)
        info: dict = {'zones': []}
        for _ in range(4):
            ahead = [z for z in zones if route_s < z < new_end]
            if not ahead:
                break
            z = ahead[0]
            j = next((i for i in range(len(lanes))
                      if lg.lanes.get(lanes[i], {}).get('junction', -1) != -1
                      and cum[i] <= z + self.zone_junction_gap_m
                      and cum[i] + lens[i] > z), None)
            if j is None:                                   # 정지선만 있다 (횡단보도)
                info['zones'].append({'s': round(z, 1), 'junction': None})
                zones = [q for q in zones if q > z]
                continue
            jid = lg.lanes[lanes[j]]['junction']
            k = j
            while (k + 1 < len(lanes)
                   and lg.lanes.get(lanes[k + 1], {}).get('junction', -1) == jid):
                k += 1
            if k + 1 >= len(lanes):
                return span_end, 'zone_no_exit', info
            j_out = cum[k] + lens[k]
            new_end = max(new_end, j_out + self.zone_exit_margin_m)
            info['zones'].append({'s': round(z, 1), 'junction': int(jid),
                                  'out': round(j_out, 1)})
            zones = [q for q in zones if q > j_out]
        info['new_end'] = round(new_end, 1)
        if new_end - span_end > self.zone_extend_max_m:
            return span_end, 'zone_extend_max', info
        for ev in route.get('events') or []:
            kind = str(ev.get('kind', ''))
            if kind.startswith('turn_'):
                if route_s <= float(ev['s']) <= new_end:
                    return span_end, 'zone_turn', info
            elif kind.startswith('lane_change'):
                a = float(ev.get('window_s0', ev.get('s', 0.0)))
                b = float(ev.get('window_s1', ev.get('s', 0.0)))
                if a <= new_end and b >= route_s:
                    return span_end, 'zone_lane_change', info
        prev_nb, prev_cum = None, None
        for i, key in enumerate(lanes):
            if cum[i] + lens[i] < route_s or cum[i] > new_end:
                continue
            nb = lg.neighbor(key, side) if key in lg.lanes else None
            if nb is None:
                return span_end, 'zone_no_through_lane', info
            # 차선변경 짝(같은 cum)은 successor 가 아니라 이웃 — 그 쌍은 건너뛴다
            if (prev_nb is not None and nb != prev_nb and prev_cum is not None
                    and cum[i] > prev_cum + 1e-6):
                try:
                    if nb not in lg.successors(prev_nb):
                        return span_end, 'zone_no_through_lane', info
                except Exception:                           # noqa: BLE001
                    pass
            prev_nb, prev_cum = nb, cum[i]
        return new_end, None, info

    def _dashed_ahead_route_m(self, planner, lg, side, route_s: float, span_m: float,
                              ego_lane) -> float:
        """E-2 전용 점선 커버리지 [m] — **경로 차로를 따라** 잰다.

        _dashed_ahead_m 의 successor 순회는 교차로 연결로(이웃 없음·마크 none)에서
        끊긴다. 연장 span 은 교차로를 지나므로 route['lanes'] 를 cum_s 로 잘라
        본다. 교차로 lane 은 차선을 넘지 않으므로 전부 인정하고, 그 밖은 side
        점선 조각과의 겹침만 센다. 차선변경 짝(같은 cum)은 자차 차로가 그 안에
        있으면 그것, 아니면 뒤쪽(to_lane)을 센다.
        """
        route = getattr(planner, 'route', None) or {}
        lanes = [tuple(k) for k in (route.get('lanes') or [])]
        cum = [float(c) for c in (route.get('cum_s') or [])]
        lens = [float(x) for x in (route.get('lengths') or [])]
        if lg is None or not lanes or len(cum) != len(lanes) or len(lens) != len(lanes):
            return 0.0
        lo, hi = float(route_s), float(route_s) + float(span_m)
        cover = 0.0
        i = 0
        while i < len(lanes):
            grp = [i]
            while i + 1 < len(lanes) and abs(cum[i + 1] - cum[grp[0]]) < 1e-6:
                i += 1
                grp.append(i)
            i += 1
            pick = next((g for g in grp if lanes[g] == ego_lane), grp[-1])
            key, a, b = lanes[pick], cum[pick], cum[pick] + lens[pick]
            s0, s1 = max(lo, a) - a, min(hi, b) - a
            if s1 <= s0:
                continue
            rec = lg.lanes.get(key)
            if rec is None:
                continue
            if rec.get('junction', -1) != -1:
                cover += s1 - s0
                continue
            try:
                runs = self._crossable_runs(lg, key, side)
            except Exception:                               # noqa: BLE001
                runs = []
            for r0, r1 in runs:
                cover += max(0.0, min(s1, r1) - max(s0, r0))
        return cover

    def _side_pass(self, ap, planner, ego_speed: float, chain: dict, preempt: bool,
                   lg, ego_lane, local_s: float, n_pass: int) -> bool:
        """side 루프 한 바퀴 (좌측 우선). 시프트를 적용했으면 True.

        n_pass ≥ 2 면 solid 게이트를 건너뛴다 (B-3). 기각 라벨은
        `{side}:{gate}@p{n}` 이고 last_avoid 에 'pass' 를 남긴다.
        chain 은 _chain() 결과 — first 가 게이트 거리 기준, first~last 가 span (B-9).
        """
        actor, last = chain['first'], chain['last']
        chain_last = None if last is actor else last
        # BREAKOUT 사다리 — 단계는 solid·occupied 가 아니라 zone·geom 완화에 쓴다
        # (B-2). solid 는 두 바퀴 구조(B-3)가, occupied 는 lvl<1 이 그대로 본다.
        lvl = self.bo_level if self.bo_state == 'BREAKOUT' else 0
        skip_solid = n_pass >= 2
        # side 선택 — 게이트를 통과한 side 의 **적용 인자만** 모으고, 적용은 루프
        # 밖에서 한 번만 한다. 게이트 본문은 그대로다. 스위치가 꺼져 있으면 첫 성공
        # side 에서 break 하므로 현행(좌측 우선)과 글자 그대로 같다.
        plans: dict = {}
        pick_on = self.side_pick and self.shift_entry      # 기준(플래토)이 있어야 한다
        # 차로 지도(커밋 B)가 고른 쪽을 **먼저** 본다. 게이트는 그대로다 —
        # 순서만 바꾸므로, 지도가 고른 쪽이 게이트에서 떨어지면 반대쪽으로 간다
        # (이전 동작으로 자연히 되돌아온다). 스위치가 꺼져 있으면 좌측 우선 그대로.
        _lp = self.last_lane_plan or {}
        _order = (('right', 'left') if _lp.get('side') == 'right'
                  else ('left', 'right'))
        for side in _order:                                # 기본은 좌측 추월 우선
            def reject(gate, **extra):
                self.last_overtake = f'{side}:{gate}@p{n_pass}'
                la = self.last_avoid if self.last_avoid is not None else {}
                la.update({'reject': f'{side}:{gate}', 'pass': n_pass, **extra})
                # 양쪽·두 바퀴의 기각을 전부 남긴다 — 최종 라벨은 마지막 side 것뿐이라
                # (C-8 로 no_neighbor 도 라벨이 붙어) 앞선 사유가 가려진다.
                la.setdefault('rejects', []).append(f'{side}:{gate}@p{n_pass}')

            # 차로 지도가 두 칸을 고르면 게이트도 **최종 목표 차로**를 봐야 한다.
            # 옛 코드는 언제나 바로 옆(1칸)만 봤다 — 그래서 두 칸 회피에서
            # 중간 차로가 막혀 있으면 `occupied` 로 기각되고, 지도가 찾은
            # 빈 차로로 영영 못 간다 (실측 07 rs 2814: lane_plan 이 (2076,3,4)를
            # 골랐는데 rejects=['right:occupied@p1'] 로 매 틱 기각 → 58 s 갇힘).
            n_hops = self._lm_hops(side) or 1
            target = self._nth_neighbor(lg, ego_lane, side, n_hops)
            if target is None:
                reject('no_neighbor')                      # C-8: 무라벨이던 기각
                continue

            # 중앙선(황색)은 **어느 BREAKOUT 단계·어느 바퀴에서도** 넘지 않는다.
            if self._is_center_mark(lg, ego_lane, side, local_s):
                reject('center_line')
                continue
            # ④ 점선 게이트 — 고정 하한(min_corridor_m)이 아니라 **실제로 밟을
            # span 전체**가 점선인지 본다. dashed_corridor_m 은 '점선 조각의
            # 길이' 라 이미 지나온 조각도 통과시킨다 (실측: 앞 점선 5 m 인데
            # 76.4 반환 → span 84.1 m 가 전 구간 실선 위에 얹혔다).
            trans_m = max(self.ot_trans_m, self.shift_k_s * max(ego_speed, 0.1))
            span_m = (2.0 * trans_m + self.ot_before_m + self.ot_after_m
                      + chain['extent_m'])                 # 연쇄면 첫~끝 객체 길이만큼
            # 전이 시작 여유 — L3 이상은 shift_ahead_l3_m (B-2). 정지 상태에서
            # need = 12 + 1 + 2 = 15 m 가 된다. 게이트와 실제 시프트가 같은 값을 쓴다.
            ahead_m = (self.shift_ahead_l3_m if lvl >= self.geom_relax_lvl
                       else self.shift_ahead_m)
            # ⑥ 기하 완성 게이트 (B-12) — **전이가 장애물 도달 전에 끝나는가**.
            # 전이는 자차 앞 ahead_m 에서 시작해 trans_m 만큼 간다. 그 끝점이
            # 장애물보다 뒤면 장애물 지점의 횡이동이 거의 0 이다 (실측 s_rel
            # 5.3 m 에서 3.0 m 중 0.0046 m = 0.15 %).
            # 여기 두는 이유: 뺄셈 하나뿐이라 7게이트 중 가장 싸다. 뒤쪽
            # solid(successor 순회)·kappa/lc_overlap(cKDTree ≈4.6 ms)보다 먼저
            # 걸러야 값싼 게이트 우선 원칙에 맞는다.
            # 거리는 **선택된 actor 기준**으로 잰다 — REACTIVE 경로에서 actor 가
            # _blocker 로 바뀌므로 corridor[0] 의 거리와 다를 수 있다.
            if self.geom_gate:
                pr = self._project(planner, actor.get_location().x,
                                   actor.get_location().y)
                need = trans_m + ahead_m + self.geom_margin_m
                # pr is None = 전방 창에서 투영 실패. 판단 근거가 없으므로
                # 기각하지 않는다 (다른 게이트의 '못 재면 통과' 관례와 같다).
                if pr is not None and need > pr[0]:
                    reject('geom', need_geom=round(need, 1),
                           s_rel_actor=round(pr[0], 1), margin=round(pr[0] - need, 1))
                    continue
            # ② 시프트 기하 게이트 — 나가는 전이 + 복귀 전이 span 전체가
            # 정지선 경계(zone_gate_margin_m) **앞에서 끝나야** 시작한다.
            # 시프트 도중에 억제 구역으로 들어가면 되돌릴 수 없다(급조향 금지).
            # BREAKOUT lvl ≥ zone_gate_relax_level 이면 건너뛴다 (B-2).
            extra_after = self.ot_after_m
            zone_ext = None                                # E-2 연장량 [m] (None = 미적용)
            route_s = float(planner.route_s[planner.route_index])
            # E-8 ②: 완화 한정 모드에서는 L2 이상에서도 게이트를 평가한다 — 해제되는
            # 사유(zone_no_exit / zone_extend_max / 평가 불가)만 통과시키고 회전·차선변경·
            # 통과 차로 없음은 유지한다. 한정 모드가 아니면 L2 부터 게이트 전체 생략(이전).
            if lvl < self.zone_relax_lvl or self.zone_relax_limited:
                span_end = route_s + span_m
                zone_lo = self._next_stopzone_s(planner)
                if zone_lo is not None and span_end > zone_lo:
                    # E-2: 기각 대신 연장을 먼저 본다. 실제 span 끝은 전이 시작이
                    # 자차 앞 ahead_m 이상이라 route_s + span_m 보다 뒤다 — 연장량은
                    # 그 추정치(span_end_est)에서 잰다. 평가 불가(목 플래너 등 경로
                    # 차로 없음)면 옛 라벨 그대로 기각한다.
                    why, new_end, zinfo = 'span_into_zone', span_end, {}
                    if self.zone_extend:
                        pr0 = self._project(planner, actor.get_location().x,
                                            actor.get_location().y)
                        ext_x = float(getattr(getattr(actor, 'bounding_box', None),
                                              'extent', None).x) \
                            if getattr(actor, 'bounding_box', None) is not None else 0.0
                        start_est = (max(pr0[0] - ext_x - trans_m - self.ot_before_m, ahead_m)
                                     if pr0 is not None else 0.0)
                        span_end_est = route_s + start_est + span_m
                        new_end, why, zinfo = self._zone_extension(
                            planner, lg, side, route_s, span_end_est)
                        if why == 'zone_no_route':
                            why = 'span_into_zone'
                        elif why is None:
                            zone_ext = max(0.0, new_end - span_end_est)
                    if why is not None:
                        relaxed = (lvl >= self.zone_relax_lvl and self.zone_relax_limited
                                   and why in self.ZONE_RELAXABLE)
                        if not relaxed:
                            reject(why, span_end=round(span_end, 1),
                                   zone_lo=round(zone_lo, 1), **zinfo)
                            continue
                        (self.last_avoid or {}).update({'zone_relaxed': why, **zinfo})
                        zone_ext = None                    # 연장 없이 원 span 으로 진행
                    if zone_ext is not None:
                        extra_after += zone_ext
                        span_m += zone_ext
                        (self.last_avoid or {}).update(
                            {'zone_extended': True, 'extended_by': round(zone_ext, 1),
                             'span_m': round(span_m, 1), **zinfo})
            if not skip_solid:
                if zone_ext is not None:
                    # 연장 span 은 교차로 연결로를 지나므로 successor 순회가 끊긴다 —
                    # 경로 차로를 따라 잰다 (교차로 안은 차선을 넘지 않아 전부 인정).
                    cover = self._dashed_ahead_route_m(planner, lg, side, route_s,
                                                       span_m, ego_lane)
                else:
                    cover = self._dashed_ahead_m(lg, ego_lane, side, local_s, span_m)
                if cover < span_m - self.ot_dash_slack_m:
                    self.ot_pass_solid = True              # 2바퀴 사유
                    reject('solid', dash_m=round(cover, 1), span_m=round(span_m, 1))
                    continue
            # occupied 완화(L1+)는 **정지 stuck_hard_s 경과** 를 요구한다 (E-3 안전
            # 가드) — 주행 중 시계로 오른 단계로 점유 차로에 밀지 않는다. 시계가
            # 꺼져 있으면 L1 자체가 정지 stuck_hard_s 뒤라 조건이 항상 참 = 이전 동작.
            occ_relaxed = lvl >= 1 and (not self.bo_reject_clock
                                        or self.bo_stop_ticks >= self.bo_hard_ticks)
            if not occ_relaxed and not self._side_is_clear(lg, planner, ap, target):
                reject('occupied')
                continue
            # 중간 차로는 **목적지가 아니라 지나가는 구간**이다. 전체 점유가 아니라
            # 램프가 실제로 훑는 s 구간에 객체가 있는지만 본다.
            if n_hops >= 2 and not self._mid_lanes_clear(
                    lg, planner, ap, ego_lane, side, n_hops):
                reject('occupied_mid')
                continue
            ppm = float(getattr(planner, 'points_per_meter', 10))
            # ⑤ 기하 계단 검사 (B-7 임시 가드) — 다른 게이트를 다 통과한 뒤에만
            # 잰다 (읽기 전용이지만 span 하나에 1.7~5.2 ms 든다). 상태를 남기지
            # 않으므로 기각은 **그 시점 그 경로 한정**이고 다음 틱에 다시 시도한다.
            geom = self._planned_shift_geom(planner, actor, side, trans_m, ahead_m,
                                            last_actor=chain_last, after_m=extra_after)
            if geom is not None:
                kap, lc_var = geom
                (self.last_avoid or {}).update(
                    {'shift_kappa': round(kap, 4), 'lc_var': round(lc_var, 3)})
                if self.shift_k_reject > 0.0 and kap > self.shift_k_reject:
                    reject('kappa')
                    continue
                # 계획 LC 와 겹치면 기각 (B-11) — 합산 횡이동을 만들지 않는다
                if self.lc_overlap_m > 0.0 and lc_var > self.lc_overlap_m:
                    reject('lc_overlap')
                    continue
            # ⑥ span 국소성 — 시프트 시작점이 자차보다 span_gate_max_m 이상 앞이면
            # 순환 코스의 한 바퀴 뒤 구간이 잡힌 것이다 (실측 4550~4595 m). 기록이
            # 없으면(plan_shift_span 실패) "못 재면 통과" 관례대로 검사하지 않는다.
            if self.span_gate and self.last_span_plan is not None:
                span_off_m = (self.last_span_plan[0] - int(planner.route_index)) / ppm
                if span_off_m >= self.span_gate_max_m:
                    reject('span_too_far', span_off_m=round(span_off_m, 1))
                    continue
            # ⑦ 전이 배치 — 나가는 전이가 중간 객체를, 복귀 전이가 목표 차로 객체를
            # 스치지 않게 시작 지연(delay)·복귀 단축(after)을 고른다. 적용은 route.py
            # 인자(min_start_ahead / extra_length_after)로만 한다. 기록이 없으면
            # (plan 실패) 검사 없이 현행 배치.
            ahead_eff = ahead_m
            pinfo = None
            if self.shift_entry and self.last_span_plan is not None:
                delay, after_eff, pinfo = self._shift_placement(
                    planner, ap, trans_m, ahead_m, extra_after, zone_ext is not None)
                (self.last_avoid or {}).update(pinfo)
                if pinfo and delay is None:
                    reject('entry_block', block_id=pinfo.get('entry_block_ids'),
                           best_clear=pinfo.get('entry_clear'))
                    continue
                if delay is not None:
                    ahead_eff = ahead_m + delay
                    extra_after = after_eff
            # 게이트 전부 통과 — 적용 인자와 이 side 의 계측을 담아 둔다.
            plans[side] = {'trans_m': trans_m, 'ahead_eff': ahead_eff,
                           'extra_after': extra_after, 'span_m': span_m,
                           'pinfo': dict(pinfo or {}),
                           # 이 side 의 span 계획 — 양측을 다 재면 last_span_plan 은
                           # **마지막(진) side** 것으로 남는다. gap_fit 이 그걸 읽으면
                           # 목표 차로가 반대쪽이라 옆 차로 객체를 못 본다 (실측
                           # 2026-09-05 202508: objs 에 id3 만 남아 no_better).
                           'span_plan': self.last_span_plan,
                           'avoid': dict(self.last_avoid or {})}
            if not pick_on or not plans[side]['pinfo'].get('entry_plateau_ids'):
                # 단락 — 플래토가 비면 _pick_side 는 어느 경우에도 이 side 를 고른다
                # (한쪽만 비면 그쪽, 둘 다 비면 좌측 유지). 반대쪽을 재지 않으므로
                # 추가 비용이 0 이다. 스위치가 꺼져 있어도 여기서 끊어 현행과 같다.
                break
        if not plans:
            return False
        side, why = self._pick_side(plans, pick_on)
        p = plans[side]
        trans_m, ahead_eff = p['trans_m'], p['ahead_eff']
        extra_after, span_m = p['extra_after'], p['span_m']
        # 선택된 side 의 계측(κ·lc_var·배치)을 되살린다 — 반대쪽을 재는 동안 덮였다.
        # 기각 누적(reject/rejects/pass)은 살아 있는 쪽 값을 그대로 둔다.
        live = self.last_avoid if isinstance(self.last_avoid, dict) else {}
        merged = dict(p['avoid'])
        for k in ('reject', 'rejects', 'pass'):
            if k in live:
                merged[k] = live[k]
        # 배치 키는 `.update()` 누적이라 앞 side 가 쓰고 뒤 side 가 안 쓰면 그대로
        # 남는다 (예: 좌측 entry_plateau_ids 가 우측 진단에 섞인다). 승자가 쓰지
        # 않은 키는 지우고 승자 값으로만 채운다.
        for k in ('entry_base_clear', 'entry_block_ids', 'entry_clear',
                  'entry_delay_m', 'entry_start_rel_m', 'exit_after_m',
                  'entry_plateau_ids'):
            merged.pop(k, None)
        merged.update(p['pinfo'])
        self.last_avoid = merged
        # ⑧ gap_fit — 옆 차로 빈 구간에 맞춰 (전이, 앞뒤 여유) 를 다시 고른다.
        # 게이트가 아니다: 기각하지 않고, 더 나은 배치를 못 찾으면 현행 그대로 간다.
        # 전이를 줄이는 방향이라 geom need(trans+ahead+margin)도 함께 줄어 이미
        # 통과한 게이트를 다시 깨지 않는다. 적용은 아래 route.py 인자뿐이다.
        gap_before = self.ot_before_m
        (trans_m, gap_before, extra_after), ginfo = self._gap_fit_search(
            planner, ap, actor, chain, trans_m, ahead_eff, extra_after,
            span_plan=p.get('span_plan'))
        if ginfo:
            self.last_avoid.update(ginfo)
        # ⑨ gap_fit 속도 연동 — 고른 전이는 v ≤ trans / shift_k_s 에서만 실제로
        # 만들어진다 (trans_m = max(transition_m, shift_k_s·v)). 지금 만들면 다음
        # 틱에 trans 가 다시 3·v 로 늘어 같은 span 이 되므로, **속도를 먼저 줄이고**
        # 생성을 보류한다. 속도는 위 apply 의 min() 후보(gap_v_req)로만 낮춘다 —
        # 여기서 제어에 직접 손대지 않는다.
        # 보류는 자기제한적이다: s_rel 이 새 전이의 geom need 밑으로 내려가면
        # 더 기다릴 이유가 없어 현행 배치로 즉시 생성한다. 무한 대기가 없다.
        if (self.gap_fit_speed and ginfo and ginfo.get('gap_T_sel') is not None):
            v_req = trans_m / max(self.shift_k_s, 1e-6)
            pr_h = self._project(planner, actor.get_location().x,
                                 actor.get_location().y)
            need_new = trans_m + ahead_eff + self.geom_margin_m
            room = (pr_h[0] > need_new) if pr_h is not None else False
            if ego_speed > v_req and room:
                self.gap_v_req = v_req
                self.gap_hold_ticks += 1
                self.last_avoid.update(
                    {'gap_v_req': round(v_req, 2), 'gap_speed_hold': True,
                     'gap_speed_hold_s': round(self.gap_hold_ticks / self.hz, 1),
                     'gap_need_new': round(need_new, 1)})
                return True                                # 기각이 아니다 — 보류
            self.last_avoid.update(
                {'gap_v_req': round(v_req, 2), 'gap_speed_hold': False,
                 'gap_speed_hold_s': round(self.gap_hold_ticks / self.hz, 1),
                 'gap_speed_release': 'v_ok' if ego_speed <= v_req else 'no_room'})
        self.gap_hold_ticks = 0
        # 차로 지도(커밋 B)가 두 칸을 고르면 **한 번에** 두 칸을 연다. 전이 길이도
        # 칸 수에 맞춰 늘린다 — 같은 길이로 두 칸을 가면 횡가속이 4배가 되고
        # _shift_speed_cap 이 곧바로 상한을 바닥까지 내린다.
        n_steps = self._lm_hops(side)
        # 커밋 C — 복귀를 데드라인까지 미룬다 (원래 차로 우선권 없음).
        nr = self._lm_no_return_m(planner, ap, ego_speed, chain)
        if nr is not None:
            extra_after = max(extra_after, nr)
        extra_kw = {}
        if n_steps is not None:
            lane_w = self._lane_width_at(planner.lg, ego_lane, local_s)
            trans_m = max(trans_m, self._ramp_len_m(n_steps, lane_w,
                                                    max(ego_speed, self.lm_avoid_v)))
            # 인자를 **조건부로만** 넘긴다 — off 에서는 호출 서명이 이전과 글자
            # 그대로여야 한다 (목 플래너를 쓰는 테스트가 그 서명을 흉내낸다).
            extra_kw['n_steps'] = n_steps
        span_m = 2.0 * trans_m + gap_before + extra_after + chain['extent_m']
        span = planner.shift_route_around_actors(
            actor, chain_last,
            obstacle_direction='right' if side == 'left' else 'left',
            transition_length=trans_m * ppm,
            extra_length_before=gap_before * ppm,
            extra_length_after=extra_after * ppm,          # E-2 연장·복귀 단축 포함
            # 전이 시작을 자차 **앞**으로 — 뒤에서 시작하면 현재 위치의
            # 경로가 옆으로 밀려 정지 상태에서 조향이 풀락된다
            min_start_ahead=ahead_eff * ppm,
            **extra_kw)
        planner._kd = _cKDTree(planner.route_points[:, :2])
        self.ot_span = span
        self.lm_hop_n = int(n_steps or 1)                  # 몇 칸짜리 시프트였나
        self.ot_ids = list(chain['ids'])                   # 대상 상실 판정의 기준
        self.ot_side = side                                # 연장이 같은 방향을 쓴다
        self.span_extend_n = 0
        self.ot_blocked_ticks = 0
        self.preempt_latch_id = None                       # E-4 래치 해제 (시프트 성공)
        self.last_overtake = side
        order = [s for s in ('left', 'right') if s in plans]
        self.last_avoid.update(
            {'shift': side, 'span': list(span), 'trans_m': round(trans_m, 1),
             'ahead_m': round(ahead_m, 1), 'preempt': preempt, 'pass': n_pass,
             'solid_relaxed': skip_solid, 'chain': list(chain['ids']),
             'span_m': round(span_m, 1), 'after_m': round(extra_after, 1)})
        if pick_on:
            # 스위치가 켜진 틱에만 남긴다 — off 에서 진단 키가 늘면 리플레이
            # 동일성 비교가 "결정은 같은데 로그는 다르다" 가 되어 쓸모가 준다.
            self.last_avoid['side_pick'] = {
                'evaluated': order, 'picked': side, 'why': why,
                'base_clear': {s: plans[s]['pinfo'].get('entry_base_clear')
                               for s in order},
                'plateau': {s: plans[s]['pinfo'].get('entry_plateau_ids')
                            for s in order},
                # 좌측 플래토가 비어 우측을 아예 재지 않은 틱 (추가 비용 0)
                'short_circuit': bool(order == ['left']
                                      and not plans['left']['pinfo']
                                      .get('entry_plateau_ids'))}
        print(f'[kr_rules] 정적 장애물 회피 — {side} 로 경로 시프트 '
              f'(id={chain["ids"]}, 구간 {span[0]}~{span[1]}, p{n_pass}, {why})',
              flush=True)
        return True

    def _pick_side(self, plans: dict, pick_on: bool):
        """게이트를 통과한 side 중 하나를 고른다 → (side, 근거).

        확정 규칙 (2026-09-04 승인):
          후보 1개          → 그것                       'single'
          한쪽만 플래토 빔   → 그쪽                       'plateau_empty'
          둘 다 플래토 빔    → 좌측 유지 (현행 동작)        'both_empty_keep_left'
          둘 다 플래토 있음  → base_clear 큰 쪽 (동률 좌측)  'base_clear'

        1차 키가 배치 점수가 아니라 `entry_plateau_ids` 인 이유: `_shift_placement`
        는 "격자 어느 (delay, after) 에서도 임계를 못 넘는 객체" 를 목적함수에서
        빼므로(플래토 제외), **막힌 쪽이 오히려 높은 점수를 낸다**. 실측
        2026-09-03 104648 t=42.16: 좌측 점수 1.31(플래토 [5,6,8], 원본 최소 이격
        −1.73) 대 우측 0.68(플래토 없음, +0.68). 플래토가 비었다는 것은 그 왜곡이
        없다는 뜻이라 1차 키로 쓸 수 있다. 2차 키도 같은 이유로 플래토 제외 **전**
        값인 base_clear 다.

        "둘 다 비면 좌측 유지" 는 회귀 방지다 — 이격만으로 비교하면 지금 성공 중인
        시프트가 뒤집힌다 (실측 2026-09-04 추월집중_01 t=10.60: 좌 +1.02 실측 OBB
        1.02 인데 우 +1.20 으로 뒤집힘). 바꿀 이유가 없는 곳은 바꾸지 않는다.

        pick_on 이 거짓(스위치 off 또는 shift_entry off)이면 plans 에는 단락으로
        첫 성공 side 하나만 들어 있으므로 결과가 현행과 같다.
        """
        order = [s for s in ('left', 'right') if s in plans]
        if len(order) == 1 or not pick_on:
            return order[0], 'single'
        empty = [s for s in order if not plans[s]['pinfo'].get('entry_plateau_ids')]
        if len(empty) == 1:
            return empty[0], 'plateau_empty'
        if len(empty) == 2:
            return 'left', 'both_empty_keep_left'

        def bc(s):
            v = plans[s]['pinfo'].get('entry_base_clear')
            return -1e9 if v is None else float(v)

        return (('left', 'base_clear') if bc('left') >= bc('right')
                else ('right', 'base_clear'))

    @staticmethod
    def _ego_lane(lg, ap):
        """자차가 선 차로. 지도에서 못 찾으면 None.

        None 이 되는 **실제** 경로는 lg.locate 뿐이다 — courseRespawn 이나 이탈로
        자차가 레인그래프 밖에 있으면 매칭이 없다. get_location 은 감싸지 않는다:
        프로덕션 VtdEgo 는 항상 가지고 있고, 없으면 그건 조립 오류라 조용히
        삼키면 안 된다.
        """
        if lg is None:
            return None
        loc = ap._vehicle.get_location()
        vx, vy = frame.from_carla_xy(loc.x, loc.y)
        try:
            m = lg.locate(vx, vy)
        except Exception:                                  # noqa: BLE001
            return None
        return m.lane if m is not None else None

    def _next_stopline(self, planner):
        """전방 신호 정지선 → (뒷축거리 [m], 상태명, 신호 id). 없으면 None.

        색을 가리지 않고 그대로 준다 — 색 해석은 호출자 한 곳
        (_stop_target)에서만 한다.
        """
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

    def _yellow_latch(self, planner, ego_speed: float, ap) -> None:
        """황색 원샷 판정 — 접근당 1회 STOP/GO 를 정하고 래치한다.

        대회 채점표 편향이 **STOP 우선**을 강제한다: 황색 정지를 감점하는 항목이
        없고(항목8 5 s 카운트는 녹색 틱만 센다), 반대로 적색 통과·걸침은 항목7
        중대다. 게다가 score.detect_red_light 는 **통과 순간의 신호**로 판정하므로
        GO 로 나갔다가 적색에 걸리면 그대로 중대가 된다 (실측 2026-08-30
        실전주행_02: 적신호 통과 2 + 정지선 침범 2 = 항목7 4건).

        그래서 판정 감속은 **확실히 실행 가능한 최대**(speed.a_yellow, 기본
        a_dec_max 와 같은 4.0)를 쓴다 — STOP 영역을 최대화하고 GO 는 물리적으로
        설 수 없는 영역에만 남긴다.

            v ≤ √(2·a_yellow·(d − s0))  → STOP (적색과 동일 취급)
            그 외                        → GO   (신호·정지선 유래 후보 미생성)

        래치는 접근당 유지한다. GO 중 적색으로 바뀌어도 번복하지 않는다 —
        번복하면 교차로 한복판 급제동이 된다. 해제는 ① 다른 신호로 넘어감
        ② 녹색 복귀 ③ 교차로 통과 가드 종료.
        """
        nxt = self._next_stopline(planner)
        if nxt is None:
            self._yellow_reset()
            return
        d_line, state, tl_id = nxt
        if self.y_ctrl is not None and tl_id != self.y_ctrl:
            self._yellow_reset()                      # 다음 교차로
        if state == 'Green':
            self._yellow_reset()                      # 녹색 복귀
            return
        if self.a_yellow <= 0.0 or self.y_decision is not None or state != 'Yellow':
            return                                    # 비활성 / 이미 래치 / 황색 아님
        if self.y_gate_m > 0.0 and d_line > self.y_gate_m:
            return                                    # 딜레마 구간 밖 — 래치하지 않는다
        # 판정도 같은 램프를 겪는다 — 실행 축과 **같은 보정**을 쓴다 ([2]).
        d_eff = (d_line - self._s0(ap)
                 - self._jerk_ramp_m(self.a_yellow, ego_speed))
        v_allow = _math.sqrt(2.0 * self.a_yellow * max(0.0, d_eff))
        self.y_decision = 'stop' if ego_speed <= v_allow else 'go'
        self.y_ctrl = tl_id
        self.y_v_allow = v_allow
        # 판정 **순간**만 기록한다 — 사후 분류(어느 접근이 STOP/GO 였나)의 근거.
        # 매 틱 싣지 않는 이유: 판정은 접근당 1회고, 그 1틱이 조건을 다 담는다.
        self.last_yellow = {'decision': self.y_decision, 'ctrl': tl_id,
                            'v': round(float(ego_speed), 2),
                            'v_allow': round(float(v_allow), 2),
                            'd_line': round(float(d_line), 2),
                            'a_judge': self.a_yellow}

    def _yellow_reset(self) -> None:
        self.y_decision = None
        self.y_ctrl = None
        self.y_v_allow = None

    def on_reset(self) -> None:
        """courseRespawn — 순간이동 전 래치는 전부 무효 (run_agent 가 부른다).

        특히 GO 래치를 살려 두면, 리스폰으로 정지선 **뒤로** 되돌아간 뒤에도
        "이미 가기로 했다"가 유지돼 적신호를 그대로 통과한다 (항목7 중대).
        """
        self._yellow_reset()
        self.cross_guard = False
        self.cross_s = None
        self.cross_junction_seen = False
        self.sl_hold_left = 0
        self.sl_stopped = False                       # 정지 연속성도 끊긴다 (B-1)
        self.sl_stop_ticks = 0
        self.latched = False
        # 보행자 의도 (P4) — 자차가 순간이동하면 경로 투영이 불연속이 되어
        # 직전 횡거리와의 차분이 가짜 '경로 쪽 횡속도' 를 만든다. 전부 버린다.
        self.ped_lat.clear()
        self.ped_intent.clear()
        self.ped_static.clear()
        self.ped_clear.clear()
        self.ped_hold.clear()
        self.cw_wait.clear()
        self.ped_walkin.clear()
        self.ped_hold_ids.clear()
        self.ped_miss.clear()
        self.ped_last.clear()
        self._creep_open_latched = False              # 크립 delay 래치 (문맥 불연속)
        self._sig_go = False                          # B-3(b) 시한 출발 래치
        self._sig_go_tl = None
        self._sig_wait_ticks = 0
        self._rtor_reset()                            # RTOR 래치·정지 누적 (순간이동 = 새 접근)
        self._creep_hold_ticks = 0
        self.j_reject_ticks = 0                       # 교차로 해제 시계 (A1) — 새 문맥
        self.ns_ticks = 0                             # never_stall 시계 (A2) — 새 문맥
        self.ns_ref_s = None
        self.ns_level = 0
        self.ns_turn_ticks = 0

    def _s0(self, ap) -> float:
        """계획 정지점의 뒷축 gap — PDM 주입값이 단일 출처."""
        return float(getattr(ap.config, 'idm_red_light_minimum_distance',
                             self.stop_gap_sl_fallback))

    def _jerk_ramp_m(self, a_eff: float, v: float) -> float:
        """감속 `a_eff` 에 **도달하기까지** jerk 램프가 더 가는 거리 [m].

        t_ramp = a_eff / jerk_rate 동안 감속이 0 → a_eff 로 선형 증가하므로
        평균 감속이 a_eff/2 다. 같은 t 를 완전감속으로 갔다면 줄었을 거리와의
        차이가 v·t/2 다 (2차항은 상쇄된다).

        √ 프로파일에서 이 거리를 **미리 빼면** 상한이 그만큼 일찍 구속하고,
        램프가 다 서는 시점에 계획 정지점에 선다. 스위치가 꺼지면 0 이다.
        """
        if not self.stop_jerk_comp or self.jerk_rate <= 0.0 or a_eff <= 0.0:
            return 0.0
        return max(0.0, float(v)) * (float(a_eff) / self.jerk_rate) / 2.0

    def _cross_guard(self, planner, ap, d_line) -> bool:
        """교차로 통과 가드 — 앞범퍼가 정지선을 넘은 뒤 교차로를 벗어날 때까지
        **신호·정지선 유래 정지 후보를 만들지 않는다** (보행자·선행차 후보는
        min() 의 다른 갈래라 그대로 산다).

        가드가 없으면 교차로 한복판에서 뒤쪽 정지선을 향해 제동하거나, 바로
        다음 정지선에 성급히 반응한다. 해제는 교차로를 벗어났을 때, 또는 진입이
        관측되지 않은 채 yellow_guard_max_m 를 지났을 때(상한 — 고착 방지).
        """
        route_s = float(planner.route_s[planner.route_index])
        if not self.cross_guard and d_line is not None and (d_line - self.front) <= 0.0:
            self.cross_guard = True                   # 앞범퍼가 정지선을 넘었다
            self.cross_s = route_s
            self.cross_junction_seen = False
        if not self.cross_guard:
            return False
        in_j = bool(getattr(ap, 'junction', False))
        if in_j:
            self.cross_junction_seen = True
        elif self.cross_junction_seen or (
                self.cross_s is not None       # 0.0 은 falsy — or 로 폴백하면 안 된다
                and route_s - self.cross_s > self.y_guard_max_m):
            self.cross_guard = False
            self.cross_s = None
            self._yellow_reset()
            return False
        return True

    def _flash_tick(self, planner, ego_speed: float) -> None:
        """적색 점멸 일시정지 시계 (B3, 항목 9). apply 가 틱당 1회 부른다.

        규정: **범퍼 기준 정지선 2 m 이내에서 0.5 s 이상 정지 1회**, 그 뒤 통과.
        무정차 통과는 중대(−6)다. 채점기 `detect_blink_stop` 과 **같은 축**을
        쓴다 — 임계는 `scoring.stop_ok_m` · `scoring.stop_hold_s` 가 정본이고
        여기서 새 상수를 만들지 않는다 (같은 주행이 항목별로 다르게 판정되면
        안 된다).

        멈추는 일 자체는 이 함수가 하지 않는다. 9910 state 6 이
        `TrafficLightState.FlashRed` 로 매핑되면 **PDM 의 적신호 IDM 이 정지선까지
        정지 프로파일을 그린다** — 상한형이 아니라 정지 축이다(CLAUDE.md 확정
        사실). 여기서는 "충분히 섰다" 를 판정해 `signal_release` 로 풀어 줄
        뿐이다. 6 은 지속 플래그라 램프 위상을 볼 필요가 없다.

        래치는 **신호 id 단위**다. 한 번 허가가 나면 그 정지선을 지날 때까지
        유지되므로 같은 선에 다시 서지 않는다. 다음 신호로 넘어가면 해제된다.
        """
        if not self.flash_stop:
            return
        nxt = self._next_stopline(planner)
        if nxt is None:
            self.flash_tl = None
            self.flash_hold_n = 0
            return
        d_line, state, tl_id = nxt
        if state != 'FlashRed':
            # 이 신호가 점멸이 아니면 시계도 래치도 이 신호에 대해선 무의미하다.
            if self.flash_tl == tl_id:
                self.flash_tl = None
                self.flash_hold_n = 0
            return
        if self.flash_tl != tl_id:                     # 새 점멸 신호
            self.flash_tl = tl_id
            self.flash_hold_n = 0
            if self.flash_latch_tl != tl_id:
                self.flash_latch_tl = None
        # 부호 규약은 로그 `world.stop_line_front_m` · 채점기와 **같다**:
        # 앞범퍼 기준, **음수 = 아직 선 앞**. (d_line 은 뒷축 기준 남은 거리다.)
        front_m = self.front - d_line
        near = front_m >= -self.flash_ok_m             # 선까지 ok_m 안
        # 정지 유지는 **선을 넘기 전에만** 센다 — 넘어가 선 것은 일시정지가
        # 아니다 (채점기가 front_m ≤ 0 만 인정하는 것과 같은 규약).
        if near and front_m <= 0.0 and ego_speed < self.flash_stop_v:
            self.flash_hold_n += 1
        elif ego_speed >= self.flash_stop_v:
            self.flash_hold_n = 0
        if self.flash_hold_n >= self.flash_hold_ticks:
            self.flash_latch_tl = tl_id                # 통과 허가
        self.last_flash = {'tl': int(tl_id) if tl_id is not None else None,
                           'front_m': round(front_m, 2),
                           'hold_s': round(self.flash_hold_n / self.hz, 2),
                           'need_s': round(self.flash_hold_ticks / self.hz, 2),
                           'released': bool(self.flash_latch_tl == tl_id)}

    def _flash_release(self, planner) -> bool:
        """점멸 통과 허가가 이번 틱 유효한가 — signal_release 의 한 갈래.

        허가는 **그 신호 id 에만** 붙는다. 교차 차량·보행자는 여기서 새로 보지
        않는다: PDM 의 IDM·OBB 와 `_ped_intent`·보행자 홀드가 min() 의 다른
        갈래로 그대로 살아 있어, 허가가 나도 교차 차량이 있으면 그쪽이 세운다.
        (`signal_release` 는 **신호 유래 감속만** 건너뛴다.)
        """
        if not self.flash_stop or self.flash_latch_tl is None:
            return False
        nxt = self._next_stopline(planner)
        if nxt is None:
            return False
        return nxt[2] == self.flash_latch_tl and nxt[1] == 'FlashRed'

    def signal_release(self, ap, _distance_to_traffic_light=None) -> bool:
        """PDM 의 적신호 IDM 을 이번 틱 건너뛸 것인가 — autopilot 조기 반환 조건.

        kr_rules 는 min() 에 후보를 **덧대기만** 하므로 PDM 이 스스로 만드는
        적신호 감속을 없앨 수 없다. 황색 GO 와 교차로 통과 가드는 "감속하지
        말 것"이 요지라, 이 규칙만 예외적으로 판단(여기)과 소비(autopilot 한 줄)가
        분리된다. 녹색일 때 IDM 을 건너뛰는 것과 같은 메커니즘이고 IDM 본문은
        무수정이다.

        참이 되는 경우는 둘뿐이다:
          · 황색 GO 래치 — 접근당 1회 판정으로 "설 수 없다" 가 확정된 상태.
            여기서 PDM 이 감속하면 어중간히 늦춰 정지선에 걸친다 (실측 2026-08-30
            실전주행_02: 황색 3 s 동안 IDM 이 가속↔감속을 오가다 slf=+1.35).
          · 교차로 통과 가드 — 앞범퍼가 이미 정지선을 넘었다. 여기서 제동하면
            걸친 채로 선다.
        보행자·선행차 후보는 min() 의 다른 갈래라 그대로 살아 있다.
        B-3(b) 미보고 적신호 시한 출발(_sig_go, 기본 off)도 같은 자리에서 푼다.
        RTOR 래치(_rtor_go, 기본 off)도 같은 자리 — 래치가 붙은 신호일 때만.
        """
        planner = getattr(ap, '_waypoint_planner', None)
        return bool(self.y_decision == 'go' or self.cross_guard or self._sig_go
                    or self._rtor_active(planner)
                    # B3: 적색 점멸 — 규정 유지시간을 채운 신호에 한해 통과.
                    or self._flash_release(planner))

    def _signal_timeout_tick(self, ap, planner, ego_speed: float) -> None:
        """B-3(b) 시계 — apply 가 틱당 1회 부른다 (_tick_cache 뒤: 회랑·stale 필요).

        성립 조건 (전부): 스위치 on ∧ 다음 신호 controller 미보고(stale) ∧ 그 신호가
        정지 후보를 만드는 상태(Red / 황색 STOP) ∧ 정지 중(v < latch_v) ∧ 앞차 없음
        (정지 회랑 객체 0 ∧ signal_timeout_clear_m 안 이동 차량 0) ∧ 보행자 래치·
        PDM 보행자 플래그 없음. 하나라도 깨지면 시계 0. signal_unknown_timeout_s 를
        채우면 _sig_go 래치 — 같은 신호 id 이고 여전히 stale 인 동안만 산다 (신호가
        다시 보고되면 그 state 가 즉시 우선한다).
        """
        if not self.sig_timeout_go:
            return
        sig = self.last_signal
        nxt = self._next_stopline(planner)
        tl_id = nxt[2] if nxt else None
        stale = bool(sig and sig.get('signal_stale'))
        if self._sig_go and (not stale or tl_id != self._sig_go_tl):
            self._sig_go = False                          # 보고 재개 / 다음 신호로 넘어감
            self._sig_go_tl = None
        # A3: 앞차 판정. 기본은 이전 동작 — 회랑에 정지 객체가 하나라도 있으면
        # 시계가 안 돈다. 스위치를 켜면 **정지** 객체는 막지 않는다: 시한이
        # 만료돼도 푸는 것은 신호 유래 정지 후보뿐이고(_stop_target → None)
        # PDM 의 선행차 IDM 은 그대로 살아 앞차 뒤에 선다. "신호를 못 봐서 서
        # 있는 것" 만 풀고 "앞차 때문에 서 있는 것" 은 안 푼다.
        lead_block = bool(self._tick_corridor) and not self.sig_lead_ok
        ok = stale and self._stop_target_raw(planner, ap) is not None \
            and ego_speed < self.latch_v and not lead_block \
            and not (self.ped_intent or self.ped_hold_ids) \
            and not (getattr(ap, 'walker_hazard', False) or getattr(ap, 'walker_close', False))
        if ok:
            # 스위치가 꺼져 있으면 static_ok=True 그대로 (정지 객체까지 센다).
            # 켜면 **실제로 움직이는** 차량만 본다 — 임계는 blocker_speed_max
            # ("회피가 정지로 보는 속도")를 그대로 읽는다.
            near = self._corridor_blockers(
                ap, planner,
                static_ok=(lambda a: float(getattr(a, 'speed', 0.0)) >= self.ot_v_max)
                if self.sig_lead_ok else (lambda _a: True))
            ok = not any(b[0] <= self.sig_timeout_clear_m for b in near)
        self._sig_wait_ticks = self._sig_wait_ticks + 1 if ok else 0
        if ok and self._sig_wait_ticks >= self.sig_timeout_ticks and not self._sig_go:
            self._sig_go = True
            self._sig_go_tl = tl_id
        if sig is not None:
            sig['timeout_s'] = round(self._sig_wait_ticks / self.hz, 1)
            sig['timeout_go'] = self._sig_go

    # ── RTOR 적색 신호 우회전 (2026-09-06 승인, 기본 off) ─────────────────────
    def _rtor_reset(self) -> None:
        self._rtor_go = False
        self._rtor_go_tl = None
        self._rtor_stop_s = None
        self._rtor_junction_seen = False
        self._rtor_hold_cnt = 0

    def _rtor_active(self, planner) -> bool:
        """RTOR 래치가 **지금 전방 신호**에 붙어 있는가 — 소비처 2곳의 게이트.

        래치는 (신호 id, 정지선 s) 에 붙는다. 뒷축이 정지선을 지나 플래너가 다음
        정지선으로 넘어가면 id 가 달라져 여기서 거짓이 된다 — 다음 신호에는 절대
        적용되지 않는다 (교차로 안은 _cross_guard 가 그대로 맡는다).
        """
        if not self._rtor_go or planner is None:
            return False
        nxt = self._next_stopline(planner)
        return nxt is not None and nxt[2] == self._rtor_go_tl

    def _rtor_sig(self, planner) -> tuple:
        """전방 신호의 보고 상태 → ('fresh' | 'stale' | None, controller ids).

        `_signal_stale` 과 같은 식(observe_lights 의 보고 시각, signal_stale_s)을
        읽되 그 함수는 건드리지 않는다 — 그쪽은 B 스위치가 게이트라 B 가 꺼져
        있어도 RTOR 가 stale 을 알아야 하기 때문이다. 관측이 한 번도 없으면(목)
        fresh 로 본다.
        """
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
        """정지선 앞 앞차 → 사유 문자열, 없으면 None.

        B 큐(_tick_queue) 또는 standoff 회랑(_tick_corridor, 정지 ≥ standoff_stop_s)
        의 정지선 앞 객체, 또는 PDM 선행차 판정(compute_leading_vehicles — 이동 중
        포함)의 정지선 앞 차량. B-3 의 80 m 회랑 전체 판정은 쓰지 않는다 — 정지선
        너머 차량은 앞차가 아니다.
        """
        if self._tick_queue:
            return 'queue'
        for b in (self._tick_corridor or []):
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
        """보행자 차단 사유, 없으면 None.

        1) 회랑 홀드·의도 래치·PDM walker 플래그 (B-3 시계 조건과 같은 플래그).
        2) 회랑(ped_release_lat_m)은 경로 좌우 2.5 m 뿐이라 횡단보도 대기 보행자를
           덮지 못한다 — 정지선~회전 종료 경로 주변 rtor_ped_guard_m 안 보행자를
           추가로 본다 (서 있는 보행자 포함: 항목 10 은 횡단이 끝날 때까지다).
        """
        if self.ped_hold_ids or self.ped_intent:
            return 'latch'
        if getattr(ap, 'walker_hazard', False) or getattr(ap, 'walker_close', False):
            return 'pdm'
        try:
            walkers = list(ap._world.get_actors().filter('*walker*'))
        except Exception:                                  # noqa: BLE001
            return None
        g = self.rtor_ped_guard_m
        s_lo, s_hi = stop_s - route_s - g, end_s - route_s + g
        for w in walkers:
            loc = w.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is not None and s_lo <= pr[0] <= s_hi and abs(pr[1]) <= g:
                return f'near:{int(getattr(w, "id", -1))}'
        return None

    def _rtor_cross_block(self, ap, lg, ego_lane, jid):
        """교차 차량 차단 사유(id), 없으면 None. 자차 프레임은 VTD 좌표계(좌 = +lat).

        1차: lg.locate 로 차량 차로를 잡아, 그 차로(또는 successor)가 자차 다음
             교차로 jid 에 속하고 자차 도로가 아닌 것만 후보. 차로를 알았는데
             후보가 아니면 제외. locate 실패면 2차만 (보수적).
        2차: 좌측(lat>0) 또는 전방(lon>0)에서 자차 쪽으로 접근 중이고
             거리 < rtor_cross_gap_m 또는 거리/속도 < rtor_cross_ttc_s.
        정지 차량(speed < ped_stop_v)은 제외 — 신호 대기열은 교차 위협이 아니다.
        """
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
            # 자차 쪽으로 오는가 — 차량 진행 방향 · (자차 − 차량) > 0
            if (-dx) * _math.cos(ayaw) + (-dy) * _math.sin(ayaw) <= 0.0:
                continue
            dist = _math.hypot(dx, dy)
            if dist < self.rtor_cross_gap_m or dist / max(v, 0.1) < self.rtor_cross_ttc_s:
                return f'{int(a.id)}'
        return None

    def _rtor_tick(self, ap, planner, ego_speed: float) -> None:
        """RTOR 상태기 — apply 가 B-3 시계 직후 틱당 1회 부른다.

        상태: off(조건 1~4·6 미충족) → hold(정지 구역 안 정지 누적 중) → wait
        (보행자·교차 차량 차단) → go(래치). 래치 중 리셋 = fresh 녹색 / 뒷축이
        교차로 차로를 벗어남 / 정지선 + rtor_release_dist_m. fresh red 재관측·
        신호 id 변경으로는 리셋하지 않는다 (B-3 의 not-stale 리셋과 분리 — 그래서
        변수도 따로다). 진단은 reasons.signal.rtor (스위치 on 일 때만 키 생성).
        """
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

        # ── 조건 1~4, 6 ──────────────────────────────────────────────────
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

        # ── 조건 5: 정지 구역 안 정지 누적 (구역 밖·이동이면 0 부터) ────────
        zone = (d_line - self.front) <= self.rtor_zone_m and ego_speed <= self.rtor_stop_v
        self._rtor_hold_cnt = self._rtor_hold_cnt + 1 if zone else 0
        diag['hold_s'] = round(self._rtor_hold_cnt / self.hz, 1)
        # ── 조건 7·8 (hold 중에도 평가해 진단에 남긴다) ────────────────────
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

    def _rtor_log(self, diag: dict) -> None:
        """reasons.signal.rtor — last_signal 이 없으면(B·B-3 off) 여기서 만든다."""
        if self.last_signal is None:
            self.last_signal = {}
        self.last_signal['rtor'] = diag

    def _stop_target(self, planner, ap) -> tuple | None:
        """정지 후보 대상 — B-3(b) 시한 출발 래치 또는 RTOR 래치(붙은 신호에 한함)가
        살아 있으면 None (그 외는 raw)."""
        if self._sig_go or self._rtor_active(planner):
            return None
        return self._stop_target_raw(planner, ap)

    def _stop_target_raw(self, planner, ap) -> tuple | None:
        """정지 후보를 만들 대상이면 (뒷축거리, 실행 감속 a_eff), 아니면 None.

        색 해석의 **단일 출처**다 — 프로파일과 홀드가 같은 판정을 본다.
          · 적색            → (d, stop_profile_a)
          · 황색 + STOP 래치 → (d, stop_profile_a)  적색과 **완전히 동일** 취급
          · 황색 + GO 래치   → None
          · 적색점멸 (B3)    → 유지시간 채우기 전까지 (d, stop_profile_a),
                              채운 뒤(래치)는 None — "정지 후 진행" 이 규정이다
          · 녹색 / 신호 없음 → None

        판정(a_yellow=4.0)과 실행(stop_profile_a=3.0)의 상수가 **다른 것이
        의도다**. 같게 두면(초기 설계안 B) STOP 판정의 정의상 진입 시
        v ≤ v_allow 라 프로파일이 느슨하고, v_allow 가 v 밑으로 내려올 때까지
        구속하지 못한다. 그 시점엔 최대 감속을 여유 0 으로 요구해 jerk 램프인에
        진다 — 폐루프 12조건에서 걸침 6건. 실행 상수를 작게 두면 진입 즉시
        구속되고 a_dec_max 까지 여유가 남는다 (같은 12조건에서 걸침 0건,
        전부 −1.52~−0.97). 2026-08-30 검증.
        교차로 통과 가드가 걸려 있으면 무조건 None.
        """
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
        # B3 적색 점멸 — 적색과 **같은 실행 축**(④′ 정지 프로파일)으로 세운다.
        # PDM 의 적신호 IDM 은 차간모형이라 정지 컨트롤러가 아니어서, 이걸 안
        # 달면 점멸에서 5.4 m/s 로 정지선을 지난다 (replay 실측 2026-09-09).
        # 유지시간을 채우면 래치가 붙어 여기서 None 이 되고, 같은 틱에
        # signal_release 가 PDM 의 적신호 IDM 도 건너뛴다 = 재출발.
        if state == 'FlashRed' and not self._flash_release(planner):
            return (d_line, self.stop_profile_a)
        return None

    def _red_stopline_dist(self, planner) -> float | None:
        """구 인터페이스 — 적색이면 뒷축거리. 색 해석은 _stop_target 이 한다."""
        nxt = self._next_stopline(planner)
        if nxt is None or nxt[1] != 'Red':
            return None
        return nxt[0]

    def _ped_intent(self, planner, ap, ego_speed: float):
        """정지 관찰 중이던 보행자가 **경로 쪽으로** 걸어나오는 순간의 정지 후보.

        반환 `(v_allow, a_req, ped_id)` 또는 None.

        왜 PDM 예측을 못 기다리는가 — `forecast_walkers` 는 등속 2 s 직선 예측이고
        속도는 `min_walker_speed` (0.5) 로만 하한을 둔다. 예측 도달거리가
        `v_ped·2 + pedestrian_minimum_extent(1.5)` 라, 서 있는 보행자는 2.5 m 밖에
        못 뻗는다. 실측 2026-09-01 실전주행_교통류_01 id7: 횡 6.35 m 에서 걸어나오는
        데 v_ped 가 1.70 m/s 가 되어서야(0.5 s 뒤) 회랑에 닿았고, 그 사이 필요
        감속이 3.31 → 4.29 m/s² 로 올라 `a_dec_max` (4.0) 를 넘겨 접촉했다.

        여기서는 교차를 기다리지 않고 **경로 횡거리의 감소율**로 의도를 읽는다.
        횡거리는 경로 기하에 대한 값이라 자차 운동과 무관하다 (자차 프레임 횡거리와
        다르다 — 자차가 돌면 서 있는 보행자도 움직이는 것처럼 보인다).

        게이트 (모두 만족해야 래치):
          · `obj_static_s` 이상 정지 관찰을 마친 id (`ped_static`)
          · 보행자 자신의 속도 ≥ `ped_intent_v`
          · **경로 쪽** 횡속도 = −d|lat|/dt ≥ `ped_intent_v`
            → 멀어지는 방향(+)·경로와 나란한 이동(≈0)은 부호/크기에서 걸러진다
          · 전방 (`0 < s_rel ≤ detect_max_m`)

        래치는 보행자가 지나가면(뒤로 감) 또는 관측이 끊기면 풀린다. 정지 목표는
        보행자의 **횡단 지점**이고 gap 은 PDM 주입값
        `idm_pedestrian_minimum_distance` 를 그대로 읽는다 (단일 출처).

        위치 기반 해제 (A-1, `_ped_release_tick`) — 횡단을 **마친** 보행자는 뒤로
        가지 않아 위 두 해제로는 절대 풀리지 않는다 (실측 2026-09-02 좌회전8 id4:
        |lat| 8.8 m 에 서 있는데 로그 끝까지 정지). 회랑 밖(|lat| > ped_release_lat_m)
        에서 경로 쪽으로 오지 않고(v_toward ≤ 0) **멈췄거나 차도 밖이면**
        ped_release_s 동안 지속 시 해제. 도로 위를 계속 걷는 동안은 유지한다
        (되돌아올 수 있다). backstop: 회랑 밖이면 ped_backstop_s 뒤 조건 무관 해제.
        """
        if self.ped_intent_v <= 0.0 or self.stop_profile_a <= 0.0:
            return None
        try:
            walkers = list(ap._world.get_actors().filter('*walker*'))
        except Exception:                                  # noqa: BLE001
            return None
        live = set()
        best = None
        self.ped_diag = {}
        self.ped_released = {}
        self.ped_all = {}
        # 회랑 폭이 0(A-1 비활성)이면 회랑 자체가 정의되지 않아 홀드도 비활성이다.
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
                # 투영 불가 틱 — 직전 횡거리는 버리고(차분이 무의미) 해제 카운터도
                # 리셋한다. 래치와 hold 는 유지 — 관측이 끊긴 것이 아니다.
                self.ped_lat.pop(wid, None)
                self.ped_clear.pop(wid, None)
                # P4-M: 홀드 래치가 살아 있으면 투영 불가 틱도 coast 로 다룬다 —
                # 래치 on 인데 v_allow 가 사라지는 틱을 만들지 않는다 (상한은 coast).
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
            # 경로 쪽 횡속도 — |lat| 이 줄어드는 속도, 멀어지면 음수. 래치 여부와
            # 무관하게 매 틱 산출한다 (해제 판정과 진단이 같은 값을 본다).
            v_toward = None if prev is None else (abs(prev) - abs(lat)) * self.hz
            if v_toward is not None:
                self.ped_vt[wid] = v_toward                # 다음 틱 큐 보행자 가드용 (C-3)
            if self._static_ok(w):
                self.ped_static.add(wid)               # 정지 관찰 완료 (지속 기억)
            if not (0.0 < s_rel <= self.detect_max_m):
                self._ped_unlatch(wid)                 # 지나갔거나 범위 밖
                continue
            self.ped_miss[wid] = 0                     # 관측됨 — coast 카운터 리셋
            # ── 회랑 홀드 래치 (P4-M) ───────────────────────────────────────
            # 회랑(|lat| < ped_release_lat_m) 안에 있는 보행자는 **누구든** 홀드한다 —
            # 의도 래치 여부·몇 명인지·어느 id 가 best 로 뽑히는지와 무관하다. 회랑
            # 폭은 A-1 해제·C-3 큐 가드와 같은 값을 그대로 읽는다 (판정 기준 불변).
            if (multi and abs(lat) < self.ped_release_lat
                    and wid not in self.ped_hold_ids):
                self.ped_hold_ids.add(wid)
                self.ped_clear.setdefault(wid, 0)
                self.ped_hold.setdefault(wid, 0)
            if wid not in self.ped_intent:
                latch_ok = v_toward is not None
                if latch_ok:
                    w_speed = float(getattr(w, 'speed', 0.0))
                    if wid in self.ped_static:
                        latch_ok = (w_speed >= self.ped_intent_v
                                    and v_toward >= self.ped_intent_v)
                    else:
                        latch_ok = self._ped_walkin(wid, w_speed, v_toward, lat)
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
            # ── 순서 보장: **해제 판정이 먼저**, v_allow 산출은 그 다음 ──────────
            # 홀드 래치가 살아 있는 한 이 아래에서 v_allow 는 0 이다. 다른 계산 경로
            # (프로파일·coast·비상)가 홀드보다 앞서 값을 만들 수 없다.
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
            # 회랑 홀드 = 0. 회랑 밖 의도 래치 = 횡단 지점 앞 정지 프로파일 (이전 그대로).
            v_allow = 0.0 if hold else _math.sqrt(2.0 * self.stop_profile_a * max(0.0, d_eff))
            # 분모 하한 0.5 m — d_eff→0 에서 a_req 가 발산해 로그가 못 쓰게 된다.
            # 판정(임계 초과 여부)에는 영향이 없다: 하한을 써도 이미 임계 위다.
            a_req = ego_speed * ego_speed / (2.0 * max(d_eff, 0.5))
            if multi:
                self.ped_last[wid] = (v_allow, a_req)
                self.ped_all[wid] = {'s_rel': round(float(s_rel), 2),
                                     'lat': round(float(lat), 2),
                                     'v_allow': round(float(v_allow), 2),
                                     'latched': wid in self.ped_intent, 'hold': bool(hold)}
            if best is None or v_allow < best[0]:
                best = (v_allow, a_req, wid)
        # 관측이 끊긴 id 정리 (obj_ticks 의 grace 와 별개 — 이전 동작은 즉시 해제).
        # P4-M coast: ped_hold_coast_s 안의 미관측은 이탈이 아니다 — 직전 틱 기여
        # (홀드면 0, 의도 래치면 직전 프로파일)를 그대로 낸다. 두 타이머의 관계:
        #   · coast (ped_hold_coast_s, 미관측 연속 틱) — **관측이 없을 때만** 센다.
        #     만료 + 그 사이 재관측 없음 → 해제. 재관측되면 0 으로 리셋.
        #   · release (ped_release_s, _ped_release_tick) — **관측된 틱에서만** 센다.
        #     회랑 밖 ∧ 멀어짐 ∧ (멈춤 ∨ 차도 밖) 연속 시 해제.
        #   둘은 서로 다른 틱에서만 진행하므로 합산되지 않는다 — 과홀드 상한은
        #   max(release 경로, coast 경로) 이지 합이 아니다.
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
        for wid in list(self.ped_static):
            if wid not in live and wid not in self.obj_ticks:
                self.ped_static.discard(wid)
        for wid in list(self.ped_lat):
            if wid not in live:
                self.ped_lat.pop(wid, None)
        for wid in list(self.ped_vt):
            if wid not in live:
                self.ped_vt.pop(wid, None)
        for wid in list(self.ped_walkin):
            if wid not in live:
                self.ped_walkin.pop(wid, None)
        return best

    def _ped_walkin(self, wid, w_speed: float, v_toward: float, lat: float) -> bool:
        """걷는 채로 등장한 보행자의 래치 조건 (A-4) — 정지 관찰(ped_static) 없이.

        GT 범위(80 m) 안으로 걸어 들어오거나 걷는 상태로 스폰된 보행자는 ped_static
        전제 때문에 래치되지 않았고, A-2 로 PDM 상자가 0.5 가 되어 그런 보행자의 PDM
        검출 시점이 3.44 → 2.44 m 로 늦어진다 (무단횡단 시나리오 직결). 여기서는
        연속 ped_walkin_s 동안 보행자 속도 ≥ ped_intent_v ∧ 경로 쪽 횡속도 ≥ ped_walkin_v
        ∧ |lat| < ped_walkin_lat_m 이면 래치한다. 정지 관찰 경로(0.3, 1틱)보다 임계·
        시간을 두는 이유는 노이즈 여유 — 첫 관측 틱의 횡거리 차분은 스폰·코스팅으로
        튈 수 있다. 한 틱이라도 깨지면 처음부터. 래치 후 동작·해제는 A-1 그대로.
        """
        if not self.walkin_enable:
            return False
        ok = (w_speed >= self.ped_intent_v and v_toward >= self.walkin_v
              and abs(lat) < self.walkin_lat)
        self.ped_walkin[wid] = self.ped_walkin.get(wid, 0) + 1 if ok else 0
        return self.ped_walkin[wid] >= self.walkin_ticks

    def _crosswalk_zones(self, planner) -> list:
        """경로상 횡단보도 route_s 구간 [(s0, s1) …]. 시작 시 1회 (A-3).

        출처는 레인그래프 lane 레코드의 crosswalks — world.ahead 의 'crosswalk' 와 같은
        데이터다. 경로 전체에 횡단보도가 하나도 없으면 정지선(_all_stopline_s)을
        점 구간으로 쓴다 (횡단보도 마킹이 빠진 지도 구간 대비).
        """
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
        """횡단보도 앞 서행 후보 (A-3) → `(v_allow, 진단)` 또는 None.

        대상: 전방 횡단보도 ±ped_crosswalk_zone_m 안, 회랑 밖(|lat| ≥ ped_release_lat_m)
        이고 ped_crosswalk_lat_m 안에 **서 있는**(speed < ped_stop_v) 보행자. 회랑 안은
        정지(PDM·래치)의 몫이고, _ped_intent 래치가 선 id 는 그쪽이 우선이다.

        동작: 대기 단계는 보행자 앞(앞범퍼 + idm_pedestrian_minimum_distance)에 서는
        정지 프로파일 v = √(2·a·d_eff) — 멀면 제한속도보다 커서 스스로 비활성이다.
        대기 틱은 **정지 중**(v < ped_stop_v ∧ 계획 정지점 앞 zone_m 안) 에만 센다 —
        감지 직후부터 세면 80 m 밖에서 3 s 만 지나면 서행이 시작되고, 먼 적신호 정지가
        대기를 채워도 안 된다. PDM 이 몇 m 앞에 먼저 세운 경우는 채워진다.
        ped_crosswalk_wait_s 를 채우면 보행자를 지날 때까지 v ≤ ped_crosswalk_creep_v.
        보행자가 걷기 시작하면(speed ≥ ped_stop_v) 대상에서 빠지고 래치가 이어받는다.
        """
        if not self.cw_enable or self.stop_profile_a <= 0.0:
            return None
        try:
            walkers = list(ap._world.get_actors().filter('*walker*'))
        except Exception:                                  # noqa: BLE001
            return None
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
            # 대기는 정지 중 + 계획 정지점 근처(d_eff ≤ zone_m)에서만 센다 — PDM 이
            # 몇 m 앞에 먼저 세워도 채워지고, 먼 적신호 정지(30 m 밖)는 세지 않는다.
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
                self.cw_wait.pop(wid, None)              # 지나감·걷기 시작·소실 → 처음부터
        return best

    def _ped_unlatch(self, wid) -> None:
        """래치 해제 + 해제 카운터 정리 (지나감·끊김·위치 해제 공통).

        의도 래치와 회랑 홀드 래치(P4-M)는 **같이** 풀린다 — 해제 조건은 하나
        (_ped_release_tick / 지나감 / coast 만료)이고, 그 뒤에야 v_allow 가 사라진다.
        """
        self.ped_intent.discard(wid)
        self.ped_hold_ids.discard(wid)
        self.ped_clear.pop(wid, None)
        self.ped_hold.pop(wid, None)
        self.ped_miss.pop(wid, None)
        self.ped_last.pop(wid, None)

    def _ped_release_tick(self, wid, w, lat: float, v_toward) -> str | None:
        """래치된 보행자의 **위치 기반 해제** 판정 (A-1). 투영이 된 틱마다 부른다.

        카운터를 갱신하고 해제 사유('clear' / 'backstop') 또는 None 을 준다.
          · clear   : 회랑 밖(|lat| > ped_release_lat_m) ∧ 경로 쪽으로 오지 않음
                      (v_toward ≤ 0) ∧ (보행자 정지 speed < ped_stop_v ∨ 차도 밖
                      |lat| > ped_offroad_lat_m) 이 ped_release_s 연속. 한 틱이라도
                      깨지면 처음부터 — 되돌아오는 보행자는 즉시 다시 구속한다.
          · backstop: 래치 후 ped_backstop_s 가 지났고 **지금 회랑 밖**이면 조건
                      무관 해제. 회랑 안(|lat| ≤ ped_release_lat_m)이면 유지.
        보행자 속도(자차 아님)를 보는 이유 — 도로 위를 걷는 동안은 방향을 바꿔
        되돌아올 수 있어, 멈추거나 차도를 완전히 벗어나야 '지나갔다' 고 본다.
        """
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

    def _standoff_profile(self, ego_speed: float) -> float | None:
        """WAIT/관찰 중 **장애물 앞 standoff 에 서도록** 하는 속도 상한 — min() 후보.

        ④′ 정지선 프로파일과 같은 형태다: v_allow = √(2·a_stop·(d − standoff)).
        standoff 는 시프트 전이가 들어갈 공간이라 전이 길이 공식과 정합시킨다:
            standoff = max(standoff_floor_m, shift_k_s · v)
        바닥은 **standoff_floor_m** 이다 — shift_latest_m 이 아니다. 둘은 기본값이
        같지만(25.0) 소비처가 다르다: shift_latest_m 은 _shift_speed_cap 의
        미리보기 창과 _try_overtake 의 PREEMPT 시간 예산이 쓰고, 여기만 이 값을
        읽는다. 정지 거리만 조정하려면 standoff_floor_m 을 움직인다.
        IDM 의 정지 gap 과 충돌하지 않는다 — 둘 다 상한이고 min() 이 낮은 쪽을 쓴다.
        """
        if self.wait_target_d is None or self.stop_profile_a <= 0.0:
            return None
        standoff = max(self.standoff_floor_m, self.shift_k_s * max(ego_speed, 0.1))
        d = self.wait_target_d - standoff
        if d >= 0.0:
            v = _math.sqrt(2.0 * self.stop_profile_a * d)      # 기준선 밖 — 무수정
            j_rel = self._junction_release()
            if not j_rel and self.ns_level < 1:
                return v
            # ① 모드 A (A1) — d 가 기준선에 정확히 얹히면 √(2a·0) = 0 인데,
            # 이 갈래는 _standoff_creep 을 부르지 않으므로 크립 게이트에
            # **도달조차 못 한다** (시뮬: 장애물 24 m → d 22.0 에서 60 s 무진전,
            # creep_* 키 없음). 무장된 동안은 여기도 크립을 거쳐 바닥을 깔고,
            # 회전 중이므로 상한으로 자른다. 배제(stop_gap·ped_hold·cause)는
            # _standoff_creep 이 그대로 판정해 0 을 돌려주고, 그때는 프로파일
            # 값 v 가 그대로 남는다 (바닥만 없어질 뿐 더 세우지 않는다).
            out = max(v, self._standoff_creep(standoff, ego_speed))
            return min(out, self.j_creep_cap) if j_rel else out
        return self._standoff_creep(standoff, ego_speed)       # 기준선 안 — 바닥

    def _red_intervals(self, planner) -> list:
        """경로 위 붉은 구간 [(진입 route_s, 이탈 route_s)] — 경로당 1회.

        `turn_intervals` 와 같은 관례로 시작 시 한 번만 만든다. 재료는
        `route_waypoints[i].key/.s` 와 lane_graph 의 `red_spans` 다 — 제한속도
        배열(`speed_limits`)로 역추정하지 않는다. 그 배열은 carry 가 섞여 있어
        "구간에서 물고 나온 30" 과 "구간 안" 이 구분되지 않는다.

        이탈 쪽은 `red_zone.exit_margin_m` 을 얹은 지점이다 —
        `speed_limit_at` 이 캡을 푸는 지점과 같은 축이라야 후보가 어긋나지 않는다.
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
        """붉은 구간 **진입 전** 감속 상한 — min() 후보. ④′ 와 같은 형태다.

            v_ceiling = √(v_zone² + 2·a·d)        d = 진입점까지 남은 경로거리

        구간 안에서는 후보를 내지 않는다 — 거기서는 제한속도(`red_zone.limit_kph`
        − `speed.margin_kph`)가 이미 상한이고, 두 상한을 겹치면 어느 쪽이 묶는지
        로그에서 갈리지 않는다.

        왜 필요한가: 50 도로(45 km/h)에서 27 km/h 로 줄이려면 a=2.0 에 **25 m** 가
        든다. 붉은 구간은 중앙 25.1 m 이고 124개 중 43개가 18 m 미만이라, 진입
        **뒤에** 줄이기 시작하면 구간 안에서 초과가 난다 (항목 2 는 초과 1 km/h
        넘으면 경미, 5 넘으면 중대다).
        실측 2026-09-05: 27경로 붉은 구간 진입 19건 중 3건은 앞 구간과의 틈이
        1.0~9.0 m 뿐이라, 그 사이에서 50 으로 가속하면 되돌릴 방법이 없다.
        이 후보는 그 틈에서 천장을 각각 27.9 / 34.6 km/h 로 눌러 애초에 못
        올라가게 한다.
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

    def _creep_geom_need(self, ego_speed: float) -> float | None:
        """기하 완성 게이트(⑥)의 need — `_side_pass` 와 **같은 식**이다.

        need = trans_m + ahead_m + shift_geom_margin_m,
        trans_m = max(transition_m, shift_k_s·v), ahead_m 은 BREAKOUT L3 이상에서
        shift_ahead_l3_m. 상수를 복제하지 않고 같은 필드를 읽는다 — 어긋나면
        "여기서 아직 가능하다고 본 것"과 "저기서 기각하는 것"이 달라진다.
        게이트가 꺼져 있으면 기하상 못 간다는 판정 자체가 없으므로 None.
        """
        if not self.geom_gate:
            return None
        trans_m = max(self.ot_trans_m, self.shift_k_s * max(ego_speed, 0.1))
        ahead_m = (self.shift_ahead_l3_m if self.bo_level >= self.geom_relax_lvl
                   else self.shift_ahead_m)
        return trans_m + ahead_m + self.geom_margin_m

    def _creep_gate(self, d: float, ego_speed: float):
        """크립 지연 게이트 — (open_why, hold_why, need).

        크립은 **비가역**이다: s_rel 이 geom need 아래로 내려가면 시프트가 영영
        불가능해진다. 그래서 아직 기하상 가능한 동안에는 크립을 보류하고
        시프트·BREAKOUT 사다리에 먼저 기회를 준다 (실측 2026-09-05 155735:
        01 은 크립이 occupied 창을 4.95 s 로 잘랐고, 02 는 크립이 bo_stuck_ticks
        를 리셋해 활성 span 이 안 풀려 시프트 시도가 0회였다).

        여는 조건 (하나라도):
          ① d < need      — 이미 기하적으로 불가. 기다릴 이유가 없다
          ② BREAKOUT L4   — 사다리 끝까지 갔는데 시프트 실패
          ③ 보류 누적 ≥ delay — 안전망 (사다리가 아예 안 도는 경우). 시계는
             _creep_hold_ticks 로 **배제(적신호·보행자·큐)에 걸린 틱은 빼고**
             센다 — ot_blocked_ticks 를 쓰면 적신호 대기가 그대로 쌓여 녹색
             직후 지연이 이미 만료된다 (실측 155049 02: hold_s 22.6).
        보류 중에는 v_allow = 0 이라 자차가 서 있고, 그래서 ot_blocked_ticks 와
        bo_stuck_ticks 가 **정상적으로 쌓인다** — 지연이 사다리를 굶기지 않는다.
        """
        if self.ns_level >= 1:
            # ⓪′ never_stall (A2) — 여기서 기다리는 '시프트 가능성' 이 무엇이든,
            # deadlock_max_s 를 넘겼으면 그 기다림 자체가 stall 이다.
            return 'never_stall', None, self._creep_geom_need(ego_speed)
        if self._junction_release():
            # ⓪ 교차로 해제 (A1) — 다른 조건보다 먼저다. 여기서 기다리는 대상인
            # '시프트 가능성' 이 교차로 lane 에서는 **원리적으로 0** 이라
            # (reject='junction'), need 를 보고 보류하는 것 자체가 무의미하다.
            return 'junction', None, self._creep_geom_need(ego_speed)
        if self.creep_delay_ticks <= 0:
            return 'no_gate', None, None                   # 지연 없음 (이전 동작)
        need = self._creep_geom_need(ego_speed)
        if need is not None and d < need:
            return 'need', None, need                      # ①
        if self.bo_level >= self.BO_CREEP:
            return 'breakout', None, need                  # ②
        if self._creep_hold_ticks >= self.creep_delay_ticks:
            return 'delay', None, need                     # ③
        why = 'need' if need is not None else 'breakout'
        return None, why, need

    def _standoff_creep(self, standoff: float, ego_speed: float = 0.0) -> float:
        """기준선 **안쪽**(d < standoff)의 바닥 — 0 고정 대신 크립 [m/s].

        왜 필요한가: d < standoff 이면 √(2a·max(0, d−standoff)) 가 항구적으로
        0 이고 min() 사다리에 이를 되올릴 후보가 없다 → 영구 정지. 실측
        2026-09-04 7건이 이 형태로 11.5~78.2 s 멈췄다. standoff_floor_m 을
        낮춰도 사라지지 않는다 — 오버슛 0.7~2.6 m 로 매번 바닥을 넘어 들어가
        잠금 **지점만** 앞으로 옮겨진다.

        min() 안의 후보다. 이 함수는 `_standoff_profile` 의 **반환값만** 바꾸고,
        호출처(apply)의 병합은 여전히 `so < candidate` 하나다 — 신호·보행자·
        정지선·종점 후보가 더 낮으면 그쪽이 이긴다. 오버라이드가 아니다.

        진짜 정지 거리는 객체 크기를 반영한다:
            d_stop = front(뒷축→앞범퍼) + 객체 반길이 + standoff_creep_gap_m
        d 가 **뒷축 → 객체 중심** 축이기 때문이다. 고정값(예: 2 m)으로 두면
        앞범퍼가 객체 중심을 1.8 m 지나쳐 서서 충돌이 된다.

        배제는 새로 만들지 않고 `_obstacle_cause` 를 그대로 쓴다 — 보행자
        (walker_hazard/close)·적황 신호(traffic_light_hazard·y_decision·
        cross_guard·sl_hold_left)·큐 대기(_tick_queue)·경로 종점(latched·
        d_end ≤ active_m)·정지표지를 이미 전부 본다. `ped_hold_ids`(횡단보도
        홀드 래치)는 PDM 플래그와 축이 달라 2차 방어로 따로 본다.
        """
        j_rel = self._junction_release()
        ns = self.ns_level >= 1
        if not self.standoff_creep and not j_rel and not ns:
            return 0.0          # 이전 동작. 진단도 남기지 않는다 — off 는 로그까지 동일
        # 교차로 해제가 무장되면 크립 속도의 바닥을 올리고 상한으로 자른다 (A1).
        # 바닥이 필요한 이유는 모드 A: 기준선 **밖**에서 부르는 경로라 여기서
        # standoff_creep_v(0.8)만 돌려주면 프로파일 값보다 낮아 아무 효과가 없다.
        v_creep = self.standoff_creep_v
        if ns:
            v_creep = max(v_creep, self._ns_creep_v())      # A2 단계 바닥
        if j_rel:
            v_creep = min(max(v_creep, self.j_creep_floor), self.j_creep_cap)
        d = float(self.wait_target_d)
        diag = {'creep_d': round(d, 1), 'creep_standoff': round(standoff, 1)}
        if j_rel:
            diag['creep_junction'] = True
        if ns:
            diag['creep_ns_lvl'] = self.ns_level

        # 진단 키가 'so_creep' 인 이유: BREAKOUT 진단이 같은 last_avoid 에 'creep'
        # 을 나중에 써서(아래 bo_state 블록) 이름이 겹치면 덮인다 — 실측 02_직진3
        # 27틱에서 크립 발동이 False 로 뒤집혔다 (2026-09-05).
        def block(why):
            # 배제로 막힌 틱은 지연 시계에 넣지 않는다 — 적신호 21 s 를 세면
            # 녹색 직후 지연이 이미 만료된 상태가 된다 (위 _creep_hold_ticks 주석).
            self._creep_hold_ticks = 0
            self._creep_diag = dict(diag, so_creep=False, creep_block=why)
            # 배제(신호·보행자·큐)는 문맥이 바뀐 것이다 — delay 래치도 시계와 같이
            # 버린다 (비용은 최대 지연 1회). stop_gap·no_size 는 크립 완료·크기
            # 미상이라 문맥이 그대로다 — 래치 유지.
            if self._creep_open_latched and why in ('cause', 'ped_hold'):
                self._creep_open_latched = False
                self._creep_diag['creep_latch_why'] = 'release:' + why
            return 0.0

        # A2 1단계는 ③ no_size · ① stop_gap 을 건너뛴다 — 둘 다 상한 없는 0 이다.
        # 건너뛰어도 **강제 전진이 아니다**: 이 값은 min() 상한이라 PDM 의 IDM·OBB
        # 가 그대로 자기 간격에서 세운다. 그걸 무효화하는 것은 3단계뿐이다.
        if self.standoff_half_len is None:
            if not ns:
                return block('no_size')                 # 크기 미상 → 정지 거리 불명
            diag['creep_ns_relax'] = 'no_size'
            d_stop = None
        else:
            d_stop = self.front + self.standoff_half_len + self.standoff_creep_gap_m
            diag['creep_stop_m'] = round(d_stop, 2)
        if d_stop is not None and d <= d_stop:
            if not ns:
                return block('stop_gap')                # 진짜 정지
            diag['creep_ns_relax'] = 'stop_gap'
        if self.ped_hold_ids:
            return block('ped_hold')                    # 보행자는 어떤 단계에서도 유지
        ap = self._ap
        planner = getattr(ap, '_waypoint_planner', None) if ap is not None else None
        if planner is None:
            return block('cause')
        # A2 는 UNKNOWN 큐까지 원인으로 본다 (_ns_cause). 적신호·보행자·종점은
        # 그 함수도 그대로 제외한다 — 배제 목록은 여전히 한 곳이다.
        if not (self._ns_cause(planner, ap) if ns
                else self._obstacle_cause(planner, ap)):
            return block('cause')
        # 지연 게이트 — 시프트가 확실히 불가능해지기 전에는 열지 않는다 (_creep_gate).
        if self.standoff_id != self._creep_hold_id:     # 대상이 바뀌면 새로 센다
            self._creep_hold_ticks = 0
            self._creep_hold_id = self.standoff_id
            self._creep_open_latched = False              # 래치도 대상별이다
        if self._creep_open_latched and self.ot_span != self._creep_latch_span:
            self._creep_open_latched = False              # 시프트가 성립했다 — 새 문맥
            self._creep_hold_ticks = 0                    # 시계도 새로 (만료값이 남으면 즉시 재래치)
        if self._creep_open_latched:
            # delay 래치 — 게이트를 다시 묻지 않고 시계도 건드리지 않는다.
            need = self._creep_geom_need(ego_speed)
            self._creep_diag = dict(diag, so_creep=True, creep_v=v_creep,
                                    creep_open_why='delay', creep_open_latched=True,
                                    creep_latch_why='delay',
                                    creep_need_m=round(need, 1) if need is not None else None,
                                    creep_hold_s=round(self._creep_hold_ticks / self.hz, 1),
                                    creep_bo_lvl=self.bo_level)
            return v_creep
        open_why, hold_why, need = self._creep_gate(d, ego_speed)
        gate = {'creep_need_m': round(need, 1) if need is not None else None,
                'creep_hold_s': round(self._creep_hold_ticks / self.hz, 1),
                'creep_bo_lvl': self.bo_level}
        if open_why is None:
            self._creep_hold_ticks += 1
            self._creep_diag = dict(diag, so_creep=False, creep_hold=True,
                                    creep_hold_why=hold_why, **gate)
            return 0.0
        if open_why == 'delay':
            # 시계가 곧 조건이다 — 0 으로 되돌리면 다음 틱에 닫힌다. 래치로 연다.
            self._creep_open_latched = True
            self._creep_latch_span = self.ot_span
            self._creep_diag = dict(diag, so_creep=True, creep_v=v_creep,
                                    creep_open_why=open_why, creep_open_latched=True,
                                    creep_latch_why='delay', **gate)
            return v_creep
        self._creep_hold_ticks = 0                       # need·breakout: 조건이 지속된다
        self._creep_diag = dict(diag, so_creep=True, creep_v=v_creep,
                                creep_open_why=open_why, **gate)
        return v_creep

    def _curvature_cap(self, planner) -> float | None:
        """앞 경로 곡률 상한 — `v ≤ √(a_lat_max / κ_max)` (B1). min() 후보.

        실측 실경로_01_PathShape03 rs 89~107: 연결로 797(R 6.47 m)에 6.8 m/s 로
        들어가 조향이 −0.480 에 **9 m 포화**하고 t_off 1.36 m 까지 벌어진 뒤
        +0.480 으로 되튀었다. 그 구간 reasons.curvature 는 전 틱 None 이었다 —
        곡률을 보는 후보가 아예 없었다.

        κ 는 **실제로 따라갈 경로점**(planner.route_points)에서 잰다. lane_graph
        의 차로 R 을 쓰지 않는 이유: 회피 시프트·차선변경 블렌드가 경로를 옆으로
        밀면 실제 곡률이 차로 곡률과 달라진다. 따라가는 선에서 재야 맞다.

        3점 외접원으로 κ = 4·A / (a·b·c). 표본 간격은 2 m 로 둔다 — 0.1 m 격자
        그대로 쓰면 좌표 잡음이 κ 를 부풀려 직선에서도 상한이 생긴다.
        curvature_min_kappa 미만(≈ R 200 m 초과)은 직선으로 보고 버린다.
        """
        if not self.curv_cap or self.curv_a_lat <= 0.0:
            return None
        pts = getattr(planner, 'route_points', None)
        if pts is None or len(pts) < 3:
            return None
        ppm = float(getattr(planner, 'points_per_meter', 10) or 10)
        step = max(1, int(round(2.0 * ppm)))               # 2 m 간격 표본
        i0 = int(planner.route_index)
        n_ahead = int(round(self.curv_look_m * ppm))
        idx = list(range(i0, min(i0 + n_ahead + 1, len(pts)), step))
        if len(idx) < 3:
            return None
        k_max, k_at = 0.0, None
        for j in range(len(idx) - 2):
            (ax, ay), (bx, by), (cx, cy) = (pts[idx[j]][:2], pts[idx[j + 1]][:2],
                                            pts[idx[j + 2]][:2])
            a = _math.hypot(bx - ax, by - ay)
            b = _math.hypot(cx - bx, cy - by)
            c = _math.hypot(cx - ax, cy - ay)
            if a < 1e-6 or b < 1e-6 or c < 1e-6:
                continue
            area2 = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))   # 2·A
            k = 2.0 * area2 / (a * b * c)                  # 4A/(abc), area2 = 2A
            if k > k_max:
                k_max, k_at = k, idx[j + 1]
        if k_max < self.curv_min_k:
            return None                                    # 직선 — 후보 없음
        v = _math.sqrt(self.curv_a_lat / k_max)
        self.last_curv_info = {'kappa': round(k_max, 4), 'R_m': round(1.0 / k_max, 1),
                               'v_cap': round(v, 2),
                               'ahead_m': None if k_at is None
                               else round((k_at - i0) / ppm, 1)}
        return v

    def _lc_speed_cap(self, planner) -> float | None:
        """계획 차선변경 창 안·직전의 속도 상한 ([3](a)). min() 후보.

        실측 실경로_01_PathShape03 rs 413~455: 도로 418 에서 차선변경 4회가
        연속인데 자차가 그 구간에서 8 → 13.3 m/s 로 **가속**했다. 램프 길이는
        v × lc_move_s 라 속도에 비례하므로 39 m 로 구워졌고 hop 간격 20 m 를
        넘어 겹쳤다 — 경로가 2차로를 한 번에 건너는 S자가 되어 heading_err
        +31°, 조향 포화 4회, |t_off| 1.85 m (항목 3 중대 + 항목 6).

        창은 플래너가 만든 램프(`planner.lc_ramps`, route_s 구간)를 그대로
        읽는다 — 제어기가 창을 다시 계산하면 플래너가 실제로 민 구간과 어긋난다.

        **연속 hop 은 하나로 본다**: 다음 창이 이번 창 끝 + lc_hop_chain_sep_m
        안에서 시작하면 사이에서도 상한을 유지한다. 사이에서 가속하면 다음
        램프가 다시 길어져 겹친다.

        창 **진입 전**에는 정지선 프로파일과 같은 식으로 미리 줄인다:
            v = √(cap² + 2·a_stop·d)
        멀면 제한속도보다 커서 min() 에 지므로 스스로 비활성이다.
        """
        if not self.lc_cap or self.lc_cap_v <= 0.0 or self.stop_profile_a <= 0.0:
            return None
        wins = getattr(planner, 'lc_ramps', None)
        if not wins:
            return None
        if self._lc_windows is None:
            merged: list = []
            for w0, w1 in sorted(wins):
                if merged and w0 - merged[-1][1] <= self.lc_chain_sep_m:
                    merged[-1][1] = max(merged[-1][1], w1)
                else:
                    merged.append([float(w0), float(w1)])
            self._lc_windows = merged
        route_s = float(planner.route_s[planner.route_index])
        for a, b in self._lc_windows:
            if b < route_s:
                continue                                   # 이미 지난 창
            if a <= route_s:
                self.last_lc_cap = self.lc_cap_v
                return self.lc_cap_v                       # 창 안 (연속 hop 포함)
            d = a - route_s
            if d > self.lc_look_m:
                return None                                # 아직 멀다 (진단도 안 남긴다)
            v = _math.sqrt(self.lc_cap_v * self.lc_cap_v
                           + 2.0 * self.stop_profile_a * d)
            self.last_lc_cap = v
            return v                                       # 진입 전 감속
        return None

    def _no_accel_red_cap(self, planner, ap, ego_speed: float) -> float | None:
        """적신호 정지 대상이 **가까이** 있으면 목표를 현재 속도로 덮는다 — 상한 후보.

        "적신호를 향해 가속하지 않는다" 는 것뿐이다. 감속은 ④′·IDM 이 그대로 한다.
        거리 한계는 `speed.red_lookahead_m` 를 그대로 읽는다 (새 상수 없음) —
        300 m 밖 적색에까지 걸면 정상 주행이 죽는다.
        """
        if not self.no_accel_red:
            return None
        tgt = self._stop_target(planner, ap)
        if tgt is None:
            return None
        d_line = float(tgt[0])
        if d_line > self.red_look_m:
            return None
        return max(0.0, float(ego_speed))

    def _stopline_profile(self, planner, ap, ego_speed: float = 0.0) -> float | None:
        """적신호 정지선까지의 **정지 프로파일 속도 상한** — min() 후보.

        PDM 의 적신호 IDM 은 차간모형이라 정지 컨트롤러가 아니다: 평형이
        s* (= s0 + vT + v²/2√(ab)) 라, 남은 거리가 s* 보다 조금만 커도 **가속을
        요구한다** (실측 2026-08-30 실전주행: 접근 92틱 중 41틱이 v 보다 높은
        목표, err/dt 최대 +10.5 m/s²). 종방향이 그 요구에 브레이크를 풀면
        타행으로 2.5 m 를 먹고 정지선을 넘는다 (앞범퍼 −1.50 목표에 −0.12 착지).

        여기서는 남은 거리로부터 **단조 감소하는 속도 상한**을 만들어 덧댄다:

            d_stop  = 정지선거리 − s0        (s0 = PDM 과 동일값, 아래 참조)
            v_allow = √(2 · a_stop · d_stop)

        · 단조라 감속 커맨드에 부호 반전이 없다 → jerk 리미터가 되감을 일이
          없다 (재제동 지연이 구조적으로 사라진다).
        · d_stop → 0 에서 v_allow → 0 이라 점근 크립이 아니라 유한 시간 도달.
          목표 0 은 종방향의 a_hold 분기(target<1e-5 ∧ v<0.2)로 자연 접속된다.
        · 멀리서는 v_allow 가 제한속도보다 커서 min() 에 지므로 스스로 비활성
          이다 (별도 발동 거리 상수를 두지 않는 이유).

        s0 는 PDM 에 주입된 idm_red_light_minimum_distance 를 그대로 읽는다 —
        run_agent.build_pdm_config 가 params 의 stop_gap_stopline_m + 앞범퍼로
        채우는 값이라, 여기서 다시 계산하면 단일 출처가 깨진다.
        """
        tgt = self._stop_target(planner, ap)
        if tgt is None:
            return None
        d_line, a_eff = tgt
        if a_eff <= 0.0:
            return None
        # jerk 램프 보정 ([2]) — 스위치가 꺼지면 0 이라 이전 식 그대로다.
        d_eff = d_line - self._s0(ap) - self._jerk_ramp_m(a_eff, ego_speed)
        return _math.sqrt(2.0 * a_eff * max(0.0, d_eff))

    def _stopline_hold(self, planner, ego_speed: float) -> float | None:
        """적신호 정지선 정지의 최소 유지 (speed.stopline_hold_s) — 목표 0 후보.

        규정은 "0.5 s 이상 정지" 다 (실측 0.4 s 재출발이 감점 대상이라 도입).

        **B-1 확정 스펙 (2026-09-01)** — 옛 동작은 적색·근접·저속인 동안 잔여를
        **매 틱 다시 채웠다.** 그래서 녹색이 되는 순간 항상 최대 stopline_hold_s
        만큼 잔여가 남아 출발이 그만큼 늦었다 (실측 잔여 0.30 / 1.00 / 1.10 /
        1.15 s). 지금은:

          1. **연속 정지 중 리필 금지** — 한 번의 정지에 대해 무장은 1회다.
          2. **정지 연속성이 깨질 때만 재무장** — 굴러갔다(latch_v 이상) 다시
             서면 그것은 새 정지이므로 다시 채운다. 적색이 지속되는 동안 목표 0 을
             유지하는 일은 홀드가 아니라 ④′ 프로파일(_stopline_profile)의 몫이다.
          3. **녹색 전환 시 잔여 클리어** — 단 그 정지가 아직 최소 시간
             (stopline_hold_min_s) 을 못 채웠으면 모자란 만큼만 남긴다.

        신호 정보가 없는 환경(목 플래너 등)에서는 개입하지 않는다.
        """
        tgt = self._stop_target(planner, self._ap)
        stopped = ego_speed < self.latch_v
        near = tgt is not None and (tgt[0] - self.front) < self.sl_near_m

        if stopped:
            if near and not self.sl_stopped:        # 이 정지에 대한 1회 무장
                self.sl_stopped = True
                self.sl_stop_ticks = 0
                self.sl_hold_left = self.sl_hold_ticks
        else:                                       # 연속성이 깨졌다 — 다음 정지에 재무장
            self.sl_stopped = False
            self.sl_stop_ticks = 0

        # 정지 대상 소멸(녹색 전환·통과) — 최소 시간 미충족분만 남긴다.
        # sl_stop_ticks 는 **이번 틱을 세기 전** 값이라, 남기는 수가 곧 "앞으로 더
        # 서 있어야 할 틱" 이 된다 (여기서 세고 빼면 1틱 모자란다).
        if tgt is None and self.sl_hold_left > 0:
            self.sl_hold_left = min(self.sl_hold_left,
                                    max(0, self.sl_min_ticks - self.sl_stop_ticks))
        if stopped:
            self.sl_stop_ticks += 1

        if self.sl_hold_left > 0:
            self.sl_hold_left -= 1
            return 0.0
        # 정지 대상이 **아직 살아 있는데** 이미 선 상태면 계속 0 이다 ((C)).
        # 안 그러면 ④′·IDM 이 0.03~0.12 m/s 를 내주어 0.2 m 씩 기어간다 —
        # 실측 20260909_093415: 02 정지 3회 중 2회, 07 5회 중 2회.
        # 녹색이 되면 tgt 가 None 이 되어 저절로 풀린다 (출발 지연 0).
        if self.stop_creep_latch and self.sl_stopped and tgt is not None:
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
        self.last_stop_profile = None
        self.last_red_zone = None
        # standoff 대상은 **매 틱 새로** 정한다 (B-8). 예전에는 _try_overtake 의
        # 회랑 블록에서만 리셋해, 억제 반환·SHIFT_HOLD·span 활성 조기 반환 틱마다
        # 직전 값이 얼어붙었다 (실측 2026-09-03 5로그 전부: standoff_d 39.0/37.2/
        # 33.0/14.3 이 수십 초 유지, 001829/01 은 시프트 직후 24.2 < 25 로 v_allow 0
        # → 시프트를 만들고도 정지).
        self.wait_target_d = None
        self.standoff_id = None
        self.standoff_half_len = None
        self.fg_dropped = 0
        self.last_curv = None
        self.last_curv_info = None
        self.last_lc_cap = None
        self.last_cap_binds = False
        self.last_flash = None
        self.last_lane_map = None
        self._creep_diag = None
        self.gap_v_req = None
        self.last_d_end = d_end
        self._ap = ap
        self.last_yellow = None
        self.last_ped = None
        self.ped_emergency = False
        # 황색 원샷 판정 — 프로파일·홀드보다 먼저 정해야 같은 틱에 반영된다
        self._yellow_latch(planner, ego_speed, ap)
        # 차로 지도 (커밋 A) — 읽기 전용. 스위치가 꺼져 있으면 None 이라
        # 진단 키도 안 생긴다 (off 는 로그까지 이전과 동일해야 지문이 성립한다).
        self.last_lane_map = self.lane_map(ap, planner)
        # 적색 점멸 일시정지 시계 (B3) — 같은 신호 축이라 여기서 같이 돈다.
        # 세우는 것은 PDM 의 적신호 IDM 이고, 이 시계는 "충분히 섰나" 만 본다.
        self._flash_tick(planner, ego_speed)
        # 녹색 연속 틱 (C-3 큐 해제 기준). 신호 id 가 바뀌면 0.
        nxt = self._next_stopline(planner)
        tl_id, state = (nxt[2], nxt[1]) if nxt else (None, None)
        if tl_id != self.green_tl_id:
            self.green_since_ticks = 0
            self.green_tl_id = tl_id
        self.green_since_ticks = self.green_since_ticks + 1 if state == 'Green' else 0

        # 정적 장애물 회피 — 경로를 밀면 PDM 의 선행차 판정에서 빠져 다시 달린다.
        self.last_avoid = None
        red_pause = self._red_pause(planner) is not None   # E-7: 거리 상한 적용
        self._update_obj_timers(ap, paused=red_pause)
        # 틱당 1회 캐시 (C-2) — standoff 축 회랑과 큐 판정. q_ticks 는 **여기서만**
        # 증가한다: _try_overtake 와 _obstacle_cause 가 각자 _is_queue 를 부르면
        # 틱당 두 번 세어 해제 시한이 절반이 된다. legacy 는 계산하지 않는다
        # (그쪽은 _try_overtake 안에서 옛 위치·옛 횟수로 부른다).
        self._tick_cache(ap, planner)
        # A2 시계 — _tick_cache 뒤(큐 판정 필요), _breakout_tick 앞(단계가 그 안에서
        # CREEP_FAIL 재진입을 여는 입력이다).
        self._never_stall_tick(planner, ap, ego_speed)
        self._signal_timeout_tick(ap, planner, ego_speed)
        self._rtor_tick(ap, planner, ego_speed)
        if self.bo_enabled:
            self._breakout_tick(planner, ap, ego_speed)
        # 차로 지도 결정 (커밋 B) — 시프트 방향 힌트로 쓰이므로 _try_overtake 앞이다.
        # 스위치가 꺼져 있으면 None 이라 아무 것도 안 바뀐다 (진단 키도 안 생긴다).
        self.last_lane_plan = None
        self.lane_plan(ap, planner)
        # (지시등은 lat_shift 를 보므로 시프트를 자동으로 따라온다)
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

        # WAIT/관찰 감속 — standoff 앞에 서도록 하는 속도 상한 (④′ 형태)
        so = self._standoff_profile(ego_speed)
        if so is not None and (candidate is None or so < candidate):
            candidate = so
        if so is not None:
            self.last_avoid = dict(self.last_avoid or {'state': 'STANDOFF'},
                                   standoff_d=round(float(self.wait_target_d), 1),
                                   standoff_id=self.standoff_id, standoff_v=round(so, 2),
                                   **(self._creep_diag or {}))

        # 연결로 곡률 상한 (B1) — min() 후보. 오버라이드가 아니라 상한이라
        # 신호·보행자·standoff 가 더 낮으면 그쪽이 이긴다.
        #
        # 곡률·차선변경은 **정상상태 속도 상한**이다 (목표점 0 정지 프로파일이
        # 아니다). 종방향이 이 둘을 IDM 의 "1틱 전방 목표" 로 오해하면 err/dt 가
        # 20배 증폭돼 0.2 m/s 초과에도 −4.0 이 나간다 — 실주행 2차 [1]. 그래서
        # 상한형 후보의 최저값을 따로 들고 있다가, 최종 목표를 이쪽이 정했을 때만
        # 종방향에 cap_target 으로 알린다.
        cap_cand: float | None = None
        cv = self._curvature_cap(planner)
        self.last_curv = cv
        if cv is not None and (candidate is None or cv < candidate):
            candidate = cv
        if cv is not None and (cap_cand is None or cv < cap_cand):
            cap_cand = cv

        # 차선변경 구간 속도 상한 ([3](a)) — 같은 자리의 min() 후보.
        lcv = self._lc_speed_cap(planner)
        if lcv is not None and (candidate is None or lcv < candidate):
            candidate = lcv
        if lcv is not None and (cap_cand is None or lcv < cap_cand):
            cap_cand = lcv

        # 붉은 구간 진입 전 감속 (2b-B) — 구간 **밖**에서만 산다. min() 후보.
        # 이것도 **정상상태 상한**이다: 목표가 0 이 아니라 구간 제한속도 v_allow
        # 라, 곡률·LC 와 같은 실행축에 태운다 (실주행 2차 [1]). 실측 rs 55.0:
        # v_allow 6.79 인데 v 7.71 (err −0.92) 만으로 −4.0 이 나갔고, 구간에
        # 들어가 후보가 사라진 뒤에도 jerk 램프 때문에 10 m 를 더 감속해
        # 7.26 → 4.25 m/s 가 됐다.
        rz = self._red_approach_profile(planner)
        if rz is not None and (candidate is None or rz < candidate):
            candidate = rz
        if rz is not None and (cap_cand is None or rz < cap_cand):
            cap_cand = rz

        # 차로 지도 회피 속도 상한 (커밋 B) — 트리거 즉시 걸어 램프를 짧게 만든다.
        # **상한형이다**: "이 속도를 넘지 마라" 지 "1틱 뒤에 이 속도가 되어라" 가
        # 아니므로 cap 축에 태운다 (CLAUDE.md 확정 사실 — err/dt 금지).
        lmv = (self.last_lane_plan or {}).get('v_cap') if self.last_lane_plan else None
        if lmv is not None and (self.last_lane_plan or {}).get('why') != 'no_candidate':
            lmv = float(lmv)
            if candidate is None or lmv < candidate:
                candidate = lmv
            if cap_cand is None or lmv < cap_cand:
                cap_cand = lmv

        # 적신호 접근 가속 금지 ((C)) — "이 속도를 넘지 마라" 이므로 cap 축이다.
        nar = self._no_accel_red_cap(planner, ap, ego_speed)
        if nar is not None:
            if candidate is None or nar < candidate:
                candidate = nar
            if cap_cand is None or nar < cap_cand:
                cap_cand = nar

        # 시프트 전이 횡가속 상한 (P1) — 진행 중인 회피 시프트에서만 산다.
        cap = self._shift_speed_cap(planner, ego_speed)
        if cap is not None and (candidate is None or cap < candidate):
            candidate = cap
        if cap is not None:
            self.last_avoid = dict(self.last_avoid or {}, shift_cap=round(cap, 2))
        # gap_fit 속도 연동 — 짧은 전이를 만들려면 v ≤ trans / shift_k_s 여야 한다.
        # `_shift_speed_cap` 과 **같은 자리·같은 방식**의 min() 후보다. 오버라이드가
        # 아니라 상한이라, 신호·보행자·standoff 가 더 낮으면 그쪽이 이긴다.
        vr = self.gap_v_req
        if vr is not None and (candidate is None or vr < candidate):
            candidate = vr

        # BREAKOUT 크립 — 훅이 PDM 후보를 무효화한 뒤, 상한은 여전히 min() 이다.
        if self.breakout_creep() and (candidate is None or self.bo_creep_v < candidate):
            candidate = self.bo_creep_v

        # 정지선 정지 프로파일 (④′) — 적색일 때만, min 으로 합류.
        # route_end 는 대상이 아니다 (검증 통과 후 별건).
        prof = self._stopline_profile(planner, ap, ego_speed)
        self.last_stop_profile = prof
        if prof is not None and (candidate is None or prof < candidate):
            candidate = prof

        # 정지선 0.5 s 유지 홀드 — route_end 후보와 min 으로 합류
        hold = self._stopline_hold(planner, ego_speed)
        if hold is not None and (candidate is None or hold < candidate):
            candidate = hold

        # RTOR 진행 상한 — 래치가 살아 있는 동안(리셋까지) min() 후보.
        if self._rtor_go and self.rtor_go_v > 0.0 and (candidate is None or self.rtor_go_v < candidate):
            candidate = self.rtor_go_v

        # 보행자 의도 후보 (P4) — PDM 예측선 교차를 기다리지 않는다.
        ped = self._ped_intent(planner, ap, ego_speed)
        ped_bind = False
        if ped is not None:
            v_allow, a_req, wid = ped
            # '구속' = 이 후보가 min() 의 최저값이다 (동률 포함). 동률까지 세는
            # 이유는 PDM 의 walker 후보가 뒤늦게 같은 값에 도달했을 때도 비상
            # 우회가 이어져야 하기 때문이다.
            ped_bind = candidate is None or v_allow <= candidate + 1e-9
            if candidate is None or v_allow < candidate:
                candidate = v_allow
            self.last_ped = {'id': int(wid), 'v_allow': round(float(v_allow), 2),
                             'a_req': round(float(a_req), 2), 'wins': bool(ped_bind),
                             **self.ped_diag.get(wid, {})}
            if self.ped_released:                       # 다른 id 가 같은 틱에 해제됨
                self.last_ped['released'] = {int(k): v for k, v in self.ped_released.items()}
        elif self.ped_released:
            # 해제 틱 — 후보는 없지만 사유·직전 계측을 남긴다 (A-1 검증용).
            wid, why = next(iter(self.ped_released.items()))
            self.last_ped = {'id': int(wid), 'wins': False, 'release': why,
                             **self.ped_diag.get(wid, {})}
        # P4-M: 평가된 보행자 **전부**를 남긴다 (id 별 종·횡거리·v_allow 기여·래치·홀드).
        # 이전에는 best 1명만 기록돼 다중 보행자 창에서 다른 id 의 상태를 알 수 없었다.
        if self.ped_multi and (self.ped_all or self.ped_hold_ids):
            if self.last_ped is None:
                self.last_ped = {'id': None, 'wins': False}
            self.last_ped['all'] = {int(k): v for k, v in self.ped_all.items()}
            self.last_ped['hold'] = sorted(int(k) for k in self.ped_hold_ids)

        # 횡단보도 앞 서행 (A-3) — 서 있는 보행자 앞 3 s 정지 후 creep 상한. min() 후보.
        cw = self._ped_crosswalk(planner, ap, ego_speed)
        if cw is not None:
            v_cw, info = cw
            if candidate is None or v_cw < candidate:
                candidate = v_cw
            cw_wins = bool(v_cw <= min(target_speed, candidate) + 1e-9)   # 최종 목표를 구속하나
            if self.last_ped is None:
                self.last_ped = {'id': int(info['id']), 'wins': cw_wins, 'crosswalk': info}
            else:
                self.last_ped['crosswalk'] = info

        # 보행자 비상 우회 — **보행자 후보가 최종 목표를 구속하는 틱 한정**이다.
        # 선행차·신호·종점·크립 후보가 이긴 틱에서는 절대 발동하지 않는다.
        final_t = target_speed if candidate is None else min(target_speed, candidate)
        emg = bool(ped_bind and self.ped_emg_ratio > 0.0
                   and ped[1] > self.ped_emg_ratio * self.a_dec_max
                   and final_t <= ped[0] + 1e-9)
        if emg:
            self.ped_emergency = True
            if self.last_ped is not None:
                self.last_ped['emergency'] = True

        if candidate is not None:
            self.last_candidate = candidate
        kr_wins = candidate is not None and candidate < target_speed
        # ── 순수 제한속도도 '상한' 이다 ────────────────────────────────────
        # PDM 의 중재 전 목표(제한속도 ∧ 교차로 상한)를 **아무도 줄이지 않은**
        # 틱이면 최종 목표는 IDM 의 1틱 전방 값이 아니라 그냥 속도 상한이다.
        # 그때도 err/dt 축을 쓰면 0.3 m/s 초과에 −4.0 이 나간다 (실측 rs 55.6~
        # 65.0: 제한 6.94 에 v 7.26 인데 급제동 → 4.25 m/s 까지 밀렸다).
        # 적신호·선행차·정지 프로파일이 이기면 target < initial 이라 여기 안 온다.
        init_t = getattr(ap, 'initial_target', None)
        pure_limit = bool(self.cap_limit and init_t is not None and not emg
                          and not kr_wins and target_speed > 1e-5
                          and abs(float(target_speed) - float(init_t)) < 1e-6)
        if kr_wins or emg or pure_limit:
            if kr_wins:
                target_speed = candidate
            # 최종 목표를 **상한형 후보가 정했나**. 동률이면 상한이 아니라고 본다
            # — 정지 프로파일·보행자·홀드가 같은 값이면 그쪽 실행축(err/dt)이
            # 맞기 때문이다. 안전한 쪽으로 틀린다 (이전 동작).
            cap_binds = bool(not emg and (
                pure_limit
                or (cap_cand is not None and abs(target_speed - cap_cand) < 1e-9)))
            self.last_cap_binds = cap_binds
            # 순수 제한속도인데 이미 목표 이하면 본류 명령과 같다 — 되감아
            # 다시 부르지 않는다 (가속측은 cap 여부와 무관하게 같은 축이라
            # 결과가 같고, 불필요한 rewind 로 jerk 이력을 건드리지 않는다).
            skip = pure_limit and not kr_wins and ego_speed <= target_speed
            if not skip:
                # 종방향 재계산 — 본류가 이번 틱 이미 호출했으므로 되감고 다시
                # (되감지 않으면 두 호출이 jerk 창을 나눠 갖는 핑퐁 — rewind_last)
                hazard = target_speed < 1e-5
                ap._longitudinal_controller.rewind_last()
                if emg:
                    accel, brake = ap._longitudinal_controller.emergency()
                else:
                    accel, brake = ap._longitudinal_controller.get_throttle_and_brake(
                        hazard, target_speed, ego_speed, cap_target=cap_binds)
                control.accel = accel
                control.throttle = accel
                control.brake = float(brake)

        if self.q_reject:
            self.last_avoid = dict(self.last_avoid or {}, queue_reject=self.q_reject)
        if self.last_lc_cap is not None:
            self.last_avoid = dict(self.last_avoid or {},
                                   lc_cap=round(self.last_lc_cap, 2))
        if self.last_lane_map is not None:
            self.last_avoid = dict(self.last_avoid or {}, lane_map=self.last_lane_map)
        if self.last_flash is not None:
            # 스위치가 꺼져 있으면 항상 None 이라 키가 안 생긴다 — off 는 로그까지
            # 이전과 동일해야 회귀 비교(지문)가 성립한다.
            self.last_avoid = dict(self.last_avoid or {}, flash=self.last_flash)
        if self.last_curv_info is not None:
            # 스위치가 꺼져 있으면 항상 None 이라 키가 안 생긴다 — off 는 로그까지
            # 이전과 동일해야 회귀 비교(51 지문)가 성립한다.
            self.last_avoid = dict(self.last_avoid or {}, curv=self.last_curv_info)
        if self.last_lane_plan is not None:
            # 스위치가 꺼져 있으면 항상 None 이라 키가 안 생긴다 — off 는 로그까지
            # 이전과 동일해야 회귀 비교(58 지문)가 성립한다.
            self.last_avoid = dict(self.last_avoid or {},
                                   lane_plan=self.last_lane_plan)
        if self.fg_dropped:
            # 스위치가 꺼져 있으면 항상 0 이라 키가 안 생긴다 — off 는 로그까지
            # 이전과 동일해야 회귀 비교(51 지문)가 성립한다.
            self.last_avoid = dict(self.last_avoid or {},
                                   finish_gate={'dropped': int(self.fg_dropped),
                                                'finish_s': round(self.finish_s, 1)})
        if self.ns_info:
            # A2 진단. 스위치가 꺼져 있으면 ns_info 가 항상 None 이라 키가 안 생긴다
            # — off 는 로그까지 이전과 동일해야 회귀 비교(51 지문)가 성립한다.
            self.last_avoid = dict(self.last_avoid or {}, never_stall=self.ns_info)
        if self.bo_state is not None:
            self.last_avoid = dict(self.last_avoid or {}, **{
                'state': self.bo_state, 'level': self.bo_level,
                'paused': self.bo_paused,
                'stall_s': round(self.bo_stall_ticks / self.hz, 1),
                'creep': self.breakout_creep(),
                # L2 이상은 감점 가능한 완화다 — 단계와 사유를 반드시 남긴다
                # (B-2: 단계는 zone·geom 완화에 쓴다. 실선은 두 바퀴(B-3)가 본다)
                'relax': self._relax_label()})
        elif self.bo_exit:
            self.last_avoid = dict(self.last_avoid or {}, exit=self.bo_exit)
            self.bo_exit = None

        self.last_target = float(target_speed)
        return control, target_speed
