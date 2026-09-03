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
        # 황색 딜레마 원샷 판정 (C). 0 이면 비활성 = 황색을 PDM 원문에만 맡긴다.
        self.a_yellow = float(sp.get('a_yellow', 0.0))
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

        self.latched = False
        self.stop_s: float | None = None           # 시작 시 1회 계산 캐시 (매 틱 투영 금지)
        self.sl_hold_left = 0                      # 정지선 홀드 잔여 틱
        self.sl_stopped = False                    # 정지 연속성 (B-1 재무장 판정)
        self.sl_stop_ticks = 0                     # 현재 정지의 지속 틱
        self.last_candidate: float | None = None   # 이번 틱 route_end 후보 (로그용)
        self.last_target: float | None = None      # 이번 틱 최종 목표속도 (로그용)
        self.last_d_end: float | None = None
        self.last_stop_profile: float | None = None   # 이번 틱 정지 프로파일 상한 (로그용)
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
        self._sl_all: list | None = None           # 경로상 전 정지선 route_s (1회)
        # BREAKOUT 상태
        self.bo_state: str | None = None           # None | 'BREAKOUT' | 'CREEP_FAIL'
        self.bo_level = 0
        self.bo_stuck_ticks = 0
        self.bo_stop_ticks = 0                     # 순수 정지(v<eps) 지속 틱 (E-3 안전 가드)
        self.ot_reject_ticks = 0                   # 회피 시도가 양쪽 다 기각된 연속 틱 (E-3)
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
        for iv in self.sig_plan:
            if route_s > iv['end_s']:
                continue
            remain = iv['ev_s'] - route_s
            if remain > max(ego_speed * self.turn_lead_s, self.sig_lead_min_m):
                continue
            key = (max(0.0, remain), 0)              # 0 = 회전 우선
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

    def _corridor_blockers(self, ap, planner, static_ok=None):
        """전방 detect_max_m 안에서 **주행 회랑을 침범한 정지 객체** 목록.

        반환: [(s_rel, lat, half_w, actor)] — s_rel 은 자차 기준 전방거리.
        침범 판정은 선행차 판정([route.py] compute_leading_vehicles) 과 같은 축:
        |lat| < 자차반폭 + 객체반폭 + obstacle_clearance_m.
        static_ok: '정지' 판정 함수. 기본 _static_ok(정지 관찰 3 s, 신호 중 일시정지).
        standoff 대상은 _stop_ok(신호 무관 1.5 s) 로 따로 뽑는다 (B-5).
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
        out = []
        for a in actors:
            if a.id == ego_id or not static_ok(a):
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
            if abs(lat) < half_ego + hw + clr:
                out.append((s_rel, lat, hw, a))
        out.sort(key=lambda z: z[0])
        return out

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
        sig_hold = (state in ('Red', 'Yellow')
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
    def _obstacle_cause(self, planner, ap) -> bool:
        """지금 정지 원인이 **장애물 계열**인가. 하나라도 아니면 거짓.

        BREAKOUT 은 제약을 풀고 전진을 강제하므로, 원인이 신호·보행자·종점
        이면 **절대 발동하면 안 된다**. PDM 이 매 틱 세우는 hazard 플래그와
        kr_rules 자신의 래치를 모두 본다.
        """
        if getattr(ap, 'traffic_light_hazard', False):
            return False
        if getattr(ap, 'walker_hazard', False) or getattr(ap, 'walker_close', False):
            return False
        if getattr(ap, 'stop_sign_hazard', False):
            return False
        if self.latched or self.sl_hold_left > 0:          # 종점 래치 / 정지선 홀드
            return False
        if self.y_decision is not None or self.cross_guard:  # 황색 래치 / 통과 가드
            return False
        if self.last_d_end is not None and self.last_d_end <= self.active_m:
            return False                                   # route_end 유령차 사정권
        if self.suppress_mode == 'legacy':
            if self._red_ahead(planner) is not None:           # 절대 규칙 (신호 > 회피)
                return False
            if self._signal_zone(planner, ap) is not None:     # 규칙 1
                return False
        elif self._tick_queue:                             # 큐 뒤에 선 것은 데드락이 아니다 (C-5)
            return False
        if self._blocker(ap, planner) is not None:         # 실제로 앞이 막혀 있을 것
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
        """
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
        if self.suppress_mode != 'legacy' and self._in_junction_lane(ap):
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
        if rejected and self.last_avoid is not None:
            self.last_avoid['reject_s'] = round(self.ot_reject_ticks / self.hz, 1)

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
            if self.span_active_standoff:
                corridor = self._corridor_blockers(ap, planner)
                self._standoff_target(ap, planner, corridor)
                self._blocked_account(ap, planner, ego_speed)
                self.last_avoid = {'state': 'SHIFT_ACTIVE', 'span': list(self.ot_span),
                                   'blocker': corridor[0][3].id if corridor else None,
                                   's_rel': round(corridor[0][0], 1) if corridor else None}
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

    def _restore_span(self, planner) -> None:
        """지나간 시프트 span 원복 (다음 장애물용) — E-6 으로 호출처가 둘이 됐다."""
        a, b = self.ot_span
        planner.route_points[a:b] = planner.original_route_points[a:b]
        planner.commands[a:b] = planner.commands_orig[a:b]
        planner.lat_shift[a:b] = planner._lat_build[a:b]
        planner._kd = _cKDTree(planner.route_points[:, :2])
        self.ot_span = None
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
            return
        if self.standoff_stop_ticks > 0:
            so_objs = self._corridor_blockers(ap, planner, static_ok=self._stop_ok)
            if so_objs:
                self.wait_target_d = so_objs[0][0]
                self.standoff_id = so_objs[0][3].id
        elif corridor:
            self.wait_target_d = corridor[0][0]

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
        for side in ('left', 'right'):                     # 좌측 추월 우선
            def reject(gate, **extra):
                self.last_overtake = f'{side}:{gate}@p{n_pass}'
                la = self.last_avoid if self.last_avoid is not None else {}
                la.update({'reject': f'{side}:{gate}', 'pass': n_pass, **extra})
                # 양쪽·두 바퀴의 기각을 전부 남긴다 — 최종 라벨은 마지막 side 것뿐이라
                # (C-8 로 no_neighbor 도 라벨이 붙어) 앞선 사유가 가려진다.
                la.setdefault('rejects', []).append(f'{side}:{gate}@p{n_pass}')

            target = lg.neighbor(ego_lane, side)
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
            span = planner.shift_route_around_actors(
                actor, chain_last,
                obstacle_direction='right' if side == 'left' else 'left',
                transition_length=trans_m * ppm,
                extra_length_before=self.ot_before_m * ppm,
                extra_length_after=extra_after * ppm,      # E-2 연장 포함
                # 전이 시작을 자차 **앞**으로 — 뒤에서 시작하면 현재 위치의
                # 경로가 옆으로 밀려 정지 상태에서 조향이 풀락된다
                min_start_ahead=ahead_m * ppm)
            planner._kd = _cKDTree(planner.route_points[:, :2])
            self.ot_span = span
            self.ot_blocked_ticks = 0
            self.preempt_latch_id = None                   # E-4 래치 해제 (시프트 성공)
            self.last_overtake = side
            (self.last_avoid or {}).update(
                {'shift': side, 'span': list(span), 'trans_m': round(trans_m, 1),
                 'ahead_m': round(ahead_m, 1), 'preempt': preempt, 'pass': n_pass,
                 'solid_relaxed': skip_solid, 'chain': list(chain['ids']),
                 'span_m': round(span_m, 1), 'after_m': round(extra_after, 1)})
            print(f'[kr_rules] 정적 장애물 회피 — {side} 로 경로 시프트 '
                  f'(id={chain["ids"]}, 구간 {span[0]}~{span[1]}, p{n_pass})', flush=True)
            return True
        return False

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
        v_allow = _math.sqrt(2.0 * self.a_yellow * max(0.0, d_line - self._s0(ap)))
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

    def _s0(self, ap) -> float:
        """계획 정지점의 뒷축 gap — PDM 주입값이 단일 출처."""
        return float(getattr(ap.config, 'idm_red_light_minimum_distance',
                             self.stop_gap_sl_fallback))

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
        """
        return bool(self.y_decision == 'go' or self.cross_guard)

    def _stop_target(self, planner, ap) -> tuple | None:
        """정지 후보를 만들 대상이면 (뒷축거리, 실행 감속 a_eff), 아니면 None.

        색 해석의 **단일 출처**다 — 프로파일과 홀드가 같은 판정을 본다.
          · 적색            → (d, stop_profile_a)
          · 황색 + STOP 래치 → (d, stop_profile_a)  적색과 **완전히 동일** 취급
          · 황색 + GO 래치   → None
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
            if wid not in self.ped_intent:
                if v_toward is None:
                    continue
                w_speed = float(getattr(w, 'speed', 0.0))
                if wid in self.ped_static:
                    if w_speed < self.ped_intent_v or v_toward < self.ped_intent_v:
                        continue
                elif not self._ped_walkin(wid, w_speed, v_toward, lat):
                    continue
                self.ped_intent.add(wid)
                self.ped_clear[wid] = 0
                self.ped_hold[wid] = 0
                self.ped_walkin.pop(wid, None)
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
            v_allow = _math.sqrt(2.0 * self.stop_profile_a * max(0.0, d_eff))
            # 분모 하한 0.5 m — d_eff→0 에서 a_req 가 발산해 로그가 못 쓰게 된다.
            # 판정(임계 초과 여부)에는 영향이 없다: 하한을 써도 이미 임계 위다.
            a_req = ego_speed * ego_speed / (2.0 * max(d_eff, 0.5))
            if best is None or v_allow < best[0]:
                best = (v_allow, a_req, wid)
        # 관측이 끊긴 id 정리 (obj_ticks 의 grace 와 별개 — 여기선 즉시)
        for wid in list(self.ped_intent):
            if wid not in live:
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
        """래치 해제 + 해제 카운터 정리 (지나감·끊김·위치 해제 공통)."""
        self.ped_intent.discard(wid)
        self.ped_clear.pop(wid, None)
        self.ped_hold.pop(wid, None)

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
            standoff = max(shift_latest_m, shift_k_s · v)
        IDM 의 정지 gap 과 충돌하지 않는다 — 둘 다 상한이고 min() 이 낮은 쪽을 쓴다.
        """
        if self.wait_target_d is None or self.stop_profile_a <= 0.0:
            return None
        standoff = max(self.shift_latest_m, self.shift_k_s * max(ego_speed, 0.1))
        d = self.wait_target_d - standoff
        return _math.sqrt(2.0 * self.stop_profile_a * max(0.0, d))

    def _stopline_profile(self, planner, ap) -> float | None:
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
        return _math.sqrt(2.0 * a_eff * max(0.0, d_line - self._s0(ap)))

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
        # standoff 대상은 **매 틱 새로** 정한다 (B-8). 예전에는 _try_overtake 의
        # 회랑 블록에서만 리셋해, 억제 반환·SHIFT_HOLD·span 활성 조기 반환 틱마다
        # 직전 값이 얼어붙었다 (실측 2026-09-03 5로그 전부: standoff_d 39.0/37.2/
        # 33.0/14.3 이 수십 초 유지, 001829/01 은 시프트 직후 24.2 < 25 로 v_allow 0
        # → 시프트를 만들고도 정지).
        self.wait_target_d = None
        self.standoff_id = None
        self.last_d_end = d_end
        self._ap = ap
        self.last_yellow = None
        self.last_ped = None
        self.ped_emergency = False
        # 황색 원샷 판정 — 프로파일·홀드보다 먼저 정해야 같은 틱에 반영된다
        self._yellow_latch(planner, ego_speed, ap)
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
        if self.bo_enabled:
            self._breakout_tick(planner, ap, ego_speed)
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
                                   standoff_id=self.standoff_id, standoff_v=round(so, 2))

        # 시프트 전이 횡가속 상한 (P1) — 진행 중인 회피 시프트에서만 산다.
        cap = self._shift_speed_cap(planner, ego_speed)
        if cap is not None and (candidate is None or cap < candidate):
            candidate = cap
        if cap is not None:
            self.last_avoid = dict(self.last_avoid or {}, shift_cap=round(cap, 2))

        # BREAKOUT 크립 — 훅이 PDM 후보를 무효화한 뒤, 상한은 여전히 min() 이다.
        if self.breakout_creep() and (candidate is None or self.bo_creep_v < candidate):
            candidate = self.bo_creep_v

        # 정지선 정지 프로파일 (④′) — 적색일 때만, min 으로 합류.
        # route_end 는 대상이 아니다 (검증 통과 후 별건).
        prof = self._stopline_profile(planner, ap)
        self.last_stop_profile = prof
        if prof is not None and (candidate is None or prof < candidate):
            candidate = prof

        # 정지선 0.5 s 유지 홀드 — route_end 후보와 min 으로 합류
        hold = self._stopline_hold(planner, ego_speed)
        if hold is not None and (candidate is None or hold < candidate):
            candidate = hold

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
        if (candidate is not None and candidate < target_speed) or emg:
            if candidate is not None and candidate < target_speed:
                target_speed = candidate
            # 종방향 재계산 — 본류가 이번 틱 이미 호출했으므로 되감고 다시
            # (되감지 않으면 두 호출이 jerk 창을 나눠 갖는 핑퐁 — rewind_last 참고)
            hazard = target_speed < 1e-5
            ap._longitudinal_controller.rewind_last()
            if emg:
                accel, brake = ap._longitudinal_controller.emergency()
            else:
                accel, brake = ap._longitudinal_controller.get_throttle_and_brake(
                    hazard, target_speed, ego_speed)
            control.accel = accel
            control.throttle = accel
            control.brake = float(brake)

        if self.q_reject:
            self.last_avoid = dict(self.last_avoid or {}, queue_reject=self.q_reject)
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
