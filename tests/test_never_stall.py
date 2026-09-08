"""
A2 — 절대 안 멈춤 (never_stall_enable).

작업1 전수에서 **상한이 없는 정지 조건 6가지**가 남아 있었다:
  ① _standoff_creep block('stop_gap') — 진짜 정지 거리 안. 실측 33.3 s
     (logs/run_20260907_154131 t=52.1~85.4, d=2.4 m, reject=right:no_neighbor)
  ② BREAKOUT CREEP_FAIL — 탈출이 'cause_gone' 하나뿐이라 장애물이 안 치워지면 영구
  ③ block('no_size') — 객체 반길이 미보고
  ④ 모드 A — d 가 standoff_floor_m 에 얹혀 _standoff_creep 미호출 (시뮬)
  ⑤ 시프트 전 사유 기각 (no_neighbor·solid·occupied·geom·zone·kappa)
  ⑥ UNKNOWN 큐 — _is_queue_v2 가 "해제 시한이 없다" 고 명시. 같은 상황의 다른
     안전망 _signal_timeout_tick 은 `not _tick_corridor` 를 요구해 큐에서는 시계가
     안 돈다 — 둘이 동시에 죽는다.

여기서 지키는 불변:
  · 시계는 **하나**다. 원인별 시계를 늘리지 않는다.
  · 배제 목록도 **하나**다 — `_ns_cause` 는 `_obstacle_cause` 를 그대로 쓰고
    UNKNOWN 큐만 넓힌다. 적신호·보행자·정지표지·종점은 어떤 단계에서도 안 푼다.
  · 1·2단계는 min() **상한**만 올린다 (강제 전진이 아니다). PDM 의 IDM·OBB 를
    무효화하는 것은 3단계뿐이고, 그래서 속도가 never_stall_force_v 로 묶인다.
  · (c) 회전 차로 복귀 보류에도 **시한이 있다** — never_stall 이 새로운 무한
    대기를 만들면 안 된다.
"""
import copy
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402
from test_avoid import Ap, Box, GeomPlanner, HZ, LgOne             # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
MAX_TICKS = int(round(OT['deadlock_max_s'] * HZ))
STEP_TICKS = int(round(OT['never_stall_step_s'] * HZ))
TURN_HOLD_TICKS = int(round(OT['never_stall_turn_hold_s'] * HZ))
NS_CREEP = OT['never_stall_creep_kph'] / 3.6
FORCE_V = OT['never_stall_force_v']
TURN_MARGIN = OT['never_stall_turn_margin_m']
# 실측 고착과 같은 배치 — 자차가 이미 d_stop(≈4.87 m) 안이라 standoff 는 stop_gap
STUCK_X = 4.0


def on_cfg(**over):
    c = copy.deepcopy(CFG)
    c['overtake']['never_stall_enable'] = True
    c['overtake'].update(over)
    return c


def off_cfg(**over):
    """이전 동작 사본. **기본값을 읽지 않는다** — params 가 true 로 바뀌어도
    off 경로 커버리지를 잃지 않기 위해서다 (CLAUDE.md 2026-09-07 드리프트 원칙).
    params 값이 정본이고 여기서는 끄기만 한다."""
    c = copy.deepcopy(CFG)
    c['overtake']['never_stall_enable'] = False
    c['overtake']['junction_creep_release_enable'] = False
    c['overtake'].update(over)
    return c


class LgBlocked(LgOne):
    """양쪽 다 이웃 없음 — 시프트가 어느 쪽으로도 불가 (⑤)."""

    def neighbor(self, key, side):
        return None


class LgJunction(LgOne):
    def __init__(self):
        super().__init__()
        self.lanes = {k: {'junction': 8} for k in self.lanes}


def rig(cfg=None, lg=None, obj_x=STUCK_X, route=None):
    cfg = CFG if cfg is None else cfg
    p = GeomPlanner(d_tl=float('inf'))
    p.lg = LgBlocked() if lg is None else lg
    if route is not None:
        p.route = route
    kr = KrRules(cfg)
    kr._sl_all = []
    ap = Ap(p, actors=[Box(2, obj_x, 0.0, 0.0, half_w=0.9)])
    ap._kr_ego_lane = (1, 0, -1)
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    kr.last_d_end = 1e6
    kr._ap = ap
    for _ in range(int(cfg['overtake']['wait_before_shift_s'] * HZ) + 5):
        kr._update_obj_timers(ap)
    return kr, p, ap


def step(kr, p, ap, n=1, v=0.0):
    """apply 와 같은 순서: 캐시 → ns 시계 → 사다리 → 회피 → standoff."""
    so = None
    for _ in range(n):
        kr._update_obj_timers(ap)
        kr._tick_cache(ap, p)
        kr._never_stall_tick(p, ap, v)
        kr._breakout_tick(p, ap, v)
        kr._try_overtake(ap, p, v)
        so = kr._standoff_profile(v)
    return so


# ── 시계와 단계 ────────────────────────────────────────────────────────────
def test_off_never_arms_and_leaves_no_log_key():
    """51 지문 회귀의 근거 — off 는 시계도 안 돌고 로그 키도 안 생긴다."""
    kr, p, ap = rig(off_cfg())
    so = step(kr, p, ap, MAX_TICKS + 3 * STEP_TICKS + 20)
    assert kr.ns_level == 0 and kr.ns_ticks == 0 and kr.ns_info is None
    assert so == pytest.approx(0.0)                    # 이전 동작 그대로 영구 정지
    assert (kr._creep_diag or {}).get('creep_block') == 'stop_gap'


def test_levels_reach_1_2_3_at_expected_times():
    kr, p, ap = rig(on_cfg())
    step(kr, p, ap, MAX_TICKS - 1)
    assert kr.ns_level == 0
    step(kr, p, ap, 1)
    assert kr.ns_level == 1
    step(kr, p, ap, STEP_TICKS)
    assert kr.ns_level == 2
    step(kr, p, ap, STEP_TICKS)
    assert kr.ns_level == 3
    step(kr, p, ap, STEP_TICKS * 3)
    assert kr.ns_level == 3                            # 3 이 상한


def test_progress_resets_the_clock():
    """조금씩이라도 가고 있으면 stall 이 아니다 — 기준을 옮기고 0 으로."""
    kr, p, ap = rig(on_cfg())
    step(kr, p, ap, MAX_TICKS - 20)
    assert kr.ns_ticks > 0
    p.route_index += int(OT['progress_m'] * p.points_per_meter) + 1
    step(kr, p, ap, 1)
    assert kr.ns_ticks == 0 and kr.ns_level == 0


def test_on_reset_clears_the_clock():
    kr, p, ap = rig(on_cfg())
    step(kr, p, ap, MAX_TICKS + 5)
    kr.on_reset()
    assert (kr.ns_ticks, kr.ns_level, kr.ns_ref_s, kr.ns_turn_ticks) == (0, 0, None, 0)


# ── 배제: 절대 풀지 않는 것 ────────────────────────────────────────────────
@pytest.mark.parametrize('flag', ['traffic_light_hazard', 'walker_hazard',
                                  'walker_close', 'stop_sign_hazard'])
def test_hazards_keep_the_clock_at_zero(flag):
    """신호·보행자·정지표지는 어떤 단계에서도 풀지 않는다 — 시계 자체가 안 돈다."""
    kr, p, ap = rig(on_cfg())
    setattr(ap, flag, True)
    step(kr, p, ap, MAX_TICKS + 3 * STEP_TICKS + 20)
    assert kr.ns_level == 0 and kr.ns_ticks == 0


def test_route_end_latch_keeps_the_clock_at_zero():
    kr, p, ap = rig(on_cfg())
    kr.last_d_end = 1.0                                # 종점 래치 사정권
    step(kr, p, ap, MAX_TICKS + 20)
    assert kr.ns_level == 0


def test_route_end_exclusion_is_the_latch_window_not_active_m():
    """active_m(150)은 유령차 **후보 창**이지 "종점이 세웠다" 가 아니다.

    그대로 쓰면 경로 마지막 150 m 에서 never_stall 이 영영 무장하지 못한다
    (실측: 정적회피집중 계열은 경로가 짧아 정지 구간 대부분이 그 안이다).
    실제로 세우는 것은 래치이므로 그 축의 unlatch_m 까지만 제외한다.
    BREAKOUT 이 쓰는 _obstacle_cause 의 기본값은 건드리지 않는다.
    """
    end = CFG['route_end']
    kr, p, ap = rig(on_cfg())
    step(kr, p, ap, 5)

    kr.last_d_end = end['active_m'] - 10.0             # 150 창 안, 래치 밖
    assert kr._obstacle_cause(p, ap) is False          # BREAKOUT 은 옛 동작 그대로
    assert kr._ns_cause(p, ap) is True                 # never_stall 은 여기서 산다

    kr.last_d_end = end['unlatch_m'] - 1.0             # 래치 사정권
    assert kr._ns_cause(p, ap) is False                # 종점 정지는 그대로 지킨다


def test_clock_runs_in_the_last_100m_of_the_route():
    kr, p, ap = rig(on_cfg())
    kr.last_d_end = 100.0
    step(kr, p, ap, MAX_TICKS)
    assert kr.ns_level == 1


def test_ped_hold_still_blocks_creep_at_every_level():
    """보행자 홀드는 3단계에서도 유지된다."""
    kr, p, ap = rig(on_cfg())
    step(kr, p, ap, MAX_TICKS + 2 * STEP_TICKS + 5)
    assert kr.ns_level == 3
    kr.ped_hold_ids.add(99)
    so = step(kr, p, ap, 1)
    assert so == pytest.approx(0.0)
    assert (kr._creep_diag or {}).get('creep_block') == 'ped_hold'


# ── PDM 차량·OBB hazard 도 원인으로 본다 ─────────────────────────────────
def test_pdm_vehicle_hazard_counts_as_obstacle_cause():
    """시프트 span 이 활성이면 kr 회랑이 비어 있어 자기 힘으로는 원인을 못 본다.

    실측 replay(정적회피집중_01_좌회전2): 목표 0 인 489틱 **전부**
    vehicle_hazard=True 인데 blocker=False · reject_pending=False 라
    _obstacle_cause 가 거짓이었고, 24 s 정지에 아무 안전망도 안 걸렸다.
    """
    kr, p, ap = rig(on_cfg(), obj_x=200.0)             # 회랑 안에 아무것도 없다
    step(kr, p, ap, 5)
    kr.ot_span = (0, 10)                               # SHIFT_ACTIVE
    kr.ot_reject_ticks = 0
    ap.vehicle_hazard = True
    assert kr._blocker(ap, p) is None
    assert kr._obstacle_cause(p, ap) is False          # BREAKOUT 은 옛 동작 그대로
    assert kr._ns_cause(p, ap) is True


def test_pdm_hazard_does_not_override_protected_causes():
    """보호 배제를 통과한 뒤에만 본다 — 신호·보행자가 서 있으면 여전히 거짓."""
    kr, p, ap = rig(on_cfg(), obj_x=200.0)
    step(kr, p, ap, 5)
    ap.vehicle_hazard = True
    for flag in ('traffic_light_hazard', 'walker_hazard', 'stop_sign_hazard'):
        setattr(ap, flag, True)
        assert kr._ns_cause(p, ap) is False, flag
        setattr(ap, flag, False)
    kr.y_decision = 'stop'                             # 황색 래치 (실측 339틱)
    assert kr._ns_cause(p, ap) is False
    kr.y_decision = None
    kr.latched = True
    assert kr._ns_cause(p, ap) is False


# ── ⑥ UNKNOWN 큐만 넓힌다 ─────────────────────────────────────────────────
def test_unknown_queue_is_counted_but_reported_red_queue_is_not():
    kr, p, ap = rig(on_cfg())
    step(kr, p, ap, 5)
    kr._tick_queue = True
    kr.q_info = {'queue_by_unknown': True}
    assert kr._obstacle_cause(p, ap) is False          # BREAKOUT 은 여전히 제외
    assert kr._ns_cause(p, ap) is True                 # never_stall 만 시한을 준다
    kr.q_info = {'cond': 'B'}                          # 보고된 적색 큐
    assert kr._ns_cause(p, ap) is False


def test_obstacle_cause_ignore_queue_defaults_to_old_behaviour():
    kr, p, ap = rig(on_cfg())
    step(kr, p, ap, 5)
    kr._tick_queue = True
    kr.q_info = {'queue_by_unknown': True}
    assert kr._obstacle_cause(p, ap) is False
    assert kr._obstacle_cause(p, ap, ignore_queue=True) is True


# ── 1단계: stop_gap · no_size 완화 ────────────────────────────────────────
def test_level1_relaxes_stop_gap():
    kr, p, ap = rig(on_cfg())
    assert step(kr, p, ap, MAX_TICKS - 1) == pytest.approx(0.0)
    assert (kr._creep_diag or {}).get('creep_block') == 'stop_gap'
    so = step(kr, p, ap, 1)
    assert so >= NS_CREEP
    d = kr._creep_diag or {}
    assert d.get('creep_ns_relax') == 'stop_gap' and d.get('creep_ns_lvl') == 1
    assert d.get('creep_open_why') == 'never_stall'


def test_level1_relaxes_no_size():
    """객체 크기가 안 오면 정지 거리를 못 구해 0 으로 굳는다 — 1단계가 푼다.

    half_len 은 매 틱 _standoff_target 이 다시 채우므로, 그 값만 지운 상태로
    _standoff_creep 을 직접 부른다 (게이트만 보는 검사다).
    """
    kr, p, ap = rig(on_cfg())
    step(kr, p, ap, MAX_TICKS)
    assert kr.ns_level == 1
    kr.standoff_half_len = None
    assert kr._standoff_creep(kr.standoff_floor_m, 0.0) >= NS_CREEP
    assert (kr._creep_diag or {}).get('creep_ns_relax') == 'no_size'

    kr2, p2, ap2 = rig(off_cfg())                            # off 는 이전 동작
    step(kr2, p2, ap2, MAX_TICKS)
    kr2.standoff_half_len = None
    assert kr2._standoff_creep(kr2.standoff_floor_m, 0.0) == pytest.approx(0.0)
    assert (kr2._creep_diag or {}).get('creep_block') == 'no_size'


def test_level3_uses_force_speed():
    kr, p, ap = rig(on_cfg())
    so = step(kr, p, ap, MAX_TICKS + 2 * STEP_TICKS + 5)
    assert kr.ns_level == 3
    assert so == pytest.approx(FORCE_V)


# ── 2단계: CREEP_FAIL 재진입 ─────────────────────────────────────────────
def test_level2_reopens_creep_fail():
    kr, p, ap = rig(on_cfg())
    step(kr, p, ap, MAX_TICKS + 1)
    assert kr.bo_state == 'CREEP_FAIL'                 # 1단계까지는 갇힌 채다
    step(kr, p, ap, STEP_TICKS)
    assert kr.ns_level == 2 and kr.bo_state == 'BREAKOUT'


def test_creep_fail_stays_terminal_when_off():
    kr, p, ap = rig(off_cfg())
    step(kr, p, ap, MAX_TICKS + 3 * STEP_TICKS + 20)
    assert kr.bo_state == 'CREEP_FAIL'                 # 이전 동작 — 탈출구 없음


# ── 3단계: 훅 강제 개방 ──────────────────────────────────────────────────
def test_level3_forces_the_creep_hook_even_inside_a_junction():
    """교차로에서는 A1 이 훅을 닫지만, 3단계는 그보다 우선한다 (최후 수단)."""
    kr, p, ap = rig(on_cfg(junction_creep_release_enable=True), lg=LgJunction())
    step(kr, p, ap, MAX_TICKS + STEP_TICKS + 5)
    assert kr.ns_level == 2 and kr.breakout_creep() is False
    step(kr, p, ap, STEP_TICKS)
    assert kr.ns_level == 3 and kr.breakout_creep() is True


def test_hook_stays_shut_below_level3():
    kr, p, ap = rig(on_cfg(junction_creep_release_enable=False), lg=LgJunction())
    step(kr, p, ap, MAX_TICKS + 1)
    assert kr.ns_level == 1 and kr.breakout_creep() is False


# ── (c) 회전 차로 복귀 우선 ──────────────────────────────────────────────
def route_turn(left_m=18.0, lanes=((1, 0, -2),), target='pair'):
    """교차로가 left_m 앞. 유효 진입 차로가 lanes 이고 자차 (1,0,-1) 은 거기 없다."""
    return {'lanes': [], 'cum_s': [], 'total_length': 600.0,
            'waypoint_s': [0.0, float(left_m), 120.0],
            'valid_entry_lanes': [{'seg': 0, 'target': target, 'turn': 'left',
                                   'lanes': [list(k) for k in lanes],
                                   'chosen': [1, 0, -1], 'in_set': False}]}


def test_turn_lane_return_holds_escalation_then_expires():
    kr, p, ap = rig(on_cfg(), route=route_turn())
    step(kr, p, ap, MAX_TICKS + 1)
    assert kr.ns_level == 0                            # 단계 상승 보류
    assert (kr.ns_info or {}).get('state') == 'TURN_LANE_RETURN'
    step(kr, p, ap, TURN_HOLD_TICKS)
    assert kr.ns_level >= 1                            # 보류에도 시한이 있다
    assert (kr.ns_info or {}).get('state') == 'NEVER_STALL'


def test_turn_lane_return_restores_the_shift_span():
    """전진 대신 복귀 — 밀어 둔 경로를 원복해야 계획 경로가 회전 차로로 데려간다."""
    kr, p, ap = rig(on_cfg(), route=route_turn())
    # _restore_span 이 실제로 만지는 배열들 (목 planner 에는 없다)
    p.commands = np.zeros(len(p.route_s))
    p.commands_orig = p.commands.copy()
    p.lat_shift = np.zeros(len(p.route_s))
    p._lat_build = p.lat_shift.copy()
    step(kr, p, ap, MAX_TICKS - 1)
    kr.ot_span = (0, 10)
    step(kr, p, ap, 1)
    # last_overtake 로는 못 본다 — 같은 틱의 _try_overtake 가 뒤에 덮어쓴다.
    assert kr.ot_span is None
    assert (kr.ns_info or {}).get('state') == 'TURN_LANE_RETURN'

    kr2, p2, ap2 = rig(on_cfg())                       # 예외가 없으면 원복도 없다
    step(kr2, p2, ap2, MAX_TICKS - 1)
    kr2.ot_span = (0, 10)
    step(kr2, p2, ap2, 1)
    assert kr2.ot_span == (0, 10)


@pytest.mark.parametrize('route,why', [
    (route_turn(left_m=200.0), '복귀 거리가 넉넉하면 대상이 아니다'),
    (route_turn(lanes=((1, 0, -1),)), '이미 유효 차로에 있으면 대상이 아니다'),
    (route_turn(lanes=()), '빈 집합 = 유효 차로 없음 — 복귀할 곳이 없다'),
    (route_turn(target='finish'), "target='finish' 는 제약 없음이다"),
    ({'lanes': [], 'cum_s': [], 'total_length': 600.0}, 'valid_entry_lanes 필드가 없다'),
])
def test_turn_lane_exception_does_not_fire(route, why):
    kr, p, ap = rig(on_cfg(), route=route)
    step(kr, p, ap, MAX_TICKS + 1)
    assert kr.ns_level == 1, why
    assert (kr.ns_info or {}).get('state') == 'NEVER_STALL', why


def test_turn_lane_window_uses_shift_transition_length():
    """창 = max(transition_m, shift_k_s·v) + never_stall_turn_margin_m — 상수 복제 금지."""
    kr, p, ap = rig(on_cfg())
    need = OT['transition_m'] + TURN_MARGIN
    p.route = route_turn(left_m=need - 1.0)
    step(kr, p, ap, MAX_TICKS + 1)
    assert (kr.ns_info or {}).get('state') == 'TURN_LANE_RETURN'
    kr2, p2, ap2 = rig(on_cfg(), route=route_turn(left_m=need + 1.0))
    step(kr2, p2, ap2, MAX_TICKS + 1)
    assert (kr2.ns_info or {}).get('state') == 'NEVER_STALL'
