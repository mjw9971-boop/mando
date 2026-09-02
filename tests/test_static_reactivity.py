"""
묶음 E — 정적 장애물 반응성 (2026-09-03, 020439 3건 + 003759 5건 리플레이 분석).

증상: 옆 차로가 비었는데 25 m 앞에 서서 10~30 s 기다리거나 끝내 못 갔다.
  E-1 장애물 클래스(cls=='obstacle') fast path — 관찰 없이 정적, 큐 제외
  E-2 span_into_zone → 기각 대신 연장 (횡단보도 정지선 통과 / 교차로 출구 뒤 복귀)
  E-3 BREAKOUT 시계를 첫 기각부터 (주행 중 포함), stuck_hard 4 s / escalate 1.5 s
      안전 가드: occupied 완화(L1)·크립(L4)은 정지 stuck_hard_s 경과를 요구
  E-4 WAIT 예산 6 → 3 s + 예산 소진 래치 (기각돼도 WAIT 로 안 돌아감)
  E-6 SHIFT_HOLD: span 통과 원복이 홀드보다 먼저, 홀드 중 standoff·회계
  E-7 적색 일시정지 거리 상한 red_pause_max_m (관찰 pause·큐 B·BREAKOUT pause)

여기서 지키는 불변:
  · 킬 스위치를 전부 끄면 D-verified 와 동일 (replay 9건 diff 0 은 별도 확인).
  · 큐는 차량만 — 박스는 어느 위치에 있어도 큐 형태에 들어가지 않는다.
  · 연장은 교차로를 직진 관통할 때만 — 회전·차선변경·통과 차로 없음·출구 미상은 기각.
  · 주행 중 시계로 오른 단계는 occupied 를 풀지 않고 크립에 닿지 않는다.
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

from kr_rules import KrRules                                          # noqa: E402
from test_avoid import (HZ, STATIC_TICKS, Ap, Box, GeomPlanner, LgOne,  # noqa: E402
                        blocked_rig, make, try_overtake)
from test_queue_only import LgNone, TL, rig as q_rig, set_signal      # noqa: E402
from test_side_pass import LgSolidOccupied                            # noqa: E402
from test_standoff_chain import RecPlanner, rig as chain_rig          # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
HARD = int(round(OT['stuck_hard_s'] * HZ))
ESC = int(round(OT['escalate_s'] * HZ))
WAIT_TICKS = int(OT['wait_before_shift_s'] * HZ) + 5
TRANS, BEFORE, AFTER = OT['transition_m'], OT['extra_before_m'], OT['extra_after_m']
EXIT_M = OT['zone_exit_margin_m']


def cfg_with(**kw):
    c = copy.deepcopy(CFG)
    c['overtake'].update(kw)
    return c


def obstacle(oid, x, y=0.0):
    b = Box(oid, x, y, 0.0, half_w=0.23)
    b.cls = 'obstacle'                                   # world.classify 가 붙이는 필드
    return b


def vehicle(oid, x, y=0.0):
    b = Box(oid, x, y, 0.0, half_w=0.9)
    b.cls = 'vehicle'
    return b


def test_params_present():
    assert OT['obstacle_class_fastpath_enable'] is True
    assert OT['zone_extend_enable'] is True and OT['zone_extend_max_m'] == 120.0
    assert OT['zone_exit_margin_m'] == 5.0
    assert OT['breakout_reject_clock_enable'] is True
    assert OT['stuck_hard_s'] == 4.0 and OT['escalate_s'] == 1.5
    assert OT['wait_before_shift_s'] == 3.0 and OT['preempt_latch_enable'] is True
    assert OT['shift_hold_restore_enable'] is True
    assert OT['red_pause_max_m'] == 100.0


# ══════════════════════════════════════════════════════════════════════════
# E-1 장애물 클래스 fast path
# ══════════════════════════════════════════════════════════════════════════
def test_obstacle_is_static_without_observation():
    """박스는 첫 틱부터 회랑·standoff 대상. 차량은 여전히 obj_static_s 관찰."""
    kr, p = make(d_tl=float('inf'))
    ap = Ap(p, [obstacle(1, 20.0), vehicle(2, 40.0)])
    kr._update_obj_timers(ap)                            # 1틱
    ids = [a.id for _s, _l, _h, a in kr._corridor_blockers(ap, p)]
    assert ids == [1]
    assert kr._stop_ok(ap._world.get_actors()[0]) is True
    for _ in range(STATIC_TICKS):
        kr._update_obj_timers(ap)
    ids = [a.id for _s, _l, _h, a in kr._corridor_blockers(ap, p)]
    assert ids == [1, 2]


def test_obstacle_is_never_a_queue():
    """적색 60 m 앞 박스 1개 = 큐 아님 (obstacle_class). 같은 자리 차량 = 큐 B."""
    kr, p, ap = q_rig(xs=(10.0,), state=TrafficLightState.Red, d_tl=60.0)
    for a in ap._world.get_actors():
        a.cls = 'obstacle'
    kr._tick_cache(ap, p)
    assert kr._tick_queue is False and kr.q_reject == 'obstacle_class'
    for a in ap._world.get_actors():
        a.cls = 'vehicle'
    kr._tick_cache(ap, p)
    assert kr._tick_queue is True


def test_mixed_queue_counts_vehicles_only():
    """박스 + 차량이 섞이면 큐 판정은 차량만 본다 (박스가 선두여도 큐 B 는 차량 기준)."""
    kr, p, ap = q_rig(xs=(10.0, 30.0), state=TrafficLightState.Red, d_tl=60.0)
    ap._world.get_actors()[1].cls = 'obstacle'           # 선두(30 m)가 박스
    kr._tick_cache(ap, p)
    assert kr._tick_queue is True
    assert kr.q_info['head_id'] == 2 and kr.q_info['n'] == 1


def test_obstacle_preempts_on_first_tick_when_budget_short():
    """관찰 없이 t_left < budget 이면 첫 틱에 PREEMPT (40 m, 10 m/s → t_left 1.0 < 3.0)."""
    p = GeomPlanner(d_tl=float('inf'))
    p.lg = LgOne()
    kr = KrRules(CFG)
    kr._sl_all = []
    ap = Ap(p, [obstacle(2, 40.0)])
    ap._kr_ego_lane = (1, 0, -1)
    kr._update_obj_timers(ap)
    try_overtake(kr, ap, p, ego_speed=10.0)
    assert kr.last_avoid['state'] == 'PREEMPT' and kr.ot_span is not None


def test_obstacle_fastpath_kill_switch():
    kr, p, ap = q_rig(xs=(10.0,), state=TrafficLightState.Red, d_tl=60.0,
                      cfg=cfg_with(obstacle_class_fastpath_enable=False))
    for a in ap._world.get_actors():
        a.cls = 'obstacle'
    kr._tick_cache(ap, p)
    assert kr._tick_queue is True                        # 옛 동작: 박스도 큐
    kr2, p2 = make(cfg_with(obstacle_class_fastpath_enable=False), d_tl=float('inf'))
    ap2 = Ap(p2, [obstacle(1, 20.0)])
    kr2._update_obj_timers(ap2)
    assert kr2._corridor_blockers(ap2, p2) == []         # 옛 동작: 관찰 필요


# ══════════════════════════════════════════════════════════════════════════
# E-2 span_into_zone → 연장
# ══════════════════════════════════════════════════════════════════════════
A, J, B = (1, 0, -1), (9, 0, -1), (2, 0, -1)
AN, JN, BN = (1, 0, -2), (9, 0, -2), (2, 0, -2)


class LgJ(LgOne):
    """A(0~64) → J(64~104, 교차로 5) → B(104~304). 오른쪽 이웃 An → Jn → Bn 이 같은
    순서로 successor 로 이어진다 (직진 관통). j_nb=False 면 교차로 안 이웃 없음,
    j_link=False 면 An 의 successor 가 Jn 이 아니다."""

    def __init__(self, j_nb=True, j_link=True):
        self.lanes = {
            A: {'junction': -1, 'right_nb': AN, 'left_nb': None, 'next': [J]},
            J: {'junction': 5, 'right_nb': JN if j_nb else None, 'left_nb': None,
                'next': [B]},
            B: {'junction': -1, 'right_nb': BN, 'left_nb': None, 'next': []},
            AN: {'junction': -1, 'right_nb': None, 'left_nb': A,
                 'next': [JN] if j_link else []},
            JN: {'junction': 5, 'right_nb': None, 'left_nb': J, 'next': [BN]},
            BN: {'junction': -1, 'right_nb': None, 'left_nb': B, 'next': []},
        }
        self._len = {A: 64.0, J: 40.0, B: 200.0}

    def neighbor(self, key, side):
        return self.lanes[key]['right_nb' if side == 'right' else 'left_nb']

    def successors(self, key):
        return self.lanes[key]['next']

    def length(self, key):
        return self._len.get(key, 1000.0)


V0 = 10.0                                            # 시도 속도 → trans 30, span 75, need 37
SL = 64.0                                            # 정지선 (교차로 진입)
J_OUT = 104.0                                        # 교차로 출구


def zone_rig(cfg=CFG, lg=None, lanes=(A, J, B), cum=(0.0, SL, J_OUT),
             lens=(SL, J_OUT - SL, 200.0), events=(), stopline=SL, obj_x=38.0):
    """정지선 64 m, 차량 38 m, 10 m/s. span 75 > 64 라 옛 게이트는 기각한다.
    (차량이 정지선 25 m 안이면 큐 A 가 먼저 억제하므로 선두→정지선 26 m 로 둔다.
     geom need 37 < 38 통과.)"""
    p = RecPlanner(d_tl=float('inf'))
    p.lg = LgJ() if lg is None else lg
    p.route = {'lanes': list(lanes), 'cum_s': list(cum), 'lengths': list(lens),
               'events': list(events), 'total_length': 300.0}
    kr = KrRules(cfg)
    kr._sl_all = [float(stopline)]
    ap = Ap(p, [vehicle(2, obj_x)])
    ap._kr_ego_lane = A
    for _ in range(WAIT_TICKS):
        kr._update_obj_timers(ap)
    try_overtake(kr, ap, p, ego_speed=V0)
    return kr, p, ap


def test_zone_crossing_allowed_at_midblock_stopline():
    """정지선 뒤에 교차로가 없으면(횡단보도) 연장 없이 그대로 통과 — 실측 020439/02."""
    kr, p, ap = zone_rig(lg=LgJ(), lanes=(A,), cum=(0.0,), lens=(304.0,))
    assert kr.ot_span is not None and kr.last_overtake == 'right'
    a = kr.last_avoid
    assert a['zone_extended'] is True and a['extended_by'] == 0.0
    assert a['after_m'] == pytest.approx(AFTER) and a['zones'][0]['junction'] is None
    assert p.calls[-1][2]['extra_length_after'] == pytest.approx(AFTER * 10)


def test_zone_extends_past_straight_junction():
    """교차로를 직진 관통하는 옆 차로가 있으면 출구 + 여유까지 extra_after 를 늘린다."""
    kr, p, ap = zone_rig()
    assert kr.ot_span is not None and kr.last_overtake == 'right'
    a = kr.last_avoid
    # 실제 span 끝 추정 = 시작 여유 5 + span 75 = 80 → 교차로 출구 104 + 5 = 109 → +29
    trans = OT['shift_k_s'] * V0
    span_end_est = OT['shift_ahead_m'] + (2 * trans + BEFORE + AFTER)
    expect = J_OUT + EXIT_M - span_end_est
    assert a['zone_extended'] is True
    assert a['extended_by'] == pytest.approx(expect, abs=0.1)
    assert a['after_m'] == pytest.approx(AFTER + expect, abs=0.1)
    assert a['zones'][0] == {'s': SL, 'junction': 5, 'out': J_OUT}
    assert p.calls[-1][2]['extra_length_after'] == pytest.approx((AFTER + expect) * 10, abs=1)


@pytest.mark.parametrize('events,label', [
    ([{'kind': 'turn_right', 's': SL}], 'zone_turn'),
    ([{'kind': 'lane_change_left', 's': 10.0, 'window_s0': 10.0, 'window_s1': 25.0}],
     'zone_lane_change'),
])
def test_zone_rejects_when_route_turns_or_changes_lane(events, label):
    kr, p, ap = zone_rig(events=events)
    assert kr.ot_span is None and kr.last_overtake == f'right:{label}@p1'
    assert kr.last_avoid['reject'] == f'right:{label}'


@pytest.mark.parametrize('lg', [LgJ(j_nb=False), LgJ(j_link=False)])
def test_zone_rejects_without_through_lane(lg):
    """교차로 안 이웃이 없거나 이웃이 successor 로 이어지지 않으면 기각."""
    kr, p, ap = zone_rig(lg=lg)
    assert kr.ot_span is None and kr.last_overtake == 'right:zone_no_through_lane@p1'


def test_zone_rejects_when_junction_has_no_exit_lane():
    kr, p, ap = zone_rig(lanes=(A, J), cum=(0.0, SL), lens=(SL, J_OUT - SL))
    assert kr.ot_span is None and kr.last_overtake == 'right:zone_no_exit@p1'


def test_zone_rejects_beyond_extend_max():
    kr, p, ap = zone_rig(cfg=cfg_with(zone_extend_max_m=10.0))
    assert kr.ot_span is None and kr.last_overtake == 'right:zone_extend_max@p1'


def test_zone_extend_kill_switch_keeps_old_label():
    kr, p, ap = zone_rig(cfg=cfg_with(zone_extend_enable=False))
    assert kr.ot_span is None and kr.last_overtake == 'right:span_into_zone@p1'
    assert kr.last_avoid['zone_lo'] == pytest.approx(SL)


def test_zone_no_route_info_falls_back_to_old_label():
    """경로 차로 정보가 없는 목 플래너에서는 평가 불가 → 옛 라벨 그대로."""
    kr, p, ap = zone_rig(lanes=(), cum=(), lens=())
    assert kr.ot_span is None and kr.last_overtake == 'right:span_into_zone@p1'


def test_route_dashed_cover_counts_junction_as_ok():
    kr = KrRules(CFG)
    p = RecPlanner(d_tl=float('inf'))
    p.route = {'lanes': [A, J, B], 'cum_s': [0.0, 30.0, 70.0], 'lengths': [30.0, 40.0, 200.0]}
    lg = LgJ()
    lg.dashed_runs = lambda key, side: [(0.0, 10.0)] if key == A else [(0.0, 1000.0)]
    cover = kr._dashed_ahead_route_m(p, lg, 'right', 0.0, 80.0, A)
    assert cover == pytest.approx(10.0 + 40.0 + 10.0)    # A 점선 10 + 교차로 40 + B 10


# ══════════════════════════════════════════════════════════════════════════
# E-3 BREAKOUT 시계를 첫 기각부터
# ══════════════════════════════════════════════════════════════════════════
def clock_rig(cfg=CFG, lg=None, obj_x=60.0):
    """장애물 60 m(blocker_dist_max 밖) + 이웃 없음 → 주행 중 매 틱 양쪽 기각."""
    p = GeomPlanner(d_tl=float('inf'))
    p.lg = LgNone() if lg is None else lg
    kr = KrRules(cfg)
    kr._sl_all = []
    kr.last_d_end = 1e6
    ap = Ap(p, [vehicle(2, obj_x)])
    ap._kr_ego_lane = (1, 0, -1)
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    for _ in range(WAIT_TICKS):
        kr._update_obj_timers(ap)
    return kr, p, ap


def step(kr, p, ap, n, v):
    """apply 와 같은 순서: 타이머 → 캐시 → BREAKOUT → 회피 시도."""
    for _ in range(n):
        kr._update_obj_timers(ap)
        kr._tick_cache(ap, p)
        kr._breakout_tick(p, ap, v)
        kr._try_overtake(ap, p, v)


def test_reject_clock_enters_breakout_while_moving():
    kr, p, ap = clock_rig()
    step(kr, p, ap, 1, 5.0)
    assert kr.last_avoid['state'] == 'WAIT_EXPIRED' and kr.ot_reject_ticks == 1
    assert kr._blocker(ap, p) is None                    # 30 m 밖 — 옛 조건으로는 원인 아님
    step(kr, p, ap, HARD - 1, 5.0)
    assert kr.bo_state is None
    step(kr, p, ap, 1, 5.0)
    assert (kr.bo_state, kr.bo_level) == ('BREAKOUT', 1)
    assert kr.last_avoid['reject_s'] == pytest.approx((HARD + 1) / HZ, abs=0.06)
    step(kr, p, ap, ESC, 5.0)
    assert kr.bo_level == 2
    step(kr, p, ap, ESC, 5.0)
    assert kr.bo_level == 3


def test_creep_needs_stationary_stuck_hard_s():
    """주행 중 시계로는 L3 에서 멈춘다. 정지가 stuck_hard_s 차면 L4."""
    kr, p, ap = clock_rig()
    step(kr, p, ap, HARD + ESC * 3 + 5, 5.0)
    assert kr.bo_level == 3 and kr.breakout_creep() is False
    step(kr, p, ap, HARD - 1, 0.0)
    assert kr.bo_level == 3
    step(kr, p, ap, 1, 0.0)
    assert kr.bo_level == 4 and kr.breakout_creep() is True


def test_progress_does_not_exit_while_still_rejected():
    kr, p, ap = clock_rig()
    step(kr, p, ap, HARD + 1, 5.0)
    assert kr.bo_state == 'BREAKOUT'
    p.route_index += 30                                  # 3 m 전진 (progress_m 2 초과)
    step(kr, p, ap, 1, 5.0)
    assert kr.bo_state == 'BREAKOUT'
    ap._world._a[:] = []                                 # 차단물 소멸 → 기각 끊김
    step(kr, p, ap, 2, 5.0)                              # 시계는 직전 틱 기각을 보므로 2틱
    assert kr.bo_state is None and kr.bo_exit in ('cause_gone', 'progress')


def test_occupied_relax_needs_stationary_stuck_hard_s():
    """L1 이어도 정지 stuck_hard_s 전에는 점유 차로로 밀지 않는다."""
    from test_avoid import _geom_rig
    kr, p, ap = _geom_rig(CFG, 52.3)
    p.lg = LgSolidOccupied()
    ap._world._a.append(Box(9, 10.0, -3.5))
    kr.bo_state, kr.bo_level = 'BREAKOUT', 1
    kr.bo_stop_ticks = HARD - 1
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is None and 'occupied' in kr.last_overtake
    kr.bo_stop_ticks = HARD
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None


def test_reject_clock_kill_switch():
    kr, p, ap = clock_rig(cfg=cfg_with(breakout_reject_clock_enable=False))
    step(kr, p, ap, HARD + ESC * 4, 5.0)
    assert kr.bo_state is None and kr.ot_reject_ticks > 0
    assert kr._reject_pending() is False


def test_stationary_clock_still_enters_after_stuck_hard_s():
    """옛 경로(정지) 그대로 — 기각 시계와 무관하게 정지 stuck_hard_s 로 진입."""
    kr, p, ap = blocked_rig()
    for _ in range(HARD):
        kr._update_obj_timers(ap)
        kr._tick_cache(ap, p)
        kr._breakout_tick(p, ap, 0.0)
    assert (kr.bo_state, kr.bo_level) == ('BREAKOUT', 1)


# ══════════════════════════════════════════════════════════════════════════
# E-4 WAIT 예산 3 s + 예산 소진 래치
# ══════════════════════════════════════════════════════════════════════════
def latch_rig(cfg=CFG):
    """차량 40 m, 이웃 없음. 10 m/s: standoff 30 → t_left 1.0 < budget 1.5 → PREEMPT."""
    p = GeomPlanner(d_tl=float('inf'))
    p.lg = LgNone()
    kr = KrRules(cfg)
    kr._sl_all = []
    ap = Ap(p, [vehicle(2, 40.0)])
    ap._kr_ego_lane = (1, 0, -1)
    for _ in range(STATIC_TICKS):
        kr._update_obj_timers(ap)
    return kr, p, ap


def test_budget_latch_keeps_preempt_after_reject():
    kr, p, ap = latch_rig()
    try_overtake(kr, ap, p, ego_speed=10.0)
    assert kr.last_avoid['state'] == 'PREEMPT' and kr.last_avoid['latched'] is False
    assert kr.ot_reject_ticks == 1
    kr._update_obj_timers(ap)
    try_overtake(kr, ap, p, ego_speed=2.0)               # t_left 7.5 > budget — 래치로 유지
    assert kr.last_avoid['state'] == 'PREEMPT' and kr.last_avoid['latched'] is True
    assert kr.ot_reject_ticks == 2


def test_budget_latch_clears_on_new_blocker():
    kr, p, ap = latch_rig()
    try_overtake(kr, ap, p, ego_speed=10.0)
    assert kr.preempt_latch_id == 2
    ap._world._a[:] = [vehicle(3, 45.0)]
    for _ in range(STATIC_TICKS):
        kr._update_obj_timers(ap)
    try_overtake(kr, ap, p, ego_speed=2.0)
    assert kr.last_avoid['state'] == 'WAIT' and kr.preempt_latch_id is None


def test_budget_latch_kill_switch():
    kr, p, ap = latch_rig(cfg=cfg_with(preempt_latch_enable=False))
    try_overtake(kr, ap, p, ego_speed=10.0)
    assert kr.last_avoid['state'] == 'PREEMPT'
    kr._update_obj_timers(ap)
    try_overtake(kr, ap, p, ego_speed=2.0)
    assert kr.last_avoid['state'] == 'WAIT'              # 옛 동작: 되돌아간다


# ══════════════════════════════════════════════════════════════════════════
# E-6 SHIFT_HOLD 정리
# ══════════════════════════════════════════════════════════════════════════
def _hold_after_span(cfg):
    kr, p, ap, boxes = chain_rig(cfg=cfg, xs=(52.3,))
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None
    n = len(p.route_s)
    p.commands = np.zeros(n)
    p.commands_orig = np.zeros(n)
    p.lat_shift = np.zeros(n)
    p._lat_build = np.zeros(n)
    tl = type('TL', (), {'state': TrafficLightState.Red, 'id': 7})()
    p.distances_to_next_traffic_lights[:] = 60.0
    p.next_traffic_lights = [tl] * n
    p.route_index = kr.ot_span[1] + 1                    # span 을 지났다
    ap._world._a[:] = []                                 # 다음 장애물 없음
    try_overtake(kr, ap, p, ego_speed=5.0)
    return kr


def test_passed_span_is_restored_even_under_red():
    kr = _hold_after_span(CFG)
    assert kr.ot_span is None and kr.last_overtake == 'restored'


def test_hold_restore_kill_switch_keeps_old_order():
    kr = _hold_after_span(cfg_with(shift_hold_restore_enable=False))
    assert kr.ot_span is not None and kr.last_avoid['state'] == 'SHIFT_HOLD'


def test_hold_tick_runs_blocked_accounting():
    """홀드 중에도 막힘 회계가 돈다 (정지 + 30 m 안 차단물 → ot_blocked_ticks)."""
    kr, p, ap, boxes = chain_rig(xs=(52.3,))
    try_overtake(kr, ap, p, ego_speed=0.0)
    tl = type('TL', (), {'state': TrafficLightState.Red, 'id': 7})()
    p.distances_to_next_traffic_lights[:] = 60.0
    p.next_traffic_lights = [tl] * len(p.route_s)
    ap._world._a.append(vehicle(5, 20.0))                # 복귀 전이 위 다음 차단물
    for _ in range(STATIC_TICKS):
        kr._update_obj_timers(ap)
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.last_avoid['state'] == 'SHIFT_HOLD' and kr.last_avoid['blocker'] == 5
    assert kr.ot_blocked_ticks == 1


# ══════════════════════════════════════════════════════════════════════════
# E-7 적색 일시정지 거리 상한
# ══════════════════════════════════════════════════════════════════════════
def test_red_pause_is_distance_bounded():
    kr, p = make(d_tl=456.0)
    p.next_traffic_lights = [TL(TrafficLightState.Red)] * len(p.route_s)
    assert kr._red_ahead(p) == pytest.approx(456.0)      # 신호 계층은 그대로 본다
    assert kr._red_pause(p) is None                      # 회피 계층은 없는 것으로
    set_signal(p, TrafficLightState.Red, 60.0)
    assert kr._red_pause(p) == pytest.approx(60.0)
    kr0 = KrRules(cfg_with(red_pause_max_m=0.0))
    set_signal(p, TrafficLightState.Red, 456.0)
    assert kr0._red_pause(p) == pytest.approx(456.0)     # 0 = 무제한 (이전 동작)


def test_far_red_does_not_make_cond_B_queue():
    """실측 020439/03: 456 m 앞 적신호 + 정지 차량 1대 → 큐가 아니다."""
    kr, p, ap = q_rig(xs=(23.0,), state=TrafficLightState.Red, d_tl=456.0)
    assert kr._tick_queue is False and kr.q_reject == 'head_far'
    kr, p, ap = q_rig(xs=(23.0,), state=TrafficLightState.Red, d_tl=60.0)
    assert kr._tick_queue is True and kr.q_info['cond'] == 'B'
    assert kr.q_info['d_sl'] == pytest.approx(60.0) and kr.q_info['head_sl'] == pytest.approx(37.0)


def test_far_red_does_not_pause_breakout():
    kr, p, ap = blocked_rig(d_tl=456.0)
    p.next_traffic_lights = [TL(TrafficLightState.Red)] * len(p.route_s)
    for _ in range(HARD):
        kr._update_obj_timers(ap, paused=kr._red_pause(p) is not None)
        kr._tick_cache(ap, p)
        kr._breakout_tick(p, ap, 0.0)
    assert kr.bo_paused is False and kr.bo_state == 'BREAKOUT'
    kr, p, ap = blocked_rig(d_tl=60.0)
    p.next_traffic_lights = [TL(TrafficLightState.Red)] * len(p.route_s)
    for _ in range(HARD):
        kr._update_obj_timers(ap, paused=kr._red_pause(p) is not None)
        kr._tick_cache(ap, p)
        kr._breakout_tick(p, ap, 0.0)
    assert kr.bo_paused is True and kr.bo_state is None

# ══════════════════════════════════════════════════════════════════════════
# E-8 실차 전 보정
# ══════════════════════════════════════════════════════════════════════════
class LgNoneMark(LgOne):
    """점선은 없고(dashed_runs 빈 목록) 오른쪽 마킹이 'none' 인 구간만 있는 목."""

    def __init__(self):
        super().__init__()
        self.lanes[(1, 0, -1)]['right_mark'] = [(0.0, 20.0, 'solid', 'standard', False),
                                                (20.0, 60.0, 'none', 'standard', False),
                                                (60.0, 1000.0, 'solid', 'standard', False)]

    def dashed_runs(self, key, side):
        return []


def test_none_marking_counts_as_crossable():
    """① 선이 없는 구간은 넘어도 위반이 아니다 — 점선처럼 cover 에 든다. 스위치 off 면 0."""
    kr = KrRules(CFG)
    assert kr._dashed_ahead_m(LgNoneMark(), (1, 0, -1), 'right', 10.0, 40.0) == pytest.approx(30.0)
    kr0 = KrRules(cfg_with(none_marking_crossable=False))
    assert kr0._dashed_ahead_m(LgNoneMark(), (1, 0, -1), 'right', 10.0, 40.0) == 0.0
    assert kr._dashed_ahead_m(LgOne(), (1, 0, -1), 'right', 0.0, 40.0) == pytest.approx(40.0)  # 마크 없는 목: 점선만

