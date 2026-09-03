"""전이 배치 (shift_entry_enable) — 2026-09-03 실주행 100310/100458.

나가는 전이가 목표 차로의 중간 객체를, 복귀 전이가 목표 차로 뒤쪽 객체를 스치지
않도록 시작 지연(delay)·복귀 단축(after)을 고른다. 지키는 것:
  · 기본 off = 이전 동작 (배치 검사 없음, 진단 키 없음).
  · 기본 배치가 임계 이상이면 현행 그대로 (delay 0, after 현행).
  · 중간 객체 → delay 가 생기고 시프트 시작이 뒤로 간다. 전이 길이는 그대로.
  · 복귀 쪽 객체 → after 가 줄고 span 끝이 앞으로 온다.
  · 어디에도 못 두면 entry_block 기각, ot_span None.
  · _corridor_blockers(lat_band=None) 은 이전과 같다.
"""
import copy
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                        # noqa: E402
from test_avoid import HZ, Ap, Box, LgOne, Planner, try_overtake      # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
TRANS = OT['transition_m']
AHEAD = OT['shift_ahead_m']
AFTER = OT['extra_after_m']
BEFORE = OT['extra_before_m']
CLR = CFG['percep']['obstacle_clearance_m']
D = 3.0                                                # 목표 차로 오프셋 [m]
FAR = 22.4                                             # 표적 거리 (100310 의 id4; geom need 19 통과, 시작 = 자차 +5 바닥)


class Car(Box):
    """길이·폭이 있는 객체 (실주행 2.08×0.85 / 4.39×1.81 급)."""

    def __init__(self, oid, x, y, length=2.08, width=0.85, yaw=0.0):
        super().__init__(oid, x, y, 0.0, half_w=width / 2.0)
        self.bounding_box.extent.x = length / 2.0
        self.yaw_deg = yaw


class ShiftPlanner(Planner):
    """x 축 직선 경로 + 우측 차로 −D. plan/shift 는 route.py 와 같은 식(순수 함수)."""

    def __init__(self, **kw):
        super().__init__(n=6000, **kw)
        self._lat_build = np.zeros(6000)
        self.lat_shift = np.zeros(6000)
        self.commands = np.zeros(6000, dtype=int)
        self.commands_orig = self.commands.copy()
        self.calls = []

    def plan_shift_span(self, first_actor, last_actor=None, obstacle_direction='right',
                        transition_length=120.0, lane_transition_factor=1.0,
                        extra_length_before=0.0, extra_length_after=0.0, min_start_ahead=0):
        ppm = self.points_per_meter
        fi = int(round(first_actor.get_location().x * ppm))
        ext = first_actor.bounding_box.extent.x
        a = fi - int(ext * ppm + transition_length + extra_length_before)
        b = fi + int(ext * ppm + transition_length + extra_length_after)
        floor = self.route_index + int(min_start_ahead)
        if a < floor:
            a = min(floor, b - 1)
        return a, b, obstacle_direction == 'right'

    def planned_lateral_offsets(self, a, b, left, step_pts=10):
        n = max(1, (int(b) - int(a)) // max(1, step_pts))
        return np.full(n, D if left else -D)

    def shift_route_around_actors(self, first_actor, last_actor=None, obstacle_direction='right',
                                  transition_length=120.0, lane_transition_factor=1.0,
                                  extra_length_before=0.0, extra_length_after=0.0,
                                  min_start_ahead=0):
        self.calls.append(dict(min_start_ahead=min_start_ahead, after=extra_length_after,
                               trans=transition_length))
        a, b, left = self.plan_shift_span(first_actor, last_actor, obstacle_direction,
                                          transition_length, lane_transition_factor,
                                          extra_length_before, extra_length_after,
                                          min_start_ahead)
        return a, b


def rig(cfg, objs, lvl=1):
    p = ShiftPlanner(d_tl=float('inf'))
    p.lg = LgOne()
    kr = KrRules(cfg)
    kr._sl_all = []
    ap = Ap(p, actors=list(objs))
    ap._kr_ego_lane = (1, 0, -1)
    for _ in range(int(cfg['overtake']['wait_before_shift_s'] * HZ) + 5):
        kr._update_obj_timers(ap)
    if lvl:                                            # 우측 차로 객체 = occupied → L1 완화
        kr.bo_state, kr.bo_level = 'BREAKOUT', lvl
        kr.bo_stop_ticks = kr.bo_hard_ticks
    return kr, p, ap


def on_cfg():
    c = copy.deepcopy(CFG)
    c['overtake']['shift_entry_enable'] = True
    return c


def test_params_present_default_off():
    assert OT['shift_entry_enable'] is False
    assert OT['shift_entry_max_delay_m'] == 15.0 and OT['shift_entry_step_m'] == 0.5
    assert OT['shift_exit_min_after_m'] == 0.0


def test_lat_band_default_is_previous_behaviour():
    kr, p, ap = rig(CFG, [Car(2, FAR, 0.0), Car(3, 11.0, -D)], lvl=0)
    kr._tick_cache(ap, p)
    ids = [a.id for *_, a in kr._corridor_blockers(ap, p)]
    assert ids == [2]                                  # 우측 차로 객체는 회랑 밖
    ids = [a.id for *_, a in kr._corridor_blockers(ap, p, lat_band=(-D, 0.0))]
    assert ids == [3, 2]                               # 띠 안에서는 둘 다


def test_off_no_placement_keys_and_same_call():
    kr, p, ap = rig(CFG, [Car(2, FAR, 0.0), Car(3, 11.0, -D)])
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None and kr.last_overtake == 'right'
    assert 'entry_delay_m' not in kr.last_avoid
    assert p.calls[-1]['min_start_ahead'] == pytest.approx(AHEAD * 10)
    assert p.calls[-1]['after'] == pytest.approx(AFTER * 10)


def test_clear_base_keeps_current_placement():
    kr, p, ap = rig(on_cfg(), [Car(2, FAR, 0.0)])
    try_overtake(kr, ap, p, ego_speed=0.0)
    a = kr.last_avoid
    assert kr.ot_span is not None and a['entry_delay_m'] == 0.0
    assert a['exit_after_m'] == pytest.approx(AFTER) and a['entry_clear'] >= CLR
    assert p.calls[-1]['min_start_ahead'] == pytest.approx(AHEAD * 10)


def test_intermediate_object_delays_entry():
    """100310 재현: 우측 차로 11 m 의 2.08×0.85 객체 — 시작이 뒤로 가고 이격이 선다."""
    kr, p, ap = rig(on_cfg(), [Car(2, FAR, 0.0), Car(3, 11.0, -D)])
    try_overtake(kr, ap, p, ego_speed=0.0)
    a = kr.last_avoid
    assert kr.ot_span is not None and kr.last_overtake == 'right'
    assert a['entry_base_clear'] < CLR and a['entry_block_ids'] == [3]
    assert a['entry_delay_m'] > 0.0 and a['entry_clear'] >= CLR
    assert p.calls[-1]['min_start_ahead'] == pytest.approx((AHEAD + a['entry_delay_m']) * 10)
    assert p.calls[-1]['trans'] == pytest.approx(TRANS * 10)      # 전이 길이 불변
    assert kr.ot_span[0] == pytest.approx(p.route_index + (AHEAD + a['entry_delay_m']) * 10)


def test_exit_object_shortens_after():
    """100458 재현: 복귀 전이 위(표적 +17 m) 우측 차로 객체 — after 가 줄고 끝이 앞으로."""
    kr, p, ap = rig(on_cfg(), [Car(2, FAR, 0.0), Car(5, FAR + 17.0, -D)])
    try_overtake(kr, ap, p, ego_speed=0.0)
    a = kr.last_avoid
    assert kr.ot_span is not None
    assert a['entry_block_ids'] == [5] and a['exit_after_m'] < AFTER
    assert a['entry_clear'] >= CLR and a['entry_delay_m'] == 0.0
    assert p.calls[-1]['after'] == pytest.approx(a['exit_after_m'] * 10)


def test_no_placement_rejects_entry_block():
    """우측 차로에 8·14·20 m 로 빈틈없이 선 객체 — 표적(22.4) 앞에서 나갈 자리가 없다 → 기각."""
    kr, p, ap = rig(on_cfg(), [Car(2, FAR, 0.0), Car(3, 8.0, -D), Car(4, 14.0, -D), Car(6, 20.0, -D)])
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is None
    assert kr.last_overtake == 'right:entry_block@p1'
    a = kr.last_avoid
    assert a['reject'] == 'right:entry_block' and set(a['block_id']) <= {3, 4, 6}
    assert a['best_clear'] < CLR


def test_plateau_object_is_not_a_placement_problem():
    """연쇄 표적(22.4·30) 의 뒤쪽 객체 옆 우측 차로 객체(104807 의 id4·id5 배치) —
    플래토 위라 진입·복귀를 옮겨도 이격이 안 변한다. 판정에서 빼고 이전처럼
    시프트를 만들며 진단에 plateau id 가 남는다 (그 객체는 시프트 뒤 standoff 몫)."""
    kr, p, ap = rig(on_cfg(), [Car(2, FAR, 0.0), Car(5, FAR + 7.6, 0.0), Car(3, FAR + 7.6, -D)])
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None and kr.last_overtake == 'right'
    a = kr.last_avoid
    assert a['chain'] == [2, 5]
    assert a['entry_plateau_ids'] == [3] and a['entry_block_ids'] == []
    assert a['entry_delay_m'] == 0.0 and a['exit_after_m'] == pytest.approx(AFTER)
