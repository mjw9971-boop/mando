"""span 국소성 게이트 (2026-09-03 실주행 100310/100458).

`plan_shift_span` 은 남은 경로 전체에서 차단물 최근접 경로점을 잡으므로 순환
코스에서 4.5 km 앞 구간이 밀렸다 (실측 14틱 4550.1~4595.2 m). 시프트 시작점이
자차 route_index 보다 span_gate_max_m 이상 앞이면 span_too_far 로 기각한다.

지키는 것:
  · 킬 스위치 span_gate_enable=false(기본) → 거리 무관 = 이전 동작.
  · _planned_shift_geom 의 반환 형태 (κ, lc_var) / None 불변, last_span_plan 은
    부속 기록. 기록이 없으면(plan 실패) "못 재면 통과".
  · 기각은 기존 reject 경로 — ot_span 은 None 이고 span_off_m 이 실측으로 남는다.
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
from test_avoid import HZ, Ap, Box, GeomPlanner, LgOne, try_overtake  # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
MAX_M = OT['span_gate_max_m']
FAR = 52.3


class PlanPlanner(GeomPlanner):
    """plan_shift_span 이 route_index + off_pts 에서 시작하는 span 을 돌려준다."""

    def __init__(self, off_pts, fail=False, **kw):
        super().__init__(n=60000, **kw)
        self.off_pts = int(off_pts)
        self.fail = fail

    def plan_shift_span(self, first_actor, last_actor=None, obstacle_direction='right',
                        transition_length=120.0, lane_transition_factor=1.0,
                        extra_length_before=0.0, extra_length_after=0.0, min_start_ahead=0):
        if self.fail:
            raise RuntimeError('no span')
        a = self.route_index + self.off_pts
        return a, a + 400, obstacle_direction == 'right'

    def planned_lateral_offsets(self, a, b, left, step_pts=10):
        return np.zeros(max(3, (b - a) // step_pts))


def rig(cfg, off_m, route_index=450, fail=False):
    p = PlanPlanner(off_m * 10, fail=fail, d_tl=float('inf'))
    p.lg = LgOne()
    p.route_index = route_index
    kr = KrRules(cfg)
    kr._sl_all = []
    ap = Ap(p, actors=[Box(2, FAR + route_index / 10.0, 0.0, 0.0)])
    ap._kr_ego_lane = (1, 0, -1)
    for _ in range(int(cfg['overtake']['wait_before_shift_s'] * HZ) + 5):
        kr._update_obj_timers(ap)
    return kr, p, ap


def on_cfg():
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['span_gate_enable'] = True
    return cfg


def test_params_present_and_default_off():
    assert OT['span_gate_enable'] is False
    assert MAX_M == pytest.approx(100.0)


def test_far_span_rejected_with_measured_offset():
    """실측 100310 t=29.69: a=45951, route_index=450 → 4550.1 m 앞 → 기각."""
    kr, p, ap = rig(on_cfg(), 0.0, route_index=450)
    p.off_pts = 45951 - 450
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is None
    assert kr.last_overtake == 'right:span_too_far@p1'
    a = kr.last_avoid
    assert a['reject'] == 'right:span_too_far'
    assert a['span_off_m'] == pytest.approx(4550.1, abs=0.05)
    assert 'right:span_too_far@p1' in a['rejects']
    assert kr.last_span_plan[0] == 45951


def test_near_span_passes():
    """정상 시프트 실측 5.0~34.7 m — 통과."""
    for off in (5.0, 9.5, 10.7, 34.7):
        kr, p, ap = rig(on_cfg(), off)
        try_overtake(kr, ap, p, ego_speed=0.0)
        assert kr.ot_span is not None and kr.last_overtake == 'right', off


def test_boundary_is_max_m():
    kr, p, ap = rig(on_cfg(), MAX_M - 0.1)
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None
    kr, p, ap = rig(on_cfg(), MAX_M)
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is None and 'span_too_far' in kr.last_overtake


def test_kill_switch_off_ignores_distance():
    kr, p, ap = rig(CFG, 4550.1)
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None and kr.last_overtake == 'right'
    assert 'span_off_m' not in (kr.last_avoid or {})


def test_no_plan_record_falls_through():
    """plan_shift_span 실패 → geom None, last_span_plan None → 검사 없이 통과 (관례)."""
    kr, p, ap = rig(on_cfg(), 4550.1, fail=True)
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.last_span_plan is None
    assert kr.ot_span is not None and kr.last_overtake == 'right'


def test_geom_return_shape_unchanged():
    kr, p, ap = rig(on_cfg(), 20.0)
    kr._tick_cache(ap, p)
    actor = ap._world.get_actors()[0] if hasattr(ap._world, 'get_actors') else None
    geom = kr._planned_shift_geom(p, actor or Box(2, FAR, 0.0, 0.0), 'right', 12.0)
    assert isinstance(geom, tuple) and len(geom) == 2
    assert kr.last_span_plan == (p.route_index + 200, p.route_index + 600, False)
