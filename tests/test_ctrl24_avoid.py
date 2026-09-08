"""
ctrl24 커밋 3 — 정적 장애물 PREEMPT · side 게이트 1개(span_too_far) · NOOP · P2 pre-pass.

여기서 지키는 불변:
  · 회랑 안 정지 객체는 관찰·대기 없이 **첫 틱**에 시프트를 시도한다.
  · 좌측 우선, 좌측 변위 0(이웃 없음 — 플래너 폴백)이면 우측. 둘 다 0 이면 NOOP:
    ot_span 없음, 요동 방지 집합에도 안 들어간다 (다음 틱 다시 시도).
  · 게이트는 span_too_far 하나. 그 밖의 기하는 묻지 않는다.
  · 보행자는 회피 대상이 아니다 (K3 의 정지 대상).
  · pre_pass 가 돈 틱에는 apply 가 회피를 다시 돌리지 않는다.
"""
import copy
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import VehicleControl
from vtd_adapter.config import load_params_yaml
from vtd_adapter.control import VtdLongitudinalController

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from ctrl24 import Ctrl24                                       # noqa: E402
from test_avoid import Ap, Box, Planner                         # noqa: E402
from test_ctrl24_long import Walker, cfg_with                   # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
LANE = 3.0


class ShiftPlanner(Planner):
    """x 축 직선 경로 위에서 route.py 의 시프트 표면을 흉내낸다.

    좌/우 이웃 유무를 켜고 끈다. 이웃이 없으면 route.py 폴백처럼 **경로를 그대로 둔다**
    (변위 0). 좌 시프트 = +y, 우 = −y, 한 칸 3.0 m. _shift_target_steps 는 ref_index 의
    현재 밀림에서 k+1 을 낸다 (route.py 의 ctrl24 동작과 같은 규약).
    """

    def __init__(self, left=True, right=True, **kw):
        super().__init__(**kw)
        n = len(self.route_s)
        self.has = {'left': left, 'right': right}
        self.commands = np.zeros(n, dtype=int)
        self.commands_orig = self.commands.copy()
        self.lat_shift = np.zeros(n)
        self._lat_build = self.lat_shift.copy()
        self._kd = None
        self.calls = []

    @staticmethod
    def _idx(actor):
        return int(round(actor.get_location().x / 0.1))

    def plan_shift_span(self, first_actor, last_actor=None, obstacle_direction='right',
                        transition_length=120.0, extra_length_before=0.0,
                        extra_length_after=0.0, min_start_ahead=0):
        ext = first_actor.bounding_box.extent.x * 10
        fi = self._idx(first_actor)
        a = fi - int(ext + transition_length + extra_length_before)
        li = fi if last_actor is None else self._idx(last_actor)
        b = li + int(ext + transition_length + extra_length_after)
        floor = self.route_index + int(min_start_ahead)
        if a < floor:
            a = min(floor, b - 1)
        return a, b, obstacle_direction == 'right'

    def _shift_target_steps(self, left, ref_index=None):
        side = 'left' if left else 'right'
        if not self.has[side]:
            return 1
        i = self.route_index if ref_index is None else int(ref_index)
        d1 = LANE if left else -LANE
        k = int(round(self.route_points[i, 1] / d1))
        return max(0, k + 1)

    def planned_lateral_offsets(self, a, b, left, step_pts=10, ref_index=None):
        side = 'left' if left else 'right'
        n = len(range(int(a), int(b), step_pts))
        if not self.has[side]:
            return np.zeros(n)
        steps = self._shift_target_steps(left, ref_index)
        return np.full(n, (LANE if left else -LANE) * steps)

    def shift_route_smoothly(self, a, b, left, transition_length=120.0,
                             lane_transition_factor=1.0, transition_length_back=None,
                             ref_index=None):
        self.calls.append((a, b, left, transition_length, transition_length_back))
        side = 'left' if left else 'right'
        if not self.has[side]:
            return
        steps = self._shift_target_steps(left, ref_index)
        target = (LANE if left else -LANE) * steps
        back = transition_length if transition_length_back is None else transition_length_back
        for idx in range(int(a), int(b)):
            f = 1.0
            if idx <= a + transition_length and idx - a < b - idx:
                f = -math.cos(float(idx - a) / transition_length * math.pi) / 2 + 0.5
            elif idx >= b - back:
                f = -math.cos(float(b - idx) / back * math.pi) / 2 + 0.5
            y = f * target + (1.0 - f) * self.route_points[idx, 1]
            self.route_points[idx, 1] = y
            self.lat_shift[idx] = y


class Car(Box):
    def __init__(self, oid, x, y=0.0, speed=0.0):
        super().__init__(oid, x, y, speed, half_w=0.9)
        self.bounding_box.extent.x = 2.2
        self.type_id = 'vehicle.vtd.object'

    def get_location(self):
        s = self

        class L:
            x, y, z = s._x, s._y, 0.0

            def distance(self_, o):
                return math.hypot(s._x - o.x, s._y - o.y)
        return L()


def car(oid, x, y=0.0, speed=0.0):
    return Car(oid, x, y, speed)


def rig(cfg=CFG, actors=(), left=True, right=True):
    p = ShiftPlanner(left=left, right=right, d_tl=float('inf'))
    ap = Ap(p, list(actors))
    ap._longitudinal_controller = VtdLongitudinalController(cfg)
    kr = Ctrl24(cfg)
    kr._sl_all = []
    return kr, p, ap


def apply(kr, ap, v, target=12.5):
    ap._vehicle.speed = v
    ap._longitudinal_controller.get_throttle_and_brake(False, target, v)
    return kr.apply(VehicleControl(steer=0.0, accel=1.0), target, ap)


# ── A: 첫 틱 PREEMPT ────────────────────────────────────────────────────
def test_preempt_on_first_tick_at_80m_left_first():
    kr, p, ap = rig(actors=[car(2, 78.0)])
    apply(kr, ap, v=12.5)
    a = kr.last_avoid
    assert a['state'] == 'PREEMPT' and a['shift'] == 'left' and a['trigger'] == 'corridor'
    assert a['trans_m'] == pytest.approx(C['trans_k'] * 12.5, abs=0.05)
    assert kr.ot_span is not None and 2 in kr._shifted_for
    assert p.route_points[kr.ot_span[0] + 600, 1] == pytest.approx(LANE, abs=0.01)   # 플래토 = 좌 3 m
    assert p.route_points[kr.ot_span[1] - 1, 1] == pytest.approx(0.0, abs=0.02)


def test_transition_floor_when_stopped():
    kr, p, ap = rig(actors=[car(2, 12.0)])
    apply(kr, ap, v=0.0)
    assert kr.last_avoid['trans_m'] == pytest.approx(C['trans_min_m'])
    assert kr.ot_span[0] == int(C['shift_ahead_m'] * 10)      # 코앞이라 전이 시작 = 자차 앞 5 m


def test_no_observation_clock_even_for_briefly_stopped_car():
    """정지한 첫 틱에 곧바로 시도 — 관찰 시계가 없다."""
    c = car(2, 60.0, speed=3.0)
    kr, p, ap = rig(actors=[c])
    apply(kr, ap, v=10.0)
    assert kr.last_avoid is None and kr.ot_span is None         # 이동 중 — 대상 아님
    c.speed = 0.0
    apply(kr, ap, v=10.0)
    assert kr.ot_span is not None and kr.last_avoid['state'] == 'PREEMPT'


def test_right_when_left_has_no_neighbour():
    kr, p, ap = rig(actors=[car(2, 60.0)], left=False)
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['shift'] == 'right' and a['rejects'] == ['left:noop']
    assert p.route_points[kr.ot_span[0] + 400, 1] == pytest.approx(-LANE)


def test_noop_when_no_neighbour_either_side():
    kr, p, ap = rig(actors=[car(2, 60.0)], left=False, right=False)
    before = p.route_points.copy()
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['state'] == 'NOOP' and a['rejects'] == ['left:noop', 'right:noop']
    assert kr.ot_span is None and not kr._shifted_for
    assert np.array_equal(p.route_points, before)
    apply(kr, ap, v=8.0)                                        # 다음 틱 다시 시도한다
    assert kr.last_avoid['state'] == 'NOOP'


def test_only_gate_is_span_too_far():
    kr, p, ap = rig(cfg_with(span_gate_max_m=20.0), actors=[car(2, 78.0)])
    apply(kr, ap, v=12.5)
    a = kr.last_avoid
    assert a['reject'] == 'left:span_too_far' and a['rejects'] == ['left:span_too_far']
    assert a['span_off_m'] >= 20.0 and kr.ot_span is None
    assert len(p.calls) == 0                                     # 반대쪽도 안 민다 (기하 동일)


def test_shifted_object_leaves_corridor_and_is_not_retried():
    kr, p, ap = rig(actors=[car(2, 60.0)])
    apply(kr, ap, v=8.0)
    n_calls = len(p.calls)
    apply(kr, ap, v=8.0)
    assert kr.last_avoid['state'] == 'SHIFT_ACTIVE' and len(p.calls) == n_calls
    assert kr._corridor == []                                    # 밀린 경로 회랑 밖


def test_restore_after_passing_span_end():
    kr, p, ap = rig(actors=[car(2, 60.0)])
    apply(kr, ap, v=8.0)
    a, b = kr.ot_span
    p.route_index = b + 5
    apply(kr, ap, v=8.0)
    assert kr.ot_span is None and kr.last_overtake == 'restored'
    assert np.array_equal(p.route_points[a:b], p.original_route_points[a:b])
    assert not kr._shifted_for


def test_chain_merges_objects_within_gap():
    kr, p, ap = rig(actors=[car(2, 40.0), car(3, 52.0), car(4, 90.0)])
    apply(kr, ap, v=5.0)
    a = kr.last_avoid
    assert a['chain'] == [2, 3]
    assert kr.ot_span[1] == 520 + int(22 + C['trans_min_m'] * 10 + C['extra_after_m'] * 10) \
        or kr.ot_span[1] > 520


def test_pedestrian_is_not_an_avoid_target():
    kr, p, ap = rig(actors=[Walker(9, 40.0, 0.5, speed=0.0)])
    apply(kr, ap, v=5.0)
    assert kr.ot_span is None and kr.last_avoid is None
    assert kr.last_kr['ped_intent'] == 0.0                       # K3 홀드가 세운다


def test_avoid_switch_off():
    kr, p, ap = rig(cfg_with(avoid_enable=False), actors=[car(2, 60.0)])
    apply(kr, ap, v=8.0)
    assert kr.ot_span is None and kr.last_avoid is None


# ── P2 pre-pass ─────────────────────────────────────────────────────────
class ObbAp(Ap):
    """OBB 예측 표면만 있는 autopilot 목 — hit_ids 의 정지 객체가 교차한다고 답한다."""

    def __init__(self, planner, actors, hit_ids=()):
        super().__init__(planner, actors)
        self._longitudinal_controller = VtdLongitudinalController(CFG)
        self.config = type('C', (), {'idm_red_light_minimum_distance': 5.299,
                                     'detection_radius': 50.0, 'bicycle_frame_rate': 20,
                                     'forecast_length_lane_change': 1.1,
                                     'default_forecast_length': 2.0})()
        self.hit_ids = set(hit_ids)
        self.forecast_calls = 0
        self._vehicle.get_transform = lambda: None

    def is_near_lane_change(self, v, route_np):
        return False

    def forecast_ego_agent(self, tr, v, n, target, route_np):
        self.forecast_calls += 1
        return list(range(n))

    def predict_other_actors_bounding_boxes(self, plant, actors, loc, n, near_lc):
        return {a.id: [a.id] * n for a in actors}          # 상자 자리에 id 를 싣는다

    def check_obb_intersection(self, ebb, obb):
        return obb in self.hit_ids


def test_prepass_obb_triggers_shift_for_object_outside_corridor():
    c = car(2, 40.0, 2.5)                                        # 회랑 반폭 2.14 m 밖
    kr, p, _ap = rig(actors=[c])
    ap = ObbAp(p, [c], hit_ids={2})
    ap._vehicle.speed = 8.0
    kr.pre_pass(ap, p.route_points, [c], 12.5, 8.0)
    a = kr.last_avoid
    assert a['trigger'] == 'obb' and a['state'] == 'PREEMPT' and kr.ot_span is not None
    assert ap.forecast_calls == 1 and kr._prepass_done is True
    assert kr.last_prepass_ms is not None
    calls = len(p.calls)
    apply(kr, ap, v=8.0)                                        # 같은 틱 apply — 재시도 없음
    assert len(p.calls) == calls and kr._prepass_done is False
    assert kr.last_avoid['trigger'] == 'obb' and kr.last_avoid['shift'] == 'left'   # 진단 유지
    apply(kr, ap, v=8.0)                                        # 훅 없는 다음 틱 — 새 진단
    assert kr.last_avoid['state'] in ('SHIFT_ACTIVE', 'HANDLED')


def test_prepass_trigger_both_when_in_corridor_and_obb():
    c = car(2, 40.0, 0.0)
    kr, p, _ap = rig(actors=[c])
    ap = ObbAp(p, [c], hit_ids={2})
    kr.pre_pass(ap, p.route_points, [c], 12.5, 8.0)
    assert kr.last_avoid['trigger'] == 'both'


def test_prepass_skips_forecast_without_stopped_vehicles():
    c = car(2, 40.0, 2.5, speed=5.0)
    kr, p, _ap = rig(actors=[c])
    ap = ObbAp(p, [c], hit_ids={2})
    kr.pre_pass(ap, p.route_points, [c], 12.5, 8.0)
    assert ap.forecast_calls == 0 and kr.ot_span is None


def test_prepass_obb_switch_off_uses_corridor_only():
    c = car(2, 40.0, 2.5)
    kr, p, _ap = rig(cfg_with(prepass_obb_enable=False), actors=[c])
    ap = ObbAp(p, [c], hit_ids={2})
    kr.pre_pass(ap, p.route_points, [c], 12.5, 8.0)
    assert ap.forecast_calls == 0 and kr.ot_span is None
