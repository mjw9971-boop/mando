"""span 연장 (span_extend_enable) — 2026-09-03 배치 연쇄장애물_01.

활성 span 의 끝 뒤에 standoff 차단물이 있고 막힌 채 정지가 이어지면, 자차 인덱스에서
shift_route_smoothly 를 같은 방향으로 다시 불러 끝만 뒤로 민다. 지키는 것:
  · 기본 off = 이전 동작 (연장 없음, 진단 키 없음).
  · 네 조건이 다 맞을 때만 연장. 하나라도 빠지면 연장 없음.
  · 연장 후 자차 뒤·발밑 경로점은 그대로, ot_span 시작은 그대로, 끝만 커진다.
  · ④ 예상 이격 미달이면 연장하지 않고 extend_skip 에 수치를 남긴다 (기각 아님).
  · 횟수 상한·경로 끝·정지 구역 가드. 목 플래너에 shift_route_smoothly 가 없으면 건너뛴다.
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
D = 3.0
TRANS = OT['transition_m']
STUCK = int(round(OT['span_extend_stuck_s'] * HZ))


class Car(Box):
    def __init__(self, oid, x, y, length=2.08, width=0.85):
        super().__init__(oid, x, y, 0.0, half_w=width / 2.0)
        self.bounding_box.extent.x = length / 2.0
        self.yaw_deg = 0.0


class ExtPlanner(Planner):
    """x 축 직선 경로, 우측 차로 y=−D. plan/shift 는 route.py 와 같은 식(이웃 = 원 경로 −D)."""

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
        li = fi if last_actor is None else int(round(last_actor.get_location().x * ppm))
        lext = ext if last_actor is None else last_actor.bounding_box.extent.x
        a = fi - int(ext * ppm + transition_length + extra_length_before)
        b = li + int(lext * ppm + transition_length + extra_length_after)
        floor = self.route_index + int(min_start_ahead)
        if a < floor:
            a = min(floor, b - 1)
        return a, b, obstacle_direction == 'right'

    def planned_lateral_offsets(self, a, b, left, step_pts=10):
        n = max(1, (int(b) - int(a)) // max(1, step_pts))
        return np.full(n, D if left else -D)

    def shift_route_smoothly(self, start_index, end_index, shift_to_left_lane,
                             transition_length=120.0, lane_transition_factor=1.0):
        self.calls.append((start_index, end_index, shift_to_left_lane, transition_length))
        L = float(transition_length)
        for idx in range(int(start_index), int(end_index)):
            loc = self.original_route_points[idx].copy()
            loc[1] = D if shift_to_left_lane else -D
            f = 1.0
            if idx <= start_index + L and idx - start_index < end_index - idx:
                f = -np.cos((idx - start_index) / L * np.pi) / 2 + 0.5
            elif idx >= end_index - L:
                f = -np.cos((end_index - idx) / L * np.pi) / 2 + 0.5
            self.route_points[idx] = f * loc + (1 - f) * self.route_points[idx]
            d = float(np.linalg.norm(self.route_points[idx][:2] - self.original_route_points[idx][:2]))
            self.lat_shift[idx] = self._lat_build[idx] + (d if shift_to_left_lane else -d)

    def shift_route_around_actors(self, first_actor, last_actor=None, obstacle_direction='right',
                                  transition_length=120.0, lane_transition_factor=1.0,
                                  extra_length_before=0.0, extra_length_after=0.0,
                                  min_start_ahead=0):
        a, b, left = self.plan_shift_span(first_actor, last_actor, obstacle_direction,
                                          transition_length, lane_transition_factor,
                                          extra_length_before, extra_length_after, min_start_ahead)
        self.shift_route_smoothly(a, b, left, transition_length, lane_transition_factor)
        return a, b


class NoSmoothPlanner(ExtPlanner):
    """shift_route_smoothly 가 없는 목 — getattr 가드 검증."""
    shift_route_smoothly = None

    def shift_route_around_actors(self, first_actor, last_actor=None, obstacle_direction='right',
                                  transition_length=120.0, lane_transition_factor=1.0,
                                  extra_length_before=0.0, extra_length_after=0.0,
                                  min_start_ahead=0):
        a, b, _ = self.plan_shift_span(first_actor, last_actor, obstacle_direction,
                                       transition_length, lane_transition_factor,
                                       extra_length_before, extra_length_after, min_start_ahead)
        return a, b


def on_cfg():
    c = copy.deepcopy(CFG)
    c['overtake']['span_extend_enable'] = True
    return c


def rig(cfg, objs, planner_cls=ExtPlanner):
    p = planner_cls(d_tl=float('inf'))
    p.lg = LgOne()
    kr = KrRules(cfg)
    kr._sl_all = []
    ap = Ap(p, actors=list(objs))
    ap._kr_ego_lane = (1, 0, -1)
    for _ in range(int(cfg['overtake']['wait_before_shift_s'] * HZ) + 5):
        kr._update_obj_timers(ap)
    return kr, p, ap


def make_shift(kr, p, ap):
    """첫 시프트(표적 22.4 m)를 만든다 → ot_span 활성."""
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None and kr.ot_side == 'right'
    return kr.ot_span


def advance_into_span(kr, p, ap, rel_m):
    """자차를 span 안 rel_m 지점으로 옮기고 그 위치 경로점을 기억."""
    p.route_index = int(rel_m * 10)
    ap._vehicle._x = float(rel_m)


def stuck_ticks(kr, p, ap, n, ego_speed=0.0):
    """n 틱 진행. 연장 진단은 그 틱의 reasons.avoid 에만 실리므로(다음 틱은 새 dict)
    'extend' 가 찍힌 틱의 last_avoid 를 돌려준다 (없으면 None)."""
    hit = None
    for _ in range(n):
        kr._update_obj_timers(ap)
        kr._tick_cache(ap, p)
        kr._try_overtake(ap, p, ego_speed)
        if kr.last_avoid and 'extend' in kr.last_avoid:
            hit = dict(kr.last_avoid)
    return hit


def test_params_present():
    assert OT['span_extend_stuck_s'] == 4.0 and OT['span_extend_max_n'] == 3
    assert 'span_extend_enable' in OT


def test_off_no_extension_and_no_keys():
    # 표적 22.4 뒤 span 끝(≈47.4) 너머 id7(60) — 조건은 맞지만 스위치 off
    kr, p, ap = rig(CFG, [Car(2, 22.4, 0.0), Car(7, 60.0, 0.0)])
    kr.span_extend = False
    a, b = make_shift(kr, p, ap)
    advance_into_span(kr, p, ap, 36.0)
    snap = p.route_points.copy()
    stuck_ticks(kr, p, ap, STUCK + 5)
    assert kr.ot_span == (a, b)
    assert np.array_equal(p.route_points, snap)
    assert 'extend' not in kr.last_avoid and 'extend_skip' not in kr.last_avoid


def test_extends_when_all_four_hold_and_keeps_footprint():
    kr, p, ap = rig(on_cfg(), [Car(2, 22.4, 0.0), Car(7, 60.0, 0.0)])
    a, b = make_shift(kr, p, ap)
    assert b < 600                                     # id7(60 m) 은 span 밖
    advance_into_span(kr, p, ap, 36.0)
    i0 = p.route_index
    before = p.route_points.copy()
    stuck_ticks(kr, p, ap, STUCK - 1)                  # ③ 아직 미달
    assert kr.ot_span == (a, b) and 'extend' not in kr.last_avoid
    hit = stuck_ticks(kr, p, ap, 2)                    # ③ 충족 → 연장
    assert kr.ot_span[0] == a and kr.ot_span[1] > b
    assert kr.ot_span[1] > 600                         # id7 을 덮는다
    assert kr.span_extend_n == 1
    assert hit is not None, '연장 틱의 진단이 없다'
    ex = hit['extend']
    assert ex['span_before'] == [a, b] and ex['span_after'] == list(kr.ot_span)
    assert 7 in ex['chain'] and ex['clear'] >= CFG['percep']['obstacle_clearance_m']
    # 자차 뒤·발밑은 그대로, 앞쪽만 바뀐다
    assert np.array_equal(p.route_points[:i0 + 1], before[:i0 + 1])
    assert not np.array_equal(p.route_points[i0 + 1:], before[i0 + 1:])
    assert p.calls[-1][0] == i0 and p.calls[-1][2] is False      # 자차 인덱스에서 우측(같은 방향)


def test_no_extension_when_blocker_inside_span():
    kr, p, ap = rig(on_cfg(), [Car(2, 22.4, 0.0), Car(7, 60.0, 0.0)])
    a, b = make_shift(kr, p, ap)
    # 차단물(id7) 이 span 안에 있도록 span 끝을 임의로 늘려 둔다
    kr.ot_span = (a, 700)
    advance_into_span(kr, p, ap, 36.0)
    stuck_ticks(kr, p, ap, STUCK + 5)
    assert kr.ot_span == (a, 700) and 'extend' not in kr.last_avoid


def test_clearance_fail_skips_with_diag():
    """연장 경로(우측 차로) 위에 id5 가 서 있으면 ④ 미달 → 연장 안 함, 진단만."""
    kr, p, ap = rig(on_cfg(), [Car(2, 22.4, 0.0), Car(7, 60.0, 0.0), Car(5, 50.0, -D)])
    a, b = make_shift(kr, p, ap)
    advance_into_span(kr, p, ap, 36.0)
    snap = p.route_points.copy()
    stuck_ticks(kr, p, ap, STUCK + 5)
    assert kr.ot_span == (a, b)
    assert np.array_equal(p.route_points, snap)
    sk = kr.last_avoid['extend_skip']
    assert sk['reason'] == 'clearance' and sk['worst_id'] == 5
    assert sk['clear'] < CFG['percep']['obstacle_clearance_m']
    assert 'extend' not in kr.last_avoid


def test_max_n_guard():
    cfg = on_cfg()
    cfg['overtake']['span_extend_max_n'] = 0
    kr, p, ap = rig(cfg, [Car(2, 22.4, 0.0), Car(7, 60.0, 0.0)])
    a, b = make_shift(kr, p, ap)
    advance_into_span(kr, p, ap, 36.0)
    stuck_ticks(kr, p, ap, STUCK + 5)
    assert kr.ot_span == (a, b) and kr.last_avoid['extend_skip']['reason'] == 'max_n'


def test_mock_without_shift_route_smoothly_is_skipped():
    kr, p, ap = rig(on_cfg(), [Car(2, 22.4, 0.0), Car(7, 60.0, 0.0)], planner_cls=NoSmoothPlanner)
    a, b = make_shift(kr, p, ap)
    advance_into_span(kr, p, ap, 36.0)
    stuck_ticks(kr, p, ap, STUCK + 5)                  # AttributeError 없이
    assert kr.ot_span == (a, b) and kr.last_avoid['extend_skip']['reason'] == 'no_planner'


def test_restore_resets_side_and_count():
    kr, p, ap = rig(on_cfg(), [Car(2, 22.4, 0.0), Car(7, 60.0, 0.0)])
    make_shift(kr, p, ap)
    advance_into_span(kr, p, ap, 36.0)
    stuck_ticks(kr, p, ap, STUCK + 5)
    assert kr.span_extend_n == 1
    kr._restore_span(p)
    assert kr.ot_span is None and kr.ot_side is None and kr.span_extend_n == 0
