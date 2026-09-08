"""
ctrl24 커밋 4 — 중첩 시프트 (C).

여기서 지키는 불변:
  · 시프트 활성 중 **밀린 경로 회랑**에 정지 객체가 있으면 같은 방향으로 한 칸 더 (목표 =
    현재 차로의 이웃 — 밀림은 새 span 플래토에서 잰다). 횟수 제한 없음.
  · ot_span 은 합집합. 복귀는 원 경로로 — 새 끝이 옛 끝을 넘으면 복귀 전이는 trans·√(D/D1).
  · 같은 객체에는 다시 시도하지 않는다 (요동 방지). 두 번째 칸에 이웃이 없으면 NOOP →
    ot_span 은 그대로, 자차는 P1/P2 가 세우는 대로 선다.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from test_ctrl24_avoid import LANE, apply, car, rig            # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']


def test_nested_shift_one_more_lane_with_union_span_and_scaled_return():
    a_car = car(2, 60.0)
    kr, p, ap = rig(actors=[a_car])
    apply(kr, ap, v=8.0)
    a1, b1 = kr.ot_span
    trans = C['trans_k'] * 8.0
    # 밀린 차로(y=+3) 플래토 위 65 m 에 정지 차량 → 밀린 경로 회랑 안
    b_car = car(3, 65.0, LANE)
    ap._world.get_actors().append(b_car)
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['state'] == 'SHIFT_NESTED' and a['shift'] == 'left' and a['blocker'] == 3
    assert a['steps'] == 2 and a['nested'] == 1 and kr.nested == 1
    a2, b2 = a['span_new']
    assert kr.ot_span == (min(a1, a2), max(b1, b2)) and b2 > b1
    assert a['back_m'] == pytest.approx(trans * math.sqrt(2.0), abs=0.05)
    assert p.route_points[a2 + int(trans * 10) + 10, 1] == pytest.approx(2 * LANE, abs=0.05)   # 새 플래토 = 2칸
    assert p.route_points[kr.ot_span[1] - 1, 1] == pytest.approx(0.0, abs=0.05)   # 원 경로로 복귀
    assert kr._shifted_for == {2, 3}


def test_nested_shift_ending_inside_first_span_keeps_plain_return():
    a_car = car(2, 70.0)
    kr, p, ap = rig(actors=[a_car])
    apply(kr, ap, v=3.0)
    a1, b1 = kr.ot_span
    b_car = car(3, 65.0, LANE)                                   # 플래토 안, 새 끝이 옛 끝 안쪽
    ap._world.get_actors().append(b_car)
    apply(kr, ap, v=3.0)
    a = kr.last_avoid
    assert a['state'] == 'SHIFT_NESTED' and a['steps'] == 2
    assert a['span_new'][1] <= b1 and a['back_m'] == pytest.approx(a['trans_m'])
    assert kr.ot_span == (min(a1, a['span_new'][0]), b1)


def test_nested_noop_when_second_neighbour_missing_keeps_first_span():
    kr, p, ap = rig(actors=[car(2, 60.0)], right=False)
    apply(kr, ap, v=8.0)
    span1 = kr.ot_span
    p.has['left'] = False                                        # 두 번째 칸에는 이웃이 없다
    ap._world.get_actors().append(car(3, 65.0, LANE))
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['state'] == 'NOOP' and a['rejects'] == ['left:noop', 'right:noop']
    assert kr.ot_span == span1 and kr.nested == 0 and 3 not in kr._shifted_for
    assert kr._corridor and kr._corridor[0][3].id == 3            # P1/P2 대상으로 남는다


def test_no_retry_for_same_object_but_new_object_is_handled():
    kr, p, ap = rig(actors=[car(2, 60.0)])
    apply(kr, ap, v=8.0)
    n = len(p.calls)
    for _ in range(3):
        apply(kr, ap, v=8.0)
    assert len(p.calls) == n                                     # id 2 재시도 없음
    ap._world.get_actors().append(car(3, 65.0, LANE))
    apply(kr, ap, v=8.0)
    assert len(p.calls) == n + 1 and kr.nested == 1


def test_restore_clears_nesting_and_shift_set():
    kr, p, ap = rig(actors=[car(2, 60.0), car(3, 65.0, LANE)])
    apply(kr, ap, v=8.0)
    apply(kr, ap, v=8.0)
    assert kr.nested == 1
    p.route_index = kr.ot_span[1] + 1
    apply(kr, ap, v=8.0)
    assert kr.ot_span is None and kr.nested == 0 and not kr._shifted_for
    assert np.array_equal(p.route_points, p.original_route_points)
