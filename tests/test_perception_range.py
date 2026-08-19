"""
GT 객체 80 m / 30개 컷오프 처리 (공식 확인 사양).

목록에서 사라진 것이 소멸을 뜻하지 않는다는 게 핵심이다.
"""
import math
import pathlib

import pytest

from conftest import PARAMS_YAML
from hlfma.core.comm import OBJ_COUNT, build_frame, parse
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.perception import Perception
from hlfma.nodes.params import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml(PARAMS_YAML)

EGO = (508.79968, -168.28766, 42.0, 0.52727822, 0.0, 0.0)

pytestmark = pytest.mark.skipif(not GRAPH.exists(), reason='lane_graph.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


def obj(oid, dx, dy, speed=0.0, length=4.5, width=1.9, height=1.5):
    return (oid, EGO[0] + dx, EGO[1] + dy, 0.0, 0.0, speed, length, width, height)


def feed(per, objects, t):
    return per.update(parse(build_frame(EGO, objects, [(0, 0)]), t_recv=t))


def test_param_exists_and_is_80m():
    assert CFG['percep']['gt_range_m'] == pytest.approx(80.0)


def test_object_near_range_edge_is_dropped_not_coasted(lg):
    """80 m 근처에서 사라지면 '범위 밖으로 나간 것' — 외삽하지 않는다."""
    per = Perception(lg, None, CFG)
    feed(per, [obj(7, 78.0, 0.0)], 100.0)
    ws = feed(per, [], 100.05)
    assert [o.id for o in ws.objects] == []
    assert ws.flags.get('obj_left_range') == 1
    assert 'obj_coasting' not in ws.flags


def test_close_object_that_vanishes_is_coasted(lg):
    """가까이 있던 게 사라지면 가림일 수 있다 — coast_s 동안 유지한다."""
    per = Perception(lg, None, CFG)
    feed(per, [obj(7, 20.0, 0.0, speed=5.0)], 100.0)
    ws = feed(per, [], 100.05)
    assert [o.id for o in ws.objects] == [7]
    o = ws.objects[0]
    assert o.coasting is True
    assert o.age == pytest.approx(0.05, abs=1e-6)
    assert ws.flags.get('obj_coasting') == 1


def test_coasted_object_is_extrapolated(lg):
    """등속 외삽으로 위치가 굴러가야 한다."""
    per = Perception(lg, None, CFG)
    feed(per, [obj(7, 20.0, 0.0, speed=10.0)], 100.0)   # heading 0 -> +x
    x0 = 20.0 + EGO[0]
    ws = feed(per, [], 100.5)
    assert ws.objects[0].x == pytest.approx(x0 + 10.0 * 0.5, abs=0.2)


def test_coasting_expires_after_coast_s(lg):
    per = Perception(lg, None, CFG)
    feed(per, [obj(7, 20.0, 0.0)], 100.0)
    ws = feed(per, [], 100.0 + CFG['percep']['coast_s'] + 0.1)
    assert ws.objects == []
    assert ws.flags.get('obj_lost') == 1


def test_full_list_keeps_far_object_coasting(lg):
    """
    30칸이 꽉 차면 더 먼 객체가 밀려나 안 온다. 이건 소멸이 아니므로
    범위 가장자리라도 외삽을 유지해야 한다.
    """
    per = Perception(lg, None, CFG)
    feed(per, [obj(99, 78.0, 0.0)], 100.0)
    full = [obj(i, 1.0 + i * 0.5, 0.0) for i in range(1, OBJ_COUNT + 1)]
    ws = feed(per, full, 100.05)
    assert ws.flags.get('obj_list_full') is True
    assert 99 in [o.id for o in ws.objects], '목록 포화로 밀려난 객체는 유지해야 한다'
    assert ws.flags.get('obj_coasting') == 1


def test_flags_report_range_and_count(lg):
    per = Perception(lg, None, CFG)
    ws = feed(per, [obj(1, 10.0, 0.0), obj(2, 50.0, 3.0)], 100.0)
    assert ws.flags['obj_n'] == 2
    assert ws.flags['obj_max_dist'] == pytest.approx(math.hypot(50.0, 3.0), abs=0.1)
    assert 'obj_beyond_gt_range' not in ws.flags


def test_object_beyond_spec_range_is_flagged(lg):
    """80 m 를 넘는 객체가 오면 사양 재확인이 필요하다 — 플래그로 남긴다."""
    per = Perception(lg, None, CFG)
    ws = feed(per, [obj(1, 120.0, 0.0)], 100.0)
    assert ws.flags.get('obj_beyond_gt_range') == pytest.approx(120.0, abs=0.5)


def test_visibility_limit_matches_sight_distance():
    from hlfma.core.planner import Planner
    pl = Planner(None, None, CFG)
    expect = math.sqrt(2 * CFG['speed']['a_comf'] *
                       (CFG['percep']['gt_range_m'] - CFG['speed']['stop_gap_m']))
    assert pl.visibility_limit() == pytest.approx(expect)
    assert 15.0 < pl.visibility_limit() < 16.0     # 80 m 기준 약 15.4 m/s
