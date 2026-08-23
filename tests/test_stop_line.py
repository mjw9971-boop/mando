"""
적색 정지선 정지 — 3중 결함 회귀 (2026-08-23 12:51 런: 3/3 완전정지 실패, 앞범퍼 3~5 m 초과).

  (d) light 소멸은 래치 해제 사유가 아니다 — 해제는 녹색류 / RTOR go / 명백한 통과(+5 m)뿐
  (b) 정지 목표는 앞범퍼 기준: 유효거리 = dist − stop_gap − (wheelbase + front_overhang)
  (a) 제어 지연 보상 stop_lag_s: 목표를 v·lag 만큼 앞당긴다
"""
import math
import pathlib
import pickle

import pytest

from conftest import PARAMS_YAML
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.planner import STOP_LINE, Planner
from hlfma.nodes.params import load_params_yaml
from test_turn_signal import make_world

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route.pkl'
CFG = load_params_yaml(PARAMS_YAML)
RED, GREEN = 1, 3
STOP_S = 50.03                       # 도로 30 정지선 (route_s)
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
GAP = CFG['speed']['stop_gap_m']

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='lane_graph.pkl / route.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def route():
    return pickle.load(open(ROUTE, 'rb'))


def test_stop_target_is_front_bumper(lg, route):
    """뒷바퀴축이 정지선 − gap − front 에 있으면 목표 0, 그보다 앞(먼)이면 양수."""
    pl = Planner(lg, route, CFG)
    at_target = STOP_S - GAP - FRONT
    d = pl.plan(make_world(lg, route, at_target, speed=0.0, t=0.0, light=(3, RED)))
    assert d.reasons['stop_line'] == 0.0 and d.state == STOP_LINE
    pl2 = Planner(lg, route, CFG)
    d2 = pl2.plan(make_world(lg, route, at_target - 10.0, speed=0.0, t=0.0, light=(3, RED)))
    # 정지점 10 m 앞: sqrt(2·a_plan·10) (v=0 이라 lag 보상 0)
    a = CFG['speed']['a_comf'] * CFG['speed']['a_plan_factor']
    assert d2.reasons['stop_line'] == pytest.approx(math.sqrt(2 * a * 10.0), abs=0.02)


def test_lag_compensation_scales_with_speed(lg, route):
    pl = Planner(lg, route, CFG)
    s = STOP_S - GAP - FRONT - 20.0
    slow = pl.plan(make_world(lg, route, s, speed=0.0, t=0.0, light=(3, RED))).reasons['stop_line']
    fast = Planner(lg, route, CFG).plan(make_world(lg, route, s, speed=10.0, t=0.0, light=(3, RED))).reasons['stop_line']
    a = CFG['speed']['a_comf'] * CFG['speed']['a_plan_factor']
    lag = CFG['speed']['stop_lag_s'] * 10.0
    assert fast < slow
    assert fast == pytest.approx(math.sqrt(2 * a * (20.0 - lag)), abs=0.02)


def test_light_loss_does_not_release_latch_while_moving(lg, route):
    """래치 후 light 가 사라져도(교차로 진입) 감속 중이면 정지 유지."""
    pl = Planner(lg, route, CFG)
    at_target = STOP_S - GAP - FRONT
    pl.plan(make_world(lg, route, at_target - 0.5, speed=1.0, t=0.0, light=(3, RED)))
    assert pl._stop_latch
    d = pl.plan(make_world(lg, route, at_target + 0.5, speed=1.0, t=0.1, light=None))
    assert d.reasons['stop_line'] == 0.0 and d.reasons['rtor'] == 'hold: light lost'
    assert pl._stop_latch


def test_latch_released_by_green_or_clear_pass(lg, route):
    pl = Planner(lg, route, CFG)
    at_target = STOP_S - GAP - FRONT
    pl.plan(make_world(lg, route, at_target, speed=0.0, t=0.0, light=(3, RED)))
    assert pl._stop_latch
    d = pl.plan(make_world(lg, route, at_target, speed=0.0, t=0.1, light=(3, GREEN)))
    assert not pl._stop_latch and 'stop_line' not in d.reasons
    # 명백한 통과: 정지선 +5 m 를 넘어 light 없음 → 해제 (교차로 안에 서지 않는다)
    pl = Planner(lg, route, CFG)
    pl.plan(make_world(lg, route, at_target, speed=0.0, t=0.0, light=(3, RED)))
    d = pl.plan(make_world(lg, route, STOP_S + 6.0, speed=3.0, t=0.1, light=None))
    assert not pl._stop_latch and 'stop_line' not in d.reasons


def test_rtor_go_then_light_loss_releases(lg, route):
    """RTOR go 후 연결로 진입으로 light 가 사라지면 래치를 푼다 (교차로 안 재정지 금지)."""
    pl = Planner(lg, route, CFG)
    at_target = STOP_S - GAP - FRONT
    for t in (0.0, 0.1, 0.1 + CFG['signal']['stop_dwell_s'] + 0.05):
        d = pl.plan(make_world(lg, route, at_target, speed=0.0, t=t, light=(3, RED)))
    assert d.reasons['rtor'] == 'go'
    d = pl.plan(make_world(lg, route, at_target + 2.0, speed=3.0, t=2.0, light=None))
    assert not pl._stop_latch and 'stop_line' not in d.reasons
