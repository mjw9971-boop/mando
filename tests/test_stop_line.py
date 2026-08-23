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
ROUTE = ROOT / 'data' / 'route_example.pkl'   # 고정 테스트 경로 (data/route.pkl 은 시나리오마다 바뀐다)
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


# ── 통과한 정지선 게이트 / 래치 freeze (2026-08-23 17:49 런) ────────────────
def test_no_candidate_for_passed_stop_line(lg, route):
    """앞범퍼가 지난 정지선으로는 정지 후보를 만들지 않는다.

    황색으로 딜레마존 통과 직후 적색으로 바뀌면 dist_stop_line 이 아직 양수
    (뒷바퀴축 기준)라 예전에는 교차로 한복판에서 급정거했다 (-3.94 m/s²).
    """
    pl = Planner(lg, route, CFG)
    # 앞범퍼가 정지선을 1 m 지난 지점 (뒷바퀴축은 아직 2.8 m 못 미침)
    s = STOP_S - FRONT + 1.0
    w = make_world(lg, route, s, speed=8.0, t=0.0, light=(3, RED))
    assert 0 < w.summ['dist_stop_line'] < FRONT, '전제: 뒷축은 못 미치고 앞범퍼는 지난 상태'
    d = pl.plan(w)
    assert 'stop_line' not in d.reasons
    assert not pl._stop_latch
    # 앞범퍼가 아직 못 미친 지점이면 정상적으로 후보가 생긴다
    d2 = Planner(lg, route, CFG).plan(
        make_world(lg, route, STOP_S - FRONT - 10.0, speed=8.0, t=0.0, light=(3, RED)))
    assert 'stop_line' in d2.reasons


def test_yellow_dilemma_pass_then_red_does_not_brake(lg, route):
    """황색 통과 → 적색 전환 시퀀스에서 급정거가 생기지 않는다."""
    pl = Planner(lg, route, CFG)
    v = 9.0
    prev = None
    for s in [STOP_S - FRONT - 25.0 + 2.0 * k for k in range(20)]:
        light = (3, 2) if s < STOP_S - FRONT else (3, 1)     # 정지선 통과 순간 적색
        d = pl.plan(make_world(lg, route, s, speed=v, t=s / v, light=light))
        if prev is not None:
            # 목표속도가 한 번에 무너지지 않는다 (급정거 유발 지점이 없다)
            assert d.v_target > 1.0 or prev <= 1.0, \
                f's={s:.1f} 에서 v_target 이 {prev:.2f} → {d.v_target:.2f} 로 붕괴'
        prev = d.v_target
    assert not pl._stop_latch


def test_latch_reference_is_frozen_and_releases(lg, route):
    """래치의 _stop_line_s 는 고정되고, 그 정지선을 5 m 지나면 해제된다."""
    pl = Planner(lg, route, CFG)
    at = STOP_S - GAP - FRONT
    pl.plan(make_world(lg, route, at, speed=0.5, t=0.0, light=(3, RED)))
    assert pl._stop_latch
    frozen = pl._stop_line_s
    assert frozen == pytest.approx(STOP_S, abs=0.5)
    # 다음 교차로 정지선이 lookahead 에 들어와도 기준이 바뀌지 않는다
    w = make_world(lg, route, at + 1.0, speed=0.5, t=0.2, light=(3, RED))
    w.summ = dict(w.summ, dist_stop_line=200.0)
    pl.plan(w)
    assert pl._stop_latch and pl._stop_line_s == pytest.approx(frozen)
    # 고정된 정지선을 5 m 지나면 해제
    d = pl.plan(make_world(lg, route, STOP_S + 6.0, speed=3.0, t=1.0, light=(3, RED)))
    assert not pl._stop_latch
    assert 'stop_line' not in d.reasons
