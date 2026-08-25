"""
lanegraph 정지선 조회 — 옛 test_stop_line.py 에서 lanegraph 부분만 발췌 (phase1).

(정지 목표·래치·RTOR 등 판단 케이스는 자체 planner 삭제로 함께 제거 —
 phase3 이후 PDM-Lite 의 적신호 IDM 이 같은 역할을 하고, 검증은 실기 로그로 한다.)

route_example 기준 검증값: 도로 30 정지선은 경로 누적거리(route_s) 50.03 m 에
있고 신호(signal_ids)가 걸려 있다.
"""
import pathlib
import pickle

import pytest

from vtd_adapter.lanegraph import LaneGraph

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route_example.pkl'
STOP_S = 50.03                       # 도로 30 정지선 (route_s) — 실측 검증값

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='lane_graph.pkl / route_example.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def route():
    return pickle.load(open(ROUTE, 'rb'))


def test_stop_line_at_expected_route_s(lg, route):
    """경로 시작에서 본 첫 정지선이 50.03 m (도로 30 — 신호 매핑 없는 정지선)."""
    ahead = lg.lookahead(route, 0, float(route.get('start_s_in_lane', 0.0)), 200.0)
    stop = next(a for a in ahead if a.kind == 'stop_line')
    assert stop.dist == pytest.approx(STOP_S, abs=0.5)


def test_summarize_reports_first_stop_line(lg, route):
    ahead = lg.lookahead(route, 0, float(route.get('start_s_in_lane', 0.0)), 200.0)
    summ = lg.summarize(ahead)
    assert summ['dist_stop_line'] == pytest.approx(STOP_S, abs=0.5)


def test_signalized_stop_line_exposes_controllers(lg):
    """신호 걸린 정지선은 controller_ids 를 갖는다 — 9910 light_id 대조의 근거.

    (실측: 도로 72 진입부 정지선 signal_ids [1..6], controller [1,2])
    """
    from vtd_adapter.route import stop_line_controllers

    rec = lg.lanes[(72, 0, 4)]
    sl = rec['stop_lines'][0]
    assert sl['signal_ids'] and sl['controller_ids']

    class Item:                      # lookahead Ahead 흉내 (lane, s_in_lane 만 쓴다)
        lane = (72, 0, 4)
        s_in_lane = sl['s']

    assert stop_line_controllers(lg, Item()) == list(sl['controller_ids'])


def test_stop_line_distance_decreases_with_progress(lg, route):
    """5 m 전진하면 정지선 거리도 5 m 줄어든다 (누적거리 계산 회귀)."""
    s0 = float(route.get('start_s_in_lane', 0.0))
    d0 = lg.summarize(lg.lookahead(route, 0, s0, 200.0))['dist_stop_line']
    d5 = lg.summarize(lg.lookahead(route, 0, s0 + 5.0, 200.0))['dist_stop_line']
    assert d0 - d5 == pytest.approx(5.0, abs=0.05)
