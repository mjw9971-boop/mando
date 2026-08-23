"""
정차 차량 추월 (blocked 판정 + 합성 차선변경).

2026-08-23 18:33 런: 정차 차량 뒤 605 m 에서 갇혀 완주 불가.

계약 (전부 만족해야 blocked):
  · 선행차 속도 < 0.5 m/s
  · 전방에 **활성 신호 정지 후보 없음** — 신호 대기 중 추월은 즉시 실격
  · 교차로 내부 아님
  · 앞범퍼 기준 전방 정지선이 lead.blocked_ignore_stopline_m 보다 멀거나 없음
  · 위가 lead.blocked_dwell_s 이상 연속
"""
import pathlib
import pickle

import pytest

from conftest import PARAMS_YAML
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.planner import Planner
from hlfma.core.types import TrackedObject
from hlfma.nodes.params import load_params_yaml
from test_turn_signal import make_world

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route_example.pkl'
CFG = load_params_yaml(PARAMS_YAML)
DWELL = CFG['lead']['blocked_dwell_s']
IGN = CFG['lead']['blocked_ignore_stopline_m']
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
RED, GREEN = 1, 3

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='lane_graph.pkl / route_example.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def route():
    return pickle.load(open(ROUTE, 'rb'))


def stopped(oid=2, s_rel=12.0, speed=0.0, lane=None):
    return TrackedObject(id=oid, x=0.0, y=0.0, heading=0.0, speed=speed, length=4.4, width=1.8,
                         height=1.4, cls='vehicle', lane=lane, on_route=True, s_rel=s_rel,
                         lat_off=0.0, v_rel=0.0, ttc=float('inf'), will_enter_lane=False,
                         age=0.0, coasting=False)


def feed(pl, lg, route, s, t, objects, light=None, summ_over=None):
    w = make_world(lg, route, s, speed=0.0, t=t, light=light, objects=objects)
    if summ_over:
        w.summ = dict(w.summ, **summ_over)
    return pl.plan(w)


def far_from_stopline(lg, route):
    """전방 정지선이 blocked_ignore_stopline_m 보다 먼 route_s 를 찾는다."""
    for s in range(0, 40):
        w = make_world(lg, route, float(s), speed=0.0, t=0.0)
        d = w.summ.get('dist_stop_line')
        if d is None or (d - FRONT) > IGN:
            return float(s)
    pytest.skip('정지선에서 충분히 먼 지점이 없다')


def test_blocked_after_dwell(lg, route):
    """정차 선행차 + 신호 무관 + 정지선 멀면 dwell 후 blocked."""
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    d0 = feed(pl, lg, route, s, 0.0, [stopped()])
    assert d0.reasons['blocked'] is False, 'dwell 전에는 아니다'
    assert 'dwell' in d0.reasons['blocked_why']
    d1 = feed(pl, lg, route, s, DWELL + 0.1, [stopped()])
    assert d1.reasons['blocked'] is True


def test_moving_lead_is_not_blocked(lg, route):
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    feed(pl, lg, route, s, 0.0, [stopped(speed=3.0)])
    d = feed(pl, lg, route, s, DWELL + 0.1, [stopped(speed=3.0)])
    assert d.reasons['blocked'] is False
    assert 'lead 주행' in d.reasons['blocked_why']


def test_signal_stop_blocks_overtake(lg, route):
    """★ 신호 정지 후보가 있으면 절대 blocked 가 아니다 (신호 대기 추월 = 실격)."""
    pl = Planner(lg, route, CFG)
    # 정지선 앞 + 적색 → stop_line 후보 활성
    s = 20.0
    for t in (0.0, DWELL + 0.1, DWELL + 5.0):
        d = feed(pl, lg, route, s, t, [stopped()], light=(3, RED))
        assert 'stop_line' in d.reasons
        assert d.reasons['blocked'] is False
        assert '신호' in d.reasons['blocked_why']


def test_near_stopline_is_not_blocked(lg, route):
    """정지선이 가까우면(신호 대기 행렬 가능) 추월하지 않는다."""
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    over = {'dist_stop_line': FRONT + IGN - 5.0, 'stop_signal_ids': []}
    for t in (0.0, DWELL + 0.1):
        d = feed(pl, lg, route, s, t, [stopped()], summ_over=over)
    assert d.reasons['blocked'] is False
    assert '정지선' in d.reasons['blocked_why']


def test_in_junction_is_not_blocked(lg, route):
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    for t in (0.0, DWELL + 0.1):
        d = feed(pl, lg, route, s, t, [stopped()], summ_over={'in_junction': True})
    assert d.reasons['blocked'] is False
    assert '교차로' in d.reasons['blocked_why']


def test_dwell_resets_on_condition_break(lg, route):
    """조건이 한 번 깨지면 dwell 이 처음부터 다시 센다."""
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    feed(pl, lg, route, s, 0.0, [stopped()])
    feed(pl, lg, route, s, DWELL - 0.2, [stopped(speed=3.0)])      # 조건 깨짐
    d = feed(pl, lg, route, s, DWELL + 0.1, [stopped()])
    assert d.reasons['blocked'] is False


def test_blocked_gap_is_wider(lg, route):
    """blocked 이면 조향 여유를 위해 lead.blocked_gap_m 만큼 더 멀리 선다."""
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    feed(pl, lg, route, s, 0.0, [stopped(s_rel=40.0)])
    d = feed(pl, lg, route, s, DWELL + 0.1, [stopped(s_rel=40.0)])
    assert d.reasons['blocked'] is True
    pl2 = Planner(lg, route, CFG)
    d2 = feed(pl2, lg, route, s, 0.0, [stopped(s_rel=40.0)])       # blocked 아님
    assert d.reasons['lead'] < d2.reasons['lead'], '더 넓은 간격 = 더 낮은 목표속도'
