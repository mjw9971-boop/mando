"""
계획 차선변경의 점등 선행 조건 + 창 확장/lead_short + 신호 폴백.

2026-08-23 11:57 런: 회전1 연결로 끝 = LC1 창 시작이라 회전 중 RIGHT 가 켜져 있다가
창 진입과 동시에 LEFT 점등·즉시 실행 → lead 0.0 s. 같은 런 시작 교차로에서
정지선 signal_ids 가 비어 적색(3,1)을 비신호로 보고 통과 → S5.1.01.

계약:
  · 계획 LC 는 해당 방향 지시등이 signal.lc_lead_min_s(3 s) 이상 **연속** 점등 후 실행
  · 창 안이어도 미달이면 깜빡이만 켠 채 원차로 대기
  · 대기로 남은 창 < 전이거리면 창 확장 시도, 불가하면 즉시 실행 + lead_short
  · shield 긴급 회피(emergency_avoid)는 선행 조건 면제
  · 정지선 signal_ids 가 비어도 9910 light 가 오면 그 state 를 따른다(폴백)
"""
import pathlib
import pickle

import pytest

from conftest import PARAMS_YAML
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.planner import FOLLOW, LANE_CHANGE, TURN_LEFT, TURN_RIGHT, Planner
from hlfma.nodes.params import load_params_yaml
from test_turn_signal import lcs, make_world, sig_of

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route.pkl'
CFG = load_params_yaml(PARAMS_YAML)
LEAD_MIN = float(CFG['signal']['lc_lead_min_s'])

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='lane_graph.pkl / route.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def route():
    return pickle.load(open(ROUTE, 'rb'))


def drive(pl, lg, route, s0, s1, v, step=0.5, lane=None):
    """s0→s1 등속 주행 틱 열. [(route_s, Decision)]"""
    out = []
    s = s0
    while s <= s1:
        out.append((s, pl.plan(make_world(lg, route, s, speed=v, lane=lane))))
        s += step
    return out


def test_lc_waits_for_lead_after_turn(lg, route):
    """LC1: 회전1 중 RIGHT → 창 진입 시 LEFT 점등, 3 s 연속 점등 전에는 실행하지 않는다."""
    ev = lcs(route)[0]
    assert sig_of(ev['kind']) == TURN_LEFT
    v = 7.0
    pl = Planner(lg, route, CFG)
    ticks = drive(pl, lg, route, ev['window_s0'] - 30.0, ev['window_s1'], v)
    first_left = next(s for s, d in ticks if d.turn_signal == TURN_LEFT)
    start = next((s for s, d in ticks if d.state == LANE_CHANGE), None)
    assert start is not None, 'LC1 이 실행되지 않았다'
    lead_s = (start - first_left) / v
    # 창(41 m)이 3 s 대기 + 전이거리(21 m)를 못 담으므로 lead_short 로 당겨 실행된다.
    # 그래도 즉시(0 s)가 아니라 창이 허락하는 만큼 기다려야 한다.
    d_start = next(d for s, d in ticks if s == start)
    assert d_start.reasons.get('sig_lead_s', 0) >= 1.0
    assert lead_s >= 1.0
    assert 'lead_short' in d_start.reasons or d_start.reasons['sig_lead_s'] >= LEAD_MIN
    # 대기 중 원차로 직진 + 깜빡이 유지
    waiting = [d for s, d in ticks if first_left <= s < start]
    assert all(d.state == FOLLOW and d.turn_signal == TURN_LEFT for d in waiting)


def test_lc_with_long_lead_unchanged(lg, route):
    """LC2: lead 가 이미 3 s 를 넘으므로 창 진입 즉시 실행 (동작 불변)."""
    ev = lcs(route)[1]
    v = 12.0
    pl = Planner(lg, route, CFG)
    ticks = drive(pl, lg, route, ev['window_s0'] - 80.0, ev['window_s0'] + 10.0, v)
    start = next((s for s, d in ticks if d.state == LANE_CHANGE), None)
    assert start is not None
    assert start - ev['window_s0'] <= 1.0
    d_start = next(d for s, d in ticks if s == start)
    assert d_start.reasons['sig_lead_s'] >= LEAD_MIN
    assert 'lead_short' not in d_start.reasons


def test_emergency_avoid_skips_lead(lg, route):
    ev = lcs(route)[0]
    pl = Planner(lg, route, CFG)
    pl.emergency_avoid = True
    d = pl.plan(make_world(lg, route, ev['window_s0'] + 1.0, speed=7.0))
    assert d.state == LANE_CHANGE
    assert d.reasons.get('lc_lead_exempt') is True


def test_window_extension_helper(lg, route):
    """LC1 원차로 (72,0,-3) 는 successor 가 없어 확장 불가 → None."""
    pl = Planner(lg, route, CFG)
    ev = lcs(route)[0]
    assert pl._try_extend_window(ev, 'left', need=10.0) is None


def test_red_light_with_unlinked_stop_line_stops(lg, route):
    """정지선 signal_ids 가 비어도 9910 light 가 적색이면 정지 후보가 나온다."""
    pl = Planner(lg, route, CFG)
    w = make_world(lg, route, 20.0, speed=8.0)
    assert w.summ['dist_stop_line'] is not None and not w.summ['stop_signal_ids']
    w.light = (3, 1)                                       # RED
    d = pl.plan(w)
    assert 'stop_line' in d.reasons and d.reasons.get('signal_fallback') is True
    assert d.reasons['stop_line'] < d.reasons['limit']
    w.light = (3, 3)                                       # GREEN → 통과
    d = Planner(lg, route, CFG).plan(w)
    assert 'stop_line' not in d.reasons
    w.light = None                                         # light 없음 → 비신호 정지선
    d = Planner(lg, route, CFG).plan(w)
    assert 'stop_line' not in d.reasons
