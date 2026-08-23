"""
RTOR(적신호 우회전) + 신호 폴백.

적색 + 다음 회전이 우회전: 정지선 앞 완전 정지 → stop_dwell_s 유지 → 통과 경로/횡단보도
무객체·접근 TTC 여유 → rtor_speed_kph 서행. 직진·좌회전은 녹색 대기.
"""
import copy
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
ROUTE = ROOT / 'data' / 'route.pkl'
CFG = load_params_yaml(PARAMS_YAML)
RED, GREEN = 1, 3
RTOR_V = CFG['signal']['rtor_speed_kph'] / 3.6
DWELL = CFG['signal']['stop_dwell_s']

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='lane_graph.pkl / route.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def route():
    return pickle.load(open(ROUTE, 'rb'))


def obj(oid, s_rel, lat_off=0.0, on_route=True, ttc=float('inf'), enter=False):
    return TrackedObject(id=oid, x=0.0, y=0.0, heading=0.0, speed=0.0, length=0.6, width=0.6,
                         height=1.7, cls='pedestrian', lane=None, on_route=on_route, s_rel=s_rel,
                         lat_off=lat_off, v_rel=0.0, ttc=ttc, will_enter_lane=enter, age=0.0,
                         coasting=False)


def at_line(lg, route, t, speed=0.0, objects=(), light=(3, RED)):
    """도로 30 정지선(50.03 m) stop_gap 앞에 선 자차. 정지선 signal_ids 는 비어 있다(폴백 경로)."""
    w = make_world(lg, route, 49.0, speed=speed, t=t, light=light, objects=objects)
    assert w.summ['next_turn'] == 'turn_right' and not w.summ['stop_signal_ids']
    return w


def test_red_right_turn_no_objects_goes_after_dwell(lg, route):
    pl = Planner(lg, route, CFG)
    d0 = pl.plan(at_line(lg, route, t=0.0))           # 이 틱에서 정지 래치
    assert d0.v_target == 0.0 and d0.reasons['winner'] == 'stop_line'
    assert d0.reasons.get('signal_fallback') is True
    d1 = pl.plan(at_line(lg, route, t=0.1))           # 완전 정지 확인 → dwell 시작
    assert d1.v_target == 0.0 and d1.reasons['rtor'].startswith('dwell')
    d1 = pl.plan(at_line(lg, route, t=0.1 + DWELL * 0.5))
    assert d1.v_target == 0.0 and d1.reasons['rtor'].startswith('dwell')
    d2 = pl.plan(at_line(lg, route, t=0.1 + DWELL + 0.05))
    assert d2.reasons['rtor'] == 'go'
    # RTOR 는 정지선 후보를 서행 상한으로 바꾼다. 최종 v_target 은 다른 후보(회전 곡률)와의 min.
    assert d2.reasons['stop_line'] == pytest.approx(RTOR_V, abs=1e-6)
    assert 0.0 < d2.v_target <= RTOR_V + 1e-6


def test_red_right_turn_with_pedestrian_holds(lg, route):
    pl = Planner(lg, route, CFG)
    ped = [obj(7, s_rel=6.0)]                       # 통과 경로(연결로) 위
    pl.plan(at_line(lg, route, t=0.0, objects=ped))
    pl.plan(at_line(lg, route, t=0.1, objects=ped))
    d = pl.plan(at_line(lg, route, t=DWELL + 0.2, objects=ped))
    assert d.v_target == 0.0 and d.reasons['rtor'].startswith('hold')
    # 보행자가 사라지면 매 틱 재평가해 통과
    d = pl.plan(at_line(lg, route, t=DWELL + 0.3))
    assert d.reasons['rtor'] == 'go'


def test_crosswalk_and_ttc_block(lg, route):
    pl = Planner(lg, route, CFG)
    pl.plan(at_line(lg, route, t=0.0))
    pl.plan(at_line(lg, route, t=0.1))
    assert pl.plan(at_line(lg, route, t=DWELL + 0.2)).reasons['rtor'] == 'go'
    w = at_line(lg, route, t=DWELL + 0.2)
    dcw = w.summ['dist_crosswalk']
    cw = obj(8, s_rel=dcw + 1.0, lat_off=5.0, on_route=False)      # 횡단보도 폴리곤 안, 옆 차로
    assert pl.plan(at_line(lg, route, t=DWELL + 0.2, objects=[cw])).reasons['rtor'].startswith('hold')
    car = obj(9, s_rel=-20.0, lat_off=6.0, on_route=False, ttc=2.0)  # 좌측 접근, TTC 2 s
    assert pl.plan(at_line(lg, route, t=DWELL + 0.3, objects=[car])).reasons['rtor'].startswith('hold')
    walker = obj(10, s_rel=3.0, lat_off=4.0, on_route=False, enter=True)   # 진입 예측
    assert pl.plan(at_line(lg, route, t=DWELL + 0.4, objects=[walker])).reasons['rtor'].startswith('hold')


def test_red_straight_waits_for_green(lg, route):
    pl = Planner(lg, route, CFG)
    for t in (0.0, DWELL + 0.1, DWELL + 5.0):
        w = at_line(lg, route, t=t)
        w.summ['next_turn'] = 'turn_left'          # 좌회전/직진 경로로 가정
        d = pl.plan(w)
        assert d.v_target == 0.0 and 'rtor' not in d.reasons
    w = at_line(lg, route, t=DWELL + 6.0, light=(3, GREEN))
    w.summ['next_turn'] = 'turn_left'
    assert 'stop_line' not in pl.plan(w).reasons


def test_rtor_disabled_waits(lg, route):
    cfg = copy.deepcopy(CFG)
    cfg['signal']['rtor_enabled'] = False
    pl = Planner(lg, route, cfg)
    pl.plan(at_line(lg, route, t=0.0))
    d = pl.plan(at_line(lg, route, t=DWELL + 5.0))
    assert d.v_target == 0.0 and 'rtor' not in d.reasons


def test_fallback_skipped_when_ctrl_mismatch_or_id0(lg, route):
    pl = Planner(lg, route, CFG)
    w = make_world(lg, route, 20.0, speed=8.0, light=(0, RED))
    assert 'stop_line' not in pl.plan(w).reasons
    w = make_world(lg, route, 20.0, speed=8.0, light=(3, RED))
    w.flags['light_ctrl_match'] = False
    assert 'stop_line' not in Planner(lg, route, CFG).plan(w).reasons
