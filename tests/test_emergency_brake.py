"""
TTC 비상제동 (shield._emergency_brake) — **최후 방어**.

정상 감속(정지선 / 횡단보도 보행자 / 선행차)이 전부 실패해 실제 충돌이 임박한
경우에만 걸린다.

계약:
  · min TTC < ttc.emergency_s(1.5) → v_target=0, state=E_STOP, reasons['shield'] 기록
  · Control 은 E_STOP 에서 speed.a_emergency(-8.0) 를 그대로 낸다 (저크·a_min 무시)
  · 대상은 on_route 객체 + will_enter_lane 예측 객체뿐
  · 해제는 TTC > ttc.brake_s(2.5) 회복 시 (히스테리시스 — 경계에서 껌뻑이지 않는다)
"""
import pathlib
import pickle

import pytest

from conftest import PARAMS_YAML
from hlfma.core.control import Control
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.planner import Planner
from hlfma.core.shield import Shield
from hlfma.core.types import TrackedObject
from hlfma.nodes.params import load_params_yaml
from test_turn_signal import make_world

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route_example.pkl'   # 고정 테스트 경로 (data/route.pkl 은 시나리오마다 바뀐다)
CFG = load_params_yaml(PARAMS_YAML)
EMERGENCY_S = CFG['ttc']['emergency_s']
BRAKE_S = CFG['ttc']['brake_s']
A_EMERGENCY = CFG['speed']['a_emergency']

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='lane_graph.pkl / route.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def route():
    return pickle.load(open(ROUTE, 'rb'))


def obj(ttc, on_route=True, enter=False, oid=42, s_rel=8.0):
    return TrackedObject(id=oid, x=0.0, y=0.0, heading=0.0, speed=0.0, length=4.5, width=1.9,
                         height=1.5, cls='vehicle', lane=None, on_route=on_route, s_rel=s_rel,
                         lat_off=0.0, v_rel=5.0, ttc=ttc, will_enter_lane=enter, age=0.0,
                         coasting=False)


def pipeline(lg, route):
    pl = Planner(lg, route, CFG)
    return pl, Shield(lg, CFG, planner=pl), Control(CFG)


def step(pl, sh, lg, route, s, speed, objects, t):
    w = make_world(lg, route, s, speed=speed, t=t, objects=objects)
    return w, sh.apply(w, pl.plan(w))


# ── 발동 ──────────────────────────────────────────────────────────────────
def test_fires_below_emergency_ttc(lg, route):
    pl, sh, ct = pipeline(lg, route)
    w, d = step(pl, sh, lg, route, 20.0, 10.0, [obj(ttc=EMERGENCY_S - 0.5)], t=2.0)
    assert d.state == 'E_STOP'
    assert d.v_target == 0.0
    assert 'emergency_brake' in d.reasons['shield']
    assert 'id=42' in d.reasons['shield']['emergency_brake']
    # Control 이 a_emergency 를 그대로 낸다 (a_min -6.0 클램프·저크 제한 무시)
    cmd = ct.compute(w, d)
    assert cmd.accel == pytest.approx(A_EMERGENCY)
    assert cmd.accel < CFG['speed']['a_min']


def test_will_enter_lane_object_also_fires(lg, route):
    pl, sh, _ = pipeline(lg, route)
    _w, d = step(pl, sh, lg, route, 20.0, 10.0,
                 [obj(ttc=EMERGENCY_S - 0.5, on_route=False, enter=True)], t=2.0)
    assert d.state == 'E_STOP' and d.v_target == 0.0


# ── 오탐 없음 ─────────────────────────────────────────────────────────────
def test_normal_lead_following_does_not_fire(lg, route):
    """정상 선행차 추종(TTC 5 s)에서는 발동하지 않는다."""
    pl, sh, ct = pipeline(lg, route)
    w, d = step(pl, sh, lg, route, 20.0, 10.0, [obj(ttc=5.0)], t=2.0)
    assert d.state != 'E_STOP'
    assert 'emergency_brake' not in d.reasons['shield']
    assert ct.compute(w, d).accel >= CFG['speed']['a_min']


def test_off_route_object_is_not_a_target(lg, route):
    """옆 차로를 스쳐 가는 객체(on_route=False, 진입 예측 없음)는 대상이 아니다."""
    pl, sh, _ = pipeline(lg, route)
    _w, d = step(pl, sh, lg, route, 20.0, 10.0,
                 [obj(ttc=0.2, on_route=False, enter=False)], t=2.0)
    assert d.state != 'E_STOP'
    assert 'emergency_brake' not in d.reasons['shield']


def test_no_objects_does_not_fire(lg, route):
    pl, sh, _ = pipeline(lg, route)
    _w, d = step(pl, sh, lg, route, 20.0, 10.0, [], t=2.0)
    assert d.state != 'E_STOP'


# ── 해제 / 히스테리시스 ───────────────────────────────────────────────────
def test_releases_when_ttc_recovers_above_brake_s(lg, route):
    pl, sh, _ = pipeline(lg, route)
    _w, d = step(pl, sh, lg, route, 20.0, 10.0, [obj(ttc=EMERGENCY_S - 0.5)], t=2.0)
    assert d.state == 'E_STOP'
    _w, d = step(pl, sh, lg, route, 21.0, 5.0, [obj(ttc=BRAKE_S + 0.5)], t=2.2)
    assert d.state != 'E_STOP'
    assert 'emergency_brake' not in d.reasons['shield']
    assert not sh._estop


def test_latch_holds_between_thresholds(lg, route):
    """발동 후 TTC 가 emergency_s~brake_s 사이로 올라와도 유지 (껌뻑임 방지)."""
    pl, sh, _ = pipeline(lg, route)
    step(pl, sh, lg, route, 20.0, 10.0, [obj(ttc=EMERGENCY_S - 0.5)], t=2.0)
    mid = (EMERGENCY_S + BRAKE_S) / 2.0
    _w, d = step(pl, sh, lg, route, 20.5, 6.0, [obj(ttc=mid)], t=2.2)
    assert d.state == 'E_STOP' and sh._estop
    # 객체가 사라지면(TTC 무한) 해제
    _w, d = step(pl, sh, lg, route, 21.0, 4.0, [], t=2.4)
    assert d.state != 'E_STOP' and not sh._estop


def test_does_not_fire_at_exactly_threshold(lg, route):
    pl, sh, _ = pipeline(lg, route)
    _w, d = step(pl, sh, lg, route, 20.0, 10.0, [obj(ttc=EMERGENCY_S)], t=2.0)
    assert d.state != 'E_STOP'
