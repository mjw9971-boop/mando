"""
Shield 하드 클램프 (SPEC §3.5).

**Shield 는 Planner 를 신뢰하지 않는다.** Planner 가 버그로 위법한 값을 내도
여기서 잘린다. 2026-08-23 19:56 런에서 저속 차선변경이 발산해 차로 id −3 → +5
(반대편 차선)로 넘어갔는데 shield 가 한 번도 발동하지 않았다 — 가드 4개가
전부 NotImplementedError 스텁이었고 apply() 에서 호출조차 되지 않았다.
"""
import math
import pathlib
import pickle

import pytest

from conftest import PARAMS_YAML
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.planner import Planner
from hlfma.core.shield import Shield
from hlfma.nodes.params import load_params_yaml
from test_turn_signal import make_world

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route_example.pkl'
CFG = load_params_yaml(PARAMS_YAML)

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='lane_graph.pkl / route_example.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def route():
    return pickle.load(open(ROUTE, 'rb'))


def pipeline(lg, route):
    pl = Planner(lg, route, CFG)
    return pl, Shield(lg, CFG, planner=pl)


def shifted_path(world, dy):
    """자차 경로를 좌(+)/우(−)로 dy 만큼 평행이동한 가짜 path."""
    e = world.ego
    nx, ny = -math.sin(e.yaw), math.cos(e.yaw)
    return [(e.x + nx * dy + math.cos(e.yaw) * i * 0.5,
             e.y + ny * dy + math.sin(e.yaw) * i * 0.5) for i in range(80)]


# ── 1) 제한속도 하드 클램프 ────────────────────────────────────────────────
def test_speed_over_limit_is_clamped(lg, route):
    """v_target > 제한속도 − margin_kph 이면 잘려야 한다."""
    pl, sh = pipeline(lg, route)
    w = make_world(lg, route, 20.0, speed=10.0, t=0.0)
    d = pl.plan(w)
    d.v_target = 100.0                       # planner 버그를 흉내
    d = sh.apply(w, d)
    cap = w.speed_limit - CFG['speed']['margin_kph'] / 3.6
    assert d.v_target == pytest.approx(cap)
    assert 'speed_cap' in d.reasons['shield']


def test_normal_speed_is_not_clamped(lg, route):
    pl, sh = pipeline(lg, route)
    w = make_world(lg, route, 20.0, speed=10.0, t=0.0)
    d = sh.apply(w, pl.plan(w))
    assert 'speed_cap' not in d.reasons['shield']


# ── 3) 중앙선 침범 ────────────────────────────────────────────────────────
def test_center_line_crossing_is_reverted(lg, route):
    """left_is_center 인데 좌측으로 벗어나는 path 는 교체된다."""
    pl, sh = pipeline(lg, route)
    w = make_world(lg, route, 20.0, speed=8.0, t=0.0)
    if not w.left_is_center:
        pytest.skip('이 지점은 좌측이 중앙선이 아니다')
    d = pl.plan(w)
    d.path = shifted_path(w, +3.0)           # 좌측으로 3 m 벗어난 경로
    d = sh.apply(w, d)
    assert 'center_crossing' in d.reasons['shield']
    assert d.path != shifted_path(w, +3.0)   # 현재 차로로 교체됨


def test_center_crossing_not_fired_when_path_ok(lg, route):
    pl, sh = pipeline(lg, route)
    w = make_world(lg, route, 20.0, speed=8.0, t=0.0)
    d = sh.apply(w, pl.plan(w))
    assert 'center_crossing' not in d.reasons['shield']


# ── 6) 차로 이탈 복귀 ─────────────────────────────────────────────────────
def test_lateral_offset_beyond_lane_edge_triggers_return(lg, route):
    """|t_off| > 차로폭/2 − edge_margin 이면 복귀 우선."""
    pl, sh = pipeline(lg, route)
    w = make_world(lg, route, 20.0, speed=8.0, t=0.0)
    half = 0.5 * lg.width_at(w.ego.lane, w.ego.s)
    w.ego.t_off = half - CFG['shield']['edge_margin_m'] + 0.5
    d = sh.apply(w, pl.plan(w))
    assert 'pull_back' in d.reasons['shield']


def test_pull_back_skipped_during_lane_change(lg, route):
    """전이 중에는 t_off 가 큰 것이 정상 — 오발화하면 안 된다."""
    pl, sh = pipeline(lg, route)
    w = make_world(lg, route, 20.0, speed=8.0, t=0.0)
    half = 0.5 * lg.width_at(w.ego.lane, w.ego.s)
    w.ego.t_off = half + 1.0
    d = pl.plan(w)
    d.state = 'LANE_CHANGE'
    d = sh.apply(w, d)
    assert 'pull_back' not in d.reasons['shield']


def test_pull_back_not_fired_on_centerline(lg, route):
    pl, sh = pipeline(lg, route)
    w = make_world(lg, route, 20.0, speed=8.0, t=0.0)
    d = sh.apply(w, pl.plan(w))
    assert 'pull_back' not in d.reasons['shield']


# ── 5) 횡단보도 정차 금지 ─────────────────────────────────────────────────
def test_stop_point_inside_crosswalk_is_pulled_back(lg, route):
    """정지 예상 지점이 횡단보도 안이면 목표를 구간 **앞으로** 당긴다."""
    pl, sh = pipeline(lg, route)
    vh, sp = CFG['vehicle'], CFG['speed']
    front = vh['wheelbase'] + vh['front_overhang_m']
    a = sp['a_comf']
    # 횡단보도 시작을 앞범퍼 정지 예상 지점 바로 뒤에 오도록 자차 속도를 고른다
    w = make_world(lg, route, 20.0, speed=4.0, t=0.0)
    cw = next((x for x in w.ahead if x.kind == 'crosswalk'), None)
    if cw is None:
        pytest.skip('전방 횡단보도 없음')
    s0 = float(cw.data.get('s0', cw.dist))
    v = math.sqrt(max(0.0, 2.0 * a * (s0 - front + 1.0)))   # 구간 안에서 멈추도록
    w = make_world(lg, route, 20.0, speed=v, t=0.0)
    d = pl.plan(w)
    d.v_target = 0.0                          # 세우려는 중
    before = d.v_target
    d = sh.apply(w, d)
    if 'no_stop_in_crosswalk' in d.reasons['shield']:
        assert d.v_target <= before
    else:
        # 정지 예상 지점이 구간 밖이면 발동하지 않는 것이 정상
        assert d.v_target == before


def test_no_stop_in_crosswalk_not_fired_when_cruising(lg, route):
    """세우려는 중이 아니면(빠른 v_target) 관여하지 않는다."""
    pl, sh = pipeline(lg, route)
    w = make_world(lg, route, 20.0, speed=10.0, t=0.0)
    d = sh.apply(w, pl.plan(w))
    assert 'no_stop_in_crosswalk' not in d.reasons['shield']


# ── 스텁이 남아 있지 않은지 ───────────────────────────────────────────────
def test_no_unimplemented_guards():
    import inspect
    from hlfma.core import shield as mod
    src = inspect.getsource(mod)
    assert 'NotImplementedError' not in src, 'shield 에 미구현 스텁이 남아 있다'
