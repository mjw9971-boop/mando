"""
횡단보도 보행자 → 정지선 앞 정지 (2026-08-23 14:31 런 대응).

계약:
  · 전방 횡단보도 폴리곤 안 객체(타입 무관, 보수적) 또는 진입 예측이면 정지 후보 활성
  · **신호와 무관** — 녹색이어도 선다 (도로교통법 27조 / "보행자 출현" 채점)
  · 목표는 그 횡단보도에 선행하는 정지선. 없으면 횡단보도 경계 − stop_gap.
    정지 프로파일은 정지선과 동일(전장 보정 + stop_lag)
  · 보행자가 폴리곤을 벗어나면 해제 후 재출발
  · 이미 정지선을 지나쳤으면 활성화하지 않는다 (횡단보도 정차 금지 S6.3.03)
  · RTOR 안전 확인과 **같은 판정**(crosswalk_blockers)을 쓴다
"""
import pathlib
import pickle

import pytest

from conftest import PARAMS_YAML
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.planner import Planner, crosswalk_blockers
from hlfma.core.types import TrackedObject
from hlfma.nodes.params import load_params_yaml
from test_turn_signal import make_world

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route_example.pkl'   # 고정 테스트 경로 (data/route.pkl 은 시나리오마다 바뀐다)
CFG = load_params_yaml(PARAMS_YAML)
GREEN, RED = 3, 1
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


def ped(oid, s_rel, lat_off=0.0, enter=False, ttc=float('inf')):
    return TrackedObject(id=oid, x=0.0, y=0.0, heading=0.0, speed=1.2, length=0.6, width=0.6,
                         height=1.7, cls='pedestrian', lane=None, on_route=False, s_rel=s_rel,
                         lat_off=lat_off, v_rel=0.0, ttc=ttc, will_enter_lane=enter, age=0.0,
                         coasting=False)


def approach(lg, route, s, speed, objects=(), light=(3, GREEN)):
    return make_world(lg, route, s, speed=speed, t=s / max(speed, 0.1),
                      light=light, objects=objects)


# ── 판정 헬퍼 ─────────────────────────────────────────────────────────────
def test_blocker_polygon_bounds():
    p = CFG['percep']
    d = 20.0
    inside = ped(1, s_rel=d, lat_off=0.0)
    assert crosswalk_blockers([inside], d, CFG) == [inside]
    assert crosswalk_blockers([ped(2, s_rel=d, lat_off=p['crosswalk_half_w_m'] + 1)], d, CFG) == []
    assert crosswalk_blockers([ped(3, s_rel=d - p['crosswalk_back_m'] - 1)], d, CFG) == []
    assert crosswalk_blockers([ped(4, s_rel=d + p['crosswalk_fwd_m'] + 1)], d, CFG) == []
    # 진입 예측은 횡단보도 근처만
    assert crosswalk_blockers([ped(5, s_rel=d, lat_off=10.0, enter=True)], d, CFG)
    assert crosswalk_blockers([ped(6, s_rel=d + 60.0, enter=True)], d, CFG) == []
    assert crosswalk_blockers([inside], None, CFG) == []


# ── 정지 ──────────────────────────────────────────────────────────────────
def test_pedestrian_on_crosswalk_stops_even_on_green(lg, route):
    """녹색이어도 횡단보도 보행자가 있으면 정지 후보가 나온다."""
    pl = Planner(lg, route, CFG)
    w = approach(lg, route, 20.0, speed=8.0)
    d_cw = w.summ['dist_crosswalk']
    assert d_cw is not None
    clear = pl.plan(w)
    assert 'crosswalk_ped' not in clear.reasons          # 보행자 없으면 후보 없음

    pl2 = Planner(lg, route, CFG)
    d = pl2.plan(approach(lg, route, 20.0, speed=8.0, objects=[ped(7, s_rel=d_cw)]))
    assert 'crosswalk_ped' in d.reasons
    assert d.reasons['crosswalk_ped'] < d.reasons['limit']
    assert d.reasons['crosswalk'].startswith('slow')
    assert 'stop_line' not in d.reasons                  # 녹색이라 신호 정지는 아니다


def test_stops_before_stop_line_and_releases_when_clear(lg, route):
    """정지선 앞에서 멈추고(래치), 보행자가 사라지면 풀려 재출발한다."""
    pl = Planner(lg, route, CFG)
    s = 20.0
    v = 8.0
    stopped_at = None
    while s < 55.0:
        w = approach(lg, route, s, speed=v)
        d_cw = w.summ['dist_crosswalk']
        objs = [ped(7, s_rel=d_cw)] if d_cw is not None else []
        d = pl.plan(approach(lg, route, s, speed=v, objects=objs))
        if pl._cw_ped_latch:
            stopped_at = s
            break
        s += 0.5
    assert stopped_at is not None, '보행자가 있는데 정지하지 않았다'
    # 래치 지점의 앞범퍼는 정지선 앞에 있어야 한다 (정지선 s=50.03)
    front = stopped_at + FRONT
    assert front < 50.03, f'앞범퍼가 정지선을 넘었다: {front:.2f}'
    assert pl.plan(approach(lg, route, stopped_at, speed=0.0,
                            objects=[ped(7, s_rel=w.summ['dist_crosswalk'])])).v_target == 0.0
    # 보행자 사라짐 → 해제
    d = pl.plan(approach(lg, route, stopped_at, speed=0.0))
    assert not pl._cw_ped_latch and 'crosswalk_ped' not in d.reasons
    assert d.v_target > 0.0


def test_will_enter_lane_also_stops(lg, route):
    pl = Planner(lg, route, CFG)
    w = approach(lg, route, 20.0, speed=8.0)
    d_cw = w.summ['dist_crosswalk']
    d = pl.plan(approach(lg, route, 20.0, speed=8.0,
                         objects=[ped(9, s_rel=d_cw, lat_off=9.0, enter=True)]))
    assert 'crosswalk_ped' in d.reasons


def test_no_stop_inside_crosswalk(lg, route):
    """정지선을 이미 지난 뒤 감지되면 서지 않는다 (횡단보도 정차 금지)."""
    pl = Planner(lg, route, CFG)
    w = approach(lg, route, 49.5, speed=6.0)
    d_cw = w.summ.get('dist_crosswalk')
    if d_cw is None:
        pytest.skip('그 지점에 전방 횡단보도가 없다')
    d = pl.plan(approach(lg, route, 49.5, speed=6.0, objects=[ped(11, s_rel=d_cw)]))
    assert d.reasons.get('crosswalk', '').startswith('pass')
    assert 'crosswalk_ped' not in d.reasons


def test_target_falls_back_to_crosswalk_without_stop_line(lg, route):
    """정지선이 없으면 횡단보도 경계를 목표로 한다."""
    pl = Planner(lg, route, CFG)
    w = approach(lg, route, 20.0, speed=8.0)
    d_cw = w.summ['dist_crosswalk']
    w.summ = dict(w.summ, dist_stop_line=None)
    w.objects = [ped(12, s_rel=d_cw)]
    d = pl.plan(w)
    assert 'crosswalk_ped' in d.reasons
    import math
    a = CFG['speed']['a_comf'] * CFG['speed']['a_plan_factor']
    room = d_cw - GAP - FRONT - CFG['speed']['stop_lag_s'] * 8.0
    assert d.reasons['crosswalk_ped'] == pytest.approx(math.sqrt(2 * a * room), rel=1e-6)


# ── 지시등 오점등 회귀 (off_route) ────────────────────────────────────────
def test_off_route_does_not_select_lane_change(lg, route):
    """경로 이탈 중에는 계획 LC 를 고르지 않는다 — route_s 요동으로 지시등이 오점등했다."""
    pl = Planner(lg, route, CFG)
    # 회전 연결로와 겹치지 않는 LC 를 고른다 (겹치면 회전 지시등이 우선한다)
    turns = [(t['s'], t['end_s']) for t in pl._turns]
    lc = next(e for e in route['events'] if e['kind'].startswith('lane_change')
              and not any(a - 40.0 <= e['window_s0'] <= b + 1.0 for a, b in turns))
    on = pl.plan(make_world(lg, route, lc['window_s0'] - 10.0, speed=8.0))
    assert on.turn_signal != 0 and on.reasons.get('sig_src') == 'lc'
    w = make_world(lg, route, lc['window_s0'] - 10.0, speed=8.0)
    w.flags['off_route'] = True
    d = Planner(lg, route, CFG).plan(w)
    assert d.reasons.get('sig_src') != 'lc'
