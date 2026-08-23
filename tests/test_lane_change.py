"""
차선변경 (planner + shield).

핵심 계약:
  · 점선 구간(window) 안에서만 시작한다 — 실선 차로변경은 감점(S2.2.05)
  · 지시등을 signal.lead_s 만큼 미리 켠다 — 채점 항목
  · 목표 차로가 비었을 때만 실행, 아니면 대기
  · 전이는 점프 없이 이어지고 **끝까지 간다** (중간에 멈추면 차로 한복판에 남는다)
"""
import math
import pathlib
import pickle

import pytest

from conftest import PARAMS_YAML
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.planner import FOLLOW, LANE_CHANGE, Planner
from hlfma.core.shield import Shield
from hlfma.core.types import EgoState, TrackedObject, WorldState
from hlfma.nodes.params import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
# 고정 테스트 경로. data/route.pkl 은 시나리오마다 다시 빌드되므로 쓰지 않는다
# (2026-08-23: 경로 재생성 때마다 하드코딩된 정지선·이벤트 위치가 어긋나 29건 실패).
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


@pytest.fixture(scope='module')
def lc_event(route):
    """바로 앞에 교차로 회전이 없는 첫 LC 이벤트.

    회전 직후의 LC(예: 회전1 연결로 끝 = LC1 창 시작)는 회전 중 반대 방향
    지시등이 우선해 lead 가 짧아지는 별도 케이스다 (tests/test_lc_lead.py)."""
    turns = [e['s'] for e in route['events'] if e['kind'].startswith('turn_')]
    evs = [e for e in route['events'] if e['kind'].startswith('lane_change')
           and not any(e['window_s0'] - 80.0 <= t <= e['window_s0'] for t in turns)]
    assert evs, '경로에 (회전과 겹치지 않는) 차선변경 이벤트가 없다'
    return evs[0]


def make_world(lg, route, route_s, lane=None, speed=5.56, objects=()):
    """경로상 route_s 지점의 자차 상태를 만든다."""
    cum = route['cum_s']
    # 차선변경 이음매는 from/to 차로의 cum_s 가 같다(평행). 동률이면 **앞쪽(from)**
    # 차로를 고른다 — 뒤쪽(to)을 고르면 t_off=0 인 자차가 목표 차로에 이미 안착한
    # 것으로 판정돼 전이가 시작 즉시 끝난다.
    best = max(cum[i] for i in range(len(cum)) if cum[i] <= route_s)
    idx = min(i for i in range(len(cum)) if cum[i] == best)
    k = route['lanes'][idx]
    s_in = route_s - cum[idx]
    s_in = max(0.0, min(s_in, lg.length(k)))
    x, y, _z, h = lg.point_at(k, s_in)
    ego = EgoState(x=x, y=y, z=0.0, yaw=h, pitch=0.0, roll=0.0,
                   speed=speed, accel=0.0, lane=lane or k, s=s_in,
                   route_s=route_s, t_off=0.0, heading_err=0.0)
    # t 는 route_s / speed — 등속 주행 가정. 틱 시퀀스 테스트에서 지시등 연속
    # 점등 시간(sig_lead_s) 이 실제처럼 흐르게 한다.
    return WorldState(t=route_s / max(speed, 0.1), ego=ego, objects=list(objects), light=None, ahead=[], summ={},
                      speed_limit=13.9, school_zone=False, left_solid=False,
                      right_solid=False, left_is_center=False, valid=True, flags={})


def blocker(lane, s_rel):
    return TrackedObject(id=99, x=0.0, y=0.0, heading=0.0, speed=0.0,
                         length=4.5, width=1.9, height=1.5, cls='vehicle',
                         lane=tuple(lane), on_route=True, s_rel=s_rel, lat_off=0.0,
                         v_rel=0.0, ttc=float('inf'), will_enter_lane=False,
                         age=0.0, coasting=False)


# ── 지시등 ────────────────────────────────────────────────────────────────
def test_signal_on_before_window(lg, route):
    """창 시작 (lead_s + margin_s) 초 전부터 켜져야 한다."""
    pl = Planner(lg, route, CFG)
    sig = CFG['signal']
    v = 5.56
    lead_m = v * (sig['lead_s'] + sig['margin_s'])
    # 바로 앞에 교차로 회전이 붙은 LC(예: 회전1 연결로 → LC1)는 회전 지시등이
    # 우선해 LC 방향이 가려진다(tests/test_turn_signal.py 가 그 규칙을 검사한다).
    # 여기서는 회전과 겹치지 않는 LC 이벤트로 LC 자체의 lead 를 본다.
    turns = [e['s'] for e in route['events'] if e['kind'].startswith('turn_')]
    lc_event = next(e for e in route['events'] if e['kind'].startswith('lane_change')
                    and not any(e['window_s0'] - 80.0 <= t <= e['window_s0'] for t in turns))
    w0 = lc_event['window_s0']
    expect = 1 if lc_event['kind'].endswith('left') else 2

    far = pl.plan(make_world(lg, route, w0 - lead_m - 15.0, speed=v))
    assert far.turn_signal == 0, '너무 일찍 켜졌다'

    near = pl.plan(make_world(lg, route, w0 - lead_m + 2.0, speed=v))
    assert near.turn_signal == expect


def test_signal_lead_time_is_at_least_lead_s(lg, route, lc_event):
    """켜진 시점부터 창 시작까지 최소 lead_s 초는 확보돼야 한다."""
    pl = Planner(lg, route, CFG)
    v = 5.56
    w0 = lc_event['window_s0']
    on_at = None
    s = w0 - 60.0
    while s < w0:
        if pl.plan(make_world(lg, route, s, speed=v)).turn_signal != 0:
            on_at = s
            break
        s += 0.5
    assert on_at is not None, '창 진입 전에 지시등이 안 켜졌다'
    assert (w0 - on_at) / v >= CFG['signal']['lead_s'] - 0.5


# ── 실행 조건 ─────────────────────────────────────────────────────────────
def test_no_lane_change_outside_window(lg, route, lc_event):
    """창 앞에서는 절대 시작하지 않는다 (실선 구간)."""
    pl = Planner(lg, route, CFG)
    d = pl.plan(make_world(lg, route, lc_event['window_s0'] - 5.0))
    assert d.state != LANE_CHANGE


def test_lane_change_starts_inside_window(lg, route, lc_event):
    pl = Planner(lg, route, CFG)
    v = 5.56
    for s in [lc_event['window_s0'] - 30, lc_event['window_s0'] - 20,
              lc_event['window_s0'] - 10, lc_event['window_s0'] + 1.0]:
        d = pl.plan(make_world(lg, route, s, speed=v))
    assert d.state == LANE_CHANGE
    assert d.turn_signal != 0


def test_blocked_target_lane_waits(lg, route, lc_event):
    """목표 차로에 차가 있으면 대기한다."""
    pl = Planner(lg, route, CFG)
    v = 5.56
    tgt = tuple(lc_event['to_lane'])
    for s in [lc_event['window_s0'] - 30, lc_event['window_s0'] - 20, lc_event['window_s0'] - 10]:
        pl.plan(make_world(lg, route, s, speed=v))
    d = pl.plan(make_world(lg, route, lc_event['window_s0'] + 1.0, speed=v,
                           objects=[blocker(tgt, 10.0)]))
    assert d.state != LANE_CHANGE
    assert d.reasons.get('lc_clear') is False


def test_object_behind_beyond_back_m_does_not_block(lg, route, lc_event):
    pl = Planner(lg, route, CFG)
    v = 5.56
    tgt = tuple(lc_event['to_lane'])
    far_back = -(CFG['lane_change']['back_m'] + 20.0)
    for s in [lc_event['window_s0'] - 30, lc_event['window_s0'] - 20, lc_event['window_s0'] - 10]:
        pl.plan(make_world(lg, route, s, speed=v))
    d = pl.plan(make_world(lg, route, lc_event['window_s0'] + 1.0, speed=v,
                           objects=[blocker(tgt, far_back)]))
    assert d.state == LANE_CHANGE


# ── 전이 경로 ─────────────────────────────────────────────────────────────
def test_path_transition_is_smooth_and_progresses(lg, route, lc_event):
    """
    시작 직후 경로는 현재 차로에 붙어 있고(점프 없음),
    진행할수록 목표 차로 쪽으로 넘어가야 한다.
    """
    pl = Planner(lg, route, CFG)
    v = 5.56
    w0 = lc_event['window_s0']
    for s in [w0 - 30, w0 - 20, w0 - 10]:
        pl.plan(make_world(lg, route, s, speed=v))

    w = make_world(lg, route, w0 + 1.0, speed=v)
    d0 = pl.plan(w)
    assert d0.state == LANE_CHANGE
    # 첫 점은 자차 바로 앞 — 급격한 횡방향 점프가 없어야 한다
    first = d0.path[0]
    assert math.hypot(first[0] - w.ego.x, first[1] - w.ego.y) < 3.0

    def lateral(dec, world):
        px, py = dec.path[min(20, len(dec.path) - 1)]
        dx, dy = px - world.ego.x, py - world.ego.y
        return -dx * math.sin(world.ego.yaw) + dy * math.cos(world.ego.yaw)

    w1 = make_world(lg, route, w0 + 12.0, speed=v)
    d1 = pl.plan(w1)
    # 진행할수록 목표 차로 쪽 횡오프셋이 커진다 (전이가 실제로 진척된다)
    assert abs(lateral(d1, w1)) > abs(lateral(d0, w)) + 0.2, \
        f'전이가 진척되지 않는다: {lateral(d0, w):.2f} → {lateral(d1, w1):.2f}'


def test_speed_is_not_reduced_during_lane_change(lg, route, lc_event):
    """
    전이 중 차선변경을 이유로 감속하지 않는다 (녹색신호 통과 감점 방지).

    v_target 은 제한속도·곡률·정지선이 정한다. 차선변경이 그 값을 더 깎지 않아야 한다.
    """
    pl = Planner(lg, route, CFG)
    v = 5.56
    for s in [lc_event['window_s0'] - 30, lc_event['window_s0'] - 20, lc_event['window_s0'] - 10]:
        pl.plan(make_world(lg, route, s, speed=v))

    w_in = make_world(lg, route, lc_event['window_s0'] + 1.0, speed=v)
    d_in = pl.plan(w_in)
    assert d_in.state == LANE_CHANGE

    # 같은 지점을 차선변경 없이 (새 planner) 계산한 값과 비교
    pl2 = Planner(lg, route, CFG)
    d_ref = pl2.plan(make_world(lg, route, lc_event['window_s0'] + 1.0, speed=v))
    assert d_in.v_target >= d_ref.v_target - 1e-6, \
        f'차선변경이 속도를 깎았다: {d_ref.v_target:.2f} → {d_in.v_target:.2f}'
    assert 'debug_const' not in d_in.reasons, '기본 경로에 상수속도가 남아 있다'


# ── shield ────────────────────────────────────────────────────────────────
def test_shield_aborts_lane_change_on_ttc(lg, route, lc_event):
    pl = Planner(lg, route, CFG)
    sh = Shield(lg, CFG, planner=pl)
    v = 5.56
    for s in [lc_event['window_s0'] - 30, lc_event['window_s0'] - 20, lc_event['window_s0'] - 10]:
        pl.plan(make_world(lg, route, s, speed=v))

    # 목표 차로에 두면 planner 가 애초에 시작하지 않는다(간격 확인에서 걸림).
    # 전이가 시작된 뒤 다른 차로의 위험이 잡히는 상황을 만든다.
    w0 = make_world(lg, route, lc_event['window_s0'] + 1.0, speed=v)
    d0 = pl.plan(w0)
    assert d0.state == LANE_CHANGE, '전이가 시작되지 않아 중단을 시험할 수 없다'

    danger = blocker(w0.ego.lane, 8.0)          # 현재 차로의 선행차
    # ttc.emergency_s(1.5) 와 ttc.warn_s(4.0) 사이 — 차선변경 중단만 보고
    # 비상제동(E_STOP)은 건드리지 않는다 (그건 tests/test_emergency_brake.py)
    danger.ttc = 2.0
    w = make_world(lg, route, lc_event['window_s0'] + 3.0, speed=v, objects=[danger])
    d = sh.apply(w, pl.plan(w))
    assert 'lc_abort_ttc' in d.reasons['shield']
    assert d.state == FOLLOW
    assert d.turn_signal == 0
