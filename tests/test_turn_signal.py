"""
지시등 (교차로 회전 + 차선변경 우선순위 + 래치/재선택 회귀).

2026-08-23 10:26 런에서 잡힌 세 가지:
  · 회전 지시등 미구현 → 우회전 1 이 LC1 lead 의 LEFT 로, 우회전 3 은 미점등
  · 완료한 LC 이벤트를 재선택 → 2틱 주기로 23회 깜빡임
  · 점등 래치가 pending 이벤트 교체 시 안 풀림 → LC2 완료 후 30 s 연속 점등

계약:
  · 회전: lead(signal.lead_s + margin_s)초 전부터 회전 방향 점등, 연결로를 벗어나면 소등
  · 차선변경과 겹치면 **더 가까운 이벤트** 가 이긴다 (동률이면 회전)
  · 완료한 LC 는 창 s1 전까지 다시 고르지 않는다
  · pending 이 바뀌면 래치를 풀고 새 이벤트의 lead 조건으로 다시 판단한다
"""
import pathlib
import pickle

import pytest

from conftest import PARAMS_YAML
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.planner import FOLLOW, LANE_CHANGE, TURN_LEFT, TURN_OFF, TURN_RIGHT, Planner
from hlfma.core.types import EgoState, WorldState
from hlfma.nodes.params import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route_example.pkl'   # 고정 테스트 경로 (data/route.pkl 은 시나리오마다 바뀐다)
LOG = ROOT / 'logs' / 'run_20260823_102634.jsonl'
CFG = load_params_yaml(PARAMS_YAML)
LEAD_S = float(CFG['signal']['lead_s']) + float(CFG['signal']['margin_s'])

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='lane_graph.pkl / route.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def route():
    return pickle.load(open(ROUTE, 'rb'))


def make_world(lg, route, route_s, lane=None, speed=10.0, t=None, light=None, objects=()):
    cum = route['cum_s']
    best = max(cum[i] for i in range(len(cum)) if cum[i] <= route_s)
    idx = min(i for i in range(len(cum)) if cum[i] == best)   # 동률이면 from 차로
    k = route['lanes'][idx]
    s_in = max(0.0, min(route_s - cum[idx], lg.length(k)))
    x, y, _z, h = lg.point_at(k, s_in)
    ego = EgoState(x=x, y=y, z=0.0, yaw=h, pitch=0.0, roll=0.0,
                   speed=speed, accel=0.0, lane=lane or k, s=s_in,
                   route_s=route_s, t_off=0.0, heading_err=0.0)
    ahead = lg.lookahead(route, idx, s_in, 200.0)
    # t 는 route_s / speed — 등속 주행 가정. 틱 시퀀스 테스트에서 지시등 연속
    # 점등 시간(sig_lead_s) 이 실제처럼 흐르게 한다.
    return WorldState(t=route_s / max(speed, 0.1) if t is None else t, ego=ego,
                      objects=list(objects), light=light, ahead=ahead,
                      summ=lg.summarize(ahead), speed_limit=13.9, school_zone=False,
                      left_solid=False, right_solid=False, left_is_center=False,
                      valid=True, flags={})


def turns(route):
    return [e for e in route['events'] if e['kind'].startswith('turn_')]


def lcs(route):
    return [e for e in route['events'] if e['kind'].startswith('lane_change')]


def sig_of(kind):
    return TURN_LEFT if kind.endswith('left') else TURN_RIGHT


# ── 회전 ──────────────────────────────────────────────────────────────────
def test_turn_signal_lead_and_direction(lg, route):
    """첫 회전: lead 밖에서는 꺼져 있고 lead 안에 들어오면 회전 방향으로 켜진다."""
    t0 = turns(route)[0]
    v = 10.0
    lead_m = v * LEAD_S
    pl = Planner(lg, route, CFG)
    far = pl.plan(make_world(lg, route, t0['s'] - lead_m - 5.0, speed=v))
    assert far.turn_signal == TURN_OFF, '너무 일찍 켜졌다'
    near = pl.plan(make_world(lg, route, t0['s'] - lead_m + 5.0, speed=v))
    assert near.turn_signal == sig_of(t0['kind'])
    assert near.reasons.get('sig_src') == 'turn'


def test_turn_signal_stays_on_in_connector_and_off_after(lg, route):
    """연결로 안에서는 회전 지시등이 켜지고, 연결로 끝(end_s)을 지나면 회전 근거가 사라진다.

    끝난 직후 LC 창이 바로 시작하는 경로도 있으므로(회전1 end_s = LC1 window_s0)
    "무조건 소등"이 아니라 **회전이 더는 점등 근거가 아니다**(sig_src != 'turn')로 본다.
    """
    pl = Planner(lg, route, CFG)
    last = pl._turns[-1]
    inside = pl.plan(make_world(lg, route, (last['s'] + last['end_s']) / 2, speed=10.0))
    assert inside.turn_signal == last['signal']
    assert inside.reasons.get('turn_dist') == 0.0
    after = pl.plan(make_world(lg, route, last['end_s'] + 3.0, speed=10.0))
    assert after.reasons.get('sig_src') != 'turn'
    assert after.reasons.get('turn_dist') is None


def test_build_turns_end_is_junction_exit(lg, route):
    """end_s 는 같은 junction 연결로의 끝 = 다음 비교차로 차로의 시작."""
    pl = Planner(lg, route, CFG)
    assert len(pl._turns) == len(turns(route))
    lanes, cum = route['lanes'], route['cum_s']
    for t in pl._turns:
        j = next(i for i in range(len(lanes)) if abs(cum[i] - t['end_s']) < 0.5)
        assert lg.lanes[lanes[j]]['junction'] == -1
        assert t['end_s'] > t['s']


# ── 우선순위 ──────────────────────────────────────────────────────────────
def test_nearer_event_wins_turn_before_lane_change(lg, route):
    """우회전 1 (50 m) 과 LC1 좌 (72 m) 가 둘 다 lead 안일 때 — 가까운 회전이 이긴다."""
    t0, lc0 = turns(route)[0], lcs(route)[0]
    assert t0['s'] < lc0['window_s0'] and sig_of(t0['kind']) != sig_of(lc0['kind'])
    v = 10.0
    s = t0['s'] - 5.0
    assert lc0['window_s0'] - s <= v * LEAD_S, '테스트 전제: LC 도 lead 안이어야 한다'
    pl = Planner(lg, route, CFG)
    d = pl.plan(make_world(lg, route, s, speed=v))
    assert d.turn_signal == sig_of(t0['kind'])
    assert d.reasons['sig_src'] == 'turn'


def test_lane_change_takes_over_after_turn_ends(lg, route):
    """연결로를 벗어나 LC 창에 들어서면 LC 방향으로 바뀐다 (역방향 점등 금지)."""
    lc0 = lcs(route)[0]
    pl = Planner(lg, route, CFG)
    end = pl._turns[0]['end_s']
    assert abs(end - lc0['window_s0']) < 1.0
    d = pl.plan(make_world(lg, route, lc0['window_s0'] + 1.0, speed=10.0))
    assert d.turn_signal == sig_of(lc0['kind'])
    assert d.reasons['sig_src'] == 'lc'


# ── 완료 기억 / 래치 ──────────────────────────────────────────────────────
def _drive_lc_to_done(pl, lg, route, ev, v=5.56):
    """창 앞에서부터 0.5 m 틱으로 주행해 LC 가 시작되면 목표 차로에 안착시켜 완료.
    시작 route_s 를 돌려준다 (점등 선행 조건 때문에 창 진입 직후가 아닐 수 있다)."""
    w0 = ev['window_s0']
    s = w0 - 30.0
    start = None
    while s < ev['window_s1']:
        d = pl.plan(make_world(lg, route, s, speed=v))
        if d.state == LANE_CHANGE:
            start = s
            break
        s += 0.5
    assert start is not None, 'LC 가 창 안에서 시작되지 않았다'
    # 목표 차로 중심선에 안착 → 완료
    pl.plan(make_world(lg, route, start + 1.0, lane=tuple(ev['to_lane']), speed=v))
    assert pl.lc_done == 1
    return start


def test_finished_lane_change_is_not_reselected(lg, route):
    """안착 후 창 안에 남아 있어도 같은 이벤트를 다시 고르지 않는다 (깜빡임 0)."""
    ev = lcs(route)[0]
    pl = Planner(lg, route, CFG)
    start = _drive_lc_to_done(pl, lg, route, ev)
    sigs, states = [], []
    s = start + 2.0
    while s < ev['window_s1']:
        d = pl.plan(make_world(lg, route, s, lane=tuple(ev['to_lane']), speed=5.56))
        sigs.append(d.turn_signal)
        states.append(d.state)
        s += 0.5
    assert pl.lc_done == 1
    assert LANE_CHANGE not in states
    assert all(x == TURN_OFF for x in sigs), sigs
    assert ev['window_s0'] in pl._lc_finished


def test_finished_set_clears_on_reset_before_window(lg, route):
    """리셋으로 창 앞까지 되돌아가면 완료 기록을 풀어 다시 실행한다."""
    ev = lcs(route)[0]
    pl = Planner(lg, route, CFG)
    _drive_lc_to_done(pl, lg, route, ev)
    pl.plan(make_world(lg, route, ev['window_s0'] - 20.0, speed=5.56))
    assert ev['window_s0'] not in pl._lc_finished


def test_signal_latch_resets_when_pending_changes(lg, route):
    """LC1 완료 후 pending 이 LC2 로 바뀌면 래치가 풀려 LC2 lead 전까지는 꺼진다."""
    ev1, ev2 = lcs(route)[0], lcs(route)[1]
    pl = Planner(lg, route, CFG)
    _drive_lc_to_done(pl, lg, route, ev1)
    v = 5.56
    lead_m = v * LEAD_S
    mid = ev2['window_s0'] - lead_m - 30.0
    d = pl.plan(make_world(lg, route, mid, speed=v))
    assert d.turn_signal == TURN_OFF, 'LC1 래치가 LC2 로 넘어갔다'
    assert d.state == FOLLOW
    near = pl.plan(make_world(lg, route, ev2['window_s0'] - lead_m + 2.0, speed=v))
    assert near.turn_signal == sig_of(ev2['kind'])


# ── 실제 런 리플레이 회귀 ─────────────────────────────────────────────────
@pytest.mark.skipif(not LOG.exists(), reason='런 로그 없음')
def test_replay_run_20260823_has_no_flicker(route):
    """실제 런 재생: 깜빡임 0, 경로의 모든 LC 수행, 경로에 없는 방향은 켜지지 않는다.

    기대치는 **route 에서 유도**한다 — route.pkl 은 시나리오마다 다시 만들어진다.
    """
    import sys
    sys.path.insert(0, str(ROOT / 'tools'))
    from replay import flicker_count, replay_signals, segments
    _orig, new, pl = replay_signals(str(LOG), route='data/route_example.pkl')
    segs = segments(new)
    assert flicker_count(segs) == 0
    n_lc = len(lcs(route))
    assert pl.lc_done == n_lc and pl.lc_aborted == 0
    on = [g for g in segs if g['sig'] != TURN_OFF]
    assert on, '점등 구간이 없다'
    # 경로 이벤트가 요구하는 방향만 켜져야 한다 (역방향 점등 금지)
    want = {sig_of(e['kind']) for e in route['events']
            if e['kind'].startswith(('turn_', 'lane_change'))}
    assert {g['sig'] for g in on} <= want
    # 토막나지 않았는지는 flicker_count 가 본다. OFF 를 사이에 둔 같은 방향 연속
    # 점등(회전2, 회전3)은 서로 다른 이벤트라 정상이다.
