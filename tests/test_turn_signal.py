"""
kr_rules 방향지시등 — 채점 동적항목 "방향지시등 n초 전" (SPEC §3.3).

계약:
  · 회전/차선변경 이벤트 전 lead 거리 안에서 점등, 구간 끝을 지나면 소등
  · lead 거리 = max(v · lead_s, lead_min_m) — 저속·정지에서도 켜진다
  · 겹치면 남은거리가 짧은 쪽, 동률이면 회전 우선
  · 지나간 구간은 다시 켜지지 않는다 (재선택 깜빡임 실사고 §6-9)
"""
import pathlib
import pickle
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.control import VtdLongitudinalController

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from autopilot import AutoPilot                            # noqa: E402
from config import GlobalConfig                            # noqa: E402
from kr_rules import (SIG_LEFT, SIG_OFF, SIG_RIGHT,        # noqa: E402
                      KrRules, signal_intervals)
from vtd_adapter.carla_types import VehicleControl         # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
SIG = CFG['signal']
TOTAL = 400.0
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE_PKL = ROOT / 'data' / 'route_example.pkl'


class FakeLaneGraph:
    """_turn_end_s 가 쓰는 표면만: lanes[key]['junction']."""

    def __init__(self, junctions: dict):
        self.lanes = {k: {'junction': j} for k, j in junctions.items()}


class FakePlanner:
    """kr_rules 표면: route(total_length/events/lanes/cum_s/lengths), route_s, lg."""

    LC_TRANSITION_M = 25.0

    def __init__(self, events=(), lanes=(), cum=(), lens=(), junctions=None):
        self.route = {'total_length': TOTAL, 'events': list(events),
                      'lanes': list(lanes), 'cum_s': list(cum), 'lengths': list(lens)}
        self.route_s = np.arange(0.0, TOTAL + 60.0, 0.1)
        self.route_index = 0
        self.lg = FakeLaneGraph(junctions or {})

    def set_route_s(self, rs):
        self.route_index = int(round(rs / 0.1))


class FakeEgo:
    def __init__(self):
        self.speed = 0.0

    def get_velocity(self):
        ego = self

        class V:
            def length(self):
                return ego.speed
        return V()


def make_ap(planner):
    a = AutoPilot()
    a.setup(world=None, world_map=None, waypoint_planner=planner,
            longitudinal_controller=VtdLongitudinalController(CFG),
            ego_vehicle=FakeEgo(), config=GlobalConfig())
    a.kr_rules = KrRules(CFG)
    return a


def sig_at(ap, route_s, v):
    """route_s·속도를 주고 이번 틱 지시등을 얻는다 (apply 경로 그대로)."""
    ap._waypoint_planner.set_route_s(route_s)
    ap._vehicle.speed = v
    ap.kr_rules.apply(VehicleControl(steer=0.0, accel=1.0), 12.5, ap)
    return ap.kr_rules.last_turn_signal


# 도로 10(일반) → 20/21(교차로 연결로, junction 7) → 30(일반)
TURN_LANES = [(10, 0, -1), (20, 0, -1), (21, 0, -1), (30, 0, -1)]
TURN_CUM = [0.0, 100.0, 110.0, 125.0]
TURN_LENS = [100.0, 10.0, 15.0, 200.0]
TURN_JUNC = {(10, 0, -1): -1, (20, 0, -1): 7, (21, 0, -1): 7, (30, 0, -1): -1}
TURN_EV = {'kind': 'turn_left', 's': 100.0, 'lane': (20, 0, -1), 'junction': 7}


@pytest.fixture()
def turn_ap():
    return make_ap(FakePlanner([TURN_EV], TURN_LANES, TURN_CUM, TURN_LENS, TURN_JUNC))


def test_turn_lead_and_hold(turn_ap):
    """lead 거리 밖은 소등, 안쪽은 점등, 연결로 끝(125)을 지나면 소등."""
    v = 10.0
    lead = max(v * SIG['turn_lead_s'], SIG['lead_min_m'])
    assert sig_at(turn_ap, 100.0 - lead - 5.0, v) == SIG_OFF
    assert sig_at(turn_ap, 100.0 - lead + 1.0, v) == SIG_LEFT
    assert sig_at(turn_ap, 100.0, v) == SIG_LEFT          # 연결로 진입
    assert sig_at(turn_ap, 120.0, v) == SIG_LEFT          # 연결로 안 — 유지
    assert sig_at(turn_ap, 126.0, v) == SIG_OFF           # 연결로 끝 통과


def test_turn_end_spans_chained_connectors(turn_ap):
    """연결로가 둘(20,21) 이어져도 마지막 끝(125)까지 유지한다."""
    ivs = signal_intervals(turn_ap._waypoint_planner)
    assert len(ivs) == 1
    assert ivs[0]['ev_s'] == pytest.approx(100.0)
    assert ivs[0]['end_s'] == pytest.approx(125.0)


def test_lead_min_m_covers_standstill(turn_ap):
    """적신호 대기(v=0)에서도 거리 하한 안이면 켜진다 — 시간 기준만 쓰면 0 이 된다."""
    assert sig_at(turn_ap, 100.0 - SIG['lead_min_m'] + 1.0, 0.0) == SIG_LEFT
    assert sig_at(turn_ap, 100.0 - SIG['lead_min_m'] - 5.0, 0.0) == SIG_OFF


def test_no_reactivation_after_pass(turn_ap):
    """구간을 지난 뒤에는 다시 켜지지 않는다 (깜빡임 실사고 §6-9)."""
    seen = [sig_at(turn_ap, rs, 10.0) for rs in np.arange(60.0, 200.0, 1.0)]
    on = [i for i, s in enumerate(seen) if s != SIG_OFF]
    assert on, '한 번은 켜져야 한다'
    assert on == list(range(on[0], on[-1] + 1)), f'점등이 끊겼다 재점등: {seen}'


def test_lane_change_window():
    """LC 는 창 시작 전 lead 부터 블렌드 끝(창끝과 전이길이 중 짧은 쪽)까지."""
    ev = {'kind': 'lane_change_right', 's': 200.0, 'window_s0': 200.0,
          'window_s1': 260.0, 'from_lane': (10, 0, -1), 'to_lane': (10, 0, -2)}
    ap = make_ap(FakePlanner([ev]))
    ivs = signal_intervals(ap._waypoint_planner)
    assert ivs[0]['end_s'] == pytest.approx(225.0)        # 200 + LC_TRANSITION_M

    v = 10.0
    lead = max(v * SIG['lc_lead_s'], SIG['lead_min_m'])
    assert sig_at(ap, 200.0 - lead - 5.0, v) == SIG_OFF
    assert sig_at(ap, 200.0 - lead + 1.0, v) == SIG_RIGHT
    assert sig_at(ap, 226.0, v) == SIG_OFF


def test_overlap_prefers_nearer_then_turn():
    """겹치면 남은거리 짧은 쪽. 동률이면 회전 우선 (SPEC §3.3)."""
    lc = {'kind': 'lane_change_left', 's': 100.0, 'window_s0': 100.0, 'window_s1': 160.0}
    turn = {'kind': 'turn_right', 's': 110.0, 'lane': (20, 0, -1), 'junction': 7}
    ap = make_ap(FakePlanner([lc, turn], TURN_LANES, TURN_CUM, TURN_LENS, TURN_JUNC))
    # 95 m: LC 가 5 m 앞, 회전은 15 m 앞 → 가까운 LC
    assert sig_at(ap, 95.0, 10.0) == SIG_LEFT

    tie_lc = {'kind': 'lane_change_left', 's': 100.0, 'window_s0': 100.0, 'window_s1': 160.0}
    tie_turn = {'kind': 'turn_right', 's': 100.0, 'lane': (20, 0, -1), 'junction': 7}
    ap2 = make_ap(FakePlanner([tie_lc, tie_turn], TURN_LANES, TURN_CUM, TURN_LENS, TURN_JUNC))
    assert sig_at(ap2, 95.0, 10.0) == SIG_RIGHT           # 동률 → 회전(우) 우선


def test_no_events_stays_off():
    """이벤트 없는 경로(목 플래너 포함)에서는 개입하지 않는다."""
    ap = make_ap(FakePlanner())
    assert sig_at(ap, 50.0, 10.0) == SIG_OFF
    assert ap.kr_rules.last_sig_src is None


@pytest.mark.skipif(not (GRAPH.exists() and ROUTE_PKL.exists()),
                    reason='lane_graph.pkl / route_example.pkl 없음')
def test_real_route_intervals_are_sane():
    """실제 경로 자산으로 구간이 만들어지고 순서·길이가 성립하는지."""
    from vtd_adapter.lanegraph import LaneGraph
    from vtd_adapter.route import VtdRoutePlanner

    with open(ROUTE_PKL, 'rb') as f:
        route = pickle.load(f)
    planner = VtdRoutePlanner(LaneGraph(str(GRAPH)), route, CFG, config=GlobalConfig())
    ivs = signal_intervals(planner)

    n_ev = sum(1 for e in route.get('events', [])
               if str(e['kind']).startswith(('turn_', 'lane_change_')))
    assert len(ivs) == n_ev
    assert ivs == sorted(ivs, key=lambda d: d['ev_s'])
    for iv in ivs:
        assert iv['end_s'] > iv['ev_s'], iv
        assert iv['sig'] in (SIG_LEFT, SIG_RIGHT)
