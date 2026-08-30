"""
차선변경 회랑(연속 점선 길이) — 창이 짧아 LC 가 실패하는 경로를 애초에 안 고르게.

2026-08-21 실사고: 창 6.1 m 차선변경이 실패 → 헤딩오차 46°, 조향 풀락 포화,
도로이탈 + courseRespawn. 실선을 밟은 게 아니라 **전이를 끝낼 거리가 없었다.**

계약:
  · dashed_corridor_m 은 차로 경계를 넘어 **뒤로** 이어붙인다 (창이 뒤로 당겨지므로)
  · 차로 안에서 점선이 끝나면 그 조각 길이가 곧 회랑
  · 이웃이 없거나 점선이 없으면 0
  · 탐색 비용은 회랑이 전이거리를 못 채운 만큼 급증한다 (금지가 아니라 가중)
  · 실제 대회 경로는 회랑이 전이거리를 넘는다 (회귀 검사)
"""
import pathlib
import pickle

import pytest

from vtd_adapter.lanegraph import LaneGraph
from conftest import PARAMS_YAML  # noqa: F401  (경로 설정 목적)

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
# 테스트 기준은 tests/fixtures 에 고정 — data/route.pkl 은 사용자 작업용이라
# 경로 시각화·대회 CSV 투입으로 언제든 바뀐다 (tests/fixtures/README.md).
ROUTE_PKL = ROOT / 'tests' / 'fixtures' / 'route.pkl'

D = lambda a, b: (a, b, 'broken', 'white', True)      # noqa: E731  점선
S = lambda a, b: (a, b, 'solid', 'white', False)      # noqa: E731  실선


class FakeLaneGraph:
    """dashed_corridor_m 이 쓰는 표면만 실제 구현에서 빌려 온다."""

    MARK_JOIN_M = LaneGraph.MARK_JOIN_M
    MARK_EDGE_M = LaneGraph.MARK_EDGE_M
    CORRIDOR_MAX_LANES = LaneGraph.CORRIDOR_MAX_LANES
    dashed_runs = LaneGraph.dashed_runs
    dashed_corridor_m = LaneGraph.dashed_corridor_m
    _corridor_extend = LaneGraph._corridor_extend

    def __init__(self, spec):
        """spec: {key: (길이, 마크, 이웃있음, prev)}"""
        self.lanes = {k: {'junction': -1, 'left_mark': m, 'right_mark': m,
                          '_len': L, '_nb': nb, '_prev': list(pv)}
                      for k, (L, m, nb, pv) in spec.items()}
        self._corridor_cache = {}

    def length(self, k):
        return self.lanes[k]['_len']

    def neighbor(self, k, side):
        return ('nb',) if self.lanes[k]['_nb'] else None

    def predecessors(self, k):
        return self.lanes[k]['_prev']

    def successors(self, k):
        return [o for o in self.lanes if k in self.lanes[o]['_prev']]


A, B, C = (1, 0, -1), (2, 0, -1), (3, 0, -1)


def test_no_neighbor_is_zero():
    lg = FakeLaneGraph({A: (50.0, [D(0, 50)], False, [])})
    assert lg.dashed_corridor_m(A, 'left') == 0.0


def test_no_dashed_is_zero():
    lg = FakeLaneGraph({A: (50.0, [S(0, 50)], True, [])})
    assert lg.dashed_corridor_m(A, 'left') == 0.0


def test_run_ending_inside_lane():
    """점선이 차로 안에서 끝나면 그 조각이 곧 회랑 (뒤로 못 당긴다)."""
    lg = FakeLaneGraph({A: (50.0, [D(10, 22), S(22, 50)], True, [])})
    assert lg.dashed_corridor_m(A, 'left') == pytest.approx(12.0)


def test_extends_backward_across_lanes():
    """차로 셋이 이어져 전부 점선이면 합쳐서 잰다 — laneSection 이 쪼개도 무관."""
    lg = FakeLaneGraph({
        A: (20.0, [D(0, 20)], True, []),
        B: (10.0, [D(0, 10)], True, [A]),
        C: (15.0, [D(0, 15)], True, [B]),
    })
    assert lg.dashed_corridor_m(C, 'left') == pytest.approx(45.0)


def test_backward_stops_at_solid():
    """앞 차로 끝이 실선이면 거기서 끊긴다."""
    lg = FakeLaneGraph({
        A: (20.0, [D(0, 12), S(12, 20)], True, []),
        B: (15.0, [D(0, 15)], True, [A]),
    })
    assert lg.dashed_corridor_m(B, 'left') == pytest.approx(15.0)


def test_backward_stops_at_fork():
    """앞이 갈림길이면 어느 쪽으로 올지 몰라 보수적으로 멈춘다."""
    lg = FakeLaneGraph({
        A: (20.0, [D(0, 20)], True, []),
        B: (20.0, [D(0, 20)], True, []),
        C: (15.0, [D(0, 15)], True, [A, B]),
    })
    assert lg.dashed_corridor_m(C, 'left') == pytest.approx(15.0)


def test_cost_grows_when_corridor_short():
    """탐색 비용: 회랑이 전이거리를 못 채운 만큼 커진다 (금지가 아니라 가중)."""
    import build_route as br

    class Stub:
        def __init__(self, corr):
            self.corr = corr

        def dashed_corridor_m(self, key, side):
            return self.corr

    full = br.lc_cost(Stub(br.LC_MIN_CORRIDOR_M), A, 'left')
    short = br.lc_cost(Stub(1.5), A, 'left')
    assert full == pytest.approx(br.LC_PENALTY)
    assert short > full
    # 1.5 m 회랑은 우회 수백 m 와 맞먹어야 대안이 있을 때 안 골린다
    assert short - full > 400.0
    # 그래도 유한하다 — 유일한 길이면 여전히 고를 수 있어야 한다
    assert short < float('inf')


@pytest.mark.skipif(not (GRAPH.exists() and ROUTE_PKL.exists()),
                    reason='lane_graph.pkl / route.pkl 없음')
def test_real_route_lc_corridors_fit_transition():
    """대회 경로의 차선변경은 전부 전이거리를 채우는 회랑 위에 있어야 한다."""
    import build_route as br

    lg = LaneGraph(str(GRAPH))
    with open(ROUTE_PKL, 'rb') as f:
        route = pickle.load(f)
    lcs = [e for e in route['events'] if e['kind'].startswith('lane_change')]
    assert lcs, '이 경로에 차선변경이 없다 — 회귀 검사가 무의미하다'
    for e in lcs:
        side = e['kind'].split('_')[-1]
        corr = lg.dashed_corridor_m(tuple(e['from_lane']), side)
        assert corr >= br.LC_MIN_CORRIDOR_M, (e['kind'], e['s'], corr)
