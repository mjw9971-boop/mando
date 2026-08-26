"""
gen_scenarios 스폰-경로 정합 게이트 (2026-08-26 실기 사고 회귀).

사고: walk 경로 완주속도_01_속도전환2 — XML 의 Ego 스폰(Path01 첫 waypoint +
PathRef StartS/StartLane)은 진행방향을 못 박는데, 첫 waypoint 뒤(−s)로도 둘째
waypoint 도로에 닿는 우회가 있어 VTD 가 Path 를 역방향 차로망으로 해석,
반대편 연결로 (3174,0,-1) 에 스폰 → heading 차 ~175° → 조향 포화·courseRespawn.

lane_graph.pkl 은 .gitignore 대상이라 없을 수 있다 → 그때는 skip.
"""
import math
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'

pytestmark = pytest.mark.skipif(not GRAPH.exists(),
                                reason='data/lane_graph.pkl 없음 (gitignore 대상)')


@pytest.fixture(scope='module')
def lg():
    from vtd_adapter.lanegraph import LaneGraph
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def gen_cfg():
    from gen_scenarios import load_themes
    _routes, _themes, cfg = load_themes()
    return cfg


CHAIN_2818 = [(2818, 0, -1), (2818, 1, -1), (2818, 2, -2), (2818, 3, -3),
              (2818, 4, -2), (2818, 5, -2), (2818, 6, -1), (3084, 0, -1),
              (2816, 0, -2)]          # 실기 사고 경로의 실제 앞부분
CHAIN_30 = [(30, 0, -1), (550, 0, -1), (72, 0, -3)]   # waypoints.csv 경로 앞부분


def synth_rt(lg, chain, start_s):
    """명시한 차로 체인으로 합성 route. 체인이 successor 로 이어지는지 검증한다.

    spawn_gate 가 보는 필드(lanes/cum_s/lengths/total_length/start_s_in_lane)만
    만든다 — build_route 관례(cum[0] = -start_s)와 같다.
    """
    for a, b in zip(chain, chain[1:]):
        assert b in lg.successors(a), f'{a} → {b} 가 successor 가 아니다 (체인 오기)'
    lengths = [lg.length(k) for k in chain]
    cum = [-start_s]
    for i in range(1, len(chain)):
        cum.append(cum[i - 1] + lengths[i - 1])
    return {'lanes': chain, 'cum_s': cum, 'lengths': lengths,
            'total_length': cum[-1] + lengths[-1], 'start_s_in_lane': start_s}


def test_reverse_ambiguous_start_rejected(lg, gen_cfg):
    """대향 차로 스폰을 만드는 케이스(실기 사고 재현): 도로 2818 시작.

    시작 waypoint 뒤가 교차로 94 — 반대 차로 (2818,0,1)→(3174,0,-1) 로 둘째
    waypoint 도로(2816)까지 주행 가능한 우회가 있다 → 게이트가 폐기해야 한다.
    """
    from gen_scenarios import spawn_gate
    rt = synth_rt(lg, CHAIN_2818, 8.0)
    why = spawn_gate(lg, rt, gen_cfg)
    assert why is not None, '역방향 스폰 모호 케이스를 게이트가 통과시켰다'
    assert '3174' in why            # 실측 스폰 차로가 역방향 해석으로 재현돼야 한다


def test_reverse_spawn_reproduces_field_observation(lg):
    """역방향 되짚기가 실기 스폰(3174,0,-1)@≈7.5 m, heading ≈ −4° 를 재현하는지."""
    from gen_scenarios import _reverse_spawn
    rt = synth_rt(lg, CHAIN_2818, 8.0)
    wp1_road_s = lg.road_s_at((2818, 0, -1), 0.5)
    rev = _reverse_spawn(lg, rt, wp1_road_s)
    assert rev is not None
    lane, s, hdg = rev
    assert lane == (3174, 0, -1)
    assert s == pytest.approx(7.5, abs=0.5)
    assert abs(math.degrees(hdg)) < 15.0        # 실측 −3.4°


def test_dead_end_behind_start_passes(lg, gen_cfg):
    """정상 케이스(waypoints.csv 기본 경로 시작): 도로 30.

    반대 차로 (30,0,1) 은 successors 가 없어(맵 가장자리) 역방향 해석으로 Path 를
    완성할 수 없다 → 게이트 통과.
    """
    from gen_scenarios import spawn_gate
    assert lg.successors((30, 0, 1)) == []      # 판별 속성 자체를 고정
    rt = synth_rt(lg, CHAIN_30, 0.03)
    assert spawn_gate(lg, rt, gen_cfg) is None


def test_build_route_start_respects_heading(lg):
    """(작업1-d 회귀) build_route 의 시작 차로 선택이 CSV 첫 두 점 방향을 따른다.

    같은 지점이라도 yaw 가 서쪽이면 (2818,0,-1), 동쪽이면 반대 차로가 잡혀야 한다
    — lg.locate 의 yaw 필터(±70°)가 시작 차로 선택 경로에 실제로 걸려 있다.
    """
    x, y, _z, h_west = lg.point_at((2818, 0, -1), 8.0)
    m = lg.locate(x, y, yaw=h_west)
    assert m is not None and m.lane == (2818, 0, -1)
    m_rev = lg.locate(x, y, yaw=h_west + math.pi)
    assert m_rev is not None and m_rev.lane != (2818, 0, -1)
    assert lg.lanes[m_rev.lane]['dir'] != lg.lanes[(2818, 0, -1)]['dir'] \
        or m_rev.lane[0] != 2818
