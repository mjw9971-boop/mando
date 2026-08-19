"""
시나리오 초기위치가 (465, 2, -1), t ≈ 0.05 로 매칭되는지 (SPEC §1.1).

lane_graph.pkl 은 .gitignore 대상이라 없을 수 있다 → 그때는 skip.
"""
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'

# SPEC §1.1 검증 완료 값
START_X, START_Y = 508.80, -168.29
START_YAW = 0.52727821887430437
EXPECT_ROAD, EXPECT_LANE_ID = 465, -1
EXPECT_T = 0.052

pytestmark = pytest.mark.skipif(not GRAPH.exists(),
                                reason='data/lane_graph.pkl 없음 (gitignore 대상)')


def test_scenario_start_matches_road_465_lane_minus1():
    from hlfma.core.lanegraph import LaneGraph
    lg = LaneGraph(str(GRAPH))
    m = lg.locate(START_X, START_Y, yaw=START_YAW)

    assert m is not None, '초기위치에서 차로 매칭 실패'
    road_id, _section, lane_id = m.lane
    assert road_id == EXPECT_ROAD
    assert lane_id == EXPECT_LANE_ID
    # VTD ModuleManager 의 Off=0.052 와 일치해야 한다
    assert abs(abs(m.t) - EXPECT_T) < 0.05
    assert m.dist < 0.5


def test_locate_rejects_opposite_heading():
    """yaw 를 180도 뒤집으면 같은 차로로 매칭되면 안 된다."""
    import math

    from hlfma.core.lanegraph import LaneGraph
    lg = LaneGraph(str(GRAPH))
    m = lg.locate(START_X, START_Y, yaw=START_YAW + math.pi)
    assert m is None or m.lane[2] != EXPECT_LANE_ID


@pytest.mark.skip(reason='TODO: Perception 구현 후 활성화')
def test_speed_estimated_from_position_delta():
    """ego 에 속도 필드가 없으므로 위치 미분으로 추정해야 한다 (SPEC §1.1)."""


@pytest.mark.skip(reason='TODO: Perception 구현 후 활성화')
def test_object_classification_by_size():
    """보행자 width 0.5–0.8 / height 1.5–2.0 / length < 1.0, 차량 length > 3.0."""


@pytest.mark.skip(reason='TODO: Perception 구현 후 활성화')
def test_missing_object_coasts_for_coast_s():
    """직전 틱에 있던 id 가 사라지면 COAST_S 동안 외삽 유지."""
