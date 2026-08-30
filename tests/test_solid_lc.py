"""
S2.2.05 실선 차선변경 금지 — 제어기 쪽 가드.

경로 생성(build_route)은 이미 점선 구간에서만 차선변경을 만든다
(has_broken / dashed_runs). 이 테스트가 지키는 것은 **제어기가 마지막 관문**
이라는 것: 손으로 만든·옛 route.pkl 이 실선 횡단을 요구해도 VtdRoutePlanner 가
넘는 자리를 점선 안으로 밀어 넣거나, 못 하면 경고를 남긴다.

  · LaneGraph.dashed_runs 가 점선 판정의 단일 출처 (build_route 도 이걸 쓴다)
  · _clip_to_dashed: 실선이 섞이면 점선 구간으로 좁힌다
  · 전이거리를 채우는 가장 이른 점선 구간을 고른다
  · 점선이 아예 없으면 기하는 그대로 두고 경고만 (임의로 옮기면 목표 차로 미달)
"""
import pathlib
import pickle

import pytest

from vtd_adapter.config import load_params_yaml
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.route import VtdRoutePlanner
from conftest import PARAMS_YAML

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
# 테스트 기준은 tests/fixtures 에 고정 — data/route.pkl 은 사용자 작업용이라
# 경로 시각화·대회 CSV 투입으로 언제든 바뀐다 (tests/fixtures/README.md).
ROUTE_PKL = ROOT / 'tests' / 'fixtures' / 'route.pkl'
CFG = load_params_yaml(PARAMS_YAML)

KEY = (10, 0, -1)


class FakeLaneGraph:
    """dashed_runs 만 쓰는 표면 — 실제 구현을 그대로 빌려 쓴다."""

    MARK_JOIN_M = LaneGraph.MARK_JOIN_M
    dashed_runs = LaneGraph.dashed_runs

    def __init__(self, marks):
        # mark 튜플: (s0, s1, type, color, lane_change_ok)
        self.lanes = {KEY: {'left_mark': marks, 'right_mark': marks}}


def planner_with(marks):
    """_build 를 돌리지 않고 _clip_to_dashed 만 시험한다."""
    pl = object.__new__(VtdRoutePlanner)
    pl.lg = FakeLaneGraph(marks)
    pl.lc_solid_warnings = []
    pl.lc_clipped = []
    return pl


def test_dashed_runs_merges_and_splits():
    """인접 점선은 잇고, 실선에서 끊는다."""
    lg = FakeLaneGraph([(0.0, 10.0, 'broken', 'white', True),
                        (10.0, 20.0, 'broken', 'white', True),
                        (20.0, 30.0, 'solid', 'white', False),
                        (30.0, 45.0, 'broken', 'white', True)])
    assert lg.dashed_runs(KEY, 'left') == [(0.0, 20.0), (30.0, 45.0)]


def test_all_dashed_is_noop():
    """창 전체가 점선이면 손대지 않는다 (현재 대회 경로가 이 경우)."""
    pl = planner_with([(0.0, 100.0, 'broken', 'white', True)])
    assert pl._clip_to_dashed(KEY, 0.0, 10.0, 60.0, 'left') == (10.0, 60.0)
    assert pl.lc_clipped == []
    assert pl.lc_solid_warnings == []


def test_leading_solid_is_clipped_forward():
    """창 앞부분이 실선이면 시작점을 점선 시작으로 민다."""
    pl = planner_with([(0.0, 30.0, 'solid', 'white', False),
                       (30.0, 100.0, 'broken', 'white', True)])
    w0, w1 = pl._clip_to_dashed(KEY, 0.0, 10.0, 80.0, 'left')
    assert (w0, w1) == (30.0, 80.0)
    assert len(pl.lc_clipped) == 1
    assert pl.lc_solid_warnings == []


def test_trailing_solid_is_clipped_back():
    """창 뒷부분이 실선이면 끝점을 점선 끝으로 당긴다."""
    pl = planner_with([(0.0, 40.0, 'broken', 'white', True),
                       (40.0, 100.0, 'solid', 'white', False)])
    assert pl._clip_to_dashed(KEY, 0.0, 5.0, 80.0, 'left') == (5.0, 40.0)


def test_picks_earliest_run_that_fits_transition():
    """점선 조각이 여럿이면 전이거리를 채우는 **가장 이른** 것."""
    pl = planner_with([(0.0, 8.0, 'broken', 'white', True),      # 8 m — 부족
                       (8.0, 20.0, 'solid', 'white', False),
                       (20.0, 60.0, 'broken', 'white', True),    # 40 m — 충분
                       (60.0, 70.0, 'solid', 'white', False),
                       (70.0, 140.0, 'broken', 'white', True)])  # 더 길지만 늦다
    w0, w1 = pl._clip_to_dashed(KEY, 0.0, 0.0, 130.0, 'left')
    assert (w0, w1) == (20.0, 60.0)
    assert w1 - w0 >= VtdRoutePlanner.LC_TRANSITION_M


def test_falls_back_to_longest_when_none_fits():
    """전이거리를 채우는 구간이 없으면 가장 긴 점선을 쓴다."""
    pl = planner_with([(0.0, 5.0, 'broken', 'white', True),
                       (5.0, 20.0, 'solid', 'white', False),
                       (20.0, 32.0, 'broken', 'white', True)])   # 12 m — 최장
    assert pl._clip_to_dashed(KEY, 0.0, 0.0, 40.0, 'left') == (20.0, 32.0)


def test_no_dashed_warns_and_keeps_geometry(capsys):
    """점선이 아예 없으면 기하는 그대로, 경고만 남긴다."""
    pl = planner_with([(0.0, 100.0, 'solid', 'white', False)])
    assert pl._clip_to_dashed(KEY, 0.0, 10.0, 60.0, 'left') == (10.0, 60.0)
    assert len(pl.lc_solid_warnings) == 1
    assert 'S2.2.05' in capsys.readouterr().out


def test_base_offset_applied():
    """점선 구간은 차로 s, 창은 route_s — base 만큼 옮겨 비교해야 한다."""
    pl = planner_with([(0.0, 30.0, 'solid', 'white', False),
                       (30.0, 100.0, 'broken', 'white', True)])
    assert pl._clip_to_dashed(KEY, 500.0, 510.0, 580.0, 'left') == (530.0, 580.0)


@pytest.mark.skipif(not (GRAPH.exists() and ROUTE_PKL.exists()),
                    reason='lane_graph.pkl / route.pkl 없음')
def test_real_route_needs_no_clipping():
    """대회 경로는 build_route 게이트를 탔으므로 제어기가 손댈 일이 없다.
    여기서 clipped/warning 이 생기면 build_route 쪽 게이트가 깨진 것이다."""
    import sys
    sys.path.insert(0, str(ROOT / 'team_code'))
    from config import GlobalConfig

    with open(ROUTE_PKL, 'rb') as f:
        route = pickle.load(f)
    pl = VtdRoutePlanner(LaneGraph(str(GRAPH)), route, CFG, config=GlobalConfig())
    assert pl.lc_solid_warnings == []
    assert pl.lc_clipped == []
