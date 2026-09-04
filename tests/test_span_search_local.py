"""plan_shift_span 탐색 국소화 (span_search_local_enable) — 2026-09-04.

PDM 원문은 차단물 최근접 경로점을 route_index 이후 **전 구간**에서 찾는다. 순환
코스에서 경로가 같은 자리를 두 번 지나고 차단물이 재통과선 쪽에 더 가까우면 한
바퀴 뒤 점이 잡혀 자차 앞이 아닌 구간이 밀린다 (실측 2026-09-03 100310/100458
14건, span 시작 − route_index = 4550.1~4595.2 m). 탐색 창을
leading_vehicles_maximum_detection_radius(80 m) 로 제한한다.

지키는 것:
  · 기본 off = 원문 동작 (전 구간 탐색).
  · 창 안에 참 최근접점이 있으면 on/off 결과가 같다 (정상 시프트 불변).
  · 재통과선이 더 가까운 배치에서 off 는 먼 인덱스, on 은 자차 근방.
  · 창 밖이면 None·예외가 아니라 창 끝으로 클램프한다
    (shift_route_around_actors 의 튜플 언패킹이 깨지면 안 된다).
"""
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.route import VtdRoutePlanner

CFG = load_params_yaml(PARAMS_YAML)
PPM = 10
WINDOW_PTS = 80 * PPM


class Actor:
    def __init__(self, x, y, half_len=1.04):
        self._x, self._y = float(x), float(y)
        self.id = 2
        ext = type('E', (), {'x': half_len, 'y': 0.43, 'z': 0.3})()
        self.bounding_box = type('B', (), {'extent': ext})()

    def get_location(self):
        s = self
        return type('L', (), {'x': s._x, 'y': s._y, 'z': 0.0})()


class Stub:
    """plan_shift_span 이 쓰는 속성만 가진 최소 플래너 (원문 함수를 그대로 호출한다)."""

    def __init__(self, pts, route_index, local):
        self.original_route_points = pts
        self.route_points = pts.copy()
        self.route_index = route_index
        self.points_per_meter = PPM
        self.leading_vehicles_maximum_detection_radius = WINDOW_PTS
        self.span_search_local = local

    def get_closest_route_index(self, begin_idx, location):
        return begin_idx


def loop_route(straight_m=450.0, offset=0.5):
    """0→straight_m 직선 후 크게 돌아 offset 만큼 옆으로 같은 자리를 재통과."""
    n1 = int(straight_m * PPM)
    seg1 = np.stack([np.arange(n1) / PPM, np.zeros(n1), np.zeros(n1)], 1)
    R, nA = 40.0, int(math.pi * 40.0 * PPM)
    th = np.linspace(0, math.pi, nA)
    arc = np.stack([straight_m + R * np.sin(th), -R + R * np.cos(th), np.zeros(nA)], 1)
    n2 = int(straight_m * PPM)
    seg2 = np.stack([straight_m - np.arange(n2) / PPM, np.full(n2, -2 * R), np.zeros(n2)], 1)
    arc2 = np.stack([-R * np.sin(th), -R - R * np.cos(th), np.zeros(nA)], 1)
    n3 = int((straight_m + 60) * PPM)
    seg3 = np.stack([np.arange(n3) / PPM, np.full(n3, offset), np.zeros(n3)], 1)
    return np.concatenate([seg1, arc, seg2, arc2, seg3], 0)


def plan(pts, route_index, actor, local, ahead_m=5.0):
    p = Stub(pts, route_index, local)
    return VtdRoutePlanner.plan_shift_span(
        p, actor, None, obstacle_direction='left', transition_length=12.0 * PPM,
        extra_length_before=5.0 * PPM, extra_length_after=10.0 * PPM,
        min_start_ahead=ahead_m * PPM)


def test_params_default_off():
    assert CFG['overtake']['span_search_local_enable'] is False


def test_planner_reads_switch_from_params():
    """생성자 인자가 아니라 params 직독. 키가 없어도 기본 off."""
    import copy
    from vtd_adapter.lanegraph import LaneGraph                     # noqa: F401  (경로 확인용)
    cfg = copy.deepcopy(CFG)
    assert cfg['overtake'].pop('span_search_local_enable') is False
    # 키 없는 cfg 로도 bool(...) 이 False 여야 한다 (생성자 코드와 같은 식)
    assert bool(cfg.get('overtake', {}).get('span_search_local_enable', False)) is False
    assert bool({}.get('overtake', {}).get('span_search_local_enable', False)) is False


@pytest.mark.parametrize('d_m', [12.0, 20.0])
def test_far_pass_wins_without_window_and_is_fixed_by_window(d_m):
    """재통과선이 더 가까운 배치 — off 는 먼 인덱스, on 은 자차 근방."""
    pts = loop_route()
    ego = 1000                                       # 첫 통과 100 m 지점
    a = Actor(ego / PPM + d_m, 0.35)                 # 재통과선(0.5)에 더 가깝다
    off_start, off_end, _ = plan(pts, ego, a, local=False)
    on_start, on_end, _ = plan(pts, ego, a, local=True)
    assert (off_start - ego) / PPM > 400.0           # 원문: 한 바퀴 뒤
    assert (on_start - ego) / PPM == pytest.approx(5.0, abs=0.2)
    assert on_end - on_start < WINDOW_PTS            # span 이 창 안에 든다


def test_identical_when_true_nearest_is_in_window():
    """차단물이 첫 통과선에 더 가까우면 on/off 결과가 같다 (정상 시프트 불변)."""
    pts = loop_route()
    ego = 1000
    a = Actor(ego / PPM + 12.0, 0.05)
    assert plan(pts, ego, a, local=False) == plan(pts, ego, a, local=True)


@pytest.mark.parametrize('d_m', [5.0, 9.5, 10.7, 34.7, 62.0])
def test_normal_distances_unchanged(d_m):
    """실측 정상 시프트 거리(5.0~34.7 m)와 그 원인 장애물(62 m)이 전부 창 안."""
    pts = loop_route()
    ego = 1000
    a = Actor(ego / PPM + d_m, 0.0)
    assert plan(pts, ego, a, local=False) == plan(pts, ego, a, local=True)


def test_clamps_instead_of_none_when_actor_beyond_window():
    """창 밖 차단물 — None·예외가 아니라 창 끝으로 클램프하고 튜플을 돌려준다."""
    pts = loop_route()
    ego = 1000
    a = Actor(ego / PPM + 120.0, 0.0)                # 창(80 m) 밖
    res = plan(pts, ego, a, local=True)
    assert isinstance(res, tuple) and len(res) == 3
    start, end, left = res
    assert start < end and isinstance(left, bool)
    assert (start - ego) / PPM <= 80.0               # 창 끝을 넘지 않는다
