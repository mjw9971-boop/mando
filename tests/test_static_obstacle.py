"""
정적 장애물 인식 — 자동차 크기가 아니어도 경로를 막으면 선행 객체다.

PDM 원문의 선행차 판정은 "경로 2.5 m 이내 + 진행방향 35° 이내" 인데, 정적
장애물(라바콘·공사 자재·비스듬히 선 차·파손물)은 heading 이 임의라 35° 조건에서
빠진다. 그러면 IDM 감속이 안 걸리고 OBB 충돌예측이 코앞에서 급제동하는 것만
남는다 (compute_target_speed_wrt_leading_vehicle 은 leading_vehicle_ids 만 본다).

계약:
  · 정지(< obstacle_speed_max) 객체는 heading 무관하게 판정한다
  · 막는가 = 경로 중심까지 횡거리 < 차폭/2 + 객체폭/2 + clearance (크기 비례)
  · 갓길에 비켜 선 물체는 잡지 않는다
  · 움직이는 객체는 원문 규칙(2.5 m / 35°) 그대로
"""
import math
import pathlib
import pickle

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.actor import VtdActor
from vtd_adapter.config import load_params_yaml
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.route import VtdRoutePlanner

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
# 테스트 기준은 tests/fixtures 에 고정 — data/route.pkl 은 사용자 작업용이라
# 경로 시각화·대회 CSV 투입으로 언제든 바뀐다 (tests/fixtures/README.md).
ROUTE_PKL = ROOT / 'tests' / 'fixtures' / 'route.pkl'
CFG = load_params_yaml(PARAMS_YAML)
PC = CFG['percep']
VW = CFG['vehicle']['width']
IDX = 1250

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE_PKL.exists()),
                                reason='lane_graph.pkl / route.pkl 없음')


@pytest.fixture(scope='module')
def pl():
    import sys
    sys.path.insert(0, str(ROOT / 'team_code'))
    from config import GlobalConfig
    with open(ROUTE_PKL, 'rb') as f:
        route = pickle.load(f)
    p = VtdRoutePlanner(LaneGraph(str(GRAPH)), route, CFG, config=GlobalConfig())
    p.route_index = IDX
    return p


def ego_actor(pl):
    a = VtdActor(0, 'vehicle')
    a.x, a.y = pl.route_points[IDX, :2]
    a.yaw_deg = float(pl.rotation_angles[IDX])
    a.length, a.width, a.height = 4.848, VW, 1.507
    return a


def obj(pl, ahead_idx, oid=2, speed=0.0, yaw_off_deg=0.0, lateral=0.0,
        size=(0.4, 0.4, 0.7)):
    """경로 ahead_idx 지점에서 lateral 만큼 옆으로 벗어난 객체."""
    a = VtdActor(oid, 'vehicle')
    yaw = math.radians(float(pl.rotation_angles[ahead_idx]))
    px, py = pl.route_points[ahead_idx, :2]
    a.x = px - lateral * math.sin(yaw)
    a.y = py + lateral * math.cos(yaw)
    a.yaw_deg = float(pl.rotation_angles[ahead_idx]) + yaw_off_deg
    a.speed = speed
    a.length, a.width, a.height = size
    return a


def leading(pl, actors):
    return set(pl.compute_leading_vehicles(actors, 0))


def test_small_static_object_on_path_is_detected(pl):
    """라바콘 크기(0.4 m)라도 경로 위에 서 있으면 선행 객체다."""
    ego = ego_actor(pl)
    cone = obj(pl, IDX + 100)                       # 10 m 앞, 경로 중심
    assert 2 in leading(pl, [ego, cone])


def test_heading_is_ignored_for_static(pl):
    """정지 객체는 90° 돌아서 있어도 잡는다 (원문 35° 조건에서 빠지던 것)."""
    ego = ego_actor(pl)
    cone = obj(pl, IDX + 100, yaw_off_deg=90.0)
    assert 2 in leading(pl, [ego, cone])


def test_moving_object_still_needs_heading(pl):
    """움직이는 객체는 원문 규칙 그대로 — 가로지르는 차는 선행차가 아니다."""
    ego = ego_actor(pl)
    crossing = obj(pl, IDX + 100, speed=5.0, yaw_off_deg=90.0, size=(4.4, 1.8, 1.4))
    assert 2 not in leading(pl, [ego, crossing])


def test_object_beside_path_is_ignored(pl):
    """갓길에 비켜 선 물체는 잡지 않는다 (과잉 정지 방지)."""
    ego = ego_actor(pl)
    far = obj(pl, IDX + 100, lateral=3.0)           # 3 m 옆
    assert 2 not in leading(pl, [ego, far])


def test_threshold_scales_with_object_width(pl):
    """막는 폭은 크기에 비례한다 — 넓은 물체는 더 멀리서도 막는 것으로 본다."""
    ego = ego_actor(pl)
    lat = VW / 2.0 + 0.9 + PC['obstacle_clearance_m'] - 0.1   # 폭 1.8 짜리 기준 안쪽
    # 둘 다 90도 돌려 원문 규칙(2.5 m + 35도)이 아니라 정적 규칙만 시험한다
    wide = obj(pl, IDX + 100, lateral=lat, yaw_off_deg=90.0, size=(4.4, 1.8, 1.4))
    narrow = obj(pl, IDX + 100, oid=3, lateral=lat, yaw_off_deg=90.0,
                 size=(0.4, 0.4, 0.7))
    ids = leading(pl, [ego, wide, narrow])
    assert 2 in ids, '넓은 물체는 잡혀야 한다'
    assert 3 not in ids, '같은 자리라도 좁은 물체는 안 막는다'


def test_fast_object_is_not_static(pl):
    """obstacle_speed_max 이상이면 정적이 아니다."""
    ego = ego_actor(pl)
    rolling = obj(pl, IDX + 100, speed=PC['obstacle_speed_max'] + 0.5, yaw_off_deg=90.0)
    assert 2 not in leading(pl, [ego, rolling])


def test_original_rule_still_applies_to_aligned_objects(pl):
    """방향이 맞는 객체는 원문 규칙(2.5 m)이 그대로 잡는다 — 크기 무관."""
    ego = ego_actor(pl)
    aligned = obj(pl, IDX + 100, lateral=2.0, size=(0.4, 0.4, 0.7))
    assert 2 in leading(pl, [ego, aligned])
