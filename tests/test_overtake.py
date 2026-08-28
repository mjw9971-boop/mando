"""
정적 장애물 회피 시프트 (kr_rules) — 2026-08-28 차로폭협착_01 실사고 회귀.

사고: 정차 차량에 막혀 353.7/821 m 에서 49 s 정지 후 시나리오 종료(미완주).
PDM 은 장애물을 "속도 0 인 선행차" 로 보고 뒤에 설 뿐이고, 회피를 발동하던
_manage_route_obstacle_scenarios 는 이식 때 stub 이 됐다.

계약:
  · 막힌 채 정지가 trigger_s 지속되면 옆 차로로 경로를 민다 (좌측 우선)
  · 게이트 — 이웃 있음 · 교차로 아님 · 점선 회랑 충분(S2.2.05) · 목표 차로 비어 있음
  · 1회만 (래치). 지나가면 원복해 다음 장애물에 다시 쓴다
  · 시프트가 lat_shift 를 갱신하므로 지시등이 자동으로 따라온다
"""
import math
import pathlib
import pickle
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter import frame
from vtd_adapter.actor import VtdActor
from vtd_adapter.config import load_params_yaml
from vtd_adapter.control import VtdLongitudinalController
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.route import VtdRoutePlanner

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from autopilot import AutoPilot                    # noqa: E402
from config import GlobalConfig                    # noqa: E402
from kr_rules import KrRules                       # noqa: E402
from vtd_adapter.carla_types import VehicleControl  # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE_PKL = ROOT / 'data' / 'route.pkl'
# 평범한 직진 구간 — 차선변경 블렌드 밖이어야 한다 (블렌드 위에 놓으면
# 장애물이 목표 차로로 매칭돼 clear 게이트가 잘못 걸린다).
IDX = 1250                   # lane (72,2,-2), 좌 이웃 (72,2,-1), 점선 회랑 105.7 m

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE_PKL.exists()),
                                reason='lane_graph.pkl / route.pkl 없음')


class FakeActorList(list):
    def filter(self, pat):
        p = pat.strip('*')
        return FakeActorList(a for a in self if p in a.type_id)


class FakeWorld:
    def __init__(self, actors):
        self._actors = FakeActorList(actors)

    def get_actors(self):
        return self._actors


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture()
def rig(lg):  # noqa: D401
    """실제 경로·지도 위에 자차와 장애물을 놓는다."""
    with open(ROUTE_PKL, 'rb') as f:
        route = pickle.load(f)
    pl = VtdRoutePlanner(lg, route, CFG, config=GlobalConfig())
    pl.route_index = IDX

    ego = VtdActor(0, 'vehicle')
    ego.type_id = 'vehicle.hyundai.ioniq6'
    ego.x, ego.y = pl.route_points[IDX, :2]
    ego.yaw_deg = float(pl.rotation_angles[IDX])
    ego.speed = 0.0
    ego.length, ego.width, ego.height = 4.848, 1.886, 1.507

    def actor_at(idx, oid=2, speed=0.0):
        """경로 위 idx 지점에 차를 놓는다 (자차 차로)."""
        a = VtdActor(oid, 'vehicle')
        a.x, a.y = pl.route_points[idx, :2]
        a.yaw_deg = float(pl.rotation_angles[idx])
        a.speed = speed
        a.length, a.width, a.height = 4.39, 1.81, 1.35
        return a

    def actor_on_left_lane(idx, oid=3, ahead_m=6.0):
        """왼쪽 이웃 차로 중심선 위에 차를 놓는다 (좌표계 부호에 안 휘둘리게
        lanegraph 중심선을 직접 쓴다)."""
        vx, vy = frame.from_carla_xy(*pl.route_points[idx, :2])
        m = lg.locate(vx, vy)
        target = lg.neighbor(m.lane, 'left')
        x, y, _z, hdg = lg.point_at(target, min(m.s + ahead_m, lg.length(target)))
        cx, cy = frame.to_carla_xy(x, y)
        a = VtdActor(oid, 'vehicle')
        a.x, a.y = cx, cy
        a.yaw_deg = frame.to_carla_yaw_deg(hdg)
        a.speed = 0.0
        a.length, a.width, a.height = 4.39, 1.81, 1.35
        return a

    def make(actors):
        ap = AutoPilot()
        ap.setup(world=FakeWorld([ego] + list(actors)), world_map=None,
                 waypoint_planner=pl,
                 longitudinal_controller=VtdLongitudinalController(CFG),
                 ego_vehicle=ego, config=GlobalConfig())
        ap.kr_rules = KrRules(CFG)
        return ap

    return pl, ego, actor_at, actor_on_left_lane, make


def run(ap, n):
    for _ in range(n):
        ap.kr_rules.apply(VehicleControl(steer=0.0, accel=0.0), 12.5, ap)
    return ap.kr_rules


def ticks_needed(kr):
    return kr.ot_ticks + 2


def test_no_blocker_no_shift(rig):
    pl, ego, actor_at, actor_on_left, make = rig
    ap = make([])
    kr = run(ap, 100)
    assert kr.ot_span is None


def test_moving_lead_is_not_static(rig):
    """앞차가 굴러가고 있으면 회피 대상이 아니다 (그냥 따라간다)."""
    pl, ego, actor_at, actor_on_left, make = rig
    ap = make([actor_at(IDX + 80, speed=OT['blocker_speed_max'] + 2.0)])
    kr = run(ap, 100)
    assert kr.ot_span is None


def test_blocked_triggers_left_shift(rig):
    """막힌 채 정지가 지속되면 좌측으로 경로를 민다."""
    pl, ego, actor_at, actor_on_left, make = rig
    before = pl.route_points.copy()
    ap = make([actor_at(IDX + 80)])
    kr = run(ap, ticks_needed(KrRules(CFG)))
    assert kr.ot_span is not None, kr.last_overtake
    assert kr.last_overtake == 'left'
    a, b = kr.ot_span
    assert np.abs(pl.route_points[a:b] - before[a:b]).max() > 1.0


def test_needs_sustained_block(rig):
    """한두 틱 멈춘 것으로는 발동하지 않는다."""
    pl, ego, actor_at, actor_on_left, make = rig
    ap = make([actor_at(IDX + 80)])
    kr = run(ap, 2)
    assert kr.ot_span is None


def test_occupied_target_lane_blocks(rig):
    """목표 차로에 차가 있으면 발동하지 않는다 (lc_clear 대용)."""
    pl, ego, actor_at, actor_on_left, make = rig
    side_car = actor_on_left(IDX)                        # 왼쪽 차로 위
    ap = make([actor_at(IDX + 80), side_car])
    kr = run(ap, ticks_needed(KrRules(CFG)))
    assert kr.ot_span is None
    assert kr.last_overtake and 'occupied' in kr.last_overtake


def test_shift_updates_lat_shift_for_signal(rig):
    """시프트가 lat_shift 를 갱신 → 지시등이 자동으로 따라온다."""
    pl, ego, actor_at, actor_on_left, make = rig
    ap = make([actor_at(IDX + 80)])
    kr = run(ap, ticks_needed(KrRules(CFG)))
    a, b = kr.ot_span
    assert np.abs(pl.lat_shift[a:b] - pl._lat_build[a:b]).max() > CFG['signal']['lat_shift_on_m']


def test_shift_once_then_restore(rig):
    """1회만 발동하고, 지나가면 경로를 원복한다."""
    pl, ego, actor_at, actor_on_left, make = rig
    orig = pl.route_points.copy()
    ap = make([actor_at(IDX + 80)])
    kr = run(ap, ticks_needed(KrRules(CFG)))
    a, b = kr.ot_span
    assert not np.allclose(pl.route_points[a:b], orig[a:b])

    pl.route_index = b + 1                                # 구간을 지났다
    run(ap, 1)
    assert kr.ot_span is None
    assert kr.last_overtake == 'restored'
    assert np.allclose(pl.route_points, orig)
    assert np.allclose(pl.lat_shift, pl._lat_build)


def test_disabled_switch(rig):
    """overtake.enabled=false 면 아무 일도 하지 않는다."""
    import copy
    pl, ego, actor_at, actor_on_left, make = rig
    ap = make([actor_at(IDX + 80)])
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['enabled'] = False
    ap.kr_rules = KrRules(cfg)
    kr = run(ap, 200)
    assert kr.ot_span is None
