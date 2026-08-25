"""
vtd_adapter CARLA 흉내 층 검증 (phase2).

  (a) 직선 도로: VtdMap.get_waypoint → next(1) 가 lanegraph 와 일치
  (b) 우회전 연결로 (664,0,-1): yaw 변화 부호가 CARLA 관례(시계방향 양수)
      — 실제 lane_graph 데이터로 검증 (합성 데이터 금지)
  (c) 합성 9910 프레임의 객체가 get_actors 에서 vehicle/walker 로 분류
  (d) 자차 속도 추정이 기존 test_speed_estimate 기대값과 동일 경로로 VtdEgo 에 반영

주: 원래 지시의 후보 도로 1241/766 은 실측 Δhdg 가 +1°/+11° 라 우회전이 아니어서
(664,0,-1) (Δhdg −93.5°, 진입 차로 (62,0,-1)) 로 대체했다.
"""
import math
import pathlib

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter import frame
from vtd_adapter.carla_types import Location, RoadOption
from vtd_adapter.comm import build_frame, parse
from vtd_adapter.config import load_params_yaml
from vtd_adapter.control import VtdLongitudinalController, command_from_control
from vtd_adapter.carla_types import VehicleControl
from vtd_adapter.ego import EgoTracker
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.map import VtdMap
from vtd_adapter.types import RawPacket
from vtd_adapter.world import VtdWorld

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route_example.pkl'
CFG = load_params_yaml(PARAMS_YAML)

# test_lanegraph_locate 와 같은 검증 위치 (도로 465 차로 -1 위)
START_X, START_Y, START_YAW = 508.80, -168.29, 0.52727821887430437

pytestmark = pytest.mark.skipif(not GRAPH.exists(), reason='lane_graph.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def vmap(lg):
    return VtdMap(lg)


# ── frame.py 변환 규칙 ─────────────────────────────────────────────────────
def test_frame_roundtrip():
    x, y = 123.4, -56.7
    assert frame.from_carla_xy(*frame.to_carla_xy(x, y)) == (x, y)
    h = 1.234
    assert frame.from_carla_yaw_deg(frame.to_carla_yaw_deg(h)) == pytest.approx(h)


def test_frame_left_turn_becomes_ccw_negative_yaw():
    """VTD 좌회전(heading 증가) → CARLA yaw 감소 (시계방향 양수 관례)."""
    assert frame.to_carla_yaw_deg(0.1) < frame.to_carla_yaw_deg(0.0)


def test_steer_sign_roundtrip():
    """CARLA 양수 조향(미러 프레임의 우회전) → 9910 음수 rad (좌 + 관례)."""
    assert frame.steer_to_vtd(0.5, 0.48) == pytest.approx(-0.24)
    assert frame.steer_to_vtd(-1.0, 0.48) == pytest.approx(0.48)


# ── (a) 직선 도로 next(1) ─────────────────────────────────────────────────
def test_map_waypoint_matches_lanegraph_on_straight(lg, vmap):
    cx, cy = frame.to_carla_xy(START_X, START_Y)
    wp = vmap.get_waypoint(Location(cx, cy, 0.0))
    assert wp.key[0] == 465 and wp.key[2] == -1

    nxts = wp.next(1.0)
    assert len(nxts) >= 1
    nxt = nxts[0]
    ex, ey, _ez, ehdg = lg.point_at(wp.key, wp.s + 1.0)
    ecx, ecy = frame.to_carla_xy(ex, ey)
    assert nxt.transform.location.x == pytest.approx(ecx, abs=1e-6)
    assert nxt.transform.location.y == pytest.approx(ecy, abs=1e-6)
    assert nxt.transform.rotation.yaw == pytest.approx(frame.to_carla_yaw_deg(ehdg), abs=1e-6)
    # 진행 방향과 위치 증분이 일치 (미러 프레임 자기일관성)
    dx = nxt.transform.location.x - wp.transform.location.x
    dy = nxt.transform.location.y - wp.transform.location.y
    yaw = math.radians(wp.transform.rotation.yaw)
    assert dx * math.cos(yaw) + dy * math.sin(yaw) == pytest.approx(1.0, abs=0.05)


def test_map_waypoint_attributes(lg, vmap):
    cx, cy = frame.to_carla_xy(START_X, START_Y)
    wp = vmap.get_waypoint(Location(cx, cy, 0.0))
    assert wp.lane_width == pytest.approx(lg.width_at(wp.key, wp.s))
    assert wp.is_junction is False
    assert wp.road_id == 465 and wp.lane_id == -1


# ── (b) 우회전 연결로 — CARLA 관례로 yaw 가 증가해야 한다 ────────────────
def test_right_turn_connector_yaw_is_clockwise_positive(lg, vmap):
    """진입 차로 (62,0,-1) 끝 → 연결로 (664,0,-1) 통과. VTD Δhdg −93.5° 가
    CARLA 프레임에서 **+** (시계방향 양수) 로 나와야 한다."""
    entry_key = (62, 0, -1)
    assert (664, 0, -1) in lg.successors(entry_key), '그래프 전제가 깨졌다'

    L = lg.length(entry_key)
    x, y, _z, _h = lg.point_at(entry_key, L - 1.0)
    cx, cy = frame.to_carla_xy(x, y)
    wp = vmap.get_waypoint(Location(cx, cy, 0.0))
    assert wp.key == entry_key

    # 연결로 전 구간을 2 m 간격으로 훑는다 — 분기에서는 연결로(664) 가지를 고른다
    yaws = [wp.transform.rotation.yaw]
    for d in np.arange(2.0, lg.length((664, 0, -1)) + 2.0, 2.0):
        cands = wp.next(float(d))
        on = [w for w in cands if w.key[0] == 664] or \
             [w for w in cands if w.key in lg.successors((664, 0, -1))]
        assert on, f'd={d}: 연결로 가지가 없다 ({[w.key for w in cands]})'
        yaws.append(on[0].transform.rotation.yaw)

    turned = np.rad2deg(np.unwrap(np.deg2rad(yaws)))[-1] - yaws[0]
    assert turned > 45.0, f'우회전인데 CARLA yaw 변화가 {turned:.1f}° (양수여야 한다)'
    # 그리고 총량이 실측 Δhdg(−93.5° → CARLA +93.5°) 근처
    assert turned == pytest.approx(93.5, abs=15.0)


# ── (c) 객체 분류 ─────────────────────────────────────────────────────────
def _ego_state(x=START_X, y=START_Y, yaw=START_YAW, speed=5.0):
    from vtd_adapter.types import EgoState
    return EgoState(x=x, y=y, z=0.0, yaw=yaw, pitch=0.0, roll=0.0,
                    speed=speed, accel=0.0, lane=None, s=0.0, route_s=0.0,
                    t_off=0.0, heading_err=0.0)


def test_world_classifies_vehicle_and_walker():
    objs = [
        # (id, x, y, z, heading, speed, length, width, height)
        (7, START_X + 20.0, START_Y, 0.0, START_YAW, 8.0, 4.5, 1.8, 1.5),   # 차량
        (8, START_X + 15.0, START_Y + 3.0, 0.0, 0.0, 1.2, 0.5, 0.6, 1.7),   # 보행자
    ]
    pkt = parse(build_frame((START_X, START_Y, 0.0, START_YAW, 0.0, 0.0),
                            objs, [(0, 0)]), t_recv=100.0)
    world = VtdWorld(CFG)
    world.update(pkt, _ego_state())

    vehicles = world.get_actors().filter('*vehicle*')
    walkers = world.get_actors().filter('*walker*')
    assert [a.id for a in vehicles] == [7]
    assert [a.id for a in walkers] == [8]
    # 좌표·속도가 CARLA 프레임으로 나온다
    v = vehicles[0]
    # (struct '<f' float32 패킹이라 1e-3 허용)
    assert v.get_location().y == pytest.approx(-START_Y, abs=1e-3)
    assert v.get_velocity().length() == pytest.approx(8.0, abs=1e-3)
    assert v.get_transform().rotation.yaw == pytest.approx(-math.degrees(START_YAW), abs=1e-3)
    # 보행자 진행방향 (forecast_walkers 가 읽는 표면)
    w = walkers[0]
    d = w.get_control().direction
    assert (d.x, d.y) == (pytest.approx(1.0), pytest.approx(0.0))
    # bounding box extent = 반치수
    assert v.bounding_box.extent.x == pytest.approx(2.25)


def test_world_coasts_lost_object_then_drops():
    objs = [(7, START_X + 20.0, START_Y, 0.0, START_YAW, 8.0, 4.5, 1.8, 1.5)]
    world = VtdWorld(CFG)
    world.update(parse(build_frame((START_X, START_Y, 0.0, START_YAW, 0.0, 0.0),
                                   objs, [(0, 0)]), t_recv=100.0), _ego_state())
    x0 = world.get_actor(7).x
    # 다음 틱: 객체가 목록에서 빠짐 (80 m 근처 아님, 30개 미만) → coast
    world.update(parse(build_frame((START_X, START_Y, 0.0, START_YAW, 0.0, 0.0),
                                   [], [(0, 0)]), t_recv=100.1), _ego_state())
    a = world.get_actor(7)
    assert a is not None and a.coasting
    assert a.x - x0 == pytest.approx(8.0 * 0.1 * math.cos(-START_YAW), abs=0.02)
    # coast_s(1.5 s) 초과 → 소멸
    world.update(parse(build_frame((START_X, START_Y, 0.0, START_YAW, 0.0, 0.0),
                                   [], [(0, 0)]), t_recv=102.0), _ego_state())
    assert world.get_actor(7) is None


# ── (d) 자차 속도 추정 → VtdEgo ───────────────────────────────────────────
def test_ego_velocity_matches_estimator(lg):
    """40/80 ms 불규칙 틱 등속 14.7 m/s — test_speed_estimate 와 같은 시나리오.
    VtdEgo.get_velocity().length() 가 추정값(±1 %)과 일치해야 한다."""
    tracker = EgoTracker(lg, None, CFG)
    world = VtdWorld(CFG)
    v = 14.7
    x, y, t = START_X, START_Y, 100.0
    ws = None
    for dt in [0.04, 0.04, 0.08] * 20:
        t += dt
        x += v * dt * math.cos(START_YAW)
        y += v * dt * math.sin(START_YAW)
        ws = tracker.update(RawPacket(t_recv=t, ego=(x, y, 0.0, START_YAW, 0.0, 0.0),
                                      objects=[], lights=[]))
        world.update(RawPacket(t_recv=t, ego=(x, y, 0.0, START_YAW, 0.0, 0.0),
                               objects=[], lights=[]), ws.ego)
    assert ws.ego.speed == pytest.approx(v, rel=0.01)
    assert world.ego.get_velocity().length() == pytest.approx(ws.ego.speed, abs=1e-9)
    # 속도 벡터 방향도 CARLA 프레임의 자차 heading 과 일치
    vel = world.ego.get_velocity()
    ang = math.degrees(math.atan2(vel.y, vel.x))
    assert ang == pytest.approx(-math.degrees(START_YAW), abs=0.5)


# ── 종방향 컨트롤러 + 명령 변환 ───────────────────────────────────────────
def test_longitudinal_controller_follows_and_holds():
    lc = VtdLongitudinalController(CFG)
    # 큰 오차 → jerk 제한 때문에 첫 틱은 jerk_max*dt 만
    a1, brake = lc.get_throttle_and_brake(False, 10.0, 0.0)
    assert not brake and a1 == pytest.approx(CFG['speed']['jerk_max'] / CFG['comm']['send_hz'])
    # 수십 틱 뒤 a_max 포화
    for _ in range(50):
        a, _b = lc.get_throttle_and_brake(False, 10.0, 0.0)
    assert a == pytest.approx(CFG['speed']['a_max'])
    # 정지 유지
    a, brake = lc.get_throttle_and_brake(True, 0.0, 0.05)
    assert brake and a == pytest.approx(CFG['speed']['a_hold'])
    # forecast 용은 무상태 (실주행 jerk 이력을 오염시키지 않는다)
    prev = lc._prev_accel
    lc.get_throttle_extrapolation(10.0, 0.0)
    assert lc._prev_accel == prev


def test_command_from_control_steer_conversion():
    cmd = command_from_control(VehicleControl(steer=0.5, accel=-1.2),
                               max_steer_rad=CFG['vehicle']['max_steer'], turn_signal=2)
    assert cmd.steering == pytest.approx(-0.24)
    assert cmd.accel == pytest.approx(-1.2)
    assert cmd.turn_signal == 2
