"""dataclass <-> 메시지 왕복 검증. 변환에서 값이 새면 여기서 잡힌다."""
import math

import pytest

pytest.importorskip('hlfma_msgs', reason='colcon build + source install 필요')

from hlfma.convert import (command_to_msg, decision_to_msg, msg_to_command,  # noqa: E402
                           msg_to_decision, msg_to_packet, msg_to_world,
                           packet_to_msg, world_to_msg)
from hlfma.core.types import (Command, Decision, EgoState, RawPacket,  # noqa: E402
                              TrackedObject, WorldState)

STAMP = None  # header.stamp 은 기본값으로 둔다


def test_packet_roundtrip():
    pkt = RawPacket(
        t_recv=1234.5,
        ego=(508.8, -168.29, 42.0, 0.5272, 0.01, -0.02),
        objects=[(7, 100.0, 20.0, 0.0, 0.1, 5.0, 4.5, 1.9, 1.5),
                 (12, 30.0, -3.0, 0.0, 1.5, 1.2, 0.6, 0.6, 1.7)],
        lights=[(3, 1)],
    )
    from builtin_interfaces.msg import Time
    back = msg_to_packet(packet_to_msg(pkt, Time()))

    assert back.t_recv == pytest.approx(pkt.t_recv)
    assert back.ego == pytest.approx(pkt.ego, rel=1e-5)
    assert [o[0] for o in back.objects] == [7, 12]
    assert back.lights == [(3, 1)]


def _world(**kw):
    ego = EgoState(x=1.0, y=2.0, z=3.0, yaw=0.4, pitch=0.0, roll=0.0,
                   speed=5.0, accel=0.5, lane=(465, 2, -1), s=10.0,
                   route_s=20.0, t_off=-0.05, heading_err=0.01)
    base = dict(t=99.0, ego=ego, objects=[], light=(3, 2), ahead=[], summ={},
                speed_limit=8.33, school_zone=True, left_solid=True,
                right_solid=False, left_is_center=True, valid=True, flags={})
    base.update(kw)
    return WorldState(**base)


def test_world_roundtrip_scalars():
    from builtin_interfaces.msg import Time
    ws = _world()
    back = msg_to_world(world_to_msg(ws, Time()))

    assert back.ego.lane == (465, 2, -1)
    assert back.ego.speed == pytest.approx(5.0)
    assert back.ego.t_off == pytest.approx(-0.05)
    assert back.light == (3, 2)
    assert back.school_zone is True
    assert back.left_is_center is True
    assert back.right_solid is False
    assert back.valid is True


def test_world_lane_none_survives():
    from builtin_interfaces.msg import Time
    ws = _world()
    ws.ego.lane = None
    assert msg_to_world(world_to_msg(ws, Time())).ego.lane is None


def test_tracked_object_inf_ttc_roundtrip():
    """ttc=inf 는 메시지에 그대로 못 담는다 → sentinel 로 보내고 inf 로 복원."""
    from builtin_interfaces.msg import Time
    obj = TrackedObject(
        id=7, x=1.0, y=2.0, heading=0.1, speed=3.0, length=4.5, width=1.9, height=1.5,
        cls='vehicle', lane=None, on_route=True, s_rel=30.0, lat_off=0.2,
        v_rel=1.5, ttc=math.inf, will_enter_lane=False, age=0.0, coasting=False)
    back = msg_to_world(world_to_msg(_world(objects=[obj]), Time()))
    o = back.objects[0]
    assert math.isinf(o.ttc)
    assert o.cls == 'vehicle'
    assert o.on_route is True
    assert o.lane is None


def test_world_ahead_and_summ_survive_json():
    """ahead 의 data dict 는 항목마다 스키마가 달라 JSON 으로 싣는다."""
    from builtin_interfaces.msg import Time

    class FakeAhead:
        kind = 'stop_line'
        dist = 42.5
        lane = (465, 2, -1)
        s_in_lane = 12.0
        data = {'signal_ids': [30, 31]}

    ws = _world(ahead=[FakeAhead()], summ={'dist_stop_line': 42.5, 'stop_signal_ids': [30, 31]})
    back = msg_to_world(world_to_msg(ws, Time()))

    assert len(back.ahead) == 1
    a = back.ahead[0]
    assert a.kind == 'stop_line'
    assert a.dist == pytest.approx(42.5)
    assert a.lane == (465, 2, -1)
    assert a.data['signal_ids'] == [30, 31]
    assert back.summ['dist_stop_line'] == pytest.approx(42.5)


def test_decision_roundtrip():
    from builtin_interfaces.msg import Time
    d = Decision(v_target=5.56, path=[(1.0, 2.0), (3.0, 4.0)], turn_signal=1,
                 state='FOLLOW', reasons={'const': 5.56, 'shield': {}})
    back = msg_to_decision(decision_to_msg(d, Time()))
    assert back.v_target == pytest.approx(5.56)
    assert back.path == [(1.0, 2.0), (3.0, 4.0)]
    assert back.turn_signal == 1
    assert back.state == 'FOLLOW'
    assert back.reasons['const'] == pytest.approx(5.56)


def test_command_roundtrip():
    from builtin_interfaces.msg import Time
    c = Command(steering=-0.12, accel=1.5, turn_signal=2)
    back = msg_to_command(command_to_msg(c, Time()))
    assert back.steering == pytest.approx(-0.12)
    assert back.accel == pytest.approx(1.5)
    assert back.turn_signal == 2
