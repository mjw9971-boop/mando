"""
dataclass <-> ROS 메시지 변환.

core/ 는 ROS 를 모른다(단위 테스트/standalone 실행에 그대로 쓰기 위해서).
노드가 메시지를 받으면 여기서 dataclass 로 바꿔 core 에 넣고, 결과를 다시
메시지로 바꿔 퍼블리시한다. **변환은 전부 여기 한 곳에만 있다.**

무한대(ttc) 와 None(lane) 은 메시지에 그대로 담을 수 없어 아래 규약을 쓴다.
  - inf      -> INF_SENTINEL (1e30) 로 보내고 되돌릴 때 inf 로 복원
  - lane None-> has_lane=False
  - ahead/summ/flags/reasons -> JSON 문자열 (스키마가 항목마다 달라 필드로 못 편다)
"""
from __future__ import annotations

import json
import math

from hlfma_msgs.msg import Cmd, Decision, GtState, Object, TrackedObject, TrafficLight, WorldState

from hlfma.core.types import Command
from hlfma.core.types import Decision as DecisionDC
from hlfma.core.types import EgoState, RawPacket
from hlfma.core.types import TrackedObject as TrackedObjectDC
from hlfma.core.types import WorldState as WorldStateDC

INF_SENTINEL = 1e30


def _f(v) -> float:
    """inf/nan 을 메시지에 실을 수 있는 값으로."""
    f = float(v)
    if math.isinf(f):
        return INF_SENTINEL if f > 0 else -INF_SENTINEL
    if math.isnan(f):
        return 0.0
    return f


def _unf(v) -> float:
    """_f 의 역변환."""
    f = float(v)
    if f >= INF_SENTINEL:
        return math.inf
    if f <= -INF_SENTINEL:
        return -math.inf
    return f


def _dumps(obj) -> str:
    return json.dumps(_jsonable(obj), ensure_ascii=False)


def _loads(s: str):
    return json.loads(s) if s else None


def _jsonable(d):
    # bool 은 int 의 하위형이라 반드시 먼저 걸러야 한다 (안 그러면 True -> 1.0).
    # int 도 float 로 바꾸지 않는다 — 신호 id 가 431.0 으로 남으면 대조가 번거롭다.
    if isinstance(d, bool) or isinstance(d, str) or d is None:
        return d
    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, int):
        return d
    if isinstance(d, float):
        return _f(d)
    if hasattr(d, 'kind') and hasattr(d, 'dist'):          # lanegraph.Ahead
        return {'kind': d.kind, 'dist': _f(d.dist),
                'lane': list(d.lane) if d.lane else None,
                's_in_lane': _f(d.s_in_lane), 'data': _jsonable(d.data)}
    return str(d)


# ══════════════════════════════════════════════════════════════════════════
# RawPacket  <->  GtState
# ══════════════════════════════════════════════════════════════════════════
def packet_to_msg(pkt: RawPacket, stamp) -> GtState:
    m = GtState()
    m.header.stamp = stamp
    m.header.frame_id = 'map'
    m.t_recv = float(pkt.t_recv)
    (m.ego_x, m.ego_y, m.ego_z, m.ego_heading, m.ego_pitch, m.ego_roll) = \
        [float(v) for v in pkt.ego]

    for o in pkt.objects:
        om = Object()
        (om.id, om.x, om.y, om.z, om.heading, om.speed,
         om.length, om.width, om.height) = (int(o[0]),) + tuple(float(v) for v in o[1:9])
        m.objects.append(om)

    for lid, state in pkt.lights:
        lm = TrafficLight()
        lm.id = int(lid)
        lm.state = int(state)
        m.lights.append(lm)
    return m


def msg_to_packet(m: GtState) -> RawPacket:
    return RawPacket(
        t_recv=float(m.t_recv),
        ego=(m.ego_x, m.ego_y, m.ego_z, m.ego_heading, m.ego_pitch, m.ego_roll),
        objects=[(o.id, o.x, o.y, o.z, o.heading, o.speed, o.length, o.width, o.height)
                 for o in m.objects],
        lights=[(int(l.id), int(l.state)) for l in m.lights],
    )


# ══════════════════════════════════════════════════════════════════════════
# WorldState  <->  WorldState.msg
# ══════════════════════════════════════════════════════════════════════════
def world_to_msg(ws: WorldStateDC, stamp) -> WorldState:
    m = WorldState()
    m.header.stamp = stamp
    m.header.frame_id = 'map'
    m.t = _f(ws.t)

    e = ws.ego
    m.ego_x, m.ego_y, m.ego_z = _f(e.x), _f(e.y), _f(e.z)
    m.ego_yaw, m.ego_pitch, m.ego_roll = _f(e.yaw), _f(e.pitch), _f(e.roll)
    m.ego_speed, m.ego_accel = _f(e.speed), _f(e.accel)
    m.has_lane = e.lane is not None
    m.lane = [int(v) for v in (e.lane or (0, 0, 0))]
    m.s, m.route_s = _f(e.s), _f(e.route_s)
    m.t_off, m.heading_err = _f(e.t_off), _f(e.heading_err)

    for o in ws.objects:
        om = TrackedObject()
        om.id = int(o.id)
        om.x, om.y = _f(o.x), _f(o.y)
        om.heading, om.speed = _f(o.heading), _f(o.speed)
        om.length, om.width, om.height = _f(o.length), _f(o.width), _f(o.height)
        om.cls = o.cls
        om.has_lane = o.lane is not None
        om.lane = [int(v) for v in (o.lane or (0, 0, 0))]
        om.on_route = bool(o.on_route)
        om.s_rel, om.lat_off, om.v_rel = _f(o.s_rel), _f(o.lat_off), _f(o.v_rel)
        om.ttc = _f(o.ttc)
        om.will_enter_lane = bool(o.will_enter_lane)
        om.age = _f(o.age)
        om.coasting = bool(o.coasting)
        m.objects.append(om)

    m.has_light = ws.light is not None
    if ws.light is not None:
        m.light.id, m.light.state = int(ws.light[0]), int(ws.light[1])

    m.speed_limit = _f(ws.speed_limit)
    m.school_zone = bool(ws.school_zone)
    m.left_solid, m.right_solid = bool(ws.left_solid), bool(ws.right_solid)
    m.left_is_center = bool(ws.left_is_center)
    m.valid = bool(ws.valid)

    m.ahead_json = _dumps(ws.ahead)
    m.summ_json = _dumps(ws.summ)
    m.flags_json = _dumps(ws.flags)
    return m


class _Ahead:
    """ahead_json 을 되돌린 항목. lanegraph.Ahead 와 같은 속성을 노출한다."""

    __slots__ = ('dist', 'kind', 'lane', 's_in_lane', 'data')

    def __init__(self, d: dict) -> None:
        self.dist = _unf(d.get('dist', 0.0))
        self.kind = d.get('kind', '')
        self.lane = tuple(d['lane']) if d.get('lane') else None
        self.s_in_lane = _unf(d.get('s_in_lane', 0.0))
        self.data = d.get('data') or {}

    def __repr__(self) -> str:
        return f'Ahead({self.kind}, {self.dist:.1f}m)'


def msg_to_world(m: WorldState) -> WorldStateDC:
    ego = EgoState(
        x=m.ego_x, y=m.ego_y, z=m.ego_z, yaw=m.ego_yaw, pitch=m.ego_pitch, roll=m.ego_roll,
        speed=m.ego_speed, accel=m.ego_accel,
        lane=tuple(int(v) for v in m.lane) if m.has_lane else None,
        s=m.s, route_s=m.route_s, t_off=m.t_off, heading_err=m.heading_err,
    )
    objects = [
        TrackedObjectDC(
            id=int(o.id), x=o.x, y=o.y, heading=o.heading, speed=o.speed,
            length=o.length, width=o.width, height=o.height, cls=o.cls,
            lane=tuple(int(v) for v in o.lane) if o.has_lane else None,
            on_route=o.on_route, s_rel=o.s_rel, lat_off=o.lat_off,
            v_rel=o.v_rel, ttc=_unf(o.ttc), will_enter_lane=o.will_enter_lane,
            age=o.age, coasting=o.coasting,
        )
        for o in m.objects
    ]
    return WorldStateDC(
        t=m.t, ego=ego, objects=objects,
        light=(int(m.light.id), int(m.light.state)) if m.has_light else None,
        ahead=[_Ahead(d) for d in (_loads(m.ahead_json) or [])],
        summ=_loads(m.summ_json) or {},
        speed_limit=m.speed_limit, school_zone=m.school_zone,
        left_solid=m.left_solid, right_solid=m.right_solid,
        left_is_center=m.left_is_center, valid=m.valid,
        flags=_loads(m.flags_json) or {},
    )


# ══════════════════════════════════════════════════════════════════════════
# Decision / Command
# ══════════════════════════════════════════════════════════════════════════
def decision_to_msg(d: DecisionDC, stamp) -> Decision:
    m = Decision()
    m.header.stamp = stamp
    m.header.frame_id = 'map'
    m.v_target = _f(d.v_target)
    m.path_x = [float(p[0]) for p in d.path]
    m.path_y = [float(p[1]) for p in d.path]
    m.turn_signal = int(d.turn_signal)
    m.state = d.state
    m.reasons_json = _dumps(d.reasons)
    return m


def msg_to_decision(m: Decision) -> DecisionDC:
    return DecisionDC(
        v_target=m.v_target,
        path=list(zip([float(v) for v in m.path_x], [float(v) for v in m.path_y])),
        turn_signal=int(m.turn_signal),
        state=m.state,
        reasons=_loads(m.reasons_json) or {},
    )


def command_to_msg(c: Command, stamp) -> Cmd:
    m = Cmd()
    m.header.stamp = stamp
    m.header.frame_id = 'base_link'
    m.steering = _f(c.steering)
    m.accel = _f(c.accel)
    m.turn_signal = int(c.turn_signal)
    return m


def msg_to_command(m: Cmd) -> Command:
    return Command(steering=m.steering, accel=m.accel, turn_signal=int(m.turn_signal))
