"""
CARLA 기하·제어 타입 흉내 — PDM-Lite(team_code)가 쓰는 최소 표면만.

autopilot.py 의 OBB(SAT)·forecast·IDM 코드가 이 타입들 위에서 **원문 그대로**
돌아야 한다. 값은 전부 CARLA 관례(frame.py 로 변환된 좌표·deg yaw)다.

주의: Rotation 벡터는 yaw 만 반영한다 (pitch/roll=0 전제 — autopilot 이 OBB 를
만들 때 carla.Rotation(pitch=0, yaw=…, roll=0) 로만 만들고, SAT 는 상자가
y 대칭이라 right 벡터 부호에도 결과가 불변이다: phase0 §0-3).
"""
from __future__ import annotations

import math
from enum import IntEnum


class Vector3D:
    __slots__ = ('x', 'y', 'z')

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = float(x), float(y), float(z)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def dot(self, o: 'Vector3D') -> float:
        return self.x * o.x + self.y * o.y + self.z * o.z

    def __add__(self, o):
        return type(self)(self.x + o.x, self.y + o.y, self.z + o.z)

    def __sub__(self, o):
        # Location - Location 도 Vector3D 가 되는 CARLA 규약
        return Vector3D(self.x - o.x, self.y - o.y, self.z - o.z)

    def __mul__(self, k: float):
        return type(self)(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    def __repr__(self) -> str:
        return f'{type(self).__name__}({self.x:.3f}, {self.y:.3f}, {self.z:.3f})'


class Location(Vector3D):
    def distance(self, o: 'Location') -> float:
        return math.sqrt((self.x - o.x) ** 2 + (self.y - o.y) ** 2 + (self.z - o.z) ** 2)


class Rotation:
    __slots__ = ('pitch', 'yaw', 'roll')

    def __init__(self, pitch: float = 0.0, yaw: float = 0.0, roll: float = 0.0) -> None:
        self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)

    def get_forward_vector(self) -> Vector3D:
        cp = math.cos(math.radians(self.pitch))
        return Vector3D(cp * math.cos(math.radians(self.yaw)),
                        cp * math.sin(math.radians(self.yaw)),
                        math.sin(math.radians(self.pitch)))

    def get_right_vector(self) -> Vector3D:
        # yaw + 90° 방향 (pitch/roll 무시 — 위 모듈 docstring)
        return Vector3D(-math.sin(math.radians(self.yaw)),
                        math.cos(math.radians(self.yaw)), 0.0)

    def get_up_vector(self) -> Vector3D:
        return Vector3D(0.0, 0.0, 1.0)

    def __repr__(self) -> str:
        return f'Rotation(pitch={self.pitch:.2f}, yaw={self.yaw:.2f}, roll={self.roll:.2f})'


class Transform:
    __slots__ = ('location', 'rotation')

    def __init__(self, location: Location | None = None,
                 rotation: Rotation | None = None) -> None:
        self.location = location if location is not None else Location()
        self.rotation = rotation if rotation is not None else Rotation()

    def transform(self, point: Vector3D) -> Location:
        """로컬 점 → 월드 (yaw 회전 + 평행이동)."""
        c = math.cos(math.radians(self.rotation.yaw))
        s = math.sin(math.radians(self.rotation.yaw))
        return Location(self.location.x + c * point.x - s * point.y,
                        self.location.y + s * point.x + c * point.y,
                        self.location.z + point.z)

    def get_forward_vector(self) -> Vector3D:
        return self.rotation.get_forward_vector()

    def __repr__(self) -> str:
        return f'Transform({self.location!r}, {self.rotation!r})'


class BoundingBox:
    __slots__ = ('location', 'extent', 'rotation')

    def __init__(self, location: Location, extent: Vector3D) -> None:
        self.location = location            # OBB 중심 (월드 or 액터 로컬 — CARLA 와 동일 관례)
        self.extent = extent                # 반치수 (length/2, width/2, height/2)
        self.rotation = Rotation()

    def __repr__(self) -> str:
        return f'BoundingBox({self.location!r}, extent={self.extent!r})'


class TrafficLightState(IntEnum):
    Red = 0
    Yellow = 1
    Green = 2
    Off = 3
    Unknown = 4


class RoadOption(IntEnum):
    """carla agents.navigation.local_planner.RoadOption 과 같은 값."""
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANEFOLLOW = 4
    CHANGELANELEFT = 5
    CHANGELANERIGHT = 6


class VehicleControl:
    """carla.VehicleControl + VTD 확장 필드 accel [m/s²] (phase0 §0-4 (b))."""
    __slots__ = ('steer', 'throttle', 'brake', 'accel')

    def __init__(self, steer: float = 0.0, throttle: float = 0.0,
                 brake: float = 0.0, accel: float = 0.0) -> None:
        self.steer, self.throttle, self.brake = float(steer), float(throttle), float(brake)
        self.accel = float(accel)


class WalkerControl:
    """보행자 진행 방향 — forecast_walkers 가 get_control().direction 을 읽는다."""
    __slots__ = ('direction', 'speed')

    def __init__(self, direction: Vector3D, speed: float = 0.0) -> None:
        self.direction = direction
        self.speed = float(speed)
