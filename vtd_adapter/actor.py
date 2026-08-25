"""
carla.Actor 흉내 — 9910 객체와 자차를 CARLA 관례로 감싼다.

VtdActor 의 좌표·yaw 는 전부 CARLA 프레임이다 (world.py 가 frame.py 로 변환해서
넣는다). PDM-Lite 가 쓰는 표면: id / type_id / attributes / bounding_box /
get_location / get_transform / get_velocity / get_control.

9910 에는 타차의 조작량(steer/throttle/brake)이 없다 → get_control() 은 (0,0,0).
kinematic 모델 forecast 가 등속·직진 외삽이 된다 (phase0 §0-1 — 허용 근사).
"""
from __future__ import annotations

import math

from .carla_types import (BoundingBox, Location, Rotation, Transform, Vector3D,
                          VehicleControl, WalkerControl)


class VtdActor:
    """9910 객체 1개. world.py 가 매 틱 update() 로 갱신한다."""

    def __init__(self, oid: int, cls: str) -> None:
        self.id = int(oid)
        self.cls = cls                                   # 'vehicle' | 'pedestrian' | ...
        self.type_id = ('walker.vtd.pedestrian' if cls == 'pedestrian'
                        else 'vehicle.vtd.object')
        self.attributes: dict = {}                       # base_type 없음 → bicycle 분기 비활성
        self.is_alive = True
        # CARLA 프레임 상태 (world.update 가 채운다)
        self.x = self.y = self.z = 0.0
        self.yaw_deg = 0.0
        self.speed = 0.0
        self.length = self.width = self.height = 0.0
        self.age = 0.0
        self.coasting = False

    def update(self, x: float, y: float, z: float, yaw_deg: float, speed: float,
               length: float, width: float, height: float) -> None:
        self.x, self.y, self.z = float(x), float(y), float(z)
        self.yaw_deg = float(yaw_deg)
        self.speed = float(speed)
        self.length, self.width, self.height = float(length), float(width), float(height)

    # ── CARLA 표면 ────────────────────────────────────────────────────────
    @property
    def bounding_box(self) -> BoundingBox:
        # location(0,0,0) = 액터 기준점이 곧 상자 중심 — transform.transform(bb.location)
        # 이 항등이 되어 autopilot 의 전역 상자 계산이 그대로 성립한다.
        return BoundingBox(Location(0.0, 0.0, 0.0),
                           Vector3D(self.length / 2.0, self.width / 2.0, self.height / 2.0))

    def get_location(self) -> Location:
        return Location(self.x, self.y, self.z)

    def get_transform(self) -> Transform:
        return Transform(Location(self.x, self.y, self.z), Rotation(yaw=self.yaw_deg))

    def get_velocity(self) -> Vector3D:
        yaw = math.radians(self.yaw_deg)
        return Vector3D(self.speed * math.cos(yaw), self.speed * math.sin(yaw), 0.0)

    def get_control(self):
        if self.cls == 'pedestrian':
            yaw = math.radians(self.yaw_deg)
            return WalkerControl(Vector3D(math.cos(yaw), math.sin(yaw), 0.0), self.speed)
        return VehicleControl()                           # 조작량 미상 → (0,0,0)

    def __repr__(self) -> str:
        return f'VtdActor(id={self.id}, {self.type_id}, v={self.speed:.1f})'


class VtdEgo(VtdActor):
    """자차. 속도는 EgoSpeedEstimator(EgoTracker) 추정값으로 채운다 (9910 에 없다).

    id=0 — 9910 객체 id 는 양수라(0 = 빈 슬롯) 충돌하지 않는다.
    """

    def __init__(self, vehicle_cfg: dict) -> None:
        super().__init__(0, 'vehicle')
        self.type_id = 'vehicle.hyundai.ioniq6'
        self.length = float(vehicle_cfg['length'])
        self.width = float(vehicle_cfg['width'])
        self.height = float(vehicle_cfg['height'])

    def update_from_ego(self, ego_state, yaw_deg: float, x: float, y: float) -> None:
        """EgoTracker 의 EgoState + CARLA 변환값으로 갱신 (world.py 가 호출)."""
        self.x, self.y, self.z = float(x), float(y), float(ego_state.z)
        self.yaw_deg = float(yaw_deg)
        self.speed = max(0.0, float(ego_state.speed))
