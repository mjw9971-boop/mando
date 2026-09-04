"""
좌표계 변환의 **유일한** 자리 (phase0 §0-3 결정).

  VTD/OpenDRIVE : 우수계(z-up, y 좌측), heading rad 반시계(+ = 좌회전)
  CARLA         : 좌수계(y 가 ODR 의 −y), rotation.yaw 도(deg), 수치상 시계방향 증가

변환 규칙 (전부 자기 역함수 성질 — 두 번 적용하면 원상복귀):

  위치   x_c = x_v,   y_c = −y_v,   z_c = z_v
  각도   yaw_c[rad] = −heading_v[rad]      (deg 는 to_carla_yaw_deg)
  조향   steering_vtd[rad] = −steer_carla × vehicle.max_steer

map.py/actor.py/world.py/route.py 는 **밖으로 내주는 모든 좌표·각도를 CARLA
관례로** 통일하고, 반드시 이 모듈을 거친다. 어댑터 안에서 변환을 두 번 겹치면
부호가 되돌아가므로, 여기 함수 외의 곳에서 y 나 yaw 의 부호를 만지지 말 것.

lanegraph/comm/logger 는 VTD 원 좌표를 쓴다 (로그 스키마 유지).
"""
from __future__ import annotations

import math

import numpy as np


# ── 위치 ──────────────────────────────────────────────────────────────────
def to_carla_xy(x: float, y: float) -> tuple[float, float]:
    return float(x), float(-y)


def from_carla_xy(x: float, y: float) -> tuple[float, float]:
    return float(x), float(-y)          # 미러라 역변환이 같은 식이다


def to_carla_np(pts: np.ndarray) -> np.ndarray:
    """(N,2|3) VTD 점열 → CARLA. 복사본을 돌려준다."""
    out = np.array(pts, dtype=float, copy=True)
    out[:, 1] = -out[:, 1]
    return out


# ── 각도 ──────────────────────────────────────────────────────────────────
def to_carla_yaw_rad(heading_v: float) -> float:
    return -float(heading_v)


def to_carla_yaw_deg(heading_v: float) -> float:
    return -math.degrees(float(heading_v))


def from_carla_yaw_rad(yaw_c: float) -> float:
    return -float(yaw_c)


def from_carla_yaw_deg(yaw_c_deg: float) -> float:
    return -math.radians(float(yaw_c_deg))


# ── 조향 ──────────────────────────────────────────────────────────────────
def steer_to_vtd(steer_carla: float, max_steer_rad: float) -> float:
    """CARLA 정규화 조향 [-1,1] → 9910 조향 [rad, 좌 +].

    미러 프레임에서 계산된 조향은 실세계에서 반대 방향이므로 부호를 되돌린다.
    comm.steer_sign(+1.0) 은 pack_command 에서 따로 곱한다 — 여기서 겹치지 않는다.
    """
    return -float(steer_carla) * float(max_steer_rad)
