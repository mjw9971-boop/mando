"""
파이프라인 각 단계가 주고받는 dataclass 정의.

SPEC §2: 각 단계는 앞 단계 출력만 받는다. 역방향 참조 금지.
  RawPacket → EgoState/TrackedObject → WorldState → Decision → Command

단위는 전부 m, m/s, m/s^2, rad. km/h 는 config/로그에서만 쓴다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# lanegraph 의 lane key 와 동일: (road_id, section_idx, lane_id)
LaneKey = tuple[int, int, int]


@dataclass
class RawPacket:
    """9910 에서 받은 1109 B 한 프레임을 그대로 푼 것. 해석/판단 없음."""
    t_recv: float                          # time.monotonic()
    ego: tuple[float, ...]                 # x, y, z, heading, pitch, roll
    objects: list[tuple]                   # id != 0 인 것만
    #                                        (id, x, y, z, heading, speed, length, width, height)
    lights: list[tuple[int, int]]          # (id, state), id == 0 이면 제외
    # 접속 후 지금까지 소켓에 도착한 9910 프레임 수 (버린 백로그 포함).
    # 프레임 하나 = 시뮬레이션 1/send_hz(50 ms) 이므로 이 카운터의 차분이
    # 시뮬레이션 경과시간이다 — 프레임에 sim 타임스탬프 필드가 없어서(§1.1)
    # RTF(sim/wall) 판별은 이 값으로만 가능하다. 0 = 카운터 없음(리플레이 등).
    frames_total: int = 0


@dataclass
class EgoState:
    """자차 상태. speed/accel 은 위치 미분으로 추정한 값(패킷에 없다)."""
    x: float
    y: float
    z: float
    yaw: float
    pitch: float
    roll: float
    speed: float                           # 위치 미분 + LPF
    accel: float                           # speed 미분
    lane: LaneKey | None
    s: float                               # 차로 내 s
    route_s: float                         # 경로 누적거리 = cum_s[idx] + s
    t_off: float                           # 중심선 횡오프셋 (좌 +)
    heading_err: float


@dataclass
class TrackedObject:
    """주변 객체 1개. 타입 필드가 없어 크기로 분류한다(SPEC §1.1)."""
    id: int
    x: float
    y: float
    heading: float
    speed: float
    length: float
    width: float
    height: float
    cls: str                               # 'vehicle' | 'pedestrian' | 'obstacle' | 'unknown'
    lane: LaneKey | None
    on_route: bool                         # 경로 차로 위인지
    s_rel: float                           # 경로 기준 종방향 상대거리 (+ = 앞)
    lat_off: float                         # 내 경로 중심선 기준 횡거리
    v_rel: float                           # 접근 속도 (+ = 접근)
    ttc: float                             # inf 가능
    will_enter_lane: bool                  # 1~2 s 등속 외삽 시 내 차로 진입
    age: float                             # 마지막 수신 후 경과 (coasting 용)
    coasting: bool                         # 이번 틱 수신 없어 외삽 중


@dataclass
class WorldState:
    """Perception 출력. 여기까지는 '무엇이 있다'만 말하고 판단하지 않는다."""
    t: float
    ego: EgoState
    objects: list[TrackedObject]
    light: tuple[int, int] | None          # (id, state)
    ahead: list                            # lanegraph.Ahead 리스트
    summ: dict                             # lanegraph.summarize 결과
    speed_limit: float                     # 현재 유효 제한속도 [m/s] (carry 반영)
    school_zone: bool
    left_solid: bool
    right_solid: bool
    left_is_center: bool
    valid: bool                            # 패킷 신선도/좌표 점프 없음
    flags: dict = field(default_factory=dict)


@dataclass
class Decision:
    """Planner 출력. Shield 가 이걸 깎는다."""
    v_target: float                        # [m/s]
    path: list[tuple[float, float]]        # 추종 목표 점열 (월드 좌표)
    turn_signal: int                       # 0=OFF 1=LEFT 2=RIGHT
    state: str                             # FSM 상태명
    reasons: dict = field(default_factory=dict)   # 각 속도 후보값 (로그/디버깅용)


@dataclass
class Command:
    """9910 으로 나가는 최종 제어값. Comm.send 가 그대로 packing 한다."""
    steering: float                        # [rad] 좌 +  (steer_sign 은 Comm 에서 적용)
    accel: float                           # [m/s^2] 음수 = 감속
    turn_signal: int                       # 0=OFF 1=LEFT 2=RIGHT
