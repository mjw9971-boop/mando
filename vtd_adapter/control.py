"""
종방향 제어 + 명령 변환 (phase0 §0-4 (b) 결정).

9910 은 targetAccel[m/s²] 를 직접 받으므로 CARLA 의 throttle/brake 회귀
컨트롤러 대신 **accel 을 직접 내는** 컨트롤러를 쓴다. 내부는 이 VTD·이 차량에서
검증된 P 제어다: kp 0.8, a_min/a_max 클램프, jerk_max 변화율 제한, 정지 유지
a_hold. (IDM 이 이미 감속 프로파일을 반영한 target_speed 를 주므로 컨트롤러는
추종만 하면 된다.)

인터페이스는 PDM-Lite LongitudinalController 와 호환: get_throttle_and_brake /
get_throttle_extrapolation / save / load. 반환의 첫 값이 throttle 이 아니라
**accel [m/s²]** 인 점만 다르다 — autopilot 접합부(phase3)가 `# VTD:` 주석과
함께 이 값을 VehicleControl.accel 로 넘긴다.
"""
from __future__ import annotations

from . import frame
from .carla_types import VehicleControl
from .types import Command


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class VtdLongitudinalController:
    """target_speed [m/s] → accel [m/s²]. PDM LongitudinalController 시그니처 호환."""

    def __init__(self, cfg: dict) -> None:
        s, c = cfg['speed'], cfg['control']
        self.kp = float(c['kp'])
        self.a_min = float(s['a_min'])
        self.a_max = float(s['a_max'])
        self.jerk_max = float(s['jerk_max'])
        self.a_hold = float(s['a_hold'])
        self.dt = 1.0 / float(cfg['comm']['send_hz'])
        self._prev_accel = 0.0
        self._saved_accel = 0.0

    def get_throttle_and_brake(self, hazard_brake: bool, target_speed: float,
                               current_speed: float) -> tuple[float, bool]:
        """(accel [m/s²], brake_bool). brake_bool 은 로그·rolling-back 방지 판정용.

        · hazard_brake 또는 목표 0 인데 이미 저속 → a_hold (정지 유지 — 굴러가지 않게)
        · 그 외 → P 추종 + a_min/a_max 클램프 + jerk 제한
        """
        if (hazard_brake or target_speed < 1e-5) and current_speed < 0.2:
            self._prev_accel = self.a_hold
            return self.a_hold, True

        target = 0.0 if hazard_brake else float(target_speed)
        accel = _clamp(self.kp * (target - float(current_speed)), self.a_min, self.a_max)

        max_dj = self.jerk_max * self.dt
        accel = _clamp(accel, self._prev_accel - max_dj, self._prev_accel + max_dj)
        self._prev_accel = accel
        return accel, bool(hazard_brake or target_speed < 1e-5)

    def get_throttle_extrapolation(self, target_speed: float,
                                   current_speed: float) -> float:
        """forecast 용 accel — **무상태**다. forecast 가 여러 스텝을 굴려도
        실주행의 jerk 이력(_prev_accel)을 오염시키면 안 된다."""
        target = float(target_speed)
        if target < 1e-5 and current_speed < 0.2:
            return self.a_hold
        return _clamp(self.kp * (target - float(current_speed)), self.a_min, self.a_max)

    def save(self) -> None:
        self._saved_accel = self._prev_accel

    def load(self) -> None:
        self._prev_accel = self._saved_accel


def command_from_control(control: VehicleControl, max_steer_rad: float,
                         turn_signal: int = 0) -> Command:
    """VehicleControl(CARLA 관례) → 9910 Command.

    조향은 frame.steer_to_vtd 한 곳에서만 변환한다 (부호 이중 적용 금지 —
    comm.pack_command 의 steer_sign 은 그대로 +1.0).
    accel 은 VtdLongitudinalController 가 이미 m/s² 로 채웠다.
    """
    return Command(
        steering=frame.steer_to_vtd(control.steer, max_steer_rad),
        accel=float(control.accel),
        turn_signal=int(turn_signal),
    )
