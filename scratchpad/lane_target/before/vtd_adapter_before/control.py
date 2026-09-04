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
    """target_speed [m/s] → accel [m/s²]. PDM LongitudinalController 시그니처 호환.

    가속: 기존 검증 P (kp 0.8) + a_max 클램프.
    감속: **IDM 의 의도를 그대로 실행한다** — PDM 의 target_speed 는 IDM 이 낸
    "1틱(t_bound 0.05 s) 전방 속도"라, P 로만 따르면 의도 감속이 kp·dt(1/25)로
    희석돼 정지선을 시스템적으로 지나친다 (폐루프 시뮬: 11.4 m 오버런).
    (target − v)/dt 가 곧 IDM 의 의도 가감속이고, 이를 control.a_dec_max 로
    클램프해 실행한다 (계단형 목표 하향 — 제한속도 전환 — 에서의 폭주 방지).
    a_min 은 hazard/비상 축으로 남는다.
    """

    def __init__(self, cfg: dict) -> None:
        s, c = cfg['speed'], cfg['control']
        self.kp = float(c['kp'])
        self.a_min = float(s['a_min'])
        self.a_max = float(s['a_max'])
        self.a_dec_max = float(c['a_dec_max'])
        self.jerk_dec_mult = float(c['jerk_dec_mult'])
        self.jerk_max = float(s['jerk_max'])
        self.a_hold = float(s['a_hold'])
        self.a_emergency = float(s['a_emergency'])
        # B-2 좁은 패치 (params 참조). release_eps 0 이면 비활성.
        self.release_eps = float(c.get('release_eps', 0.0))
        self.release_v = float(c.get('release_v', 0.5))
        self.dt = 1.0 / float(cfg['comm']['send_hz'])
        self._prev_accel = 0.0
        self._undo_accel = 0.0
        self._saved_accel = 0.0

    def _raw_accel(self, target: float, v: float) -> float:
        err = target - v
        if err < 0.0:
            # IDM 의도 감속 (err/dt), 상한 a_dec_max — a_min 은 hazard 전용
            return max(err / self.dt, self.a_dec_max)
        return min(self.kp * err, self.a_max)

    def get_throttle_and_brake(self, hazard_brake: bool, target_speed: float,
                               current_speed: float) -> tuple[float, bool]:
        """(accel [m/s²], brake_bool). brake_bool 은 로그·rolling-back 방지 판정용.

        · hazard_brake 또는 목표 0 인데 이미 저속 → a_hold (정지 유지 — 굴러가지 않게)
        · 그 외 → _raw_accel (모듈 docstring) + jerk 제한 (감속 방향은 jerk_dec_mult 배 완화)
        """
        self._undo_accel = self._prev_accel        # rewind_last() 용 (kr_rules 재호출)
        if (hazard_brake or target_speed < 1e-5) and current_speed < 0.2:
            self._prev_accel = self.a_hold
            return self.a_hold, True

        # 비상 제동(a_emergency)에서 빠져나오는 틱 — jerk 창의 기준을 정상 축
        # (a_dec_max)까지 끌어올린다. 안 그러면 −8.0 → 0 복귀에 4 s 가 걸린다.
        # 비상이 계속 필요하면 kr_rules 가 emergency() 로 다시 덮는다.
        if self._prev_accel < self.a_dec_max:
            self._prev_accel = self.a_dec_max

        # ── B-2 좁은 패치 ────────────────────────────────────────────────
        # 정지 후보가 하나도 없고(hazard=False, 목표 > release_v) 자차가 사실상
        # 서 있는데 제동 명령이 남아 있으면, 램프를 태우지 않고 즉시 턴다.
        # 실측 2026-09-01 t=414.1~417.1: winner=none·v_target=8.33 인데 직전
        # 급제동의 잔량(−4.0)이 +0.1/틱으로만 풀려 3.00 s 를 더 서 있었다.
        # 목표 0(홀드·④′ 종단)은 위 a_hold 분기가 먼저 반환하므로 여기 안 온다.
        if (self.release_eps > 0.0 and not hazard_brake
                and float(target_speed) > self.release_v
                and float(current_speed) < self.release_eps
                and self._prev_accel < 0.0):
            self._prev_accel = 0.0

        target = 0.0 if hazard_brake else float(target_speed)
        accel = self._raw_accel(target, float(current_speed))

        accel = _clamp(accel,
                       self._prev_accel - self.jerk_dec_mult * self.jerk_max * self.dt,
                       self._prev_accel + self.jerk_max * self.dt)
        self._prev_accel = accel
        return accel, bool(hazard_brake or target_speed < 1e-5)

    def emergency(self, accel: float | None = None) -> tuple[float, bool]:
        """jerk 램프를 건너뛰고 즉시 accel 을 낸다 — **보행자 비상 한정**.

        kr_rules 가 보행자 의도 후보의 필요 감속이 |a_dec_max|·ped_emergency_ratio
        를 넘을 때만 부른다. 실측 2026-09-01: 후보 생성 뒤 −4.0 에 도달하기까지
        1.00 s 가 걸려 그 사이 12.1 m 를 갔고, 그 1 초가 정지/접촉을 갈랐다.

        `_prev_accel` 을 그대로 덮으므로 다음 틱의 jerk 기준도 이 값이다. 비상이
        풀리면 get_throttle_and_brake 가 기준을 a_dec_max 로 끌어올려 복귀한다.
        """
        a = self.a_emergency if accel is None else float(accel)
        self._undo_accel = self._prev_accel
        self._prev_accel = a
        return a, True

    def rewind_last(self) -> None:
        """직전 get_throttle_and_brake 호출을 무효화한다 (jerk 기준 복원).

        kr_rules 가 같은 틱에 더 낮은 목표로 재계산할 때 쓴다 — 되감지 않으면
        본류 호출(높은 목표, accel↑)과 재호출(낮은 목표, accel↓)이 jerk 창을
        나눠 갖는 핑퐁이 되어 순감속이 0 에 수렴한다 (2026-08-26 mock 실측:
        route_end 발동에도 v 6.9→6.65 밖에 못 줄여 종점 통과).
        """
        self._prev_accel = self._undo_accel

    def get_throttle_extrapolation(self, target_speed: float,
                                   current_speed: float) -> float:
        """forecast 용 accel — **무상태**다. forecast 가 여러 스텝을 굴려도
        실주행의 jerk 이력(_prev_accel)을 오염시키면 안 된다."""
        target = float(target_speed)
        if target < 1e-5 and current_speed < 0.2:
            return self.a_hold
        return self._raw_accel(target, float(current_speed))

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
