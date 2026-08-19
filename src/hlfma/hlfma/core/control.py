"""
Decision → Command  (SPEC §3.6)

**Control 은 교통법을 모른다.** 주어진 목표 점열과 목표 속도를 추종할 뿐이다.

횡: Pure Pursuit. 좌표 원점이 뒷바퀴 축 중심이라 보정 없이 그대로 쓴다(SPEC §1.3).
종: PI + 저크 제한.
"""
from __future__ import annotations

import math

from .types import Command, Decision, WorldState


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


class SteerSignMonitor:
    """
    조향 명령과 실제 회전 방향이 같은 부호인지 감시한다.

    comm.steer_sign 이 뒤집히면 제어 루프가 양의 피드백이 되어 직선에서도 발산하고,
    차는 도로를 벗어나 courseRespawn 리셋을 반복한다. 2026-08-19 실측에서
    corr(steering, yaw_rate) = -0.93 으로 이 상태가 확인됐다.

    주행 시작 후 window_s 동안 표본을 모아 상관계수를 낸다. 음수면 부호가 반대다.
    표본은 아래를 만족하는 틱만 쓴다(그 외에는 상관이 의미 없다):
      - 리셋(순간이동) 직후가 아니다
      - 속도가 min_speed 이상   (정지 중에는 조향이 회전을 만들지 않는다)
      - 조향이 min_steer 이상   (미세 조향은 노이즈에 묻힌다)
    """

    def __init__(self, window_s: float = 5.0, min_speed: float = 1.0,
                 min_steer: float = 0.01, min_samples: int = 20,
                 max_s: float = 30.0) -> None:
        # window_s: 가장 이른 판정 시점. 추종이 좋으면 조향이 작아 표본이 잘 안 쌓이므로
        # 표본이 찰 때까지 max_s 까지 기다린다 (반대 부호일 때는 조향이 커서 금방 찬다).
        self.max_s = max_s
        self.window_s = window_s
        self.min_speed = min_speed
        self.min_steer = min_steer
        self.min_samples = min_samples
        self._steer: list[float] = []
        self._yaw_rate: list[float] = []
        self._prev = None            # (t, yaw)
        self._t0 = None
        self.verdict: str | None = None      # None=판정중, 'ok' | 'inverted' | 'unknown'
        self.corr: float | None = None

    def update(self, t: float, yaw: float, speed: float, steering: float,
               was_reset: bool) -> None:
        if self.verdict is not None:
            return
        if self._t0 is None:
            self._t0 = t

        prev, self._prev = self._prev, (t, yaw)
        if was_reset:
            self._prev = None                 # 순간이동 구간은 표본에서 뺀다
        elif prev is not None:
            dt = t - prev[0]
            if 1e-4 < dt < 0.5 and speed >= self.min_speed and abs(steering) >= self.min_steer:
                dyaw = (yaw - prev[1] + math.pi) % (2 * math.pi) - math.pi
                self._steer.append(steering)
                self._yaw_rate.append(dyaw / dt)

        elapsed = t - self._t0
        if elapsed >= self.window_s and len(self._steer) >= self.min_samples:
            self._decide()
        elif elapsed >= self.max_s:
            self._decide()          # 표본이 모자라도 여기서 끝낸다 ('unknown')

    def _decide(self) -> None:
        n = len(self._steer)
        if n < self.min_samples:
            self.verdict = 'unknown'
            return
        mx = sum(self._steer) / n
        my = sum(self._yaw_rate) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(self._steer, self._yaw_rate))
        sxx = sum((a - mx) ** 2 for a in self._steer)
        syy = sum((b - my) ** 2 for b in self._yaw_rate)
        if sxx <= 0 or syy <= 0:
            self.verdict = 'unknown'
            return
        self.corr = sxy / math.sqrt(sxx * syy)
        self.verdict = 'inverted' if self.corr < 0 else 'ok'

    @property
    def samples(self) -> int:
        return len(self._steer)


class Control:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        v = cfg['vehicle']
        c = cfg['control']
        s = cfg['speed']

        self.wheelbase = float(v['wheelbase'])
        self.max_steer = float(v['max_steer'])
        self.kp = float(c['kp'])
        self.ki = float(c['ki'])
        self.k_ld = float(c['k_ld'])
        self.ld_min = float(c['ld_min'])
        self.ld_max = float(c['ld_max'])
        self.steer_rate_max = float(c['steer_rate_max'])
        self.ki_band = float(c.get('ki_band_mps', 1e9))
        self.k_curv = float(c.get('k_curv', 0.0))
        self.ld_curve_min = float(c.get('ld_curve_min', 3.0))
        self.last_curv = 0.0
        self.a_min = float(s['a_min'])
        self.a_max = float(s['a_max'])
        self.jerk_max = float(s['jerk_max'])
        self.a_hold = float(s.get('a_hold', -1.0))

        # STEER_SIGN 은 Comm.send 에서 곱한다. 여기서는 내부 부호(좌 +)로 둔다.
        self._i_term = 0.0
        self._prev_steer = 0.0
        self._prev_accel = 0.0
        self._prev_t: float | None = None
        self.last_target: tuple[float, float] | None = None
        self.last_ld: float = 0.0

        # 조향 부호 감시 (comm.steer_sign 이 뒤집히면 5초 안에 잡는다)
        self.sign_monitor = SteerSignMonitor()

    def compute(self, world: WorldState, decision: Decision) -> Command:
        """횡(Pure Pursuit) + 종(PI) → Command."""
        # VTD 리셋(리스폰) 직후에는 적분항과 직전 조향이 전부 무의미하다.
        # 그대로 두면 리셋 전의 누적 오차와 최대 락 조향을 새 위치에서 이어받는다.
        if world.flags.get('reset'):
            self.reset()

        dt = self._dt(world.t)
        steering = self._pure_pursuit(world, decision.path, dt)
        accel = self._longitudinal(world, decision, dt)

        self.sign_monitor.update(world.t, world.ego.yaw, world.ego.speed, steering,
                                 bool(world.flags.get('reset')))

        return Command(steering=steering, accel=accel, turn_signal=int(decision.turn_signal))

    def _dt(self, t: float) -> float:
        if self._prev_t is None:
            self._prev_t = t
            return 1.0 / float(self.cfg['comm']['send_hz'])
        dt = t - self._prev_t
        self._prev_t = t
        if dt <= 1e-4 or dt > 1.0:          # 이상치는 공칭 주기로 대체
            return 1.0 / float(self.cfg['comm']['send_hz'])
        return dt

    # ── 횡방향 ────────────────────────────────────────────────────────────
    def _pure_pursuit(self, world: WorldState, path: list[tuple[float, float]],
                      dt: float) -> float:
        """
        L_d      = clamp(K_LD * v, LD_MIN, LD_MAX)
        목표점   = path 위에서 ego 로부터 L_d 이상 떨어진 **첫 점**
        alpha    = 목표점의 차량좌표계 방위각
        delta    = atan2(2 * WHEELBASE * sin(alpha), L_d)
        steering = clamp(delta, -MAX_STEER, MAX_STEER)  + 변화율 제한

        경로가 없으면 조향을 0 으로 되돌린다(변화율 제한은 그대로 적용).
        """
        ego = world.ego
        v = max(0.0, ego.speed)
        ld = _clamp(self.k_ld * v, self.ld_min, self.ld_max)

        # 곡률이 크면 lookahead 를 줄인다.
        # 47 km/h 면 L_d 가 10 m 인데 교차로 연결로 반경이 13 m 라, 그대로 두면
        # 목표점이 코너 안쪽에 놓여 경로를 잘라 먹는다.
        #   L_d = clamp(k_ld*v, ld_min, ld_max) / (1 + k_curv*|curv|)
        curv = self._path_curvature(path, ld)
        self.last_curv = curv
        if self.k_curv > 0.0 and curv > 0.0:
            ld = max(self.ld_curve_min, ld / (1.0 + self.k_curv * curv))
        self.last_ld = ld

        target = self._find_target(path, ego.x, ego.y, ld)
        self.last_target = target

        if target is None:
            return self._rate_limit(0.0, dt)

        # 월드 → 차량좌표계 (x 전방, y 좌측)
        dx, dy = target[0] - ego.x, target[1] - ego.y
        c, s = math.cos(-ego.yaw), math.sin(-ego.yaw)
        lx = dx * c - dy * s
        ly = dx * s + dy * c

        dist = math.hypot(lx, ly)
        if dist < 1e-3:
            return self._rate_limit(0.0, dt)

        alpha = math.atan2(ly, lx)
        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), max(dist, 1e-3))
        return self._rate_limit(_clamp(delta, -self.max_steer, self.max_steer), dt)

    @staticmethod
    def _path_curvature(path: list[tuple[float, float]], span_m: float) -> float:
        """
        경로 앞쪽 span_m 구간의 최대 |곡률| [1/m].

        LaneGraph 를 참조하지 않고 **따라갈 경로 자체**에서 낸다. 실제로 추종해야
        하는 곡률이 그것이고, core 가 지도에 의존하지 않아도 된다.
        연속 세 점의 외접원 반지름(Menger 곡률)을 쓴다.
        """
        if len(path) < 3:
            return 0.0
        acc = 0.0
        worst = 0.0
        for i in range(1, len(path) - 1):
            ax, ay = path[i - 1]
            bx, by = path[i]
            cx, cy = path[i + 1]
            acc += math.hypot(bx - ax, by - ay)
            if acc > max(span_m, 1.0):
                break
            d1 = math.hypot(bx - ax, by - ay)
            d2 = math.hypot(cx - bx, cy - by)
            d3 = math.hypot(cx - ax, cy - ay)
            if d1 < 1e-6 or d2 < 1e-6 or d3 < 1e-6:
                continue
            # 삼각형 넓이 * 2
            cross = abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
            k = 2.0 * cross / (d1 * d2 * d3)
            if k > worst:
                worst = k
        return worst

    @staticmethod
    def _find_target(path: list[tuple[float, float]], x: float, y: float,
                     ld: float) -> tuple[float, float] | None:
        """
        ego 에서 L_d 이상 떨어진 첫 점. 전부 그보다 가까우면 마지막 점을 쓴다
        (경로 끝에서 조향이 갑자기 0 이 되지 않도록).
        """
        if not path:
            return None
        for px, py in path:
            if math.hypot(px - x, py - y) >= ld:
                return (px, py)
        return path[-1]

    def _rate_limit(self, steer: float, dt: float) -> float:
        """조향 변화율 제한 [rad/s]."""
        max_delta = self.steer_rate_max * dt
        steer = _clamp(steer, self._prev_steer - max_delta, self._prev_steer + max_delta)
        self._prev_steer = steer
        return steer

    # ── 종방향 ────────────────────────────────────────────────────────────
    def _longitudinal(self, world: WorldState, decision: Decision, dt: float) -> float:
        """
        accel = KP*(v_target - v) + KI*∫,  clamp(A_MIN, A_MAX), 저크 제한.

        - `state == 'E_STOP'` 이면 저크 제한 해제
        - 정지 유지: v_target == 0 && v < 0.2 → accel = `speed.a_hold`
        - 적분 와인드업 방지: clamp 에 걸린 동안 적분을 되돌린다
        """
        v = world.ego.speed
        v_t = decision.v_target

        # 정지 유지 — 브레이크를 계속 물고 있어야 굴러가지 않는다
        if v_t <= 1e-6 and v < 0.2:
            self._i_term = 0.0
            self._prev_accel = self.a_hold
            return self.a_hold

        err = v_t - v
        # 적분은 목표 근처에서만 쌓는다.
        # 정지→목표속도처럼 오차가 큰 구간에서 계속 적분하면, 목표에 도달한 뒤에도
        # 적분항이 가속을 밀어 제한속도를 넘긴다 (실측: 27 목표에 28.4 까지 오버슛).
        # 출력 포화 시 되돌리는 기존 방어만으로는 포화 전 구간을 못 막는다.
        if abs(err) <= self.ki_band:
            self._i_term += err * dt
        else:
            self._i_term *= 0.9          # 대역 밖에서는 서서히 흘려보낸다
        raw = self.kp * err + self.ki * self._i_term

        accel = _clamp(raw, self.a_min, self.a_max)
        if accel != raw:                       # 와인드업 방지
            self._i_term -= err * dt

        if decision.state != 'E_STOP':         # 비상정지는 저크 제한 해제
            max_djerk = self.jerk_max * dt
            accel = _clamp(accel, self._prev_accel - max_djerk, self._prev_accel + max_djerk)

        self._prev_accel = accel
        return accel

    def reset(self) -> None:
        """적분항/이력 초기화 (리플레이·테스트용)."""
        self._i_term = 0.0
        self._prev_steer = 0.0
        self._prev_accel = 0.0
        self._prev_t = None
