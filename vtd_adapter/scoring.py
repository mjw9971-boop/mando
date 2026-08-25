"""
주행 품질 집계 — 제한속도 준수가 채점 항목(S1.1.01 / S1.1.02)이라 상시 본다.

**위반 기준은 법규 제한속도 자체**다. params 의 speed.margin_kph 는 우리가 스스로
둔 여유이므로 그걸 넘었다고 감점되지 않는다. 둘을 따로 센다.

ROS 를 모른다 — 노드와 tools/score.py 가 같은 코드를 쓴다.
"""
from __future__ import annotations


class SpeedMonitor:
    """구간별(제한속도 × 스쿨존) 실제속도 통계와 초과 집계."""

    def __init__(self, margin_kph: float = 3.0, school_cap_kph: float = 28.0) -> None:
        self.margin = float(margin_kph)
        self.school_cap = float(school_cap_kph)
        self.groups: dict = {}
        self.n = 0
        self.t_first = None
        self.t_last = None
        self.route_s_max = 0.0

    def feed(self, speed_mps: float, limit_mps: float, school_zone: bool,
             route_s: float = 0.0, t: float | None = None) -> None:
        v = speed_mps * 3.6
        lim = round(limit_mps * 3.6)
        key = (lim, bool(school_zone))
        g = self.groups.setdefault(key, {'n': 0, 'sum': 0.0, 'max': 0.0,
                                         'over': 0, 'over_max': 0.0,
                                         'tight': 0, 'over_at': None})
        legal = float(lim)
        target = min(lim - self.margin, self.school_cap) if school_zone else lim - self.margin
        g['n'] += 1
        g['sum'] += v
        g['max'] = max(g['max'], v)
        g['legal'] = legal
        g['target'] = target
        if v > legal:
            g['over'] += 1
            if v - legal > g['over_max']:
                g['over_max'] = v - legal
                g['over_at'] = route_s
        if v > target:
            g['tight'] += 1

        self.n += 1
        self.route_s_max = max(self.route_s_max, route_s)
        if t is not None:
            if self.t_first is None:
                self.t_first = t
            self.t_last = t

    # ── 집계 ──────────────────────────────────────────────────────────────
    @property
    def n_over(self) -> int:
        return sum(g['over'] for g in self.groups.values())

    @property
    def over_max(self) -> float:
        return max((g['over_max'] for g in self.groups.values()), default=0.0)

    @property
    def n_tight(self) -> int:
        return sum(g['tight'] for g in self.groups.values())

    @property
    def duration(self) -> float:
        if self.t_first is None or self.t_last is None:
            return 0.0
        return self.t_last - self.t_first

    @property
    def avg_kph(self) -> float:
        d = self.duration
        return (self.route_s_max / d * 3.6) if d > 1e-6 else 0.0

    def lines(self) -> list[str]:
        """사람이 읽는 요약. 노드 로거로 한 줄씩 내보낸다."""
        out = [
            f'주행 요약: {self.route_s_max:.1f} m / {self.duration:.1f} s / '
            f'평균 {self.avg_kph:.1f} km/h  ({self.n} 틱)',
            f'  {"제한":>5} {"스쿨존":>7} {"틱":>6} {"평균":>7} {"최대":>7} '
            f'{"법규위반":>9} {"최대초과":>9} {"목표":>6} {"목표초과":>9}',
        ]
        for key in sorted(self.groups):
            lim, sz = key
            g = self.groups[key]
            om = f"+{g['over_max']:.2f}" if g['over'] else '—'
            out.append(
                f'  {lim:5d} {str(sz):>7} {g["n"]:6d} {g["sum"]/g["n"]:7.1f} '
                f'{g["max"]:7.1f} {g["over"]:9d} {om:>9} {g["target"]:6.1f} {g["tight"]:9d}')
        if self.n_over:
            out.append(f'  → **제한속도 위반 {self.n_over}틱, 최대 +{self.over_max:.2f} km/h** '
                       f'(채점 감점 대상)')
        else:
            out.append('  → 제한속도 위반 없음')
        out.append(f'     목표 초과 {self.n_tight}틱 = 우리 여유(margin)를 쓴 것, 감점 아님')
        return out
