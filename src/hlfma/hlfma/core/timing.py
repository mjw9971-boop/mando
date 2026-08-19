"""
단계별 소요시간 계측.

20 Hz 루프에서 한 틱이 50 ms 를 넘으면 그만큼 제어가 늦어진다. 실측에서 틱 간격이
최대 2 초까지 벌어지는 스톨이 관측됐고, 어느 단계가 먹는지 알아야 고칠 수 있다.

ROS 를 모른다 — core 어디서든 쓰고, 노드가 요약을 찍는다.
"""
from __future__ import annotations

import time


class Stage:
    """한 단계의 소요시간 통계."""

    def __init__(self, name: str, buckets_ms=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)) -> None:
        self.name = name
        self.buckets = tuple(buckets_ms)
        self.n = 0
        self.total = 0.0
        self.max = 0.0
        self.max_at: float | None = None
        self._hist = [0] * (len(self.buckets) + 1)

    def record(self, dt_s: float, at: float | None = None) -> None:
        ms = dt_s * 1000.0
        self.n += 1
        self.total += ms
        if ms > self.max:
            self.max = ms
            self.max_at = at
        for i, b in enumerate(self.buckets):
            if ms <= b:
                self._hist[i] += 1
                return
        self._hist[-1] += 1

    @property
    def mean(self) -> float:
        return self.total / self.n if self.n else 0.0

    def over(self, ms: float) -> int:
        """ms 를 넘은 횟수."""
        cnt = 0
        for i, b in enumerate(self.buckets):
            if b > ms:
                cnt += self._hist[i]
        return cnt + self._hist[-1]

    def summary(self) -> str:
        if not self.n:
            return f'{self.name}: 표본 없음'
        at = f' @t={self.max_at:.1f}s' if self.max_at is not None else ''
        return (f'{self.name}: n={self.n} 평균 {self.mean:.2f} ms '
                f'최대 {self.max:.1f} ms{at}  >50ms {self.over(50)}회')

    def histogram(self) -> str:
        if not self.n:
            return ''
        parts = []
        prev = 0
        for i, b in enumerate(self.buckets):
            if self._hist[i]:
                parts.append(f'{prev}~{b}ms:{self._hist[i]}')
            prev = b
        if self._hist[-1]:
            parts.append(f'>{self.buckets[-1]}ms:{self._hist[-1]}')
        return '  ' + ' '.join(parts)


class Timer:
    """with Timer(stage, at): ... 로 감싼다."""

    __slots__ = ('stage', 'at', '_t')

    def __init__(self, stage: Stage, at: float | None = None) -> None:
        self.stage = stage
        self.at = at

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.stage.record(time.perf_counter() - self._t, self.at)
        return False
