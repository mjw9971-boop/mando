"""
로그 jsonl → Comm 대체 소스  (SPEC §3.7)

VTD 없이 Perception/Planner/Shield/Control 을 다시 돌린다.
판단 로직을 고친 뒤 회귀 테스트하는 기반.

    python3 src/main.py --replay logs/run_xxx.jsonl
"""
from __future__ import annotations

from hlfma.core.types import Command, RawPacket


class ReplaySource:
    """Comm 과 같은 인터페이스(recv/send/stale/connected/close)를 흉내낸다."""

    def __init__(self, path: str) -> None:
        self.path = path
        # TODO: jsonl 열고 한 줄씩 읽을 준비

    def recv(self) -> RawPacket | None:
        """다음 줄의 raw 필드를 RawPacket 으로 복원. 끝나면 None."""
        # TODO: 구현
        raise NotImplementedError('replay.recv')

    def send(self, cmd: Command) -> bool:
        """리플레이에서는 실제로 보내지 않는다. 원본 Command 와 비교하면 회귀 검출."""
        # TODO: 원본 로그의 Command 와 diff 를 집계
        return True

    def stale(self, now: float | None = None) -> bool:
        return False

    @property
    def connected(self) -> bool:
        return True

    def close(self) -> None:
        # TODO: 파일 닫기 + diff 요약 출력
        pass
