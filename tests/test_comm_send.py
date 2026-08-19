"""
Comm.send 의 EAGAIN 재시도 처리 (논블로킹 소켓의 backpressure).

예전에는 EAGAIN 을 치명적 오류로 보고 연결을 닫아, 주행 중 송신이 잠깐 밀리기만
해도 링크가 끊겼다. 여기서 검증하는 계약:
  - 일시적 EAGAIN 은 재시도로 흡수한다
  - 재시도 소진 시 이번 틱을 포기하되 **명령을 쌓지 않는다**
  - 프레임 중간 조각은 반드시 보존한다 (9 바이트 경계)
  - 연속 실패가 watchdog_s 를 넘으면 그때 소켓을 닫는다
"""
import time

import pytest

from hlfma.core.comm import CTRL_SIZE, Comm
from hlfma.core.types import Command

CMD = Command(steering=0.1, accel=-1.0, turn_signal=0)


class FakeSock:
    """send() 동작을 시나리오로 지정하는 가짜 소켓."""

    def __init__(self, script, default=None):
        # script: 각 send 호출에 대해 int(보낸 바이트) 또는 예외 인스턴스
        # default: script 소진 후의 기본 동작 (None = 전량 송신 성공)
        self.script = list(script)
        self.default = default
        self.sent = bytearray()
        self.closed = False
        self.calls = 0

    def send(self, data):
        self.calls += 1
        if self.script:
            act = self.script.pop(0)
        elif self.default is not None:
            act = self.default() if callable(self.default) else self.default
        else:
            act = len(data)
        if isinstance(act, BaseException):
            raise act
        n = min(act, len(data))
        self.sent.extend(data[:n])
        return n

    def close(self):
        self.closed = True


def make_comm(sock, watchdog_s=1.0, retries=3):
    c = Comm('127.0.0.1', 9910, watchdog_s=watchdog_s, steer_sign=1.0,
             send_retries=retries, send_retry_delay_s=0.0,
             logger=_Silent())
    c.sock = sock
    return c


class _Silent:
    def info(self, msg): pass
    def warn(self, msg): pass


# ── 정상 ──────────────────────────────────────────────────────────────────
def test_normal_send_writes_full_frame():
    s = FakeSock([])
    c = make_comm(s)
    assert c.send(CMD) is True
    assert len(s.sent) == CTRL_SIZE
    assert c.send_ok == 1 and c.send_retry_count == 0 and c.send_skipped == 0


# ── 일시적 EAGAIN ─────────────────────────────────────────────────────────
def test_transient_eagain_is_retried_then_succeeds():
    s = FakeSock([BlockingIOError(11, 'EAGAIN'), BlockingIOError(11, 'EAGAIN'), CTRL_SIZE])
    c = make_comm(s)
    assert c.send(CMD) is True
    assert c.send_retry_count == 2
    assert c.send_skipped == 0
    assert len(s.sent) == CTRL_SIZE
    assert not s.closed, 'EAGAIN 으로 연결을 닫으면 안 된다'


def test_eagain_does_not_close_socket_when_giving_up():
    s = FakeSock([], default=lambda: BlockingIOError(11))
    c = make_comm(s, watchdog_s=10.0)
    assert c.send(CMD) is False
    assert c.send_skipped == 1
    assert not s.closed, 'watchdog 이내면 연결 유지'


# ── 밀린 명령 누적 금지 ───────────────────────────────────────────────────
def test_dropped_command_is_not_queued():
    """통째로 못 보낸 명령은 버린다 — 다음 틱에 최신 명령만 나가야 한다."""
    s = FakeSock([], default=lambda: BlockingIOError(11))
    c = make_comm(s, watchdog_s=10.0)
    for _ in range(5):
        c.send(CMD)
    assert c.send_dropped_frames == 5
    assert len(c._pending) == 0, '완결 프레임을 쌓아두면 안 된다'

    # 이제 소켓이 뚫리면 최신 명령 한 개만 나간다
    s.default = None
    assert c.send(Command(0.2, 1.0, 1)) is True
    assert len(s.sent) == CTRL_SIZE


# ── 프레임 경계 보존 ──────────────────────────────────────────────────────
def test_partial_frame_remainder_is_preserved():
    """4 바이트만 나가면 남은 5 바이트를 보관했다가 다음에 마저 보낸다."""
    s = FakeSock([4], default=lambda: BlockingIOError(11))
    c = make_comm(s, watchdog_s=10.0)
    assert c.send(CMD) is False
    assert len(c._pending) == CTRL_SIZE - 4

    s.default = None
    assert c.send(CMD) is True
    # 첫 프레임 9 바이트가 온전히 복원되고, 두 번째 프레임도 나갔다
    assert len(s.sent) == 2 * CTRL_SIZE
    assert len(s.sent) % CTRL_SIZE == 0, '프레임 경계가 깨졌다'


def test_pending_never_reaches_full_frame():
    """_pending 은 항상 미완 조각(<9B)이어야 한다."""
    s = FakeSock([3], default=lambda: BlockingIOError(11))
    c = make_comm(s, watchdog_s=10.0)
    for _ in range(6):
        c.send(CMD)
        assert len(c._pending) < CTRL_SIZE


# ── watchdog ──────────────────────────────────────────────────────────────
def test_blocked_longer_than_watchdog_closes_socket():
    s = FakeSock([], default=lambda: BlockingIOError(11))
    c = make_comm(s, watchdog_s=0.05)
    assert c.send(CMD) is False
    assert not s.closed
    time.sleep(0.08)
    c.send(CMD)
    assert s.closed, 'watchdog 초과 시 재접속 절차로 넘겨야 한다'


def test_success_clears_blocked_timer():
    s = FakeSock([], default=lambda: BlockingIOError(11))
    c = make_comm(s, watchdog_s=0.05)
    c.send(CMD)                      # 포기 -> blocked 시작
    assert c._blocked_since is not None
    s.default = None
    assert c.send(CMD) is True
    assert c._blocked_since is None, '성공하면 연속실패 타이머가 풀려야 한다'


# ── 진짜 오류는 그대로 치명 ───────────────────────────────────────────────
def test_real_oserror_still_closes():
    s = FakeSock([ConnectionResetError(104, 'reset')])
    c = make_comm(s)
    assert c.send(CMD) is False
    assert s.closed, 'EAGAIN 이 아닌 오류는 연결을 닫아야 한다'


def test_reconnect_clears_pending():
    """끊긴 연결의 조각을 새 연결로 흘리면 정렬이 깨진다."""
    s = FakeSock([4], default=lambda: BlockingIOError(11))
    c = make_comm(s, watchdog_s=10.0)
    c.send(CMD)
    assert c._pending
    c.connect = lambda *a, **k: True     # 실제 접속은 하지 않는다
    c.reconnect()
    assert c._pending == b''


def test_stats_string_has_counters():
    s = FakeSock([], default=lambda: BlockingIOError(11))
    c = make_comm(s, watchdog_s=10.0)
    c.send(CMD)
    stats = c.send_stats()
    for key in ('ok=', 'retry=', 'skip=', 'drop=', 'pending='):
        assert key in stats
