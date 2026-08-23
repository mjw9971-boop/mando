"""
9910 TCP 송수신. SPEC §1.1 / §1.2 / §3.2 를 그대로 구현한다.

수신: 1109 B 고정 프레임, 헤더 없음, little-endian, 20 Hz
송신: 9 B, `<ffB` (steering, targetAccel, turnSignal)

스트림에는 길이 필드가 없다. recv 가 1109 배수로 떨어지지 않으므로 버퍼에
쌓아 1109 씩 잘라내고, **밀린 프레임은 버리고 가장 최신 것만** 쓴다(지연 누적 방지).
"""
from __future__ import annotations

import socket
import struct
import time

from .types import Command, RawPacket

# ── SPEC §1.1 수신 프레임 레이아웃 ────────────────────────────────────────
EGO_FMT = struct.Struct('<6f')          # x, y, z, heading, pitch, roll
OBJ_FMT = struct.Struct('<I8f')         # id, x, y, z, heading, speed, length, width, height
TL_FMT = struct.Struct('<iB')           # id, state

EGO_SIZE = EGO_FMT.size                 # 24
OBJ_SIZE = OBJ_FMT.size                 # 36
TL_SIZE = TL_FMT.size                   # 5

OBJ_COUNT = 30
TL_COUNT = 1

OBJ_BASE = EGO_SIZE                     # 24
TL_BASE = OBJ_BASE + OBJ_SIZE * OBJ_COUNT   # 1104
FRAME_SIZE = TL_BASE + TL_SIZE * TL_COUNT   # 1109

# ── SPEC §1.2 송신 ────────────────────────────────────────────────────────
CTRL_FMT = struct.Struct('<ffB')        # steering, targetAccel, turnSignal
CTRL_SIZE = CTRL_FMT.size               # 9

# 신호등 state (SPEC §1.1)
TL_UNASSIGNED, TL_RED, TL_YELLOW, TL_GREEN, TL_LEFT, TL_GREEN_LEFT, TL_FLASH = range(7)

TURN_OFF, TURN_LEFT, TURN_RIGHT = 0, 1, 2


def parse(frame: bytes, t_recv: float | None = None, frames_total: int = 0) -> RawPacket:
    """1109 B 한 프레임 → RawPacket. id == 0 인 슬롯은 버린다."""
    if len(frame) < FRAME_SIZE:
        raise ValueError(f'프레임이 짧다: {len(frame)} < {FRAME_SIZE}')

    ego = EGO_FMT.unpack_from(frame, 0)

    objects: list[tuple] = []
    for i in range(OBJ_COUNT):
        rec = OBJ_FMT.unpack_from(frame, OBJ_BASE + i * OBJ_SIZE)
        if rec[0] == 0:          # 빈 슬롯은 전 필드 0
            continue
        objects.append(rec)

    lights: list[tuple[int, int]] = []
    for i in range(TL_COUNT):
        tid, state = TL_FMT.unpack_from(frame, TL_BASE + i * TL_SIZE)
        if tid == 0:             # 지금 볼 신호등이 없으면 (0, 0)
            continue
        lights.append((tid, state))

    return RawPacket(
        t_recv=time.monotonic() if t_recv is None else t_recv,
        ego=ego, objects=objects, lights=lights, frames_total=frames_total,
    )


def build_frame(ego: tuple, objects=(), lights=()) -> bytes:
    """parse 의 역함수. 테스트/리플레이 전용."""
    buf = bytearray(FRAME_SIZE)
    EGO_FMT.pack_into(buf, 0, *ego)
    for i, o in enumerate(list(objects)[:OBJ_COUNT]):
        OBJ_FMT.pack_into(buf, OBJ_BASE + i * OBJ_SIZE, int(o[0]), *[float(v) for v in o[1:9]])
    for i, (tid, state) in enumerate(list(lights)[:TL_COUNT]):
        TL_FMT.pack_into(buf, TL_BASE + i * TL_SIZE, int(tid), int(state))
    return bytes(buf)


def pack_command(cmd: Command, steer_sign: float = 1.0) -> bytes:
    """
    Command → 9 B.

    steer_sign 은 실측으로 +1.0 확정 (2026-08-19, VTD 2025.2). VTD 는 우리가 보낸
    조향을 그대로 적용한다. -1.0 이면 제어가 양의 피드백이 되어 직선에서도 발산한다.
    """
    return CTRL_FMT.pack(
        float(steer_sign) * float(cmd.steering),
        float(cmd.accel),
        int(cmd.turn_signal) if cmd.turn_signal in (0, 1, 2) else 0,
    )


class Comm:
    """9910 TCP 클라이언트. VTD 가 서버, 우리가 클라이언트다."""

    def __init__(self, host: str, port: int, *, watchdog_s: float = 1.0,
                 steer_sign: float = 1.0, connect_retry_s: float = 1.0,
                 recv_bufsize: int = 65536, logger=None,
                 send_retries: int = 3, send_retry_delay_s: float = 0.002) -> None:
        self.host = host
        self.port = port
        self.watchdog_s = watchdog_s
        self.steer_sign = steer_sign
        self.connect_retry_s = connect_retry_s
        self.recv_bufsize = recv_bufsize
        self._log = logger
        self.send_retries = max(1, int(send_retries))
        self.send_retry_delay_s = float(send_retry_delay_s)

        self.sock: socket.socket | None = None
        self._buf = bytearray()
        self.last_rx: float = 0.0      # 마지막으로 프레임을 받은 monotonic 시각
        self.frames_seen = 0
        self.frames_dropped = 0        # 밀려서 버린 프레임 수

        # 송신 상태/카운터
        self._pending = b''            # 못 다 보낸 프레임 조각 (항상 CTRL_SIZE 미만)
        self._blocked_since: float | None = None
        self.send_ok = 0
        self.send_retry_count = 0      # EAGAIN 으로 다시 시도한 횟수
        self.send_skipped = 0          # 재시도 소진해 이번 틱을 포기한 횟수
        self.send_dropped_frames = 0   # 통째로 못 보내고 버린 명령 수
        self._last_send_warn = 0.0

    # ── 연결 ──────────────────────────────────────────────────────────────
    def connect(self, timeout: float | None = None) -> bool:
        """접속될 때까지 재시도. timeout 을 주면 그 시간까지만 시도한다."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect((self.host, self.port))
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.setblocking(False)
                self.sock = s
                self._buf.clear()
                self.last_rx = time.monotonic()
                self._info(f'연결 성공 {self.host}:{self.port}')
                return True
            except OSError as e:
                self._warn(f'{self.host}:{self.port} 연결 실패 ({e}) — {self.connect_retry_s}s 후 재시도')
                if deadline is not None and time.monotonic() >= deadline:
                    return False
                time.sleep(self.connect_retry_s)

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def reconnect(self) -> bool:
        self.close()
        # 끊긴 연결의 프레임 조각을 새 연결로 흘려보내면 정렬이 깨진다
        self._pending = b''
        self._blocked_since = None
        return self.connect()

    # ── 수신 ──────────────────────────────────────────────────────────────
    def recv(self) -> RawPacket | None:
        """
        지금 소켓에 와 있는 것을 전부 읽고 **가장 최신 프레임 하나**만 돌려준다.
        완성된 프레임이 없으면 None (호출자는 직전 명령을 유지하면 된다).
        """
        if self.sock is None:
            return None

        while True:
            try:
                chunk = self.sock.recv(self.recv_bufsize)
            except (BlockingIOError, InterruptedError):
                break
            except OSError as e:
                self._warn(f'수신 오류: {e}')
                self.close()
                return None
            if not chunk:                      # 상대가 닫음
                self._warn('상대가 연결을 닫았다')
                self.close()
                return None
            self._buf.extend(chunk)

        if len(self._buf) < FRAME_SIZE:
            return None

        n_full = len(self._buf) // FRAME_SIZE
        if n_full > 1:
            self.frames_dropped += n_full - 1   # 밀린 것은 버린다
        last = bytes(self._buf[(n_full - 1) * FRAME_SIZE: n_full * FRAME_SIZE])
        del self._buf[:n_full * FRAME_SIZE]

        self.frames_seen += 1
        self.last_rx = time.monotonic()
        return parse(last, t_recv=self.last_rx,
                     frames_total=self.frames_seen + self.frames_dropped)

    # ── 송신 ──────────────────────────────────────────────────────────────
    def send(self, cmd: Command) -> bool:
        """
        제어 한 프레임 송신. 성공하면 True.

        소켓이 논블로킹이라 송신 버퍼가 차면 EAGAIN(BlockingIOError)이 난다.
        이건 **치명적 오류가 아니라 일시적 backpressure** 다. 예전에는 이걸
        연결 종료로 처리해서 주행 중에 링크가 끊겼다.

        처리 방식
          1. 수 ms 간격으로 send_retries 회까지 재시도
          2. 그래도 안 나가면 이번 틱은 포기 — 다음 틱에 **최신** 명령을 보낸다.
             못 보낸 명령을 쌓아두지 않는다(밀린 명령 누적 금지).
          3. 다만 프레임 중간까지 나간 조각은 반드시 남겨 다음에 마저 보낸다.
             9 바이트 경계가 깨지면 수신측 프레임 정렬이 통째로 어긋난다.
          4. 연속 실패가 watchdog_s 를 넘으면 소켓을 닫아 재접속 절차로 넘긴다.
        """
        if self.sock is None:
            return False

        # 이전 조각(항상 미완 프레임)을 앞에 붙여 경계를 지킨다
        data = self._pending + pack_command(cmd, self.steer_sign)
        sent = 0

        for attempt in range(self.send_retries):
            try:
                while sent < len(data):
                    sent += self.sock.send(data[sent:])
                # 전부 나갔다
                self._pending = b''
                self._blocked_since = None
                self.send_ok += 1
                return True
            except (BlockingIOError, InterruptedError):
                self.send_retry_count += 1
                if attempt < self.send_retries - 1:
                    time.sleep(self.send_retry_delay_s)
            except OSError as e:
                # EAGAIN 이 아닌 진짜 오류 — 연결을 닫는다
                self._warn(f'송신 실패(치명): {e}')
                self._pending = b''
                self._blocked_since = None
                self.close()
                return False

        # ── 재시도 소진: 이번 틱 포기 ────────────────────────────────────
        self.send_skipped += 1
        rem = data[sent:]
        if len(rem) >= CTRL_SIZE:
            # 완결 프레임이 통째로 안 나갔다 -> 그 명령은 버린다(다음 틱에 최신 명령).
            # 미완 조각만 남겨 경계를 지킨다.
            self._pending = rem[:len(rem) - CTRL_SIZE]
            self.send_dropped_frames += 1
        else:
            self._pending = rem

        now = time.monotonic()
        if self._blocked_since is None:
            self._blocked_since = now
        blocked = now - self._blocked_since

        if now - self._last_send_warn >= 1.0:
            self._last_send_warn = now
            self._warn(f'송신 지연 {blocked:.2f}s — {self.send_stats()}')

        if blocked > self.watchdog_s:
            self._warn(f'송신이 {blocked:.2f}s 막힘(watchdog {self.watchdog_s}s 초과) '
                       f'— 재접속. {self.send_stats()}')
            self._pending = b''
            self._blocked_since = None
            self.close()
        return False

    def send_stats(self) -> str:
        """로그용 카운터 요약."""
        return (f'ok={self.send_ok} retry={self.send_retry_count} '
                f'skip={self.send_skipped} drop={self.send_dropped_frames} '
                f'pending={len(self._pending)}B')

    # ── 워치독 ────────────────────────────────────────────────────────────
    def stale(self, now: float | None = None) -> bool:
        """
        watchdog_s 동안 패킷이 없었는가.

        주의(SPEC §3.2): 적신호 대기가 13~18 s 이므로 **속도 0 을 이유로 재시작하면
        안 된다.** 오직 패킷 수신 여부로만 판단한다.
        """
        now = time.monotonic() if now is None else now
        if self.last_rx == 0.0:
            return True
        return (now - self.last_rx) > self.watchdog_s

    @property
    def connected(self) -> bool:
        return self.sock is not None

    # ── 로깅 ──────────────────────────────────────────────────────────────
    def _info(self, msg: str) -> None:
        if self._log:
            self._log.info(msg)
        else:
            print(f'[comm] {msg}', flush=True)

    def _warn(self, msg: str) -> None:
        if self._log:
            self._log.warn(msg)
        else:
            print(f'[comm] {msg}', flush=True)


SAFE_STOP = Command(steering=0.0, accel=-3.0, turn_signal=TURN_OFF)
"""워치독 발동 시 내보낼 안전정지 명령. accel 은 config 의 a_min 보다 완만하게."""
