"""
SCP(Simulation Control Protocol) 클라이언트 — VTD 를 파이썬에서 제어한다.

    python3 tools/scp_client.py listen --seconds 10
    python3 tools/scp_client.py load ~/VIRES/.../HL_FMA_VTD_LivingLab.xml
    python3 tools/scp_client.py start
    python3 tools/scp_client.py stop
    python3 tools/scp_client.py reset-player --player Ego
    python3 tools/scp_client.py raw '<SimCtrl><Pause/></SimCtrl>'

자동 회귀 테스트의 기반이다. 시나리오를 올리고 → 시작하고 → 우리 컨트롤러를 붙이고
→ 결과를 채점하고 → 다시 초기화하는 루프를 사람 손 없이 돌릴 수 있다.

[와이어 규격 — 실측 검증 완료]
    포트 48179 (viSCPIcd.h `SCP_DEFAULT_PORT`, simServer.xml `PORT_TC_2_SCP`)
    헤더 136 B:  magicNo(u16)=40108  version(u16)=0x0001
                 sender[64]  receiver[64]  dataSize(u32)
    (예전 코드가 쓰던 magic 0x53435000 / headerSize 필드는 **틀린 추정**이었다.
     실제 헤더에는 headerSize 가 없다. 위 값은 구동 중인 VTD 에서 수신해 확인했다.)

[명령 문법 출처] Doc/SCP_HTML/docu.html, VtdToolkit/Scp/ScpBuilder.h
    <SimCtrl><LoadScenario filename="..."/></SimCtrl>
    <SimCtrl><Init/></SimCtrl>  <SimCtrl><Start tMax=".."/></SimCtrl>
    <SimCtrl><Stop/></SimCtrl>  <SimCtrl><Pause/></SimCtrl>
    <SimCtrl><Restart/></SimCtrl>  <SimCtrl><Step/></SimCtrl>
    <Traffic><ResetPlayer name="Ego"/></Traffic>
    위치 속성은 x,y,z,hDeg,pDeg,rDeg — **hDeg 는 도(degree)** 다.
"""
from __future__ import annotations

import argparse
import math
import socket
import struct
import sys
import time

SCP_PORT = 48179
SCP_MAGIC = 40108
SCP_VERSION = 0x0001
NAME_LEN = 64

HDR = struct.Struct(f'<HH{NAME_LEN}s{NAME_LEN}sI')
HDR_SIZE = HDR.size          # 136
assert HDR_SIZE == 136, HDR_SIZE

NUL = b'\x00'


def cstr(b: bytes) -> str:
    return b.rstrip(NUL).decode('utf-8', 'replace')


def pack(payload: str, sender: str = 'hlfma', receiver: str = 'any') -> bytes:
    """SCP 메시지 한 통. 페이로드는 NUL 종료 문자열로 보낸다."""
    data = payload.encode('utf-8') + NUL
    return HDR.pack(SCP_MAGIC, SCP_VERSION,
                    sender.encode('utf-8')[:NAME_LEN - 1],
                    receiver.encode('utf-8')[:NAME_LEN - 1],
                    len(data)) + data


def unpack(buf: bytearray):
    """버퍼에서 완성된 메시지를 하나 꺼낸다 → (sender, receiver, payload) 또는 None."""
    if len(buf) < HDR_SIZE:
        return None
    magic, ver, snd, rcv, dsize = HDR.unpack_from(buf, 0)
    if magic != SCP_MAGIC:
        raise ValueError(f'SCP magic 불일치: {magic} (기대 {SCP_MAGIC})')
    if len(buf) < HDR_SIZE + dsize:
        return None
    data = bytes(buf[HDR_SIZE:HDR_SIZE + dsize])
    del buf[:HDR_SIZE + dsize]
    return cstr(snd), cstr(rcv), cstr(data)


class ScpClient:
    """
    VTD TaskControl 의 SCP 포트에 붙는다.

    with ScpClient() as scp:
        scp.load_scenario('/path/to.xml')
        scp.init(); scp.start()
    """

    def __init__(self, host: str = '127.0.0.1', port: int = SCP_PORT,
                 sender: str = 'hlfma', timeout: float = 5.0,
                 dry_run: bool = False, verbose: bool = True) -> None:
        self.host, self.port = host, port
        self.sender = sender
        self.timeout = timeout
        self.dry_run = dry_run
        self.verbose = verbose
        self.sock: socket.socket | None = None
        self._buf = bytearray()

    # ── 연결 ──────────────────────────────────────────────────────────────
    def connect(self) -> 'ScpClient':
        if self.dry_run:
            self._log(f'[dry-run] {self.host}:{self.port} 연결 생략')
            return self
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        s.connect((self.host, self.port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = s
        self._log(f'SCP 연결 {self.host}:{self.port}')
        return self

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *_):
        self.close()

    # ── 저수준 ────────────────────────────────────────────────────────────
    def send(self, payload: str, receiver: str = 'any') -> None:
        self._log(f'→ {payload}')
        if self.dry_run:
            return
        if self.sock is None:
            raise RuntimeError('connect() 를 먼저 호출할 것')
        self.sock.sendall(pack(payload, self.sender, receiver))

    def poll(self, seconds: float = 1.0):
        """받은 메시지들을 [(sender, receiver, payload)] 로. 응답 확인/디버깅용."""
        out = []
        if self.dry_run or self.sock is None:
            return out
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            self.sock.settimeout(max(0.05, end - time.monotonic()))
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                break
            except OSError:
                break
            if not chunk:
                break
            self._buf.extend(chunk)
            while True:
                msg = unpack(self._buf)
                if msg is None:
                    break
                out.append(msg)
        self.sock.settimeout(self.timeout)
        return out

    # ── 시뮬레이션 제어 ───────────────────────────────────────────────────
    def load_scenario(self, path: str) -> None:
        """시나리오 파일 로드. 이후 init() → start() 순서."""
        self.send(f'<SimCtrl><LoadScenario filename="{path}"/></SimCtrl>')

    def init(self) -> None:
        self.send('<SimCtrl><Init/></SimCtrl>')

    def start(self, t_max: float | None = None) -> None:
        tag = '<Start/>' if t_max is None else f'<Start tMax="{t_max:.3f}"/>'
        self.send(f'<SimCtrl>{tag}</SimCtrl>')

    def stop(self) -> None:
        self.send('<SimCtrl><Stop/></SimCtrl>')

    def pause(self) -> None:
        self.send('<SimCtrl><Pause/></SimCtrl>')

    def restart(self) -> None:
        self.send('<SimCtrl><Restart/></SimCtrl>')

    def step(self, n: int = 1) -> None:
        self.send(f'<SimCtrl><Step dt="{n}"/></SimCtrl>')

    def apply(self) -> None:
        self.send('<SimCtrl><Apply/></SimCtrl>')

    # ── 플레이어 ──────────────────────────────────────────────────────────
    def reset_player(self, player: str = 'Ego') -> None:
        """플레이어를 시나리오 초기 상태로 되돌린다 (회귀 테스트 사이 초기화용)."""
        self.send(f'<Traffic><ResetPlayer name="{player}"/></Traffic>')

    def set_player_pos(self, player: str, x: float, y: float, z: float = 0.0,
                       heading_rad: float = 0.0) -> None:
        """
        플레이어를 임의 좌표로 옮긴다.

        ⚠ **미검증**: VTD 2025.2 SCP 문서에서 "임의 위치로 순간이동" 전용 명령을
        찾지 못했다. PosInertial 은 문서상 signal/path 문맥에서 설명돼 있다.
        아래는 문서화된 태그·속성명을 조합한 **후보**이며, 실제 VTD 에서 먹는지
        확인이 필요하다(먹지 않으면 조용히 무시될 수 있다).

        확실히 동작하는 대안이 있다:
            tools/set_ego_start.py 로 시나리오 xml 의 초기 위치를 바꾼 뒤
            load_scenario() → init() → start()
        회귀 테스트에는 이 경로를 권한다.
        """
        h_deg = math.degrees(heading_rad)
        self.send(f'<Player name="{player}">'
                  f'<PosInertial x="{x:.5f}" y="{y:.5f}" z="{z:.5f}" hDeg="{h_deg:.5f}"/>'
                  f'</Player>')

    # ── 회귀 테스트용 조합 ────────────────────────────────────────────────
    def load_and_run(self, scenario: str, settle_s: float = 2.0,
                     t_max: float | None = None) -> list:
        """시나리오 로드 → Init → Start. 도중 받은 메시지를 돌려준다."""
        msgs = []
        self.load_scenario(scenario)
        msgs += self.poll(settle_s)
        self.init()
        msgs += self.poll(settle_s)
        self.start(t_max)
        msgs += self.poll(settle_s)
        return msgs

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f'[scp] {msg}', flush=True)


# ══════════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='VTD SCP 클라이언트')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=SCP_PORT)
    ap.add_argument('--dry-run', action='store_true', help='보내지 않고 명령만 출력')
    ap.add_argument('--wait', type=float, default=1.0, help='[s] 명령 후 응답 수신 시간')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('listen', help='들어오는 SCP 메시지만 구경 (송신 없음)')
    p.add_argument('--seconds', type=float, default=10.0)

    p = sub.add_parser('load', help='시나리오 로드')
    p.add_argument('scenario')
    p.add_argument('--run', action='store_true', help='로드 후 Init+Start 까지')

    sub.add_parser('init')
    p = sub.add_parser('start'); p.add_argument('--tmax', type=float, default=None)
    sub.add_parser('stop')
    sub.add_parser('pause')
    sub.add_parser('restart')
    p = sub.add_parser('step'); p.add_argument('-n', type=int, default=1)

    p = sub.add_parser('reset-player'); p.add_argument('--player', default='Ego')

    p = sub.add_parser('set-pos', help='플레이어 위치 설정 (미검증)')
    p.add_argument('--player', default='Ego')
    p.add_argument('x', type=float); p.add_argument('y', type=float)
    p.add_argument('--z', type=float, default=0.0)
    p.add_argument('--heading', type=float, default=0.0, help='[rad]')

    p = sub.add_parser('raw', help='임의 SCP XML 전송')
    p.add_argument('xml')

    a = ap.parse_args(argv)

    scp = ScpClient(a.host, a.port, dry_run=a.dry_run)
    try:
        scp.connect()
    except OSError as e:
        print(f'SCP 연결 실패 {a.host}:{a.port} — {e}\n'
              f'VTD 가 실행 중인지 확인할 것 (vtdStart.sh)', file=sys.stderr)
        return 1

    try:
        if a.cmd == 'listen':
            end = time.monotonic() + a.seconds
            n = 0
            while time.monotonic() < end:
                for snd, rcv, payload in scp.poll(1.0):
                    n += 1
                    print(f'  #{n} [{snd} → {rcv}] {payload[:200]}')
            print(f'수신 {n}건')
            return 0

        if a.cmd == 'load':
            if a.run:
                scp.load_and_run(a.scenario)
            else:
                scp.load_scenario(a.scenario)
        elif a.cmd == 'init':
            scp.init()
        elif a.cmd == 'start':
            scp.start(a.tmax)
        elif a.cmd == 'stop':
            scp.stop()
        elif a.cmd == 'pause':
            scp.pause()
        elif a.cmd == 'restart':
            scp.restart()
        elif a.cmd == 'step':
            scp.step(a.n)
        elif a.cmd == 'reset-player':
            scp.reset_player(a.player)
        elif a.cmd == 'set-pos':
            scp.set_player_pos(a.player, a.x, a.y, a.z, a.heading)
        elif a.cmd == 'raw':
            scp.send(a.xml)

        for snd, rcv, payload in scp.poll(a.wait):
            print(f'  ← [{snd}] {payload[:200]}')
    finally:
        scp.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
