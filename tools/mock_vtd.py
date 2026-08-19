"""
VTD 대역 목 서버 — ROS 없이도 제어 루프 전체를 돌려보기 위한 것.

진짜 대회 브릿지와 같은 자리에 선다.
  - TCP 9910 서버: 1109 B 상태를 20 Hz 로 보내고, 9 B 제어를 받는다
  - 받은 제어를 자전거 모델에 넣어 자차를 움직인다 (닫힌 루프)

    python3 tools/mock_vtd.py                       # 기본 9910
    python3 tools/mock_vtd.py --port 19910 --seconds 30

[조향 부호]
  실제 VTD 는 우리가 보낸 조향을 그대로 적용한다(실측 2026-08-19).
  이 목도 같은 규약이라 기본값이 +1.0 이고, **받은 값을 그대로** 쓴다.

  예전에는 기본값이 -1.0 이라 목이 부호를 한 번 더 뒤집었다. 그 탓에 config 의
  comm.steer_sign 이 -1.0(틀린 값)이어도 목에서는 두 번 뒤집혀 정상 주행했고,
  실차에서만 발산하는 문제를 가려버렸다. 목 테스트가 실차 부호를 검증하려면
  이 값은 실제 VTD 와 같아야 한다.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import socket
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'src' / 'hlfma'))

from hlfma.core.comm import CTRL_FMT, CTRL_SIZE, build_frame  # noqa: E402

WHEELBASE = 2.944
MAX_STEER = 0.48

# SPEC §1.1 검증 완료된 시나리오 초기 위치
START_X, START_Y, START_YAW = 508.79968, -168.28766, 0.52727822


class MockVehicle:
    """자전거 모델. 제어 입력이 실제로 차를 움직이는지만 본다."""

    def __init__(self, x: float, y: float, yaw: float) -> None:
        self.x, self.y, self.yaw, self.v = x, y, yaw, 0.0
        self.steer = self.accel = 0.0
        self.signal = 0

    def step(self, dt: float) -> None:
        self.v = max(0.0, self.v + self.accel * dt)
        if self.v > 1e-3:
            self.yaw += self.v / WHEELBASE * math.tan(
                max(-MAX_STEER, min(MAX_STEER, self.steer))) * dt
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt


# 신호 주기: (state, 지속시간[s]).  SPEC §1.1 state 코드
LIGHT_CYCLE = [(1, 8.0),    # 적색
               (3, 10.0),   # 녹색
               (2, 3.0)]    # 황색
CYCLE_LEN = sum(d for _, d in LIGHT_CYCLE)


def _light_at(a: argparse.Namespace, t: float) -> tuple[int, int]:
    """
    현재 신호등 (id, state).

    --lights 를 안 주면 (0, 0) — 실제 브릿지가 "지금 볼 신호등 없음" 을 그렇게 보낸다.
    주면 적→녹→황을 돌리고, --light-switch-s 마다 id 를 바꿔 교차로 전환을 흉내낸다
    (SPEC §7-1: id 가 교차로마다 바뀌는지 미확인 → 수신측 관측 로직을 시험한다).
    """
    if not a.lights:
        return (0, 0)
    phase = t % CYCLE_LEN
    state = LIGHT_CYCLE[-1][0]
    acc = 0.0
    for st, dur in LIGHT_CYCLE:
        acc += dur
        if phase < acc:
            state = st
            break
    lid = a.light_id
    if a.light_switch_s > 0:
        lid += int(t // a.light_switch_s)
    return (lid, state)


def serve(a: argparse.Namespace) -> int:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((a.bind, a.port))
    srv.listen(1)
    print(f'[mock] TCP {a.bind}:{a.port} 대기 (상태 1109 B @{a.hz} Hz / 제어 9 B)', flush=True)

    conn, addr = srv.accept()
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.setblocking(False)
    print(f'[mock] 제어기 접속: {addr}', flush=True)

    veh = MockVehicle(a.start_x, a.start_y, a.start_yaw)
    dt = 1.0 / a.hz
    buf = bytearray()
    t0 = time.monotonic()
    n = 0

    try:
        while a.seconds <= 0 or time.monotonic() - t0 < a.seconds:
            try:
                while True:
                    chunk = conn.recv(4096)
                    if not chunk:
                        print('[mock] 제어기가 연결을 닫았다', flush=True)
                        return 0
                    buf.extend(chunk)
            except BlockingIOError:
                pass

            while len(buf) >= CTRL_SIZE:
                steer_in, veh.accel, veh.signal = CTRL_FMT.unpack_from(buf, 0)
                del buf[:CTRL_SIZE]
                veh.steer = a.steer_sign * steer_in

            veh.step(dt)
            conn.sendall(build_frame(
                (veh.x, veh.y, 42.0, veh.yaw, 0.0, 0.0), [],
                [_light_at(a, time.monotonic() - t0)]))

            n += 1
            if n % (a.hz * a.print_every) == 0:
                print(f'[mock] t={time.monotonic() - t0:6.1f}s '
                      f'pos=({veh.x:8.2f},{veh.y:8.2f}) yaw={math.degrees(veh.yaw):+7.1f}° '
                      f'v={veh.v:5.2f} m/s  steer={veh.steer:+.3f} accel={veh.accel:+.2f} '
                      f'sig={veh.signal}', flush=True)
            time.sleep(dt)
    except (KeyboardInterrupt, ConnectionResetError, BrokenPipeError) as e:
        print(f'[mock] 종료: {type(e).__name__}', flush=True)
    finally:
        conn.close()
        srv.close()
    print(f'[mock] {n} 프레임 송신', flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='HL FMA 대회 브릿지 목 서버')
    ap.add_argument('--bind', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=9910)
    ap.add_argument('--hz', type=int, default=20)
    ap.add_argument('--seconds', type=float, default=0.0, help='0 = 무제한')
    ap.add_argument('--steer-sign', type=float, default=1.0,
                    help='받은 조향에 곱할 부호. 기본 +1.0 = 실제 VTD 와 같은 규약 '
                         '(받은 값을 그대로 적용). 바꾸지 말 것 — 목이 부호를 되돌리면 '
                         'steer_sign 오류를 가려버린다')
    ap.add_argument('--start-x', type=float, default=START_X)
    ap.add_argument('--start-y', type=float, default=START_Y)
    ap.add_argument('--start-yaw', type=float, default=START_YAW)
    ap.add_argument('--print-every', type=int, default=5, help='[s] 상태 출력 주기')
    ap.add_argument('--lights', action='store_true',
                    help='신호등을 적→녹→황으로 돌린다 (기본: 신호등 없음 = (0,0))')
    ap.add_argument('--light-id', type=int, default=3, help='시작 신호등 id')
    ap.add_argument('--light-switch-s', type=float, default=0.0,
                    help='[s] 이 주기마다 신호등 id 를 바꾼다 (0 = 고정)')
    return serve(ap.parse_args(argv))


if __name__ == '__main__':
    raise SystemExit(main())
