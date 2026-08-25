"""
9910 덤프/디코드 도구.

SPEC §1.5 는 이 파일을 "기존 자산"으로 적고 있으나 저장소에 없어서 새로 썼다.
프레임 규격(§1.1/§1.2)이 확정돼 있으므로 src/comm.py 의 파서를 그대로 재사용한다.

    python3 tools/probe_9910.py --host 127.0.0.1        # 실시간 디코드 출력
    python3 tools/probe_9910.py --host ... --raw out.bin # 원본 바이트도 저장
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from vtd_adapter.comm import FRAME_SIZE, parse  # noqa: E402

STATE_NAME = {0: '미할당', 1: '적', 2: '황', 3: '녹', 4: '좌회전', 5: '녹+좌', 6: '점멸'}


def decode(frame: bytes):
    """1109 B → RawPacket. vtd_adapter.comm.parse 와 동일."""
    return parse(frame)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='9910 프레임 덤프')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=9910)
    ap.add_argument('--raw', default=None, help='원본 바이트를 저장할 파일')
    ap.add_argument('--seconds', type=float, default=0.0, help='0 = 무제한')
    args = ap.parse_args(argv)

    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((args.host, args.port))
    print(f'연결됨 {args.host}:{args.port}  (frame={FRAME_SIZE} B)')

    raw_f = open(args.raw, 'wb') if args.raw else None
    buf = bytearray()
    t0 = time.monotonic()
    n = 0
    try:
        while True:
            if args.seconds and time.monotonic() - t0 > args.seconds:
                break
            chunk = s.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
            if raw_f:
                raw_f.write(chunk)
            while len(buf) >= FRAME_SIZE:
                frame = bytes(buf[:FRAME_SIZE])
                del buf[:FRAME_SIZE]
                p = decode(frame)
                n += 1
                if n % 20 == 1:                        # 20 Hz → 1초에 한 줄
                    x, y, _z, h = p.ego[0], p.ego[1], p.ego[2], p.ego[3]
                    lights = ', '.join(f'{i}:{STATE_NAME.get(st, st)}' for i, st in p.lights)
                    # TODO(SPEC §7-2): 거리 컷오프 확인용 — 최원거리 객체를 같이 찍는다
                    far = max((((o[1] - x) ** 2 + (o[2] - y) ** 2) ** 0.5 for o in p.objects),
                              default=0.0)
                    print(f'#{n:6d} ego=({x:9.2f},{y:9.2f}) hdg={h:+.3f} '
                          f'obj={len(p.objects):2d} 최원={far:6.1f}m 신호=[{lights}]')
    except KeyboardInterrupt:
        pass
    finally:
        s.close()
        if raw_f:
            raw_f.close()
    print(f'총 {n} 프레임')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
