"""
경유점 CSV 를 진행 방향의 수직(횡)으로 흔든 CSV 를 만든다.

대회 배포 CSV 의 좌표가 차로 중앙에서 조금 어긋나 있어도 build_route 가 같은
차로를 고르는지 미리 확인하기 위한 것. 가장 위험한 실패는 에러 없이 조용히 옆
차로로 경로가 잡히는 것이라, 흔든 CSV → build_route → 차로열 비교로 잡는다.

    python3 tools/jitter_csv.py waypoints.csv -o /tmp/w_L03.csv --offset 0.3
    python3 tools/jitter_csv.py waypoints.csv -o /tmp/w_R03.csv --offset -0.3

부호는 진행 방향 기준 **좌(+) / 우(−)** 다 (heading (dx,dy) 의 좌법선 (−dy,dx)).
횡방향은 앞뒤 경유점을 잇는 방향으로 잡고, 첫 점·끝점은 인접 구간 방향을 쓴다.

읽기 전용 도구다 — 입력 CSV 는 건드리지 않고 --out 에만 쓴다.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'tools'))

from build_route import read_waypoints_csv       # noqa: E402 — CSV 해석 단일 출처


def headings(pts: list[tuple[float, float]]) -> list[float]:
    """각 점의 진행 방향 [rad].

    중간 점은 앞뒤 점을 잇는 방향(i-1 → i+1), 첫 점은 0→1, 끝점은 n-2→n-1.
    앞뒤 점이 겹쳐 방향이 0 이면 가장 가까운 유효 구간 방향으로 대신한다
    (경유점이 0.1 m 간격으로 붙어 있으면 방향이 노이즈에 지배되기 때문).
    """
    n = len(pts)
    if n < 2:
        raise SystemExit('경유점이 2개 미만이다')

    def seg(i: int, j: int):
        dx, dy = pts[j][0] - pts[i][0], pts[j][1] - pts[i][1]
        return (dx, dy) if math.hypot(dx, dy) > 1e-9 else None

    out = []
    for i in range(n):
        if i == 0:
            cands = [seg(0, j) for j in range(1, n)]
        elif i == n - 1:
            cands = [seg(j, n - 1) for j in range(n - 2, -1, -1)]
        else:
            cands = [seg(i - 1, i + 1), seg(i, i + 1), seg(i - 1, i)]
        d = next((c for c in cands if c is not None), None)
        if d is None:
            raise SystemExit(f'경유점 {i}: 모든 이웃이 같은 좌표라 방향을 못 잡는다')
        out.append(math.atan2(d[1], d[0]))
    return out


def jitter(rows: list, offset_m: float) -> list[tuple]:
    """[(seq, x, y)] → 좌법선으로 offset_m 만큼 민 [(seq, x, y)]."""
    pts = [(r[1], r[2]) for r in rows]
    out = []
    for (seq, x, y), h in zip(rows, headings(pts)):
        # 좌법선 = heading 을 +90° 돌린 방향. offset>0 이면 진행 방향 왼쪽.
        out.append((seq, x + offset_m * -math.sin(h), y + offset_m * math.cos(h)))
    return out


def write_csv(path: str, rows: list[tuple]) -> None:
    body = 'seq,x,y\n' + ''.join(f'{int(s)},{x:.3f},{y:.3f}\n' for s, x, y in rows)
    pathlib.Path(path).write_text(body, encoding='utf-8')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='경유점 CSV 를 횡방향으로 흔든다')
    ap.add_argument('csv', help='입력 경유점 CSV (seq,x,y — 읽기만 한다)')
    ap.add_argument('-o', '--out', required=True, help='출력 CSV 경로')
    ap.add_argument('--offset', type=float, required=True,
                    help='[m] 횡 이동량. 진행 방향 기준 양수=좌 / 음수=우')
    a = ap.parse_args(argv)

    if pathlib.Path(a.out).resolve() == pathlib.Path(a.csv).resolve():
        raise SystemExit('출력이 입력과 같은 경로다 — 원본을 덮어쓸 수 없다')

    rows = read_waypoints_csv(a.csv)
    out = jitter(rows, a.offset)
    write_csv(a.out, out)
    side = '좌' if a.offset > 0 else '우'
    print(f'{a.csv} → {a.out}  ({len(out)}점, {side} {abs(a.offset):g} m)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
