#!/usr/bin/env python3
"""
gen_odd_waypoints.py ─ 홀수 경유점 CSV 표본 생성 (작업22-1, 분석 전용)

주최 공식 형식은 `[시작, (진입,진출)×N, 종료]` = **짝수**지만 당일 홀수로 올 수
있다고 확인됐다. `build_route --pair-offset auto` 의 홀수 판정(작업21-2)을 채점
하려면 **정답 offset 을 아는 홀수 표본**이 있어야 하는데 실배포본에는 없다.
그래서 짝수 원본에서 점 하나를 빼 만든다 — 어느 점을 뺐는지가 곧 정답이다.

    유형 B  첫 점 제거   [(진입,진출)×N, 종료]   정답 offset 0
    유형 C  끝 점 제거   [시작, (진입,진출)×N]   정답 offset 1
    유형 D  중간 점 제거  짝 구조 파괴            정답 none (판정 불가)

유형 D 는 "빼도 짝처럼 보이는" 경우를 일부러 포함한다. 예를 들어 6점
[S,I1,O1,I2,O2,E] 에서 I1 을 빼면 [S,O1,I2,O2,E] 가 되는데, offset 0 해석의
첫 짝 (S→O1) 이 교차로를 실제로 지나므로 유형 B 와 신호가 구분되지 않는다.
이런 표본이 자동 판정의 상한을 정한다.

    python3 tools/gen_odd_waypoints.py --out tests/fixtures/odd

원본은 **짝 구간이 2개 이상**(= 6점 이상 짝수)인 CSV 전부다. 4점(짝 1개)은
빼도 짝이 하나뿐이라 비율이 0 또는 1 로만 나와 판별력을 잴 수 없다.
"""
import argparse
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from build_route import read_waypoints_csv, junction_segments   # noqa: E402

# 원본 후보 — 저장소 안의 경유점 CSV 전부에서 조건에 맞는 것만 고른다.
SEARCH_DIRS = ('data', 'tests/fixtures', 'scenarios', '.')
SKIP_PARTS = ('logs', '.git', 'templates')


def find_originals(root: str):
    """짝수 & 6점 이상(짝 구간 2개 이상)인 경유점 CSV 경로 목록."""
    out = []
    for d in SEARCH_DIRS:
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            dns[:] = [x for x in dns if x not in SKIP_PARTS and not x.startswith('.')]
            if any(p in dp.split(os.sep) for p in SKIP_PARTS):
                continue
            for fn in fns:
                if not fn.endswith('.csv'):
                    continue
                p = os.path.normpath(os.path.join(dp, fn))
                if p in out:
                    continue
                try:
                    rows = read_waypoints_csv(p)
                except Exception:                       # noqa: BLE001 — 경유점 CSV 가 아니다
                    continue
                n = len(rows)
                if n >= 6 and n % 2 == 0:
                    out.append(p)
    return sorted(set(out))


def write_csv(path: str, rows, renumber: bool) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')   # 원본 CSV 와 같은 LF
        w.writerow(['seq', 'x', 'y'])
        for i, (seq, x, y) in enumerate(rows, 1):
            w.writerow([i if renumber else seq, f'{x:.3f}', f'{y:.3f}'])


def samples(rows, drop_middle_max: int):
    """(유형, 뺀 인덱스, 정답 offset, 남은 행) 목록."""
    n = len(rows)
    out = [('B', 0, 0, rows[1:]),
           ('C', n - 1, 1, rows[:-1])]
    # 유형 D — 중간 점. 앞뒤 균형 있게 고른다 (첫 진입, 첫 진출, 마지막 진출).
    mids = [1, 2, n - 2]
    seen = set()
    for k in mids:
        if 1 <= k <= n - 2 and k not in seen:
            seen.add(k)
            out.append(('D', k, None, rows[:k] + rows[k + 1:]))
        if len(seen) >= drop_middle_max:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description='홀수 경유점 CSV 표본 생성 (작업22-1)')
    ap.add_argument('--out', default=os.path.join(ROOT, 'tests/fixtures/odd'),
                    help='표본을 쓸 디렉터리')
    ap.add_argument('--root', default=ROOT, help='원본 CSV 를 찾을 저장소 루트')
    ap.add_argument('--csv', nargs='*', default=None,
                    help='원본을 직접 지정 (없으면 자동 탐색)')
    ap.add_argument('--drop-middle', type=int, default=3,
                    help='원본당 유형 D 표본 수 (기본 3)')
    ap.add_argument('--no-renumber', action='store_true',
                    help='seq 열을 원본 값 그대로 둔다 (기본은 1..n 재번호)')
    a = ap.parse_args()

    originals = a.csv if a.csv else find_originals(a.root)
    if not originals:
        print('원본 CSV 를 못 찾았다', file=sys.stderr)
        return 1

    manifest = []
    for src in originals:
        rows = read_waypoints_csv(src)
        n = len(rows)
        # 같은 basename 이 여러 디렉터리에 있다 (waypoints.csv 가 루트와
        # tests/fixtures 양쪽에 있다) — 부모 디렉터리를 붙여 표본 이름을 가른다.
        stem = os.path.splitext(os.path.basename(src))[0]
        parent = os.path.basename(os.path.dirname(os.path.abspath(src)))
        if parent not in ('', os.path.basename(os.path.abspath(a.root))):
            stem = f'{parent}-{stem}'
        for kind, drop_i, want, kept in samples(rows, a.drop_middle):
            tag = kind if kind != 'D' else f'D{drop_i}'
            name = f'{stem}__{tag}.csv'
            path = os.path.join(a.out, name)
            write_csv(path, kept, not a.no_renumber)
            manifest.append({
                'sample': name,
                'path': os.path.relpath(path, a.root),
                'source': os.path.relpath(src, a.root),
                'kind': kind,
                'dropped_index': drop_i,
                'n_src': n,
                'n': len(kept),
                'expect_offset': want,          # None = 판정 불가가 정답
                'expect_pairs': (sorted(junction_segments(len(kept), want))
                                 if want is not None else None),
            })

    mpath = os.path.join(a.out, 'manifest.json')
    with open(mpath, 'w') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    kinds = {}
    for m in manifest:
        kinds[m['kind']] = kinds.get(m['kind'], 0) + 1
    print(f'원본 {len(originals)}개 → 표본 {len(manifest)}개 '
          f'({", ".join(f"{k} {v}" for k, v in sorted(kinds.items()))})')
    print(f'manifest {mpath}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
