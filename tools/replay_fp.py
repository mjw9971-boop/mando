#!/usr/bin/env python3
"""
replay 회귀 지문 — 로그를 전 스택으로 재생하고 **결정값을 값으로** 비교한다.

`run_agent.py --replay` 는 개루프다: 궤적은 로그에서 오고, 우리는 같은 입력에
같은 결정을 내는지만 본다. 그래서 리팩터·게이트 제거의 "동작 불변" 증명에 쓴다.

**줄 단위 해시는 쓰지 않는다.** 같은 설정으로 두 번 돌려도 해시가 갈린다
(2026-09-10 실측: 같은 로그·같은 설정 4회에서 sha256 이 매번 달랐다). 반면
틱별 (v_target, steering, accel) 값은 4회 전건 동일했다 — 즉 제어기는
결정적이고, 흔들리는 것은 **줄의 직렬화/순서**뿐이다. 그래서 값으로 비교한다.

사용:
    # 기준 뜨기
    python3 tools/replay_fp.py --out /tmp/fp_before
    # 코드를 고친 뒤
    python3 tools/replay_fp.py --out /tmp/fp_after --compare /tmp/fp_before

기본 목록은 tests/replay_quick.txt (대표 10건). route pkl 은 같은 배치의
routes/route_<이름>.pkl 을 자동으로 찾는다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIELDS = ('v_target', 'steering', 'accel')


def logs_from(list_path: pathlib.Path) -> list[pathlib.Path]:
    out = []
    for line in list_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        p = ROOT / line
        if p.exists():
            out.append(p)
        else:
            print(f'  없음: {line}', file=sys.stderr)
    return out


def route_for(log: pathlib.Path) -> pathlib.Path | None:
    p = log.parent / 'routes' / f'route_{log.stem}.pkl'
    return p if p.exists() else None


def decisions(path: pathlib.Path) -> list[tuple]:
    """틱별 (v_target, steering, accel). 설정 스냅샷 줄은 `"raw"` 가 없어 빠진다."""
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '"raw"' not in line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            c, de = d.get('cmd') or {}, d.get('decision') or {}
            out.append((de.get('v_target'), c.get('steering'), c.get('accel')))
    return out


def run(log: pathlib.Path, route: pathlib.Path, out: pathlib.Path) -> bool:
    r = subprocess.run(
        [sys.executable, 'run_agent.py', '--replay', str(log),
         '--route', str(route), '--log', str(out)],
        cwd=str(ROOT), capture_output=True, text=True)
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--list', default='tests/replay_quick.txt')
    ap.add_argument('--out', required=True, help='재생 산출을 쓸 디렉터리')
    ap.add_argument('--compare', default=None, help='이 디렉터리와 값으로 대조')
    a = ap.parse_args()

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    logs = logs_from(ROOT / a.list)
    if not logs:
        print('재생할 로그가 없다', file=sys.stderr)
        return 2

    print(f"{'로그':38} {'틱':>7}" + (f" {'불일치':>7}  결과" if a.compare else ''))
    total = mismatch = 0
    missing = []
    for log in logs:
        route = route_for(log)
        if route is None:
            print(f'{log.stem[:38]:38} route pkl 없음')
            continue
        dst = out_dir / f'{log.stem}.jsonl'
        if not run(log, route, dst):
            print(f'{log.stem[:38]:38} 재생 실패')
            return 1
        cur = decisions(dst)
        total += len(cur)
        if not a.compare:
            print(f'{log.stem[:38]:38} {len(cur):7}')
            continue
        ref_path = pathlib.Path(a.compare) / f'{log.stem}.jsonl'
        if not ref_path.exists():
            missing.append(log.stem)
            print(f'{log.stem[:38]:38} {len(cur):7} {"—":>7}  기준 없음')
            continue
        ref = decisions(ref_path)
        n = min(len(ref), len(cur))
        d = sum(1 for i in range(n) if ref[i] != cur[i]) + abs(len(ref) - len(cur))
        mismatch += d
        print(f'{log.stem[:38]:38} {len(cur):7} {d:7}  '
              + ('동일' if d == 0 else '**차이**'))

    print(f'\n총 {total:,} 틱')
    if a.compare:
        print(f'총 불일치 {mismatch:,} 틱'
              + ('  → 결정 불변' if mismatch == 0 else '  → **동작이 바뀌었다**'))
        if missing:
            print(f'기준 없음 {len(missing)}건: {", ".join(missing)}')
        return 0 if mismatch == 0 and not missing else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
