"""
시나리오 xml 의 Ego 초기 위치를 경유점 CSV 의 seq1 로 맞춘다.

    python3 tools/set_ego_start.py waypoints.csv \
            --scenario ~/VIRES/VTD.2025.2/Data/Projects/SampleProject/Scenarios/HL_FMA_VTD_LivingLab.xml
    python3 tools/set_ego_start.py waypoints.csv --scenario ... --dry-run

경로를 새로 받으면 시나리오의 출발 위치도 같이 옮겨야 한다. 손으로 고치면
숫자 하나 틀려도 차가 도로 밖에서 시작하므로, 차로 위인지 검증한 뒤에만 쓴다.

[xml 속성 이름]
  TrafficControl/Player/Init/PosAbsolute 의 X, Y, Z, Direction 이다.
  (PosX/PosY/Heading 이 아니다. Direction 이 헤딩 [rad])
"""
from __future__ import annotations

import argparse
import math
import pathlib
import re
import shutil
import sys
import time
import xml.etree.ElementTree as ET

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))                                   # build_route
sys.path.insert(0, str(ROOT.parent / 'src' / 'hlfma'))          # hlfma.core

from build_route import RouteError, read_waypoints_csv          # noqa: E402
from hlfma.core.lanegraph import LaneGraph                      # noqa: E402

NUM = r'[-+0-9.eE]+'


def fmt(v: float) -> str:
    """VTD 가 쓰는 표기에 맞춘다."""
    return f'{v:.16e}'


def read_current(scenario: str) -> dict:
    """현재 Ego 초기 포즈."""
    root = ET.parse(scenario).getroot()
    pos = root.find('.//TrafficControl/Player/Init/PosAbsolute')
    if pos is None:
        raise RouteError(f'{scenario}: TrafficControl/Player/Init/PosAbsolute 를 찾지 못했다')
    name = root.findtext('.//TrafficControl/Player/Description') or ''
    desc = root.find('.//TrafficControl/Player/Description')
    return {
        'name': desc.get('Name') if desc is not None else '?',
        'control': desc.get('Control') if desc is not None else '?',
        'x': float(pos.get('X')), 'y': float(pos.get('Y')), 'z': float(pos.get('Z')),
        'dir': float(pos.get('Direction')),
        'align': pos.get('AlignToRoad'),
    }


def replace_pos(text: str, x: float, y: float, z: float, direction: float) -> str:
    """
    PosAbsolute 의 X/Y/Z/Direction 만 바꾼다.

    ElementTree 로 다시 쓰면 397 KB 짜리 파일 전체가 재포맷돼 diff 를 볼 수 없다.
    파일에 PosAbsolute 가 하나뿐인 것을 확인하고 텍스트로 치환한다.
    """
    hits = re.findall(r'<PosAbsolute\b[^>]*/>', text)
    if len(hits) != 1:
        raise RouteError(f'PosAbsolute 가 {len(hits)}개다 (1개를 기대). 수동 확인 필요')
    old = hits[0]
    new = old
    for attr, val in (('X', x), ('Y', y), ('Z', z), ('Direction', direction)):
        pat = rf'({attr}=")({NUM})(")'
        if not re.search(pat, new):
            raise RouteError(f'PosAbsolute 에 {attr} 속성이 없다: {old}')
        new = re.sub(pat, lambda m, v=val: f'{m.group(1)}{fmt(v)}{m.group(3)}', new, count=1)
    return text.replace(old, new, 1)


def backup(path: pathlib.Path) -> pathlib.Path:
    """
    .bak 이 이미 있으면 **덮지 않는다** — 그게 원본일 수 있다.
    대신 타임스탬프 백업을 새로 만든다.
    """
    plain = path.with_suffix(path.suffix + '.bak')
    if not plain.exists():
        shutil.copy2(path, plain)
        return plain
    stamped = path.with_suffix(path.suffix + f".{time.strftime('%Y%m%d_%H%M%S')}.bak")
    shutil.copy2(path, stamped)
    return stamped


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='시나리오 Ego 초기 위치를 seq1 로 설정')
    ap.add_argument('waypoints', help='csv: seq,x,y')
    ap.add_argument('--scenario', required=True, help='시나리오 xml 경로')
    ap.add_argument('--graph', default=str(ROOT.parent / 'data' / 'lane_graph.pkl'))
    ap.add_argument('--radius', type=float, default=3.0,
                    help='[m] 이 거리 안에 차로가 없으면 중단')
    ap.add_argument('--yaw-min-dist', type=float, default=2.0,
                    help='[m] 헤딩 계산에 쓸 최소 거리')
    ap.add_argument('--heading', type=float, default=None,
                    help='[rad] 헤딩을 직접 지정 (seq1→seq2 추정 대신)')
    ap.add_argument('--keep-z', action='store_true',
                    help='Z 를 그대로 둔다 (기본: lane_graph 의 노면 높이로 갱신)')
    ap.add_argument('--dry-run', action='store_true', help='파일을 쓰지 않는다')
    a = ap.parse_args(argv)

    scenario = pathlib.Path(a.scenario).expanduser()
    if not scenario.exists():
        print(f'시나리오 파일이 없다: {scenario}', file=sys.stderr)
        return 1

    rows = read_waypoints_csv(a.waypoints)
    if len(rows) < 2 and a.heading is None:
        raise RouteError('경유점이 2개 미만이면 헤딩을 추정할 수 없다 (--heading 을 주면 된다)')
    seq0, x, y = rows[0]

    # ── 헤딩 ─────────────────────────────────────────────────────────────
    if a.heading is not None:
        yaw, yaw_src = a.heading, '--heading 지정'
    else:
        ref = None
        for j in range(1, len(rows)):
            d = math.hypot(rows[j][1] - x, rows[j][2] - y)
            if d >= a.yaw_min_dist:
                ref = (j, d)
                break
        if ref is None:
            ref = (1, math.hypot(rows[1][1] - x, rows[1][2] - y))
            print(f'  [경고] 모든 경유점이 {a.yaw_min_dist:g} m 이내다. 헤딩이 부정확할 수 있다',
                  file=sys.stderr)
        j, d = ref
        yaw = math.atan2(rows[j][2] - y, rows[j][1] - x)
        yaw_src = f'seq {seq0}→{rows[j][0]} 방향 ({d:.1f} m)'

    # ── 차로 위인지 검증 ─────────────────────────────────────────────────
    lg = LaneGraph(a.graph)
    m = lg.locate(x, y, yaw=yaw, max_dist=a.radius)
    if m is None:
        d, _ = lg.kd.query((x, y), k=1)
        print(f'\n[중단] seq {seq0} ({x:.3f},{y:.3f}) 이 차로 위가 아니다.\n'
              f'       반경 {a.radius:g} m 내 진행방향이 맞는 차로 없음 '
              f'(최근접 차로점 {float(d):.2f} m).\n'
              f'       좌표계가 다르거나 헤딩이 반대일 수 있다. '
              f'--heading 으로 직접 지정해 볼 것.', file=sys.stderr)
        return 2

    zx, zy, zz, zh = lg.point_at(m.lane, m.s)
    cur = read_current(str(scenario))
    new_z = cur['z'] if a.keep_z else zz

    # ── 출력 ─────────────────────────────────────────────────────────────
    print(f'시나리오 : {scenario}')
    print(f'플레이어 : {cur["name"]} (Control={cur["control"]}, AlignToRoad={cur["align"]})')
    print(f'경유점   : {a.waypoints}  seq {seq0}')
    print(f'헤딩 근거: {yaw_src}')
    print(f'차로 검증: lane={m.lane}  s={m.s:.2f} m  중심선까지 {m.dist:.3f} m  '
          f'헤딩오차 {math.degrees(m.heading_err):+.2f}°')
    print()
    print(f'{"":9} {"전":>22} {"후":>22}')
    for label, o, n in (('X', cur['x'], x), ('Y', cur['y'], y),
                        ('Z', cur['z'], new_z), ('Direction', cur['dir'], yaw)):
        print(f'  {label:8} {o:22.6f} {n:22.6f}')
    print(f'  {"(deg)":8} {math.degrees(cur["dir"]):22.2f} {math.degrees(yaw):22.2f}')
    moved = math.hypot(x - cur['x'], y - cur['y'])
    print(f'\n  이동 거리 {moved:.1f} m')

    if a.dry_run:
        print('\n[dry-run] 파일을 쓰지 않았다')
        return 0

    bak = backup(scenario)
    text = scenario.read_text(encoding='utf-8')
    scenario.write_text(replace_pos(text, x, y, new_z, yaw), encoding='utf-8')
    print(f'\n백업 : {bak.name}')
    print(f'수정 : {scenario.name}')

    after = read_current(str(scenario))
    ok = (abs(after['x'] - x) < 1e-6 and abs(after['y'] - y) < 1e-6
          and abs(after['dir'] - yaw) < 1e-9)
    print('검증 : ' + ('OK' if ok else '[경고] 다시 읽은 값이 다르다 — 확인 필요'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
