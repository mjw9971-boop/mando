"""
최소 주행 루프용 임시 경로 생성기.

주최 경유점을 받기 전까지 "일단 굴러가는지" 확인하려면 route.pkl 이 필요하다.
시작점에서 lane_graph 의 successors 를 따라 약 1.5 km 걸어가며 중심점을 뽑아
waypoints csv 를 만들고, tools/build_route.py 를 호출해 data/route.pkl 을 만든다.

    python3 tools/make_test_route.py
    python3 tools/make_test_route.py --length 3000 -o data/route.pkl

분기에서는 아무 successor 나 하나 고르되, **막다른 차로(next 없음)는 피한다.**
이미 지나온 차로도 피해서 제자리를 돌지 않게 한다.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src' / 'hlfma'))

from hlfma.core.lanegraph import LaneGraph  # noqa: E402

# SPEC §1.1 검증 완료된 시나리오 초기 위치
START_X, START_Y = 508.79968, -168.28766
START_YAW = 0.52727822


def pick_next(lg: LaneGraph, lane, visited: set):
    """
    다음 차로 하나 고르기.

    우선순위: 안 가본 + 그 다음이 또 있는 차로 > 안 가본 차로 > 없으면 None.
    막다른 차로로 들어가면 거기서 경로가 끝나버리므로 뒤가 있는 쪽을 먼저 본다.
    """
    nxts = [k for k in lg.successors(lane) if k in lg.lanes]
    fresh = [k for k in nxts if k not in visited]
    pool = fresh or nxts
    if not pool:
        return None
    with_future = [k for k in pool if lg.successors(k)]
    return (with_future or pool)[0]


def walk(lg: LaneGraph, start_lane, start_s: float, target_len: float, wp_step: float):
    """
    successors 를 따라 target_len 만큼 진행하며 wp_step 마다 중심점을 뽑는다.
    반환: (waypoints[(x,y)], 실제 진행 거리)
    """
    wps: list[tuple[float, float]] = []
    lane, s = start_lane, start_s
    visited = {lane}
    travelled = 0.0
    next_wp_at = 0.0

    while travelled < target_len:
        L = lg.length(lane)
        # 이 차로 안에서 wp_step 간격으로 점 뽑기
        while next_wp_at <= travelled + (L - s) and next_wp_at <= target_len:
            s_at = s + (next_wp_at - travelled)
            x, y, _z, _h = lg.point_at(lane, min(s_at, L))
            wps.append((x, y))
            next_wp_at += wp_step

        travelled += L - s
        nxt = pick_next(lg, lane, visited)
        if nxt is None:
            print(f'  막다른 차로 {lane} 에서 종료 ({travelled:.0f} m)')
            break
        lane, s = nxt, 0.0
        visited.add(lane)

    # 마지막 지점은 반드시 포함 (경로 끝을 명확히)
    x, y, _z, _h = lg.point_at(lane, min(s + max(0.0, target_len - travelled), lg.length(lane)))
    if not wps or (abs(wps[-1][0] - x) + abs(wps[-1][1] - y)) > 1.0:
        wps.append((x, y))
    return wps, travelled


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='임시 테스트 경로 생성')
    ap.add_argument('--graph', default=str(ROOT / 'data' / 'lane_graph.pkl'))
    ap.add_argument('-o', '--out', default=str(ROOT / 'data' / 'route.pkl'))
    ap.add_argument('--csv', default=str(ROOT / 'data' / 'test_route_waypoints.csv'))
    ap.add_argument('--length', type=float, default=1500.0, help='[m] 목표 경로 길이')
    ap.add_argument('--wp-step', type=float, default=100.0, help='[m] 경유점 간격')
    ap.add_argument('--plot', default=str(ROOT / 'docs' / 'test_route.png'))
    ap.add_argument('--no-plot', action='store_true')
    args = ap.parse_args(argv)

    lg = LaneGraph(args.graph)
    m = lg.locate(START_X, START_Y, yaw=START_YAW)
    if m is None:
        print('시작점에서 차로를 찾지 못했다', file=sys.stderr)
        return 1
    print(f'시작: lane={m.lane} s={m.s:.2f} t={m.t:+.3f} (중심선 {m.dist:.3f} m)')

    wps, travelled = walk(lg, m.lane, m.s, args.length, args.wp_step)
    print(f'경유점 {len(wps)}개 / 진행 {travelled:.0f} m')
    if len(wps) < 2:
        print('경유점이 부족하다', file=sys.stderr)
        return 1

    pathlib.Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, 'w', encoding='utf-8') as f:
        f.write('# x,y  (make_test_route.py 자동 생성 — 주최 경유점 받으면 교체)\n')
        for x, y in wps:
            f.write(f'{x:.4f},{y:.4f}\n')
    print(f'저장: {args.csv}')

    # build_route.py 호출 (경로 규격은 그쪽이 단일 출처)
    # 여기 경유점은 successors 를 따라 균등 간격으로 뽑은 것이지 대회 공식 형식의
    # 교차로 진입·진출 짝이 아니다. 짝으로 해석하면 엉뚱한 제약이 걸린다.
    cmd = [sys.executable, str(ROOT / 'tools' / 'build_route.py'), args.graph, args.csv,
           '-o', args.out, '--start-yaw', repr(START_YAW), '--no-pairs']
    print('$', ' '.join(cmd))
    if subprocess.call(cmd) != 0:
        return 1

    if not args.no_plot:
        pathlib.Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        pcmd = [sys.executable, str(ROOT / 'tools' / 'plot_lane_graph.py'), args.graph,
                '--route', args.out, '-o', args.plot]
        print('$', ' '.join(pcmd))
        subprocess.call(pcmd)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
