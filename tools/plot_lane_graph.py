#!/usr/bin/env python3
"""
plot_lane_graph.py ─ lane_graph.pkl 을 그림으로 확인

    python3 plot_lane_graph.py lane_graph.pkl -o map.png                       # 전체
    python3 plot_lane_graph.py lane_graph.pkl -o zoom.png --center 15 42 --radius 120   # 특정 지점 확대
    python3 plot_lane_graph.py lane_graph.pkl -o route.png --route route.pkl   # 경로 오버레이

색:
    회색 = driving 차로 중심선 (방향 화살표 조금)
    파랑 = 제한속도 30, 주황 = 스쿨존(RM_518), 초록 = 제한속도 50
    빨강 짧은 선 = 정지선, 빨강 점 = 신호등 걸린 정지선
    보라 = 횡단보도(pelican), 하늘색 점 = 횡단보도 예고 다이아몬드
    굵은 검정 = 경로, 별 = 경유점
"""
import argparse, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def travel_pt(rec, s):
    return np.array([np.interp(s, rec['s'], rec['pts'][:, 0]), np.interp(s, rec['s'], rec['pts'][:, 1])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pkl')
    ap.add_argument('-o', '--out', default='map.png')
    ap.add_argument('--center', type=float, nargs=2)
    ap.add_argument('--radius', type=float, default=150)
    ap.add_argument('--route')
    ap.add_argument('--dpi', type=int, default=150)
    a = ap.parse_args()
    g = pickle.load(open(a.pkl, 'rb'))
    lanes = g['lanes']
    fig, ax = plt.subplots(figsize=(16, 16))
    for k, r in lanes.items():
        P = r['pts']
        if a.center is not None:
            if np.min(np.hypot(P[:, 0] - a.center[0], P[:, 1] - a.center[1])) > a.radius * 1.5:
                continue
        col = '0.6'
        lw = 0.5
        if r['speed_limit'] == 30 and r['school_zone']:
            col, lw = 'orange', 1.2
        elif r['speed_limit'] == 30:
            col, lw = 'tab:blue', 1.2
        elif r['speed_limit'] == 50:
            col, lw = 'tab:green', 1.0
        ax.plot(P[:, 0], P[:, 1], color=col, lw=lw, alpha=0.9)
        # 방향 화살표 (중간 지점)
        if a.center is not None and r['length'] > 6:
            i = len(P) // 2
            d = P[min(i + 2, len(P) - 1)] - P[i]
            ax.arrow(P[i, 0], P[i, 1], d[0] * 0.5, d[1] * 0.5, head_width=0.8, color=col, alpha=0.7, length_includes_head=True)
        for sl in r['stop_lines']:
            p = travel_pt(r, sl['s'])
            hd = np.interp(sl['s'], r['s'], np.unwrap(r['hdg'].astype(float)))
            n = np.array([-np.sin(hd), np.cos(hd)]) * 1.4
            ax.plot([p[0] - n[0], p[0] + n[0]], [p[1] - n[1], p[1] + n[1]], color='red', lw=1.5)
            if sl['signal_ids']:
                ax.plot(p[0], p[1], 'o', color='red', ms=3)
        for s0, s1, kind in r['crosswalks']:
            if kind == 'pelican':
                p = travel_pt(r, 0.5 * (s0 + s1))
                ax.plot(p[0], p[1], 's', color='purple', ms=6)
        for s in r['crosswalk_warn']:
            p = travel_pt(r, s)
            ax.plot(p[0], p[1], '.', color='deepskyblue', ms=3)
    if a.route:
        rt = pickle.load(open(a.route, 'rb'))
        for k in rt['lanes']:
            P = lanes[k]['pts']
            ax.plot(P[:, 0], P[:, 1], color='black', lw=2.5, alpha=0.9)
        for i, (x, y) in enumerate(rt.get('waypoints', [])):
            ax.plot(x, y, '*', color='gold', ms=16, mec='black')
            ax.annotate(str(i), (x, y), fontsize=10)
        for ev in rt.get('events', []):
            k, s_in = ev['lane'], ev['s_in_lane']
            p = travel_pt(lanes[k], s_in)
            ax.plot(p[0], p[1], '^', color='magenta', ms=8)
            ax.annotate(ev['kind'], (p[0], p[1]), fontsize=7, color='magenta')
    ax.set_aspect('equal')
    if a.center is not None:
        ax.set_xlim(a.center[0] - a.radius, a.center[0] + a.radius)
        ax.set_ylim(a.center[1] - a.radius, a.center[1] + a.radius)
    ax.set_title(f"lanes={g['meta']['n_lanes']}  points={g['meta']['n_points']}  (gray=driving, blue=30, orange=school30, green=50, red=stopline)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(a.out, dpi=a.dpi)
    print('saved', a.out)


if __name__ == '__main__':
    main()
