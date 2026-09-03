#!/usr/bin/env python3
"""
perturb_route_test.py ─ 경유점 흔들기로 build_route 의 강건성을 잰다 (분석 전용)

    python3 tools/perturb_route_test.py [--graph data/lane_graph.pkl] [--radius 8]
        [--offsets 1.5,3.0,4.5] [--seeds 30] [--modes ball,k40] [--csv A.csv B.csv]
        [--boundary] [--ties]

**build_route 를 고치지 않는다.** import 해서 부르고, 스위치는 params 대신
모듈 캐시(_CAND_CFG / _START_OVERRIDE)를 직접 세팅해 켜고 끈다.

왜 흔드나
    대회 안내문 형식은 [시작, (진입,진출)×N, 종료] 다. 시작·종료를 뺀 경유점이
    전부 교차로 근처에 찍히고, 교차로는 kd 밀도가 최대(반경 8 m 안 330점)이며
    차로가 겹쳐 지나간다. 사람이 찍는 좌표는 중심선에서 1~5 m 벗어난다
    (2026-09-03 대회장 CSV 실측: 중앙 1.11 m, 최대 1.67 m, seq 9 정답 차로까지
    5.02 m). 즉 "정확한 좌표" 를 전제한 후보 선택은 현장에서 그대로 깨진다.

    원본(무흔들) 빌드의 차로 열을 정답으로 두고, 경유점을 반경 L 원판 안에서
    균일하게 흔들었을 때 같은 차로 열이 나오는지를 본다. 좌표만 흔든다 —
    CSV 에서 유도되는 출발 헤딩이 같이 변하는 건 현장과 같으므로 그대로 둔다.

유형 (build_route.junction_segments 의 짝 규칙 그대로)
    index 0 = start, 홀수 = entry(교차로 진입), 짝수 = exit(진출), 마지막 = end
"""
import argparse, contextlib, io, math, pathlib, re, sys, zlib
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'tools'))

import build_route as BR                                        # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                     # noqa: E402

DEFAULT_CSVS = [
    'data/official_route.csv',
    'data/test_route_waypoints.csv',
    'tests/fixtures/waypoints.csv',
    'tests/fixtures/venue_20260903_waypoints.csv',
] + sorted(str(p) for p in (_ROOT / 'scenarios' / '정적회피집중').glob('*.csv'))

TIE_EPS = 1e-6            # 이보다 가까우면 "완전 동률" 로 본다
WARN_RE = re.compile(r'출발 차로 불일치.*?점수 ([\d.]+)\).*?점수 ([\d.]+)\)', re.S)


def kind_of(i, n):
    """경유점 유형 — 대회 형식 [시작, (진입,진출)×N, 종료]."""
    if i == 0:
        return 'start'
    if i == n - 1:
        return 'end'
    return 'entry' if i % 2 == 1 else 'exit'


def set_mode(mode):
    """ball = 반경 기반(현재 기본) / k40 = 개수 기반(HEAD 동작)."""
    BR._CAND_CFG = (mode == 'ball', 5000)
    BR._START_OVERRIDE = (mode == 'ball')


def build(lg, wps, seqs, radius):
    """(rt, stderr) 또는 (None, 실패메시지). 표준출력은 버린다."""
    yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    err = io.StringIO()
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rt = BR.build_route(lg, wps, radius, yaw,
                                junction_segs=BR.junction_segments(len(wps)),
                                seqs=seqs, finish_tail_m=BR.finish_tail_cfg())
        return rt, err.getvalue()
    except BaseException as e:                        # RouteError 는 SystemExit 파생
        return None, str(e).split('\n')[0]


def targets_of(rt):
    """{경유점 index: 그 경유점을 목표로 뽑힌 차로} — segment_span 이 단일 출처."""
    return {wi + 1: rt['lanes'][i1] for wi, _i0, i1 in rt['segment_span']}


def tier_of(lg, wps, wi, chosen, radius, arrive_yaw):
    """선택 차로가 tier 몇에서 잡혔는지 (1/2/3). 계산 근거는 tier 루프와 같다.

    build_route 는 (d0+0.5, d0+2.0, radius) 순으로 층을 넓히다 **처음 연결이
    되는 층에서 멈춘다**. 그래서 선택 차로의 거리가 들어가는 가장 낮은 층이
    곧 사용된 층이다 — 더 낮은 층에서 연결이 됐다면 거기서 멈췄을 테니까.
    """
    tg = BR.candidates(lg, wps[wi][0], wps[wi][1], radius, arrive_yaw)
    if not tg:
        return None, None
    d0 = tg[0][2]
    d = next((c[2] for c in tg if c[0] == chosen), None)
    if d is None:
        return None, None
    for t, hi in enumerate((d0 + 0.5, d0 + 2.0, radius), start=1):
        if d <= hi:
            return t, d
    return 3, d


def wp_max_dist(lg, rt):
    """경유점 → 경로 차로 중심선 최대 거리 [m]. 조용한 오경로의 눈금."""
    out = 0.0
    for x, y in rt['waypoints']:
        out = max(out, min(lg.project(k, x, y)[2] for k in rt['lanes']))
    return out


def seed_of(*parts):
    """재현 가능한 시드. 파이썬 hash() 는 프로세스마다 달라져서 못 쓴다
    (str 해시 랜덤화) — 같은 인자로 두 번 돌리면 다른 표본이 나온다."""
    return zlib.crc32('|'.join(map(str, parts)).encode()) & 0xFFFFFFFF


def perturb(wps, L, rng):
    """각 경유점을 반경 L 원판 안에서 균일하게 이동 (좌표만)."""
    r = L * np.sqrt(rng.random(len(wps)))
    th = rng.uniform(0, 2 * math.pi, len(wps))
    return [(x + rr * math.cos(tt), y + rr * math.sin(tt))
            for (x, y), rr, tt in zip(wps, r, th)]


# ── 4-2 흔들기 ──────────────────────────────────────────────────────────
def run_perturb(lg, csvs, offsets, seeds, radius, modes):
    rows = []                     # (csv, mode, L, seed, ok, match, first_bad, ...)
    base = {}
    for f in csvs:
        recs = BR.read_waypoints_csv(f)
        wps = [(r[1], r[2]) for r in recs]
        seqs = [r[0] for r in recs]
        set_mode('ball')
        rt, _ = build(lg, wps, seqs, radius)
        if rt is None:
            print(f'  [건너뜀] {f}: 무흔들 빌드 실패', file=sys.stderr)
            continue
        base[f] = (wps, seqs, targets_of(rt), rt['lanes'])

    for f, (wps, seqs, tgt0, lanes0) in base.items():
        n = len(wps)
        for mode in modes:
            set_mode(mode)
            for L in offsets:
                for sd in range(seeds):
                    rng = np.random.default_rng(seed_of(f, mode, L, sd))
                    pw = perturb(wps, L, rng)
                    rt, err = build(lg, pw, seqs, radius)
                    rec = dict(csv=f, mode=mode, L=L, seed=sd, ok=rt is not None,
                               match=False, first_bad=None, kind=None, got=None,
                               want=None, tier=None, wpmax=None, ov=False, ov_gap=None,
                               err=None if rt is not None else err)
                    if rt is not None:
                        m = WARN_RE.search(err)
                        rec['ov'] = '출발 차로 불일치' in err
                        if m:
                            rec['ov_gap'] = float(m.group(1)) - float(m.group(2))
                        rec['wpmax'] = wp_max_dist(lg, rt)
                        tgt = targets_of(rt)
                        rec['match'] = rt['lanes'] == lanes0
                        if not rec['match']:
                            for i in sorted(tgt0):
                                if tgt.get(i) != tgt0[i]:
                                    ay = math.atan2(pw[i][1] - pw[i - 1][1],
                                                    pw[i][0] - pw[i - 1][0])
                                    t, _d = tier_of(lg, pw, i, tgt.get(i), radius, ay)
                                    rec.update(first_bad=i, kind=kind_of(i, n),
                                               got=tgt.get(i), want=tgt0[i], tier=t)
                                    break
                            else:
                                rec.update(first_bad=-1, kind='(차로열만 상이)')
                    rows.append(rec)
    return rows


def report_perturb(rows, offsets, modes):
    print('\n' + '=' * 96)
    print('[4-2] 경유점 흔들기 — 반경 L 원판 균일, 좌표만')
    print('=' * 96)
    print('\n L [m]  mode  표본  빌드성공   차로열 일치   출발override   경유점→경로 최대거리 p50/p95/max')
    for mode in modes:
        for L in offsets:
            r = [x for x in rows if x['mode'] == mode and x['L'] == L]
            if not r:
                continue
            ok = [x for x in r if x['ok']]
            mt = [x for x in ok if x['match']]
            wm = np.array([x['wpmax'] for x in ok]) if ok else np.zeros(1)
            ov = sum(1 for x in ok if x['ov'])
            print(' %4.1f  %-5s %4d  %3d (%5.1f%%)  %3d (%5.1f%%)  %6d      %.2f / %.2f / %.2f' % (
                L, mode, len(r), len(ok), 100 * len(ok) / len(r),
                len(mt), 100 * len(mt) / max(1, len(ok)), ov,
                np.percentile(wm, 50), np.percentile(wm, 95), wm.max()))

    print('\n 불일치 유형별 (첫 갈라지는 경유점 기준)')
    print(' L [m]  mode  start  entry   exit    end   (차로열만)   빌드실패')
    for mode in modes:
        for L in offsets:
            r = [x for x in rows if x['mode'] == mode and x['L'] == L]
            if not r:
                continue
            c = {k: sum(1 for x in r if x['kind'] == k)
                 for k in ('start', 'entry', 'exit', 'end', '(차로열만 상이)')}
            print(' %4.1f  %-5s %5d  %5d  %5d  %5d   %8d   %8d' % (
                L, mode, c['start'], c['entry'], c['exit'], c['end'],
                c['(차로열만 상이)'], sum(1 for x in r if not x['ok'])))

    print('\n 불일치 시 선택 차로가 잡힌 tier')
    print(' L [m]  mode   tier1  tier2  tier3')
    for mode in modes:
        for L in offsets:
            r = [x for x in rows if x['mode'] == mode and x['L'] == L and x['tier']]
            if not r:
                continue
            print(' %4.1f  %-5s %6d %6d %6d' % (
                L, mode, sum(1 for x in r if x['tier'] == 1),
                sum(1 for x in r if x['tier'] == 2), sum(1 for x in r if x['tier'] == 3)))

    ov = [x for x in rows if x['ov']]
    print('\n 출발 차로 override 발동 %d건' % len(ov))
    for x in ov[:12]:
        print('   %-46s %-5s L=%.1f seed=%2d  점수여유 %s' % (
            x['csv'][-46:], x['mode'], x['L'], x['seed'],
            '%.3f' % x['ov_gap'] if x['ov_gap'] is not None else '-'))

    print('\n CSV 별 차로열 일치율 (%s)' % ' / '.join('L=%.1f' % L for L in offsets))
    for f in sorted({x['csv'] for x in rows}):
        line = []
        for mode in modes:
            cells = []
            for L in offsets:
                r = [x for x in rows if x['csv'] == f and x['mode'] == mode and x['L'] == L]
                ok = [x for x in r if x['ok']]
                cells.append('%3.0f%%' % (100 * sum(1 for x in ok if x['match']) / max(1, len(r))))
            line.append('%-5s %s' % (mode, ' '.join(cells)))
        print('   %-46s  %s' % (f[-46:], '  |  '.join(line)))


# ── 경계선(동률) 케이스 ─────────────────────────────────────────────────
def tie_nodes(lg, eps=TIE_EPS):
    """두 개 이상의 차로가 같은 좌표를 공유하는 kd 점 index — 차로 경계 노드."""
    pts = np.asarray(lg.kd_pts)
    pairs = lg.kd.query_pairs(eps, output_type='ndarray')
    lane = np.asarray(lg.kd_lane)
    keep = pairs[lane[pairs[:, 0]] != lane[pairs[:, 1]]]
    return pts, np.unique(keep[:, 0]) if len(keep) else np.array([], dtype=int)


def run_ties(lg, csvs, radius):
    """무흔들 빌드의 각 세그먼트에서 후보 거리 동률이 있었는지, 무엇이 갈랐는지."""
    print('\n' + '=' * 96)
    print('[4-3] 경계선(동률) 케이스 — 거리차 < %g m 인 후보가 있었나' % TIE_EPS)
    print('=' * 96)
    print('\n %-42s %-4s %-22s %s' % ('csv', 'seq', '선택 차로', '동률 후보 / dijkstra 연결'))
    set_mode('ball')
    for f in csvs:
        recs = BR.read_waypoints_csv(f)
        wps = [(r[1], r[2]) for r in recs]
        seqs = [r[0] for r in recs]
        rt, _ = build(lg, wps, seqs, radius)
        if rt is None:
            continue
        jseg = BR.junction_segments(len(wps))
        banned, _thr = BR.infeasible_connectors(lg)
        for wi, i0, i1 in rt['segment_span']:
            i = wi + 1
            ay = math.atan2(wps[i][1] - wps[i - 1][1], wps[i][0] - wps[i - 1][0])
            tg = BR.candidates(lg, wps[i][0], wps[i][1], radius, ay)
            if not tg:
                continue
            chosen = rt['lanes'][i1]
            d_ch = next((c[2] for c in tg if c[0] == chosen), None)
            if d_ch is None:
                continue
            tied = [c for c in tg if c[0] != chosen and abs(c[2] - d_ch) < TIE_EPS]
            if not tied:
                continue
            start = (rt['lanes'][i0], 0.0) if i0 else (rt['lanes'][0], rt['start_s_in_lane'])
            det = []
            for k, s, _d in tied:
                r = BR.dijkstra(lg, [start], {k: s},
                                allow_lane_change=wi not in jseg, banned=banned)
                det.append('%s %s' % (k, '연결 cost=%.1f' % r[0] if r else '연결 X'))
            print(' %-42s %-4s %-22s [%s] %s' % (
                f[-42:], seqs[i], str(chosen), kind_of(i, len(wps)), ' ; '.join(det)))


def run_boundary(lg, csvs, radius, snap_max=3.0):
    """경유점을 가까운 차로 경계 노드로 스냅해 동률을 강제로 만들어 본다."""
    print('\n' + '=' * 96)
    print('[4-3b] 경유점을 차로 경계 노드로 스냅 (스냅거리 <= %.1f m)' % snap_max)
    print('=' * 96)
    pts, tn = tie_nodes(lg)
    if not len(tn):
        print('  경계 노드 없음'); return
    from scipy.spatial import cKDTree
    kd = cKDTree(pts[tn])
    set_mode('ball')
    print('\n %-42s %5s %8s %8s  %s' % ('csv', '스냅', '무흔들', '스냅후', '첫 갈라지는 경유점'))
    for f in csvs:
        recs = BR.read_waypoints_csv(f)
        wps = [(r[1], r[2]) for r in recs]
        seqs = [r[0] for r in recs]
        rt0, _ = build(lg, wps, seqs, radius)
        if rt0 is None:
            continue
        d, ii = kd.query(np.array(wps), k=1)
        sw = [tuple(pts[tn[j]].astype(float)) if dd <= snap_max else w
              for w, dd, j in zip(wps, d, ii)]
        n_snap = sum(1 for dd in d if dd <= snap_max)
        rt1, _ = build(lg, sw, seqs, radius)
        if rt1 is None:
            print(' %-42s %5d %8s %8s  -' % (f[-42:], n_snap, 'OK', 'FAIL')); continue
        t0, t1 = targets_of(rt0), targets_of(rt1)
        bad = next((i for i in sorted(t0) if t0[i] != t1.get(i)), None)
        print(' %-42s %5d %8s %8s  %s' % (
            f[-42:], n_snap, 'OK', 'OK',
            '없음' if bad is None else 'seq %s [%s] %s -> %s' % (
                seqs[bad], kind_of(bad, len(wps)), t0[bad], t1.get(bad))))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--graph', default=str(_ROOT / 'data' / 'lane_graph.pkl'))
    ap.add_argument('--radius', type=float, default=8.0)
    ap.add_argument('--offsets', default='1.5,3.0,4.5')
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--modes', default='ball,k40')
    ap.add_argument('--csv', nargs='*', default=None)
    ap.add_argument('--ties', action='store_true', help='동률 감사만')
    ap.add_argument('--boundary', action='store_true', help='경계 노드 스냅만')
    a = ap.parse_args(argv)

    lg = LaneGraph(a.graph)
    csvs = a.csv or [str(_ROOT / c) for c in DEFAULT_CSVS]
    csvs = [c for c in csvs if pathlib.Path(c).exists()]
    offsets = [float(v) for v in a.offsets.split(',')]
    modes = a.modes.split(',')

    only = a.ties or a.boundary
    if not only:
        rows = run_perturb(lg, csvs, offsets, a.seeds, a.radius, modes)
        report_perturb(rows, offsets, modes)
    if a.ties or not only:
        run_ties(lg, csvs, a.radius)
    if a.boundary or not only:
        run_boundary(lg, csvs, a.radius)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
