#!/usr/bin/env python3
"""
build_route.py ─ 대회 공식 경유점 CSV → route.pkl   (대회날 경로 받으면 실행)

    python3 build_route.py lane_graph.pkl waypoints.csv -o route.pkl \
            [--radius 8] [--start-yaw 0.53 | --ego-yaw 0.53] [--no-pairs]

[공식 CSV 형식]
    헤더 있는 CSV: seq,x,y   (VTD 월드 직교좌표 [m] — ego/objects 와 같은 좌표계)
      · 첫 지점(seq 최소) = 시작점, 마지막 = 종료점
      · 중간 지점은 2개씩 짝: (2,3)=첫 교차로 진입·진출, (4,5)=둘째 교차로 ...
        짝 사이 구간은 **교차로 내부** 이므로 차선변경을 금지하고 junction 통과를 확인한다
      · 경로는 최단거리 기준, seq 순서대로 통과해야 한다 (이탈 시 감점)
    헤더/seq 열이 없으면 "x,y" 만 있는 옛 형식으로 읽고 순서를 그대로 쓴다.

route.pkl:
    lanes        : [lane_key ...]        차로 순서
    cum_s        : [float ...]           각 차로 시작점의 경로 누적거리
    lengths      : [float ...]
    total_length : float
    waypoints    : [(x,y) ...]
    waypoint_s   : [float ...]           각 경유점의 경로 누적거리
    events       : [{kind, s, lane, s_in_lane, ...}]
                   kind = turn_left / turn_right / lane_change_left / lane_change_right
                   lane_change 는 window_s0/window_s1 (경로 누적거리, 점선 구간) 포함
탐색 규칙: 차로 길이 = 비용, 차선변경 = +25m 비용 (점선 구간이 있을 때만 허용), 막다른 차로 자동 회피
"""
import argparse, heapq, math, pickle, sys
import numpy as np
# core/ 는 ROS 패키지 안(src/hlfma/hlfma/core)에 있다. 그 패키지 루트를 올린다.
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent / 'src' / 'hlfma'))
from hlfma.core.lanegraph import LaneGraph, wrap

LC_PENALTY = 25.0
# 같은 거리 층 안에서 목표 차로를 고를 때의 가중치 [비용/m] — 경유점에 가까운 쪽 우선
TARGET_DIST_W = 5.0


def candidates(lg, x, y, radius, yaw=None):
    """경유점 근처 후보 (lane_key, s, dist)"""
    d, ii = lg.kd.query((x, y), k=40)
    seen = {}
    for dist, i in zip(d, ii):
        if not np.isfinite(dist) or dist > radius:
            continue
        key = lg.lane_keys[lg.kd_lane[i]]
        if key in seen:
            continue
        s, t, dd, j = lg.project(key, x, y, idx_hint=int(lg.kd_i[i]))
        if yaw is not None:
            hd = float(np.interp(s, lg.lanes[key]['s'], np.unwrap(lg.lanes[key]['hdg'].astype(float))))
            if abs(wrap(yaw - hd)) > math.radians(70):
                continue
        # 폭이 너무 좁은 지점(포켓 시작 등)은 배제
        if lg.width_at(key, s) < 2.0:
            continue
        seen[key] = (key, s, dd)
    return sorted(seen.values(), key=lambda c: c[2])


def has_broken(lg, key, side):
    if lg.neighbor(key, side) is None:
        return False
    return any(ok for _, _, _, _, ok in lg.lanes[key]['left_mark' if side == 'left' else 'right_mark'])


def dijkstra(lg, starts, targets, allow_lane_change=True):
    """starts: [(lane, s_start)]  targets: {lane: s_target} → (cost, [ (lane, s_enter) ... ])

    allow_lane_change=False 면 successor 링크만 따라간다 (교차로 내부 구간용)."""
    tgt = dict(targets)
    best = {}
    heap = []
    for key, s in starts:
        if key in tgt and tgt[key] >= s - 1e-6:
            # 같은 차로 안에서 도달
            heapq.heappush(heap, (tgt[key] - s, key, s, None, key, True))
        heapq.heappush(heap, (lg.length(key) - s, key, s, None, key, False))
    parent = {}
    result = None
    while heap:
        cost, key, s_enter, par, root, done = heapq.heappop(heap)
        state = (key, done)
        if state in best:
            continue
        best[state] = (cost, par, s_enter)
        if done:
            result = (cost, key, s_enter)
            break
        # key 끝에 도달한 상태 (cost = 끝까지). 다음 후보들
        r = lg.lanes[key]
        for k2 in r['next']:
            if (k2, False) in best and (k2, True) in best:
                continue
            L2 = lg.length(k2)
            if k2 in tgt:
                heapq.heappush(heap, (cost + tgt[k2], k2, 0.0, (key, s_enter), root, True))
            heapq.heappush(heap, (cost + L2, k2, 0.0, (key, s_enter), root, False))
        # 차선변경: 같은 s 로 옆 차로에 진입 (진입 지점은 이 차로 시작 s_enter 이후 아무 데나 → 여기선 s_enter 로 근사)
        if not allow_lane_change:
            continue
        for side in ('left', 'right'):
            if not has_broken(lg, key, side):
                continue
            k2 = lg.neighbor(key, side)
            L2 = lg.length(k2)
            s2 = min(s_enter, L2)
            c_lc = cost - (r['length'] - s_enter) + LC_PENALTY  # 이 차로를 s_enter 에서 떠나는 비용으로 되돌림
            if k2 in tgt and tgt[k2] >= s2:
                heapq.heappush(heap, (c_lc + (tgt[k2] - s2), k2, s2, (key, s_enter), root, True))
            heapq.heappush(heap, (c_lc + (L2 - s2), k2, s2, (key, s_enter), root, False))
    if result is None:
        return None
    cost, key, s_enter = result
    path = [(key, s_enter)]
    cur = (key, True)
    par = best[cur][1]
    while par is not None:
        path.append(par)
        pk, ps = par
        # 부모 상태는 done=False
        par = best[(pk, False)][1] if (pk, False) in best else None
    path.reverse()
    return cost, path


class RouteError(SystemExit):
    """경로 생성 실패. 어느 seq 에서 왜 막혔는지 메시지에 담는다."""


def read_waypoints_csv(path):
    """
    공식 CSV(seq,x,y) 를 읽어 [(seq, x, y)] 로. seq 순으로 정렬한다.

    헤더 유무와 seq 열 유무를 자동 인식한다. seq 가 없으면 파일 순서를 seq 로 쓴다.
    """
    rows, header = [], None
    with open(path, newline='', encoding='utf-8-sig') as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            parts = [c.strip() for c in line.replace(';', ',').replace('\t', ',').split(',')]
            parts = [c for c in parts if c != '']
            if not parts:
                continue
            # 숫자로 안 읽히면 헤더로 본다 (맨 처음 한 번만)
            try:
                [float(c) for c in parts]
            except ValueError:
                if header is None and not rows:
                    header = [c.lower() for c in parts]
                    continue
                raise RouteError(f'{path}: 숫자로 읽을 수 없는 줄 → {line!r}')
            rows.append(parts)

    if not rows:
        raise RouteError(f'{path}: 경유점이 하나도 없다')

    # 열 위치 결정
    if header and 'x' in header and 'y' in header:
        ix, iy = header.index('x'), header.index('y')
        iseq = header.index('seq') if 'seq' in header else None
    elif len(rows[0]) >= 3:
        iseq, ix, iy = 0, 1, 2          # 헤더가 없어도 3열이면 seq,x,y 로 본다
    else:
        iseq, ix, iy = None, 0, 1

    out = []
    for n, parts in enumerate(rows, 1):
        need = max(i for i in (iseq, ix, iy) if i is not None)
        if len(parts) <= need:
            raise RouteError(f'{path}: {n}번째 줄의 열이 부족하다 → {parts}')
        seq = int(float(parts[iseq])) if iseq is not None else n
        out.append((seq, float(parts[ix]), float(parts[iy])))

    out.sort(key=lambda r: r[0])
    seqs = [r[0] for r in out]
    if len(set(seqs)) != len(seqs):
        dup = sorted({v for v in seqs if seqs.count(v) > 1})
        raise RouteError(f'{path}: seq 가 중복된다 → {dup}')
    return out


def junction_segments(n_points):
    """
    교차로 내부 구간(0-based 세그먼트 인덱스) 집합.

    seq 1=시작, (2,3)(4,5)... 짝, 마지막=종료 이므로 0-based 로는
    waypoints[1]→[2], [3]→[4], ... 즉 **홀수 인덱스 세그먼트**가 교차로 내부다.
    """
    return {wi for wi in range(1, n_points - 1, 2)}


def build_route(lg, waypoints, radius=8.0, start_yaw=None, junction_segs=frozenset(),
                seqs=None):
    seq = []   # [(lane, s_enter)]
    wp_s = []
    total = 0.0
    prev_end = None  # (lane, s) 이전 경유점 위치
    seg_span = []    # 세그먼트별 seq 경로상 차로 구간 [(wi, i0, i1)]
    def label(wi):
        return f'seq {seqs[wi]}' if seqs else f'waypoint {wi}'
    def nearest_report(x, y):
        d, _ii = lg.kd.query((x, y), k=1)
        return float(np.atleast_1d(d)[0])
    for wi in range(len(waypoints) - 1):
        x0, y0 = waypoints[wi]
        x1, y1 = waypoints[wi + 1]
        if prev_end is None:
            # 출발점: 헤딩 포함해서 하나로 확정 (여러 후보를 주면 바로 앞 차로가 선택되는 문제)
            m0 = lg.locate(x0, y0, start_yaw, max_dist=radius)
            if m0 is not None:
                starts = [(m0.lane, m0.s)]
            else:
                starts = [(k, s) for k, s, d in candidates(lg, x0, y0, radius, start_yaw)[:6]]
            if not starts:
                raise RouteError(
                    f'{label(wi)} ({x0:.2f},{y0:.2f}): 반경 {radius:g}m 내 차로 없음 '
                    f'(최근접 {nearest_report(x0, y0):.1f}m)')
        else:
            starts = [prev_end]

        # 도착 헤딩 = 이 구간의 진행방향. 경유점은 "여기를 이 방향으로 지난다" 는
        # 뜻이므로 반대편 차로는 후보에서 빼야 한다 (안 그러면 유턴 경로가 생긴다).
        arrive_yaw = math.atan2(y1 - y0, x1 - x0)
        tg = candidates(lg, x1, y1, radius, arrive_yaw)
        if not tg:
            loose = candidates(lg, x1, y1, radius)
            hint = ''
            if loose:
                hint = (f' (헤딩을 무시하면 {loose[0][0]} 가 {loose[0][2]:.2f}m 에 있다'
                        f' — 진행방향이 반대일 수 있다)')
            raise RouteError(
                f'{label(wi + 1)} ({x1:.2f},{y1:.2f}): 반경 {radius:g}m 내 '
                f'진행방향이 맞는 차로 없음 (최근접 {nearest_report(x1, y1):.1f}m){hint}')
        allow_lc = wi not in junction_segs
        # 경유점은 **어느 차로인지까지** 지정한다. 가까운 차로부터 층을 넓혀가며 찾고
        # 층 안에서만 비용으로 고른다. 이렇게 안 하면 차선변경 비용(25 m)을 피하려고
        # 3 m 옆 차로에서 구간을 끝내버려 다음 교차로 연결이 끊긴다.
        best = None
        d0 = tg[0][2]
        for tier in (d0 + 0.5, d0 + 2.0, radius):
            for k, s, d in tg:
                if d > tier:
                    continue
                res = dijkstra(lg, starts, {k: s}, allow_lane_change=allow_lc)
                if res is None:
                    continue
                score = res[0] + TARGET_DIST_W * d
                if best is None or score < best[0]:
                    best = (score, res[1], k, s)
            if best is not None:
                break
        if best is None:
            # 제약을 풀지 않는다. 풀면 경로가 통째로 엉뚱한 데로 새면서
            # "성공했지만 완전히 틀린 경로" 가 나온다. 실패를 그대로 보고한다.
            near = ', '.join(f'{k}@{d:.2f}m' for k, s, d in tg[:4])
            extra = ''
            if not allow_lc:
                extra = ('\n       이 구간은 교차로 내부(짝)라 차선변경이 금지돼 있다.'
                         '\n       진입 차로가 이미 틀렸을 가능성이 크다 — 앞 구간이 끝난 차로를 확인할 것.')
                if any(dijkstra(lg, starts, {k: s}, allow_lane_change=True)
                       for k, s, d in tg[:4]):
                    extra += '\n       (차선변경을 허용하면 연결된다 = 차로 선택 문제)'
            raise RouteError(
                f'{label(wi)} ({x0:.2f},{y0:.2f}) -> {label(wi + 1)} ({x1:.2f},{y1:.2f}): '
                f'경로 없음\n       출발 차로 {starts[0][0]}   후보 {near}{extra}')
        _score, path, k_end, s_end = best
        if wi == 0:
            wp_s.append(0.0)
        # path 를 seq 에 이어붙임 (첫 원소는 prev_end 와 같은 차로면 중복 제거)
        i0 = max(0, len(seq) - 1)
        for j, (k, s_en) in enumerate(path):
            if seq and seq[-1][0] == k:
                continue
            seq.append((k, s_en))
        seg_span.append((wi, i0, len(seq) - 1))
        # 누적거리: seq 기준으로 재계산 후 경유점 위치
        prev_end = (k_end, s_end)
        # 경유점 누적거리 계산은 마지막에
        wp_s.append(None)
    # 누적거리: dist(lane i, s_in_lane) = cum[i] + s_in_lane,  출발점이 0
    lanes = [k for k, _ in seq]
    lengths = [lg.length(k) for k in lanes]
    s_first = seq[0][1]
    cum = [-s_first]
    for i in range(1, len(lanes)):
        cum.append(cum[i - 1] + lengths[i - 1])
    total = cum[-1] + lengths[-1]
    # 경유점 누적거리: 경로 차로들에 투영해서 가장 가까운 것
    wp_dist = [0.0]
    for wi in range(1, len(waypoints)):
        x, y = waypoints[wi]
        best = None
        for i, k in enumerate(lanes):
            s_p, t_p, d_p, _ = lg.project(k, x, y)
            if d_p <= radius and (best is None or d_p < best[0]):
                best = (d_p, cum[i] + s_p)
        wp_dist.append(best[1] if best else None)
    # 이벤트: 회전 / 차선변경
    events = []
    for i, k in enumerate(lanes):
        r = lg.lanes[k]
        # 차선변경: 다음 차로가 successor 가 아니라 이웃이면
        if i + 1 < len(lanes):
            k2 = lanes[i + 1]
            if k2 not in r['next']:
                side = 'left' if r['left_nb'] == k2 else ('right' if r['right_nb'] == k2 else None)
                if side:
                    segs = [(a, b) for a, b, typ, col, ok in r['left_mark' if side == 'left' else 'right_mark'] if ok]
                    s_en = seq[i][1] if i == 0 else 0.0
                    w0, w1 = (segs[0][0], segs[-1][1]) if segs else (s_en, r['length'])
                    w0 = max(w0, s_en)
                    events.append({'kind': f'lane_change_{side}', 's': cum[i] + w0, 'lane': k, 's_in_lane': w0,
                                   'window_s0': cum[i] + w0, 'window_s1': cum[i] + w1, 'to_lane': k2})
        # 회전: 교차로 연결도로에서 헤딩 변화
        if r['junction'] != -1 and (i == 0 or lg.lanes[lanes[i - 1]]['junction'] == -1):
            h = np.unwrap(r['hdg'].astype(float))
            dh = float(h[-1] - h[0])
            # 연결도로가 여러 개 이어질 수 있어 뒤로 합침
            j = i + 1
            while j < len(lanes) and lg.lanes[lanes[j]]['junction'] == r['junction']:
                h2 = np.unwrap(lg.lanes[lanes[j]]['hdg'].astype(float))
                dh += float(h2[-1] - h2[0])
                j += 1
            if abs(dh) > math.radians(25):
                events.append({'kind': 'turn_left' if dh > 0 else 'turn_right', 's': cum[i], 'lane': k, 's_in_lane': 0.0,
                               'junction': r['junction'], 'delta_heading_deg': math.degrees(dh)})
    # ── 짝(공식 CSV) 기준 회전 보정 ──────────────────────────────────────
    # 위 검출은 교차로 **연결로 자체의 곡률**만 본다. 진입로→진출로 전체로는
    # 확실히 꺾이는데 연결로가 완만해서 25° 임계를 못 넘는 경우가 있고, 그러면
    # turn 이벤트가 안 생겨 방향지시등이 안 켜진다(감점 항목).
    # 공식 CSV 의 짝은 "여기가 교차로다" 라는 확정 정보이므로 이를 근거로 보정한다.
    for wi, i0, i1 in seg_span:
        if wi not in junction_segs or i1 <= i0:
            continue
        h0 = np.unwrap(lg.lanes[lanes[i0]]['hdg'].astype(float))
        h1 = np.unwrap(lg.lanes[lanes[i1]]['hdg'].astype(float))
        dh = math.degrees(wrap(float(h1[-1]) - float(h0[0])))
        if abs(dh) <= 25.0:
            continue
        kind = 'turn_left' if dh > 0 else 'turn_right'
        # 이 교차로 구간에 이미 같은 방향 회전 이벤트가 있으면 건드리지 않는다
        s_lo, s_hi = cum[i0], cum[i1] + lengths[i1]
        if any(e['kind'] == kind and s_lo - 1e-6 <= e['s'] <= s_hi + 1e-6 for e in events):
            continue
        jids = [lg.lanes[k]['junction'] for k in lanes[i0:i1 + 1]
                if lg.lanes[k]['junction'] != -1]
        # 회전 시작점 = 교차로 진입 차로의 시작
        i_start = next((i for i in range(i0, i1 + 1)
                        if lg.lanes[lanes[i]]['junction'] != -1), i0)
        events.append({'kind': kind, 's': cum[i_start], 'lane': lanes[i_start],
                       's_in_lane': 0.0, 'junction': jids[0] if jids else None,
                       'delta_heading_deg': dh, 'source': 'pair'})

    events.sort(key=lambda e: e['s'])
    return {'lanes': lanes, 'cum_s': cum, 'lengths': lengths, 'total_length': total, 'start_s_in_lane': s_first,
            'waypoints': [tuple(w) for w in waypoints], 'waypoint_s': wp_dist, 'events': events,
            'waypoint_seq': list(seqs) if seqs else list(range(1, len(waypoints) + 1)),
            'junction_segments': sorted(junction_segs), 'segment_span': seg_span}


def turn_kind(delta_deg, straight_deg=25.0):
    if delta_deg > straight_deg:
        return '좌회전'
    if delta_deg < -straight_deg:
        return '우회전'
    return '직진'


def report(lg, rt, radius, warn_dev=None):
    """
    검증 리포트. 대회날 경로를 받자마자 눈으로 확인하는 용도다.
    문제가 있으면 [경고] 로 표시하고, 경고 개수를 돌려준다.
    """
    warn_dev = radius if warn_dev is None else warn_dev
    lanes, cum = rt['lanes'], rt['cum_s']
    warns = 0

    print(f"\n{'=' * 72}")
    print(f"경로 검증 리포트")
    print('=' * 72)

    # ── 1) seq 점이 경로에서 얼마나 떨어져 있나 ──────────────────────────
    print(f"\n[1] 경유점 이탈 (허용 {warn_dev:g} m)")
    for wi, (x, y) in enumerate(rt['waypoints']):
        sq = rt['waypoint_seq'][wi]
        best_d, best_s, best_lane = None, None, None
        for i, k in enumerate(lanes):
            s_p, _t, d_p, _ = lg.project(k, x, y)
            if best_d is None or d_p < best_d:
                best_d, best_s, best_lane = d_p, cum[i] + s_p, k
        flag = ''
        if best_d > warn_dev:
            flag = '   <= [경고] 경로가 이 지점을 스치지 않는다'
            warns += 1
        print(f"  seq {sq:>3}  ({x:9.2f},{y:9.2f})  이탈 {best_d:6.2f} m  "
              f"경로 s={best_s:8.1f} m  lane={best_lane}{flag}")

    # ── 2) 짝(교차로) 구간 ───────────────────────────────────────────────
    jsegs = set(rt.get('junction_segments') or [])
    spans = {wi: (i0, i1) for wi, i0, i1 in rt.get('segment_span') or []}
    print(f"\n[2] 교차로 짝 구간")
    if not jsegs:
        print('  (짝 정보 없음 — --no-pairs 이거나 경유점이 홀수 개)')
    for wi in sorted(jsegs):
        sq_in, sq_out = rt['waypoint_seq'][wi], rt['waypoint_seq'][wi + 1]
        i0, i1 = spans.get(wi, (None, None))
        if i0 is None:
            continue
        seg_lanes = lanes[i0:i1 + 1]
        jids = []
        for k in seg_lanes:
            j = lg.lanes[k]['junction']
            if j != -1 and j not in jids:
                jids.append(j)
        # 진입→진출 헤딩 변화로 좌/우/직진 판정
        h0 = np.unwrap(lg.lanes[seg_lanes[0]]['hdg'].astype(float))
        h1 = np.unwrap(lg.lanes[seg_lanes[-1]]['hdg'].astype(float))
        dh = math.degrees(wrap(float(h1[-1]) - float(h0[0])))
        # 진짜 위반은 "교차로 연결로 위에서의 차선변경" 이다.
        # 거리 구간으로 세면 점선 구간이 열리는 지점(s)이 진출 경유점보다 몇십 cm
        # 앞선다는 이유로 정상 차선변경까지 잡힌다. 이벤트가 일어나는 차로가
        # junction 차로인지로 판정한다.
        seg_keys = set(seg_lanes)
        lc = sum(1 for e in rt['events']
                 if e['kind'].startswith('lane_change')
                 and e.get('lane') in seg_keys
                 and lg.lanes[e['lane']]['junction'] != -1)
        flag = ''
        if not jids:
            flag = '   <= [경고] junction 차로를 안 거친다'
            warns += 1
        if lc:
            flag += f'   <= [경고] 교차로 내부에서 차선변경 {lc}회'
            warns += 1
        print(f"  seq {sq_in:>3}→{sq_out:<3}  junction={jids if jids else '없음'}  "
              f"Δheading={dh:+7.1f}°  {turn_kind(dh)}  차로 {len(seg_lanes)}개{flag}")

    # ── 3) 총계 ──────────────────────────────────────────────────────────
    ev = rt['events']
    n_l = sum(1 for e in ev if e['kind'] == 'turn_left')
    n_r = sum(1 for e in ev if e['kind'] == 'turn_right')
    n_lc = sum(1 for e in ev if e['kind'].startswith('lane_change'))

    # 스쿨존 구간 (연속 묶음)
    zones, cur = [], None
    for i, k in enumerate(lanes):
        if lg.lanes[k]['school_zone']:
            end = cum[i] + lg.length(k)
            if cur is None:
                cur = [cum[i], end]
            else:
                cur[1] = end
        elif cur is not None:
            zones.append(tuple(cur)); cur = None
    if cur is not None:
        zones.append(tuple(cur))

    # 정지선 (경로 진행 범위 안의 것만)
    stops = []
    for i, k in enumerate(lanes):
        for sl in lg.lanes[k]['stop_lines']:
            s_abs = cum[i] + sl['s']
            if -1.0 <= s_abs <= rt['total_length'] + 1.0:
                stops.append((s_abs, sl.get('signal_ids') or []))
    stops.sort()
    unsignalized = sum(1 for _s, ids in stops if not ids)

    print(f"\n[3] 총계")
    print(f"  총 길이        {rt['total_length']:8.1f} m   차로 {len(lanes)}개")
    print(f"  좌회전 {n_l}회 / 우회전 {n_r}회 / 차선변경 {n_lc}회")
    print(f"  정지선 {len(stops)}개 (신호 없는 정지선 {unsignalized}개)")
    if zones:
        print(f"  스쿨존 {len(zones)}구간:")
        for a0, b0 in zones:
            print(f"    s {a0:8.1f} ~ {b0:8.1f} m  ({b0 - a0:6.1f} m)")
    else:
        print('  스쿨존 없음')

    print(f"\n[4] 이벤트")
    for e in ev:
        extra = ''
        if 'window_s0' in e:
            extra = f"  window {e['window_s0']:.0f}-{e['window_s1']:.0f} m"
        if 'delta_heading_deg' in e:
            extra = (f"  Δ{e['delta_heading_deg']:+.1f}°  junction={e.get('junction')}"
                     + ('  [짝 기준 보정]' if e.get('source') == 'pair' else ''))
        print(f"  {e['s']:8.1f} m  {e['kind']:<20}{extra}")

    print(f"\n{'=' * 72}")
    print('경고 없음' if warns == 0 else f'[경고] {warns}건 — 위 표시된 항목을 확인할 것')
    print('=' * 72)
    return warns


def main():
    ap = argparse.ArgumentParser(description='대회 공식 경유점 CSV → route.pkl')
    ap.add_argument('pkl')
    ap.add_argument('waypoints', help='csv: seq,x,y (헤더 있어도 됨)')
    ap.add_argument('-o', '--out', default='route.pkl')
    ap.add_argument('--radius', type=float, default=8.0, help='[m] 경유점 매칭 반경')
    ap.add_argument('--start-yaw', type=float, default=None,
                    help='[rad] 출발 헤딩. 없으면 seq1→seq2 방향으로 자동 추정')
    ap.add_argument('--ego-yaw', type=float, default=None,
                    help='[rad] 9910 에서 받은 실제 ego heading (--start-yaw 보다 우선)')
    ap.add_argument('--no-pairs', action='store_true',
                    help='중간 지점을 교차로 진입·진출 짝으로 해석하지 않는다')
    ap.add_argument('--warn-dev', type=float, default=None,
                    help='[m] 경유점 이탈 경고 임계 (기본: --radius)')
    ap.add_argument('--yaw-min-dist', type=float, default=2.0,
                    help='[m] 헤딩 자동추정에 쓸 최소 거리. 이보다 가까운 경유점은 건너뛴다')
    a = ap.parse_args()

    lg = LaneGraph(a.pkl)
    rows = read_waypoints_csv(a.waypoints)
    seqs = [r[0] for r in rows]
    wps = [(r[1], r[2]) for r in rows]
    if len(wps) < 2:
        raise RouteError('경유점이 2개 미만이다')

    # ── 출발 헤딩 ────────────────────────────────────────────────────────
    if a.ego_yaw is not None:
        start_yaw, src = a.ego_yaw, '--ego-yaw (9910 실측)'
    elif a.start_yaw is not None:
        start_yaw, src = a.start_yaw, '--start-yaw'
    else:
        # seq1→seq2 가 너무 가까우면 방향이 노이즈에 지배돼 반대로 잡힌다.
        # (실제로 0.11 m 떨어진 두 점에서 180° 틀린 헤딩이 나왔다)
        # yaw_min_dist 이상 떨어진 첫 경유점을 쓴다.
        ref = None
        for j in range(1, len(wps)):
            d = math.hypot(wps[j][0] - wps[0][0], wps[j][1] - wps[0][1])
            if d >= a.yaw_min_dist:
                ref = (j, d)
                break
        if ref is None:
            ref = (1, math.hypot(wps[1][0] - wps[0][0], wps[1][1] - wps[0][1]))
            print(f'  [경고] 모든 경유점이 시작점에서 {a.yaw_min_dist:g}m 이내다. '
                  f'헤딩 추정이 부정확할 수 있으니 --ego-yaw 를 주는 편이 안전하다',
                  file=sys.stderr)
        j, d = ref
        dx, dy = wps[j][0] - wps[0][0], wps[j][1] - wps[0][1]
        start_yaw = math.atan2(dy, dx)
        src = f'자동 추정 (seq {seqs[0]}→{seqs[j]} 방향, {d:.1f} m)'
    print(f'출발 헤딩 {start_yaw:+.5f} rad ({math.degrees(start_yaw):+.1f}°)  ← {src}')

    # ── 교차로 짝 ────────────────────────────────────────────────────────
    jsegs = set()
    if not a.no_pairs:
        if len(wps) % 2 != 0:
            print(f'  [경고] 경유점이 {len(wps)}개(홀수)다. 공식 형식은 '
                  f'시작 + 짝*N + 종료 = 짝수여야 한다. 짝 해석을 건너뛴다', file=sys.stderr)
        elif len(wps) == 2:
            print('  경유점이 시작/종료뿐 — 교차로 짝 없음')
        else:
            jsegs = junction_segments(len(wps))
            pairs = [(seqs[wi], seqs[wi + 1]) for wi in sorted(jsegs)]
            print(f'  교차로 짝 {len(pairs)}개: ' +
                  ', '.join(f'({i}→{o})' for i, o in pairs))

    rt = build_route(lg, wps, a.radius, start_yaw, junction_segs=jsegs, seqs=seqs)

    with open(a.out, 'wb') as f:
        pickle.dump(rt, f, protocol=4)

    warns = report(lg, rt, a.radius, a.warn_dev)
    print(f'saved {a.out}')
    return 1 if warns else 0


if __name__ == '__main__':
    raise SystemExit(main())
