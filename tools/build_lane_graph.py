#!/usr/bin/env python3
"""
build_lane_graph.py  ─  xodr(OpenDRIVE) → lane_graph.pkl   (대회 전 1번 실행)

사용:
    python3 build_lane_graph.py HL_FMA_VTD_LivingLab.xodr  -o lane_graph.pkl  [--ds 0.5]

만드는 것 (차로 하나마다):
    pts        : 주행 방향 순서의 중심선 점 (x, y, z)  ds 간격
    s          : 차로 시작점부터의 누적 거리 (주행 방향)
    road_s     : 각 점의 도로 s
    hdg, curv  : 주행 방향 헤딩 / 곡률
    width      : 차로 폭
    left_mark / right_mark : (s0, s1, type, color, lane_change_ok) 구간 리스트  (운전자 기준 좌/우)
    left_nb / right_nb     : 같은 방향 옆 차로 (없으면 None)
    left_is_center         : 왼쪽이 중앙선/중앙분리 (넘으면 중앙선 침범)
    next / prev            : 주행 방향 다음/이전 차로들 (교차로 연결 포함)
    stop_lines : [{s, signal_ids}]     정지선 (주행 s)
    signals    : [{id, stop_s, explicit}] 이 차로에 걸린 신호등
    crosswalks : [(s0, s1, kind)]      kind = 'pelican'(실제 객체) / 'inferred'(정지선 뒤 8m, xodr에 없어서 추정)
    crosswalk_warn / yield_marks / arrows / speed_marks / markings
    speed_limit, school_zone, speed_src

전제/주의 (README 참고):
  - 신호등 방향: validity 태그 > hOffset(0→우측차로 +s, π→좌측차로 -s) > t 부호  (이 맵에서 검증됨)
  - 제한속도: 표준 <speed> 필드가 없어서 노면표시 객체(RM_517_50 / roadmark_speed_30 / RM_518)로 도로 단위 추정
  - RM_518 = 어린이보호구역 속도제한 표시(30)로 가정 → 텍스처 확인 필요
  - 실제 횡단보도 객체는 2개(pelican)뿐. 교차로 횡단보도는 xodr에 없음 → 정지선 뒤 8m를 'inferred' 존으로 표시
"""
import argparse, math, pickle, sys, time, collections
import xml.etree.ElementTree as ET
import numpy as np
from scipy.integrate import cumulative_trapezoid

DRIVING = ('driving',)  # 그래프 노드가 되는 lane type

# ────────────────────────────────────────────────────────────────────────────
# 0. 유틸
# ────────────────────────────────────────────────────────────────────────────
def poly_eval(pieces, s, key='sOffset'):
    """pieces: [(s_start, a, b, c, d)] 중 s_start<=s 인 마지막 piece로 a+b*ds+c*ds²+d*ds³"""
    if not pieces or s < pieces[0][0] - 1e-9:
        return 0.0          # 첫 piece 이전 구간은 0 (laneOffset 등)
    st, a, b, c, d = pieces[0]
    for p in pieces:
        if p[0] <= s + 1e-9:
            st, a, b, c, d = p
        else:
            break
    ds = s - st
    return a + b * ds + c * ds * ds + d * ds * ds * ds


def poly_eval_arr(pieces, s_arr):
    out = np.zeros_like(s_arr, dtype=float)
    if not pieces:
        return out
    starts = np.array([p[0] for p in pieces])
    idx = np.searchsorted(starts, s_arr + 1e-9, side='right') - 1   # -1 = 첫 piece 이전 → 0 유지
    for k, (st, a, b, c, d) in enumerate(pieces):
        m = idx == k
        if m.any():
            ds = s_arr[m] - st
            out[m] = a + b * ds + c * ds ** 2 + d * ds ** 3
    return out


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def piece_at(pieces, s):
    """(sOffset, ...) 리스트에서 sOffset<=s 인 마지막 원소"""
    cur = pieces[0]
    for p in pieces:
        if p[0] <= s + 1e-9:
            cur = p
        else:
            break
    return cur


# ────────────────────────────────────────────────────────────────────────────
# 1. planView 기하 (line / arc / spiral / poly3)
# ────────────────────────────────────────────────────────────────────────────
def geom_line(g, sl):
    return g['x'] + sl * math.cos(g['hdg']), g['y'] + sl * math.sin(g['hdg']), np.full_like(sl, g['hdg'])


def geom_arc(g, sl):
    k = g['curvature']
    h0 = g['hdg']
    h = h0 + k * sl
    x = g['x'] + (np.sin(h) - math.sin(h0)) / k
    y = g['y'] - (np.cos(h) - math.cos(h0)) / k
    return x, y, h


def geom_spiral(g, sl):
    c0, c1, L = g['curvStart'], g['curvEnd'], g['length']
    n = max(3, int(L / 0.05) + 2)
    u = np.linspace(0.0, L, n)
    h = g['hdg'] + c0 * u + (c1 - c0) * u * u / (2.0 * L)
    X = g['x'] + cumulative_trapezoid(np.cos(h), u, initial=0.0)
    Y = g['y'] + cumulative_trapezoid(np.sin(h), u, initial=0.0)
    x = np.interp(sl, u, X)
    y = np.interp(sl, u, Y)
    hh = g['hdg'] + c0 * sl + (c1 - c0) * sl * sl / (2.0 * L)
    return x, y, hh


def geom_poly3(g, sl):
    a, b, c, d, L, h0 = g['a'], g['b'], g['c'], g['d'], g['length'], g['hdg']
    umax = L * 1.5 + 1.0
    n = max(3, int(umax / 0.05) + 2)
    u = np.linspace(0.0, umax, n)
    dv = b + 2 * c * u + 3 * d * u ** 2
    arc = cumulative_trapezoid(np.sqrt(1 + dv ** 2), u, initial=0.0)
    us = np.interp(sl, arc, u)
    v = a + b * us + c * us ** 2 + d * us ** 3
    dvs = b + 2 * c * us + 3 * d * us ** 2
    x = g['x'] + us * math.cos(h0) - v * math.sin(h0)
    y = g['y'] + us * math.sin(h0) + v * math.cos(h0)
    return x, y, h0 + np.arctan(dvs)


GEOM_FN = {'line': geom_line, 'arc': geom_arc, 'spiral': geom_spiral, 'poly3': geom_poly3}


def parse_geometries(road_el):
    gs = []
    for ge in road_el.find('planView').findall('geometry'):
        g = {k: float(ge.get(k)) for k in ('s', 'x', 'y', 'hdg', 'length')}
        child = list(ge)[0]
        g['kind'] = child.tag
        for k, v in child.attrib.items():
            g[k] = float(v)
        gs.append(g)
    return gs


def ref_line(geoms, s_arr):
    """도로 s 배열 → 기준선 x, y, hdg"""
    x = np.zeros_like(s_arr)
    y = np.zeros_like(s_arr)
    h = np.zeros_like(s_arr)
    starts = np.array([g['s'] for g in geoms])
    idx = np.searchsorted(starts, s_arr + 1e-9, side='right') - 1
    idx = np.clip(idx, 0, len(geoms) - 1)
    for k, g in enumerate(geoms):
        m = idx == k
        if not m.any():
            continue
        sl = np.clip(s_arr[m] - g['s'], 0.0, g['length'])
        fn = GEOM_FN.get(g['kind'])
        if fn is None:
            raise ValueError('unsupported geometry ' + g['kind'])
        xx, yy, hh = fn(g, sl)
        x[m], y[m], h[m] = xx, yy, hh
    return x, y, h


# ────────────────────────────────────────────────────────────────────────────
# 2. 도로 파싱
# ────────────────────────────────────────────────────────────────────────────
# 곡률 dh/dst 를 신뢰할 수 있는 최소 세그먼트 길이 [m].
# 섹션 끝에는 직전 점과 1 mm 떨어진 중복점이 흔한데, 그 dst 로 나누면 값이 폭발한다
# (실측: 도로 2011 섹션 경계 Δs=0.001 m → 곡률 0.1546 = 반경 6.5 m, 실제는 직선).
CURV_MIN_DS = 0.05


def _fill_nan(c):
    """NaN 을 가장 가까운 유효값으로 채운다 (앞→뒤, 뒤→앞)."""
    n = len(c)
    ok = np.isfinite(c)
    if not ok.any():
        return np.zeros(n)
    idx = np.where(ok, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)
    fwd = np.where(idx >= 0, c[np.maximum(idx, 0)], np.nan)
    idx2 = np.where(ok, np.arange(n), n)
    idx2 = np.minimum.accumulate(idx2[::-1])[::-1]
    bwd = np.where(idx2 < n, c[np.minimum(idx2, n - 1)], np.nan)
    return np.where(np.isfinite(fwd), fwd, bwd)


def smooth_curvature(cv, k=5):
    """
    곡률 점열의 **스파이크 제거**. 끝점 대체 + 중앙값 필터.

    끝점 곡률을 믿을 수 없는 이유가 둘이다:
      1) 내부 점은 앞뒤 세그먼트 헤딩의 **벡터평균**을 쓰는데 끝점만 마지막
         세그먼트 헤딩을 그대로 쓴다 → 마지막 두 점 사이 dh 가 과대해진다.
      2) 섹션 끝의 중복점(Δs ≈ 1 mm)에서 dh/dst 가 폭발한다. 그 값은
         **cv[-2] 와 cv[-1] 둘 다**에 들어가므로 한 점만 고쳐서는 안 된다.
    실측(2026-08-23): 직선 도로 2011 의 섹션 경계에서 곡률 0.1546(반경 6.5 m)이
    찍혀, 플래너가 헤어핀으로 오인하고 45 → 16 km/h 로 감속했다.

    진짜 커브는 여러 점에 걸쳐 있어(연결로 반경 4.5 m 호 ≈ 10 점) 중앙값 필터가
    그대로 보존한다. 고립된 스파이크만 사라진다.
    """
    n = len(cv)
    if n < 4:
        return np.zeros(n, dtype=np.float32) if n else cv
    c = _fill_nan(np.asarray(cv, dtype=float).copy())
    # 구성상 오염되는 자리: 앞 1점(hd[0] 이 원시 헤딩), 뒤 2점(마지막 세그먼트)
    c[0] = c[1]
    c[-1] = c[-2] = c[-3]
    if n < k:
        return c.astype(np.float32)
    pad = k // 2
    ext = np.concatenate([np.repeat(c[0], pad), c, np.repeat(c[-1], pad)])
    try:
        out = np.median(np.lib.stride_tricks.sliding_window_view(ext, k), axis=-1)
    except AttributeError:                       # numpy < 1.20
        out = np.array([np.median(ext[i:i + k]) for i in range(n)])
    return out.astype(np.float32)


def parse_pieces(el, tag, keys=('a', 'b', 'c', 'd'), skey='sOffset'):
    out = []
    for e in el.findall(tag):
        out.append((float(e.get(skey, e.get('s', 0.0))),) + tuple(float(e.get(k, 0.0)) for k in keys))
    out.sort(key=lambda p: p[0])
    return out


def parse_lane(lane_el):
    lane = {
        'id': int(lane_el.get('id')),
        'type': lane_el.get('type', 'none'),
        'width': parse_pieces(lane_el, 'width'),
        'marks': [],
        'pred': [], 'succ': [],
    }
    for rm in lane_el.findall('roadMark'):
        lane['marks'].append((float(rm.get('sOffset', 0.0)), rm.get('type', 'none'),
                              rm.get('color', 'standard'), rm.get('laneChange', 'both')))
    lane['marks'].sort(key=lambda p: p[0])
    lk = lane_el.find('link')
    if lk is not None:
        lane['pred'] = [int(p.get('id')) for p in lk.findall('predecessor')]
        lane['succ'] = [int(p.get('id')) for p in lk.findall('successor')]
    return lane


def parse_road(road_el):
    r = {
        'id': int(road_el.get('id')),
        'length': float(road_el.get('length')),
        'junction': int(road_el.get('junction', -1)),
        'geoms': parse_geometries(road_el),
        'elev': parse_pieces(road_el.find('elevationProfile'), 'elevation', skey='s')
        if road_el.find('elevationProfile') is not None else [],
        'lane_offset': parse_pieces(road_el.find('lanes'), 'laneOffset', skey='s'),
        'pred': None, 'succ': None,
        'sections': [],
        'objects': [], 'signals': [],
    }
    lk = road_el.find('link')
    if lk is not None:
        for tag in ('predecessor', 'successor'):
            e = lk.find(tag)
            if e is not None:
                r[tag[:4]] = (e.get('elementType'), int(e.get('elementId')), e.get('contactPoint'))
    for sec_el in road_el.find('lanes').findall('laneSection'):
        sec = {'s': float(sec_el.get('s')), 'lanes': {}, 'center_marks': []}
        c = sec_el.find('center')
        if c is not None:
            for ln in c.findall('lane'):
                for rm in ln.findall('roadMark'):
                    sec['center_marks'].append((float(rm.get('sOffset', 0.0)), rm.get('type', 'none'),
                                                rm.get('color', 'standard'), rm.get('laneChange', 'both')))
        sec['center_marks'].sort(key=lambda p: p[0])
        for side in ('left', 'right'):
            sd = sec_el.find(side)
            if sd is None:
                continue
            for ln in sd.findall('lane'):
                L = parse_lane(ln)
                sec['lanes'][L['id']] = L
        r['sections'].append(sec)
    r['sections'].sort(key=lambda s: s['s'])
    ob = road_el.find('objects')
    if ob is not None:
        for o in ob.findall('object'):
            r['objects'].append({
                'id': o.get('id'), 'name': o.get('name', ''), 'type': o.get('type', ''),
                's': float(o.get('s')), 't': float(o.get('t')), 'hdg': float(o.get('hdg', 0.0)),
                'length': float(o.get('length', 0.0)), 'width': float(o.get('width', 0.0)),
                'zOffset': float(o.get('zOffset', 0.0)),
            })
    sg = road_el.find('signals')
    if sg is not None:
        for s in sg.findall('signal'):
            v = s.find('validity')
            r['signals'].append({
                'id': int(s.get('id')), 'name': s.get('name', ''), 's': float(s.get('s')), 't': float(s.get('t')),
                'hOffset': float(s.get('hOffset', 0.0)), 'type': s.get('type'), 'subtype': s.get('subtype'),
                'dynamic': s.get('dynamic'), 'orientation': s.get('orientation'),
                'validity': (int(v.get('fromLane')), int(v.get('toLane'))) if v is not None else None,
            })
    return r


def parse_controllers(root):
    """
    <controller id> → 제어하는 signal id 목록.

    9910 이 주는 light_id 는 **개별 signal id 가 아니라 이 controller id** 다.
    (실측: 정지선 signal_ids=[101..106] 인 곳에서 9910 은 id=27 을 줬고,
     ctrl027 이 제어하는 신호가 101/102/105 였다)
    signal id 와 controller id 는 숫자 공간이 겹치므로 반드시 이 표로 대조해야 한다.
    """
    ctrl2sig, sig2ctrl = {}, {}
    for c in root.findall('controller'):
        if c.get('id') is None:
            continue
        cid = int(c.get('id'))
        sids = sorted(int(x.get('signalId')) for x in c.findall('control')
                      if x.get('signalId') is not None)
        ctrl2sig[cid] = sids
        for sid in sids:
            sig2ctrl.setdefault(sid, []).append(cid)
    return ctrl2sig, sig2ctrl


def parse_junctions(root):
    J = {}
    for j in root.findall('junction'):
        conns = []
        for c in j.findall('connection'):
            conns.append({
                'incoming': int(c.get('incomingRoad')), 'connecting': int(c.get('connectingRoad')),
                'contact': c.get('contactPoint'),
                'links': [(int(l.get('from')), int(l.get('to'))) for l in c.findall('laneLink')],
            })
        J[int(j.get('id'))] = conns
    return J


# ────────────────────────────────────────────────────────────────────────────
# 3. 차로 기하 샘플링
# ────────────────────────────────────────────────────────────────────────────
def section_bounds(road, i):
    s0 = road['sections'][i]['s']
    s1 = road['sections'][i + 1]['s'] if i + 1 < len(road['sections']) else road['length']
    return s0, s1


def lane_bounds_at(road, s):
    """도로 s에서 각 lane의 (t_inner, t_outer). 반환: sec_idx, {lane_id: (t_in, t_out, type)}"""
    secs = road['sections']
    i = 0
    for k, sec in enumerate(secs):
        if sec['s'] <= s + 1e-9:
            i = k
    sec = secs[i]
    ds = s - sec['s']
    tc = poly_eval(road['lane_offset'], s)
    out = {}
    t = tc
    for lid in sorted([l for l in sec['lanes'] if l < 0], reverse=True):
        w = poly_eval(sec['lanes'][lid]['width'], ds)
        out[lid] = (t, t - w, sec['lanes'][lid]['type'])
        t -= w
    t = tc
    for lid in sorted([l for l in sec['lanes'] if l > 0]):
        w = poly_eval(sec['lanes'][lid]['width'], ds)
        out[lid] = (t, t + w, sec['lanes'][lid]['type'])
        t += w
    return i, out


def lane_at_t(road, s, t, prefer_driving=True, tol=0.0):
    """도로 (s,t)에 놓인 lane id (없으면 가장 가까운 driving lane). 반환 (sec_idx, lane_id, dist_outside)"""
    i, b = lane_bounds_at(road, s)
    best = None
    for lid, (tin, tout, typ) in b.items():
        lo, hi = min(tin, tout), max(tin, tout)
        if prefer_driving and typ not in DRIVING:
            continue
        d = 0.0 if lo - tol <= t <= hi + tol else min(abs(t - lo), abs(t - hi))
        if best is None or d < best[1]:
            best = (lid, d)
    if best is None:
        return i, None, None
    return i, best[0], best[1]


def sample_road(road, ds):
    """도로 전체 기준선 샘플 (섹션 경계 포함)"""
    L = road['length']
    s = np.arange(0.0, L, ds)
    extra = [sec['s'] for sec in road['sections']] + [L]
    s = np.unique(np.concatenate([s, np.array(extra)]))
    s = s[(s >= 0) & (s <= L + 1e-9)]
    # 부동소수 오차로 생긴 거의 같은 s 제거 (중복 점 → 헤딩 계산 오류 방지)
    if len(s) > 1:
        keep = np.concatenate([[True], np.diff(s) > 1e-6])
        s = s[keep]
    x, y, h = ref_line(road['geoms'], s)
    z = poly_eval_arr(road['elev'], s) if road['elev'] else np.zeros_like(s)
    off = poly_eval_arr(road['lane_offset'], s) if road['lane_offset'] else np.zeros_like(s)
    return s, x, y, h, z, off


def compress_marks(s_travel, mark_seq):
    """샘플별 mark 튜플 → (s0, s1, type, color, lane_change_ok) 구간"""
    segs = []
    if len(mark_seq) == 0:
        return segs
    start = 0
    for i in range(1, len(mark_seq) + 1):
        if i == len(mark_seq) or mark_seq[i] != mark_seq[start]:
            typ, col, lc = mark_seq[start]
            ok = (typ == 'broken') and (lc != 'none')
            segs.append((float(s_travel[start]), float(s_travel[min(i, len(s_travel) - 1)]), typ, col, ok))
            start = i
    return segs


def build_lanes(road, ds, warnings):
    """도로 하나의 모든 driving lane 레코드 생성. 반환 dict lane_key → rec"""
    s_all, x_all, y_all, h_all, z_all, off_all = sample_road(road, ds)
    lanes = {}
    for i, sec in enumerate(road['sections']):
        s0, s1 = section_bounds(road, i)
        m = (s_all >= s0 - 1e-9) & (s_all <= s1 + 1e-9)
        if m.sum() < 2:
            # 길이 0 섹션 → 양끝 강제
            m = np.zeros_like(s_all, dtype=bool)
            k0 = int(np.argmin(np.abs(s_all - s0)))
            m[k0] = True
            k1 = min(k0 + 1, len(s_all) - 1)
            m[k1] = True
        s = s_all[m]
        x, y, h, z = x_all[m], y_all[m], h_all[m], z_all[m]
        # laneOffset 은 섹션 경계에서 불연속일 수 있음 → 섹션 끝점은 왼쪽 극한(자기 섹션 piece)으로 평가
        s_eval = s.astype(float).copy()
        if len(s_eval) >= 2 and s_eval[-1] >= s1 - 1e-9 and s1 > s0 + 1e-6:
            s_eval[-1] = s1 - 1e-6
        off = poly_eval_arr(road['lane_offset'], s_eval) if road['lane_offset'] else np.zeros_like(s)
        nx = -np.sin(h)
        ny = np.cos(h)
        dsec = s - sec['s']
        # 폭 누적 → 각 lane t 범위
        t_in = {}
        t_out = {}
        wid = {}
        t = off.copy()
        for lid in sorted([l for l in sec['lanes'] if l < 0], reverse=True):
            w = poly_eval_arr(sec['lanes'][lid]['width'], dsec)
            t_in[lid] = t.copy()
            t = t - w
            t_out[lid] = t.copy()
            wid[lid] = w
        t = off.copy()
        for lid in sorted([l for l in sec['lanes'] if l > 0]):
            w = poly_eval_arr(sec['lanes'][lid]['width'], dsec)
            t_in[lid] = t.copy()
            t = t + w
            t_out[lid] = t.copy()
            wid[lid] = w

        def marks_of(lid, ds_arr):
            if lid == 0:
                pieces = sec['center_marks']
            else:
                pieces = sec['lanes'][lid]['marks']
            if not pieces:
                return [('none', 'standard', 'both')] * len(ds_arr)
            return [piece_at(pieces, d)[1:] for d in ds_arr]

        for lid, L in sec['lanes'].items():
            if L['type'] not in DRIVING:
                continue
            tc = 0.5 * (t_in[lid] + t_out[lid])
            px = x + tc * nx
            py = y + tc * ny
            pz = z
            d = 1 if lid < 0 else -1  # 주행 방향: 음수 lane = +s
            order = np.arange(len(s)) if d == 1 else np.arange(len(s))[::-1]
            P = np.stack([px[order], py[order], pz[order]], axis=1)
            seg = np.hypot(np.diff(P[:, 0]), np.diff(P[:, 1]))
            st = np.concatenate([[0.0], np.cumsum(seg)])
            # heading (travel)
            if len(P) >= 2:
                hh_ref = np.array([wrap(a) for a in (h[order] + (0.0 if d == 1 else math.pi))])
                dx = np.diff(P[:, 0]); dy = np.diff(P[:, 1])
                seg_ok = seg > 1e-4
                seg_h = np.where(seg_ok, np.arctan2(dy, dx), np.nan)
                hd = np.empty(len(P))
                hd[:-1] = seg_h
                hd[-1] = seg_h[-1] if len(seg_h) else np.nan
                # 앞뒤 세그먼트 헤딩 평균(내부점), 없으면 기준선 헤딩
                mid = 0.5 * (seg_h[:-1] + seg_h[1:]) if len(seg_h) > 1 else np.array([])
                # 각도 평균은 벡터 평균으로
                if len(seg_h) > 1:
                    cx = np.cos(seg_h[:-1]) + np.cos(seg_h[1:]); cy = np.sin(seg_h[:-1]) + np.sin(seg_h[1:])
                    mid = np.arctan2(cy, cx)
                    hd[1:-1] = mid
                bad = ~np.isfinite(hd)
                hd[bad] = hh_ref[bad]
                hd = np.array([wrap(a) for a in hd])
                dh = np.diff(np.unwrap(hd))
                dst = np.diff(st)
                cv = np.zeros_like(st)
                with np.errstate(divide='ignore', invalid='ignore'):
                    # 너무 짧은 세그먼트(중복점)는 무효로 두고 이웃 값으로 채운다.
                    # 예전 문턱 1e-6 은 Δs=1 mm 도 통과시켜 곡률을 폭발시켰다.
                    c_mid = np.where(dst > CURV_MIN_DS, dh / dst, np.nan)
                cv[:-1] = c_mid
                cv[-1] = c_mid[-1] if len(c_mid) else 0.0
                cv = smooth_curvature(cv)         # 섹션 경계 스파이크 제거
            else:
                hd = np.array([wrap(h[order][0] + (0.0 if d == 1 else math.pi))])
                cv = np.zeros(1)
            inner_id = lid + 1 if lid < 0 else lid - 1  # 중앙 쪽 이웃 (0이면 center)
            own = marks_of(lid, dsec)
            inner = marks_of(inner_id, dsec)
            own_t = [own[k] for k in order]
            inner_t = [inner[k] for k in order]
            # 운전자 기준: left = 중앙 쪽(inner), right = 바깥(own)   (좌/우 lane 모두 동일)
            left_mark = compress_marks(st, inner_t)
            right_mark = compress_marks(st, own_t)
            # 옆 차로 (같은 방향 driving)
            def nb(idn):
                if idn in sec['lanes'] and sec['lanes'][idn]['type'] in DRIVING and (idn * lid > 0):
                    return (road['id'], i, idn)
                return None
            left_nb = nb(inner_id) if inner_id != 0 else None
            right_nb = nb(lid - 1 if lid < 0 else lid + 1)
            rec = {
                'road': road['id'], 'sec': i, 'lane_id': lid, 'dir': d, 'type': L['type'],
                'junction': road['junction'],
                'pts': P.astype(np.float32), 's': st.astype(np.float32),
                'road_s': s[order].astype(np.float32),
                'hdg': hd.astype(np.float32), 'curv': cv.astype(np.float32),
                'width': wid[lid][order].astype(np.float32),
                'length': float(st[-1]),
                'left_mark': left_mark, 'right_mark': right_mark,
                'left_nb': left_nb, 'right_nb': right_nb,
                'left_is_center': left_nb is None,
                'next': [], 'prev': [],
                'stop_lines': [], 'signals': [], 'crosswalks': [], 'crosswalk_warn': [],
                'yield_marks': [], 'arrows': [], 'speed_marks': [], 'markings': [],
                'speed_limit': None, 'school_zone': False, 'speed_src': None,
                # 링크용 원본
                '_succ_ids': L['succ'], '_pred_ids': L['pred'],
            }
            lanes[(road['id'], i, lid)] = rec
    return lanes


def road_s_to_travel(rec, road_s):
    """차로 rec에서 도로 s → 주행 s"""
    rs = rec['road_s']
    st = rec['s']
    if rec['dir'] == 1:
        return float(np.interp(road_s, rs, st))
    return float(np.interp(road_s, rs[::-1], st[::-1]))


# ────────────────────────────────────────────────────────────────────────────
# 4. 연결 (섹션 간 / 도로 간 / 교차로)
# ────────────────────────────────────────────────────────────────────────────
def entering_lanes(lanes, road_id, contact, roads):
    """road_id 에 contact(start/end)로 진입할 때 후보 lane 들 (주행 시작점이 그 끝인 lane)"""
    road = roads[road_id]
    if contact == 'start':
        sec = 0
        want_dir = 1
    else:
        sec = len(road['sections']) - 1
        want_dir = -1
    return [k for k, r in lanes.items() if r['road'] == road_id and r['sec'] == sec and r['dir'] == want_dir]


def nearest_start(lanes, cands, end_pt, end_hdg, tol=2.0):
    best = None
    for k in cands:
        p = lanes[k]['pts'][0]
        d = math.hypot(p[0] - end_pt[0], p[1] - end_pt[1])
        dh = abs(wrap(float(lanes[k]['hdg'][0]) - end_hdg))
        if d < tol and dh < math.radians(60):
            sc = d + dh
            if best is None or sc < best[1]:
                best = (k, sc)
    return best[0] if best else None


def link_lanes(lanes, roads, junctions, warnings):
    stats = collections.Counter()
    for key, rec in lanes.items():
        road = roads[rec['road']]
        i, lid, d = rec['sec'], rec['lane_id'], rec['dir']
        nsec = len(road['sections'])
        end_pt = rec['pts'][-1]
        end_hdg = float(rec['hdg'][-1])
        nxt = []
        # 4-1 같은 도로 안 다음 섹션
        if (d == 1 and i < nsec - 1) or (d == -1 and i > 0):
            j = i + 1 if d == 1 else i - 1
            ids = rec['_succ_ids'] if d == 1 else rec['_pred_ids']
            for lid2 in ids:
                k2 = (road['id'], j, lid2)
                if k2 in lanes:
                    nxt.append(k2)
            if not nxt:
                cands = [k for k, r in lanes.items() if r['road'] == road['id'] and r['sec'] == j and r['dir'] == d]
                k2 = nearest_start(lanes, cands, end_pt, end_hdg, tol=1.5)
                if k2:
                    nxt.append(k2)
                    stats['sec_geom'] += 1
                else:
                    stats['sec_dead'] += 1
        else:
            # 4-2 도로 끝 → link
            link = road['succ'] if d == 1 else road['pred']
            if link is None:
                stats['no_link'] += 1
            elif link[0] == 'road':
                r2, contact = link[1], link[2]
                if r2 not in roads:
                    stats['missing_road'] += 1
                else:
                    cands = entering_lanes(lanes, r2, contact, roads)
                    ids = rec['_succ_ids'] if d == 1 else rec['_pred_ids']
                    sec2 = 0 if contact == 'start' else len(roads[r2]['sections']) - 1
                    for lid2 in ids:
                        k2 = (r2, sec2, lid2)
                        if k2 in lanes and k2 in cands:
                            nxt.append(k2)
                    if not nxt:
                        k2 = nearest_start(lanes, cands, end_pt, end_hdg, tol=2.5)
                        if k2:
                            nxt.append(k2)
                            stats['road_geom'] += 1
                        else:
                            stats['road_dead'] += 1
            elif link[0] == 'junction':
                jid = link[1]
                for c in junctions.get(jid, []):
                    if c['incoming'] != road['id']:
                        continue
                    for fr, to in c['links']:
                        if fr != lid:
                            continue
                        r2 = c['connecting']
                        if r2 not in roads:
                            continue
                        sec2 = 0 if c['contact'] == 'start' else len(roads[r2]['sections']) - 1
                        k2 = (r2, sec2, to)
                        if k2 in lanes:
                            p = lanes[k2]['pts'][0]
                            dd = math.hypot(p[0] - end_pt[0], p[1] - end_pt[1])
                            if dd > 4.0:
                                warnings.append(f'junction {jid}: lane {key}->{k2} gap {dd:.1f}m')
                            nxt.append(k2)
                if not nxt:
                    stats['junction_dead'] += 1
        rec['next'] = sorted(set(nxt))
    for key, rec in lanes.items():
        for k2 in rec['next']:
            lanes[k2]['prev'].append(key)
    for rec in lanes.values():
        rec['prev'] = sorted(set(rec['prev']))
        del rec['_succ_ids']
        del rec['_pred_ids']
    return stats


# ────────────────────────────────────────────────────────────────────────────
# 5. 객체 / 신호 배치
# ────────────────────────────────────────────────────────────────────────────
STOP_NAME = 'Rm_StopLine'
CROSSWARN_NAME = 'Rm_Warning_Crosswalk'
YIELD_NAME = 'Rm_Give_Way'
SPEED_MARKS = {'RM_517_50.flt': (50, False), 'roadmark_speed_30.flt': (30, False), 'RM_518.flt': (30, True)}
ARROWS = {'RM_537_ST.flt': 'S', 'RM_537_LT.flt': 'L', 'RM_537_RT.flt': 'R', 'RM_538_SRT.flt': 'SR',
          'RM_538_SLT.flt': 'SL', 'RM_539_LUT.flt': 'LU'}


def dir_from_hdg(hdg):
    return 1 if abs(wrap(hdg)) < math.pi / 2 else -1


def lanes_of_dir_at(lanes, road, s, d):
    i, b = lane_bounds_at(road, s)
    out = []
    for lid, (tin, tout, typ) in b.items():
        if typ in DRIVING and ((lid < 0 and d == 1) or (lid > 0 and d == -1)):
            k = (road['id'], i, lid)
            if k in lanes:
                out.append(k)
    return out


def assign_objects(lanes, roads, warnings, sig2ctrl=None):
    stats = collections.Counter()
    for road in roads.values():
        # 정지선 클러스터 (방향별, s 3m 이내)
        clusters = []  # [dir, s_list, t_list]
        for o in road['objects']:
            if STOP_NAME in o['name']:
                d = dir_from_hdg(o['hdg'])
                for c in clusters:
                    if c[0] == d and abs(c[1][0] - o['s']) < 3.0:
                        c[1].append(o['s'])
                        c[2].append(o['t'])
                        break
                else:
                    clusters.append([d, [o['s']], [o['t']]])
        road['_stop_clusters'] = []
        for d, ss, ts in clusters:
            sc = float(np.mean(ss))
            i, b = lane_bounds_at(road, min(max(sc, 0.0), road['length']))
            hit = []
            for lid, (tin, tout, typ) in b.items():
                if typ not in DRIVING or (lid < 0) != (d == 1):
                    continue
                lo, hi = min(tin, tout) - 1.6, max(tin, tout) + 1.6
                if any(lo <= t <= hi for t in ts):
                    hit.append((road['id'], i, lid))
            if not hit:
                hit = lanes_of_dir_at(lanes, road, min(max(sc, 0.0), road['length']), d)
            entry = {'dir': d, 's': sc, 'lanes': [k for k in hit if k in lanes], 'signal_ids': []}
            road['_stop_clusters'].append(entry)
            stats['stop_clusters'] += 1

        # 신호등
        for sg in road['signals']:
            s = min(max(sg['s'], 0.0), road['length'])
            explicit = sg['validity'] is not None
            if explicit:
                lo, hi = sorted(sg['validity'])
                d = 1 if lo < 0 else -1
                i, b = lane_bounds_at(road, s)
                lk = [(road['id'], i, lid) for lid in range(lo, hi + 1)
                      if lid in b and b[lid][2] in DRIVING and (road['id'], i, lid) in lanes]
                if not lk:  # validity가 driving 이 아니면 방향 전체
                    lk = lanes_of_dir_at(lanes, road, s, d)
                    stats['sig_validity_fallback'] += 1
            else:
                h = sg['hOffset'] % (2 * math.pi)
                if h < 0.5 or h > 2 * math.pi - 0.5:
                    d = 1
                elif abs(h - math.pi) < 0.5:
                    d = -1
                else:
                    d = 1 if math.copysign(1.0, sg['t']) < 0 else -1
                    warnings.append(f"signal {sg['id']} road {road['id']}: side ambiguous (hOffset {h:.2f}), used t sign → dir {d}")
                lk = lanes_of_dir_at(lanes, road, s, d)
            # 정지선: 같은 방향 클러스터 중 가장 가까운 것 (20m 이내)
            best = None
            for c in road['_stop_clusters']:
                if c['dir'] == d and abs(c['s'] - s) < 20.0:
                    if best is None or abs(c['s'] - s) < abs(best['s'] - s):
                        best = c
            stop_s = best['s'] if best else s
            if best:
                best['signal_ids'].append(sg['id'])
            else:
                stats['sig_no_stopline'] += 1
            for k in lk:
                rec = lanes[k]
                rec['signals'].append({'id': sg['id'], 'stop_s': road_s_to_travel(rec, stop_s),
                                       'explicit': explicit, 'type': sg['type'], 'subtype': sg['subtype']})
            stats['signals'] += 1

        # 정지선 → lane 기록 (신호 id 포함)
        for c in road['_stop_clusters']:
            for k in c['lanes']:
                rec = lanes[k]
                sids = sorted(c['signal_ids'])
                # 9910 light_id 대조용. 한 정지선의 신호가 두 controller 로
                # 갈리는 곳이 많다(직진/좌회전) — 그래서 리스트다.
                cids = sorted({cid for sid in sids for cid in (sig2ctrl or {}).get(sid, [])})
                rec['stop_lines'].append({'s': road_s_to_travel(rec, c['s']),
                                          'signal_ids': sids, 'controller_ids': cids})

        # 나머지 객체: t 로 lane 찾기
        side_speed = {1: [], -1: []}
        for o in road['objects']:
            nm = o['name']
            if STOP_NAME in nm:
                continue
            is_pelican = o['type'] == 'crosswalk'
            if not (CROSSWARN_NAME in nm or YIELD_NAME in nm or nm in SPEED_MARKS or nm in ARROWS
                    or nm.startswith('RM_') or nm.startswith('Rm_') or is_pelican):
                continue
            s = min(max(o['s'], 0.0), road['length'])
            if is_pelican:
                # 도로 전폭 → 양방향 모든 driving lane
                half = max(o['length'], 3.0) / 2.0
                for d in (1, -1):
                    for k in lanes_of_dir_at(lanes, road, s, d):
                        rec = lanes[k]
                        a, b2 = road_s_to_travel(rec, s - half), road_s_to_travel(rec, s + half)
                        rec['crosswalks'].append((min(a, b2), max(a, b2), 'pelican'))
                stats['pelican'] += 1
                continue
            i, lid, dout = lane_at_t(road, s, o['t'], prefer_driving=True)
            if lid is None:
                continue
            k = (road['id'], i, lid)
            if k not in lanes:
                continue
            rec = lanes[k]
            st = road_s_to_travel(rec, s)
            if dout is not None and dout > 1.0:
                # 차로 밖 (중앙선 근처 등) → 헤딩으로 방향 잡아 그 방향 전체 lane 에 표기하진 않고 스킵 기록
                stats['obj_outside_lane'] += 1
            if CROSSWARN_NAME in nm:
                rec['crosswalk_warn'].append(st)
            elif YIELD_NAME in nm:
                rec['yield_marks'].append(st)
            elif nm in SPEED_MARKS:
                v, school = SPEED_MARKS[nm]
                rec['speed_marks'].append((st, v, nm))
                side_speed[rec['dir']].append((v, school, nm))
            elif nm in ARROWS:
                rec['arrows'].append((st, ARROWS[nm]))
            else:
                rec['markings'].append((st, nm))
        # 제한속도: 도로+방향 단위 (표시 없으면 None → 런타임에서 이전 값 유지)
        road['speed_by_dir'] = {}
        for d, lst in side_speed.items():
            if not lst:
                road['speed_by_dir'][d] = (None, False, None)
                continue
            vals = [v for v, sc, nm in lst]
            school = any(sc for v, sc, nm in lst)
            v = min(vals)
            src = ','.join(sorted(set(nm for _, _, nm in lst)))
            road['speed_by_dir'][d] = (v, school, src)
        both = [x for x in road['speed_by_dir'].values() if x[0] is not None]
        road['speed_limit'] = min(x[0] for x in both) if both else None
        road['school_zone'] = any(x[1] for x in both)
        for k, rec in lanes.items():
            if rec['road'] != road['id']:
                continue
            v, school, src = road['speed_by_dir'][rec['dir']]
            if v is None and road['speed_limit'] is not None:
                # 반대편에만 표시가 있으면 보수적으로 같은 값 적용 (src에 표시)
                v, school, src = road['speed_limit'], road['school_zone'], 'other_side'
            rec['speed_limit'], rec['school_zone'], rec['speed_src'] = v, school, src
    # 정지선 뒤 추정 횡단보도 존 (8m) — 실제 횡단보도는 xodr 에 없음
    ZONE = 8.0
    for k, rec in lanes.items():
        for sl in rec['stop_lines']:
            s0 = sl['s']
            s1 = s0 + ZONE
            rec['crosswalks'].append((s0, min(s1, rec['length']), 'inferred'))
            rem = s1 - rec['length']
            if rem > 0:
                for k2 in rec['next']:
                    lanes[k2]['crosswalks'].append((0.0, min(rem, lanes[k2]['length']), 'inferred'))
    for rec in lanes.values():
        rec['stop_lines'].sort(key=lambda d: d['s'])
        rec['signals'].sort(key=lambda d: d['stop_s'])
        rec['crosswalks'].sort()
        rec['crosswalk_warn'].sort()
        rec['yield_marks'].sort()
        rec['arrows'].sort()
        rec['speed_marks'].sort()
        rec['markings'].sort()
    return stats


# ────────────────────────────────────────────────────────────────────────────
# 6. 메인
# ────────────────────────────────────────────────────────────────────────────
def build(xodr_path, ds=0.5):
    t0 = time.time()
    warnings = []
    root = ET.parse(xodr_path).getroot()
    roads = {}
    for rel in root.findall('road'):
        r = parse_road(rel)
        roads[r['id']] = r
    junctions = parse_junctions(root)
    ctrl2sig, sig2ctrl = parse_controllers(root)
    print(f'[parse] roads={len(roads)} junctions={len(junctions)}  {time.time()-t0:.1f}s')

    lanes = {}
    for r in roads.values():
        lanes.update(build_lanes(r, ds, warnings))
    print(f'[geom ] driving lanes={len(lanes)}  points={sum(len(l["pts"]) for l in lanes.values())}  {time.time()-t0:.1f}s')

    st = link_lanes(lanes, roads, junctions, warnings)
    print('[link ]', dict(st))
    so = assign_objects(lanes, roads, warnings, sig2ctrl)
    print('[objs ]', dict(so))

    # KD-tree 용 전역 인덱스
    lane_keys = sorted(lanes.keys())
    key_idx = {k: i for i, k in enumerate(lane_keys)}
    P, LI, PI, H = [], [], [], []
    for k in lane_keys:
        rec = lanes[k]
        n = len(rec['pts'])
        P.append(rec['pts'][:, :2])
        LI.append(np.full(n, key_idx[k], dtype=np.int32))
        PI.append(np.arange(n, dtype=np.int32))
        H.append(rec['hdg'])
    kd_pts = np.concatenate(P).astype(np.float32)
    kd_lane = np.concatenate(LI)
    kd_i = np.concatenate(PI)
    kd_hdg = np.concatenate(H).astype(np.float32)

    # 도로 요약
    roads_out = {}
    for rid, r in roads.items():
        roads_out[rid] = {
            'length': r['length'], 'junction': r['junction'], 'pred': r['pred'], 'succ': r['succ'],
            'sections': [sec['s'] for sec in r['sections']],
            'speed_limit': r.get('speed_limit'), 'school_zone': r.get('school_zone', False),
            'speed_by_dir': r.get('speed_by_dir', {}),
            'stop_clusters': [{'dir': c['dir'], 's': c['s'], 'signal_ids': sorted(c['signal_ids']),
                               'controller_ids': sorted({cid for sid in c['signal_ids']
                                                         for cid in (sig2ctrl or {}).get(sid, [])})}
                              for c in r.get('_stop_clusters', [])],
            'lanes': [k for k in lane_keys if k[0] == rid],
        }
    signals_out = {}
    for k, rec in lanes.items():
        for sg in rec['signals']:
            e = signals_out.setdefault(sg['id'], {'road': rec['road'], 'lanes': [], 'type': sg['type'],
                                                  'subtype': sg['subtype'],
                                                  'controller_ids': sig2ctrl.get(sg['id'], [])})
            e['lanes'].append(k)

    # 통계
    dead = [k for k, r in lanes.items() if not r['next']]
    heads = [k for k, r in lanes.items() if not r['prev']]
    n_pts = int(len(kd_pts))
    graph = {
        'meta': {
            'source': xodr_path, 'built': time.strftime('%Y-%m-%d %H:%M:%S'), 'ds': ds,
            'n_roads': len(roads), 'n_junctions': len(junctions), 'n_lanes': len(lanes), 'n_points': n_pts,
            'total_lane_length_m': float(sum(r['length'] for r in lanes.values())),
            'lanes_without_next': len(dead), 'lanes_without_prev': len(heads),
            'assumptions': [
                'signal side: validity > hOffset(0:+s lanes<0, pi:-s lanes>0) > t sign',
                'speed limit from road-marking objects, per road & direction; None → runtime carries previous value',
                'RM_518 assumed school-zone 30 (verify texture)',
                'crosswalk zones after stop lines are INFERRED (8 m); only 2 real crosswalk objects (pelican)',
                'left_mark = mark toward road center, right_mark = outer mark (driver frame, both lane sides)',
                '9910 light_id == xodr <controller> id (NOT signal id); stop_lines carry controller_ids',
            ],
            'n_controllers': len(ctrl2sig),
            'n_signals_controlled': len(sig2ctrl),
            'warnings': warnings[:200],
        },
        'lanes': lanes, 'lane_keys': lane_keys, 'roads': roads_out, 'junctions': junctions, 'signals': signals_out,
        # 9910 light_id(=controller id) ↔ xodr signal id
        'controllers': ctrl2sig, 'signal_to_controller': sig2ctrl,
        'kd_pts': kd_pts, 'kd_lane': kd_lane, 'kd_i': kd_i, 'kd_hdg': kd_hdg,
    }
    print(f'[done ] lanes={len(lanes)} pts={n_pts} dead_ends={len(dead)} heads={len(heads)} warnings={len(warnings)}  {time.time()-t0:.1f}s')
    return graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('xodr')
    ap.add_argument('-o', '--out', default='lane_graph.pkl')
    ap.add_argument('--ds', type=float, default=0.5)
    a = ap.parse_args()
    g = build(a.xodr, a.ds)
    with open(a.out, 'wb') as f:
        pickle.dump(g, f, protocol=4)
    import os
    print(f'saved {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)')
    if g['meta']['warnings']:
        print('--- warnings (first 15) ---')
        for w in g['meta']['warnings'][:15]:
            print('  ', w)


if __name__ == '__main__':
    main()
