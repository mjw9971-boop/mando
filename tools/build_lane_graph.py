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
    sidewalk_left_m / sidewalk_right_m : 차로 중심선→보도 안쪽 경계 횡거리 프로파일
                 (운전자 기준 좌/우, s 정렬 배열, 그 쪽에 보도 없으면 None.
                  좌측은 반대 차선 건너편 보도까지 포함 — 월선 후 보도 침범 판정용)

전제/주의 (README 참고):
  - 신호등 방향: validity 태그 > hOffset(0→우측차로 +s, π→좌측차로 -s) > t 부호  (이 맵에서 검증됨)
  - 제한속도: 표준 <speed> 필드가 없어서 노면표시 객체(RM_517_50 / roadmark_speed_30 / RM_518)로 도로 단위 추정
  - RM_518 = 어린이보호구역 속도제한 표시(30)로 가정 → 텍스처 확인 필요
  - 실제 횡단보도 객체는 2개(pelican)뿐. 교차로 횡단보도는 xodr에 없음 → 정지선 뒤 8m를 'inferred' 존으로 표시
"""
import argparse, json, math, os, pickle, sys, time, collections
import xml.etree.ElementTree as ET
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree

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


_MAP_CFG = None


def map_cfg(reload=False):
    """params.yaml map.* — xodr 파싱 상수의 단일 출처.

    이 도구는 params 없이 단독 실행될 수 있어야 하므로(현장 롤백·최소 환경),
    읽기 실패는 기본값 폴백으로 삼킨다. build_route.candidates_cfg 와 같은 관례.
    """
    global _MAP_CFG
    if _MAP_CFG is None or reload:
        try:
            import pathlib as _pl, sys as _sy
            _sy.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
            from vtd_adapter.config import load_params_yaml
            mc = load_params_yaml().get('map') or {}
            _MAP_CFG = (bool(mc.get('laneoffset_stable_order', True)),
                        float(mc.get('laneoffset_tie_tol_m', 1e-3)))
        except Exception:                            # noqa: BLE001 — 독립 실행 폴백
            _MAP_CFG = (True, 1e-3)
    return _MAP_CFG


_RED_CFG = None
_SPEED_CFG = None


def red_zone_cfg(reload=False):
    """params.yaml red_zone.* — 붉은 노면(감속 구간) 포장 정점 파일과 매핑 상수.

    map_cfg 와 같은 관례로 params 없이도 동작한다 (현장 롤백·최소 환경).
    """
    global _RED_CFG
    if _RED_CFG is None or reload:
        d = {}
        try:
            import pathlib as _pl, sys as _sy
            _sy.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
            from vtd_adapter.config import load_params_yaml
            d = dict(load_params_yaml().get('red_zone') or {})
        except Exception:                            # noqa: BLE001 — 독립 실행 폴백
            d = {}
        _RED_CFG = {
            'verts_json': str(d.get('verts_json', 'data/red_surface_verts.json')),
            'k': int(d.get('kd_candidates', 24)),
            'half_margin_m': float(d.get('half_margin_m', 0.30)),
            'gap_m': float(d.get('span_gap_m', 3.0)),
            'min_verts': int(d.get('min_verts', 20)),
            'min_len_m': float(d.get('min_len_m', 1.0)),
        }
    return _RED_CFG


def speed_cfg(reload=False):
    """params.yaml red_zone.roadmark_30_as_limit — 30 노면표시를 제한속도로 쓸지.

    대회 규칙상 **붉은 노면만** 보호구역이고, 붉지 않은 도로는 30 표시가 있어도
    도로 기본 제한(50)이 적용된다. 이 맵의 30 표시(roadmark_speed_30 · RM_518)는
    붉은 포장과 일치하지 않는다 — 실측 2026-09-05: speed_limit=30 인 386 차로
    14,548 m 중 **10,237 m 가 붉지 않다**. 그래서 기본을 false 로 둔다.
    true 면 이전 동작(30 표시 = 제한속도 30).
    """
    global _SPEED_CFG
    if _SPEED_CFG is None or reload:
        try:
            import pathlib as _pl, sys as _sy
            _sy.path.insert(0, str(_pl.Path(__file__).resolve().parent.parent))
            from vtd_adapter.config import load_params_yaml
            sc = load_params_yaml().get('red_zone') or {}
            _SPEED_CFG = bool(sc.get('roadmark_30_as_limit', False))
        except Exception:                            # noqa: BLE001 — 독립 실행 폴백
            _SPEED_CFG = False
    return _SPEED_CFG


def parse_pieces(el, tag, keys=('a', 'b', 'c', 'd'), skey='sOffset'):
    """(s, a, b, c, d) 조각 리스트를 s 오름차순으로.

    **같은 s 의 레코드는 xodr 문서 순서를 지킨다** (map.laneoffset_stable_order).
    list.sort 는 안정 정렬이라 키가 *정확히* 같으면 문서 순서가 보존되지만,
    이 맵의 laneOffset 은 같은 지점을 e-12 ~ e-9 만큼 다른 s 로 적어 놓아
    정렬이 문서 순서를 뒤집는다. poly_eval / poly_eval_arr 은 "s 이하의 마지막
    조각"을 고르므로, 뒤로 밀린 앞 구간 종결자(b=0)가 다음 구간의 램프(b≠0)를
    덮어쓴다 -- 도로 2803 에서 1.195 m, 1603 에서 2.700 m 짜리 중심선 오차.
    자세한 근거는 params.yaml map: 주석.
    """
    stable, tol = map_cfg()
    out = []
    for i, e in enumerate(el.findall(tag)):
        out.append((float(e.get(skey, e.get('s', 0.0))),) + tuple(float(e.get(k, 0.0)) for k in keys))
    if not stable:
        out.sort(key=lambda p: p[0])                 # 이전 동작
        return out
    # s 로 정렬하되, 같은 s 묶음 안에서는 문서 순서(원래 인덱스)를 보조키로.
    order = sorted(range(len(out)), key=lambda i: out[i][0])
    ranked = []                                      # (묶음 대표 s, 문서 인덱스)
    for n, i in enumerate(order):
        if n and out[i][0] - out[order[n - 1]][0] < tol:
            ranked.append((ranked[-1][0], i))        # 앞 묶음에 합류
        else:
            ranked.append((out[i][0], i))
    ranked.sort()
    return [out[i] for _s, i in ranked]


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

        # 보도 프로파일용: 이 섹션의 sidewalk 차로들 (t_in/t_out 은 위에서 전 타입 계산됨)
        sw_ids = [l for l in sec['lanes'] if sec['lanes'][l]['type'] == 'sidewalk']

        def sidewalk_profile(tc, side_sign):
            """차로 중심선 tc(샘플별)에서 도로 프레임 side_sign(+1 = +t 쪽) 방향으로
            가장 가까운 보도의 **안쪽 경계**까지 횡거리 [m]. 그 쪽에 보도 없으면 None.
            사이에 낀 차로(border/none/parking 등)의 폭은 t 좌표 차에 자동 포함된다."""
            best = None
            for l in sw_ids:
                c_sw = 0.5 * (t_in[l] + t_out[l])
                if (float(np.mean(c_sw)) - float(np.mean(tc))) * side_sign <= 0:
                    continue                                   # 반대쪽 보도
                edge = np.minimum(t_in[l], t_out[l]) if side_sign > 0 else np.maximum(t_in[l], t_out[l])
                d = (edge - tc) * side_sign
                if best is None or float(np.mean(d)) < float(np.mean(best)):
                    best = d                                   # 같은 쪽에 여럿이면 가까운 것
            return best

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
            # 운전자 기준 좌/우 → 도로 프레임 t 부호 (d=+1 이면 +t 가 왼쪽)
            sw_l = sidewalk_profile(tc, 1 if d == 1 else -1)
            sw_r = sidewalk_profile(tc, -1 if d == 1 else 1)
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
                'sidewalk_left_m': None if sw_l is None else sw_l[order].astype(np.float32),
                'sidewalk_right_m': None if sw_r is None else sw_r[order].astype(np.float32),
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


def _lane_t_span(road, s):
    """도로 s 에서 전체 차로가 덮는 t 범위 (양끝 ±1.6 여유). 차로가 없으면 None."""
    _, b = lane_bounds_at(road, min(max(s, 0.0), road['length']))
    if not b:
        return None
    lo = min(min(tin, tout) for tin, tout, _ in b.values()) - 1.6
    hi = max(max(tin, tout) for tin, tout, _ in b.values()) + 1.6
    return lo, hi


def stopline_damaged(road, o) -> bool:
    """이 정지선 레코드의 s/t 가 깨졌는가.

    조건은 **절대 t** 다: s == 0.0 이면서 |t| 가 그 도로 어떤 차로의 t 범위도
    벗어나면 s 와 t 가 서로 뒤바뀐 것이다 (|t| ≈ 도로 길이).
    |t|/L 비율을 조건으로 쓰면 안 된다 — 짧은 도로의 정상 정지선을 오탐하고
    (L=6.09 에 t=7.5 가 3차선 중심), L=13~20 m 손상 레코드는 |t|/L 이 1.03 까지
    벌어져 놓친다 (2026-09-01 실측: 비율 0.99~1.01 창은 24개 중 19개만 잡았다).
    """
    if o['s'] != 0.0:
        return False
    span = _lane_t_span(road, 0.0)
    return span is not None and not (span[0] <= o['t'] <= span[1])


def repair_stopline_dir(lanes, road, covered, warnings):
    """손상 정지선의 진행방향 — hdg 는 쓰지 않는다.

    hdg 는 이 레코드들에서 못 믿는다: road 2819 는 주행차로가 전부 dir=+1 인
    일방통행인데 손상 6건의 hdg 가 모두 dir=-1 을 가리켰다. 그래서 폴백
    (lanes_of_dir_at)까지 공집합이 돼 정지선이 그래프에서 사라졌고, controller
    217 적신호를 무감속 통과했다 (2026-09-01).

    판정 우선순위 — 0·1 만으로 실측 24/24 가 결정된다:
      0) 이미 정상 정지선이 덮은 방향은 배제 (손상분은 반대편이다)
      1) 일방통행 배제 (그 방향에 주행차로가 없으면 반대로 확정)
    2·3 은 **교차 검증 전용**이다. 불일치해도 보정하지 않고 경고만 남긴다:
      2) 같은 방향 신호의 road_s (s≈L 이면 dir=+1)
      3) L-4.00 화살표 t 부호 (음수 t = 우측차로 = dir=+1)
    """
    L = road['length']
    avail = {d: bool(lanes_of_dir_at(lanes, road, min(max(L, 0.0), L), d)) or
                bool(lanes_of_dir_at(lanes, road, 0.0, d)) for d in (1, -1)}
    d0 = None
    if covered == {-1} and avail[1]:
        d0 = 1
    elif covered == {1} and avail[-1]:
        d0 = -1
    d1 = 1 if (avail[1] and not avail[-1]) else (-1 if (avail[-1] and not avail[1]) else None)
    d = d0 if d0 is not None else d1
    # ── 교차 검증 (보정하지 않는다) ──────────────────────────────────────
    sig_s = sorted({sg['s'] for sg in road['signals']})
    at_L = any(abs(s - L) < 0.5 for s in sig_s)
    at_0 = any(abs(s) < 0.5 for s in sig_s)
    d2 = 1 if (at_L and not at_0) else (-1 if (at_0 and not at_L) else None)
    arrow_t = [o['t'] for o in road['objects']
               if o['name'] in ARROWS and abs(o['s'] - (L - 4.0)) < 1.5]
    neg = sum(1 for x in arrow_t if x < 0)
    d3 = (1 if neg > len(arrow_t) - neg else
          (-1 if len(arrow_t) - neg > neg else None)) if arrow_t else None
    for tag, dx in (('신호 road_s', d2), ('화살표 t', d3)):
        if d is not None and dx is not None and dx != d:
            warnings.append(
                f"stopline road {road['id']}: 방향 교차검증 불일치 — 판정 {d:+d} 인데 "
                f"{tag} 는 {dx:+d} (보정은 판정대로, 확인 필요)")
    return d


# 신호 재귀속 (2026-09-02) — 대향(far-side) 설치 신호가 접근로가 아니라 교차로
# 내부 연결로의 객체로 기록된 경우를 바로잡는다.
SIG_REHOME_RADIUS_M = 40.0    # 정지선 ↔ 대향 신호 거리 상한 (실측 24.4 / 26.3 m)
SIG_REHOME_HDG_DEG = 10.0     # 절대방향 vs 진행방향 허용 오차 (실측 0.9 / 0.4 deg)


def _road_xy(road, s, t=0.0):
    """도로 (s, t) 의 월드 좌표와 그 s 의 기준선 heading."""
    s = min(max(s, 0.0), road['length'])
    xx, yy, hh = ref_line(road['geoms'], np.array([s], dtype=float))
    return (float(xx[0] - t * math.sin(hh[0])), float(yy[0] + t * math.cos(hh[0])), float(hh[0]))


def rehome_signal(sg, road, roads):
    """hOffset 이 자기 도로 진행방향과 어긋난 신호를 실제 관장 접근로로 되돌린다.

    근거 (2026-09-02 실측): 정상 신호 640개에서 **신호 절대방향(도로 hdg + hOffset)
    은 그 신호가 관장하는 차량의 진행방향과 같다** — hOffset 0 → +s 진행,
    pi → -s 진행. 이 규칙이 자기 도로와 어긋나는(hOffset = 3pi/2) 신호는 지도
    전체에서 6개뿐이다: road 556 의 30·31·34, road 791 의 965·966·967. 둘 다
    교차로 내부 연결로이고, 절대방향은 각각 road 30 dir+1(오차 0.4 deg)·
    road 190 dir-1(오차 0.9 deg) 의 진행방향과 일치한다. 그 두 접근로는
    정지선만 있고 신호가 0개였다 — 정확히 뒤집힌 짝이다.

    t 부호 폴백으로 두면 신호가 교차로 내부 차로에 붙어 정지선을 못 만나고
    (sig_no_stopline), 상류 접근로는 무신호로 남는다. road 2819 가 controller
    217 적신호를 무감속 통과한 것과 같은 실패다 (2026-09-01).

    후보는 **정확히 1개일 때만** 채택한다. 오귀속은 엉뚱한 접근로에 적신호를
    만들어 급정거를 부르므로, 애매하면 재귀속하지 않고 현행 폴백에 맡긴다.

    반환: (host_road, cluster, dir, lane_keys, hdg_err_deg, dist_m) 또는 None.
    """
    sx, sy, _ = _road_xy(road, sg['s'], sg['t'])
    _, _, rh = _road_xy(road, sg['s'], 0.0)
    sig_hdg = wrap(rh + sg['hOffset'])
    tol = math.radians(SIG_REHOME_HDG_DEG)
    out = []
    for cand in roads.values():
        for c in cand.get('_stop_clusters') or ():
            if not c['lanes']:
                continue
            cx, cy, ch = _road_xy(cand, c['s'], 0.0)
            th = wrap(ch + (0.0 if c['dir'] == 1 else math.pi))
            err = wrap(sig_hdg - th)
            if abs(err) > tol:
                continue
            dx, dy = sx - cx, sy - cy
            dist = math.hypot(dx, dy)
            if dist > SIG_REHOME_RADIUS_M:
                continue
            # 대향 설치이므로 신호는 정지선보다 **진행 전방**에 선다.
            if dx * math.cos(th) + dy * math.sin(th) <= 0.0:
                continue
            out.append((cand, c, c['dir'], list(c['lanes']), math.degrees(err), dist))
    return out[0] if len(out) == 1 else None


def assign_objects(lanes, roads, warnings, sig2ctrl=None, synth_stopline=True):
    stats = collections.Counter()
    repaired = []          # (road, s_orig, t_orig, dir_hdg, s_true, dir_fixed)
    reassigned = []        # 대향 신호 재귀속 내역 (무음 보정 금지)
    synthesized = []       # 합성 정지선 내역 (무음 보정 금지)
    for road in roads.values():
        # 정지선 클러스터 (방향별, s 3m 이내)
        # 좌표가 깨진 정지선 보정 (2026-09-01). 종방향은 s_true = min(|t|, L) 로
        # 복원된다 — 13 road 전부에서 도로 끝을 집고, 같은 방향 신호 s 와 일치한다.
        # **횡방향(t)은 복원 불가**다: s=0.0 이라 t 정보가 파일에 없고 부호도 차로
        # 좌우와 무관하다(road 2819: 실제 +2/-4, 기록 +3/-3). t 가 어떤 차로에도
        # 안 들어가므로 아래 배정에서 폴백(방향 전체 차로)을 타게 된다 — 이미
        # 383개 클러스터 중 186개가 그렇게 동작하던 경로다.
        stop_objs = [o for o in road['objects'] if STOP_NAME in o['name']]
        dmg_ids = {id(o) for o in stop_objs if stopline_damaged(road, o)}
        rep_d = None
        if dmg_ids:
            covered = {dir_from_hdg(o['hdg']) for o in stop_objs if id(o) not in dmg_ids}
            rep_d = repair_stopline_dir(lanes, road, covered, warnings)
            if rep_d is None:
                warnings.append(f"stopline road {road['id']}: 손상 {len(dmg_ids)}건의 "
                                f"방향을 정할 수 없다 — 보정하지 않고 원본대로 둔다")
                stats['stop_repair_undecided'] += len(dmg_ids)

        clusters = []  # [dir, s_list, t_list]
        for o in stop_objs:
            if id(o) in dmg_ids and rep_d is not None:
                d = rep_d
                s_o = min(abs(o['t']), road['length'])
                repaired.append({'road': road['id'], 'road_length': round(road['length'], 4),
                                 's_orig': round(o['s'], 4), 't_orig': round(o['t'], 4),
                                 'dir_hdg': dir_from_hdg(o['hdg']),
                                 's_true': round(s_o, 4), 'dir_fixed': d})
                stats['stop_repaired'] += 1
            else:
                d = dir_from_hdg(o['hdg'])
                s_o = o['s']
            for c in clusters:
                if c[0] == d and abs(c[1][0] - s_o) < 3.0:
                    c[1].append(s_o)
                    c[2].append(o['t'])
                    break
            else:
                clusters.append([d, [s_o], [o['t']]])
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
                stats['stop_lane_fallback'] += 1
            # ── 위생 검사 (관측만 — 배정은 위에서 이미 끝났다) ──────────────
            # 1차 판정은 **절대 t**: 정지선 t 가 그 도로 어떤 차로의 t 범위도
            # 벗어나면 좌표가 깨진 것이다. |t|/L 비율을 1차로 쓰면 짧은 도로에서
            # 정상 정지선을 오탐하고(L=6.09 에 t=7.5 인 3차선 중심), 반대로
            # L=13~20 m 손상 레코드는 |t|/L 이 1.03 까지 벌어져 놓친다
            # (2026-09-01 실측: 비율 0.99~1.01 창은 24개 중 19개만 잡았다).
            lo_all = min(min(tin, tout) for tin, tout, _ in b.values()) - 1.6
            hi_all = max(max(tin, tout) for tin, tout, _ in b.values()) + 1.6
            out_t = [t for t in ts if not (lo_all <= t <= hi_all)]
            if out_t:
                L = road['length']
                ratio = max(abs(t) / L for t in out_t) if L > 0 else float('inf')
                # s==0 + |t|≈L 은 s/t 가 서로 뒤바뀐 손상 패턴이다. 종방향은
                # s_true=min(|t|,L) 로 복원되지만 횡방향(t)은 파일에 없다.
                # s 는 클러스터 **평균**이라 정상 레코드와 병합되면 0 에서 밀린다
                # (road 1043: 정상 s=0.15 3개 + 손상 s=0.0 2개 → 평균 0.09).
                # 그래서 원본 s 목록(ss)을 본다.
                kind = ('s/t 손상 의심' if any(s0 == 0.0 for s0 in ss)
                        and any(abs(abs(t) - L) < 0.05 * L for t in out_t)
                        else 't 범위 이탈')
                warnings.append(
                    f"stopline road {road['id']} dir {d}: {kind} — "
                    f"t={[round(x, 2) for x in out_t]} 가 차로 t 범위 "
                    f"[{lo_all:.2f}, {hi_all:.2f}] 밖 (s={sc:.2f}, L={L:.2f}, |t|/L={ratio:.4f})")
                stats['stop_t_out_of_range'] += 1
            entry = {'dir': d, 's': sc, 'lanes': [k for k in hit if k in lanes], 'signal_ids': []}
            if not entry['lanes']:
                # 여기서 조용히 사라진다 — 아래 'lane 기록' 루프가 c['lanes'] 를
                # 순회하므로 빈 리스트면 어느 차로에도 안 남고 통계에도 안 잡혔다.
                # 실측(2026-09-01): road 2819 는 일방통행(주행차로 전부 dir=+1)인데
                # 손상 레코드의 hdg 가 dir=-1 을 가리켜 폴백까지 공집합이 됐고,
                # controller 217 적신호를 무감속 통과했다.
                warnings.append(
                    f"stopline road {road['id']} dir {d} s {sc:.2f}: 배정된 차로가 없다 "
                    f"— 이 정지선은 그래프에서 사라진다 (그 방향 주행차로 "
                    f"{len(lanes_of_dir_at(lanes, road, min(max(sc, 0.0), road['length']), d))}개)")
                stats['stop_no_lane'] += 1
            road['_stop_clusters'].append(entry)
            stats['stop_clusters'] += 1

    # ── 신호등 1/3: 방향·귀속 도로 해석 ───────────────────────────────────
    # 정지선 클러스터가 **모든 도로에서** 끝난 뒤에 돈다. 재귀속(rehome_signal)이
    # 다른 도로의 클러스터를 봐야 하기 때문이다.
    resolved = []          # (sg, host_road, dir, lane_keys, host_cluster|None)
    for road in roads.values():
        for sg in road['signals']:
            s = min(max(sg['s'], 0.0), road['length'])
            explicit = sg['validity'] is not None
            host, host_c, lk = road, None, None   # host = 정지선을 찾을 도로
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
                    # hOffset 이 자기 도로 진행방향과 어긋난다 = 대향 신호가 엉뚱한
                    # 도로에 기록됐다. 실제 관장 접근로로 되돌린다.
                    rh = rehome_signal(sg, road, roads)
                    if rh is None:
                        d = 1 if math.copysign(1.0, sg['t']) < 0 else -1
                        warnings.append(f"signal {sg['id']} road {road['id']}: side ambiguous "
                                        f"(hOffset {h:.2f}), 재귀속 후보가 유일하지 않다 → t 부호로 dir {d}")
                    else:
                        host, host_c, d, lk, err, dist = rh
                        reassigned.append({'signal': sg['id'], 'from_road': road['id'],
                                           'from_s': round(sg['s'], 4), 'from_t': round(sg['t'], 4),
                                           'hOffset': round(sg['hOffset'], 4),
                                           'to_road': host['id'], 'to_dir': d,
                                           'stop_s': round(host_c['s'], 4),
                                           'lanes': [list(k) for k in lk],
                                           'hdg_err_deg': round(err, 2), 'dist_m': round(dist, 2)})
                        stats['signal_reassigned'] += 1
                if lk is None:
                    lk = lanes_of_dir_at(lanes, road, s, d)
            resolved.append((sg, host, d, lk, host_c, explicit))

    # ── 신호등 2/3: 접근로 정지선 합성 ────────────────────────────────────
    # 신호가 있는데 그 방향에 정지선 클러스터가 **0개**인 접근로에, 지도 자신의
    # 규약대로 정지선을 만든다.
    #
    # 근거 (2026-09-02 실측): 정상 매칭된 신호 625개에서 (정지선 s - 신호 s) 는
    # 최빈 +0.15 (302건) / -0.15 (227건), p1 -0.158 · p50 +0.146 · p99 +0.531,
    # |offset| > 1 m 은 3건뿐이다. 신호 자신은 중앙값 기준 도로 끝에 정확히 선다.
    # 즉 이 지도에서 "신호 s ± 0.15" 는 규약이고, 합성 위치 오차는 최대 0.5 m 다.
    #
    # 대상은 road 1928(dir-1) · 2575 · 2806 · 3142 의 15개 신호다. 넷 다 하류
    # 연결로에 폭이 맞는 정지선이 하나씩 있지만 14~29 m 떨어져 있다 — 규약의
    # 40~190배 밖이라 접근로 정지선이 아니다. 그 정지선을 끌어오는 안(B2)은
    # 채택하지 않았다: 별개 지물(교차로 내부 정지선)일 가능성을 배제하지 못했고,
    # 같은 형태가 junction 42·77 에 3건 더 있는데 그쪽 접근로는 이미 자기
    # 정지선을 갖고 있어 끌어오면 중복이 된다.
    #
    # 교차로 내부(junction != -1)에는 만들지 않는다 — 정지선을 이미 지난 지점이다.
    if synth_stopline:
        for sg, host, d, lk, host_c, _ in resolved:
            if host_c is not None or host['junction'] != -1 or not lk:
                continue
            if any(c['dir'] == d for c in host['_stop_clusters']):
                continue                      # 그 방향에 정지선이 있으면 건드리지 않는다
            L = host['length']
            s_syn = min(max(min(max(sg['s'], 0.0), L) - 0.15 * d, 0.0), L)
            entry = {'dir': d, 's': s_syn, 'lanes': list(lk), 'signal_ids': []}
            host['_stop_clusters'].append(entry)
            end = L if d == 1 else 0.0
            synthesized.append({'road': host['id'], 'dir': d, 's': round(s_syn, 4),
                                'signal': sg['id'], 'signal_s': round(sg['s'], 4),
                                'road_length': round(L, 4),
                                'road_end_gap_m': round(abs(sg['s'] - end), 4),
                                'lanes': [list(k) for k in lk]})
            stats['stop_synthesized'] += 1

    # ── 신호등 3/3: 정지선 매칭 + 차로 기록 ───────────────────────────────
    for sg, host, d, lk, host_c, explicit in resolved:
        # 정지선: 같은 방향 클러스터 중 가장 가까운 것 (20m 이내).
        # 재귀속된 신호는 host_c 가 이미 정해져 있다 (sg['s'] 는 원래 도로 좌표라
        # host 에서 다시 재면 안 된다).
        s = min(max(sg['s'], 0.0), host['length'])
        best = host_c
        if best is None:
            for c in host['_stop_clusters']:
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

    for road in roads.values():

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
        # red_zone.roadmark_30_as_limit 이 false 면 **값이 30 인 표시는 제한속도로
        # 쓰지 않는다** (speed_cfg 주석 참조 — 붉은 포장과 일치하지 않는다).
        # RM_517_50 같은 다른 값은 그대로다. 이 도로에 30 표시밖에 없으면
        # speed_by_dir 가 None 이 되어 런타임 carry 가 앞 값(대개 50)을 물고 간다.
        # speed_marks 원본은 지우지 않는다 — 진단·되돌리기 근거로 남긴다.
        use30 = speed_cfg()
        road['speed_by_dir'] = {}
        for d, lst in side_speed.items():
            if not use30:
                lst = [t for t in lst if t[0] != 30]
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

    # 신호는 붙었는데 정지선이 없는 차로 — 정지선이 없으면 route.collect_stops 가
    # traffic_lights 를 만들지 않아 next_traffic_light 이 None 이 되고, 제어기의
    # 적신호 IDM 이 **호출조차 되지 않는다**. 채점기의 stop_ctrl_ids 도 비어
    # 항목 7 판정이 함께 죽는다. 그래서 별도 지표로 요약에 올린다.
    if repaired:
        # 보정 내역은 **전건**을 남긴다 — 무음 보정 금지. 어느 레코드를 어떻게
        # 옮겼는지 사후에 되짚을 수 없으면 오보정을 발견할 방법이 없다.
        print(f'[fix  ] 정지선 좌표 보정 {len(repaired)}건 '
              f"(road {len({r['road'] for r in repaired})}개)")
        for r in sorted(repaired, key=lambda x: (x['road'], x['t_orig'])):
            print(f"         road {r['road']:<5} s {r['s_orig']:>6.2f}→{r['s_true']:>7.2f}  "
                  f"dir {r['dir_hdg']:+d}→{r['dir_fixed']:+d}  "
                  f"(t={r['t_orig']:+.2f} 복원 불가, 방향 전체 차로에 배정)")
    if synthesized:
        # 무음 보정 금지 — xodr 에 없는 객체를 만들었으므로 전건을 남긴다.
        print(f'[synth] 접근로 정지선 합성 {len(synthesized)}건 '
              f"(road {len({r['road'] for r in synthesized})}개)")
        for r in sorted(synthesized, key=lambda x: (x['road'], x['dir'])):
            print(f"         road {r['road']:<5} dir {r['dir']:+d} s={r['s']:>8.2f} "
                  f"(신호 {r['signal']} s={r['signal_s']:.2f}, 도로끝까지 "
                  f"{r['road_end_gap_m']:.2f}m, 차로 {len(r['lanes'])}개)")
    if reassigned:
        # 무음 보정 금지 — 어느 신호를 어디로 옮겼는지 전건을 남긴다.
        print(f'[sig  ] 대향 신호 재귀속 {len(reassigned)}건')
        for r in sorted(reassigned, key=lambda x: (x['from_road'], x['signal'])):
            print(f"         sig {r['signal']:<5} road {r['from_road']} -> road {r['to_road']} "
                  f"dir {r['to_dir']:+d}  정지선 s={r['stop_s']:.2f}  "
                  f"(방향오차 {r['hdg_err_deg']:+.2f}deg, 거리 {r['dist_m']:.1f}m, 차로 {len(r['lanes'])}개)")
    orphan = [k for k, rec in lanes.items()
              if rec['signals'] and not rec['stop_lines']]
    stats['lanes_signal_no_stopline'] = len(orphan)
    for k in sorted(orphan)[:20]:
        warnings.append(f'lane {k}: 신호 {len(lanes[k]["signals"])}개가 붙었으나 정지선 없음 '
                        f'— 이 차로에서는 적신호 감속이 발동하지 않는다')
    return stats, repaired, reassigned, synthesized


# ────────────────────────────────────────────────────────────────────────────
# 5-b. 붉은 노면 (감속 구간) — osgb 텍스처에서 뽑은 월드 정점 → 차로 s 구간
# ────────────────────────────────────────────────────────────────────────────
def _seg_project(P, i, x, y):
    """폴리라인 P 의 i-1·i 두 구간 중 (x,y) 에 가까운 쪽으로 투영.

    반환 (구간 인덱스 j, 구간 내 비율 u, 부호 없는 횡거리). LaneGraph.project 와
    같은 방식이다 — 여기서 다시 쓰는 이유는 build 시점에 LaneGraph 객체가 아직
    없어서다 (그래프를 지금 만드는 중이다).
    """
    best = None
    for j in (i - 1, i):
        if j < 0 or j + 1 >= len(P):
            continue
        ax, ay = float(P[j][0]), float(P[j][1])
        bx, by = float(P[j + 1][0]), float(P[j + 1][1])
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        if L2 <= 1e-12:
            continue
        u = ((x - ax) * dx + (y - ay) * dy) / L2
        u = min(1.0, max(0.0, u))
        px, py = ax + u * dx, ay + u * dy
        d = math.hypot(x - px, y - py)
        if best is None or d < best[2]:
            best = (j, u, d)
    return best


def apply_red_spans(lanes, warnings):
    """붉은 노면 정점 → rec['red_spans'] = [(s0, s1), …] (주행 s).

    **이것이 감속 구간의 단일 출처다.** 붉은 포장은 xodr 에 없고 osgb 텍스처
    (StyleSrfBikeway, R131 G59 B59)로만 있어 정점을 미리 뽑아 두었다. 이름은
    Bikeway 지만 도로 **전폭**을 덮는다 — 실측 2026-09-05: 정점 64,914개의
    최근접 차로 중심선 거리 p50 1.45 m 로 차로 반폭과 같은 축이고, 98.4% 가
    차로 폭 안에 든다.

    **차로 단위**로 판정한다. road 2312 는 dir −1 의 lane 2·3·4 만 붉고 반대
    방향은 아니다 — road 단위로 묶으면 통행방향 하나가 통째로 잘못 감속한다.

    잡음 걸러내기: 인접 차로 끝점에 스치는 정점이 s=0 또는 s=length 에 파편을
    만든다. 실측으로 실체 124구간 4,252.9 m 대 파편 174개 합 4.2 m 로 1000배
    갈리므로(min_verts 20 · min_len_m 1.0) 임계 선택에 민감하지 않다.

    기존 `school_zone` bool 은 `bool(red_spans)` 로 남긴다 — 하위호환.
    파일이 없으면 아무것도 하지 않고 경고만 남긴다 (이전 동작).
    """
    cfg = red_zone_cfg()
    path = cfg['verts_json']
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path)
    for rec in lanes.values():
        rec['red_spans'] = []
    if not os.path.exists(path):
        warnings.append(f'붉은 노면 정점 파일이 없다: {path} — red_spans 를 만들지 않는다 '
                        f'(school_zone 은 노면표시 기반 이전 값 유지)')
        return {'file': path, 'found': False, 'verts': 0, 'lanes': 0, 'spans': 0,
                'length_m': 0.0, 'dropped_spans': 0, 'dropped_length_m': 0.0}
    with open(path, encoding='utf-8') as f:
        V = json.load(f).get('verts') or []
    if not V:
        warnings.append(f'붉은 노면 정점 파일이 비어 있다: {path}')
        return {'file': path, 'found': True, 'verts': 0, 'lanes': 0, 'spans': 0,
                'length_m': 0.0, 'dropped_spans': 0, 'dropped_length_m': 0.0}
    V = np.asarray(V, dtype=float)

    keys = list(lanes.keys())
    idx_of = {k: i for i, k in enumerate(keys)}
    P, LI, PI = [], [], []
    for k in keys:
        pts = lanes[k]['pts']
        P.append(pts[:, :2])
        LI.append(np.full(len(pts), idx_of[k], dtype=np.int32))
        PI.append(np.arange(len(pts), dtype=np.int32))
    kd_pts = np.concatenate(P)
    kd_lane = np.concatenate(LI)
    kd_i = np.concatenate(PI)
    tree = cKDTree(kd_pts)
    _d, nn = tree.query(V, k=cfg['k'], workers=-1)

    hit = {}
    for vi in range(len(V)):
        x, y = V[vi]
        seen = set()
        for i in np.atleast_1d(nn[vi]):
            i = int(i)
            if i >= len(kd_lane):
                continue
            key = keys[kd_lane[i]]
            if key in seen:
                continue
            seen.add(key)
            rec = lanes[key]
            pr = _seg_project(rec['pts'], int(kd_i[i]), x, y)
            if pr is None:
                continue
            j, u, d = pr
            sv = float(rec['s'][j] + u * (rec['s'][j + 1] - rec['s'][j]))
            hw = 0.5 * float(np.interp(sv, rec['s'], rec['width']))
            if d <= hw + cfg['half_margin_m'] and 0.0 <= sv <= rec['length']:
                hit.setdefault(key, []).append(sv)

    n_span = n_drop = 0
    len_span = len_drop = 0.0
    for key, ss in hit.items():
        ss = np.sort(np.asarray(ss))
        runs, a, b = [], ss[0], ss[0]
        for v in ss[1:]:
            if v - b > cfg['gap_m']:
                runs.append((a, b)); a = v
            b = v
        runs.append((a, b))
        keep = []
        for s0, s1 in runs:
            n = int(((ss >= s0) & (ss <= s1)).sum())
            if n < cfg['min_verts'] or (s1 - s0) < cfg['min_len_m']:
                n_drop += 1; len_drop += float(s1 - s0)
                continue
            # 차로 길이를 넘지 않게 클램프 — round(,3) 이 length 를 5e-4 만큼
            # 넘길 수 있고, 소비자(제어기·채점기)는 s ≤ length 를 전제한다.
            keep.append((max(0.0, round(float(s0), 3)),
                         min(float(lanes[key]['length']), round(float(s1), 3))))
            n_span += 1; len_span += float(s1 - s0)
        lanes[key]['red_spans'] = keep

    for rec in lanes.values():
        rec['school_zone'] = bool(rec['red_spans'])

    n_lanes = sum(1 for r in lanes.values() if r['red_spans'])
    return {'file': path, 'found': True, 'verts': int(len(V)), 'lanes': n_lanes,
            'spans': n_span, 'length_m': round(len_span, 1),
            'dropped_spans': n_drop, 'dropped_length_m': round(len_drop, 2)}


# ────────────────────────────────────────────────────────────────────────────
# 6. 메인
# ────────────────────────────────────────────────────────────────────────────
def build(xodr_path, ds=0.5, synth_stopline=True):
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
    so, repairs, resigs, synths = assign_objects(lanes, roads, warnings, sig2ctrl,
                                                 synth_stopline=synth_stopline)
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

    # 붉은 노면 (감속 구간) — 노면표시가 아니라 **포장 정점**이 단일 출처다.
    # 반드시 제한속도 부여 뒤에 온다: school_zone 을 red_spans 로 덮어쓴다.
    red_stats = apply_red_spans(lanes, warnings)
    print(f"[red  ] verts={red_stats['verts']} lanes={red_stats['lanes']} "
          f"spans={red_stats['spans']} len={red_stats['length_m']:.1f}m "
          f"(파편 {red_stats['dropped_spans']}개 {red_stats['dropped_length_m']:.1f}m 제외)"
          if red_stats['found'] else f"[red  ] 정점 파일 없음 — {red_stats['file']}")

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
                'school_zone comes from red_surface_verts.json (red pavement), NOT RM_518',
                'roadmark 30 marks (roadmark_speed_30 / RM_518) are ignored as speed limit '
                'unless speed.roadmark_30_as_limit',
                'crosswalk zones after stop lines are INFERRED (8 m); only 2 real crosswalk objects (pelican)',
                'left_mark = mark toward road center, right_mark = outer mark (driver frame, both lane sides)',
                '9910 light_id == xodr <controller> id (NOT signal id); stop_lines carry controller_ids',
                'sidewalk_left_m/right_m: driver-frame lateral distance from lane center to nearest sidewalk inner edge (None if absent on that side; left crosses opposite carriageway)',
            ],
            # 정지선 좌표 보정 내역 — 무음 보정 금지. 어느 레코드를 어떻게 옮겼는지
            # 산출물 안에 남겨야 오보정을 사후에 찾을 수 있다.
            'stop_repairs': repairs,
            # 대향 신호 재귀속 내역 (2026-09-02) — 교차로 내부 연결로에 잘못 기록된
            # 신호를 실제 접근로로 되돌린 건. junction_ctrl_map.json 은 이 결과에서
            # 파생되므로 그래프를 바꾸면 함께 재생성해야 한다.
            'signal_reassigned': resigs,
            # 합성 정지선 내역 (2026-09-02) — xodr 에 없는 객체를 만든 유일한 지점이다.
            # --no-synth-stopline 으로 끄면 이전 동작(정지선 없음)으로 돌아간다.
            'stop_synthesized': synths,
            'stop_synth_enabled': bool(synth_stopline),
            # 붉은 노면 (감속 구간) 반영 내역 — 파일 유무·건수·길이를 산출물에 남긴다.
            'red_zone': red_stats,
            'roadmark_30_as_limit': bool(speed_cfg()),
            'stop_repaired': so.get('stop_repaired', 0),
            'stop_repair_undecided': so.get('stop_repair_undecided', 0),
            'lanes_signal_no_stopline': so.get('lanes_signal_no_stopline', 0),
            'sig_no_stopline': so.get('sig_no_stopline', 0),
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
    ap.add_argument('--no-synth-stopline', action='store_true',
                    help='신호는 있는데 정지선이 없는 접근로에 정지선을 합성하지 않는다')
    a = ap.parse_args()
    g = build(a.xodr, a.ds, synth_stopline=not a.no_synth_stopline)
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
