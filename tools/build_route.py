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
                   차선변경 hop 은 평행 차로라 증가분 0 (실주행거리와 일치시킨다)
    lengths      : [float ...]
    total_length : float
    waypoints    : [(x,y) ...]
    waypoint_s   : [float ...]           각 경유점의 경로 누적거리
    finish_xy    : [x, y]                CSV 마지막 행 원본 좌표 = 종료선.
                   scoring.finish_xy 가 null 이면 kr_rules._resolve_stop_s 와
                   score.py 가 이 값을 자동으로 쓴다 (제어·채점 단일 출처)
    events       : [{kind, s, lane, s_in_lane, ...}]
                   kind = turn_left / turn_right / lane_change_left / lane_change_right
                   lane_change 는 window_s0/window_s1 (경로 누적거리, 점선 구간) 포함
                   창 시작점은 laneSection 경계를 넘어 최대한 앞으로 당긴다
탐색 규칙: 차로 길이 = 비용, 차선변경 = +25m 비용 (점선 구간이 있을 때만 허용), 막다른 차로 자동 회피
"""
import argparse, collections as _collections, contextlib, heapq, io, math, pickle, sys
import numpy as np
import pathlib as _pathlib, sys as _sys
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent.parent))
from vtd_adapter.lanegraph import LaneGraph, wrap

LC_PENALTY = 25.0
# 차선변경 회랑(연속 점선 길이)이 이 거리를 못 채우면 전이를 끝낼 수 없다.
# route.py 의 LC_TRANSITION_M 과 같은 축 — 창이 짧아 LC 가 실패하면 헤딩오차·조향
# 포화·도로이탈로 이어진다 (2026-08-21 실사고: 창 6.1 m, 헤딩오차 46°, courseRespawn).
LC_MIN_CORRIDOR_M = 25.0
# 부족분 1 m 당 비용 [m 환산]. 금지가 아니라 **비싸게** 매긴다 — 유일한 길이면
# 여전히 고를 수 있어야 경로 자체가 실패하지 않는다. 20 이면 회랑 1.5 m 짜리가
# 우회 495 m 와 맞먹어, 대안이 있으면 사실상 안 고른다.
LC_SHORT_PENALTY_PER_M = 20.0


def lc_cost(lg, key, side):
    """차선변경 비용 [m 환산] — 회랑이 전이거리를 못 채울수록 급증."""
    short = max(0.0, LC_MIN_CORRIDOR_M - lg.dashed_corridor_m(key, side))
    return LC_PENALTY + LC_SHORT_PENALTY_PER_M * short
# 같은 거리 층 안에서 목표 차로를 고를 때의 가중치 [비용/m] — 경유점에 가까운 쪽 우선
TARGET_DIST_W = 5.0


# 후보 수집이 개수 기반이던 시절의 k. candidates_ball_query_enable=false 로
# 되돌릴 때만 쓴다 (현장 롤백용) — 이 값이 왜 부족한지는 candidates() 주석 참고.
CANDIDATES_K = 40

_TAPER_CFG = None


def taper_cfg(reload=False):
    """params.yaml route.taper_* + vehicle.width — 소멸(테이퍼) 차로 판정의 단일 출처.

    소멸 차로 = 끝 폭 < vehicle.width. vtd_adapter/route.py 의 taper_blend 와
    tools/score.py 의 차로유지 제외가 같은 기준을 쓴다 — 셋이 어긋나면 "경로는
    피하지 않는데 채점은 제외" 같은 틈이 생긴다.
      taper_penalty_enable  탐색(dijkstra)에서 소멸 차로 진입에 벌점을 줄지
      taper_penalty_m       그 벌점 [m 환산]
    dijkstra 는 경유점마다 여러 번 불리므로 한 번 읽고 캐시한다.
    """
    global _TAPER_CFG
    if _TAPER_CFG is None or reload:
        from vtd_adapter.config import load_params_yaml
        cfg = load_params_yaml()
        rc = cfg.get('route') or {}
        _TAPER_CFG = (bool(rc.get('taper_penalty_enable', False)),
                      float(rc.get('taper_penalty_m', 0.0)),
                      float(cfg['vehicle']['width']))
    return _TAPER_CFG


def is_taper_lane(lg, key, veh_width=None) -> bool:
    """끝 폭이 차폭 미만으로 소멸하는 차로인가 (route.py taper_blend · score.py 와 같은 기준)."""
    if veh_width is None:
        veh_width = taper_cfg()[2]
    return lg.width_at(key, lg.length(key)) < veh_width


_TURN_CONSTRAINT = None


def turn_constraint_on(reload=False) -> bool:
    """params.yaml route.turn_lane_constraint_enable — 회전 차로 **제약**.

    가산점(turn_lane_bias_m)이 아니라 후보 제외다. false = 이전 동작.
    """
    global _TURN_CONSTRAINT
    if _TURN_CONSTRAINT is None or reload:
        from vtd_adapter.config import load_params_yaml
        _TURN_CONSTRAINT = bool((load_params_yaml().get('route') or {})
                                .get('turn_lane_constraint_enable', False))
    return _TURN_CONSTRAINT


def arrow_allows(lg, key, turn: str):
    """노면 화살표가 이 차로에서 `turn`('L'/'S'/'R')을 허용하나. 화살표 없으면 None.

    **laneLink 는 쓰면 안 된다.** junction connection 의 laneLink 는 노면 표시보다
    관대하다 — 실측(2026-09-09, 진입 차로 552개 중 화살표 보유 472개):
    laneLink 와 화살표가 같은 것 327, **laneLink 가 더 관대한 것 69**,
    화살표가 더 관대한 것 76. 그래서 "회전 방향이 laneLink 에 없으면 제외" 는
    회전 차로 위반 19건 중 **한 건도 못 거른다** (전부 laneLink 에는 있다).
    예: (146,0,1) 화살표 L(좌회전 전용)인데 laneLink 는 L·S·R 전부 있다.

    표기는 조합 문자열이다 — 'SR'(직진+우회전) · 'LU'(좌회전+유턴) · 'SL'.
    """
    a = lg.lanes[key].get('arrows') or []
    if not a:
        return None
    return any(turn in t for _s, t in a)


def _same_dir_siblings(lg, key):
    """같은 방향 이웃 전부 (자기 포함) — lg.neighbor 만 쓴다."""
    out = [key]
    for side in ('left', 'right'):
        k = key
        for _ in range(8):
            k = lg.neighbor(k, side)
            if k is None or k in out:
                break
            out.append(k)
    return out


def turn_lane_blocked(lg, key, turn: str) -> bool:
    """회전 차로 제약 — 이 차로에서 `turn` 이 **금지**되나.

    참이 되는 조건은 둘 다여야 한다:
      1. 이 차로의 노면 화살표가 `turn` 을 허용하지 않는다.
      2. **같은 방향 이웃 중 허용하는 차로가 있다.**

    2번이 안전망이다 — 접근로 전체에 그 회전 화살표가 하나도 없으면 지도의
    데이터 공백이므로(실측: road 62·100 의 우회전 등) 막으면 경로가 통째로
    불가능해진다. 그때는 이전 동작 그대로 통과시킨다.
    """
    ok = arrow_allows(lg, key, turn)
    if ok is None or ok:
        return False
    return any(arrow_allows(lg, k, turn) for k in _same_dir_siblings(lg, key)
               if k != key)


_TURN_BIAS = None
_TURN_KIND: dict = {}
_SIDE_N: dict = {}


def turn_bias_m(reload=False) -> float:
    """params.yaml route.turn_lane_bias_m — 회전 방향 차로 선호 가중치 [m 환산].

    0 = 끔(이전 동작). 다음이 우회전이면 진입 차로에서 **오른쪽에 남은 차로 수**
    만큼, 좌회전이면 왼쪽에 남은 수만큼 비용을 더한다. dijkstra 가 경유점마다
    여러 번 불리므로 한 번 읽고 캐시한다.
    """
    global _TURN_BIAS
    if _TURN_BIAS is None or reload:
        from vtd_adapter.config import load_params_yaml
        _TURN_BIAS = float((load_params_yaml().get('route') or {})
                           .get('turn_lane_bias_m', 0.0))
    return _TURN_BIAS


def connector_turn(lg, key) -> str | None:
    """연결로의 회전 방향 — 시작·끝 헤딩 차이. 교차로 차로가 아니면 None.

    노면 화살표(lanes[k]['arrows'])가 아니라 **기하**로 판정한다: 화살표는
    비교차로 차로에만 있고(지도 전체 1773개 중 696개), 연결로에는 없다.
    """
    if key in _TURN_KIND:
        return _TURN_KIND[key]
    r = lg.lanes[key]
    out = None
    if r.get('junction', -1) != -1:
        h = r.get('hdg')
        if h is not None and len(h) >= 2:
            d = math.degrees((float(h[-1]) - float(h[0]) + math.pi)
                             % (2.0 * math.pi) - math.pi)
            if abs(d) <= 135.0:
                out = 'left' if d > 25.0 else ('right' if d < -25.0 else None)
    _TURN_KIND[key] = out
    return out


def lanes_on_side(lg, key, side: str) -> int:
    """key 에서 side 로 남은 같은 방향 차로 수 (0 = 그쪽 끝 차로)."""
    ck = (key, side)
    if ck in _SIDE_N:
        return _SIDE_N[ck]
    n, cur = 0, key
    while n <= 6:
        nb = lg.neighbor(cur, side)
        if nb is None or nb == key:
            break
        n += 1
        cur = nb
    _SIDE_N[ck] = n
    return n


_CAND_CFG = None


def candidates_cfg(reload=False):
    """params.yaml route.candidates_* — 후보 수집 방식의 단일 출처.

    load_params_yaml 은 부를 때마다 YAML 을 다시 파싱한다. candidates() 는
    경유점마다 여러 번 불리므로 여기서 한 번만 읽고 캐시한다.
    """
    global _CAND_CFG
    if _CAND_CFG is None or reload:
        try:
            from vtd_adapter.config import load_params_yaml
            rc = load_params_yaml().get('route') or {}
            _CAND_CFG = (bool(rc.get('candidates_ball_query_enable', True)),
                         int(rc.get('candidates_max_points', 5000)))
        except Exception:                                # noqa: BLE001 — 독립 실행 폴백
            _CAND_CFG = (True, 5000)
    return _CAND_CFG


_START_OVERRIDE = None


def start_override_cfg(reload=False):
    """params.yaml route.start_lane_ball_override_enable — 출발 차로 우회 스위치.

    false 면 lg.locate() 결과를 그대로 쓴다 (2d8a7e1 이전 동작). candidates 스위치와
    따로 두는 이유: 후보 수집 전환과 출발 차로 우회는 독립된 판단이라, 현장에서
    한쪽만 되돌려야 할 수 있다.
    """
    global _START_OVERRIDE
    if _START_OVERRIDE is None or reload:
        try:
            from vtd_adapter.config import load_params_yaml
            rc = load_params_yaml().get('route') or {}
            _START_OVERRIDE = bool(rc.get('start_lane_ball_override_enable', True))
        except Exception:                                # noqa: BLE001 — 독립 실행 폴백
            _START_OVERRIDE = True
    return _START_OVERRIDE


def candidates(lg, x, y, radius, yaw=None, ball=None, max_points=None):
    """경유점 근처 후보 (lane_key, s, dist)

    **반경 안의 kd 점을 전부** 본다. 개수 기반(kd.query(k=N))은 경유점이 어떤
    차로 중심선 위에 얹히면 그 차로 점들이 N 을 다 채워 인접 차로·연결로가
    후보에서 통째로 빠진다. kd 샘플 간격이 0.5 m 라 k=40 은 20 m 어치일 뿐이다.

    2026-09-03 실측 (data/lane_graph.pkl, ds 0.5 m, radius 8 m):
      · 반경 안 점 수 중앙 106 / p99 253 / **최대 330** (교차로에서 최대)
      · k=40 이 실제로 도달하는 거리는 경유점 대부분에서 3.4 ~ 5.4 m
      · 대회장 CSV seq 9 는 정답 차로 (836,0,-1) 가 5.02 m 인데 k=40 도달이
        4.84 m — 0.18 m 차이로 잘려 "seq 8 -> seq 9 경로 없음" 이 났다.
        seq 8->9 는 교차로 짝 구간이라 차선변경이 금지돼, 잘려나간 그 차로가
        유일한 연결이었다.
    k=200 으로 키우는 건 같은 병의 다른 크기일 뿐이다 (최악 330 > 200).
    반경 안을 전부 보면 아래 `dist > radius` 컷과 의미가 같아지고, 점 몰림에
    영향받지 않는다.

    ball / max_points 는 테스트용 명시 오버라이드다. None 이면 params 를 읽는다.
    """
    if ball is None or max_points is None:
        cfg_ball, cfg_max = candidates_cfg()
        ball = cfg_ball if ball is None else ball
        max_points = cfg_max if max_points is None else max_points

    if ball:
        ii = np.asarray(lg.kd.query_ball_point((x, y), radius), dtype=np.intp)
        if len(ii):
            P = np.asarray(lg.kd_pts)[ii]
            d = np.hypot(P[:, 0].astype(np.float64) - x, P[:, 1].astype(np.float64) - y)
            # 거리순 정렬은 선택이 아니다. project(idx_hint) 는 힌트 주변
            # 세그먼트 두 개만 보므로, **그 차로에서 처음 만난 kd 점**이 투영
            # 결과를 정한다. 순서가 흐트러지면 같은 차로의 먼 점이 힌트가 돼
            # (s, dist) 가 엉뚱한 국소 최소로 간다.
            order = np.argsort(d, kind='stable')
            ii, d = ii[order], d[order]
            if len(ii) > max_points:
                print(f'  [경고] ({x:.2f},{y:.2f}) 반경 {radius:g}m 안 kd 점 {len(ii)}개 > '
                      f'route.candidates_max_points {max_points} — 가까운 순으로 자른다',
                      file=sys.stderr)
                ii, d = ii[:max_points], d[:max_points]
        else:
            d = np.empty(0, dtype=np.float64)
    else:
        d, ii = lg.kd.query((x, y), k=CANDIDATES_K)
        d, ii = np.atleast_1d(d), np.atleast_1d(ii)

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


_PAIR_CFG = None


def pair_cfg(reload=False):
    """params.yaml route.waypoint_lane_is_hint / turn_connect_max_m /
    turn_heading_thr_deg — 짝 공동 선택의 단일 출처.

    candidates_cfg 와 같은 이유로 여기서 한 번만 읽고 캐시한다.
    """
    global _PAIR_CFG
    if _PAIR_CFG is None or reload:
        try:
            from vtd_adapter.config import load_params_yaml
            rc = load_params_yaml().get('route') or {}
            _PAIR_CFG = (bool(rc.get('waypoint_lane_is_hint', True)),
                         float(rc.get('turn_connect_max_m', 400.0)),
                         float(rc.get('turn_heading_thr_deg', 25.0)))
        except Exception:                                # noqa: BLE001 — 독립 실행 폴백
            _PAIR_CFG = (True, 400.0, 25.0)
    return _PAIR_CFG


def road_lane_pool(lg, cands, x, y, yaw, all_sections=False):
    """후보집합에 등장한 **모든 도로**의 같은 섹션·같은 통행방향 driving 차로.

    all_sections=True 면 섹션 제한을 푼다 (같은 도로·같은 방향 전체 차로).
    짝 탐색이 0개로 끝나 폴백하기 직전에만 쓴다 —
    route.pair_fallback_widen_enable. 기본 False = 이전 동작.

    경유점이 어느 차로에 찍혔는지는 정보가 아니다 (주최측 2026-09-03). 도로를
    하나로 확정하지도 않는다 — 최근접 후보의 도로만 보면 경유점을 1.5 m 흔들었을
    때 15 %, 4.5 m 에서 28.6 % 가 엉뚱한 도로를 집는다 (실측).

    반환: [(lane_key, s, dist)] — 반경 밖으로 확장된 차로도 project 로 s·거리를
    채운다. 거리는 스텝4 의 동점 깨기 항에만 쓰인다.
    """
    # 후보 자신을 먼저 넣는다 — 확장 풀은 candidates() 의 **상위집합**이어야 한다.
    # 확장은 차로 중점에서 헤딩·폭을 보는데, 짧은 연결로처럼 중점 헤딩이 투영
    # 지점과 크게 다른 차로는 그 필터에 걸려 자기 자신조차 빠진다 (실측:
    # test_route_waypoints seq 15 에서 진출 풀이 통째로 비어 폴백했다).
    out = list(cands)
    seen_key = {c[0] for c in cands}
    seen_road = set()
    for k0, _s0, _d0 in cands:
        if k0[0] in seen_road:
            continue
        seen_road.add(k0[0])
        for kk in lg.lanes_of_road(k0[0]):
            if kk in seen_key or (not all_sections and kk[1] != k0[1]):
                continue
            r = lg.lanes[kk]
            if r.get('type') != 'driving' or (kk[2] > 0) != (k0[2] > 0):
                continue
            hd = float(np.interp(0.5 * r['length'], r['s'],
                                 np.unwrap(r['hdg'].astype(float))))
            if yaw is not None and abs(wrap(yaw - hd)) > math.radians(70):
                continue
            if lg.width_at(kk, 0.5 * r['length']) < 2.0:
                continue
            seen_key.add(kk)
            s_p, _t, d_p, _j = lg.project(kk, x, y)
            out.append((kk, s_p, d_p))
    return sorted(out, key=lambda c: c[2])


def road_heading(lg, road_id, like_key):
    """도로 단위 헤딩 [rad].

    리포트 [2] 의 Δheading 은 **차로** 시작·끝 헤딩으로 낸다(build_route.report).
    짝 공동 선택은 차로를 정하기 **전에** 회전 방향이 필요하므로 같은 도로·같은
    통행방향의 driving 차로 헤딩으로 대신한다 — 같은 도로에서 통행방향이 같으면
    차로 헤딩이 같으므로 **등가 대체**이지, 같은 코드의 재사용은 아니다.
    회전 방향은 표시용(WARN·리포트·turn 이벤트)이고 후보 필터가 아니다.
    """
    for kk in lg.lanes_of_road(road_id):
        r = lg.lanes[kk]
        if r.get('type') != 'driving' or (kk[2] > 0) != (like_key[2] > 0):
            continue
        h = np.unwrap(r['hdg'].astype(float))
        return float(h[0]), float(h[-1])
    return None, None


def turn_connect(lg, ka, s_a, kb, s_b, banned, cap):
    """진입 차로 ka → 진출 차로 kb 를 **차선변경 없이** 잇는 비용 [m]. 안 되면 None.

    cap 상한이 없으면 successor 만 따라가도 블록을 5 km 돌아 "연결됨" 이 된다
    (실측 p90 1455 m / 최대 5109 m). 실제 짝 43개가 쓴 연결 비용은 최대 363 m 다.
    """
    r = dijkstra(lg, [(ka, s_a)], {kb: s_b}, allow_lane_change=False, banned=banned)
    if r is None or r[0] > cap:
        return None
    return r


def pair_turn_ok(lg, rt, wi, banned=frozenset(), cap=None):
    """짝 세그먼트 wi 의 (진입 차로 → 진출 차로) 가 차선변경 없이 이어지는가.

    반환: (ok, k_in, k_out, cost) — 안 되면 cost 는 None.
    report() 와 작업10(직선 구간 유효 차로 집합)이 같이 쓴다. turn_connect 를
    감싸기만 하고 판정을 새로 만들지 않는다.
    """
    spans = {w: (a, b) for w, a, b in rt.get('segment_span') or []}
    if wi not in spans:
        return None, None, None, None
    i0, i1 = spans[wi]
    k_in, k_out = rt['lanes'][i0], rt['lanes'][i1]
    wps = rt['waypoints']
    s_in = lg.project(k_in, wps[wi][0], wps[wi][1])[0]
    s_out = lg.project(k_out, wps[wi + 1][0], wps[wi + 1][1])[0]
    if cap is None:
        cap = pair_cfg()[1]
    r = turn_connect(lg, k_in, s_in, k_out, s_out, banned, cap)
    return (r is not None), k_in, k_out, (None if r is None else r[0])


def locate_score(lg, key, s, dd, yaw):
    """lanegraph.locate 의 후보 점수(prefer 없음) — 거리 + 헤딩오차 가중.

    build_route 는 출발 차로 판정에서 **locate 와 같은 규칙, 다른 후보집합**을
    쓰려고 이걸 따로 갖는다. 규칙까지 다르면 locate 가 헤딩을 보고 내린 판단을
    통째로 덮어써 버린다 — 여기서 고치려는 건 후보집합 절단뿐이다.
    lanegraph.locate 가 바뀌면 이 함수도 같이 맞춰야 한다.
    """
    if yaw is None:
        return dd
    hd = float(np.interp(s, lg.lanes[key]['s'], np.unwrap(lg.lanes[key]['hdg'].astype(float))))
    return dd + 0.5 * abs(wrap(yaw - hd))


def has_broken(lg, key, side):
    if lg.neighbor(key, side) is None:
        return False
    return any(ok for _, _, _, _, ok in lg.lanes[key]['left_mark' if side == 'left' else 'right_mark'])


def is_lane_change_hop(lg, k, k2) -> bool:
    """route['lanes'] 의 k -> k2 가 차선변경(평행 이웃)인가. successor 면 False."""
    if k2 in lg.successors(k):
        return False
    return k2 in (lg.neighbor(k, 'left'), lg.neighbor(k, 'right'))


def advance(lg, k, k2, length_k) -> float:
    """
    k 를 떠나 k2 로 갈 때 **경로 누적거리** 증가분 [m].

    차선변경은 평행한 이웃 차로로 옮겨 타는 것이라 진행거리가 늘지 않는다.
    successor 처럼 차로 길이를 더하면 route_s 가 통째로 과대계상된다
    (2026-08-21 주행: 실주행 221.3 m 인데 route_s 는 273.8 m -- 52.5 m 초과).
    route_s 는 차선변경 창 판정과 `_blend_path` 의 전이 진행도 기준이라
    이게 틀리면 창이 엉뚱한 물리적 위치에 놓인다.

    평행 차로는 같은 laneSection 안에서 주행방향 s 가 정렬돼 있으므로
    (곡률 차이로 길이가 몇 cm 다른 정도) 증가분 0 으로 두면 된다.
    """
    return 0.0 if is_lane_change_hop(lg, k, k2) else length_k


# 차선변경 창이 이보다 짧으면 리포트에서 경고한다.
# planner 의 전이거리 = max(lane_change.transition_s * v, lane_change.transition_min_m)
# 이고 transition_min_m 이 20 m 다 (params.yaml). 창이 그보다 짧으면 전이를
# 끝낼 수 없다 — 2026-08-21 주행에서 창 6.1 m 짜리 차선변경이 실패해
# 헤딩오차 46°, 조향 풀락 포화, 도로이탈 + courseRespawn 으로 끝났다.
MIN_LC_WINDOW_M = 20.0

def min_hop_gap_m():
    """전이 하나가 먹는 최소 진행거리 [m]. params 가 단일 출처."""
    return float((route_cfg() or {}).get('min_hop_gap_m', MIN_LC_WINDOW_M))


_HOPSEP_CFG = None


def hop_sep_cfg(reload=False):
    """(sep_m, speed_enable) — route.lc_hop_sep_m · lc_hop_sep_speed_enable.

    **연속 차선변경 hop 사이에 필요한 간격**이다. min_hop_gap_m 과 축이 다르다:
    저쪽은 "전이 하나가 목표 차로에서 먹는 진행거리"(dijkstra 비용식), 이쪽은
    "앞 전이가 끝나고 다음이 시작되기까지 필요한 거리"(제어기 램프 + 지시등 선행).

    speed_enable 이면 구간 제한속도로 계산한다 —
    lc_move_len(v) + v x signal.lc_lead_s.
    실측(2026-09-06): 50 km/h 에서 37.5 + 37.5 = 75 m 가 필요한데 상수 45 는
    30 m 모자라고, min_hop_gap_m 20 m 는 1.60 s 로 규정 3 s 의 절반이다.
    기본 false = 상수(45, 현재 동작).
    """
    global _HOPSEP_CFG
    if _HOPSEP_CFG is None or reload:
        r = route_cfg() or {}
        _HOPSEP_CFG = (float(r.get('lc_hop_sep_m', 45.0)),
                       bool(r.get('lc_hop_sep_speed_enable', False)))
    return _HOPSEP_CFG


def hop_sep_for(lg, key):
    """차로 key 에서의 연속 차선변경 필요 간격 [m]. hop_sep_cfg 참조."""
    sep, speed_on = hop_sep_cfg()
    if not speed_on:
        return sep
    from vtd_adapter.config import load_params_yaml
    cfg = load_params_yaml()
    r = cfg['route']
    kph, _sc = lg.speed_limit_at(key)
    kph = float(kph) if kph is not None else float(cfg.get('default_speed_kph', 50.0))
    v = max(0.0, kph - float((cfg.get('speed') or {}).get('margin_kph', 0.0))) / 3.6
    ramp = min(float(r['lc_move_max_m']), max(float(r['lc_move_min_m']),
                                              v * float(r['lc_move_s'])))
    return ramp + v * float((cfg.get('signal') or {}).get('lc_lead_s', 3.0))


def hop_spacing_cost_enable():
    """탐색이 hop 간격을 비용으로 보는가 (작업19-3). false = 이전 동작."""
    return bool((route_cfg() or {}).get('hop_spacing_cost_enable', True))


def hop_room(lg, rt, gap=None):
    """차선변경마다 (인덱스, cum, from, to, 누적 필요거리, 차로 여유, 연쇄번째).

    **탐색(dijkstra)과 같은 축이다.** 전이 하나는 목표 차로 안에서 gap 만큼의
    진행거리를 먹는다. 연속 hop 은 커서가 누적되고, successor 전이는 커서를
    0 으로 되돌린다(새 차로에서 다시 시작).

    경로 누적거리(cum) 축으로는 이걸 못 잰다 -- advance() 가 hop 의 진행거리를
    0 으로 두므로 같은 차로 안의 hop 은 전부 간격 0 이고, "29.8 m 에 3회"와
    "70.5 m 에 3회"가 똑같이 보인다. 앞쪽은 주행 불가, 뒤쪽은 정상이다.

    연쇄의 **첫 hop(nth=0)은 여기서 판정하지 않는다** -- 그건 창 검사
    (window_s1 - window_s0 >= MIN_LC_WINDOW_M)가 이미 본다. 창은
    lane_change_window 가 laneSection 경계를 넘어 이어 붙이므로, 17 m 짜리
    짧은 차로라도 후행 차로까지 회랑이 이어지면 정상으로 잡힌다. 차로 길이만
    보는 여기서 첫 hop 까지 재면 그런 정상 경로를 과탐한다
    (실측: (2801,0,3)->(2801,0,2) 차로 17.07 m / 회랑 20.8 m, 계단 0).
    """
    # gap=None 이면 dijkstra 비용식과 같은 축(min_hop_gap_m). 호출부가
    # route.lc_hop_sep_speed_enable 을 켜면 차로별 필요 간격을 넘겨 준다.
    if gap is None:
        gap = min_hop_gap_m()
    lanes = [tuple(k) for k in rt['lanes']]
    cum = rt['cum_s']
    cur = 0.0
    nth = 0
    out = []
    for i in range(len(lanes) - 1):
        a, b = lanes[i], lanes[i + 1]
        if not is_lane_change_hop(lg, a, b):
            cur, nth = 0.0, 0                    # successor -> 커서 리셋
            continue
        room = lg.length(b)
        need = cur + (gap(b) if callable(gap) else gap)
        out.append((i + 1, float(cum[i + 1]), a, b, need, room, nth))
        cur = min(need, room)
        nth += 1
    return out


# 점선 구간 두 개가 이만큼 안쪽으로 붙어 있으면 하나로 잇는다 (샘플 경계 오차)
MARK_JOIN_M = 1e-6
# 점선 구간이 차로 끝까지 닿았다고 볼 허용오차 [m]
MARK_EDGE_M = 0.5


def dashed_runs(lg, key, side):
    """side 방향 연속 점선 구간 — lanegraph 가 단일 출처다 (제어기와 같은 답)."""
    return lg.dashed_runs(key, side)


def lane_change_window(lg, lanes, cum, seq, i, side, target):
    """
    차선변경 창 -> (window_s0, window_s1, lane_idx, s_in_lane)

    **끝점은 기존 그대로** — 지금 차로 lanes[i] 의 마지막 점선 구간 끝이다.
    **시작점만 최대한 앞으로 당긴다**: 거기서 뒤로 거슬러 올라가며 "차선변경
    가능" 이 끊기지 않는 가장 이른 지점을 찾는다.

    laneSection 은 OpenDRIVE 의 차로구성 변경 단위라 차선변경에 필요한 거리와
    아무 상관이 없다 (도로 128 은 12 m, 도로 1648 은 6 m). 그래서 같은 차로가
    successor 로 끊김 없이 이어지는 동안은 section 경계를 넘어 병합한다.

    뒤로 못 가는 조건 (여기서 멈춘다):
      · 앞 차로가 route 상 successor 가 아니다 (차선변경으로 들어온 차로)
      · 앞 차로가 교차로 연결로다 (junction != -1)
      · 앞 차로에 side 이웃이 없거나, 그 이웃이 target 으로 이어지지 않는다
        (= 목표 차로가 거기엔 아직 없다)
      · 앞 차로의 그 방향이 실선이다 (roadMark type != broken)
      · 그 차로에 route 가 진입한 지점 (그 앞은 우리가 지나온 길이 아니다)
    """
    k = lanes[i]
    runs = dashed_runs(lg, k, side)
    if not runs:
        # 점선이 아예 없다 = 원래 넘을 수 없는 자리. 탐색이 has_broken 으로
        # 걸러 주므로 여기 오면 안 되지만, 오면 기존 폴백을 그대로 쓴다.
        s_en = seq[i][1]
        return cum[i] + s_en, cum[i] + lg.length(k), i, s_en

    w1 = runs[-1][1]
    j, s0 = i, runs[-1][0]
    while True:
        entry = seq[j][1]
        if s0 > entry + 1e-6:
            break                       # 이 차로 안에서 시작한다 — 더는 못 당긴다
        s0 = entry
        if j == 0:
            break
        p = lanes[j - 1]
        if lanes[j] not in lg.successors(p):
            break                       # 차선변경으로 들어온 차로
        if lg.lanes[p]['junction'] != -1:
            break                       # 교차로 연결로에서는 차선변경 금지
        nb_p, nb_j = lg.neighbor(p, side), lg.neighbor(lanes[j], side)
        if nb_p is None or nb_j is None or nb_j not in lg.successors(nb_p):
            break                       # 목표 차로가 거기까지 이어지지 않는다
        runs_p = dashed_runs(lg, p, side)
        if not runs_p or runs_p[-1][1] < lg.length(p) - MARK_EDGE_M:
            break                       # 앞 차로 끝이 실선 — 창이 거기서 끊긴다
        j -= 1
        s0 = runs_p[-1][0]
    return cum[j] + s0, cum[i] + w1, j, s0


def finish_tail_cfg():
    """params.yaml route.finish_tail_* — 종료선 뒤 꼬리 연장 요구량 [m]. 0 = 끔.

    plan_stop_s(team_code/kr_rules.py)가 finish_s + finish_clearance + stop_gap
    + wheelbase + front_overhang ≤ total − end_slack 을 요구한다 (현재 params 합
    10.799 m). 기본 12.0 은 그 요구량 + 여유다.
    """
    try:
        from vtd_adapter.config import load_params_yaml
        rc = load_params_yaml().get('route') or {}
        if not rc.get('finish_tail_enable', True):
            return 0.0
        return float(rc.get('finish_tail_m', 12.0))
    except Exception:                                    # noqa: BLE001 — 독립 실행 폴백
        return 12.0


_ROUTE_CFG = None


def route_cfg(reload=False):
    """params.yaml route.* 원본 dict (캐시). 개별 소비자는 필요한 키만 읽는다."""
    global _ROUTE_CFG
    if _ROUTE_CFG is None or reload:
        try:
            from vtd_adapter.config import load_params_yaml
            _ROUTE_CFG = dict(load_params_yaml().get('route') or {})
        except Exception:                                # noqa: BLE001 — 독립 실행 폴백
            _ROUTE_CFG = {}
    return _ROUTE_CFG


def route_check_cfg():
    """params.yaml route_check.* — 검증 리포트 임계의 단일 출처.

    키가 없으면 조용히 기본값으로 도는 대신 죽는다 (설정 두 벌 금지 규칙).
    """
    from vtd_adapter.config import load_params_yaml
    return load_params_yaml()['route_check']


def min_turn_radius_m():
    """차량 최소회전반경 [m] = 축거 / tan(최대조향). params.yaml 을 읽는다."""
    try:
        from vtd_adapter.config import load_params_yaml
        cfg = load_params_yaml()
        vh = cfg['vehicle']
        return (float(vh['wheelbase']) / math.tan(float(vh['max_steer'])),
                float(vh.get('min_turn_margin', 1.2)))
    except Exception:                                    # noqa: BLE001 — 독립 실행 폴백
        return 2.944 / math.tan(0.48), 1.2


def lane_r_min(lg, key):
    """차로의 최소 곡률반경 [m]. 곡률 0 이면 inf."""
    cv = np.abs(lg.lanes[key]['curv'])
    m = float(cv.max()) if len(cv) else 0.0
    return (1.0 / m) if m > 1e-6 else float('inf')


def banned_r_min_m():
    """연결로 통행 금지 임계 R_min [m] — route.banned_r_min_m (기본 3.0).

    옛 임계는 최소회전반경 × vehicle.min_turn_margin (= 5.65 m) 이었다.
    2026-09-08 완화: 지도가 실도로 기반이라 연결로는 실차가 도는 길이고,
    이탈 실측 4/4 는 커브 감속 없이 25 km/h 로 진입한 결과였다. 이제
    speed.curvature_cap 이 R 에 맞춰 눌러 준다. 규정상 우회가 급회전보다
    나쁘므로(경로 이탈 감점), 금지는 **지도 결함 안전망**으로만 남긴다.
    """
    return float((route_cfg() or {}).get('banned_r_min_m', 3.0))


def curvature_a_lat_max_m_s2():
    """제어기 커브 감속의 허용 횡가속 [m/s²] — speed.curvature_a_lat_max.

    리포트에서 "급회전 연결로에 몇 m/s 로 들어가는가" 를 보여 주기 위해서만
    읽는다 (제어기 파트 키다 — 여기서는 읽기 전용). 커브 감속이 꺼져 있으면
    상한이 없다는 뜻이라 None 대신 0 을 쓰지 않고, 값 자체는 그대로 보여 준다.
    """
    try:
        from vtd_adapter.config import load_params_yaml
        return float((load_params_yaml().get('speed') or {})
                     .get('curvature_a_lat_max', 2.5))
    except Exception:                                    # noqa: BLE001 — 독립 실행 폴백
        return 2.5


def tight_turn_r_m():
    """'급회전' 경고 상한 [m] = 기하 최소회전반경 × vehicle.min_turn_margin.

    banned 임계와 이 값 사이 구간은 **통행하되 감속 진입**으로 다룬다.
    """
    r_need, margin = min_turn_radius_m()
    return r_need * margin


def infeasible_connectors(lg):
    """
    물리적으로 돌 수 없는 교차로 연결로 집합.

    R_min < route.banned_r_min_m (기본 3.0 m) 인 junction 연결로만 금지한다.
    9_school_route 실측(2026-08-24)의 (1576,0,-1) R_min 2.55 m — 조향 포화
    1.8 s 끝에 호를 이탈해 off_route 정지 — 는 이 임계에도 계속 걸린다.
    곡률 스파이크는 빌드 단계에서 이미 걸렀으므로(중앙값 필터) 남은 값은 진짜
    기하다 — 그대로 평가한다.
    """
    thr = banned_r_min_m()
    out = {}
    for key, rec in lg.lanes.items():
        if rec['junction'] == -1:
            continue
        r = lane_r_min(lg, key)
        if r < thr:
            out[key] = r
    return out, thr


def dijkstra(lg, starts, targets, allow_lane_change=True, banned=frozenset(),
             lc_in_junction=True):
    """starts: [(lane, s_start)]  targets: {lane: s_target} → (cost, [ (lane, s_enter) ... ])

    allow_lane_change=False 면 successor 링크만 따라간다 (교차로 내부 구간용).
    banned: 통행 금지 차로 (회전 불가 연결로 — 비용 무한 대신 아예 확장하지 않는다).
    lc_in_junction=False 면 **교차로 차로 위에서만** 차선변경을 막는다 (짝 형식에
    기대지 않는 형식 무관 규칙 — 채점 항목 6). 기본 True = 이전 동작."""
    tgt = dict(targets)
    best = {}
    heap = []
    # 전이 하나가 먹는 진행거리 [m]. 0 이면 이전 동작(간격을 비용에서 무시).
    hop_gap = min_hop_gap_m() if hop_spacing_cost_enable() else 0.0
    # 소멸(테이퍼) 차로 진입 벌점 [m 환산]. 0 이면 이전 동작.
    # 왜: 2026-09-06 실전주행_교통류_01 junction 7 우회전이 끝 폭 0.05 m 로
    # 소멸하는 연결로 (1154,0,-2) 를 탔다. 옆에 폭이 유지되는 (1154,0,-3) 이 있고
    # 지도 전체에 같은 꼴의 연결로가 29개, 그중 28개에 오른쪽 대안이 있다.
    # 벌점은 successor 진입에만 붙는다 — 같은 도로 안 소멸 차로(차로 수 감소)는
    # 대안이 차선변경뿐이라 LC 비용 축과 섞이지 않게 둔다.
    tp_on, tp_m, veh_w = taper_cfg()
    taper_pen = tp_m if (tp_on and tp_m > 0.0) else 0.0
    # 회전 방향 차로 선호 [m/칸]. 0 이면 이전 동작 (계산도 안 한다).
    turn_bias = turn_bias_m()
    # 회전 차로 **제약** — 노면 화살표가 금지하는 회전은 후보에서 뺀다.
    turn_con = turn_constraint_on()
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
            if k2 in banned:
                continue                     # 물리적으로 돌 수 없는 연결로
            if (k2, False) in best and (k2, True) in best:
                continue
            L2 = lg.length(k2)
            pen = taper_pen if (taper_pen > 0.0 and lg.lanes[k2].get('junction', -1) != -1
                                and is_taper_lane(lg, k2, veh_w)) else 0.0
            if turn_bias > 0.0 or turn_con:
                tk = connector_turn(lg, k2)
                if tk is not None:
                    if turn_con and turn_lane_blocked(lg, key, 'L' if tk == 'left' else 'R'):
                        continue          # 노면 화살표가 금지한다 — 후보에서 제외
                    if turn_bias > 0.0:
                        # 우회전 연결로에 드는데 진입 차로 오른쪽에 차로가 남아
                        # 있으면 그 칸 수만큼 문다 (좌회전은 왼쪽).
                        pen += turn_bias * lanes_on_side(lg, key, tk)
            if k2 in tgt:
                heapq.heappush(heap, (cost + tgt[k2] + pen, k2, 0.0, (key, s_enter), root, True))
            heapq.heappush(heap, (cost + L2 + pen, k2, 0.0, (key, s_enter), root, False))
        # 차선변경: 같은 s 로 옆 차로에 진입 (진입 지점은 이 차로 시작 s_enter 이후 아무 데나 → 여기선 s_enter 로 근사)
        if not allow_lane_change:
            continue
        for side in ('left', 'right'):
            if not has_broken(lg, key, side):
                continue
            k2 = lg.neighbor(key, side)
            if not lc_in_junction and (r['junction'] != -1
                                       or lg.lanes[k2]['junction'] != -1):
                continue
            L2 = lg.length(k2)
            if hop_gap > 0.0:
                # 전이 하나가 목표 차로 안에서 hop_gap 만큼의 진행거리를 먹는다.
                # 연속 hop 은 s_enter 가 누적되므로 짧은 차로에 몰아넣으면
                # 부족분을 문다 — 이게 없으면 "3회를 100 m 에"와 "29.8 m 에"가
                # 같은 비용이다 (작업19-3). + hop_gap 은 전이 중 실제로 달리는
                # 거리라 회계상 맞다.
                s_req = s_enter + hop_gap
                s2 = min(s_req, L2)
                extra = hop_gap + LC_SHORT_PENALTY_PER_M * max(0.0, s_req - L2)
            else:
                s2 = min(s_enter, L2)          # 이전 동작
                extra = 0.0
            # 이 차로를 s_enter 에서 떠나는 비용으로 되돌리고 + 차선변경 비용(회랑 반영)
            c_lc = cost - (r['length'] - s_enter) + lc_cost(lg, key, side) + extra
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


def junction_segments(n_points, offset=1):
    """
    교차로 내부 구간(0-based 세그먼트 인덱스) 집합.

    offset=1 (기본): seq 1=시작, (2,3)(4,5)... 짝, 마지막=종료. 0-based 로는
    waypoints[1]→[2], [3]→[4], ... 즉 **홀수 인덱스 세그먼트**가 교차로 내부다.
    주최 공식 형식(짝수 경유점)이 이것이다.

    offset=0: 첫 점이 곧 진입점 — [(진입,진출)×N, 종료]. 경유점이 홀수로 올 때의
    또 다른 해석이다 (작업21). 어느 쪽인지는 pair_offset_auto 가 고른다.
    """
    return {wi for wi in range(offset, n_points - 1, 2)}


# ── 짝 해석 자동 판정 (작업21) ────────────────────────────────────────────
def pair_auto_cfg():
    """params.yaml route.pair_auto_* — 자동 판정 임계의 단일 출처.

    폴백 기본값은 params 의 값과 **같아야 한다**. 키가 항상 있어서 실제로는
    안 쓰이지만, 두 곳에 다른 수가 적혀 있으면 나중에 읽는 사람이 어느 쪽이
    사는 값인지 헷갈린다 (2026-09-05 임계 1.00 상향 때 0.75 로 남아 있었다).
    """
    rc = route_cfg() or {}
    return (float(rc.get('pair_auto_min_ratio', 1.00)),
            float(rc.get('pair_auto_min_margin', 0.50)))


def pair_junction_ratio(lg, rt):
    """(짝 구간 junction 경유 비율, 짝 개수, 비짝 구간 비율, 비짝 개수).

    **짝 구간이 교차로 연결로를 실제로 지나는가** — 짝 해석이 맞는지를 가르는
    유일하게 결정적인 신호다. 짝수 기준 CSV 25개 실측(2026-09-04): 정답
    offset 65/65 = 100 %, 오답 offset 4/86 = 5 %.

    비짝 비율은 보조 표시다. 단독으로는 못 쓴다 — 정답 offset 에서도
    0.00~0.80 으로 흔들린다(official_route 0.50, venue 0.80). 다만 오답
    offset 에서는 25/25 전부 1.00 이었으므로, **짝 비율이 높은데 비짝 비율도
    1.00 이면 의심 신호**다.
    """
    spans = {w: (a, b) for w, a, b in rt.get('segment_span') or []}
    js = set(rt.get('junction_segments') or [])

    def has_junction(wi):
        if wi not in spans:
            return None
        i0, i1 = spans[wi]
        return any(lg.lanes[k]['junction'] != -1 for k in rt['lanes'][i0:i1 + 1])

    pair = [v for v in (has_junction(w) for w in sorted(js)) if v is not None]
    other = [v for v in (has_junction(w) for w in range(len(rt['waypoints']) - 1)
                         if w not in js) if v is not None]
    return (sum(pair) / len(pair) if pair else 0.0, len(pair),
            sum(other) / len(other) if other else 0.0, len(other))


def pair_offset_auto(lg, waypoints, radius, start_yaw, seqs, finish_tail_m,
                     build_fn=None, dp_radius=None):
    """짝 해석을 고른다 → (offset|None, 근거 문자열, {offset: 신호}).

    짝수면 시험 빌드 없이 offset 1 이다 (주최 공식 형식). 홀수일 때만 0/1 을
    각각 지어 보고 pair_junction_ratio 로 고른다 — 시험 빌드 2회가 늘어난다.

    판정이 서지 않으면 None(짝 해석 안 함) 으로 떨어진다. 그건 현재 동작과
    같고, **교차로 내부 차선변경 금지가 안 걸린다**는 뜻이라 리포트가 크게
    표시한다.
    """
    n = len(waypoints)
    if n <= 2:
        return None, f'경유점 {n}개 — 시작/종료뿐, 짝 없음', {}
    if n % 2 == 0:
        return 1, f'짝수 {n}점 — 주최 공식 형식 (offset 1)', {}

    fn = build_fn or build_route
    ev = {}
    for off in (0, 1):
        try:
            # dp_radius 를 같이 넘겨야 한다 — 이 시험 빌드가 짝 해석을 정하는데,
            # 여기만 params 기본 반경으로 지으면 **본 빌드와 다른 경로**를 보고
            # 고르게 된다 (2026-09-08).
            rt = fn(lg, waypoints, radius, start_yaw,
                    junction_segs=junction_segments(n, off),
                    seqs=seqs, finish_tail_m=finish_tail_m, dp_radius=dp_radius)
        except RouteError as e:                 # SystemExit 파생 — 이것만 잡는다.
            ev[off] = dict(ok=False, why=str(e).splitlines()[0], ratio=-1.0,
                           n=0, other=0.0, n_other=0)
            continue
        r, np_, ro, no = pair_junction_ratio(lg, rt)
        ev[off] = dict(ok=True, why=None, ratio=r, n=np_, other=ro, n_other=no)

    min_ratio, min_margin = pair_auto_cfg()
    win, lose = (0, 1) if ev[0]['ratio'] >= ev[1]['ratio'] else (1, 0)
    margin = ev[win]['ratio'] - ev[lose]['ratio']
    detail = ' / '.join(
        f"offset{o}: " + ('빌드실패' if not ev[o]['ok']
                          else f"junction {ev[o]['ratio']:.2f}({ev[o]['n']}개)")
        for o in (0, 1))
    if ev[win]['ok'] and ev[win]['ratio'] >= min_ratio and margin >= min_margin:
        return win, (f'홀수 {n}점 — offset {win} 채택 ({detail}; '
                     f'margin {margin:.2f} ≥ {min_margin:g})'), ev
    return None, (f'홀수 {n}점 — **판정 불가** ({detail}; margin {margin:.2f} < '
                  f'{min_margin:g} 또는 비율 < {min_ratio:g})'), ev


def _pair_diag(lg, pool_in, pool_out, starts, allow_prev, banned, cap):
    """진입 차로별 (앞 세그먼트 도달 / 진출 연결 최소비용) — WARN·RouteError 용."""
    out = []
    for ka, sa, da in pool_in:
        pre = dijkstra(lg, starts, {ka: sa}, allow_lane_change=allow_prev, banned=banned)
        best = None
        for kb, sb, _db in pool_out:
            c = turn_connect(lg, ka, sa, kb, sb, banned, cap)
            if c is not None and (best is None or c[0] < best[0]):
                best = (c[0], kb)
        out.append((ka, da, None if pre is None else pre[0], best))
    return out


def _fmt_pair_diag(diag):
    rows = []
    for ka, da, pre, best in diag:
        rows.append('%s @%.2fm 앞도달 %s 회전 %s' % (
            ka, da, '-' if pre is None else '%.0fm' % pre,
            '불가' if best is None else '%s %.0fm' % (best[1], best[0])))
    return '\n              '.join(rows)


def _pair_choice(lg, starts, wps, wi, radius, junction_segs, banned, cap, label, seqs,
                 fallbacks=None):
    """교차로 짝 (진입, 진출) 차로를 함께 고른다.

    스텝 (2026-09-03 주최측 답변 반영):
      1. 진입·진출 경유점 각각 candidates() 후보집합에 등장한 **모든 도로**의
         같은 섹션·같은 통행방향 driving 차로로 넓힌다. 도로를 하나로 확정하지
         않는다 — 경유점이 찍힌 차로도, 그 차로의 도로도 정보가 아니다.
      2. 회전 방향은 도로 단위 헤딩으로 낸다. **표시용**이고 필터가 아니다.
      3. 진입 차로 → 진출 차로가 차선변경 없이 이어지고 그 비용이 cap 이하인
         짝만 남긴다.
      4. cost(앞→진입) + cost(진입→진출) + W×진입거리 + W×진출거리 최소.
         거리 항은 동점 깨기지 결정 요인이 아니다.
      5. 짝이 0개면 None 을 돌려 호출부가 기존 탐욕으로 폴백하게 한다 (+WARN).
         route.pair_fallback_widen_enable 이면 폴백 **직전에** 진출 풀을 같은
         도로·같은 방향 **전체 차로**(섹션 제한 해제)로 넓혀 한 번 더 찾는다.
         왜: 짝 사이는 차선변경 금지라 진출 경유점이 찍힌 섹션에 연결 가능한
         차로가 없으면 후보가 통째로 죽는데, 같은 도로의 다른 섹션 차로는
         successor 만으로 이어지는 경우가 많다 (2026-09-06 재현: 진입
         [2152,2190] → 진출 [2012] 좌회전. 폴백 경로가 205 m 대신 2226 m).

    fallbacks: 폴백이 나면 진단을 append 할 리스트 (호출부가 리포트에 쓴다).

    tier 층은 쓰지 않는다 — 실패를 만든 게 "가까운 후보부터 좁게 본다" 였고,
    거리는 이미 4의 비용에 들어 있다.

    반환: (path_in, k_in, s_in, path_out, k_out, s_out, used_banned) 또는 None
    """
    x1, y1 = wps[wi + 1]
    x2, y2 = wps[wi + 2]
    ay_in = math.atan2(y1 - wps[wi][1], x1 - wps[wi][0])
    ay_out = math.atan2(y2 - y1, x2 - x1)
    ca = candidates(lg, x1, y1, radius, ay_in)
    cb = candidates(lg, x2, y2, radius, ay_out)
    if not ca or not cb:
        return None
    pool_in = road_lane_pool(lg, ca, x1, y1, ay_in)
    pool_out = road_lane_pool(lg, cb, x2, y2, ay_out)
    allow_prev = wi not in junction_segs

    # ── 연속 짝 공동 선택 (2교차로 lookahead) ──────────────────────────────
    # 다음 짝이 바로 이어지고 그 사이 도로가 짧으면, 진출 차로를 **다음 짝
    # 진입 차로에 차선변경 없이 닿는 것**으로 좁힌다. 좁혀서 아무것도 안
    # 남으면 좁히기 전 결과를 쓴다 — 이 필터가 경로를 없애지는 않는다.
    chain_keep = None
    ch_on, ch_gap = pair_chain_cfg()
    if (ch_on and (wi + 3) in junction_segs and wi + 4 < len(wps)
            and math.hypot(wps[wi + 3][0] - x2, wps[wi + 3][1] - y2) < ch_gap):
        x3, y3 = wps[wi + 3]
        x4, y4 = wps[wi + 4]
        ay3 = math.atan2(y3 - y2, x3 - x2)
        ay4 = math.atan2(y4 - y3, x4 - x3)
        c3 = candidates(lg, x3, y3, radius, ay3)
        c4 = candidates(lg, x4, y4, radius, ay4)
        pool3 = road_lane_pool(lg, c3, x3, y3, ay3) if c3 else []
        pool4 = road_lane_pool(lg, c4, x4, y4, ay4) if c4 else []
        # 다음 짝의 진입 차로 중 **그 회전을 실제로 끝낼 수 있는** 것만 본다.
        # 진입 풀 전체로 보면 걸러지는 게 없다 — 직진 차로도 풀에 들어 있어서다.
        need = [(kc, sc) for kc, sc, _dc in pool3
                if any(turn_connect(lg, kc, sc, kd, sd, banned, cap) is not None
                       for kd, sd, _dd in pool4)]
        if need:
            chain_keep = {kb for kb, sb, _d in pool_out
                          if any(turn_connect(lg, kb, sb, kc, sc, banned, cap) is not None
                                 for kc, sc in need)}
            if not chain_keep or len(chain_keep) == len({k for k, _s, _d in pool_out}):
                chain_keep = None          # 걸러지는 게 없으면 이전과 같은 길

    def search(bans, keep=None):
        best = None
        for ka, sa, da in pool_in:
            pre = dijkstra(lg, starts, {ka: sa}, allow_lane_change=allow_prev, banned=bans)
            if pre is None:
                continue
            for kb, sb, db in pool_out:
                if keep is not None and kb not in keep:
                    continue
                c = turn_connect(lg, ka, sa, kb, sb, bans, cap)
                if c is None:
                    continue
                score = pre[0] + c[0] + TARGET_DIST_W * da + TARGET_DIST_W * db
                if best is None or score < best[0]:
                    best = (score, pre[1], ka, sa, c[1], kb, sb)
        return best

    widen_enable, _is_err = pair_fallback_cfg()
    best, used_banned = None, []
    chained = False
    if chain_keep is not None:
        best = search(banned, chain_keep)
        chained = best is not None
    if best is None:
        best = search(banned)
    widened = False
    if best is None:
        # 금지 연결로를 풀면 되는가 — 기존 탐욕과 같은 취급(불가피하면 허용 + 기록)
        best = search(frozenset())
        if best is None and widen_enable:
            # 진출 풀을 같은 도로·같은 방향 전체 차로로 넓혀 한 번 더 (제안②)
            wide = road_lane_pool(lg, cb, x2, y2, ay_out, all_sections=True)
            if len(wide) > len(pool_out):
                pool_out = wide
                widened = True
                best = search(banned)
                if best is None:
                    best = search(frozenset())
                else:
                    used_banned = []
        if best is None:
            widened = False
            roads_in = sorted({k[0] for k, _s, _d in pool_in})
            roads_out = sorted({k[0] for k, _s, _d in pool_out})
            h0, _ = road_heading(lg, ca[0][0][0], ca[0][0])
            _, h1 = road_heading(lg, cb[0][0][0], cb[0][0])
            kind = ('?' if h0 is None or h1 is None
                    else turn_kind(math.degrees(wrap(h1 - h0))))
            print(f'  [경고] {label(wi + 1)}→{label(wi + 2)} 회전 가능한 (진입,진출) '
                  f'짝이 없다 — 진입 도로 {roads_in} 진출 도로 {roads_out} {kind}, '
                  f'연결 상한 {cap:g}m. 기존 탐욕으로 폴백한다\n'
                  f'              {_fmt_pair_diag(_pair_diag(lg, pool_in, pool_out, starts, allow_prev, banned, cap))}',
                  file=sys.stderr)
            if fallbacks is not None:
                # wi 는 **짝 구간의 세그먼트 인덱스**로 남긴다 (= wi+1).
                # report() 의 [2] 가 junction_segments 를 그 인덱스로 돈다.
                fallbacks.append({'wi': wi + 1, 'label': f'{label(wi + 1)}→{label(wi + 2)}',
                                  'roads_in': roads_in, 'roads_out': roads_out,
                                  'kind': kind, 'cap_m': float(cap),
                                  'widen_tried': bool(widen_enable),
                                  'connectors': []})
            return None
        used_banned = [kk for kk, _ in (best[1] + best[4]) if kk in banned]
    if chained:
        print(f'  [주의] {label(wi + 1)}→{label(wi + 2)} 다음 교차로까지 보고 진출 차로를 '
              f'{best[5]} 로 골랐다 (route.pair_chain_enable, 사이 도로 '
              f'{math.hypot(wps[wi + 3][0] - x2, wps[wi + 3][1] - y2):.1f} m)', file=sys.stderr)
    if widened:
        print(f'  [주의] {label(wi + 1)}→{label(wi + 2)} 진출 풀을 같은 도로 전체 '
              f'차로로 넓혀 짝을 찾았다 (route.pair_fallback_widen_enable) — '
              f'진입 {best[2]} → 진출 {best[5]}', file=sys.stderr)
    _sc, path_in, k_in, s_in, path_out, k_out, s_out = best
    return path_in, k_in, s_in, path_out, k_out, s_out, used_banned


_PAIRFB_CFG = None
_PAIRCHAIN_CFG = None


def pair_chain_cfg(reload=False):
    """(enable, gap_m) — route.pair_chain_enable · route.pair_chain_gap_m.

    연속 짝 공동 선택(2교차로 lookahead): 앞 교차로 진출 차로를 고를 때 **다음
    교차로 진입 차로까지** 본다. 왜: 교차로 사이 도로가 짧으면 진출 차로를
    잘못 고른 뒤 차선변경할 자리가 없다. 짝 사이는 차선변경 금지라 다음 짝의
    후보가 통째로 죽고 탐욕 폴백으로 넘어간다 (2026-09-06 G-2: 도로 2152 는
    27.6 m 인데 진입 후보의 목표 s 가 27.57 이라, 차선변경 착지점 s+20 이
    3 cm 차이로 목표를 지나쳐 후보가 탈락한다).
    기본 false = 이전 동작.
    """
    global _PAIRCHAIN_CFG
    if _PAIRCHAIN_CFG is None or reload:
        from vtd_adapter.config import load_params_yaml
        r = load_params_yaml().get('route') or {}
        _PAIRCHAIN_CFG = (bool(r.get('pair_chain_enable', False)),
                          float(r.get('pair_chain_gap_m', 30.0)))
    return _PAIRCHAIN_CFG


def pair_fallback_cfg(reload=False):
    """(widen_enable, is_error) — route.pair_fallback_widen_enable ·
    route_check.pair_fallback_is_error. 둘 다 기본 False = 이전 동작."""
    global _PAIRFB_CFG
    if _PAIRFB_CFG is None or reload:
        from vtd_adapter.config import load_params_yaml
        cfg = load_params_yaml()
        _PAIRFB_CFG = (bool((cfg.get('route') or {}).get('pair_fallback_widen_enable', False)),
                       bool((cfg.get('route_check') or {}).get('pair_fallback_is_error', False)))
    return _PAIRFB_CFG


def _fmt_fb_connectors(fb, n=3):
    """폴백이 고른 연결로 요약 — 앞 n 개만 R_min 과 함께, 나머지는 개수로."""
    cs = fb.get('connectors') or []
    if not cs:
        return '(없음)'
    head = '  '.join(f'{tuple(k)} R_min {r:.2f} m' for k, r in cs[:n])
    return head + (f'  … 총 {len(cs)}개' if len(cs) > n else '')


def valid_entry_lanes_enable():
    """route.valid_entry_lanes_enable (기본 true). false 면 필드를 안 넣는다."""
    return bool((route_cfg() or {}).get('valid_entry_lanes_enable', True))


def valid_entry_lanes(lg, rt, radius=8.0, banned=None, cap=None):
    """짝이 아닌 세그먼트마다 "다음 제약을 차선변경 없이 만족시키는 차로 집합" (작업10).

    쓰임: 회피 시프트로 옆 차로에 나간 뒤 **원래 차로로 복귀해야 하는가**를
    제어기가 판단할 재료다. 복귀 공간이 부족하면 곡률이 커져 속도가 하한에
    붙거나(2026-09-03 정적회피집중_01, 27.7 s 크립) span 게이트가 회피를 기각해
    장애물 앞에 선다. 직진 구간이면 굳이 복귀하지 않고 그 차로로 다음 교차로에
    들어가도 되는 경우가 많다 — 그걸 여기서 알려 준다.

    판정은 **작업7-2 의 turn_connect 를 그대로** 쓴다. 새 탐색을 만들지 않는다:
      · 후보 풀은 _pair_choice 와 같은 방식 — candidates() 후보집합에 등장한
        모든 도로의 같은 섹션·같은 통행방향 driving 차로(road_lane_pool).
      · 진입 후보 ka 가 진출 후보 kb 중 하나로 차선변경 없이 cap 안에 이어지면
        유효하다.

    좌·우회전이면 진입 차로가 하나로 좁혀져 집합 크기가 1 이 되고, 그게 곧
    **복귀 강제**다. 직진이면 여러 개가 남는다.

    반환: [{seg, seq, target, turn, lanes, chosen, in_set}] — target 은
      'pair'   다음 세그먼트가 짝이다. lanes 는 그 짝의 진입 차로 후보.
      'finish' 다음 짝이 없는 마지막 세그먼트. lanes 는 종료 경유점까지
               차선변경 없이 갈 수 있는 차로들 (같은 축으로 계산한다).
      None     다음 제약이 없다 (짝 해석이 없는 경로의 중간 세그먼트).
    """
    if banned is None:
        banned, _thr = infeasible_connectors(lg)
    if cap is None:
        cap = pair_cfg()[1]
    thr = pair_cfg()[2]
    wps = rt['waypoints']
    n = len(wps)
    jsegs = set(rt.get('junction_segments') or [])
    spans = {w: (a, b) for w, a, b in rt.get('segment_span') or []}
    seqs = rt.get('waypoint_seq') or list(range(1, n + 1))
    lanes_r = rt['lanes']
    out = []
    for wi in range(n - 1):
        if wi in jsegs:
            continue                                  # 짝 구간 자체는 대상이 아니다
        nxt = wi + 1
        if nxt in jsegs and wi + 2 < n:
            target, i_ent, i_ext = 'pair', wi + 1, wi + 2
        elif nxt == n - 1:
            # 마지막 세그먼트(마지막 진출 → 종료). 다음 짝이 없으므로 **차로 제약이
            # 없다** — 종료선은 점이고 채점은 뒷축의 route_s 로 판정하므로 어느
            # 평행 차로에 있어도 완주가 성립한다. 그래서 빈 집합을 넣는다.
            # target='finish' 가 "유효 차로가 없다"가 아니라 "제약이 없다"임을
            # 구분한다 — 소비자는 target 을 먼저 봐야 한다.
            out.append(dict(seg=wi, seq=(seqs[wi], seqs[wi + 1]), target='finish',
                            turn=None, lanes=[],
                            chosen=(tuple(lanes_r[spans[wi][1]]) if wi in spans else None),
                            in_set=None))
            continue
        else:
            out.append(dict(seg=wi, seq=(seqs[wi], seqs[wi + 1]), target=None,
                            turn=None, lanes=[], chosen=None, in_set=None))
            continue
        # 후보 풀 — _pair_choice 와 같은 방식
        ax, ay = wps[i_ent]
        bx, by = wps[i_ext]
        ay_in = math.atan2(ay - wps[i_ent - 1][1], ax - wps[i_ent - 1][0]) if i_ent else start_yaw_of(rt)
        ay_out = math.atan2(by - ay, bx - ax)
        ca = candidates(lg, ax, ay, radius, ay_in)
        cb = candidates(lg, bx, by, radius, ay_out)
        pool_in = road_lane_pool(lg, ca, ax, ay, ay_in) if ca else []
        pool_out = road_lane_pool(lg, cb, bx, by, ay_out) if cb else []
        # 진입 후보에서 **교차로 연결로를 뺀다**. 진입 경유점은 교차로 어귀에
        # 찍히므로 후보집합에 접근 차로와 그 successor 연결로가 같이 들어온다.
        # 연결로는 "어느 평행 차로에 있을 것인가" 의 선택지가 아니라 그 다음
        # 단계라, 넣어 두면 집합 크기가 실제 선택지의 두 배로 부풀고 좌회전인데
        # 2개로 보이는 착시가 생긴다 (실측: venue 4개 짝 전부).
        # pool_out 은 그대로 둔다 — 저건 turn_connect 의 목적지일 뿐이고,
        # 진출 경유점이 연결로 위에 찍혀도 "거기까지 간다" 는 성립한다.
        pool_in = [(k, sv, d) for k, sv, d in pool_in if lg.lanes[k]['junction'] == -1]
        ok = []
        for ka, sa, _da in pool_in:
            if any(turn_connect(lg, ka, sa, kb, sb, banned, cap) is not None
                   for kb, sb, _db in pool_out):
                ok.append(ka)
        # 회전 종류 — report() 와 같은 축(짝 구간 진입 시작 → 진출 끝 헤딩차)
        turn = None
        if target == 'pair' and nxt in spans:
            i0, i1 = spans[nxt]
            h0 = np.unwrap(lg.lanes[lanes_r[i0]]['hdg'].astype(float))
            h1 = np.unwrap(lg.lanes[lanes_r[i1]]['hdg'].astype(float))
            dh = math.degrees(wrap(float(h1[-1]) - float(h0[0])))
            turn = 'straight' if abs(dh) <= thr else ('left' if dh > 0 else 'right')
        # 경로가 실제로 고른 차로 (그 세그먼트가 끝나는 차로)
        # 경로가 실제로 고른 진입 차로. 세그먼트가 연결로 위에서 끝나는 경우가
        # 있어(경유점이 교차로 어귀에 찍히면 그렇다) **마지막 일반 도로 차로**를
        # 쓴다 — 집합과 같은 축이라야 in_set 이 뜻이 있다.
        chosen = None
        if wi in spans:
            i0, i1 = spans[wi]
            for i in range(i1, i0 - 1, -1):
                if lg.lanes[lanes_r[i]]['junction'] == -1:
                    chosen = tuple(lanes_r[i])
                    break
        okt = [tuple(k) for k in ok]
        out.append(dict(seg=wi, seq=(seqs[i_ent], seqs[i_ext]), target=target, turn=turn,
                        lanes=okt, chosen=chosen,
                        in_set=(chosen in okt) if chosen is not None else None))
    return out


_DP_CFG = None


def polyline_step_thr():
    """폴리라인 연속성 임계 [m] — configs/themes.yaml gen.max_polyline_step_m 이
    단일 출처다 (생성기 게이트와 같은 값을 봐야 한다). 없으면 0 = 검사 안 함."""
    try:
        import yaml
        p = _pathlib.Path(__file__).resolve().parent.parent / 'configs' / 'themes.yaml'
        g = (yaml.safe_load(open(p, encoding='utf-8')) or {}).get('gen') or {}
        return float(g.get('max_polyline_step_m', 0.0))
    except Exception:                                # noqa: BLE001
        return 0.0


DPCfg = _collections.namedtuple(
    'DPCfg', 'enable radius ratio floor detour_penalty step_penalty sep compare '
             'retry radius_max dev_penalty finish_lock retry_ratio retry_junction',
    # 뒤 두 개만 기본값을 준다 — 밖에서 11개짜리 위치인자로 DPCfg 를 짓는
    # 자리(tests/test_global_dp.py)가 있어서, 기본값 없이 늘리면 그쪽이 깨진다.
    defaults=(True, 0.0, False))


def dp_cfg(reload=False):
    """DPCfg — route.global_dp_* / dp_*.

    형식 무관 전역 경로 탐색(작업 R). 경유점 형식이 (진입,진출) 짝인지, 홑점인지,
    공유점인지 **모르는 채로** 최적 차로 열을 찾는다. 기본 false = 이전 동작.

    detour: 한 구간의 경로거리가 max(floor, ratio × 직선거리) 를 넘으면 **넘은
    만큼 × penalty** 를 비용에 더한다. 막지는 않는다 — 층 단위로 잘라내면 그
    층에서 살아남은 상태가 다음 층에서 전부 막히는 일이 생긴다 (실측:
    waypoints_pair_banned 에서 32 m 짜리 전이가 앞 층 가지치기로 사라졌다).
    왜 벌점이 필요한가: 경유점에서 가장 가까운 차로를 확정하고 넘어가면, 그
    차로에서 다음 경유점으로 가는 길이 블록 한 바퀴여도 알 수 없다 (2026-09-06
    PathShape03 CSV seq 9→10 이 직선 20 m 인데 경로 2081 m). 정상 구간 748개의
    비율은 중앙 1.00 · p99 1.41 이고 2.0 초과 3건이 전부 그 병증이다.
    """
    global _DP_CFG
    if _DP_CFG is not None and not isinstance(_DP_CFG, DPCfg):
        _DP_CFG = DPCfg(*_DP_CFG)        # 밖에서 평범한 튜플로 덮어써도 필드로 읽힌다
    if _DP_CFG is None or reload:
        from vtd_adapter.config import load_params_yaml
        r = load_params_yaml().get('route') or {}
        _DP_CFG = DPCfg(
            bool(r.get('global_dp_enable', False)),
            float(r.get('dp_match_radius_m', 8.0)),
            float(r.get('dp_detour_ratio', 3.0)),
            float(r.get('dp_detour_floor_m', 400.0)),
            float(r.get('dp_detour_penalty', 10.0)),
            float(r.get('dp_step_penalty', 0.0)),
            None,                      # 간격은 hop_sep_for 가 차로별로 준다
            bool(r.get('dp_compare_enable', True)),
            bool(r.get('dp_radius_retry_enable', False)),
            float(r.get('dp_radius_max_m', 16.0)),
            float(r.get('dp_radius_dev_penalty', 1.0)),
            bool(r.get('dp_finish_lane_lock', True)),
            float(r.get('dp_retry_ratio', 2.0)),
            bool(r.get('dp_retry_junction_enable', False)))
    return _DP_CFG


def polyline_max_step(lg, rt):
    """제어기가 실제로 따라갈 재샘플(VtdRoutePlanner, 10 cm)의 최대 점 간격 [m].

    생성기 폴리라인 게이트(gen_scenarios.polyline_gate)와 **같은 잣대**다.
    DP 가 고른 차로 열이 그 게이트에 걸리면 시나리오가 통째로 폐기되므로,
    여기서 미리 재 보고 탐욕 경로가 더 매끈하면 그쪽을 쓴다.
    """
    from vtd_adapter.config import load_params_yaml
    from vtd_adapter.route import VtdRoutePlanner
    pl = VtdRoutePlanner(lg, rt, load_params_yaml())
    pts = np.asarray(pl.route_points)
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    i = int(np.argmax(d))
    return float(d[i]), float(pl.route_s[i])


def lc_stack_excess(lg, path, sep_m=None):
    """경로 조각 안에서 연속 차선변경 hop 사이 **간격 부족분** 합 [m].

    폴리라인 불연속의 실제 원인이다. 차선변경 hop 은 route_s 를 진행시키지 않아
    (advance()=0) 두 번을 같은 자리에서 겹칠 수 있는데, 제어기 램프는 한 번에
    필요 간격(route.lc_hop_sep_m) 까지 쓴다 — 앞 램프가 끝나기 전에 다음이 시작하면
    재샘플 폴리라인이 튄다 (2026-09-06 실측: 간격 0.0 m → 3.14 m, 31.8 m → 0.42 m).
    """
    s = 0.0
    last = None
    ex = 0.0
    for i in range(len(path) - 1):
        k, k2 = path[i][0], path[i + 1][0]
        if k2 not in lg.lanes[k]['next']:
            if last is not None:
                need = sep_m if sep_m is not None else hop_sep_for(lg, k2)
                ex += max(0.0, need - (s - last))
            last = s
        s += advance(lg, k, k2, lg.length(k))
    return ex


def dp_point_yaw(waypoints, k, start_yaw=None):
    """점 k 의 진행방향 — 0번은 출발 헤딩(없으면 다음 점 방향), 그 외는 직전 점에서."""
    x, y = waypoints[k]
    if k == 0:
        if start_yaw is not None:
            return float(start_yaw)
        return math.atan2(waypoints[1][1] - y, waypoints[1][0] - x)
    return math.atan2(y - waypoints[k - 1][1], x - waypoints[k - 1][0])


def dp_point_candidates(lg, waypoints, k, radius, start_yaw=None):
    """점 k 의 후보 → ([(lane, s, dist)], 진단). 0개면 헤딩 완화 → 반경 1.5배."""
    x, y = waypoints[k]
    yaw = dp_point_yaw(waypoints, k, start_yaw)
    notes = []
    c = candidates(lg, x, y, radius, yaw)
    if not c:
        c = candidates(lg, x, y, radius)
        if c:
            notes.append(f'점 {k + 1}: 헤딩 필터 완화로 후보 {len(c)}개')
    if not c:
        c = candidates(lg, x, y, radius * 1.5)
        if c:
            notes.append(f'점 {k + 1}: 반경 {radius * 1.5:g} m 로 넓혀 후보 {len(c)}개')
    return c, notes


WideR = _collections.namedtuple('WideR', 'radius allow near')


def dp_wide_radius(lg, waypoints, k, base_r, max_r, junc=None):
    """점 k 를 덮어야 할 반경 [m] 과 그때 후보로 허용할 차로 → WideR.

    지도에 같은 방향 3차로 이상인 도로가 78개 있고 그중 33개는 첫↔끝 차로
    중심거리가 8 m 를 넘는다 (최대 15.84 m, 도로 1926). 경유점이 한쪽 차로에
    찍히면 반대쪽 끝 차로가 기본 반경 밖이라 DP 후보에 아예 안 들어온다 —
    그 교차로 회전이 반대쪽 차로에서만 되면 못 본다 (2026-09-07 분석).

    **최근접 후보가 교차로 연결로면 그 연결로의 폭을 봐서는 안 된다.**
    연결로는 언제나 1차로라 "도로 전폭" 이 1차로 폭으로 계산되고, 그러면
    반경이 base_r 그대로여서 재시도가 아예 무장하지 않는다. 대회 공식 형식은
    경유점이 전부 교차로 진입·진출부라 이 경우가 기본값이다 (2026-09-08
    PathShape04 seq 3: 최근접 (1940,0,-1) = junction 39 연결로 → wide 8.0,
    필요한 차로 (1926,0,6) 은 12.0 m 밖, seq 3→4 가 29 m 직선에 771 m 우회).

    그래서 교차로 안 점은 **접근 도로** 를 본다: 연결로의 진입 도로
    (predecessor) 를 우선으로 그 도로 같은 방향 전폭까지 넓히고, 후보도 그
    도로의 같은 방향 차로로 제한한다. 진입 도로가 max_r 밖이면 — 긴 연결로의
    **진출단** 에 찍힌 점이다 (실측 seq 4: 진입 도로 1869 이 53.9 m, 진출 도로
    1927 이 9.7 m) — 진출 도로(successor)로 대신한다. 둘 다 밖이면 안 넓힌다.

    allow 는 **넓힌 만큼에만** 적용되는 화이트리스트다 — base_r 안에서 이미
    잡히던 후보는 그대로 남는다. 재시도는 후보를 더하기만 해야 한다.
    """
    x, y = waypoints[k]
    base_c = candidates(lg, x, y, base_r)
    near = base_c or candidates(lg, x, y, max_r)
    if not near:
        return WideR(base_r, None, None)
    k0 = near[0][0]

    def _span(ref):
        """ref 도로 섹션의 같은 방향 주행차로 → (가장 가까운 차로까지, 가장 먼
        차로까지, 차로 집합). 거리는 lg.project — 점이 그 섹션 **옆**에 있을 때만
        가로 폭을 뜻한다. 섹션 끝을 지난 점에서는 세로 거리가 지배하므로
        (실측: seq1 이 접근 도로 1846 에서 42.58 m 인데 2차로 편차가 0.24 m
        뿐이다 — 폭이 아니라 길이를 잰 것이다), 가까운 쪽(lo)으로 그 섹션을
        쓸지 말지를 거른다."""
        allow, lo, hi = set(), float('inf'), base_r
        for kk in lg.lanes_of_road(ref[0]):
            rr = lg.lanes[kk]
            if rr.get('type') != 'driving' or kk[1] != ref[1] or (kk[2] > 0) != (ref[2] > 0):
                continue
            d = lg.project(kk, x, y)[2]
            allow.add(kk)
            lo, hi = min(lo, d), max(hi, d + 0.5)
        return lo, min(float(max_r), hi), allow

    if junc is None:
        junc = dp_cfg().retry_junction
    if not junc:                     # route.dp_retry_junction_enable=false = 이전 동작
        _lo, hi, _aw = _span(k0)
        return WideR(hi, None, k0)
    refs = [k0]
    if lg.lanes[k0].get('junction', -1) != -1:
        # 연결로는 언제나 1차로라 자기 폭으로는 못 넓힌다. 접근 도로를 본다:
        # 진입(predecessor) 우선, 그쪽이 max_r 밖이면(긴 연결로의 진출단에 찍힌
        # 점) 진출(successor). 둘 다 밖이면 이전처럼 안 넓힌다.
        pre = [kk for kk in lg.predecessors(k0) if lg.lane(kk).get('junction', -1) == -1]
        suc = [kk for kk in lg.successors(k0) if lg.lane(kk).get('junction', -1) == -1]
        pre = [kk for kk in pre if _span(kk)[0] <= max_r]
        suc = [kk for kk in suc if _span(kk)[0] <= max_r]
        refs = pre or suc or [k0]

    r, allow = base_r, set()
    for ref in refs:
        _lo, hi, aw = _span(ref)
        r = max(r, hi)
        allow |= aw
    # base_r 안 후보는 무조건 살린다 — 재시도는 후보를 **더하기만** 해야 한다.
    allow |= {kk for kk, _s, _d in (base_c or [])}
    return WideR(min(float(max_r), r), frozenset(allow), k0)


def _wide_cands(lg, waypoints, k, wide, start_yaw):
    """넓힌 반경의 후보 — WideR.allow 화이트리스트로 거른다.

    다른 도로·반대 방향은 반경 안이어도 뺀다. 넓히는 목적은 "같은 도로의 반대쪽
    끝 차로"를 보는 것이지 옆 도로를 끌어오는 것이 아니다. allow 는 base_r 안
    후보를 이미 품고 있어서 (dp_wide_radius) 이 필터가 기존 후보를 줄이지는
    않는다 — 재시도는 후보를 더하기만 한다.
    """
    c, _nt = dp_point_candidates(lg, waypoints, k, wide.radius, start_yaw)
    if not wide.allow:
        return c
    return [x for x in c if x[0] in wide.allow]


def _wide_warn(wide, pick):
    """재시도가 경유점이 찍힌 도로가 **아닌** 도로의 차로를 골랐으면 경고 꼬리."""
    if wide.near is None or pick[0] == wide.near[0]:
        return ''
    return f'  <= [경고] 경유점이 찍힌 도로는 {wide.near[0]} 다'


def dp_candidates(lg, waypoints, radius, start_yaw=None):
    """점마다 후보 차로 → ([ [(lane,s,dist)] ... ], [진단 문자열])

    후보가 0개면 **헤딩 필터 완화 → 반경 1.5배** 순으로 다시 잡는다. 그래도
    0이면 그 점은 None 으로 두고(=DP 에서 건너뛴다) 진단에 남긴다.
    """
    out, notes = [], []
    for k in range(len(waypoints)):
        c, nt = dp_point_candidates(lg, waypoints, k, radius, start_yaw)
        notes += nt
        if not c:
            notes.append(f'점 {k + 1}: 반경 {radius * 1.5:g} m 안에 차로 없음 — 이 점을 건너뛴다')
        out.append(c or None)
    return out, notes


def dp_chain(lg, waypoints, radius, start_yaw, banned, seqs=None, cfg=None):
    """형식 무관 전역 탐색 → (seq, seg_span, info)

    · 점 k 의 후보 C_k (dp_candidates)
    · 전이 비용 = dijkstra(c_k → c_k+1). 길이 · 차선변경(hop_gap) · 회전 불가
      연결로 금지 · 소멸 차로 벌점이 전부 그 안에 있고, 교차로 차로 위
      차선변경만 여기서 추가로 막는다 (lc_in_junction=False).
    · Viterbi 로 전체 최소. 동률이면 **경유점 거리 합**이 작은 쪽.
    · 마지막 층은 직전 구간 진행방향에 맞는 후보만 남는다 (dp_candidates 의
      헤딩 필터가 그 방향으로 잡는다).

    반환 seq/seg_span 은 탐욕 경로가 만드는 것과 같은 형식이다 — 뒤쪽
    조립(누적거리·이벤트·valid_entry_lanes·리포트)은 손대지 않는다.
    """
    C = cfg or dp_cfg()
    ratio, floor, dpen, spen, sep_m = C.ratio, C.floor, C.detour_penalty, C.step_penalty, C.sep
    cands, notes = dp_candidates(lg, waypoints, radius, start_yaw)
    idx = [k for k, c in enumerate(cands) if c]
    if len(idx) < 2:
        raise RouteError('전역 DP: 후보가 있는 경유점이 2개 미만이다')

    def lab(k):
        return f'seq {seqs[k]}' if seqs else f'waypoint {k}'

    INF = float('inf')
    best = [(0.0, c[2]) for c in cands[idx[0]]]      # (누적비용, 누적 경유점거리)
    parent = [[None] * len(cands[idx[0]])]           # 층별 부모 인덱스
    seg_path = [{}]                                  # 층별 {(i,j): dijkstra path}
    best_hist = [best]                               # 층별 best (반경 재시도 되감기용)
    lims = [0.0]                                     # 층별 우회 상한
    relaxed, n_edge, retries = [], 0, []

    def layer(ka, kb, cand_b, lim, dev0=0.0, best_in=None, cand_a=None):
        """한 층 계산 → (cur, par, pth, edges). dev0>0 이면 그 반경을 넘은 후보에
        경유점 거리 비례 벌점을 물린다 (반경 재시도로 들어온 먼 차로)."""
        cur, par, pth, ed = [], [], {}, 0
        bin_ = best if best_in is None else best_in
        ca = cands[ka] if cand_a is None else cand_a
        for attempt in (0, 1):
            bans = banned if attempt == 0 else frozenset()
            cur, par, pth = [], [], {}
            for j, (kj, sj, dj) in enumerate(cand_b):
                bi, bc, bd = None, INF, INF
                dev = C.dev_penalty * max(0.0, dj - dev0) if dev0 > 0 else 0.0
                for i, (ki, si, _di) in enumerate(ca):
                    if bin_[i][0] == INF:
                        continue
                    res = dijkstra(lg, [(ki, si)], {kj: sj}, allow_lane_change=True,
                                   banned=bans, lc_in_junction=False)
                    ed += 1
                    if res is None:
                        continue
                    pen = dpen * max(0.0, res[0] - lim) + dev
                    if spen > 0.0:
                        pen += spen * lc_stack_excess(lg, res[1], sep_m)
                    c = bin_[i][0] + res[0] + pen
                    d = bin_[i][1] + dj
                    if c < bc - 1e-9 or (abs(c - bc) <= 1e-9 and d < bd - 1e-9):
                        bi, bc, bd = i, c, d
                        pth[j] = (res[1], res[0], pen)
                cur.append((bc, bd))
                par.append(bi)
            if any(v[0] < INF for v in cur):
                break
            if attempt == 0:
                relaxed.append(f'{lab(ka)}→{lab(kb)} 회전 불가 연결로를 빼면 연결 없음 — 금지 해제')
        return cur, par, pth, ed

    for t in range(1, len(idx)):
        ka, kb = idx[t - 1], idx[t]
        straight = math.dist(waypoints[ka], waypoints[kb])
        lim = max(floor, ratio * straight)
        # 금지 연결로는 하드 제약이다. 그것 때문에 층이 통째로 막히면 그때만
        # 풀고 기록한다 (탐욕의 "대안이 없으면 불가피하게 허용하고 기록" 과 같다).
        cur, par, pth, ed = layer(ka, kb, cands[kb], lim)
        n_edge += ed
        # ── 반경 재시도 (route.dp_radius_retry_enable) ─────────────────────
        # 최선 전이가 우회 벌점을 물었거나, 경로/직선 비율이 dp_retry_ratio 를
        # 넘었거나, 전부 막혔으면 — 그 점만 "도로 같은 방향 전체 차로를 덮는
        # 반경"(교차로 연결로면 접근 도로, dp_retry_junction_enable)으로 후보를
        # 다시 잡고 이 층만 재계산한다.
        # 마지막 경유점은 재시도 대상이 아니다 — 그 점이 완주 판정의 기준
        # (finish_xy)이라, 먼 차로로 옮기면 경로가 종점을 8 m 밖으로 비껴간다
        # (실측: venue 계열 6경로에서 마지막 경유점 투영이 사라졌다).
        if C.retry and kb != idx[-1]:
            jbest = min(range(len(cur)), key=lambda j: cur[j][0]) if cur else None
            raw = (pth.get(jbest) or (None, 0.0, 0.0))[1] if jbest is not None else 0.0
            # 우회 벌점만으로는 부족하다: 벌점은 raw > max(floor 400, ratio×직선)
            # 일 때만 붙으므로 **400 m 아래 우회는 전부 안 보인다**. 실측
            # 2026-09-08 PathShape04 seq 3→4 는 직선 29 m 에 경로 771 m 라
            # 벌점이 붙었지만, 같은 병증이 350 m 로 났으면 아무 표시가 없다.
            # 그래서 경로/직선 비율도 함께 본다 (dp_retry_ratio, 0 = 끔).
            over = (C.retry_ratio > 0.0 and straight > 1.0
                    and raw > C.retry_ratio * straight)
            bad = (jbest is None or cur[jbest][0] == INF
                   or (pth.get(jbest) or (None, 0.0, 0.0))[2] > 0.0 or over)
            base_best = cur[jbest][0] if jbest is not None else INF
            # (a) 진출점 kb 를 넓혀 본다
            if bad:
                w2 = dp_wide_radius(lg, waypoints, kb, radius, C.radius_max,
                                    junc=C.retry_junction)
                r2 = w2.radius
                c2 = (_wide_cands(lg, waypoints, kb, w2, start_yaw)
                      if r2 > radius + 1e-9 else None)
                if c2 and len(c2) > len(cands[kb]):
                    cur2, par2, pth2, ed2 = layer(ka, kb, c2, lim, dev0=radius)
                    n_edge += ed2
                    j2 = min(range(len(cur2)), key=lambda j: cur2[j][0]) if cur2 else None
                    if j2 is not None and cur2[j2][0] < base_best - 1e-9:
                        retries.append(
                            f'{lab(kb)} 반경 재시도 {radius:g} → {r2:.1f} m, '
                            f'채택 차로 {c2[j2][0]} 이탈 {c2[j2][2]:.2f} m'
                            + _wide_warn(w2, c2[j2][0]))
                        cands[kb] = c2
                        cur, par, pth = cur2, par2, pth2
                        base_best = cur2[j2][0]
                        bad = False
            # (b) 진입점 ka 를 넓힌다 — 앞 층까지 되감아 다시 센다.
            # 넓혀야 할 쪽은 대개 **진입점**이다: 경유점이 한쪽 차로에 찍혀서
            # 회전 가능한 반대쪽 차로가 후보에 없는 것이 원래 문제다.
            if bad:
                w3 = dp_wide_radius(lg, waypoints, ka, radius, C.radius_max,
                                    junc=C.retry_junction)
                r3 = w3.radius
                c3 = (_wide_cands(lg, waypoints, ka, w3, start_yaw)
                      if r3 > radius + 1e-9 else None)
                if c3 and len(c3) > len(cands[ka]):
                    if t == 1:                     # 첫 층 — best 를 다시 깐다
                        b3 = [(C.dev_penalty * max(0.0, c[2] - radius), c[2]) for c in c3]
                        par3, pth3, ed3 = [None] * len(c3), {}, 0
                    else:                          # 앞 층을 c3 로 다시 계산
                        b3, par3, pth3, ed3 = layer(idx[t - 2], ka, c3, lims[t - 1],
                                                    dev0=radius, best_in=best_hist[t - 2])
                        n_edge += ed3
                    if any(v[0] < INF for v in b3):
                        cur3, par3b, pth3b, ed4 = layer(ka, kb, cands[kb], lim,
                                                        best_in=b3, cand_a=c3)
                        n_edge += ed4
                        j3 = min(range(len(cur3)), key=lambda j: cur3[j][0]) if cur3 else None
                        if j3 is not None and cur3[j3][0] < base_best - 1e-9:
                            i3 = par3b[j3]
                            retries.append(
                                f'{lab(ka)} 반경 재시도 {radius:g} → {r3:.1f} m, '
                                f'채택 차로 {c3[i3][0]} 이탈 {c3[i3][2]:.2f} m'
                                + _wide_warn(w3, c3[i3][0]))
                            cands[ka] = c3
                            best = best_hist[t - 1] = b3
                            parent[t - 1] = par3
                            seg_path[t - 1] = pth3
                            cur, par, pth = cur3, par3b, pth3b
        if not any(v[0] < INF for v in cur):
            raise RouteError(
                f'전역 DP: {lab(ka)} → {lab(kb)} 구간에 연결 가능한 차로 조합이 없다 '
                f'(후보 {len(cands[ka])}×{len(cands[kb])})')
        best, _ = cur, None
        parent.append(par)
        seg_path.append(pth)
        best_hist.append(best)
        lims.append(lim)

    # ── 종점 차로 고정 (route.dp_finish_lane_lock) ────────────────────────
    # 완주 규칙이 "뒷축이 **두 콘 사이** 종료선 통과" 라, 경로가 종료 좌표가
    # 놓인 차로의 옆 차로로 끝나면 종료선을 넘고도 콘 밖이 된다. 실측
    # 2026-09-07: 생성 경로 21개 중 9개가 그랬고(경로 중심선에서 ±3~6 m),
    # 그중 5개는 종료선 콘이 주행 회랑 안으로 들어왔다.
    #
    # 그래서 마지막 층에서는 **비용보다 차로 정합을 우선**한다: 종료 좌표가
    # 실제로 올라앉은 차로가 후보에 있고 도달 가능하면 비용과 무관하게 채택하고,
    # 도달 불가일 때만 최소비용 후보로 떨어진다.
    #
    # "올라앉았다" 의 기준은 그 차로 **반폭 안** 이다 — 좌표가 차로 경계 밖이면
    # 어느 차로의 것인지가 애매하므로 고정하지 않는다. cands 는 거리순 정렬이라
    # (candidates() 의 sorted(key=dist)) 첫 후보가 가장 가까운 차로다.
    #
    # 반경 재시도와는 충돌하지 않는다: 재시도는 `C.retry and kb != idx[-1]` 로
    # **마지막 경유점을 이미 제외**하므로 cands[idx[-1]] 을 건드리지 않는다.
    # 이 고정은 그 후보 목록을 읽기만 한다.
    order = sorted(range(len(best)), key=lambda j: (best[j][0], best[j][1]))
    endj = order[0]
    lock = None
    if C.finish_lock and cands[idx[-1]]:
        k0, s0, d0 = cands[idx[-1]][0]
        on_lane = d0 <= lg.width_at(k0, s0) / 2.0
        j0 = next((j for j, c in enumerate(cands[idx[-1]]) if c[0] == k0), None)
        lock = {'lane': list(k0), 'dev_m': round(float(d0), 3),
                'on_lane': bool(on_lane), 'was': list(cands[idx[-1]][endj][0]),
                'last_point_used': idx[-1] == len(waypoints) - 1}
        if not on_lane:
            lock['why'] = (f'종료 좌표가 {k0} 경계 밖({d0:.2f} m > 반폭 '
                           f'{lg.width_at(k0, s0) / 2.0:.2f} m) — 고정하지 않는다')
        elif j0 is None or best[j0][0] == INF:
            lock['why'] = f'{k0} 로 이어지는 차로 조합이 없다 — 최소비용 후보로 떨어진다'
        elif j0 == endj:
            lock['why'] = '최소비용 후보가 이미 종료 좌표 차로다'
        else:
            lock['why'] = (f'비용 {best[endj][0]:.0f} → {best[j0][0]:.0f} 를 물고 '
                           f'종료 좌표 차로 {k0} 로 고정')
            endj = j0
        lock['applied'] = (endj == j0 and j0 is not None and on_lane)
    picks = [None] * len(idx)
    picks[-1] = endj
    for t in range(len(idx) - 1, 0, -1):
        picks[t - 1] = parent[t][picks[t]]
    total_cost, total_d = best[endj]

    # ── seq / seg_span 조립 (탐욕과 같은 방식) ────────────────────────────
    seq, seg_span, detours = [], [], []
    for t in range(1, len(idx)):
        path, raw_cost, pen = seg_path[t][picks[t]]
        if pen > 0:
            ka, kb = idx[t - 1], idx[t]
            # pen 은 우회 벌점과 **반경 재시도 거리 벌점**의 합이다. 둘을 다
            # "우회" 로 찍으면 우회가 없는 구간에도 경고가 뜬다 (2026-09-08:
            # 재시도로 고친 PathShape04 가 277/216 m 구간에 우회 경고를 냈다).
            why = ('우회 벌점' if raw_cost > lims[t] + 1e-9 else '반경 재시도 거리 벌점')
            detours.append(f'{lab(ka)}→{lab(kb)} 경로 {raw_cost:.0f} m / 직선 '
                           f'{math.dist(waypoints[ka], waypoints[kb]):.0f} m — '
                           f'{why}을 물고 채택')
        i0 = max(0, len(seq) - 1)
        for k, s_en in path:
            if seq and seq[-1][0] == k:
                continue
            seq.append((k, s_en))
        # 건너뛴 점이 있으면 그 구간들을 한 세그먼트로 본다 (앞 wi 로 기록)
        for wi in range(idx[t - 1], idx[t]):
            seg_span.append((wi, i0, len(seq) - 1))
    # 회전 불가 연결로를 불가피하게 포함했으면 **구간 인덱스**로 기록한다.
    # 리포트가 "구간 {wi}" 로 찍으므로 seq 인덱스를 넣으면 엉뚱한 번호가 된다
    # (end_pos 와 같은 종류의 실수 — 2026-09-06 감사).
    forced, seen_bad = [], set()
    for wi, i0, i1 in seg_span:
        for j in range(i0, min(i1 + 1, len(seq))):
            k = seq[j][0]
            if k in banned and k not in seen_bad:
                seen_bad.add(k)
                forced.append((wi, k, banned[k]))
    # 마지막 경유점이 실제로 앉은 (차로, s). 꼬리 계산이 이걸 봐야 한다 —
    # seq[-1] 은 마지막 차로에 **진입한** s 라 경유점 위치가 아니다.
    k_end, s_end, _d_end = cands[idx[-1]][picks[-1]]
    info = {
        'used': True,
        'end_pos': (k_end, float(s_end)),
        'forced_infeasible': forced,
        'n_points': len(waypoints),
        'n_used_points': len(idx),
        'skipped_points': [k for k, c in enumerate(cands) if not c],
        'cand_counts': [0 if not c else len(c) for c in cands],
        'edges': n_edge,
        'cost': round(float(total_cost), 1),
        'wp_dist_sum': round(float(total_d), 2),
        'picks': [list(cands[k][picks[t]][0]) for t, k in enumerate(idx)],
        'finish_lock': lock,
        'notes': notes,
        'relaxed': relaxed + detours,
        'retries': retries,
    }
    return seq, seg_span, info


def start_yaw_of(rt):
    """경로 시작 헤딩 (첫 두 경유점) — valid_entry_lanes 의 진입 헤딩 폴백."""
    w = rt['waypoints']
    return math.atan2(w[1][1] - w[0][1], w[1][0] - w[0][0])


def build_route(lg, waypoints, radius=8.0, start_yaw=None, junction_segs=frozenset(),
                seqs=None, finish_tail_m=0.0, pair_meta=None, _dp=True, dp_radius=None):
    """pair_meta: {'offset','source','why'} — 짝 해석을 pkl 에 기록만 한다.
    경로 계산에는 쓰지 않는다 (계산은 junction_segs 가 전부다)."""
    # 회전 불가 연결로 (R_min < 최소회전반경 × 여유) — dijkstra 에서 통행 금지
    banned, turn_thr = infeasible_connectors(lg)
    forced_infeasible: list = []   # 대안이 없어 불가피하게 포함시킨 연결로
    pair_fallbacks: list = []      # 짝 탐색이 0개라 기존 탐욕으로 폴백한 구간
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
    pair_hint, turn_cap, _thr = pair_cfg()
    # ── 형식 무관 전역 탐색 (작업 R) ──────────────────────────────────────
    # 켜면 아래 구간별 탐욕 대신 DP 가 차로 열을 정한다. 짝 해석은 대조용으로만
    # 돈다 (호출부 build_route(_dp=False)). 끄면 이전 동작 그대로다.
    dp_on = dp_cfg()[0] and _dp and len(waypoints) >= 2
    dp_info = None
    if dp_on:
        # 후보 반경은 route.dp_match_radius_m 이 단일 출처다. CLI --radius 를
        # 사람이 준 경우에만 그쪽이 이긴다 (dp_radius 로 넘어온다).
        r_dp = float(dp_radius) if dp_radius is not None else float(dp_cfg()[1])
        seq, seg_span, dp_info = dp_chain(lg, waypoints, r_dp, start_yaw, banned,
                                          seqs=seqs)
        dp_info['radius_m'] = r_dp
        wp_s = [0.0] + [None] * (len(waypoints) - 1)
        prev_end = dp_info['end_pos']       # 마지막 경유점이 앉은 (차로, s) — 꼬리 계산용
        forced_infeasible.extend(dp_info.pop('forced_infeasible', []))
    skip_next = False
    for wi in range(len(waypoints) - 1):
        if dp_on:
            break                           # 차로 열은 위에서 DP 가 정했다
        if skip_next:                       # 앞 반복에서 짝으로 함께 처리했다
            skip_next = False
            continue
        x0, y0 = waypoints[wi]
        x1, y1 = waypoints[wi + 1]
        if prev_end is None:
            # 출발점: 헤딩 포함해서 하나로 확정 (여러 후보를 주면 바로 앞 차로가 선택되는 문제)
            #
            # locate() 는 lanegraph 안에서 kd.query(k=16) 으로 후보를 뽑는다 —
            # candidates 가 갖고 있던 것과 **같은 개수 절단 결함**이다. 출발점이
            # 중심선에서 벗어나 있으면 조용히 옆 차로를 고르고, 그러면 에러 없이
            # 경로 전체가 어긋난다 (가장 위험한 실패 모드).
            # 2026-09-03 실측: 대회장 CSV 출발점에서 k=16 도달거리는 3.06 m 뿐이고
            # 정답 차로가 0.14 m 라 우연히 맞았다 — 3 m 넘게 벗어나면 안 맞는다.
            #
            # lanegraph.py 는 제어기 런타임(EgoTracker)이 같이 쓰는 파일이라 여기서
            # 고치지 않는다. 대신 **locate 의 점수 규칙을 반경 기반 후보집합에 그대로
            # 적용**해 보고, 더 나은 차로가 나오면 그쪽을 쓰고 경고한다. 바꾸는 건
            # 후보집합뿐이라 locate 가 헤딩을 보고 일부러 조금 먼 차로를 고른 판단은
            # 그대로 살아 있고, candidates 만의 필터(폭 2 m 미만 포켓 배제)로 빠진
            # 차로 때문에 더 가까운 정답을 밀어내는 일도 없다.
            cand0 = candidates(lg, x0, y0, radius, start_yaw)
            m0 = lg.locate(x0, y0, start_yaw, max_dist=radius)
            override = None
            if m0 is not None and cand0 and start_override_cfg():
                sc_m0 = locate_score(lg, m0.lane, m0.s, m0.dist, start_yaw)
                best0 = min(cand0, key=lambda c: locate_score(lg, c[0], c[1], c[2], start_yaw))
                sc_b0 = locate_score(lg, best0[0], best0[1], best0[2], start_yaw)
                if best0[0] != m0.lane and sc_b0 < sc_m0 - 1e-9:
                    print(f'  [경고] {label(wi)} ({x0:.2f},{y0:.2f}) 출발 차로 불일치 — '
                          f'locate {m0.lane} @{m0.dist:.2f}m (점수 {sc_m0:.3f}) vs '
                          f'반경후보 {best0[0]} @{best0[2]:.2f}m (점수 {sc_b0:.3f}). '
                          f'반경후보를 쓴다 — locate 는 kd.query(k=16) 절단 '
                          f'(vtd_adapter/lanegraph.py)', file=sys.stderr)
                    override = [(best0[0], best0[1])]
            if override is not None:
                starts = override
            elif m0 is not None:
                starts = [(m0.lane, m0.s)]
            else:
                starts = [(k, s) for k, s, d in cand0[:6]]
            if not starts:
                raise RouteError(
                    f'{label(wi)} ({x0:.2f},{y0:.2f}): 반경 {radius:g}m 내 차로 없음 '
                    f'(최근접 {nearest_report(x0, y0):.1f}m)')
        else:
            starts = [prev_end]

        # ── 교차로 짝 공동 선택 ──────────────────────────────────────────
        # 진입 경유점(waypoints[wi+1])의 차로를 확정하기 **전에** 진출
        # 경유점(waypoints[wi+2])까지 함께 본다. 짝 사이는 차선변경 금지라
        # 진입이 틀리면 복구가 없고, 현재 탐욕은 진출점을 보지 않는다.
        if pair_hint and (wi + 1) in junction_segs and wi + 2 < len(waypoints):
            got = _pair_choice(lg, starts, waypoints, wi, radius, junction_segs,
                               banned, turn_cap, label, seqs,
                               fallbacks=pair_fallbacks)
            if got is not None:
                (pa, k_in, s_in, pb, k_out, s_out, ub) = got
                for kk in ub:
                    forced_infeasible.append((wi, kk, banned[kk]))
                if wi == 0:
                    wp_s.append(0.0)
                i0 = max(0, len(seq) - 1)
                for k, s_en in pa:
                    if seq and seq[-1][0] == k:
                        continue
                    seq.append((k, s_en))
                seg_span.append((wi, i0, len(seq) - 1))
                i0b = max(0, len(seq) - 1)
                for k, s_en in pb:
                    if seq and seq[-1][0] == k:
                        continue
                    seq.append((k, s_en))
                seg_span.append((wi + 1, i0b, len(seq) - 1))
                prev_end = (k_out, s_out)
                wp_s.append(None); wp_s.append(None)
                skip_next = True
                continue
            # 회전 가능 짝이 0개 — 아래 기존 탐욕으로 폴백한다 (WARN 은 _pair_choice 가 냈다)

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
                res = dijkstra(lg, starts, {k: s}, allow_lane_change=allow_lc,
                               banned=banned)
                used_banned = []
                if res is None:
                    # 금지를 풀면 연결되는가 — 대안이 없어 **불가피**한 경우만 허용하되
                    # 조용히 넘기지 않고 기록한다 (리포트에 ⚠).
                    res = dijkstra(lg, starts, {k: s}, allow_lane_change=allow_lc)
                    if res is None:
                        continue
                    used_banned = [kk for kk, _ in res[1] if kk in banned]
                score = res[0] + TARGET_DIST_W * d
                if best is None or score < best[0]:
                    best = (score, res[1], k, s, used_banned)
            if best is not None:
                break
        if best is None:
            # 제약을 풀지 않는다. 풀면 경로가 통째로 엉뚱한 데로 새면서
            # "성공했지만 완전히 틀린 경로" 가 나온다. 실패를 그대로 보고한다.
            #
            # 현장에서 이 메시지만 보고 판단해야 하므로 **후보별 거리와 dijkstra
            # 결과를 전부** 싣는다 (예전에는 어느 seq 가 왜 막혔는지 알려면 따로
            # 스크립트를 짜야 했다).
            lines = []
            for k, s, d in tg:
                r_lc = dijkstra(lg, starts, {k: s}, allow_lane_change=allow_lc,
                                banned=banned)
                r_free = None if r_lc is not None else dijkstra(
                    lg, starts, {k: s}, allow_lane_change=True)
                lines.append('%s @%.2fm  %s%s' % (
                    k, d,
                    '연결 %.0fm' % r_lc[0] if r_lc else '연결 X',
                    '' if r_lc or r_free is None else ' (차선변경 허용하면 %.0fm)' % r_free[0]))
            extra = ''
            if not allow_lc:
                extra = ('\n       이 구간은 교차로 내부(짝)라 차선변경이 금지돼 있다.'
                         '\n       진입 차로가 이미 틀렸을 가능성이 크다 — 앞 구간이 끝난 차로를 확인할 것.')
                if any(l.endswith(')') for l in lines):
                    extra += '\n       (차선변경을 허용하면 연결된다 = 차로 선택 문제)'
            if pair_hint and (wi + 1) in junction_segs:
                extra += ('\n       짝 공동 선택이 회전 가능한 (진입,진출) 짝을 찾지 못해'
                          ' 탐욕으로 폴백한 뒤 실패했다 (위 [경고] 참조).')
            raise RouteError(
                f'{label(wi)} ({x0:.2f},{y0:.2f}) -> {label(wi + 1)} ({x1:.2f},{y1:.2f}): '
                f'경로 없음\n       출발 차로 {starts[0][0]} @s={starts[0][1]:.1f}'
                f'\n       진출 후보 {len(tg)}개:\n              ' + '\n              '.join(lines)
                + extra)
        _score, path, k_end, s_end, used_banned = best
        for kk in used_banned:
            forced_infeasible.append((wi, kk, banned[kk]))
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
        cum.append(cum[i - 1] + advance(lg, lanes[i - 1], lanes[i], lengths[i - 1]))
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
                    w_s0, w_s1, j0, s_in = lane_change_window(lg, lanes, cum, seq, i, side, k2)
                    events.append({'kind': f'lane_change_{side}', 's': w_s0,
                                   'lane': lanes[j0], 's_in_lane': s_in,
                                   'window_s0': w_s0, 'window_s1': w_s1, 'to_lane': k2,
                                   'from_lane': k})
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
            if abs(dh) > math.radians(pair_cfg()[2]):
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
        if abs(dh) <= pair_cfg()[2]:
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

    # ── 종료선 뒤 경로 꼬리 확보 ─────────────────────────────────────────
    # 마지막 경유점이 차로 끝 근처에 매칭되면 꼬리가 우연히 짧아져 finish 정지가
    # 클립되고 뒷축이 종료선 앞에 선다 (plan_stop_s 요구량은 finish_tail_cfg 참조).
    # 이벤트·경유점 투영이 끝난 뒤에 연장한다 — 꼬리 차로가 junction 연결로여도
    # 가짜 turn 이벤트(지시등 점등)가 생기지 않게 하기 위해서다.
    # (2026-09-03 실측: 대회형식 waypoints.csv 꼬리 15.0 m, tests/fixtures 0.2 m)
    tail = tail0 = total - (cum[-1] + prev_end[1])
    if finish_tail_m > 0:
        added = []
        while tail < finish_tail_m and len(added) < 8:
            succs = lg.successors(lanes[-1])
            if not succs:
                print(f'  [경고] 종료선 뒤 꼬리 {tail:.1f} m < 요구 {finish_tail_m:g} m — '
                      f'successor 가 없어 연장 불가 (finish 정지가 클립될 수 있다)',
                      file=sys.stderr)
                break
            # 회전 불가 연결로는 피하되 그것뿐이면 그냥 쓴다 — 꼬리는 계획
            # 정지점 뒤라 실제로 끝까지 달리지 않는다 (기하 확보용).
            pool = [k for k in succs if k not in banned] or succs
            h_end = float(np.unwrap(lg.lanes[lanes[-1]]['hdg'].astype(float))[-1])
            k2 = min(pool, key=lambda k: abs(wrap(float(lg.lanes[k]['hdg'][0]) - h_end)))
            cum.append(cum[-1] + lengths[-1])
            lanes.append(k2)
            lengths.append(lg.length(k2))
            tail += lengths[-1]
            added.append(k2)
        if added:
            total = cum[-1] + lengths[-1]
            print(f'  경로 꼬리 연장: 잔여 {tail0:.1f} m < 요구 {finish_tail_m:g} m → '
                  f'{" → ".join(str(k) for k in added)}  (꼬리 {tail:.1f} m)')

    # 급회전 연결로 (banned 임계 ≤ R_min < 기하 최소회전반경) — 통행은 하되
    # 리포트 WARN + 여기에 기록한다. 커브 감속(speed.curvature_cap)이 이 R 에
    # 맞춰 속도를 눌러 주므로 금지 대신 '감속 진입'으로 다룬다 (2026-09-08).
    tight_thr = tight_turn_r_m()
    tight_turns = []
    for i, k in enumerate(lanes):
        if lg.lanes[k]['junction'] == -1:
            continue
        r = lane_r_min(lg, k)
        if turn_thr <= r < tight_thr:
            tight_turns.append({'lane': list(k), 'r_min_m': round(r, 2),
                                's_m': round(float(cum[i]), 1),
                                'junction': int(lg.lanes[k]['junction'])})

    rt = {'lanes': lanes, 'cum_s': cum, 'lengths': lengths, 'total_length': total, 'start_s_in_lane': s_first,
            'infeasible_forced': forced_infeasible, 'turn_radius_thr_m': turn_thr,
            'tight_turns': tight_turns, 'tight_turn_thr_m': tight_thr,
            'pair_fallbacks': pair_fallbacks, 'dp': dp_info,
            'finish_xy': [float(waypoints[-1][0]), float(waypoints[-1][1])],
            'waypoints': [tuple(w) for w in waypoints], 'waypoint_s': wp_dist, 'events': events,
            'waypoint_seq': list(seqs) if seqs else list(range(1, len(waypoints) + 1)),
            'junction_segments': sorted(junction_segs), 'segment_span': seg_span,
            # 어느 짝 해석으로 지었는지 (작업21). 리포트·현장 확인·사후 추적용.
            'pair_offset': (pair_meta or {}).get('offset'),
            'pair_offset_source': (pair_meta or {}).get('source', 'caller'),
            'pair_offset_why': (pair_meta or {}).get('why', '')}
    # 폴백 구간이 **실제로 무엇을 골랐는지** 채운다 (제안③). 경고만 보고는
    # 사후 추적이 안 됐다 — 경고는 stderr 로만 나가고 어느 연결로를 탔는지는
    # 어디에도 안 남았다 (2026-09-06 G 분석).
    if pair_fallbacks:
        span = {}
        for wi_s, i0, i1 in seg_span:
            a, b = span.get(wi_s, (i1, i0))
            span[wi_s] = (min(a, i0), max(b, i1))
        for fb in pair_fallbacks:
            got_span = span.get(fb['wi'])
            if got_span is None:
                continue
            i0, i1 = got_span
            fb['connectors'] = [[list(k), round(lane_r_min(lg, k), 2)]
                                for k in lanes[i0:i1 + 1]
                                if lg.lanes[k]['junction'] != -1]
            fb['segment_m'] = round(float(cum[min(i1, len(cum) - 1)] - cum[i0]), 1)
        if pair_fallback_cfg()[0]:
            for fb in pair_fallbacks:
                print(f"  [경고] {fb['label']} 폴백이 고른 연결로: "
                      f"{_fmt_fb_connectors(fb)}  구간 {fb.get('segment_m', '?')} m",
                      file=sys.stderr)

    # ── 직선 구간 유효 차로 집합 (작업10) — 정보 필드다. 경로 계산에 안 쓴다.
    # 아무도 안 읽으면 영향 0 이어야 하므로 여기서 **덧붙이기만** 한다.
    if valid_entry_lanes_enable():
        rt['valid_entry_lanes'] = valid_entry_lanes(lg, rt, radius, banned, turn_cap)

    # ── 짝 해석 대조 (DP 로 지었을 때만) ──────────────────────────────────
    # 선택은 자동이다 — DP 결과를 쓰고, 짝 해석은 **다르면 알리기 위해서만** 돈다.
    if dp_on and junction_segs and dp_cfg()[7]:
        cmp_out = io.StringIO()
        try:
            with contextlib.redirect_stdout(cmp_out), contextlib.redirect_stderr(cmp_out):
                rt_p = build_route(lg, waypoints, radius, start_yaw, junction_segs,
                                   seqs, finish_tail_m, pair_meta, _dp=False,
                                   dp_radius=dp_radius)
            same = ([tuple(k) for k in rt_p['lanes']] == [tuple(k) for k in rt['lanes']])
            dp_info['pair_cmp'] = {
                'ok': True, 'same': same,
                'pair_total': round(float(rt_p['total_length']), 1),
                'pair_lanes': len(rt_p['lanes']),
                'pair_fallbacks': len(rt_p.get('pair_fallbacks') or []),
            }
            # 연속성 안전망 — DP 가 고른 차로 열이 폴리라인 게이트에 걸리면
            # 시나리오가 통째로 폐기된다. 탐욕 쪽이 더 매끈하면 그쪽을 쓴다.
            thr = polyline_step_thr()
            if not same and thr > 0:
                with contextlib.redirect_stdout(cmp_out), contextlib.redirect_stderr(cmp_out):
                    d_dp, s_dp = polyline_max_step(lg, rt)
                    d_p, _s_p = polyline_max_step(lg, rt_p)
                dp_info['polyline_step_m'] = round(d_dp, 3)
                dp_info['pair_cmp']['polyline_step_m'] = round(d_p, 3)
                if d_dp > thr and d_p < d_dp:
                    dp_info['fallback_to_pair'] = (
                        f'DP 경로 폴리라인 불연속 {d_dp:.2f} m (>{thr:g}, route_s≈{s_dp:.0f} m) — '
                        f'짝 경로({d_p:.2f} m)를 쓴다')
                    rt_p['dp'] = dp_info
                    return rt_p
        except RouteError as e:
            dp_info['pair_cmp'] = {'ok': False, 'same': False,
                                   'error': str(e).splitlines()[0][:160]}
    return rt


def turn_kind(delta_deg, straight_deg=None):
    """직진/좌회전/우회전. 임계는 params route.turn_heading_thr_deg 단일 출처."""
    if straight_deg is None:
        straight_deg = pair_cfg()[2]
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
    rc = route_check_cfg()
    hw_k = float(rc['wp_dev_halfwidth_k'])
    pair_exempt = bool(rc.get('pair_waypoint_exempt_enable', True))
    max_dist = float((route_cfg() or {}).get('check_waypoint_max_dist_m', 6.0))
    banned, _thr0 = infeasible_connectors(lg)
    turn_cap = pair_cfg()[1]
    # 경고 심각도 2단 (2026-09-04). 가르는 기준 하나: **이 경로로 달리면
    # 물리적으로 실패하거나 채점 위반이 확정되는가.** 그러면 ERROR(rc=1),
    # 아니면 WARN(정보성, rc 에 반영 안 함).
    #   ERROR  교차로 내부 차선변경 / 회전 수행 불가 / LC 창 부족 /
    #          회전 불가 기하 / 불가피 포함된 회전 불가 연결로
    #   WARN   경유점 이탈 / junction 차로 미경유 / 총 길이 비율
    # warn_affects_rc=true 면 WARN 도 rc=1 을 낸다 (이전 동작).
    warn_rc = bool(rc.get('warn_affects_rc', False))
    # 짝 폴백을 ERROR 로 볼지 (route_check.pair_fallback_is_error, 기본 false).
    # 대회 당일은 rc=1 로 시나리오가 통째로 버려지는 것보다 '큰 WARN' 이 낫다.
    fb_is_err = pair_fallback_cfg()[1]
    errs = 0
    lanes, cum = rt['lanes'], rt['cum_s']
    jsegs = set(rt.get('junction_segments') or [])
    # 짝 경유점 = 진입(wi) + 진출(wi+1). 이 점들은 "찍힌 차로 = 주행 차로" 가
    # 아니므로(주최측 2026-09-03) 반폭 이탈로 판정하지 않는다 — [2] 에서
    # "그 회전을 수행 가능한가" 로 본다.
    pair_wps = {w for wi in jsegs for w in (wi, wi + 1)}
    warns = 0

    print(f"\n{'=' * 72}")
    print(f"경로 검증 리포트")
    print('=' * 72)

    # ── 1) seq 점이 경로에서 얼마나 떨어져 있나 ──────────────────────────
    # **반폭 기준은 폐기했다.** 그건 "경유점이 찍힌 차로 = 주행 차로" 를 전제로
    # "반폭을 넘으면 옆 차로로 잘못 잡힌 것" 이라 보는 판정인데, 주최측 답변
    # (2026-09-03 "좌표는 대략적", "좌회전 구간에 3차로 경유지가 올 수 있다")
    # 으로 그 전제가 무효가 됐다. 짝 공동 선택(route.waypoint_lane_is_hint)이
    # 회전 가능한 차로를 고르면 경유점에서 한 차로(~3 m) 멀어지는 게 정상이고,
    # 반폭(~1.5 m)으로 재면 정상 경로가 경고를 받는다 (실측 2026-09-04:
    # 정적회피집중_01 seq 2 이탈 2.86 / 한계 1.54 → rc=1 → batch 가 시나리오 폐기).
    #   · 짝 경유점(진입·진출) → 이 판정에서 **제외**. [2] 의 회전 가능 판정이 대신한다.
    #   · 그 밖(시작·종료·직선 구간) → route.check_waypoint_max_dist_m (도로 폭 급).
    # --warn-dev 는 여전히 모든 경유점에 대한 고정 임계 override 다.
    head = (f'허용 {warn_dev:g} m 고정 (--warn-dev)' if warn_dev is not None
            else (f'허용 {max_dist:g} m (route.check_waypoint_max_dist_m); '
                  f'짝 경유점은 [2] 에서 판정' if pair_exempt
                  else f'허용 = 매칭 차로 반폭 × {hw_k:g}'))
    print(f"\n[1] 경유점 이탈 ({head};  매칭 반경 --radius {radius:g} m 는 별개)")
    for wi, (x, y) in enumerate(rt['waypoints']):
        sq = rt['waypoint_seq'][wi]
        best_d, best_s, best_sp, best_lane = None, None, None, None
        for i, k in enumerate(lanes):
            s_p, _t, d_p, _ = lg.project(k, x, y)
            if best_d is None or d_p < best_d:
                best_d, best_s, best_sp, best_lane = d_p, cum[i] + s_p, s_p, k
        half = 0.5 * lg.width_at(best_lane, best_sp)
        is_pair = pair_exempt and wi in pair_wps
        if warn_dev is not None:
            lim = warn_dev
        elif not pair_exempt:
            lim = hw_k * half
        else:
            lim = max_dist
        flag = ''
        if is_pair and warn_dev is None:
            flag = '   (짝 경유점 — [2] 회전 가능 판정)'
        elif best_d > lim:
            flag = (f'   <= [경고] 허용 {lim:.2f} m 초과 — 경로가 이 경유점에서 '
                    f'너무 멀다 (차로 반폭 {half:.2f} m)')
            warns += 1                     # WARN — 주행은 가능하다
        print(f"  seq {sq:>3}  ({x:9.2f},{y:9.2f})  이탈 {best_d:6.2f} / 한계 "
              f"{'—' if is_pair and warn_dev is None else f'{lim:4.2f}'} m  "
              f"경로 s={best_s:8.1f} m  lane={best_lane}{flag}")

    # ── 1b) 전역 DP (작업 R) ─────────────────────────────────────────────
    dpi = rt.get('dp')
    if dpi:
        cc = dpi.get('cand_counts') or []
        fb = dpi.get('fallback_to_pair')
        print(f"\n[DP] 형식 무관 전역 탐색 "
              f"{'— 짝 경로로 복귀' if fb else '채택'} (route.global_dp_enable)")
        print(f"  경유점 {dpi['n_points']}개 중 {dpi['n_used_points']}개 사용"
              f"{'' if not dpi['skipped_points'] else '  건너뜀 ' + str([i + 1 for i in dpi['skipped_points']])}"
              f"   후보 {min(cc) if cc else 0}~{max(cc) if cc else 0}개"
              f"   전이 {dpi['edges']}회   비용 {dpi['cost']:.0f} m"
              f"   경유점거리합 {dpi['wp_dist_sum']:.2f} m")
        for n in (dpi.get('notes') or []):
            print(f'  · {n}')
            warns += 1
        for n in (dpi.get('relaxed') or []):
            print(f'  · {n}   <= [경고] 제약 해제')
            warns += 1
        for n in (dpi.get('retries') or []):
            print(f'  · {n}   <= [경고] 반경 재시도')
            warns += 1
        if fb:
            print(f'  · {fb}   <= [경고] 연속성 안전망')
            warns += 1
        cm = dpi.get('pair_cmp')
        if fb:
            pass                     # 최종 경로가 짝 경로다 — 아래 대조 문구는 오해를 부른다
        elif cm is None:
            print('  짝 대조: 안 함 (짝 해석 없음)')
        elif not cm.get('ok'):
            print(f"  짝 대조: 짝 해석은 경로를 못 짓는다 — {cm.get('error')}")
        elif cm.get('same'):
            print('  짝 대조: 같은 경로')
        else:
            print(f"  짝 대조: **다른 경로** — 짝 {cm['pair_total']:.0f} m/"
                  f"{cm['pair_lanes']}차로  vs  DP {rt['total_length']:.0f} m/"
                  f"{len(lanes)}차로   <= [경고] 두 해석이 갈린다 (DP 를 쓴다)")
            warns += 1

    # ── 2) 짝(교차로) 구간 ───────────────────────────────────────────────
    spans = {wi: (i0, i1) for wi, i0, i1 in rt.get('segment_span') or []}
    # ── 짝 해석 머리 (작업21) — 대회 당일 사람이 눈으로 확인하는 한 줄이다.
    p_off = rt.get('pair_offset')
    p_src = rt.get('pair_offset_source') or 'caller'
    p_why = rt.get('pair_offset_why') or ''
    r_pair, n_pair, r_other, n_other = pair_junction_ratio(lg, rt)
    head = f"짝 해석: {'offset ' + str(p_off) if p_off is not None else '없음'} ({p_src})"
    print(f"\n[2] 교차로 짝 구간   {head}")
    if p_why:
        print(f"    근거: {p_why}")
    print(f"    junction 경유: 짝 {r_pair:.2f} ({n_pair}개) / 비짝 {r_other:.2f} ({n_other}개)"
          + ('   <= [경고] 짝·비짝이 둘 다 높다 — 한 칸 밀렸을 수 있다'
             if (n_pair and n_other and r_pair >= 0.75 and r_other >= 0.999) else ''))
    if not jsegs:
        # ERROR 는 아니지만 **가장 위험한 상태**다. 당일에 이 줄을 놓치면 안 된다.
        print('  ' + '!' * 68)
        print('  !! 짝 해석 없음 — 이 경로는 교차로 내부 차선변경 금지가 걸리지 않았다.')
        print('  !! 교차로 안에서 차선을 바꾸는 경로가 나올 수 있다 (채점 항목 6 위험).')
        print('  !! --pair-offset 0 또는 1 로 강제해 재빌드하고 [2] 표를 대조할 것.')
        print('  ' + '!' * 68)
        print('  (짝 정보 없음 — --pair-offset none 이거나 auto 판정 불가)')
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
            # WARN — 연결로 사이 링크 도로·같은 도로 다음 섹션으로 이어지는
            # 정당한 짝이 실재한다 (2026-09-04 실측 43개 중 6개). 오탐률이 높다.
            flag = '   <= [경고] junction 차로를 안 거친다'
            warns += 1
        if lc:
            # ERROR — 짝 구간은 allow_lane_change=False 로 만든다. 그런데 LC
            # 이벤트가 있으면 경로 생성과 이벤트 생성이 모순이고, 실주행에선
            # 실선 차로변경(채점 항목 6) 위험이다.
            flag += f'   <= [오류] 교차로 내부에서 차선변경 {lc}회'
            errs += 1
        # 짝 경유점의 진짜 판정 기준 — 고른 진입 차로에서 진출 차로로 차선변경
        # 없이 갈 수 있는가. 경유점과의 거리가 아니라 이게 성립해야 정상이다.
        # 짝 탐색이 0개라 기존 탐욕으로 폴백한 구간인가 (제안①·③).
        fb = next((f for f in rt.get('pair_fallbacks') or [] if f.get('wi') == wi), None)
        # 둘 다 off 면 이전과 같이 아무 줄도 안 낸다 (킬 스위치 규칙).
        if fb is not None and (fb_is_err or pair_fallback_cfg()[0]):
            note = (f"   <= [{'오류' if fb_is_err else '경고'}] 회전 가능한 (진입,진출) 짝이 "
                    f"없어 탐욕 폴백 — 진입 도로 {fb['roads_in']} 진출 도로 "
                    f"{fb['roads_out']} {fb['kind']}, 폴백 연결로 "
                    f"{_fmt_fb_connectors(fb)}, 구간 {fb.get('segment_m', '?')} m")
            flag += note
            if fb_is_err:
                errs += 1
            else:
                warns += 1
        ok, k_in, k_out, cost = pair_turn_ok(lg, rt, wi, banned, turn_cap)
        turn_txt = ''
        if ok is None:
            turn_txt = '  회전 —'
        elif ok:
            turn_txt = f'  회전 OK ({k_in}→{k_out}, {cost:.0f} m)'
        else:
            turn_txt = f'  회전 X ({k_in}→{k_out})'
            # ERROR — 짝 사이는 차선변경 금지다. 이게 안 되면 물리적으로
            # 주행 불가능한 경로다.
            # 단 DP 로 지은 경로에서는 WARN 이다: 짝 판정은 경유점이 (진입,진출)
            # 짝이라는 전제 위에 있고, DP 는 그 전제를 안 쓴다. 형식이 짝이
            # 아니면 이 판정이 틀린 쪽이다 (작업 R).
            if rt.get('dp'):
                flag += ('   <= [경고] 짝 해석으로는 진입→진출이 차선변경 없이 '
                         '안 이어진다 (DP 경로 — 짝 형식이 아닐 수 있다)')
                warns += 1
            else:
                flag += (f'   <= [오류] 진입 차로에서 진출 차로로 차선변경 없이 갈 수 없다'
                         f' (상한 {turn_cap:g} m)')
                errs += 1
        print(f"  seq {sq_in:>3}→{sq_out:<3}  junction={jids if jids else '없음'}  "
              f"Δheading={dh:+7.1f}°  {turn_kind(dh)}  차로 {len(seg_lanes)}개"
              f"{turn_txt}{flag}")

    # ── 2b) 소멸(테이퍼) 차로 통과 — WARN ─────────────────────────────────
    # 끝 폭이 차폭 미만으로 사라지는 차로. 제어기(route.py taper_blend 15 m)가
    # successor 중심선으로 블렌드해 주행은 되지만(2026-09-06 실측: 끝점에서
    # successor 선 기준 ±0.06 m), 인계 첫 틱의 차로 매칭 t_off 가 −1.3 m 로 튀어
    # 채점 차로유지에 잡힌다 (같은 날 4건 중 3건). 대안 연결로가 합법 도달 가능한지는
    # 여기서 판정하지 않는다 — 포켓 진입 창이 MIN_LC_WINDOW_M 미만인 곳이 있어
    # (146,0,2: 점선 13.2 m) 불가피한 경우가 실재한다. 그래서 ERROR 가 아니다.
    _tp_on, _tp_m, _veh_w = taper_cfg()
    tapers = [(i, k) for i, k in enumerate(lanes) if is_taper_lane(lg, k, _veh_w)]
    if tapers:
        print(f"\n[2b] 소멸 차로 통과 (끝 폭 < 차폭 {_veh_w:.3f} m; 탐색 벌점 "
              f"{'on %.0f m' % _tp_m if _tp_on else 'off'})")
        for i, k in tapers:
            L = lg.length(k)
            kind = '연결로' if lg.lanes[k].get('junction', -1) != -1 else '도로 안'
            nxt = lanes[i + 1] if i + 1 < len(lanes) else None
            print(f"  route_s {cum[i]:7.1f}  {k}  {kind}  len {L:5.1f}  폭 "
                  f"{lg.width_at(k, 0.0):.2f}→{lg.width_at(k, L):.2f}  → {nxt}"
                  f"   <= [경고] 소멸 차로 통과 (인계 지점 차로유지 판정 주의)")
            warns += 1

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

    # 경유점을 직선으로 이은 길이 대비 실제 경로 길이. 옆 차로로 잘못 잡히면
    # 되돌아오는 우회가 붙어 이 비율이 튄다 (실측: 정상 1.06~1.18 / 오선택 1.37).
    wps = rt['waypoints']
    straight = sum(math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
                   for i in range(len(wps) - 1))
    ratio = rt['total_length'] / straight if straight > 1e-9 else float('inf')
    r_max = float(rc['length_ratio_max'])
    r_flag = ''
    if ratio > r_max:
        # WARN — 경유점이 성기면 정상 경로도 넘는다 (params 주석 참조).
        r_flag = (f'   <= [경고] 임계 {r_max:g} 초과 — 옆 차로·먼 길로 잡혔을 수 있다')
        warns += 1

    print(f"\n[3] 총계")
    print(f"  총 길이        {rt['total_length']:8.1f} m   차로 {len(lanes)}개")
    print(f"  경유점 직선연결 {straight:8.1f} m   실제/직선 {ratio:.3f}{r_flag}")
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
            w = e['window_s1'] - e['window_s0']
            # 회랑은 **떠나는 차로**(from_lane) 기준 — e['lane'] 은 창이 시작되는
            # 차로라 뒤로 당겨진 결과이고, 탐색이 본 값과 축이 다르다.
            fl = e.get('from_lane')
            corr = (lg.dashed_corridor_m(fl, e['kind'].split('_')[-1])
                    if fl in lg.lanes else None)
            extra = (f"  window {e['window_s0']:.1f}-{e['window_s1']:.1f} m  ({w:.1f} m)"
                     + (f"  회랑 {corr:.1f} m" if corr is not None else ''))
            if w < MIN_LC_WINDOW_M:
                # ERROR — 근거는 **횡이동 전이거리** 하나다. 2026-08-21 실사고:
                # 창 6.1 m LC 가 실패해 헤딩오차 46°, 조향 풀락 포화, 도로이탈 +
                # courseRespawn. MIN_LC_WINDOW_M 은 transition_min_m 20 m 에서 왔다.
                # ※ 여기 있던 "지시등 선행 3 s 도 못 낸다" 는 **인과가 아니다**
                #   (2026-09-05 반증, BACKLOG B-16). 지시등은 창 안이 아니라 창
                #   **앞**의 도로에서 켜지므로(kr_rules._lane_shift 의 look 은
                #   현재 위치 기준 전방 탐색이고 램프는 창 맨 앞에 놓인다) 창
                #   길이와 lead 는 무관하다. 실측: on_s 3 s 미달 6건의 창 중앙값
                #   51.2 m > 통과 38건의 47.1 m 이고, 통과 최소 창은 20.8 m 였다.
                extra += (f'   <= [오류] 창이 {MIN_LC_WINDOW_M:.0f} m 미만 — '
                          f'전이거리(max(transition_s*v, transition_min_m))를 못 채운다')
                errs += 1
        if 'delta_heading_deg' in e:
            extra = (f"  Δ{e['delta_heading_deg']:+.1f}°  junction={e.get('junction')}"
                     + ('  [짝 기준 보정]' if e.get('source') == 'pair' else ''))
        print(f"  {e['s']:8.1f} m  {e['kind']:<20}{extra}")

    # ── 차선변경 여유 적합 (작업19-2 도입 / 19-3 축 교체) ─────────────────
    # 위 창 검사와 축이 다르다. 창은 "이 전이를 어디서 시작할 수 있나" 이고,
    # 여기는 "앞 전이가 끝나기 전에 다음이 시작되지 않나" 다.
    sep_m, sep_speed = hop_sep_cfg()
    # 속도 의존 스위치가 켜졌을 때만 게이트 축을 필요 간격으로 바꾼다.
    # 끄면 이전과 같은 축(min_hop_gap_m)이다 — 물리적으로 불가능한 구간
    # (도로 418 의 3연속 차선변경)에 새 오류를 만들지 않기 위해서다.
    rooms = hop_room(lg, rt, (lambda k: hop_sep_for(lg, k)) if sep_speed else None)
    if rooms:
        gap_on = bool(rc.get('hop_gap_enable', True))
        thr_txt = ('구간 제한속도별 (램프+지시등)' if sep_speed
                   else f'{min_hop_gap_m():g} m')
        print(f"  ── 차선변경 여유 적합 (전이 하나에 {thr_txt}"
              f"{'' if gap_on else ', 검사 꺼짐'})")
        for _i, cum_i, fl, tl, need, room, nth in rooms:
            note = ''
            if nth > 0 and need > room + 1e-9:
                # ERROR — 한 차로 안에서 전이를 여러 번 끝내야 한다. planner
                # 램프는 cum_s 당 hop 하나만 블렌드하므로 나머지는 경로 점열에
                # 차로 폭짜리 계단으로 남는다 (venue_20260903 실측:
                # 29.8 m 에 hop 3개 -> 3.402 m / 6.800 m 계단).
                note = (f'   <= [오류] 누적 {need:.1f} m > 차로 {room:.1f} m — '
                        f'앞 전이가 끝나기 전에 다음이 시작된다')
                if gap_on:
                    errs += 1
                else:
                    note += ' (검사 꺼짐 — rc 무관)'
            mark = '연쇄' if nth else '단발'
            print(f"    {cum_i:8.1f} m  {str(fl)} -> {str(tl)}   "
                  f"{mark} 누적 {need:5.1f} / 차로 {room:6.1f} m{note}")

    # ── [5] 회전 가능성 — 회전 이벤트가 지나는 연결로들의 최소 곡률반경 ──────
    # 금지는 route.banned_r_min_m (기본 3.0) 미만 = 지도 결함 안전망뿐이다.
    # 그 위 ~ 기하 최소회전반경 사이는 **급회전**: 통행하되 WARN 이고,
    # speed.curvature_cap 이 √(a_lat_max·R) 로 진입속도를 눌러 준다.
    # (9_school_route 실측 R 2.55 m 연결로 → 호 이탈 → off_route 정지는
    #  3.0 임계에도 계속 걸린다.)
    r_need, margin = min_turn_radius_m()
    thr = rt.get('turn_radius_thr_m', banned_r_min_m())
    tight_thr = rt.get('tight_turn_thr_m', r_need * margin)
    a_lat = curvature_a_lat_max_m_s2()
    # 임계는 **그 경로를 지을 때 쓴 값**(pkl 기록)이다 — 지금 params 와 다를 수
    # 있으므로(옛 pkl 재검증) 같을 때만 키 이름을 붙인다.
    src = ' = route.banned_r_min_m' if abs(thr - banned_r_min_m()) < 1e-9 else ' (이 pkl 을 지을 때 값)'
    print(f"\n[5] 회전 가능성  (금지 임계 R < {thr:.2f} m{src} — 지도 결함 안전망;"
          f"\n     {thr:.2f}~{tight_thr:.2f} m 는 '급회전' — 통행 허용, 커브 감속 진입)")
    lanes_list = rt['lanes']
    cum = rt['cum_s']
    for e in ev:
        if not e['kind'].startswith('turn'):
            continue
        junc = e.get('junction')
        conns = [(i, k) for i, k in enumerate(lanes_list)
                 if lg.lanes[k]['junction'] == junc and junc is not None
                 and abs(cum[i] - e['s']) < 60.0]
        for i, k in conns:
            r = lane_r_min(lg, k)
            bad = r < thr
            tight = (not bad) and r < tight_thr
            if bad:
                note = '   <= ⚠ 회전 불가 기하'
            elif tight:
                note = (f"   <= [경고] 급회전 연결로 R {r:.2f} — 감속 진입"
                        f" (v ≤ {math.sqrt(a_lat * r):.2f} m/s)")
            else:
                note = ''
            print(f"  {e['s']:8.1f} m  {e['kind']:<11} 연결로 {str(k):<16} "
                  f"R_min {r:8.2f} m{note}")
            if bad:
                errs += 1              # ERROR — 풀락으로도 못 도는 기하
    # 급회전 집계는 rt['tight_turns'] 를 정본으로 센다 — 위 이벤트 루프는 한
    # 연결로를 여러 이벤트에서 다시 찍을 수 있고, 회전 이벤트가 안 붙은
    # 직진 통과 연결로는 아예 안 찍힌다.
    for tt in rt.get('tight_turns') or []:
        r = float(tt['r_min_m'])
        print(f"  [경고] {tt['s_m']:8.1f} m  급회전 연결로 {tuple(tt['lane'])} "
              f"R {r:.2f} m — 감속 진입 (v ≤ {math.sqrt(a_lat * r):.2f} m/s)")
        warns += 1                     # WARN — 통행 가능, 커브 감속이 받쳐 준다
    forced = rt.get('infeasible_forced', [])
    for wi, k, r in forced:
        print(f"  ⚠ 구간 {wi}: 대안 경로가 없어 회전 불가 연결로 {k} (R_min {r:.2f} m) 를 "
              f"**불가피하게 포함** — 실주행에서 이탈 가능성 높음")
        errs += 1                      # ERROR — #7 과 같은 사유, 대안까지 없다

    rc_n = (errs + warns) if warn_rc else errs
    print(f"\n{'=' * 72}")
    if errs == 0 and warns == 0:
        print('경고 없음')
    else:
        parts = []
        if errs:
            parts.append(f'[오류] {errs}건')
        if warns:
            parts.append(f'[경고] {warns}건' + ('' if warn_rc else ' (정보성 — rc 무관)'))
        print('  '.join(parts) + ' — 위 표시된 항목을 확인할 것')
    print('=' * 72)
    # 반환값 = **rc 를 유발하는 건수**. main() 의 rc 와 gen_scenarios.route_check
    # 의 합격 기준이 같은 값을 봐야 생성 시점과 실행 시점이 어긋나지 않는다.
    return rc_n


def main():
    ap = argparse.ArgumentParser(description='대회 공식 경유점 CSV → route.pkl')
    ap.add_argument('pkl')
    ap.add_argument('waypoints', help='csv: seq,x,y (헤더 있어도 됨)')
    ap.add_argument('-o', '--out', default='route.pkl')
    ap.add_argument('--radius', type=float, default=None,
                    help='[m] 경유점 매칭 반경 (기본 8. 주면 DP 후보 반경'
                         ' route.dp_match_radius_m 보다 우선한다)')
    ap.add_argument('--start-yaw', type=float, default=None,
                    help='[rad] 출발 헤딩. 없으면 seq1→seq2 방향으로 자동 추정')
    ap.add_argument('--ego-yaw', type=float, default=None,
                    help='[rad] 9910 에서 받은 실제 ego heading (--start-yaw 보다 우선)')
    ap.add_argument('--no-pairs', action='store_true',
                    help='중간 지점을 교차로 진입·진출 짝으로 해석하지 않는다 '
                         '(= --pair-offset none)')
    ap.add_argument('--pair-offset', choices=('auto', 'none', '0', '1'), default='auto',
                    help='짝 해석 시작점. auto=짝수면 1, 홀수면 시험 빌드로 판정 '
                         '(안 서면 none). 0=첫 점부터 짝, 1=두 번째 점부터 짝 '
                         '(주최 공식 형식), none=짝 해석 안 함')
    ap.add_argument('--warn-dev', type=float, default=None,
                    help='[m] 경유점 이탈 경고 임계 (기본: --radius)')
    ap.add_argument('--yaw-min-dist', type=float, default=2.0,
                    help='[m] 헤딩 자동추정에 쓸 최소 거리. 이보다 가까운 경유점은 건너뛴다')
    a = ap.parse_args()
    # 매칭 반경 기본 8 m — 안 주면 그 값을 쓰고, DP 후보 반경은 params 키가 정한다.
    radius = 8.0 if a.radius is None else float(a.radius)

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

    # ── 교차로 짝 (작업21) ────────────────────────────────────────────────
    # 주최 형식은 [시작, (진입,진출)*N, 종료] = 짝수지만 홀수로 올 수도 있다.
    # 홀수면 해석이 둘이라 auto 가 시험 빌드로 고른다. 어느 해석을 썼는지는
    # 리포트 [2] 머리와 pkl 에 남는다 — 당일 눈으로 확인할 근거다.
    tail_m = finish_tail_cfg()
    mode = 'none' if a.no_pairs else a.pair_offset
    if mode == 'none':
        offset, why, src = None, '--pair-offset none (짝 해석 안 함)', 'forced'
    elif mode == 'auto':
        offset, why, _ev = pair_offset_auto(lg, wps, radius, start_yaw, seqs, tail_m,
                                           dp_radius=a.radius)
        src = 'auto'
    else:
        offset, why, src = int(mode), f'--pair-offset {mode} (사람이 지정)', 'forced'
    print(f'  짝 해석: {"offset " + str(offset) if offset is not None else "없음"} ({src})')
    print(f'    근거: {why}')

    jsegs = junction_segments(len(wps), offset) if offset is not None else set()
    if jsegs:
        pairs = [(seqs[wi], seqs[wi + 1]) for wi in sorted(jsegs)]
        print(f'  교차로 짝 {len(pairs)}개: ' +
              ', '.join(f'({i}→{o})' for i, o in pairs))
    elif offset is not None:
        print('  경유점이 시작/종료뿐 — 교차로 짝 없음')

    rt = build_route(lg, wps, radius, start_yaw, junction_segs=jsegs, seqs=seqs,
                     dp_radius=a.radius,
                     finish_tail_m=tail_m,
                     pair_meta={'offset': offset, 'source': src, 'why': why})

    with open(a.out, 'wb') as f:
        pickle.dump(rt, f, protocol=4)

    warns = report(lg, rt, radius, a.warn_dev)
    print(f'saved {a.out}')
    return 1 if warns else 0


if __name__ == '__main__':
    raise SystemExit(main())
