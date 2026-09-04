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
import argparse, heapq, math, pickle, sys
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


def road_lane_pool(lg, cands, x, y, yaw):
    """후보집합에 등장한 **모든 도로**의 같은 섹션·같은 통행방향 driving 차로.

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
            if kk in seen_key or kk[1] != k0[1]:
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
        need = cur + gap
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


def infeasible_connectors(lg):
    """
    물리적으로 돌 수 없는 교차로 연결로 집합.

    junction 연결로의 R_min 이 (최소회전반경 × vehicle.min_turn_margin) 미만이면
    풀락으로도 호를 못 따라간다 — 9_school_route 실측(2026-08-24): junction 25 의
    (1576,0,-1) R_min 2.55 m 를 최단이라고 골랐다가 조향 포화 1.8 s 끝에 호를
    이탈해 off_route 정지. 같은 교차로에 R 56.6 m 대안(1573)이 있었다.
    곡률 스파이크는 빌드 단계에서 이미 걸렀으므로(중앙값 필터) 남은 값은 진짜
    기하다 — 그대로 평가한다.
    """
    r_min, margin = min_turn_radius_m()
    thr = r_min * margin
    out = {}
    for key, rec in lg.lanes.items():
        if rec['junction'] == -1:
            continue
        r = lane_r_min(lg, key)
        if r < thr:
            out[key] = r
    return out, thr


def dijkstra(lg, starts, targets, allow_lane_change=True, banned=frozenset()):
    """starts: [(lane, s_start)]  targets: {lane: s_target} → (cost, [ (lane, s_enter) ... ])

    allow_lane_change=False 면 successor 링크만 따라간다 (교차로 내부 구간용).
    banned: 통행 금지 차로 (회전 불가 연결로 — 비용 무한 대신 아예 확장하지 않는다)."""
    tgt = dict(targets)
    best = {}
    heap = []
    # 전이 하나가 먹는 진행거리 [m]. 0 이면 이전 동작(간격을 비용에서 무시).
    hop_gap = min_hop_gap_m() if hop_spacing_cost_enable() else 0.0
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


def junction_segments(n_points):
    """
    교차로 내부 구간(0-based 세그먼트 인덱스) 집합.

    seq 1=시작, (2,3)(4,5)... 짝, 마지막=종료 이므로 0-based 로는
    waypoints[1]→[2], [3]→[4], ... 즉 **홀수 인덱스 세그먼트**가 교차로 내부다.
    """
    return {wi for wi in range(1, n_points - 1, 2)}


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


def _pair_choice(lg, starts, wps, wi, radius, junction_segs, banned, cap, label, seqs):
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

    def search(bans):
        best = None
        for ka, sa, da in pool_in:
            pre = dijkstra(lg, starts, {ka: sa}, allow_lane_change=allow_prev, banned=bans)
            if pre is None:
                continue
            for kb, sb, db in pool_out:
                c = turn_connect(lg, ka, sa, kb, sb, bans, cap)
                if c is None:
                    continue
                score = pre[0] + c[0] + TARGET_DIST_W * da + TARGET_DIST_W * db
                if best is None or score < best[0]:
                    best = (score, pre[1], ka, sa, c[1], kb, sb)
        return best

    best, used_banned = search(banned), []
    if best is None:
        # 금지 연결로를 풀면 되는가 — 기존 탐욕과 같은 취급(불가피하면 허용 + 기록)
        best = search(frozenset())
        if best is None:
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
            return None
        used_banned = [kk for kk, _ in (best[1] + best[4]) if kk in banned]
    _sc, path_in, k_in, s_in, path_out, k_out, s_out = best
    return path_in, k_in, s_in, path_out, k_out, s_out, used_banned


def build_route(lg, waypoints, radius=8.0, start_yaw=None, junction_segs=frozenset(),
                seqs=None, finish_tail_m=0.0):
    # 회전 불가 연결로 (R_min < 최소회전반경 × 여유) — dijkstra 에서 통행 금지
    banned, turn_thr = infeasible_connectors(lg)
    forced_infeasible: list = []   # 대안이 없어 불가피하게 포함시킨 연결로
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
    skip_next = False
    for wi in range(len(waypoints) - 1):
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
                               banned, turn_cap, label, seqs)
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

    return {'lanes': lanes, 'cum_s': cum, 'lengths': lengths, 'total_length': total, 'start_s_in_lane': s_first,
            'infeasible_forced': forced_infeasible, 'turn_radius_thr_m': turn_thr,
            'finish_xy': [float(waypoints[-1][0]), float(waypoints[-1][1])],
            'waypoints': [tuple(w) for w in waypoints], 'waypoint_s': wp_dist, 'events': events,
            'waypoint_seq': list(seqs) if seqs else list(range(1, len(waypoints) + 1)),
            'junction_segments': sorted(junction_segs), 'segment_span': seg_span}


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

    # ── 2) 짝(교차로) 구간 ───────────────────────────────────────────────
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
            flag += (f'   <= [오류] 진입 차로에서 진출 차로로 차선변경 없이 갈 수 없다'
                     f' (상한 {turn_cap:g} m)')
            errs += 1
        print(f"  seq {sq_in:>3}→{sq_out:<3}  junction={jids if jids else '없음'}  "
              f"Δheading={dh:+7.1f}°  {turn_kind(dh)}  차로 {len(seg_lanes)}개"
              f"{turn_txt}{flag}")

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
                # ERROR — 2026-08-21 실사고: 창 6.1 m LC 가 실패해 헤딩오차 46°,
                # 조향 풀락 포화, 도로이탈 + courseRespawn. 지시등 선행 3 s 도 못 낸다.
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
    rooms = hop_room(lg, rt)
    if rooms:
        thr = min_hop_gap_m()
        gap_on = bool(rc.get('hop_gap_enable', True))
        print(f"  ── 차선변경 여유 적합 (전이 하나에 {thr:g} m"
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
    # R_min < 최소회전반경 × vehicle.min_turn_margin 이면 풀락으로도 못 돈다
    # (9_school_route 실측: R 2.55 m 연결로 선택 → 호 이탈 → off_route 정지).
    r_need, margin = min_turn_radius_m()
    thr = rt.get('turn_radius_thr_m', r_need * margin)
    print(f"\n[5] 회전 가능성  (금지 임계 R < {thr:.2f} m = 최소회전반경 {r_need:.2f} × {margin:g};"
          f"  {thr:.2f}~{r_need:.2f} m 는 '빠듯' — 포화·차로폭 여유로 통과)")
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
            tight = (not bad) and r < r_need
            note = '   <= ⚠ 회전 불가 기하' if bad else ('   (빠듯 — 조향 포화 예상)' if tight else '')
            print(f"  {e['s']:8.1f} m  {e['kind']:<11} 연결로 {str(k):<16} "
                  f"R_min {r:8.2f} m{note}")
            if bad:
                errs += 1              # ERROR — 풀락으로도 못 도는 기하
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

    rt = build_route(lg, wps, a.radius, start_yaw, junction_segs=jsegs, seqs=seqs,
                     finish_tail_m=finish_tail_cfg())

    with open(a.out, 'wb') as f:
        pickle.dump(rt, f, protocol=4)

    warns = report(lg, rt, a.radius, a.warn_dev)
    print(f'saved {a.out}')
    return 1 if warns else 0


if __name__ == '__main__':
    raise SystemExit(main())
