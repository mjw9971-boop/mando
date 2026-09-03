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
            s2 = min(s_enter, L2)
            # 이 차로를 s_enter 에서 떠나는 비용으로 되돌리고 + 차선변경 비용(회랑 반영)
            c_lc = cost - (r['length'] - s_enter) + lc_cost(lg, key, side)
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
    rc = route_check_cfg()
    hw_k = float(rc['wp_dev_halfwidth_k'])
    lanes, cum = rt['lanes'], rt['cum_s']
    warns = 0

    print(f"\n{'=' * 72}")
    print(f"경로 검증 리포트")
    print('=' * 72)

    # ── 1) seq 점이 경로에서 얼마나 떨어져 있나 ──────────────────────────
    # 허용치는 **매칭된 차로의 반폭** 이다 — 이걸 넘으면 옆 차로가 더 가까워져
    # 조용히 옆 차로로 경로가 잡힐 수 있다. --radius(경유점 매칭 반경)는 이 판정과
    # 무관한 별개 값이라 쓰지 않는다 (--warn-dev 로 고정 임계를 줄 수는 있다).
    head = (f'허용 {warn_dev:g} m 고정 (--warn-dev)' if warn_dev is not None
            else f'허용 = 매칭 차로 반폭 × {hw_k:g}')
    print(f"\n[1] 경유점 이탈 ({head};  매칭 반경 --radius {radius:g} m 는 별개)")
    for wi, (x, y) in enumerate(rt['waypoints']):
        sq = rt['waypoint_seq'][wi]
        best_d, best_s, best_sp, best_lane = None, None, None, None
        for i, k in enumerate(lanes):
            s_p, _t, d_p, _ = lg.project(k, x, y)
            if best_d is None or d_p < best_d:
                best_d, best_s, best_sp, best_lane = d_p, cum[i] + s_p, s_p, k
        half = 0.5 * lg.width_at(best_lane, best_sp)
        lim = warn_dev if warn_dev is not None else hw_k * half
        flag = ''
        if best_d > lim:
            flag = (f'   <= [경고] 차로 반폭 {half:.2f} m 를 벗어난다 — '
                    f'옆 차로로 잡힐 수 있다')
            warns += 1
        print(f"  seq {sq:>3}  ({x:9.2f},{y:9.2f})  이탈 {best_d:6.2f} / 한계 {lim:4.2f} m  "
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

    # 경유점을 직선으로 이은 길이 대비 실제 경로 길이. 옆 차로로 잘못 잡히면
    # 되돌아오는 우회가 붙어 이 비율이 튄다 (실측: 정상 1.06~1.18 / 오선택 1.37).
    wps = rt['waypoints']
    straight = sum(math.hypot(wps[i + 1][0] - wps[i][0], wps[i + 1][1] - wps[i][1])
                   for i in range(len(wps) - 1))
    ratio = rt['total_length'] / straight if straight > 1e-9 else float('inf')
    r_max = float(rc['length_ratio_max'])
    r_flag = ''
    if ratio > r_max:
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
                extra += (f'   <= [경고] 창이 {MIN_LC_WINDOW_M:.0f} m 미만 — '
                          f'전이거리(max(transition_s*v, transition_min_m))를 못 채운다')
                warns += 1
        if 'delta_heading_deg' in e:
            extra = (f"  Δ{e['delta_heading_deg']:+.1f}°  junction={e.get('junction')}"
                     + ('  [짝 기준 보정]' if e.get('source') == 'pair' else ''))
        print(f"  {e['s']:8.1f} m  {e['kind']:<20}{extra}")

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
                warns += 1
    forced = rt.get('infeasible_forced', [])
    for wi, k, r in forced:
        print(f"  ⚠ 구간 {wi}: 대안 경로가 없어 회전 불가 연결로 {k} (R_min {r:.2f} m) 를 "
              f"**불가피하게 포함** — 실주행에서 이탈 가능성 높음")
        warns += 1

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

    rt = build_route(lg, wps, a.radius, start_yaw, junction_segs=jsegs, seqs=seqs,
                     finish_tail_m=finish_tail_cfg())

    with open(a.out, 'wb') as f:
        pickle.dump(rt, f, protocol=4)

    warns = report(lg, rt, a.radius, a.warn_dev)
    print(f'saved {a.out}')
    return 1 if warns else 0


if __name__ == '__main__':
    raise SystemExit(main())
