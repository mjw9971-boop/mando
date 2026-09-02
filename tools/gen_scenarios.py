"""
시나리오 자동 생성기 — 주제 이름만 주면 그날치 테스트 시나리오를 만든다.

    python3 tools/gen_scenarios.py 보행자집중 급정거집중 --count 3 --seed 1
    python3 tools/gen_scenarios.py 교차로집중 --hours 2
    python3 tools/gen_scenarios.py --list                      # 주제 목록
    python3 tools/gen_scenarios.py --from-yaml scenarios/보행자집중/보행자집중_01_기본.yaml

주제는 configs/themes.yaml 의 프리셋 이름이다. 여러 개를 나열하면 합쳐서
하나의 batch 목록으로 만든다.

  --hours N : 총 실행시간 예산. 시나리오당 예상시간(경로길이/27 km/h + 60 s)으로
              환산해 주제별로 라운드로빈 배분한다.
  --count N : 주제당 개수를 직접 지정.
  (없으면)  : 유효 조합 전수, 주제당 상한 30 (초과분은 --seed 로 샘플링).

산출물 (scenarios/<주제>/ 아래):
  <이름>.xml   VTD 시나리오 — templates/9_clean_drive.xml 에 검증된 블록
               (2_lead_brake / 3_static_vehicle / 4_rtor_red_ped /
                5_red60_first_junction / ped_crosswalk_static 에서 추출) 삽입
  <이름>.csv   대회형식 경유점 (시작 + 교차로 진입/진출 짝 + 종료)
  <이름>.yaml  확정 정의 — 이 파일 하나로 단건 재생성 가능 (--from-yaml)
  scenarios/batch_<주제>.json     주제별 batch_run.py 용 목록
  scenarios/batch_all.json        이번 실행 전체 통합 목록 (이름 중복은 통합 기준 검사)

좌표 조회는 전부 data/lane_graph.pkl (LaneGraph) 로 한다 — xodr 재파싱 금지.
신호 조작용 교차로별 접근 컨트롤러 매핑은 최초 1회 lane_graph(=xodr 파싱 결과)
에서 만들어 data/junction_ctrl_map.json 으로 캐시한다.

로직이 아직 대응 못 하는 이벤트(cut_in / oncoming / narrow)도 생성은 한다 —
실행해서 나온 실패 목록이 곧 미구현 우선순위 자료다. 요약 표에 표시된다.
"""
from __future__ import annotations

import argparse
import collections
import heapq
import itertools
import json
import math
import pathlib
import random
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

from build_route import (RouteError, build_route, junction_segments,   # noqa: E402
                         read_waypoints_csv, report as route_report)
from vtd_adapter.lanegraph import LaneGraph, wrap                       # noqa: E402

TEMPLATE = ROOT / 'templates' / '9_clean_drive.xml'
THEMES_YAML = ROOT / 'configs' / 'themes.yaml'
CTRL_MAP_JSON = ROOT / 'data' / 'junction_ctrl_map.json'

AVG_SPEED_MPS = 27.0 / 3.6      # 예상시간 환산용 평균속도
OVERHEAD_S = 60.0               # 시나리오당 로드/리셋 오버헤드
MAX_PER_THEME = 30              # 개수 미지정 시 주제당 상한
LAT_WARN_M = 3.0                # ego 차선 이벤트의 경로 횡거리 경고 임계

CAR = 'AlfaRomeo_Brera_10_BiancoSpino'
BUS = 'MercedesTravego_10_HoneyYellow'
CAR_LEN, CAR_W = 4.7, 1.9
BUS_LEN = 12.0

# 컨트롤러 로직이 아직 대응하지 않는(대응 확인 안 된) 이벤트 — 요약에 표시만 한다
UNSUPPORTED = {'cut_in': '컷인 대응 미구현', 'oncoming': '대향차 인지/추월 판단 미구현',
               'narrow': '협착 통과 코리더 미검증'}

AXIS_DEFAULTS = {
    '위치': [0.3, 0.5, 0.75],
    '보행속도': [1.0, 1.5, 2.0],
    '트리거거리': [15, 25, 35],
    '신호': ['기본', '적색'],
    '감속강도': [3.0, 5.0, 7.0],
    '발동거리': [0.35, 0.55, 0.75],
    '재출발': ['있음', '없음'],
    '차선': ['자차로', '좌측차로'],
    '대향차': ['없음', '있음'],
    '개수': [3, 4, 6],
    '속도': [30, 50],
    '간격': [30, 50, 80],
    '지점': ['횡단보도', '도로중간'],
    '차폐물': ['버스', '승용차'],
    '침범폭': [0.4, 0.7, 1.0],
    '경로변형': [1, 2],
    '방향': ['우측', '좌측'],
    '교통류대수': [12, 20, 30],
    '교통류밀도': ['조밀', '보통', '성김'],
}


class GenError(SystemExit):
    """생성 실패 — 어디서 왜 막혔는지 메시지에 담는다 (무음 실패 금지)."""


class EventUnfeasible(GenError):
    """이 지점에서 이 이벤트가 기하적으로 성립하지 않는다.

    resolve_events 가 위치를 옮겨 재시도하고, 그래도 안 되면 (min_keep 가 있는
    주제에서) **그 이벤트만** 버린다. 시나리오째 죽이지 않는다 — 성립 안 하는
    이벤트를 억지로 넣는 것보다 하나 적게 넣는 편이 낫다 (2026-08-30 결정).
    """


def fnum(v: float) -> str:
    return f'{float(v):.4f}'


# ════════════════════════════════════════════════════════════════════════
# 1. 경로
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Route:
    name: str          # 풀 이름 (+변형)
    rows: list         # [(seq, x, y)]
    rt: dict           # build_route 결과 (차로 체인 포함)
    start_yaw: float


def _start_yaw(rows) -> float:
    for j in range(1, len(rows)):
        if math.hypot(rows[j][1] - rows[0][1], rows[j][2] - rows[0][2]) >= 2.0:
            return math.atan2(rows[j][2] - rows[0][2], rows[j][1] - rows[0][1])
    return math.atan2(rows[1][2] - rows[0][2], rows[1][1] - rows[0][1])


def _build_from_rows(lg, name, rows) -> Route:
    wps = [(x, y) for _, x, y in rows]
    if len(wps) % 2 != 0:
        raise GenError(f'경로 {name}: 경유점 {len(wps)}개(홀수) — 대회형식(짝수)이 아니다')
    yaw = _start_yaw(rows)
    rt = build_route(lg, wps, 8.0, yaw, junction_segs=junction_segments(len(wps)),
                     seqs=[r[0] for r in rows])
    return Route(name, rows, rt, yaw)


# ── walk: lane_graph 를 걸어 대회형식 경로를 합성 ────────────────────────

def _turn_kind(dh_deg: float) -> str:
    if abs(dh_deg) > 135.0:
        return 'uturn'                  # 되돌아오는 연결로 — right/left 로 오분류 금지
    if dh_deg > 25.0:
        return 'left'
    if dh_deg < -25.0:
        return 'right'
    return 'straight'


def _hdg_end(lg, k):
    return float(np.unwrap(lg.lanes[k]['hdg'].astype(float))[-1])


def _hdg_start(lg, k):
    return float(np.unwrap(lg.lanes[k]['hdg'].astype(float))[0])


def _junction_options(lg, approach):
    """approach 차로에서 나가는 교차로 통과 옵션들.

    → [(kind, approach_lane, [교차로/짧은 링크 차로들…], exit_lane)]
    연결로가 짧은 링크 도로로 이어지는 클러스터(waypoints.csv 경로처럼 미니 교차로
    연쇄)는 길이 25 m 이상의 비교차로 차로가 나올 때까지 직진 우선으로 삼킨다.
    """
    opts = []
    for nk in lg.successors(approach):
        if lg.lanes[nk]['junction'] == -1:
            continue
        chain, cur, exit_lane = [nk], nk, None
        for _ in range(40):
            nxts = lg.successors(cur)
            if not nxts:
                break
            h0 = _hdg_end(lg, cur)
            nxt = min(nxts, key=lambda k2: abs(
                (_hdg_start(lg, k2) - h0 + math.pi) % (2 * math.pi) - math.pi))
            if lg.lanes[nxt]['junction'] == -1 and lg.length(nxt) >= 25.0:
                exit_lane = nxt                    # 진짜 출구
                break
            chain.append(nxt)
            cur = nxt
        if exit_lane is None:
            continue
        dh = math.degrees((_hdg_start(lg, exit_lane) - _hdg_end(lg, approach)
                           + math.pi) % (2 * math.pi) - math.pi)
        opts.append((_turn_kind(dh), approach, chain, exit_lane))
    return opts


# 연장(min_length_m) 중단 사유 계측 — 3 km 도달률 진단·보고용 (실행 단위 누적)
WALK_EXT_STATS: collections.Counter = collections.Counter()
WALK_EXT_LENS: dict = collections.defaultdict(list)     # 사유 → 중단 시점 route_len


# ── 목표 길이 탐색 (백트랙 DFS) — 주행 검증형 연장 구간 전용 ──────────────

def _corridor_ahead(lg, lane, fwd_cap: float):
    """lane 에서 교차로 옵션이 나올 때까지 비교차로 successor 로 강제 전진.

    → (지나온 차로들[lane 제외], 추가 길이, 끝 차로, 비 U턴 교차로 옵션들)
    옵션이 빈 목록이면 막다른 회랑이다. 이동 모델은 _walk_once 의 전진 루프와
    같다 — 이웃(같은 방향 driving) 접근도 옵션 수집에 포함한다."""
    passed, add = [], 0.0
    cur = lane
    for _ in range(60):
        approaches = [cur]
        for side in ('left', 'right'):
            nb = lg.neighbor(cur, side)
            if nb is not None and lg.lanes[nb]['dir'] == lg.lanes[cur]['dir'] \
                    and lg.lanes[nb]['type'] == 'driving':
                approaches.append(nb)
        opts = []
        for ap in approaches:
            opts += _junction_options(lg, ap)
        opts = [o for o in opts if o[0] != 'uturn']     # U턴 배제 (대회 최단거리 기준)
        if opts:
            return passed, add, cur, opts
        nxts = [k for k in lg.successors(cur) if lg.lanes[k]['junction'] == -1]
        if not nxts or add > fwd_cap:
            return passed, add, cur, []
        cur = nxts[0]
        passed.append(cur)
        add += lg.length(cur)
    return passed, add, cur, []


def _visit_roads(visited: set, prev_road, roads):
    """road 순서열을 단순 경로 규칙으로 반영 → (새 visited, 새 prev).

    연속된 같은 road 는 한 번의 통과로 본다 (같은 도로의 다음 섹션으로 전진하는
    것은 재방문이 아니다 — 이 구분이 없으면 존재율이 0%로 나온다). 이미 떠난
    road 로 되돌아오면 위반 → (None, None)."""
    v, p, added = visited, prev_road, []
    for r in roads:
        if r == p:
            continue
        if r in v or r in added:
            return None, None
        added.append(r)
        p = r
    return (v | set(added)) if added else v, p


def _search_walk(lg, rng, start, turns, start_len, min_len_m, used_roads,
                 budget, fwd_cap, max_depth, max_gap_m=None):
    """start 에서 필수 turns 를 만족시키며 목표 길이까지 잇는 단순 경로 탐색.

    막다른 회랑을 만나면 직전 교차로로 되돌아가 다른 출구를 시도한다 — 앞만
    보고 걷는 그리디가 3 km 를 못 채우던 원인(2026-08-29 실측: 미달 24건 전부
    막다른 회랑)을 해소한다. 필수 turns 는 탐색의 접두 제약으로 들어가므로
    그 구간에서도 재방문·막다름이 생기지 않는다.

    출구 시도 순서는 그리디와 같은 미방문 road 가중(Efraimidis-Spirakis:
    u^(1+방문횟수) 내림차순)이다 — 커버리지 지렛대를 탐색에서도 유지한다.
    → (crossings[(ap, jchain, exit)…], 최종 길이, 끝 차로, chain 추가분) 또는 None
    """
    exp = [0]

    def rec(lane, length, vis, prev, crossings, extra, depth, gap_base):
        if exp[0] >= budget or depth > max_depth:
            return None
        exp[0] += 1
        passed, add, end_lane, opts = _corridor_ahead(lg, lane, fwd_cap)
        vis2, prev2 = _visit_roads(vis, prev, [k[0] for k in passed])
        if vis2 is None:
            return None                     # 강제 회랑이 재방문 — 이 가지 폐기
        length2 = length + add
        extra2 = extra + passed
        if depth >= len(turns) and length2 >= min_len_m:
            return crossings, length2, end_lane, extra2
        if not opts:
            return None                     # 막다른 회랑 — 호출자가 다른 출구를 시도
        if max_gap_m is not None and depth > 0 and gap_base + add > max_gap_m:
            return None
        want = turns[depth] if depth < len(turns) else 'any'
        pick = opts if want == 'any' else [o for o in opts if o[0] == want]
        for kind, ap, jchain, ex in sorted(
                pick, reverse=True,
                key=lambda o: rng.random() ** (1.0 + used_roads.get(o[3][0], 0))):
            vis3, prev3 = _visit_roads(vis2, prev2, [k[0] for k in jchain] + [ex[0]])
            if vis3 is None:
                continue
            r = rec(ex, length2 + sum(lg.length(k) for k in jchain) + lg.length(ex),
                    vis3, prev3, crossings + [(ap, jchain, ex)],
                    extra2 + ([ap] if ap != end_lane else []) + jchain + [ex],
                    depth + 1, lg.length(ex))
            if r is not None:
                return r
        return None

    return rec(start, start_len, {start[0]}, start[0], [], [], 0, 0.0)


def _walk_once(lg, rng, start, turns, tail_m, max_gap_m,
               min_len_m: float = 0.0, ext_max: int = 0,
               used_roads: dict | None = None,
               ext_forward_max_m: float = 700.0,
               exit_clear_cap_m: float = 0.0,
               search: dict | None = None):
    """start 차로에서 turns 정책대로 걷는다. 성공 → (chain, waypoint rows), 실패 → None.

    · turns 소진 후 경로 길이가 min_len_m 미만이면 'any' 교차로 통과를 최대
      ext_max 회 자동 연장한다. 연장 중 실패(전진 불가·옵션 없음·gap 초과)는
      walk 전체 실패가 아니라 연장 중단이다 — 필수 turns 는 이미 충족됐다.
    · 'any' 의 교차로 출구는 균등이 아니라 **미방문 road 가중**으로 뽑는다
      (실행 내 방문 횟수 n → 가중 1/(1+n), Efraimidis-Spirakis 1개 추첨) —
      used_roads 는 RoutePool 이 실행 단위로 유지·기록한다.
    · search 가 주어지면(주행 검증형: min_length_m ≥ search_min_length_m) 필수
      turns 이후의 연장을 그리디 대신 **백트랙 탐색**(_search_extend)으로 한다 —
      막다른 회랑에서 되돌아 나올 수 있어 3 km 도달률이 오른다. 필수 turns
      구간은 기존 로직 그대로다.
    · 비교차로 전진의 successors[0] 은 실측상 아무것도 버리지 않는다
      (2026-08-28 전수: 일반도로 lane 의 비교차로 successor 는 0 또는 1개)."""
    if used_roads is None:
        used_roads = {}
    chain = [start]
    s0 = min(8.0, lg.length(start) * 0.2)
    visited_roads = None    # 탐색 모드에서만 채운다 — 꼬리 주행의 재방문 가드
    entries = []            # (entry_lane, exit_lane)
    cur = start
    dist_since_exit = lg.length(start) - s0
    route_len = dist_since_exit
    ti = 0
    while True:
        if search is not None:
            # 주행 검증형: 필수 turns + 연장을 한 번의 백트랙 탐색으로 만든다
            res = _search_walk(lg, rng, start, turns, route_len, min_len_m,
                               used_roads, int(search['expand_max']),
                               ext_forward_max_m, int(search['max_depth']), max_gap_m)
            if res is None:
                WALK_EXT_STATS['탐색 실패(목표 길이 단순 경로 없음)'] += 1
                return None                  # 다른 출발 후보로 재시도
            crossings, route_len, cur, extra = res
            chain += extra
            entries += [(ap, ex) for ap, _jc, ex in crossings]
            visited_roads = {start[0]} | {k[0] for k in extra}
            WALK_EXT_STATS['탐색 목표 달성'] += 1
            WALK_EXT_LENS['탐색 목표 달성'].append(route_len)
            break
        required = ti < len(turns)
        if required:
            want = turns[ti]
        elif route_len < min_len_m and ti - len(turns) < ext_max:
            want = 'any'                        # 목표 길이까지 연장
        else:
            if min_len_m:
                why = '목표 달성' if route_len >= min_len_m else 'ext_max 소진'
                WALK_EXT_STATS[why] += 1
                WALK_EXT_LENS[why].append(route_len)
            break
        ti += 1
        # 다음 교차로까지 전진. 700 m 상한은 "교차로를 못 찾고 헤매는 것"을
        # 막는 가드다 — 필수 turns 구간에만 적용하고, 연장 구간은 별도 상한
        # (walk_ext_forward_max_m)으로 풀어 긴 직선 회랑도 길이에 기여시킨다.
        fwd_cap = 700.0 if required else ext_forward_max_m
        fail = None
        guard = 0
        while guard < 60:
            guard += 1
            # 이 차로(또는 같은 방향 이웃)에서 교차로 옵션이 있는가
            approaches = [cur]
            for side in ('left', 'right'):
                nb = lg.neighbor(cur, side)
                if nb is not None and lg.lanes[nb]['dir'] == lg.lanes[cur]['dir'] \
                        and lg.lanes[nb]['type'] == 'driving':
                    approaches.append(nb)
            opts = []
            for ap in approaches:
                opts += _junction_options(lg, ap)
            if opts:
                break
            nxts = [k for k in lg.successors(cur) if lg.lanes[k]['junction'] == -1]
            if not nxts:
                fail = '막다른 회랑(successor 없음)'
                break
            cur = nxts[0]
            chain.append(cur)
            dist_since_exit += lg.length(cur)
            route_len += lg.length(cur)
            if dist_since_exit > fwd_cap:
                fail = '전방 상한'
                break
        else:
            fail = 'guard 소진'
        if fail is None and max_gap_m is not None and ti > 1 and dist_since_exit > max_gap_m:
            fail = 'max_gap 위반'
        pick = []
        if fail is None:
            pick = ([o for o in opts if o[0] == want] if want != 'any'
                    else [o for o in opts if o[0] != 'uturn'])
            if not pick:
                fail = '옵션 없음(원하는 방향/비U턴 출구 부재)'
        if fail is not None:
            if required:
                return None
            WALK_EXT_STATS[fail] += 1
            WALK_EXT_LENS[fail].append(route_len)
            break                               # 연장 중단 — 필수 구간은 완성
        if want == 'any':
            # 출구 가중 = 미방문(1/(1+방문)) × 전방 잔여 회랑 길이(정규화, cap 0=미사용)
            def _w(o):
                w = 1.0 / (1.0 + used_roads.get(o[3][0], 0))
                if exit_clear_cap_m > 0:
                    clear = _forward_clear_m(lg, o[3], exit_clear_cap_m)
                    w *= max(clear, 1.0) / exit_clear_cap_m
                return w
            kind, ap, jchain, exit_lane = max(
                pick, key=lambda o: rng.random() ** (1.0 / _w(o)))
        else:
            kind, ap, jchain, exit_lane = pick[rng.randrange(len(pick))]
        if ap is not cur and ap != cur:
            chain.append(ap)                     # 진입 전 차선변경 (이웃 차로로)
        entries.append((ap, exit_lane))
        chain += jchain
        chain.append(exit_lane)
        route_len += sum(lg.length(k) for k in jchain) + lg.length(exit_lane)
        cur = exit_lane
        dist_since_exit = lg.length(exit_lane)
    # 꼬리 주행
    tail = tail_m - (lg.length(cur) - min(3.0, lg.length(cur) * 0.3))
    guard = 0
    while tail > 0 and guard < 40:
        guard += 1
        nxts = [k for k in lg.successors(cur) if lg.lanes[k]['junction'] == -1]
        if not nxts:
            break
        if visited_roads is not None and nxts[0][0] != cur[0] \
                and nxts[0][0] in visited_roads:
            break                            # 꼬리가 지나온 도로로 되돌아간다 — 중단
        cur = nxts[0]
        chain.append(cur)
        if visited_roads is not None:
            visited_roads.add(cur[0])
        tail -= lg.length(cur)
    end_s = max(lg.length(cur) - max(5.0, min(15.0, lg.length(cur) * 0.2)), lg.length(cur) * 0.5)
    # 경유점 rows: 시작 + (진입, 진출)*N + 종료
    out, seq = [], 1
    x, y, _, _ = lg.point_at(start, s0)
    out.append((seq, x, y)); seq += 1
    for ap, ex in entries:
        x, y, _, _ = lg.point_at(ap, lg.length(ap) - 1.0)
        out.append((seq, x, y)); seq += 1
        x, y, _, _ = lg.point_at(ex, min(3.0, lg.length(ex) * 0.3))
        out.append((seq, x, y)); seq += 1
    x, y, _, _ = lg.point_at(cur, end_s)
    out.append((seq, x, y))
    return chain, out


_BLINK_CTRLS = None       # TEMPLATE 의 점멸 SignalController id 집합 (1회 추출)


def blink_ctrls() -> set:
    """점멸(blink) 페이즈를 가진 SignalController id — TEMPLATE 에서 읽는다.

    **하드코딩하지 않는 이유**: 지도·배포본이 바뀌면 id 가 따라 바뀐다. 생성기가
    이미 베이스로 쓰는 TEMPLATE 이 곧 시나리오가 될 XML 이므로, 거기서 뽑으면
    산출물과 항상 일치한다 (2026-09-01 현재 이 맵은 117 하나뿐).
    """
    global _BLINK_CTRLS
    if _BLINK_CTRLS is None:
        ls = ET.parse(TEMPLATE).getroot().find('LightSigns')
        _BLINK_CTRLS = {int(sc.get('Id')) for sc in (ls.findall('SignalController') if ls is not None else [])
                        if any(p.get('Type') == 'blink' for p in sc.findall('Phase'))}
    return _BLINK_CTRLS


# 점멸 정지선을 규율받는 접근 차로 id. 이 접근로(road 2312)는 정지선 하나에
# controller 두 개(116 빈 컨트롤러 / 117 점멸)가 함께 걸려 있고, 9910 은 접근로당
# 하나만 준다 — 차로 중심 t 대조(lane3 +5.29 ↔ 117 녹376 +5.20, lane2 +2.10 ↔
# 116 적371 +2.00)와 실측(lane 3 에서 117 수신)으로 3·4 만 점멸을 받는다.
# lane 2 를 섞으면 점멸이 안 잡히는 시나리오가 생겨 "채점기가 틀렸나 신호가 안
# 왔나"를 가릴 수 없다 (2026-09-01 결정).
BLINK_LANE_IDS = (3, 4)


def _blink_stopline(lg, v) -> bool:
    """이 차로의 정지선이 점멸 controller 에 걸려 있는가."""
    b = blink_ctrls()
    return any(b & set(sl.get('controller_ids') or []) for sl in (v.get('stop_lines') or []))


def _upstream_starts(lg, pred, back_m=180.0):
    """조건(pred)에 맞는 차로의 상류 차로들 — require 경로의 출발 후보."""
    outs = []
    for k, v in lg.lanes.items():
        if not pred(v) or v['junction'] != -1:
            continue
        cur, acc, guard = k, 0.0, 0
        while acc < back_m and guard < 30:
            guard += 1
            preds = [p for p in lg.predecessors(cur) if lg.lanes[p]['junction'] == -1
                     and lg.lanes[p]['type'] == 'driving']
            if not preds:
                break
            cur = preds[0]
            acc += lg.length(cur)
        if cur != k and lg.length(cur) >= 25.0:
            outs.append(cur)
    return sorted(set(outs))


def _check_require(lg, chain, require) -> bool:
    if require == 'school_zone':
        return any(lg.lanes[k]['school_zone'] for k in chain)
    if require == 'speed_change':
        vals = {(lg.lanes[k]['speed_limit'], lg.lanes[k]['school_zone'])
                for k in chain if lg.lanes[k]['speed_limit'] is not None
                or lg.lanes[k]['school_zone']}
        return len(vals) >= 2
    if require in ('signalized', 'signalized_slow', 'signalized_fast'):
        # 교차로 앞 접근 차로에 신호가 매핑돼 있어야 한다.
        # *_slow / *_fast 는 그 접근 차로의 제한속도까지 본다 — 정지선 정지
        # 프로파일(speed.stop_profile_a)을 저속·고속 진입 양쪽에서 밟기 위한
        # 변형이다 (2026-08-30, ④′ 검증).
        for i, k in enumerate(chain[:-1]):
            if lg.lanes[k]['junction'] == -1 and lg.lanes[chain[i + 1]]['junction'] != -1 \
                    and lg.lanes[k]['signals']:
                if require == 'signalized':
                    return True
                v = lg.lanes[k]['speed_limit']
                slow = lg.lanes[k]['school_zone'] or (v is not None and v <= 30)
                if (require == 'signalized_slow') == bool(slow):
                    return True
        return False
    if require == 'blink':
        # 교차로 직전 접근 차로가 점멸 정지선을 물고 있고, 그 차로가 실제로
        # 점멸을 수신하는 차로(BLINK_LANE_IDS)여야 한다. lane 2 를 통과시키면
        # 9910 이 116(빈 컨트롤러)을 줄 수 있어 점멸이 안 잡힌다.
        for i, k in enumerate(chain[:-1]):
            if lg.lanes[k]['junction'] == -1 and lg.lanes[chain[i + 1]]['junction'] != -1 \
                    and _blink_stopline(lg, lg.lanes[k]) and k[2] in BLINK_LANE_IDS:
                return True
        return False
    return True


_START_POOL_CACHE: dict = {}
_EV_CFG = None       # params.yaml gen_events 캐시 (장거리 다중이벤트 배치 상수)


def ev_cfg() -> dict:
    global _EV_CFG
    if _EV_CFG is None:
        from vtd_adapter.config import load_params_yaml
        _EV_CFG = load_params_yaml()['gen_events']
    return _EV_CFG


_TRG_CFG = None      # params.yaml event_trigger 캐시 (보행자 트리거 역산 상수)


def trg_cfg() -> dict:
    global _TRG_CFG
    if _TRG_CFG is None:
        from vtd_adapter.config import load_params_yaml
        _TRG_CFG = load_params_yaml()['event_trigger']
    return _TRG_CFG


_PULK_CFG = None     # params.yaml pulk 캐시 (VTD 네이티브 교통류 PulkDef 상수)


def pulk_cfg() -> dict:
    global _PULK_CFG
    if _PULK_CFG is None:
        from vtd_adapter.config import load_params_yaml
        _PULK_CFG = load_params_yaml()['pulk']
    return _PULK_CFG


_PLC_CFG = None      # params.yaml gen_placement 캐시 (이벤트별 배치 성립 조건)


def plc_cfg() -> dict:
    global _PLC_CFG
    if _PLC_CFG is None:
        from vtd_adapter.config import load_params_yaml
        _PLC_CFG = load_params_yaml()['gen_placement']
    return _PLC_CFG


_COV_CFG = None      # params.yaml gen_coverage 캐시 (후보 선별·추첨 상수의 단일 출처)


def cov_cfg() -> dict:
    global _COV_CFG
    if _COV_CFG is None:
        from vtd_adapter.config import load_params_yaml
        _COV_CFG = load_params_yaml()['gen_coverage']
    return _COV_CFG


def _forward_clear_m(lg, k, need: float) -> float:
    """k 시작(오프셋 8 m 제외)부터 비교차로 successor 를 따라 확보되는 전진
    거리 [m]. need 에 닿으면 조기 반환 — 짧은 차로의 가속 구간 보완용."""
    acc = lg.length(k) - 8.0
    cur = k
    for _ in range(20):
        if acc >= need:
            break
        nxts = [n for n in lg.successors(cur) if lg.lanes[n]['junction'] == -1]
        if not nxts:
            break
        cur = nxts[0]
        acc += lg.length(cur)
    return acc


def _quad4(h: float) -> int:
    """heading [rad] → 4분면 인덱스 (0=E, 1=N, 2=W, 3=S) — 셀 키의 방향 축."""
    return int(((math.degrees(h) % 360.0) + 45.0) // 90.0) % 4


def _cell_of(lg, k, cell_m: float) -> tuple:
    """시작 후보의 추첨 셀 키 = (격자 x, 격자 y, 시작 heading 4분면).

    방향 축이 없으면 같은 회랑의 역방향 쌍둥이(2026-08-28 실측: pool 426개 중
    반대차로 쌍 323, 그중 190쌍이 같은 200 m 셀)가 셀 감쇠에 함께 눌려,
    실행 내에서 도로당 한 방향만 뽑혔다 (60건 중 양방향 사용 도로 3/30개).
    방향을 키에 넣어 사용 이력 가중을 방향별로 분리한다 — 같은 위치라도
    미사용 방향은 감쇠 없이 우선 추첨된다."""
    x, y, _, h = lg.point_at(k, 0.0)
    return (int(x // cell_m), int(y // cell_m), _quad4(h))


def _spatial_order(lg, cands, rng, used_cells: dict) -> list:
    """공간·방향 분산 추첨 순서 — 맵을 (격자 셀 × 방향 4분면)으로 나눠 셀을
    먼저 뽑고 셀 안에서 추첨.

    셀 순열은 라운드마다 가중 무작위(Efraimidis-Spirakis: u^(1/w) 내림차순,
    같은 실행에서 이미 쓴 셀은 w=1/(1+사용횟수) 로 감쇠). 게이트 폐기 후
    재시도는 이 순서를 따라가므로 자연히 "풀 전체 재추첨"이다 — 인접 후보로
    밀리며 편향이 쌓이는 일이 없다."""
    cell_m = float(cov_cfg()['grid_cell_m'])
    boost = float(cov_cfg()['reverse_dir_boost'])
    by: dict = {}
    for k in cands:
        by.setdefault(_cell_of(lg, k, cell_m), []).append(k)
    for lst in by.values():
        rng.shuffle(lst)

    def _exp(c):
        # E-S 가중 u^(1/w): 지수 = 1/w. 기본 w=1/(1+사용횟수) → 지수 1+사용횟수.
        # 미사용 셀인데 같은 격자의 역방향 셀이 사용됐으면 w=reverse_dir_boost
        # — 방문 회랑의 미사용 방향을 능동 우선한다 (2026-08-28, 방향 편향 해소).
        u = used_cells.get(c, 0)
        if u == 0 and used_cells.get((c[0], c[1], (c[2] + 2) % 4), 0) > 0:
            return 1.0 / boost
        return 1.0 + u

    order = []
    while True:
        live = [c for c in by if by[c]]
        if not live:
            return order
        keyed = sorted(live, reverse=True, key=lambda c: rng.random() ** _exp(c))
        for c in keyed:
            order.append(by[c].pop())


def _forward_junction_ok(lg, start, horizon_m=700.0) -> bool:
    """start 에서 successor 를 따라가면 회전 가능한(비 U턴) 교차로 옵션이 나오는가."""
    cur, acc = start, 0.0
    for _ in range(60):
        approaches = [cur]
        for side in ('left', 'right'):
            nb = lg.neighbor(cur, side)
            if nb is not None and lg.lanes[nb]['dir'] == lg.lanes[cur]['dir'] \
                    and lg.lanes[nb]['type'] == 'driving':
                approaches.append(nb)
        for ap in approaches:
            if any(o[0] != 'uturn' for o in _junction_options(lg, ap)):
                return True
        nxts = [k for k in lg.successors(cur) if lg.lanes[k]['junction'] == -1]
        if not nxts:
            return False
        acc += lg.length(cur)
        if acc > horizon_m:
            return False
        cur = nxts[0]
    return False


def start_pool(lg) -> list:
    """맵 전체(도로 651개)에서 걷기 출발 후보를 1회 수집해 캐시.

    조건 (params gen_coverage 가 단일 출처): 일반 도로(junction=-1)의 주행
    차선, 길이 ≥ start_min_lane_m, 출발 오프셋(≤8 m) 이후 전진 거리(짧은
    차로는 비교차로 successor 누적)가 start_accel_m 이상, 전방이 회전 가능한
    연결로로 이어짐. 구 조건(차로 단독 58 m)은 실효 가속 50 m 과 같았고,
    successor 누적 보완으로 후보를 늘리되 가속 확보는 유지한다 (2026-08-28
    커버리지 분석: 58 m 필터가 후보를 1773→309 로 깎아 시작점이 63/651
    도로에 몰렸다)."""
    cov = cov_cfg()
    min_len = float(cov['start_min_lane_m'])
    accel = float(cov['start_accel_m'])
    key = (id(lg), min_len, accel)
    if key not in _START_POOL_CACHE:
        _START_POOL_CACHE[key] = [
            k for k, v in sorted(lg.lanes.items())
            if v['type'] == 'driving' and v['junction'] == -1
            and v['length'] >= min_len and v['next']
            and _forward_clear_m(lg, k, accel) >= accel
            and _forward_junction_ok(lg, k)]
    return _START_POOL_CACHE[key]


def _route_feasible(rt) -> bool:
    """빠른 사전 판정 — 짧은 차선변경 창(물리 불가)·회전불가 연결로 포함이면 탈락."""
    if rt['infeasible_forced']:
        return False
    for e in rt['events']:
        if e['kind'].startswith('lane_change') and e['window_s1'] - e['window_s0'] < 20.0:
            return False
    return True


def route_check(lg, rt):
    """build_route 의 report 를 그대로 돌려 경고 수를 센다 (출력은 삼킨다).

    batch_run 은 build_route rc=1(경고 존재)을 route 실패로 처리하므로, 생성
    시점 합격 기준을 실행 시점과 **완전히 같은 코드**로 맞춘다 — 실기에서
    회전 불가 연결로(R 1.75/2.55) 포함 경로가 3/5 실패한 사고의 재발 방지.
    → (경고 수, 첫 ⚠ 줄)
    """
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        warns = route_report(lg, rt, 8.0)
    first = next((line.strip() for line in buf.getvalue().splitlines() if '⚠' in line), '')
    return warns, first


# ── 스폰-경로 정합 게이트 ────────────────────────────────────────────────
# XML 의 Ego 스폰(Path01 첫 waypoint + PathRef StartS/StartLane)은 도로·s 만
# 지정하고 진행방향을 못 박는다. 실측(2026-08-26, 완주속도_01_속도전환2):
# 첫 waypoint 뒤쪽(−s)에도 둘째 waypoint 도로로 이어지는 우회가 있으면
# (무방향 608 m vs 정방향 587 m) VTD 가 Path 를 역방향 차로망으로 해석해
# 반대편 연결로에 스폰했다 — 되짚기 예측 (3174,0,-1)@7.5 m·hdg −4.2° 가
# 실측 (1423.0,1163.2)·hdg −3.4° 와 일치. 정상 케이스(도로 30)는 시작점
# 뒤가 막다른 끝(반대 차로 successors=[])이라 역방향 해석 자체가 없다.

GATE_STATS = {'ok': 0, 'reject': 0}     # main 이 리셋·요약 출력


def _reverse_spawn(lg, rt, wp1_road_s):
    """역방향 해석의 스폰 되짚기 → (lane, s_in_lane, heading). 반대 차로망이 없으면 None.

    Path01 첫 waypoint(road_s=wp1_road_s)를 도로 −s 방향(반대 dir 차로망) 위치로
    보고, 그 차로의 주행방향으로 StartS 만큼 전진한 지점 — VTD 가 역방향으로
    해석했을 때 Ego 가 놓이는 자리다."""
    start_key = tuple(rt['lanes'][0])
    start_s = float(rt['start_s_in_lane'])
    opp = lg.opposite_of(start_key)
    if opp is None:
        return None
    r = lg.lanes[opp]
    rs = np.asarray(r['road_s'], float)
    ls = np.asarray(r['s'], float)
    # dir=-1 차로는 road_s 가 감소 배열 — np.interp 는 x 증가 필요
    if rs[0] > rs[-1]:
        s_cur = float(np.interp(wp1_road_s, rs[::-1], ls[::-1]))
    else:
        s_cur = float(np.interp(wp1_road_s, rs, ls))
    cur, rem = opp, start_s
    for _ in range(30):
        left = lg.length(cur) - s_cur
        if rem <= left:
            s_fin = s_cur + rem
            _x, _y, _z, h = lg.point_at(cur, s_fin)
            return cur, s_fin, h
        rem -= left
        nxts = lg.successors(cur)
        if not nxts:
            L = lg.length(cur)
            _x, _y, _z, h = lg.point_at(cur, L)
            return cur, L, h
        cur = min(nxts, key=lambda k2: abs(wrap(_hdg_start(lg, k2) - _hdg_end(lg, cur))))
        s_cur = 0.0
    _x, _y, _z, h = lg.point_at(cur, s_cur)
    return cur, s_cur, h


def _reverse_reaches(lg, seed_lane, target_road, cap_m) -> bool:
    """seed_lane 에서 **주행방향(successor)으로** target_road 까지 cap_m 안에 닿는가.

    역방향 해석이 실재하려면 그 스폰 지점에서 차가 실제로 굴러갈 경로가 wp2
    도로로 이어져야 한다 — 실측 케이스도 successor 체인(864 m)으로 설명된다.
    반대 차로가 막다른 끝(successors 없음, 예: 도로 30)이면 VTD 는 그 해석으로
    Path 를 완성할 수 없으므로 통과다."""
    heap = [(0.0, seed_lane)]
    seen = set()
    while heap:
        c, k = heapq.heappop(heap)
        if k in seen:
            continue
        seen.add(k)
        if k[0] == target_road:
            return True
        if c > cap_m:
            continue
        for k2 in lg.successors(k):
            if k2 not in seen:
                heapq.heappush(heap, (c + lg.length(k2), k2))
    return False


def spawn_gate(lg, rt, gen_cfg):
    """스폰-경로 정합 게이트. 문제가 있으면 사유 문자열, 통과면 None.

    "XML 에 쓸 값을 lane_graph 로 되짚은 스폰 해석"과 "경로 첫 차로 heading"을
    대조한다. 정방향 해석은 경로 시작 차로 그 자체(차 0°)이므로, 역방향 해석이
    존재하고(반대 차로망으로 둘째 waypoint 도로 도달, 정방향 경로거리 ×
    reverse_ratio_max 이내) heading 차가 spawn_heading_max_deg 를 넘으면 폐기."""
    thr_deg = float(gen_cfg['spawn_heading_max_deg'])
    ratio = float(gen_cfg['reverse_ratio_max'])
    try:
        wps = path_waypoints(lg, rt)
    except GenError as e:
        return f'Path01 waypoint 구성 불가: {e}'
    start_key = tuple(rt['lanes'][0])
    _x, _y, _z, h_route = lg.point_at(start_key, float(rt['start_s_in_lane']))
    rev = _reverse_spawn(lg, rt, wps[0][1])
    if rev is None:
        return None                     # 시작점 뒤로 나가는 차로망이 없다 — 모호성 없음
    lane_r, s_r, h_r = rev
    d_deg = math.degrees(abs(wrap(h_r - h_route)))
    if d_deg <= thr_deg:
        return None
    wp2_road = wps[1][0]
    fwd_m = next((rt['cum_s'][i] for i, k in enumerate(rt['lanes']) if k[0] == wp2_road),
                 rt['total_length'])
    if not _reverse_reaches(lg, lane_r, wp2_road, max(float(fwd_m), 50.0) * ratio):
        return None                     # 역방향으로는 wp2 도로에 못 간다 — VTD 가 택할 수 없음
    return (f'스폰 방향 모호 — 역방향 해석 {lane_r}@{s_r:.1f} m, 경로 시작과 heading 차 '
            f'{d_deg:.0f}° (>{thr_deg:g}°), 뒤쪽으로 waypoint2 도로({wp2_road}) 우회 존재')


_PARAMS_CFG = None     # polyline_gate 용 params.yaml 캐시 (VtdRoutePlanner 재샘플에 필요)


def polyline_gate(lg, rt, gen_cfg):
    """경로 폴리라인 연속성 게이트. 문제가 있으면 사유 문자열, 통과면 None.

    컨트롤러가 실제로 따라갈 재샘플(VtdRoutePlanner._build, 10 cm 간격)을 그대로
    만들어 연속 점 간격을 검사한다 — 소멸 차로 킹크(2026-08-26 실기 1.5 m 점프
    → 조향 풀포화·차선이탈 진동) 같은 불연속 경로를 생성 시점에 폐기한다.
    route.py 의 taper_blend_m 수정과 독립적인 이중 방어 — 저쪽이 퇴행해도
    여기서 잡힌다.
    """
    global _PARAMS_CFG
    thr = float(gen_cfg['max_polyline_step_m'])
    if _PARAMS_CFG is None:
        from vtd_adapter.config import load_params_yaml
        _PARAMS_CFG = load_params_yaml()
    from vtd_adapter.route import VtdRoutePlanner
    pl = VtdRoutePlanner(lg, rt, _PARAMS_CFG)
    pts = pl.route_points
    d = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
    i = int(np.argmax(d))
    if d[i] > thr:
        return (f'경로 폴리라인 불연속 {d[i]:.2f} m (>{thr:g} m) — '
                f'route_s≈{float(pl.route_s[i]):.1f} m 부근 (테이퍼 킹크 의심)')
    return None


def synth_walk(lg, rng, name, spec, gen_cfg, used_cells: dict | None = None,
               used_roads: dict | None = None,
               min_length_m: float | None = None) -> Route:
    """walk 사양 → 검증까지 마친 Route. 실패는 GenError (무음 실패 금지).

    used_cells: 같은 실행에서 이미 시작점을 뽑은 격자 셀 → 사용 횟수. 공간
    분산 추첨(_spatial_order)의 감쇠 가중치로 쓰고, 성공 시 여기 기록한다.
    used_roads: 실행 내 방문 road → 횟수 — 'any' 출구 가중과 연장 통과가
    미방문 도로를 우선하게 한다 (커버리지 지렛대, params gen_coverage)."""
    turns = spec.get('turns', ['any'])
    tail = float(spec.get('tail_m', 100))
    require = spec.get('require')
    max_gap = spec.get('max_gap_m')
    cov = cov_cfg()
    # 경로 최소 길이 우선순위: 주제(min_length_m 인자) > routes 개별(spec) > 전역 기본
    min_len = float(min_length_m if min_length_m is not None
                    else spec.get('min_length_m', cov['walk_min_length_m']))
    ext_max = int(cov['walk_ext_turns_max'])
    if used_roads is None:
        used_roads = {}
    if require == 'school_zone':
        cands = _upstream_starts(lg, lambda v: v['school_zone'])
    elif require == 'speed_change':
        cands = _upstream_starts(lg, lambda v: v['speed_limit'] == 30 or v['school_zone'])
    elif require == 'signalized':
        cands = _upstream_starts(lg, lambda v: bool(v['signals']))
    elif require == 'signalized_slow':
        cands = _upstream_starts(lg, lambda v: bool(v['signals']) and (
            v['school_zone'] or (v['speed_limit'] is not None and v['speed_limit'] <= 30)))
    elif require == 'signalized_fast':
        cands = _upstream_starts(lg, lambda v: bool(v['signals'])
                                 and not v['school_zone']
                                 and (v['speed_limit'] or 0) >= 50)
    elif require == 'blink':
        # 점멸 신호는 이 맵에 접근로 하나뿐(junction 60 / road 2312)이라 후보가
        # 2개로 좁다 — 변이는 시작점이 아니라 경로 길이·꼬리에서 난다.
        cands = _upstream_starts(lg, lambda v: _blink_stopline(lg, v))
    else:
        cands = start_pool(lg)          # 맵 전체 후보 풀(1회 수집)에서 시드 샘플링
    if not cands:
        raise GenError(f'경로 {name}: 출발 후보 차로가 없다 (require={require})')
    if used_cells is None:
        used_cells = {}
    redraw_max = int(cov_cfg()['redraw_max'])
    # 주행 검증형(목표 길이 ≥ search_min_length_m)만 연장을 탐색으로 — 짧은
    # 경로에 탐색 비용을 쓸 이유가 없고, 이벤트형은 그리디로 이미 충족된다
    search = ({'expand_max': cov['search_expand_max'],
               'max_depth': cov['search_max_depth']}
              if min_len >= float(cov['search_min_length_m']) else None)
    order = _spatial_order(lg, cands, rng, used_cells)
    for start in order[:redraw_max]:
        res = _walk_once(lg, rng, start, turns, tail, max_gap,
                         min_len_m=min_len, ext_max=ext_max, used_roads=used_roads,
                         ext_forward_max_m=float(cov['walk_ext_forward_max_m']),
                         exit_clear_cap_m=float(cov['walk_exit_clear_cap_m']),
                         search=search)
        if res is None:
            continue
        chain, rows = res
        if not _check_require(lg, chain, require):
            continue
        try:
            route = _build_from_rows(lg, name, rows)
        except (RouteError, SystemExit):
            continue                     # 합성점이 빌드에 실패 — 다른 출발점으로
        if not _route_feasible(route.rt):
            continue                     # 차선변경 창 부족·회전불가 포함 — 빠른 탈락
        warns, first = route_check(lg, route.rt)
        if warns:
            continue                     # build_route 경고가 하나라도 있으면 폐기 후 재시도
        why = spawn_gate(lg, route.rt, gen_cfg) or polyline_gate(lg, route.rt, gen_cfg)
        if why:
            GATE_STATS['reject'] += 1    # 역방향 스폰 가능 또는 폴리라인 킹크 — 재시도
            continue
        GATE_STATS['ok'] += 1
        cell = _cell_of(lg, start, float(cov_cfg()['grid_cell_m']))
        used_cells[cell] = used_cells.get(cell, 0) + 1
        for rd in {k[0] for k in chain}:
            used_roads[rd] = used_roads.get(rd, 0) + 1
        return route
    raise GenError(f'경로 {name}: {min(len(order), redraw_max)}개 출발 후보로 걸어도 빌드 경고 0 · '
                   f'스폰 게이트 통과인 경로를 못 만들었다 (turns={turns}, require={require})')


class RoutePool:
    """경로 풀 — 같은 (이름, 변형, salt) 은 한 번만 만든다.

    salt 는 주제의 `start: 자유` 모드용이다: 주제 이름이 들어가 같은 경로 풀
    이름이라도 주제·경로변형마다 다른 시드 → 다른 시작점에서 걷는다.
    csv 경로는 시작점이 파일에 고정이라 salt 의 영향이 없다.
    """

    def __init__(self, lg, defs: dict, seed: int, gen_cfg: dict):
        self.lg, self.defs, self.seed = lg, defs, seed
        self.gen_cfg = gen_cfg
        self.cache: dict = {}
        self.used_cells: dict = {}      # 같은 실행 내 시작 셀 사용 이력 (공간 분산)
        self.used_roads: dict = {}      # 같은 실행 내 방문 road 이력 ('any' 출구 가중)

    def get(self, name: str, variant: int = 1, salt: str = '',
            min_length_m: float | None = None) -> Route:
        """min_length_m: 주제 지정 경로 최소 길이 — 우선순위는 주제 > routes
        개별(spec) > gen_coverage 기본 (synth_walk 이 해석). 캐시 키에 포함해
        같은 경로 이름이라도 주제 목표 길이가 다르면 따로 만든다."""
        if name not in self.defs:
            raise GenError(f'routes 풀에 "{name}" 이 없다 (configs/themes.yaml)')
        d = self.defs[name]
        if 'csv' in d:
            variant, salt, min_length_m = 1, '', None   # csv 는 파일에 고정
        key = (name, variant, salt, min_length_m)
        if key in self.cache:
            cached = self.cache[key]
            if isinstance(cached, BaseException):   # GenError 는 SystemExit 계열이다
                raise cached            # 한 번 탈락한 경로는 변형마다 재검사하지 않는다
            return cached
        if 'csv' in d:
            rows = read_waypoints_csv(str(ROOT / d['csv']))
            r = _build_from_rows(self.lg, name, rows)
            # csv 는 재시도할 다른 시작점이 없다 — 경고가 있으면 경로째 탈락시키고
            # 이 경로를 쓰는 변형 전부를 사유와 함께 제외한다 (조용히 포함 금지)
            warns, first = route_check(self.lg, r.rt)
            if warns:
                err = GenError(f'경로 {name}({d["csv"]}): build_route 경고 {warns}건 — {first}')
                self.cache[key] = err
                raise err
            why = (spawn_gate(self.lg, r.rt, self.gen_cfg)
                   or polyline_gate(self.lg, r.rt, self.gen_cfg))
            if why:                     # csv 는 재시도할 다른 시작점이 없다 — 경로째 탈락
                GATE_STATS['reject'] += 1
                err = GenError(f'경로 {name}({d["csv"]}): 경로 게이트 탈락 — {why}')
                self.cache[key] = err
                raise err
            GATE_STATS['ok'] += 1
        elif 'walk' in d:
            seed_str = f'{self.seed}:route:{name}:{variant}' + (f':{salt}' if salt else '')
            rng = random.Random(seed_str)
            label = name if variant == 1 else f'{name}{variant}'
            r = synth_walk(self.lg, rng, label, d['walk'], self.gen_cfg,
                           used_cells=self.used_cells, used_roads=self.used_roads,
                           min_length_m=min_length_m)
        else:
            raise GenError(f'경로 {name}: csv 또는 walk 정의가 필요하다')
        self.cache[key] = r
        return r


# ── 경로 위 기하 헬퍼 ────────────────────────────────────────────────────

def lane_at(rt, s: float):
    """route_s → (i, lane, s_in_lane). 평행(차선변경) 중복 구간은 첫 차로."""
    lanes, cum, lens = rt['lanes'], rt['cum_s'], rt['lengths']
    for i, k in enumerate(lanes):
        if cum[i] - 1e-6 <= s <= cum[i] + lens[i] + 1e-6:
            return i, k, min(max(s - cum[i], 0.0), lens[i])
    raise GenError(f'route_s {s:.1f} 가 경로 밖이다 (total {rt["total_length"]:.1f})')


def route_pt(lg, rt, s: float, t_off: float = 0.0):
    """route_s(+횡오프셋, 좌 +) → (x, y, z, hdg)"""
    _, k, sl = lane_at(rt, s)
    x, y, z, h = lg.point_at(k, sl)
    return x - t_off * math.sin(h), y + t_off * math.cos(h), z, h


def usable_spans(lg, rt, start_margin=60.0, end_margin=40.0, junction_margin=25.0):
    """이벤트를 놓을 수 있는 route_s 구간 (교차로·시점·종점 여유 제외)."""
    lanes, cum, lens = rt['lanes'], rt['cum_s'], rt['lengths']
    total = rt['total_length']
    jiv = []                        # 교차로 구간들
    for i, k in enumerate(lanes):
        if lg.lanes[k]['junction'] != -1:
            jiv.append((cum[i] - junction_margin, cum[i] + lens[i] + junction_margin))
    # 병합
    jiv.sort()
    merged = []
    for a, b in jiv:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))
    spans, lo = [], start_margin
    for a, b in merged + [(total - end_margin, total)]:
        if a > lo + 5.0:
            spans.append((lo, a))
        lo = max(lo, b)
    return [(a, b) for a, b in spans if b - a >= 20.0]


def pick_s(spans, frac: float, need: float = 0.0) -> float:
    """구간들을 이어붙인 길이축에서 비율 frac 지점의 route_s. need 만큼 뒤 여유 확보."""
    flat = sum(b - a for a, b in spans)
    if flat <= need + 1.0:
        raise GenError(f'이벤트를 놓을 구간이 부족하다 (가용 {flat:.0f} m, 필요 {need:.0f} m)')
    tgt = frac * (flat - need)
    acc = 0.0
    for a, b in spans:
        if tgt <= acc + (b - a):
            return a + (tgt - acc)
        acc += b - a
    return spans[-1][1] - need - 1.0


def road_width_at(lg, rt, s: float):
    """ego 차로 기준 좌/우 차도 가장자리까지 거리 (+left, +right).

    lanegraph.roadway_edges 공용 — 반대 통행방향 차로 폭을 왼쪽에 포함한다.
    2026-08-25 정지 고착: 반대차로를 빼고 계산한 횡단 폭 때문에 보행자가
    반대 차선 위에 멈췄고, 컨트롤러(같은 차도 판정)가 영원히 대기했다.
    """
    _, k, sl = lane_at(rt, s)
    return lg.roadway_edges(k, sl)


# ════════════════════════════════════════════════════════════════════════
# 2. 교차로 접근 컨트롤러 매핑 (1회 생성 → data/junction_ctrl_map.json 캐시)
# ════════════════════════════════════════════════════════════════════════

def junction_ctrl_map(lg) -> dict:
    """{junction_id(str): {"road,dir": [controller_id…]}}.

    xodr 를 다시 파싱하지 않는다 — lane_graph.pkl(=xodr 파싱 결과)의 차로별
    signal / signal_to_controller 로 만든다. 최초 1회만 만들고 JSON 캐시.
    """
    if CTRL_MAP_JSON.exists():
        return json.loads(CTRL_MAP_JSON.read_text(encoding='utf-8'))
    s2c = lg.g['signal_to_controller']
    out: dict = {}
    for k, v in lg.lanes.items():
        if not v['signals'] or v['junction'] != -1:
            continue
        jid = lg.junction_ahead(k)
        if jid is None:
            continue
        key = f'{k[0]},{v["dir"]}'
        ctrls = out.setdefault(str(jid), {}).setdefault(key, set())
        for sig in v['signals']:
            for c in s2c.get(sig['id'], []):
                ctrls.add(int(c))
    out = {j: {a: sorted(c) for a, c in appr.items()} for j, appr in out.items()}
    CTRL_MAP_JSON.write_text(json.dumps(out, indent=1, sort_keys=True), encoding='utf-8')
    print(f'교차로 접근 컨트롤러 매핑 생성 → {CTRL_MAP_JSON} (교차로 {len(out)}개)')
    return out


def route_signal_approaches(lg, rt, ctrl_map):
    """경로의 신호화 접근들 → [(junction, stopline_route_s, [controller…])]"""
    lanes, cum, lens = rt['lanes'], rt['cum_s'], rt['lengths']
    out = []
    for i, k in enumerate(lanes):
        if lg.lanes[k]['junction'] == -1 or (i > 0 and lg.lanes[lanes[i - 1]]['junction'] != -1):
            continue
        if i == 0:
            continue
        ap = lanes[i - 1]
        v = lg.lanes[ap]
        jid = lg.lanes[k]['junction']
        ctrls = set()
        for sig in v['signals']:
            for c in lg.g['signal_to_controller'].get(sig['id'], []):
                ctrls.add(int(c))
        for c in ctrl_map.get(str(jid), {}).get(f'{ap[0]},{v["dir"]}', []):
            ctrls.add(int(c))
        stop_s = None
        for sl in v['stop_lines']:
            if sl['signal_ids'] or sl['controller_ids']:
                stop_s = cum[i - 1] + sl['s']
                break
        if stop_s is None:
            stop_s = cum[i]                     # 정지선 미매핑 → 교차로 진입점
        if ctrls:
            out.append((jid, stop_s, sorted(ctrls)))
    return out


# ════════════════════════════════════════════════════════════════════════
# 3. XML 조립 — 치환·삽입은 전부 개수 검증 (무음 실패 금지)
# ════════════════════════════════════════════════════════════════════════

class XmlDoc:
    def __init__(self, template_path: pathlib.Path):
        if not template_path.exists():
            raise GenError(f'템플릿이 없다: {template_path}')
        self.text = template_path.read_text(encoding='utf-8')

    def _sub1(self, pattern, repl_text, what):
        new, n = re.subn(pattern, lambda m: repl_text, self.text, count=1, flags=re.S)
        if n != 1:
            hits = len(re.findall(pattern, self.text, flags=re.S))
            raise GenError(f'XML 치환 실패: {what} (패턴 일치 {hits}개, 1개 기대)')
        self.text = new

    def set_path(self, waypoints):
        """Path01 을 경로 도로열로 교체. waypoints = [(track_id, road_s)…]"""
        lines = '\n'.join(
            f'            <Waypoint PathOption="shortest" s="{fnum(s)}" TrackId="{tid}"/>'
            for tid, s in waypoints)
        block = f'<Path Name="Path01" PathId="1">\n{lines}\n        </Path>'
        self._sub1(r'<Path Name="Path01" PathId="1">.*?</Path>', block, 'Path01 교체')

    def set_ego(self, start_s, target_s, start_lane):
        self._sub1(
            r'<PathRef StartS="[^"]*" EndAction="continue" TargetS="[^"]*" StartLane="[^"]*" PathId="1"/>',
            f'<PathRef StartS="{fnum(start_s)}" EndAction="continue" '
            f'TargetS="{fnum(target_s)}" StartLane="{start_lane}" PathId="1"/>',
            'Ego PathRef')

    def add_players(self, blocks: list):
        if not blocks:
            return
        anchor = ('        <Player>\n'
                  '            <Description Driver="DefaultDriver" Control="external"')
        if self.text.count(anchor) != 1:
            raise GenError(f'Player 삽입 앵커가 {self.text.count(anchor)}개다 (1개 기대)')
        ins = ''.join(b.rstrip() + '\n' for b in blocks)
        self.text = self.text.replace(anchor, ins + anchor, 1)

    def add_player_actions(self, blocks: list):
        if not blocks:
            return
        anchor = '<PlayerActions Player="Ego"/>'
        if self.text.count(anchor) != 1:
            raise GenError('PlayerActions 삽입 앵커를 못 찾았다')
        ins = anchor + ''.join('\n' + b.rstrip() for b in blocks)
        self.text = self.text.replace(anchor, ins, 1)

    def add_moving(self, blocks: list):
        if not blocks:
            return
        anchor = '    <MovingObjectsControl>\n    </MovingObjectsControl>'
        if self.text.count(anchor) != 1:
            raise GenError('MovingObjectsControl 앵커를 못 찾았다 (템플릿이 9_clean_drive 인가?)')
        body = ''.join('\n' + b.rstrip() for b in blocks)
        self.text = self.text.replace(
            anchor, f'    <MovingObjectsControl>{body}\n    </MovingObjectsControl>', 1)

    def set_pulk(self, attrs: str):
        """빈 <PulkTraffic/> 을 <PulkTraffic><PulkDef …/></PulkTraffic> 로 채운다.

        대회 제공 XML 11행에 이미 빈 태그가 있으므로 **삽입이 아니라 치환**이다
        — 새로 붙이면 태그가 둘이 된다. _sub1 이 일치 1개를 강제하므로 중복
        생성도, 앵커 실종도 조용히 지나가지 않는다.
        """
        self._sub1(r'<PulkTraffic\s*/>|<PulkTraffic>.*?</PulkTraffic>',
                   '<PulkTraffic>\n'
                   f'        <PulkDef {attrs}/>\n'
                   '    </PulkTraffic>', 'PulkTraffic 채우기')

    def set_signal(self, ctrl_id: int, go: float, attention: float, stop: float):
        m = re.search(rf'<SignalController Id="{ctrl_id}" [^>]*>.*?</SignalController>',
                      self.text, flags=re.S)
        if not m:
            raise GenError(f'SignalController Id={ctrl_id} 가 템플릿에 없다')
        block = m.group(0)
        new = block
        for typ, dur in (('go', go), ('attention', attention), ('stop', stop)):
            new, n = re.subn(rf'(<Phase Duration=")[^"]*(" Type="{typ}"/>)',
                             rf'\g<1>{dur:.1f}\g<2>', new, count=1)
            if n != 1:
                raise GenError(f'SignalController {ctrl_id}: Phase({typ}) 치환 실패')
        self.text = self.text.replace(block, new, 1)

    def final(self, name: str) -> str:
        try:
            ET.fromstring(self.text)
        except ET.ParseError as e:
            raise GenError(f'{name}: 생성된 XML 이 파싱되지 않는다 — {e}')
        return self.text


# ── 검증된 블록 템플릿 (templates/ 의 2/3/4/5/ped 시나리오에서 추출) ──────

def blk_vehicle_pathref(name, vtype, speed, start_s, target_s, start_lane):
    return f'''        <Player>
            <Description Driver="DefaultDriver" Control="internal" AdaptDriverToVehicleType="true" Type="{vtype}" Name="{name}"/>
            <Init>
                <Speed Value="{fnum(speed)}"/>
                <PosRoute/>
                <PathRef StartS="{fnum(start_s)}" EndAction="continue" TargetS="{fnum(target_s)}" StartLane="{start_lane}" PathId="1"/>
            </Init>
        </Player>
'''


def blk_vehicle_posabs(name, vtype, x, y, z, direction):
    return f'''        <Player>
            <Description Driver="DefaultDriver" Control="internal" AdaptDriverToVehicleType="true" Type="{vtype}" Name="{name}"/>
            <Init>
                <Speed Value="0.0"/>
                <PosAbsolute X="{fnum(x)}" Y="{fnum(y)}" Z="{fnum(z)}" Direction="{fnum(direction)}" AlignToRoad="true"/>
            </Init>
        </Player>
'''


def blk_vehicle_pathshape(name, vtype, shape_id, start_s, target_s):
    return f'''        <Player>
            <Description Driver="DefaultDriver" Control="internal" AdaptDriverToVehicleType="true" Type="{vtype}" Name="{name}"/>
            <Init>
                <Speed Value="0.0"/>
                <PosPathShape/>
                <PathShapeRef StartS="{fnum(start_s)}" EndAction="continue" TargetS="{fnum(target_s)}" UsedFor="steering" PathShapeId="{shape_id}"/>
            </Init>
        </Player>
'''


def _act(name, trig_x, trig_y, radius, pivot, body, delay=0.0):
    return (f'            <Action Name="{name}">\n'
            f'                <PosAbsolute CounterID="" CounterComp="COMP_EQ" Radius="{fnum(radius)}" '
            f'X="{fnum(trig_x)}" Y="{fnum(trig_y)}" NetDist="false" CounterVal="0" Pivot="{pivot}"/>\n'
            f'{body}'
            f'            </Action>\n')


def _speed_change(rate, target, delay=0.0):
    return (f'                <SpeedChange Rate="{fnum(rate)}" Target="{fnum(target)}" '
            f'Force="true" ExecutionTimes="1" ActiveOnEnter="true" DelayTime="{fnum(delay)}"/>\n')


def blk_player_actions(player, actions):
    return (f'        <PlayerActions Player="{player}">\n'
            + ''.join(actions)
            + '        </PlayerActions>\n')


def blk_stay_action(player, x, y):
    return blk_player_actions(player, [_act('stay', x, y, 20.0, player, _speed_change(10.0, 0.0))])


def blk_pathshape(shape_id, name, pts):
    """pts = [(x, y, z, yaw)…]"""
    wp = ''.join(
        f'            <Waypoint X="{fnum(x)}" Y="{fnum(y)}" Options="0x00000000" Z="{fnum(z)}" '
        f'Weight="1.0" Yaw="{fnum(yaw)}" Pitch="0.0" Roll="0.0" Time="0.0"/>\n'
        for x, y, z, yaw in pts)
    return (f'        <PathShape ShapeId="{shape_id}" ShapeType="polyline" Closed="false" Name="{name}">\n'
            f'{wp}        </PathShape>\n')


def blk_character(name, x, y, z, direction):
    return (f'        <Character CharacterType="male_adult" Class="pedestrian" '
            f'Appearance="Christian" Name="{name}">\n'
            f'            <StartPosAbs X="{fnum(x)}" Y="{fnum(y)}" Z="{fnum(z)}" Direction="{fnum(direction)}"/>\n'
            f'        </Character>\n')


def blk_character_actions(name, trig_x, trig_y, radius, walk_speed, shape_id):
    return (f'        <CharacterActions Character="{name}">\n'
            f'            <Action Name="cross">\n'
            f'                <PosAbsolute CounterID="" CounterComp="COMP_EQ" Radius="{fnum(radius)}" '
            f'X="{fnum(trig_x)}" Y="{fnum(trig_y)}" NetDist="false" CounterVal="0" Pivot="Ego"/>\n'
            f'                <Motion Move="walk" Rate="0.0" Speed="{fnum(walk_speed)}" Force="false" '
            f'ExecutionTimes="1" ActiveOnEnter="true" DelayTime="0.0"/>\n'
            f'                <CharacterPath Loop="false" PathShape="{shape_id}" ExecutionTimes="1" '
            f'ActiveOnEnter="true" DelayTime="0.0" Beam="true" ClampToGround="true"/>\n'
            f'            </Action>\n'
            f'        </CharacterActions>\n')


def blk_object(name, x, y, z):
    return (f'        <Object Type="other" Name="{name}" Definition="Fuelcan01">\n'
            f'            <StartPosAbs X="{fnum(x)}" Y="{fnum(y)}" Z="{fnum(z)}" '
            f'Direction="0.0" Pitch="0.0" Roll="0.0"/>\n'
            f'        </Object>\n')


# ════════════════════════════════════════════════════════════════════════
# 4. 이벤트 배치
# ════════════════════════════════════════════════════════════════════════

@dataclass
class Ctx:
    lg: LaneGraph
    route: Route
    rng: random.Random
    ctrl_map: dict
    players: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    moving: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)      # ctrl_id → (go, att, stop)
    checks: list = field(default_factory=list)       # (x, y, tag)  tag=ego_lane 만 3 m 검사
    occupied: list = field(default_factory=list)
    shape_seq: int = 0
    player_seq: int = 0

    def claim(self, s0, s1, what):
        for a, b in self.occupied:
            if s0 < b and a < s1:
                raise GenError(f'{what}: 이벤트 구간이 겹친다 [{s0:.0f},{s1:.0f}] vs [{a:.0f},{b:.0f}]')
        self.occupied.append((s0, s1))

    def next_shape(self):
        self.shape_seq += 1
        return self.shape_seq

    def next_name(self, base):
        self.player_seq += 1
        return f'{base}{self.player_seq:02d}'

    @property
    def spans(self):
        return usable_spans(self.lg, self.route.rt)


def _crossing_shape(ctx, s, from_side, extra=0.0):
    """s 지점을 가로지르는 보행 경로점들과 진행방향. from_side: 우측→좌측 / 좌측→우측."""
    lg, rt = ctx.lg, ctx.route.rt
    left, right = road_width_at(lg, rt, s)
    m = float(trg_cfg()['ped_start_margin_m'])     # 역산(ped_trigger)과 같은 값이어야 한다
    if from_side == '우측':
        t0, t1 = -(right + m + extra), left + m
    else:
        t0, t1 = left + m + extra, -(right + m)
    pts = []
    for u in np.linspace(0.0, 1.0, 4):
        t = t0 + (t1 - t0) * float(u)
        x, y, z, h = route_pt(lg, rt, s, t)
        yaw = h + (math.pi / 2 if t1 > t0 else -math.pi / 2)
        pts.append((x, y, z, yaw % (2 * math.pi)))
    return pts


def ped_walk_speed(v, kind):
    """보행속도 [m/s] — 축(vary 보행속도)이 있으면 그 값, 없으면 이벤트별 기본값.

    무단횡단(jaywalk/ped_blind)은 뛰어서 건너므로 정상 횡단(pedestrian)보다 빠르다.
    기본값은 params event_trigger.walk_speed_default.* 가 단일 출처다.
    """
    if '보행속도' in v:
        return float(v['보행속도'])
    return float(trg_cfg()['walk_speed_default'][kind])


def ped_trigger(ctx, s, walk_speed, from_side, kind):
    """보행자 트리거 거리를 조우가 성립하도록 역산한다 (params event_trigger.*).

    고정 트리거거리(옛 25 m)는 도로 폭에 따라 필요값이 27~87 m 로 변해 성립하지
    않았다 — 2026-08-30 실측: 보행자가 ego 통과 4 s 뒤에 차로 진입, 무반응 통과.

    ego 가 이벤트 지점 s 에 닿는 순간 보행자가 **차로 근접 가장자리를 lead_s 만큼
    지난** 위치에 오도록 잡는다. 조우 창은 차로폭/보행속도(1~2 s)뿐이라 lead_s 를
    크게 잡으면 반대로 이미 건너가 버린다.

        t_near = (횡시작거리 − 차로반폭) / 보행속도      # 근접 가장자리 도달
        t_far  = (횡시작거리 + 차로반폭) / 보행속도      # 차로 이탈
        trig_d = v_exp × (t_near + lead_s) − radius_m   # 반경만큼 일찍 터진다

    trig_d 가 [trig_min_m, trig_max_m] 를 벗어나면 보행속도를
    [walk_speed_min_mps, walk_speed_max_mps] 안에서 자동 조정해 성립시키고,
    그래도 안 되면 GenError — 호출자가 그 이벤트만 버린다(min_keep_ratio).

    반환 dict 는 그대로 시나리오 yaml 에 실려 검증(tools/event_check.py)의
    기대값이 된다.
    """
    et = trg_cfg()
    lg, rt = ctx.lg, ctx.route.rt
    left, right = road_width_at(lg, rt, s)
    lat = (right if from_side == '우측' else left) + float(et['ped_start_margin_m'])
    _, k, sl = lane_at(rt, s)
    half = 0.5 * float(lg.width_at(k, sl))
    lim = lg.lanes[k]['speed_limit'] or float(et['default_limit_kph'])
    v_exp = float(lim) / 3.6 * float(et['speed_factor'])
    lead, rad = float(et['lead_s']), float(et['radius_m'])
    lo, hi = float(et['trig_min_m']), float(et['trig_max_m'])
    ws_lo, ws_hi = float(et['walk_speed_min_mps']), float(et['walk_speed_max_mps'])
    if lat <= half:
        raise EventUnfeasible(f'{kind}: 보행자 시작점이 이미 차로 안이다 (횡 {lat:.1f} m)')

    def trig_of(ws):
        return v_exp * ((lat - half) / ws + lead) - rad

    # 보행속도를 상·하한 안에서 조정해 trig_d 를 [lo, hi] 로 넣는다.
    #   trig_d ≤ hi  ⟺  ws ≥ (lat−half) / ((hi+rad)/v_exp − lead)
    #   trig_d ≥ lo  ⟺  ws ≤ (lat−half) / ((lo+rad)/v_exp − lead)
    ws = float(walk_speed)
    for bound, want_fast in ((hi, True), (lo, False)):
        budget = (bound + rad) / v_exp - lead
        if budget <= 0:
            raise EventUnfeasible(
                f'{kind}: 트리거 한계 {bound:.0f} m 가 lead_s 를 못 넘는다 '
                f'(제한 {lim:.0f} kph)')
        need = (lat - half) / budget
        if want_fast and ws < need:
            ws = need                     # (b) 자동 상향
        if (not want_fast) and ws > need:
            ws = need                     # 하한 미달 — 보행속도를 낮춰 트리거를 늘린다
    if not (ws_lo - 1e-9 <= ws <= ws_hi + 1e-9):
        why = ('너무 빨라야' if ws > ws_hi else
               '너무 느려야')          # 느림 = 트리거가 하한보다 짧아진다
        raise EventUnfeasible(
            f'{kind}: 조우가 성립하려면 보행속도가 {why} 한다 ({ws:.2f} m/s, 허용 '
            f'[{ws_lo:g}, {ws_hi:g}]) — 횡 {lat:.1f} m, 차로 {2 * half:.1f} m, '
            f'제한 {lim:.0f} kph')
    trig_d = trig_of(ws)
    if not (lo - 1e-6 <= trig_d <= hi + 1e-6):
        raise EventUnfeasible(f'{kind}: 트리거 거리 {trig_d:.0f} m 가 [{lo:g}, {hi:g}] 밖이다')
    if s - trig_d < float(et['trig_min_route_s_m']):
        raise EventUnfeasible(
            f'{kind}: 트리거 지점 route_s {s - trig_d:.0f} m 가 경로 앞부분이라 스폰 즉시 '
            f'발동한다 (이벤트 s={s:.0f}, trig_d={trig_d:.0f})')

    t_near, t_far = (lat - half) / ws, (lat + half) / ws
    return {'walk_speed': round(ws, 3), 'trigger_d': round(trig_d, 2),
            'lat_start_m': round(lat, 2), 'lane_w_m': round(2 * half, 2),
            'v_exp_mps': round(v_exp, 2), 'limit_kph': round(float(lim), 1),
            't_cross_s': round(lat / ws, 2),
            't_near_s': round(t_near, 2), 't_far_s': round(t_far, 2),
            # ego 도착 순간의 보행자 횡위치 (차로 중심 기준, + = 아직 출발측)
            'meet_lat_m': round(half - ws * lead, 2),
            'trigger_radius_m': rad}


def ped_claim(ctx, s, trg, kind):
    """보행자 이벤트 점유 구간 — 트리거 지점까지 잡아야 다른 이벤트가 안 낀다."""
    et = trg_cfg()
    ctx.claim(s - trg['trigger_d'] - float(et['claim_back_m']),
              s + float(et['claim_fwd_m']), kind)


def _add_ped(ctx, s, trg, from_side, tag):
    pts = _crossing_shape(ctx, s, from_side)
    sid = ctx.next_shape()
    name = ctx.next_name('Ped')
    tx, ty, _, _ = route_pt(ctx.lg, ctx.route.rt, s - trg['trigger_d'])
    ctx.moving.append(blk_pathshape(sid, f'{name}Path', pts))
    ctx.moving.append(blk_character(name, pts[0][0], pts[0][1], pts[0][2], pts[0][3]))
    ctx.moving.append(blk_character_actions(name, tx, ty, trg['trigger_radius_m'],
                                            trg['walk_speed'], sid))
    cx, cy, _, _ = route_pt(ctx.lg, ctx.route.rt, s)
    ctx.checks.append((cx, cy, tag))
    return {'ped': name, 'route_s': round(s, 2), 'from': from_side, **trg}


def _crosswalk_list(ctx):
    lg, rt = ctx.lg, ctx.route.rt
    lanes, cum = rt['lanes'], rt['cum_s']
    out = []
    for i, k in enumerate(lanes):
        for a, b, kind in lg.lanes[k]['crosswalks']:
            rs = cum[i] + a
            if 60.0 <= rs <= rt['total_length'] - 25.0:
                out.append(rs)
    out.sort()
    dedup = []
    for rs in out:
        if not dedup or rs - dedup[-1] > 3.0:
            dedup.append(rs)
    return dedup


def _red60_for(ctx, stop_s, ctrls):
    t_arr = stop_s / AVG_SPEED_MPS + 2.0
    go = 8.0 if t_arr >= 12.0 else max(2.0, t_arr - 6.0)
    for c in ctrls:
        ctx.signals[c] = (go, 3.0, 60.0)


def ev_pedestrian(ctx, v):
    cws = _crosswalk_list(ctx)
    if not cws:
        raise GenError('경로에 횡단보도가 없다 — pedestrian 이벤트 불가')
    idx = round(v['위치'] * (len(cws) - 1))
    s = cws[idx]
    side = v.get('방향', '우측')
    trg = ped_trigger(ctx, s, ped_walk_speed(v, 'pedestrian'), side, 'pedestrian')
    ped_claim(ctx, s, trg, 'pedestrian')
    out = _add_ped(ctx, s, trg, side, 'ego_lane')
    out['kind'] = 'pedestrian'
    if v.get('신호') == '적색':
        appr = [(j, ss, cc) for j, ss, cc in
                route_signal_approaches(ctx.lg, ctx.route.rt, ctx.ctrl_map)
                if abs(ss - s) < 40.0]
        if appr:
            _red60_for(ctx, appr[0][1], appr[0][2])
            out['signal'] = {'junction': appr[0][0], 'controllers': appr[0][2], 'timing': '적60'}
    return out


def ev_jaywalk(ctx, v):
    if v.get('지점') == '횡단보도':
        cws = _crosswalk_list(ctx)
        if cws:
            s = cws[round(v['위치'] * (len(cws) - 1))]
        else:
            s = pick_s(ctx.spans, v['위치'])
    else:
        s = pick_s(ctx.spans, v['위치'])
    side = v.get('방향', '우측')
    trg = ped_trigger(ctx, s, ped_walk_speed(v, 'jaywalk'), side, 'jaywalk')
    ped_claim(ctx, s, trg, 'jaywalk')
    out = _add_ped(ctx, s, trg, side, 'ego_lane')
    out.update(kind='jaywalk', spot=v.get('지점', '도로중간'))
    return out


def ev_ped_blind(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    s = pick_s(ctx.spans, v['위치'], need=40.0)
    bus = v.get('차폐물', '버스') == '버스'
    vtype, vlen = (BUS, BUS_LEN) if bus else (CAR, CAR_LEN)
    _, k, sl = lane_at(rt, s)
    w = lg.width_at(k, sl)
    t_block = -(w / 2.0 + 1.1)                      # 차로 우측 가장자리 바깥에 정차
    # 보행자 트리거를 먼저 역산해야 점유 구간(트리거점 ~ 차폐물 뒤)을 한 번에 잡는다.
    ped_s0 = s + vlen / 2.0 + 2.0
    trg = ped_trigger(ctx, ped_s0, ped_walk_speed(v, 'ped_blind'), '우측', 'ped_blind')
    et = trg_cfg()
    ctx.claim(min(s - 40.0, ped_s0 - trg['trigger_d'] - float(et['claim_back_m'])),
              max(s + 40.0, ped_s0 + float(et['claim_fwd_m'])), 'ped_blind')
    bx, by, bz, bh = route_pt(lg, rt, s, t_block)
    name = ctx.next_name('Blocker')
    ctx.players.append(blk_vehicle_posabs(name, vtype, bx, by, bz, bh))
    ctx.actions.append(blk_stay_action(name, bx, by))
    # 보행자: 차폐물 앞머리 쪽에서 우→좌 횡단 (가려져 있다가 출현)
    ped_s = s + vlen / 2.0 + 2.0
    out = _add_ped(ctx, ped_s, trg, '우측', 'ego_lane')
    out.update(kind='ped_blind', blocker=vtype, blocker_s=round(s, 2))
    return out


def ev_lead_brake(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    trig_s = pick_s(ctx.spans, v['발동거리'])
    ctx.claim(trig_s - 50, trig_s + 60, 'lead_brake')
    ego_start = rt['start_s_in_lane']
    gap0 = 80.0
    v0 = 30.0 / 3.6
    name = ctx.next_name('LeadCar')
    sx, sy, _, _ = route_pt(lg, rt, min(gap0, rt['total_length'] * 0.3))
    tx, ty, _, _ = route_pt(lg, rt, trig_s)
    ctx.players.append(blk_vehicle_pathref(
        name, CAR, v0, ego_start + gap0, ego_start + rt['total_length'], _start_lane_id(rt)))
    acts = [_act('hold_speed', sx, sy, 20.0, name, _speed_change(3.0, v0)),
            _act('hard_brake', tx, ty, 10.0, 'Ego', _speed_change(v['감속강도'], 0.0))]
    resume = v.get('재출발', '없음') == '있음'
    if resume:
        acts.append(_act('resume', tx, ty, 10.0, 'Ego', _speed_change(2.0, 40.0 / 3.6, delay=15.0)))
    ctx.actions.append(blk_player_actions(name, acts))
    ctx.checks.append((tx, ty, 'ego_lane'))
    return {'kind': 'lead_brake', 'lead': name, 'trigger_s': round(trig_s, 2),
            'decel': v['감속강도'], 'resume': resume, 'gap0': gap0}


def ev_slow_lead(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    ego_start = rt['start_s_in_lane']
    gap = float(v.get('간격', 50))
    speed_kph = float(v.get('_속도_kph', 15.0))
    vt = speed_kph / 3.6
    name = ctx.next_name('SlowLead')
    sx, sy, _, _ = route_pt(lg, rt, min(gap, rt['total_length'] * 0.3))
    ctx.players.append(blk_vehicle_pathref(
        name, CAR, vt, ego_start + gap, ego_start + rt['total_length'], _start_lane_id(rt)))
    ctx.actions.append(blk_player_actions(
        name, [_act('hold_speed', sx, sy, 20.0, name, _speed_change(3.0, vt))]))
    return {'kind': 'slow_lead', 'lead': name, 'gap0': gap, 'speed_kph': speed_kph}


def _same_dir_lane_count(lg, k, sl: float, min_w: float) -> int:
    """차로 k 의 sl 지점에서 **동일 방향 주행차로** 수 (자기 포함).

    폭 임계를 두는 이유: 이 맵에는 폭 0 인 테이퍼 차로가 실재한다
    ((173,3,-1) w=0.0, 2026-09-02 확인). 개수만 세면 "이웃이 있다" 가 참이
    되지만 비켜 설 자리는 없다. left_nb 는 중앙선을 넘지 않지만
    (lanegraph.roadway_edges 주석) 반대 통행방향은 dir 로 한 번 더 막는다.
    """
    me = lg.lanes[k]
    n = 1
    for side in ('left', 'right'):
        cur = k
        for _ in range(6):
            nb = lg.neighbor(cur, side)
            if nb is None:
                break
            o = lg.lanes[nb]
            if o['type'] != 'driving' or o['dir'] != me['dir']:
                break
            if lg.width_at(nb, min(sl, o['length'])) < min_w:
                break
            n += 1
            cur = nb
    return n


def _multi_lane_spans(ctx, min_w: float, min_span: float, step: float = 2.0):
    """동일 방향 주행차로가 2개 이상인 route_s 부분구간들 (_opposite_spans 와 같은 형태).

    구간 양끝이 **둘 다 검사를 통과한 표본**이 되도록 자른다 — 표본 사이
    (step 미만)만 미검사로 남는다. step 이 2 m 인 것은 이 맵의 차로 섹션이
    짧기 때문이다 (실측 최단 1.6~1.8 m): 10 m 간격이면 차로가 끊기는 짧은
    섹션을 통째로 건너뛴다.
    """
    out = []
    for a, b in ctx.spans:
        lo = prev = None
        u = a
        while True:
            uu = min(u, b)
            _, k, sl = lane_at(ctx.route.rt, uu)
            if _same_dir_lane_count(ctx.lg, k, sl, min_w) >= 2:
                lo = uu if lo is None else lo
                prev = uu
            else:
                if lo is not None and prev - lo >= min_span:
                    out.append((lo, prev))
                lo = prev = None
            if uu >= b:
                break
            u += step
        if lo is not None and prev - lo >= min_span:
            out.append((lo, prev))
    return out


def ev_static_vehicle(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    cfg = plc_cfg()['static_vehicle']
    gate = bool(cfg['require_multi_lane'])
    min_w = float(cfg['min_neighbor_width_m'])
    if gate:
        # 자차로를 막는 정차 차량은 옆 차로가 있어야 회피가 성립한다 —
        # 근거는 params gen_placement.static_vehicle 주석.
        mspans = _multi_lane_spans(ctx, min_w, float(cfg['min_span_m']))
        if not mspans:
            raise EventUnfeasible('static_vehicle: 동일 방향 주행차로가 2개 이상인 '
                                  '가용구간이 없다 — 자차로를 막으면 회피가 원천 불가다')
        s = _place_in_one_span(ctx, v['위치'], reach=0.0, need=30.0,
                               claim=(-40.0, 40.0), what='static_vehicle',
                               spans=mspans)
        if s is None:
            raise EventUnfeasible('static_vehicle: 2차로 이상 구간이 전부 다른 이벤트에 막혔다')
    else:
        s = pick_s(ctx.spans, v['위치'], need=30.0)
        ctx.claim(s - 40, s + 40, 'static_vehicle')
    i, k, sl = lane_at(rt, s)
    t = 0.0
    lane_tag = 'ego_lane'
    if v.get('차선') == '좌측차로':
        nb = lg.neighbor(k, 'left')
        ok = (nb is not None and lg.lanes[nb]['dir'] == lg.lanes[k]['dir']
              and lg.lanes[nb]['type'] == 'driving')
        if ok:
            t = (lg.width_at(k, sl) + lg.width_at(nb, min(sl, lg.length(nb)))) / 2.0
            lane_tag = 'side_lane'
        elif gate:
            # 조용히 t=0 으로 떨어지면 '좌측차로' 요청이 **자차로 정중앙 차단**이
            # 된다 — 요청과 정반대 시나리오다. 2026-09-02 전수에서 1차로 완전차단
            # 24건 중 6건이 이 폴백이었다 (2차로 도로라도 ego 가 왼쪽 차로면
            # 좌측 이웃이 없다). 게이트가 켜져 있으면 버리고 백필한다.
            raise EventUnfeasible(f'static_vehicle: s={s:.0f} m 에 좌측 동방향 '
                                  f'주행차로가 없다 — 자차로 차단으로 떨어지는 것을 막는다')
    x, y, z, h = route_pt(lg, rt, s, t)
    name = ctx.next_name('StoppedCar')
    ctx.players.append(blk_vehicle_posabs(name, CAR, x, y, z, h))
    ctx.actions.append(blk_stay_action(name, x, y))
    ctx.checks.append((x, y, lane_tag))
    out = {'kind': 'static_vehicle', 'name': name, 'route_s': round(s, 2),
           'lane': v.get('차선', '자차로')}
    if v.get('대향차') == '있음':
        out['oncoming'] = ev_oncoming(ctx, {'위치': None, '개수': 2, '속도': 40, '_anchor': s})
    return out


def _has_opposite(ctx, s) -> bool:
    _, k, _ = lane_at(ctx.route.rt, s)
    return ctx.lg.opposite_of(k) is not None


def _opposite_spans(ctx, step=10.0):
    """반대 통행방향 차로가 존재하는 route_s 구간들 (양방향 도로 구간)."""
    out = []
    for a, b in ctx.spans:
        lo = None
        u = a
        while u <= b:
            ok = _has_opposite(ctx, u)
            if ok and lo is None:
                lo = u
            if (not ok or u + step > b) and lo is not None:
                hi = u if not ok else b
                if hi - lo >= 60.0:
                    out.append((lo, hi))
                lo = None
            u += step
    return out


def _oncoming_polyline(ctx, anchor, span, step=5.0):
    """anchor 를 중심으로 span 안에서 반대 차로 중심선 — 대향 진행 순서(먼 쪽→가까운 쪽)."""
    lg, rt = ctx.lg, ctx.route.rt
    s_hi = min(anchor + 180.0, span[1])
    s_lo = max(anchor - 120.0, span[0])
    pts = []
    u = s_hi
    while u >= s_lo:
        _, k, sl = lane_at(rt, u)
        opp = lg.opposite_of(k)
        if opp is None:
            if pts:
                break
            u -= step
            continue
        ex, ey, _, _ = lg.point_at(k, sl)
        so, _, _, _ = lg.project(opp, ex, ey)
        x, y, z, h = lg.point_at(opp, so)
        pts.append((x, y, z, h % (2 * math.pi)))
        u -= step
    return pts


def ev_oncoming(ctx, v):
    ospans = _opposite_spans(ctx)
    if not ospans:
        raise GenError('oncoming: 경로에 대향 차로가 있는 구간이 없다 (전부 일방/분리 구간)')
    anchor = v.get('_anchor')
    if anchor is None:
        anchor = pick_s(ospans, v['위치'] if v.get('위치') is not None else 0.5)
        ctx.claim(anchor - 30, anchor + 30, 'oncoming')
    else:
        # 다른 이벤트(정차 추월 등)에 붙는 대향차 — 가장 가까운 양방향 구간으로 스냅
        anchor = min((max(a, min(anchor, b)) for a, b in ospans),
                     key=lambda u: abs(u - anchor))
    span = next(((a, b) for a, b in ospans if a - 30 <= anchor <= b + 30), ospans[0])
    n = int(v.get('개수', 2))
    kph = float(v.get('속도', 40))
    pts = _oncoming_polyline(ctx, anchor, span)
    if len(pts) < 12:
        raise GenError(f'oncoming: 반대 차로 폴리라인이 너무 짧다 ({len(pts) * 5} m) — '
                       f'경로 s≈{anchor:.0f} 부근에 대향 차로가 없다')
    length = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                 for i in range(len(pts) - 1))
    sid = ctx.next_shape()
    ctx.moving.append(blk_pathshape(sid, f'Oncoming{sid}Path', pts))
    spacing = max(18.0, (length * 0.5) / max(n, 1))
    names = []
    for i in range(n):
        start = 0.01 + i * spacing
        if start >= length - 10.0:
            break
        name = ctx.next_name('Oncoming')
        names.append(name)
        ctx.players.append(blk_vehicle_pathshape(name, CAR, sid, start, length))
        # 자기 시작점 자기-트리거로 즉시 출발 (2_lead_brake 의 hold_speed 패턴)
        px, py = _poly_at(pts, start)
        ctx.actions.append(blk_player_actions(
            name, [_act('go', px, py, 25.0, name, _speed_change(4.0, kph / 3.6))]))
    ctx.checks.append((pts[len(pts) // 2][0], pts[len(pts) // 2][1], 'oncoming'))
    return {'kind': 'oncoming', 'count': len(names), 'speed_kph': kph,
            'anchor_s': round(anchor, 2), 'players': names}


def _poly_at(pts, dist):
    acc = 0.0
    for i in range(len(pts) - 1):
        d = math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
        if acc + d >= dist:
            u = (dist - acc) / max(d, 1e-9)
            return (pts[i][0] + u * (pts[i + 1][0] - pts[i][0]),
                    pts[i][1] + u * (pts[i + 1][1] - pts[i][1]))
        acc += d
    return pts[-1][0], pts[-1][1]


def ev_cut_in(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    s = pick_s(ctx.spans, v['위치'], need=80.0)
    ctx.claim(s - 50, s + 80, 'cut_in')
    _, k, sl = lane_at(rt, s)
    nb = lg.neighbor(k, 'left')
    side = '좌측'
    if nb is None or lg.lanes[nb]['dir'] != lg.lanes[k]['dir'] or lg.lanes[nb]['type'] != 'driving':
        nb = lg.neighbor(k, 'right')
        side = '우측'
        if nb is None or lg.lanes[nb]['dir'] != lg.lanes[k]['dir'] or lg.lanes[nb]['type'] != 'driving':
            raise GenError(f'cut_in: s≈{s:.0f} 에 같은 방향 이웃 차로가 없다')
    w = (lg.width_at(k, sl) + lg.width_at(nb, min(sl, lg.length(nb)))) / 2.0
    sign = 1.0 if side == '좌측' else -1.0
    pts = []
    for u in np.arange(s - 40.0, s + 70.0, 4.0):
        if u < 5.0 or u > rt['total_length'] - 5.0:
            continue
        if u <= s:
            t = sign * w
        elif u <= s + 25.0:
            t = sign * w * (1.0 - (u - s) / 25.0)
        else:
            t = 0.0
        x, y, z, h = route_pt(lg, rt, u, t)
        pts.append((x, y, z, h % (2 * math.pi)))
    length = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
                 for i in range(len(pts) - 1))
    sid = ctx.next_shape()
    name = ctx.next_name('CutIn')
    ctx.moving.append(blk_pathshape(sid, f'{name}Path', pts))
    ctx.players.append(blk_vehicle_pathshape(name, CAR, sid, 0.01, length))
    ctx.actions.append(blk_player_actions(
        name, [_act('go', pts[0][0], pts[0][1], float(v.get('트리거거리', 30)), 'Ego',
                    _speed_change(4.0, 30.0 / 3.6))]))
    ctx.checks.append((pts[-1][0], pts[-1][1], 'ego_lane'))
    return {'kind': 'cut_in', 'name': name, 'route_s': round(s, 2), 'side': side}


def ev_signal(ctx, v):
    appr = route_signal_approaches(ctx.lg, ctx.route.rt, ctx.ctrl_map)
    if not appr:
        raise GenError('경로에 신호화된 접근이 없다 — signal 이벤트 불가 '
                       '(다른 경로를 쓰거나 junction_ctrl_map 을 확인)')
    timing = v.get('신호', '적60')
    out = []
    for jid, stop_s, ctrls in appr:
        t_arr = stop_s / AVG_SPEED_MPS + 2.0
        if timing == '적60':
            go = 8.0 if t_arr >= 12.0 else max(2.0, t_arr - 6.0)
            phases = (go, 3.0, 60.0)
        elif timing == '딜레마':
            phases = (max(3.0, t_arr - 1.5), 3.0, 60.0)
        elif timing == '짧은녹색':
            phases = (5.0, 3.0, 25.0)
        else:
            raise GenError(f'signal: 모르는 신호 값 "{timing}"')
        for c in ctrls:
            ctx.signals[c] = phases
        out.append({'junction': jid, 'controllers': ctrls,
                    'stop_s': round(stop_s, 2), 'phases': list(phases)})
    return {'kind': 'signal', 'timing': timing, 'approaches': out}


def _place_in_one_span(ctx, frac, reach, need, claim, what, spans=None):
    """s 부터 s+reach 까지가 **한 가용구간 안에** 들어가는 시작 route_s. 없으면 None.

    왜 한 구간이어야 하는가 (2026-09-02 실측): pick_s 는 구간들을 이어붙인
    길이축에서 비율 지점을 고를 뿐인데, 여러 물체를 놓는 이벤트는 그 지점부터
    raw route_s 로 뻗어 나간다. 그래서 뒷물체가 구간 밖 — 교차로 안으로 걸어
    나갔다. obstacle_chain 표본 144 에서 개수 6 → 30% / 4 → 14~18% / 3 → 5%
    가 교차로 내부에 장애물을 놓았고 (예: 좌회전 542 m 경로
    spans=[(131,193),(259,355),(463,502)] 에서 s=219.7 이 junction 54 안),
    narrow 도 같은 결함이 있었다 (실전주행_01_연속교차로24 s=1713.0).
    교차로 안 장애물은 회피 연습이 아니라 채점 왜곡이다.

    요청 위치(frac)가 들어가는 구간을 먼저 보고, 거기서 꼬리가 넘치면 구간
    안으로 당긴다. 그 구간이 겹침으로 막히면 요청 위치에서 가까운 구간 순으로
    옮겨 본다 — 개수를 줄이거나 이벤트를 버리는 것보다 위치를 옮기는 쪽이 먼저다.

    reach: 시작점에서 마지막 물체까지 [m]. need: pick_s 예약 길이 (종전 값).
    claim: (lo, hi) — s 기준 점유 구간 오프셋. need·claim 을 종전 값 그대로
    받는 이유는, 이미 한 구간에 들어가던 배치는 시작점도 점유 구간도 바뀌지
    않아야 하기 때문이다 (기존 산출물 불변 조건).
    spans: 후보 구간을 ctx.spans 가 아닌 다른 목록으로 좁힌다 (기본 ctx.spans).
    이벤트별 성립 조건으로 미리 걸러 낸 부분구간을 넘기는 용도다 — 예:
    static_vehicle 의 '동일 방향 주행차로 2개 이상' (_multi_lane_spans).
    """
    spans = ctx.spans if spans is None else spans
    try:
        want = pick_s(spans, frac, need=need)
    except GenError:
        want = None                     # 가용축 전체가 짧다 — 구간만 보고 고른다
    cands = [(a, b) for a, b in spans if b - a >= reach]
    if want is not None:                # 요청 위치를 품은 구간 → 가까운 구간 순
        cands.sort(key=lambda ab: 0.0 if ab[0] <= want <= ab[1]
                   else min(abs(ab[0] - want), abs(ab[1] - want)))
    for a, b in cands:
        lo, hi = a, b - reach           # 배치 전체가 구간 안에 남는 시작점 범위
        target = lo if want is None else min(max(want, lo), hi)
        s = _nearest_free(lo, hi, target, ctx.occupied, claim)
        if s is None:
            continue                    # 이 구간은 다른 이벤트가 통째로 막았다
        ctx.claim(s + claim[0], s + claim[1], what)
        return s
    return None


def _nearest_free(lo, hi, target, occupied, claim):
    """[lo,hi] 안에서 claim 이 기존 점유와 안 겹치는, target 에 가장 가까운 s.

    구간을 옮기기 전에 **같은 구간 안에서 먼저 민다** — 옮기면 요청 위치에서
    멀어진다. 2026-09-02 실측: 밀기 없이 바로 옆 구간으로 보내면
    정적회피집중_02_우회전 의 narrow 가 186.9 → 296.8 로 110 m 튀었다.
    수정 전에도 186.9 는 구간 안이었으므로 그건 개선이 아니라 회귀다.

    claim 은 (lo,hi) 오프셋이고 Ctx.claim 이 열린 겹침(s0 < b and a < s1)을
    쓰므로, 점유 (o0,o1) 를 피하는 금지 구간은 열린 (o0-claim[1], o1-claim[0])
    이다 — 경계에 딱 붙는 배치는 허용된다.
    """
    free = [(lo, hi)]
    for o0, o1 in occupied:
        bad0, bad1 = o0 - claim[1], o1 - claim[0]
        nxt = []
        for f0, f1 in free:
            if bad1 <= f0 or bad0 >= f1:
                nxt.append((f0, f1))
                continue
            if f0 < bad0:
                nxt.append((f0, min(bad0, f1)))
            if f1 > bad1:
                nxt.append((max(bad1, f0), f1))
        free = [iv for iv in nxt if iv[1] >= iv[0]]
    best = None
    for f0, f1 in free:
        s = min(max(target, f0), f1)
        if best is None or abs(s - target) < abs(best - target):
            best = s
    return best


def ev_obstacle_chain(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    cfg = plc_cfg()
    spans = None
    if bool(cfg['obstacle_chain']['require_multi_lane']):
        # 1차로 구간의 체인은 슬라럼이 아니라 차단이다 — 실측(2026-09-02,
        # logs/batch/20260902_231132 정적회피집중_01_좌회전0): 1차로 (2818,6,-1)
        # 에서 체인이 자차로를 직접 막아 충돌 1 + blocked 로 끝났다.
        # 폭·구간 임계는 static_vehicle 과 같은 물리량이라 그쪽 값을 읽는다
        # (값을 두 곳에 적지 않는다). 체인 전체(reach)가 2차로 부분구간 안에
        # 들어가야 하므로 _place_in_one_span 의 spans 인자로 좁힌다.
        sv = cfg['static_vehicle']
        spans = _multi_lane_spans(ctx, float(sv['min_neighbor_width_m']),
                                  float(sv['min_span_m']))
        if not spans:
            raise EventUnfeasible('obstacle_chain: 동일 방향 주행차로가 2개 이상인 '
                                  '가용구간이 없다 — 1차로의 체인은 회피가 원천 불가다')
    n0 = n = int(v.get('개수', 4))
    spacing = 18.0
    s0 = None
    while n >= 2:
        s0 = _place_in_one_span(ctx, v['위치'], reach=(n - 1) * spacing,
                                need=n * spacing + 20.0,
                                claim=(-20.0, n * spacing + 20.0),
                                what='obstacle_chain', spans=spans)
        if s0 is not None:
            break
        n -= 1                          # 어느 구간에도 안 들어간다 — 개수를 줄인다
    if s0 is None:
        raise GenError('obstacle_chain: 장애물 2개도 놓을 구간이 부족하다')
    shrunk = n != n0
    placed = []
    for i in range(n):
        s = s0 + i * spacing
        _, k, sl = lane_at(rt, s)
        w = lg.width_at(k, sl)
        t = (w / 2.0 - 0.7) * (1 if i % 2 == 0 else -1)
        x, y, z, _ = route_pt(lg, rt, s, t)
        ctx.moving.append(blk_object(ctx.next_name('Obstacle'), x, y, z))
        ctx.checks.append((x, y, 'ego_lane'))
        placed.append(round(s, 2))
    out = {'kind': 'obstacle_chain', 'count': n, 'route_s': placed}
    if shrunk:
        out['note'] = f'구간 부족으로 개수 축소 ({v.get("개수", 4)}→{n})'
    return out


# 양측 정차 차량의 (종거리 오프셋, 좌우 부호) — 우측 먼저, 14 m 뒤 좌측.
# 배치와 "한 구간 안에 들어가는가" 판정이 같은 값을 봐야 해서 상수로 뺐다.
NARROW_OFFSETS = ((0.0, -1.0), (14.0, 1.0))


def ev_narrow(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    # 두 대가 한 가용구간 안에 들어가야 한다 — 뒤차가 교차로로 넘어가면
    # 협착이 아니라 교차로 안 정차 차량이 된다 (2026-09-02: 실전주행_01_
    # 연속교차로24 의 narrow s=1713.0 이 구간 밖이었다).
    reach = max(ds for ds, _ in NARROW_OFFSETS) - min(ds for ds, _ in NARROW_OFFSETS)
    s = _place_in_one_span(ctx, v['위치'], reach=reach, need=40.0,
                           claim=(-30.0, 50.0), what='narrow')
    if s is None:
        raise GenError(f'narrow: 정차 차량 두 대({reach:.0f} m)가 통째로 들어가는 '
                       f'가용구간이 없다')
    intr = float(v.get('침범폭', 0.7))
    placed = []
    for ds, sgn in NARROW_OFFSETS:
        _, k, sl = lane_at(rt, s + ds)
        w = lg.width_at(k, sl)
        t = sgn * (w / 2.0 + CAR_W / 2.0 - intr)
        x, y, z, h = route_pt(lg, rt, s + ds, t)
        name = ctx.next_name('Parked')
        ctx.players.append(blk_vehicle_posabs(name, CAR, x, y, z, h))
        ctx.actions.append(blk_stay_action(name, x, y))
        placed.append({'name': name, 'route_s': round(s + ds, 2), 'side': '우측' if sgn < 0 else '좌측'})
        ctx.checks.append((x, y, 'narrow'))
    return {'kind': 'narrow', 'intrusion_m': intr, 'vehicles': placed}


def ev_none(ctx, v):
    return {'kind': 'none'}


EVENTS = {
    'pedestrian': ev_pedestrian, 'jaywalk': ev_jaywalk, 'ped_blind': ev_ped_blind,
    'lead_brake': ev_lead_brake, 'slow_lead': ev_slow_lead,
    'static_vehicle': ev_static_vehicle, 'cut_in': ev_cut_in, 'oncoming': ev_oncoming,
    'signal': ev_signal, 'obstacle_chain': ev_obstacle_chain, 'narrow': ev_narrow,
    'none': ev_none,
}


def _start_lane_id(rt) -> int:
    return rt['lanes'][0][2]


# ════════════════════════════════════════════════════════════════════════
# 5. 시나리오 조립·검증·저장
# ════════════════════════════════════════════════════════════════════════

def path_waypoints(lg, rt):
    """경로 차로 체인 → Path01 Waypoint (track, road_s) 목록.

    첫 점은 시작 차로의 s=0 (ego StartS 의 기준점), 마지막은 종점 차로 끝.
    같은 도로가 연달아 나오면 대표점 하나만 남긴다.
    """
    lanes = rt['lanes']
    groups = []
    for i, k in enumerate(lanes):
        if groups and groups[-1][0] == k[0]:
            groups[-1][1].append(i)
        else:
            groups.append((k[0], [i]))
    # 실기 로드 OK 인 검증 시나리오(2_lead/4_rtor/ped)의 waypoint 는 전부
    # junction=-1 일반 도로다. 교차로 연결로를 Track 으로 지정하면 VTD 가
    # "Path contains errors" 로 로드를 거부한다(실측: 1241/766 포함 경로).
    # 교차로는 PathOption="shortest" 가 스스로 잇는다 — 4_rtor 는 두 점
    # (128→1602)만으로 교차로 여러 개를 건너 로드된다.
    out = []
    for gi, (road, idxs) in enumerate(groups):
        if lg.roads[road]['junction'] != -1:
            continue
        if gi == 0:
            k = lanes[idxs[0]]
            s = lg.road_s_at(k, min(0.5, lg.length(k) * 0.25))
        elif gi == len(groups) - 1:
            k = lanes[idxs[-1]]
            L = lg.length(k)
            s = lg.road_s_at(k, max(L - 0.5, L * 0.75))
        else:
            # 그룹 중간 "인덱스" 차로는 차선수 전환 슬리버(4.7 m)일 수 있어
            # 대표점이 도로 끝 쪽에 쏠린다(실측 173@423.95) — 가장 긴 차로의
            # 중점을 쓴다 (검증 시나리오처럼 도로 몸통 안쪽 점).
            k = max((lanes[i] for i in idxs), key=lg.length)
            s = lg.road_s_at(k, lg.length(k) * 0.5)
        # 모든 waypoint 를 도로 길이 기준 [0.5, len-0.5] 로 클램프.
        # XML 은 4자리 반올림으로 적히므로 상한은 내림해 반올림 후에도 범위 안이게.
        L_road = lg.roads[road]['length']
        s = min(max(s, 0.5), math.floor((L_road - 0.5) * 1e4) / 1e4)
        out.append((road, round(s, 4)))
    # 전수 검증 (무음 실패 금지): 첫/끝 도로 유지 + 범위 + 비교차로
    if len(out) < 2:
        raise GenError(f'Path01 waypoint 가 {len(out)}개뿐이다 — 경로가 교차로 연결로 위주다')
    if out[0][0] != groups[0][0] or out[-1][0] != groups[-1][0]:
        raise GenError('Path01 첫/끝 waypoint 도로가 경로 시작/끝 도로와 다르다 '
                       '(시작 또는 종점이 교차로 연결로 위인 경로)')
    for road, s in out:
        L_road = lg.roads[road]['length']
        if lg.roads[road]['junction'] != -1 or not (0.5 - 1e-6 <= s <= L_road - 0.5 + 1e-6):
            raise GenError(f'Path01 waypoint 검증 실패: Track {road} s={s:.3f} '
                           f'(len={L_road:.3f}, junction={lg.roads[road]["junction"]})')
    return out


def lateral_check(lg, rt, checks) -> list:
    """ego_lane 태그 좌표의 경로 횡거리. 초과분 [(tag, x, y, dist)] 반환."""
    bad = []
    for x, y, tag in checks:
        if tag != 'ego_lane':
            continue
        dmin = min(lg.project(k, x, y)[2] for k in rt['lanes'])
        if dmin > LAT_WARN_M:
            bad.append((tag, x, y, dmin))
    return bad


def route_summary(lg, route: Route) -> dict:
    """시작점·경로 요약 — 리포트 출력과 커버리지 누적(지나간 도로 id)용."""
    rt = route.rt
    lanes = rt['lanes']
    roads, juncs = [], []
    for k in lanes:
        if not roads or roads[-1] != k[0]:
            roads.append(k[0])
        j = lg.lanes[k]['junction']
        if j != -1 and (not juncs or juncs[-1] != j):
            juncs.append(j)
    turns = [e['kind'] for e in rt['events'] if e['kind'].startswith('turn_')]
    x0, y0 = route.rows[0][1], route.rows[0][2]
    return {'start': {'road': lanes[0][0], 'lane': list(lanes[0]),
                      'x': round(x0, 3), 'y': round(y0, 3)},
            'roads': roads, 'junctions': juncs,
            'turns': {'left': turns.count('turn_left'), 'right': turns.count('turn_right')}}


def est_seconds(route_len: float) -> float:
    return route_len / AVG_SPEED_MPS + OVERHEAD_S


def timeout_for(route_len: float) -> int:
    return max(180, int(route_len / AVG_SPEED_MPS * 1.8 + 90))


# ── VTD 네이티브 교통류 (PulkTraffic) ────────────────────────────────────

# PulkDef 속성 순서 — 실기 검증된 Scenario Editor 2025.2 산출물과 같은 순서로
# 낸다. (params 키, 서식) 쌍이라 상수는 전부 params pulk.* 다.
_PULK_ATTRS = (
    ('SemiMajorAxis', 'semi_major_m', 'len'), ('SemiMinorAxis', 'semi_minor_m', 'len'),
    ('InnerRadius', 'inner_radius_m', 'len'), ('CenterOffset', 'center_offset_m', 'len'),
    ('AreaF', 'area_f', 'ratio'), ('AreaB', 'area_b', 'ratio'),
    ('AreaL', 'area_l', 'ratio'), ('AreaR', 'area_r', 'ratio'),
    ('OwnSide', 'own_side', 'ratio'),
    ('Cars', 'cars', 'ratio'), ('Vans', 'vans', 'ratio'), ('Buses', 'buses', 'ratio'),
    ('Trucks', 'trucks', 'ratio'), ('Bikes', 'bikes', 'ratio'),
)
_PULK_SUMS = (('Area(F/B/L/R)', ('area_f', 'area_b', 'area_l', 'area_r')),
              ('Vehicle Classes', ('cars', 'vans', 'buses', 'trucks', 'bikes')))


def pulk_def(axes: dict) -> dict:
    """축 값 + params pulk.* → PulkDef 속성 dict. 이벤트가 아니라 시나리오 전역이다.

    왜 이벤트가 아닌가: PulkTraffic 은 경로의 한 구간이 아니라 **ego 를 따라다니는
    영역**이다. 구간 배타 모델(ctx.claim)에 넣으면 다른 이벤트를 전부 밀어내고,
    event 목록에 넣으면 배치 실패·슬롯 분배 같은 구간 이벤트 논리를 타게 된다.

    축 연동 (시드 → 값): '교통류대수' → Count, '교통류밀도' → 영역 크기 프리셋
    (semi_major/minor + inner_radius). 축이 없으면 params 기본값. 밀도 프리셋은
    비율을 건드리지 않으므로 축이 어떻게 뽑혀도 아래 두 합계는 유지된다.

    두 합계(Area 4개, 차종 5개)를 여기서 검증하고 1.0 에서 벗어나면 GenError 다.
    **자동 보정하지 않는다** — 조용히 고치면 시나리오마다 다른 값이 들어가고,
    에디터가 나중에 같은 검사로 거부할 때 원인을 되짚을 수 없다.
    """
    pc = pulk_cfg()
    vals = {k: pc[k] for _, k, _ in _PULK_ATTRS}
    dens = axes.get('교통류밀도')
    if dens is not None:
        presets = pc['density']
        if dens not in presets:
            raise GenError(f'pulk: 모르는 교통류밀도 "{dens}" — '
                           f'params pulk.density 에 {", ".join(presets)} 만 있다')
        vals.update(presets[dens])
    for what, keys in _PULK_SUMS:
        tot = sum(float(vals[k]) for k in keys)
        if abs(tot - 1.0) > 1e-6:
            raise GenError(f'pulk: {what} 합이 {tot:.4f} 다 (1.0 이어야 한다) — '
                           f'params pulk.{"/".join(keys)} 를 고칠 것')
    count = int(axes.get('교통류대수') or pc['count'])
    if count < 1:
        raise GenError(f'pulk: Count 가 {count} 다 (1 이상이어야 한다)')
    out = {'CentralPlayer': 'Ego', 'Count': count, 'FillAtStart': 'true'}
    for attr, key, kind in _PULK_ATTRS:
        # 서식 고정 = 같은 축 값이면 바이트 동일. len 은 정수형(300), 비율은 0.40.
        out[attr] = f'{float(vals[key]):g}' if kind == 'len' else f'{float(vals[key]):.2f}'
    out['VisibleInArea'] = '-1'
    return out


def pulk_attrs(d: dict) -> str:
    return ' '.join(f'{k}="{v}"' for k, v in d.items())


def build_scenario(lg, ctrl_map, route: Route, events: list, axes: dict,
                   name: str, seed_key: str, min_keep: int | None = None,
                   pulk: dict | None = None):
    """이벤트 목록 → (xml_text, def_dict, warnings).

    min_keep 이 주어지면(실전주행 scale_events) 개별 이벤트의 배치 실패
    (겹침/공간 부족이 fracs 재시도로도 해소 안 됨)는 그 이벤트만 버리고
    계속한다 — 단 최종 배치 수가 min_keep 미만이면 시나리오째 폐기(GenError)
    → 호출자가 다른 경로로 백필한다.

    pulk 는 pulk_def 결과 — 이벤트가 아니라 시나리오 전역 속성이라 events 와
    별도 인자로 받아 <PulkTraffic> 에 그대로 쓴다."""
    if pulk is not None and any(ev == 'narrow' for ev, _v in events) \
            and bool(plc_cfg().get('narrow_pulk_guard_enable', True)):
        # narrow 는 pulk 교통류를 막는다 — internal-driver 차량이 협착을 못 지나
        # 그 앞에 정체를 만들고 ego 가 갇힌다 (실측 2026-09-02: 실전주행_교통류_01_
        # 좌회전24 · 정적회피집중_02_우회전, 근거는 params gen_placement 주석).
        # 테마 정의(themes.yaml)와 저장된 정의(--from-yaml) 양쪽을 여기 한 곳에서
        # 막는다. narrow 단독 검증은 pulk 없는 차로폭협착 테마가 담당한다.
        raise GenError(f'{name}: narrow 는 pulk 교통류를 막는다 — pulk 가 켜진 '
                       f'시나리오에는 배치하지 않는다 (gen_placement.narrow_pulk_guard_enable)')
    ctx = Ctx(lg, route, random.Random(seed_key), ctrl_map)
    resolved = []
    dropped = []
    for ev, v in events:
        if ev not in EVENTS:
            raise GenError(f'{name}: 모르는 이벤트 "{ev}"')
        # 구간 겹침/부족은 위치를 옮겨가며 재시도한다 (다중이벤트 대비).
        # 이벤트가 블록을 일부 추가하고 실패할 수 있으므로 스냅숏 후 복원한다.
        fracs = [v.get('위치', 0.5)] + list(v.get('_retry_fracs')
                                            or [0.15, 0.35, 0.55, 0.75, 0.9])
        last = None
        for attempt, frac in enumerate(fracs):
            snap = (len(ctx.players), len(ctx.actions), len(ctx.moving),
                    dict(ctx.signals), list(ctx.checks), list(ctx.occupied),
                    ctx.shape_seq, ctx.player_seq)
            vv = dict(v) if attempt == 0 else dict(v, 위치=frac, 발동거리=frac)
            try:
                resolved.append(EVENTS[ev](ctx, vv))
                last = None
                break
            except GenError as e:
                msg = str(e)
                (np_, na, nm, ctx.signals, ctx.checks, ctx.occupied,
                 ctx.shape_seq, ctx.player_seq) = snap
                del ctx.players[np_:], ctx.actions[na:], ctx.moving[nm:]
                if not isinstance(e, EventUnfeasible) and \
                        '겹친다' not in msg and '부족하다' not in msg:
                    raise
                last = e
        if last is not None:
            if min_keep is not None:
                dropped.append(ev)          # 이 이벤트만 포기하고 계속 (min_keep 가드)
                continue
            raise last
    if min_keep is not None and len(resolved) < min_keep:
        raise GenError(f'{name}: 이벤트 배치 {len(resolved)}/{len(events)}건 '
                       f'(최소 {min_keep}) — 공간 부족, 시나리오 폐기')
    doc = XmlDoc(TEMPLATE)
    rt = route.rt
    doc.set_path(path_waypoints(lg, rt))
    doc.set_ego(rt['start_s_in_lane'], rt['start_s_in_lane'] + rt['total_length'],
                _start_lane_id(rt))
    if pulk:
        doc.set_pulk(pulk_attrs(pulk))
    doc.add_players(ctx.players)
    doc.add_player_actions(ctx.actions)
    doc.add_moving(ctx.moving)
    for cid, (go, att, stop) in sorted(ctx.signals.items()):
        doc.set_signal(cid, go, att, stop)
    xml_text = doc.final(name)
    bad = lateral_check(lg, rt, ctx.checks)
    d = {'name': name, 'theme': None, 'seed_key': seed_key,
         'route': {'name': route.name,
                   # CSV(대회형식)는 3자리지만 재생성 정밀도를 위해 6자리로 저장
                   'rows': [[r[0], round(r[1], 6), round(r[2], 6)] for r in route.rows],
                   'length_m': round(rt['total_length'], 1),
                   # 커버리지 누적용: 시작점 + 지나간 도로/교차로/회전 요약
                   **route_summary(lg, route)},
         'axes': axes, 'events': resolved,
         **({'pulk': dict(pulk)} if pulk else {}),
         **({'events_planned': len(events), 'events_dropped': dropped}
            if min_keep is not None else {}),
         'est_s': round(est_seconds(rt['total_length']), 1),
         'timeout_s': timeout_for(rt['total_length'])}
    return xml_text, d, bad


# ════════════════════════════════════════════════════════════════════════
# 6. 주제 전개 (조합 → 개수/시간 예산)
# ════════════════════════════════════════════════════════════════════════

def axis_pool(theme_cfg: dict, axis: str):
    return list(theme_cfg.get(axis, AXIS_DEFAULTS.get(axis, [None])))


def expand_theme(theme: str, cfg: dict, seed: int, route_defs: dict | None = None) -> list:
    """주제 → 변형 목록 [{'route': (이름, 변형), 'event': [...], axes…}] (샘플링 전 전체)."""
    rng = random.Random(f'{seed}:{theme}')
    routes = list(cfg.get('routes', ['기본']))
    vary = list(cfg.get('vary', []))
    events = list(cfg.get('event', ['none']))
    # start: 자유 → walk 경로 시드에 주제 이름을 섞어 주제·변형마다 다른 시작점
    salt = theme if cfg.get('start', '고정') == '자유' else ''
    # 자유 모드에서 csv 경로(시작점 파일 고정)는 커버리지에 기여하지 못하고
    # 다양성만 깎는다 — walk 경로가 하나라도 있으면 제외한다 (routes 가 csv 뿐이면
    # 유지: 자유 지정이 무의미할 뿐 생성은 되어야 한다). csv 검증 회귀는
    # start: 고정 주제·단독 실행으로 커버한다 (2026-08-28 커버리지 분석).
    if salt and route_defs:
        walk_routes = [r for r in routes if 'walk' in (route_defs.get(r) or {})]
        if walk_routes:
            routes = walk_routes
    pools = {ax: axis_pool(cfg, ax) for ax in vary}
    combos = []
    if cfg.get('random_axes'):
        for i in range(MAX_PER_THEME):
            v = {ax: rng.choice(pools[ax]) for ax in vary}
            # 자유 시작이면 뽑기마다 다른 시작점이 되도록 변형 번호도 흔든다
            rv = v.get('경로변형', rng.randrange(1, MAX_PER_THEME + 1) if salt else 1)
            v['route'] = (rng.choice(routes), rv, salt)
            v['event'] = [rng.choice(events)]
            combos.append(v)
        return combos
    axis_names = [a for a in vary if a != '경로변형']
    rv_pool = pools.get('경로변형', [1]) if '경로변형' in vary else [1]
    ev_sets: list
    if cfg.get('combine'):
        ev_sets = [list(c) for c in itertools.combinations(events, int(cfg['combine']))]
    else:
        ev_sets = [[e] for e in events]
    prod = itertools.product(routes, rv_pool, range(len(ev_sets)),
                             *[pools[a] for a in axis_names])
    for route_name, rv, ei, *vals in prod:
        v = dict(zip(axis_names, vals))
        v['route'] = (route_name, rv, salt)
        v['event'] = ev_sets[ei]
        combos.append(v)
    if cfg.get('rotate'):
        # 이벤트 축을 조합에서 빼고, 시나리오 순번에 따라 순환 배정
        base = [c for c in combos if c['event'] == ev_sets[0]]
        for i, c in enumerate(base):
            c['event'] = [events[i % len(events)]]
        combos = base
    rng.shuffle(combos)
    if salt:
        # 캐시 고정 해제 (2026-08-28 커버리지 분석): 자유 모드는 경로변형 축
        # 유무와 무관하게 시나리오 순번(0..N-1)을 변형 번호로 써서 RoutePool
        # 캐시 키가 매번 달라지게 한다 — 구현 전에는 축이 없으면 rv=1 고정이라
        # 주제당 walk 시작점이 1개로 수렴했다. random_axes 는 기존 동작 유지.
        for i, c in enumerate(combos):
            c['route'] = (c['route'][0], i, c['route'][2])
    return combos


def allocate(themes: dict, combos: dict, count, hours, route_len, seed):
    """주제별 선택 목록 확정. combos[theme] 는 이미 시드 셔플돼 있다."""
    chosen = {}
    if count is not None:
        for th, lst in combos.items():
            if len(lst) < count and lst and lst[0]['route'][2]:
                # start: 자유 — 변형 번호가 시나리오 순번이라 축 조합을 순환하며
                # 새 순번을 붙이면 얼마든지 늘릴 수 있다 (다른 시작점의 walk).
                base = list(lst)
                i = len(lst)
                while len(lst) < count:
                    c = dict(base[i % len(base)])
                    c['route'] = (c['route'][0], i, c['route'][2])
                    lst.append(c)
                    i += 1
            if len(lst) < count:
                print(f'  [주의] {th}: 조합이 {len(lst)}개뿐이라 --count {count} 를 다 못 채운다')
            chosen[th] = lst[:count]
        return chosen
    if hours is not None:
        budget = hours * 3600.0
        idx = {th: 0 for th in combos}
        chosen = {th: [] for th in combos}
        progressed = True
        while budget > 0 and progressed:
            progressed = False
            for th, lst in combos.items():
                if idx[th] >= len(lst):
                    continue
                c = lst[idx[th]]
                try:
                    cost = est_seconds(route_len(c['route']))
                except (GenError, RouteError):
                    idx[th] += 1        # 경로가 검증 탈락 — 이 변형은 배분에서 제외
                    progressed = True
                    continue
                if cost > budget:
                    continue
                chosen[th].append(c)
                idx[th] += 1
                budget -= cost
                progressed = True
        return chosen
    for th, lst in combos.items():
        chosen[th] = lst[:MAX_PER_THEME]
    return chosen


# ════════════════════════════════════════════════════════════════════════
# 7. main
# ════════════════════════════════════════════════════════════════════════

def load_themes():
    if not THEMES_YAML.exists():
        raise GenError(f'{THEMES_YAML} 이 없다')
    data = yaml.safe_load(THEMES_YAML.read_text(encoding='utf-8'))
    gen_cfg = data.get('gen') or {}
    for k in ('spawn_heading_max_deg', 'reverse_ratio_max', 'max_polyline_step_m'):
        if k not in gen_cfg:
            raise GenError(f'{THEMES_YAML} 에 gen.{k} 가 없다 — 게이트 임계는 '
                           f'설정 파일이 단일 출처다 (하드코딩 금지)')
    return data.get('routes', {}), data.get('themes', {}), gen_cfg


def batch_item(vtd_dir: str, out_dir_name: str, theme: str, name: str, timeout_s) -> dict:
    """batch 목록 항목 — batch_run._load_one 이 읽는 스키마 그대로 (변경 금지)."""
    return {'name': name,
            'vtd_xml_path': f'{vtd_dir.rstrip("/")}/{theme}/{name}.xml',
            'route_csv': f'{out_dir_name}/{theme}/{name}.csv',
            'timeout_s': timeout_s}


def rebuild_batch_lists(out_dir: pathlib.Path, vtd_dir: str) -> tuple[int, int]:
    """디스크의 <주제>/*.yaml 을 단일 출처로 batch_<주제>.json / batch_all.json 재생성.

    "이번 호출분" 메모리로 쓰던 예전 방식은 호출마다 목록을 덮어써서, 주제를
    차례로 생성하면 batch_all.json 에 마지막 주제만 남았다 (2026-08-27 실측:
    4개 주제 생성 후 보행자집중 6개만 잔존, 완주속도는 두 번째 시드분 9개만).
    주제 순서 = 디렉터리 생성 시각 오름차순(처음 생성된 순서), 주제 안 =
    파일명(번호) 순. 이름 중복이면 아무 목록도 쓰지 않고 실패한다.
    → (전체 항목 수, 주제 수)
    """
    per_theme: list[tuple[str, list]] = []
    for d in sorted((p for p in out_dir.iterdir() if p.is_dir()),
                    key=lambda p: p.stat().st_ctime):
        items = []
        for y in sorted(d.glob('*.yaml')):
            sdef = yaml.safe_load(y.read_text(encoding='utf-8'))
            items.append(batch_item(vtd_dir, out_dir.name, d.name,
                                    sdef['name'], sdef['timeout_s']))
        if items:
            per_theme.append((d.name, items))
    all_items = [it for _th, items in per_theme for it in items]
    names = [it['name'] for it in all_items]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:                              # 통합 기준 중복 검사 — batch_run 과 같은 규칙
        raise GenError(f'통합 batch 목록에서 이름이 중복된다: {dup}')
    for th, items in per_theme:
        (out_dir / f'batch_{th}.json').write_text(
            json.dumps(items, ensure_ascii=False, indent=1), encoding='utf-8')
    (out_dir / 'batch_all.json').write_text(
        json.dumps(all_items, ensure_ascii=False, indent=1), encoding='utf-8')
    return len(all_items), len(per_theme)


def write_scenario(out_dir, theme, name, xml_text, sdef, rows):
    d = out_dir / theme
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{name}.xml').write_text(xml_text, encoding='utf-8')
    csv = 'seq,x,y\n' + ''.join(f'{r[0]},{r[1]:.3f},{r[2]:.3f}\n' for r in rows)
    (d / f'{name}.csv').write_text(csv, encoding='utf-8')
    sdef = dict(sdef, theme=theme)
    (d / f'{name}.yaml').write_text(
        '# tools/gen_scenarios.py --from-yaml <이 파일> 로 단건 재생성\n'
        + yaml.safe_dump(sdef, allow_unicode=True, sort_keys=False), encoding='utf-8')


def gen_one(lg, ctrl_map, pool, theme, cfg, variant, name, seed):
    ml = cfg.get('min_length_m')
    route = pool.get(*variant['route'],
                     min_length_m=None if ml is None else float(ml))
    events = list(variant['event'])
    scale = bool(cfg.get('scale_events'))
    min_keep = None
    if scale:
        # 실전주행: 이벤트 개수 = 경로 길이 비례 (params gen_events.per_route_m),
        # 상한 = 가용 구간 길이 / 최소 간격 (usable_spans 재사용 — 새 배치기 없음).
        # 종류는 시나리오 시드로 무작위(중복 허용) — from-yaml 재생성도 같은
        # 시드라 같은 목록이 나온다.
        ge = ev_cfg()
        flat = sum(b - a for a, b in usable_spans(lg, route.rt))
        n = max(1, round(route.rt['total_length'] / float(ge['per_route_m'])))
        n = min(n, max(1, int(flat // float(ge['min_gap_m']))))
        rng_e = random.Random(f'{seed}:{theme}:{name}:events')
        # 동행형(slow_lead/lead_brake — ego 기준 발동이라 위치 슬롯으로 분산되지
        # 않는다)과 전역형(signal — 신호 타이밍 설정이라 중복이 덮어쓰기일 뿐)은
        # 경로당 1회로 제한 — 실측(2026-08-30): 무제한이면 한 경로에 signal×3,
        # 선행차 5대가 나와 원인 분리가 안 된다. 앵커형은 중복 무방(슬롯 분산).
        unique_evs = {'slow_lead', 'lead_brake', 'signal'}
        pool_all = list(cfg['event'])
        events = []
        for _ in range(int(n)):
            avail = [e for e in pool_all
                     if e not in unique_evs or e not in events]
            events.append(rng_e.choice(avail))
        min_keep = max(1, math.ceil(n * float(ge['min_keep_ratio'])))
    ev_list = []
    n_ev = len(events)
    for j, ev in enumerate(events):
        v = dict(variant)
        if n_ev > 1:                    # 다중이벤트: 위치 슬롯을 나눠 겹침 방지
            # scale_events 는 개수가 가변이라 균등 슬롯 — pick_s 가 가용 구간
            # 이어붙인 길이축의 비율이므로 균등 비율 = 가용축 등간격(≥ min_gap).
            # 기존 다중이벤트(combine)는 슬롯 고정 유지 (산출물 불변).
            slots = ([(j2 + 0.5) / n_ev for j2 in range(n_ev)] if scale
                     else [0.3, 0.65, 0.9])
            v['위치'] = slots[j % len(slots)]
            v['발동거리'] = slots[j % len(slots)]
            if scale:
                # 겹침 재시도를 자기 슬롯 근방으로 제한 — 전역 fracs 로 옮기면
                # 다른 슬롯 이벤트와 붙어 min_gap_m 이 깨진다 (실측: 89 m).
                # ±0.15/n, ±0.3/n 은 슬롯 반간격(0.5/n)의 30/60% — 이웃 불침범.
                s0, d = v['위치'], 1.0 / n_ev
                v['_retry_fracs'] = [min(0.98, max(0.02, s0 + k * d))
                                     for k in (0.15, -0.15, 0.3, -0.3)]
        v.setdefault('위치', 0.5)
        v.setdefault('발동거리', v['위치'])
        # 보행속도 기본값은 여기서 주입하지 않는다 — 이벤트별로 다르고
        # (jaywalk/ped_blind 2.5 vs pedestrian 1.5) params 가 단일 출처다.
        # ped_walk_speed 가 "축에 있으면 축, 없으면 params" 로 고른다.
        v.setdefault('트리거거리', 25)      # cut_in 트리거 반경 (보행자와 무관)
        v.setdefault('감속강도', 5.0)
        if '속도_kph' in cfg:
            v['_속도_kph'] = float(cfg['속도_kph'])
        ev_list.append((ev, v))
    seed_key = f'{seed}:{theme}:{name}'
    axes = {k: v for k, v in variant.items() if k not in ('route', 'event')} | \
        {'route': list(variant['route']), 'event': events}
    # 교통류는 이벤트가 아니라 시나리오 전역 속성 — event 목록이 아니라 테마의
    # pulk 플래그로 켠다. 축 값(교통류대수·교통류밀도)은 axes 에 이미 들어 있어
    # --from-yaml 재생성도 같은 PulkDef 를 낸다.
    pulk = pulk_def(axes) if cfg.get('pulk') else None
    xml_text, sdef, bad = build_scenario(lg, ctrl_map, route, ev_list, axes,
                                         name, seed_key, min_keep=min_keep,
                                         pulk=pulk)
    return route, xml_text, sdef, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='주제 프리셋 → 시나리오 XML + 경로 CSV + batch 목록')
    ap.add_argument('themes', nargs='*', help='configs/themes.yaml 의 주제 이름')
    ap.add_argument('--hours', type=float, default=None, help='총 실행시간 예산 [h]')
    ap.add_argument('--count', type=int, default=None, help='주제당 시나리오 개수')
    ap.add_argument('--seed', type=int, default=0, help='샘플링 시드 (재현성)')
    ap.add_argument('--out-dir', default=str(ROOT / 'scenarios'))
    ap.add_argument('--vtd-dir', default='/home/mjw/scenarios',
                    help='batch 목록의 vtd_xml_path 프리픽스 (VTD PC 기준. '
                         '실측: scp 수신 확인, VTD 는 임의 절대경로 로드 가능)')
    ap.add_argument('--graph', default=str(ROOT / 'data' / 'lane_graph.pkl'))
    ap.add_argument('--list', action='store_true', help='주제 목록만 출력')
    ap.add_argument('--from-yaml', default=None, help='저장된 정의 YAML 로 단건 재생성')
    ap.add_argument('--rebuild-lists', action='store_true',
                    help='생성 없이 디스크(<주제>/*.yaml) 기준으로 batch 목록만 재생성')
    ap.add_argument('--coverage-report', action='store_true',
                    help='이번 생성분이 밟은 road 커버리지 요약(유니크/%%/미방문) 출력')
    a = ap.parse_args(argv)

    if a.hours is not None and a.count is not None:
        raise GenError('--hours 와 --count 는 함께 쓸 수 없다')

    # ── 복구용: 목록만 재생성 (lane_graph 불필요) ────────────────────────
    if a.rebuild_lists:
        out_dir = pathlib.Path(a.out_dir)
        if not out_dir.is_dir():
            raise GenError(f'{out_dir} 가 없다')
        n_all, n_themes = rebuild_batch_lists(out_dir, a.vtd_dir)
        print(f'batch_all.json: {n_all}개 (주제 {n_themes}개)  ({out_dir}/)')
        return 0

    route_defs, themes, gen_cfg = load_themes()
    GATE_STATS['ok'] = GATE_STATS['reject'] = 0

    if a.list:
        for th, cfg in themes.items():
            evs = ','.join(cfg.get('event', ['none']))
            print(f'  {th:12s} event={evs}  routes={",".join(cfg.get("routes", []))}')
        return 0

    lg = LaneGraph(a.graph)
    ctrl_map = junction_ctrl_map(lg)
    out_dir = pathlib.Path(a.out_dir)

    # ── 단건 재생성 ──────────────────────────────────────────────────────
    if a.from_yaml:
        sdef = yaml.safe_load(pathlib.Path(a.from_yaml).read_text(encoding='utf-8'))
        rows = [tuple(r) for r in sdef['route']['rows']]
        route = _build_from_rows(lg, sdef['route']['name'], rows)
        why = spawn_gate(lg, route.rt, gen_cfg) or polyline_gate(lg, route.rt, gen_cfg)
        if why:                         # 게이트 도입 전에 저장된 정의일 수 있다 — 재생성 거부
            raise GenError(f'{a.from_yaml}: 경로 게이트 탈락 — {why}')
        axes = sdef['axes']
        variant = {k: v for k, v in axes.items() if k not in ('route', 'event')}
        variant['route'] = tuple(axes['route'])
        variant['event'] = list(axes['event'])
        theme = sdef['theme']
        seed = int(sdef['seed_key'].split(':')[0])
        route2, xml_text, sdef2, bad = gen_one(lg, ctrl_map, _FixedPool(route),
                                               theme, themes.get(theme, {}),
                                               variant, sdef['name'], seed)
        write_scenario(out_dir, theme, sdef['name'], xml_text, sdef2, route.rows)
        for _, x, y, dm in bad:
            print(f'  ⚠ ego 차선 이벤트가 경로에서 {dm:.1f} m 벗어남 ({x:.1f},{y:.1f})')
        print(f'재생성 완료: {out_dir / theme / (sdef["name"] + ".xml")}')
        return 0

    if not a.themes:
        raise GenError('주제를 하나 이상 지정하거나 --list 를 쓸 것')
    for th in a.themes:
        if th not in themes:
            raise GenError(f'모르는 주제 "{th}" — 가능한 주제: {", ".join(themes)}')

    pool = RoutePool(lg, route_defs, a.seed, gen_cfg)
    # 커버리지 이력 지속 (2026-08-28): 매 실행이 독립이면 같은 도로·방향을
    # 반복 추첨한다 — 방문 이력(used_roads/used_cells)을 out_dir 단위 파일로
    # 이어받아 실행 간에도 미방문 도로·방향을 우선한다. 같은 out_dir 반복
    # 실행은 이력에 의존하는 게 의도다; 같은 seed 재현이 필요하면 새 out_dir
    # 을 쓰거나 이 파일을 지울 것 (테스트도 새 out_dir 기준).
    hist_path = pathlib.Path(a.out_dir) / 'coverage_history.json'
    if hist_path.exists():
        hist = json.loads(hist_path.read_text(encoding='utf-8'))
        pool.used_roads.update({int(k): int(v) for k, v in hist['used_roads'].items()})
        pool.used_cells.update({tuple(int(t) for t in k.split(',')): int(v)
                                for k, v in hist['used_cells'].items()})
        print(f'커버리지 이력 이어받음: 방문 도로 {len(pool.used_roads)}개, '
              f'시작 셀 {len(pool.used_cells)}개 ({hist_path})')
    combos = {th: expand_theme(th, themes[th], a.seed, route_defs) for th in a.themes}

    def route_len(route_key):
        return pool.get(*route_key).rt['total_length']

    chosen = allocate(themes, combos, a.count, a.hours, route_len, a.seed)

    summary, warn_total = [], 0
    skipped = []
    walk_starts: set = set()            # 합성 경로 시작 도로들 — 실기 스폰 확인 항목
    cov_roads: collections.Counter = collections.Counter()   # 생성분이 밟은 road 빈도
    for th in a.themes:
        cfg = themes[th]
        n_ok = 0
        est_total = 0.0
        target = len(chosen[th])
        # 실패한 변형은 선택되지 않은 나머지 조합으로 백필해 개수를 채운다.
        # start: 자유 주제는 조합이 소진돼도 새 순번(=새 시작점의 walk)으로
        # 이어서 백필한다 — 조합 수 == target 이면 실패 1건이 곧 개수 미달이던
        # 문제(2026-08-28 실측: 다중이벤트 17/20) 방지. 상한 3×target.
        picked = {id(c) for c in chosen[th]}
        queue = list(chosen[th]) + [c for c in combos[th] if id(c) not in picked]
        free_mode = bool(combos[th] and combos[th][0]['route'][2])
        next_rv = max((c['route'][1] for c in combos[th]), default=0) + 1
        extra_used, qi = 0, 0
        while n_ok < target:
            if qi < len(queue):
                variant = queue[qi]
                qi += 1
            elif free_mode and extra_used < 3 * target:
                base = combos[th][extra_used % len(combos[th])]
                variant = dict(base)
                variant['route'] = (base['route'][0], next_rv, base['route'][2])
                next_rv += 1
                extra_used += 1
            else:
                break
            name = f'{th}_{n_ok + 1:02d}_{variant["route"][0]}'
            if variant['route'][1] != 1 and 'walk' in route_defs.get(variant['route'][0], {}):
                name += str(variant['route'][1])
            try:
                route, xml_text, sdef, bad = gen_one(lg, ctrl_map, pool, th, cfg,
                                                     variant, name, a.seed)
            except (GenError, RouteError) as e:
                desc = f'{variant["route"][0]} / {"+".join(variant["event"])}'
                skipped.append((th, desc, str(e).splitlines()[0]))
                continue
            write_scenario(out_dir, th, xml_text=xml_text, name=name,
                           sdef=sdef, rows=route.rows)
            rs = sdef['route']
            print(f'  {name}: 시작 도로 {rs["start"]["road"]} '
                  f'({rs["start"]["x"]:.1f},{rs["start"]["y"]:.1f})  '
                  f'{rs["length_m"]:.0f} m  회전 좌{rs["turns"]["left"]}/우{rs["turns"]["right"]}  '
                  f'교차로 {rs["junctions"] or "없음"}  도로 {len(rs["roads"])}개')
            if 'walk' in route_defs.get(variant['route'][0], {}):
                walk_starts.add(rs['start']['road'])
            cov_roads.update(rs['roads'])
            for _, x, y, dm in bad:
                print(f'  ⚠ {name}: ego 차선 이벤트가 경로에서 {dm:.1f} m 벗어남 ({x:.1f},{y:.1f})')
                warn_total += 1
            n_ok += 1
            est_total += sdef['est_s']
        unsup = sorted({UNSUPPORTED[e] for e in cfg.get('event', []) if e in UNSUPPORTED})
        summary.append((th, n_ok, est_total, '; '.join(unsup)))

    # ── batch 목록: 디스크(<주제>/*.yaml)가 단일 출처 — 매번 전체 재생성 ──
    out_dir.mkdir(parents=True, exist_ok=True)
    hist_path.write_text(json.dumps(
        {'used_roads': {str(k): v for k, v in sorted(pool.used_roads.items())},
         'used_cells': {','.join(map(str, k)): v
                        for k, v in sorted(pool.used_cells.items())}},
        indent=1), encoding='utf-8')
    n_all, n_themes = rebuild_batch_lists(out_dir, a.vtd_dir)

    # ── 요약 ─────────────────────────────────────────────────────────────
    print()
    print(f'{"주제":14s} {"개수":>4s} {"예상시간":>8s}  비고')
    tot_n, tot_t = 0, 0.0
    for th, n, t, note in summary:
        print(f'{th:14s} {n:4d} {t / 60:7.1f}분  {note}')
        tot_n += n
        tot_t += t
    print(f'{"합계":14s} {tot_n:4d} {tot_t / 3600:6.2f}시간')
    if skipped:
        print('\n생성 못 한 변형 (사유별 — 미구현/경로 제약 우선순위 자료):')
        grouped: dict = {}
        for th, desc, why in skipped:
            grouped.setdefault((th, why), []).append(desc)
        for (th, why), descs in grouped.items():
            uniq = sorted(set(descs))
            print(f'  ✗ {th} ×{len(descs)} [{", ".join(uniq[:3])}'
                  f'{" 외" if len(uniq) > 3 else ""}]: {why}')
    if warn_total:
        print(f'\n⚠ 횡거리 경고 {warn_total}건 — 위 로그 확인')
    g_ok, g_rej = GATE_STATS['ok'], GATE_STATS['reject']
    line = f'\n스폰-경로 게이트: 통과 {g_ok}경로 / 폐기 {g_rej}회 (재시도 포함)'
    if g_rej > g_ok:
        line += ('  ⚠ 폐기가 통과보다 많다 — walk 시작점 선정이 "뒤쪽 탈출로 있는 '
                 '양방향 도로"에 편중됐다는 신호 (start_pool 조건 검토)')
    print(line)
    if a.coverage_report:
        all_roads = set(lg.roads)
        visited = set(cov_roads)
        pct = 100.0 * len(visited) / max(1, len(all_roads))
        print(f'\n[커버리지] 이번 생성분 road {len(visited)}개 / {len(all_roads)}개 ({pct:.1f}%)')
        print('  상위 빈도: ' + '  '.join(f'{r}×{n}' for r, n in cov_roads.most_common(20)))
        missing = sorted(all_roads - visited)
        print(f'  미방문 {len(missing)}개: ' + ' '.join(map(str, missing)))
    if walk_starts:
        print(f'\n[실기 1회 확인] 합성 경로 시작 도로 {sorted(walk_starts)} — '
              f'Ego PathRef/Path01 은 경로 기준으로 생성했지만, VTD 스폰이 실제로 '
              f'그 위치에 되는지는 새 시작점마다 실기에서 한 번 확인할 것')
    print(f'\nbatch_all.json: {n_all}개 (주제 {n_themes}개)  ({out_dir}/)')
    print(f'실행:  python3 tools/batch_run.py {out_dir / "batch_all.json"}')
    print('       (batch_run 은 목록 여러 개·glob 도 받는다 — 주제별 batch_<주제>.json 조합 가능)')
    return 0


class _FixedPool:
    """--from-yaml 재생성용 — 저장된 rows 로 만든 경로만 돌려준다."""

    def __init__(self, route: Route):
        self.route = route

    def get(self, name, variant=1, salt='', **_kw):
        return self.route


if __name__ == '__main__':
    raise SystemExit(main())
