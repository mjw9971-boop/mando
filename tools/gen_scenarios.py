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
from vtd_adapter.lanegraph import LaneGraph                             # noqa: E402

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
}


class GenError(SystemExit):
    """생성 실패 — 어디서 왜 막혔는지 메시지에 담는다 (무음 실패 금지)."""


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


def _walk_once(lg, rng, start, turns, tail_m, max_gap_m):
    """start 차로에서 turns 정책대로 걷는다. 성공 → (chain, waypoint rows), 실패 → None."""
    chain = [start]
    s0 = min(8.0, lg.length(start) * 0.2)
    rows = [(s0, start, 'start')]
    entries = []            # (entry_lane, exit_lane)
    cur = start
    dist_since_exit = lg.length(start) - s0
    for ti, want in enumerate(turns):
        # 다음 교차로까지 전진
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
                return None
            cur = nxts[0]
            chain.append(cur)
            dist_since_exit += lg.length(cur)
            if dist_since_exit > 700.0:
                return None
        else:
            return None
        if max_gap_m is not None and ti > 0 and dist_since_exit > max_gap_m:
            return None
        pick = ([o for o in opts if o[0] == want] if want != 'any'
                else [o for o in opts if o[0] != 'uturn'])
        if not pick:
            return None
        kind, ap, jchain, exit_lane = pick[rng.randrange(len(pick))]
        if ap is not cur and ap != cur:
            chain.append(ap)                     # 진입 전 차선변경 (이웃 차로로)
        entries.append((ap, exit_lane))
        chain += jchain
        chain.append(exit_lane)
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
        cur = nxts[0]
        chain.append(cur)
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
    if require == 'signalized':
        # 교차로 앞 접근 차로에 신호가 매핑돼 있어야 한다
        for i, k in enumerate(chain[:-1]):
            if lg.lanes[k]['junction'] == -1 and lg.lanes[chain[i + 1]]['junction'] != -1 \
                    and lg.lanes[k]['signals']:
                return True
        return False
    return True


_START_POOL_CACHE: dict = {}


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

    조건: 일반 도로(junction=-1)의 주행 차선, 출발 오프셋(≤8 m)을 빼고도
    가속 구간 50 m 이상, 전방이 회전 가능한 연결로로 이어짐.
    """
    key = id(lg)
    if key not in _START_POOL_CACHE:
        _START_POOL_CACHE[key] = [
            k for k, v in sorted(lg.lanes.items())
            if v['type'] == 'driving' and v['junction'] == -1
            and v['length'] >= 58.0 and v['next']
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


def synth_walk(lg, rng, name, spec) -> Route:
    """walk 사양 → 검증까지 마친 Route. 실패는 GenError (무음 실패 금지)."""
    turns = spec.get('turns', ['any'])
    tail = float(spec.get('tail_m', 100))
    require = spec.get('require')
    max_gap = spec.get('max_gap_m')
    if require == 'school_zone':
        cands = _upstream_starts(lg, lambda v: v['school_zone'])
    elif require == 'speed_change':
        cands = _upstream_starts(lg, lambda v: v['speed_limit'] == 30 or v['school_zone'])
    elif require == 'signalized':
        cands = _upstream_starts(lg, lambda v: bool(v['signals']))
    else:
        cands = start_pool(lg)          # 맵 전체 후보 풀(1회 수집)에서 시드 샘플링
    if not cands:
        raise GenError(f'경로 {name}: 출발 후보 차로가 없다 (require={require})')
    order = list(cands)
    rng.shuffle(order)
    for start in order[:300]:
        res = _walk_once(lg, rng, start, turns, tail, max_gap)
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
        return route
    raise GenError(f'경로 {name}: {min(len(order),300)}개 출발 후보로 걸어도 빌드 경고 0 인 '
                   f'경로를 못 만들었다 (turns={turns}, require={require})')


class RoutePool:
    """경로 풀 — 같은 (이름, 변형, salt) 은 한 번만 만든다.

    salt 는 주제의 `start: 자유` 모드용이다: 주제 이름이 들어가 같은 경로 풀
    이름이라도 주제·경로변형마다 다른 시드 → 다른 시작점에서 걷는다.
    csv 경로는 시작점이 파일에 고정이라 salt 의 영향이 없다.
    """

    def __init__(self, lg, defs: dict, seed: int):
        self.lg, self.defs, self.seed = lg, defs, seed
        self.cache: dict = {}

    def get(self, name: str, variant: int = 1, salt: str = '') -> Route:
        if name not in self.defs:
            raise GenError(f'routes 풀에 "{name}" 이 없다 (configs/themes.yaml)')
        d = self.defs[name]
        if 'csv' in d:
            variant, salt = 1, ''       # csv 는 시작점·변형이 파일에 고정
        key = (name, variant, salt)
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
        elif 'walk' in d:
            seed_str = f'{self.seed}:route:{name}:{variant}' + (f':{salt}' if salt else '')
            rng = random.Random(seed_str)
            label = name if variant == 1 else f'{name}{variant}'
            r = synth_walk(self.lg, rng, label, d['walk'])
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
    if from_side == '우측':
        t0, t1 = -(right + 1.5 + extra), left + 1.5
    else:
        t0, t1 = left + 1.5 + extra, -(right + 1.5)
    pts = []
    for u in np.linspace(0.0, 1.0, 4):
        t = t0 + (t1 - t0) * float(u)
        x, y, z, h = route_pt(lg, rt, s, t)
        yaw = h + (math.pi / 2 if t1 > t0 else -math.pi / 2)
        pts.append((x, y, z, yaw % (2 * math.pi)))
    return pts


def _add_ped(ctx, s, walk_speed, trig_d, from_side, tag):
    pts = _crossing_shape(ctx, s, from_side)
    sid = ctx.next_shape()
    name = ctx.next_name('Ped')
    tx, ty, _, _ = route_pt(ctx.lg, ctx.route.rt, max(1.0, s - trig_d))
    ctx.moving.append(blk_pathshape(sid, f'{name}Path', pts))
    ctx.moving.append(blk_character(name, pts[0][0], pts[0][1], pts[0][2], pts[0][3]))
    ctx.moving.append(blk_character_actions(name, tx, ty, 10.0, walk_speed, sid))
    cx, cy, _, _ = route_pt(ctx.lg, ctx.route.rt, s)
    ctx.checks.append((cx, cy, tag))
    return {'ped': name, 'route_s': round(s, 2), 'walk_speed': walk_speed,
            'trigger_d': trig_d, 'from': from_side}


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
    ctx.claim(s - 40, s + 20, 'pedestrian')
    out = _add_ped(ctx, s, v['보행속도'], v['트리거거리'], v.get('방향', '우측'), 'ego_lane')
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
    ctx.claim(s - 40, s + 20, 'jaywalk')
    out = _add_ped(ctx, s, v['보행속도'], v['트리거거리'], v.get('방향', '우측'), 'ego_lane')
    out.update(kind='jaywalk', spot=v.get('지점', '도로중간'))
    return out


def ev_ped_blind(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    s = pick_s(ctx.spans, v['위치'], need=40.0)
    ctx.claim(s - 40, s + 40, 'ped_blind')
    bus = v.get('차폐물', '버스') == '버스'
    vtype, vlen = (BUS, BUS_LEN) if bus else (CAR, CAR_LEN)
    _, k, sl = lane_at(rt, s)
    w = lg.width_at(k, sl)
    t_block = -(w / 2.0 + 1.1)                      # 차로 우측 가장자리 바깥에 정차
    bx, by, bz, bh = route_pt(lg, rt, s, t_block)
    name = ctx.next_name('Blocker')
    ctx.players.append(blk_vehicle_posabs(name, vtype, bx, by, bz, bh))
    ctx.actions.append(blk_stay_action(name, bx, by))
    # 보행자: 차폐물 앞머리 쪽에서 우→좌 횡단 (가려져 있다가 출현)
    ped_s = s + vlen / 2.0 + 2.0
    out = _add_ped(ctx, ped_s, v['보행속도'], v['트리거거리'], '우측', 'ego_lane')
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


def ev_static_vehicle(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    s = pick_s(ctx.spans, v['위치'], need=30.0)
    ctx.claim(s - 40, s + 40, 'static_vehicle')
    i, k, sl = lane_at(rt, s)
    t = 0.0
    lane_tag = 'ego_lane'
    if v.get('차선') == '좌측차로':
        nb = lg.neighbor(k, 'left')
        if nb is not None and lg.lanes[nb]['dir'] == lg.lanes[k]['dir'] \
                and lg.lanes[nb]['type'] == 'driving':
            t = (lg.width_at(k, sl) + lg.width_at(nb, min(sl, lg.length(nb)))) / 2.0
            lane_tag = 'side_lane'
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


def ev_obstacle_chain(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    n0 = n = int(v.get('개수', 4))
    spacing = 18.0
    s0 = None
    while n >= 2:
        try:
            s0 = pick_s(ctx.spans, v['위치'], need=n * spacing + 20.0)
            ctx.claim(s0 - 20, s0 + n * spacing + 20, 'obstacle_chain')
            break
        except GenError:
            n -= 1                      # 구간이 짧거나 겹치면 개수를 줄여서라도 놓는다
            s0 = None
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


def ev_narrow(ctx, v):
    lg, rt = ctx.lg, ctx.route.rt
    s = pick_s(ctx.spans, v['위치'], need=40.0)
    ctx.claim(s - 30, s + 50, 'narrow')
    intr = float(v.get('침범폭', 0.7))
    placed = []
    for ds, sgn in ((0.0, -1.0), (14.0, 1.0)):       # 우측 먼저, 14 m 뒤 좌측
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


def build_scenario(lg, ctrl_map, route: Route, events: list, axes: dict,
                   name: str, seed_key: str):
    """이벤트 목록 → (xml_text, def_dict, warnings)."""
    ctx = Ctx(lg, route, random.Random(seed_key), ctrl_map)
    resolved = []
    for ev, v in events:
        if ev not in EVENTS:
            raise GenError(f'{name}: 모르는 이벤트 "{ev}"')
        # 구간 겹침/부족은 위치를 옮겨가며 재시도한다 (다중이벤트 대비).
        # 이벤트가 블록을 일부 추가하고 실패할 수 있으므로 스냅숏 후 복원한다.
        fracs = [v.get('위치', 0.5)] + [0.15, 0.35, 0.55, 0.75, 0.9]
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
                if '겹친다' not in msg and '부족하다' not in msg:
                    raise
                last = e
        if last is not None:
            raise last
    doc = XmlDoc(TEMPLATE)
    rt = route.rt
    doc.set_path(path_waypoints(lg, rt))
    doc.set_ego(rt['start_s_in_lane'], rt['start_s_in_lane'] + rt['total_length'],
                _start_lane_id(rt))
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
         'est_s': round(est_seconds(rt['total_length']), 1),
         'timeout_s': timeout_for(rt['total_length'])}
    return xml_text, d, bad


# ════════════════════════════════════════════════════════════════════════
# 6. 주제 전개 (조합 → 개수/시간 예산)
# ════════════════════════════════════════════════════════════════════════

def axis_pool(theme_cfg: dict, axis: str):
    return list(theme_cfg.get(axis, AXIS_DEFAULTS.get(axis, [None])))


def expand_theme(theme: str, cfg: dict, seed: int) -> list:
    """주제 → 변형 목록 [{'route': (이름, 변형), 'event': [...], axes…}] (샘플링 전 전체)."""
    rng = random.Random(f'{seed}:{theme}')
    routes = list(cfg.get('routes', ['기본']))
    vary = list(cfg.get('vary', []))
    events = list(cfg.get('event', ['none']))
    # start: 자유 → walk 경로 시드에 주제 이름을 섞어 주제·변형마다 다른 시작점
    salt = theme if cfg.get('start', '고정') == '자유' else ''
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
    return combos


def allocate(themes: dict, combos: dict, count, hours, route_len, seed):
    """주제별 선택 목록 확정. combos[theme] 는 이미 시드 셔플돼 있다."""
    chosen = {}
    if count is not None:
        for th, lst in combos.items():
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
    return data.get('routes', {}), data.get('themes', {})


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
    route = pool.get(*variant['route'])
    ev_list = []
    n_ev = len(variant['event'])
    for j, ev in enumerate(variant['event']):
        v = dict(variant)
        if n_ev > 1:                    # 다중이벤트: 위치 슬롯을 나눠 겹침 방지
            slots = [0.3, 0.65, 0.9]
            v['위치'] = slots[j % len(slots)]
            v['발동거리'] = slots[j % len(slots)]
        v.setdefault('위치', 0.5)
        v.setdefault('발동거리', v['위치'])
        v.setdefault('보행속도', 1.5)
        v.setdefault('트리거거리', 25)
        v.setdefault('감속강도', 5.0)
        if '속도_kph' in cfg:
            v['_속도_kph'] = float(cfg['속도_kph'])
        ev_list.append((ev, v))
    seed_key = f'{seed}:{theme}:{name}'
    xml_text, sdef, bad = build_scenario(lg, ctrl_map, route, ev_list,
                                         {k: v for k, v in variant.items()
                                          if k not in ('route', 'event')} |
                                         {'route': list(variant['route']),
                                          'event': variant['event']},
                                         name, seed_key)
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
    a = ap.parse_args(argv)

    if a.hours is not None and a.count is not None:
        raise GenError('--hours 와 --count 는 함께 쓸 수 없다')

    route_defs, themes = load_themes()

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

    pool = RoutePool(lg, route_defs, a.seed)
    combos = {th: expand_theme(th, themes[th], a.seed) for th in a.themes}

    def route_len(route_key):
        return pool.get(*route_key).rt['total_length']

    chosen = allocate(themes, combos, a.count, a.hours, route_len, a.seed)

    batch_by_theme = {th: [] for th in a.themes}
    summary, warn_total = [], 0
    skipped = []
    walk_starts: set = set()            # 합성 경로 시작 도로들 — 실기 스폰 확인 항목
    for th in a.themes:
        cfg = themes[th]
        n_ok = 0
        est_total = 0.0
        target = len(chosen[th])
        # 실패한 변형은 선택되지 않은 나머지 조합으로 백필해 개수를 채운다
        picked = {id(c) for c in chosen[th]}
        queue = list(chosen[th]) + [c for c in combos[th] if id(c) not in picked]
        for variant in queue:
            if n_ok >= target:
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
            for _, x, y, dm in bad:
                print(f'  ⚠ {name}: ego 차선 이벤트가 경로에서 {dm:.1f} m 벗어남 ({x:.1f},{y:.1f})')
                warn_total += 1
            batch_by_theme[th].append({
                'name': name,
                'vtd_xml_path': f'{a.vtd_dir.rstrip("/")}/{th}/{name}.xml',
                'route_csv': str(pathlib.Path(a.out_dir).name + f'/{th}/{name}.csv'),
                'timeout_s': sdef['timeout_s']})
            n_ok += 1
            est_total += sdef['est_s']
        unsup = sorted({UNSUPPORTED[e] for e in cfg.get('event', []) if e in UNSUPPORTED})
        summary.append((th, n_ok, est_total, '; '.join(unsup)))

    # ── batch 목록: 주제별 + 이번 실행 전체 통합(batch_all.json) ─────────
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_all = [it for th in a.themes for it in batch_by_theme[th]]
    names = [it['name'] for it in batch_all]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:                              # 통합 기준 중복 검사 — batch_run 과 같은 규칙
        raise GenError(f'통합 batch 목록에서 이름이 중복된다: {dup}')
    for th in a.themes:
        (out_dir / f'batch_{th}.json').write_text(
            json.dumps(batch_by_theme[th], ensure_ascii=False, indent=1), encoding='utf-8')
    (out_dir / 'batch_all.json').write_text(
        json.dumps(batch_all, ensure_ascii=False, indent=1), encoding='utf-8')

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
    if walk_starts:
        print(f'\n[실기 1회 확인] 합성 경로 시작 도로 {sorted(walk_starts)} — '
              f'Ego PathRef/Path01 은 경로 기준으로 생성했지만, VTD 스폰이 실제로 '
              f'그 위치에 되는지는 새 시작점마다 실기에서 한 번 확인할 것')
    print(f'\nbatch 목록: ' + ', '.join(f'batch_{th}.json' for th in a.themes)
          + f'  +  batch_all.json  ({out_dir}/)')
    print(f'실행:  python3 tools/batch_run.py {out_dir / "batch_all.json"}')
    print('       (batch_run 은 목록 여러 개·glob 도 받는다 — 주제별 batch_<주제>.json 조합 가능)')
    return 0


class _FixedPool:
    """--from-yaml 재생성용 — 저장된 rows 로 만든 경로만 돌려준다."""

    def __init__(self, route: Route):
        self.route = route

    def get(self, name, variant=1, salt=''):
        return self.route


if __name__ == '__main__':
    raise SystemExit(main())
