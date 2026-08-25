"""
lanegraph.py ─ lane_graph.pkl 런타임 헬퍼 (인지 노드에서 import)

    from lanegraph import LaneGraph
    lg = LaneGraph('lane_graph.pkl')

    m = lg.locate(x, y, yaw)              # → LaneMatch(lane, s, t, heading_err, dist)  (ego / 객체 공용)
    lg.point_at(lane, s)                  # → x, y, z, hdg
    lg.mark_at(lane, s, 'left')           # → (type, color, lane_change_ok)
    lg.speed_limit_at(lane)               # → (limit or None, school_zone)
    lg.lookahead(route, idx, s_in_lane, horizon=200)   # 전방 프로파일 (정지선/신호/횡단보도/제한속도/실선/회전)

좌표계: xodr 월드 (VTD 9910 ego X,Y 와 동일 좌표계라고 가정 — 연습 때 확인)
s     : 차로 시작(주행 방향)부터 누적 거리 [m]
t     : 중심선 기준 횡방향 오프셋 [m], 왼쪽 + / 오른쪽 −
"""
import math, pickle
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import numpy as np
from scipy.spatial import cKDTree

LaneKey = Tuple[int, int, int]  # (road_id, section_idx, lane_id)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class LaneMatch:
    lane: LaneKey
    s: float            # 차로 내 주행 s
    t: float            # 횡 오프셋 (좌 +)
    heading_err: float  # yaw - lane heading  [rad] (yaw 없으면 0)
    dist: float         # 중심선까지 거리 [m]
    idx: int            # 가장 가까운 점 index


@dataclass
class Ahead:
    """lookahead 결과 한 항목"""
    dist: float                 # 현재 위치부터 거리 [m]
    kind: str                   # stop_line / signal / crosswalk / crosswalk_warn / yield / speed / school / mark_left / mark_right / turn / lane_change / junction_in / junction_out / dead_end / route_end
    lane: LaneKey
    s_in_lane: float
    data: Dict[str, Any] = field(default_factory=dict)


class LaneGraph:
    def __init__(self, path='lane_graph.pkl'):
        with open(path, 'rb') as f:
            g = pickle.load(f)
        self.g = g
        self.lanes: Dict[LaneKey, dict] = g['lanes']
        self.roads = g['roads']
        self.signals = g['signals']
        self.meta = g['meta']
        self.lane_keys: List[LaneKey] = g['lane_keys']
        self.kd_pts = g['kd_pts']
        self.kd_lane = g['kd_lane']
        self.kd_i = g['kd_i']
        self.kd_hdg = g['kd_hdg']
        self.kd = cKDTree(self.kd_pts)

    # ── 기본 조회 ───────────────────────────────────────────────────────
    def lane(self, key: LaneKey) -> dict:
        return self.lanes[key]

    def length(self, key: LaneKey) -> float:
        return self.lanes[key]['length']

    def successors(self, key: LaneKey) -> List[LaneKey]:
        return self.lanes[key]['next']

    def predecessors(self, key: LaneKey) -> List[LaneKey]:
        return self.lanes[key]['prev']

    def neighbor(self, key: LaneKey, side: str) -> Optional[LaneKey]:
        return self.lanes[key]['left_nb' if side == 'left' else 'right_nb']

    def point_at(self, key: LaneKey, s: float):
        """차로 s → (x, y, z, hdg)"""
        r = self.lanes[key]
        ss = r['s']
        s = float(np.clip(s, 0.0, r['length']))
        x = np.interp(s, ss, r['pts'][:, 0])
        y = np.interp(s, ss, r['pts'][:, 1])
        z = np.interp(s, ss, r['pts'][:, 2])
        h = np.interp(s, ss, np.unwrap(r['hdg'].astype(float)))
        return float(x), float(y), float(z), wrap(float(h))

    def points_ahead(self, key: LaneKey, s: float, dist: float, step: float = 0.5, route=None, idx=None):
        """
        현재 s 에서 dist 만큼 앞의 중심선 점들 (경로가 있으면 다음 차로로 이어서) → (N,2) array

        **경로의 차선변경 이음매를 뒤로 점프시키지 않는다.** route['lanes'] 는 차선변경을
        "평행한 이웃 차로를 연달아 적는" 식으로 표현한다(successor 가 아니다). 이걸
        successor 처럼 s=0 부터 이어붙이면 경로가 차로 길이만큼 **뒤로** 튀고
        (실측 6.75 / 12.4 / 30.3 m), Pure Pursuit 목표점이 차 뒤로 넘어가 조향이
        풀락으로 나간다 — 2026-08-20 주행에서 차선변경 두 곳 모두 도로이탈 +
        courseRespawn 으로 끝났다(logs/run_20260820_200930.jsonl).
        그래서 이음매에서는 route 를 따라가지 않고 **자기 successor** 로 잇는다.
        경로가 기하학적으로 끊기지 않는 것이 우선이고, 목표 차로로의 가로 이동은
        planner._blend_path 가 맡는다. 이어붙인 차로가 다시 route 위에 있으면
        그 지점부터 route 추종을 재개한다.
        """
        out = []
        acc = 0.0
        cur = key
        i = idx
        s0 = s
        while acc < dist and cur is not None:
            r = self.lanes[cur]
            L = r['length']
            if s0 < L - 1e-9:
                ss = np.arange(s0, L + 1e-9, step)
            elif not out:
                ss = np.array([min(s0, L)])       # 경로 끝에서도 최소 한 점은 낸다
            else:
                ss = None
            if ss is not None:
                xs = np.interp(ss, r['s'], r['pts'][:, 0])
                ys = np.interp(ss, r['s'], r['pts'][:, 1])
                for x, y in zip(xs, ys):
                    out.append((x, y))
            acc += max(0.0, L - s0)
            nxt = None
            if route is not None and i is not None and i + 1 < len(route['lanes']):
                nxt = route['lanes'][i + 1]
                if nxt in r['next']:
                    i += 1
                else:
                    nxt = None                    # 차선변경 이음매 — 아래로 떨어뜨린다
            if nxt is None:
                nx = r['next']
                nxt = nx[0] if nx else None
                i = self._route_pos(route, nxt)
            cur = nxt
            s0 = 0.0
        return np.array(out) if out else np.zeros((0, 2))

    @staticmethod
    def _route_pos(route, lane) -> Optional[int]:
        """route['lanes'] 안에서의 위치. 없으면 None (이후로는 successor 만 따라간다)."""
        if route is None or lane is None:
            return None
        try:
            return route['lanes'].index(tuple(lane))
        except ValueError:
            return None

    def mark_at(self, key: LaneKey, s: float, side: str):
        segs = self.lanes[key]['left_mark' if side == 'left' else 'right_mark']
        for s0, s1, typ, col, ok in segs:
            if s0 - 1e-6 <= s <= s1 + 1e-6:
                return typ, col, ok
        return ('none', 'standard', False) if not segs else segs[-1][2:]

    def lane_change_ok(self, key: LaneKey, s: float, side: str) -> bool:
        """이 지점에서 side 로 차선변경 가능? (점선 + 옆차로 존재)"""
        if self.neighbor(key, side) is None:
            return False
        return bool(self.mark_at(key, s, side)[2])

    def speed_limit_at(self, key: LaneKey):
        r = self.lanes[key]
        return r['speed_limit'], r['school_zone']

    def width_at(self, key: LaneKey, s: float) -> float:
        r = self.lanes[key]
        return float(np.interp(s, r['s'], r['width']))

    def lanes_of_road(self, road_id: int) -> List[LaneKey]:
        return list(self.roads[road_id]['lanes'])

    def road_s_at(self, key: LaneKey, s: float) -> float:
        """차로 s(주행 방향) → 도로 s. dir=-1 차로는 road_s 가 감소 방향이다."""
        r = self.lanes[key]
        rs = r['road_s']
        # np.interp 는 x 가 증가해야 한다 — 차로 s 축은 항상 증가하므로 그대로 쓴다
        return float(np.interp(np.clip(s, 0.0, r['length']), r['s'], rs))

    def opposite_of(self, key: LaneKey) -> Optional[LaneKey]:
        """같은 도로에서 통행방향이 반대인 driving 차로 중 가장 가까운(중앙선 쪽) 것.
        같은 road_s 구간을 덮는 섹션이어야 한다 (섹션 인덱스는 방향별로 다를 수 있다)."""
        road, _sec, lid = key
        me = self.lanes[key]
        rs_mid = self.road_s_at(key, 0.5 * me['length'])
        best = None
        for k in self.roads[road]['lanes']:
            o = self.lanes[k]
            if o['dir'] == me['dir'] or o['type'] != 'driving':
                continue
            lo, hi = sorted((float(o['road_s'][0]), float(o['road_s'][-1])))
            if not (lo - 1e-6 <= rs_mid <= hi + 1e-6):
                continue
            if best is None or abs(k[2] - lid) < abs(best[2] - lid):
                best = k
        return best

    def roadway_edges(self, key: LaneKey, s: float) -> Tuple[float, float]:
        """이 차로 중심선에서 차도 가장자리까지 (left, right) 거리 [m].

        같은 방향 이웃(driving)을 좌우로 합산하고, **반대 통행방향 차로 폭을
        왼쪽에 더한다** — left_nb 는 중앙선을 넘지 않는다(실측: (30,0,-1) 이웃
        None). 2026-08-25 정지 고착: 생성기가 반대차로를 빼고 횡단 폭을 계산해
        보행자가 반대 차선 위에 멈췄고, 컨트롤러는 차도 위 보행자로 보고 영원히
        대기했다. 횡단보도 보행자 판정(planner)과 시나리오 생성(gen_scenarios)이
        같은 값을 쓰도록 여기 한 곳에 둔다.
        """
        def _extent(side: str) -> float:
            ext = 0.5 * self.width_at(key, s)
            cur = key
            for _ in range(6):
                nb = self.neighbor(cur, side)
                if nb is None or self.lanes[nb]['type'] != 'driving':
                    break
                ext += self.width_at(nb, min(s, self.length(nb)))
                cur = nb
            return ext

        left, right = _extent('left'), _extent('right')
        me = self.lanes[key]
        rs = self.road_s_at(key, s)
        for kk in self.roads[key[0]]['lanes']:
            o = self.lanes[kk]
            if o['dir'] == me['dir'] or o['type'] != 'driving':
                continue
            lo, hi = sorted((float(o['road_s'][0]), float(o['road_s'][-1])))
            if lo - 1e-6 <= rs <= hi + 1e-6:
                left += self.width_at(kk, 0.5 * o['length'])
        return left, right

    def junction_ahead(self, key: LaneKey) -> Optional[int]:
        """이 차로가 흘러드는 교차로 id (successor 가 교차로 차로일 때). 없으면 None."""
        if self.lanes[key]['junction'] != -1:
            return self.lanes[key]['junction']
        for nk in self.lanes[key]['next']:
            j = self.lanes[nk]['junction']
            if j != -1:
                return j
        return None

    # ── 위치 매칭 ───────────────────────────────────────────────────────
    def project(self, key: LaneKey, x: float, y: float, idx_hint: Optional[int] = None):
        """점 (x,y) 를 차로 key 폴리라인에 투영 → (s, t, dist, idx)"""
        r = self.lanes[key]
        P = r['pts']
        if idx_hint is None:
            d = np.hypot(P[:, 0] - x, P[:, 1] - y)
            i = int(np.argmin(d))
        else:
            i = int(np.clip(idx_hint, 0, len(P) - 1))
        best = None
        for j in (i - 1, i):
            if j < 0 or j + 1 >= len(P):
                continue
            ax, ay = P[j, 0], P[j, 1]
            bx, by = P[j + 1, 0], P[j + 1, 1]
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            if L2 < 1e-12:
                continue
            u = ((x - ax) * vx + (y - ay) * vy) / L2
            u = min(1.0, max(0.0, u))
            px, py = ax + u * vx, ay + u * vy
            dist = math.hypot(x - px, y - py)
            cross = vx * (y - ay) - vy * (x - ax)   # >0 이면 왼쪽
            t = math.copysign(dist, cross)
            s = float(r['s'][j] + u * (r['s'][j + 1] - r['s'][j]))
            if best is None or dist < best[2]:
                best = (s, t, dist, j)
        if best is None:  # 점 1개짜리 차로
            dist = math.hypot(x - P[0, 0], y - P[0, 1])
            return float(r['s'][0]), 0.0, dist, 0
        return best

    def locate(self, x: float, y: float, yaw: Optional[float] = None, k: int = 16,
               max_dist: float = 8.0, max_heading_err: float = math.radians(70),
               prefer: Optional[List[LaneKey]] = None, prefer_bonus: float = 1.5) -> Optional[LaneMatch]:
        """(x,y[,yaw]) → 가장 그럴듯한 차로. yaw 주면 반대 방향 차로 배제.
        prefer: 경로 차로 리스트 (같은 거리면 경로 차로 우선, prefer_bonus[m] 만큼 유리)"""
        d, ii = self.kd.query((x, y), k=k)
        cand = {}
        for dist, i in zip(np.atleast_1d(d), np.atleast_1d(ii)):
            if not np.isfinite(dist) or dist > max_dist:
                continue
            key = self.lane_keys[self.kd_lane[i]]
            if key in cand:
                continue
            s, t, dd, j = self.project(key, x, y, idx_hint=int(self.kd_i[i]))
            r = self.lanes[key]
            hd = float(np.interp(s, r['s'], np.unwrap(r['hdg'].astype(float))))
            herr = wrap(yaw - hd) if yaw is not None else 0.0
            if yaw is not None and abs(herr) > max_heading_err:
                continue
            score = dd + (0.0 if prefer is None or key not in prefer else -prefer_bonus) + 0.5 * abs(herr)
            cand[key] = (score, LaneMatch(key, s, t, herr, dd, j))
        if not cand:
            return None
        return min(cand.values(), key=lambda v: v[0])[1]

    # ── 경로 전방 프로파일 ───────────────────────────────────────────────
    def lookahead(self, route: dict, idx: int, s_in_lane: float, horizon: float = 200.0) -> List[Ahead]:
        """route['lanes'][idx] 의 s_in_lane 에서 horizon[m] 앞까지의 이벤트 목록 (거리순).
        speed 항목: 제한속도가 None 인 차로는 이전 값 유지(carry) → data['limit'] 에 유효값"""
        items: List[Ahead] = []
        lanes = route['lanes']
        acc = 0.0
        cur_limit, cur_school = None, False
        # 현재 차로 이전 값 (carry) 찾기: 뒤로 거슬러
        for j in range(idx, -1, -1):
            v, sc = self.speed_limit_at(lanes[j])
            if v is not None:
                cur_limit, cur_school = v, sc
                break
        prev_left = prev_right = None
        events_by_lane: Dict[LaneKey, list] = {}
        for ev in route.get('events', []):
            events_by_lane.setdefault(ev['lane'], []).append(ev)
        i = idx
        s0 = s_in_lane
        while i < len(lanes) and acc < horizon:
            key = lanes[i]
            r = self.lanes[key]
            L = r['length']
            base = acc - s0  # 차로 s → 거리: dist = base + s
            # 제한속도 변경
            v, sc = self.speed_limit_at(key)
            if v is not None and (v != cur_limit or sc != cur_school):
                if i != idx or s0 <= 1e-6:
                    items.append(Ahead(max(0.0, base + 0.0), 'speed', key, 0.0, {'limit': v, 'school_zone': sc, 'prev': cur_limit}))
                cur_limit, cur_school = v, sc
            elif i == idx:
                pass
            # 정지선/신호
            for sl in r['stop_lines']:
                if sl['s'] >= s0 - 1e-6:
                    items.append(Ahead(base + sl['s'], 'stop_line', key, sl['s'],
                                       {'signal_ids': sl['signal_ids'], 'signalized': bool(sl['signal_ids'])}))
            # 횡단보도
            for a, b, kind in r['crosswalks']:
                if b >= s0 - 1e-6:
                    items.append(Ahead(base + max(a, s0), 'crosswalk', key, max(a, s0),
                                       {'s0': base + a, 's1': base + b, 'kind': kind}))
            for s in r['crosswalk_warn']:
                if s >= s0 - 1e-6:
                    items.append(Ahead(base + s, 'crosswalk_warn', key, s, {}))
            for s in r['yield_marks']:
                if s >= s0 - 1e-6:
                    items.append(Ahead(base + s, 'yield', key, s, {}))
            # 차선(실선/점선) 변경 지점
            for side, prev in (('left', prev_left), ('right', prev_right)):
                for a, b, typ, col, ok in r['left_mark' if side == 'left' else 'right_mark']:
                    if b < s0 - 1e-6:
                        continue
                    state = (typ, ok)
                    if state != prev:
                        items.append(Ahead(base + max(a, s0), 'mark_' + side, key, max(a, s0),
                                           {'type': typ, 'color': col, 'lane_change_ok': ok,
                                            'is_center': (side == 'left' and r['left_is_center'])}))
                        prev = state
                if side == 'left':
                    prev_left = prev
                else:
                    prev_right = prev
            # 교차로 진입/진출
            if r['junction'] != -1 and (i == 0 or self.lanes[lanes[i - 1]]['junction'] == -1) and s0 <= 1e-6:
                items.append(Ahead(base + 0.0, 'junction_in', key, 0.0, {'junction': r['junction']}))
            if r['junction'] != -1 and (i + 1 >= len(lanes) or self.lanes[lanes[i + 1]]['junction'] == -1):
                items.append(Ahead(base + L, 'junction_out', key, L, {'junction': r['junction']}))
            # 경로 이벤트 (회전, 차선변경)
            for ev in events_by_lane.get(key, []):
                if ev['s_in_lane'] >= s0 - 1e-6:
                    items.append(Ahead(base + ev['s_in_lane'], ev['kind'], key, ev['s_in_lane'], dict(ev)))
            # 경로 끝 / 막다른 차로.
            # route_end 는 "경로 차로열이 여기서 끝난다"다 — successor 유무와
            # 무관하다. 예전엔 successor 없는 차로에서만 방출해서, 마지막 경로
            # 차로에 successor 가 있으면(1602/-3 → 1615/1631) 완주 후에도 그대로
            # 달렸다 (2026-08-21: 완주 후 6.94 m/s 로 93 m 초과 주행).
            if i + 1 >= len(lanes):
                items.append(Ahead(base + L, 'route_end', key, L, {}))
            elif not r['next']:
                items.append(Ahead(base + L, 'dead_end', key, L, {}))
            # 다음 차로로 전진. **차선변경 이음매(평행 이웃)는 진행거리가 늘지
            # 않는다** — 같은 물리 구간을 옆 차로에서 이어 보는 것이므로 acc 를
            # 그대로 두고 s 위치도 유지한다. 이걸 successor 처럼 L-s0 를 더하면
            # 전방 정지선/횡단보도/회전까지의 거리가 이음매마다 차로 길이만큼
            # (2026-08-21 경로 실측 12/30/6 m) 과대 보고되고, route['cum_s']
            # (build_route.advance 가 hop 을 0 으로 계상)와도 어긋난다.
            if i + 1 < len(lanes) and lanes[i + 1] not in r['next'] \
                    and lanes[i + 1] in (r['left_nb'], r['right_nb']):
                s0 = min(s0, self.lanes[lanes[i + 1]]['length'])
            else:
                acc += L - s0
                s0 = 0.0
            i += 1
        items = [it for it in items if it.dist <= horizon + 1e-6]
        items.sort(key=lambda it: it.dist)
        # 평행 차로를 같은 s 에서 두 번 훑으므로 같은 물리 지점(정지선/횡단보도
        # 등)이 양쪽 차로에서 한 번씩 나온다. 종류별로 0.5 m 안에 겹치면 중복이다.
        out: List[Ahead] = []
        last_at: Dict[str, float] = {}
        for it in items:
            prev = last_at.get(it.kind)
            if prev is not None and it.dist - prev < 0.5:
                continue
            last_at[it.kind] = it.dist
            out.append(it)
        return out

    def summarize(self, ahead: List[Ahead]) -> Dict[str, Any]:
        """lookahead 결과를 판단 노드가 바로 쓰는 요약치로"""
        out = {'dist_stop_line': None, 'stop_signal_ids': [], 'dist_crosswalk': None,
               'dist_next_turn': None, 'next_turn': None, 'dist_lane_change': None, 'lane_change_dir': None,
               'dist_junction': None, 'dist_dead_end': None, 'dist_route_end': None,
               'dist_junction_out': None, 'in_junction': False, 'speed_changes': []}
        for it in ahead:
            if it.kind == 'stop_line' and out['dist_stop_line'] is None:
                out['dist_stop_line'] = it.dist
                out['stop_signal_ids'] = it.data['signal_ids']
            elif it.kind == 'crosswalk' and out['dist_crosswalk'] is None:
                out['dist_crosswalk'] = it.dist
            elif it.kind in ('turn_left', 'turn_right') and out['dist_next_turn'] is None:
                out['dist_next_turn'] = it.dist
                out['next_turn'] = it.kind
            elif it.kind in ('lane_change_left', 'lane_change_right') and out['dist_lane_change'] is None:
                out['dist_lane_change'] = it.dist
                out['lane_change_dir'] = it.kind
            elif it.kind == 'junction_in' and out['dist_junction'] is None:
                out['dist_junction'] = it.dist
            elif it.kind == 'junction_out' and out['dist_junction_out'] is None:
                out['dist_junction_out'] = it.dist
            elif it.kind == 'dead_end' and out['dist_dead_end'] is None:
                out['dist_dead_end'] = it.dist
            elif it.kind == 'route_end' and out['dist_route_end'] is None:
                out['dist_route_end'] = it.dist
            elif it.kind == 'speed':
                out['speed_changes'].append((it.dist, it.data['limit'], it.data['school_zone']))
        # 교차로 안인가: 진출(junction_out)이 앞에 있는데 진입(junction_in)이 없거나
        # 더 멀면, 이미 연결로 위다. junction 캡·회전 판정의 근거를 로그에서 구분하려고 남긴다.
        if out['dist_junction_out'] is not None:
            out['in_junction'] = (out['dist_junction'] is None
                                  or out['dist_junction'] > out['dist_junction_out'])
        return out


if __name__ == '__main__':
    import sys
    lg = LaneGraph(sys.argv[1] if len(sys.argv) > 1 else 'lane_graph.pkl')
    print(lg.meta)
    x, y = 30.0, 45.0
    m = lg.locate(x, y, yaw=0.3)
    print('locate', (x, y), '→', m)
