#!/usr/bin/env python3
"""
avoid_sim.py ─ 정적 장애물 회피 폐루프 시뮬 (분석·검증 전용)

    python3 tools/avoid_sim.py                 # 10 케이스, 차로 지도 on/off 대조
    python3 tools/avoid_sim.py --cases 2,4     # 일부만
    python3 tools/avoid_sim.py --trace 2       # 한 케이스 틱 덤프

**저장소 코드를 고치지 않는다.** 실제 결정 스택을 그대로 부른다:
`VtdRoutePlanner` + PDM `AutoPilot` + `KrRules.apply` + `VtdLongitudinalController`
+ PDM `LateralPIDController`. 자차 운동만 자전거 모델로 굴린다.

무대는 **실제 지도**다 — road 3113 sec 0 (4차로 · 200 m · κ=0 직선 · 좌우 전부
점선 · 폭 2.8~3.0 m). 합성 레인그래프를 만들지 않으므로 차로 폭·마킹·이웃
관계가 전부 실측값이다. 장애물만 합성이다.

한계: VTD 물리가 아니라 자전거 모델이다. 통신 지연·액추에이터·타이어를 안 본다
(실주행 1차 [2] 추종 시뮬과 같은 얼개, 로그 대비 |t_off| 가 낙관적이다).
**순위와 on/off 차이**를 보는 도구지 절대값을 실주행 예측으로 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'team_code'))

from vtd_adapter.carla_types import TrafficLightState              # noqa: E402
from vtd_adapter.config import load_params_yaml                    # noqa: E402
from vtd_adapter.control import VtdLongitudinalController          # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                        # noqa: E402
from vtd_adapter.route import VtdRoutePlanner                      # noqa: E402
from vtd_adapter import frame                                      # noqa: E402
from vtd_adapter.actor import VtdActor, VtdEgo                     # noqa: E402

ROAD, SEC = 3113, 0
LANES = [(ROAD, SEC, -1), (ROAD, SEC, -2), (ROAD, SEC, -3), (ROAD, SEC, -4)]
HZ = 20.0
DT = 1.0 / HZ


# ── 액터 (저장소 어댑터 타입을 그대로 쓴다) ────────────────────────────────
# 직접 만든 CARLA 스텁을 쓰면 PDM 이 만지는 표면(Transform.transform ·
# bounding_box.location ...)을 하나씩 빠뜨린다. VtdActor/VtdEgo 가 이미 그
# 표면을 정확히 갖고 있으므로 그대로 쓴다 — 실기와 같은 객체다.
class ActorList(list):
    def filter(self, pattern):
        pat = pattern.strip('*')
        return ActorList(a for a in self if pat in a.type_id)


class World:
    """carla.World 중 autopilot 이 쓰는 것만 (get_actors · get_actor).

    `get_actor(id)` 는 선행차 추종(compute_target_speed_wrt_leading_vehicle)이
    id 로 되찾을 때 쓴다 — 없으면 첫 장애물에서 바로 죽는다.
    """

    def __init__(self, actors):
        self._a = ActorList(actors)
        self._by_id = {a.id: a for a in self._a}

    def get_actors(self):
        return self._a

    def get_actor(self, actor_id):
        return self._by_id.get(actor_id)


class Map:
    def get_waypoint(self, _loc):
        return type('W', (), {'is_junction': False})()


def make_obj(aid, x_carla, y_carla, yaw_deg, speed=0.0, length=4.5, width=1.9):
    a = VtdActor(aid, 'vehicle')
    a.update(x_carla, y_carla, 0.0, yaw_deg, speed, length, width, 1.5)
    return a


# ── 무대 ──────────────────────────────────────────────────────────────────
def build_route(lg, lane):
    """road 3113 의 한 차로짜리 경로 dict (build_route 산출과 같은 형식)."""
    L = lg.length(lane)
    return {'lanes': [lane], 'cum_s': [0.0], 'lengths': [L], 'total_length': L,
            'start_s_in_lane': 0.0, 'events': [], 'waypoints': [], 'waypoint_s': [],
            'junction_segments': [], 'segment_span': [], 'infeasible_forced': [],
            'finish_xy': None}


def lane_xy(lg, lane, s, lat=0.0):
    """차로 s 지점에서 횡으로 lat [m] 옮긴 VTD 좌표 (+좌 / −우)."""
    x, y, _z, h = lg.point_at(lane, float(np.clip(s, 0.0, lg.length(lane))))
    return x - math.sin(h) * lat, y + math.cos(h) * lat


class Sim:
    def __init__(self, cfg, lg, ego_lane, actors, v0=8.33, ticks=900, setup=None):
        from autopilot import AutoPilot
        from kr_rules import KrRules
        from lateral_controller import LateralPIDController
        from run_agent import build_pdm_config

        self.cfg, self.lg, self.ego_lane = cfg, lg, ego_lane
        self.route = build_route(lg, ego_lane)
        pdm = build_pdm_config(cfg)
        self.planner = VtdRoutePlanner(lg, self.route, cfg, pdm)
        self.ego = VtdEgo(cfg['vehicle'])
        self.actors = actors
        self.ap = AutoPilot()
        self.ap.setup(world=World([self.ego] + actors), world_map=Map(),
                      waypoint_planner=self.planner,
                      longitudinal_controller=VtdLongitudinalController(cfg),
                      ego_vehicle=self.ego, config=pdm)
        self.ap.kr_rules = KrRules(cfg)
        self.ap.kr_rules._sl_all = []
        self.lat = LateralPIDController(pdm)
        if setup is not None:
            # 케이스별 무대 손질 (적신호·실선·다음 회전). 무대가 4차로 직선 점선
            # 하나뿐이라, 그 조건들은 여기서 **명시적으로** 얹지 않으면 케이스 1과
            # 같은 상황이 된다 — 실제로 그래서 6·7·8 이 1 과 똑같은 답을 냈다.
            setup(self)
        self.ticks = ticks
        self.v0 = v0

    def run(self):
        lg, lane = self.lg, self.ego_lane
        x, y = lane_xy(lg, lane, 3.0)
        _x2, _y2, _z, h = lg.point_at(lane, 3.0)
        cx, cy = frame.to_carla_xy(x, y)
        yaw = frame.to_carla_yaw_rad(h)
        v = self.v0
        rec = {'t_off': 0.0, 'sat': 0, 'min_gap': 9e9, 'signal': set(),
               'shift_v': None, 'shift_s': None, 'lane_hist': [], 'stopped': 0,
               'v_min': 9e9, 'end_s': 0.0, 'collide': 0}
        half_ego = self.cfg['vehicle']['width'] / 2.0
        prev_span = None
        for _i in range(self.ticks):
            self.ego.update(cx, cy, 0.0, math.degrees(yaw), v,
                            self.ego.length, self.ego.width, self.ego.height)
            out = self.planner.run_step(np.array([cx, cy, 0.0]))
            rp = out[0]
            # tick_data 는 autopilot 이 ego 객체에서 직접 만든다 (input_data 무관)
            ctrl = self.ap.run_step({}, _i * DT)
            accel = float(getattr(ctrl, 'accel', getattr(ctrl, 'throttle', 0.0)))
            st = float(np.clip(getattr(ctrl, 'steer', 0.0), -1.0, 1.0))
            if abs(st) > 0.999:
                rec['sat'] += 1
            sig = int(getattr(self.ap.kr_rules, 'last_turn_signal', 0) or 0)
            if sig:
                rec['signal'].add(sig)
            span = self.ap.kr_rules.ot_span
            if span is not None and prev_span is None:      # 시프트가 붙은 틱
                rec['shift_v'] = v
                rec['shift_s'] = float(self.planner.route_s[self.planner.route_index])
            prev_span = span
            # 운동 (자전거 모델)
            delta = st * float(self.cfg['vehicle']['max_steer'])
            v = max(0.0, v + accel * DT)
            cx += v * math.cos(yaw) * DT
            cy += v * math.sin(yaw) * DT
            yaw += v / float(self.cfg['vehicle']['wheelbase']) * math.tan(delta) * DT
            # 계측 — t_off 는 **자차가 실제로 있는 차로**의 중심선까지 (차로유지 축)
            vx, vy = frame.from_carla_xy(cx, cy)
            best = min((abs(lg.project(k, vx, vy)[1]), k) for k in LANES)
            rec['t_off'] = max(rec['t_off'], best[0])
            rec['lane_hist'].append(best[1])
            for a in self.actors:
                avx, avy = frame.from_carla_xy(a.x, a.y)
                d = math.hypot(vx - avx, vy - avy)
                gap = d - (half_ego + a.width / 2.0)
                rec['min_gap'] = min(rec['min_gap'], gap)
                if gap <= 0.0:
                    rec['collide'] += 1
            rec['v_min'] = min(rec['v_min'], v)
            if v < 0.3:
                rec['stopped'] += 1
            rec['end_s'] = float(self.planner.route_s[self.planner.route_index])
            if rec['end_s'] > lg.length(lane) - 8.0:
                break
        rec['lane_end'] = rec['lane_hist'][-1] if rec['lane_hist'] else None
        rec['lanes_used'] = sorted({k[2] for k in rec['lane_hist']})
        return rec


# ── 무대 손질 (케이스 6·7·8) ──────────────────────────────────────────────
class FakeTL:
    """`_next_stopline` 이 읽는 최소 표면 — state.name 과 id 뿐이다."""

    def __init__(self, name, tl_id=1):
        self.state = type('S', (), {'name': name})()
        self.id = tl_id


def red_light_at(s_line):
    """정지선을 s_line 에 두고 **적색**으로 고정한다 (케이스 6: 신호 대기 줄).

    `_red_ahead` 가 참이 되어 회피가 거리 무관으로 억제되는지를 본다.
    """
    def setup(sim):
        n = len(sim.planner.route_s)
        sim.planner.distances_to_next_traffic_lights = np.maximum(
            0.0, float(s_line) - np.asarray(sim.planner.route_s, dtype=float))
        sim.planner.next_traffic_lights = [FakeTL('Red')] * n
    return setup


def solid_line(side='right'):
    """그 방향 마킹을 **실선**으로 만든다 (케이스 7).

    `lg.dashed_runs` 가 차선변경 허용 구간의 단일 출처라 거기만 비우면 된다
    (build_route 와 제어기 플래너가 같은 함수를 본다).
    """
    def setup(sim):
        lg = sim.lg
        orig = lg.dashed_runs
        def patched(key, sd, _o=orig):
            return [] if sd == side else _o(key, sd)
        sim.lg.dashed_runs = patched
    return setup


def next_turn_left(s_turn):
    """경로에 **다음 좌회전** 이벤트를 심는다 (케이스 8).

    never_stall 의 회전 여유(never_stall_turn_margin_m)와 복귀 데드라인이
    이 이벤트를 읽는다.
    """
    def setup(sim):
        sim.planner.route['events'] = [{'kind': 'turn_left', 's': float(s_turn)}]
        sim.route['events'] = sim.planner.route['events']
    return setup


# ── 케이스 ────────────────────────────────────────────────────────────────
def obj_at(lg, lane, s, aid, lat=0.0, **kw):
    """차로 lane 의 s 지점(횡 lat) 에 정지 객체. 좌표는 CARLA 프레임으로 넘긴다."""
    x, y = lane_xy(lg, lane, s, lat)
    _x, _y, _z, h = lg.point_at(lane, s)
    cx, cy = frame.to_carla_xy(x, y)
    return make_obj(aid, cx, cy, frame.to_carla_yaw_deg(h), **kw)


def cases(lg):
    """(번호, 이름, ego 차로, 장애물들, 기대) — NIGHT 파일 목록 그대로."""
    L1, L2, L3, L4 = LANES                                  # -1 .. -4
    return [
        (1, '단독 장애물 (내 차로 60 m)', L1, [obj_at(lg, L1, 60, 2)],
         '옆 차로로 비켜 통과'),
        (2, '대회장: 1·2차로 나란히, 3차로 빔', L1,
         [obj_at(lg, L1, 60, 2), obj_at(lg, L2, 62, 3)], '두 칸으로 3차로, 충돌 0'),
        (3, '엇갈림: 1차로 60, 2차로 100', L1,
         [obj_at(lg, L1, 60, 2), obj_at(lg, L2, 100, 3)], '2차로 → 다시 1차로'),
        (4, '걸쳐 선 차 (1·2차로 경계), 3차로 빔', L1,
         [obj_at(lg, L1, 60, 2, lat=-1.5)], '3차로로 (두 차로 다 막힘)'),
        (5, '연속 3대 30 m 간격', L1,
         [obj_at(lg, L1, 60, 2), obj_at(lg, L1, 90, 3), obj_at(lg, L1, 120, 4)],
         '옆 차로 유지, 복귀 없음'),
        (6, '신호 대기 줄 (적색, 정지선 앞 3대)', L1,
         [obj_at(lg, L1, 60, 2), obj_at(lg, L1, 68, 3), obj_at(lg, L1, 76, 4)],
         '추월 안 함 (큐)'),
        (7, '실선 구간', L1, [obj_at(lg, L1, 60, 2)], '시프트 안 함 → standoff'),
        (8, '다음 좌회전, 우측 회피', L1, [obj_at(lg, L1, 60, 2)],
         '데드라인 전 복귀'),
        (9, '양쪽 다 막힘', L2,
         [obj_at(lg, L2, 60, 2), obj_at(lg, L1, 62, 3), obj_at(lg, L3, 62, 4)],
         'never_stall'),
        (10, '먼 차로에만 장애물 (내 차로 비었음)', L1, [obj_at(lg, L4, 60, 2)],
         '반응 없음 (트리거 아님)'),
    ]


def cfg_with(base, lane_map: bool):
    import copy
    c = copy.deepcopy(base)
    c['avoid_map']['lane_map_avoid_enable'] = lane_map
    return c


def main():
    ap = argparse.ArgumentParser(description='정적 장애물 회피 폐루프 시뮬')
    ap.add_argument('--graph', default='data/lane_graph.pkl')
    ap.add_argument('--cases', default='')
    ap.add_argument('--v0', type=float, default=8.33)
    ap.add_argument('--trace', type=int, default=0)
    a = ap.parse_args()

    cfg = load_params_yaml()
    lg = LaneGraph(a.graph, cfg=cfg)
    want = {int(x) for x in a.cases.split(',')} if a.cases else None

    print(f'무대: road {ROAD} sec {SEC} — 4차로 · {lg.length(LANES[0]):.0f} m · κ=0 직선 · 점선')
    print(f'진입속도 {a.v0:.2f} m/s ({a.v0 * 3.6:.0f} km/h)\n')
    hdr = (f"{'#':>2} {'케이스':<34} "
           f"{'off: 충돌':>8} {'|t_off|':>8} {'포화':>5} {'차로':>10} | "
           f"{'on: 충돌':>8} {'|t_off|':>8} {'포화':>5} {'차로':>10}")
    print(hdr); print('─' * len(hdr))
    for no, name, lane, objs, expect in cases(lg):
        if want and no not in want:
            continue
        row = []
        for lm in (False, True):
            objs2 = [make_obj(o.id, o.x, o.y, o.yaw_deg, o.speed, o.length, o.width)
                     for o in objs]
            try:
                r = Sim(cfg_with(cfg, lm), lg, lane, objs2, v0=a.v0).run()
                row.append(r)
            except Exception as e:                          # noqa: BLE001
                row.append({'err': f'{type(e).__name__}: {e}'})
        def fmt(r):
            if 'err' in r:
                return f"{'실패':>8} {r['err'][:26]:>34}"
            return (f"{r['collide']:>8} {r['t_off']:>8.2f} {r['sat']:>5} "
                    f"{str(r['lanes_used']):>10}")
        print(f'{no:>2} {name:<34} {fmt(row[0])} | {fmt(row[1])}')
        print(f"{'':>2} {'기대: ' + expect:<34}")
    print('\n주의: 자전거 모델 폐루프다 (통신 지연·액추에이터·타이어 없음).')
    print('      절대값이 아니라 on/off 차이와 순위를 본다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
