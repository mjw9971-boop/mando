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
        # 틱 훅 — 무대 손질이 **시간에 따라** 변해야 하는 케이스용 (신호가 도중에
        # 바뀌는 12번). setup 이 sim.on_tick 에 함수를 꽂는다.
        self.on_tick = None
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
               'v_min': 9e9, 'end_s': 0.0, 'collide': 0, 'shift_n': 0,
               'shift_pass': None, 'solid_relaxed': False, 'states': set(),
               'never_stall': False,
               # 12번용 — "얼어붙었나" 는 최대 t_off 가 아니라 **끝난 자리**와
               # **한 번에 얼마나 오래 섰나** 로 봐야 안다.
               't_off_end': 0.0, 'stop_max_s': 0.0, 'cause_why': {},
               'arm_t_off': None, 'arm_s': None, 'd_end_at_freeze': None}
        half_ego = self.cfg['vehicle']['width'] / 2.0
        prev_span = None
        run_stop = 0
        for _i in range(self.ticks):
            self.ego.update(cx, cy, 0.0, math.degrees(yaw), v,
                            self.ego.length, self.ego.width, self.ego.height)
            out = self.planner.run_step(np.array([cx, cy, 0.0]))
            rp = out[0]
            if self.on_tick is not None:
                self.on_tick(self, _i,
                             float(self.planner.route_s[self.planner.route_index]))
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
                rec['shift_n'] += 1
                rec['shift_pass'] = (self.ap.kr_rules.last_avoid or {}).get('pass')
                rec['solid_relaxed'] = bool(
                    (self.ap.kr_rules.last_avoid or {}).get('solid_relaxed'))
            prev_span = span
            la = self.ap.kr_rules.last_avoid or {}
            if la.get('state'):
                rec['states'].add(la['state'])
            if (self.ap.kr_rules.ns_info or {}).get('state'):
                rec['never_stall'] = True
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
                run_stop += 1
                rec['stop_max_s'] = max(rec['stop_max_s'], run_stop / HZ)
                if rec['d_end_at_freeze'] is None and run_stop > 20:
                    rec['d_end_at_freeze'] = self.ap.kr_rules.last_d_end
            else:
                run_stop = 0
            rec['t_off_end'] = best[0]
            w = (self.ap.kr_rules.last_avoid or {}).get('cause_why')
            if w:
                rec['cause_why'][w] = rec['cause_why'].get(w, 0) + 1
            rec['end_s'] = float(self.planner.route_s[self.planner.route_index])
            if rec['end_s'] > lg.length(lane) - 8.0:
                break
        rec['lane_end'] = rec['lane_hist'][-1] if rec['lane_hist'] else None
        rec['lanes_used'] = sorted({k[2] for k in rec['lane_hist']})
        return rec


# ── 무대 손질 (케이스 6·7·8) ──────────────────────────────────────────────
def make_tl(state_name, route_s, tl_id=1):
    """실제 `VtdTrafficLight` 를 그대로 쓴다 — 스텁을 만들면 PDM 이 만지는
    표면(type_id 등)을 하나씩 빠뜨린다 (액터와 같은 원칙)."""
    from vtd_adapter.route import VtdTrafficLight
    from vtd_adapter.carla_types import TrafficLightState
    tl = VtdTrafficLight([tl_id], [tl_id], float(route_s))
    tl.state = getattr(TrafficLightState, state_name)
    return tl


def red_light_at(s_line):
    """정지선을 s_line 에 두고 **적색**으로 고정한다 (케이스 6: 신호 대기 줄).

    `_red_ahead` 가 참이 되어 회피가 거리 무관으로 억제되는지를 본다.
    """
    def setup(sim):
        n = len(sim.planner.route_s)
        sim.planner.distances_to_next_traffic_lights = np.maximum(
            0.0, float(s_line) - np.asarray(sim.planner.route_s, dtype=float))
        tl = make_tl('Red', s_line)
        sim.planner.next_traffic_lights = [tl] * n
    return setup


def red_light_turns_at(s_line, arm_s, red_s=None, green_s=None):
    """정지선을 `s_line` 에 두고, 자차가 `arm_s` 를 지나는 순간 **녹→적**으로 바꾼다.

    `red_light_at` 은 처음부터 적색이라 회피가 아예 시작되지 않는다. 12번이
    재현하려는 것은 그 반대다 — **시프트가 이미 진행 중일 때** 적신호가 켜지는
    조합이다 (run_20260909_232350 t 60.7: SHIFT_ACTIVE → SHIFT_HOLD,
    t_off 1.27 에서 굳음, 정지선 99.8 m 앞).

    `distances_to_next_traffic_lights` 는 정적이라 한 번만 깔고, 신호 **상태**만
    틱 훅에서 바꾼다 (VtdTrafficLight 객체 하나를 공유하므로 그 객체의 state 를
    바꾸면 전 틱에 반영된다).
    """
    def setup(sim):
        n = len(sim.planner.route_s)
        sim.planner.distances_to_next_traffic_lights = np.maximum(
            0.0, float(s_line) - np.asarray(sim.planner.route_s, dtype=float))
        tl = make_tl('Green', s_line)
        sim.planner.next_traffic_lights = [tl] * n

        # 켜진 뒤 red_s 초마다 녹↔적을 번갈아 준다 (green_s 가 녹색 길이).
        # 231730 은 적 13 s / 녹 13 s 로 다섯 주기를 돌았고, **녹색 구간에도**
        # v_target 0 이었다 — 그래서 주기를 돌려 봐야 "신호가 원인인가" 가 갈린다.
        st = {'armed_i': None}

        def tick(sm, i, route_s):
            if st['armed_i'] is None:
                if route_s >= float(arm_s):
                    st['armed_i'] = i
                    tl.state = TrafficLightState.Red
                return
            if red_s is None:
                return
            k = int((i - st['armed_i']) / (float(red_s) * HZ))
            g = green_s if green_s is not None else red_s
            period = float(red_s) + float(g)
            dt = (i - st['armed_i']) / HZ
            tl.state = (TrafficLightState.Red if (dt % period) < float(red_s)
                        else TrafficLightState.Green)
        sim.on_tick = tick
    return setup


def solid_line(side='right'):
    """그 방향 마킹을 **실선**으로 바꾼다 (케이스 7).

    `dashed_runs` 만 비우면 안 된다 — `kr_rules._crossable_runs` 가 마킹
    타입이 `'none'`(선 없음) 인 조각도 넘을 수 있는 구간으로 더한다. 그래서
    **원본 마크 데이터**를 실선으로 바꾼다 (실제 실선 도로와 같은 상태).
    """
    field = 'left_mark' if side == 'left' else 'right_mark'
    def setup(sim):
        for k in LANES:
            marks = sim.lg.lanes[k].get(field) or []
            sim.lg.lanes[k][field] = [(a, b, 'solid', c, False)
                                      for a, b, _t, c, _ok in marks]
        # 반대편 차로가 보는 같은 경계도 같이 바꿔야 한다 (좌/우 한 쌍)
        other = 'left_mark' if field == 'right_mark' else 'right_mark'
        for k in LANES:
            marks = sim.lg.lanes[k].get(other) or []
            sim.lg.lanes[k][other] = [(a, b, 'solid', c, False)
                                      for a, b, _t, c, _ok in marks]
    return setup


def next_turn_left(s_turn, valid_lane=None):
    """경로에 **다음 좌회전**과 그 **유효 진입 차로**를 심는다 (케이스 8).

    복귀 데드라인(커밋 C)과 never_stall (c) 는 둘 다 route.pkl 의
    `valid_entry_lanes` 를 읽는다 — 그게 없으면 "제약 없음" 이라 데드라인이
    아예 생기지 않는다. 무대 경로는 build_route 산출이 아니라 손으로 만든
    한 차로짜리라 이 필드가 비어 있어서, 케이스 8 이 커밋 C 를 전혀 못 밟았다.
    """
    def setup(sim):
        lane = valid_lane or LANES[0]
        sim.planner.route['events'] = [{'kind': 'turn_left', 's': float(s_turn)}]
        sim.planner.route['waypoint_s'] = [0.0, float(s_turn)]
        sim.planner.route['valid_entry_lanes'] = [
            {'seg': 0, 'target': 'pair', 'turn': 'left', 'lanes': [list(lane)]}]
        sim.route.update(sim.planner.route)
    return setup


# ── 케이스 ────────────────────────────────────────────────────────────────
def obj_at(lg, lane, s, aid, lat=0.0, **kw):
    """차로 lane 의 s 지점(횡 lat) 에 정지 객체. 좌표는 CARLA 프레임으로 넘긴다."""
    x, y = lane_xy(lg, lane, s, lat)
    _x, _y, _z, h = lg.point_at(lane, s)
    cx, cy = frame.to_carla_xy(x, y)
    return make_obj(aid, cx, cy, frame.to_carla_yaw_deg(h), **kw)


def cases(lg):
    """(번호, 이름, ego 차로, 장애물들, 기대, 무대손질) — NIGHT 파일 목록 그대로.

    무대가 4차로 직선·점선 하나뿐이라 적신호(6)·실선(7)·다음 좌회전(8)은
    **무대손질 콜백으로 명시**해야 한다. 안 하면 셋 다 케이스 1 과 같은 상황이
    되어 같은 답이 나온다 (2026-09-09 처음 돌렸을 때 실제로 그랬다).
    """
    L1, L2, L3, L4 = LANES                                  # -1 .. -4
    return [
        (1, '단독 장애물 (내 차로 60 m)', L1, [obj_at(lg, L1, 60, 2)],
         '옆 차로로 비켜 통과', None),
        (2, '대회장: 1·2차로 나란히, 3차로 빔', L1,
         [obj_at(lg, L1, 60, 2), obj_at(lg, L2, 62, 3)], '두 칸으로 3차로, 충돌 0', None),
        (3, '엇갈림: 1차로 60, 2차로 100', L1,
         [obj_at(lg, L1, 60, 2), obj_at(lg, L2, 100, 3)], '2차로 → 다시 1차로', None),
        (4, '걸쳐 선 차 (1·2차로 경계), 3차로 빔', L1,
         [obj_at(lg, L1, 60, 2, lat=-1.5)], '3차로로 (두 차로 다 막힘)', None),
        (5, '연속 3대 30 m 간격', L1,
         [obj_at(lg, L1, 60, 2), obj_at(lg, L1, 90, 3), obj_at(lg, L1, 120, 4)],
         '옆 차로 유지, 복귀 없음', None),
        (6, '신호 대기 줄 (적색, 정지선 앞 3대)', L1,
         [obj_at(lg, L1, 60, 2), obj_at(lg, L1, 68, 3), obj_at(lg, L1, 76, 4)],
         '추월 안 함 (큐)', red_light_at(84.0)),
        (7, '실선 구간', L1, [obj_at(lg, L1, 60, 2)], '시프트 안 함 → standoff',
         solid_line('right')),
        (8, '다음 좌회전, 우측 회피', L1, [obj_at(lg, L1, 60, 2)],
         '데드라인 전 복귀', next_turn_left(150.0, valid_lane=L1)),
        (9, '양쪽 다 막힘', L2,
         [obj_at(lg, L2, 60, 2), obj_at(lg, L1, 62, 3), obj_at(lg, L3, 62, 4)],
         'never_stall', None),
        (10, '먼 차로에만 장애물 (내 차로 비었음)', L1, [obj_at(lg, L4, 60, 2)],
         '반응 없음 (트리거 아님)', None),
        # 11 은 지시안 [1] 의 재현이다 — 두 칸을 골랐는데 **중간 차로가 점유**돼
        # 옛 게이트(occupied)가 최종 목표가 아니라 중간 차로를 보고 기각하던 상황.
        # 중간 차로 객체를 clear_radius_m(30) **안**에 둬야 그 게이트가 실제로 걸린다
        # (케이스 2 는 62 m 라 반경 밖이어서 이 경로를 안 밟는다).
        (11, '두 칸 필요 + 중간 차로가 코앞에서 점유', L1,
         [obj_at(lg, L1, 55, 2), obj_at(lg, L2, 22, 3), obj_at(lg, L2, 70, 4)],
         '3차로로 (중간은 지나갈 뿐)', None),
        # 12 는 run_20260909_232350 의 재현이다 — **시프트 중(t_off≈1.3)에 93 m
        # 앞이 적색으로 바뀌는** 조합. 그 로그는 SHIFT_HOLD 로 굳어 램프 중간
        # (t_off 2.08 = 차로선 위)에서 70 s 정지했다. 얼린 것은 적신호가 아니라
        # 크립 게이트였다(`_obstacle_cause` 의 종점 배제 150 m). 그래서 이 케이스는
        # **[1] 스위치 축**(--axis creep_end)으로 돌려야 의미가 있다.
        # 배치는 d_end 를 150 밑으로 만들지 못했다(경로 3 km) — 여기 무대는 200 m
        # 라 정지 지점에서 d_end 가 자연히 150 안이다. 실측 조건과 같아진다.
        # 장애물이 **둘**이라야 재현된다: 내 차로(A)를 피해 옆 차로로 램프를
        # 그리는 도중, 그 옆 차로의 B 가 standoff 대상이 되어 **램프 중간에** 선다.
        # 232350 이 정확히 이 꼴이었다 — 왼쪽으로 갔는데 왼쪽도 막혀 있었다
        # (결정 시점 blocker id 4 → 시프트 후 blocker id 2, s_rel 20.5).
        (12, '시프트 중 93 m 앞 적신호 (232350 재현)', L1,
         [obj_at(lg, L1, 100, 2), obj_at(lg, L2, 110, 3)],
         '램프를 끝내고 차로 안으로',
         red_light_turns_at(181.0, 80.0, red_s=13.0, green_s=13.0)),
    ]


def cfg_with(base, on: bool, axis: str = 'lane_map'):
    """비교 축을 켜고 끈다.

    `lane_map` — 커밋 B·C 를 **같이** 켠다 (한 기능의 앞뒤다: 후보를 고르는
    쪽과 고른 차로에 머무는 쪽).
    `creep_end` — [1] `standoff_creep_end_narrow_enable` 만 켠다. 크립 게이트의
    종점 배제 폭을 never_stall 과 같은 축(unlatch_m 30)으로 좁힌다.
    `creep_fix` — 위에 `standoff_creep_delay_pause_enable` 까지 같이 켠다.
    `geom_b` — 전이 길이를 속도로 재계산 (조향 바닥 포함). **채택하지 않았다** —
    조향 바닥을 제대로 넣으면 geom 기각의 2 % 만 풀린다 (아래 커밋 메시지 참조).
    `side_clear` — (3) `side_clear_by_map_enable`. 지도는 양쪽 다 켜 둔다
    (지도가 꺼져 있으면 이 스위치가 아무 일도 안 하므로 대조가 성립하지 않는다).
    """
    import copy
    c = copy.deepcopy(base)
    if axis == 'creep_end':
        c['overtake']['standoff_creep_end_narrow_enable'] = on
    elif axis == 'geom_b':
        # 전이 길이를 속도로 (조향 바닥 포함). 채택하지 않은 축이지만 대조용으로 남긴다.
        c['overtake']['trans_m_by_speed_enable'] = on
    elif axis == 'side_clear':
        # (3) 목표 차로 점유 판정을 free_run 으로. 지도가 켜져 있어야 의미가 있다.
        c['avoid_map']['lane_map_avoid_enable'] = True
        c['avoid_map']['lane_map_no_return_enable'] = True
        c['overtake']['side_clear_by_map_enable'] = on
    elif axis == 'ramp_phys':
        # (4) 복귀 차로 참조가 램프·v_cap 까지 정하는 것을 끊는다. 이 축은
        # **밤에 켠 13개를 양쪽 다 켜 둔 위에서** 비교해야 의미가 있다 —
        # shift_ref 가 꺼져 있으면 mine 이 애초에 물리 차로라 차이가 0 이다.
        for k in ('lane_map_avoid_enable', 'lane_map_no_return_enable',
                  'lane_map_retarget_enable', 'lane_map_owns_shift_enable',
                  'lane_map_shift_ref_enable', 'lane_map_fallback_side_enable',
                  'lane_map_queue_mark_enable', 'lane_map_shift_on_pick_enable',
                  'lane_map_retarget_in_hold_enable'):
            c['avoid_map'][k] = True
        for k in ('standoff_creep_end_narrow_enable',
                  'standoff_creep_delay_pause_enable',
                  'queue_close_gap_enable', 'side_clear_by_map_enable'):
            c['overtake'][k] = True
        c['avoid_map']['lane_map_ramp_by_phys_lane_enable'] = on
    elif axis == 'prefer_dashed':
        # (2) 후보 순위에서 점선 쪽을 free 다음으로. ramp_phys 와 같은 바탕
        # (밤에 켠 13개 + (4) 수정) 위에서 이 스위치만 가른다.
        for k in ('lane_map_avoid_enable', 'lane_map_no_return_enable',
                  'lane_map_retarget_enable', 'lane_map_owns_shift_enable',
                  'lane_map_shift_ref_enable', 'lane_map_fallback_side_enable',
                  'lane_map_queue_mark_enable', 'lane_map_shift_on_pick_enable',
                  'lane_map_retarget_in_hold_enable',
                  'lane_map_ramp_by_phys_lane_enable'):
            c['avoid_map'][k] = True
        for k in ('standoff_creep_end_narrow_enable',
                  'standoff_creep_delay_pause_enable',
                  'queue_close_gap_enable', 'side_clear_by_map_enable'):
            c['overtake'][k] = True
        c['avoid_map']['lane_map_prefer_dashed_enable'] = on
    elif axis == 'creep_fix':
        # [1] 두 조각을 같이 켠다 — 종점 배제 폭(route_end)과 지연 시계 누적.
        # 둘 중 하나만으로는 안 풀린다 (앞의 것을 풀면 뒤의 것이 이어받는다).
        c['overtake']['standoff_creep_end_narrow_enable'] = on
        c['overtake']['standoff_creep_delay_pause_enable'] = on
    else:
        c['avoid_map']['lane_map_avoid_enable'] = on
        c['avoid_map']['lane_map_no_return_enable'] = on
    return c


def main():
    ap = argparse.ArgumentParser(description='정적 장애물 회피 폐루프 시뮬')
    ap.add_argument('--graph', default='data/lane_graph.pkl')
    ap.add_argument('--cases', default='')
    ap.add_argument('--v0', type=float, default=8.33)
    ap.add_argument('--trace', type=int, default=0)
    ap.add_argument('--ticks', type=int, default=900,
                    help='케이스당 최대 틱 (기본 900 = 45 s). 신호 주기를 여러 번 '
                         '보려면 늘린다 (12번).')
    ap.add_argument('--axis', default='lane_map',
                    choices=('lane_map', 'creep_end', 'creep_fix', 'side_clear',
                             'geom_b', 'ramp_phys', 'prefer_dashed'),
                    help='off/on 으로 비교할 스위치 축')
    a = ap.parse_args()

    cfg = load_params_yaml()
    lg = LaneGraph(a.graph, cfg=cfg)
    want = {int(x) for x in a.cases.split(',')} if a.cases else None

    print(f'무대: road {ROAD} sec {SEC} — 4차로 · {lg.length(LANES[0]):.0f} m · κ=0 직선 · 점선')
    print(f'비교 축: {a.axis}')
    print(f'진입속도 {a.v0:.2f} m/s ({a.v0 * 3.6:.0f} km/h)\n')
    hdr = (f"{'#':>2} {'케이스':<30} "
           f"{'off 충돌':>7} {'|t_off|':>7} {'포화':>4} {'시프트':>6} {'차로':>12} "
           f"{'최저v':>6} {'정지s':>6} | "
           f"{'on 충돌':>7} {'|t_off|':>7} {'포화':>4} {'시프트':>6} {'차로':>12} "
           f"{'최저v':>6} {'정지s':>6}")
    print(hdr); print('─' * len(hdr))
    # 케이스 setup 이 무대(LaneGraph)를 덮어쓰므로 원본 마킹을 떠 둔다.
    _mark0 = {k: (list(lg.lanes[k].get('left_mark') or []),
                  list(lg.lanes[k].get('right_mark') or []))
              for k in lg.lanes}
    for no, name, lane, objs, expect, setup in cases(lg):
        if want and no not in want:
            continue
        row = []
        for lm in (False, True):
            objs2 = [make_obj(o.id, o.x, o.y, o.yaw_deg, o.speed, o.length, o.width)
                     for o in objs]
            # **무대를 매 케이스 처음 상태로 되돌린다.** 케이스 7 의 setup 이
            # `lg.lanes[k]['left_mark']` 를 실선으로 **영구히** 덮어써서, 한 번
            # 돌고 나면 8~12 가 전부 실선 무대에서 돌았다 (2026-09-10 발견:
            # 12 를 단독으로 돌리면 1p1 · 정지 0.0 s 인데 전 케이스 sweep 에서는
            # 1p2! · 180.4 s 로 나온다 — 같은 설정·같은 코드인데 순서 때문이다).
            for _k, _rec in _mark0.items():
                lg.lanes[_k]['left_mark'] = list(_rec[0])
                lg.lanes[_k]['right_mark'] = list(_rec[1])
            try:
                # 12 는 신호 주기(적 13 s + 녹 13 s)를 **여러 번** 넘겨야
                # 차이가 드러난다 — 기본 900틱(45 s)은 한 주기가 채 안 돼
                # off·on 이 똑같이 나온다. 케이스가 자기 요구 틱을 갖는다.
                ticks = max(a.ticks, 4000) if no == 12 else a.ticks
                r = Sim(cfg_with(cfg, lm, a.axis), lg, lane, objs2,
                        v0=a.v0, ticks=ticks, setup=setup).run()
                row.append(r)
            except Exception as e:                          # noqa: BLE001
                row.append({'err': f'{type(e).__name__}: {e}'})
        def fmt(r):
            if 'err' in r:
                return f"{'실패':>7} {r['err'][:40]:>45}"
            sh = ('—' if not r['shift_n']
                  else f"{r['shift_n']}p{r['shift_pass']}" + ('!' if r['solid_relaxed'] else ''))
            return (f"{r['collide']:>7} {r['t_off']:>7.2f} {r['sat']:>4} {sh:>6} "
                    f"{str(r['lanes_used']):>12} {r['v_min']:>6.2f} "
                    f"{r['stopped'] / HZ:>6.1f}")

        def tail(r):
            """끝난 자리 — "램프를 끝내고 차로 안에 들어갔나" 는 최대 t_off 로는
            알 수 없다. 마지막 틱의 |t_off| 와 최장 연속 정지를 같이 본다."""
            if 'err' in r:
                return ''
            cw = ','.join('%s %d' % kv for kv in sorted(r['cause_why'].items(),
                                                        key=lambda x: -x[1])[:2])
            return (f"끝 |t_off| {r['t_off_end']:.2f} · 최장정지 {r['stop_max_s']:.1f} s"
                    f" · 도달 {r['end_s']:.0f} m"
                    + (f" · cause_why {cw}" if cw else '')
                    + (f" · d_end(정지) {r['d_end_at_freeze']:.0f} m"
                       if r.get('d_end_at_freeze') is not None else ''))
        print(f'{no:>2} {name:<30} {fmt(row[0])} | {fmt(row[1])}')
        ns = ' · never_stall' if row[0].get('never_stall') else ''
        st = ','.join(sorted(row[0].get('states') or [])) if 'err' not in row[0] else ''
        print(f"{'':>2} {'기대: ' + expect:<30}   off 상태: {st}{ns}")
        if no == 12:
            print(f"{'':>2} {'':<30}   off  {tail(row[0])}")
            print(f"{'':>2} {'':<30}   on   {tail(row[1])}")
    print('\n주의: 자전거 모델 폐루프다 (통신 지연·액추에이터·타이어 없음).')
    print('      절대값이 아니라 on/off 차이와 순위를 본다.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
