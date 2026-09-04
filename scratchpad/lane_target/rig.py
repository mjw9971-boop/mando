"""실제 VtdRoutePlanner + LaneGraph + KrRules 로 _side_pass 를 그대로 호출하는 리그.

경로 pkl 이 없는 로그(104648/104807 등)용으로, 그 시점 자차 차로의 **열 사슬**을
route dict 으로 만들어 실제 플래너를 세운다. 차로 기하는 lane_graph 원본이다.
"""
import sys, copy, math, json
sys.path.insert(0, '.'); sys.path.insert(0, 'team_code'); sys.path.insert(0, 'tests')
import numpy as np, yaml
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.route import VtdRoutePlanner
from vtd_adapter import frame
from run_agent import build_pdm_config
from kr_rules import KrRules

def load_cfg(**over):
    c = yaml.safe_load(open('config/params.yaml'))
    for k, v in over.items():
        c['overtake'][k] = v
    return c

def chain_lanes(lg, start, n=8):
    out=[start]; k=start
    for _ in range(n):
        nx=lg.successors(k)
        if not nx: break
        k=nx[0]
        if k in out: break
        out.append(k)
    return out

def make_route(lg, first_lane, n=8):
    lanes = chain_lanes(lg, first_lane, n)
    lengths=[lg.length(k) for k in lanes]
    cum=[0.0]
    for L in lengths[:-1]: cum.append(cum[-1]+L)
    return {'lanes': lanes, 'cum_s': cum, 'lengths': lengths, 'events': [],
            'start_s_in_lane': 0.0, 'total_length': float(sum(lengths)),
            'finish_xy': None}

class Box:
    def __init__(self, oid, x_v, y_v, yaw_v, L, W, speed=0.0):
        self.id=oid; self.speed=float(speed); self.type_id='vehicle.vtd.object'
        cx, cy = frame.to_carla_xy(x_v, y_v)
        self._x, self._y = cx, cy
        self.yaw_deg = frame.to_carla_yaw_deg(yaw_v)
        ext=type('E',(),{'x':L/2.0,'y':W/2.0,'z':0.7})()
        self.bounding_box=type('BB',(),{'extent':ext})()
    def get_location(self):
        s=self; return type('L',(),{'x':s._x,'y':s._y,'z':0.0})()
    def get_velocity(self):
        s=self; return type('V',(),{'length':lambda self_: s.speed})()
    def get_transform(self):
        s=self
        return type('T',(),{'rotation':type('R',(),{'yaw':s.yaw_deg})()})()

class ActorList(list):
    def filter(self, pat):
        p=pat.strip('*'); return ActorList(a for a in self if p in a.type_id)
class World:
    def __init__(self, a): self._a=ActorList(a)
    def get_actors(self): return self._a
class Ap:
    def __init__(self, planner, ego, actors):
        self._waypoint_planner=planner; self._world=World(list(actors)); self._vehicle=ego
        self.config=type('C',(),{'idm_red_light_minimum_distance':5.299})()

def build(lg, cfg, first_lane, ego_v, objs_v, ego_speed=0.0,
          first_shift=None):
    """first_shift=(a,b,left) 면 그 시프트를 먼저 적용한다 (1차 시프트 재현)."""
    pc=build_pdm_config(cfg)
    route=make_route(lg, first_lane)
    pl=VtdRoutePlanner(lg, route, cfg, config=pc)
    ex,ey,eyaw,eL,eW = ego_v
    ecx,ecy = frame.to_carla_xy(ex,ey)
    if first_shift is not None:
        a,b,left = first_shift
        pl.route_index = 0
        pl.shift_route_smoothly(a, b, left, transition_length=120.0)
        pl._kd = None
    # 자차 최근접 인덱스 (밀린 경로 기준)
    d=np.hypot(pl.route_points[:,0]-ecx, pl.route_points[:,1]-ecy)
    pl.route_index=int(np.argmin(d))
    ego=Box(0, ex, ey, eyaw, eL, eW)
    actors=[ego]+[Box(o[0],o[1],o[2],o[3],o[4],o[5]) for o in objs_v]
    ap=Ap(pl, ego, actors)
    kr=KrRules(cfg)
    return kr, pl, ap, actors
