"""31 시프트 생성 틱을 실제 _side_pass 로 재현한다.

경로는 **그 틱 자차 차로의 successor 사슬**로 만든다 (자차가 경로 위에 있으므로
t_cur = 0 → 커밋1 은 무동작 = 1 단계). 즉 이 재현은 **side_pick 효과만** 분리한다.
커밋1 의 다단(k=1) 경로는 104648/104807 을 원 경로(col2)+1차 시프트로 따로 재현한다.
"""
import sys, json, glob; sys.path.insert(0,'scratchpad/lane_target'); sys.path.insert(0,'.')
import numpy as np
from rig import *
from geo import load_ticks, EGO_L, EGO_W

def one(lg, d, sw):
    ex,ey,eyaw=d['raw']['ego'][0],d['raw']['ego'][1],d['raw']['ego'][3]
    objs=[(int(o[0]),o[1],o[2],o[4],o[6],o[7]) for o in d['raw']['objects']]
    if not objs: return None
    m=lg.locate(ex,ey)
    if m is None: return None
    cfg=load_cfg(**sw)
    kr,pl,ap,actors=build(lg,cfg,m.lane,(ex,ey,eyaw,EGO_L,EGO_W),objs)
    n=int(round(cfg['overtake']['obj_static_s']*cfg['comm']['send_hz']))+2
    for _ in range(n): kr._update_obj_timers(ap)
    kr._tick_cache(ap, pl)
    a=d['decision']['reasons'].get('avoid') or {}
    lvl=a.get('level') or 0
    if lvl:
        kr.bo_state='BREAKOUT'; kr.bo_level=lvl
        kr.bo_stop_ticks=10**6; kr.bo_stuck_ticks=10**6
    kr.ot_span=None
    corridor=kr._corridor_blockers(ap, pl)
    if not corridor: return {'err':'회랑 비어 있음'}
    chain=kr._chain(corridor, corridor[0][3])
    kr.last_avoid={'state':'PREEMPT','blocker':corridor[0][3].id}
    kr.ot_pass_solid=False
    i0=pl.route_index; before=np.copy(pl.route_points)
    ok=kr._side_pass(ap, pl, float(d['ego']['speed']), chain, True, lg,
                     kr._ego_lane(lg,ap), kr._ego_local_s(lg,ap), 1)
    la=kr.last_avoid or {}
    out={'ok':ok, 'shift':la.get('shift'), 'side_pick':la.get('side_pick'),
         'rejects':la.get('rejects'),
         'foot':float(np.abs(pl.route_points[i0]-before[i0]).max()) if ok else 0.0}
    return out
