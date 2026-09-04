import sys, json, copy; sys.path.insert(0,'scratchpad/lane_target'); sys.path.insert(0,'.')
import numpy as np
from rig import *
from geo import load_ticks, corners, obb_gap, EGO_L, EGO_W
lg=LaneGraph('data/lane_graph.pkl')

def run(path, tt, first_lane, first_shift, sw, lvl=1, tag=''):
    tk=load_ticks(path); d=[x for x in tk if x['tt']>=tt][0]
    ex,ey,eyaw=d['raw']['ego'][0],d['raw']['ego'][1],d['raw']['ego'][3]
    objs=[(int(o[0]),o[1],o[2],o[4],o[6],o[7]) for o in d['raw']['objects']]
    cfg=load_cfg(**sw)
    kr,pl,ap,actors=build(lg,cfg,first_lane,(ex,ey,eyaw,EGO_L,EGO_W),objs,
                          first_shift=first_shift)
    # 객체 정지 타이머 채우기
    n=int(round(cfg['overtake']['obj_static_s']*cfg['comm']['send_hz']))+2
    for _ in range(n): kr._update_obj_timers(ap)
    kr._tick_cache(ap, pl)
    # BREAKOUT L{lvl} 상태 (정지 오래)
    kr.bo_state='BREAKOUT'; kr.bo_level=lvl; kr.bo_stop_ticks=10**6; kr.bo_stuck_ticks=10**6
    kr.ot_span=None
    corridor=kr._corridor_blockers(ap, pl)
    if not corridor: return dict(err='회랑 비어 있음')
    actor=corridor[0][3]; chain=kr._chain(corridor, actor)
    ego_lane=kr._ego_lane(lg, ap); local_s=kr._ego_local_s(lg, ap)
    kr.last_avoid={'state':'PREEMPT','blocker':corridor[0][3].id}; kr.ot_pass_solid=False
    before=np.copy(pl.route_points)
    ok=kr._side_pass(ap, pl, 0.0, chain, True, lg, ego_lane, local_s, 1)
    a=dict(kr.last_avoid or {})
    res=dict(ok=ok, ego_lane=ego_lane, corridor=[(c[3].id, round(c[0],1), round(c[1],2)) for c in corridor],
             chain=chain['ids'], shift=a.get('shift'), span=a.get('span'),
             side_pick=a.get('side_pick'), rejects=a.get('rejects'),
             entry_base_clear=a.get('entry_base_clear'),
             entry_plateau_ids=a.get('entry_plateau_ids'),
             kappa=a.get('shift_kappa'), lc_var=a.get('lc_var'))
    if ok:
        i0=pl.route_index
        res['foot_delta']=float(np.abs(pl.route_points[i0]-before[i0]).max())
        # 평지 OBB 이격
        span=a['span']; plat=int((span[0]+span[1])/2)
        gaps={}
        for o in ap._world.get_actors():
            if o.id==0: continue
            j=int(np.argmin(np.hypot(pl.route_points[:,0]-o.get_location().x,
                                     pl.route_points[:,1]-o.get_location().y)))
            j=int(np.clip(j,1,len(pl.route_points)-2))
            t=pl.route_points[j+1,:2]-pl.route_points[j,:2]
            h=float(np.arctan2(t[1],t[0]))
            gaps[o.id]=round(obb_gap(corners(pl.route_points[j,0],pl.route_points[j,1],h,EGO_L,EGO_W),
                                     corners(o.get_location().x,o.get_location().y,
                                             np.radians(o.yaw_deg),
                                             o.bounding_box.extent.x*2, o.bounding_box.extent.y*2)),2)
        res['obb']=gaps; res['obb_min']=min(gaps.values())
        # 이동 후 회랑
        pl._kd=None
        res['corridor_after']=[c[3].id for c in kr._corridor_blockers(ap, pl)]
    return res
