import sys, yaml, numpy as np, glob
sys.path.insert(0,'.'); sys.path.insert(0,'scratchpad/lane_target/before')
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.route import VtdRoutePlanner as NEW
from vtd_adapter_before.route import VtdRoutePlanner as OLD
from run_agent import build_pdm_config, load_route
cfg=yaml.safe_load(open('config/params.yaml')); lg=LaneGraph('data/lane_graph.pkl'); pc=build_pdm_config(cfg)
print('스위치 shift_target_current_lane_enable =', cfg['overtake']['shift_target_current_lane_enable'])
allok=True
for rp in sorted(glob.glob('logs/batch/2026090*/routes/*.pkl')):
    r=load_route(rp); a=OLD(lg,r,cfg,config=pc); b=NEW(lg,r,cfg,config=pc); n=len(a.route_points)
    ok=all(np.array_equal(getattr(a,k),getattr(b,k)) for k in
           ('route_points','original_route_points','commands','lat_shift','route_s'))
    plo=True; sh=True
    for i0 in range(0,n-400,max(1,(n-400)//12)):
        for left in (True,False):
            a.route_index=b.route_index=i0
            if not np.array_equal(a.planned_lateral_offsets(i0,i0+300,left,step_pts=10),
                                  b.planned_lateral_offsets(i0,i0+300,left,step_pts=10)): plo=False
    for i0 in range(0,n-500,max(1,(n-500)//8)):
        for left in (True,False):
            a2=OLD(lg,r,cfg,config=pc); b2=NEW(lg,r,cfg,config=pc); a2.route_index=b2.route_index=i0
            a2.shift_route_smoothly(i0+50,i0+450,left,transition_length=120.0)
            b2.shift_route_smoothly(i0+50,i0+450,left,transition_length=120.0)
            if not (np.array_equal(a2.route_points,b2.route_points) and
                    np.array_equal(a2.commands,b2.commands) and
                    np.array_equal(a2.lat_shift,b2.lat_shift)): sh=False
    allok &= ok and plo and sh
    print(f"{rp.split('/')[-1]:38s} 빌드={ok} plo={plo} shift={sh}")
print('\nroute.py 층 off 동일성:', 'PASS' if allok else 'FAIL')
