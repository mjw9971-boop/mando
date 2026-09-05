import pickle, numpy as np, sys, pathlib
sys.path.insert(0,'/home/cjw/mando'); sys.path.insert(0,'/home/cjw/mando/team_code')
from vtd_adapter.config import load_params_yaml
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.route import VtdRoutePlanner
from config import GlobalConfig
cfg=load_params_yaml('config/params.yaml')
rt=pickle.load(open('logs/batch/20260904_230426/routes/route_정적회피집중_02_직진3.pkl','rb'))
P=VtdRoutePlanner(LaneGraph('data/lane_graph.pkl'), rt, cfg, config=GlobalConfig())
print('n points',P.route_points.shape,'ppm',P.points_per_meter)
rs=np.asarray(P.route_s)
print('route_s last',rs[-1])
for i in (4162,4520,4591,4870,4981):
    print(i,'route_s',round(float(rs[i]),2),'xy',np.round(P.route_points[i][:2],2))
np.save('scratchpad/stall_0230426/route_pts.npy', P.route_points)
np.save('scratchpad/stall_0230426/route_s.npy', rs)
np.save('scratchpad/stall_0230426/lat_shift.npy', np.asarray(P.lat_shift))
