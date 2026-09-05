import pickle, sys, numpy as np
sys.path.insert(0,'/home/cjw/mando'); sys.path.insert(0,'/home/cjw/mando/team_code')
from vtd_adapter.config import load_params_yaml
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.route import VtdRoutePlanner
from config import GlobalConfig
cfg=load_params_yaml('config/params.yaml')
def build(name):
    rt=pickle.load(open(f'logs/batch/20260904_230426/routes/route_{name}.pkl','rb'))
    return VtdRoutePlanner(LaneGraph('data/lane_graph.pkl'), rt, cfg, config=GlobalConfig())
def proj(P,x_vtd,y_vtd):
    XY=P.route_points[:,:2]; p=np.array([x_vtd,-y_vtd])
    d=np.hypot(XY[:,0]-p[0],XY[:,1]-p[1]); i=int(np.argmin(d))
    j=min(i+1,len(XY)-1); k=max(i-1,0)
    t=XY[j]-XY[k]; t=t/(np.hypot(*t)+1e-12); n=np.array([-t[1],t[0]])
    return i, i*0.1, float(np.dot(p-XY[i],n))
