import pickle, numpy as np, sys
sys.path.insert(0,'/home/cjw/mando'); sys.path.insert(0,'/home/cjw/mando/team_code')
from vtd_adapter.config import load_params_yaml
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.route import VtdRoutePlanner
from config import GlobalConfig
cfg=load_params_yaml('config/params.yaml')
rt=pickle.load(open('logs/batch/20260904_230426/routes/route_정적회피집중_02_직진3.pkl','rb'))
def planner():
    return VtdRoutePlanner(LaneGraph('data/lane_graph.pkl'), rt, cfg, config=GlobalConfig())
def lat_of(P, x_vtd, y_vtd):
    """자차 기준 아님 — 현재 P.route_points 기준 최근접점의 부호 있는 횡오프셋."""
    XY=P.route_points[:,:2]; p=np.array([x_vtd,-y_vtd])
    d=np.linalg.norm(XY-p,axis=1); i=int(np.argmin(d))
    j=min(i+1,len(XY)-1); k=max(i-1,0)
    t=XY[j]-XY[k]; t/=np.linalg.norm(t)+1e-12
    n=np.array([-t[1],t[0]])
    return i, float(np.dot(p-XY[i],n)), float(d[i])
