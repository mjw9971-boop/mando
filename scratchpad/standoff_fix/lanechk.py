import sys, math, numpy as np
sys.path.insert(0,'/home/cjw/mando')
from vtd_adapter.lanegraph import LaneGraph
LG=LaneGraph('data/lane_graph.pkl')
def locate(x_vtd, y_vtd):
    """(x,y) VTD → (lane key, 그 지점 기준선 heading[rad], 곡률, dir)."""
    try:
        m=LG.locate(x_vtd,y_vtd)
    except Exception:
        return None
    if m is None: return None
    key=m.lane; L=LG.lanes.get(key)
    if not L: return (key,None,None,None)
    P=np.asarray(L['pts'])[:, :2]
    d=np.hypot(P[:,0]-x_vtd,P[:,1]-y_vtd); i=int(np.argmin(d))
    hdg=L.get('hdg'); curv=L.get('curv')
    h=float(hdg[i]) if hdg is not None and i<len(hdg) else None
    c=float(curv[i]) if curv is not None and i<len(curv) else None
    return (key, h, c, L.get('dir'))
