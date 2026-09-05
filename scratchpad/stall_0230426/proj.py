import sys, numpy as np, json
sys.path.insert(0,'scratchpad/stall_0230426')
pts=np.load('scratchpad/stall_0230426/route_pts.npy')  # CARLA frame (y = -y_vtd)
XY=pts[:,:2]
tang=np.gradient(XY,axis=0); tang/= (np.linalg.norm(tang,axis=1,keepdims=True)+1e-12)
def project(x_vtd,y_vtd):
    p=np.array([x_vtd,-y_vtd])
    d=np.linalg.norm(XY-p,axis=1); i=int(np.argmin(d))
    t=tang[i]; n=np.array([-t[1],t[0]])
    lat=float(np.dot(p-XY[i],n))
    return i, i*0.1, lat, float(d[i])
def heading(i):
    t=tang[i]; return float(np.arctan2(t[1],t[0]))
