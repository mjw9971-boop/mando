import sys, math, numpy as np
sys.path.insert(0,'scratchpad/false_standoff')
from lanechk import LG, locate
def forward_centerline(x_vtd, y_vtd, max_m=140.0):
    """자차 위치의 차로 중심선을, 후속 차로를 이어 앞으로 max_m 만큼 뽑는다.
    반환 (P[N,2], s[N]) — '그대로 직진' 기준선."""
    m=LG.locate(x_vtd,y_vtd)
    if m is None: return None
    key=m.lane
    P=np.asarray(LG.lanes[key]['pts'])[:, :2]
    d=np.hypot(P[:,0]-x_vtd,P[:,1]-y_vtd); i=int(np.argmin(d))
    out=[P[i:]]
    tot=float(np.sum(np.hypot(*np.diff(P[i:],axis=0).T))) if len(P)-i>1 else 0.0
    cur=key; guard=0
    while tot<max_m and guard<12:
        guard+=1
        sc=list(LG.successors(cur) or [])
        if not sc: break
        # 헤딩이 가장 이어지는 후속 하나
        last=out[-1]
        if len(last)<2: break
        t0=last[-1]-last[-2]; h0=math.atan2(t0[1],t0[0])
        best=None
        for k in sc:
            Q=np.asarray(LG.lanes[k]['pts'])[:, :2]
            if len(Q)<2: continue
            gap=float(np.hypot(*(Q[0]-last[-1])))
            t1=Q[1]-Q[0]; dh=abs((math.atan2(t1[1],t1[0])-h0+math.pi)%(2*math.pi)-math.pi)
            sc_=gap+30.0*dh
            if best is None or sc_<best[0]: best=(sc_,k,Q)
        if best is None: break
        cur=best[1]; Q=best[2]; out.append(Q)
        tot+=float(np.sum(np.hypot(*np.diff(Q,axis=0).T)))
    P=np.vstack(out)
    s=np.concatenate([[0.0],np.cumsum(np.hypot(*np.diff(P,axis=0).T))])
    return P,s
def project_on(P,s,x,y):
    d=np.hypot(P[:,0]-x,P[:,1]-y); i=int(np.argmin(d))
    j=min(i+1,len(P)-1); k=max(i-1,0)
    t=P[j]-P[k]; L=np.hypot(*t)
    if L<1e-9: return float(s[i]),0.0,i
    t=t/L; n=np.array([-t[1],t[0]])
    v=np.array([x,y])-P[i]
    return float(s[i]+np.dot(v,t)), float(np.dot(v,n)), i
def curvature(P,s,i,win=20):
    a=max(0,i-win); b=min(len(P)-1,i+win)
    if b-a<4: return None
    t1=P[a+1]-P[a]; t2=P[b]-P[b-1]
    dh=(math.atan2(t2[1],t2[0])-math.atan2(t1[1],t1[0])+math.pi)%(2*math.pi)-math.pi
    ds=s[b]-s[a]
    return abs(dh/ds) if ds>1e-6 else None
