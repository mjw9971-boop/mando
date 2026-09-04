import sys, math, json
sys.path.insert(0,'.')
from vtd_adapter.lanegraph import LaneGraph
EGO_L, EGO_W = 4.848, 1.886
OBJ_L, OBJ_W = 4.394, 1.808
CLR = 0.30

def corners(cx,cy,yaw,L,W):
    c,s=math.cos(yaw),math.sin(yaw); hl,hw=L/2,W/2
    return [(cx+dx*c-dy*s, cy+dx*s+dy*c) for dx,dy in ((hl,hw),(hl,-hw),(-hl,-hw),(-hl,hw))]
def _proj(pts,ax,ay): 
    v=[p[0]*ax+p[1]*ay for p in pts]; return min(v),max(v)
def obb_gap(a,b):
    best=-1e9
    for poly in (a,b):
        n=len(poly)
        for i in range(n):
            x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
            ex,ey=x2-x1,y2-y1; L=math.hypot(ex,ey)
            if L<1e-9: continue
            ax,ay=-ey/L,ex/L
            amin,amax=_proj(a,ax,ay); bmin,bmax=_proj(b,ax,ay)
            best=max(best, max(bmin-amax, amin-bmax))
    return best
def load_ticks(path):
    tk=[]
    for line in open(path):
        d=json.loads(line)
        if 'decision' in d: tk.append(d)
    t0=tk[0]['t']
    for d in tk: d['tt']=d['t']-t0
    return tk
