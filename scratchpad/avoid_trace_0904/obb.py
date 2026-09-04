import math
EGO_L, EGO_W = 4.848, 1.886
def corners(cx,cy,yaw,L,W):
    c,s=math.cos(yaw),math.sin(yaw); hl,hw=L/2,W/2
    return [(cx+dx*c-dy*s, cy+dx*s+dy*c) for dx,dy in
            ((hl,hw),(hl,-hw),(-hl,-hw),(-hl,hw))]
def _proj(pts,ax,ay):
    v=[p[0]*ax+p[1]*ay for p in pts]; return min(v),max(v)
def obb_gap(a,b):
    """signed separation between two OBBs (SAT). >0 = clear gap, <=0 = overlap depth(neg)."""
    best=-1e9
    for poly in (a,b):
        n=len(poly)
        for i in range(n):
            x1,y1=poly[i]; x2,y2=poly[(i+1)%n]
            ex,ey=x2-x1,y2-y1; L=math.hypot(ex,ey)
            if L<1e-9: continue
            ax,ay=-ey/L,ex/L
            amin,amax=_proj(a,ax,ay); bmin,bmax=_proj(b,ax,ay)
            gap=max(bmin-amax, amin-bmax)
            best=max(best,gap)
    return best
