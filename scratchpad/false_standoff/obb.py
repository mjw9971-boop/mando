import math
def corners(cx,cy,yaw,L,W):
    c,s=math.cos(yaw),math.sin(yaw); hl,hw=L/2.0,W/2.0
    return [(cx+c*dx-s*dy, cy+s*dx+c*dy) for dx,dy in ((hl,hw),(hl,-hw),(-hl,-hw),(-hl,hw))]
def _seg(p,q,a,b):
    def ps(px,py,ax,ay,bx,by):
        vx,vy=bx-ax,by-ay; L2=vx*vx+vy*vy
        u=0.0 if L2==0 else max(0.0,min(1.0,((px-ax)*vx+(py-ay)*vy)/L2))
        return math.hypot(px-(ax+u*vx),py-(ay+u*vy))
    return min(ps(*p,*a,*b),ps(*q,*a,*b),ps(*a,*p,*q),ps(*b,*p,*q))
def rect_gap(A,B):
    def axes(P):
        o=[]
        for i in range(4):
            x1,y1=P[i]; x2,y2=P[(i+1)%4]; n=(-(y2-y1),x2-x1); L=math.hypot(*n)
            if L>1e-9: o.append((n[0]/L,n[1]/L))
        return o
    ov=True
    for ax,ay in axes(A)+axes(B):
        pa=[x*ax+y*ay for x,y in A]; pb=[x*ax+y*ay for x,y in B]
        if max(pa)<min(pb) or max(pb)<min(pa): ov=False; break
    if ov: return 0.0
    return min(_seg(A[i],A[(i+1)%4],B[j],B[(j+1)%4]) for i in range(4) for j in range(4))
