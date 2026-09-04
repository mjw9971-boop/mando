"""side 선택 기준 평가기 — lane_graph 차로 중심선 기준 (경로 pkl 없는 로그용).

kr_rules._shift_placement 와 같은 축:
  · 기준 프레임 = 자차 차로의 successor 사슬 중심선 (자차가 그 위에 있다)
  · 목표 오프셋 D = 목표 차로 사슬 중심선의 기준 프레임 횡오프셋 중앙값
  · 격자 delay 0..15(0.5) × after 10..0(1), 전이 = max(12, 3v)
  · base_clear = delay 0 / after 현행 의 최소 이격
  · plateau = 격자 어디서도 임계(0.3) 를 못 넘는 객체
차이: _corridor_blockers 의 lat_band 제한을 두지 않고 모든 객체를 넣는다
(띠 밖 객체는 이격이 커서 최소값에 영향이 없고 plateau 판정에도 안 걸린다).
"""
import sys, math
import numpy as np
sys.path.insert(0, 'scratchpad/lane_target'); sys.path.insert(0, '.')
from geo import EGO_L, EGO_W, corners, obb_gap
from sweep import frenet, clear_profile

CLR = 0.30

def chain_poly(lg, start, dist=140.0, step=0.25):
    P=[];H=[];S=[];K=[];k=start;acc=0.0
    seen=set()
    while k is not None and acc<dist and k not in seen:
        seen.add(k)
        L=lg.length(k); s=0.0
        while s<L and acc<dist:
            x,y,z,h=lg.point_at(k,s); P.append((x,y)); H.append(h); S.append(acc); K.append(k)
            s+=step; acc+=step
        nx=lg.successors(k); k=nx[0] if nx else None
    return np.array(P), np.array(H), np.array(S), K

def eval_side(lg, ego_lane, ego_s, ex, ey, objs, side, v, lvl, obj_dims):
    """→ dict(가능 여부, base_clear, plateau, best, D, target)"""
    nb = lg.neighbor(ego_lane, side)
    if nb is None:
        return {'ok': False, 'why': 'no_neighbor', 'target': None}
    try:
        typ, col, _ = lg.mark_at(ego_lane, float(ego_s), side)
    except Exception:
        typ, col = None, None
    if str(col) == 'yellow':
        return {'ok': False, 'why': 'center_line', 'target': nb, 'mark': f'{typ}/{col}'}
    Pe,He,Se,_ = chain_poly(lg, ego_lane)
    Pt,_,_,_   = chain_poly(lg, nb)
    s_ego,_,_,_ = frenet(Pe,He,Se, ex, ey)
    Ds=[]
    for i in range(0, len(Pt), 4):
        s,lat,_,_ = frenet(Pe,He,Se, Pt[i,0], Pt[i,1])
        if 5.0 <= s - s_ego <= 45.0: Ds.append(lat)
    if not Ds:
        return {'ok': False, 'why': 'no_offset', 'target': nb, 'mark': f'{typ}/{col}'}
    D = float(np.median(Ds))
    trans = max(12.0, 3.0*max(v, 0.1))
    ahead = 1.0 if lvl >= 3 else 5.0
    after0 = 10.0
    end0 = 2*trans + 5.0 + after0 + 40.0
    grid=[]
    for k in range(31):
        delay = k*0.5
        for after in range(11):
            mn, per = clear_profile(Pe,He,Se, s_ego, D, objs, EGO_L, EGO_W,
                                    obj_dims[0], obj_dims[1],
                                    ahead+delay, trans, end0-(after0-after))
            grid.append((delay, after, per))
    base = next(g for g in grid if g[0]==0.0 and g[1]==after0)
    ids = set(o[0] for o in objs)
    best_of = {o: max(g[2].get(o, -1e9) for g in grid) for o in ids}
    plateau = sorted(o for o,hi in best_of.items() if hi < CLR)
    relevant = [o for o in ids if o not in plateau]
    def score(per):
        v2=[per[o] for o in relevant if o in per]
        return min(v2) if v2 else float('inf')
    best_score = max(score(g[2]) for g in grid)
    return {'ok': True, 'target': nb, 'mark': f'{typ}/{col}', 'D': D,
            'base_clear': min(base[2].values()) if base[2] else float('inf'),
            'base_per': base[2], 'plateau': plateau, 'score': best_score,
            'trans': trans, 'ahead': ahead}
