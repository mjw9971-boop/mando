"""전이 배치 이격 모델 — kr_rules._span_clear_model 과 같은 축.
경로(기준 열) 폴리라인 위에서 t_e(s)=f(s)·D(s) 궤적을 그리고, 각 객체를
경로 접선 프레임으로 풀어 |t_e − lat| − w_e − ht 의 최소를 낸다.
w_e = hw_e/cos(psi) + sagitta(kappa·hl^2/2)."""
import math, numpy as np

def frenet(P, H, S, x, y):
    j = int(np.argmin(np.hypot(P[:,0]-x, P[:,1]-y)))
    h = H[j]
    dx, dy = x-P[j,0], y-P[j,1]
    lat = -math.sin(h)*dx + math.cos(h)*dy      # +좌 (VTD 우수계)
    return S[j], lat, h, j

def clear_profile(P, H, S, ego_s, D, objs, ego_L, ego_W, obj_L, obj_W,
                  start_rel, trans_m, end_rel):
    """(min_clear, per_id). start_rel/end_rel 은 자차 기준 [m]."""
    hl, hw = ego_L/2.0, ego_W/2.0
    def f(s_rel):
        if s_rel <= start_rel: return 0.0
        if s_rel >= start_rel + trans_m: return 1.0
        u = (s_rel - start_rel)/trans_m
        return -math.cos(u*math.pi)/2.0 + 0.5
    per = {}
    for oid, ox, oy, oyaw in objs:
        s_o, lat_o, h_o, _ = frenet(P, H, S, ox, oy)
        s_rel_o = s_o - ego_s
        dpsi = math.atan2(math.sin(oyaw-h_o), math.cos(oyaw-h_o))
        ht = abs(obj_W/2*math.cos(dpsi)) + abs(obj_L/2*math.sin(dpsi))   # 유효 반폭
        hs = abs(obj_L/2*math.cos(dpsi)) + abs(obj_W/2*math.sin(dpsi))   # 유효 반장
        best = 1e9
        for s_rel in np.arange(s_rel_o-hs, s_rel_o+hs+1e-9, 0.1):
            if s_rel > end_rel: continue
            te = f(s_rel)*D
            # 비스듬한 단면
            eps = 0.25
            psi = math.atan2(f(s_rel+eps)*D - f(s_rel-eps)*D, 2*eps)
            we = hw/max(math.cos(psi), 1e-3)
            d2 = (f(s_rel+eps)*D - 2*f(s_rel)*D + f(s_rel-eps)*D)/(eps*eps)
            sag = abs(d2)*hl*hl/2.0
            c = abs(te - lat_o) - (we + sag) - ht
            best = min(best, c)
        per[oid] = round(best, 3)
    return (min(per.values()) if per else 1e9), per
