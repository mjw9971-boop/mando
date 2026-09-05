import sys, math
sys.path.insert(0,'scratchpad/false_standoff')
from load import LOGS, rows as _rows
def ticks(p):
    return [d for d in _rows(p) if 'decision' in d]
def obj_raw(d, oid):
    for o in d['raw']['objects']:
        if int(o[0])==oid: return o
    return None
def obj_meta(d, oid):
    for o in d.get('objects') or []:
        if o['id']==oid: return o
    return None
def bound(d):
    """standoff 가 kr 후보를 구속하고, kr 후보가 최종 승자인 틱인가."""
    rs=d['decision']['reasons']; a=rs.get('avoid') or {}
    sv=a.get('standoff_v'); sid=a.get('standoff_id')
    if sv is None or sid is None: return None
    kr=rs.get('route_end')
    if kr is None or abs(kr-sv)>0.02: return None
    if rs.get('winner')!='route_end': return None
    lim=d['world'].get('speed_limit')
    if lim is not None and d['decision']['v_target'] >= lim-0.05: return None
    return (int(sid), float(sv), a.get('standoff_d'))
def segments(T, gap_ticks=10):
    out=[]; cur=None
    for i,d in enumerate(T):
        b=bound(d)
        if b:
            if cur and i-cur[1]<=gap_ticks: cur[1]=i
            else:
                if cur: out.append(cur)
                cur=[i,i]
        # gap 은 유지 (연속으로 안 봄)
    if cur: out.append(cur)
    # 병합
    merged=[]
    for a,b in out:
        if merged and a-merged[-1][1]<=gap_ticks: merged[-1][1]=b
        else: merged.append([a,b])
    return merged
