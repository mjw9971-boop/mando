import json, glob, os
D='logs/batch/20260904_173419'
D2='logs/batch/20260904_162407'
def load(path):
    ticks=[]; events=[]
    for line in open(path):
        d=json.loads(line)
        if 'decision' in d: ticks.append(d)
        else: events.append(d)
    t0=ticks[0]['t']
    for d in ticks: d['tt']=d['t']-t0
    return ticks, events
def logs(D=D):
    return sorted(glob.glob(D+'/*.jsonl'))
def name(p): return os.path.basename(p).replace('.jsonl','')
