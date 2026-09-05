import json, os, glob
B='logs/batch'
LOGS=[]
for d in ('20260904_230426','20260904_173419'):
    for p in sorted(glob.glob(os.path.join(B,d,'*.jsonl'))):
        LOGS.append((os.path.basename(os.path.dirname(p))+'/'+os.path.basename(p)[:-6], p))
for p in sorted(glob.glob('logs/run_*.jsonl')):
    LOGS.append(('실주행/'+os.path.basename(p)[:-6], p))
def rows(p):
    return [json.loads(l) for l in open(p,encoding='utf-8')]
