import json, os
D='logs/batch/20260904_230426'
def rows(name):
    p=os.path.join(D,name+'.jsonl')
    return [json.loads(l) for l in open(p,encoding='utf-8')]
