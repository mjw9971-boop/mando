import json, glob, os
B='logs/batch'
OFF='20260905_155049'; ON='20260905_155735'
NAMES=['정적회피집중_01_좌회전2','정적회피집중_02_직진3','정적회피집중_03_우회전5']
def rows(batch, name):
    p=os.path.join(B,batch,name+'.jsonl')
    return [json.loads(l) for l in open(p,encoding='utf-8') if '"decision"' in l]
def av(d): return (d['decision']['reasons'] or {}).get('avoid') or {}
