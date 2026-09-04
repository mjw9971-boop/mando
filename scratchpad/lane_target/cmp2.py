"""리플레이 비교 — 벽시계 잡음(timing/t)과 gc_pause 이벤트 줄을 제외한다.

제외 근거: timing.gc_s/gc_max_s 는 실행 시간 측정값이고, logger 는 GC 정지를
감지하면 틱 사이에 event 줄을 끼워 넣는다. 둘 다 코드 변경과 무관하며
'수정 전 vs 수정 전 재실행' 바닥에서도 그대로 나온다 (아래 floor 참조).
"""
import json, os, sys, glob

def norm(o):
    if isinstance(o, float): return 0.0 if o == 0.0 else o
    if isinstance(o, dict): return {k: norm(v) for k, v in o.items()}
    if isinstance(o, list): return [norm(v) for v in o]
    return o

def ticks(path):
    out = []
    for line in open(path):
        d = json.loads(line)
        if 'decision' not in d:          # gc_pause 등 이벤트 줄
            continue
        d.pop('timing', None)
        d.pop('t', None)
        out.append(norm(d))
    return out

def dec(d):
    de = d.get('decision') or {}
    return (de.get('v_target'), de.get('state'), (de.get('reasons') or {}).get('winner'),
            (d.get('cmd') or {}).get('steering'), (d.get('cmd') or {}).get('accel'))

def cmp_files(a, b):
    la, lb = ticks(a), ticks(b)
    if len(la) != len(lb):
        return {'ok': False, 'why': f'틱 수 {len(la)} vs {len(lb)}', 'n': min(len(la), len(lb))}
    nd = nc = 0; first = None; keys = {}
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            nd += 1
            if first is None: first = i
            for k in set(x) | set(y):
                if x.get(k) != y.get(k): keys[k] = keys.get(k, 0) + 1
            if dec(x) != dec(y): nc += 1
    return {'ok': nd == 0, 'n': len(la), 'ndiff': nd, 'ndec': nc, 'first': first, 'keys': keys}

if __name__ == '__main__':
    A, B = sys.argv[1], sys.argv[2]
    files = sorted(os.path.basename(p) for p in glob.glob(A + '/*.jsonl'))
    allok = True; tot = 0
    for f in files:
        pb = os.path.join(B, f)
        if not os.path.exists(pb):
            print(f'{f[:34]:34s} 상대 없음'); allok = False; continue
        r = cmp_files(os.path.join(A, f), pb); tot += r.get('n', 0); allok &= r['ok']
        if r['ok']:
            print(f'{f[:34]:34s} 동일  틱 {r["n"]:5d}')
        else:
            print(f'{f[:34]:34s} **차이** {r.get("why","")} 틱 {r.get("ndiff")} '
                  f'(결정 필드 {r.get("ndec")}) 첫 {r.get("first")} 키 {r.get("keys")}')
    print(f'\n파일 {len(files)}개, 틱 합계 {tot}  →  {"PASS (전부 동일)" if allok else "FAIL"}')
