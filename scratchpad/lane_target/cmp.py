"""리플레이 출력 두 벌을 틱 단위로 비교한다. -0.0 은 0.0 으로 정규화."""
import json, sys, os, glob

def norm(o):
    if isinstance(o, float):
        return 0.0 if o == 0.0 else o
    if isinstance(o, dict):
        return {k: norm(v) for k, v in o.items()}
    if isinstance(o, list):
        return [norm(v) for v in o]
    return o

DEC = ('v_target', 'state', 'turn_signal')
def dec(d):
    de = d.get('decision') or {}
    return (de.get('v_target'), de.get('state'), (de.get('reasons') or {}).get('winner'),
            (d.get('cmd') or {}).get('steering'), (d.get('cmd') or {}).get('accel'))

def cmp_files(a, b):
    la = [json.loads(x) for x in open(a)]
    lb = [json.loads(x) for x in open(b)]
    if len(la) != len(lb):
        return {'ok': False, 'why': f'틱 수 {len(la)} vs {len(lb)}'}
    first = None; ndiff = 0; ndec = 0
    for i, (x, y) in enumerate(zip(la, lb)):
        if norm(x) != norm(y):
            ndiff += 1
            if first is None: first = i
            if dec(x) != dec(y): ndec += 1
    return {'ok': ndiff == 0, 'n': len(la), 'ndiff': ndiff, 'ndec': ndec, 'first': first}

if __name__ == '__main__':
    A, B = sys.argv[1], sys.argv[2]
    files = sorted(os.path.basename(p) for p in glob.glob(A + '/*.jsonl'))
    allok = True; tot = 0
    for f in files:
        pb = os.path.join(B, f)
        if not os.path.exists(pb):
            print(f'{f[:34]:34s} 상대 없음'); allok = False; continue
        r = cmp_files(os.path.join(A, f), pb)
        tot += r.get('n', 0)
        allok &= r['ok']
        if r['ok']:
            print(f'{f[:34]:34s} 동일  틱 {r["n"]:5d}')
        else:
            print(f'{f[:34]:34s} **차이** {r.get("why","")} 다른 틱 {r.get("ndiff")} '
                  f'(결정 필드 {r.get("ndec")}) 첫 {r.get("first")}')
    print(f'\n파일 {len(files)}개, 틱 합계 {tot}  →  {"PASS (전부 동일)" if allok else "FAIL"}')
