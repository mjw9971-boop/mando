"""두 리플레이 산출물의 '판단' 동일성 비교.

정규화(그 사실을 보고에 명시):
  · 최상위 't' (리플레이 벽시계)와 'timing'(loop/gc/rusage) 은 제외 — 벽시계 의존
  · 부호 있는 0 (-0.0 vs 0.0) 은 같은 값으로 본다 (수치 동일, 표기만 다름)
그 외 모든 필드는 **완전 일치**를 요구한다 (허용오차 없음).
"""
import json, sys, glob, os
DROP_TOP = {'t', 'timing'}
def diff(x, y, path=''):
    if isinstance(x, dict) and isinstance(y, dict):
        out = []
        for k in sorted(set(x) | set(y)):
            out += diff(x.get(k), y.get(k), f'{path}/{k}')
        return out
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            return [(path, f'len {len(x)}', f'len {len(y)}')]
        out = []
        for i, (p, q) in enumerate(zip(x, y)):
            out += diff(p, q, f'{path}[{i}]')
        return out
    if isinstance(x, (int, float)) and isinstance(y, (int, float)):
        return [] if x == y else [(path, x, y)]        # -0.0 == 0.0
    return [] if x == y else [(path, x, y)]
def load(p):
    return [json.loads(l) for l in open(p, encoding='utf-8') if '"decision"' in l]
def cmp_dir(a, b, verbose=False):
    ok = True
    for pa in sorted(glob.glob(os.path.join(a, '*.jsonl'))):
        nm = os.path.basename(pa); pb = os.path.join(b, nm)
        if not os.path.exists(pb):
            print(f'  {nm:34s} 짝 없음'); ok = False; continue
        A, B = load(pa), load(pb)
        if len(A) != len(B):
            print(f'  {nm:34s} 틱수 {len(A)} vs {len(B)} ✗'); ok = False; continue
        bad = []
        for i, (x, y) in enumerate(zip(A, B)):
            d = diff({k: v for k, v in x.items() if k not in DROP_TOP},
                     {k: v for k, v in y.items() if k not in DROP_TOP})
            if d: bad.append((i, d))
        if bad:
            ok = False
            print(f'  {nm:34s} {len(A)}틱  차이 {len(bad)}틱 ✗  최초 i={bad[0][0]} {bad[0][1][:3]}')
            if verbose:
                for i, d in bad[:5]: print(f'      i={i} {d[:4]}')
        else:
            print(f'  {nm:34s} {len(A)}틱  ✅ 완전 동일')
    return ok
if __name__ == '__main__':
    v = '-v' in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    sys.exit(0 if cmp_dir(args[0], args[1], v) else 1)
