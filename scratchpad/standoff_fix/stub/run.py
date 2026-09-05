"""스텁 러너 — tests/*.py 의 test_* 를 모아 실행. 결과는 수정 전/후 비교용."""
import sys, os, glob, traceback, itertools, importlib, pathlib
ROOT = pathlib.Path('/home/cjw/mando')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))   # stub pytest 우선
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'tests'))
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(ROOT / 'team_code'))
import pytest as P

res = {'pass': [], 'fail': [], 'skip': [], 'error': []}
for path in sorted(glob.glob(str(ROOT / 'tests' / 'test_*.py'))):
    mod_name = os.path.basename(path)[:-3]
    try:
        mod = importlib.import_module(mod_name)
    except P.Skipped as e:
        res['skip'].append((mod_name, '<module>', str(e))); continue
    except Exception:
        res['error'].append((mod_name, '<import>', traceback.format_exc(limit=2).strip().splitlines()[-1])); continue
    pmark = getattr(mod, 'pytestmark', None)
    mod_skip = None
    if pmark is not None:
        for m in (pmark if isinstance(pmark, list) else [pmark]):
            if isinstance(m, str) or callable(m):
                pass
    fixtures = {n: f for n, f in vars(mod).items() if getattr(f, '__is_fixture__', False)}
    cache = {}
    def resolve(name):
        if name in cache: return cache[name]
        f = fixtures[name]
        args = [resolve(a) for a in f.__code__.co_varnames[:f.__code__.co_argcount]]
        v = f(*args)
        if hasattr(v, '__next__'):
            v = next(v)
        cache[name] = v
        return v
    for fname in sorted(vars(mod)):
        fn = vars(mod)[fname]
        if not (fname.startswith('test_') and callable(fn)): continue
        if getattr(fn, '__skip__', None):
            res['skip'].append((mod_name, fname, fn.__skip__)); continue
        params = getattr(fn, '__params__', [])
        argnames = list(fn.__code__.co_varnames[:fn.__code__.co_argcount])
        combos = [{}]
        for names, values in reversed(params):
            new = []
            for c in combos:
                for v in values:
                    vv = v if len(names) > 1 else (v,)
                    d = dict(c); d.update(dict(zip(names, vv))); new.append(d)
            combos = new
        for combo in combos:
            kw = {}
            missing = None
            for a in argnames:
                if a in combo: kw[a] = combo[a]
                elif a in fixtures:
                    try: kw[a] = resolve(a)
                    except Exception: missing = a; break
                else: missing = a; break
            label = fname + (('[' + ','.join(f'{k}={v}' for k, v in combo.items()) + ']') if combo else '')
            if missing:
                res['skip'].append((mod_name, label, f'fixture 미지원: {missing}')); continue
            try:
                fn(**kw)
                res['pass'].append((mod_name, label, ''))
            except P.Skipped as e:
                res['skip'].append((mod_name, label, str(e)))
            except Exception as e:
                res['fail'].append((mod_name, label, f'{type(e).__name__}: {e}'.split(chr(10))[0][:160]))
out = sys.argv[1] if len(sys.argv) > 1 else None
lines = [f"pass={len(res['pass'])} fail={len(res['fail'])} skip={len(res['skip'])} error={len(res['error'])}"]
for k in ('fail', 'error'):
    for m, t, e in res[k]: lines.append(f'{k.upper()}\t{m}::{t}\t{e}')
for m, t, e in res['skip']: lines.append(f'SKIP\t{m}::{t}\t{e}')
txt = '\n'.join(lines)
print(lines[0])
if out: open(out, 'w', encoding='utf-8').write(txt + '\n')
