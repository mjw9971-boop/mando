"""pytest 미설치 환경용 **최소 스텁** (scratchpad 전용, 저장소 밖).

CLAUDE.md 가 경고한 shim 과 같은 한계가 있다 — builtin fixture(tmp_path 등)를
지원하지 않으므로 그런 테스트는 '미실행' 으로 분류한다. 절대값이 아니라
**수정 전/후 결과 비교**에만 쓴다.
"""
import math, functools


class Approx:
    def __init__(self, v, rel=None, abs=None):
        self.v, self.rel, self.abs = v, rel, abs
    def _eq1(self, a, b):
        if isinstance(b, (list, tuple)) or isinstance(a, (list, tuple)):
            a, b = list(a), list(b)
            return len(a) == len(b) and all(self._eq1(x, y) for x, y in zip(a, b))
        if b is None or a is None:
            return a is b
        rel = 1e-6 if self.rel is None else self.rel
        ab = 1e-12 if self.abs is None else self.abs
        return abs(a - b) <= max(ab, rel * max(abs(a), abs(b)))
    def __eq__(self, other):
        return self._eq1(other, self.v)
    def __req__(self, other):
        return self.__eq__(other)
    def __repr__(self):
        return f'approx({self.v})'


def approx(v, rel=None, abs=None):
    return Approx(v, rel, abs)


class Skipped(Exception):
    pass


class Failed(Exception):
    pass


def skip(reason=''):
    raise Skipped(reason)


def fail(reason=''):
    raise Failed(reason)


def xfail(reason=''):
    raise Skipped('xfail: ' + reason)


def importorskip(name, *a, **k):
    try:
        return __import__(name)
    except ImportError:
        raise Skipped('no ' + name)


class raises:
    def __init__(self, exc, match=None):
        self.exc, self.match = exc, match
        self.value = None
    def __enter__(self):
        return self
    def __exit__(self, t, v, tb):
        if t is None:
            raise AssertionError(f'DID NOT RAISE {self.exc}')
        self.value = v
        return issubclass(t, self.exc if isinstance(self.exc, tuple) else self.exc)


def fixture(*a, **k):
    def deco(fn):
        fn.__is_fixture__ = True
        fn.__fixture_scope__ = k.get('scope', 'function')
        return fn
    if a and callable(a[0]) and not k:
        return deco(a[0])
    return deco


class _Mark:
    def parametrize(self, argnames, argvalues, **k):
        names = [s.strip() for s in argnames.split(',')] if isinstance(argnames, str) else list(argnames)
        def deco(fn):
            prev = getattr(fn, '__params__', [])
            fn.__params__ = prev + [(names, list(argvalues))]
            return fn
        return deco
    def skipif(self, cond, reason=''):
        def deco(fn):
            if cond:
                fn.__skip__ = reason
            return fn
        return deco
    def skip(self, reason=''):
        def deco(fn):
            fn.__skip__ = reason
            return fn
        return deco
    def xfail(self, *a, **k):
        def deco(fn):
            fn.__xfail__ = True
            return fn
        return deco
    def __getattr__(self, name):
        def deco(*a, **k):
            if a and callable(a[0]):
                return a[0]
            return lambda fn: fn
        return deco


mark = _Mark()
