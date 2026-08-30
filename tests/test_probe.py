"""
프리즈 진단 계측 (vtd_adapter.logger.Probe + run_agent 의 무수신 구간 기록).

2026-08-30 실사고: 틱 로그에 14.95 s 공백이 났는데 그 사이 줄이 하나도 없어
"우리 루프가 멈춘 것"과 "VTD 가 안 보낸 것"을 사후에 가릴 수 없었다.
핵심은 timing.loop_iters / gap_close.iters_in_gap — 공백 동안 루프가 돌았는지다.

계측은 **관측 전용**이다. 여기 테스트는 값이 찍히는지와 오버헤드만 본다.
"""
import gc
import time

from vtd_adapter.config import load_params_yaml
from vtd_adapter.logger import Probe


def cfg(**over):
    c = load_params_yaml()
    c['log'] = {**c.get('log', {}), **over}
    return c


def test_probe_fields_present():
    p = Probe(cfg())
    try:
        p.sample(0)                       # 1회차는 델타가 없다
        s = p.sample(20)
        assert s['loop_iters'] == 20
        assert 'cpu_dt' in s and s['cpu_dt'] >= 0.0
        assert 'nivcsw' in s and 'majflt' in s      # POSIX
    finally:
        p.close()


def test_probe_disabled_is_empty():
    p = Probe(cfg(probe_enabled=False))
    try:
        assert p.sample(5) == {}
    finally:
        p.close()


def test_probe_records_gc_pause():
    p = Probe(cfg())
    try:
        p.sample(0)
        junk = [[object() for _ in range(500)] for _ in range(200)]
        gc.collect()
        del junk
        s = p.sample(1)
        assert s.get('gc_n', 0) >= 1 and s['gc_s'] >= 0.0
    finally:
        p.close()


def test_take_long_gc_thresholds_and_resets():
    p = Probe(cfg(probe_gc_min_s=1e9))     # 사실상 절대 안 넘는 임계
    try:
        p._gc_n, p._gc_s, p._gc_max = 1, 0.5, 0.5
        assert p.take_long_gc() is None
        assert p._gc_n == 0 and p._gc_max == 0.0     # 임계 미만도 카운터는 비운다
        p._gc_n, p._gc_s, p._gc_max = 1, 0.5, 0.5
        p.gc_min_s = 0.1
        assert p.take_long_gc() == 0.5
    finally:
        p.close()


def test_rusage_cadence_is_configurable():
    """비용이 문제되면 주기를 낮춘다 — 그 틱만 nivcsw/majflt 가 빠진다."""
    p = Probe(cfg(probe_rusage_every=3))
    try:
        p.sample(0)
        got = [('nivcsw' in p.sample(i)) for i in range(1, 7)]
        assert any(got) and not all(got)
    finally:
        p.close()


def test_probe_close_unregisters_gc_callback():
    n0 = len(gc.callbacks)
    p = Probe(cfg())
    assert len(gc.callbacks) == n0 + 1
    p.close()
    assert len(gc.callbacks) == n0


def test_probe_overhead_is_negligible():
    """틱 예산(50 ms) 대비 무시할 수준이어야 한다 — 실측 ~2 µs/틱."""
    p = Probe(cfg())
    try:
        p.sample(0)
        t = time.perf_counter()
        for i in range(2000):
            p.sample(i)
        per_us = (time.perf_counter() - t) / 2000 * 1e6
        assert per_us < 50.0, f'{per_us:.1f} µs/틱 — 예산 50 ms 의 0.1% 초과'
    finally:
        p.close()
