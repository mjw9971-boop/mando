"""
ctrl24 — 적색 정지선 앞 K9 탈출 유예 (2026-09-11).

여기서 지키는 불변:
  · C = (정지선 거리 ≤ escape_red_defer_stopline_m) ∧ (그 신호 state 가 Red).
    황색·녹색·매핑 없음·거리 밖은 전부 C 거짓 → 유예 없음, 기존 동작 그대로.
  · 유예 중에는 바닥이 None 이고 가상 시프트도 안 선다. escape_red_defer_s 를 채우면
    그 뒤로는 유예하지 않는다.
  · 상태는 타이머 하나(_esc_defer_ticks). C 가 거짓이면 즉시 0 이다.
  · 기본 off — off 면 C 를 계산조차 하지 않는다.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from test_ctrl24_avoid import apply, car, rig as avoid_rig        # noqa: E402
from test_ctrl24_escape import hold, on_cfg                        # noqa: E402
from test_ctrl24_long import TL, rig as long_rig, set_signal       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
HZ = CFG['comm']['send_hz']
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']


def defer_cfg(**kw):
    c = on_cfg(escape_red_defer_enable=True)
    c['ctrl24'].update(kw)
    return c


def rig_at(d_tl, state, cfg=None):
    """정지선 d_tl m (뒷축 기준) 앞 · 주어진 신호 상태. 앞은 비어 있어 바닥이 깔릴 자리."""
    kr, p, ap = long_rig(cfg or defer_cfg(), d_tl=d_tl, state=state)
    return kr, p, ap


def stick(kr, ap, p, n=None):
    """래치가 설 때까지 정지 상태로 돌린다."""
    hold(kr, ap, p, n or int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_params_defaults():
    assert C['escape_red_defer_enable'] is False
    assert C['escape_red_defer_stopline_m'] == 15.0
    assert C['escape_red_defer_s'] == 53.0


def test_off_does_not_defer_and_keeps_the_floor():
    kr, p, ap = rig_at(10.0, TrafficLightState.Red, cfg=on_cfg())   # 유예 스위치 off
    stick(kr, ap, p)
    assert kr._esc_engaged is True and kr._esc_deferring is False
    assert kr._esc_defer_ticks == 0                                  # C 를 계산조차 않는다
    assert kr.last_escape['state'] == 'FLOOR'


# ── 6종 조건 ─────────────────────────────────────────────────────────────
def test_red_near_stopline_defers():
    kr, p, ap = rig_at(10.0, TrafficLightState.Red)
    stick(kr, ap, p)
    assert kr._esc_engaged is True and kr._esc_deferring is True
    e = kr.last_escape
    assert e['state'] == 'RED_DEFER' and e['stopline_m'] == pytest.approx(10.0)
    assert e['left_s'] == pytest.approx(C['escape_red_defer_s'] - kr._esc_defer_ticks / HZ, abs=0.06)
    assert kr._escape_floor(ap) is None                              # 바닥을 안 깐다


def test_red_near_stopline_fires_after_defer_seconds():
    kr, p, ap = rig_at(10.0, TrafficLightState.Red)
    stick(kr, ap, p)
    need = int(C['escape_red_defer_s'] * HZ)
    for _ in range(need - kr._esc_defer_ticks):
        apply(kr, ap, v=0.0, target=0.0)
    assert kr._esc_defer_ticks >= need and kr._esc_deferring is False
    assert kr.last_escape['state'] == 'FLOOR' and kr._escape_floor(ap) == pytest.approx(C['escape_v'])


def test_green_never_defers():
    kr, p, ap = rig_at(10.0, TrafficLightState.Green)
    stick(kr, ap, p)
    assert kr._esc_deferring is False and kr._esc_defer_ticks == 0
    assert kr.last_escape['state'] == 'FLOOR'


def test_yellow_never_defers():
    kr, p, ap = rig_at(10.0, TrafficLightState.Yellow)
    stick(kr, ap, p)
    assert kr._esc_deferring is False and kr._esc_defer_ticks == 0
    assert kr.last_escape['state'] == 'FLOOR'


def test_red_far_from_stopline_does_not_defer():
    kr, p, ap = rig_at(40.0, TrafficLightState.Red)
    stick(kr, ap, p)
    assert kr._esc_deferring is False and kr._esc_defer_ticks == 0
    assert kr.last_escape['state'] == 'FLOOR'


def test_no_signal_mapping_does_not_defer():
    kr, p, ap = avoid_rig(defer_cfg(), actors=[car(2, 40.0)])        # next_traffic_lights 전부 None
    stick(kr, ap, p)
    assert kr._esc_engaged is True and kr._esc_deferring is False
    assert kr._esc_defer_ticks == 0 and kr.last_escape['state'] == 'FLOOR'


# ── 타이머 리셋 ──────────────────────────────────────────────────────────
def test_timer_resets_when_the_light_turns_green():
    kr, p, ap = rig_at(10.0, TrafficLightState.Red)
    stick(kr, ap, p)
    hold(kr, ap, p, int(5 * HZ), v=0.0, target=0.0)
    assert kr._esc_defer_ticks > int(4 * HZ) and kr._esc_deferring is True
    set_signal(p, TrafficLightState.Green, 10.0)
    apply(kr, ap, v=0.0, target=0.0)
    assert kr._esc_defer_ticks == 0 and kr._esc_deferring is False
    assert kr.last_escape['state'] == 'FLOOR'


def test_reset_clears_the_defer_timer():
    kr, p, ap = rig_at(10.0, TrafficLightState.Red)
    stick(kr, ap, p)
    assert kr._esc_defer_ticks > 0
    kr.on_reset()
    assert kr._esc_defer_ticks == 0 and kr._esc_deferring is False


# ── 가상 시프트도 같은 판정 ───────────────────────────────────────────────
def test_virtual_shift_is_deferred_too():
    from test_ctrl24_virtual import rig as virt_rig, stick as virt_stick, virt_cfg
    c = virt_cfg(); c['ctrl24'].update(escape_red_defer_enable=True)
    kr, p, ap = virt_rig(c, actors=[car(2, 30.0)])
    p.next_traffic_lights = [TL(TrafficLightState.Red)] * len(p.route_s)
    p.distances_to_next_traffic_lights[:] = 10.0
    virt_stick(kr, ap, p)
    assert kr._esc_deferring is True and kr.ot_span is None           # 가상 시프트도 안 선다
