"""
우선순위 불변식 — **신호·정지선 준수 > 회피**.

절대 규칙: 다음 신호가 Red 또는 황색 STOP 래치면, 그 정지선과 자차 사이의 정지
객체는 **거리 무관** 회피 대상이 아니다. PREEMPT/WAIT/REACTIVE/BREAKOUT 전부
미발동이고, 회피가 다시 열리는 조건은 **녹색 전환 후 obj_static_s 이상 계속
정지**뿐이다. 대기열 판별(_is_queue)은 이 규칙 뒤에 오는 **2선** 판별이라
녹색·무신호에서만 동작한다.

왜 거리 무관인가 — 30 m 억제창(stopline_suppress_m)은 "정지선 근처"만 막는다.
적신호 대기열은 그보다 길게 늘어설 수 있고, 그 꼬리를 비켜가면 결국 신호를
위반한다. 실측 2026-08-30 실전주행_교통류_01(route_s 1589.8): 정지 객체 2대
뒤에 선 채 신호가 녹→황→적→녹으로 순환했다. 적색 구간에 회피가 열려 있었다면
대기열을 비켜 신호를 위반했을 것이다.

시프트 기하: 나가는 전이 + 복귀 전이 span 전체가 억제 구역 **밖에서 끝나야**
시작한다. 시프트 도중 신호가 바뀌어 억제 구역에 걸리면 원복하지 않고
(급조향 금지) 횡위치를 유지한 채 종방향을 정지 후보에 맡긴다.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402
from test_avoid import (Ap, Box, HARD_TICKS, ESC_TICKS, STATIC_TICKS,  # noqa: E402
                        Planner, World, drive, make, tick)

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
SUP_M = OT['stopline_suppress_m']
HZ = CFG['comm']['send_hz']
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
S0 = CFG['speed']['stop_gap_stopline_m'] + FRONT
A_STOP = CFG['speed']['stop_profile_a']


class TL:
    def __init__(self, state=TrafficLightState.Red, tl_id=7):
        self.state = state
        self.id = tl_id


def rig(d_tl=60.0, state=TrafficLightState.Red, obj_x=(8.0,), junction=False):
    """적신호 정지선이 d_tl 앞, 그 **사이**에 정지 객체가 있는 상황."""
    kr, p = make(d_tl=d_tl)
    p.next_traffic_lights = [TL(state)] * len(p.next_traffic_lights)
    kr._sl_all = []
    ap = Ap(p, [Box(i + 1, x, 0.0, half_w=0.9) for i, x in enumerate(obj_x)],
            junction=junction)
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    tick(kr, ap, p, STATIC_TICKS)
    kr.last_d_end = 1e6
    return kr, p, ap


# ── 절대 규칙: 거리 무관 억제 ────────────────────────────────────────────
@pytest.mark.parametrize('d_tl', [10.0, 35.0, 60.0, 120.0, 250.0])
def test_red_ahead_suppresses_at_any_distance(d_tl):
    """30 m 억제창과 달리 **거리 조건이 없다** — 250 m 밖 적신호도 막는다."""
    kr, p, ap = rig(d_tl=d_tl)
    assert kr._red_ahead(p) == pytest.approx(d_tl)
    kr._try_overtake(ap, p, 0.0)
    assert kr.last_avoid['suppress'] == 'red_ahead'
    assert kr.ot_span is None                       # 시프트 없음


def test_yellow_stop_latch_suppresses_like_red():
    kr, p, ap = rig(state=TrafficLightState.Yellow)
    kr.y_decision = 'stop'
    assert kr._red_ahead(p) is not None
    kr._try_overtake(ap, p, 0.0)
    assert kr.last_avoid['suppress'] == 'red_ahead'


def test_yellow_go_latch_does_not_suppress():
    """황색 GO 는 통과하기로 한 것이므로 이 규칙의 대상이 아니다."""
    kr, p, ap = rig(state=TrafficLightState.Yellow)
    kr.y_decision = 'go'
    assert kr._red_ahead(p) is None


def test_green_reopens_avoidance():
    """녹색 전환 후에는 회피가 열린다 (객체가 계속 정지라면)."""
    kr, p, ap = rig(state=TrafficLightState.Green)
    assert kr._red_ahead(p) is None
    kr._try_overtake(ap, p, 0.0)
    assert (kr.last_avoid or {}).get('suppress') != 'red_ahead'


# ── 회피 상태 전부 × 적신호 ─────────────────────────────────────────────
@pytest.mark.parametrize('v,label', [(10.0, 'PREEMPT/WAIT 속도대'), (0.0, 'REACTIVE 정지')])
def test_no_avoid_state_arms_under_red(v, label):
    kr, p, ap = rig()
    for _ in range(int(OT['wait_before_shift_s'] * HZ) + 40):
        kr._update_obj_timers(ap)
        kr._try_overtake(ap, p, v)
        assert kr.last_avoid['suppress'] == 'red_ahead', label
        assert kr.ot_span is None


def test_breakout_never_enters_under_red():
    """BREAKOUT L1~L4 어느 것도 서지 않는다."""
    kr, p, ap = rig()
    drive(kr, p, ap, HARD_TICKS + ESC_TICKS * 4)
    assert kr.bo_state is None
    assert kr.bo_level == 0
    assert kr.breakout_creep() is False


def test_creep_hook_false_under_red():
    kr, p, ap = rig()
    drive(kr, p, ap, HARD_TICKS + ESC_TICKS * 4)
    assert kr.breakout_creep() is False


def test_yellow_go_hook_false_under_red_stop():
    """황색 STOP 래치 중 GO 훅(signal_release)은 거짓이어야 한다."""
    kr, p, ap = rig(state=TrafficLightState.Yellow)
    kr.y_decision = 'stop'
    assert kr.signal_release(ap) is False


# ── 대기열 판별의 2선 강등 ──────────────────────────────────────────────
def test_queue_judgement_is_skipped_under_red():
    """적신호에서는 대기열 판별이 아예 돌지 않는다 (절대 규칙이 먼저)."""
    kr, p, ap = rig(obj_x=(10.0, 25.0))
    tick(kr, ap, p, STATIC_TICKS)
    assert kr._is_queue(kr._corridor_blockers(ap, p), p, ap) is False
    assert kr.q_ticks == 0


def test_queue_judgement_runs_when_green():
    kr, p, ap = rig(state=TrafficLightState.Green, obj_x=(10.0, 25.0))
    tick(kr, ap, p, STATIC_TICKS)
    kr._is_queue(kr._corridor_blockers(ap, p), p, ap)     # 예외 없이 동작
    assert kr._red_ahead(p) is None


# ── 시프트 기하 게이트 ──────────────────────────────────────────────────
def test_shift_span_into_stopzone_is_rejected():
    """span 이 억제 구역(정지선−30 m)을 넘으면 시작하지 않는다."""
    kr, p = make(d_tl=float('inf'))
    kr._sl_all = [40.0]                                   # 정지선 40 m 앞
    zone_lo = kr._next_stopzone_s(p)
    assert zone_lo == pytest.approx(40.0 - SUP_M)
    span_m = 2 * OT['transition_m'] + OT['extra_before_m'] + OT['extra_after_m']
    assert span_m > zone_lo                                # 이 span 은 기각돼야 한다


def test_stopzone_start_uses_nearest_of_signal_and_stopline():
    kr, p = make(d_tl=100.0)
    kr._sl_all = [40.0]
    assert kr._next_stopzone_s(p) == pytest.approx(40.0 - SUP_M)
    kr2, p2 = make(d_tl=20.0)
    kr2._sl_all = [400.0]
    assert kr2._next_stopzone_s(p2) == pytest.approx(20.0 - SUP_M)


def test_active_shift_is_held_not_reverted_when_red_appears():
    """시프트 진행 중 적신호가 나타나면 원복하지 않는다 — 급조향 금지."""
    kr, p, ap = rig(state=TrafficLightState.Green)
    kr.ot_span = (100, 900)                                # 시프트 진행 중으로 가정
    p.next_traffic_lights = [TL(TrafficLightState.Red)] * len(p.next_traffic_lights)
    kr._try_overtake(ap, p, 2.0)
    assert kr.ot_span == (100, 900)                        # 유지 (원복 안 함)
    assert kr.last_avoid['state'] == 'SHIFT_HOLD'


# ── 적신호 정지 위치는 그대로 (④′ 회귀) ─────────────────────────────────
@pytest.mark.parametrize('d_line', [S0 + 30.0, S0 + 10.0, S0 + 2.0, S0])
def test_red_stop_profile_unaffected_by_avoidance(d_line):
    """회피 억제가 걸려도 정지 프로파일은 그대로 — 계획 정지점 −1.50 m."""
    kr, p, ap = rig(d_tl=d_line)
    v = kr._stopline_profile(p, ap)
    assert v == pytest.approx(math.sqrt(2 * A_STOP * max(0.0, d_line - S0)))
    if d_line == S0:
        assert v == pytest.approx(0.0)                     # 계획 정지점에서 0


def test_planned_stop_is_inside_the_scoring_band():
    """계획 정지점 slf = −stop_gap_stopline_m 는 −2.0 ≤ slf ≤ −0.3 밴드 안."""
    slf = FRONT - S0
    assert -2.0 <= slf <= -0.3
