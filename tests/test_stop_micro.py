"""[2](C)(D) 2차 주행 회귀 — 정지선 접근 품질과 장애물 정지 탈출구 (2026-09-09).

`logs/batch/20260909_093415` (교통류가 실제로 접속된 첫 배치) 에서 나온 네 가지다.

(C) 정지선 앞 **찔끔찔끔** — 원인 둘
  · PDM 의 red-light IDM 은 차간모형이라 남은 거리가 s* 보다 조금만 커도
    **가속을 요구한다**. ④′ 는 √ 프로파일이라 그 구간에서 아직 v 위에 있어
    min() 에서 지고, 차가 붙었다 급제동한다.
    실측 실경로_02: 적색 접근 667틱 중 **290틱(43.5 %)** 목표 > 현재속도,
    최대 **+4.17 m/s** (slf −14.7 에서 v 5.44 인데 목표 6.15 → accel +0.57).
  · 정지 뒤 stopline_hold_s(0.5 s)가 끝나면 0.03~0.12 m/s 가 새어 0.2 m 씩
    기어간다 (실측 02 정지 3회 중 2회, 07 5회 중 2회).

(D) 장애물 앞 **영구 정지** — 원인 둘
  · 황색 판정에 거리 한계가 없어 **300 m 밖** 황색이 STOP 을 래치하고,
    그 래치가 `_obstacle_cause` 를 거짓으로 만들어 크립·BREAKOUT·never_stall 을
    전부 죽인다 (11 rs 477.7 d_line **150.8 m**, 18 rs 2590.5 **317.3 m**).
  · PDM 의 `traffic_light_hazard` 도 같은 일을 한다 — 정지선이 150 m 밖인데
    20 m 앞 정지 차량에 막힌 경우까지 신호 대기로 분류한다
    (11 정지 1635틱 중 **1167틱**이 이 분기).

전부 **끄는 스위치**를 갖는다. off = 이전 동작.
"""
import copy
import math
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                        # noqa: E402
from test_stop_profile import S0, make_ap                           # noqa: E402
from test_stopline_stop import FakePlannerTL                        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)


def cfg_on(**over):
    c = copy.deepcopy(CFG)
    c['speed'].update(no_accel_toward_red_enable=True,
                      stopline_creep_latch_enable=True,
                      tl_hazard_far_blocker_enable=True)
    c['speed'].update(over)
    return c


def cfg_off(**over):
    """이전 동작 사본 — 기본값을 읽지 않는다 (2026-09-07 드리프트 원칙)."""
    c = copy.deepcopy(CFG)
    c['speed'].update(no_accel_toward_red_enable=False,
                      stopline_creep_latch_enable=False,
                      tl_hazard_far_blocker_enable=False)
    c['speed'].update(over)
    return c


def rig(cfg, d_line):
    p = FakePlannerTL(d_tl=d_line)
    ap = make_ap(p, cfg)
    ap.kr_rules = KrRules(cfg)
    ap.kr_rules._ap = ap
    ap.kr_rules._sl_all = []
    return ap.kr_rules, p, ap


# ── (C-1) 적신호 접근 가속 금지 ──────────────────────────────────────────
def test_cap_is_the_current_speed_on_a_near_red():
    kr, p, ap = rig(cfg_on(), S0 + 20.0)
    assert kr._no_accel_red_cap(p, ap, 5.44) == pytest.approx(5.44)


def test_cap_is_absent_beyond_the_red_lookahead():
    """거리 한계는 speed.red_lookahead_m 를 그대로 읽는다 (새 상수 없음)."""
    look = CFG['speed']['red_lookahead_m']
    kr, p, ap = rig(cfg_on(), S0 + look + 20.0)
    assert kr._no_accel_red_cap(p, ap, 8.0) is None
    kr2, p2, ap2 = rig(cfg_on(), look - 1.0)
    assert kr2._no_accel_red_cap(p2, ap2, 8.0) == pytest.approx(8.0)


def test_cap_is_absent_without_a_stop_target():
    kr, p, ap = rig(cfg_on(), float('inf'))
    assert kr._no_accel_red_cap(p, ap, 8.0) is None


def test_cap_is_inert_when_off():
    kr, p, ap = rig(cfg_off(), S0 + 20.0)
    assert kr._no_accel_red_cap(p, ap, 5.44) is None


def test_cap_never_forces_a_negative_target():
    kr, p, ap = rig(cfg_on(), S0 + 20.0)
    assert kr._no_accel_red_cap(p, ap, -1.0) == 0.0


# ── (C-2) 정지 후 미세 전진 래치 ─────────────────────────────────────────
def test_latch_keeps_zero_after_the_hold_expires():
    kr, p, ap = rig(cfg_on(), S0 + 1.0)
    for _ in range(int(kr.sl_hold_ticks) + 5):
        kr._stopline_hold(p, 0.0)
    assert kr._stopline_hold(p, 0.0) == 0.0                # 홀드가 끝나도 0


def test_latch_releases_when_the_target_disappears():
    """녹색이 되면 대상이 사라져 **저절로** 풀린다 (출발 지연 0)."""
    kr, p, ap = rig(cfg_on(), S0 + 1.0)
    for _ in range(int(kr.sl_hold_ticks) + 5):
        kr._stopline_hold(p, 0.0)
    p.tl.state = type(p.tl.state)(0) if hasattr(p.tl.state, '__call__') else p.tl.state
    from vtd_adapter.carla_types import TrafficLightState
    p.tl.state = TrafficLightState.Green
    assert kr._stopline_hold(p, 0.0) is None


def test_latch_is_inert_when_off():
    kr, p, ap = rig(cfg_off(), S0 + 1.0)
    for _ in range(int(kr.sl_hold_ticks) + 5):
        kr._stopline_hold(p, 0.0)
    assert kr._stopline_hold(p, 0.0) is None               # 이전 동작


def test_latch_needs_an_actual_stop():
    """서지 않았으면 래치가 무장되지 않는다."""
    kr, p, ap = rig(cfg_on(), S0 + 1.0)
    for _ in range(int(kr.sl_hold_ticks) + 5):
        kr._stopline_hold(p, 5.0)                          # 계속 달리는 중
    assert kr._stopline_hold(p, 5.0) is None


# ── (D-1) 황색 판정 거리 한계 ────────────────────────────────────────────
def test_yellow_does_not_latch_beyond_the_gate():
    """실측 재현 — 150 m / 317 m 밖 황색은 딜레마가 아니다."""
    from vtd_adapter.carla_types import TrafficLightState
    for d in (150.8, 317.3):
        kr, p, ap = rig(cfg_on(), d)
        p.tl.state = TrafficLightState.Yellow
        p.tl.id = 7
        kr._yellow_latch(p, 8.0, ap)
        assert kr.y_decision is None, f'{d} m 밖 황색이 래치됐다'


def test_yellow_still_latches_inside_the_gate():
    from vtd_adapter.carla_types import TrafficLightState
    kr, p, ap = rig(cfg_on(), S0 + 10.0)
    p.tl.state = TrafficLightState.Yellow
    p.tl.id = 7
    kr._yellow_latch(p, 3.0, ap)
    assert kr.y_decision == 'stop'


def test_yellow_gate_zero_is_the_old_behaviour():
    from vtd_adapter.carla_types import TrafficLightState
    kr, p, ap = rig(cfg_on(yellow_decide_max_m=0.0), 317.3)
    p.tl.state = TrafficLightState.Yellow
    p.tl.id = 7
    kr._yellow_latch(p, 8.0, ap)
    assert kr.y_decision == 'stop'                         # 이전 동작


def test_yellow_gate_reads_the_same_scale_as_red_lookahead():
    """새 눈금을 만들지 않았다 — red_lookahead_m 와 같은 값이다."""
    assert CFG['speed']['yellow_decide_max_m'] == CFG['speed']['red_lookahead_m']


# ── (D-2) 먼 정지선 + 가까운 장애물 ─────────────────────────────────────
def test_hazard_relaxation_needs_a_blocker():
    kr, p, ap = rig(cfg_on(), S0 + 5.0)
    assert kr._blocker_before_stopline(ap, p) is False     # 장애물이 없다


def test_hazard_relaxation_is_inert_when_off():
    kr, p, ap = rig(cfg_off(), S0 + 5.0)
    assert kr._blocker_before_stopline(ap, p) is False


def test_relaxation_threshold_is_the_queue_head_constant():
    """새 상수를 만들지 않았다 — overtake.queue_head_max_m 를 그대로 읽는다."""
    kr, _p, _ap = rig(cfg_on(), S0 + 5.0)
    assert kr.q_head_m == CFG['overtake']['queue_head_max_m']


def test_switches_are_on_by_default():
    """실측으로 검증해 켰다 — off 는 사본에서 명시적으로 끄고 본다."""
    assert CFG['speed']['no_accel_toward_red_enable'] is True
    assert CFG['speed']['stopline_creep_latch_enable'] is True
    assert CFG['speed']['tl_hazard_far_blocker_enable'] is True
    assert CFG['signal']['signal_turn_tail_yield_enable'] is True
