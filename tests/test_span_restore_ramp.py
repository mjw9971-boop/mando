"""시프트 원복을 **전이로** 푼다 (overtake.span_restore_ramp_enable).

종전 원복(`kr_rules._restore_span`)은 span [a,b) 전체를 한 틱에 원 경로로
덮었다. 자차가 span 평지 한복판이면 발밑과 바로 앞의 경로가 차로폭만큼
**계단**으로 옮겨 붙어 다음 틱에 조향이 곧바로 풀락이 된다.

실측 2026-09-10 `20260910_202706/실전주행_교통류_02_직진11`:
  n=7020~7034  SHIFT_ACTIVE span [22586, 24201] (ppm 10 → rs 2258.6~2420.1)
  n=7035       rs 2380.5 — span **안**에서 `targets_lost` 원복
  n=7036~7046  조향 −0.480 포화 11틱 (직전 틱 0.000, t_off 0.00)
  n=7047       차로 −1 → −2, heading_err −0.609 rad, 진폭 4.4 m S자
그 구간 `avoid.state` 는 원복 뒤 끝까지 None 이다 — 회피 시프트도 계획
차선변경도 아니고 **원복 자체**가 만든 궤적이다.

시프트를 **만들 때** 전이를 두는 것과 같은 이유로 풀 때도 전이가 필요하다.
잣대도 같은 축이다 (`_ramp_len_m`, `a_lat_max`, 시작은 자차 앞 shift_ahead_m).
"""
import copy
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.route import VtdRoutePlanner

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
PPM = 10
D = 3.2                                    # 차로폭 = 시프트 변위


def on_cfg(**over):
    c = copy.deepcopy(CFG)
    c['overtake']['span_restore_ramp_enable'] = True
    c['overtake'].update(over)
    return c


def off_cfg():
    c = copy.deepcopy(CFG)
    c['overtake']['span_restore_ramp_enable'] = False
    return c


class P:
    """x 축 직선 경로에 [a,b) 가 y = −D 로 밀려 있는 상태 — 실주행 평지 재현.

    `restore_route_smoothly` 는 **route.py 원본을 그대로** 매단다. 목이 아니라
    실제 구현을 재야 계단이 사라졌는지 검증이 된다.
    """

    restore_route_smoothly = VtdRoutePlanner.restore_route_smoothly
    _smooth_transition = VtdRoutePlanner._smooth_transition

    def __init__(self, a, b, i0, n=30000):
        self.points_per_meter = PPM
        self.route_index = i0
        self.original_route_points = np.stack(
            [np.arange(n) / PPM, np.zeros(n), np.zeros(n)], axis=1)
        self.route_points = self.original_route_points.copy()
        self.route_s = np.arange(n) / PPM
        self.commands_orig = np.zeros(n, dtype=int)
        self.commands = self.commands_orig.copy()
        self._lat_build = np.zeros(n)
        self.lat_shift = np.zeros(n)
        # 평지 시프트 — 전이 없이 [a,b) 전체를 −D 로 (원복만 재는 시험이므로 충분)
        self.route_points[a:b, 1] = -D
        self.lat_shift[a:b] = -D
        self._kd = None


def kr_at(cfg, a, b, i0):
    kr = KrRules(cfg)
    kr.ot_span = (a, b)
    return kr, P(a, b, i0)


def lat_at(p, idx):
    """경로점의 횡변위 [m] (원 경로 대비, 부호 그대로)."""
    return float(p.route_points[idx, 1])


def max_step(p, lo, hi):
    """[lo,hi) 안 이웃 경로점 사이 **최대 횡 계단** [m]."""
    y = p.route_points[lo:hi, 1]
    return float(np.max(np.abs(np.diff(y)))) if len(y) > 1 else 0.0


# ── 원본 결함 고정 ──────────────────────────────────────────────────────
def test_off_steps_the_path_under_the_ego():
    """off = 이전 동작. 자차 발밑 경로가 한 틱에 차로폭만큼 옮겨 붙는다."""
    kr, p = kr_at(off_cfg(), 22586, 24201, 23805)
    assert lat_at(p, 23805) == pytest.approx(-D)
    kr._restore_span(p, ego_speed=9.6)
    assert lat_at(p, 23805) == pytest.approx(0.0)          # 계단 — 이것이 결함이다
    assert kr._restore_diag is None


def test_on_keeps_the_path_under_the_ego():
    """on = 자차 발밑·바로 앞은 그대로. 계단이 없다."""
    kr, p = kr_at(on_cfg(), 22586, 24201, 23805)
    kr._restore_span(p, ego_speed=9.6)
    assert lat_at(p, 23805) == pytest.approx(-D)           # 발밑 불변
    ahead = int(kr.shift_ahead_m * PPM)
    assert lat_at(p, 23805 + ahead - 1) == pytest.approx(-D)


def test_on_removes_the_lateral_step():
    """전이 구간의 점간 계단이 요구 횡가속 상한과 맞는 크기로 줄어든다."""
    kr, p = kr_at(on_cfg(), 22586, 24201, 23805)
    kr._restore_span(p, ego_speed=9.6)
    # 코사인 램프의 최대 기울기 = πΔ/(2L) → 점당 계단은 그 1/ppm 이다.
    L = kr._ramp_len_m(1, D, 9.6)
    assert max_step(p, 23805, 24201) < math.pi * D / (2.0 * L * PPM) * 1.05


def test_on_finishes_at_the_span_end():
    """전이가 끝나면 원 경로다 — 원복이 안 되고 남으면 항목 [3] 이 된다."""
    kr, p = kr_at(on_cfg(), 22586, 24201, 23805)
    kr._restore_span(p, ego_speed=9.6)
    assert lat_at(p, 24200) == pytest.approx(0.0, abs=1e-6)
    assert p.lat_shift[24200] == pytest.approx(0.0, abs=1e-6)


def test_ramp_is_sized_by_speed():
    """램프 길이 잣대는 생성부와 같은 축이다 — 빠를수록 길다."""
    out = []
    for v in (5.56, 9.6, 14.0):
        kr, p = kr_at(on_cfg(), 22586, 24201, 23805)
        kr._restore_span(p, ego_speed=v)
        out.append(kr._restore_diag['ramp_m'])
    assert out[0] < out[1] < out[2]
    kr, p = kr_at(on_cfg(), 22586, 24201, 23805)
    kr._restore_span(p, ego_speed=9.6)
    assert kr._restore_diag['ramp_m'] == pytest.approx(
        round(kr._ramp_len_m(1, D, 9.6), 1), abs=0.15)


def test_slow_ego_uses_the_avoid_cap_speed():
    """정지 중 원복이면 v 를 lm_avoid_v 로 본다 — 램프가 0 이 되면 다시 계단이다."""
    kr, p = kr_at(on_cfg(), 22586, 24201, 23805)
    kr._restore_span(p, ego_speed=0.0)
    assert kr._restore_diag['v'] == pytest.approx(kr.lm_avoid_v, abs=0.01)
    assert kr._restore_diag['ramp_m'] > 0.0


# ── 하드 원복이 옳은 자리 ───────────────────────────────────────────────
def test_ego_past_the_span_restores_hard():
    """자차가 span 을 지났으면 계단이 생길 자리가 없다 — 전이를 만들지 않는다."""
    kr, p = kr_at(on_cfg(), 22586, 24201, 24500)
    kr._restore_span(p, ego_speed=9.6)
    assert lat_at(p, 23805) == pytest.approx(0.0)
    assert kr._restore_diag is None


def test_ego_before_the_span_restores_hard():
    kr, p = kr_at(on_cfg(), 22586, 24201, 22000)
    kr._restore_span(p, ego_speed=9.6)
    assert lat_at(p, 23805) == pytest.approx(0.0)
    assert kr._restore_diag is None


def test_hard_flag_wins():
    """순간이동 리셋 — 경로 위 위치가 통째로 바뀐 뒤라 전이가 의미 없다."""
    kr, p = kr_at(on_cfg(), 22586, 24201, 23805)
    kr._restore_span(p, ego_speed=9.6, hard=True)
    assert lat_at(p, 23805) == pytest.approx(0.0)
    assert kr._restore_diag is None


def test_tiny_displacement_restores_hard():
    """변위가 없으면 계단도 없다 — 쓸데없는 전이를 만들지 않는다."""
    kr, p = kr_at(on_cfg(), 22586, 24201, 23805)
    p.route_points[22586:24201, 1] = -0.1
    p.lat_shift[22586:24201] = -0.1
    kr._restore_span(p, ego_speed=9.6)
    assert lat_at(p, 23805) == pytest.approx(0.0)
    assert kr._restore_diag is None


def test_short_remaining_span_compresses_the_ramp():
    """잔여 span 이 램프보다 짧으면 압축한다 — 계단(무한대)보다는 언제나 낫다."""
    b = 23805 + 200                               # 자차 앞 20 m 뿐
    kr, p = kr_at(on_cfg(), 22586, b, 23805)
    kr._restore_span(p, ego_speed=9.6)
    assert kr._restore_diag is not None
    assert kr._restore_diag['ramp_m'] < kr._ramp_len_m(1, D, 9.6)
    assert lat_at(p, 23805) == pytest.approx(-D)   # 발밑은 그래도 안 튄다


def test_no_room_for_the_lead_in_restores_hard():
    """자차 앞 전이 여유(shift_ahead_m)조차 없으면 하드 폴백."""
    kr, p = kr_at(on_cfg(), 22586, 23810, 23805)
    kr._restore_span(p, ego_speed=9.6)
    assert kr._restore_diag is None


def test_planner_without_the_helper_falls_back():
    """구 서명 목 플래너 — AttributeError 로 이전 동작에 맡긴다."""
    kr, p = kr_at(on_cfg(), 22586, 24201, 23805)
    del type(p).restore_route_smoothly
    try:
        kr._restore_span(p, ego_speed=9.6)
        assert lat_at(p, 23805) == pytest.approx(0.0)
        assert kr._restore_diag is None
    finally:
        type(p).restore_route_smoothly = VtdRoutePlanner.restore_route_smoothly


# ── 뒷정리는 스위치와 무관하다 ──────────────────────────────────────────
@pytest.mark.parametrize('cfg', [on_cfg(), off_cfg()])
def test_state_is_cleared_either_way(cfg):
    kr, p = kr_at(cfg, 22586, 24201, 23805)
    kr.ot_ids = [28]
    kr.ot_side = 'right'
    kr._restore_span(p, ego_speed=9.6)
    assert kr.ot_span is None and kr.ot_ids == [] and kr.ot_side is None
    assert kr.last_overtake == 'restored'
