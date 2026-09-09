"""
ctrl24 K7 — 시프트 전이 횡가속 상한 (shift_cap).

여기서 지키는 불변:
  · **게이트가 아니다.** 켜고 꺼도 시프트 생성·span·NOOP 판정이 같아야 한다.
  · 진행 중인 시프트(ot_span)가 없으면 후보를 내지 않는다 — 시프트 없는 구간
    (예: 교차로 연결로)에 관여하면 안 된다.
  · 값은 v ≤ max(shift_cap_min_v, √(a_lat_max/κ)), κ 는 회피 시프트 성분
    (lat_shift − _lat_build)의 횡곡률.
  · 계획 차선변경(_lat_build)만 있는 구간은 κ = 0 — 후보 없음.
  · min() 후보다. 더 낮은 후보가 있으면 그쪽이 목표를 잡는다.
"""
import copy
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from test_ctrl24_avoid import LANE, apply, car, rig             # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
OT = CFG['overtake']


def off_cfg(**over):
    """params 값이 정본 — off 경로는 사본에서 명시적으로 끈다 (B-25 관례)."""
    c = copy.deepcopy(CFG)
    c['ctrl24']['shift_cap_enable'] = False
    c['overtake'].update(over)
    return c


def on_cfg(**over):
    c = copy.deepcopy(CFG)
    c['overtake'].update(over)
    return c


def cos_cap(delta, trans_m, a_lat=None):
    """코사인 전이의 해석 상한 — κ = (Δ/2)(π/L)²."""
    a_lat = OT['a_lat_max'] if a_lat is None else a_lat
    kappa = 0.5 * delta * (math.pi / trans_m) ** 2
    return max(OT['shift_cap_min_v'], math.sqrt(a_lat / kappa))


# ── 스위치·기본값 ────────────────────────────────────────────────────────
def test_params_switch_on_and_constants_come_from_overtake_section():
    assert C['shift_cap_enable'] is True
    assert 'a_lat_max' not in C and 'shift_cap_min_v' not in C   # 값은 한 곳에서만
    kr, _p, _ap = rig()
    assert kr.a_lat_max == OT['a_lat_max'] > 0.0
    assert kr.shift_cap_min_v == OT['shift_cap_min_v']
    assert kr.shift_cap_look_m == OT['shift_latest_m']


# ── 시프트가 없으면 산출 없음 ──────────────────────────────────────────────
def test_no_candidate_without_active_span():
    kr, p, ap = rig()
    apply(kr, ap, v=12.5)
    assert kr.ot_span is None
    assert kr.last_kr['shift_cap'] is None
    assert kr._shift_speed_cap(p, 12.5) is None


def test_no_candidate_for_planned_lane_change_only():
    """계획 차선변경(_lat_build)만 있는 경로 — 회피 시프트 성분이 0이라 κ = 0."""
    kr, p, ap = rig()
    n = len(p.lat_shift)
    s = np.arange(n) / 10.0
    p._lat_build = 1.5 * (1.0 - np.cos(np.clip(s / 20.0, 0, 1) * math.pi)) / 2.0
    p.lat_shift = p._lat_build.copy()
    kr.ot_span = (0, n - 1)                                      # span 은 있다고 치더라도
    assert kr._shift_speed_cap(p, 8.0) is None


# ── 값 ───────────────────────────────────────────────────────────────────
def test_cap_matches_cosine_transition_of_the_shift_that_was_created():
    """정지 중 만든 시프트는 전이 8 m 로 굳는다 — 그 전이를 지날 상한."""
    kr, p, ap = rig(actors=[car(2, 12.0)])
    apply(kr, ap, v=0.0)                                         # trans = trans_min_m
    assert kr.last_avoid['trans_m'] == pytest.approx(C['trans_min_m'])
    cap = kr._shift_speed_cap(p, 6.0)
    assert cap == pytest.approx(cos_cap(LANE, C['trans_min_m']), abs=0.1)
    assert cap < 6.0                                             # 실제로 구속한다


def test_longer_transition_gives_higher_cap():
    """빠를 때 만든 시프트는 전이가 길어 상한이 높다 — 상한이 곡률만 본다는 확인."""
    kr, p, ap = rig(actors=[car(2, 78.0)])
    apply(kr, ap, v=12.5)
    trans = C['trans_k'] * 12.5
    assert kr._shift_speed_cap(p, 12.5) == pytest.approx(cos_cap(LANE, trans), abs=0.3)


def test_cap_never_below_floor():
    kr, p, ap = rig(cfg=on_cfg(a_lat_max=0.01), actors=[car(2, 12.0)])
    apply(kr, ap, v=0.0)
    assert kr._shift_speed_cap(p, 6.0) == pytest.approx(OT['shift_cap_min_v'])


def test_plateau_and_past_span_are_inactive():
    """평지(변위 일정)에서는 κ = 0, span 을 넘기면 창이 닫혀 후보 자체가 없다."""
    kr, p, ap = rig(actors=[car(2, 78.0)])
    apply(kr, ap, v=12.5)
    a, b = kr.ot_span
    flat = p.lat_shift.copy()
    flat[a:b] = LANE                                             # 전이 없이 평지만
    p.lat_shift = flat
    p.route_index = a + 100
    assert kr._shift_speed_cap(p, 12.5) is None                  # κ = 0
    p.route_index = b + 10                                       # span 밖 → 창이 안 열린다
    assert kr._shift_speed_cap(p, 12.5) is None


def test_window_previews_the_transition_before_reaching_it():
    """미리보기 창 — 전이에 **닿기 전에** 상한이 나와야 감속할 시간이 있다."""
    kr, p, ap = rig(actors=[car(2, 12.0)])
    apply(kr, ap, v=0.0)
    a, _b = kr.ot_span
    p.route_index = max(0, a - int(0.5 * kr.shift_cap_look_m * 10))
    assert kr._shift_speed_cap(p, 6.0) == pytest.approx(cos_cap(LANE, C['trans_min_m']), abs=0.1)


# ── 게이트가 아니다 ──────────────────────────────────────────────────────
def test_switch_off_changes_speed_only_not_the_shift():
    on = rig(cfg=on_cfg(), actors=[car(2, 12.0)])
    off = rig(cfg=off_cfg(), actors=[car(2, 12.0)])
    for kr, p, ap in (on, off):
        apply(kr, ap, v=0.0)
        apply(kr, ap, v=6.0)
    kr_on, p_on, _ = on
    kr_off, p_off, _ = off
    assert kr_on.ot_span == kr_off.ot_span and kr_on._shifted_for == kr_off._shifted_for
    assert kr_on.last_avoid['state'] == kr_off.last_avoid['state']
    assert np.allclose(p_on.route_points[:, 1], p_off.route_points[:, 1])
    assert kr_off.last_kr['shift_cap'] is None
    assert kr_on.last_kr['shift_cap'] is not None
    assert kr_on.last_target < kr_off.last_target


def test_cap_is_a_min_candidate_and_reported_in_diagnostics():
    kr, p, ap = rig(actors=[car(2, 12.0)])
    apply(kr, ap, v=0.0)
    _c, target = apply(kr, ap, v=6.0)
    cap = kr.last_kr['shift_cap']
    assert target == pytest.approx(min(12.5, cap), abs=1e-3)   # last_kr 은 3자리 반올림
    assert kr.last_kr_winner == 'shift_cap'
    assert kr.last_avoid['shift_cap'] == pytest.approx(cap, abs=0.01)


def test_lower_candidate_wins_over_cap():
    """보행자·신호가 더 낮으면 그쪽이 잡는다 — 상한이지 오버라이드가 아니다."""
    kr, p, ap = rig(actors=[car(2, 12.0)])
    apply(kr, ap, v=0.0)
    apply(kr, ap, v=6.0)
    cap = kr.last_kr['shift_cap']
    kr._rtor_cap = lambda: cap - 0.5                             # 더 낮은 후보 하나
    _c, target = apply(kr, ap, v=6.0)
    assert target == pytest.approx(cap - 0.5, abs=1e-6)
    assert kr.last_kr_winner == 'rtor_cap'
