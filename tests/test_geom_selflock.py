"""geom 자기잠금 — (b) 회피 가능성 보존 + (c) 속도로 전이 길이 재계산.

구조: standoff 가 22 m 에 세우고 → 크립이 s_rel 을 소모하고 → `need`
(= trans_m + ahead_m + geom_margin_m) 밑으로 내려가면 **영구 기각**이다.
거리를 쓴 뒤에 판단하니 되돌릴 수 없다.

실측(2026-09-10): geom 기각 3,803틱의 s_rel 중앙값 **3.9 m**, need 는 15.0/19.0.
blocker 별로 보면 기각 전에 s_rel 이 need 이상이던 적이 **있다** — 07 id 9 는
78.3 m 까지 기회가 있었는데 10.1 m 에서 기각됐다.

  (c) 옛 식 max(transition_m(12), shift_k_s·v) 는 정지 상태에서도 12 m 를
      요구한다. `_ramp_len_m` 의 L ≥ v·π·√(Y/(2·a_lat_max)) 로 바꾸면
      5 km/h 에서 4.5 m 다 (need 15.0 → 7.5).
**크립 보존(A)은 채택하지 않았다** — avoid_sim 12 케이스에서 off/on 이 전건
동일했고(효과 0), 로그 분석도 원인이 크립이 아님을 보였다. 진짜 원인은
시프트 중 `lane_plan` 이 None 인 것이다 (lane_map_shift_ref_enable).
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

from kr_rules import KrRules                                       # noqa: E402
from test_avoid import Ap, Box, Planner                            # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)


def kr(**over):
    c = copy.deepcopy(CFG)
    c['overtake'].update(over)
    return KrRules(c)


class _Lg:
    lanes = {}


# ── (c) 전이 길이 ────────────────────────────────────────────────────────
def test_trans_off_keeps_the_12m_floor_at_standstill():
    k = kr(trans_m_by_speed_enable=False)
    assert k._trans_need_m(None, None, 0.0, 0.0, 1) == pytest.approx(k.ot_trans_m)


def test_trans_on_shrinks_at_low_speed_but_not_below_the_steering_floor():
    """저속에서 줄되 **조향 한계** 밑으로는 못 간다.

    `_ramp_len_m` 은 횡가속만 보므로 v→0 에서 0 으로 수렴하는데, 실제로는
    조향각이 먼저 막힌다. 고정 4.0 을 쓰면 램프가 포화한다 — 실측
    avoid_sim 11 에서 조향 포화 0 → 27틱 (2026-09-10).
    """
    k = kr(trans_m_by_speed_enable=True)
    v = 5.0 / 3.6
    got = k._trans_need_m(None, None, 0.0, v, 1)
    floor = k._steer_min_trans_m(1, 3.2)
    assert got == pytest.approx(floor, abs=0.3)
    assert got < k.ot_trans_m


def test_steering_floor_comes_from_vehicle_params():
    """상수를 새로 만들지 않는다 — wheelbase·max_steer 에서 유도한다."""
    k = kr(trans_m_by_speed_enable=True)
    L = float(CFG['vehicle']['wheelbase']); ms = float(CFG['vehicle']['max_steer'])
    kmax = math.tan(ms) / L
    assert k._steer_min_trans_m(1, 3.2) == pytest.approx(
        math.pi * math.sqrt(3.2 / (2.0 * kmax)), rel=1e-6)


def test_steering_floor_grows_with_hops():
    k = kr(trans_m_by_speed_enable=True)
    assert k._steer_min_trans_m(2, 3.2) > k._steer_min_trans_m(1, 3.2)


def test_trans_on_never_exceeds_the_old_value():
    """목적은 저속에서 줄이는 것이지 고속에서 늘리는 것이 아니다."""
    k = kr(trans_m_by_speed_enable=True)
    for v in (0.0, 2.0, 5.56, 8.33, 12.5):
        old = max(k.ot_trans_m, k.shift_k_s * max(v, 0.1))
        assert k._trans_need_m(None, None, 0.0, v, 1) <= old + 1e-9


def test_trans_on_respects_the_floor():
    """정지 상태에서도 조향 바닥 아래로는 안 내려간다."""
    k = kr(trans_m_by_speed_enable=True, trans_min_m=4.0)
    got = k._trans_need_m(None, None, 0.0, 0.0, 1)
    assert got == pytest.approx(k._steer_min_trans_m(1, 3.2), abs=0.3)
    assert got > 4.0


def test_switch_defaults_are_off():
    o = CFG['overtake']
    assert o.get('trans_m_by_speed_enable') is False
    assert o.get('trans_min_m') == 4.0
