"""
회피 시프트 전이 — 횡가속 상한(속도 캡) + 중앙선 절대 금지 (P1).

실측 근거 (2026-09-01, logs/batch/20260901_174724/실전주행_교통류_01_좌회전24):
  · t=98.863 WAIT_EXPIRED 로 우측 시프트 생성. 그때 자차 v≈1.0 m/s 라
    trans_m = max(transition_m 12.0, shift_k_s·v) = 12.0 으로 굳었다.
  · 적신호 SHIFT_HOLD 로 21 s 정지 후 6.5 m/s 로 복귀 전이를 통과 →
    요구 횡가속 Δπ²v²/(2L²) = 4.34 m/s². 조향이 −0.480 → +0.480 양방향
    풀락으로 포화하며 경로 대비 **+1.45 m 오버슛**, 황색 중앙선을 0.94 m 침범
    (0.60 s 지속 = 항목4 중대 임계. 채점기는 0.599989 s 로 11 μs 차이로 놓쳤다).

여기서 지키는 불변:
  · 진행 중인 시프트는 **다시 밀지 않는다** — 속도만 낮춘다 (급조향 금지).
  · 상한은 lat_shift 의 2계 미분에서 나온다 — 전이 형상·길이를 가정하지 않고,
    평지·span 밖에서는 스스로 비활성이다.
  · 황색 중앙선은 BREAKOUT 단계와 무관하게 넘지 않는다 (L5 미구현).
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

from kr_rules import KrRules                                   # noqa: E402
from test_avoid import Planner                                 # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
A_LAT = OT['a_lat_max']
MIN_V = OT['shift_cap_min_v']
TRANS = OT['transition_m']


def cosine_shift(n, lo, hi, delta, trans_pts):
    """[lo, hi) 에 '나갔다 돌아오는' 코사인 전이를 심은 lat_shift 배열."""
    lat = np.zeros(n)
    for j in range(lo, min(hi, n)):
        u = j - lo
        span = hi - lo
        if u < trans_pts:                                  # 나가는 전이
            lat[j] = delta * 0.5 * (1.0 - math.cos(math.pi * u / trans_pts))
        elif u > span - trans_pts:                         # 복귀 전이
            w = (span - u) / trans_pts
            lat[j] = delta * 0.5 * (1.0 - math.cos(math.pi * w))
        else:
            lat[j] = delta                                 # 평지
    return lat


def make(delta=3.0, trans_m=TRANS, cfg=CFG, index=0, span_from=0):
    kr = KrRules(cfg)
    p = Planner()
    n = len(p.route_s)
    ppm = p.points_per_meter
    lo = span_from
    hi = lo + int((2.0 * trans_m + 15.0) * ppm)
    p.lat_shift = cosine_shift(n, lo, hi, delta, int(trans_m * ppm))
    p.route_index = index
    kr.ot_span = (lo, hi)
    return kr, p


# ── 속도 캡 ──────────────────────────────────────────────────────────────
def test_cap_matches_cosine_transition_formula():
    """코사인 전이의 최대 횡곡률 κ = Δπ²/(2L²) → v = √(a_lat_max/κ)."""
    delta, L = 3.0, TRANS
    kr, p = make(delta=delta, trans_m=L)
    got = kr._shift_speed_cap(p, 6.5)
    kappa = delta * math.pi ** 2 / (2.0 * L ** 2)
    want = math.sqrt(A_LAT / kappa)
    assert got == pytest.approx(want, rel=0.02)            # 0.5 m 스텐실 오차 이내


def test_cap_would_have_bound_the_measured_case():
    """실측 조건(Δ=3.0, L=12.0)에서 상한이 6.5 m/s 를 실제로 구속한다."""
    kr, p = make(delta=3.0, trans_m=12.0)
    assert kr._shift_speed_cap(p, 6.5) < 6.5


def test_longer_transition_allows_more_speed():
    """전이가 길수록 상한이 올라간다 (v ∝ L)."""
    a = make(trans_m=12.0)[0]._shift_speed_cap(make(trans_m=12.0)[1], 6.5)
    b = make(trans_m=30.0)[0]._shift_speed_cap(make(trans_m=30.0)[1], 6.5)
    assert b > a


def test_no_cap_without_active_span():
    """시프트가 없으면 후보를 만들지 않는다."""
    kr, p = make()
    kr.ot_span = None
    assert kr._shift_speed_cap(p, 6.5) is None


def test_no_cap_on_flat_profile():
    """lat_shift 가 평평하면(전이 없음) κ=0 → 미개입."""
    kr, p = make()
    p.lat_shift = np.zeros(len(p.route_s))
    assert kr._shift_speed_cap(p, 6.5) is None


def test_no_cap_when_transition_is_behind():
    """이미 지나온 전이는 상한을 만들지 않는다 (앞 창만 본다)."""
    kr, p = make(span_from=0)
    p.route_index = int((2.0 * TRANS + 20.0) * p.points_per_meter)
    assert kr._shift_speed_cap(p, 6.5) is None


def test_cap_has_lower_bound():
    """전이가 아무리 급해도 shift_cap_min_v 밑으로는 안 내린다."""
    kr, p = make(delta=6.0, trans_m=2.0)
    assert kr._shift_speed_cap(p, 6.5) == pytest.approx(MIN_V)


def test_switch_off_disables_cap():
    """a_lat_max = 0 이면 완전 비활성 = 이전 동작."""
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['a_lat_max'] = 0.0
    kr, p = make(cfg=cfg)
    assert kr._shift_speed_cap(p, 6.5) is None


# ── 중앙선 절대 금지 ─────────────────────────────────────────────────────
class Lg:
    """mark_at 만 있는 최소 레인그래프 목."""

    def __init__(self, color):
        self.color = color

    def mark_at(self, key, s, side):
        return ('solid', self.color, False)


def test_yellow_mark_is_center_line():
    kr = KrRules(CFG)
    assert kr._is_center_mark(Lg('yellow'), (1, 0, -1), 'left', 0.0) is True


def test_white_mark_is_not_center_line():
    kr = KrRules(CFG)
    assert kr._is_center_mark(Lg('standard'), (1, 0, -1), 'left', 0.0) is False


def test_center_line_gate_is_level_independent():
    """BREAKOUT L2(실선 허용)·L3·L4 어느 단계에서도 중앙선은 막힌다.

    dashed 게이트는 `lvl < 2` 에서만 걸리므로, 색 판정이 그 **앞**에 있어야
    L2 이상에서 황색 중앙선이 통과하지 않는다.
    """
    kr = KrRules(CFG)
    lg = Lg('yellow')
    for lvl in (0, 1, 2, 3, 4):
        kr.bo_level = lvl
        assert kr._is_center_mark(lg, (1, 0, -1), 'left', 0.0) is True


def test_missing_lane_graph_is_not_center_line():
    """lg 가 없으면(목 환경) 판정하지 않는다 — 기존 경로 회귀."""
    kr = KrRules(CFG)
    assert kr._is_center_mark(None, (1, 0, -1), 'left', 0.0) is False
    assert kr._is_center_mark(Lg('yellow'), None, 'left', 0.0) is False
