"""
ctrl24 — K10 경로 곡률 감속 (2026-09-11).

증상: 급커브 연결로에 빠르게 들어가면 조향이 한계까지 꺾여도 호를 못 따라가고
밖으로 밀린다. 기존 속도 후보 9개 중 **경로 곡률을 보는 것이 없었다**
(K7 shift_cap 은 회피 시프트의 전이 곡률이지 지도 경로 곡률이 아니다).

여기서 지키는 불변:
  · 기본 off — 꺼져 있으면 후보가 아예 None 이다.
  · 직선(κ < curvature_min_kappa)이면 None — 영향 0.
  · 값은 √(a_lat/κ) 를 거리 프로파일 √(v_pt² + 2·a·d) 로 올린 것이다.
  · **상한형 바닥** — 이 후보가 요구하는 감속은 절대 a 를 넘지 않는다.
    (넘으면 종방향이 err/dt(×20)로 a_dec_max 에 포화하고, jerk 비대칭 탓에
     목표 아래로 넘어가 선다. 2026-09-11 폐루프에서 실제로 0.05 m/s 까지 갔다.)
  · 탐색 창은 curvature_lookahead_m + v²/(2a) — 제동거리를 덮는다.
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

from ctrl24 import Ctrl24                                          # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
HZ = float(CFG['comm']['send_hz'])
A = float(CFG['speed']['approach_decel_mps2'])
PPM = 10                      # 경로점 간격 0.1 m (VtdRoutePlanner 기본)


def cfg_with(**kw):
    """params 사본에서 **명시적으로** 켠다 — 기본값이 바뀌어도 안 깨진다."""
    c = copy.deepcopy(CFG)
    c['ctrl24'].update(curvature_cap_enable=True, **kw)
    return c


class _WP:
    __slots__ = ('key', 's')

    def __init__(self, key, s):
        self.key, self.s = key, float(s)


class _LG:
    """κ 를 s 배열과 함께 들고 있는 최소 lane graph."""

    def __init__(self, lanes):
        self.lanes = lanes


class _Planner:
    """route_waypoints / route_s / route_index / lg 만 있는 최소 플래너."""

    def __init__(self, kappa_of_s, length_m, key=(1, 0, -1)):
        n = int(length_m * PPM) + 1
        ss = np.arange(n, dtype=float) / PPM
        self.route_s = ss
        self.route_waypoints = [_WP(key, s) for s in ss]
        self.route_index = 0
        self.points_per_meter = PPM
        self.lg = _LG({key: {'s': ss, 'curv': np.array([kappa_of_s(s) for s in ss])}})


def straight(_s):
    return 0.0


def curve_at(s0, s1, kappa):
    """[s0, s1] 구간만 κ, 나머지는 직선."""
    return lambda s: kappa if s0 <= s <= s1 else 0.0


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_params_present():
    assert isinstance(C['curvature_cap_enable'], bool)
    assert float(C['curvature_lookahead_m']) > 0.0
    assert float(C['curvature_a_lat_max']) > 0.0
    assert float(C['curvature_min_kappa']) > 0.0


def test_off_yields_no_candidate():
    c = copy.deepcopy(CFG)
    c['ctrl24']['curvature_cap_enable'] = False
    kr = Ctrl24(c)
    p = _Planner(curve_at(10.0, 25.0, 1 / 5.0), 120.0)
    assert kr._curvature_profile(p, 12.0) is None


# ── 직선 ─────────────────────────────────────────────────────────────────
def test_straight_yields_none():
    kr = Ctrl24(cfg_with())
    assert kr._curvature_profile(_Planner(straight, 200.0), 12.5) is None


def test_gentle_below_min_kappa_yields_none():
    """κ < curvature_min_kappa (기본 0.005 = R 200 m) 는 직선으로 본다."""
    kr = Ctrl24(cfg_with())
    p = _Planner(curve_at(10.0, 40.0, 0.004), 200.0)
    assert kr._curvature_profile(p, 12.5) is None


# ── 값 ───────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('R', [5.0, 6.5, 10.0, 15.0, 20.0, 30.0])
def test_in_curve_value_is_sqrt_a_lat_over_kappa(R):
    """커브 **안**(d = 0)에서는 v_pt = √(a_lat·R) 그대로다."""
    a_lat = float(C['curvature_a_lat_max'])
    kr = Ctrl24(cfg_with())
    p = _Planner(curve_at(0.0, 30.0, 1.0 / R), 120.0)
    v = math.sqrt(a_lat * R)
    got = kr._curvature_profile(p, v)          # 이미 v_cap 이라 바닥이 안 산다
    assert got == pytest.approx(v, abs=1e-3)


def test_distance_profile_rises_with_distance():
    """멀수록 허용속도가 높다 — √(v_pt² + 2·a·d)."""
    a_lat = float(C['curvature_a_lat_max'])
    R, d = 8.0, 10.0          # 저속의 창(curvature_lookahead_m 15) 안이어야 한다
    kr = Ctrl24(cfg_with())
    p = _Planner(curve_at(d, d + 10.0, 1.0 / R), 200.0)
    v_pt = math.sqrt(a_lat * R)
    want = math.sqrt(v_pt * v_pt + 2.0 * A * d)
    # 바닥이 안 사는 저속에서 본다 (프로파일 자체를 확인)
    got = kr._curvature_profile(p, 1.0)
    assert got == pytest.approx(want, rel=1e-3)
    assert want > v_pt        # 멀리 있으면 커브 안보다 높다


def test_far_curve_outside_braking_distance_cannot_bind():
    """d > v²/(2a) 인 커브는 v_pt 가 0 이어도 구속할 수 없다 — 창 밖이어도 된다."""
    kr = Ctrl24(cfg_with())
    v = 12.5
    far = v * v / (2.0 * A) + float(C['curvature_lookahead_m']) + 50.0
    p = _Planner(curve_at(far, far + 10.0, 1.0 / 5.0), far + 60.0)
    got = kr._curvature_profile(p, v)
    assert got is None or got >= v


# ── 상한형 바닥 ──────────────────────────────────────────────────────────
def test_floor_bounds_demanded_deceleration_to_a():
    """늦게 걸려도 한 틱에 a 보다 더 줄이라고 하지 않는다.

    이 바닥이 없으면 종방향이 (target − v)/dt = ×20 으로 실행해 0.2 m/s 만
    모자라도 a_dec_max 에 포화하고, jerk 이 내려갈 때 3배라 목표 아래로
    넘어가 선다 (2026-09-11 폐루프 실측: v 5.35 → 0.05 m/s).
    """
    kr = Ctrl24(cfg_with())
    v = 12.0
    p = _Planner(curve_at(0.0, 20.0, 1.0 / 5.0), 100.0)     # 이미 커브 안 = 최악
    got = kr._curvature_profile(p, v)
    assert got is not None
    assert got == pytest.approx(v - A / HZ, abs=1e-6)
    assert (v - got) * HZ <= A + 1e-9                        # 요구 감속 ≤ a
    assert kr.last_curvature['floored'] is True


def test_floor_does_not_lift_the_value_on_normal_approach():
    """정상 접근(창이 제동거리를 덮는다)에서는 바닥이 아무것도 안 바꾼다."""
    kr = Ctrl24(cfg_with())
    v = 12.5
    d = v * v / (2.0 * A) - 5.0                              # 제동거리 안쪽
    p = _Planner(curve_at(d, d + 15.0, 1.0 / 8.0), d + 60.0)
    got = kr._curvature_profile(p, v)
    assert got is not None
    assert kr.last_curvature['floored'] is False
    assert got == pytest.approx(kr.last_curvature['raw'], abs=1e-2)


# ── 후보 배선 ────────────────────────────────────────────────────────────
def test_candidate_name_is_mapped_for_the_logger():
    """run_agent._KR24_NAME 에 없으면 winner 계산이 KeyError 로 죽는다."""
    sys.path.insert(0, str(ROOT))
    import run_agent                                          # noqa: PLC0415
    assert 'curvature' in run_agent._KR24_NAME


def test_curvature_is_not_a_legit_stop_cause():
    """K9 탈출 바닥이 들어 올릴 수 있어야 한다 — 곡률은 정당한 정지 원인이 아니다."""
    assert 'curvature' not in Ctrl24.LEGIT_KR
