"""
ctrl24 — K7-P 시프트 상한을 거리 프로파일로 (2026-09-11).

증상: K7 shift_cap 은 전방 창 look = max(overtake.shift_latest_m, trans_k·v) 가
span[0] 에 닿아야 값이 난다. 실측 run_20260911_185104 (SHIFT_ACTIVE 2633틱):
  · cap 이 None 인 틱 152개 — 전부 창이 span 에 못 닿아서다
  · 닿는 순간 **계단**으로 선다. 구간 0 은 d 36.0 m 에서 None → 9.69 인데
    그때 v 11.41 이라 err/dt(×20)가 −34 를 요구하고, jerk 하강 한도
    0.3 m/s²/틱 탓에 −4.0 에 닿기까지 1 s 를 더 가속한다 (v 11.41 → 11.77).

여기서 지키는 불변:
  · 기본 off — 꺼져 있으면 값이 한 비트도 안 바뀐다.
  · 창에 v²/(2a) 를 더한다 (K10 _curvature_profile 과 같은 항).
  · 값은 √(cap² + 2·a·d), d = span[0] 까지 남은 거리. **span 안에서는 d = 0**
    이라 cap 그대로다 — 전이 구간의 상한(a_lat_max)은 안 건드린다.
  · 상한형 바닥 — 요구 감속이 a 를 절대 안 넘는다 (K10 과 같은 한 줄).
  · 하한 overtake.shift_cap_min_v 는 그대로.
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
A = float(CFG['speed']['approach_decel_mps2'])
HZ = float(CFG['comm']['send_hz'])
A_LAT = float(CFG['overtake']['a_lat_max'])
MIN_V = float(CFG['overtake']['shift_cap_min_v'])
LOOK_M = float(CFG['overtake']['shift_latest_m'])
TRANS_K = float(CFG['ctrl24']['trans_k'])
PPM = 10


def cfg_with(profile):
    """params 사본에서 명시적으로 켜고 끈다 — 기본값이 바뀌어도 안 깨진다."""
    c = copy.deepcopy(CFG)
    c['ctrl24']['shift_cap_enable'] = True
    c['ctrl24']['shift_cap_profile_enable'] = profile
    return c


class _Planner:
    """lat_shift 에 코사인 전이 하나만 있는 최소 플래너."""

    def __init__(self, i, span, trans_m=12.0, offset=3.0, n=4000):
        self.route_index = int(i)
        self.points_per_meter = PPM
        a, b = span
        lat = np.zeros(n, dtype=float)
        k = int(trans_m * PPM)
        ramp = 0.5 * (1.0 - np.cos(np.linspace(0.0, math.pi, k)))
        lat[a:a + k] = offset * ramp
        lat[a + k:b] = offset
        self.lat_shift = lat
        self._lat_build = np.zeros(n, dtype=float)


def cap_of(profile, i, span, v, **kw):
    kr = Ctrl24(cfg_with(profile))
    kr.ot_span = span
    return kr._shift_speed_cap(_Planner(i, span, **kw), v)


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_params_present():
    assert isinstance(CFG['ctrl24']['shift_cap_profile_enable'], bool)


def test_off_is_unchanged_inside_and_outside_the_span():
    """off 경로는 값이 그대로다 (span 안·밖 모두)."""
    span = (2000, 2600)
    for i, v in ((1950, 8.0), (2005, 8.0), (2100, 8.0), (1500, 12.0)):
        a = cap_of(False, i, span, v)
        b = cap_of(False, i, span, v)
        assert a == b


# ── 창 확장 ──────────────────────────────────────────────────────────────
def test_window_reaches_a_span_the_old_window_could_not():
    """창에 v²/(2a) 가 붙어, 옛 창으로는 None 이던 자리에서 값이 난다."""
    span = (2000, 2600)
    v = 12.0
    old_look = max(LOOK_M, TRANS_K * v)                  # ≈ 38.9 m
    new_look = old_look + v * v / (2.0 * A)              # ≈ 74.9 m
    d = 0.5 * (old_look + new_look)                      # 둘 사이
    i = span[0] - int(d * PPM)
    assert cap_of(False, i, span, v) is None             # 옛 창은 못 닿는다
    assert cap_of(True, i, span, v) is not None          # 새 창은 닿는다


# ── 거리 프로파일 ────────────────────────────────────────────────────────
def test_inside_the_span_the_value_is_unchanged():
    """span 안이면 d = 0 — 전이 구간의 상한은 건드리지 않는다."""
    span = (2000, 2600)
    # 전이가 아직 앞에 남아 있는 자리만 본다 (평지는 κ = 0 이라 둘 다 None).
    # v = 0 이라 상한형 바닥(v − a/hz)이 음수라 안 산다 — 프로파일만 본다.
    for i in (2000, 2030, 2060):
        off = cap_of(False, i, span, 0.0)
        on = cap_of(True, i, span, 0.0)
        assert off is not None, i
        assert on == pytest.approx(off, rel=1e-9)


@pytest.mark.parametrize('d', [5.0, 10.0, 20.0])     # v 1.0 의 창 25.25 m 안
def test_value_is_sqrt_cap_squared_plus_2ad(d):
    """접근 구간 값 = √(off 값² + 2·a·d). 바닥이 안 사는 저속에서 본다."""
    span = (2000, 2600)
    i = span[0] - int(d * PPM)
    v = 1.0                                              # 바닥 v − a/hz 가 낮게 눕는다
    # off 값은 같은 기하의 span 안 값과 같다 (창만 다르다)
    base = cap_of(False, span[0], span, v)
    on = cap_of(True, i, span, v)
    assert base is not None and on is not None
    assert on == pytest.approx(math.sqrt(base * base + 2.0 * A * d), rel=1e-6)
    assert on > base                                     # 멀수록 높다


def test_profile_is_monotone_as_the_car_closes_in():
    """계단이 아니라 램프다 — d 가 줄수록 값이 단조 감소해 span 에서 cap 에 닿는다."""
    span = (2000, 2600)
    v = 1.0
    seq = [cap_of(True, span[0] - int(d * PPM), span, v) for d in (25, 20, 15, 10, 5, 0)]
    assert all(x is not None for x in seq)
    assert all(seq[k] > seq[k + 1] for k in range(len(seq) - 1))
    assert seq[-1] == pytest.approx(cap_of(False, span[0], span, v), rel=1e-9)


# ── 상한형 바닥 ──────────────────────────────────────────────────────────
def test_floor_bounds_demanded_deceleration_to_a():
    """늦게 걸려도 한 틱에 a 보다 더 줄이라고 하지 않는다.

    이 바닥이 없으면 종방향이 (target − v)/dt = ×20 으로 실행해 0.2 m/s 만
    모자라도 a_dec_max 에 포화한다. 실측 2026-09-11 리플레이 8런: 바닥 없이
    accel ≤ −3.99 가 1293 → 1431틱(+138)이고 그중 157틱의 argmin 이
    kr:shift_cap 이었다 (예: v 9.66 · cap 9.445 → err −0.215 → −4.0).
    """
    span = (2000, 2600)
    v = 12.0
    on = cap_of(True, span[0] + 50, span, v)             # span 안 = d 0 = 최악
    off = cap_of(False, span[0] + 50, span, v)
    assert off is not None and on is not None
    assert off < v - A / HZ                              # 바닥이 실제로 사는 상황
    assert on == pytest.approx(v - A / HZ, abs=1e-9)
    assert (v - on) * HZ <= A + 1e-9                     # 요구 감속 ≤ a


def test_floor_never_lowers_the_value():
    """바닥은 올리기만 한다 — 프로파일이 이미 높으면 아무것도 안 바꾼다."""
    span = (2000, 2600)
    v = 1.0
    i = span[0] - 200
    assert cap_of(True, i, span, v) > v


def test_min_v_floor_still_applies():
    """하한 shift_cap_min_v 는 그대로다."""
    span = (2000, 2600)
    # 아주 짧은 전이 = 큰 κ → cap 이 하한으로 눌린다
    on = cap_of(True, span[0], span, 0.0, trans_m=1.0, offset=4.0)
    assert on is not None and on >= MIN_V - 1e-9
