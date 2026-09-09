"""[2] ④′ 정지 프로파일·황색 판정의 jerk 램프 보정
(`speed.stop_jerk_compensate_enable`, 2026-09-09).

√(2·a·d) 는 "지금부터 a 를 **즉시** 낼 수 있다" 는 식이다. 실제 종방향에는
jerk 제한이 있어 a 도달에 t = a / (control.jerk_dec_mult × speed.jerk_max)
= a / 6.0 [s] 가 걸리고, 그동안 평균 감속이 절반이라 **v·t/2 만큼 더 간다**.

실측 `20260909_093415` — 정지선 침범 9건, 최대 +3.28 m. 침범 지점의 접근 속도가
전부 43~49 km/h (12~13.6 m/s) 고, a 3.0 / jerk 6.0 → t 0.5 s 라 보정량이
3.0~3.4 m 로 침범량과 같은 자릿수다. 예: 실전주행_교통류_07 rs 870.8 은 신호가
28 m 밖에서 이미 적색인데 ④′ 가 slf −7.6 m 에서야 구속했다.

지키는 것:
  · 보정은 **거리에서 뺀다** (d_eff −= v·t/2). 상수를 깎는 것이 아니다 —
    지연분은 속도에 비례하므로 저속 접근은 영향이 거의 없어야 한다.
  · jerk 값은 control 이 실제로 쓰는 두 상수를 그대로 읽는다 (단일 출처).
  · **판정(a_yellow)과 실행(stop_profile_a) 두 축 모두**에 같은 보정이 든다.
    한쪽만 고치면 침범이 안 없어진다 (아침에 a_yellow 만 내려서 실패했다).
  · v → 0 이면 보정 → 0. 그래서 계획 정지점은 안 움직인다 (과감속 금지).
  · false = 이전 동작.
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
from test_stop_profile import A_STOP, S0, make_ap                   # noqa: E402
from test_stopline_stop import FakePlannerTL                        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
JERK_RATE = CFG['control']['jerk_dec_mult'] * CFG['speed']['jerk_max']


def on_cfg():
    c = copy.deepcopy(CFG)
    c['speed']['stop_jerk_compensate_enable'] = True
    return c


def off_cfg():
    """이전 동작 사본 — 기본값을 읽지 않는다 (2026-09-07 드리프트 원칙)."""
    c = copy.deepcopy(CFG)
    c['speed']['stop_jerk_compensate_enable'] = False
    return c


def rig(cfg, d_line):
    p = FakePlannerTL(d_tl=d_line)
    ap = make_ap(p, cfg)
    ap.kr_rules = KrRules(cfg)
    return ap.kr_rules, p, ap


# ── 램프 거리 자체 ────────────────────────────────────────────────────────
def test_jerk_rate_reads_the_controller_constants():
    kr, _, _ = rig(on_cfg(), S0 + 30.0)
    assert kr.jerk_rate == pytest.approx(JERK_RATE)


@pytest.mark.parametrize('a,v', [(3.0, 12.0), (4.0, 8.06), (3.0, 2.0)])
def test_ramp_distance_matches_the_formula(a, v):
    kr, _, _ = rig(on_cfg(), S0 + 30.0)
    assert kr._jerk_ramp_m(a, v) == pytest.approx(v * (a / JERK_RATE) / 2.0)


def test_ramp_is_proportional_to_speed():
    """상수를 깎는 것과의 결정적 차이 — 저속 접근은 거의 영향이 없다."""
    kr, _, _ = rig(on_cfg(), S0 + 30.0)
    assert kr._jerk_ramp_m(A_STOP, 0.0) == 0.0
    assert kr._jerk_ramp_m(A_STOP, 12.0) == pytest.approx(
        6.0 * kr._jerk_ramp_m(A_STOP, 2.0))


def test_ramp_is_zero_when_switch_off():
    kr, _, _ = rig(off_cfg(), S0 + 30.0)
    assert kr._jerk_ramp_m(A_STOP, 12.0) == 0.0


def test_ramp_is_zero_for_nonpositive_accel():
    kr, _, _ = rig(on_cfg(), S0 + 30.0)
    assert kr._jerk_ramp_m(0.0, 12.0) == 0.0
    assert kr._jerk_ramp_m(-3.0, 12.0) == 0.0


# ── ④′ 실행 축 ───────────────────────────────────────────────────────────
def test_profile_subtracts_the_ramp_from_the_distance():
    d = S0 + 30.0
    kr, p, ap = rig(on_cfg(), d)
    v = 12.0
    got = kr._stopline_profile(p, ap, v)
    want = math.sqrt(2.0 * A_STOP * (30.0 - kr._jerk_ramp_m(A_STOP, v)))
    assert got == pytest.approx(want)


def test_profile_binds_earlier_than_before():
    """같은 지점에서 상한이 낮다 = 더 일찍 구속한다."""
    d = S0 + 30.0
    on, p1, ap1 = rig(on_cfg(), d)
    off, p2, ap2 = rig(off_cfg(), d)
    assert on._stopline_profile(p1, ap1, 12.0) < off._stopline_profile(p2, ap2, 12.0)


def test_stop_point_does_not_move():
    """v → 0 이면 보정 → 0 — 계획 정지점은 그대로다 (과감속 금지)."""
    d = S0 + 3.0
    on, p1, ap1 = rig(on_cfg(), d)
    off, p2, ap2 = rig(off_cfg(), d)
    assert on._stopline_profile(p1, ap1, 0.0) == pytest.approx(
        off._stopline_profile(p2, ap2, 0.0))


def test_profile_never_goes_negative_under_the_root():
    """보정이 남은 거리보다 커도 0 으로 클램프된다 (수학 오류 금지)."""
    kr, p, ap = rig(on_cfg(), S0 + 0.2)
    assert kr._stopline_profile(p, ap, 30.0) == pytest.approx(0.0)


def test_default_call_has_no_compensation():
    """ego_speed 를 안 주면 0 — 옛 호출부(테스트·진단)가 그대로 산다."""
    d = S0 + 30.0
    on, p1, ap1 = rig(on_cfg(), d)
    off, p2, ap2 = rig(off_cfg(), d)
    assert on._stopline_profile(p1, ap1) == pytest.approx(off._stopline_profile(p2, ap2))


# ── 두 축이 같은 보정을 쓴다 ─────────────────────────────────────────────
def test_both_axes_use_the_same_correction():
    """판정(a_yellow)과 실행(stop_profile_a) 이 같은 식을 탄다.

    한쪽만 보정하면 안 된다 — 아침에 a_yellow 만 내렸을 때 정지선 침범이
    9건 그대로 났다 (20260909_093415).
    """
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text(encoding='utf-8')
    # `_jerk_ramp_m` 호출이 정확히 두 곳(황색 판정·④′ 프로파일)이어야 한다
    calls = src.count('self._jerk_ramp_m(')
    assert calls == 2, f'_jerk_ramp_m 호출 {calls}곳 — 판정·실행 두 축이어야 한다'
    assert 'self._jerk_ramp_m(self.a_yellow' in src


def test_a_yellow_is_back_to_the_executable_maximum():
    """보정이 지연분을 맡으므로 a_yellow 는 다시 a_dec_max 다.

    **상수 두 개가 같은 일을 하면 안 된다.**
    """
    assert CFG['speed']['a_yellow'] == pytest.approx(
        abs(float(CFG['control']['a_dec_max'])))
    assert CFG['speed']['a_yellow'] != CFG['speed']['stop_profile_a']
