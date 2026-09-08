"""정상상태 속도 상한의 감속 축 (실주행 2차 [1], 2026-09-08).

증상: 목표속도가 실속도보다 **위**인데 급제동하는 구간이 한 주행에 11곳,
min accel −3.9. rs 363.8~366.0 은 목표 5.56 m/s 인데 완전히 정지했다.

원인은 곡률·LC 상한이 accel 에 직접 개입해서가 아니다 (그런 경로는 없다 —
둘 다 min() 후보로만 들어간다). `VtdLongitudinalController._raw_accel` 의
`err/dt` 가 **1틱 전방 목표**(PDM IDM)에만 맞는 식인데, 정상상태 상한에도
그대로 쓰여 1/dt = 20배 증폭된 것이다:

    목표가 실속도보다 0.20 m/s 낮기만 해도  −0.20/0.05 = −4.0 (a_dec_max 포화)

포화 뒤 jerk 램프(+0.1/틱)로 푸는 데 1.75 s 가 걸려 그 사이 상한 **아래로**
한참 내려가고, 그 언더슛이 다음 창의 오차를 키우는 되먹임이 된다.

계약: 상한형 목표(곡률·LC·붉은구간 접근·순수 제한속도)를 따를 때는 P 로
부드럽게 붙이고 a_cap_dec_max 로만 클램프한다. 정지 요구(hazard·정지 프로파일
·보행자)는 **이전 축 그대로** — 그쪽은 err/dt 가 맞다.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from conftest import PARAMS_YAML                                # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402
from vtd_adapter.control import VtdLongitudinalController       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
DT = 1.0 / float(CFG['comm']['send_hz'])


def mk(**control):
    """control.* 만 갈아끼운 사본으로 컨트롤러를 만든다 (기본값 드리프트 무관)."""
    c = dict(CFG)
    c['control'] = {**CFG['control'], **control}
    return VtdLongitudinalController(c)


def on():
    return mk(cap_soft_dec_enable=True)


def off():
    return mk(cap_soft_dec_enable=False)


# ── params 계약 ──────────────────────────────────────────────────────────
def test_params_present():
    c = CFG['control']
    assert c['cap_soft_dec_enable'] is True
    assert c['cap_limit_target_enable'] is True
    assert float(c['a_cap_dec_max']) == -1.5
    assert float(c['kp_cap_dec']) == 0.8
    # 상한 축은 정지 축보다 **약해야** 한다 — 그게 이 기능의 전부다
    assert float(c['a_cap_dec_max']) > float(c['a_dec_max'])


# ── 증폭 자체 ────────────────────────────────────────────────────────────
def test_tiny_overshoot_saturated_before():
    """0.20 m/s 초과가 a_dec_max 포화를 냈다 — 이 배수가 회귀의 뿌리다."""
    a_dec = float(CFG['control']['a_dec_max'])
    assert -0.20 / DT <= a_dec + 1e-9          # −4.0 이하 = 포화
    assert off()._raw_accel(6.94, 7.14) == pytest.approx(a_dec)


def test_cap_target_is_gentle():
    """같은 오차를 상한으로 주면 P 로 부드럽다."""
    a = on()._raw_accel(6.94, 7.14, cap_target=True)
    assert a == pytest.approx(0.8 * (6.94 - 7.14))
    assert a > -0.2


def test_cap_target_clamped_at_a_cap_dec_max():
    """큰 오차라도 상한 축의 클램프를 안 넘는다."""
    a = on()._raw_accel(5.0, 20.0, cap_target=True)
    assert a == pytest.approx(float(CFG['control']['a_cap_dec_max']))


@pytest.mark.parametrize('err', [-0.05, -0.2, -1.0, -3.0, -10.0])
def test_cap_never_harder_than_stop_axis(err):
    """상한 축은 어떤 오차에서도 정지 축보다 세게 밟지 않는다."""
    v = 12.0
    c_on, c_off = on(), off()
    assert (c_on._raw_accel(v + err, v, cap_target=True)
            >= c_off._raw_accel(v + err, v) - 1e-9)


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_switch_off_is_previous_behaviour():
    """off 면 cap_target 을 줘도 이전 축이다 (지문 동일 요구)."""
    c = off()
    for err in (-0.05, -0.5, -2.0):
        assert (c._raw_accel(10.0 + err, 10.0, cap_target=True)
                == c._raw_accel(10.0 + err, 10.0))


def test_default_arg_is_stop_axis():
    """cap_target 을 안 주면 이전 동작이다 — 호출부를 안 고친 경로는 안 변한다."""
    c = on()
    assert c._raw_accel(9.8, 10.0) == pytest.approx(max(-0.2 / DT, c.a_dec_max))


# ── 안 건드려야 하는 축 ──────────────────────────────────────────────────
def test_acceleration_side_unchanged():
    """가속측은 상한 여부와 무관하게 같다."""
    c = on()
    assert (c._raw_accel(12.0, 8.0, cap_target=True)
            == c._raw_accel(12.0, 8.0))


def test_hazard_never_uses_cap_axis():
    """hazard 는 '상한' 이 아니라 정지 요구다 — cap_target 을 줘도 정지 축."""
    c = on()
    a_haz, brake = c.get_throttle_and_brake(True, 8.0, 10.0, cap_target=True)
    c2 = on()
    a_ref, _ = c2.get_throttle_and_brake(True, 8.0, 10.0, cap_target=False)
    assert a_haz == pytest.approx(a_ref)
    assert brake is True


def test_forecast_is_unchanged():
    """forecast(무상태)는 PDM 목표를 쓰므로 상한 축을 타지 않는다."""
    c = on()
    assert c.get_throttle_extrapolation(9.8, 10.0) == pytest.approx(
        max(-0.2 / DT, c.a_dec_max))


def test_stop_demand_still_reaches_a_dec_max():
    """정지 프로파일이 이기는 상황의 제동력은 그대로여야 한다 (정지선 회귀 방지)."""
    c = on()
    assert c._raw_accel(0.0, 8.0) == pytest.approx(float(CFG['control']['a_dec_max']))
