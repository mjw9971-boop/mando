"""
ctrl24 — K11 적색 점멸 일시정지 (채점 항목 9, 2026-09-12).

여기서 지키는 불변:
  · 점멸 판정은 전방 정지선 tl 의 controller_ids 중 하나라도 9910 **원시** state 가
    6 인지로만 한다. 지도 id 를 박지 않고, LIGHT_STATE_MAP 도 색 해석도 안 건드린다.
  · 상수는 K1 것을 읽는다 (stop_profile_a · _s0) — 새 거리 임계가 없다.
    프로파일은 뒷축 d = s0 에서 0 이고, 그 지점의 앞범퍼 여유가 채점 stop_ok_m 과 같다.
  · 정지 후 flash_stop_hold_s 유지, 끝나면 그 정지선을 다시 세우지 않는다 (1회 제한).
  · 기본 off — off 면 state 6 판정조차 하지 않는다.
  · 정지선 매핑이 없으면 None. 점멸이 아닌 state(1·3·5)면 None 이고 기존 경로 그대로다.

params 값은 팀원 소관이라 리터럴을 박지 않는다 (docs/BACKLOG.md B-25).
킬스위치 기본값과 관계만 보고, 산술이 필요한 곳은 사본에서 명시적으로 켠다.
"""
import copy
import math
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from test_ctrl24_long import TL, apply, cfg_with, rig, set_signal   # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
HZ = CFG['comm']['send_hz']
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
S0 = CFG['speed']['stop_gap_stopline_m'] + FRONT
TL_ID = 7
BLINK = 6


def on_cfg(**kw):
    """사본에서 K11 을 켠다 — 기본값이 또 바뀌어도 안 깨지게."""
    return cfg_with(flash_stop_enable=True, **kw)


def flash_rig(d_tl, cfg=None, state=TrafficLightState.Green, raw=BLINK):
    """정지선 d_tl m (뒷축) 앞. raw 가 9910 원시 state 다 (6 = 점멸)."""
    kr, p, ap = rig(cfg or on_cfg(), d_tl=d_tl, state=state)
    if raw is not None:
        kr.observe_lights([(TL_ID, raw)])
    return kr, p, ap


def hold_ticks():
    return int(round(float(C['flash_stop_hold_s']) * HZ))


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_params_defaults():
    assert C['flash_stop_enable'] is False
    assert C['flash_stop_hold_s'] > 0.0


def test_off_never_looks_at_state_6():
    kr, p, ap = flash_rig(20.0, cfg=cfg_with(flash_stop_enable=False))
    apply(kr, ap, v=6.0)
    assert kr.last_kr['flash_stop'] is None
    assert kr._flash_hold == 0 and not kr._flash_done


# ── 점멸 판정 ────────────────────────────────────────────────────────────
def test_blink_at_5m_yields_a_candidate():
    kr, p, ap = flash_rig(0.0)                      # s0 는 조립이 정한다 — 여기서 읽는다
    d = kr._s0(ap) + 5.0
    p.distances_to_next_traffic_lights[:] = d
    apply(kr, ap, v=6.0)
    v = kr.last_kr['flash_stop']
    assert v is not None
    assert v == pytest.approx(math.sqrt(2.0 * C['stop_profile_a'] * 5.0), abs=0.01)


def test_profile_is_zero_at_s0():
    """프로파일은 K1 과 같은 s0 에서 0 이 된다 — 새 거리 임계가 없다."""
    kr, p, ap = flash_rig(0.0)
    p.distances_to_next_traffic_lights[:] = kr._s0(ap)
    apply(kr, ap, v=6.0)
    assert kr.last_kr['flash_stop'] == pytest.approx(0.0, abs=1e-6)


def test_production_s0_parks_the_bumper_at_the_scoring_window():
    """실주행 s0 = speed.stop_gap_stopline_m + front 라 앞범퍼 여유가 stop_ok_m 이다."""
    assert CFG['speed']['stop_gap_stopline_m'] == pytest.approx(CFG['scoring']['stop_ok_m'])


def test_non_blink_states_are_ignored():
    """1 적색 · 3 녹색 · 5 녹색+좌 는 K11 대상이 아니다 — 기존 경로 그대로."""
    for raw in (1, 3, 5):
        kr, p, ap = flash_rig(10.0, raw=raw)
        apply(kr, ap, v=6.0)
        assert kr.last_kr['flash_stop'] is None, raw


def test_no_raw_report_yields_none():
    kr, p, ap = flash_rig(10.0, raw=None)          # observe_lights 를 안 불렀다
    apply(kr, ap, v=6.0)
    assert kr.last_kr['flash_stop'] is None


def test_no_stopline_mapping_yields_none():
    kr, p, ap = rig(on_cfg(), d_tl=float('inf'))   # next_traffic_lights 전부 None
    kr.observe_lights([(TL_ID, BLINK)])
    apply(kr, ap, v=6.0)
    assert kr.last_kr['flash_stop'] is None


def test_matches_any_controller_id_not_just_the_first():
    """tl.id 가 아니라 controller_ids 전체를 본다 — 지도 id 하드코딩이 없다."""
    kr, p, ap = rig(on_cfg(), d_tl=10.0, state=TrafficLightState.Green)
    for tl in p.next_traffic_lights:
        tl.controller_ids = [116, 117]
        tl.id = 116
    kr.observe_lights([(117, BLINK)])              # 두 번째 id 만 점멸을 보고한다
    apply(kr, ap, v=6.0)
    assert kr.last_kr['flash_stop'] is not None


# ── 정지 → 유지 → 출발 ───────────────────────────────────────────────────
def test_stop_arms_the_hold_and_releases_after_flash_stop_hold_s():
    kr, p, ap = flash_rig(0.0)
    p.distances_to_next_traffic_lights[:] = kr._s0(ap)
    apply(kr, ap, v=0.0)                            # 정지선 앞에서 섰다 → 무장
    assert kr._flash_hold == hold_ticks()
    assert kr.last_kr['flash_stop'] == pytest.approx(0.0)
    for _ in range(hold_ticks()):                   # 유지 동안 계속 0
        apply(kr, ap, v=0.0)
        assert kr.last_kr['flash_stop'] == pytest.approx(0.0) or kr._flash_hold == 0
    assert kr._flash_hold == 0 and kr._flash_done
    apply(kr, ap, v=0.0)
    assert kr.last_kr['flash_stop'] is None         # 유지가 끝나면 후보가 사라진다


def test_same_stopline_is_not_served_twice():
    """재접근해도 다시 서지 않는다."""
    kr, p, ap = flash_rig(0.0)
    p.distances_to_next_traffic_lights[:] = kr._s0(ap)
    apply(kr, ap, v=0.0)
    for _ in range(hold_ticks() + 1):
        apply(kr, ap, v=0.0)
    assert kr._flash_done
    set_signal(p, TrafficLightState.Green, 10.0)    # 같은 정지선에 다시 접근
    for tl in p.next_traffic_lights:
        tl.controller_ids = [TL_ID]
    kr.observe_lights([(TL_ID, BLINK)])
    apply(kr, ap, v=6.0)
    assert kr.last_kr['flash_stop'] is None


def test_far_approach_does_not_arm_the_hold():
    kr, p, ap = flash_rig(0.0)
    p.distances_to_next_traffic_lights[:] = kr._s0(ap) + 40.0
    apply(kr, ap, v=0.0)                            # 멀리서 섰다 — 무장하지 않는다
    assert kr._flash_hold == 0 and not kr._flash_done
    assert kr.last_kr['flash_stop'] > 0.0


# ── 리셋 ─────────────────────────────────────────────────────────────────
def test_reset_clears_the_three_dedicated_variables():
    kr, p, ap = flash_rig(0.0)
    p.distances_to_next_traffic_lights[:] = kr._s0(ap)
    apply(kr, ap, v=0.0)
    assert kr._raw_light and kr._flash_hold > 0
    kr._flash_done.add(('x', 0.0))
    kr.on_reset()
    assert not kr._raw_light and kr._flash_hold == 0 and not kr._flash_done


def test_winner_name_is_mapped():
    """후보가 이기는 틱에 run_agent 가 KeyError 를 내지 않는다."""
    import run_agent as RA
    assert RA._KR24_NAME['flash_stop'] == 'light'


# ── 선을 넘은 뒤 ─────────────────────────────────────────────────────────
def test_past_the_line_yields_none_and_closes_the_stopline():
    """앞범퍼가 선을 넘으면 후보가 사라진다 — 교차로 안에서 서지 않는다."""
    kr, p, ap = flash_rig(0.0)
    p.distances_to_next_traffic_lights[:] = FRONT - 0.1      # 범퍼가 선을 막 넘었다
    apply(kr, ap, v=6.0)
    assert kr.last_kr['flash_stop'] is None
    assert kr._flash_done                                     # 기회가 끝났다 — 닫는다


def test_stale_raw_report_does_not_restop_after_passing():
    """9910 이 보고를 멈춰도 캐시가 남아 다시 서지 않는다."""
    kr, p, ap = flash_rig(0.0)
    p.distances_to_next_traffic_lights[:] = FRONT - 0.1
    apply(kr, ap, v=6.0)
    kr.observe_lights([])                                     # 보고 중단 (캐시 유지)
    p.distances_to_next_traffic_lights[:] = -0.5              # 더 지나갔다
    apply(kr, ap, v=6.0)
    assert kr.last_kr['flash_stop'] is None
