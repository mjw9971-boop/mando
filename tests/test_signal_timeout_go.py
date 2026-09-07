"""B-3(b) 미보고 적신호 시한 출발 (signal_timeout_go_enable, 2026-09-06 승인, 기본 off).

Red 로 선 뒤 controller 보고가 끊기면 플래너 state 가 Red 로 남아 영구 정지다.
켜면 앞차 없음 ∧ 보행자 래치 없음 ∧ 정지 중 ∧ 미보고 signal_unknown_timeout_s 이상
→ 신호 정지 후보 해제(_stop_target None, signal_release True). 71 s 주기·적색 60 s
신호에서는 85 % 확률로 적신호 통과라 기본 off 다.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from test_avoid import Ap, Box, World                                  # noqa: E402
from test_queue_only import rig                                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
HZ = CFG['comm']['send_hz']
ON = copy.deepcopy(CFG)
ON['overtake']['signal_timeout_go_enable'] = True
# 이 파일의 대상 스위치(signal_timeout_go_enable)는 params 와 테스트가 일치한다
# (둘 다 off). 다만 **다른** 스위치인 signal_stale_queue_enable 이 켜져 있으면
# 그쪽이 last_signal 진단을 채워서, "off 면 진단도 없다" 를 보는 검사가 깨진다.
# 그래서 off 검사에서는 그 이웃 스위치도 함께 끈다 — 이 파일이 재는 것은
# 타임아웃 해제이지 stale 큐가 아니다.
# **params 값이 정본이다** — 두 키의 기본값은 제어기 파트(팀원) 소관이라 이
# 테스트가 정하지 않는다 (2026-09-07, docs/BACKLOG.md B-25).
OFF = copy.deepcopy(CFG)
OFF['overtake']['signal_timeout_go_enable'] = False
OFF['overtake']['signal_stale_queue_enable'] = False
STALE = int(round(OT['signal_stale_s'] * HZ))
TIMEOUT = int(round(OT['signal_unknown_timeout_s'] * HZ))


def step(kr, p, ap, lights=(), n=1, v=0.0):
    """apply 가 하는 순서: observe → 캐시 → 시한 시계."""
    for _ in range(n):
        kr.observe_lights(list(lights))
        kr._tick_cache(ap, p)
        kr._signal_timeout_tick(ap, p, v)


def red_rig(cfg=ON, xs=()):
    kr, p, ap = rig(xs=xs, state=TrafficLightState.Red, d_tl=20.0, cfg=cfg)
    return kr, p, ap


def test_params_present_default_off():
    assert isinstance(OT['signal_timeout_go_enable'], bool)
    assert OT['signal_unknown_timeout_s'] == 10.0 and OT['signal_timeout_clear_m'] == 30.0


def test_off_never_releases():
    kr, p, ap = red_rig(cfg=OFF)
    step(kr, p, ap, n=STALE + TIMEOUT + 10)
    assert kr._stop_target(p, ap) is not None and kr.signal_release(ap) is False
    assert kr.last_signal is None


def test_on_releases_after_timeout_and_only_then():
    kr, p, ap = red_rig()
    step(kr, p, ap, n=STALE)                              # stale 시작 — 이 틱부터 센다
    assert kr.last_signal['signal_stale'] is True
    assert kr.last_signal['timeout_s'] == pytest.approx(1.0 / HZ, abs=0.06)
    step(kr, p, ap, n=TIMEOUT - 2)
    assert kr._stop_target(p, ap) is not None and kr.last_signal['timeout_go'] is False
    step(kr, p, ap, n=1)
    assert kr.last_signal['timeout_go'] is True
    assert kr._stop_target(p, ap) is None and kr.signal_release(ap) is True
    assert kr.last_signal['signal_last_state'] == 'Red'   # state 는 그대로 Red


def test_reported_again_restores_stop():
    """보고가 재개되면 그 state 가 즉시 우선 — 적색이면 다시 선다."""
    kr, p, ap = red_rig()
    step(kr, p, ap, n=STALE + TIMEOUT)
    assert kr._stop_target(p, ap) is None
    step(kr, p, ap, lights=[(7, 1)], n=1)
    assert kr.last_signal['signal_stale'] is False and kr.last_signal['timeout_go'] is False
    assert kr._stop_target(p, ap) is not None and kr.signal_release(ap) is False


def test_stopped_vehicle_ahead_holds():
    kr, p, ap = red_rig(xs=(10.0,))                        # 정지 회랑 객체
    step(kr, p, ap, n=STALE + TIMEOUT + 5)
    assert kr._stop_target(p, ap) is not None and kr.last_signal['timeout_s'] == 0.0


def test_moving_vehicle_within_clear_m_holds():
    kr, p, ap = red_rig()
    mv = Box(9, OT['signal_timeout_clear_m'] - 1.0, 0.0)
    mv.speed = 5.0
    ap._world = World([mv])
    step(kr, p, ap, n=STALE + TIMEOUT + 5)
    assert kr._stop_target(p, ap) is not None


def test_moving_vehicle_beyond_clear_m_does_not_hold():
    kr, p, ap = red_rig()
    mv = Box(9, OT['signal_timeout_clear_m'] + 5.0, 0.0)
    mv.speed = 5.0
    ap._world = World([mv])
    step(kr, p, ap, n=STALE + TIMEOUT)
    assert kr._stop_target(p, ap) is None


def test_pedestrian_latch_holds():
    kr, p, ap = red_rig()
    kr.ped_hold_ids.add(4)
    step(kr, p, ap, n=STALE + TIMEOUT + 5)
    assert kr._stop_target(p, ap) is not None
    kr.ped_hold_ids.clear(); ap.walker_hazard = True
    step(kr, p, ap, n=STALE + TIMEOUT + 5)
    assert kr._stop_target(p, ap) is not None


def test_moving_ego_does_not_count():
    kr, p, ap = red_rig()
    step(kr, p, ap, n=STALE + TIMEOUT + 5, v=3.0)
    assert kr._stop_target(p, ap) is not None and kr.last_signal['timeout_s'] == 0.0


def test_green_stale_has_nothing_to_release():
    kr, p, ap = rig(xs=(), state=TrafficLightState.Green, d_tl=20.0, cfg=ON)
    step(kr, p, ap, n=STALE + TIMEOUT + 5)
    assert kr.last_signal['timeout_go'] is False and kr._stop_target(p, ap) is None


def test_reset_clears_latch():
    kr, p, ap = red_rig()
    step(kr, p, ap, n=STALE + TIMEOUT)
    assert kr._stop_target(p, ap) is None
    kr.on_reset()
    assert kr._stop_target(p, ap) is not None
