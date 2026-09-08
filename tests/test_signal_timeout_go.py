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
# A3 스위치 — off/on 사본. **기본값을 읽지 않는다** (2026-09-07 드리프트 원칙).
LEAD_OFF = copy.deepcopy(ON)
LEAD_OFF['overtake']['signal_timeout_with_lead_enable'] = False
LEAD_ON = copy.deepcopy(ON)
LEAD_ON['overtake']['signal_timeout_with_lead_enable'] = True
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
    """이전 동작 — 정지 앞차가 하나라도 있으면 시계가 아예 안 돈다."""
    kr, p, ap = red_rig(cfg=LEAD_OFF, xs=(10.0,))          # 정지 회랑 객체
    step(kr, p, ap, n=STALE + TIMEOUT + 5)
    assert kr._stop_target(p, ap) is not None and kr.last_signal['timeout_s'] == 0.0


# ── A3: 정지 앞차가 있어도 시한이 돈다 (signal_timeout_with_lead_enable) ──
# 왜: 같은 상황을 막아 주던 다른 안전망도 같이 죽는다 — _is_queue_v2 는 UNKNOWN
# 큐에 "해제 시한이 없다" 고 명시한다. 미보고 신호 + 안 움직이는 앞차 =
# **두 안전망이 동시에 죽어 무한 정지**다.
# 안전한 이유: 시한이 만료돼도 푸는 것은 신호 유래 정지 후보뿐이고
# (_stop_target → None), PDM 의 선행차 IDM 은 그대로 살아 앞차 뒤에 선다.

def test_stopped_lead_no_longer_blocks_the_clock():
    kr, p, ap = red_rig(cfg=LEAD_ON, xs=(10.0,))
    step(kr, p, ap, n=STALE + TIMEOUT + 5)
    assert kr._stop_target(p, ap) is None                  # 신호 후보만 풀렸다
    assert kr.last_signal['timeout_go'] is True


def test_stopped_lead_release_does_not_touch_the_lead_candidate():
    """푸는 것은 신호뿐이다 — 앞차는 여전히 회랑에 있고 PDM 이 그 뒤에 선다."""
    kr, p, ap = red_rig(cfg=LEAD_ON, xs=(10.0,))
    step(kr, p, ap, n=STALE + TIMEOUT + 5)
    assert kr._stop_target(p, ap) is None
    assert len(kr._tick_corridor) == 1                     # 앞차 판정은 그대로다


def test_moving_lead_still_blocks_with_the_switch_on():
    """움직이는 차량은 켜도 막는다 — 실제로 흘러가는 중이면 기다리는 게 맞다."""
    kr, p, ap = red_rig(cfg=LEAD_ON)
    mv = Box(9, OT['signal_timeout_clear_m'] - 1.0, 0.0)
    mv.speed = 5.0
    ap._world = World([mv])
    step(kr, p, ap, n=STALE + TIMEOUT + 5)
    assert kr._stop_target(p, ap) is not None
    assert kr.last_signal['timeout_s'] == 0.0


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


def test_missing_light_object_needs_no_release():
    """통신 끊김의 다른 형태 — 플래너에 신호 객체 자체가 없는 경우.

    이때는 _signal_stale 이 None 이라 시계가 안 돌지만, **풀 것도 없다**:
    _stop_target_raw 가 "신호 없음 → None" 이라 애초에 정지 후보를 안 만든다.
    무한 정지의 원인이 될 수 없다 (CLAUDE.md '무신호 정지선' 확정 사실과 같은 축).

    실제 통신 끊김(신호 객체는 남고 보고만 끊김)은 위 test_on_releases… 가 재는
    경로다 — 플래너 state 가 마지막 값(Red)으로 남고 _light_seen 나이가 쌓여
    stale 이 된다.
    """
    kr, p, ap = red_rig(cfg=LEAD_ON)
    p.next_traffic_lights = [None] * len(p.route_s)
    step(kr, p, ap, n=STALE + TIMEOUT + 5)
    assert kr._stop_target_raw(p, ap) is None              # 정지 후보 자체가 없다
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
