"""신호 미보고 큐 판정 (signal_stale_queue_enable, 2026-09-06, 020738/13 근거).

9910 이 다음 정지선 controller 를 보내지 않으면 플래너 state 는 기본 Green 이라
적신호 대기열이 green_expired 로 풀려 장애물이 됐다 (13_좌회전7: ctrl 167/168
736틱 미보고 → 회피 → junction 기각 → 고착). 켜면 미보고 signal_stale_s 이상이면
UNKNOWN — 큐 판정에서 Red 와 같고 해제 시한이 없다. 정지 후보는 그대로다.
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

from test_avoid import Ap, Box, World, try_overtake                   # noqa: E402
from test_queue_only import GREEN_TICKS, TL, rig                     # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
ON = copy.deepcopy(CFG)
ON['overtake']['signal_stale_queue_enable'] = True
# 스위치를 강제로 끈 사본. **params 값이 정본이다** — 이 키의 기본값은 제어기
# 파트(팀원) 소관이라 이 테스트가 정하지 않는다. 기본값이 무엇이든 off 경로
# (녹색 만료로 큐 해제 · 진단 키 없음)는 계속 덮어야 하므로 여기서 명시적으로
# 꺼서 검사한다 (2026-09-07, 판정 근거는 docs/BACKLOG.md B-25).
OFF = copy.deepcopy(CFG)
OFF['overtake']['signal_stale_queue_enable'] = False
# 이웃 스위치도 함께 끈다 — signal_timeout_go 가 켜져 있으면 그쪽이 last_signal
# 진단을 채워서 "off 면 진단도 없다" 가 깨진다 (2026-09-07 팀원이 기본값을 켰다).
# 이 파일이 재는 것은 stale 큐이지 시한 출발이 아니다. **params 값이 정본**이라
# 기본값을 되돌리지 않고 사본에서 끈다.
OFF['overtake']['signal_timeout_go_enable'] = False
STALE_TICKS = int(round(OT['signal_stale_s'] * CFG['comm']['send_hz']))


def feed(kr, p, ap, lights, n):
    """observe_lights n 틱 + 캐시 재계산 (틱당 1회 규칙)."""
    for _ in range(n):
        kr.observe_lights(lights)
        kr._tick_cache(ap, p)


def test_params_present():
    """키가 있는지와 부속 상수만 본다 — 기본값은 팀원 소관이라 강제하지 않는다."""
    assert isinstance(OT['signal_stale_queue_enable'], bool)
    assert OT['signal_stale_s'] == 1.0


def test_off_keeps_green_expired_and_no_diag():
    """off: 미보고여도 이전 동작 — 녹색 만료로 큐 해제, 진단 키 없음."""
    kr, p, ap = rig(xs=(40.0,), state=TrafficLightState.Green, d_tl=60.0, cfg=OFF)
    kr.green_since_ticks = GREEN_TICKS
    feed(kr, p, ap, [], STALE_TICKS + 5)
    assert kr._tick_queue is False and kr.q_reject == 'green_expired'
    assert kr.last_signal is None


def test_on_stale_holds_queue_without_expiry():
    """on + 미보고: 선두 정지선 25 m 안(A) → 큐, 녹색 만료를 적용하지 않는다."""
    kr, p, ap = rig(xs=(40.0,), state=TrafficLightState.Green, d_tl=60.0, cfg=ON)
    kr.green_since_ticks = GREEN_TICKS * 10
    feed(kr, p, ap, [], STALE_TICKS - 1)                       # 아직 stale 아님
    assert kr.last_signal['signal_stale'] is False and kr._tick_queue is False
    feed(kr, p, ap, [], 1)
    assert kr.last_signal['signal_stale'] is True
    assert kr._tick_queue is True and kr.q_info['queue_by_unknown'] is True
    feed(kr, p, ap, [], GREEN_TICKS * 5)                       # 시한 없음
    assert kr._tick_queue is True and kr.q_reject is None
    assert kr.last_signal['signal_last_state'] == 'Green'
    assert kr.last_signal['signal_stale_s'] == pytest.approx(
        (STALE_TICKS + GREEN_TICKS * 5) / CFG['comm']['send_hz'], abs=0.06)


def test_on_recent_report_is_not_stale():
    """controller 가 매 틱 보고되면 이전 동작 그대로 (녹색 만료)."""
    kr, p, ap = rig(xs=(40.0,), state=TrafficLightState.Green, d_tl=60.0, cfg=ON)
    kr.green_since_ticks = GREEN_TICKS
    feed(kr, p, ap, [(7, 3)], STALE_TICKS + 5)
    assert kr.last_signal['signal_stale'] is False
    assert kr._tick_queue is False and kr.q_reject == 'green_expired'


def test_stale_cond_B_between_ego_and_stopline():
    """B: UNKNOWN 은 Red 와 같다 — 정지 객체가 자차~정지선 사이면 큐."""
    kr, p, ap = rig(xs=(10.0,), state=TrafficLightState.Green, d_tl=60.0, cfg=ON)
    feed(kr, p, ap, [], STALE_TICKS + 1)
    assert kr._tick_queue is True and 'B' in kr.q_info['cond']
    assert kr.q_info['queue_by_unknown'] is True


def test_stale_without_blockers_is_not_queue():
    kr, p, ap = rig(xs=(), state=TrafficLightState.Green, d_tl=60.0, cfg=ON)
    feed(kr, p, ap, [], STALE_TICKS + 1)
    assert kr.last_signal['signal_stale'] is True
    assert kr._tick_queue is False


def test_stale_does_not_touch_stop_target():
    """UNKNOWN 은 Green 을 만들지도 Red 를 풀지도 않는다 — 정지 후보는 플래너 state."""
    kr, p, ap = rig(xs=(), state=TrafficLightState.Green, d_tl=60.0, cfg=ON)
    feed(kr, p, ap, [], STALE_TICKS + 1)
    assert kr._stop_target(p, ap) is None                         # Green(기본) → 통과
    kr2, p2, ap2 = rig(xs=(), state=TrafficLightState.Red, d_tl=60.0, cfg=ON)
    feed(kr2, p2, ap2, [], STALE_TICKS + 1)
    assert kr2.last_signal['signal_last_state'] == 'Red'
    assert kr2._stop_target(p2, ap2) is not None                  # Red 유지


def test_stale_queue_suppresses_avoidance_and_creep():
    kr, p, ap = rig(xs=(40.0,), state=TrafficLightState.Green, d_tl=60.0, cfg=ON)
    kr.green_since_ticks = GREEN_TICKS * 10
    feed(kr, p, ap, [], STALE_TICKS + 1)
    try_overtake(kr, ap, p)
    assert kr.last_avoid['state'] == 'SUPPRESS' and kr.last_avoid['suppress'] == 'queue'
    assert kr.last_avoid['queue']['queue_by_unknown'] is True
    assert kr._obstacle_cause(p, ap) is False                     # 크립·BREAKOUT 원인 아님


def test_head_departure_dissolves_unknown_queue():
    """앞차가 가면 따라간다 — 선두가 떠나면 blockers 에서 빠져 그 틱에 해소."""
    kr, p, ap = rig(xs=(40.0,), state=TrafficLightState.Green, d_tl=60.0, cfg=ON)
    kr.green_since_ticks = GREEN_TICKS * 10
    feed(kr, p, ap, [], STALE_TICKS + 1)
    assert kr._tick_queue is True
    ap._world = World([])                                         # 선두 출발
    feed(kr, p, ap, [], 1)
    assert kr._tick_queue is False


def test_no_observation_means_off():
    """observe_lights 가 한 번도 안 불렸으면 판정하지 않는다 — 기존 테스트 불변."""
    kr, p, ap = rig(xs=(40.0,), state=TrafficLightState.Green, d_tl=60.0, cfg=ON)
    kr.green_since_ticks = GREEN_TICKS
    kr._tick_cache(ap, p)
    assert kr.last_signal is None and kr._tick_queue is False
