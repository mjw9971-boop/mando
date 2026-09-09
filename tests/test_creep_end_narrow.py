"""크립 게이트의 종점 배제 폭 — overtake.standoff_creep_end_narrow_enable.

`_standoff_creep` 은 `_obstacle_cause` 로 "지금 정지 원인이 장애물인가" 를 묻는다.
그 함수의 종점 배제는 기본값이 `route_end.active_m`(150) 인데, 그 값은 종점
유령차 **후보를 만드는 창**이지 "종점이 세웠다" 는 뜻이 아니다. never_stall 은
2026-09-08 에 같은 이유로 `unlatch_m`(30) 으로 좁혔고(`_ns_cause` → `ns_end_m`),
크립 게이트만 150 을 그대로 쓰고 있었다.

실측 2026-09-09 run_20260909_231730 rs 485.3 (경로 578.6 m, d_end 138.1 m):
정지 차량 20.5 m 뒤에서 189 s 얼었다. 매 틱 `creep_block='cause'` 였고,
**녹색 구간에도** v_target 0 이었다 — SHIFT_HOLD(red_ahead)가 원인이 아니다.
"""
import copy
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


def cfg(narrow):
    c = copy.deepcopy(CFG)
    c['overtake']['standoff_creep_enable'] = True
    c['overtake']['standoff_creep_end_narrow_enable'] = bool(narrow)
    return c


def rig(narrow, d_end):
    """정지 차량 뒤 기준선 안쪽에서 크립을 부르는 최소 장치."""
    p = Planner(d_tl=float('inf'))
    kr = KrRules(cfg(narrow))
    kr._sl_all = []
    a = Box(2, 20.5, 0.0, 0.0)
    ap = Ap(p, actors=[a])
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = ap.vehicle_hazard = False
    kr._ap = ap
    kr._tick_queue = False
    kr._tick_corridor = []
    kr.wait_target_d = 20.5
    kr.standoff_id = 2
    kr.standoff_half_len = 2.2
    kr.last_d_end = d_end                       # 종점까지 남은 거리
    kr._creep_hold_ticks = 10 ** 6              # 지연 게이트는 이 검사의 대상이 아니다
    kr._creep_hold_id = 2
    return kr, p, ap


def why(kr):
    return (kr._creep_diag or {}).get('creep_block')


# active_m 150 / unlatch_m 30 — 그 사이 값이 이 검사의 핵심 구간이다
MID = 138.1                                     # 실측 d_end


def test_off_blocks_creep_inside_active_m():
    """이전 동작 — d_end 138.1 m 가 active_m(150) 안이라 크립이 막힌다."""
    kr, p, ap = rig(False, MID)
    kr._standoff_creep(22.0, 0.0)
    assert why(kr) == 'cause'


def test_on_opens_creep_outside_unlatch_m():
    """스위치 on — 138.1 m 는 unlatch_m(30) 밖이므로 종점이 이유가 아니다."""
    kr, p, ap = rig(True, MID)
    kr._standoff_creep(22.0, 0.0)
    assert why(kr) != 'cause'


def test_on_still_blocks_at_the_real_route_end():
    """종점 정지는 그대로 지킨다 — unlatch_m(30) 안이면 켜져도 막힌다."""
    kr, p, ap = rig(True, 12.0)
    kr._standoff_creep(22.0, 0.0)
    assert why(kr) == 'cause'


def test_switch_default_is_off():
    """params 기본값은 이전 동작이다 (배치 검증 전에는 켜지 않는다)."""
    assert CFG['overtake'].get('standoff_creep_end_narrow_enable') is False


def test_signal_axis_is_untouched():
    """이 스위치는 **종점 축만** 좁힌다 — 신호 판정은 on/off 가 같아야 한다.

    (적신호 hazard 자체의 처리는 `tl_hazard_far_blocker_enable` 소관이다.
    여기서 보는 것은 "스위치가 그 결과를 바꾸지 않는다" 뿐이다.)
    """
    out = []
    for narrow in (False, True):
        kr, p, ap = rig(narrow, 12.0)           # 종점 축은 양쪽 다 막히는 값
        ap.traffic_light_hazard = True
        kr._standoff_creep(22.0, 0.0)
        out.append(why(kr))
    assert out[0] == out[1] == 'cause'
