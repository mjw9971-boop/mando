"""크립 지연 시계의 누적 — overtake.standoff_creep_delay_pause_enable.

`_standoff_creep` 의 지연 게이트는 배제(적신호·큐·보행자)에 걸린 틱을 세지
않는다. 의도는 맞다 — 적신호 21 s 를 세면 녹색 직후 이미 만료된 상태가 된다.
문제는 **0 으로 되돌린다**는 것이다. 신호가 도는 동안 배제가 주기적으로 들어오면
시계가 영영 만료에 도달하지 못한다.

실측 avoid_sim 12 (run_20260909_232350 재현, 적 13 s / 녹 13 s):
녹색마다 `creep_hold_s` 가 0 → **9.4 s** 까지 갔다가 적색 복귀에 리셋 —
`standoff_creep_delay_s`(10.0)에 **0.6 s 모자라** 다섯 주기 182 s 를 램프
중간(t_off 1.05 = 차로선 위)에서 굳었다.

멈춰 두면(pause) 배제 틱을 안 세는 원래 의도는 그대로면서 주기를 넘겨 누적된다.
"""
import copy
import pathlib
import sys

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402
from test_avoid import Ap, Box, Planner                            # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)


def rig(pause):
    c = copy.deepcopy(CFG)
    c['overtake']['standoff_creep_enable'] = True
    c['overtake']['standoff_creep_delay_pause_enable'] = bool(pause)
    kr = KrRules(c)
    kr._sl_all = []
    a = Box(2, 20.5, 0.0, 0.0)
    ap = Ap(Planner(d_tl=float('inf')), actors=[a])
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = ap.vehicle_hazard = False
    kr._ap = ap
    kr._tick_queue = False
    kr._tick_corridor = []
    kr.wait_target_d = 20.5
    kr.standoff_id = 2
    kr.standoff_half_len = 2.2
    kr.last_d_end = None
    return kr


def test_off_resets_the_clock_on_exclusion():
    """이전 동작 — 배제 한 번에 시계가 0 으로 돌아간다."""
    kr = rig(False)
    kr._creep_hold_ticks = 90
    kr._tick_queue = True                       # 배제 (큐)
    kr._standoff_creep(22.0, 0.0)
    assert kr._creep_hold_ticks == 0


def test_on_pauses_instead_of_resetting():
    """켜면 시계가 **멈추기만** 한다 — 다음 창에서 이어 센다."""
    kr = rig(True)
    kr._creep_hold_ticks = 90
    kr._tick_queue = True
    kr._standoff_creep(22.0, 0.0)
    assert kr._creep_hold_ticks == 90


def test_on_still_does_not_count_excluded_ticks():
    """멈춘다는 것이 '센다' 는 뜻은 아니다 — 배제 틱에 늘지 않아야 한다."""
    kr = rig(True)
    kr._creep_hold_ticks = 90
    kr._tick_queue = True
    for _ in range(40):
        kr._standoff_creep(22.0, 0.0)
    assert kr._creep_hold_ticks == 90


def test_on_accumulates_across_two_windows():
    """창 두 개에 걸쳐 만료에 도달한다 — 되돌리는 쪽은 영영 도달 못 한다."""
    need = KrRules(copy.deepcopy(CFG)).creep_delay_ticks
    assert need > 0
    half = need // 2 + 1
    kr = rig(True)
    for _ in range(half):                       # 창 1 (열려 있음)
        kr._standoff_creep(22.0, 0.0)
    kr._tick_queue = True
    kr._standoff_creep(22.0, 0.0)               # 배제 — 멈추기만
    kr._tick_queue = False
    for _ in range(half):                       # 창 2
        kr._standoff_creep(22.0, 0.0)
    assert kr._creep_hold_ticks >= need


def test_switch_default_is_off():
    assert CFG['overtake'].get('standoff_creep_delay_pause_enable') is False
