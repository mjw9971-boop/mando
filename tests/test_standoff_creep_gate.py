"""크립 지연 게이트 (2026-09-05, 실차 20260905_155735 근거).

크립은 **비가역**이다 — s_rel 이 기하 완성 게이트의 need(정지 상태 19.0 =
transition_m 12 + shift_ahead_m 5 + shift_geom_margin_m 2) 아래로 한 번
내려가면 시프트가 영영 불가능해진다. 155735 실측 두 건:

  · 01_좌회전2 — 정지 sod 21.9 에서 크립이 곧바로 시작. 시프트는 매 틱
    시도됐으나 `right:occupied`(우측 차로에 id11 정차, 그 차는 t=75 에 서서
    로그 끝 149 s 까지 안 비켰다). 크립이 4.95 s 만에 s_rel 을 19.0 아래로
    끌어내려 그 뒤로는 `right:geom` 고정.
  · 02_직진3 — ot_span 이 활성인 채 크립이 돌아 **시프트 시도가 0회**였다.
    크립 0.8 m/s > stuck_eps 0.2 라 bo_stuck_ticks 가 매 틱 리셋되고,
    span 활성이면 _try_overtake_inner 가 '기각'을 반환하지 않아 reject 시계도
    죽는다 → BREAKOUT 이 안 걸려 span 이 안 풀린다.

그래서 s_rel ≥ need 인 동안에는 크립을 보류하고 시프트·BREAKOUT 사다리에
먼저 기회를 준다. 여는 조건은 ① d < need ② BREAKOUT L4 ③ 지연 경과.
"""
import copy
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'team_code'))

from conftest import PARAMS_YAML                                   # noqa: E402
from test_avoid import Ap, Box, make                               # noqa: E402
from vtd_adapter.config import load_params_yaml                    # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
NEED = OT['transition_m'] + OT['shift_ahead_m'] + OT['shift_geom_margin_m']
NEED_L3 = OT['transition_m'] + OT['shift_ahead_l3_m'] + OT['shift_geom_margin_m']


def rig(d=25.0, delay_s=10.0, cause=True):
    """d [m] 앞에 라바콘. `_ap` 를 붙여 크립 경로를 태운다.

    delay_s 는 **명시적으로** 넣는다 — params 의 현재 값이 바뀌어도 게이트
    자체의 성질을 보는 검사가 흔들리지 않게 한다 (값 자체는
    test_params_present 가 따로 본다).
    """
    kr, p = make()
    kr.standoff_creep = True
    kr.standoff_half_len = 0.075
    kr.wait_target_d = d
    kr.standoff_id = 2
    kr.creep_delay_ticks = int(round(delay_s * kr.hz))
    ap = Ap(p, actors=[Box(2, d, 0.0)])
    kr._ap = ap
    kr.last_d_end = None if cause else 0.0
    kr._blocker = (lambda *a, **k: ap._world.get_actors()[0]) if cause else (lambda *a, **k: None)
    return kr, p, ap


def test_params_present():
    assert OT['standoff_creep_delay_s'] == 10.0
    assert NEED == pytest.approx(19.0) and NEED_L3 == pytest.approx(15.0)


def test_delay_zero_is_previous_behaviour():
    """0 = 지연 없음. 20260905_155735 의 동작 그대로."""
    kr, _p, _ap = rig(d=21.0, delay_s=0.0)
    assert kr._standoff_profile(0.0) == pytest.approx(OT['standoff_creep_v'])
    assert kr._creep_diag['creep_open_why'] == 'no_gate'


def test_holds_while_shift_still_geometrically_possible():
    """d ≥ need 이면 보류 — 시프트가 아직 가능하므로 기다린다."""
    kr, _p, _ap = rig(d=21.0)                       # 21.0 ≥ need 19.0
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    dg = kr._creep_diag
    assert dg['creep_hold'] is True and dg['creep_hold_why'] == 'need'
    assert dg['creep_need_m'] == pytest.approx(NEED)
    assert dg['so_creep'] is False


def test_opens_when_below_geom_need():
    """① d < need — 이미 기하적으로 불가. 기다릴 이유가 없다."""
    kr, _p, _ap = rig(d=18.0)
    assert kr._standoff_profile(0.0) == pytest.approx(OT['standoff_creep_v'])
    assert kr._creep_diag['creep_open_why'] == 'need'


def test_opens_at_breakout_l4():
    """② 사다리 끝까지 갔는데 시프트 실패 — 더 기다릴 것이 없다."""
    kr, _p, _ap = rig(d=21.0)
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)      # 보류
    kr.bo_level = kr.BO_CREEP
    assert kr._standoff_profile(0.0) == pytest.approx(OT['standoff_creep_v'])
    assert kr._creep_diag['creep_open_why'] == 'breakout'


def test_l3_relaxation_lowers_need():
    """L3 이상은 ahead_m 이 줄어 need 가 15.0 이 된다 — _side_pass 와 같은 식."""
    kr, _p, _ap = rig(d=17.0)
    assert kr._standoff_profile(0.0) == pytest.approx(OT['standoff_creep_v'])  # 17 < 19
    kr2, _p2, _ap2 = rig(d=17.0)
    kr2.bo_level = kr2.geom_relax_lvl                            # need 19.0 → 15.0
    assert kr2._standoff_profile(0.0) == pytest.approx(0.0)      # 17 ≥ 15 → 보류
    assert kr2._creep_diag['creep_need_m'] == pytest.approx(NEED_L3)


def test_opens_after_delay_expires():
    """③ 안전망 — 사다리가 아예 안 도는 경우(신호 전이로 매번 리셋 등)."""
    kr, _p, _ap = rig(d=21.0, delay_s=1.0)
    n = int(round(1.0 * kr.hz))
    for _ in range(n):
        assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._standoff_profile(0.0) == pytest.approx(OT['standoff_creep_v'])
    assert kr._creep_diag['creep_open_why'] == 'delay'


def test_delay_clock_excludes_blocked_ticks():
    """적신호에 걸린 틱은 지연 시계에 넣지 않는다.

    ot_blocked_ticks 를 쓰면 적신호 대기 22.6 s 가 그대로 쌓여 녹색으로
    바뀌는 순간 이미 만료된다 (실측 20260905_155049 02_직진3).
    """
    kr, _p, ap = rig(d=21.0, delay_s=1.0)
    for _ in range(int(round(0.9 * kr.hz))):                     # 0.9 s 보류
        kr._standoff_profile(0.0)
    ap.traffic_light_hazard = True                               # 적신호
    for _ in range(int(round(5.0 * kr.hz))):                     # 5 s 대기
        assert kr._standoff_profile(0.0) == pytest.approx(0.0)
        assert kr._creep_diag['creep_block'] == 'cause'
    ap.traffic_light_hazard = False
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)       # 시계가 0 부터 다시
    assert kr._creep_diag['creep_hold'] is True
    assert kr._creep_diag['creep_hold_s'] == pytest.approx(0.0)


def test_delay_clock_resets_on_target_change():
    kr, _p, _ap = rig(d=21.0, delay_s=1.0)
    for _ in range(int(round(0.9 * kr.hz))):
        kr._standoff_profile(0.0)
    assert kr._creep_diag['creep_hold_s'] > 0.5
    kr.standoff_id = 99                                          # 다른 차단물
    kr._standoff_profile(0.0)
    assert kr._creep_diag['creep_hold_s'] == pytest.approx(0.0)


@pytest.mark.parametrize('flag', ['traffic_light_hazard', 'walker_hazard',
                                  'walker_close', 'stop_sign_hazard'])
def test_exclusions_still_precede_the_gate(flag):
    """지연 게이트가 배제를 바꾸지 않는다 — 신호·보행자가 먼저다."""
    kr, _p, ap = rig(d=18.0)                                     # 게이트라면 즉시 open
    setattr(ap, flag, True)
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_block'] == 'cause'
    assert 'creep_open_why' not in kr._creep_diag


def test_stop_gap_still_precedes_the_gate():
    kr, _p, _ap = rig(d=4.0)                                     # d_stop 4.87 미만
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_block'] == 'stop_gap'


def test_outside_baseline_untouched_by_gate():
    """기준선 바깥은 지연과 무관하게 기존 프로파일 그대로."""
    import math
    kr, _p, _ap = rig(d=40.0)
    assert kr._standoff_profile(0.0) == pytest.approx(
        math.sqrt(2.0 * CFG['speed']['stop_profile_a'] * (40.0 - kr.standoff_floor_m)))
    assert kr._creep_diag is None


def test_gate_off_when_geom_gate_disabled():
    """geom 게이트가 꺼져 있으면 조건 ① 이 성립할 수 없다 — 사다리를 기다린다."""
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['shift_geom_gate_enable'] = False
    cfg['overtake']['standoff_creep_delay_s'] = 10.0
    kr, p = make(cfg)
    kr.standoff_creep = True
    kr.standoff_half_len = 0.075
    kr.wait_target_d = 21.0
    kr.standoff_id = 2
    ap = Ap(p, actors=[Box(2, 21.0, 0.0)])
    kr._ap = ap
    kr.last_d_end = None
    kr._blocker = lambda *a, **k: ap._world.get_actors()[0]
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_hold_why'] == 'breakout'
    assert kr._creep_diag['creep_need_m'] is None
