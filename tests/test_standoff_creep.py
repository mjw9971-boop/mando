"""standoff 기준선 **안쪽** 크립 바닥 (2026-09-05).

    d ≥ standoff  →  √(2a(d − standoff))     기존 프로파일, 무수정
    d_stop < d < standoff  →  standoff_creep_v
    d ≤ d_stop    →  0                        (진짜 정지)

왜: d < standoff 이면 √(2a·max(0, d−standoff)) 가 항구적으로 0 이고 min()
사다리에 이를 되올릴 후보가 없다 → 영구 정지. 실측 2026-09-04 7건이 이 형태로
11.5~78.2 s 멈췄다. standoff_floor_m 을 낮춰도 오버슛(0.7~2.6 m)으로 매번 바닥을
넘어 들어가므로 잠금 **지점만** 앞으로 옮겨진다 — 여기서만 풀 수 있다.

크립은 **min() 안의 후보**다. `_standoff_profile` 의 반환값만 바꾸고, 호출처의
병합은 여전히 `so < candidate` 하나다. 신호·보행자 후보가 더 낮으면 그쪽이 이긴다.
"""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'team_code'))

from conftest import PARAMS_YAML                                   # noqa: E402
from test_avoid import Ap, Box, Planner, make                      # noqa: E402
from vtd_adapter.config import load_params_yaml                    # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
VH = CFG['vehicle']
A_STOP = CFG['speed']['stop_profile_a']
FRONT = VH['wheelbase'] + VH['front_overhang_m']                   # 뒷축 → 앞범퍼


def rig(creep=True, half_len=0.075, d=20.0, cause=True, delay_s=None):
    """d [m] 앞에 반길이 half_len 인 정지 물체. `_ap` 를 붙여 크립 경로를 태운다.

    delay_s: 크립 **지연 게이트**(2026-09-05 추가, test_standoff_creep_gate.py)를
    끄고 싶을 때 0 을 준다. 이 파일은 지연이 아니라 **크립 바닥 자체**를 보므로
    바닥을 재는 검사는 지연을 0 으로 두고 부른다.
    """
    kr, p = make()
    kr.standoff_creep = creep
    if delay_s is not None:
        kr.creep_delay_ticks = int(round(delay_s * kr.hz))
    kr.wait_target_d = d
    kr.standoff_id = 2
    kr.standoff_half_len = half_len
    ap = Ap(p, actors=[Box(2, d, 0.0)])
    kr._ap = ap
    # `_obstacle_cause` 를 통과/차단시키는 최소 조작 — 새 판정을 만들지 않는다.
    kr.last_d_end = None if cause else 0.0                         # 종점 사정권으로 차단
    kr._blocker = (lambda *a, **k: ap._world.get_actors()[0]) if cause else (lambda *a, **k: None)
    return kr, p, ap


def test_params_present_and_default_off():
    assert OT['standoff_creep_enable'] is False                    # kill switch 기본 off
    assert OT['standoff_creep_v'] == 0.8
    assert OT['standoff_creep_gap_m'] == 1.0


def test_outside_baseline_untouched():
    """d ≥ standoff 는 기존 식 그대로 — 스위치와 무관하다."""
    for creep in (False, True):
        kr, _p, _ap = rig(creep=creep, d=40.0)
        assert kr._standoff_profile(0.0) == pytest.approx(
            math.sqrt(2.0 * A_STOP * (40.0 - kr.standoff_floor_m)))


def test_off_keeps_zero_inside_baseline():
    """꺼져 있으면 기준선 안쪽은 이전 동작(0 고정)이고 진단도 남기지 않는다."""
    kr, _p, _ap = rig(creep=False, d=20.0)
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag is None


def test_on_gives_creep_inside_baseline():
    kr, _p, _ap = rig(d=20.0, delay_s=0.0)
    assert kr._standoff_profile(0.0) == pytest.approx(OT['standoff_creep_v'])
    assert kr._creep_diag['so_creep'] is True


def test_stop_distance_includes_front_overhang_and_half_length():
    """d 는 **뒷축 → 객체 중심** 축이다. 고정값으로 두면 앞범퍼가 객체 중심을
    지나쳐 선다 — 앞범퍼 오프셋과 객체 반길이를 반드시 더한다."""
    half = 2.195                                                   # 승용차 4.39 m
    d_stop = FRONT + half + OT['standoff_creep_gap_m']
    assert d_stop == pytest.approx(6.994)
    kr, _p, _ap = rig(half_len=half, d=d_stop + 0.5)
    assert kr._standoff_profile(0.0) == pytest.approx(OT['standoff_creep_v'])
    kr, _p, _ap = rig(half_len=half, d=d_stop - 0.5)
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_block'] == 'stop_gap'


def test_stop_distance_scales_with_object_size():
    """대형차(11.96 m)는 라바콘보다 훨씬 멀리 선다 — 고정 상수로는 못 덮는다."""
    d = 8.0
    kr, _p, _ap = rig(half_len=0.075, d=d)                         # 라바콘 → d_stop 4.87
    assert kr._standoff_profile(0.0) == pytest.approx(OT['standoff_creep_v'])
    kr, _p, _ap = rig(half_len=5.98, d=d)                          # 대형차 → d_stop 10.78
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)


def test_unknown_size_does_not_creep():
    kr, _p, _ap = rig(half_len=None, d=20.0)
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_block'] == 'no_size'


def test_ped_hold_latch_blocks_creep():
    """횡단보도 홀드 래치는 PDM walker 플래그와 축이 달라 2차 방어로 따로 본다."""
    kr, _p, _ap = rig(d=20.0)
    kr.ped_hold_ids = {7}
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_block'] == 'ped_hold'


@pytest.mark.parametrize('flag', ['traffic_light_hazard', 'walker_hazard',
                                  'walker_close', 'stop_sign_hazard'])
def test_hazard_flags_block_creep(flag):
    """배제는 새 함수가 아니라 `_obstacle_cause` 재사용이다 — 신호·보행자에서
    크립이 나가면 고착보다 나쁘다."""
    kr, _p, ap = rig(d=20.0)
    setattr(ap, flag, True)
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_block'] == 'cause'


def test_route_end_latch_blocks_creep():
    kr, _p, _ap = rig(d=20.0)
    kr.latched = True
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_block'] == 'cause'


def test_queue_blocks_creep():
    """선행차 큐 뒤에 선 것은 데드락이 아니다 (_tick_queue)."""
    kr, _p, _ap = rig(d=20.0)
    kr._tick_queue = True
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_block'] == 'cause'


def test_yellow_latch_blocks_creep():
    kr, _p, _ap = rig(d=20.0)
    kr.y_decision = 'STOP'
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_block'] == 'cause'


def test_stopline_hold_blocks_creep():
    kr, _p, _ap = rig(d=20.0)
    kr.sl_hold_left = 3
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)
    assert kr._creep_diag['creep_block'] == 'cause'


def test_creep_is_a_min_candidate_not_an_override():
    """크립은 상한 후보일 뿐이다 — 더 낮은 후보가 있으면 그쪽이 최종값이다.

    apply 의 병합은 `so < candidate` 하나이므로, 여기서는 그 병합식을 그대로
    재현해 크립이 다른 후보를 덮지 못함을 본다.
    """
    kr, _p, _ap = rig(d=20.0, delay_s=0.0)
    so = kr._standoff_profile(0.0)
    assert so == pytest.approx(OT['standoff_creep_v'])
    for other in (0.0, 0.3, 0.79):                                 # 신호·보행자 등
        candidate = other
        if so is not None and (candidate is None or so < candidate):
            candidate = so
        assert candidate == pytest.approx(other)                   # 크립이 이기지 않는다


def test_diag_key_is_so_creep_not_creep():
    """BREAKOUT 진단이 같은 last_avoid 에 'creep' 을 나중에 쓴다 — 이름이 겹치면
    덮인다 (실측 02_직진3 27틱에서 발동이 False 로 뒤집혔다)."""
    kr, _p, _ap = rig(d=20.0, delay_s=0.0)
    kr._standoff_profile(0.0)
    assert 'so_creep' in kr._creep_diag and 'creep' not in kr._creep_diag
    for k in ('creep_d', 'creep_standoff', 'creep_stop_m', 'creep_v'):
        assert k in kr._creep_diag
