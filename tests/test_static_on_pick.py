"""
(1) pick 이 나와도 시프트가 2.9 s 늦던 문제 — `lane_map_static_on_pick_enable`.

실측 2026-09-10 run_20260910_121651 (자차 (2756,3,5), 앞차 id 3):

    t 52.96  id 3 최초 관측 — **speed 0.00**, 그 틱부터 `blocked_by` 에 있다
    t 53.01  lane_plan pick [2756,3,6]
    t 53.91  STANDOFF
    t 54.41  WAIT   obj_s 1.5  t_left 4.5  budget 1.5
    t 55.81  WAIT   obj_s 2.9  t_left 4.4  budget 0.1
    t 55.91  WAIT_EXPIRED → 시프트

내역은 1.50 s(`_static_ok` = obj_static_s) + 1.45 s(그 뒤 WAIT)다.
뒤의 1.45 s 를 연 것은 시간 예산이 **아니라** `obj_s ≥ wait_before_shift_s` 다 —
`t_left < budget` 은 끝내 참이 되지 않는다: standoff = shift_k_s·v 가 v 를
따라가 t_left 가 4.5 에 고정되는데 budget 은 3.0 − obj_s 로 줄기만 한다.

뒤의 1.45 s 는 `lane_map_shift_on_pick_enable` 이 이미 없앤다 (그 런에서는
**꺼져 있었다** — 2칸 폴백이 `_lm_second_side` 를 거쳤으므로 owns_shift 는
켜져 있었고, 그렇다면 on_pick 이 t 54.46 에 열렸어야 한다).
앞의 1.50 s 를 없애는 것이 이 스위치다.
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

CFG = load_params_yaml(PARAMS_YAML)
EGO = (2756, 3, 5)


class A:
    def __init__(self, aid, speed=0.0):
        self.id = aid
        self.speed = speed
        self.type_id = 'vehicle.vtd.car'


def kr(on):
    c = copy.deepcopy(CFG)
    c['avoid_map']['lane_map_avoid_enable'] = True
    c['avoid_map']['lane_map_static_on_pick_enable'] = on
    k = KrRules(c)
    k.last_lane_map = {'ego_lane': list(EGO),
                       'blocked_by': {str(list(EGO)): 3},
                       'free_run': {str(list(EGO)): 66.0}, 'hops': {}}
    return k


def test_default_is_previous_behaviour():
    assert CFG['avoid_map']['lane_map_static_on_pick_enable'] is False


def test_the_wait_is_obj_static_s_long():
    """전제 — 관찰 문턱이 1.5 s (= 30틱 @20 Hz) 다."""
    k = kr(False)
    assert k.obj_static_ticks == 30
    assert float(CFG['overtake']['obj_static_s']) == 1.5


def test_off_needs_the_full_observation():
    """이전 동작 — 29틱까지는 정적이 아니다."""
    k = kr(False)
    a = A(3)
    for n in (0, 15, 29):
        k.obj_ticks = {3: n}
        assert k._static_ok(a) is False, n
    k.obj_ticks = {3: 30}
    assert k._static_ok(a) is True


def test_on_accepts_the_map_verdict_on_the_first_tick():
    """수정 후 — 지도가 내 차로 차단물로 세었으면 0틱에도 참이다."""
    k = kr(True)
    k.obj_ticks = {3: 0}
    assert k._lm_blocks_ego(A(3)) is True


def test_only_the_blocker_of_my_own_lane_counts():
    """옆 차로를 막은 객체·다른 id 는 해당 없다."""
    k = kr(True)
    k.obj_ticks = {}
    assert k._lm_blocks_ego(A(2)) is False           # 다른 id
    k.last_lane_map = {'ego_lane': list(EGO),
                       'blocked_by': {'[2756, 3, 4]': 3}, 'free_run': {}, 'hops': {}}
    assert k._lm_blocks_ego(A(3)) is False           # 옆 차로만 막았다


def test_off_never_consults_the_map():
    k = kr(False)
    k.obj_ticks = {3: 0}
    assert k.lm_static_on_pick is False
    # on_pick 의 두 번째 항이 죽어 있으므로 _static_ok 결과가 그대로다
    assert k._static_ok(A(3)) is False


def test_static_ok_itself_is_untouched():
    """축을 벌리지 않는다 — standoff·큐·보행자가 같은 함수를 쓴다."""
    k = kr(True)
    k.obj_ticks = {3: 0}
    assert k._static_ok(A(3)) is False               # 지도를 봐도 여기선 거짓
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    i = src.index('def _static_ok(')
    assert '_lm_blocks_ego' not in src[i:i + 400]


def test_no_map_no_change():
    k = kr(True)
    k.last_lane_map = None
    assert k._lm_blocks_ego(A(3)) is False
