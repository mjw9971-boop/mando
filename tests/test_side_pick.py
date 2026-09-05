"""side 선택 (2026-09-04 실주행 104648/104807).

`_side_pass` 는 ('left','right') 순으로 보고 **먼저 통과한 쪽**을 그대로 썼다.
BREAKOUT 은 occupied 게이트를 끄므로(lvl≥1 + 정지 stuck_hard_s, 진입 조건이 곧 그
조건이라 항상 참) 막힌 좌측도 전 게이트를 통과해 채택되고 우측은 평가조차 되지
않는다 — 실측 104648 t=42.16: 좌측 base_clear −1.15 / 플래토 [5] 인데 채택,
우측 +0.68 / 플래토 없음은 미평가. 경로가 원래 차로로 되돌아가 그대로 고착했다.

지키는 것:
  · 킬 스위치 side_pick_enable=false(기본) → 첫 성공 side 에서 끊는다 = 이전 동작.
  · shift_entry_enable 의존 — 1차 키 entry_plateau_ids 가 _shift_placement
    산출물이라, 꺼져 있으면 기준이 없어 현행(좌측 우선)으로 폴백한다.
  · 둘 다 플래토가 비면 **좌측 유지** (회귀 방지 — 추월집중_01 t=10.60 실측
    OBB 1.02 가 우측 +1.20 으로 뒤집히면 안 된다).
  · 게이트를 완화하지 않는다 — 후보는 전부 기존 게이트를 통과한 side 뿐이다.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']


def plan(base_clear=None, plateau=None):
    """_pick_side 가 보는 최소 형태 — pinfo 만 쓴다."""
    pinfo = {}
    if base_clear is not None:
        pinfo['entry_base_clear'] = base_clear
    if plateau is not None:
        pinfo['entry_plateau_ids'] = plateau
    return {'pinfo': pinfo}


def kr(**over):
    cfg = copy.deepcopy(CFG)
    cfg['overtake'].update(over)
    return KrRules(cfg)


# ── 스위치 ──────────────────────────────────────────────────────────────
def test_params_present_and_default_off():
    assert OT['side_pick_enable'] is False


def test_flag_read_from_params():
    assert kr(side_pick_enable=True).side_pick is True
    assert kr(side_pick_enable=False).side_pick is False


# ── 선택 규칙 ───────────────────────────────────────────────────────────
def test_single_candidate_is_taken():
    k = kr(side_pick_enable=True)
    assert k._pick_side({'right': plan(0.68, [])}, True) == ('right', 'single')
    assert k._pick_side({'left': plan(1.02, [])}, True) == ('left', 'single')


def test_plateau_empty_side_wins_even_with_lower_clearance():
    """실측 104648 t=42.16 — 좌 −1.15/플래토 [5] 대 우 +0.68/플래토 없음."""
    k = kr(side_pick_enable=True)
    plans = {'left': plan(-1.15, [5]), 'right': plan(0.68, [])}
    assert k._pick_side(plans, True) == ('right', 'plateau_empty')


def test_plateau_empty_wins_on_the_left_too():
    k = kr(side_pick_enable=True)
    plans = {'left': plan(1.01, []), 'right': plan(-0.04, [6, 7])}
    assert k._pick_side(plans, True) == ('left', 'plateau_empty')


def test_missing_plateau_key_counts_as_empty():
    """_shift_placement 는 기본 배치가 임계 이상이면 조기 반환해 키를 안 쓴다."""
    k = kr(side_pick_enable=True)
    plans = {'left': plan(-1.15, [5]), 'right': plan(0.68)}
    assert k._pick_side(plans, True) == ('right', 'plateau_empty')


def test_both_empty_keeps_left_even_if_right_is_wider():
    """회귀 방지 — 추월집중_01 t=10.60 좌 +1.02(실측 OBB 1.02) 대 우 +1.20."""
    k = kr(side_pick_enable=True)
    plans = {'left': plan(1.02, []), 'right': plan(1.20, [])}
    assert k._pick_side(plans, True) == ('left', 'both_empty_keep_left')


def test_both_plateau_falls_back_to_base_clear():
    k = kr(side_pick_enable=True)
    plans = {'left': plan(-1.15, [5]), 'right': plan(-1.78, [4, 7])}
    assert k._pick_side(plans, True) == ('left', 'base_clear')
    plans = {'left': plan(-1.78, [4, 7]), 'right': plan(-1.15, [5])}
    assert k._pick_side(plans, True) == ('right', 'base_clear')


def test_both_plateau_tie_keeps_left():
    k = kr(side_pick_enable=True)
    plans = {'left': plan(-1.0, [5]), 'right': plan(-1.0, [4])}
    assert k._pick_side(plans, True) == ('left', 'base_clear')


def test_missing_base_clear_never_beats_a_measured_one():
    k = kr(side_pick_enable=True)
    plans = {'left': plan(None, [5]), 'right': plan(-9.0, [4])}
    assert k._pick_side(plans, True) == ('right', 'base_clear')


# ── 폴백 ────────────────────────────────────────────────────────────────
def test_pick_off_takes_first_side_in_order():
    """스위치가 꺼지면 좌측 우선 — plans 가 둘이어도 순서로 고른다."""
    k = kr(side_pick_enable=False)
    plans = {'left': plan(-1.15, [5]), 'right': plan(0.68, [])}
    assert k._pick_side(plans, False) == ('left', 'single')


def test_shift_entry_off_disables_the_criterion():
    """1차 키가 _shift_placement 산출물이라 shift_entry 없이는 기준이 없다."""
    k = kr(side_pick_enable=True, shift_entry_enable=False)
    assert k.side_pick is True and k.shift_entry is False
    # _side_pass 안의 pick_on 과 같은 식
    assert (k.side_pick and k.shift_entry) is False
    plans = {'left': plan(-1.15, [5]), 'right': plan(0.68, [])}
    assert k._pick_side(plans, False) == ('left', 'single')


def test_pick_on_requires_both_flags():
    assert (kr(side_pick_enable=True, shift_entry_enable=True).side_pick
            and kr(side_pick_enable=True, shift_entry_enable=True).shift_entry) is True
    for a, b in ((True, False), (False, True), (False, False)):
        k = kr(side_pick_enable=a, shift_entry_enable=b)
        assert not (k.side_pick and k.shift_entry)
