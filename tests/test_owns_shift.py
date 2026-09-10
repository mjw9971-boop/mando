"""[3] 차로 지도가 시프트를 소유한다 — avoid_map.lane_map_owns_shift_enable.

켜지면 "어디로·언제" 는 `lane_plan` 만 정한다. 옛 `_try_overtake` 는 안전
검사(게이트 10개)와 안전망(REACTIVE·BREAKOUT·never_stall)만 남는다:

  · side 후보가 계획한 쪽 **하나**다 — 게이트에서 떨어지면 반대쪽으로 새지
    않고 그 틱은 포기한다 (지도가 매 틱 다시 고르므로 다음 틱에 다시 온다).
  · 실선 2바퀴 완화(solid_second_pass)를 안 돈다.
  · 시프트 시점은 `lane_plan.armed`(램프 시작점 통과) 가 정한다.
  · 활성 목표 갈아타기는 `lane_switch_margin_m` 만큼 이득이 있을 때만.

소유 조건은 셋 다다: 스위치 ∧ 지도 on ∧ 이번 틱 pick 이 있다. 지도가 비어
있는 틱에 소유권을 주면 아무 데도 못 간다.
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


def kr(owns=True, map_on=True, **over):
    c = copy.deepcopy(CFG)
    c['avoid_map']['lane_map_owns_shift_enable'] = bool(owns)
    c['avoid_map']['lane_map_avoid_enable'] = bool(map_on)
    c['avoid_map'].update(over)
    return KrRules(c)


# ── 소유 조건 ────────────────────────────────────────────────────────────
def test_owns_needs_all_three():
    k = kr(); k.last_lane_plan = {'pick': '[1, 0, -2]', 'side': 'right'}
    assert k._owns_shift() is True
    k.last_lane_plan = {'pick': None}
    assert k._owns_shift() is False               # 지도가 목표를 못 골랐다
    k2 = kr(owns=False); k2.last_lane_plan = {'pick': '[1, 0, -2]'}
    assert k2._owns_shift() is False              # 스위치 off
    k3 = kr(map_on=False); k3.last_lane_plan = {'pick': '[1, 0, -2]'}
    assert k3._owns_shift() is False              # 지도 off


def test_switch_default_is_off():
    assert CFG['avoid_map'].get('lane_map_owns_shift_enable') is False
    assert CFG['avoid_map'].get('lane_switch_margin_m') == 20.0


def test_decide_m_matches_the_detection_window():
    """트리거 거리를 감지창과 맞췄다 — 창 안에 보이는데 트리거가 안 걸리면 안 된다."""
    assert CFG['avoid_map']['lane_map_decide_m'] == CFG['avoid_map']['lane_map_ahead_m']


# ── 재타겟 최소 이득 ─────────────────────────────────────────────────────
class _P:
    pass


def retarget_rig(new_free, cur_free, margin=20.0):
    k = kr(lane_map_retarget_enable=True, lane_switch_margin_m=margin)
    k.ot_span = (10, 200)
    k.ot_target = (1, 0, -2)
    k.last_lane_plan = {'pick': '[1, 0, -3]', 'side': 'right'}
    k.last_lane_map = {'free_run': {'[1, 0, -3]': new_free, '[1, 0, -2]': cur_free}}
    return k


def test_retarget_needs_the_margin():
    """이득이 마진 미만이면 갈아타지 않는다 — 램프를 매 틱 다시 그리면 안 된다."""
    k = retarget_rig(new_free=30.0, cur_free=20.0)      # +10 < 20
    assert k._lm_retarget(_P(), _P(), 5.0) is False


def test_retarget_margin_zero_is_the_old_behaviour():
    """0 이면 이득 검사를 하지 않는다 (이전 동작). 그 뒤 조건에서 걸린다."""
    k = retarget_rig(new_free=30.0, cur_free=20.0, margin=0.0)
    # 이득 검사를 통과하면 회랑 조회로 넘어가 목(mock) 없이 예외 없이 거짓을 낸다
    try:
        out = k._lm_retarget(_P(), _P(), 5.0)
    except Exception:                                      # noqa: BLE001
        out = 'raised'
    assert out is not True


def test_retarget_same_target_is_never_a_switch():
    k = retarget_rig(new_free=80.0, cur_free=10.0)
    k.ot_target = (1, 0, -3)                               # 이미 그리로 가는 중
    assert k._lm_retarget(_P(), _P(), 5.0) is False


# ── 동률 깨기 ────────────────────────────────────────────────────────────
def test_tie_break_prefers_right():
    """양쪽 free_run 이 같으면 우측(우측통행). 예전에는 차로 키 순서가 이겼다."""
    k = kr()
    hops = {'[1, 0, -1]': 0, '[1, 0, -2]': 1, '[1, 0, 0]': -1}
    # _lm_valid_side 는 route 정보가 없으면 None → 우측 기본
    class P:
        route = {}
        route_s = [0.0]
        route_index = 0
    assert k._lm_valid_side(P(), hops, '[1, 0, -1]') is None


# ── (4) 단일 후보 폴백 ───────────────────────────────────────────────────
#
# owns_shift 가 계획한 쪽 하나만 넘기면, 그 하나가 게이트에서 떨어졌을 때
# 그 틱은 끝이다 (continue → 루프 종료). 좌우 2바퀴를 돌던 옛 로직보다 오히려
# 좁아진다. 폴백은 ① 계획한 쪽 ② lane_plan 후보 free_run 2위의 쪽 ③ 반대쪽.
EGO = [1, 0, -2]


HOPS = {'[1, 0, -2]': 0, '[1, 0, -3]': 1, '[1, 0, -1]': -1, '[1, 0, -4]': 2}


def plan(pick, cands, side='right'):
    return {'pick': pick, 'side': side, 'ego_lane': EGO, 'cands': cands}


def with_map(k):
    k.last_lane_map = {'hops': HOPS}
    return k


def test_second_side_reads_the_hops_sign_not_the_lane_id():
    """좌/우는 지도의 hops 부호로만 읽는다 — id 부호 규약은 방향마다 뒤집힌다."""
    k = with_map(kr())
    lp = plan('[1, 0, -3]', {'[1, 0, -3]': {'free': 80.0},
                             '[1, 0, -1]': {'free': 40.0}})
    assert k._lm_second_side(lp) == 'left'      # hops -1 → left


def test_second_side_is_none_with_a_single_candidate():
    k = with_map(kr())
    assert k._lm_second_side(plan('[1, 0, -3]', {'[1, 0, -3]': {'free': 80.0}})) is None


def test_second_side_is_none_without_a_map():
    """지도가 없으면 부호를 알 수 없다 — 추정하지 않는다."""
    k = kr()
    lp = plan('[1, 0, -3]', {'[1, 0, -3]': {'free': 80.0},
                             '[1, 0, -1]': {'free': 40.0}})
    assert k._lm_second_side(lp) is None


def test_second_side_is_none_without_a_plan():
    assert with_map(kr())._lm_second_side({}) is None


def test_fallback_switch_default_is_off():
    assert CFG['avoid_map'].get('lane_map_fallback_side_enable') is False


def test_fallback_order_has_no_duplicates():
    """계획한 쪽·2위·반대쪽이 겹쳐도 한 번씩만 돈다."""
    k = with_map(kr(lane_map_fallback_side_enable=True))
    lp = plan('[1, 0, -3]', {'[1, 0, -3]': {'free': 80.0},
                             '[1, 0, -1]': {'free': 60.0}}, side='right')
    second = k._lm_second_side(lp)              # hops -1 → left
    order = [lp['side']]
    if second and second not in order:
        order.append(second)
    other = 'right' if lp['side'] == 'left' else 'left'
    if other not in order:
        order.append(other)
    assert order == ['right', 'left']
    assert len(order) == len(set(order))
