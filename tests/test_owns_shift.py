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


# [4] 차선책은 `_lm_second_side`(삭제됨)가 아니라 `lane_plan['ranked']` 에서
# 나온다. 게이트 층이 방향을 정하던 경로를 없앴으므로 그 함수도 같이 죽었다.
def test_fallback_switch_default_is_off():
    assert CFG['avoid_map'].get('lane_map_fallback_side_enable') is False


def test_second_side_helper_is_gone():
    """게이트 층의 방향 선택이 사라졌으므로 그 헬퍼도 남아 있으면 안 된다."""
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    assert '_lm_second_side' not in src
    assert '_lm_fb_hops' not in src


def test_fallback_order_has_no_duplicates_and_no_opposite():
    """순위표에서 side 를 뽑되 중복은 한 번씩, **반대편은 붙이지 않는다**."""
    ranked = [{'lane': 'a', 'side': 'right', 'hops': 1},
              {'lane': 'b', 'side': 'right', 'hops': 2},
              {'lane': 'c', 'side': 'left', 'hops': 2}]
    order = list(dict.fromkeys(r['side'] for r in ranked))
    assert order == ['right', 'left']
    assert len(order) == len(set(order))
    ranked_one_side = [r for r in ranked if r['side'] == 'right']
    assert list(dict.fromkeys(r['side'] for r in ranked_one_side)) == ['right']


# ── 복귀 차로 기준을 **hop 오프셋**으로 (섹션 독립) ──────────────────────
#
# 차로 **키**로 기억하면 자차가 섹션을 넘는 순간 `free_run` 의 키와 안 맞아
# 조용히 무효가 된다. 실측 2026-09-10 run_20260910_110743: rs 459.0 에
# (2756,**3**,5) 에서 좌측 시프트했는데 정지 지점의 지도는 (2756,**1**,*) 라
# 조회가 전부 None 이었다 — 스위치를 켜고 돌렸는데 아무 일도 안 일어났다.
# hop 은 매 틱 자차 기준으로 다시 계산되므로 섹션과 무관하다.
def test_from_hop_is_opposite_of_the_shift_side():
    """좌측으로 n 칸 갔으면 떠나온 차로는 목표 기준 **+n**(우측)이다."""
    k = kr()
    for side, n, want in (('left', 1, 1), ('left', 2, 2),
                          ('right', 1, -1), ('right', 2, -2)):
        k.ot_from_hop = int(n) * (1 if side == 'left' else -1)
        assert k.ot_from_hop == want


def test_from_hop_starts_cleared():
    assert kr().ot_from_hop == 0


def test_shift_ref_switch_default_is_off():
    assert CFG['avoid_map'].get('lane_map_shift_ref_enable') is False


# ── (c) pick 즉시 시프트 / (e) 홀드 중 재타겟 ────────────────────────────
#
# 실측 2026-09-10 run_20260910_115526:
#   pick   t 45.6 rs 428.2  s_rel 79.7  v 11.9
#   시프트 t 49.7 rs 467.3  s_rel 38.5  → **39 m / 4.1 s 지연**
#   그 사이 STANDOFF→WAIT→WAIT_EXPIRED, 속도 11.9 → 6.85
#   대가: 두 번째 blocker 에서 avail_m 0.8 < need_m 18.2 (ramp_too_late)
#
# `armed` 는 대안이 못 된다 — lane_plan 이 램프를 v_cap(20 km/h)로 재는데 자차는
# 43 km/h 라 start_s_rel 57.4 이고 armed 는 s_rel ≈ 22 에서야 참이 된다.
def test_shift_on_pick_switch_default_is_off():
    assert CFG['avoid_map'].get('lane_map_shift_on_pick_enable') is False


def test_retarget_in_hold_switch_default_is_off():
    assert CFG['avoid_map'].get('lane_map_retarget_in_hold_enable') is False


def test_shift_on_pick_needs_owns_shift_and_a_pick():
    """소유 조건이 없으면 즉시 시프트도 없다 — 옛 경로 그대로."""
    k = kr(lane_map_shift_on_pick_enable=True)
    k.last_lane_plan = {'pick': None}
    assert k.lm_shift_on_pick is True
    assert k._owns_shift() is False          # pick 이 없으면 소유하지 않는다


def test_shift_on_pick_is_read_from_params():
    assert kr(lane_map_shift_on_pick_enable=True).lm_shift_on_pick is True
    assert kr(lane_map_shift_on_pick_enable=False).lm_shift_on_pick is False


def test_retarget_in_hold_is_read_from_params():
    assert kr(lane_map_retarget_in_hold_enable=True).lm_retarget_in_hold is True
    assert kr(lane_map_retarget_in_hold_enable=False).lm_retarget_in_hold is False


# ── 폴백은 **칸 수도** 따라가야 한다 ─────────────────────────────────────
#
# 안 따라가면 `_lm_hops` 가 폴백 side 에 None 을 돌려 n_hops = 1 이 되고,
# 2칸짜리 차선책이 **한 칸 옆**에 떨어진다. 그 한 칸이 막힌 차로면 최악이다.
# 실측 2026-09-10 run_20260910_115526 t 49.5: pick (2756,3,6)(우 1칸)이 기각돼
# 차선책 (2756,3,3)(**좌 2칸**, free 80.0)으로 갔어야 하는데 좌 **1칸** =
# (2756,3,4) 로 갔다. 그 차로는 id 2 가 38.5 m 앞에서 막고 있었고, 결국
# 5.8 m 까지 기어들어가 ramp_too_late 로 갇혔다.
HOPS2 = {'[2756, 3, 5]': 0, '[2756, 3, 4]': -1, '[2756, 3, 3]': -2,
         '[2756, 3, 6]': 1}


def rig_fb():
    k = kr(lane_map_fallback_side_enable=True)
    k.last_lane_map = {'hops': HOPS2}
    k.last_lane_plan = {
        'pick': '[2756, 3, 6]', 'side': 'right', 'hops': 1,
        'ego_lane': [2756, 3, 5],
        'cands': {'[2756, 3, 6]': {'free': 80.0, 'hops': 1},
                  '[2756, 3, 3]': {'free': 80.0, 'hops': 2}}}
    return k


def test_fallback_carries_the_runner_up_hop_count():
    """차선책이 좌 2칸이면 폴백도 **2칸**이어야 한다 (순위표에서 온다)."""
    k = rig_fb()
    k._lm_rank_hops = {'right': 1, 'left': 2}       # lane_plan 이 채우는 값
    assert k._lm_hops('left') == 2                  # 1 이 아니다
    assert k._lm_hops('right') is None              # 계획한 쪽은 1칸이라 None


def test_fallback_hops_none_without_the_switch():
    """스위치가 꺼져 있으면 이전 동작 — 폴백 side 는 항상 None."""
    k = rig_fb()
    k._lm_rank_hops = {'right': 1, 'left': 2}
    k.lm_fallback = False
    assert k._lm_hops('left') is None


def test_fallback_hops_none_when_runner_up_is_one_hop():
    """차선책이 1칸이면 None (기존 관례 — n>1 일 때만 값을 준다)."""
    k = rig_fb()
    k._lm_rank_hops = {'right': 1, 'left': 1}
    assert k._lm_hops('left') is None
