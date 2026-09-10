"""
[4] 횡 결정을 하나로 — `lane_map` 만 방향을 정한다.

실주행에서 결정자가 둘이라 실제로 사고가 났다.
실측 2026-09-10 run_20260910_144656 t 65.3~66.5, 자차 (2533,0,5)
**free_run 80.0 = 완전히 빈 차로**:

    lane_plan  pick (2533,0,3) side **left** hops 2
    게이트     ['left:occupied_mid@p1']        ← 중간 차로 4 가 id 9 로 막힘
    폴백       반대편 'right' 를 최후 수단으로 덧붙였다 → lane 6
    side_pick  picked 'right' (why 'plateau_empty') ← 여기서 한 번 더 뒤집힌다
    결과       t 70.6 자차가 lane 6 (free 21.0, id 12 **대기열**) 안에 있다

전수: `side_pick` 이 기록된 시프트 생성 **18건 중 13건(72 %)** 이
`lane_plan.side` 와 **다른 쪽**으로 실행됐다.

수정 — `_owns_shift()` 일 때:
  · 차선책은 `lane_plan['ranked']`(cands 순위표)에서 나온다. side 는 **그
    후보의 side** 지 반대편이 아니다. 반대편을 덧붙이는 코드를 없앴다.
  · `_pick_side` 를 타지 않는다. 게이트를 통과한 것 중 순위가 앞선 쪽을 쓴다.
  · 첫 성공에서 끊는다 (반대쪽을 잴 이유가 없다 — cKDTree 4.6 ms 절약).
owns_shift 가 꺼지면 옛 경로 그대로다.
"""
import copy
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402
from test_avoid import Ap, Planner                                 # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)

# run_20260910_144656 t 65.3 실값
EGO = (2533, 0, 5)
L4, L3, L6 = (2533, 0, 4), (2533, 0, 3), (2533, 0, 6)
ORDER = [L3, L4, EGO, L6]
FREE = {str(list(EGO)): 80.0, str(list(L4)): 28.7,
        str(list(L3)): 80.0, str(list(L6)): 35.5}
HOPS = {str(list(EGO)): 0, str(list(L4)): -1,
        str(list(L3)): -2, str(list(L6)): 1}
QLANES = [str(list(L6))]                       # id 12 대기열


class Lg:
    def __init__(self):
        s = np.linspace(0.0, 1000.0, 51)
        self.lanes = {k: {'junction': -1, 'dir': -1, 's': s,
                          'width': np.full_like(s, 3.0), 'length': 1000.0}
                      for k in ORDER}

    def neighbor(self, key, side):
        if key not in ORDER:
            return None
        i = ORDER.index(key) + (1 if side == 'right' else -1)
        return ORDER[i] if 0 <= i < len(ORDER) else None

    def length(self, key):
        return 1000.0

    def locate(self, x, y, prefer=None, **kw):
        return type('M', (), {'lane': EGO, 's': float(x), 't': 0.0})()


def rig(**over):
    c = copy.deepcopy(CFG)
    c['avoid_map']['lane_map_avoid_enable'] = True
    c['avoid_map']['lane_map_owns_shift_enable'] = True
    c['avoid_map']['lane_map_fallback_side_enable'] = True
    c['avoid_map']['lane_map_shift_ref_enable'] = True
    c['avoid_map'].update(over)
    kr = KrRules(c)
    kr._sl_all = []
    p = Planner(d_tl=float('inf'))
    p.lg = Lg()
    ap = Ap(p, actors=[])
    ap._kr_ego_lane = EGO
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    kr._ap = ap
    kr._tick_ego_lane = EGO
    kr._tick_queue = False
    kr._tick_corridor = []
    # 그 틱은 **시프트 중**이었다 (span [1371,3500], 우측 1칸으로 가는 중).
    # 그래서 물리 차로가 비어 보여도 `shift_ref` 가 복귀 차로(hop -1 = lane 4,
    # free 28.7)를 같이 보아 트리거가 걸렸다.
    kr.ot_span = (1371, 3500)
    kr.ot_from_hop = -1
    kr.lane_map = lambda _ap, _p: {
        'ego_lane': list(EGO), 'hops': dict(HOPS), 'free_run': dict(FREE),
        'passable': {k: True for k in FREE}, 'blocked_by': {}, 'queue_lanes': list(QLANES),
    }
    return kr, p, ap


def test_plan_ranks_only_left_candidates_in_the_log_case():
    """전제 — 그 틱의 후보는 lane 3·4 **둘 다 왼쪽**이고 lane 6 은 큐라 빠진다."""
    kr, p, ap = rig()
    plan = kr.lane_plan(ap, p)
    assert plan['side'] == 'left' and plan['pick'] == str(list(L3))
    sides = [r['side'] for r in plan['ranked']]
    assert sides and set(sides) == {'left'}, plan['ranked']
    assert str(list(L6)) not in plan['cands']       # 큐는 후보가 아니다


def test_ranked_carries_each_candidates_own_hops():
    kr, p, ap = rig()
    plan = kr.lane_plan(ap, p)
    by = {r['lane']: r for r in plan['ranked']}
    assert by[str(list(L3))]['hops'] == 2
    assert by[str(list(L3))]['side'] == 'left'


def test_ranked_is_ordered_like_cands():
    """순위표는 정렬된 cands 순서 그대로다 (free → 점선 → hop 적은 쪽 → …)."""
    kr, p, ap = rig()
    plan = kr.lane_plan(ap, p)
    assert plan['ranked'][0]['lane'] == plan['pick']


def test_no_opposite_side_is_appended_any_more():
    """오늘 버그의 직접 원인 — 반대편을 최후 수단으로 붙이던 코드가 없다."""
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    i = src.index("_ranked = _lp.get('ranked')")
    blk = src[i:i + 500]
    assert "_other" not in blk
    assert "dict.fromkeys(" in blk
    # 옛 코드의 흔적이 남아 있으면 안 된다
    assert "_other = 'right' if _lp['side'] == 'left' else 'left'" not in src


def test_pick_side_is_bypassed_under_owns_shift():
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    i = src.index('if self._owns_shift():\n            # [4] 지도가 결정자다')
    blk = src[i:i + 600]
    assert "side = next((sd for sd in _order if sd in plans), None)" in blk
    assert "why = 'lane_map'" in blk


def test_old_path_still_uses_pick_side():
    """(c) owns_shift 가 꺼지면 옛 경로 그대로."""
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    i = src.index("            side, why = self._pick_side(plans, pick_on)\n"
                  "        p = plans[side]")
    assert i > 0, 'else 분기의 _pick_side 호출이 남아 있어야 한다'


def test_fallback_hops_come_from_the_rank_table():
    """폴백 side 의 칸 수는 순위표에 적힌 그 후보의 칸 수다."""
    kr, p, ap = rig()
    kr.lane_plan(ap, p)
    kr._lm_rank_hops = {'left': 2}
    assert kr._lm_hops('left') == 2
    kr.last_lane_plan = dict(kr.last_lane_plan, side='right')
    assert kr._lm_hops('left') == 2          # 계획이 우측이어도 순위표를 본다


def test_switch_off_keeps_single_side_only():
    """lane_map_fallback_side_enable 이 꺼지면 계획한 쪽 하나뿐이다."""
    kr, p, ap = rig(lane_map_fallback_side_enable=False)
    plan = kr.lane_plan(ap, p)
    assert plan['side'] == 'left'
