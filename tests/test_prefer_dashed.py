"""
(2) 실선으로 갈린 차로를 먼저 고르고 게이트에서 기각당하던 문제
— `lane_map_prefer_dashed_enable`.

실측 2026-09-10 run_20260910_121651 t 53.0 (자차 (2756,3,5)):

    cands  (2756,3,6) free 80.0  1칸    ← pick, side=right
           (2756,3,3) free 80.0  2칸
           (2756,3,4) free 77.5  1칸
    t 55.91  reject `right:solid`, rejects ["right:solid@p1"] → 좌로 폴백

`lane_map` 의 `passable` 은 **표시를 아예 안 본다** — 존재 ∧ 같은 방향 ∧
폭 ≥ min_width ∧ 교차로 아님, 넷뿐이다. 실선 판정은 `_side_pass` 의 solid
게이트(`_dashed_ahead_m` + `dash_slack_m`)에만 있다. 그래서 지도는 실선 너머를
고르고, 게이트가 기각하고, 그제야 반대쪽으로 간다.

**실선 후보를 빼지는 않는다** — 게이트의 2바퀴(solid 완화)가 유일한 길인
경우를 잃기 때문이다. free_run 다음 순위로 점선을 올릴 뿐이다.
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

# 로그 그대로: 자차 lane 5, 우 1칸 = lane 6(실선), 좌 2칸 = lane 3(점선)
EGO   = (2756, 3, 5)
L4    = (2756, 3, 4)      # hop -1
L3    = (2756, 3, 3)      # hop -2
R6    = (2756, 3, 6)      # hop +1
ORDER = [L3, L4, EGO, R6]
LANE_W = 2.96

FREE = {str(list(EGO)): 79.6, str(list(L4)): 77.5,
        str(list(L3)): 80.0, str(list(R6)): 80.0}
HOPS = {str(list(EGO)): 0, str(list(L4)): -1,
        str(list(L3)): -2, str(list(R6)): 1}


class LgMark:
    """오른쪽만 실선, 왼쪽은 점선인 4차로 직선."""

    def __init__(self):
        s = np.linspace(0.0, 1000.0, 51)
        self.lanes = {k: {'junction': -1, 'dir': -1, 's': s,
                          'width': np.full_like(s, LANE_W), 'length': 1000.0}
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


def rig(prefer, right_dashed=False):
    c = copy.deepcopy(CFG)
    c['avoid_map']['lane_map_avoid_enable'] = True
    c['avoid_map']['lane_map_prefer_dashed_enable'] = prefer
    kr = KrRules(c)
    kr._sl_all = []
    p = Planner(d_tl=float('inf'))
    p.lg = LgMark()
    ap = Ap(p, actors=[])
    ap._kr_ego_lane = EGO
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    kr._ap = ap
    kr._tick_ego_lane = EGO
    kr._tick_queue = False
    kr._tick_corridor = []
    kr.lane_map = lambda _ap, _p: {
        'ego_lane': list(EGO), 'hops': dict(HOPS), 'free_run': dict(FREE),
        'passable': {k: True for k in FREE}, 'blocked_by': {}, 'queue_lanes': [],
    }
    # 게이트와 같은 잣대를 쓰되, 이 목에서는 방향으로 점선/실선을 준다
    kr._dashed_ahead_m = (lambda lg, lane, side, s0, span:
                          span if (side == 'left' or right_dashed) else 0.0)
    return kr, p, ap


def test_default_is_previous_behaviour():
    assert CFG['avoid_map']['lane_map_prefer_dashed_enable'] is False


def test_passable_does_not_look_at_markings_at_all():
    """전제 확인 — `passable` 은 표시를 안 본다 (사용자 가설의 정정 지점)."""
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    i = src.index('passable[k] = bool(')
    body = src[i:i + 220]
    assert 'junction' in body and 'dir' in body and 'min_w' in body
    for word in ('dash', 'solid', 'mark'):
        assert word not in body, f'passable 이 {word} 를 본다 — 이 테스트를 갱신할 것'


def test_off_picks_the_solid_right_lane():
    """이전 동작 재현 — free 동률에서 hop 이 적은 실선 우측을 고른다."""
    pl = rig(False)[0].lane_plan(*rig(False)[1:])
    kr, p, ap = rig(False)
    pl = kr.lane_plan(ap, p)
    assert pl['pick'] == str(list(R6)) and pl['side'] == 'right'


def test_on_picks_the_dashed_left_lane_two_hops_away():
    """수정 후 — 같은 free 면 넘어도 되는 선 쪽을 고른다 (2칸이어도)."""
    kr, p, ap = rig(True)
    pl = kr.lane_plan(ap, p)
    assert pl['pick'] == str(list(L3)) and pl['side'] == 'left'
    assert pl['hops'] == 2


def test_solid_candidate_is_ranked_not_removed():
    """실선 후보를 빼지 않는다 — 게이트 2바퀴가 유일한 길인 경우를 잃지 않는다."""
    kr, p, ap = rig(True)
    pl = kr.lane_plan(ap, p)
    assert str(list(R6)) in pl['cands']
    assert pl['cands'][str(list(R6))]['dashed'] is False
    assert pl['cands'][str(list(L3))]['dashed'] is True


def test_no_change_when_both_sides_are_dashed():
    """둘 다 점선이면 순위가 그대로다 — free → hop 수 → 우측."""
    a = rig(False, right_dashed=True)
    b = rig(True, right_dashed=True)
    pa = a[0].lane_plan(a[2], a[1])
    pb = b[0].lane_plan(b[2], b[1])
    assert pa['pick'] == pb['pick'] == str(list(R6))


def test_free_run_still_outranks_dashed():
    """점선은 free_run **다음** 순위다 — 더 뚫린 실선 차로를 버리지 않는다."""
    kr, p, ap = rig(True)
    free2 = dict(FREE); free2[str(list(L3))] = 30.0     # 점선 쪽이 훨씬 짧다
    kr.lane_map = lambda _ap, _p: {
        'ego_lane': list(EGO), 'hops': dict(HOPS), 'free_run': free2,
        'passable': {k: True for k in free2}, 'blocked_by': {}, 'queue_lanes': [],
    }
    pl = kr.lane_plan(ap, p)
    assert pl['pick'] == str(list(R6))
