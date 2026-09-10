"""
(3) 동률 1순위(= 다음 짝의 회전 방향)가 **한 번도 돌지 않았다**
— `avoid_map.lane_map_valid_side_by_id_enable`.

`valid_entry_lanes` 는 교차로 **직전 섹션**의 차로 키를 담는다 (실측 seg =
[(2533,0,3)…(2533,0,6)]). 그런데 자차는 그 한참 위 섹션에 있다 —
run_20260910_144656 t 12.1 은 (2533,**5**,3) 이고 지도 hops 키도 전부 섹션 5 다.
교집합이 **항상 비어** `_lm_valid_side` 가 늘 None 을 돌려주었고, 동률은
마지막 규칙(우측통행 기본)으로 갈렸다.

`ot_from_hop`(3940047)·`_lane_hops` 섹션 분할과 같은 부류의 버그다.
차로 id 는 도로 안에서 섹션이 달라도 같은 차로를 가리킨다.

참고: 이 로그의 t 12.2 동률에서는 수정해도 **결론이 같다** (좌 = 차로 id 2 는
경로의 진입 차로 집합에 없고, 우 = id 4 는 있다 → 여전히 right). 고치는 것은
"규칙이 돌게 하는 것" 이지 그 틱의 선택을 뒤집는 것이 아니다.
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

CFG = load_params_yaml(PARAMS_YAML)

EGO = (2533, 0, 5)
L4, L3, L6 = (2533, 0, 4), (2533, 0, 3), (2533, 0, 6)
ORDER = [L3, L4, EGO, L6]


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


class P:
    pass


def kr(**over):
    c = copy.deepcopy(CFG)
    c['avoid_map'].update(over)
    return KrRules(c)


# ── (3) 동률 1순위가 한 번도 돌지 않았다 ──────────────────────────────────
# `valid_entry_lanes` 는 교차로 **직전 섹션**의 차로 키를 담는데(실측 seg0 =
# [(2533,0,3)…(2533,0,6)]) 자차는 그 한참 위 섹션에 있다 — t 12.1 은
# (2533,**5**,3) 이고 지도 hops 키도 전부 섹션 5 다. 교집합이 **항상 비어**
# `_lm_valid_side` 가 늘 None 이었다.
import pickle                                                       # noqa: E402

ROUTE = ROOT / 'data' / 'route.pkl'
HOPS_S5 = {'[2533, 5, 3]': 0, '[2533, 5, 2]': -1, '[2533, 5, 4]': 1}


def _p(route, rs=98.9):
    p = P()
    p.route, p.route_index, p.route_s = route, 0, [rs]
    return p


@pytest.fixture(scope='module')
def route():
    if not ROUTE.exists():
        pytest.skip('route.pkl 없음')
    return pickle.load(open(ROUTE, 'rb'))


def test_valid_side_by_id_default_is_off():
    assert CFG['avoid_map']['lane_map_valid_side_by_id_enable'] is False


def test_route_stores_entry_lanes_of_a_different_section(route):
    """전제 — 저장된 차로는 섹션 0, 자차 지도는 섹션 5 다."""
    pair = [e for e in route['valid_entry_lanes'] if e['target'] == 'pair']
    if not pair:
        pytest.skip('pair 세그먼트 없음')
    secs = {k[1] for k in pair[0]['lanes']}
    assert secs == {0}
    assert {int(k.strip('[]').split(',')[1]) for k in HOPS_S5} == {5}


def test_off_never_matches_and_the_rule_is_dead(route):
    k = kr(lane_map_valid_side_by_id_enable=False)
    assert k._lm_valid_side(_p(route), HOPS_S5, '[2533, 5, 3]') is None


def test_on_matches_by_road_and_lane_id(route):
    k = kr(lane_map_valid_side_by_id_enable=True)
    assert k._lm_valid_side(_p(route), HOPS_S5, '[2533, 5, 3]') == 'right'


def test_key_road_lane_drops_the_section():
    k = kr()
    assert k._key_road_lane('[2533, 5, 4]') == (2533, 4)
    assert k._key_road_lane((2533, 0, 4)) == (2533, 4)
    assert k._key_road_lane('nonsense') is None


def test_hops_tie_already_prefers_the_nearer_lane():
    """(3) 뒷부분 — 'hops 작은 쪽 우선' 은 이미 순위에 있다 (free·점선 다음)."""
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    i = src.index('def _tie(c):')
    blk = src[i:i + 420]
    assert '-abs(int(hops[c[2]]))' in blk
    assert blk.index('c[0]') < blk.index('-abs(int(hops[c[2]]))')
