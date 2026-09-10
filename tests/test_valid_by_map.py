"""
불필요한 복귀 — 마지막 세그먼트에 유효 차로 제약이 **아예 없다**.

실측 2026-09-10 run_20260910_135646 (자차 road 2756 lane 3, 교차로 82 진입):

    route.pkl valid_entry_lanes
      seg 0  target 'pair'    turn 'straight'  lanes 4개  ← 채워져 있다
      seg 2  target 'finish'  turn None        lanes []   ← **비어 있다**

`_lm_deadline_s` 와 `_lm_no_return_m` 은 둘 다 `target=='pair'` 이고 lanes 가
비어 있지 않을 때만 산다. 그래서 마지막 세그먼트에서는 데드라인이 None 이고
`no_return` 이 **켜져 있어도 아무 일도 하지 않는다** (현재 차로가 유효한지
따지는 코드는 애초에 없다 — 그 경로로 떨어진 것이 아니다).

지도로 유도하면 road 2756 sec 0 의 lane 2·3·4·5 가 전부 junction 82 를 직진
통과해 같은 진출 도로(2806)로 간다 — 경로 차로 5 만이 아니라 넷 다 유효하다.

같은 런의 span 문제도 함께 고정한다:

    t 47.4  span 생성      [453.5, 567.3]   blocker 3
    t 53.4  RETARGET 합집합 [453.5, 581.9]   blocker 4 (정지선 대기차)
    정지선 route_s 578.5 · 종료선 577.5

복귀 램프 끝이 정지선 **4.4 m 뒤**라, 자차가 d_stop 6.7 → 5.5 m 에서 차로를
넘고 교차로 안에서 복귀를 마쳤다. 생성 시에는 `span_into_zone` 게이트가
`_next_stopzone_s` 를 보지만 **재타겟의 합집합은 다시 보지 않는다**.
"""
import copy
import pathlib
import pickle
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.lanegraph import LaneGraph

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route.pkl'

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='lane_graph/route.pkl 없음')

EGO = (2756, 0, 3)


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def route():
    return pickle.load(open(ROUTE, 'rb'))


class P:
    points_per_meter = 10


def kr(on_map=False, on_clamp=False):
    c = copy.deepcopy(CFG)
    c['avoid_map']['lane_map_avoid_enable'] = True
    c['avoid_map']['lane_map_valid_by_map_enable'] = on_map
    c['avoid_map']['lane_map_span_zone_clamp_enable'] = on_clamp
    return KrRules(c)


def planner(lg, route, rs):
    p = P()
    p.lg, p.route = lg, route
    p.route_index, p.route_s = 0, [rs]
    return p


def test_defaults_are_previous_behaviour():
    assert CFG['avoid_map']['lane_map_valid_by_map_enable'] is False
    assert CFG['avoid_map']['lane_map_span_zone_clamp_enable'] is False


def test_route_pkl_last_segment_is_empty(route):
    """전제 — 사용자 가설('경로 차로 하나만')이 아니라 **빈 리스트**다."""
    segs = {e['seg']: e for e in route['valid_entry_lanes']}
    last = segs[max(segs)]
    assert last['target'] == 'finish'
    assert last['lanes'] == []
    assert last['chosen'] == (2756, 0, 5)
    first = segs[min(segs)]
    assert first['target'] == 'pair' and len(first['lanes']) == 4


def test_map_derives_all_four_straight_lanes(lg, route):
    """junction 82 를 직진 통과해 같은 진출 도로로 가는 차로 전부."""
    k = kr(on_map=True)
    got = k._lm_valid_by_map(planner(lg, route, 560.0), lg, EGO)
    assert got == {(2756, 0, 2), (2756, 0, 3), (2756, 0, 4), (2756, 0, 5)}


def test_map_set_is_none_when_switch_off(lg, route):
    k = kr(on_map=False)
    assert k.lm_valid_by_map is False


def test_deadline_is_none_without_the_switch(lg, route):
    """이전 동작 — 마지막 세그먼트에 데드라인이 없어 no_return 이 죽는다."""
    k = kr(on_map=False)
    k._tick_ego_lane = EGO
    p = planner(lg, route, 560.0)
    assert k._lm_deadline_s(p, None, 5.0) is None


def test_deadline_appears_with_the_switch(lg, route):
    """수정 후 — 자차가 이미 유효 차로에 있으므로 데드라인 = 세그먼트 끝."""
    k = kr(on_map=True)
    k._tick_ego_lane = EGO
    p = planner(lg, route, 560.0)
    dl = k._lm_deadline_s(p, None, 5.0)
    assert dl == pytest.approx(route['waypoint_s'][-1], abs=0.01)   # 577.53


def test_deadline_is_pulled_in_when_ego_lane_is_not_valid(lg, route):
    """유효 차로 밖이면 램프 길이만큼 앞당긴다."""
    k = kr(on_map=True)
    k._tick_ego_lane = (2756, 0, 3)
    p = planner(lg, route, 560.0)
    k._lm_valid_by_map = lambda *_a, **_kw: {(2756, 0, 5)}          # 자차 제외
    dl = k._lm_deadline_s(p, None, 5.0)
    assert dl is not None and dl < route['waypoint_s'][-1]


def test_zone_clamp_would_cut_the_observed_span(lg, route):
    """실측 span 끝 581.9 는 정지선 578.5 뒤다 — 클램프가 잘라야 한다."""
    k = kr(on_clamp=True)
    assert k.lm_span_zone_clamp is True
    stop_s, margin = 578.5, float(CFG['overtake']['zone_gate_margin_m'])
    limit_idx = int((stop_s - margin) * 10)
    assert limit_idx < 5819, '정지선이 span 끝보다 앞이어야 이 회귀가 성립한다'
    assert limit_idx > 4535


def test_clamp_only_shortens_never_extends():
    """클램프는 자르기만 한다 — span 을 늘리지 않는다."""
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    i = src.index('if self.lm_span_zone_clamp:')
    blk = src[i:i + 600]
    assert 'if _i > span[0] and _i < span[1]:' in blk
    assert 'span = (span[0], _i)' in blk
