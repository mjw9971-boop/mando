"""
2칸 시프트 중 route_s 가 섹션 경계에서 뒤로 뛰던 문제.

실측 (2026-09-10, run_20260910_121651 — 2칸 좌 시프트 성공 런):

    t 63.32  lane (2756,2,3)→(2756,1,3)  route_s 492.4 → 480.1  Δ −12.3
    t 67.97  lane (2756,1,3)→(2756,0,3)  route_s 490.9 → 480.2  Δ −10.8
    t 73.37  lane (2756,0,3)→(2756,0,4)  route_s 506.6 → 530.4  Δ +23.9

경로 차로는 road 2756 의 **전 섹션에서 lane 5** 다. 2칸 왼쪽인 lane 3 은
경로 차로도, 그 한 칸 이웃(lane 4)도 아니다 → 옛 `_route_index` 는
`off_route` 로 떨어지며 `_route_idx` 를 **얼린다**. 그런데

    route_s = cum_s[route_idx] + m.s

의 `m.s` 는 자차가 실제로 있는 차로의 s 라 **섹션 경계마다 0 으로 돌아간다**.
얼린 cum_s + 새 섹션의 s → 건너뛴 섹션 길이만큼 통째로 역행한다.
Δ 는 우연이 아니라 섹션 길이 그 자체다 (섹션 2 = 12.44 m, 섹션 1 = 10.96 m).

`percep.route_index_hops` 를 2 로 두면 3→4→5 로 걸어가 **같은 섹션의**
경로 차로를 찾는다. 1 이면 이전 동작 그대로다.
"""
import copy
import pathlib

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.ego import EgoTracker
from vtd_adapter.lanegraph import LaneGraph

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml(PARAMS_YAML)

pytestmark = pytest.mark.skipif(not GRAPH.exists(), reason='lane_graph.pkl 없음')

# run_20260910_121651 이 쓴 경로의 road 2756 구간 (data/route.pkl 실값)
ROUTE = {
    'lanes':  [(2756, 3, 5), (2756, 2, 5), (2756, 1, 5), (2756, 0, 5)],
    'cum_s':  [393.82, 480.02, 492.47, 503.43],
}
# 이 축약 경로 안의 인덱스: 0=(3,5) 1=(2,5) 2=(1,5) 3=(0,5).
# 실제 route.pkl 에서는 11~14 지만 cum_s 값이 같아 결론은 동일하다.
IDX_SEC2 = 1


def cfg_with(hops):
    c = copy.deepcopy(CFG)
    c['percep']['route_index_hops'] = hops
    return c


def tracker(lg, hops):
    return EgoTracker(lg, ROUTE, cfg_with(hops))


def route_s(per, lane, s, flags=None):
    f = {} if flags is None else flags
    idx = per._route_index(lane, f)
    return (ROUTE['cum_s'][idx] + s), f


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


def test_default_is_one_hop():
    """기본값은 1 = 이전 동작. 켜야 바뀐다."""
    assert int(CFG['percep']['route_index_hops']) == 1


def test_map_has_the_two_hop_chain(lg):
    """전제: lane 3 →(좌/우 한 칸)→ 4 →→ 5 가 같은 섹션 안에 있다."""
    for sec in (2, 1, 0):
        one = lg.neighbor((2756, sec, 3), 'right')
        assert one == (2756, sec, 4), (sec, one)
        assert lg.neighbor(one, 'right') == (2756, sec, 5), sec


def test_one_hop_freezes_index_and_route_s_goes_backwards(lg):
    """이전 동작 재현 — 섹션 2 에서 1 로 넘어갈 때 route_s 가 뒤로 간다."""
    per = tracker(lg, 1)
    per._route_idx = IDX_SEC2                 # (2756,2,5) 에서 이탈한 상태
    a, fa = route_s(per, (2756, 2, 3), 12.41)  # 섹션 2 끝
    b, fb = route_s(per, (2756, 1, 3), 0.08)   # 섹션 1 시작
    assert fa.get('off_route') and fb.get('off_route')
    assert a == pytest.approx(492.43, abs=0.05)   # 로그 실값
    assert b == pytest.approx(480.10, abs=0.05)   # 로그 실값
    assert b - a == pytest.approx(-12.33, abs=0.1)


def test_two_hops_keeps_route_s_monotone(lg):
    """수정 후 — 같은 두 틱이 앞으로 간다."""
    per = tracker(lg, 2)
    per._route_idx = IDX_SEC2
    a, fa = route_s(per, (2756, 2, 3), 12.41)
    b, fb = route_s(per, (2756, 1, 3), 0.08)
    assert fa.get('off_route') is None and fb.get('off_route') is None
    assert fa.get('beside_route') and fa.get('beside_hops') == 2
    assert b > a, (a, b)
    assert b == pytest.approx(492.55, abs=0.05)   # cum_s[13] + 0.08


def test_two_hops_removes_the_whole_23m_shortfall(lg):
    """섹션 0 진입까지 누적 과소보고 23.4 m 가 사라진다."""
    per1, per2 = tracker(lg, 1), tracker(lg, 2)
    per1._route_idx = per2._route_idx = IDX_SEC2
    s_old, _ = route_s(per1, (2756, 0, 3), 0.13)
    s_new, _ = route_s(per2, (2756, 0, 3), 0.13)
    assert s_old == pytest.approx(480.15, abs=0.05)   # 로그 실값
    assert s_new == pytest.approx(503.56, abs=0.05)
    assert s_new - s_old == pytest.approx(23.41, abs=0.1)


def test_return_to_one_hop_lane_no_longer_jumps(lg):
    """복귀 틱(lane 4)의 +23.9 점프는 '잘못을 되돌린 것'이었다 — 이제 없다."""
    per = tracker(lg, 2)
    per._route_idx = IDX_SEC2
    before, _ = route_s(per, (2756, 0, 3), 26.54)
    after, f = route_s(per, (2756, 0, 4), 27.01)
    assert f.get('beside_route') and f.get('beside_hops') is None   # 한 칸
    assert 0.0 < after - before < 1.0, (before, after)              # 옛값 +23.87


def test_on_route_lane_is_untouched_by_the_switch(lg):
    """경로 차로 위에서는 두 설정이 완전히 같다."""
    for hops in (1, 2):
        per = tracker(lg, hops)
        per._route_idx = IDX_SEC2
        f = {}
        assert per._route_index((2756, 1, 5), f) == 2
        assert f == {}


def test_reset_drop_check_is_already_gated_by_off_route(lg):
    """
    임계 5 m 인데 −12.3 에서 리셋이 안 뜬 것은 우연이 아니다 —
    `on_route = not flags['off_route']` 가 막고 `_prev_route_s` 를 None 으로
    비운다. 그 가드가 지금도 코드에 있는지 고정한다.
    """
    src = (ROOT / 'vtd_adapter' / 'ego.py').read_text()
    assert "on_route = not flags.get('off_route')" in src
    assert 'self._prev_route_s = route_s if on_route else None' in src
