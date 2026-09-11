"""소멸(테이퍼) 차로 — 탐색 벌점 · route_check WARN · 채점 인계 제외 (P7, 2026-09-06).

실측 배경: 실전주행_교통류_01_좌회전24 junction 7 우회전이 끝 폭 0.05 m 로 소멸하는
연결로 (1154,0,-2) 를 탔다. 제어기 taper_blend 가 successor 선 ±0.06 m 로 붙였지만
인계 첫 틱의 매칭 t_off 가 −1.34 m 로 튀어 차로유지 3건이 찍혔다.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

import build_route as BR                                        # noqa: E402
import score as SC                                              # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                     # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml()
VW = float(CFG['vehicle']['width'])
TAPER = (1154, 0, -2)          # 끝 폭 0.05, successor (126,0,-3)
SUCC = (126, 0, -3)


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH), cfg=CFG)


def test_params_present():
    # taper_penalty_enable 은 2026-09-09 (경로 파트 실측, 경로 34건 중 4건이
    # 끝 폭 0.00 m 차로를 피하고 대가는 +1.2 m) 로 **true 가 정본**이다.
    # 기본값을 읽어 "꺼져 있는지" 보는 검사는 그때 드리프트가 됐다 —
    # 2026-09-07 원칙대로 키의 **존재와 타입**만 보고, off 동작은 아래
    # _dij(penalty=0) 처럼 사본에서 명시적으로 꺼서 본다.
    assert isinstance(CFG['route']['taper_penalty_enable'], bool)
    assert float(CFG['route']['taper_penalty_m']) > 0
    assert float(CFG['scoring']['lane_departure_taper_handover_m']) == 0.0


def test_is_taper_lane_matches_scorer_criterion(lg):
    assert BR.is_taper_lane(lg, TAPER, VW)
    assert not BR.is_taper_lane(lg, SUCC, VW)
    assert not BR.is_taper_lane(lg, (1154, 0, -3), VW)       # 폭이 유지되는 옆 연결로


def _dij(lg, penalty):
    """(1777,3,-1)→(1818,0,-1): 소멸 연결로 (1821,0,-2) 직진 vs 옆 차로로 옮겨
    (1821,0,-3) 로 도는 대안. 점선 36.1 m 로 합법 도달 가능한 경우다."""
    BR._TAPER_CFG = (penalty > 0.0, penalty, VW)
    try:
        return BR.dijkstra(lg, [((1777, 3, -1), 0.0)], {(1818, 0, -1): 5.0})
    finally:
        BR._TAPER_CFG = None


def test_penalty_off_takes_the_taper_connector(lg):
    cost, path = _dij(lg, 0.0)
    assert (1821, 0, -2) in [k for k, _ in path]


def test_penalty_on_avoids_when_alternative_is_cheaper(lg):
    cost0, path0 = _dij(lg, 0.0)
    cost1, path1 = _dij(lg, 500.0)
    keys1 = [k for k, _ in path1]
    assert (1821, 0, -2) not in keys1, keys1
    assert (1821, 0, -3) in keys1
    # 벌점은 비용에만 얹힌다 — 경로 길이(진행거리) 자체는 변하지 않는다
    assert cost1 > cost0


def test_penalty_below_lc_cost_changes_nothing(lg):
    """LC_PENALTY(25)보다 작은 벌점은 차선변경을 이기지 못한다 — 문서화 의도."""
    _, path0 = _dij(lg, 0.0)
    _, path1 = _dij(lg, 10.0)
    assert [k for k, _ in path0] == [k for k, _ in path1]


# ── 채점: 인계 첫 틱 제외 ────────────────────────────────────────────────
def _tick(lane, s, t_off, rs, t):
    return {'t': t, 'ego': {'lane': list(lane), 's': s, 't_off': t_off, 'route_s': rs,
                            'x': 0.0, 'y': 0.0, 'speed': 8.0},
            'world': {'valid': True, 'flags': {}},
            'decision': {'state': 'none'}}


def test_handover_ticks_excluded_only_when_on(lg):
    """실측 재현: successor (126,0,-3) s=0.0 에서 t_off −1.34/−0.68, 그 뒤 −0.17."""
    ticks = [_tick(TAPER, 29.6, -1.42, 1005.6, 0.0),      # 소멸 차로 — 원래부터 제외
             _tick(SUCC, 0.0, -1.34, 1007.2, 0.1),
             _tick(SUCC, 0.0, -0.68, 1007.2, 0.15),
             _tick(SUCC, 1.7, -0.17, 1008.8, 0.2),
             _tick(SUCC, 3.3, -0.14, 1010.5, 0.3)]
    off = SC.detect_lane_departure(ticks, 0.0, lg, VW)
    on = SC.detect_lane_departure(ticks, 0.0, lg, VW, handover_m=1.0)
    assert len(off) == 1 and off[0]['lane'] == list(SUCC) and off[0]['max_t_off'] == -1.34
    assert on == []


def test_handover_does_not_excuse_a_real_departure(lg):
    """인계 거리 밖의 진짜 이탈은 그대로 잡힌다."""
    ticks = [_tick(SUCC, 5.0, -1.30, 1012.0, 0.0), _tick(SUCC, 6.0, -1.30, 1013.0, 0.1)]
    assert len(SC.detect_lane_departure(ticks, 0.0, lg, VW, handover_m=1.0)) == 1


def test_handover_applies_only_after_a_taper(lg):
    """소멸 차로 뒤가 아닌 차로의 s<1 m 는 제외하지 않는다."""
    plain = (1927, 3, -4)
    assert not any(BR.is_taper_lane(lg, k, VW) for k in lg.lanes if plain in lg.successors(k))
    ticks = [_tick(plain, 0.0, -1.30, 100.0, 0.0), _tick(plain, 0.2, -1.30, 100.2, 0.1)]
    assert len(SC.detect_lane_departure(ticks, 0.0, lg, VW, handover_m=1.0)) == 1
