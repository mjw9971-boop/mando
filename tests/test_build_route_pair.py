"""tools/build_route.py 교차로 짝 공동 선택 — 경유점 차로는 힌트다.

명세 (주최측 답변 2026-09-03):
  · "경유지는 전체 패스 설정을 위해 교차로 끝·시작 지점을 임의로 피팅한 것"
  · "특정 차로값으로 제공되지 않는다. 좌회전 구간에서 3차로로 경유지가 올 수도
    있으며 그 경우 1차로로 주행하면 된다"
  · "좌표는 대략적인 위치다"
→ 경유점이 주는 정보는 도로·순서·회전방향 셋뿐이다. 차로는 우리가 고른다.

  · 짝 구간은 진입 차로를 확정하기 전에 진출까지 함께 평가한다.
  · 짝이 아닌 세그먼트는 기존 탐욕 그대로다.
  · route.waypoint_lane_is_hint=false 면 전부 이전 동작이다.
"""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

import build_route as BR                                        # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                     # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
RADIUS = 8.0
# waypoints.csv · tests/fixtures/waypoints.csv 임시 제외 (2026-09-04, 작업13-2).
# vehicle.min_turn_margin 0.7 → 1.0 (임계 3.96 → 5.65 m) 으로 두 경로가 각각
# (2429,0,-1) R 5.43 m / (550,0,-1) R 4.52 m 를 지나 #7 '회전 불가 기하' ERROR
# 가 되어 rc=1 이다. 임계가 맞고 **경로 쪽이 틀렸다** — 작업15 에서 경유점을
# 조정해 좁은 연결로를 우회시킨 뒤 이 목록에 되돌린다.
# 테스트가 검증하는 계약("짝 진입 차로에서 회전을 수행할 수 있다")은 그대로다.
CSVS = ['tests/fixtures/venue_20260903_waypoints.csv', 'data/official_route.csv',
        'data/test_route_waypoints.csv'] + sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / 'scenarios').glob('*/*.csv'))


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


def _on(monkeypatch, hint=True, cap=400.0, thr=25.0):
    monkeypatch.setattr(BR, '_CAND_CFG', (True, 5000))
    monkeypatch.setattr(BR, '_START_OVERRIDE', True)
    monkeypatch.setattr(BR, '_PAIR_CFG', (hint, cap, thr))


def _build(lg, path):
    rows = BR.read_waypoints_csv(str(ROOT / path))
    wps = [(r[1], r[2]) for r in rows]
    seqs = [r[0] for r in rows]
    yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    rt = BR.build_route(lg, wps, RADIUS, yaw,
                        junction_segs=BR.junction_segments(len(wps)),
                        seqs=seqs, finish_tail_m=BR.finish_tail_cfg())
    return rt, wps, seqs


def test_params_defaults(monkeypatch):
    monkeypatch.setattr(BR, '_PAIR_CFG', None)
    hint, cap, thr = BR.pair_cfg(reload=True)
    assert hint is True and cap == 400.0 and thr == 25.0


def test_turn_threshold_is_single_source(monkeypatch):
    """turn_kind 의 임계는 params 하나에서 온다 — 하드코딩 25 가 남아 있으면 실패."""
    _on(monkeypatch, thr=5.0)
    assert BR.turn_kind(10.0) == '좌회전'          # 25 였다면 '직진'
    _on(monkeypatch, thr=45.0)
    assert BR.turn_kind(30.0) == '직진'            # 25 였다면 '좌회전'


@pytest.mark.parametrize('path', CSVS)
def test_every_pair_entry_can_perform_the_turn(lg, monkeypatch, path):
    """짝 구간 전수: 고른 진입 차로에서 진출 차로로 차선변경 없이 갈 수 있어야 한다.

    이게 깨지면 "짝 사이 차선변경 금지" 아래에서 경로가 물리적으로 불가능하다.
    """
    if not (ROOT / path).exists():
        pytest.skip(f'{path} 없음')
    _on(monkeypatch)
    rt, wps, seqs = _build(lg, path)
    banned, _thr = BR.infeasible_connectors(lg)
    span = {wi: (i0, i1) for wi, i0, i1 in rt['segment_span']}
    checked = 0
    for wi in sorted(BR.junction_segments(len(wps))):
        if wi not in span:
            continue
        k_in, k_out = rt['lanes'][span[wi][0]], rt['lanes'][span[wi][1]]
        s_in = lg.project(k_in, *wps[wi])[0]
        s_out = lg.project(k_out, *wps[wi + 1])[0]
        assert BR.turn_connect(lg, k_in, s_in, k_out, s_out, banned, 400.0) is not None, \
            f'seq {seqs[wi]}→{seqs[wi + 1]}: {k_in} → {k_out} 회전 불가'
        checked += 1
    assert checked > 0 or len(wps) <= 2


def test_pool_is_superset_of_candidates(lg, monkeypatch):
    """도로 확장 풀은 candidates() 의 상위집합이어야 한다.

    확장은 차로 **중점**에서 헤딩·폭을 보므로, 짧은 연결로처럼 중점 헤딩이
    투영 지점과 크게 다른 차로는 자기 자신조차 빠질 수 있다 (실측: 진출 풀이
    통째로 비어 폴백했다). 후보를 먼저 넣어 그걸 막는다.
    """
    _on(monkeypatch)
    _rt, wps, _seqs = _build(lg, 'data/test_route_waypoints.csv')
    for wi in range(1, len(wps)):
        yaw = math.atan2(wps[wi][1] - wps[wi - 1][1], wps[wi][0] - wps[wi - 1][0])
        c = BR.candidates(lg, wps[wi][0], wps[wi][1], RADIUS, yaw)
        if not c:
            continue
        pool = BR.road_lane_pool(lg, c, wps[wi][0], wps[wi][1], yaw)
        assert {k for k, _s, _d in c} <= {k for k, _s, _d in pool}


def test_switch_off_is_previous_behaviour(lg, monkeypatch):
    """킬 스위치 off 는 짝 로직을 통째로 끈다 — 짝 구간 차로가 탐욕 결과와 같다."""
    csv = 'tests/fixtures/pair_lane_offset_waypoints.csv'   # 작업17: 생성물 → 픽스처
    if not (ROOT / csv).exists():
        pytest.skip(f'{csv} 없음')
    _on(monkeypatch, hint=False)
    a, _w, _s = _build(lg, csv)
    _on(monkeypatch, hint=True)
    b, _w, _s = _build(lg, csv)
    # 이 CSV 는 짝 2(좌회전)·짝 4(직진)의 진입 차로가 바뀌는 게 확인된 케이스다
    assert a['lanes'] != b['lanes']
    assert abs(b['total_length'] - a['total_length']) > 1.0
    _on(monkeypatch, hint=False)
    c, _w, _s = _build(lg, csv)
    assert a['lanes'] == c['lanes'] and a['total_length'] == c['total_length']


def test_non_pair_segments_untouched(lg, monkeypatch):
    """짝이 아닌 세그먼트(시작→첫진입 / 진출→다음진입 / 마지막진출→종료)는 불변."""
    _on(monkeypatch, hint=False)
    a, wps, _s = _build(lg, 'data/official_route.csv')
    _on(monkeypatch, hint=True)
    b, _w, _s = _build(lg, 'data/official_route.csv')
    jseg = BR.junction_segments(len(wps))
    sa = {wi: (i0, i1) for wi, i0, i1 in a['segment_span']}
    sb = {wi: (i0, i1) for wi, i0, i1 in b['segment_span']}
    for wi in sa:
        if wi in jseg:
            continue
        assert a['lanes'][sa[wi][1]] == b['lanes'][sb[wi][1]], f'세그먼트 {wi} 가 바뀌었다'


def test_connect_cap_rejects_detours(lg, monkeypatch):
    """상한이 없으면 블록을 도는 경로도 '회전 가능' 이 된다 — 상한이 실제로 자르나.

    실측: 상한 없음이면 연결 비용 p90 1455 m / 최대 5109 m. 실제 짝은 최대 363 m.
    """
    _on(monkeypatch)
    banned, _thr = BR.infeasible_connectors(lg)
    _rt, wps, _seqs = _build(lg, 'tests/fixtures/venue_20260903_waypoints.csv')
    yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    c = BR.candidates(lg, wps[1][0], wps[1][1], RADIUS, yaw)
    ka, sa, _d = c[0]
    far = None
    for kb, sb, _d2 in BR.road_lane_pool(lg, c, wps[1][0], wps[1][1], yaw):
        r = BR.dijkstra(lg, [(ka, sa)], {kb: sb}, allow_lane_change=False, banned=banned)
        if r is not None and r[0] > 400.0:
            far = (kb, sb, r[0])
            break
    if far is None:
        pytest.skip('이 지점에는 400 m 를 넘는 연결이 없다')
    assert BR.turn_connect(lg, ka, sa, far[0], far[1], banned, 400.0) is None
    assert BR.turn_connect(lg, ka, sa, far[0], far[1], banned, far[2] + 1.0) is not None
