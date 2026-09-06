"""
형식 무관 전역 경로 탐색 (route.global_dp_enable) — 작업 R, 2026-09-06.

빌더는 경유점이 (진입,진출) 짝이라고 전제하고 구간별 탐욕으로 차로를 정한다.
경유점마다 가장 가까운 차로를 확정하고 되돌리지 않으므로, 그 차로에서 다음
경유점으로 가는 길이 블록 한 바퀴여도 알 수 없다. 실측(PathShape03 CSV, 10점):
seq 9 에서 0.52 m 짜리 차로를 잡아 마지막 구간이 2081 m 가 됐다 — 2.42 m 짜리
옆 차로면 23 m 다.

DP 는 점마다 후보를 모두 살려 Viterbi 로 전체 최소를 찾는다. 형식(짝·홑점·
공유점·교차로당 1점)을 전제하지 않는다.
"""
import csv
import io
import contextlib
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import build_route as BR                                        # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                     # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml()
FIX = ROOT / 'tests' / 'fixtures' / 'dp'
DP_ON = (True, 8.0, 3.0, 400.0, 10.0)
DP_OFF = (False, 8.0, 3.0, 400.0, 10.0)


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH), cfg=CFG)


@contextlib.contextmanager
def dp(cfg):
    old = BR._DP_CFG
    BR._DP_CFG = cfg
    try:
        yield
    finally:
        BR._DP_CFG = old


def wps(path):
    return [(float(r[1]), float(r[2])) for r in csv.reader(open(path))
            if r[0] != 'seq' and len(r) >= 3]


def build(lg, pts):
    err = io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
        rt = BR.build_route(lg, pts, 8.0, None,
                            junction_segs=BR.junction_segments(len(pts)),
                            seqs=list(range(1, len(pts) + 1)))
    return rt


def max_wp_dev(lg, rt):
    lanes = [tuple(k) for k in rt['lanes']]
    return max(min(lg.project(k, x, y)[2] for k in lanes) for x, y in rt['waypoints'])


def test_params_present_default_off():
    r = CFG['route']
    assert r['global_dp_enable'] is False
    assert float(r['dp_match_radius_m']) == 8.0
    BR._DP_CFG = None
    assert BR.dp_cfg(reload=True) == (False, 8.0, 3.0, 400.0, 10.0)


def test_off_does_not_touch_route(lg):
    """스위치 off 면 DP 는 아예 안 돈다 (rt['dp'] 가 None)."""
    pts = wps(FIX / 'dp_03_one_per_junction.csv')
    with dp(DP_OFF):
        rt = build(lg, pts)
    assert rt['dp'] is None


def test_dijkstra_lc_in_junction_gate(lg):
    """교차로 차로 위 차선변경은 lc_in_junction=False 에서 막힌다 (지도에 70곳)."""
    k, nb = (1931, 0, -1), (1931, 0, -2)
    assert BR.has_broken(lg, k, 'right') and lg.neighbor(k, 'right') == nb
    L = lg.length(k)
    a = BR.dijkstra(lg, [(k, 0.0)], {nb: L * 0.5}, allow_lane_change=True)
    b = BR.dijkstra(lg, [(k, 0.0)], {nb: L * 0.5}, allow_lane_change=True,
                    lc_in_junction=False)
    assert a is not None
    assert b is None or b[0] > a[0] + 100        # 막히면 없거나 한참 돌아간다


def test_candidates_relax_heading_then_radius(lg):
    """후보 0개면 헤딩 완화 → 반경 1.5배 순으로 만든다."""
    pts = wps(FIX / 'dp_02_shared_point.csv')
    cands, notes = BR.dp_candidates(lg, pts, 8.0, None)
    assert all(c for c in cands), '모든 점에 후보가 생겨야 한다'
    assert any('헤딩 필터 완화' in n for n in notes)
    # 탐욕은 같은 점에서 헤딩 필터에 걸려 죽는다
    with dp(DP_OFF):
        with pytest.raises(BR.RouteError, match='진행방향이 맞는 차로 없음'):
            build(lg, pts)


def test_dp_not_longer_than_greedy_on_one_per_junction(lg):
    """교차로당 1점 형식에서 DP 가 탐욕보다 길어지지 않는다."""
    pts = wps(FIX / 'dp_03_one_per_junction.csv')
    with dp(DP_OFF):
        a = build(lg, pts)
    with dp(DP_ON):
        b = build(lg, pts)
    # 여유 1 m: 출발점 s 가 후보에 따라 몇 cm 달라진다 (경로 자체는 같다)
    assert b['total_length'] <= a['total_length'] + 1.0
    assert max_wp_dev(lg, b) < 8.0


@pytest.mark.parametrize('name', ['dp_01_drop_point', 'dp_02_shared_point',
                                  'dp_03_one_per_junction', 'dp_04_jitter3m',
                                  'dp_05_short_gap_2152'])
def test_form_variants_build_and_match(lg, name):
    """형식 5종 — 전부 지어지고, 경유점이 매칭 반경 안에 든다."""
    pts = wps(FIX / f'{name}.csv')
    with dp(DP_ON):
        rt = build(lg, pts)
    assert rt['dp'] and rt['dp']['used']
    assert max_wp_dev(lg, rt) < 8.0
    assert rt['total_length'] > 0


def test_form_variants_agree_with_each_other(lg):
    """같은 코스를 형식만 바꿔 준 셋은 같은 경로가 나와야 한다 (형식 무관)."""
    outs = []
    for name in ('dp_01_drop_point', 'dp_02_shared_point', 'dp_03_one_per_junction'):
        with dp(DP_ON):
            rt = build(lg, wps(FIX / f'{name}.csv'))
        outs.append([tuple(k) for k in rt['lanes']])
    assert outs[0] == outs[1] == outs[2]


def test_detour_is_penalised_not_blocked(lg):
    """우회는 막지 않고 값을 매긴다 — 벌점을 0 으로 두면 먼 길도 고를 수 있다."""
    pts = wps(FIX / 'dp_03_one_per_junction.csv')
    with dp(DP_ON):
        good = build(lg, pts)
    with dp((True, 8.0, 3.0, 400.0, 0.0)):
        free = build(lg, pts)
    assert good['total_length'] <= free['total_length'] + 1e-6
    assert free['dp']['used']
