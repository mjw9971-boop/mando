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
DP_ON = BR.DPCfg(True, 8.0, 3.0, 400.0, 10.0, 1.0, 45.0, True, False, 16.0, 1.0)
DP_OFF = DP_ON._replace(enable=False)


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


def test_params_present():
    r = CFG['route']
    assert r['global_dp_enable'] is True          # 2026-09-06 검증 뒤 기본 채택
    assert r['dp_compare_enable'] is True
    assert float(r['dp_match_radius_m']) == 8.0
    BR._DP_CFG = None
    assert BR.dp_cfg(reload=True)[:5] == (True, 8.0, 3.0, 400.0, 10.0)


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
    with dp(DP_ON._replace(detour_penalty=0.0)):
        free = build(lg, pts)
    assert good['total_length'] <= free['total_length'] + 1e-6
    assert free['dp']['used']


# ── 넓은 도로 후보 반경 재시도 (route.dp_radius_retry_enable) ─────────────
# 지도에 같은 방향 3차로 이상인 도로가 78개, 그중 33개는 첫↔끝 차로 중심거리가
# 8 m 를 넘는다 (최대 15.84 m). 경유점이 한쪽 차로에 찍히면 회전 가능한 반대쪽
# 차로가 기본 반경 밖이라 DP 가 못 본다 (2026-09-07).
WIDE_CSV = FIX / 'dp_06_wide_road_1926.csv'
RETRY_ON = DP_ON._replace(retry=True, compare=False)
RETRY_OFF = DP_ON._replace(retry=False, compare=False)


def test_retry_params_present():
    r = CFG['route']
    assert r['dp_radius_retry_enable'] is True     # 2026-09-07 검증 뒤 기본 채택
    assert float(r['dp_radius_max_m']) == 16.0
    assert float(r['dp_radius_dev_penalty']) == 1.0


def test_wide_radius_covers_road_lanes(lg):
    """도로 1926 섹션 0 은 5차로·첫↔끝 12.87 m — 넓힌 반경이 그걸 덮는다."""
    pts = wps(WIDE_CSV)
    # 2026-09-11: 반환이 float → WideR(radius, allow, near) 로 바뀌었다
    # (연결로에서 접근 도로를 보려면 허용 차로 집합을 같이 돌려줘야 한다).
    r = BR.dp_wide_radius(lg, pts, 1, 8.0, 16.0).radius
    assert 12.87 < r <= 16.0
    c8, _ = BR.dp_point_candidates(lg, pts, 1, 8.0, None)
    cw, _ = BR.dp_point_candidates(lg, pts, 1, r, None)
    assert {k for k, _s, _d in c8} < {k for k, _s, _d in cw}
    assert (1926, 0, 6) in {k for k, _s, _d in cw}


def test_retry_off_takes_the_detour(lg):
    pts = wps(WIDE_CSV)
    with dp(RETRY_OFF):
        rt = build(lg, pts)
    assert not (rt['dp'].get('retries') or [])
    assert tuple(rt['dp']['picks'][1]) == (1926, 0, 4)      # 반경 8 안의 차로
    ok, _ki, _ko, _c = BR.pair_turn_ok(lg, rt, rt['junction_segments'][0])
    assert ok is False                                      # 그 차로에서는 회전이 안 된다


def test_retry_on_finds_the_far_lane(lg):
    pts = wps(WIDE_CSV)
    with dp(RETRY_ON):
        rt = build(lg, pts)
    rr = rt['dp'].get('retries') or []
    assert len(rr) == 1 and '반경 재시도' in rr[0]
    assert tuple(rt['dp']['picks'][1]) == (1926, 0, 6)      # 12.87 m 밖의 회전 차로
    assert rt['dp']['cost'] < 500                           # 우회(3060) 대신 짧은 길
    ok, _ki, _ko, _c = BR.pair_turn_ok(lg, rt, rt['junction_segments'][0])
    assert ok is True


def test_retry_never_moves_the_last_waypoint(lg):
    """마지막 경유점은 완주 판정 기준(finish_xy)이라 재시도 대상이 아니다.
    실측: venue 계열 6경로가 종점을 11.45 m 밖 차로로 옮겨 경유점 투영이 사라졌다."""
    p = ROOT / 'tests' / 'fixtures' / 'venue_20260903_waypoints.csv'
    if not p.exists():
        pytest.skip('venue 픽스처 없음')
    pts = wps(p)
    with dp(RETRY_ON):
        rt = build(lg, pts)
    assert not any('seq %d ' % len(pts) in r for r in (rt['dp'].get('retries') or []))
    assert rt['waypoint_s'][-1] is not None                 # 종점이 반경 안에 남는다
