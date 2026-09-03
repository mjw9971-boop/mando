"""tools/build_route.py candidates() — 반경 기반 후보 수집.

명세 (2026-09-03):
  · 개수 기반 kd.query(k=40) 은 경유점이 차로 중심선에 얹히면 그 차로 점들이
    k 를 다 채워 인접 차로·연결로를 후보에서 떨어뜨린다. 반경 안 점을 전부
    보면 몰림에 영향받지 않는다.
  · 대회장 CSV(waypoints.csv)는 k=40 으로 seq 8->9 가 "경로 없음" 이고,
    반경 기반이면 목표 차로 (836,0,-1) 로 빌드된다.
  · 킬 스위치 route.candidates_ball_query_enable=false 는 옛 동작 그대로다
    (현장 롤백 경로라 반드시 재현돼야 한다).
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
VENUE_CSV = ROOT / 'waypoints.csv'
RADIUS = 8.0
# 대회장 CSV seq 8 -> seq 9 (4번째 교차로 진입->진출). k=40 이 이 차로를
# 후보에서 떨어뜨려 실패했다 — 5.02 m 인데 k=40 도달이 4.84 m 였다.
VENUE_SEQ8_TARGET = (836, 0, -1)


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def venue():
    if not VENUE_CSV.exists():
        pytest.skip('waypoints.csv 없음')
    rows = BR.read_waypoints_csv(str(VENUE_CSV))
    return [(r[1], r[2]) for r in rows], [r[0] for r in rows]


def _build(lg, wps, seqs, **kw):
    yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    return BR.build_route(lg, wps, RADIUS, yaw, junction_segs=BR.junction_segments(len(wps)),
                          seqs=seqs, finish_tail_m=BR.finish_tail_cfg(), **kw)


def _set(monkeypatch, ball, max_points=5000):
    monkeypatch.setattr(BR, '_CAND_CFG', (ball, max_points))


def test_default_is_ball_query(monkeypatch):
    """params 기본값 = 반경 기반. 기본이 뒤집히면 대회장 CSV 가 다시 깨진다."""
    monkeypatch.setattr(BR, '_CAND_CFG', None)
    ball, max_points = BR.candidates_cfg(reload=True)
    assert ball is True
    assert max_points == 5000


def test_ball_superset_of_k40(lg, venue):
    """반경 기반 후보는 k=40 후보의 상위집합이고, 공통 항목은 (s, dist) 가 같다.

    같아야 하는 이유: k=40 은 반경 안을 '거리순 앞에서 자른' 것뿐이다. 값이
    갈리면 project(idx_hint) 에 들어가는 kd 점 순서가 어긋났다는 뜻이다.
    """
    wps, _ = venue
    grew = 0
    for wi, (x, y) in enumerate(wps):
        yaw = None if wi == 0 else math.atan2(y - wps[wi - 1][1], x - wps[wi - 1][0])
        b = {k: (s, d) for k, s, d in BR.candidates(lg, x, y, RADIUS, yaw, ball=True)}
        n = {k: (s, d) for k, s, d in BR.candidates(lg, x, y, RADIUS, yaw, ball=False)}
        assert set(n) <= set(b), f'경유점 {wi}: k=40 에만 있는 차로 {set(n) - set(b)}'
        for k in n:
            assert n[k] == pytest.approx(b[k], abs=1e-9), f'경유점 {wi} 차로 {k}'
        if len(b) > len(n):
            grew += 1
    assert grew >= 1, 'k=40 이 아무것도 안 자르면 이 테스트가 아무것도 안 지킨다'


def test_all_candidates_within_radius(lg, venue):
    """dist > radius 컷은 유지된다 — tier 3 상한이 radius 라는 전제가 여기 걸려 있다."""
    wps, _ = venue
    for x, y in wps:
        for _k, _s, d in BR.candidates(lg, x, y, RADIUS, None, ball=True):
            assert d <= RADIUS + 1e-9


def test_max_points_truncates(lg, venue, capsys):
    """캡을 넘기면 가까운 순으로 자르고 경고를 남긴다 (조용히 자르지 않는다)."""
    x, y = venue[0][8]                       # seq 9 — 반경 8 m 안 145점
    full = BR.candidates(lg, x, y, RADIUS, None, ball=True, max_points=5000)
    cut = BR.candidates(lg, x, y, RADIUS, None, ball=True, max_points=20)
    assert 'candidates_max_points' in capsys.readouterr().err
    assert 0 < len(cut) < len(full)
    # 잘린 뒤에도 가장 가까운 차로는 그대로여야 한다 (거리순 절단이므로)
    assert cut[0] == full[0]


def test_venue_csv_builds_with_ball(lg, venue, monkeypatch):
    """대회장 CSV 는 반경 기반에서 빌드되고, seq 8->9 목표 차로가 확정된다."""
    _set(monkeypatch, True)
    wps, seqs = venue
    rt = _build(lg, wps, seqs)
    target = {seqs[wi + 1]: rt['lanes'][i1] for wi, _i0, i1 in rt['segment_span']}
    assert target[9] == VENUE_SEQ8_TARGET


def test_venue_csv_fails_with_k40(lg, venue, monkeypatch):
    """킬 스위치 off = 옛 동작. 대회장 CSV 는 seq 8->9 에서 그대로 실패한다.

    현장 롤백 경로라 '옛 동작 그대로' 가 지켜지는지 확인한다.
    """
    _set(monkeypatch, False)
    wps, seqs = venue
    with pytest.raises(BR.RouteError) as e:
        _build(lg, wps, seqs)
    assert 'seq 8' in str(e.value) and 'seq 9' in str(e.value)


def test_centerline_csvs_unchanged(lg, monkeypatch):
    """중심선 샘플 CSV 는 반경 기반/개수 기반이 같은 경로를 낸다.

    이 CSV 들은 경유점이 목표 차로 중심선 위(거리 ~0)라 k=40 이 잘라낸 후보가
    전부 '안 쓸 옆 차로' 였다 — 그래서 옛 동작으로도 통과했다. 회귀 방지용.
    """
    csvs = [ROOT / 'data' / 'official_route.csv',
            ROOT / 'data' / 'test_route_waypoints.csv',
            ROOT / 'tests' / 'fixtures' / 'waypoints.csv']
    csvs = [c for c in csvs if c.exists()]
    if not csvs:
        pytest.skip('기준 CSV 없음')
    for c in csvs:
        rows = BR.read_waypoints_csv(str(c))
        wps = [(r[1], r[2]) for r in rows]
        seqs = [r[0] for r in rows]
        _set(monkeypatch, True)
        a = _build(lg, wps, seqs)
        _set(monkeypatch, False)
        b = _build(lg, wps, seqs)
        assert a['lanes'] == b['lanes'], c.name
        assert a['total_length'] == pytest.approx(b['total_length'], abs=1e-9), c.name
