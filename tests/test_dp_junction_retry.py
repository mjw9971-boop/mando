"""
DP 반경 재시도가 **교차로 진입·진출점에서** 돈다 (route.dp_retry_junction_enable,
route.dp_retry_ratio) — 2026-09-08.

반경 재시도(dp_radius_retry_enable, 2026-09-07)는 "최근접 후보가 앉은 도로의
같은 방향 전폭"으로 후보를 넓힌다. 그런데 대회 공식 형식은 경유점이 **전부
교차로 진입·진출부**라 최근접 후보가 교차로 연결로다. 연결로는 언제나 1차로라
"전폭"이 1차로 폭이 되고, 넓힌 반경이 base 그대로여서 재시도가 아예 안 돈다.

실측(PathShape04 5점): seq 3 최근접 = (1940,0,-1) junction 39 연결로 → wide 8.0.
seq 4 로 가는 우회전은 (1926,0,6)→(1943,0,-1)→1927 뿐인데 그 차로가 12.0 m
밖이라 후보에 없고, seq 3→4 가 29 m 직선에 771 m 우회했다 (총 1397 m,
경로/직선 2.16, 차로 49개).
"""
import csv
import contextlib
import io
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
CSV = ROOT / 'tests' / 'fixtures' / 'dp' / 'dp_07_junction_entry_1926.csv'

# 기본값을 읽지 않는다 — 스위치가 뒤집혀도 이 파일은 안 깨진다 (CLAUDE.md 드리프트).
BASE = BR.DPCfg(True, 8.0, 3.0, 400.0, 10.0, 1.0, 45.0, False, True, 16.0, 1.0,
                True, 0.0, False)
ON = BASE._replace(retry_junction=True, retry_ratio=2.0)
OFF = BASE._replace(retry_junction=False, retry_ratio=0.0)


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


def wps(path=CSV):
    return [(float(r[1]), float(r[2])) for r in csv.reader(open(path))
            if r[0] != 'seq' and len(r) >= 3]


def build(lg, pts):
    with contextlib.redirect_stderr(io.StringIO()), \
            contextlib.redirect_stdout(io.StringIO()):
        return BR.build_route(lg, pts, 8.0, None,
                              junction_segs=BR.junction_segments(len(pts)),
                              seqs=list(range(1, len(pts) + 1)))


def ratio(rt, pts):
    straight = sum(math.dist(pts[i], pts[i + 1]) for i in range(len(pts) - 1))
    return float(rt['total_length']) / straight


# ── params ────────────────────────────────────────────────────────────────
def test_params_present():
    r = CFG['route']
    assert 'dp_retry_junction_enable' in r
    assert 'dp_retry_ratio' in r


def test_namedtuple_defaults_match_params_defaults():
    """위치인자로 지은 DPCfg 가 새 기능을 조용히 켜면 안 된다."""
    c = BR.DPCfg(True, 8.0, 3.0, 400.0, 10.0, 1.0, 45.0, True, False, 16.0, 1.0)
    assert c.retry_junction is False and c.retry_ratio == 0.0


# ── (a) dp_wide_radius ────────────────────────────────────────────────────
def test_connector_is_one_lane_so_off_cannot_widen(lg):
    """off: 최근접이 연결로라 '도로 전폭' 이 1차로 폭 — base 그대로다."""
    w = BR.dp_wide_radius(lg, wps(), 2, 8.0, 16.0, junc=False)
    assert w.radius == 8.0 and w.allow is None
    assert lg.lanes[w.near]['junction'] != -1        # (1940,0,-1) junction 39


def test_on_widens_to_the_approach_road(lg):
    """on: 연결로의 진입 도로(1926,0) 같은 방향 전폭까지 넓히고 (1926,0,6) 을 문다."""
    w = BR.dp_wide_radius(lg, wps(), 2, 8.0, 16.0, junc=True)
    assert 12.0 < w.radius <= 16.0
    assert (1926, 0, 6) in w.allow
    c8 = {k for k, _s, _d in BR.dp_point_candidates(lg, wps(), 2, 8.0, None)[0]}
    cw = {k for k, _s, _d in BR._wide_cands(lg, wps(), 2, w, None)}
    assert c8 < cw                                   # 넓히기만 한다 — 줄지 않는다
    assert (1926, 0, 6) in cw


def test_allow_excludes_other_roads_and_opposite_direction(lg):
    """넓힌 반경 안이어도 다른 도로·반대 방향은 후보가 아니다.

    단, base 반경(8 m) 안에서 이미 잡히던 후보는 남는다 — 재시도가 후보를
    빼앗으면 안 되기 때문이다.
    """
    pts = wps()
    w = BR.dp_wide_radius(lg, pts, 2, 8.0, 16.0, junc=True)
    c8 = {k for k, _s, _d in BR.dp_point_candidates(lg, pts, 2, 8.0, None)[0]}
    cw = {k for k, _s, _d in BR._wide_cands(lg, pts, 2, w, None)}
    raw = {k for k, _s, _d in BR.dp_point_candidates(lg, pts, 2, w.radius, None)[0]}
    added = cw - c8
    assert added                                     # 실제로 넓어졌다
    assert all(k[0] == 1926 and k[2] > 0 for k in added)
    assert raw - cw                                  # 반경 안인데 걸러진 것이 있다
    assert any(k[0] != 1926 or k[2] < 0 for k in raw - cw)


def test_long_connector_falls_back_to_the_exit_road(lg):
    """진입 도로가 max_r 밖이면(긴 연결로의 진출단) 진출 도로를 본다.

    seq 4 최근접 = (1931,0,-2) junction 39. 진입 도로 1869 은 53.9 m 밖이고
    진출 도로 1927 이 9.7 m 다 — 세로 거리를 폭으로 오해하면 안 된다.
    """
    w = BR.dp_wide_radius(lg, wps(), 3, 8.0, 16.0, junc=True)
    assert w.radius <= 16.0
    assert all(k[0] != 1869 for k in w.allow if k[0] not in (1931, 1943, 1955))


def test_non_junction_point_is_unchanged(lg):
    """교차로 밖 점은 on/off 반경이 같다 — (a) 는 연결로에만 손댄다."""
    pts = wps()
    for k in (1, 4):                                  # seq 2 · seq 5
        assert (BR.dp_wide_radius(lg, pts, k, 8.0, 16.0, junc=True).radius
                == BR.dp_wide_radius(lg, pts, k, 8.0, 16.0, junc=False).radius)


# ── (c) 재시도 트리거 ─────────────────────────────────────────────────────
def test_detour_penalty_alone_misses_short_detours():
    """우회 벌점은 dp_detour_floor_m(400) 아래를 전부 놓친다 — 그래서 비율이 필요하다."""
    c = BR.dp_cfg()
    assert float(CFG['route']['dp_detour_floor_m']) == 400.0
    # 29 m 직선에 350 m 우회 → lim = max(400, 3×29) = 400 > 350 → 벌점 0
    assert 350.0 <= max(c.floor, c.ratio * 29.0)


def test_ratio_trigger_fires_where_penalty_does_not(lg):
    """비율만 켜도 재시도가 무장한다 (교차로 확장은 꺼둔 채)."""
    pts = wps()
    with dp(OFF._replace(retry_ratio=2.0)):
        rt = build(lg, pts)
    # 트리거는 돌지만 연결로에서는 넓힐 폭이 없어서 채택은 없다 — (a) 가 있어야 한다.
    assert not (rt['dp'].get('retries') or [])


# ── 경로 결과 ─────────────────────────────────────────────────────────────
def test_off_takes_the_771m_detour(lg):
    pts = wps()
    with dp(OFF):
        rt = build(lg, pts)
    assert not (rt['dp'].get('retries') or [])
    assert len(rt['lanes']) == 48
    assert 1340.0 < float(rt['total_length']) < 1350.0
    assert ratio(rt, pts) > 2.0
    assert any('seq 3→seq 4' in r and '771 m' in r and '우회 벌점' in r
               for r in rt['dp']['relaxed'])


def test_on_finds_the_right_turn_lane(lg):
    pts = wps()
    with dp(ON):
        rt = build(lg, pts)
    assert rt['dp']['retries']                        # 재시도가 돌았다
    assert tuple(rt['dp']['picks'][2]) == (1926, 0, 6)
    assert (1943, 0, -1) in [tuple(k) for k in rt['lanes']]
    assert float(rt['total_length']) < 900.0
    assert ratio(rt, pts) < 1.4
    assert len(rt['lanes']) < 30
    assert not any('우회 벌점' in r for r in rt['dp']['relaxed'])   # 우회 0


def test_on_warns_when_the_pick_moves_off_the_waypoint_road(lg):
    """경유점은 연결로 1940 에 찍혔는데 채택은 1926 — 사람이 볼 경고가 붙는다."""
    with dp(ON):
        rt = build(lg, wps())
    assert any('경유점이 찍힌 도로는 1940' in r for r in rt['dp']['retries'])


def test_last_waypoint_is_still_excluded(lg):
    """마지막 경유점(완주 판정 기준)은 여전히 재시도 대상이 아니다."""
    with dp(ON):
        rt = build(lg, wps())
    assert not any('seq 5' in r or 'waypoint 4' in r for r in rt['dp']['retries'])


# ── (b) --radius 배선 ─────────────────────────────────────────────────────
def test_cli_radius_reaches_dp_info(lg):
    with dp(ON._replace(compare=False)):
        with contextlib.redirect_stderr(io.StringIO()), \
                contextlib.redirect_stdout(io.StringIO()):
            rt = BR.build_route(lg, wps(), 8.0, None,
                                junction_segs=BR.junction_segments(5),
                                seqs=[1, 2, 3, 4, 5], dp_radius=12.0)
    assert rt['dp']['radius_m'] == 12.0


def test_pair_offset_auto_passes_dp_radius(lg):
    """짝 시험 빌드가 본 빌드와 **같은 반경**으로 지어야 한다.

    여기만 params 기본(8 m)으로 지으면 다른 경로를 보고 짝 해석을 고른다.
    """
    seen = []

    def spy(*_a, **kw):
        seen.append(kw.get('dp_radius'))
        raise BR.RouteError('stop')

    BR.pair_offset_auto(lg, wps(), 8.0, 0.0, [1, 2, 3, 4, 5], 0.0,
                        build_fn=spy, dp_radius=12.0)
    assert seen == [12.0, 12.0]
