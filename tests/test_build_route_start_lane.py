"""tools/build_route.py 출발 차로 확정 — locate() 개수 절단 우회.

명세 (2026-09-03):
  · lanegraph.locate 는 kd.query(k=16) 으로 후보를 뽑는다. candidates 가 갖고
    있던 것과 같은 절단 결함이라, 출발점이 중심선에서 벗어나면 조용히 옆 차로를
    고를 수 있다 — 에러 없이 경로 전체가 어긋나는 가장 위험한 실패 모드다.
  · lanegraph.py 는 제어기 런타임이 같이 쓰므로 고치지 않는다. build_route 는
    **locate 와 같은 점수 규칙을 반경 기반 후보집합에 적용**해 더 나은 차로가
    나오면 그쪽을 쓰고 경고한다.
  · 바꾸는 건 후보집합뿐이다 — locate 가 헤딩을 보고 일부러 조금 먼 차로를
    고른 판단은 살아 있어야 하고, candidates 만의 필터(폭 2 m 미만 포켓 배제)로
    빠진 차로 때문에 더 가까운 정답을 밀어내서도 안 된다.
"""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

import build_route as BR                                        # noqa: E402
from vtd_adapter.lanegraph import LaneGraph, LaneMatch, wrap    # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CSV = ROOT / 'tests' / 'fixtures' / 'waypoints.csv'
# 출발점에 후보 차로가 여럿인 CSV — 덮어쓰기 경로를 실제로 밟으려면 필요하다
MULTI_CSV = ROOT / 'data' / 'official_route.csv'
RADIUS = 8.0

# 2026-09-03 지도 전수 탐색에서 찾은, locate 가 **더 가까운** 차로를 고르는 지점.
# candidates 는 폭 2 m 미만 포켓을 배제하므로 후보집합에서 (3258,0,-3) 이 빠진다.
# 여기서 덮어쓰면 0.07 m 차로를 1.97 m 차로로 바꿔버린다 — 발동하면 안 된다.
LOCATE_BETTER = (971.2777, 665.8487, -2.673638)


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


def _args(path):
    if not path.exists():
        pytest.skip(f'{path.name} 없음')
    rows = BR.read_waypoints_csv(str(path))
    wps = [(r[1], r[2]) for r in rows]
    return wps, [r[0] for r in rows], math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])


@pytest.fixture(scope='module')
def route_args():
    return _args(CSV)


@pytest.fixture(scope='module')
def multi_args():
    return _args(MULTI_CSV)


def _build(lg, args):
    wps, seqs, yaw = args
    return BR.build_route(lg, wps, RADIUS, yaw, junction_segs=BR.junction_segments(len(wps)),
                          seqs=seqs, finish_tail_m=BR.finish_tail_cfg())


def test_locate_score_reproduces_lanegraph_ranking(lg):
    """locate_score 가 lanegraph.locate 의 규칙과 같은지 — 같은 후보집합이면 같은 답.

    이게 깨지면 build_route 가 locate 의 헤딩 판단을 통째로 덮어쓰기 시작한다.
    lanegraph.locate 의 점수식이 바뀌면 이 테스트가 먼저 운다.
    """
    import numpy as np
    rng = np.random.default_rng(3)
    pts = np.asarray(lg.kd_pts)
    checked = 0
    for _ in range(400):
        i = int(rng.integers(len(pts)))
        th, r = rng.uniform(0, 2 * math.pi), rng.uniform(0.5, 6.0)
        x, y = float(pts[i, 0]) + r * math.cos(th), float(pts[i, 1]) + r * math.sin(th)
        yaw = float(rng.uniform(-math.pi, math.pi))
        m = lg.locate(x, y, yaw, max_dist=RADIUS)
        if m is None:
            continue
        # locate 가 실제로 본 후보집합(k=16)을 그대로 재구성해 점수를 다시 매긴다
        d, ii = lg.kd.query((x, y), k=16)
        seen = {}
        for dist, j in zip(np.atleast_1d(d), np.atleast_1d(ii)):
            if not np.isfinite(dist) or dist > RADIUS:
                continue
            key = lg.lane_keys[lg.kd_lane[j]]
            if key in seen:
                continue
            s, _t, dd, _ = lg.project(key, x, y, idx_hint=int(lg.kd_i[j]))
            hd = float(np.interp(s, lg.lanes[key]['s'],
                                 np.unwrap(lg.lanes[key]['hdg'].astype(float))))
            if abs(wrap(yaw - hd)) > math.radians(70):
                continue
            seen[key] = (key, s, dd)
        if not seen:
            continue
        best = min(seen.values(), key=lambda c: BR.locate_score(lg, c[0], c[1], c[2], yaw))
        assert best[0] == m.lane, f'({x:.3f},{y:.3f}) yaw={yaw:.4f}'
        checked += 1
    assert checked > 100, '표본이 너무 적어 아무것도 안 지킨다'


def test_no_override_when_locate_is_closer(lg):
    """locate 가 더 나은 차로를 골랐으면 덮어쓰지 않는다 (폭 필터로 빠진 차로)."""
    x, y, yaw = LOCATE_BETTER
    m = lg.locate(x, y, yaw, max_dist=RADIUS)
    cand = BR.candidates(lg, x, y, RADIUS, yaw, ball=True)
    assert m is not None and cand
    assert cand[0][0] != m.lane, '전제가 깨졌다 — 이 지점은 불일치 케이스여야 한다'
    sc_m = BR.locate_score(lg, m.lane, m.s, m.dist, yaw)
    sc_b = min(BR.locate_score(lg, k, s, d, yaw) for k, s, d in cand)
    assert sc_b >= sc_m, '더 나쁜 차로로 덮어쓰려 하고 있다'


def test_override_fires_and_recovers(lg, multi_args, monkeypatch, capsys):
    """locate 가 엉뚱한 차로를 주면 경고하고 반경 후보로 복구한다.

    실제 절단이 나는 좌표는 드물어서(지도 랜덤 5,639 표본 중 4건, 점수 여유
    0.1 이하) locate 를 직접 가짜로 바꿔 경로만 검증한다.
    """
    monkeypatch.setattr(BR, '_CAND_CFG', (True, 5000))
    ref = _build(lg, multi_args)
    wps, _seqs, yaw0 = multi_args
    x0, y0 = wps[0]
    cand = BR.candidates(lg, x0, y0, RADIUS, yaw0, ball=True)
    good = min(cand, key=lambda c: BR.locate_score(lg, c[0], c[1], c[2], yaw0))
    worse = max(cand, key=lambda c: BR.locate_score(lg, c[0], c[1], c[2], yaw0))
    assert worse[0] != good[0], '후보가 하나뿐이면 이 테스트가 성립하지 않는다'

    monkeypatch.setattr(lg, 'locate', lambda *a, **kw: LaneMatch(
        worse[0], worse[1], 0.0, 0.0, worse[2], 0))
    got = _build(lg, multi_args)
    err = capsys.readouterr().err
    assert '출발 차로 불일치' in err
    assert str(worse[0]) in err and str(good[0]) in err
    assert got['lanes'] == ref['lanes'], '반경 후보로 복구되지 않았다'


def test_locate_none_falls_back_to_candidates(lg, route_args, monkeypatch, capsys):
    """locate 가 None 이면 기존 candidates()[:6] 폴백을 그대로 쓴다 (경고 없음)."""
    monkeypatch.setattr(BR, '_CAND_CFG', (True, 5000))
    ref = _build(lg, route_args)
    monkeypatch.setattr(lg, 'locate', lambda *a, **kw: None)
    got = _build(lg, route_args)
    assert '출발 차로 불일치' not in capsys.readouterr().err
    assert got['lanes'] == ref['lanes']


def test_real_csvs_do_not_trigger(lg, monkeypatch, capsys):
    """기준 CSV 들에서는 발동하지 않는다 — 이 우회가 기존 경로를 건드리지 않는다."""
    monkeypatch.setattr(BR, '_CAND_CFG', (True, 5000))
    csvs = [ROOT / 'waypoints.csv',
            ROOT / 'tests' / 'fixtures' / 'venue_20260903_waypoints.csv',
            ROOT / 'data' / 'official_route.csv',
            ROOT / 'data' / 'test_route_waypoints.csv', CSV]
    csvs = [c for c in csvs if c.exists()]
    if not csvs:
        pytest.skip('기준 CSV 없음')
    for c in csvs:
        rows = BR.read_waypoints_csv(str(c))
        wps = [(r[1], r[2]) for r in rows]
        yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
        BR.build_route(lg, wps, RADIUS, yaw, junction_segs=BR.junction_segments(len(wps)),
                       seqs=[r[0] for r in rows], finish_tail_m=BR.finish_tail_cfg())
        assert '출발 차로 불일치' not in capsys.readouterr().err, c.name
