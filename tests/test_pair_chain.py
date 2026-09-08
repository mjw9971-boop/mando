"""
연속 짝 공동 선택 — 2교차로 lookahead (route.pair_chain_enable). 2026-09-06 G-3.

앞 교차로 진출 차로를 고를 때 다음 교차로 진입 차로까지 본다. 왜: 교차로 사이
도로가 짧으면 진출 차로를 잘못 고른 뒤 바꿀 자리가 없다. 도로 2152 는 27.6 m 인데
다음 짝 진입 후보의 목표 s 가 27.57 이라, 차선변경 착지점(s + min_hop_gap_m 20)이
3 cm 차이로 목표를 지나쳐 회전 가능한 후보 둘이 통째로 탈락한다.

경로 59개 실측: 교차로 사이 도로 < 30 m 인 연속 짝 10건, 그중 앞 진출차로와 뒤
진입차로가 달라 차선변경이 필요한 것 1건.
"""
import csv
import io
import contextlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import build_route as BR                                        # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                     # noqa: E402
from conftest import banned_r_min, legacy_banned_r_min          # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml()
CSV_2152 = ROOT / 'tests' / 'fixtures' / 'pair_chain_2152_waypoints.csv'
# 짝 앞 구간까지 차선변경 금지로 강제한다. 공식 offset(0·1)에서는 이 배치가 안
# 나오지만, "진출 차로를 잘못 고르면 복구가 없다" 는 성질만 떼어 보기 위한 것이다.
FORCED_SEGS = frozenset({1, 2, 3})


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH), cfg=CFG)


# 이 파일은 짝 파이프라인의 계약을 검사한다 — 전역 DP(작업 R)가 켜지면 차로 열을
# DP 가 정해서 여기서 보려는 동작이 일어나지 않는다. 파일 단위로 DP 를 끈다.
@pytest.fixture(autouse=True)
def _dp_off(monkeypatch):
    monkeypatch.setattr(BR, '_DP_CFG', BR.dp_cfg()._replace(enable=False))


@contextlib.contextmanager
def chain(enable, gap=30.0):
    old = BR._PAIRCHAIN_CFG
    BR._PAIRCHAIN_CFG = (enable, gap)
    try:
        yield
    finally:
        BR._PAIRCHAIN_CFG = old


def wps(path=CSV_2152):
    return [(float(r[1]), float(r[2])) for r in csv.reader(open(path)) if r[0] != 'seq']


def build(lg, points, jsegs=None, yaw=None):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rt = BR.build_route(lg, points, 8.0, yaw,
                            junction_segs=(BR.junction_segments(len(points))
                                           if jsegs is None else jsegs))
    return rt, err.getvalue()


def test_params_present_default_off():
    assert CFG['route']['pair_chain_enable'] is False
    assert float(CFG['route']['pair_chain_gap_m']) == 30.0
    BR._PAIRCHAIN_CFG = None
    assert BR.pair_chain_cfg(reload=True) == (False, 30.0)


def test_off_falls_back_on_2152(lg):
    """짝 연쇄 off ⇒ 폴백이 220 m 자리에 먼 길을 고른다.

    금지 임계를 옛 값(5.65 m)으로 고정해 둔다 — 이 '먼 길' 의 길이는 어느
    연결로가 막혀 있느냐에 딸린 값이고, 여기서 보려는 계약은 "off 면 폴백이
    돌고 그 결과가 220 m 가 아니다" 다 (2026-09-08, docs/BACKLOG.md B-29).
    완화된 기본값(3.0)에서는 같은 폴백이 1934 m 를 고른다 — 여전히 먼 길이다.
    """
    with banned_r_min(legacy_banned_r_min()), chain(False):
        rt, err = build(lg, wps(), FORCED_SEGS)
    assert len(rt['pair_fallbacks']) == 1
    assert rt['pair_fallbacks'][0]['roads_in'] == [2152, 2190]
    assert rt['total_length'] > 2000                # 220 m 자리에 먼 길
    assert '짝이 없다' in err


def test_on_picks_exit_lane_that_reaches_next_entry(lg):
    with chain(True):
        rt, err = build(lg, wps(), FORCED_SEGS)
    assert not rt['pair_fallbacks']
    assert '다음 교차로까지 보고 진출 차로를' in err
    assert '(2152, 0, 1)' in err
    assert rt['total_length'] < 300
    lanes = [tuple(k) for k in rt['lanes']]
    assert (2152, 0, 1) in lanes and (2152, 0, 2) not in lanes
    for wi in rt['junction_segments']:
        if wi in {w for w, _a, _b in rt['segment_span']}:
            ok, _ki, _ko, _c = BR.pair_turn_ok(lg, rt, wi)
            assert ok is not False


def test_gap_threshold_gates_it(lg):
    """사이 도로가 문턱보다 길면 lookahead 를 안 한다 = off 와 같다."""
    with chain(False):
        a, _ = build(lg, wps(), FORCED_SEGS)
    with chain(True, gap=1.0):
        b, err = build(lg, wps(), FORCED_SEGS)
    assert '다음 교차로까지' not in err
    assert a['lanes'] == b['lanes'] and a['total_length'] == b['total_length']


def test_healthy_route_unchanged(lg):
    """정상 경로는 스위치로 바뀌지 않는다 (좁혀도 이긴 후보가 같다)."""
    p = ROOT / 'scenarios' / '실전주행_교통류' / '실전주행_교통류_01_좌회전24.csv'
    if not p.exists():
        pytest.skip('scenarios/ 생성물 없음')
    pts = wps(p)
    with chain(False):
        a, _ = build(lg, pts)
    with chain(True):
        b, _ = build(lg, pts)
    assert a['lanes'] == b['lanes']
    assert a['total_length'] == b['total_length']
    assert not a['pair_fallbacks'] and not b['pair_fallbacks']
