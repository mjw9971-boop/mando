"""연속 차선변경(hop) 간격 검사 (작업19-2).

hop 은 평행 차로로 옮겨 타는 것이라 advance() 가 진행거리를 0 으로 둔다.
그래서 탐색 비용은 "차선변경 3회를 100 m 에 나눠 함"과 "29.8 m 에 몰아넣음"을
구분하지 못한다 — 대회장 배포 CSV(venue_20260903)에서 실제로 후자가 총비용
0.54 m(0.26 %) 차이로 뽑혔다. VtdRoutePlanner 램프는 hop 하나만 블렌드하므로
경로 점열에 3.402 m · 6.800 m 계단이 남는다. 물리적으로 주행 불가다.

창(window_s0/s1) 검사가 이걸 못 잡는 이유: lane_change_window 가 창 시작을
최대한 앞으로 당겨서, 같은 29.8 m 를 공유하는 hop 3 개가 각각 "29.8 m 창"
으로 보고된다. hop 지점 사이의 간격이 정확한 축이다.
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
VENUE = 'tests/fixtures/venue_20260903_waypoints.csv'
# hop 이 2개 이상이면서 정상인 기준 경로 — 임계에 걸리면 안 된다.
CLEAN = ['data/official_route.csv',
         'tests/fixtures/pair_lane_offset_waypoints.csv',
         'data/test_route_waypoints.csv']


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


def _build(lg, path):
    rows = BR.read_waypoints_csv(str(ROOT / path))
    wps = [(r[1], r[2]) for r in rows]
    yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    rt = BR.build_route(lg, wps, 8.0, yaw,
                        junction_segs=BR.junction_segments(len(wps)),
                        seqs=[r[0] for r in rows],
                        finish_tail_m=BR.finish_tail_cfg())
    return rt[0] if isinstance(rt, tuple) else rt


def test_threshold_comes_from_params():
    """임계는 params 가 단일 출처다 (상수 두 벌 금지)."""
    from vtd_adapter.config import load_params_yaml
    assert BR.min_hop_gap_m() == pytest.approx(
        float(load_params_yaml()['route']['min_hop_gap_m']))


def test_venue_has_zero_gap_hops(lg):
    """venue 경로는 같은 cum_s 에 hop 이 3 개 쌓여 간격 0 이 두 번 나온다."""
    rt = _build(lg, VENUE)
    gaps = BR.hop_gaps(lg, rt)
    zero = [g for g in gaps if g[2] < 1e-6]
    assert len(zero) == 2, gaps
    # 셋 다 도로 418 섹션 2 안이고, 그 차로 길이는 29.8 m 다 — 전이 3회에 60 m 필요.
    for _i, cum_i, _gap, fl, tl in zero:
        assert fl[0] == 418 and tl[0] == 418
        assert lg.length(fl) < 3 * BR.min_hop_gap_m()


def test_gap_is_route_distance_between_hops(lg):
    """간격은 앞 hop 지점부터 이 hop 지점까지의 경로 누적거리다."""
    rt = _build(lg, VENUE)
    lanes = [tuple(k) for k in rt['lanes']]
    cum = rt['cum_s']
    hops = [i for i in range(len(lanes) - 1)
            if BR.is_lane_change_hop(lg, lanes[i], lanes[i + 1])]
    gaps = BR.hop_gaps(lg, rt)
    assert len(gaps) == len(hops) - 1
    for (idx, cum_i, gap, _fl, _tl), a, b in zip(gaps, hops, hops[1:]):
        assert idx == b
        assert cum_i == pytest.approx(cum[b])
        assert gap == pytest.approx(cum[b] - cum[a])


def test_venue_report_errors(lg, capsys):
    """검사가 켜져 있으면 venue 는 ERROR — 배치가 이 경로를 버려야 한다."""
    rt = _build(lg, VENUE)
    errs = BR.report(lg, rt, 8.0)
    out = capsys.readouterr().out
    assert '앞 전이가 끝나기 전에 다음이 시작된다' in out
    assert errs >= 2


def test_kill_switch_disables_rc(lg, monkeypatch, capsys):
    """hop_gap_enable=false 면 표시는 하되 rc 에 반영하지 않는다 (이전 동작)."""
    rt = _build(lg, VENUE)
    base = BR.route_check_cfg()
    monkeypatch.setattr(BR, 'route_check_cfg',
                        lambda: dict(base, hop_gap_enable=False))
    errs_off = BR.report(lg, rt, 8.0)
    out = capsys.readouterr().out
    assert '검사 꺼짐' in out
    monkeypatch.setattr(BR, 'route_check_cfg', lambda: dict(base, hop_gap_enable=True))
    errs_on = BR.report(lg, rt, 8.0)
    capsys.readouterr()
    assert errs_on - errs_off == 2


@pytest.mark.parametrize('csv', CLEAN)
def test_clean_routes_have_no_hop_gap_error(lg, csv):
    """정상 기준 경로는 걸리지 않는다 (오탐 0)."""
    if not (ROOT / csv).exists():
        pytest.skip(f'{csv} 없음')
    rt = _build(lg, csv)
    thr = BR.min_hop_gap_m()
    bad = [g for g in BR.hop_gaps(lg, rt) if g[2] < thr]
    assert not bad, bad


def test_report_prints_gaps(lg, capsys):
    """리포트 [4] 에 hop 간격이 표시된다."""
    rt = _build(lg, VENUE)
    BR.report(lg, rt, 8.0)
    out = capsys.readouterr().out
    assert '연속 차선변경 간격' in out
    assert out.count('간격') >= 5
