"""홀수 경유점 짝 해석 자동 판정 (작업21).

주최 형식은 [시작, (진입,진출)×N, 종료] = 짝수지만 홀수로 올 수도 있다고
확인됐다. 홀수면 해석이 둘이다:
    offset 0 = [(진입,진출)×N, 종료]      첫 점이 곧 진입점
    offset 1 = [시작, (진입,진출)×N]       마지막 점이 곧 진출점
예전에는 홀수면 짝 해석을 통째로 건너뛰어, 짝 공동 선택도 교차로 내부
차선변경 금지도 안 걸렸다.

판정 신호는 **짝 구간이 junction 차로를 실제로 지나는 비율** 하나다.
짝수 기준 CSV 25개 실측(2026-09-04): 정답 offset 65/65 = 100 %,
오답 offset 4/86 = 5 %, margin 최소 0.50. 다른 후보(turn_connect 성공률,
|Δθ|)는 판별력이 없거나 약하다 — 근거는 params.yaml route.pair_auto_* 주석.
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
EVEN = ['data/official_route.csv', 'tests/fixtures/venue_20260903_waypoints.csv',
        'tests/fixtures/waypoints.csv', 'tests/fixtures/pair_lane_offset_waypoints.csv']
ODD_AMBIGUOUS = 'data/test_route_waypoints.csv'      # 17점 합성 경로 — 짝 구조 없음
ODD_CLEAR = 'data/test_waypoints.csv'                # 3점 — offset 1 만 성립


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


def _wps(path):
    rows = BR.read_waypoints_csv(str(ROOT / path))
    return [(r[1], r[2]) for r in rows], [r[0] for r in rows]


def _yaw(wps):
    j = next((k for k in range(1, len(wps))
              if math.hypot(wps[k][0] - wps[0][0], wps[k][1] - wps[0][1]) >= 2.0), 1)
    return math.atan2(wps[j][1] - wps[0][1], wps[j][0] - wps[0][0])


def _auto(lg, path):
    wps, seqs = _wps(path)
    return BR.pair_offset_auto(lg, wps, 8.0, _yaw(wps), seqs, BR.finish_tail_cfg())


def _build(lg, path, offset):
    wps, seqs = _wps(path)
    js = BR.junction_segments(len(wps), offset) if offset is not None else set()
    return BR.build_route(lg, wps, 8.0, _yaw(wps), junction_segs=js, seqs=seqs,
                          finish_tail_m=BR.finish_tail_cfg())


# ── junction_segments 확장 ────────────────────────────────────────────
def test_default_offset_is_unchanged():
    """기본 인자는 예전 동작 그대로 — 기존 호출부가 전부 이걸 쓴다."""
    for n in (4, 6, 8, 10, 17, 32):
        assert BR.junction_segments(n) == {wi for wi in range(1, n - 1, 2)}


def test_offset_zero_starts_at_first_point():
    assert sorted(BR.junction_segments(9, 0)) == [0, 2, 4, 6]
    assert sorted(BR.junction_segments(9, 1)) == [1, 3, 5, 7]


def test_thresholds_from_params():
    from vtd_adapter.config import load_params_yaml
    rc = load_params_yaml()['route']
    assert BR.pair_auto_cfg() == (pytest.approx(float(rc['pair_auto_min_ratio'])),
                                  pytest.approx(float(rc['pair_auto_min_margin'])))


# ── 신호 ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize('csv', EVEN)
def test_correct_offset_hits_every_junction(lg, csv):
    """정답 해석에서는 짝 구간이 전부 junction 을 지난다 (실측 65/65)."""
    if not (ROOT / csv).exists():
        pytest.skip(f'{csv} 없음')
    r, n, _ro, _no = BR.pair_junction_ratio(lg, _build(lg, csv, 1))
    assert n > 0 and r == pytest.approx(1.0)


@pytest.mark.parametrize('csv', EVEN)
def test_wrong_offset_collapses(lg, csv):
    """오답 해석은 확실히 탈락한다 — 빌드 실패 아니면 비율이 크게 떨어진다."""
    if not (ROOT / csv).exists():
        pytest.skip(f'{csv} 없음')
    try:
        rt = _build(lg, csv, 0)
    except BR.RouteError:
        return                                    # 빌드 실패도 탈락 신호다
    r, n, _ro, _no = BR.pair_junction_ratio(lg, rt)
    min_ratio, min_margin = BR.pair_auto_cfg()
    assert n > 0 and 1.0 - r >= min_margin, (csv, r)


# ── auto 판정 ─────────────────────────────────────────────────────────
@pytest.mark.parametrize('csv', EVEN)
def test_auto_keeps_even_behaviour(lg, csv):
    """짝수는 시험 빌드 없이 offset 1 — 기존 동작 유지."""
    if not (ROOT / csv).exists():
        pytest.skip(f'{csv} 없음')
    off, why, ev = _auto(lg, csv)
    assert off == 1 and ev == {} and '짝수' in why


def test_auto_picks_offset_for_clear_odd(lg):
    """3점 CSV: offset 0 은 빌드 실패, offset 1 은 junction 1/1 → offset 1."""
    if not (ROOT / ODD_CLEAR).exists():
        pytest.skip('없음')
    off, why, ev = _auto(lg, ODD_CLEAR)
    assert off == 1, why
    assert ev[0]['ok'] is False                   # 첫 점을 진입점으로 보면 끊긴다
    assert ev[1]['ratio'] == pytest.approx(1.0)


def test_auto_falls_back_when_ambiguous(lg):
    """17점 합성 경로: 0.75 vs 0.50 = margin 0.25 → 판정 불가 → 짝 해석 안 함."""
    if not (ROOT / ODD_AMBIGUOUS).exists():
        pytest.skip('없음')
    off, why, ev = _auto(lg, ODD_AMBIGUOUS)
    assert off is None and '판정 불가' in why
    assert abs(ev[0]['ratio'] - ev[1]['ratio']) < BR.pair_auto_cfg()[1]


def test_route_error_only_is_swallowed(lg, monkeypatch):
    """시험 빌드는 RouteError 만 삼킨다 — KeyboardInterrupt 는 그대로 올라간다."""
    wps, seqs = _wps(ODD_CLEAR)

    def boom(*_a, **_k):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        BR.pair_offset_auto(lg, wps, 8.0, _yaw(wps), seqs, 0.0, build_fn=boom)


# ── pkl 기록 · 리포트 ─────────────────────────────────────────────────
def test_pkl_records_interpretation(lg):
    wps, seqs = _wps('data/official_route.csv')
    rt = BR.build_route(lg, wps, 8.0, _yaw(wps),
                        junction_segs=BR.junction_segments(len(wps), 1), seqs=seqs,
                        pair_meta={'offset': 1, 'source': 'auto', 'why': '짝수 8점'})
    assert rt['pair_offset'] == 1
    assert rt['pair_offset_source'] == 'auto'
    assert '짝수' in rt['pair_offset_why']


def test_report_prints_interpretation(lg, capsys):
    """[2] 머리에 해석과 junction 비율이 찍힌다 — 당일 눈으로 볼 한 줄."""
    wps, seqs = _wps('data/official_route.csv')
    rt = BR.build_route(lg, wps, 8.0, _yaw(wps),
                        junction_segs=BR.junction_segments(len(wps), 1), seqs=seqs,
                        pair_meta={'offset': 1, 'source': 'auto', 'why': '짝수 8점'})
    BR.report(lg, rt, 8.0)
    out = capsys.readouterr().out
    assert '짝 해석: offset 1 (auto)' in out
    assert 'junction 경유: 짝' in out and '비짝' in out


def test_report_shouts_when_no_pairs(lg, capsys):
    """짝 해석 없음 = 교차로 내부 차선변경 금지가 안 걸린 상태. 크게 표시한다."""
    wps, seqs = _wps('data/official_route.csv')
    rt = BR.build_route(lg, wps, 8.0, _yaw(wps), junction_segs=set(), seqs=seqs,
                        pair_meta={'offset': None, 'source': 'forced', 'why': 'none'})
    BR.report(lg, rt, 8.0)
    out = capsys.readouterr().out
    assert '교차로 내부 차선변경 금지가 걸리지 않았다' in out
    assert '--pair-offset' in out
