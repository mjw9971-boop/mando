"""batch_run 완주 판정을 채점과 같은 규칙(finish_s)으로.

명세 (2026-09-04):
  · 폴백 임계 total − end_margin_m 는 route_end.target_mode 가 'route_total' 이던
    시절 유도식이다. 지금 기본은 'finish' 라 계획 정지점이 finish_s +
    finish_clearance 이고, 경로 꼬리(route.finish_tail_m 12 m) 때문에 임계가
    계획보다 항상 위에 있다 — 11개 CSV 전부 계획대로 서면 미완주였다.
  · 새 임계는 score.detect_finish 와 **같은 함수**로 낸다 (규칙 두 벌 금지).
  · batch.finish_judge_use_finish_s=false 면 이전 동작.
"""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

import batch_run as BT                                          # noqa: E402
import build_route as BR                                        # noqa: E402
import score as SC                                              # noqa: E402
from team_code.kr_rules import plan_stop_s                       # noqa: E402
from vtd_adapter.config import end_margin_m, load_params_yaml    # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                      # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml()
MARGIN = end_margin_m(CFG)
CSVS = ['tests/fixtures/venue_20260903_waypoints.csv', 'data/official_route.csv',
        'data/test_route_waypoints.csv', 'tests/fixtures/waypoints.csv',
        'waypoints.csv'] + sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / 'scenarios').glob('*/*.csv'))


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


def _route(lg, path):
    rows = BR.read_waypoints_csv(str(ROOT / path))
    wps = [(r[1], r[2]) for r in rows]
    yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    return BR.build_route(lg, wps, 8.0, yaw,
                          junction_segs=BR.junction_segments(len(wps)),
                          seqs=[r[0] for r in rows],
                          finish_tail_m=BR.finish_tail_cfg())


def _cfg(use):
    c = dict(CFG)
    c['batch'] = dict(CFG['batch'])
    c['batch']['finish_judge_use_finish_s'] = use
    return c


def test_default_switch_is_on():
    assert CFG['batch'].get('finish_judge_use_finish_s') is True


@pytest.mark.parametrize('path', CSVS)
def test_planned_stop_is_inside_threshold(lg, path):
    """계획대로 정지하면 완주로 잡혀야 한다 — 옛 임계는 11/11 이 실패했다."""
    if not (ROOT / path).exists():
        pytest.skip(f'{path} 없음')
    rt = _route(lg, path)
    thr, basis = BT.finish_threshold(_cfg(True), lg, rt, MARGIN)
    assert basis == 'finish_xy'
    fs = SC.project_route_s(lg, rt, *rt['finish_xy'])
    stop_s, _clip = plan_stop_s(CFG, rt['total_length'], fs)
    rear = (stop_s - float(CFG['speed']['stop_gap_route_end_m'])
            - float(CFG['vehicle']['wheelbase'])
            - float(CFG['vehicle']['front_overhang_m']))
    assert rear > thr, f'{path}: 계획 뒷축 {rear:.2f} ≤ 임계 {thr:.2f}'
    # 여유는 finish_clearance_m 그대로여야 한다 (다른 상수가 새어들면 실패)
    assert rear - thr == pytest.approx(float(CFG['scoring']['finish_clearance_m']), abs=1e-6)


@pytest.mark.parametrize('path', CSVS[:5])
def test_same_threshold_as_score(lg, path):
    """score.detect_finish 와 비트 단위로 같은 임계 — 같은 함수를 쓰므로."""
    if not (ROOT / path).exists():
        pytest.skip(f'{path} 없음')
    rt = _route(lg, path)
    fxy = SC.resolve_finish_xy(CFG, rt)
    sc_thr = (SC.project_route_s(lg, rt, float(fxy[0]), float(fxy[1])) if fxy
              else rt['total_length'] - MARGIN)
    bt_thr, _basis = BT.finish_threshold(_cfg(True), lg, rt, MARGIN)
    assert bt_thr == sc_thr


def test_switch_off_is_previous_behaviour(lg):
    rt = _route(lg, 'tests/fixtures/venue_20260903_waypoints.csv')
    thr, basis = BT.finish_threshold(_cfg(False), lg, rt, MARGIN)
    assert basis.startswith('route_s_margin')
    assert thr == pytest.approx(rt['total_length'] - MARGIN)


def test_fallback_without_finish_xy(lg):
    """finish_xy 없는 옛 pkl 은 폴백 (하위호환)."""
    rt = _route(lg, 'data/official_route.csv')
    rt2 = dict(rt)
    rt2.pop('finish_xy')
    thr, basis = BT.finish_threshold(_cfg(True), lg, rt2, MARGIN)
    assert basis.startswith('route_s_margin')
    assert thr == pytest.approx(rt['total_length'] - MARGIN)


def test_fallback_without_lane_graph(lg):
    rt = _route(lg, 'data/official_route.csv')
    thr, basis = BT.finish_threshold(_cfg(True), None, rt, MARGIN)
    assert basis.startswith('route_s_margin')
    assert thr == pytest.approx(rt['total_length'] - MARGIN)


def _tick(t, route_s, v=0.0, v_target=0.0):
    return {'t': t, 'ego': {'route_s': route_s, 'speed': v},
            'decision': {'v_target': v_target, 'reasons': {'winner': 'none'}},
            'world': {'light': None}}


def test_endjudge_uses_thr():
    """thr 를 주면 그 값이 임계다. 안 주면 기존 total − margin (하위호환)."""
    j = BT.EndJudge(total=1000.0, margin=8.8, thr=900.0)
    assert j.feed(0.0, _tick(0.0, 899.0)) != '완주'
    assert j.feed(1.0, _tick(1.0, 900.0)) == '완주'
    j2 = BT.EndJudge(total=1000.0, margin=8.8)
    assert j2.feed(0.0, _tick(0.0, 990.0)) != '완주'
    assert j2.feed(1.0, _tick(1.0, 991.3)) == '완주'
