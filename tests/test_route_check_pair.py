"""tools/build_route.py 경유점 이탈 경고 기준 정정 (작업11).

명세 (2026-09-04):
  · 차로 반폭 기준은 폐기했다. 그건 "경유점이 찍힌 차로 = 주행 차로" 전제인데
    주최측 답변("좌표는 대략적", "좌회전에 3차로 경유지가 올 수 있다")으로
    무효가 됐다. 짝 공동 선택이 회전 가능한 차로를 고르면 경유점에서 한 차로
    (~3 m) 멀어지는 게 정상이고, 반폭(~1.5 m)으로 재면 정상 경로가 경고를 받아
    rc=1 → batch_run 이 시나리오를 폐기한다 (정적회피집중_01 실측).
  · 짝 경유점 → [1] 이탈 판정에서 제외, [2] 의 "회전 수행 가능" 으로 판정.
  · 그 밖 → route.check_waypoint_max_dist_m (도로 폭 급, 기본 6.0).
  · route_check.pair_waypoint_exempt_enable=false 면 이전 동작.
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
# 짝 진입 차로가 경유점에서 한 차로 벗어나는, 실측으로 확인된 CSV
PAIR_CSV = 'scenarios/정적회피집중/정적회피집중_01_좌회전2.csv'
CSVS = ['tests/fixtures/venue_20260903_waypoints.csv', 'data/official_route.csv',
        'tests/fixtures/waypoints.csv', 'waypoints.csv'] + sorted(
    str(p.relative_to(ROOT)) for p in (ROOT / 'scenarios').glob('*/*.csv'))


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


def _build(lg, path):
    rows = BR.read_waypoints_csv(str(ROOT / path))
    wps = [(r[1], r[2]) for r in rows]
    yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    return BR.build_route(lg, wps, 8.0, yaw,
                          junction_segs=BR.junction_segments(len(wps)),
                          seqs=[r[0] for r in rows],
                          finish_tail_m=BR.finish_tail_cfg())


def _patch(monkeypatch, **over):
    base = BR.route_check_cfg()
    monkeypatch.setattr(BR, 'route_check_cfg', lambda: {**base, **over})


def test_params_defaults():
    assert BR.route_check_cfg().get('pair_waypoint_exempt_enable') is True
    assert BR.route_cfg(reload=True).get('check_waypoint_max_dist_m') == 6.0


@pytest.mark.parametrize('path', CSVS)
def test_no_warnings(lg, capsys, path):
    """짝 경유점 면제 후 기준 CSV 는 경고 0 이어야 한다 (rc=0 = 배치가 안 버린다)."""
    if not (ROOT / path).exists():
        pytest.skip(f'{path} 없음')
    rt = _build(lg, path)
    assert BR.report(lg, rt, 8.0, None) == 0, capsys.readouterr().out


def test_pair_waypoints_exempted_and_turn_reported(lg, capsys):
    """짝 경유점은 이탈로 안 잡고, 대신 회전 가능 여부가 표에 찍힌다."""
    rt = _build(lg, PAIR_CSV)
    warns = BR.report(lg, rt, 8.0, None)
    out = capsys.readouterr().out
    assert warns == 0
    assert '짝 경유점 — [2] 회전 가능 판정' in out
    assert '회전 OK' in out
    # 면제된 그 경유점이 실제로 반폭을 넘는지 — 안 넘으면 이 테스트가 공허하다
    assert '이탈   2.86' in out or '이탈   3.05' in out


def test_switch_off_restores_halfwidth(lg, monkeypatch, capsys):
    """킬 스위치 off = 이전 동작. 같은 CSV 가 반폭 경고 2건을 다시 낸다."""
    _patch(monkeypatch, pair_waypoint_exempt_enable=False)
    rt = _build(lg, PAIR_CSV)
    assert BR.report(lg, rt, 8.0, None) == 2
    assert '차로 반폭' in capsys.readouterr().out


def test_warn_dev_still_overrides(lg, capsys):
    """--warn-dev 는 짝 포함 모든 경유점에 대한 고정 임계 override 로 남는다."""
    rt = _build(lg, PAIR_CSV)
    assert BR.report(lg, rt, 8.0, warn_dev=1.0) >= 2   # 2.86·3.05 가 1.0 을 넘는다
    assert '허용 1 m 고정' in capsys.readouterr().out


def test_non_pair_uses_max_dist(lg, monkeypatch, capsys):
    """짝이 아닌 경유점은 check_waypoint_max_dist_m 로 잰다."""
    rt = _build(lg, PAIR_CSV)
    monkeypatch.setattr(BR, '_ROUTE_CFG', {'check_waypoint_max_dist_m': 0.001})
    warns = BR.report(lg, rt, 8.0, None)
    out = capsys.readouterr().out
    assert '허용 0.001 m' in out
    # 시작·종료(짝 아님)는 이탈 0.00 이라도 임계 0.001 을 넘지 않는다 → 경고 0.
    # 임계가 실제로 쓰이는지는 헤더 문구로 확인한다.
    assert warns == 0


def test_turn_incapable_pair_warns(lg, capsys):
    """회전 불가 짝은 새 경고를 낸다 — 없으면 이 정정이 판정을 통째로 잃은 것이다.

    실제 CSV 에는 그런 짝이 없으므로(작업7-2 로 43/43 회전 가능) 경로의 진입
    차로를 옆 차로로 바꿔치기해 만든다. pkl 은 건드리지 않는다.
    """
    rt = _build(lg, PAIR_CSV)
    jseg = sorted(BR.junction_segments(len(rt['waypoints'])))
    spans = {wi: (i0, i1) for wi, i0, i1 in rt['segment_span']}
    wi = jseg[0]
    i0, i1 = spans[wi]
    k_in = rt['lanes'][i0]
    banned, _t = BR.infeasible_connectors(lg)
    cap = BR.pair_cfg()[1]
    # 같은 도로에서 진출 차로로 못 가는 이웃 차로를 찾는다
    bad = None
    for side in ('left', 'right'):
        nb = lg.neighbor(k_in, side)
        if nb is None:
            continue
        s_out = lg.project(rt['lanes'][i1], *rt['waypoints'][wi + 1])[0]
        if BR.turn_connect(lg, nb, 0.0, rt['lanes'][i1], s_out, banned, cap) is None:
            bad = nb
            break
    if bad is None:
        pytest.skip('이 짝에는 회전 불가 이웃 차로가 없다')
    rt2 = dict(rt)
    rt2['lanes'] = list(rt['lanes'])
    rt2['lanes'][i0] = bad
    assert BR.report(lg, rt2, 8.0, None) >= 1
    out = capsys.readouterr().out
    assert '회전 X' in out
    assert '차선변경 없이 갈 수 없다' in out


def test_pair_turn_ok_helper(lg):
    """pair_turn_ok 는 turn_connect 를 감싸기만 한다 (작업10 이 재사용)."""
    rt = _build(lg, PAIR_CSV)
    banned, _t = BR.infeasible_connectors(lg)
    for wi in sorted(BR.junction_segments(len(rt['waypoints']))):
        ok, k_in, k_out, cost = BR.pair_turn_ok(lg, rt, wi, banned)
        assert ok is True and cost is not None and cost >= 0.0
        assert k_in in rt['lanes'] and k_out in rt['lanes']
    assert BR.pair_turn_ok(lg, rt, 999, banned) == (None, None, None, None)
