"""
교차로 짝 탐색이 0개일 때의 폴백 — 진출 풀 확장(제안②) · 폴백 흔적 기록(제안③) ·
ERROR 승격(제안①). 2026-09-06 G 분석.

배경: 짝 사이는 차선변경 금지라, 진출 경유점이 찍힌 **섹션**에 진입 차로에서
차선변경 없이 닿는 차로가 하나도 없으면 후보가 통째로 죽고 기존 탐욕으로
폴백한다. 폴백은 조용히 먼 길을 잡는다 (재현: 진입 [2152,2190] → 진출 [2012]
좌회전에서 정상 205 m 대신 2226 m). 게다가 그 경고는 stderr 로만 나가고
리포트·route pkl 어디에도 안 남아 사후 추적이 불가능했다.
"""
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

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml()

# 진입 [2152,2190] → 진출 [2012] 좌회전. 앞 구간까지 짝으로 두면(=앞 차선변경 금지)
# 후보 4개가 모두 죽는다: 회전 가능한 (2152,0,1)/(2190,0,-1) 은 앞에서 못 닿고,
# 닿는 (2152,0,2)/(2190,0,-2) 는 진출로 못 간다.
FB_WPS = [(1028.7, -620.2), (978.5, -560.2), (906.0, -537.0), (880.0, -549.0)]
FB_YAW = 2.2675
# 진출 섹션 제한만으로 죽는 경우 — (1928,1,-5) → 도로 1869 섹션 0
WIDEN_WPS = [(931.5, -867.0), (962.8, -906.7), (893.7, -1000.1), (858.8, -1034.5)]


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH), cfg=CFG)


# 이 파일은 짝 파이프라인의 계약을 검사한다 — 전역 DP(작업 R)가 켜지면 차로 열을
# DP 가 정해서 여기서 보려는 동작이 일어나지 않는다. 파일 단위로 DP 를 끈다.
@pytest.fixture(autouse=True)
def _dp_off(monkeypatch):
    monkeypatch.setattr(BR, '_DP_CFG', (False,) + tuple(BR.dp_cfg()[1:]))


@contextlib.contextmanager
def switches(widen, is_error):
    """pair_fallback_cfg 캐시를 직접 갈아끼운다 (params 파일을 안 건드린다)."""
    old = BR._PAIRFB_CFG
    BR._PAIRFB_CFG = (widen, is_error)
    try:
        yield
    finally:
        BR._PAIRFB_CFG = old


def build(lg, wps, yaw=None, jsegs=None):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rt = BR.build_route(lg, wps, 8.0, yaw,
                            junction_segs=(BR.junction_segments(len(wps))
                                           if jsegs is None else frozenset(jsegs)))
    return rt, err.getvalue()


def report_text(lg, rt):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = BR.report(lg, rt, 8.0)
    return rc, buf.getvalue()


def test_params_present_default_off():
    assert CFG['route']['pair_fallback_widen_enable'] is False
    assert CFG['route_check']['pair_fallback_is_error'] is False
    BR._PAIRFB_CFG = None
    assert BR.pair_fallback_cfg(reload=True) == (False, False)


def test_pool_all_sections_is_superset(lg):
    import math
    x, y = 906.0, -537.0
    yaw = math.atan2(y + 560.2, x - 978.5)
    cb = BR.candidates(lg, x, y, 8.0, yaw)
    narrow = {k for k, _s, _d in BR.road_lane_pool(lg, cb, x, y, yaw)}
    wide = {k for k, _s, _d in BR.road_lane_pool(lg, cb, x, y, yaw, all_sections=True)}
    assert narrow <= wide and len(wide) > len(narrow)
    assert {k[1] for k in narrow} == {4}          # 경유점 섹션만
    assert {k[1] for k in wide} > {4}             # 다른 섹션도 들어온다
    assert all(k[0] == 2012 and k[2] > 0 for k in wide)   # 같은 도로·같은 방향


def test_fallback_recorded_with_connectors(lg):
    with switches(False, False):
        rt, err = build(lg, FB_WPS, FB_YAW, jsegs={0, 1})
    assert '짝이 없다' in err
    fb = rt['pair_fallbacks']
    assert len(fb) == 1
    assert fb[0]['roads_in'] == [2152, 2190] and fb[0]['roads_out'] == [2012]
    assert fb[0]['kind'] == '좌회전'
    assert fb[0]['wi'] in set(rt['junction_segments'])     # 짝 구간 인덱스로 남는다
    assert fb[0]['connectors'], '폴백이 고른 연결로가 비었다'
    assert fb[0]['segment_m'] > 1000                       # 조용히 먼 길로 샜다
    assert all(len(k) == 3 and r > 0 for k, r in fb[0]['connectors'])


def test_report_silent_when_both_off(lg):
    with switches(False, False):
        rt, _ = build(lg, FB_WPS, FB_YAW, jsegs={0, 1})
        _rc, txt = report_text(lg, rt)
    assert '탐욕 폴백' not in txt


def test_report_flags_fallback_when_is_error(lg):
    with switches(False, True):
        rt, _ = build(lg, FB_WPS, FB_YAW, jsegs={0, 1})
        rc_on, txt_on = report_text(lg, rt)
    with switches(False, False):
        rt2, _ = build(lg, FB_WPS, FB_YAW, jsegs={0, 1})
        rc_off, _ = report_text(lg, rt2)
    assert '[오류] 회전 가능한 (진입,진출) 짝이 없어 탐욕 폴백' in txt_on
    assert 'R_min' in txt_on and '진입 도로 [2152, 2190]' in txt_on
    assert rc_on == rc_off + 1


def test_widen_rescues_section_limited_exit(lg):
    """진출 섹션에만 갇혀 실패하던 경로가 확장으로 살아난다."""
    with switches(False, False):
        with pytest.raises(BR.RouteError):
            build(lg, WIDEN_WPS)
    with switches(True, False):
        rt, err = build(lg, WIDEN_WPS)
    assert '넓혀 짝을 찾았다' in err
    assert not rt['pair_fallbacks']
    ok, _k_in, _k_out, _cost = BR.pair_turn_ok(lg, rt, rt['junction_segments'][0])
    assert ok is True


def test_widen_does_not_change_healthy_route(lg):
    """정상 경로는 확장 스위치로 바뀌지 않는다 (확장은 폴백 직전에만 돈다)."""
    wps = [(1028.7, -620.2), (978.5, -560.2), (906.0, -537.0), (880.0, -549.0)]
    with switches(False, False):
        a, _ = build(lg, wps, FB_YAW)
    with switches(True, False):
        b, _ = build(lg, wps, FB_YAW)
    assert a['lanes'] == b['lanes'] and a['total_length'] == b['total_length']
    assert not a['pair_fallbacks'] and not b['pair_fallbacks']


def test_left_turn_connector_r_min_above_threshold(lg):
    """G (a): 2152/2190 → 2012 좌회전 연결로는 (2207,0,-1), R_min 10.69 m.
    작업13 임계(최소회전반경×여유)를 넘어 금지 목록에 안 든다 — 그래서 [5] 게이트는
    조용하다. 폴백이 위험한 건 R_min 이 아니라 '엉뚱한 길'이라는 점이다."""
    banned, thr = BR.infeasible_connectors(lg)
    r = BR.lane_r_min(lg, (2207, 0, -1))
    assert abs(r - 10.69) < 0.05
    assert r > thr and (2207, 0, -1) not in banned
