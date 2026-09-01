"""
적색점멸집중 테마 — 경로가 점멸 접근로(junction 60 / road 2312)를 반드시 경유한다.

이 맵의 점멸 신호는 controller 117 하나뿐이라, 다른 테마처럼 이벤트를 경로
아무 곳에나 배치하는 방식이 통하지 않는다. require: blink 가 출발 후보를 그
접근로 상류로 한정하고, _check_require 가 정지선 차로까지 검사한다.

lane 2 를 걸러내는 이유: 정지선 하나에 controller 116(빈 컨트롤러)과 117(점멸)이
함께 걸려 있고 9910 은 접근로당 하나만 준다. lane 2 는 116 이 잡힐 수 있어
점멸이 안 오면 "채점기가 틀렸나 신호가 안 왔나"를 가릴 수 없다.
"""
import xml.etree.ElementTree as ET

import pytest
import yaml

import gen_scenarios as gs     # noqa: E402 (conftest 가 tools 경로 추가)


def themes():
    return yaml.safe_load(open('configs/themes.yaml', encoding='utf-8'))


# ── blink controller 추출 (하드코딩 금지) ────────────────────────────────
def test_blink_ctrls_read_from_template_not_hardcoded():
    """TEMPLATE 에서 읽는다 — 지도·배포본이 바뀌면 따라간다."""
    got = gs.blink_ctrls()
    ls = ET.parse(gs.TEMPLATE).getroot().find('LightSigns')
    expect = {int(sc.get('Id')) for sc in ls.findall('SignalController')
              if any(p.get('Type') == 'blink' for p in sc.findall('Phase'))}
    assert got == expect and got, '템플릿에 blink 컨트롤러가 있어야 한다'


def test_blink_stopline_predicate_matches_only_that_approach():
    lg = gs.LaneGraph('data/lane_graph.pkl')
    hits = [k for k, v in lg.lanes.items()
            if v['junction'] == -1 and gs._blink_stopline(lg, v)]
    assert hits, '점멸 정지선을 가진 접근 차로가 있어야 한다'
    assert {k[0] for k in hits} == {2312}, '점멸 접근로는 road 2312 하나뿐이다'


# ── require: blink 배선 ──────────────────────────────────────────────────
def test_upstream_starts_limited_to_blink_approach():
    lg = gs.LaneGraph('data/lane_graph.pkl')
    st = gs._upstream_starts(lg, lambda v: gs._blink_stopline(lg, v))
    assert st, '출발 후보가 있어야 한다'
    assert {k[0] for k in st} == {2312}


def test_check_require_rejects_lane_two():
    """lane 2 는 116(빈 컨트롤러)이 잡힐 수 있어 통과시키지 않는다."""
    lg = gs.LaneGraph('data/lane_graph.pkl')
    assert 2 not in gs.BLINK_LANE_IDS
    lane2 = [(2312, 0, 2), (2370, 0, -1)]          # 접근차로 → 교차로 연결로
    lane3 = [(2312, 0, 3), (2370, 0, -2)]
    assert gs._check_require(lg, lane3, 'blink') is True
    assert gs._check_require(lg, lane2, 'blink') is False


def test_check_require_rejects_route_without_blink():
    lg = gs.LaneGraph('data/lane_graph.pkl')
    other = [(2192, 0, 3), (2192, 1, 3)]
    assert gs._check_require(lg, other, 'blink') is False


@pytest.mark.parametrize('req', ['school_zone', 'speed_change', 'signalized',
                                 'signalized_slow', 'signalized_fast', None])
def test_existing_require_values_untouched(req):
    """기존 require 값의 판정 경로에 blink 분기가 끼어들면 안 된다."""
    lg = gs.LaneGraph('data/lane_graph.pkl')
    chain = [(2312, 0, 3), (2370, 0, -2)]
    gs._check_require(lg, chain, req)              # 예외 없이 동작하면 된다


# ── 테마 정의 ────────────────────────────────────────────────────────────
def test_theme_registered():
    t = themes()
    assert t['routes']['적색점멸']['walk']['require'] == 'blink'
    cfg = t['themes']['적색점멸집중']
    assert cfg['routes'] == ['적색점멸']
    assert cfg['min_length_m'] == 400


def test_theme_excludes_signal_event():
    """signal 이 들어가면 set_signal(117) 이 blink 를 덮어쓰려다 GenError 로 죽는다."""
    cfg = themes()['themes']['적색점멸집중']
    assert 'signal' not in cfg['event'], 'signal 이벤트는 117 blink 와 충돌한다'


def test_theme_vary_axes_are_approach_conditions():
    """신호 조건이 고정이라 변이는 접근 조건에서 잡는다."""
    cfg = themes()['themes']['적색점멸집중']
    assert set(cfg['vary']) <= set(gs.AXIS_DEFAULTS), '축은 AXIS_DEFAULTS 에 있어야 한다'
    assert {'속도', '감속강도'} <= set(cfg['vary'])
    # 선행차 유무는 event 로 준다
    assert {'none', 'slow_lead', 'lead_brake'} == set(cfg['event'])


def test_set_signal_on_blink_controller_still_errors():
    """가드 회귀: 117 에 set_signal 을 걸면 여전히 GenError 여야 한다.
    (조용히 성공하면 blink 페이즈가 덮어써져 테마 목적이 사라진다)"""
    doc = gs.XmlDoc(gs.TEMPLATE)
    cid = sorted(gs.blink_ctrls())[0]
    with pytest.raises(gs.GenError):
        doc.set_signal(cid, 15.0, 3.0, 18.0)
