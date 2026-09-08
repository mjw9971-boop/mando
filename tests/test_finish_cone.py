"""종료선 콘 게이트 (2026-09-07, 주최측 공지 2건).

공지: 종료 지점은 정지선 위 라바콘 2개로 표시되고 뒷축이 두 콘 사이 종료선을
지나면 완주다. 다만 **콘은 대략적 위치 표시이고 실제 판정은 운영측 제공 종료
지점 좌표 기준**이다. 그래서 이 저장소의 규칙은:

  · 완주 판정은 종전 그대로 finish_s (route.pkl 의 finish_xy 투영) — 불변
  · 콘은 시각 확인용 배치 + 채점 리포트의 **표시**일 뿐

이 파일이 지키는 것은 그 경계다: 콘이 어디에 서든 done 이 바뀌지 않는다.
"""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import finish_cone as fc                                        # noqa: E402
import score                                                    # noqa: E402
from conftest import mk_tick                                    # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
LANE = (1, 0, -1)
LEN = 200.0


class _Match:
    def __init__(self, lane, s, dist):
        self.lane, self.s, self.dist = lane, s, dist


class FakeLG:
    """+x 축을 따라 y=0 에 놓인 직선 차로 하나. 기하가 손으로 검산된다."""

    def __init__(self, width=3.0):
        self.width = width

    def project(self, key, x, y, idx_hint=None, tangent_ends=False):
        s = min(max(float(x), 0.0), LEN)
        t = float(y)                       # 좌(+y) 가 +t — 진행방향 +x 기준
        outside = not (0.0 <= x <= LEN)
        dist = abs(t) if (tangent_ends or not outside) else math.hypot(x - s, y)
        return s, t, dist, 0

    def point_at(self, key, s):
        return (min(max(float(s), 0.0), LEN), 0.0, 10.0, 0.0)    # hdg 0 = +x

    def width_at(self, key, s):
        return self.width

    def locate(self, x, y, **kw):
        return _Match(LANE, min(max(float(x), 0.0), LEN), abs(float(y)))


def route(finish_x=100.0, finish_y=0.0):
    return {'lanes': [LANE], 'cum_s': [0.0], 'lengths': [LEN],
            'total_length': LEN, 'finish_xy': [finish_x, finish_y]}


def cfg_with(**cone):
    cfg = load_params_yaml()
    cfg = dict(cfg)
    cfg['gen_placement'] = dict(cfg['gen_placement'], **cone)
    return cfg


# ── 배치 기하 ────────────────────────────────────────────────────────────
def test_cones_straddle_finish_coordinate():
    """콘 2개는 종료 좌표를 **정확히 가운데 두고** 차로 t 방향으로 선다."""
    g = fc.finish_gate(FakeLG(3.0), route(), cfg_with(cone_margin_m=0.3))
    L, R = g['left'], g['right']
    assert (L[0] + R[0]) / 2 == pytest.approx(100.0)     # 중점 = 종료 좌표
    assert (L[1] + R[1]) / 2 == pytest.approx(0.0)
    assert L[1] == pytest.approx(+1.8)                   # 3.0/2 + 0.3, 좌 +
    assert R[1] == pytest.approx(-1.8)
    assert L[0] == pytest.approx(R[0])                   # 종방향으로는 같은 자리


def test_cone_offset_is_half_lane_plus_margin():
    for w, m in ((3.0, 0.3), (2.6, 0.5), (5.5, 0.3)):
        g = fc.finish_gate(FakeLG(w), route(), cfg_with(cone_margin_m=m))
        assert g['half_gate'] == pytest.approx(w / 2 + m)
        assert g['left'][1] == pytest.approx(w / 2 + m)


def test_cones_sit_outside_the_lane_edge():
    """콘은 **차로 경계 밖**이다 — 경계(반폭)보다 margin 만큼 더 나가야 한다."""
    w, m = 3.0, 0.3
    g = fc.finish_gate(FakeLG(w), route(), cfg_with(cone_margin_m=m))
    assert abs(g['left'][1]) - w / 2 == pytest.approx(m)
    assert abs(g['right'][1]) - w / 2 == pytest.approx(m)


def test_model_and_switch_come_from_params():
    g = fc.finish_gate(FakeLG(), route(), cfg_with(cone_model='X01'))
    assert g['model'] == 'X01'
    assert fc.cone_cfg(cfg_with(finish_cone_enable=False))['enable'] is False


def test_default_model_is_the_confirmed_pylon():
    """카탈로그에서 확인한 주황 라바콘 (2026-09-07 사용자 확인)."""
    assert load_params_yaml()['gen_placement']['cone_model'] == 'RdMiscPylon03-32cm'


# ── 회랑 여유 — kr_rules 와 같은 식 ──────────────────────────────────────
def test_reach_matches_kr_rules_formula():
    """reach = 자차반폭 + 객체반폭 + percep.obstacle_clearance_m (값 두 벌 금지)."""
    cfg = cfg_with()
    got = fc.corridor_reach(cfg, 0.16)
    want = cfg['vehicle']['width'] / 2 + 0.16 + cfg['percep']['obstacle_clearance_m']
    assert got == pytest.approx(want)


def test_clearance_is_lat_minus_reach():
    cfg = cfg_with(cone_margin_m=0.3, cone_half_width_m=0.16)
    g = fc.finish_gate(FakeLG(3.0), route(), cfg)
    assert g['clear_left'] == pytest.approx(abs(g['lat_left']) - g['reach'])
    assert g['clear_min'] == pytest.approx(min(abs(g['lat_left']),
                                               abs(g['lat_right'])) - g['reach'])


def test_wider_margin_helps_only_when_finish_is_lane_centred():
    """margin 을 올리면 **중앙에 찍힌** 종료선에서는 여유가 그만큼 는다."""
    a = fc.finish_gate(FakeLG(2.4), route(), cfg_with(cone_margin_m=0.3))
    b = fc.finish_gate(FakeLG(2.4), route(), cfg_with(cone_margin_m=0.7))
    assert b['clear_min'] - a['clear_min'] == pytest.approx(0.4)


def test_wider_margin_hurts_when_finish_is_one_lane_over():
    """경유점이 옆 차로에 찍히면 margin 을 올릴수록 안쪽 콘이 회랑으로 들어온다.

    실측 2026-09-07 실전주행_교통류_02_직진11: 종료 좌표가 경로 차로 (2815,3,-3)
    가 아니라 (2815,3,-2) 에 있어 t +2.90 m — margin 0.3→1.0 에서 여유가
    −0.32 → −1.02 로 **악화**했다. margin 튜닝이 해법이 아니라는 근거다.
    """
    r = route(finish_y=2.9)                       # 종료 좌표가 경로에서 좌로 2.9 m
    a = fc.finish_gate(FakeLG(3.0), r, cfg_with(cone_margin_m=0.3,
                                                cone_at_road_edge=False))
    b = fc.finish_gate(FakeLG(3.0), r, cfg_with(cone_margin_m=1.0,
                                                cone_at_road_edge=False))
    assert a['clear_min'] < 0.0 and b['clear_min'] < a['clear_min']
    # 그래서 margin 이 아니라 **기준점**을 바꿨다 (cone_at_road_edge, 2026-09-08):
    # 차로 중심선 기준이면 종료 좌표가 얼마나 치우쳐 있든 콘은 경계에 선다.
    c = fc.finish_gate(FakeLG(3.0), r, cfg_with(cone_margin_m=0.3))
    assert c['clear_min'] > 0.0


# ── 테이퍼 차로 게이트 바닥 ──────────────────────────────────────────────
def test_gate_never_narrower_than_the_car():
    """폭 0.75 m 테이퍼 차로에서도 콘 사이로 차가 지나갈 수 있어야 한다.

    실측 2026-09-07 실전주행_교통류_16_직진20: 배치 차로 (2076,2,-4) 폭 0.75 m
    → 받침 없이는 콘 간격 0.27 m 로 차(1.886 m)가 물리적으로 못 지나간다.
    """
    cfg = cfg_with(cone_margin_m=0.3)
    g = fc.finish_gate(FakeLG(0.75), route(), cfg)
    lo, hi = fc.gate_span(g)
    assert g['gate_floored'] is True
    assert hi - lo >= cfg['vehicle']['width']
    g_legacy = fc.finish_gate(FakeLG(0.75), route(),
                              cfg_with(cone_margin_m=0.3, cone_at_road_edge=False))
    assert g_legacy['gate_floored'] is True             # 이전 경로에서도 받친다


def test_gate_floor_is_off_by_switch():
    g = fc.finish_gate(FakeLG(0.75), route(),
                       cfg_with(cone_margin_m=0.3, cone_min_gate_from_vehicle=False))
    assert g['gate_floored'] is False
    assert g['half_gate'] == pytest.approx(0.75 / 2 + 0.3)      # 이전 동작 재현


def test_gate_floor_does_not_touch_normal_lanes():
    g = fc.finish_gate(FakeLG(3.0), route(), cfg_with(cone_margin_m=0.3))
    assert g['gate_floored'] is False
    assert g['half_gate'] == pytest.approx(1.8)


# ── 콘 사이 통과 표시 ────────────────────────────────────────────────────
def test_between_cones_uses_inner_faces():
    g = fc.finish_gate(FakeLG(3.0), route(), cfg_with(cone_margin_m=0.3,
                                                      cone_half_width_m=0.16))
    lo, hi = fc.gate_span(g)
    assert (lo, hi) == pytest.approx((-1.64, 1.64))   # ±(1.8 − 0.16)
    assert fc.between_cones(g, 0.0)
    assert fc.between_cones(g, 1.6)
    assert not fc.between_cones(g, 1.7)
    assert not fc.between_cones(g, -1.7)


def test_between_cones_follows_the_finish_coordinate_when_legacy():
    """이전 동작(cone_at_road_edge=false): 콘이 종료 좌표를 중심으로 선다.

    **기본값을 읽지 않고 사본에서 명시적으로 끈다** — 기본이 바뀌어도 이 경로의
    커버리지를 잃지 않기 위해서다 (CLAUDE.md 2026-09-07 드리프트 원칙).
    """
    g = fc.finish_gate(FakeLG(3.0), route(finish_y=2.9),
                       cfg_with(cone_margin_m=0.3, cone_at_road_edge=False))
    assert fc.between_cones(g, 2.9)
    assert not fc.between_cones(g, 0.0)


def test_road_edge_places_cones_at_the_lane_edge_not_around_the_finish_point():
    """기본(도로 끝): 종료 좌표가 치우쳐 있어도 콘은 차로 경계 + margin 에 선다.

    실측 실경로_01_PathShape03: 폭 2.4 m 연결로에서 종료 좌표가 중심선에서
    0.406 m 치우쳐 있었고, 종료 좌표 기준 대칭 배치는 오른쪽 콘을 회랑 안
    (여유 −0.309 m)에 놓아 자차가 종료선 17 m 앞에서 영구 정지했다(미완주).
    """
    off = fc.finish_gate(FakeLG(2.4), route(finish_y=0.406),
                         cfg_with(cone_margin_m=0.3, cone_at_road_edge=False))
    on = fc.finish_gate(FakeLG(2.4), route(finish_y=0.406),
                        cfg_with(cone_margin_m=0.3))
    # 이전: 종료 좌표 기준 대칭 → 한쪽이 회랑 안
    assert off['lat_left'] == pytest.approx(0.406 + 1.5)
    assert off['lat_right'] == pytest.approx(0.406 - 1.5)
    assert off['clear_min'] < 0
    # 지금: 차로 중심선 기준 ±(반폭 + margin)
    assert on['lat_left'] == pytest.approx(1.5)
    assert on['lat_right'] == pytest.approx(-1.5)
    assert on['clear_min'] > 0


def test_road_edge_spans_all_same_direction_lanes():
    """편도 다차로면 **바깥 차로 경계**까지 나간다 — 도로 한가운데 세우지 않는다."""
    class MultiLG(FakeLG):
        """배치 차로 왼쪽에 3.0 m 차로가 하나 더 있는 편도 2차로."""

        def neighbor(self, key, side):
            return (1, 0, -2) if (side == 'left' and key == LANE) else None

    g = fc.finish_gate(MultiLG(3.0), route(), cfg_with(cone_margin_m=0.3))
    assert g['lat_left'] == pytest.approx(1.5 + 3.0 + 0.3)  # 반폭 + 이웃 폭 + margin
    assert g['lat_right'] == pytest.approx(-1.8)         # 오른쪽은 이웃이 없다


# ── 판정 불변 (이 작업의 핵심 경계) ──────────────────────────────────────
def _ticks(peak_s, lat=0.0, n=40):
    return [mk_tick(t=i * 0.05, x=i * (peak_s / (n - 1)), y=lat, speed=10.0,
                    route_s=i * (peak_s / (n - 1))) for i in range(n)]


def _run(ticks, lat_cone_margin=0.3, finish_s=100.0):
    lg, cfg = FakeLG(3.0), cfg_with(cone_margin_m=lat_cone_margin)
    fin = score.detect_finish(ticks, 0.0, route(), cfg, finish_s)
    rep = score.finish_gate_report(ticks, 0.0, lg, route(), cfg, finish_s, fin)
    return fin, rep


def test_verdict_is_independent_of_cone_params():
    """margin 을 어떻게 흔들어도 done/완주시간은 한 글자도 안 바뀐다."""
    ticks = _ticks(150.0, lat=1.2)
    base = score.detect_finish(ticks, 0.0, route(), cfg_with(), 100.0)['summary']
    for m in (0.0, 0.3, 1.0, 5.0):
        got = score.detect_finish(ticks, 0.0, route(), cfg_with(cone_margin_m=m),
                                  100.0)['summary']
        assert got == base


def test_outside_the_cones_is_still_a_finish():
    """'콘 밖' 은 표시일 뿐 — done 은 True 로 남는다 (판정은 finish_s 기준)."""
    fin, rep = _run(_ticks(150.0, lat=3.0))
    assert fin['summary']['done'] is True
    assert rep['between'] is False


def test_between_reported_when_inside():
    fin, rep = _run(_ticks(150.0, lat=0.4))
    assert fin['summary']['done'] is True and rep['between'] is True
    assert rep['lat'] == pytest.approx(0.4, abs=1e-6)


def test_crossing_report_has_time_route_s_and_margin():
    ticks = _ticks(150.0)
    fin, rep = _run(ticks)
    hit = fin['hit_i']
    assert rep['t_s'] == pytest.approx(round(ticks[hit]['t'], 1))
    assert rep['route_s'] == pytest.approx(round(ticks[hit]['ego']['route_s'], 1))
    assert rep['margin_m'] == pytest.approx(
        round(ticks[hit]['ego']['route_s'] - 100.0, 2))
    assert rep['margin_m'] >= 0.0
    assert rep['remain_m'] is None


def test_not_finished_reports_remaining_distance():
    fin, rep = _run(_ticks(60.0))
    assert fin['summary']['done'] is False
    assert rep['route_s'] is None and rep['between'] is None
    assert rep['remain_m'] == pytest.approx(100.0 - 60.0, abs=0.2)


def test_gate_report_is_none_without_route_or_finish_s():
    ticks = _ticks(150.0)
    fin = score.detect_finish(ticks, 0.0, route(), cfg_with(), 100.0)
    assert score.finish_gate_report(ticks, 0.0, FakeLG(), route(), cfg_with(),
                                    None, fin) is None
    assert score.finish_gate_report(ticks, 0.0, None, route(), cfg_with(),
                                    100.0, fin) is None


# ── 생성기 산출물 ────────────────────────────────────────────────────────
@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_generated_xml_has_two_cones_of_the_configured_model():
    import xml.etree.ElementTree as ET

    import gen_scenarios as gs
    from vtd_adapter.lanegraph import LaneGraph
    lg = LaneGraph(str(GRAPH))
    route_defs, _themes, gen_cfg = gs.load_themes()
    pool = gs.RoutePool(lg, route_defs, 3, gen_cfg)
    rt = pool.get('직진', 0, '콘검사', min_length_m=300.0)
    xml, sdef, _bad = gs.build_scenario(lg, gs.junction_ctrl_map(lg), rt, [],
                                        {}, '콘검사', '0:콘:검사')
    objs = [o for o in ET.fromstring(xml).iter('Object')
            if (o.get('Name') or '').startswith('FinishCone')]
    assert len(objs) == 2
    model = gs.plc_cfg()['cone_model']
    assert {o.get('Definition') for o in objs} == {model}
    # 정의(yaml)에 좌표가 남아 재생성 없이 대조할 수 있어야 한다
    assert sdef['finish_cone']['model'] == model
    L, R = sdef['finish_cone']['left'], sdef['finish_cone']['right']
    lat = sdef['finish_cone']['lat']
    # 도로 끝 규약: 두 콘은 배치 차로 **중심선**을 사이에 두고 서고, 좌우 거리는
    # 각 방향 바깥 경계까지라 대칭이 아닐 수 있다. 종료 좌표 중점이 아니다.
    assert lat[0] > 0 > lat[1]
    assert L[2] == pytest.approx(R[2], abs=1e-6)        # 같은 높이
    assert sdef['finish_cone']['clear_m'] == pytest.approx(
        [abs(lat[0]) - sdef['finish_cone']['reach_m'],
         abs(lat[1]) - sdef['finish_cone']['reach_m']], abs=1e-6)


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_switch_off_reproduces_cone_free_xml(monkeypatch):
    import xml.etree.ElementTree as ET

    import gen_scenarios as gs
    from vtd_adapter.lanegraph import LaneGraph
    monkeypatch.setattr(gs, '_PLC_CFG', dict(gs.plc_cfg(), finish_cone_enable=False))
    lg = LaneGraph(str(GRAPH))
    route_defs, _themes, gen_cfg = gs.load_themes()
    pool = gs.RoutePool(lg, route_defs, 3, gen_cfg)
    rt = pool.get('직진', 0, '콘검사', min_length_m=300.0)
    xml, sdef, _bad = gs.build_scenario(lg, gs.junction_ctrl_map(lg), rt, [],
                                        {}, '콘끔', '0:콘:끔')
    assert not [o for o in ET.fromstring(xml).iter('Object')
                if (o.get('Name') or '').startswith('FinishCone')]
    assert 'finish_cone' not in sdef


def test_obstacle_chain_keeps_its_own_model():
    """체인은 회피 **대상**이고 콘은 아니다 — 모델을 같이 바꾸지 않는다."""
    import gen_scenarios as gs
    assert 'Definition="Fuelcan01"' in gs.blk_object('O', 0.0, 0.0, 0.0)
