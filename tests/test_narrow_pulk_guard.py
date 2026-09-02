"""pulk 교통류가 켜진 시나리오에는 narrow 를 배치하지 않는다 (2026-09-02).

결함: narrow(양측 정차, 침범폭 0.7)는 ego 기준으로는 통과 가능하지만
(잔여폭 2.10~2.14 m > 차폭 1.9 m, 엇갈림 14 m) VTD internal-driver 교통류는
이 협착을 지나지 못하고 그 앞에 정체를 만들며, ego 가 정체에 갇혀 blocked 로
끝난다. 실측:
  · 실전주행_교통류_01_좌회전24 (logs/batch/20260902_230229) — narrow 20.7 m
    앞에서 pulk 차량(이력상 최대 15.6 m/s) 정지, ego 는 그 뒤 9.6 m 에 고착.
  · 정적회피집중_02_우회전 (logs/batch/20260902_231132) — ego 가 narrow 5.1 m
    앞 정지(blocked), 뒤로 pulk 3대 정체.

수정: 테마 정의에서 narrow 제거(실전주행_교통류·정적회피집중) + build_scenario
의 안전장치(gen_placement.narrow_pulk_guard_enable, 기본 true) — 저장된 정의
(--from-yaml)로 오는 경우까지 한 곳에서 막는다. narrow 단독 검증은 pulk 없는
차로폭협착 테마가 담당한다.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import gen_scenarios as gs                                      # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'


# ── 테마 정의: narrow 와 pulk 는 같은 테마에 없어야 한다 ──────────────────
def test_no_theme_combines_narrow_with_pulk():
    _routes, themes, _gen = gs.load_themes()
    bad = [name for name, cfg in themes.items()
           if cfg.get('pulk') and 'narrow' in (cfg.get('event') or [])]
    assert bad == [], f'pulk 테마에 narrow 가 들어 있다: {bad}'


def test_narrow_verification_stays_with_pulkless_theme():
    """narrow 단독 검증 담당(차로폭협착)은 pulk 없이 narrow 를 유지해야 한다."""
    _routes, themes, _gen = gs.load_themes()
    cfg = themes['차로폭협착']
    assert 'narrow' in cfg['event']
    assert not cfg.get('pulk')


def test_traffic_count_axis_excludes_30():
    """30대/보통은 정체가 교차로 안까지 밀린다 (실전주행_교통류_02_좌회전8,
    logs/batch/20260902_212135 — in_junction 정차 10.6 s)."""
    _routes, themes, _gen = gs.load_themes()
    assert themes['실전주행_교통류']['교통류대수'] == [12, 20]


def test_static_focus_keeps_combine_without_narrow():
    _routes, themes, _gen = gs.load_themes()
    cfg = themes['정적회피집중']
    assert cfg['event'] == ['static_vehicle', 'obstacle_chain']
    assert cfg.get('combine') == 2


# ── build_scenario 안전장치 ──────────────────────────────────────────────
def guard_cfg(monkeypatch, enable):
    cfg = dict(gs.plc_cfg())
    cfg['narrow_pulk_guard_enable'] = enable
    monkeypatch.setattr(gs, '_PLC_CFG', cfg)


def real_route():
    from vtd_adapter.lanegraph import LaneGraph
    lg = LaneGraph(str(GRAPH))
    route_defs, _themes, gen_cfg = gs.load_themes()
    pool = gs.RoutePool(lg, route_defs, 7, gen_cfg)
    route = pool.get('직진', 0, '협착가드', min_length_m=400.0)
    return lg, gs.junction_ctrl_map(lg), route


PULK_AXES = {'교통류대수': 12, '교통류밀도': '보통'}


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_guard_rejects_narrow_with_pulk(monkeypatch):
    guard_cfg(monkeypatch, True)
    lg, ctrl_map, route = real_route()
    with pytest.raises(gs.GenError, match='narrow 는 pulk 교통류를 막는다'):
        gs.build_scenario(lg, ctrl_map, route,
                          [('narrow', {'위치': 0.5, '침범폭': 0.7})],
                          dict(PULK_AXES), '가드검사', '0:가드:검사',
                          pulk=gs.pulk_def(PULK_AXES))


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_narrow_without_pulk_is_untouched(monkeypatch):
    """차로폭협착 경로 — pulk 가 없으면 가드가 관여하지 않는다."""
    guard_cfg(monkeypatch, True)
    lg, ctrl_map, route = real_route()
    xml_text, sdef, _bad = gs.build_scenario(
        lg, ctrl_map, route, [('narrow', {'위치': 0.5, '침범폭': 0.7})],
        {}, '가드검사', '0:가드:검사', pulk=None)
    assert any(e['kind'] == 'narrow' for e in sdef['events'])


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_guard_switch_off_restores_old_behaviour(monkeypatch):
    """off 면 종전 동작(동시 배치 허용) 그대로 — 기존 산출물 재현 수단."""
    guard_cfg(monkeypatch, False)
    lg, ctrl_map, route = real_route()
    xml_text, sdef, _bad = gs.build_scenario(
        lg, ctrl_map, route, [('narrow', {'위치': 0.5, '침범폭': 0.7})],
        dict(PULK_AXES), '가드검사', '0:가드:검사', pulk=gs.pulk_def(PULK_AXES))
    assert any(e['kind'] == 'narrow' for e in sdef['events'])
    assert '<PulkDef' in xml_text


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_other_events_pass_with_pulk(monkeypatch):
    """가드는 narrow 만 본다 — static_vehicle/obstacle_chain 은 pulk 와 공존."""
    guard_cfg(monkeypatch, True)
    lg, ctrl_map, route = real_route()
    xml_text, sdef, _bad = gs.build_scenario(
        lg, ctrl_map, route, [('static_vehicle', {'위치': 0.5})],
        dict(PULK_AXES), '가드검사', '0:가드:검사', pulk=gs.pulk_def(PULK_AXES))
    assert any(e['kind'] == 'static_vehicle' for e in sdef['events'])
