"""
VTD 네이티브 교통류 (gen_scenarios.pulk_def / XmlDoc.set_pulk)
+ 관측 지표 (event_check.traffic_metrics).

왜 있는가: 기존 실전주행은 ego 앞 1대(slow_lead/lead_brake)뿐이라 "정적 장애물을
피하려는데 옆·뒤 차가 방해한다" 를 만들 수 없었다. 회피 판단은 옆 차로가 비었는지에
달려 있는데, 그 조건을 만드는 이벤트가 없었다.

왜 ev_traffic(Path01 고정 배치)이 아닌가 (2026-08-30 실기 확인): 고정 배치는
(1) 6대 × 45 m = 265 m 대역뿐이라 3 km 경로를 못 채우고 (2) ego 와 속도가 다르면
3 km 동안 수백 m 벌어져 후반부가 빈다. PulkTraffic 은 영역을 벗어난 차량을
가장자리에 재배치하므로 두 문제가 모두 사라진다.

관측 지표를 같이 두는 이유: 2026-08-30 보행자 이벤트가 XML 에만 있고 실주행에는
안 나타났는데 리포트가 조용했다. 교통류도 같은 실패를 할 수 있다.
"""
import math
import re

import pytest
import yaml

import event_check as ec       # noqa: E402 (conftest 가 tools 경로 추가)
import gen_scenarios as gs     # noqa: E402
from vtd_adapter.config import load_params_yaml


def tc():
    return load_params_yaml()['gen_traffic']


def pc():
    return load_params_yaml()['pulk']


# ── PulkDef 속성 ─────────────────────────────────────────────────────────
def test_defaults_come_from_params_not_code():
    """상수 단일 출처 — 코드 기본값이 없으므로 params 값이 그대로 나와야 한다."""
    c, d = pc(), gs.pulk_def({})
    assert d['Count'] == int(c['count'])
    assert d['SemiMajorAxis'] == f"{float(c['semi_major_m']):g}"
    assert d['InnerRadius'] == f"{float(c['inner_radius_m']):g}"
    assert d['CenterOffset'] == f"{float(c['center_offset_m']):g}"
    assert (d['CentralPlayer'], d['FillAtStart'], d['VisibleInArea']) \
        == ('Ego', 'true', '-1')


def test_params_yaml_has_all_pulk_keys():
    """키가 빠지면 KeyError 로 죽는다 — 코드 기본값으로 조용히 메우지 않는다."""
    c = pc()
    for k in ('count', 'semi_major_m', 'semi_minor_m', 'inner_radius_m',
              'center_offset_m', 'area_f', 'area_b', 'area_l', 'area_r',
              'own_side', 'cars', 'vans', 'buses', 'trucks', 'bikes', 'density'):
        assert k in c, k


def test_ratios_are_zero_to_one_decimals():
    """에디터 스펙: 비율은 전부 0~1 소수 (퍼센트 아님)."""
    d = gs.pulk_def({})
    for k in ('AreaF', 'AreaB', 'AreaL', 'AreaR', 'OwnSide',
              'Cars', 'Vans', 'Buses', 'Trucks', 'Bikes'):
        assert 0.0 <= float(d[k]) <= 1.0, (k, d[k])


def test_circle_is_equal_semi_axes():
    """Circle 은 SemiMajorAxis == SemiMinorAxis 로만 표현된다 — 구분 속성 없음."""
    d = gs.pulk_def({})
    assert d['SemiMajorAxis'] == d['SemiMinorAxis']
    assert not any('circle' in k.lower() or 'shape' in k.lower() for k in d)


# ── 두 합계 제약 (에디터가 검사한다) ─────────────────────────────────────
def _sum(d, keys):
    return sum(float(d[k]) for k in keys)


AREA_KEYS = ('AreaF', 'AreaB', 'AreaL', 'AreaR')
CLASS_KEYS = ('Cars', 'Vans', 'Buses', 'Trucks', 'Bikes')


def test_both_sums_are_one():
    d = gs.pulk_def({})
    assert _sum(d, AREA_KEYS) == pytest.approx(1.0, abs=1e-6)
    assert _sum(d, CLASS_KEYS) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('key', ['area_f', 'cars'])
def test_broken_sum_is_gen_error_not_silent_fix(monkeypatch, key):
    """자동 보정 금지 — 조용히 고치면 시나리오마다 다른 값이 들어간다."""
    monkeypatch.setattr(gs, 'pulk_cfg', lambda: dict(pc(), **{key: 0.5}))
    with pytest.raises(gs.GenError, match='합이'):
        gs.pulk_def({})


def test_sums_hold_for_every_axis_combination():
    """축 값이 어떻게 뽑혀도 두 제약은 유지된다 (밀도는 크기만 바꾼다)."""
    for n in gs.AXIS_DEFAULTS['교통류대수']:
        for dens in gs.AXIS_DEFAULTS['교통류밀도']:
            d = gs.pulk_def({'교통류대수': n, '교통류밀도': dens})
            assert _sum(d, AREA_KEYS) == pytest.approx(1.0, abs=1e-6), (n, dens)
            assert _sum(d, CLASS_KEYS) == pytest.approx(1.0, abs=1e-6), (n, dens)


# ── 축 연동 (시드 → 값) ─────────────────────────────────────────────────
def test_count_axis_wins_over_params():
    assert gs.pulk_def({'교통류대수': 33})['Count'] == 33


def test_density_axis_changes_area_only():
    """교통류밀도 → 영역 크기 프리셋. 비율은 손대지 않는다."""
    base, tight, wide = (gs.pulk_def({}), gs.pulk_def({'교통류밀도': '조밀'}),
                         gs.pulk_def({'교통류밀도': '성김'}))
    assert float(tight['SemiMajorAxis']) < float(wide['SemiMajorAxis'])
    assert float(tight['InnerRadius']) < float(wide['InnerRadius'])
    assert tight['SemiMajorAxis'] == tight['SemiMinorAxis']
    assert wide['SemiMajorAxis'] == wide['SemiMinorAxis']
    for k in AREA_KEYS + CLASS_KEYS + ('OwnSide', 'CenterOffset'):
        assert tight[k] == wide[k] == base[k], k


def test_unknown_density_is_gen_error():
    with pytest.raises(gs.GenError, match='교통류밀도'):
        gs.pulk_def({'교통류밀도': '없는값'})


def test_axes_registered_in_axis_defaults():
    for ax in ('교통류대수', '교통류밀도'):
        assert gs.AXIS_DEFAULTS.get(ax), ax
    # ev_traffic 전용 축은 사라졌다 (PulkDef 에 속도 속성이 없다)
    assert '교통류속도' not in gs.AXIS_DEFAULTS


def test_same_axes_give_byte_identical_attrs():
    """같은 시드 = 같은 축 값 = 바이트 동일 PulkDef (서식 고정)."""
    a = {'교통류대수': 20, '교통류밀도': '보통'}
    assert gs.pulk_attrs(gs.pulk_def(a)) == gs.pulk_attrs(gs.pulk_def(dict(a)))


# ── XML 삽입 ─────────────────────────────────────────────────────────────
def test_fills_existing_empty_tag_without_duplicating():
    """대회 제공 XML 11행에 빈 <PulkTraffic/> 이 있다 — 새로 붙이면 태그가 둘이 된다."""
    doc = gs.XmlDoc(gs.TEMPLATE)
    assert doc.text.count('PulkTraffic') == 1        # 빈 태그 1개
    doc.set_pulk(gs.pulk_attrs(gs.pulk_def({})))
    root = __import__('xml.etree.ElementTree', fromlist=['ET']).fromstring(doc.text)
    pulks = root.findall('PulkTraffic')
    assert len(pulks) == 1 and len(pulks[0].findall('PulkDef')) == 1
    # Scenario 직속 자식이고 <TrafficElements/> 와 <TrafficControl> 사이다
    tags = [c.tag for c in root]
    assert tags.index('TrafficElements') < tags.index('PulkTraffic') \
        < tags.index('TrafficControl')


def test_second_set_pulk_replaces_rather_than_appends():
    doc = gs.XmlDoc(gs.TEMPLATE)
    doc.set_pulk(gs.pulk_attrs(gs.pulk_def({'교통류대수': 12})))
    doc.set_pulk(gs.pulk_attrs(gs.pulk_def({'교통류대수': 30})))
    assert doc.text.count('<PulkDef ') == 1
    assert 'Count="30"' in doc.text and 'Count="12"' not in doc.text


def test_attr_order_matches_verified_editor_output():
    """실기 검증된 Scenario Editor 2025.2 산출물과 같은 속성 순서."""
    attrs = gs.pulk_attrs(gs.pulk_def({}))
    names = re.findall(r'(\w+)=', attrs)
    assert names == ['CentralPlayer', 'Count', 'FillAtStart',
                     'SemiMajorAxis', 'SemiMinorAxis', 'InnerRadius', 'CenterOffset',
                     'AreaF', 'AreaB', 'AreaL', 'AreaR', 'OwnSide',
                     'Cars', 'Vans', 'Buses', 'Trucks', 'Bikes', 'VisibleInArea']


# ── 이벤트가 아니다 ──────────────────────────────────────────────────────
def test_traffic_is_not_an_event_anymore():
    """전역 속성 — 이벤트 목록에 넣으면 구간 배치·슬롯 분배 논리를 타게 된다."""
    assert 'traffic' not in gs.EVENTS
    assert not hasattr(gs, 'ev_traffic')
    th = yaml.safe_load(open('configs/themes.yaml', encoding='utf-8'))['themes']
    cfg = th['실전주행_교통류']
    assert 'traffic' not in cfg['event']
    assert cfg.get('pulk') is True                   # 플래그로 켠다
    assert {'교통류대수', '교통류밀도'} <= set(cfg['vary'])


def test_theme_without_pulk_flag_gets_no_pulkdef():
    """기존 실전주행 회귀 — pulk 플래그가 없으면 XML 이 그대로다."""
    th = yaml.safe_load(open('configs/themes.yaml', encoding='utf-8'))['themes']
    assert not th['실전주행'].get('pulk')
    doc = gs.XmlDoc(gs.TEMPLATE)
    assert '<PulkDef' not in doc.text


# ── 관측 지표 ────────────────────────────────────────────────────────────
def veh_tick(objs, ex=0.0, yaw=0.0):
    """objs = [(id, 종거리, 횡거리, 속도)] — ego 프레임으로 준다."""
    out = []
    for oid, lon, lat, sp in objs:
        out.append({'id': oid, 'cls': 'vehicle', 'speed': sp,
                    'x': ex + lon * math.cos(yaw) - lat * math.sin(yaw),
                    'y': lon * math.sin(yaw) + lat * math.cos(yaw)})
    return {'ego': {'x': ex, 'y': 0.0, 'yaw': yaw}, 'objects': out}


def test_params_yaml_keeps_observation_keys():
    c = tc()
    for k in ('observe_moving_mps', 'observe_corridor_m'):
        assert k in c, k


def test_metrics_count_moving_vehicles_in_corridor():
    ts = [veh_tick([(1, 30.0, 0.0, 8.0), (2, -20.0, 3.0, 8.0)])]
    m = ec.traffic_metrics(ts, planned=2, tc=tc())
    assert (m['planned'], m['observed']) == (2, 2)
    assert (m['min_ahead_m'], m['min_behind_m']) == (30.0, 20.0)
    assert m['min_lon_m'] == -20.0          # 절댓값이 가장 작은 쪽


def test_metrics_exclude_parked_and_far_lateral():
    """정차 차량(narrow/static_vehicle)과 회랑 밖 차량은 교통류가 아니다."""
    c = tc()
    ts = [veh_tick([(1, 10.0, 0.0, 0.0),                     # 정차
                    (2, 12.0, c['observe_corridor_m'] + 1, 8.0)])]   # 회랑 밖
    m = ec.traffic_metrics(ts, planned=2, tc=c)
    assert (m['observed'], m['min_lon_m']) == (0, None)


def test_metrics_zero_observed_is_the_warning_case():
    """배치했는데 한 대도 안 보이면 리포트에 경고가 남아야 한다."""
    rep = {'ok': 0, 'total': 0, 'unreached': 0, 'events': [],
           'traffic': ec.traffic_metrics([], planned=20, tc=tc())}
    txt = ec.render(rep, 'x')
    assert '배치 20대 / 관측 0대' in txt and '경고' in txt


def test_metrics_holds_under_rotated_heading():
    for yaw in (0.0, 1.0, -2.5, math.pi):
        m = ec.traffic_metrics([veh_tick([(1, 25.0, 1.0, 8.0)], ex=100.0, yaw=yaw)],
                               planned=1, tc=tc())
        assert m['observed'] == 1 and m['min_ahead_m'] == pytest.approx(25.0, abs=0.01)


def test_check_scenario_reports_traffic_from_pulk_count(tmp_path):
    """배치 대수의 출처는 전역 속성 pulk.Count 다 (교통류는 이벤트가 아니다)."""
    y = tmp_path / 's.yaml'
    y.write_text(yaml.safe_dump({'events': [], 'pulk': gs.pulk_def({'교통류대수': 20})},
                                allow_unicode=True), encoding='utf-8')
    log = tmp_path / 'r.jsonl'
    log.write_text(__import__('json').dumps(
        {**veh_tick([(1, 12.0, 0.0, 9.0)]), 'raw': {},
         'ego': {'x': 0.0, 'y': 0.0, 'yaw': 0.0, 'route_s': 10.0}}) + '\n',
        encoding='utf-8')
    rep = ec.check_scenario(str(y), str(log))
    assert rep['total'] == 0                      # 보행자 판정과 섞이지 않는다
    assert (rep['traffic']['planned'], rep['traffic']['observed']) == (20, 1)
    assert '교통류(관측)' in ec.render(rep, 's')


def test_check_scenario_omits_traffic_without_pulk(tmp_path):
    y = tmp_path / 's.yaml'
    y.write_text(yaml.safe_dump({'events': [{'kind': 'narrow'}]}), encoding='utf-8')
    log = tmp_path / 'r.jsonl'
    log.write_text('', encoding='utf-8')
    assert ec.check_scenario(str(y), str(log))['traffic'] is None
