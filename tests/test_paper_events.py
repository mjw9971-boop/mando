"""논문 실험 세트의 생성기 기능 (2026-10-02) — 새 이벤트 · 균형 생성 · 경로 품질 게이트.

지키는 것:
  · 근접 정지 출발: 장애물 중심 = (축거 + 앞 오버행) + 거리 + 차량 길이/2 — 거리는 자차
    앞범퍼 ~ 장애물 뒤끝이다. E(출발 거리)는 이 유형에만 걸지 않는다.
  · 앞뒤 연속: 범퍼 간 간격이 지정값이고 전부 같은 차로다.
  · 나란히: 자차가 가운데 차로이고, 막힌 쪽은 열린 쪽의 반대다.
  · 지그재그: 두 번째 차는 첫 차 차로의 옆 차로다 (같은 차로가 아니다).
  · 옆 차로 교통: 교통 차량은 옆 차로에만 있다 (자차로에 없다).
  · 신호 대기 줄: 신호가 적색으로 시작하고, 대기 차량은 녹색 시각에 풀린다. 줄은 경로의
    차로변경 창과 겹치지 않고, 풀린 차량은 자기 차로로 직진한다 (2026-10-03).
  · 균형 생성: 조건마다 per_condition 곳, 한 조건 안에서는 서로 다른 도로.
  · 경로 품질 게이트: 꼬리·급회전·콘·두 빌더 — 켜져 있으면 폐기, 끄면 통과, 실경로는 기록만.

생성 검사는 다른 브랜치 빌더 게이트를 **끄고** 돈다 (그 커밋이 없는 클론에서도 돌게).
그 게이트 자체는 빌더 호출을 바꿔치기해 따로 본다.
"""
import copy
import json
import math
import pathlib
import re
import sys
import xml.etree.ElementTree as ET

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import gen_scenarios as gs                                      # noqa: E402

THEMES = ['논문2_근접정지', '논문3_앞뒤연속', '논문4_나란히', '논문5_지그재그',
          '논문6_옆차로교통', '논문7_오탐_신호줄']


def _cfg(**gates):
    cfg = copy.deepcopy(gs.plc_cfg())
    cfg['route_gates']['other_builder_enable'] = False
    cfg['route_gates'].update(gates)
    for key in gs._AVOID_KEYS.values():                 # A·E 켬 / D·F 끔 을 명시 (params 와 무관)
        cfg['avoid_geom'][key] = key in ('neighbor_run_enable', 'start_clear_enable')
    return cfg


@pytest.fixture(scope='module')
def lg():
    return gs.LaneGraph(str(ROOT / 'data' / 'lane_graph.pkl'))


@pytest.fixture(scope='module')
def out(tmp_path_factory):
    """유형마다 조건당 1곳씩 생성 (모듈 픽스처라 설정은 직접 바꾸고 되돌린다)."""
    d = tmp_path_factory.mktemp('paper_events')
    saved = gs._PLC_CFG
    gs._PLC_CFG = _cfg()
    try:
        assert gs.main(THEMES + ['--count', '1', '--seed', '0', '--out-dir', str(d)]) == 0
    finally:
        gs._PLC_CFG = saved
    return d


def _defs(out, theme):
    res = []
    for y in sorted((out / theme).glob('*.yaml')):
        d = yaml.safe_load(y.read_text(encoding='utf-8'))
        root = ET.parse(y.with_suffix('.xml')).getroot()
        pos = {}
        for pl in root.iter('Player'):
            pa = pl.find('Init').find('PosAbsolute')
            if pa is not None:
                pos[pl.find('Description').get('Name')] = (
                    float(pa.get('X')), float(pa.get('Y')), float(pa.get('Direction')))
        rec = json.loads((y.parent / (y.stem + '.placement.json')).read_text(encoding='utf-8'))
        res.append((d, pos, root, rec, y))
    return res


def _lane(lg, p):
    return tuple(lg.locate(p[0], p[1], yaw=p[2]).lane)


def test_counts_and_private_keys_not_in_yaml(out):
    n = {t: len(list((out / t).glob('*.xml'))) for t in THEMES}
    assert n == {'논문2_근접정지': 2, '논문3_앞뒤연속': 4, '논문4_나란히': 2, '논문5_지그재그': 3,
                 '논문6_옆차로교통': 2, '논문7_오탐_신호줄': 1}
    for t in THEMES:
        for d, _pos, _root, rec, _y in _defs(out, t):
            assert not any(k.startswith('_') for k in d)        # 내부용 키는 정의에 안 남는다
            assert 'route_quality' in rec


def test_near_stop_distance_is_bumper_to_rear(out):
    veh = gs.full_cfg()['vehicle']
    front = float(veh['wheelbase']) + float(veh['front_overhang_m'])
    got = {}
    for d, _pos, _root, rec, _y in _defs(out, '논문2_근접정지'):
        e = d['events'][0]
        got[e['bumper_gap_m']] = e['route_s']
        assert e['placement'] == '근접정지' and e['objects'] == [e['name']]
        assert e['route_s'] == pytest.approx(front + e['bumper_gap_m'] + gs.car_len() / 2, abs=0.01)
        r = rec['events'][0]
        assert r['A']['pass'] and r['E']['enabled'] is False        # E 는 이 유형만 예외
    assert sorted(got) == [8.0, 30.0]


def test_parked_row_gap_and_same_lane(out, lg):
    seen = set()
    for d, pos, _root, _rec, _y in _defs(out, '논문3_앞뒤연속'):
        e = d['events'][0]
        cars = [pos[n] for n in e['objects']]
        assert len(cars) == e['count'] and e['count'] in (2, 3)
        for a, b in zip(cars, cars[1:]):
            assert math.hypot(b[0] - a[0], b[1] - a[1]) - gs.car_len() == pytest.approx(e['gap_m'], abs=0.6)
        seen.add(e['gap_m'])
    assert seen == {10.0, 20.0, 30.0, 45.0}


def test_side_by_side_blocks_ego_and_one_neighbor(out, lg):
    opened = set()
    for d, pos, _root, rec, _y in _defs(out, '논문4_나란히'):
        e = d['events'][0]
        ego, side = pos[e['ego_lane']], pos[e['side_lane']]
        k = _lane(lg, ego)
        left, right = lg.neighbor(k, 'left'), lg.neighbor(k, 'right')
        assert left is not None and right is not None               # 자차 = 가운데 차로
        blocked = right if e['open'] == '좌' else left
        assert _lane(lg, side) == tuple(blocked)
        dx, dy = side[0] - ego[0], side[1] - ego[1]
        assert abs(dx * math.cos(ego[2]) + dy * math.sin(ego[2])) < 0.5      # 같은 위치(종)
        open_side = 'left' if e['open'] == '좌' else 'right'
        assert rec['events'][0]['A'][open_side]['ok']               # 열린 쪽으로 회피가 성립
        opened.add(e['open'])
    assert opened == {'좌', '우'}


def test_zigzag_second_car_is_in_the_neighbor_lane(out, lg):
    gaps = set()
    for d, pos, _root, _rec, _y in _defs(out, '논문5_지그재그'):
        e = d['events'][0]
        a, b = pos[e['ego_lane']], pos[e['side_lane']]
        assert _lane(lg, a) != _lane(lg, b)
        assert math.hypot(b[0] - a[0], b[1] - a[1]) > e['gap_m']
        gaps.add(e['gap_m'])
    assert gaps == {30.0, 45.0, 60.0}


def test_side_traffic_only_in_the_neighbor_lane(out, lg):
    for d, pos, root, _rec, _y in _defs(out, '논문6_옆차로교통'):
        e = d['events'][0]
        car_lane = _lane(lg, pos[e['name']])
        side = 'left' if e['traffic_side'] == '좌' else 'right'
        shapes = {ps.get('Name'): [(float(w.get('X')), float(w.get('Y'))) for w in ps.iter('Waypoint')]
                  for ps in root.iter('PathShape')}
        assert len(e['traffic']) == e['traffic_count'] >= 2
        for name in e['traffic']:
            pts = shapes[name + 'Path']
            x, y = pts[0]
            m = lg.locate(x, y)
            assert tuple(m.lane)[:1] != () and abs(m.t) < 0.6       # 차로 중심 위
            assert tuple(m.lane) != car_lane                         # 자차로가 아니다
        # 줄 머리는 장애물 10 m 뒤라 앞 차로 섹션일 수 있다 — 장애물 옆 지점으로 비교한다.
        cx, cy, ch = pos[e['name']]
        near = min(shapes[e['traffic'][0] + 'Path'], key=lambda p: math.hypot(p[0] - cx, p[1] - cy))
        assert tuple(lg.locate(*near).lane) == tuple(lg.neighbor(car_lane, side))
        lat = -(near[0] - cx) * math.sin(ch) + (near[1] - cy) * math.cos(ch)
        assert (lat > 2.0) if side == 'left' else (lat < -2.0)
        assert e['objects'] == [e['name']]                           # 지나가야 할 것은 정차차뿐


def test_signal_queue_starts_red_and_releases_at_green(out, lg):
    (d, pos, root, _rec, y), = _defs(out, '논문7_오탐_신호줄')
    e = d['events'][0]
    text = y.with_suffix('.xml').read_text(encoding='utf-8')
    for c in e['controllers']:
        m = re.search(rf'<SignalController Id="{c}" [^>]*>(.*?)</SignalController>', text, re.S)
        ph = re.findall(r'<Phase Duration="([0-9.]+)" Type="(\w+)"/>', m.group(1))
        assert [t for _d, t in ph] == ['stop', 'go', 'attention']    # 적색으로 시작
        assert float(ph[0][0]) == pytest.approx(e['green_at_s'], abs=0.06)
    assert len(e['queue']) == e['count']
    # 2026-10-03: 대기 줄(뒤끝 − queue_lc_clear_m ~ 정지선)은 경로의 차로변경 창과 겹치지 않는다
    pe = gs.pe_cfg()
    lo = e['route_s'][-1] - gs.car_len() / 2.0 - float(pe['queue_lc_clear_m'])
    assert not any(w0 < e['stop_s'] and w1 > lo for w0, w1 in e['lc_windows'])
    shapes = {ps.get('Name'): [(float(w.get('X')), float(w.get('Y')), float(w.get('Yaw')))
                               for w in ps.iter('Waypoint')] for ps in root.iter('PathShape')}
    pitch = gs.car_len() + float(pe['queue_car_gap_m'])
    for i, name in enumerate(e['queue']):
        m = re.search(rf'<PlayerActions Player="{name}">(.*?)</PlayerActions>', text, re.S)
        delays = [float(x) for x in re.findall(r'DelayTime="([0-9.]+)"', m.group(1))]
        targets = [float(x) for x in re.findall(r'Target="([0-9.]+)"', m.group(1))]
        assert targets[0] == 0.0 and targets[1] > 5.0                # 정지 → 출발
        assert delays[0] == 0.0 and delays[1] >= e['green_at_s'] - 0.01
        # 풀린 뒤 자기 차로로 직진 — 경로가 차로 중심 위이고 옆으로 꺾이는 점이 없다
        pts = shapes[name + 'Path']
        plen = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
        assert plen == pytest.approx(e['release_path_m'] + i * pitch, abs=5.0)
        for x, y_, _h in pts:
            assert abs(lg.locate(x, y_).t) < 0.6
        for a, b in zip(pts, pts[1:]):
            assert abs((b[2] - a[2] + math.pi) % (2 * math.pi) - math.pi) < math.radians(25)


def test_own_lane_path_takes_the_straightest_branch(lg):
    """갈림(교차로 연결로)에서는 방향이 가장 덜 바뀌는 다음 차로를 탄다."""
    def turn(a, b):
        return abs((b - a + math.pi) % (2 * math.pi) - math.pi)
    k = next(k for k, v in sorted(lg.lanes.items())
             if v['type'] == 'driving' and v['junction'] == -1 and len(lg.successors(k)) >= 2
             and len({round(turn(lg.point_at(k, lg.length(k))[3], lg.point_at(n, lg.length(n))[3]), 1)
                      for n in lg.successors(k)}) >= 2)
    L = lg.length(k)
    h_end = lg.point_at(k, L)[3]
    want = min(lg.successors(k), key=lambda n: (turn(h_end, lg.point_at(n, lg.length(n))[3]), n))
    pts = gs._own_lane_path(lg, k, max(L - 6.0, 0.0), 6.0 + lg.length(want) / 2.0)
    x, y, _z, _h = pts[-1]
    assert tuple(lg.locate(x, y).lane) == tuple(want) or lg.locate(x, y).lane in lg.successors(want)


def test_conditions_use_distinct_roads(tmp_path, monkeypatch):
    monkeypatch.setattr(gs, '_PLC_CFG', _cfg())
    assert gs.main(['논문3_앞뒤연속', '--count', '2', '--seed', '0', '--out-dir', str(tmp_path)]) == 0
    by_gap: dict = {}
    for y in sorted((tmp_path / '논문3_앞뒤연속').glob('*.placement.json')):
        rec = json.loads(y.read_text(encoding='utf-8'))
        d = yaml.safe_load(y.with_name(y.name.replace('.placement.json', '.yaml')).read_text(encoding='utf-8'))
        by_gap.setdefault(d['axes']['간격'], []).append(rec['events'][0]['lane'][0])
    assert set(by_gap) == {10, 20, 30, 45}
    for gap, roads in by_gap.items():
        assert len(roads) == 2 and len(set(roads)) == 2, (gap, roads)


# ── 경로 품질 게이트 (빌더 호출을 바꿔치기) ──────────────────────────────
def _fake_cli(outputs):
    def run(root, _csv):
        key = 'other' if 'mando_builder_' in str(root) else 'own'
        return outputs[key]
    return run


SDEF = {'finish_cone': {'clear_m': [0.3, 0.3]},
        '_route_objs': [{'name': 'StoppedCar01', 'x': 0.0, 'y': 0.0, 'lane': [10, 0, -1]}]}
ROWS = [(1, 0.0, 0.0), (2, 100.0, 0.0)]
GOOD = (0, '', {'lanes': [(10, 0, -1), (11, 0, -1)], 'total_length': 200.0})


@pytest.mark.parametrize('code,own,other,sdef', [
    ('T', (0, '  [경고] 종료선 뒤 꼬리 8.8 m < 요구 12 m — …', GOOD[2]), GOOD, SDEF),
    ('R', (0, '급회전 연결로 (2921, 0, -1) R 5.06 m — 감속 진입', GOOD[2]), GOOD, SDEF),
    ('C', GOOD, GOOD, dict(SDEF, finish_cone={'clear_m': [0.2, -0.0]})),
    ('B', (0, '', {'lanes': [(10, 0, -2)], 'total_length': 200.0}), GOOD, SDEF),
    ('W', GOOD, (0, '', {'lanes': [(10, 0, -2)], 'total_length': 209.0}), SDEF),
])
def test_route_gate_codes_and_kill_switches(monkeypatch, tmp_path, code, own, other, sdef):
    monkeypatch.setattr(gs, 'other_builder_dir', lambda ref: tmp_path / 'mando_builder_x')
    monkeypatch.setattr(gs, '_cli_route', _fake_cli({'own': own, 'other': other}))
    monkeypatch.setattr(gs, '_PLC_CFG', _cfg(other_builder_enable=True))
    with pytest.raises(gs.RouteGateError) as ei:
        gs.route_quality(sdef, ROWS, enforce=True)
    assert ei.value.gate == code
    # 그 게이트만 끄면 통과한다
    key = {'T': 'tail_enable', 'R': 'sharp_turn_enable', 'C': 'cone_enable',
           'B': 'builder_enable', 'W': 'other_builder_enable'}[code]
    monkeypatch.setattr(gs, '_PLC_CFG', _cfg(**{'other_builder_enable': True, key: False}))
    q = gs.route_quality(sdef, ROWS, enforce=True)
    assert isinstance(q, dict)
    # 실경로(enforce=False)는 걸려도 버리지 않고 기록만 한다 — 다른 브랜치 빌더 결과도 남는다
    monkeypatch.setattr(gs, '_PLC_CFG', _cfg(other_builder_enable=True))
    q = gs.route_quality(sdef, ROWS, enforce=False)
    assert 'other_builder' in q and q['other_builder']['same_lanes'] == (own[2]['lanes'] == other[2]['lanes'])


def test_all_gates_pass_on_a_clean_route(monkeypatch, tmp_path):
    monkeypatch.setattr(gs, 'other_builder_dir', lambda ref: tmp_path / 'mando_builder_x')
    monkeypatch.setattr(gs, '_cli_route', _fake_cli({'own': GOOD, 'other': GOOD}))
    monkeypatch.setattr(gs, '_PLC_CFG', _cfg(other_builder_enable=True))
    q = gs.route_quality(SDEF, ROWS, enforce=True)
    assert q['tail_short_m'] is None and q['sharp_turn_r_m'] == [] and q['builder_off_lane'] == []
    assert q['other_builder']['off_lane'] == [] and q['other_builder']['same_lanes'] is True
