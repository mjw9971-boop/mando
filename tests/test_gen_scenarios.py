"""tools/gen_scenarios.py — 시나리오 자동 생성기 검증.

명세: 대표 주제 4개를 --count 2 --seed 1 로 생성하면
  · XML 파싱 전부 통과
  · ego 차선 이벤트의 경로 횡거리 경고 0
  · CSV 는 대회형식 (시작 + 교차로 진입/진출 짝 + 종료 = 짝수 행)
  · 생성물을 build_route 로 빌드하면 "경로 없음" 이 없다
  · 같은 seed 재실행 시 동일 산출물 (재현성)
  · 정의 YAML 로 단건 재생성하면 동일 산출물
"""
import json
import math
import pathlib
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import gen_scenarios as gs                                      # noqa: E402
from build_route import build_route, junction_segments, read_waypoints_csv  # noqa: E402

THEMES = ['보행자집중', '급정거집중', '교차로집중', '대향차직진']
SEED = ['--seed', '1']


def _gen(out_dir, capsys=None):
    rc = gs.main(THEMES + ['--count', '2'] + SEED + ['--out-dir', str(out_dir)])
    assert rc == 0
    return capsys.readouterr().out if capsys else None


@pytest.fixture(scope='module')
def outputs(tmp_path_factory):
    out = tmp_path_factory.mktemp('gen')
    # 모듈 픽스처라 capsys 를 못 쓴다 — 경고는 별도 테스트에서 다시 생성해 확인
    _gen(out)
    return out


def test_xml_parse_all(outputs):
    xmls = sorted(outputs.glob('*/*.xml'))
    assert len(xmls) == 8, f'주제 4개 × 2 = 8개 기대, 실제 {len(xmls)}'
    for f in xmls:
        ET.parse(f)                                  # 실패 시 ParseError


def test_no_lateral_warning(tmp_path, capsys):
    stdout = _gen(tmp_path, capsys)
    assert 'ego 차선 이벤트' not in stdout, f'횡거리 경고가 있다:\n{stdout}'
    # 경고 있는 경로(기본 등)의 제외는 정상이지만, 조용히 빠지면 안 된다 —
    # 제외가 있으면 반드시 사유 목록이 출력돼야 한다
    if 'build_route 경고' in stdout:
        assert '생성 못 한 변형' in stdout


def test_csv_competition_format(outputs):
    for f in sorted(outputs.glob('*/*.csv')):
        rows = read_waypoints_csv(str(f))
        assert len(rows) >= 4, f'{f.name}: 경유점 {len(rows)}개 (최소 4)'
        assert len(rows) % 2 == 0, f'{f.name}: 홀수 행 — 시작+짝*N+종료 형식이 아니다'
        assert [r[0] for r in rows] == list(range(1, len(rows) + 1))


def test_build_route_from_outputs(outputs):
    lg = gs.LaneGraph(str(ROOT / 'data' / 'lane_graph.pkl'))
    csvs = sorted(outputs.glob('*/*.csv'))[:2]
    assert csvs
    for f in csvs:
        rows = read_waypoints_csv(str(f))
        wps = [(x, y) for _, x, y in rows]
        yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
        rt = build_route(lg, wps, 8.0, yaw,
                         junction_segs=junction_segments(len(wps)))   # 실패 시 RouteError
        assert rt['total_length'] > 100.0


def test_batch_json(outputs):
    """주제별 batch_<주제>.json + 이번 실행 전체 통합 batch_all.json."""
    per_theme = {th: json.loads((outputs / f'batch_{th}.json').read_text(encoding='utf-8'))
                 for th in THEMES}
    assert all(len(v) == 2 for v in per_theme.values())
    items = json.loads((outputs / 'batch_all.json').read_text(encoding='utf-8'))
    assert len(items) == 8
    assert items == [it for th in THEMES for it in per_theme[th]]
    names = [it['name'] for it in items]
    assert len(set(names)) == len(names)
    for it in items:
        assert it['vtd_xml_path'].startswith('/home/mjw/scenarios/')
        assert it['vtd_xml_path'].endswith('.xml')
        assert (outputs / pathlib.Path(*pathlib.Path(it['route_csv']).parts[1:])).exists()
        assert isinstance(it['timeout_s'], int) and it['timeout_s'] >= 180


def test_batch_run_merge_and_dup_check(outputs):
    """batch_run 은 목록 여러 개(glob 포함)를 합치고, 이름 중복은 통합 후 기준으로 잡는다."""
    from batch_run import load_scenarios
    per_theme_files = [str(outputs / f'batch_{th}.json') for th in THEMES]
    merged = load_scenarios(per_theme_files)
    all_items = load_scenarios([str(outputs / 'batch_all.json')])
    assert merged == all_items
    # glob 패턴 하나로 여러 목록을 합친다 ('*집중' → 보행자/급정거/교차로집중)
    globbed = load_scenarios([str(outputs / 'batch_*집중.json')])
    expect = {it['name'] for it in merged if not it['name'].startswith('대향차직진')}
    assert {it['name'] for it in globbed} == expect
    # 통합 기준 중복: batch_all 과 주제별 목록을 같이 주는 실수를 잡아야 한다
    with pytest.raises(SystemExit, match='중복'):
        load_scenarios([str(outputs / 'batch_all.json'), per_theme_files[0]])
    with pytest.raises(SystemExit, match='없다'):
        load_scenarios([str(outputs / 'batch_없는파일.json')])


def test_reproducible(outputs, tmp_path):
    _gen(tmp_path)
    a = sorted(p.relative_to(outputs) for p in outputs.glob('*/*'))
    b = sorted(p.relative_to(tmp_path) for p in tmp_path.glob('*/*'))
    assert a == b, '같은 seed 인데 파일 목록이 다르다'
    for rel in a:
        fa, fb = (outputs / rel).read_bytes(), (tmp_path / rel).read_bytes()
        assert fa == fb, f'같은 seed 인데 {rel} 내용이 다르다'


def test_from_yaml_regen(outputs, tmp_path):
    src = sorted(outputs.glob('*/*.yaml'))[0]
    rc = gs.main(['--from-yaml', str(src), '--out-dir', str(tmp_path)])
    assert rc == 0
    rel = src.relative_to(outputs)
    for ext in ('.xml', '.csv'):
        orig = (outputs / rel).with_suffix(ext)
        regen = (tmp_path / rel).with_suffix(ext)
        assert regen.read_bytes() == orig.read_bytes(), f'{rel.with_suffix(ext)} 재생성 불일치'


def test_free_start(tmp_path):
    """start=자유 주제(완주속도)를 5개 생성하면 시작 도로가 전부 다르고,
    XML·CSV·빌드·재현성이 전부 성립해야 한다."""
    import yaml
    out = tmp_path / 'a'
    rc = gs.main(['완주속도', '--count', '5'] + SEED + ['--out-dir', str(out)])
    assert rc == 0
    yamls = sorted(out.glob('완주속도/*.yaml'))
    assert len(yamls) == 5
    starts, roads_all = [], set()
    for yf in yamls:
        sdef = yaml.safe_load(yf.read_text(encoding='utf-8').split('\n', 1)[1])
        r = sdef['route']
        assert set(r['start']) == {'road', 'lane', 'x', 'y'}
        assert r['roads'] and r['roads'][0] == r['start']['road']   # 커버리지 목록
        starts.append(r['start']['road'])
        roads_all |= set(r['roads'])
        ET.parse(yf.with_suffix('.xml'))
    assert len(set(starts)) == 5, f'시작 도로가 겹친다: {starts}'
    assert len(roads_all) >= 10                                     # 맵 커버리지 확장
    # 생성 CSV 하나를 빌드 — 경로 없음이 없어야 한다
    lg = gs.LaneGraph(str(ROOT / 'data' / 'lane_graph.pkl'))
    rows = read_waypoints_csv(str(yamls[0].with_suffix('.csv')))
    wps = [(x, y) for _, x, y in rows]
    yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    build_route(lg, wps, 8.0, yaw, junction_segs=junction_segments(len(wps)))
    # 같은 seed 재실행 → 동일 산출물
    out2 = tmp_path / 'b'
    rc = gs.main(['완주속도', '--count', '5'] + SEED + ['--out-dir', str(out2)])
    assert rc == 0
    for yf in yamls:
        rel = yf.relative_to(out)
        for ext in ('.yaml', '.xml', '.csv'):
            assert (out2 / rel).with_suffix(ext).read_bytes() == \
                yf.with_suffix(ext).read_bytes(), f'{rel}{ext} 재현 불일치'


def test_routes_build_clean(outputs):
    """생성된 모든 경로 CSV 는 build_route 경고 0 이어야 한다 — 실기에서
    회전 불가 연결로 포함 경로가 route 실패(rc=1)로 죽은 사고의 회귀 방지."""
    lg = gs.LaneGraph(str(ROOT / 'data' / 'lane_graph.pkl'))
    csvs = sorted(outputs.glob('*/*.csv'))
    assert csvs
    for f in csvs:
        rows = read_waypoints_csv(str(f))
        wps = [(x, y) for _, x, y in rows]
        yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
        rt = build_route(lg, wps, 8.0, yaw, junction_segs=junction_segments(len(wps)))
        warns, first = gs.route_check(lg, rt)
        assert warns == 0, f'{f.name}: 빌드 경고 {warns}건 — {first}'


def test_warned_csv_route_rejected():
    """경고 있는 csv 경로(현재의 waypoints.csv 기본)는 풀에서 사유와 함께 탈락한다.

    waypoints.csv 가 나중에 경고 0 으로 고쳐지면 이 테스트는 '탈락하지 않음' 분기로
    지나간다 — 그때는 기본 경로 재사용이 다시 허용되는 것이 맞다.
    """
    lg = gs.LaneGraph(str(ROOT / 'data' / 'lane_graph.pkl'))
    _routes, _themes, gen_cfg = gs.load_themes()
    pool = gs.RoutePool(lg, {'기본': {'csv': 'waypoints.csv'}}, seed=1, gen_cfg=gen_cfg)
    try:
        route = pool.get('기본')
    except gs.GenError as e:
        assert 'build_route 경고' in str(e) or '스폰 게이트' in str(e)
        with pytest.raises(gs.GenError):     # 두 번째 조회도 같은 사유로 즉시 탈락(캐시)
            pool.get('기본')
    else:
        warns, _ = gs.route_check(lg, route.rt)
        assert warns == 0


def test_path_waypoints_interior(outputs):
    """Path01 waypoint 는 laneSection 경계·도로 끝의 극단값이면 안 된다.

    실기 사고: 첫 waypoint 가 섹션 경계값(1222@137.0100 > 경계 137.00999)에
    놓여 VTD 가 "Path01 contains errors" 로 로드를 거부, 배치가 no-data 로 죽었다.
    검증된 시나리오처럼 주행 구간 안쪽 점이어야 한다.
    """
    import re
    lg = gs.LaneGraph(str(ROOT / 'data' / 'lane_graph.pkl'))
    xmls = sorted(outputs.glob('*/*.xml'))
    assert xmls
    for f in xmls:
        text = f.read_text(encoding='utf-8')
        m = re.search(r'<Path Name="Path01" PathId="1">(.*?)</Path>', text, re.S)
        assert m, f'{f.name}: Path01 없음'
        wps = [(int(tid), float(s))
               for s, tid in re.findall(r's="([^"]+)" TrackId="(\d+)"', m.group(1))]
        assert len(wps) >= 2
        for tid, s in wps:
            rd = lg.roads[tid]
            assert rd['junction'] == -1, \
                f'{f.name}: Track {tid} 은 교차로 연결로 (junction={rd["junction"]}) — ' \
                f'VTD 가 Path 생성을 거부한다 (실기 확인, 검증 시나리오는 전부 일반 도로)'
            assert 0.5 - 1e-6 <= s <= rd['length'] - 0.5 + 1e-6, \
                f'{f.name}: Track {tid} s={s} 가 [0.5, len-0.5] 밖 (len={rd["length"]:.3f})'
            for b in rd['sections']:
                if b > 0:
                    assert abs(s - b) > 0.05, \
                        f'{f.name}: Track {tid} s={s} 가 섹션 경계 {b:.3f} 위'


def test_ped_crossing_ends_off_roadway(outputs):
    """생성된 보행자 횡단 경로의 양 끝은 차도(반대차로 포함) 밖이어야 한다.

    2026-08-25 정지 고착: 횡단 폭 계산이 반대차로를 빼먹어 보행자가 반대 차선
    위에 멈췄고, 컨트롤러(차도 위 보행자 대기)가 172 s 재출발 불가 → 완주 실패.
    생성기와 planner 가 같은 lanegraph.roadway_edges 를 쓰는지 이 테스트가 지킨다.
    """
    import re
    import yaml
    lg = gs.LaneGraph(str(ROOT / 'data' / 'lane_graph.pkl'))
    checked = 0
    for yf in sorted(outputs.glob('보행자집중/*.yaml')):
        sdef = yaml.safe_load(yf.read_text(encoding='utf-8').split('\n', 1)[1])
        route = gs._build_from_rows(lg, sdef['route']['name'],
                                    [tuple(r) for r in sdef['route']['rows']])
        xml = yf.with_suffix('.xml').read_text(encoding='utf-8')
        m = re.search(r'<PathShape .*?</PathShape>', xml, re.S)
        assert m, f'{yf.name}: 보행자 PathShape 없음'
        wps = re.findall(r'X="([^"]+)" Y="([^"]+)"', m.group(0))
        ev = next(e for e in sdef['events'] if 'route_s' in e)
        _, k, sl = gs.lane_at(route.rt, ev['route_s'])
        left, right = lg.roadway_edges(k, sl)
        for x, y in (wps[0], wps[-1]):
            _, t_signed, _, _ = lg.project(k, float(x), float(y))
            edge = left if t_signed > 0 else right       # 좌 +, 우 − (lanegraph 규약)
            assert abs(t_signed) > edge + 0.4, \
                f'{yf.name}: 횡단 끝점 t={t_signed:+.2f} 가 차도 안 ' \
                f'(그쪽 가장자리 {edge:.2f} m — 사고 재현: 반대차선 위 정지 보행자)'
        checked += 1
    assert checked > 0, '보행자 시나리오가 없다'


def test_start_pool_conditions():
    """시작점 후보 풀: 일반 도로 주행 차선 + 가속 확보(successor 누적 허용) +
    전방 회전 가능. 임계는 params gen_coverage 가 단일 출처."""
    lg = gs.LaneGraph(str(ROOT / 'data' / 'lane_graph.pkl'))
    cov = gs.cov_cfg()
    min_len, accel = float(cov['start_min_lane_m']), float(cov['start_accel_m'])
    pool = gs.start_pool(lg)
    assert len(pool) >= 100, f'후보가 너무 적다: {len(pool)}'
    assert len({k[0] for k in pool}) >= 50                          # 여러 도로에 분포
    for k in pool[:20]:
        v = lg.lanes[k]
        assert v['type'] == 'driving' and v['junction'] == -1
        assert v['length'] >= min_len and v['next']
        assert gs._forward_clear_m(lg, k, accel) >= accel           # 가속 구간 확보
    assert gs.start_pool(lg) is pool                                # 1회 수집 캐시


def test_signal_phases_applied(outputs):
    """교차로집중 시나리오는 접근 컨트롤러의 Phase 가 실제로 바뀌어 있어야 한다."""
    import re
    import yaml
    ydir = outputs / '교차로집중'
    for yf in sorted(ydir.glob('*.yaml')):
        sdef = yaml.safe_load(yf.read_text(encoding='utf-8').split('\n', 1)[1])
        ev = next(e for e in sdef['events'] if e['kind'] == 'signal')
        xml_text = yf.with_suffix('.xml').read_text(encoding='utf-8')
        for ap in ev['approaches']:
            go, att, stop = ap['phases']
            for cid in ap['controllers']:
                m = re.search(rf'<SignalController Id="{cid}" .*?</SignalController>',
                              xml_text, re.S)
                assert m, f'{yf.name}: 컨트롤러 {cid} 없음'
                assert f'Duration="{go:.1f}" Type="go"' in m.group(0)
                assert f'Duration="{stop:.1f}" Type="stop"' in m.group(0)
