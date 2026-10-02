"""gen_placement.avoid_geom — 회피가 기하적으로 성립하는 자리에만 정적 장애물을 놓는다 (2026-10-01).

목적은 논문 실험용이다: 어떤 제어기로도·어떤 속도로도 지나갈 수 없는 자리만
생성 단계에서 뺀다. 조건은 A(이웃 차로 연속 길이) · D(연쇄 엇갈림) · E(출발 거리) ·
F(회피 창 겹침) 넷이고 각각 킬스위치가 있다.

여기서 지키는 것:
  · 스위치를 전부 끄면 산출물은 블록이 없을 때와 바이트 동일하다.
  · 길이 상수를 새로 두지 않는다 — 전부 기존 params 키에서 계산한다.
  · [0] 실측 표본의 판정이 유지된다: (173,0,2) 의 포켓 차로 직전 자리는 A 에 걸리고,
    같은 차로 35 m 뒤(무신호 정지선 구역 — 구현 특이 실패)는 걸리지 않는다.
  · 켜면 시나리오마다 <이름>.placement.json 이 남고, 기록된 배치는 전부 통과다.

스위치를 켜고 끄는 검사는 params 기본값을 읽지 않고 **사본에서 명시적으로 켜고 끈다**
(on_cfg) — params 값이 정본이고 기본값이 바뀌어도 깨지지 않는다 (2026-10-01 현재
A·E 켬 / D·F 끔, docs/BACKLOG.md B-35).
"""
import copy
import json
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import gen_scenarios as gs                                      # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
THEMES = ['정적회피집중', '연쇄장애물']
ARGS = ['--count', '3', '--seed', '0']


def on_cfg(monkeypatch, conds='ADEF'):
    """avoid_geom 스위치만 사본에서 켠다 (params 기본값과 무관)."""
    cfg = copy.deepcopy(gs.plc_cfg())
    for c, key in gs._AVOID_KEYS.items():
        cfg['avoid_geom'][key] = c in conds
    monkeypatch.setattr(gs, '_PLC_CFG', cfg)
    return gs.avoid_cfg()


@pytest.fixture(scope='module')
def lg():
    return gs.LaneGraph(str(GRAPH))


def _xml_bytes(out):
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in sorted(out.glob('*/*.xml'))}


# ── 기본값 · 상수 출처 ───────────────────────────────────────────────────
def test_switches_present_and_all_off_is_inert(monkeypatch):
    """네 스위치가 params 에 bool 로 있고, 전부 끄면 파생 길이조차 읽지 않는다."""
    g = gs.plc_cfg()['avoid_geom']
    assert all(isinstance(g[k], bool) for k in gs._AVOID_KEYS.values())
    assert on_cfg(monkeypatch, '') == {'on': dict.fromkeys('ADEF', False), 'any': False}


def test_missing_block_is_off(monkeypatch):
    """블록이 없는 설정(다른 테스트가 _PLC_CFG 를 좁혀 쓰는 경우)도 꺼진 것으로 본다."""
    monkeypatch.setattr(gs, '_PLC_CFG', {'static_vehicle': gs.plc_cfg()['static_vehicle']})
    assert gs.avoid_cfg()['any'] is False


def test_lengths_come_from_existing_keys(monkeypatch):
    av = on_cfg(monkeypatch)
    full = gs.full_cfg()
    c24 = full['ctrl24']
    v_ref = float(c24['span_v_req_min'])
    assert av['v_ref'] == v_ref
    assert av['trans'] == max(float(c24['trans_min_m']), float(c24['trans_k']) * v_ref)
    assert (av['xb'], av['xa']) == (float(c24['extra_before_m']), float(c24['extra_after_m']))
    assert av['min_w'] == float(gs.plc_cfg()['static_vehicle']['min_neighbor_width_m'])
    assert av['corr'] == pytest.approx(
        float(full['vehicle']['width']) + 2 * float(full['percep']['obstacle_clearance_m']))
    assert av['start_min'] == pytest.approx(
        float(c24['detect_max_m']) + v_ref ** 2 / (2 * float(full['speed']['a_max'])))


# ── A: 이웃 차로 연속 길이 (장애물 차로 계보) ────────────────────────────
LANE_173 = (173, 0, 2)      # 왼쪽 포켓 차로가 차로 시작 바로 앞에서 열린다 ([0] 실측)


def _veh_eval(lg, av, k, sl, s0=500.0, bodies=()):
    x, y, _z, _h = lg.point_at(k, sl)
    return gs._avoid_eval(lg, av, k, sl, s0, 0.0, gs.CAR_LEN / 2.0,
                          [(x, y, gs.CAR_W / 2.0)], list(bodies), cap=300.0)


def test_A_rejects_spot_right_after_pocket_lane_opens(lg, monkeypatch):
    """정적회피집중_05_좌회전6 (2026-09-03 blocked): 이웃이 장애물 중심 10.6 m 앞에서야 열린다."""
    av = on_cfg(monkeypatch)
    res = _veh_eval(lg, av, LANE_173, 9.6)
    left = res['sides']['left']
    assert res['fail'] == 'A' and not res['passed']['A']
    assert left['present'] and not left['ok']
    assert left['back'] < res['need'][-1] <= left['back'] + 2.0      # 1 m 안팎 모자란다
    assert not res['sides']['right']['present']


def test_A_keeps_implementation_specific_failure_spot(lg, monkeypatch):
    """정적회피집중_02_직진3 (무신호 정지선 9 m 뒤 — kr_rules span_into_zone 기각):
    기하로는 성립하므로 거르면 안 된다."""
    av = on_cfg(monkeypatch)
    res = _veh_eval(lg, av, LANE_173, 44.2)
    assert res['fail'] is None
    assert res['sides']['left']['ok'] and res['free_sides'] == ['left']


def test_A_follows_obstacle_lane_not_route_lane(lg, monkeypatch):
    """추월집중_01_우회전0: 경로는 (2423,6,3) → (2423,6,2) 로 계획 차선변경을 하지만
    장애물 차로의 계보에는 왼쪽 이웃이 100 m 넘게 이어진다 — 경로 차로 기준이면 걸린다."""
    av = on_cfg(monkeypatch)
    res = _veh_eval(lg, av, (2423, 6, 3), 51.5)
    assert res['fail'] is None
    assert res['sides']['left']['front'] > 100.0


def test_A_off_does_not_reject(lg, monkeypatch):
    av = on_cfg(monkeypatch, 'DEF')
    res = _veh_eval(lg, av, LANE_173, 9.6)
    assert res['fail'] is None and res['passed']['A'] is False      # 값은 재되 판정에 안 쓴다


# ── E: 출발 거리 ─────────────────────────────────────────────────────────
def test_E_start_distance(lg, monkeypatch):
    av = on_cfg(monkeypatch)
    half = gs.CAR_LEN / 2.0
    near = _veh_eval(lg, av, LANE_173, 44.2, s0=av['start_min'] + half - 0.1)
    far = _veh_eval(lg, av, LANE_173, 44.2, s0=av['start_min'] + half)
    assert near['fail'] == 'E' and far['fail'] is None


# ── F: 회피 창 겹침 ──────────────────────────────────────────────────────
def test_F_body_in_neighbor_lane_blocks_only_side(lg, monkeypatch):
    av = on_cfg(monkeypatch)
    nb = lg.neighbor(LANE_173, 'left')
    x, y, _z, _h = lg.point_at(nb, 44.2 + 8.0)                       # 창 안, 이웃 차로 한가운데
    body = (x, y, gs.CAR_LEN / 2.0, gs.CAR_W / 2.0, 'Parked01')
    res = _veh_eval(lg, av, LANE_173, 44.2, bodies=[body])
    assert res['fail'] == 'F' and res['sides']['left']['hits'] == ['Parked01']
    x, y, _z, _h = lg.point_at(nb, 44.2 + 40.0)                      # 창 밖
    res = _veh_eval(lg, av, LANE_173, 44.2, bodies=[(x, y, body[2], body[3], 'Parked01')])
    assert res['fail'] is None


def test_F_is_symmetric(lg, monkeypatch):
    """나중에 놓이는 물체가 먼저 놓인 이벤트의 창을 막아도 버린다."""
    av = on_cfg(monkeypatch)
    res = _veh_eval(lg, av, LANE_173, 44.2)

    class Ctx:
        pass
    ctx = Ctx()
    ctx.lg = lg
    ctx.avoid_evs = [{'name': 'StoppedCar01',
                      'sides': {sd: res['sides'][sd]['segs'] for sd in res['free_sides']}}]
    nb = lg.neighbor(LANE_173, 'left')
    x, y, _z, _h = lg.point_at(nb, 44.2 - 6.0)
    ok, _ = gs._avoid_reverse(ctx, [(x, y, 0.25, 0.25, '')])
    assert ok is False
    x, y, _z, _h = lg.point_at(LANE_173, 44.2 - 6.0)                # 자차로 안 — 이웃 창 밖
    ok, evs = gs._avoid_reverse(ctx, [(x, y, 0.25, 0.25, '')])
    assert ok is True and list(evs[0]['sides']) == ['left']


# ── D: 연쇄 엇갈림 ───────────────────────────────────────────────────────
def _chain(lg, k, sl, n, lanes=None, spacing=18.0, half=0.25):
    """ev_obstacle_chain 과 같은 엇갈림 (차로 가장자리 0.7 m 안쪽, 좌우 번갈아)."""
    st = []
    for i in range(n):
        kk = k if lanes is None else lanes[i]
        x, y, _z, h = lg.point_at(kk, sl + i * spacing)
        t = (lg.width_at(kk, sl + i * spacing) / 2.0 - 0.7) * (1 if i % 2 == 0 else -1)
        st.append((x - t * math.sin(h), y + t * math.cos(h), half))
    return st


def test_D_neighbor_lane_is_the_free_corridor(lg, monkeypatch):
    av = on_cfg(monkeypatch)
    width, lane = gs._chain_free_lane(lg, LANE_173, 60.0, _chain(lg, LANE_173, 60.0, 3),
                                      [0.0, 18.0, 36.0])
    assert tuple(lane) == tuple(lg.neighbor(LANE_173, 'left'))
    assert width >= av['corr']


def test_D_own_lane_common_gap_is_too_narrow(lg, monkeypatch):
    """엇갈린 체인은 자차로의 공통 빈 구간이 (차로폭 − 1.9 m) 뿐이다 — 이웃이 없으면 걸린다."""
    av = on_cfg(monkeypatch)
    k = (2818, 6, -1)                                               # 1차로 (params obstacle_chain 주석)
    assert lg.neighbor(k, 'left') is None or \
        lg.lanes[lg.neighbor(k, 'left')]['dir'] != lg.lanes[k]['dir']
    width, _lane = gs._chain_free_lane(lg, k, 20.0, _chain(lg, k, 20.0, 2), [0.0, 18.0])
    assert width == pytest.approx(lg.width_at(k, 20.0) - 1.9, abs=0.25)
    assert width < av['corr']


def test_D_chain_alternating_across_both_lanes_fails(lg, monkeypatch):
    """두 차로에 번갈아 놓인 물체(수동 배치 run_20260903_1003xx 형태)는 어느 차로도 비지 않는다."""
    av = on_cfg(monkeypatch)
    nb = lg.neighbor(LANE_173, 'left')
    st = []
    for i, kk in enumerate((LANE_173, nb, LANE_173)):
        x, y, _z, _h = lg.point_at(kk, 60.0 + i * 18.0)
        st.append((x, y, gs.CAR_W / 2.0))                           # 각 차로 정중앙의 차폭 물체
    width, _lane = gs._chain_free_lane(lg, LANE_173, 60.0, st, [0.0, 18.0, 36.0])
    assert width < av['corr']


# ── 생성: 끄면 바이트 동일 · 켜면 기록이 남고 전부 통과 ──────────────────
@pytest.fixture(scope='module')
def gen_off(tmp_path_factory):
    """스위치를 전부 **명시적으로 끄고** 만든 기준 산출물 (모듈 픽스처라 monkeypatch 대신 직접 복원)."""
    out = tmp_path_factory.mktemp('avoid_off')
    saved = gs._PLC_CFG
    cfg = copy.deepcopy(gs.plc_cfg())
    for key in gs._AVOID_KEYS.values():
        cfg['avoid_geom'][key] = False
    gs._PLC_CFG = cfg
    try:
        assert gs.main(THEMES + ARGS + ['--out-dir', str(out)]) == 0
    finally:
        gs._PLC_CFG = saved
    return out


def test_off_matches_config_without_block(gen_off, tmp_path, monkeypatch):
    """avoid_geom 블록을 통째로 뺀 설정(= 이 기능 이전)과 XML·정의가 바이트 동일하다."""
    cfg = {k: v for k, v in copy.deepcopy(gs.plc_cfg()).items() if k != 'avoid_geom'}
    cfg['obstacle_chain'].pop('object_extent_m')
    monkeypatch.setattr(gs, '_PLC_CFG', cfg)
    assert gs.main(THEMES + ARGS + ['--out-dir', str(tmp_path)]) == 0
    assert _xml_bytes(tmp_path) == _xml_bytes(gen_off)
    for p in sorted(gen_off.glob('*/*.yaml')):
        assert (tmp_path / p.relative_to(gen_off)).read_bytes() == p.read_bytes()
    assert not list(gen_off.glob('*/*.placement.json'))


def test_on_writes_records_and_all_pass(tmp_path, monkeypatch, capsys):
    on_cfg(monkeypatch)
    assert gs.main(THEMES + ARGS + ['--out-dir', str(tmp_path), '--coverage-report']) == 0
    out = capsys.readouterr().out
    assert '[배치 조건] gen_placement.avoid_geom  A=on D=on E=on F=on' in out
    xmls = sorted(tmp_path.glob('*/*.xml'))
    assert len(xmls) == 6                                           # 목표 개수를 채운다
    # 판정 기록은 *.yaml 이 아니다 — batch 목록(<주제>/*.yaml 전수)이 오염되지 않는다.
    assert len(json.loads((tmp_path / 'batch_all.json').read_text(encoding='utf-8'))) == 6
    n_ev = 0
    for x in xmls:
        rec = json.loads((x.parent / (x.stem + '.placement.json')).read_text(encoding='utf-8'))
        assert rec['switches'] == dict.fromkeys('ADEF', True)
        for e in rec['events']:
            assert e['applies']
            n_ev += 1
            assert e['A']['pass'] and e['E']['pass'] and e['F']['pass']
            side = e['A']['left'] if e['A']['left']['ok'] else e['A']['right']
            assert side['back_m'] >= e['A']['need_back_m'] - 1e-6
            assert side['front_m'] >= e['A']['need_front_m'] - 1e-6
            assert e['E']['near_edge_s_m'] >= e['E']['need_m']
            if e['kind'] == 'obstacle_chain':
                assert e['D']['applies'] and e['D']['pass'] and e['D']['free_m'] >= e['D']['need_m'] - 0.01
            else:
                assert e['D']['applies'] is False
    assert n_ev == 3 * 2 + 3                                        # 정적회피집중 2건씩 + 연쇄 1건씩


def test_on_keeps_placements_that_already_pass(gen_off, tmp_path, monkeypatch):
    """조건이 건드리지 않는 시나리오는 스위치를 켜도 XML 이 같다 (옮긴 것만 다르다).

    폐기·백필이 끼면 뒤 순번의 경로가 달라지므로, 같은 이름에 **같은 경로**가 나온
    시나리오만 비교한다 (seed·테마가 바뀌어도 깨지지 않게).
    """
    import yaml
    on_cfg(monkeypatch)
    assert gs.main(THEMES + ARGS + ['--out-dir', str(tmp_path)]) == 0
    off, on = _xml_bytes(gen_off), _xml_bytes(tmp_path)
    compared = 0
    for p in sorted(tmp_path.glob('*/*.placement.json')):
        rec = json.loads(p.read_text(encoding='utf-8'))
        key = f'{p.parent.name}/{rec["name"]}.xml'
        y_off = gen_off / p.parent.name / f'{rec["name"]}.yaml'
        if not y_off.exists():
            continue
        rows_on = yaml.safe_load((p.parent / f'{rec["name"]}.yaml').read_text(encoding='utf-8'))
        rows_off = yaml.safe_load(y_off.read_text(encoding='utf-8'))
        if rows_on['route']['rows'] != rows_off['route']['rows']:
            continue
        compared += 1
        if not any(e['candidates']['moved'] for e in rec['events']):
            assert on[key] == off[key], key
    assert compared >= 1
