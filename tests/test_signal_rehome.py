"""
대향 신호 재귀속 (build_lane_graph.rehome_signal).

2026-09-02 조사: 신호 절대방향(도로 hdg + hOffset)은 **그 신호가 관장하는 차량의
진행방향**과 같다 — 정상 신호 640개에서 hOffset 0 → +s 진행, pi → -s 진행이다.
이 규칙이 자기 도로와 어긋나는 신호는 지도 전체에서 6개뿐이고(hOffset = 3pi/2),
전부 교차로 내부 연결로(road 556 j2, road 791 j10)에 기록된 대향 신호다.

그대로 두면 신호가 교차로 내부 차로에 붙어 정지선을 못 만나고(sig_no_stopline),
상류 접근로 road 30·190 은 정지선만 있고 신호가 0개인 무신호 접근로가 된다 —
road 2819 가 controller 217 적신호를 무감속 통과한 것과 같은 실패다.
"""
import math
import xml.etree.ElementTree as ET

import pytest

XODR = 'data/HL_FMA_VTD_LivingLab.xodr'
AMBIG_ROADS = {556, 791}
AMBIG_SIGS = {30, 31, 34, 965, 966, 967}


def _mod():
    import sys
    sys.path.insert(0, 'tools')
    import build_lane_graph as B
    return B


# ── 전제: 모호 신호는 지도 전체에서 이 6개뿐 ─────────────────────────────
def test_ambiguous_hoffset_is_exactly_six_signals():
    root = ET.parse(XODR).getroot()
    amb = []
    for rd in root.iter('road'):
        for sg in rd.iter('signal'):
            h = float(sg.get('hOffset', 0.0)) % (2 * math.pi)
            if not (h < 0.5 or h > 2 * math.pi - 0.5 or abs(h - math.pi) < 0.5):
                amb.append((int(rd.get('id')), int(sg.get('id'))))
    assert {a[1] for a in amb} == AMBIG_SIGS
    assert {a[0] for a in amb} == AMBIG_ROADS


def test_ambiguous_signals_have_no_validity():
    """validity 가 있으면 그쪽이 이깁니다 — 모호 분기를 타는지 확인."""
    root = ET.parse(XODR).getroot()
    for rd in root.iter('road'):
        if int(rd.get('id')) not in AMBIG_ROADS:
            continue
        for sg in rd.iter('signal'):
            assert sg.find('validity') is None


def test_host_roads_had_stopline_but_no_signal():
    """뒤집힌 짝 — road 30·190 은 정지선만 있고 신호가 0개였다."""
    root = ET.parse(XODR).getroot()
    for rid in (30, 190):
        rd = [r for r in root.iter('road') if int(r.get('id')) == rid][0]
        assert len([o for o in rd.iter('object') if 'Rm_StopLine' in (o.get('name') or '')]) > 0
        assert len(list(rd.iter('signal'))) == 0


# ── 빌드 결과 ────────────────────────────────────────────────────────────
@pytest.fixture(scope='module')
def built():
    return _mod().build(XODR, ds=0.5)


def test_six_signals_reassigned(built):
    r = built['meta']['signal_reassigned']
    assert len(r) == 6
    assert {x['signal'] for x in r} == AMBIG_SIGS
    assert {x['from_road'] for x in r} == AMBIG_ROADS


def test_reassigned_to_road_30_and_190(built):
    r = {x['signal']: x for x in built['meta']['signal_reassigned']}
    for sid in (30, 31, 34):
        assert (r[sid]['to_road'], r[sid]['to_dir']) == (30, 1)
    for sid in (965, 966, 967):
        assert (r[sid]['to_road'], r[sid]['to_dir']) == (190, -1)


def test_heading_and_distance_are_well_inside_tolerance(built):
    """실측 오차는 1 deg / 27 m 미만이다. 임계(10 deg / 40 m)에 아슬아슬하지 않다."""
    for x in built['meta']['signal_reassigned']:
        assert abs(x['hdg_err_deg']) < 1.0
        assert x['dist_m'] < 30.0


def test_host_lane_gets_controller(built):
    """road 30 에 controller 3, road 190 에 312 가 붙는다."""
    want = {(30, 1): 3, (190, -1): 312}
    for (rid, d), cid in want.items():
        ks = [k for k, v in built['lanes'].items() if k[0] == rid and v['dir'] == d]
        assert ks
        hit = [k for k in ks
               if any(cid in (sl.get('controller_ids') or []) for sl in built['lanes'][k]['stop_lines'])]
        assert hit, f'road {rid} dir {d} 정지선에 controller {cid} 가 없다'
        for k in hit:
            assert built['lanes'][k]['signals'], '같은 차로에 신호도 붙어야 한다'


def test_junction_interior_lanes_lose_the_signal(built):
    """556·791 의 고아 차로 4개에서 신호가 빠진다 — 교차로 내부엔 정지선이 없다."""
    orphan = [(556, 0, -1), (556, 0, -2), (556, 0, -3), (791, 0, -1)]
    for k in orphan:
        assert built['lanes'][k]['signals'] == []
        assert built['lanes'][k]['stop_lines'] == []


def test_metrics_drop_by_exactly_the_reassigned_amount(built):
    """재귀속 6건 → sig_no_stopline 21→15, 고아 차로 15→11."""
    assert built['meta']['sig_no_stopline'] == 15
    assert built['meta']['lanes_signal_no_stopline'] == 11


def test_signal_total_unchanged(built):
    """재귀속은 신호를 옮길 뿐 만들거나 지우지 않는다."""
    root = ET.parse(XODR).getroot()
    n = sum(len(list(rd.iter('signal'))) for rd in root.iter('road'))
    seen = {s['id'] for v in built['lanes'].values() for s in v['signals']}
    assert AMBIG_SIGS <= seen
    assert n == 646


def test_normal_signals_stay_on_their_own_road(built):
    """hOffset 0 / pi 인 신호는 재귀속 분기를 타지 않는다."""
    moved = {x['signal'] for x in built['meta']['signal_reassigned']}
    assert moved == AMBIG_SIGS


# ── 유일성 조건 (오귀속 방지의 전부) ─────────────────────────────────────
def _line_road(B, rid, x, y, hdg, L, clusters=()):
    return {'id': rid, 'length': L,
            'geoms': [{'s': 0.0, 'length': L, 'kind': 'line', 'x': x, 'y': y, 'hdg': hdg}],
            '_stop_clusters': list(clusters)}


def _cluster(d, s, lanes=(('x',),)):
    return {'dir': d, 's': s, 'lanes': list(lanes), 'signal_ids': []}


def test_rehome_needs_exactly_one_candidate():
    """후보가 2개면 재귀속하지 않는다 — 애매하면 현행 폴백에 맡긴다."""
    B = _mod()
    host = _line_road(B, 1, 0.0, 0.0, 0.0, 100.0, [_cluster(1, 10.0)])
    twin = _line_road(B, 2, 0.0, 5.0, 0.0, 100.0, [_cluster(1, 10.0)])
    sig = {'id': 9, 's': 20.0, 't': 0.0, 'hOffset': 0.0}
    holder = _line_road(B, 3, 20.0, 0.0, 0.0, 100.0)
    assert B.rehome_signal(sig, holder, {1: host}) is not None
    assert B.rehome_signal(sig, holder, {1: host, 2: twin}) is None


def test_rehome_requires_signal_ahead_of_stopline():
    """대향 설치 — 정지선보다 진행 후방에 있는 신호는 후보가 아니다."""
    B = _mod()
    host = _line_road(B, 1, 0.0, 0.0, 0.0, 100.0, [_cluster(1, 50.0)])
    holder_front = _line_road(B, 3, 70.0, 0.0, 0.0, 10.0)   # 정지선 앞 20 m
    holder_back = _line_road(B, 3, 30.0, 0.0, 0.0, 10.0)    # 정지선 뒤 20 m
    sig = {'id': 9, 's': 0.0, 't': 0.0, 'hOffset': 0.0}
    assert B.rehome_signal(sig, holder_front, {1: host}) is not None
    assert B.rehome_signal(sig, holder_back, {1: host}) is None


def test_rehome_rejects_heading_mismatch():
    """진행방향이 10 deg 넘게 어긋나면 후보가 아니다."""
    B = _mod()
    host = _line_road(B, 1, 0.0, 0.0, math.radians(30), 100.0, [_cluster(1, 10.0)])
    holder = _line_road(B, 3, 20.0, 12.0, 0.0, 10.0)
    sig = {'id': 9, 's': 0.0, 't': 0.0, 'hOffset': 0.0}   # 절대방향 0 deg
    assert B.rehome_signal(sig, holder, {1: host}) is None


def test_rehome_skips_clusters_without_lanes():
    """차로가 안 붙은 정지선 클러스터는 후보에서 뺀다 — 옮겨봐야 사라진다."""
    B = _mod()
    host = _line_road(B, 1, 0.0, 0.0, 0.0, 100.0, [_cluster(1, 10.0, lanes=())])
    holder = _line_road(B, 3, 20.0, 0.0, 0.0, 10.0)
    sig = {'id': 9, 's': 0.0, 't': 0.0, 'hOffset': 0.0}
    assert B.rehome_signal(sig, holder, {1: host}) is None
