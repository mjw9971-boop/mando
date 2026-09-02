"""
정지선 좌표 위생 검사 (build_lane_graph.assign_objects) — 관측 전용.

2026-09-01 실사고: xodr 정지선 24개(13 road)가 s=0.0 · |t|≈도로길이 로 깨져 있다.
road 2819 는 6개 전부가 손상이고 **일방통행**(주행차로가 전부 dir=+1)인데 손상
레코드의 hdg 가 dir=-1 을 가리켜, t 폴백(lanes_of_dir_at)까지 공집합이 됐다.
그 결과 정지선이 그래프에서 조용히 사라졌고 controller 217 적신호를 무감속
통과했다 — 어떤 통계에도 잡히지 않았다.

검출 기준은 **절대 t** 다. |t|/L 비율을 1차로 쓰면
  · 짧은 도로의 정상 정지선을 오탐한다 (L=6.09 에 t=7.5 가 3차선 중심)
  · L=13~20 m 손상 레코드를 놓친다 (|t|/L 이 1.03 까지 벌어진다)
실측: 0.99~1.01 창은 24개 중 19개만 잡았고, 절대 t 는 24/24 · 오탐 0 이었다.
"""
import xml.etree.ElementTree as ET

import pytest

XODR = 'data/HL_FMA_VTD_LivingLab.xodr'
STOP = 'Rm_StopLine'
BAD_ROADS = {1043, 1603, 2077, 2096, 2097, 2115, 2116, 2139,
             2152, 2172, 2195, 2241, 2819}


def stop_objects():
    root = ET.parse(XODR).getroot()
    out = []
    for rd in root.iter('road'):
        L = float(rd.get('length'))
        for o in rd.iter('object'):
            if STOP in (o.get('name') or ''):
                out.append((int(rd.get('id')), L, float(o.get('s')), float(o.get('t'))))
    return out


# ── 손상 레코드의 성질 (보정 작업의 전제) ────────────────────────────────
def test_damaged_records_are_exactly_24():
    """s==0 이고 |t| 가 도로 길이에 붙는 레코드가 정확히 24개, 13 road."""
    bad = [r for r in stop_objects() if r[2] == 0.0 and abs(abs(r[3]) - r[1]) < 0.05 * r[1]]
    assert len(bad) == 24
    assert {r[0] for r in bad} == BAD_ROADS


def test_s_zero_alone_is_not_the_criterion():
    """s==0 인 정상 정지선이 있다 (road 2802/2805, t=±1.5) — s 만으로 가르면 오탐."""
    z = [r for r in stop_objects() if r[2] == 0.0]
    normal = [r for r in z if abs(r[3]) < 3.0]
    assert normal, 's==0 이지만 t 가 정상인 레코드가 있어야 한다'
    assert {r[0] for r in normal}.isdisjoint(BAD_ROADS)


def test_ratio_window_would_miss_some():
    """|t|/L 0.99~1.01 창은 24개를 다 못 잡는다 — 절대 t 를 1차로 쓰는 근거."""
    bad = [r for r in stop_objects() if r[2] == 0.0 and abs(abs(r[3]) - r[1]) < 0.05 * r[1]]
    caught = [r for r in bad if 0.99 <= abs(r[3]) / r[1] <= 1.01]
    assert len(caught) < len(bad), '비율 창이 전수를 잡으면 이 테스트의 전제가 바뀐 것'


# ── 빌드 결과: 보정과 잔여 경고 ─────────────────────────────────────────
@pytest.fixture(scope='module')
def built():
    import sys
    sys.path.insert(0, 'tools')
    import build_lane_graph as B
    return B.build(XODR, ds=0.5)


def road_of(w):
    """경고 문자열에서 road 번호 — 'stopline road 1043 dir -1: …' / 'road 1043: …'"""
    tok = w.split()[2].rstrip(':')
    return int(tok)


def test_all_damaged_records_repaired(built):
    """24건 전부 보정된다 — 무음 통과도, 무음 폐기도 없다."""
    assert built['meta']['stop_repaired'] == 24
    assert built['meta']['stop_repair_undecided'] == 0


def test_repair_direction_is_plus_one_everywhere(built):
    """0(덮인 방향 배제)·1(일방통행 배제)로 24/24 가 dir=+1 로 결정된다."""
    rep = built['meta']['stop_repairs']
    assert len(rep) == 24
    assert {r['dir_fixed'] for r in rep} == {1}
    assert {r['road'] for r in rep} == BAD_ROADS


def test_repair_s_is_min_abs_t_and_length(built):
    """s_true = min(|t|, L) — 13 road 전부에서 도로 끝을 집는다."""
    for r in built['meta']['stop_repairs']:
        assert r['s_orig'] == 0.0
        assert abs(r['s_true'] - min(abs(r['t_orig']), r['road_length'])) < 1e-6


def test_road_2819_now_has_stopline_with_controller_217(built):
    """실사고 지점 — 일방통행이라 hdg 방향에 차로가 없어 사라졌던 정지선."""
    sl = [s for k, v in built['lanes'].items() if k[0] == 2819
          for s in (v.get('stop_lines') or [])]
    assert sl, 'road 2819 에 정지선이 붙어야 한다'
    assert any(217 in (s.get('controller_ids') or []) for s in sl)


def test_orphan_signal_lanes_reduced(built):
    """신호는 있는데 정지선이 없는 차로가 줄어든다 (보정 전 27 → 15)."""
    assert built['meta']['lanes_signal_no_stopline'] < 27


def test_t_remains_unrecoverable_and_warned(built):
    """t 는 복원 불가다 — 보정 후에도 범위 이탈 경고는 남아야 한다."""
    w = [x for x in built['meta']['warnings'] if x.startswith('stopline ') and 't 범위' in x]
    assert {road_of(x) for x in w} == BAD_ROADS


def test_ratio_reported_as_secondary(built):
    """비율은 보조 지표로 메시지에 남는다 (판정은 절대 t)."""
    w = [x for x in built['meta']['warnings'] if x.startswith('stopline ') and 't 범위' in x]
    assert w and all('|t|/L=' in x for x in w)


def test_cross_check_mismatch_warns_but_does_not_override(built):
    """2·3 은 교차검증 전용 — 불일치해도 판정(+1)을 뒤집지 않고 경고만."""
    w = [x for x in built['meta']['warnings'] if '교차검증 불일치' in x]
    if w:                                   # 실측: road 1043·2097·2172 (신호가 반대편 것)
        for r in built['meta']['stop_repairs']:
            assert r['dir_fixed'] == 1, '교차검증이 판정을 뒤집으면 안 된다'


def test_untouched_records_not_moved(built):
    """조건 밖(정상 정지선)은 건드리지 않는다 — s==0 이지만 t 가 정상인 것 포함."""
    moved = {(r['road'], r['t_orig']) for r in built['meta']['stop_repairs']}
    for rid, L, s, tt in stop_objects():
        if not (s == 0.0 and abs(abs(tt) - L) < 0.05 * L):
            assert (rid, round(tt, 2)) not in moved
