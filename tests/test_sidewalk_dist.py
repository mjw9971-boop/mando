"""
lg.sidewalk_dist_at — 보도 안쪽 경계까지 횡거리 (항목 5 보도 침범 판정 데이터).

실제 xodr 검증값 (2026-08-27 실측, tools/build_lane_graph.py 확장):
  · (30, 0, 1)  우측: 반폭 1.5 + border 0.5 = 2.0 m (driving–border–sidewalk 표준 배치)
                좌측: 반대 차선 건너편 보도까지 5.0 m — 좌/우가 달라야 한다
  · (308, 0, -1) 보도 없는 도로 — 양쪽 None (판정 대상 아님)
독립 경로 대조: 같은 값을 build_lane_graph.lane_bounds_at (xodr 폭 다항식 직접
평가)로 재계산해 일치를 확인한다 — 그래프 저장 경로와 계산 경로가 분리된 검증.
"""
import pathlib
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from vtd_adapter.lanegraph import LaneGraph

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
XODR = ROOT / 'data' / 'HL_FMA_VTD_LivingLab.xodr'

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and XODR.exists()),
                                reason='lane_graph.pkl / xodr 없음')

STD_KEY = (30, 0, 1)          # driving–border(0.5)–sidewalk 표준 배치
STD_RIGHT_M = 2.0             # 반폭 1.5 + border 0.5
STD_LEFT_M = 5.0              # 반대 차선(3.0) 건너 보도
NONE_KEY = (308, 0, -1)       # 보도 없는 도로


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


def _expected_from_xodr(key, s_in_lane, side):
    """xodr 폭 다항식 직접 평가(lane_bounds_at)로 같은 값을 독립 재계산."""
    from build_lane_graph import lane_bounds_at, parse_road
    root = ET.parse(str(XODR)).getroot()
    road_el = next(r for r in root.findall('road') if int(r.get('id')) == key[0])
    road = parse_road(road_el)
    rec = pickle.load(open(GRAPH, 'rb'))['lanes'][key]
    road_s = float(np.interp(s_in_lane, rec['s'], rec['road_s']))
    _i, b = lane_bounds_at(road, road_s)
    lid = key[2]
    tc = 0.5 * (b[lid][0] + b[lid][1])
    d = 1 if lid < 0 else -1
    sign = (1 if d == 1 else -1) if side == 'left' else (-1 if d == 1 else 1)
    edges = [min(tin, tout) if sign > 0 else max(tin, tout)
             for l, (tin, tout, typ) in b.items()
             if typ == 'sidewalk' and (0.5 * (tin + tout) - tc) * sign > 0]
    assert edges, f'{key} {side}: xodr 에 그 쪽 보도가 없다'
    return min((e - tc) * sign for e in edges)


def test_standard_border_road(lg):
    """표준 배치: 값 = 차로 반폭 + 사이 border 폭 (xodr 독립 계산과 일치)."""
    s = 0.5 * lg.length(STD_KEY)
    got = lg.sidewalk_dist_at(STD_KEY, s, 'right')
    assert got == pytest.approx(STD_RIGHT_M, abs=0.02)
    assert got == pytest.approx(_expected_from_xodr(STD_KEY, s, 'right'), abs=0.02)


def test_sides_differ(lg):
    """좌측은 반대 차선 건너편 보도 — 우측과 달라야 하고 역시 xodr 합과 일치."""
    s = 0.5 * lg.length(STD_KEY)
    left = lg.sidewalk_dist_at(STD_KEY, s, 'left')
    assert left == pytest.approx(STD_LEFT_M, abs=0.02)
    assert left == pytest.approx(_expected_from_xodr(STD_KEY, s, 'left'), abs=0.02)
    assert left != pytest.approx(lg.sidewalk_dist_at(STD_KEY, s, 'right'), abs=0.5)


def test_no_sidewalk_is_none(lg):
    """보도 없는 도로는 None — score 쪽 '판정 대상 아님'이 명시적이어야 한다."""
    s = 0.5 * lg.length(NONE_KEY)
    assert lg.sidewalk_dist_at(NONE_KEY, s, 'left') is None
    assert lg.sidewalk_dist_at(NONE_KEY, s, 'right') is None


def test_old_pkl_field_missing_returns_none(lg):
    """구 pkl 호환: 레코드에 신규 필드가 없으면 None (rec.get 방어)."""
    fake = ('_old', 0, 1)
    rec = {k: v for k, v in lg.lanes[STD_KEY].items()
           if k not in ('sidewalk_left_m', 'sidewalk_right_m')}
    lg.lanes[fake] = rec
    try:
        assert lg.sidewalk_dist_at(fake, 1.0, 'right') is None
    finally:
        del lg.lanes[fake]


def test_unknown_key_raises(lg):
    """width_at 과 같은 관례 — 없는 차로는 KeyError."""
    with pytest.raises(KeyError):
        lg.sidewalk_dist_at((99999, 0, 1), 0.0, 'right')
