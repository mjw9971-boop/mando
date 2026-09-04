"""laneOffset 중복 s 타이브레이크 (작업18).

이 맵의 laneOffset 표에는 같은 지점을 두 레코드로 적어 놓은 곳이 51 군데다 —
앞 구간을 닫는 종결자(b=0)와 다음 구간을 여는 램프(b≠0). 둘의 s 가 부동소수
오차만큼(도로 2803 은 6.4e-12) 어긋나 있어 순수 s 정렬이 문서 순서를 뒤집고,
poly_eval / poly_eval_arr 이 "s 이하의 마지막 조각"을 고르는 탓에 종결자가
램프를 덮어쓴다. 결과는 중심선이 통째로 옆으로 밀리는 것이다 —
도로 2803 은 1.195 m(V 노치), 도로 1603 은 2.700 m(꺾임 없이 조용히).

여기서 지키는 계약:
  1. 같은 s 묶음 안에서는 xodr 문서 순서를 따른다.
  2. 킬 스위치 map.laneoffset_stable_order=false 면 이전 동작(순수 s 정렬).
  3. 산출된 lane_graph.pkl 에 그 노치가 남아 있지 않다.
"""
import math
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

import build_lane_graph as BLG                                  # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                     # noqa: E402

XODR = ROOT / 'data' / 'HL_FMA_VTD_LivingLab.xodr'
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
# 실측으로 확인된 노치 차로와 그 원인 도로 (작업18-1). road_s 구간은 노치 위치.
NOTCH_LANES = [(2801, 2, -2), (2801, 2, -1), (2801, 2, 2),
               (2803, 2, -2), (2803, 2, -1), (2803, 2, 2),
               (2445, 3, -4), (2445, 3, -3), (2445, 3, -2), (2445, 3, -1),
               (2445, 3, 2), (2445, 3, 3), (2445, 3, 4)]


@pytest.fixture(scope='module')
def road_els():
    import xml.etree.ElementTree as ET
    if not XODR.exists():
        pytest.skip('xodr 없음')
    return {int(r.get('id')): r for r in ET.parse(XODR).getroot().iter('road')}


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


def _offsets(road_el):
    return BLG.parse_pieces(road_el.find('lanes'), 'laneOffset', skey='s')


def _kink_deg(rec):
    """차로 중심선의 인접 세그먼트 간 최대 헤딩 변화 [deg]."""
    p = rec['pts'][:, :2]
    d = np.diff(p, axis=0)
    th = np.arctan2(d[:, 1], d[:, 0])
    if len(th) < 2:
        return 0.0
    dth = np.degrees((np.diff(th) + np.pi) % (2 * np.pi) - np.pi)
    return float(np.abs(dth).max())


def test_tie_keeps_document_order(road_els, monkeypatch):
    """도로 2803: 종결자(b=0)가 먼저, 램프(b=0.5833)가 뒤 — 문서 순서 그대로."""
    monkeypatch.setattr(BLG, '_MAP_CFG', (True, 1e-3))
    tie = [p for p in _offsets(road_els[2803]) if abs(p[0] - 25.951) < 1e-3]
    assert len(tie) == 2, tie
    assert tie[0][2] == pytest.approx(0.0)          # 종결자가 앞
    assert tie[1][2] == pytest.approx(0.583333, abs=1e-5)   # 램프가 뒤 = 이긴다


def test_ramp_wins_between_breakpoints(road_els, monkeypatch):
    """25.951~28.351 구간에서 오프셋이 1.9 고정이 아니라 램프를 탄다."""
    monkeypatch.setattr(BLG, '_MAP_CFG', (True, 1e-3))
    pieces = _offsets(road_els[2803])
    # 구간 끝 직전: 종결자면 1.9, 램프면 1.9 + 0.583333*2.4 = 3.3
    assert BLG.poly_eval(pieces, 28.35) == pytest.approx(3.3, abs=1e-3)
    assert BLG.poly_eval_arr(pieces, np.array([28.0]))[0] == pytest.approx(3.0953, abs=1e-3)


def test_kill_switch_restores_old_order(road_els, monkeypatch):
    """map.laneoffset_stable_order=false 면 순수 s 정렬 = 이전 동작."""
    monkeypatch.setattr(BLG, '_MAP_CFG', (False, 1e-3))
    pieces = _offsets(road_els[2803])
    assert BLG.poly_eval(pieces, 28.35) == pytest.approx(1.9, abs=1e-3)   # 종결자가 이긴다
    ss = [p[0] for p in pieces]
    assert ss == sorted(ss)


def test_no_reverse_flip_anywhere(road_els, monkeypatch):
    """교정이 '램프 → 종결자' 로 뒤집는 곳은 맵 전체에 하나도 없어야 한다.

    반대 방향 전환이 생기면 멀쩡한 구간을 새로 망가뜨린다는 뜻이다.
    """
    reverse = []
    for rid, el in road_els.items():
        monkeypatch.setattr(BLG, '_MAP_CFG', (False, 1e-3))
        old = _offsets(el)
        monkeypatch.setattr(BLG, '_MAP_CFG', (True, 1e-3))
        new = _offsets(el)
        if old == new:
            continue
        L = float(el.get('length'))
        for s in np.arange(0.0, L, 0.5):
            po, pn = BLG.piece_at(old, float(s)), BLG.piece_at(new, float(s))
            if po == pn:
                continue
            if abs(pn[2]) < 1e-12 <= abs(po[2]):     # 램프 → 종결자
                reverse.append((rid, float(s)))
    assert not reverse, f'역방향 전환 {len(reverse)} 건: {reverse[:5]}'


@pytest.mark.parametrize('key', NOTCH_LANES)
def test_notch_gone_in_graph(lg, key):
    """산출된 그래프에 V 노치가 남아 있지 않다 (꺾임 30°/샘플 초과 없음)."""
    if key not in lg.lanes:
        pytest.skip(f'{key} 없음')
    assert _kink_deg(lg.lanes[key]) < 30.0


def test_only_known_kinks_remain(lg):
    """맵 전체에서 30°/샘플 초과 꺾임은 도로 1927 뿐이다.

    1927 은 원인이 다르다 — lane 4 의 첫 width 조각이 sOffset 13.2 부터라
    그 앞이 폭 0 으로 평가되고 3.3 m 계단이 생긴다 (OpenDRIVE 규격 해석 별건,
    docs/BACKLOG.md). 여기에 다른 도로가 끼면 새 파싱 결함이라는 신호다.
    """
    bad = sorted({k[0] for k, r in lg.lanes.items() if _kink_deg(r) > 30.0})
    assert bad == [1927], bad
