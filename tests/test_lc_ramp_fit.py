"""
실주행 1차 [3] — 연속 차선변경 구간 (램프 맞춤 + 속도 상한).

실측 logs/batch/20260908_130919/실경로_01_PathShape03 rs 413~455: 도로 418 에서
차선변경 4회가 연속인데
  · 자차가 그 구간에서 8 → 13.3 m/s 로 **가속**했고 (붉은 구간이 rs 290 에서
    끝나 제한 50),
  · 램프 길이는 v × lc_move_s 라 39 m 로 구워졌는데 hop 간격은 20 m 였다.
램프가 겹쳐 경로가 2차로를 한 번에 건너는 S자가 됐다 — heading_err +31°,
조향 포화 4회, |t_off| 1.85 m (항목 3 차로유지 중대 + 항목 6 실선 변경).

둘 다 있어야 한다:
  · 속도만 낮추면 램프는 이미 구워진 채고,
  · 램프만 줄이면 48 km/h 에 17 m 램프는 요구 횡가속 5 m/s² 라 더 흔들린다.
"""
import copy
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
CAP_V = CFG['speed']['lc_speed_cap_kph'] / 3.6
SEP = CFG['speed']['lc_hop_chain_sep_m']
A_STOP = CFG['speed']['stop_profile_a']
GAP = CFG['route']['lc_ramp_hop_gap_m']


def on_cfg(**over):
    c = copy.deepcopy(CFG)
    c['speed']['lc_speed_cap_enable'] = True
    c['speed'].update(over)
    return c


def off_cfg(**over):
    """이전 동작 사본 — 기본값을 읽지 않는다 (2026-09-07 드리프트 원칙)."""
    c = copy.deepcopy(CFG)
    c['speed']['lc_speed_cap_enable'] = False
    c['speed'].update(over)
    return c


class P:
    """route_s 축만 있는 최소 planner 목 — LC 창은 직접 준다."""

    def __init__(self, wins, rs=0.0, n=12000, ppm=10):
        self.points_per_meter = ppm
        self.route_s = np.arange(n) / ppm
        self.route_index = int(rs * ppm)
        self.lc_ramps = list(wins)


# ── (a) 속도 상한 ─────────────────────────────────────────────────────────
def test_off_makes_no_candidate():
    assert KrRules(off_cfg())._lc_speed_cap(P([(100.0, 120.0)], rs=110.0)) is None


def test_no_windows_makes_no_candidate():
    assert KrRules(on_cfg())._lc_speed_cap(P([], rs=110.0)) is None


def test_inside_window_is_capped():
    assert KrRules(on_cfg())._lc_speed_cap(
        P([(100.0, 120.0)], rs=110.0)) == pytest.approx(CAP_V)


def test_before_window_decelerates_on_the_stopline_profile():
    """창 진입 전 감속은 정지선 프로파일과 **같은 식**이다: √(cap² + 2·a·d)."""
    d = 10.0
    v = KrRules(on_cfg())._lc_speed_cap(P([(100.0, 120.0)], rs=100.0 - d))
    assert v == pytest.approx(math.sqrt(CAP_V ** 2 + 2 * A_STOP * d))
    assert v > CAP_V                                   # 아직 창 밖이라 더 빠르다


def test_far_before_window_makes_no_candidate():
    """창이 lc_speed_cap_look_m 밖이면 후보도 진단도 만들지 않는다.

    접근 프로파일은 멀면 값이 커서 어차피 min() 에 지지만, 후보를 계속 만들면
    진단이 전 구간에 찍혀 "언제 실제로 걸렸나" 를 못 읽는다 (실측: 거리 제한
    없이 두니 18,868틱 중 18,017틱에 후보가 존재했다).
    """
    look = CFG['speed']['lc_speed_cap_look_m']
    kr = KrRules(on_cfg())
    assert kr._lc_speed_cap(P([(100.0, 120.0)], rs=100.0 - look - 5.0)) is None
    assert kr.last_lc_cap is None
    assert kr._lc_speed_cap(P([(100.0, 120.0)], rs=100.0 - look + 5.0)) is not None


def test_after_all_windows_no_candidate():
    assert KrRules(on_cfg())._lc_speed_cap(P([(100.0, 120.0)], rs=200.0)) is None


def test_chained_hops_hold_the_cap_between_windows():
    """연속 hop 사이에서 가속하면 다음 램프가 다시 길어져 겹친다 — 유지한다."""
    kr = KrRules(on_cfg())
    wins = [(100.0, 118.0), (118.0 + SEP - 1.0, 140.0)]      # 사이 간격 < SEP
    assert kr._lc_speed_cap(P(wins, rs=119.0)) == pytest.approx(CAP_V)


def test_far_apart_hops_are_not_chained():
    kr = KrRules(on_cfg())
    wins = [(100.0, 118.0), (118.0 + SEP + 10.0, 160.0)]      # 사이 간격 > SEP
    v = kr._lc_speed_cap(P(wins, rs=119.0))
    assert v > CAP_V                                   # 창 밖 — 감속 프로파일


# ── (b) 램프를 다음 hop 안으로 맞춘다 ─────────────────────────────────────
# 실측(실경로_01_PathShape03, 도로 418): hop 이 20 m 간격인데 램프가 37.5 m 로
# 구워져 겹쳤다. 켜면 18.0 m 로 잘려 겹침이 사라지고, 따라가는 경로의 최소
# 곡률반경이 **3.0 m → 20.6 m** 로 회복된다.

def route_cfg(on):
    c = copy.deepcopy(CFG)
    c['route']['lc_ramp_fit_hop_enable'] = bool(on)
    return c


class FakeLG:
    """직선 도로 하나에 이웃 차로만 있는 최소 목 — 램프 창 계산만 본다."""

    def __init__(self, kph=50.0):
        self.kph = kph

    def speed_limit_at(self, key):
        return self.kph, None

    def neighbor(self, key, side):
        return (1, 0, key[2] + 1) if side == 'left' else None


def build_ramps(cfg, hop_gap_m):
    """hop_gap_m 간격으로 세 번 왼쪽으로 옮기는 계획 → 램프 창 목록."""
    from vtd_adapter.route import VtdRoutePlanner
    p = object.__new__(VtdRoutePlanner)
    p.lg = FakeLG()
    p.cfg = cfg
    rt = cfg['route']
    p.lc_move_s = float(rt['lc_move_s'])
    p.lc_move_min_m = float(rt['lc_move_min_m'])
    p.lc_move_max_m = float(rt['lc_move_max_m'])
    p.lc_ramp_fit_hop = bool(rt.get('lc_ramp_fit_hop_enable', False))
    p.lc_ramp_hop_gap_m = float(rt.get('lc_ramp_hop_gap_m', 2.0))
    p.lc_ramps = []
    lanes = [(1, 0, -4), (1, 0, -3), (1, 0, -2), (1, 0, -1)]
    cum = [0.0, hop_gap_m, 2 * hop_gap_m, 3 * hop_gap_m]
    lens = [500.0] * 4
    evs = [{'from_lane': lanes[i], 'to_lane': lanes[i + 1],
            'window_s0': cum[i + 1], 'window_s1': cum[i + 1] + 300.0}
           for i in range(3)]
    p._road_entry_s = lambda *_a: 0.0
    p._ramp_is_dashed = lambda *_a: True
    p._build_lc_ramps(lanes, cum, lens, evs)
    return p.lc_ramps


def test_ramps_overlap_when_off():
    """이전 동작 — 램프가 hop 간격을 넘어 다음 창을 침범한다."""
    r = build_ramps(route_cfg(False), 20.0)
    assert len(r) == 3
    assert r[0][1] > r[1][0]                           # 겹친다
    assert r[0][1] - r[0][0] == pytest.approx(37.5, abs=0.1)


def test_ramps_fit_inside_the_next_hop_when_on():
    r = build_ramps(route_cfg(True), 20.0)
    assert len(r) == 3
    for (a0, a1), (b0, _b1) in zip(r, r[1:]):
        assert a1 <= b0 - GAP + 1e-6                   # 다음 창 전에 끝난다
        assert a1 - a0 == pytest.approx(20.0 - GAP, abs=0.1)   # 18.0 m


def test_last_ramp_is_not_shortened():
    """다음 hop 이 없으면 자를 이유가 없다 — 마지막은 원래 길이 그대로."""
    r = build_ramps(route_cfg(True), 20.0)
    assert r[-1][1] - r[-1][0] == pytest.approx(37.5, abs=0.1)


def test_wide_hops_are_untouched():
    """간격이 넉넉하면 켜도 안 바뀐다."""
    off = build_ramps(route_cfg(False), 60.0)
    on = build_ramps(route_cfg(True), 60.0)
    assert off == pytest.approx(on)
