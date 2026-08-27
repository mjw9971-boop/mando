"""
detect_center_line — 대회 항목 4 중앙선 침범 (합성 틱 경계값).

깊이 = t_off + 차폭/2 − 차로폭/2 ≥ center_depth_m 가 center_hold_s 이상 지속.
left_is_center 틱만, junction 차로 제외, reset 틱 제외.
"""
import score as sc_mod
from conftest import mk_tick

SC = {'center_depth_m': 0.6, 'center_hold_s': 0.6}
VEH_W = 1.9
LANE_W = 3.5                       # half = 1.75 → depth = t_off + 0.95 − 1.75
K = (10, 0, -1)


class FakeLG:
    def __init__(self, junction=-1, color='yellow'):
        self.lanes = {K: {'junction': junction}}
        self.color = color

    def width_at(self, key, s):
        if key not in self.lanes:
            raise KeyError(key)
        return LANE_W

    def mark_at(self, key, s, side):
        return ('solid', self.color, False)


def t_off_for(depth):
    return depth - VEH_W / 2.0 + LANE_W / 2.0


def run_ticks(depth, n=8, dt=0.1, lic=True, reset_at=None, lg=None):
    ticks = [mk_tick(t=i * dt, speed=5.0, lane=K, t_off=t_off_for(depth),
                     left_is_center=lic, reset=(i == reset_at)) for i in range(n)]
    return sc_mod.detect_center_line(ticks, 0.0, lg or FakeLG(), VEH_W, SC)


# ── 깊이 경계 ─────────────────────────────────────────────────────────────
def test_depth_just_under_threshold_ignored():
    assert run_ticks(depth=0.59) == []


def test_depth_over_threshold_detected_major_depth():
    evs = run_ticks(depth=0.61)
    assert len(evs) == 1
    assert evs[0]['depth_m'] == 0.61
    assert evs[0]['left_mark_yellow'] is True


# ── 지속시간 경계 (8틱×0.1 s → span 0.7 s / 6틱 → 0.5 s) ────────────────
def test_hold_just_under_ignored():
    assert run_ticks(depth=0.7, n=6) == []                 # 0.5 s < 0.6


def test_hold_at_threshold_counts():
    assert len(run_ticks(depth=0.7, n=7)) == 1             # 0.6 s


# ── 대상 한정 ─────────────────────────────────────────────────────────────
def test_not_left_is_center_ignored():
    assert run_ticks(depth=1.0, lic=False) == []


def test_junction_lane_excluded():
    assert run_ticks(depth=1.0, lg=FakeLG(junction=5)) == []


def test_reset_tick_breaks_episode():
    assert run_ticks(depth=1.0, reset_at=4) == []          # 양쪽 각 0.3 s < hold


def test_unknown_lane_key_skipped():
    ticks = [mk_tick(t=i * 0.1, speed=5.0, lane=(99, 0, -1), t_off=t_off_for(1.0),
                     left_is_center=True) for i in range(8)]
    assert sc_mod.detect_center_line(ticks, 0.0, FakeLG(), VEH_W, SC) == []


def test_white_left_mark_recorded_not_judged():
    evs = run_ticks(depth=0.8, lg=FakeLG(color='standard'))
    assert len(evs) == 1 and evs[0]['left_mark_yellow'] is False
