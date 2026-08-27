"""
detect_sidewalk — 대회 항목 5 보도 침범 (합성 틱 경계값).

깊이 = |t_off| + 차폭/2 − sidewalk_dist_at ≥ sidewalk_depth_m 가
sidewalk_hold_s 이상 지속 = 중대. 그 쪽 보도 없음(None)은 대상 아님.
"""
import score as sc_mod
from conftest import mk_tick

SC = {'sidewalk_depth_m': 0.5, 'sidewalk_hold_s': 0.3}
VEH_W = 1.9
DIST_R, DIST_L = 3.0, 5.0          # 우/좌 보도 안쪽 경계까지 (차로 중심 기준)
K = (10, 0, -1)


class FakeLG:
    def __init__(self, right=DIST_R, left=DIST_L):
        self.d = {'right': right, 'left': left}

    def sidewalk_dist_at(self, key, s, side):
        if key != K:
            raise KeyError(key)
        return self.d[side]


def t_off_for(depth, side='right'):
    off = depth - VEH_W / 2.0 + (DIST_R if side == 'right' else DIST_L)
    return -off if side == 'right' else off


def run_ticks(depth, side='right', n=8, dt=0.05, reset_at=None, lg=None):
    ticks = [mk_tick(t=i * dt, speed=3.0, lane=K, t_off=t_off_for(depth, side),
                     reset=(i == reset_at)) for i in range(n)]
    return sc_mod.detect_sidewalk(ticks, 0.0, lg or FakeLG(), VEH_W, SC)


# ── 깊이 경계 ─────────────────────────────────────────────────────────────
def test_depth_just_under_ignored():
    assert run_ticks(depth=0.49) == []


def test_depth_over_detected():
    evs = run_ticks(depth=0.51)
    assert len(evs) == 1
    assert evs[0]['depth_m'] == 0.51
    assert evs[0]['side'] == 'right'


# ── 지속시간 경계 (n틱×0.05 s → span (n−1)×0.05) ─────────────────────────
def test_hold_just_under_ignored():
    assert run_ticks(depth=0.8, n=6) == []                 # 0.25 s < 0.3


def test_hold_at_threshold_counts():
    assert len(run_ticks(depth=0.8, n=7)) == 1             # 0.3 s


# ── side 선택·대상 한정 ──────────────────────────────────────────────────
def test_left_side_uses_left_distance():
    evs = run_ticks(depth=0.6, side='left')
    assert len(evs) == 1 and evs[0]['side'] == 'left'
    assert evs[0]['t_off'] > 0


def test_no_sidewalk_none_skipped():
    assert run_ticks(depth=2.0, lg=FakeLG(right=None, left=None)) == []


def test_old_graph_missing_field_skipped():
    class OldLG:                                            # 구 pkl: 항상 None
        def sidewalk_dist_at(self, key, s, side):
            return None
    assert run_ticks(depth=2.0, lg=OldLG()) == []


def test_reset_tick_breaks_episode():
    assert run_ticks(depth=0.8, reset_at=4) == []          # 양쪽 각 0.15 s < hold


def test_unknown_lane_key_skipped():
    ticks = [mk_tick(t=i * 0.05, speed=3.0, lane=(99, 0, -1),
                     t_off=t_off_for(1.0)) for i in range(8)]
    assert sc_mod.detect_sidewalk(ticks, 0.0, FakeLG(), VEH_W, SC) == []
