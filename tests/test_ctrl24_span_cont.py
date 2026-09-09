"""
ctrl24 K8 — span 이웃 연속성 (B-30).

여기서 지키는 불변:
  · **게이트가 아니다.** 이웃이 전 구간 이어지면 이전과 똑같이 동작한다.
  · 장애물 지점에 그 side 목표가 없으면(섹션에 그 차로가 없음) 그 side 를 건너뛰고
    **반대편을 본다**. 양쪽 다 없을 때만 NOOP.
  · 뒤가 끊기면 span 을 자르지 않는다. 창에 전이 2회가 들어가는 속도를 min() 후보로
    내고(K8), v ≤ v_req 인 틱에 시프트를 만든다.
  · 창이 하한(span_v_req_min)으로도 부족하면 담지 않는다.
  · route_waypoints·_shift_target_wp 가 없는 조립(구형 플래너)에서는 판정을 건너뛴다.
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

from ctrl24 import Ctrl24                                       # noqa: E402
from test_avoid import Ap                                       # noqa: E402
from test_ctrl24_avoid import LANE, ShiftPlanner, apply, car     # noqa: E402
from vtd_adapter.control import VtdLongitudinalController        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
PPM = 10


class WP:
    """차로 키만 가진 웨이포인트 — _nb_ok 는 키가 바뀔 때만 재조회한다."""

    def __init__(self, key):
        self.key = key


class ContPlanner(ShiftPlanner):
    """이웃 유무를 **구간별로** 켜고 끄는 플래너.

    gaps = {'left': [(i0, i1), ...]} — 그 인덱스 구간에는 좌 이웃이 없다.
    차로 키는 100점(10 m)마다 바뀌게 두어 캐시 경로를 실제로 태운다.
    """

    def __init__(self, gaps=None, **kw):
        super().__init__(**kw)
        self.gaps = {'left': [], 'right': [], **(gaps or {})}
        n = len(self.route_s)
        self.route_waypoints = [WP(('R', i // 100)) for i in range(n)]

    def _shift_target_wp(self, idx, left, n_steps):
        side = 'left' if left else 'right'
        if not self.has[side]:
            return None
        if any(a <= idx < b for a, b in self.gaps[side]):
            return None
        return WP(('N', idx // 100))


def rig(cfg=CFG, actors=(), gaps=None, left=True, right=True):
    p = ContPlanner(gaps=gaps, left=left, right=right, d_tl=float('inf'))
    ap = Ap(p, list(actors))
    ap._longitudinal_controller = VtdLongitudinalController(cfg)
    kr = Ctrl24(cfg)
    kr._sl_all = []
    return kr, p, ap


def off_cfg():
    c = copy.deepcopy(CFG)
    c['ctrl24']['span_v_req_enable'] = False
    return c


def v_req_of(window_m):
    return (window_m - C['extra_before_m'] - C['extra_after_m']) / (2 * C['trans_k'])


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_params_switch_on_and_floor():
    assert C['span_v_req_enable'] is True
    assert C['span_v_req_min'] == 2.5
    kr, _p, _ap = rig()
    assert kr.span_cont is True and kr.span_v_min == 2.5


# ── 연속이면 이전과 같다 ──────────────────────────────────────────────────
def test_continuous_route_behaves_exactly_as_before():
    on = rig(actors=[car(2, 78.0)])
    off = rig(cfg=off_cfg(), actors=[car(2, 78.0)])
    for kr, _p, ap in (on, off):
        apply(kr, ap, v=12.5)
    a_on, a_off = on[0].last_avoid, off[0].last_avoid
    assert on[0].ot_span == off[0].ot_span
    assert a_on['state'] == a_off['state'] == 'PREEMPT' and a_on['shift'] == a_off['shift']
    assert np.allclose(on[1].route_points[:, 1], off[1].route_points[:, 1])
    assert on[0].span_v_req is None and on[0].last_kr['span_v_req'] is None


def test_planner_without_waypoints_skips_the_check():
    """구형 조립(_shift_target_wp 없음) — 판정을 건너뛰고 이전 동작."""
    from test_ctrl24_avoid import rig as plain_rig
    kr, p, ap = plain_rig(actors=[car(2, 78.0)])
    apply(kr, ap, v=12.5)
    assert kr.ot_span is not None and kr.last_avoid['state'] == 'PREEMPT'
    assert kr.span_v_req is None


# ── 유형 (b): 장애물 지점에 목표가 없다 → 반대편 side ──────────────────────
def test_no_target_on_left_falls_through_to_right():
    kr, p, ap = rig(actors=[car(2, 60.0)], gaps={'left': [(400, 900)]})
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['shift'] == 'right' and 'left:no_target' in a['rejects']
    assert p.route_points[kr.ot_span[0] + 400, 1] == pytest.approx(-LANE, abs=0.05)
    assert kr.span_v_req is None


def test_noop_when_neither_side_has_a_target_at_the_obstacle():
    kr, p, ap = rig(actors=[car(2, 60.0)],
                    gaps={'left': [(400, 900)], 'right': [(400, 900)]})
    before = p.route_points.copy()
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['state'] == 'NOOP'
    assert a['rejects'] == ['left:no_target', 'right:no_target']
    assert kr.ot_span is None and np.allclose(p.route_points, before)


# ── 유형 (a): 뒤가 끊긴다 → 자르지 않고 감속 후보 ─────────────────────────
def test_tail_break_defers_and_emits_speed_candidate():
    # 장애물 60 m, 좌 이웃은 0~800(=80 m)까지만 → 창 80 m, v_req 10.0 < 현재 12.5
    kr, p, ap = rig(actors=[car(2, 60.0)], gaps={'left': [(800, 6000)],
                                                 'right': [(0, 6000)]})
    _c, target = apply(kr, ap, v=12.5)
    a = kr.last_avoid
    assert kr.ot_span is None                                  # 만들지 않았다
    assert a['state'] == 'SPAN_WAIT_V' and 'left:span_v_req' in a['rejects']
    want = v_req_of(80.0)
    assert a['span_v_req'] == pytest.approx(want, abs=0.02)
    assert kr.last_kr['span_v_req'] == pytest.approx(want, abs=0.02)
    assert target == pytest.approx(want, abs=0.02)             # min() 후보로 나갔다
    assert kr.last_kr_winner == 'span_v_req'
    assert 2 not in kr._shifted_for                            # 다음 틱 다시 시도한다


def test_creates_once_speed_is_below_v_req():
    kr, p, ap = rig(actors=[car(2, 60.0)], gaps={'left': [(800, 6000)],
                                                 'right': [(0, 6000)]})
    apply(kr, ap, v=12.5)
    assert kr.ot_span is None
    apply(kr, ap, v=v_req_of(80.0) - 0.1)                     # v_req 아래로 내려왔다
    a = kr.last_avoid
    assert a['state'] == 'PREEMPT' and a['shift'] == 'left'
    assert kr.ot_span is not None and kr.span_v_req is None
    assert kr.last_kr['span_v_req'] is None                    # 만든 틱엔 후보가 없다


def test_span_is_not_cut_when_it_is_created():
    """자르지 않는다 — 만들어진 span 은 평소와 같은 규칙으로 잡힌다."""
    v = v_req_of(80.0) - 0.1
    cut = rig(actors=[car(2, 60.0)], gaps={'left': [(800, 6000)], 'right': [(0, 6000)]})
    free = rig(actors=[car(2, 60.0)], gaps={'right': [(0, 6000)]})
    apply(cut[0], cut[2], v=12.5)
    apply(cut[0], cut[2], v=v)
    apply(free[0], free[2], v=v)
    assert cut[0].ot_span == free[0].ot_span
    assert cut[0].last_avoid['trans_m'] == free[0].last_avoid['trans_m']


# ── 창이 하한으로도 부족하면 담지 않는다 ──────────────────────────────────
def test_window_too_short_is_not_taken():
    # 창 30 m → v_req = (30 − 15)/6.48 = 2.31 < 2.5 하한
    kr, p, ap = rig(actors=[car(2, 15.0)], gaps={'left': [(300, 6000)],
                                                 'right': [(0, 6000)]})
    before = p.route_points.copy()
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['state'] == 'NOOP' and 'left:span_no_room' in a['rejects']
    assert a['left_v_req'] == pytest.approx(v_req_of(30.0), abs=0.02)
    assert kr.ot_span is None and np.allclose(p.route_points, before)
    assert kr.span_v_req is None                               # 감속을 요구하지 않는다


def test_switch_off_restores_previous_behaviour():
    kr, p, ap = rig(cfg=off_cfg(), actors=[car(2, 60.0)],
                    gaps={'left': [(1000, 6000)], 'right': [(0, 6000)]})
    apply(kr, ap, v=12.5)
    assert kr.ot_span is not None                              # 끊김을 묻지 않고 만든다
    assert kr.last_avoid['state'] == 'PREEMPT' and kr.span_v_req is None


# ── 연속성 조회 자체 ─────────────────────────────────────────────────────
def test_nb_ok_matches_shift_target_wp_and_is_cached():
    kr, p, ap = rig(gaps={'left': [(1000, 2000)]})
    ok = kr._nb_ok(p, True, 1)
    assert ok is not None and len(ok) == len(p.route_waypoints)
    for i in (0, 500, 999, 1000, 1500, 1999, 2000, 3000):
        assert bool(ok[i]) == (p._shift_target_wp(i, True, 1) is not None), i
    assert kr._nb_ok(p, True, 1) is ok                          # 두 번째는 캐시
    assert ('left', 1) in kr._nb_cache


def test_cont_window_spans_only_the_continuous_run():
    ok = np.ones(1000, dtype=bool)
    ok[200:300] = False
    ok[700:800] = False
    assert Ctrl24._cont_window(ok, 400, 500) == (300, 700)
    assert Ctrl24._cont_window(ok, 250, 250) is None             # 끊긴 구간 안
    assert Ctrl24._cont_window(ok, 100, 400) is None             # 끊김을 가로지른다
