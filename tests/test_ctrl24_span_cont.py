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


EXT = 2.2                                                        # Car 반길이 (test_ctrl24_avoid)


def v_front(front_m):
    """복귀 쪽 여유가 내는 상한 — 앞여유 − extra_after − 객체 반길이."""
    return (front_m - C['extra_after_m'] - EXT) / C['trans_k']


def v_behind(behind_m):
    return (behind_m - C['extra_before_m'] - EXT) / C['trans_k']


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
    # 장애물 60 m, 좌 이웃은 0~1000(=100 m)까지만 → 앞여유 40 m 가 상한을 낸다
    kr, p, ap = rig(actors=[car(2, 60.0)], gaps={'left': [(1000, 6000)],
                                                 'right': [(0, 6000)]})
    _c, target = apply(kr, ap, v=12.5)
    a = kr.last_avoid
    assert kr.ot_span is None                                  # 만들지 않았다
    assert a['state'] == 'SPAN_WAIT_V' and 'left:span_v_req' in a['rejects']
    want = min(v_front(40.0), v_behind(60.0))
    assert a['span_v_req'] == pytest.approx(want, abs=0.02)
    assert kr.last_kr['span_v_req'] == pytest.approx(want, abs=0.02)
    assert target == pytest.approx(want, abs=0.02)             # min() 후보로 나갔다
    assert kr.last_kr_winner == 'span_v_req'
    assert 2 not in kr._shifted_for                            # 다음 틱 다시 시도한다


def test_creates_once_speed_is_below_v_req():
    kr, p, ap = rig(actors=[car(2, 60.0)], gaps={'left': [(1000, 6000)],
                                                 'right': [(0, 6000)]})
    apply(kr, ap, v=12.5)
    assert kr.ot_span is None
    apply(kr, ap, v=min(v_front(40.0), v_behind(60.0)) - 0.1)                     # v_req 아래로 내려왔다
    a = kr.last_avoid
    assert a['state'] == 'PREEMPT' and a['shift'] == 'left'
    assert kr.ot_span is not None and kr.span_v_req is None
    assert kr.last_kr['span_v_req'] is None                    # 만든 틱엔 후보가 없다


def test_span_is_not_cut_when_it_is_created():
    """자르지 않는다 — 만들어진 span 은 평소와 같은 규칙으로 잡힌다."""
    v = min(v_front(40.0), v_behind(60.0)) - 0.1
    cut = rig(actors=[car(2, 60.0)], gaps={'left': [(1000, 6000)], 'right': [(0, 6000)]})
    free = rig(actors=[car(2, 60.0)], gaps={'right': [(0, 6000)]})
    apply(cut[0], cut[2], v=12.5)
    apply(cut[0], cut[2], v=v)
    apply(free[0], free[2], v=v)
    assert cut[0].ot_span == free[0].ot_span
    assert cut[0].last_avoid['trans_m'] == free[0].last_avoid['trans_m']


# ── 창이 하한으로도 부족하면 담지 않는다 ──────────────────────────────────
def test_window_too_short_is_not_taken():
    # 장애물 15 m, 좌 이웃 0~300 → 앞여유 15 m → (15 − 10 − 2.2)/3.24 = 0.86 < 2.5
    kr, p, ap = rig(actors=[car(2, 15.0)], gaps={'left': [(300, 6000)],
                                                 'right': [(0, 6000)]})
    before = p.route_points.copy()
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['state'] == 'NOOP' and 'left:span_no_room' in a['rejects']
    assert a['left_fit']['v_req'] == pytest.approx(v_front(15.0), abs=0.02)
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
    """목표 유무와 일치하되, 끊김에서 **빠져나오는 한 점**은 계단이라 False 다."""
    kr, p, ap = rig(gaps={'left': [(1000, 2000)]})
    ok = kr._nb_ok(p, True, 1)
    assert ok is not None and len(ok) == len(p.route_waypoints)
    for i in (0, 500, 999, 1000, 1500, 1999, 3000):
        assert bool(ok[i]) == (p._shift_target_wp(i, True, 1) is not None), i
    assert not ok[2000] and ok[2001]        # 없던 목표가 생기는 점 = 계단
    assert kr._nb_ok(p, True, 1) is ok                          # 두 번째는 캐시
    assert ('left', 1) in kr._nb_cache


def test_cont_window_spans_only_the_continuous_run():
    ok = np.ones(1000, dtype=bool)
    ok[200:300] = False
    ok[700:800] = False
    assert Ctrl24._cont_window(ok, 400, 500) == (300, 700)
    assert Ctrl24._cont_window(ok, 250, 250) is None             # 끊긴 구간 안
    assert Ctrl24._cont_window(ok, 100, 400) is None             # 끊김을 가로지른다


# ── 교차로 관통 연장 (2026-09-10 결정) ────────────────────────────────────
class FakeLG:
    """차로 3개짜리 최소 lane_graph — 경로 차로와 그 좌 이웃, 교차로 연결로."""

    def __init__(self, lanes):
        self.lanes = lanes

    def neighbor(self, key, side):
        return self.lanes.get(key, {}).get('left_nb' if side == 'left' else 'right_nb')

    def successors(self, key):
        return self.lanes.get(key, {}).get('next') or []


def jx_planner(through=True, turn=False, gaps=None):
    """정지선 60 m · 교차로 60~80 m · 출구 80 m 인 경로. through 면 옆 차로도 관통."""
    p = ContPlanner(gaps=gaps or {}, d_tl=float('inf'))
    IN, JN, OUT = ('R', 0, -1), ('J', 0, -1), ('R', 1, -1)
    NIN, NJN, NOUT = ('R', 0, -2), ('J', 0, -2), ('R', 1, -2)
    lanes = {
        IN: {'junction': -1, 'left_nb': NIN, 'right_nb': None, 'next': [JN]},
        JN: {'junction': 7, 'left_nb': NJN if through else None, 'right_nb': None,
             'next': [OUT]},
        OUT: {'junction': -1, 'left_nb': NOUT, 'right_nb': None, 'next': []},
        NIN: {'junction': -1, 'left_nb': None, 'right_nb': IN, 'next': [NJN]},
        NJN: {'junction': 7, 'left_nb': None, 'right_nb': JN, 'next': [NOUT]},
        NOUT: {'junction': -1, 'left_nb': None, 'right_nb': OUT, 'next': []},
    }
    p.lg = FakeLG(lanes)
    p.route = {'lanes': [IN, JN, OUT], 'cum_s': [0.0, 60.0, 80.0],
               'lengths': [60.0, 20.0, 200.0],
               'events': ([{'kind': 'turn_left', 's': 65.0}] if turn else [])}
    return p


def jx_rig(cfg=CFG, actors=(), **kw):
    p = jx_planner(**kw)
    ap = Ap(p, list(actors))
    ap._longitudinal_controller = VtdLongitudinalController(cfg)
    kr = Ctrl24(cfg)
    kr._sl_all = [60.0]                                          # 교차로 진입 정지선
    return kr, p, ap


def test_junction_extension_reaches_exit_plus_margin():
    kr, p, _ap = jx_rig()
    new_end, why, info = kr._junction_extend(p, 'left', 0.0, 65.0)
    assert why is None
    assert new_end == pytest.approx(80.0 + CFG['overtake']['zone_exit_margin_m'])
    assert info['zones'][0]['junction'] == 7


def test_junction_extension_rejected_when_side_lane_does_not_pass_through():
    kr, p, _ap = jx_rig(through=False)
    new_end, why, info = kr._junction_extend(p, 'left', 0.0, 65.0)
    assert why == 'zone_no_through_lane' and new_end == 65.0
    assert info['break_lane'] == ['J', 0, -1]


def test_junction_extension_rejected_on_turn_event():
    kr, p, _ap = jx_rig(turn=True)
    _new_end, why, _info = kr._junction_extend(p, 'left', 0.0, 65.0)
    assert why == 'zone_turn'


def test_tail_break_records_the_pass_through_verdict_and_never_makes_a_step():
    """뒤가 끊기면 관통 판정을 남기고, 관통이 안 서면 만들지 않는다 (계단 금지).

    실측 2026-09-10: 연결로에 side 이웃이 없어서 끊긴 것이므로 "관통 불가" 와
    "끊김" 은 같은 사실이다 — 그래서 연장이 서는 경우는 이 지도에서 관찰되지 않았다
    (docs/BACKLOG.md B-30). 여기서 지키는 것은 **계단을 만들지 않는다** 는 것이다.
    """
    kr, p, _ap = jx_rig(gaps={'left': [(700, 6000)]})             # 70 m 뒤로 이웃 없음
    v, v_req, b_new, info = kr._span_fit(p, True, 1, 300, 750, 600, 600,
                                         trans=20.0, back=20.0, ext_m=2.2, route_s=0.0)
    assert info['jx'] == 'zone_no_through_lane'                   # 관통 판정을 남긴다
    assert v != 'extend' and b_new == 750                         # 늘리지 않았다
    assert v == 'no_room'                                         # 앞여유 10 m 로는 못 담는다
    assert v_req == pytest.approx(v_front(10.0), abs=0.02)


def test_switch_off_skips_the_extension():
    c = copy.deepcopy(CFG)
    c['ctrl24']['span_junction_extend_enable'] = False
    kr, p, _ap = jx_rig(cfg=c, gaps={'left': [(1200, 6000)]})
    v, _vr, b, info = kr._span_fit(p, True, 1, 300, 950, 600, 600,
                                   trans=20.0, back=20.0, ext_m=2.2, route_s=0.0)
    assert v != 'extend' and 'jx' not in info and b == 950
