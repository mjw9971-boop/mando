"""
ctrl24 — 가상 차로 시프트 (B-33). 탈출 규칙의 한 갈래다.

여기서 지키는 불변:
  · 발동은 **정상 시프트가 불가능하다고 확정된 뒤**다 — 고착 래치가 서 있고,
    양쪽에 목표가 없어 NOOP 이고, 정당한 정지 원인이 없을 때만.
  · 방향 우선순위: 좌(대향 차로 있음) → 우(보도까지 여유 ≥ D) → 시도 안 함.
  · 목표는 이웃 차로가 아니라 원 경로를 D 만큼 민 폴리라인이다 — 계단이 정의상 없다.
  · 좌(대향) 진입 전에는 대향 이동 차량이 escape_oncoming_clear_m 안에 있으면 대기.
  · 가상 span 안에서는 escape_virtual_v 상한이 min() 후보로 나간다.
  · 기본 off.
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

from ctrl24 import Ctrl24                                                    # noqa: E402
from test_ctrl24_avoid import LANE, ShiftPlanner, apply, car                 # noqa: E402
from test_ctrl24_escape import hold, move, on_cfg                              # noqa: E402
from test_avoid import Ap                                                      # noqa: E402
from vtd_adapter.control import VtdLongitudinalController                      # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
HZ = CFG['comm']['send_hz']
HALF_EGO = CFG['vehicle']['width'] / 2.0
CLR = CFG['percep']['obstacle_clearance_m']


def virt_cfg(**kw):
    c = on_cfg(escape_virtual_shift_enable=True)
    c['ctrl24'].update(kw)
    return c


def D_of(obj_half_w=0.9):
    return HALF_EGO + obj_half_w + C['escape_virtual_extra_m'] + CLR


class LG:
    """도로 하나짜리 최소 lane_graph — 반대 방향 차로와 보도 여유를 켜고 끈다."""

    def __init__(self, oncoming=True, right_room=0.0):
        self.lanes = {
            (1, 0, -1): {'road': 1, 'sec': 0, 'lane_id': -1, 'dir': 1, 'junction': -1,
                         'left_nb': None, 'right_nb': None,
                         'sidewalk_right_m': np.full(4, float(right_room)),
                         'sidewalk_left_m': None},
        }
        if oncoming:
            self.lanes[(1, 0, 1)] = {'road': 1, 'sec': 0, 'lane_id': 1, 'dir': -1,
                                     'junction': -1, 'left_nb': None, 'right_nb': None,
                                     'sidewalk_right_m': None, 'sidewalk_left_m': None}

    def neighbor(self, key, side):
        return self.lanes.get(key, {}).get('left_nb' if side == 'left' else 'right_nb')


class VirtPlanner(ShiftPlanner):
    """offset_m 을 지원하는 플래너 목 — route.py 의 오프셋 폴리라인과 같은 규약.

    경로가 x 축 직선이라 좌측 법선은 +y 다. 이웃 차로는 조회하지 않는다.
    """

    def shift_route_smoothly(self, a, b, left, transition_length=120.0,
                             lane_transition_factor=1.0, transition_length_back=None,
                             ref_index=None, offset_m=None):
        if offset_m is None:
            return super().shift_route_smoothly(
                a, b, left, transition_length, lane_transition_factor,
                transition_length_back, ref_index)
        self.calls.append((a, b, left, transition_length, transition_length_back, offset_m))
        target = float(offset_m) * (1.0 if left else -1.0)
        back = transition_length if transition_length_back is None else transition_length_back
        for idx in range(int(a), int(b)):
            f = 1.0
            if idx <= a + transition_length and idx - a < b - idx:
                f = -math.cos(float(idx - a) / transition_length * math.pi) / 2 + 0.5
            elif idx >= b - back:
                f = -math.cos(float(b - idx) / back * math.pi) / 2 + 0.5
            y = f * target + (1.0 - f) * self.route_points[idx, 1]
            self.route_points[idx, 1] = y
            self.lat_shift[idx] = y


def rig(cfg, actors=(), oncoming=True, right_room=0.0):
    """양쪽에 이웃이 없어 정상 시프트가 NOOP 인 조립."""
    p = VirtPlanner(left=False, right=False, d_tl=float('inf'))
    ap = Ap(p, list(actors))
    ap._longitudinal_controller = VtdLongitudinalController(cfg)
    kr = Ctrl24(cfg)
    kr._sl_all = []
    p.lg = LG(oncoming=oncoming, right_room=right_room)
    kr._tick_lg = p.lg
    kr._ego_lane = lambda *_a, **_k: (1, 0, -1)
    return kr, p, ap


def stick(kr, ap, p, n=None):
    """고착 래치가 설 때까지 돌리고, 가상 시프트가 도는 틱의 진단을 돌려준다.

    회피 틱은 후보 계산보다 먼저라 래치가 서는 틱에는 아직 걸려 있지 않다 —
    가상 시프트는 그 **다음 틱**에 돈다.
    """
    n = n or int(C['escape_stuck_s'] * HZ) + 6
    snap = None
    for _ in range(n):
        apply(kr, ap, v=0.0, target=0.0)
        st = (kr.last_avoid or {}).get('state')
        if st and st.startswith('ESCAPE_VIRTUAL'):
            snap = dict(kr.last_avoid)
        if kr.ot_span is not None:
            break
    return snap or dict(kr.last_avoid or {})


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_params_defaults():
    assert C['escape_virtual_shift_enable'] is False
    assert C['escape_virtual_extra_m'] == 0.3 and C['escape_virtual_v'] == 3.0
    assert C['escape_oncoming_clear_m'] == 80.0


def test_off_by_default_never_shifts():
    kr, p, ap = rig(on_cfg(), actors=[car(2, 30.0)])
    before = p.route_points.copy()
    stick(kr, ap, p)
    assert kr._esc_engaged is True and kr.ot_span is None
    assert np.allclose(p.route_points, before)


def test_does_not_fire_before_the_latch():
    """정상 시프트가 불가능해도 고착이 아니면 가상 차로로 가지 않는다."""
    kr, p, ap = rig(virt_cfg(), actors=[car(2, 30.0)])
    before = p.route_points.copy()
    apply(kr, ap, v=8.0)                                  # 첫 틱 — 래치 전
    assert kr.last_avoid['state'] == 'NOOP' and kr.ot_span is None
    assert np.allclose(p.route_points, before)


# ── 방향 우선순위 ────────────────────────────────────────────────────────
def test_left_when_an_oncoming_lane_exists():
    kr, p, ap = rig(virt_cfg(), actors=[car(2, 30.0)], oncoming=True)
    a = stick(kr, ap, p)
    assert a['state'] == 'ESCAPE_VIRTUAL' and a['side'] == 'left'
    assert a['why'] == 'oncoming_lane'
    assert a['D'] == pytest.approx(D_of(), abs=0.01)
    assert kr.ot_span is not None
    lo, hi = kr.ot_span
    assert p.route_points[(lo + hi) // 2, 1] == pytest.approx(a['D'], abs=0.05)


def test_right_when_there_is_shoulder_room_and_no_oncoming():
    kr, p, ap = rig(virt_cfg(), actors=[car(2, 30.0)], oncoming=False,
                    right_room=D_of() + 0.5)
    a = stick(kr, ap, p)
    assert a['side'] == 'right' and a['why'].startswith('shoulder')
    lo, hi = kr.ot_span
    assert p.route_points[(lo + hi) // 2, 1] == pytest.approx(-a['D'], abs=0.05)


def test_no_side_when_neither_oncoming_nor_room():
    kr, p, ap = rig(virt_cfg(), actors=[car(2, 30.0)], oncoming=False, right_room=0.5)
    before = p.route_points.copy()
    a = stick(kr, ap, p)
    assert a['state'] == 'ESCAPE_VIRTUAL_NONE' and a['side'] is None
    assert a['why'] == 'no_room'
    assert kr.ot_span is None and np.allclose(p.route_points, before)


# ── 대향 안전 ────────────────────────────────────────────────────────────
def test_waits_while_an_oncoming_vehicle_is_close():
    moving = car(3, 50.0, y=0.0, speed=8.0)               # 이동 차량 (대향 위험)
    kr, p, ap = rig(virt_cfg(), actors=[car(2, 30.0), moving])
    a = stick(kr, ap, p)
    assert a['state'] == 'ESCAPE_VIRTUAL_WAIT' and kr.ot_span is None
    assert a['oncoming_m'] == pytest.approx(50.0, abs=0.2)
    assert a['wait_ticks'] >= 1
    moving.speed = 0.0                                    # 더는 이동 객체가 아니다
    apply(kr, ap, v=0.0, target=0.0)
    assert kr.last_avoid['state'] == 'ESCAPE_VIRTUAL' and kr.ot_span is not None


def test_oncoming_beyond_the_window_does_not_block():
    far = car(3, C['escape_oncoming_clear_m'] + 10.0, speed=8.0)
    kr, p, ap = rig(virt_cfg(), actors=[car(2, 30.0), far])
    a = stick(kr, ap, p)
    assert a['state'] == 'ESCAPE_VIRTUAL' and kr.ot_span is not None


# ── 속도 상한 ────────────────────────────────────────────────────────────
def test_virtual_span_caps_the_speed():
    kr, p, ap = rig(virt_cfg(), actors=[car(2, 30.0)])
    stick(kr, ap, p)
    lo, _hi = kr.ot_span
    p.route_index = lo + 10
    _c, t = apply(kr, ap, v=5.0, target=12.5)
    assert kr.last_kr['virtual_cap'] == pytest.approx(C['escape_virtual_v'])
    assert t <= C['escape_virtual_v'] + 1e-9        # 상한이 걸렸다 (K7 이 더 낮을 수 있다)


def test_cap_lifts_after_the_span():
    kr, p, ap = rig(virt_cfg(), actors=[car(2, 30.0)])
    stick(kr, ap, p)
    _lo, hi = kr.ot_span
    p.route_index = hi + 5
    apply(kr, ap, v=5.0, target=12.5)
    assert kr.last_kr['virtual_cap'] is None and kr._virt_span is None


# ── 계단 ─────────────────────────────────────────────────────────────────
def test_offset_polyline_has_no_step():
    """목표가 연속 폴리라인이라 계단이 정의상 없다."""
    kr, p, ap = rig(virt_cfg(), actors=[car(2, 30.0)])
    stick(kr, ap, p)
    lo, hi = kr.ot_span
    y = p.route_points[lo:hi, 1]
    assert float(np.abs(np.diff(y)).max()) < 0.05        # 한 점 사이 도약 없음
    d2 = np.abs(y[10:] - 2 * y[5:-5] + y[:-10]) / 0.25    # 0.5 m 스텐실
    assert float(d2.max()) < 1.0
