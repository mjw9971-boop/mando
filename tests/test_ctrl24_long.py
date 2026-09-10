"""
ctrl24 커밋 1 — 종방향 후보 K1~K5 · 황색 원샷 · 교차로 가드 · 훅 호환.

여기서 지키는 불변:
  · 속도는 전부 min() 후보다 — apply 는 candidate < target 일 때만 종방향을 되감아 재계산한다.
  · K1 은 kr_rules ④′ 와 같은 식·같은 s0(PDM 주입값)이다.
  · K2 는 정지당 1회 무장, 대상 소멸 시 최소 시간 미충족분만 남는다 (B-1 스펙).
  · K3 는 walkin 단일 경로 — ped_walkin_s 0 이면 조건 성립 첫 틱에 래치.
  · breakout_creep 훅은 항상 False (오버라이드는 signal_release 하나).
"""
import copy
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState, VehicleControl
from vtd_adapter.config import load_params_yaml
from vtd_adapter.control import VtdLongitudinalController

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from ctrl24 import Ctrl24                                     # noqa: E402
from test_avoid import Ap, Box, Planner                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
HZ = CFG['comm']['send_hz']
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
S0 = 5.299                                                    # test_avoid.Ap 의 주입값
A = C['stop_profile_a']


class Walker(Box):
    def __init__(self, oid, x, y, speed=0.0):
        super().__init__(oid, x, y, speed, half_w=0.3)
        self.type_id = 'walker.vtd.pedestrian'

    def move(self, x, y):
        self._x, self._y = float(x), float(y)


class TL:
    def __init__(self, state, tl_id=7):
        self.state = state
        self.id = tl_id
        self.controller_ids = [tl_id]


def rig(cfg=CFG, actors=(), d_tl=float('inf'), state=None, junction=False):
    p = Planner(d_tl=d_tl)
    if state is not None:
        p.next_traffic_lights = [TL(state)] * len(p.route_s)
    ap = Ap(p, list(actors), junction=junction)
    ap._longitudinal_controller = VtdLongitudinalController(cfg)
    kr = Ctrl24(cfg)
    kr._sl_all = []
    return kr, p, ap


def set_signal(p, state, d):
    p.next_traffic_lights = [TL(state)] * len(p.route_s)
    p.distances_to_next_traffic_lights[:] = float(d)


def apply(kr, ap, v, target=12.5):
    ap._vehicle.speed = v
    ap._longitudinal_controller.get_throttle_and_brake(False, target, v)   # 본류 역할
    ctrl = VehicleControl(steer=0.0, accel=1.0)
    return kr.apply(ctrl, target, ap)


def cfg_with(**kw):
    c = copy.deepcopy(CFG)
    c['ctrl24'].update(kw)
    return c


# ── 골격 ─────────────────────────────────────────────────────────────────
def test_params_section_present_and_enabled():
    assert C['enable'] is True
    assert C['ped_walkin_s'] == 0.0 and C['rtor_stop_hold_s'] == 0.5
    assert C['trans_min_m'] == 8.0 and C['trans_k'] == 3.24
    assert C['ped_crosswalk_creep_enable'] is False


def test_breakout_hook_is_always_false():
    kr, _p, _ap = rig()
    assert kr.breakout_creep() is False


def test_apply_without_candidates_keeps_pdm_target():
    kr, p, ap = rig()
    ctrl, t = apply(kr, ap, v=5.0, target=12.5)
    assert t == 12.5 and ctrl.accel == 1.0             # 되감지 않았다 (본류 값 그대로)
    assert kr.last_kr_winner is None
    assert set(kr.last_kr) == {'stop_profile', 'stop_hold', 'rtor_cap', 'ped_intent',
                               'crosswalk', 'red_zone', 'shift_cap',
                               'span_v_req', 'virtual_cap'}


# ── K1 ───────────────────────────────────────────────────────────────────
def test_k1_profile_matches_kr_rules_formula():
    kr, p, ap = rig(d_tl=30.0, state=TrafficLightState.Red)
    _c, t = apply(kr, ap, v=10.0)
    assert t == pytest.approx(math.sqrt(2.0 * A * (30.0 - S0)))
    assert kr.last_kr_winner == 'stop_profile'


def test_k1_absent_on_green():
    kr, p, ap = rig(d_tl=30.0, state=TrafficLightState.Green)
    _c, t = apply(kr, ap, v=10.0)
    assert t == 12.5 and kr.last_kr['stop_profile'] is None


def test_k1_uses_pdm_injected_s0():
    kr, p, ap = rig(d_tl=S0 + 2.0, state=TrafficLightState.Red)
    _c, t = apply(kr, ap, v=3.0)
    assert t == pytest.approx(math.sqrt(2.0 * A * 2.0))


# ── K2 ───────────────────────────────────────────────────────────────────
def test_k2_arms_once_per_stop_and_releases_on_green_with_min():
    kr, p, ap = rig(d_tl=FRONT + 1.5, state=TrafficLightState.Red)
    hold = int(C['stopline_hold_s'] * HZ)
    mn = int(C['stopline_hold_min_s'] * HZ)
    for _ in range(5):
        _c, t = apply(kr, ap, v=0.0)
        assert t == 0.0
    assert kr.sl_hold_left == hold - 5
    set_signal(p, TrafficLightState.Green, FRONT + 1.5)
    _c, t = apply(kr, ap, v=0.0)                        # 녹색 — 최소 미충족분만 남긴다
    assert t == 0.0 and kr.sl_hold_left == mn - 5 - 1
    for _ in range(mn - 5 - 1):
        _c, t = apply(kr, ap, v=0.0)
        assert t == 0.0
    _c, t = apply(kr, ap, v=0.0)
    assert t == 12.5                                    # 정확히 0.5 s 채우고 출발


def test_k2_does_not_rearm_while_still_stopped():
    kr, p, ap = rig(d_tl=FRONT + 3.0, state=TrafficLightState.Red)   # 계획 정지점 1.5 m 앞
    for _ in range(int(C['stopline_hold_s'] * HZ) + 5):
        apply(kr, ap, v=0.0)
    assert kr.sl_hold_left == 0
    _c, t = apply(kr, ap, v=0.0)
    assert t == pytest.approx(math.sqrt(2.0 * A * 1.5))  # 홀드가 아니라 K1 이 잡는다 (재무장 없음)


# ── 황색 원샷 · 가드 · signal_release ───────────────────────────────────
def test_yellow_one_shot_stop_then_go_latched():
    kr, p, ap = rig(d_tl=30.0, state=TrafficLightState.Yellow)
    apply(kr, ap, v=5.0)
    assert kr.y_decision == 'stop' and kr.last_yellow['decision'] == 'stop'
    assert kr.signal_release(ap) is False
    kr2, p2, ap2 = rig(d_tl=10.0, state=TrafficLightState.Yellow)
    apply(kr2, ap2, v=12.0)
    assert kr2.y_decision == 'go' and kr2.signal_release(ap2) is True
    set_signal(p2, TrafficLightState.Red, 8.0)           # 적색 전환에도 번복 없음
    _c, t = apply(kr2, ap2, v=12.0)
    assert kr2.y_decision == 'go' and t == 12.5


def test_cross_guard_after_front_passes_line():
    kr, p, ap = rig(d_tl=FRONT - 0.5, state=TrafficLightState.Red)
    _c, t = apply(kr, ap, v=6.0)
    assert kr.cross_guard is True and t == 12.5 and kr.signal_release(ap) is True
    ap.junction = True
    apply(kr, ap, v=6.0)
    assert kr.cross_junction_seen is True
    ap.junction = False
    set_signal(p, TrafficLightState.Red, 100.0)        # 플래너가 다음 정지선으로 넘어갔다
    _c, t = apply(kr, ap, v=6.0)
    assert kr.cross_guard is False and kr.signal_release(ap) is False
    assert kr.last_kr['stop_profile'] == pytest.approx(math.sqrt(2.0 * A * (100.0 - S0)), abs=1e-3)


# ── K3 ───────────────────────────────────────────────────────────────────
def test_k3_walkin_latches_on_first_tick_with_v_toward():
    w = Walker(9, 30.0, 6.0, speed=1.5)
    kr, p, ap = rig(actors=[w])
    _c, t = apply(kr, ap, v=10.0)
    assert t == 12.5                                   # 첫 관측 — 횡속도 없음
    w.move(30.0, 5.9)                                  # 경로 쪽으로 0.1 m/틱 = 2 m/s
    _c, t = apply(kr, ap, v=10.0)
    assert 9 in kr.ped_intent
    d_eff = 30.0 - FRONT - 4.0
    assert t == pytest.approx(math.sqrt(2.0 * A * d_eff))
    assert kr.last_kr_winner == 'ped_intent' and kr.last_ped['wins'] is True


def test_k3_walkin_delay_switch_restores_half_second():
    w = Walker(9, 30.0, 6.0, speed=1.5)
    kr, p, ap = rig(cfg_with(ped_walkin_s=0.5), actors=[w])
    apply(kr, ap, v=10.0)
    for k in range(9):
        w.move(30.0, 5.9 - 0.1 * k)
        apply(kr, ap, v=10.0)
        assert 9 not in kr.ped_intent
    w.move(30.0, 4.9)
    apply(kr, ap, v=10.0)
    assert 9 in kr.ped_intent


def test_k3_corridor_hold_is_zero_and_releases_when_clear():
    w = Walker(9, 20.0, 1.0, speed=0.0)
    kr, p, ap = rig(actors=[w])
    _c, t = apply(kr, ap, v=3.0)
    assert t == 0.0 and 9 in kr.ped_hold_ids and kr.last_ped['hold'] == [9]
    w.move(20.0, 4.0)                                  # 회랑 밖 · 멈춤 · 멀어짐
    apply(kr, ap, v=0.0)
    for _ in range(int(C['ped_release_s'] * HZ)):
        _c, t = apply(kr, ap, v=0.0)
    assert 9 not in kr.ped_hold_ids and t == 12.5


def test_k4_emergency_bypasses_jerk_ramp():
    w = Walker(9, 12.0, 6.0, speed=2.0)
    kr, p, ap = rig(actors=[w])
    apply(kr, ap, v=12.0)
    w.move(12.0, 5.8)
    ctrl, _t = apply(kr, ap, v=12.0)
    assert kr.ped_emergency is True and kr.last_ped['emergency'] is True
    assert ctrl.accel == pytest.approx(CFG['speed']['a_emergency'])


# ── K5 ───────────────────────────────────────────────────────────────────
def test_k5_off_by_default_and_on_when_enabled():
    w = Walker(9, 10.0, 3.0, speed=0.0)
    kr, p, ap = rig(actors=[w])
    kr._sl_all = [10.0]
    _c, t = apply(kr, ap, v=5.0)
    assert kr.last_kr['crosswalk'] is None
    kr2, p2, ap2 = rig(cfg_with(ped_crosswalk_creep_enable=True), actors=[w])
    kr2._sl_all = [10.0]
    _c, t = apply(kr2, ap2, v=5.0)
    assert kr2.last_kr['crosswalk'] is not None and t < 5.0


def test_on_reset_clears_latches():
    kr, p, ap = rig(d_tl=30.0, state=TrafficLightState.Yellow)
    apply(kr, ap, v=5.0)
    assert kr.y_decision == 'stop'
    kr.on_reset()
    assert kr.y_decision is None and kr.sl_hold_left == 0 and not kr.ped_intent


# ── K6 붉은 구간 진입 전 감속 (2026-09-09 복원) ──────────────────────────
class RedPlanner(Planner):
    """붉은 구간이 entry_s 부터 있는 경로 목 — route_waypoints/lg.red_spans 를 흉내낸다."""

    def __init__(self, entry_s=40.0, exit_s=52.0, **kw):
        super().__init__(**kw)
        key = (7, 0, -1)
        self.lg = type('LG', (), {
            'lanes': {key: {'red_spans': [(entry_s, exit_s)], 'junction': -1}},
            '_red_cfg': (True, 30.0, 0.0)})()
        self.route_waypoints = [type('W', (), {'key': key, 's': float(x)})()
                                for x in self.route_s]


def red_rig(cfg=CFG, entry_s=40.0):
    p = RedPlanner(entry_s=entry_s, d_tl=float('inf'))
    ap = Ap(p, [])
    ap._longitudinal_controller = VtdLongitudinalController(cfg)
    kr = Ctrl24(cfg)
    kr._sl_all = []
    return kr, p, ap


def test_k6_reads_existing_speed_constants():
    kr, _p, _ap = red_rig()
    assert kr.red_approach is True
    assert kr.red_a == CFG['speed']['approach_decel_mps2']
    assert kr.red_look_m == CFG['speed']['red_lookahead_m']
    assert kr.red_v_zone == pytest.approx(CFG['speed']['red_zone_target_kph'] / 3.6)


def test_k6_ceiling_matches_kr_rules_formula_and_stops_inside_zone():
    kr, p, ap = red_rig(entry_s=40.0)
    p.route_index = int(20.0 / 0.1)                  # 진입점 20 m 앞
    _c, t = apply(kr, ap, v=12.0)
    vz = CFG['speed']['red_zone_target_kph'] / 3.6
    want = math.sqrt(vz * vz + 2.0 * CFG['speed']['approach_decel_mps2'] * 20.0)
    assert t == pytest.approx(want)
    assert kr.last_kr['red_zone'] == pytest.approx(round(want, 3))
    assert kr.last_kr_winner == 'red_zone'
    assert kr.last_red_zone['entry_s'] == 40.0 and kr.last_red_zone['d'] == 20.0
    p.route_index = int(45.0 / 0.1)                  # 구간 안 — 후보 없음 (제한속도 소관)
    _c, t = apply(kr, ap, v=7.0)
    assert kr.last_kr['red_zone'] is None and t == 12.5


def test_k6_silent_beyond_lookahead_and_when_off():
    kr, p, ap = red_rig(entry_s=200.0)               # lookahead 60 m 밖
    _c, t = apply(kr, ap, v=12.0)
    assert kr.last_kr['red_zone'] is None and t == 12.5
    kr2, p2, ap2 = red_rig(cfg_with(red_zone_enable=False), entry_s=40.0)
    p2.route_index = int(20.0 / 0.1)
    _c, t = apply(kr2, ap2, v=12.0)
    assert kr2.last_kr['red_zone'] is None and t == 12.5
