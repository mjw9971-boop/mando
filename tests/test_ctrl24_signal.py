"""
ctrl24 커밋 2 — 래치(timeout GO) · RTOR(정지 유지 0.5 s) · 지시등.

여기서 지키는 불변:
  · RTOR 래치는 정지 구역 안 정지 rtor_stop_hold_s 누적 뒤에만 선다 (결정 Q5).
    래치 뒤 K2 잔여는 최소 시간(stopline_hold_min_s) 미충족분뿐이다 — 틱 단위 검증.
  · rtor_stop_hold_s 0 이면 정지 없이 래치된다 (= "정지 0 s" 의 근거).
  · timeout GO 는 회랑 정지 객체가 하나라도 있으면 시계가 0 이다.
  · 지시등은 회전(이벤트)·차로 이동(lat_shift) 두 후보, 유지는 min_on/off_delay 뿐.
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

from ctrl24 import SIG_LEFT, SIG_OFF, SIG_RIGHT, Ctrl24        # noqa: E402
from test_avoid import Ap, Box, Planner                         # noqa: E402
from test_ctrl24_long import TL, Walker, cfg_with               # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
HZ = CFG['comm']['send_hz']
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
S0 = 5.299


class EgoT(Box):
    """RTOR 교차 차량 검사가 자차 transform 을 묻는다."""

    def __init__(self):
        super().__init__(0, 0.0, 0.0, 0.0, 0.94)

    def get_transform(self):
        return type('T', (), {'rotation': type('R', (), {'yaw': 0.0})()})()


def rig(cfg=CFG, actors=(), d_tl=float('inf'), state=None, events=()):
    p = Planner(d_tl=d_tl)
    p.route['events'] = list(events)
    if state is not None:
        p.next_traffic_lights = [TL(state)] * len(p.route_s)
    ap = Ap(p, list(actors))
    ap._vehicle = EgoT()
    ap._longitudinal_controller = VtdLongitudinalController(cfg)
    kr = Ctrl24(cfg)
    kr._sl_all = []
    return kr, p, ap


def apply(kr, ap, v, target=12.5, lights=((7, 1),)):
    """observe_lights → apply. lights=None 이면 이번 틱 미보고."""
    if lights is not None:
        kr.observe_lights(list(lights))
    else:
        kr.observe_lights([])
    ap._vehicle.speed = v
    ap._longitudinal_controller.get_throttle_and_brake(False, target, v)
    return kr.apply(VehicleControl(steer=0.0, accel=1.0), target, ap)


# ── 지시등 ─────────────────────────────────────────────────────────────
def test_turn_signal_from_event_with_lead_and_hold():
    kr, p, ap = rig(events=[{'kind': 'turn_right', 's': 50.0, 'junction': 3}])
    p.route_index = int(35.0 / 0.1)                     # 남은 15 m ≤ max(v·4, 20)
    apply(kr, ap, v=3.0)
    assert kr.last_turn_signal == SIG_RIGHT and kr.last_sig_src == 'turn'
    assert kr.last_sig_lead_s == pytest.approx(15.0 / 3.0)
    p.route_index = int(60.0 / 0.1)                     # end_s(=ev_s, lg 없음) 통과
    apply(kr, ap, v=3.0)
    assert kr.last_turn_signal == SIG_RIGHT and kr.last_sig_src == 'hold'
    for _ in range(int(C['sig_min_on_s'] * HZ)):       # min_on(1.0 s) 이 off_delay 보다 길다
        apply(kr, ap, v=3.0)
    assert kr.last_turn_signal == SIG_OFF


def test_lane_shift_signal_follows_lat_shift_array():
    kr, p, ap = rig()
    n = len(p.route_s)
    lat = np.zeros(n)
    lat[int(10.0 / 0.1):] = 3.0                         # 10 m 앞에서 좌로 3 m
    p.lat_shift = lat
    apply(kr, ap, v=2.0)
    assert kr.last_turn_signal == SIG_LEFT and kr.last_sig_src == 'lc'


# ── timeout GO ──────────────────────────────────────────────────────────
def test_timeout_go_latches_after_stale_wait_and_resets_on_report():
    kr, p, ap = rig(d_tl=10.0, state=TrafficLightState.Red)
    apply(kr, ap, v=0.0)                                # fresh 보고 1회
    n = int(C['signal_unknown_timeout_s'] * HZ) + int(C['signal_stale_s'] * HZ) + 2
    for _ in range(n):
        _c, t = apply(kr, ap, v=0.0, lights=None)
    assert kr._sig_go is True and kr.last_signal['timeout_go'] is True
    assert kr.last_kr['stop_profile'] is None and kr.signal_release(ap) is True
    assert t == 12.5
    _c, t = apply(kr, ap, v=0.0)                        # 보고 재개 → 그 state 가 우선
    assert kr._sig_go is False and kr.last_kr['stop_profile'] is not None and t < 12.5


def test_timeout_go_clock_is_zero_with_stopped_lead_in_corridor():
    lead = Box(3, 6.0, 0.0, 0.0, half_w=0.9)
    lead.type_id = 'vehicle.vtd.object'
    kr, p, ap = rig(actors=[lead], d_tl=10.0, state=TrafficLightState.Red)
    apply(kr, ap, v=0.0)
    for _ in range(int(C['signal_unknown_timeout_s'] * HZ) + 40):
        apply(kr, ap, v=0.0, lights=None)
    assert kr._sig_go is False and kr.last_signal['timeout_s'] == 0.0


# ── RTOR ────────────────────────────────────────────────────────────────
def _rtor_rig(cfg=CFG, actors=()):
    d = FRONT + 1.0                                     # 앞범퍼가 정지선 1.0 m 앞 = 구역 안
    ev = [{'kind': 'turn_right', 's': d + 0.2, 'junction': 3}]
    return rig(cfg, actors=actors, d_tl=d, state=TrafficLightState.Red, events=ev)


def test_rtor_latches_after_half_second_stop_and_caps_speed():
    kr, p, ap = _rtor_rig()
    hold = int(C['rtor_stop_hold_s'] * HZ)
    assert hold == 10
    states = []
    for _ in range(hold - 1):
        _c, t = apply(kr, ap, v=0.0)
        states.append(kr.last_signal['rtor']['state'])
        assert t == 0.0                                 # K1(d−s0<0)·K2 홀드가 세운다
    assert states == ['hold'] * (hold - 1) and kr._rtor_go is False
    _c, t = apply(kr, ap, v=0.0)                        # 10번째 정지 틱 = 0.5 s → 래치
    assert kr._rtor_go is True and kr.last_signal['rtor']['reason'] == 'latch'
    assert kr.signal_release(ap) is True and kr.last_kr['stop_profile'] is None
    # K2 잔여 — 이 틱 전 정지 9틱, 최소 10틱 → 1틱 남아 이 틱에 소비된다 (t=0), 다음 틱 출발.
    # 즉 정지 총 10틱 = 0.5 s, RTOR hold 와 K2 최소가 같은 틱에 끝난다.
    assert kr.sl_hold_left == 0 and t == 0.0
    _c, t = apply(kr, ap, v=0.0)
    assert t == pytest.approx(C['rtor_go_speed_kph'] / 3.6)
    assert kr.last_kr['stop_hold'] is None and kr.last_kr_winner == 'rtor_cap'


def test_rtor_k2_residual_when_min_hold_longer_than_rtor_hold():
    """rtor 0.5 < stopline_hold_min 이면 그 차이만큼 K2 가 남는다 (틱 단위)."""
    kr, p, ap = _rtor_rig(cfg_with(stopline_hold_min_s=0.8))
    for _ in range(9):
        apply(kr, ap, v=0.0)
    _c, t = apply(kr, ap, v=0.0)                        # 래치 틱 (정지 10틱째)
    assert kr._rtor_go is True
    # sl_stop_ticks 가 이번 틱 전 9 → 잔여 min(11, 16−9) = 7, 이번 틱 소비 후 6
    assert kr.sl_hold_left == 6 and t == 0.0
    for _ in range(6):
        _c, t = apply(kr, ap, v=0.0)
        assert t == 0.0
    _c, t = apply(kr, ap, v=0.0)
    assert t == pytest.approx(C['rtor_go_speed_kph'] / 3.6)


def test_rtor_hold_zero_means_no_stop_at_all():
    """정지 0 s 의 근거: hold 0 이면 조건 1~4·6 만으로 첫 틱에 래치 → 정지 후보 소멸."""
    kr, p, ap = _rtor_rig(cfg_with(rtor_stop_hold_s=0.0))
    _c, t = apply(kr, ap, v=6.0)                        # 6 m/s 로 접근 중 (정지 없음)
    assert kr._rtor_go is True and kr.last_signal['rtor']['reason'] == 'latch'
    assert kr.last_kr['stop_profile'] is None and kr.last_kr['stop_hold'] is None
    assert t == pytest.approx(C['rtor_go_speed_kph'] / 3.6)


def test_rtor_blocked_by_lead_and_by_pedestrian():
    lead = Box(3, 2.0, 0.0, 0.0, half_w=0.9)
    lead.type_id = 'vehicle.vtd.object'
    kr, p, ap = _rtor_rig(actors=[lead])
    for _ in range(12):
        apply(kr, ap, v=0.0)
    assert kr._rtor_go is False and kr.last_signal['rtor']['reason'] == 'lead'
    w = Walker(9, 8.0, 3.0, speed=0.0)
    kr2, p2, ap2 = _rtor_rig(actors=[w])
    for _ in range(12):
        apply(kr2, ap2, v=0.0)
    assert kr2._rtor_go is False and kr2.last_signal['rtor']['state'] == 'wait'


def test_rtor_resets_on_fresh_green():
    kr, p, ap = _rtor_rig()
    for _ in range(11):
        apply(kr, ap, v=0.0)
    assert kr._rtor_go is True
    p.next_traffic_lights = [TL(TrafficLightState.Green)] * len(p.route_s)
    apply(kr, ap, v=0.0)
    assert kr._rtor_go is False and kr.last_signal['rtor']['reason'] == 'reset:green'


def test_rtor_off_switch_never_latches():
    kr, p, ap = _rtor_rig(cfg_with(rtor_enable=False))
    for _ in range(30):
        _c, t = apply(kr, ap, v=0.0)
    assert kr._rtor_go is False and t == 0.0 and 'rtor' not in (kr.last_signal or {})
