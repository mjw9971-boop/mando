"""
억제 단일화 (C) — suppress_mode: queue_only.

목표 규칙:
  억제      _is_queue 하나. 적색·정지선·교차로는 억제가 아니라 일시정지/게이트 입력.
  큐        정지 객체 ≥1 ∧ (A 선두가 정지선 queue_head_max_m 안
                          ∨ B 신호 Red/Yellow ∧ 정지 객체 전부가 자차~정지선 사이)
  해제      신호 있음 → 녹색 q_green_release_s 경과 (선두가 떠나면 즉시 해소)
            무신호   → q_nosignal_release_s 경과 ∧ 보행자 가드 아님
  가드      회랑 안(|lat| < ped_release_lat_m) 또는 접근 중(v_toward > 0) 보행자.
            서 있는 인도 보행자는 가드가 아니다.
  standoff  큐 여부와 무관하게 항상. 정지선 너머 객체 제외.
  캐시      q_ticks 는 apply 의 _tick_cache 에서만 +1 (틱당 1회).
  legacy    옛 3중 억제 바이트 동일 (test_avoid·test_priority_signal 이 legacy 로 검증).
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                          # noqa: E402
from test_avoid import (HZ, STATIC_TICKS, Ap, Box, GeomPlanner, LgOne,  # noqa: E402
                        blocked_rig, drive, legacy_cfg, make, try_overtake)
from test_ped_intent import Walker                                    # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
STOP_TICKS = int(round(OT['standoff_stop_s'] * HZ))
GREEN_TICKS = int(round(OT['q_green_release_s'] * HZ))
NOSIG_TICKS = int(round(OT['q_nosignal_release_s'] * HZ))
QHEAD = OT['queue_head_max_m']


class TL:
    def __init__(self, state, tl_id=7):
        self.state, self.id = state, tl_id


def set_signal(p, state, d_tl):
    p.distances_to_next_traffic_lights[:] = float(d_tl)
    p.next_traffic_lights = [None if state is None else TL(state)] * len(p.route_s)


def rig(xs=(10.0,), state=TrafficLightState.Red, d_tl=60.0, cfg=CFG, ticks=None):
    """정지 객체 xs [m] 앞, 신호 state 가 d_tl 앞. 정지 관찰을 쌓고 캐시까지 채운다."""
    kr, p = make(cfg, d_tl=d_tl)
    if state is not None:
        p.next_traffic_lights = [TL(state)] * len(p.route_s)
    kr._sl_all = []
    ap = Ap(p, [Box(i + 2, x, 0.0, half_w=0.9) for i, x in enumerate(xs)])
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    kr.last_d_end = 1e6
    for _ in range((STATIC_TICKS + 2) if ticks is None else ticks):
        kr._update_obj_timers(ap)
    kr._tick_cache(ap, p)
    return kr, p, ap


def test_params_present():
    assert OT['suppress_mode'] == 'queue_only'
    assert OT['q_green_release_s'] == 3.0 and OT['q_nosignal_release_s'] == 10.0
    assert OT['standoff_stop_s'] == 1.0


# ── 큐 1대 판정 ───────────────────────────────────────────────────────────
def test_single_vehicle_queue_cond_A_head_near_stopline():
    """녹색이어도 선두가 정지선 25 m 안이면 큐 (A)."""
    kr, p, ap = rig(xs=(40.0,), state=TrafficLightState.Green, d_tl=60.0)  # 선두-정지선 20
    assert kr._tick_queue is True and kr.q_info['cond'] == 'A'
    try_overtake(kr, ap, p)
    assert kr.last_avoid['suppress'] == 'queue' and kr.last_avoid['queue']['head_id'] == 2


def test_single_vehicle_queue_cond_B_red_between_ego_and_stopline():
    """적색이고 정지 객체가 자차~정지선 사이면 정지선까지 멀어도 큐 (B)."""
    kr, p, ap = rig(xs=(10.0,), state=TrafficLightState.Red, d_tl=60.0)   # 선두-정지선 50
    assert kr._tick_queue is True and kr.q_info['cond'] == 'B'


def test_neither_condition_is_obstacle_not_queue():
    """녹색 ∧ 선두가 정지선 25 m 밖 → 큐 아님 = 장애물 (PREEMPT/WAIT 경로)."""
    kr, p, ap = rig(xs=(10.0,), state=TrafficLightState.Green, d_tl=60.0)
    assert kr._tick_queue is False and kr.q_reject == 'head_far'
    try_overtake(kr, ap, p)
    assert kr.last_avoid['state'] in ('PREEMPT', 'WAIT', 'WAIT_EXPIRED', 'REACTIVE')


def test_red_beyond_stopline_object_is_not_queue():
    """적색인데 정지 객체가 정지선 너머면 B 도 아니다."""
    kr, p, ap = rig(xs=(70.0,), state=TrafficLightState.Red, d_tl=60.0)
    assert kr._tick_queue is False


def test_no_red_ahead_suppression_in_queue_only():
    """적색 자체는 억제가 아니다 — 회랑이 비면 SUPPRESS 가 없다."""
    kr, p = make(d_tl=60.0)
    p.next_traffic_lights = [TL(TrafficLightState.Red)] * len(p.route_s)
    ap = Ap(p, [])
    try_overtake(kr, ap, p)
    assert (kr.last_avoid or {}).get('suppress') is None


# ── 해제 ──────────────────────────────────────────────────────────────────
def test_green_release_after_3s_when_head_still_stopped():
    kr, p, ap = rig(xs=(40.0,), state=TrafficLightState.Green, d_tl=60.0)
    for i in range(GREEN_TICKS):
        kr.green_since_ticks = i                                  # apply 가 세는 값 모사
        kr._tick_cache(ap, p)
        assert kr._tick_queue is True, i
    kr.green_since_ticks = GREEN_TICKS
    kr._tick_cache(ap, p)
    assert kr._tick_queue is False and kr.q_reject == 'green_expired'
    try_overtake(kr, ap, p)
    assert kr.last_avoid['state'] in ('PREEMPT', 'WAIT', 'WAIT_EXPIRED')


def test_green_counter_tracks_signal_state_in_apply():
    """green_since_ticks 는 녹색 연속 틱, 다른 색이면 0, 신호 id 바뀌면 0."""
    from vtd_adapter.carla_types import VehicleControl
    from test_ped_intent import make_ap
    p = GeomPlanner(d_tl=60.0)
    ap = make_ap(p, [])
    kr = ap.kr_rules
    set_signal(p, TrafficLightState.Green, 60.0)
    for _ in range(5):
        kr.apply(VehicleControl(), 12.5, ap)
    assert kr.green_since_ticks == 5
    set_signal(p, TrafficLightState.Red, 60.0)
    kr.apply(VehicleControl(), 12.5, ap)
    assert kr.green_since_ticks == 0
    set_signal(p, TrafficLightState.Green, 60.0)
    kr.apply(VehicleControl(), 12.5, ap)
    p.next_traffic_lights = [TL(TrafficLightState.Green, tl_id=9)] * len(p.route_s)
    kr.apply(VehicleControl(), 12.5, ap)
    assert kr.green_since_ticks == 1                              # id 바뀌어 리셋 후 +1


def test_head_departure_dissolves_queue_immediately():
    kr, p, ap = rig(xs=(10.0,), state=TrafficLightState.Red, d_tl=60.0)
    assert kr._tick_queue is True
    ap._world._a[0].speed = OT['blocker_speed_max'] + 1.0            # 선두 출발
    kr._update_obj_timers(ap)
    kr._tick_cache(ap, p)
    assert kr._tick_queue is False and kr.q_ticks == 0


def test_no_signal_release_after_10s_without_ped_guard():
    kr, p, ap = rig(xs=(10.0,), state=None, d_tl=float('inf'))
    kr._sl_all = [30.0]                                            # 무신호 정지선 (선두 20 m 앞)
    kr._tick_cache(ap, p)
    assert kr._tick_queue is True and kr.q_info['cond'] == 'A'
    for _ in range(NOSIG_TICKS - 2):
        kr._tick_cache(ap, p)
    assert kr._tick_queue is True
    kr._tick_cache(ap, p)
    assert kr._tick_queue is False and kr.q_reject == 'hold_expired'
    assert kr.q_info['ped_guard'] is False


def test_no_signal_release_blocked_by_ped_in_corridor_or_approaching():
    kr, p, ap = rig(xs=(10.0,), state=None, d_tl=float('inf'))
    kr._sl_all = [30.0]
    w = Walker(4, 15.0, 1.0)                                       # 회랑 안 (|lat| 1.0 < 2.5)
    ap._world._a.append(w)
    for _ in range(NOSIG_TICKS + 5):
        kr._update_obj_timers(ap)
        kr._tick_cache(ap, p)
    assert kr._tick_queue is True and kr.q_info['ped_guard'] is True
    w._y = 5.0                                                     # 인도로 나갔지만 접근 중
    kr.ped_vt[4] = +0.5
    kr._tick_cache(ap, p)
    assert kr._tick_queue is True and kr.q_info['ped_guard'] is True


def test_standing_sidewalk_pedestrian_is_not_a_guard():
    kr, p, ap = rig(xs=(10.0,), state=None, d_tl=float('inf'))
    kr._sl_all = [30.0]
    w = Walker(4, 15.0, 5.0)                                       # 인도에 서 있음
    ap._world._a.append(w)
    kr.ped_vt[4] = 0.0
    for _ in range(NOSIG_TICKS + 2):
        kr._update_obj_timers(ap)
        kr._tick_cache(ap, p)
    assert kr._tick_queue is False and kr.q_reject == 'hold_expired'
    assert kr.q_info['ped_guard'] is False


def test_ped_vt_is_stored_by_ped_intent():
    """_ped_intent 가 매 틱 v_toward 를 ped_vt 에 남긴다 (다음 틱 가드용)."""
    kr, p = make()
    w = Walker(4, 20.0, -5.0)
    ap = Ap(p, actors=[w])
    kr._update_obj_timers(ap)
    kr._ped_intent(p, ap, 0.0)
    w._y = -4.9
    kr._ped_intent(p, ap, 0.0)
    assert kr.ped_vt[4] == pytest.approx(0.1 * HZ, abs=0.05)
    ap._world._a.remove(w)
    kr._ped_intent(p, ap, 0.0)
    assert 4 not in kr.ped_vt


# ── 캐시 / 이중 증가 ───────────────────────────────────────────────────────
def test_q_ticks_increments_once_per_tick():
    kr, p, ap = rig(xs=(10.0,), state=TrafficLightState.Red, d_tl=60.0)
    q0 = kr.q_ticks
    kr._tick_cache(ap, p)
    kr._try_overtake(ap, p, 0.0)
    kr._obstacle_cause(p, ap)
    kr._breakout_tick(p, ap, 0.0)
    assert kr.q_ticks == q0 + 1


def test_obstacle_cause_false_behind_queue():
    kr, p, ap = rig(xs=(10.0,), state=TrafficLightState.Green, d_tl=30.0)   # cond A
    assert kr._tick_queue is True
    assert kr._obstacle_cause(p, ap) is False
    kr2, p2, ap2 = rig(xs=(10.0,), state=TrafficLightState.Green, d_tl=60.0)  # 장애물
    assert kr2._tick_queue is False
    assert kr2._obstacle_cause(p2, ap2) is True


def test_breakout_pauses_in_junction_instead_of_reset():
    kr, p, ap = blocked_rig(junction=True)
    drive(kr, p, ap, 5)
    assert kr.bo_paused is True and kr.bo_state is None


def test_breakout_pauses_under_red_and_resumes_green():
    """적색 = 일시정지 (억제 아님). 녹색 복귀 후 카운터가 이어진다."""
    kr, p, ap = rig(xs=(8.0,), state=TrafficLightState.Green, d_tl=60.0)
    drive(kr, p, ap, 20)
    assert kr.bo_stuck_ticks == 20
    p.next_traffic_lights = [TL(TrafficLightState.Red)] * len(p.route_s)
    kr._tick_cache(ap, p)
    drive(kr, p, ap, 20)
    assert kr.bo_paused is True and kr.bo_stuck_ticks == 20
    p.next_traffic_lights = [TL(TrafficLightState.Green)] * len(p.route_s)
    kr._tick_cache(ap, p)
    drive(kr, p, ap, 1)
    assert kr.bo_stuck_ticks == 21


# ── standoff ──────────────────────────────────────────────────────────────
def test_standoff_target_excludes_objects_beyond_stopline():
    kr, p, ap = rig(xs=(10.0, 70.0), state=TrafficLightState.Red, d_tl=60.0)
    try_overtake(kr, ap, p)
    assert kr.wait_target_d == pytest.approx(10.0) and kr.standoff_id == 2
    kr2, p2, ap2 = rig(xs=(70.0,), state=TrafficLightState.Red, d_tl=60.0)
    try_overtake(kr2, ap2, p2)
    assert kr2.wait_target_d is None


def test_standoff_alive_while_queued():
    """큐로 억제돼도 standoff 는 산다 — 큐 뒤 25 m 정차가 설계."""
    kr, p, ap = rig(xs=(10.0,), state=TrafficLightState.Red, d_tl=60.0)
    try_overtake(kr, ap, p)
    assert kr.last_avoid['suppress'] == 'queue'
    assert kr.wait_target_d == pytest.approx(10.0)


def test_standoff_target_uses_stop_ok_not_static():
    """정지 1.0 s 면 standoff 대상 (정적 관찰 3 s 전)."""
    kr, p, ap = rig(xs=(40.0,), state=TrafficLightState.Green, d_tl=200.0, ticks=STOP_TICKS + 1)
    try_overtake(kr, ap, p)
    assert kr.wait_target_d == pytest.approx(40.0)


# ── 라벨 형식 (C-8) ───────────────────────────────────────────────────────
class LgNone(LgOne):
    def neighbor(self, key, side):
        return None


class LgJunction(LgOne):
    def __init__(self):
        super().__init__()
        self.lanes = {k: {'junction': 8} for k in self.lanes}


def _side_rig(lg):
    from test_avoid import _geom_rig
    kr, p, ap = _geom_rig(CFG, 52.3)
    p.lg = lg
    return kr, p, ap


def test_no_neighbor_label_per_side():
    kr, p, ap = _side_rig(LgNone())
    try_overtake(kr, ap, p)
    assert kr.last_overtake == 'right:no_neighbor@p1'
    assert kr.last_avoid['reject'] == 'right:no_neighbor' and kr.last_avoid['pass'] == 1


def test_junction_and_no_lane_labels():
    kr, p, ap = _side_rig(LgJunction())
    try_overtake(kr, ap, p)
    assert kr.last_overtake == 'junction' and kr.last_avoid['reject'] == 'junction'
    kr, p, ap = _side_rig(None)
    ap._kr_ego_lane = None
    try_overtake(kr, ap, p)
    assert kr.last_overtake == 'no_lane' and kr.last_avoid['reject'] == 'no_lane'


# ── legacy 모드 ───────────────────────────────────────────────────────────
def test_legacy_mode_keeps_red_ahead_suppression_and_no_cache():
    kr, p, ap = rig(xs=(10.0,), state=TrafficLightState.Red, d_tl=60.0, cfg=legacy_cfg())
    assert kr._tick_queue is False and kr._tick_corridor == []
    try_overtake(kr, ap, p)
    assert kr.last_avoid['suppress'] == 'red_ahead'
    assert kr.wait_target_d is None                                 # 옛 동작: 적색이면 리셋


def test_legacy_mode_keeps_15s_hold_and_two_vehicle_shape():
    kr, p, ap = rig(xs=(10.0, 25.0), state=None, d_tl=40.0, cfg=legacy_cfg())
    corr = kr._corridor_blockers(ap, p)
    assert kr._is_queue(corr, p, ap) is True
    for _ in range(kr.q_hold_ticks + 1):
        kr._is_queue(corr, p, ap)
    assert kr.q_reject == 'hold_expired'
