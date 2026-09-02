"""
B-8 standoff 잔류 + B-9 연쇄 장애물 병합.

실측 근거 (2026-09-03 정적회피집중 배치 5로그):
  · B-8: 억제 반환·SHIFT_HOLD·span 활성 조기 반환 틱마다 wait_target_d 가 직전
    값으로 얼어 standoff_d 39.0/37.2/33.0/14.3 이 수십 초 유지. 001829/01 은
    시프트 직후 24.2 < 25 로 v_allow 0 → 시프트를 만들고도 정지. 또 25 m 에
    서면 _blocker(20 m) 가 못 잡아 REACTIVE·BREAKOUT 이 서지 않았다.
  · B-9: 003759/01 t=74.6 id3(47.2)·id4(65.3) 18 m 간격 — id3 만 보고 span 을
    만들어 복귀 전이가 id4 위에 떨어졌고 OBB 후보가 5.7 m 에서야 정지 → 접촉.

여기서 지키는 불변:
  · wait_target_d / standoff_id 는 apply 틱 머리에서 None. 억제 반환 틱엔 None.
  · standoff 정지 거리(25)에서 _blocker 가 대상을 잡는다 (blocker_dist_max ≥ 30).
  · chain_gap_m 안에 이어지는 정지 객체는 한 span (first 앞 ~ last 뒤). 밖이면 단일.
  · span 활성 중에도 standoff 가 산출된다. 킬 스위치는 각각 이전 동작.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState, VehicleControl
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                        # noqa: E402
from test_avoid import (HZ, STATIC_TICKS, Ap, Box, GeomPlanner,     # noqa: E402
                        LgOne, legacy_cfg, make, try_overtake)
from test_ped_intent import make_ap                                 # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
STANDOFF = OT['shift_latest_m']
CHAIN = OT['chain_gap_m']
TRANS = OT['transition_m']
BEFORE, AFTER = OT['extra_before_m'], OT['extra_after_m']


def test_params_present():
    assert OT['blocker_dist_max'] >= STANDOFF + 5.0
    assert CHAIN == pytest.approx(AFTER + TRANS)
    assert OT['span_active_standoff_enable'] is True


class RecPlanner(GeomPlanner):
    """shift_route_around_actors 인자를 기록한다 (연쇄 span 검증용)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls = []

    def shift_route_around_actors(self, first_actor, last_actor=None, **kw):
        self.calls.append((first_actor, last_actor, kw))
        return super().shift_route_around_actors(first_actor, last_actor, **kw)


def rig(cfg=CFG, xs=(40.0,), obs_ticks=None):
    """xs [m] 앞에 정지 객체들 (id 2,3,4,…). side 루프까지 들어가는 배치."""
    if obs_ticks is None:
        obs_ticks = int(cfg['overtake']['wait_before_shift_s'] * HZ) + 5
    p = RecPlanner(d_tl=float('inf'))
    p.lg = LgOne()
    kr = KrRules(cfg)
    kr._sl_all = []
    boxes = [Box(2 + i, x, 0.0, 0.0, half_w=0.9) for i, x in enumerate(xs)]
    ap = Ap(p, actors=boxes)
    ap._kr_ego_lane = (1, 0, -1)
    for _ in range(obs_ticks):
        kr._update_obj_timers(ap)
    return kr, p, ap, boxes


# ── B-8 (1): 틱 머리 리셋 ─────────────────────────────────────────────────
def test_apply_resets_standoff_target_each_tick():
    """apply 틱 머리에서 None → 대상이 사라지면 그 틱에 바로 없어진다."""
    p = GeomPlanner(d_tl=float('inf'))
    b = Box(2, 40.0, 0.0, half_w=0.9)
    ap = make_ap(p, [b])
    kr = ap.kr_rules
    for _ in range(STATIC_TICKS + 2):
        kr._update_obj_timers(ap)
    kr.apply(VehicleControl(), 12.5, ap)
    assert kr.wait_target_d == pytest.approx(40.0) and kr.standoff_id == 2
    ap._world._a.remove(b)                                      # 대상 소멸
    kr.apply(VehicleControl(), 12.5, ap)
    assert kr.wait_target_d is None and kr.standoff_id is None


def test_suppression_tick_has_no_standoff():
    """억제 반환(적색, legacy) 틱에는 standoff 가 없다 — 직전 값이 얼지 않는다.
    (queue_only 는 적색이 억제가 아니고 큐 뒤에서도 standoff 가 사는 것이 설계라
    legacy 로 검증한다.)"""
    p = GeomPlanner(d_tl=float('inf'))
    b = Box(2, 40.0, 0.0, half_w=0.9)
    ap = make_ap(p, [b], cfg=legacy_cfg())
    kr = ap.kr_rules
    for _ in range(STATIC_TICKS + 2):
        kr._update_obj_timers(ap)
    kr.apply(VehicleControl(), 12.5, ap)
    assert kr.wait_target_d is not None
    tl = type('TL', (), {'state': TrafficLightState.Red, 'id': 7})()
    p.distances_to_next_traffic_lights[:] = 60.0
    p.next_traffic_lights = [tl] * len(p.route_s)
    kr.apply(VehicleControl(), 12.5, ap)
    assert (kr.last_avoid or {}).get('suppress') == 'red_ahead'
    assert kr.wait_target_d is None and kr.standoff_id is None
    assert 'standoff_d' not in (kr.last_avoid or {})


def test_shift_hold_tick_has_no_stale_standoff():
    """SHIFT_HOLD(적색 + span 활성) 틱에도 standoff 잔류가 없다."""
    kr, p, ap, boxes = rig(xs=(52.3,))
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None
    tl = type('TL', (), {'state': TrafficLightState.Red, 'id': 7})()
    p.distances_to_next_traffic_lights[:] = 60.0
    p.next_traffic_lights = [tl] * len(p.route_s)
    kr.wait_target_d, kr.standoff_id = None, None                # apply 머리 리셋 모사
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.last_avoid['state'] == 'SHIFT_HOLD'
    assert kr.wait_target_d is None


# ── B-8 (2): standoff 정지에서 _blocker ───────────────────────────────────
def test_blocker_catches_target_at_standoff_distance():
    kr, p = make()
    ap = Ap(p, actors=[Box(2, STANDOFF, 0.0, half_w=0.9)])
    assert kr._blocker(ap, p) is not None
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['blocker_dist_max'] = 20.0                   # 옛 값이면 못 잡는다
    kr2 = KrRules(cfg)
    assert kr2._blocker(ap, p) is None


def test_blocked_accounting_arms_at_standoff_stop():
    """25 m 에 정지해 있으면 ot_blocked_ticks 가 쌓여 REACTIVE 가 무장된다."""
    kr, p, ap, _ = rig(xs=(STANDOFF,), obs_ticks=STATIC_TICKS + 2)
    for _ in range(kr.ot_ticks + 1):
        try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_blocked_ticks >= kr.ot_ticks


# ── B-9 (4): 연쇄 병합 ────────────────────────────────────────────────────
def test_two_obstacles_within_gap_merge_into_one_span():
    """실측 재현: 47.2·65.3 (18 m 간격) → first=id2, last=id3 로 한 span."""
    kr, p, ap, boxes = rig(xs=(47.2, 65.3))
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None
    first, last, kw = p.calls[-1]
    assert first.id == 2 and last is not None and last.id == 3
    a = kr.last_avoid
    assert a['chain'] == [2, 3]
    assert a['span_m'] == pytest.approx(2 * TRANS + BEFORE + AFTER + (65.3 - 47.2), abs=0.2)


def test_three_obstacles_chain_iteratively():
    kr, p, ap, boxes = rig(xs=(40.0, 55.0, 70.0))
    try_overtake(kr, ap, p, ego_speed=0.0)
    first, last, _ = p.calls[-1]
    assert (first.id, last.id) == (2, 4)
    assert kr.last_avoid['chain'] == [2, 3, 4]
    assert kr.last_avoid['span_m'] == pytest.approx(2 * TRANS + BEFORE + AFTER + 30.0, abs=0.2)


def test_obstacle_beyond_gap_is_not_chained():
    kr, p, ap, boxes = rig(xs=(40.0, 40.0 + CHAIN + 1.0))
    try_overtake(kr, ap, p, ego_speed=0.0)
    first, last, _ = p.calls[-1]
    assert first.id == 2 and last is None
    assert kr.last_avoid['chain'] == [2]
    assert kr.last_avoid['span_m'] == pytest.approx(2 * TRANS + BEFORE + AFTER, abs=0.2)


def test_chain_breaks_at_first_gap():
    """2번째까지 붙고 3번째가 멀면 2대만 묶는다."""
    kr, p, ap, boxes = rig(xs=(40.0, 55.0, 55.0 + CHAIN + 1.0))
    try_overtake(kr, ap, p, ego_speed=0.0)
    first, last, _ = p.calls[-1]
    assert (first.id, last.id) == (2, 3)


def test_geom_need_uses_first_obstacle():
    """geom 게이트 거리는 첫 객체 기준 — 뒤 객체가 멀어도 첫 객체가 가까우면 기각."""
    kr, p, ap, boxes = rig(xs=(5.3, 20.0))
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is None and 'geom' in kr.last_overtake
    assert kr.last_avoid['s_rel_actor'] == pytest.approx(5.3, abs=0.1)


def test_zone_gate_uses_extended_span():
    """연쇄로 늘어난 span 이 정지선 경계를 넘으면 기각, 단일이면 통과."""
    # queue_only 에서는 정지선 25 m 안의 정지 객체가 큐라 side 루프에 못 들어간다 —
    # 게이트 기하만 보는 검사라 legacy 로 돌린다.
    span1 = 2 * TRANS + BEFORE + AFTER
    kr, p, ap, boxes = rig(legacy_cfg(), xs=(40.0, 55.0))
    kr._sl_all = [span1 + 5.0]                                    # 단일 span 은 통과할 경계
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is None and 'span_into_zone' in kr.last_overtake
    kr, p, ap, boxes = rig(legacy_cfg(), xs=(40.0,))
    kr._sl_all = [span1 + 5.0]
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None


def test_chain_kill_switch():
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['chain_gap_m'] = 0.0
    kr, p, ap, boxes = rig(cfg, xs=(47.2, 65.3))
    try_overtake(kr, ap, p, ego_speed=0.0)
    first, last, _ = p.calls[-1]
    assert first.id == 2 and last is None and kr.last_avoid['chain'] == [2]


# ── B-9 (5): span 활성 중 standoff ────────────────────────────────────────
def test_standoff_evaluated_while_span_active():
    """span 활성 중 회랑(밀린 경로 기준)에 정지 객체가 있으면 standoff 가 선다."""
    kr, p, ap, boxes = rig(xs=(52.3,))
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None
    # 복귀 구간 위 새 장애물 (목 경로는 안 밀리므로 그냥 앞에 둔다)
    b4 = Box(9, 70.0, 0.0, half_w=0.9)
    ap._world._a.append(b4)
    for _ in range(int(OT['standoff_stop_s'] * HZ) + 1):
        kr._update_obj_timers(ap)
    kr.wait_target_d, kr.standoff_id = None, None                # apply 머리 리셋 모사
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.last_avoid['state'] == 'SHIFT_ACTIVE'
    assert kr.wait_target_d == pytest.approx(52.3)              # 가장 가까운 정지 객체
    assert kr._standoff_profile(0.0) == pytest.approx(
        (2.0 * CFG['speed']['stop_profile_a'] * (52.3 - STANDOFF)) ** 0.5)
    assert kr.ot_span is not None                                # 생성은 건너뛴다 (그대로)
    assert len(p.calls) == 1


def test_span_active_kill_switch_restores_old_behaviour():
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['span_active_standoff_enable'] = False
    kr, p, ap, boxes = rig(cfg, xs=(52.3,))
    try_overtake(kr, ap, p, ego_speed=0.0)
    kr.wait_target_d, kr.standoff_id, kr.last_avoid = None, None, None   # apply 머리 모사
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.wait_target_d is None and kr.last_avoid is None             # 아무것도 안 본다
