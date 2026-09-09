"""[6] 차로 지도 커밋 B — 후보 선택·시점·감속 (`avoid_map.lane_map_*`).

커밋 A 의 `lane_map()` 이 재료고, 여기서 **어느 차로로 언제 옮길지**를 정한다.
개입은 둘뿐이다 — `min()` 에 덧대는 **상한형** 속도 후보와, 시프트 **방향 힌트**.
기존 게이트 7개는 그대로 통과해야 한다 (외부 오버라이드 금지).

왜 필요한가 — 실측 `20260909_093415` 에서 quick 7건 중 **4건이 같은 모양으로
죽었다**: 20~22 m 앞 정지 차량, `v_target = 0`, 30 s 무전진.
`tools/avoid_sim` 케이스 2·4 가 같은 상황을 재현하고, 현행 코드는 옆 차로로
한 칸만 시프트한 뒤 **23~25 초** 서 있는다 (3차로는 비어 있는데 못 간다).

지키는 것:
  · 트리거는 **내 차로**의 free_run 뿐이다. 다른 차로가 막힌 것은 트리거가 아니다.
  · 램프 길이는 상수가 아니라 전이 기하에서 나온다 (L = v·π·√(Y/2a)),
    a 는 `overtake.a_lat_max` 를 그대로 읽는다 (_shift_speed_cap 과 같은 축).
  · 후보 = passable ∧ free_run > 램프 + shift_ahead_m. 선택 = free_run 최대,
    동률이면 가까운 쪽.
  · 늦었으면 감속해 램프를 줄이고, 크립보다 느려야 하면 후보 없음(standoff).
  · 두 칸은 **한 번에** 연다 — 한 칸씩 가면 첫 전이가 끝나기 전에 서 버린다.
  · 스위치 off = 이전 동작 (진단 키도 안 생긴다).
"""
import copy
import math
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                        # noqa: E402
from test_avoid import Ap, Box, Planner                             # noqa: E402
from test_lane_map import L0, L1, L2, Lg3, off_cfg, on_cfg, rig     # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
A_LAT = CFG['overtake']['a_lat_max']
AHEAD = CFG['overtake']['shift_ahead_m']
DECIDE = CFG['avoid_map']['lane_map_decide_m']
V_CAP = CFG['avoid_map']['lane_map_avoid_speed_kph'] / 3.6


def obj(oid, x, y):
    """CARLA 좌표에 정지 객체 (목의 오른쪽 이웃은 CARLA y = −3·−6)."""
    return Box(oid, x, y, 0.0, half_w=0.9)


def plan(cfg, actors, **kw):
    kr, p, ap = rig(cfg, actors=actors, **kw)
    return kr, kr.lane_plan(ap, p)


# ── 램프 길이 ─────────────────────────────────────────────────────────────
def test_ramp_length_follows_the_transition_geometry():
    """L = v·π·√(Y / (2·a_lat_max)) — shift_route_smoothly 의 형상에서 나온다."""
    kr, _ = plan(on_cfg(), [])
    for hops, w, v in ((1, 3.0, 5.56), (2, 3.0, 5.56), (2, 3.5, 8.0)):
        Y = hops * w
        assert kr._ramp_len_m(hops, w, v) == pytest.approx(
            v * math.pi * math.sqrt(Y / (2.0 * A_LAT)))


def test_ramp_uses_the_shared_lateral_limit():
    """새 상수를 만들지 않는다 — overtake.a_lat_max 를 그대로 읽는다."""
    kr, _ = plan(on_cfg(), [])
    assert kr.a_lat_max == A_LAT


def test_ramp_grows_with_hops_and_speed():
    kr, _ = plan(on_cfg(), [])
    assert kr._ramp_len_m(2, 3.0, 5.0) > kr._ramp_len_m(1, 3.0, 5.0)
    assert kr._ramp_len_m(1, 3.0, 10.0) > kr._ramp_len_m(1, 3.0, 5.0)
    assert kr._ramp_len_m(0, 3.0, 5.0) == 0.0


# ── 트리거 ────────────────────────────────────────────────────────────────
def test_switch_off_makes_no_plan():
    kr, pl = plan(off_cfg(), [obj(2, 20.0, 0.0)])
    assert pl is None
    assert kr.last_lane_plan is None


def test_far_obstacle_is_not_a_trigger():
    """내 차로가 decide_m 밖까지 뚫려 있으면 개입하지 않는다."""
    _kr, pl = plan(on_cfg(), [obj(2, DECIDE + 20.0, 0.0)])
    assert pl is None


def test_other_lane_blocked_is_not_a_trigger():
    """다른 차로만 막힌 것은 트리거가 아니다 (avoid_sim 케이스 10)."""
    _kr, pl = plan(on_cfg(), [obj(2, 20.0, -6.0)])
    assert pl is None


def test_my_lane_blocked_triggers():
    _kr, pl = plan(on_cfg(), [obj(2, 25.0, 0.0)])
    assert pl is not None
    assert pl['trigger_free_m'] == pytest.approx(25.0, abs=1.0)


# ── 후보 선택 ─────────────────────────────────────────────────────────────
def test_picks_the_free_lane_one_hop_away():
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0)])
    assert pl['pick'] == str(list(L1))
    assert pl['hops'] == 1
    assert pl['side'] == 'right'


def test_skips_a_blocked_neighbour_and_takes_two_hops():
    """avoid_sim 케이스 2 의 배치 — 1·2차로가 막히고 3차로가 비었다."""
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0), obj(3, 42.0, -3.0)])
    assert pl['pick'] == str(list(L2))
    assert pl['hops'] == 2


def test_no_candidate_when_everything_is_blocked():
    """옆 차로들이 **램프도 못 넣을 만큼** 가까이 막혔으면 후보가 없다.

    (옆 차로가 내 차로보다 조금이라도 더 뚫려 있고 램프가 들어가면 그건
     정당한 후보다 — 40/42/42 는 후보가 있는 배치다.)
    """
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0), obj(3, 8.0, -3.0),
                              obj(4, 8.0, -6.0)])
    assert pl['pick'] is None
    assert pl['why'] == 'no_candidate'


def test_impassable_lane_is_not_a_candidate():
    """소멸 차로(폭 < min_width)는 후보가 아니다 — 옆 칸으로 밀린다."""
    narrow = Lg3(w2=0.5)
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0), obj(3, 8.0, -3.0)], lg=narrow)
    assert pl['pick'] is None                          # L1 은 가깝게 막히고 L2 는 좁다
    kr2, pl2 = plan(on_cfg(), [obj(2, 40.0, 0.0)], lg=narrow)
    assert pl2['pick'] == str(list(L1))                # L1 이 뚫렸으면 그쪽
    assert str(list(L2)) not in (pl2.get('cands') or {})


def test_candidate_needs_room_for_the_ramp():
    """free_run 이 램프 + shift_ahead_m 보다 짧으면 후보가 아니다."""
    kr, _ = plan(on_cfg(), [])
    need = kr._ramp_len_m(1, 3.0, V_CAP) + AHEAD
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0), obj(3, need - 2.0, -3.0)])
    assert pl['pick'] != str(list(L1))


# ── 시점·감속 ─────────────────────────────────────────────────────────────
def test_speed_cap_is_the_avoid_speed():
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0)])
    assert pl['v_cap'] == pytest.approx(V_CAP, abs=0.01)


def test_late_decision_lowers_the_cap_instead_of_giving_up():
    """램프 시작점을 지났으면 **더 감속해** 램프를 줄인다."""
    kr, pl = plan(on_cfg(), [obj(2, 18.0, 0.0)])
    assert pl['pick'] is not None
    assert pl['late'] is True
    assert pl['v_cap'] < V_CAP


def test_too_late_gives_up_to_standoff():
    """크립보다 느려야 겨우 되는 램프면 후보 없음 — standoff 에 맡긴다."""
    _kr, pl = plan(on_cfg(), [obj(2, AHEAD + 0.5, 0.0)])
    assert pl['pick'] is None
    assert pl['why'] == 'ramp_too_late'


def test_start_s_is_before_the_first_obstacle():
    kr, pl = plan(on_cfg(), [obj(2, 45.0, 0.0)])
    ramp = pl['ramp_m']
    assert pl['start_s_rel'] == pytest.approx(45.0 - AHEAD - ramp, abs=1.0)


# ── 다단 시프트 ───────────────────────────────────────────────────────────
def test_two_hops_are_opened_at_once():
    """한 칸씩 두 번은 못 쓴다 — 첫 전이가 끝나기 전에 서 버린다."""
    kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0), obj(3, 42.0, -3.0)])
    assert kr._lm_hops('right') == 2
    assert kr._lm_hops('left') is None                 # 지도가 고른 쪽만


def test_one_hop_passes_no_step_override():
    """한 칸이면 route.py 에 n_steps 를 넘기지 않는다 (이전 서명 그대로)."""
    kr, _pl = plan(on_cfg(), [obj(2, 40.0, 0.0)])
    assert kr._lm_hops('right') is None


def test_hops_are_none_when_off():
    kr, _pl = plan(off_cfg(), [obj(2, 40.0, 0.0), obj(3, 42.0, -3.0)])
    assert kr._lm_hops('right') is None


# ── 큐는 그대로 억제된다 ─────────────────────────────────────────────────
def test_queue_is_still_suppressed():
    """신호 대기 줄은 지도에서 빠지므로 트리거되지 않는다 (억제 기준 단일 출처)."""
    _kr, pl = plan(on_cfg(), [obj(2, 25.0, 0.0)], queue=True)
    assert pl is None


# ── 커밋 C: 복귀 없음 ────────────────────────────────────────────────────
def nr_cfg(**over):
    c = on_cfg(**over)
    c['avoid_map']['lane_map_no_return_enable'] = True
    return c


def with_vel(p, turn_s=150.0, lanes=(L0,)):
    """route.pkl 의 valid_entry_lanes 를 심는다 — 데드라인의 유일한 재료다."""
    p.route = dict(getattr(p, 'route', None) or {})
    p.route['waypoint_s'] = [0.0, float(turn_s)]
    p.route['valid_entry_lanes'] = [
        {'seg': 0, 'target': 'pair', 'turn': 'left',
         'lanes': [list(k) for k in lanes]}]
    return p


def test_deadline_uses_valid_entry_lanes():
    """데드라인 = 세그먼트 끝 − 복귀 전이거리 − never_stall_turn_margin_m.

    `_ns_turn_lane_pending`(never_stall (c)) 와 **같은 식**이어야 한다.
    """
    kr, p, ap = rig(nr_cfg(), actors=[])
    with_vel(p, turn_s=150.0)
    need = max(kr.ot_trans_m, kr.shift_k_s * 8.0) + kr.ns_turn_margin_m
    assert kr._lm_deadline_s(p, ap, 8.0) == pytest.approx(150.0 - need)


def test_no_deadline_without_valid_entry_lanes():
    """제약이 없으면 미루지 않는다 — 경로 끝까지 밀면 span 이 영영 안 풀린다."""
    kr, p, ap = rig(nr_cfg(), actors=[])
    assert kr._lm_deadline_s(p, ap, 8.0) is None


def test_no_deadline_for_a_finish_segment():
    """target='finish' 는 '제약 없음' 이라 데드라인이 아니다."""
    kr, p, ap = rig(nr_cfg(), actors=[])
    with_vel(p)
    p.route['valid_entry_lanes'][0]['target'] = 'finish'
    assert kr._lm_deadline_s(p, ap, 8.0) is None


def test_extra_after_is_pushed_to_the_deadline():
    kr, p, ap = rig(nr_cfg(), actors=[obj(2, 40.0, 0.0)])
    with_vel(p, turn_s=150.0)
    kr.lane_plan(ap, p)
    chain = {'first': ap._world.get_actors()[0], 'last': ap._world.get_actors()[0]}
    got = kr._lm_no_return_m(p, ap, 8.0, chain)
    need = max(kr.ot_trans_m, kr.shift_k_s * 8.0) + kr.ns_turn_margin_m
    assert got == pytest.approx((150.0 - need) - 40.0, abs=1.0)
    assert got > kr.ot_after_m                             # 기본 10 m 보다 멀리


def test_switch_off_keeps_the_old_return():
    kr, p, ap = rig(on_cfg(), actors=[obj(2, 40.0, 0.0)])   # no_return 은 off
    with_vel(p)
    kr.lane_plan(ap, p)
    chain = {'first': ap._world.get_actors()[0], 'last': ap._world.get_actors()[0]}
    assert kr._lm_no_return_m(p, ap, 8.0, chain) is None


def test_no_push_when_the_deadline_is_already_close():
    """데드라인이 기존 extra_after 보다 가까우면 미루지 않는다 (되돌리지 않는다)."""
    kr, p, ap = rig(nr_cfg(), actors=[obj(2, 40.0, 0.0)])
    with_vel(p, turn_s=55.0)
    kr.lane_plan(ap, p)
    chain = {'first': ap._world.get_actors()[0], 'last': ap._world.get_actors()[0]}
    assert kr._lm_no_return_m(p, ap, 8.0, chain) is None


def test_no_push_without_a_plan():
    """지도가 목표를 안 골랐으면 관여하지 않는다."""
    kr, p, ap = rig(nr_cfg(), actors=[obj(2, 40.0, 0.0)])
    with_vel(p)
    kr.last_lane_plan = None
    chain = {'first': ap._world.get_actors()[0], 'last': ap._world.get_actors()[0]}
    assert kr._lm_no_return_m(p, ap, 8.0, chain) is None


# ── 두 칸 시프트: 게이트가 최종 목표를 본다 ──────────────────────────────
def test_nth_neighbor_walks_the_chain():
    kr, p, ap = rig(nr_cfg(), actors=[])
    lg = p.lg
    assert kr._nth_neighbor(lg, L0, 'right', 1) == L1
    assert kr._nth_neighbor(lg, L0, 'right', 2) == L2
    assert kr._nth_neighbor(lg, L0, 'right', 3) is None     # 끊기면 None
    assert kr._nth_neighbor(lg, L0, 'left', 1) is None


def test_gate_target_is_the_final_lane_not_the_middle_one():
    """옛 코드는 언제나 바로 옆(1칸)만 봤다 — 두 칸 회피에서 중간 차로가 막히면
    `occupied` 로 매 틱 기각돼 지도가 찾은 빈 차로로 영영 못 갔다.

    실측 `tools/avoid_sim` 케이스 11 (내 차로 55 m, 중간 차로 **22 m**,
    끝 차로 빔): `right:occupied@p1` **129 → 0**, 정지 **9.1 → 5.3 s**.
    (케이스 2·4 는 중간 객체가 clear_radius_m(30) 밖이라 이 경로를 안 밟는다.)
    """
    kr, p, ap = rig(nr_cfg(), actors=[obj(2, 40.0, 0.0), obj(3, 42.0, -3.0)])
    kr.lane_plan(ap, p)
    assert kr._lm_hops('right') == 2
    assert kr._nth_neighbor(p.lg, L0, 'right', kr._lm_hops('right')) == L2


def test_middle_lane_is_checked_only_inside_the_ramp():
    """중간 차로는 목적지가 아니다 — 램프가 훑는 s 구간 밖 객체는 안 본다."""
    kr, p, ap = rig(nr_cfg(), actors=[obj(2, 40.0, 0.0), obj(3, 42.0, -3.0)])
    lp = kr.lane_plan(ap, p)
    reach = lp['ramp_m'] + kr.shift_ahead_m
    assert reach < 42.0                                    # 중간 객체는 램프 밖
    assert kr._mid_lanes_clear(p.lg, p, ap, L0, 'right', 2) is True


def test_middle_lane_blocks_when_inside_the_ramp():
    kr, p, ap = rig(nr_cfg(), actors=[obj(2, 40.0, 0.0), obj(3, 6.0, -3.0)])
    kr.lane_plan(ap, p)
    assert kr._mid_lanes_clear(p.lg, p, ap, L0, 'right', 2) is False


def test_one_hop_keeps_the_old_target():
    """한 칸이면 이전과 글자 그대로 같다 — `_lm_hops` 가 None 이라 n=1 이다."""
    kr, p, ap = rig(nr_cfg(), actors=[obj(2, 40.0, 0.0)])
    kr.lane_plan(ap, p)
    assert kr._lm_hops('right') is None
    assert kr._nth_neighbor(p.lg, L0, 'right', 1) == L1


def test_moving_object_does_not_block_the_middle_lane():
    """움직이는 것은 지나간다 — 정지 객체만 램프를 막는다."""
    kr, p, ap = rig(nr_cfg(), actors=[obj(2, 40.0, 0.0), obj(3, 42.0, -3.0)])
    kr.lane_plan(ap, p)
    mover = Box(9, 6.0, -3.0, 5.0, half_w=0.9)
    ap._world._a.append(mover)
    assert kr._mid_lanes_clear(p.lg, p, ap, L0, 'right', 2) is True


# ── [2] 보행자는 회피 대상이 아니다 ──────────────────────────────────────
class Walker(Box):
    """`VtdActor(cls='pedestrian')` 과 같은 표면 — type_id 로 갈린다."""

    def __init__(self, oid, x, y, speed=0.0):
        super().__init__(oid, x, y, speed, half_w=0.3)
        self.type_id = 'walker.vtd.pedestrian'


def test_pedestrian_does_not_shorten_free_run():
    """서 있는 보행자가 free_run 을 깎으면 시프트 후보가 만들어진다 —
    보행자는 **세우는** 축(_ped_intent·walker_hazard)이 맡는다."""
    kr, p, ap = rig(on_cfg(), actors=[Walker(2, 20.0, 0.0)])
    m = kr.lane_map(ap, p)
    assert m['blocked_by'] == {}
    assert m['free_run'][str(list(L0))] == pytest.approx(kr.lane_map_ahead_m)


def test_pedestrian_does_not_trigger_a_plan():
    _kr, pl = plan(on_cfg(), [Walker(2, 20.0, 0.0)])
    assert pl is None


def test_vehicle_still_shortens_free_run():
    """대조 — 같은 자리 차량은 그대로 잡힌다 (필터가 너무 넓지 않다)."""
    kr, p, ap = rig(on_cfg(), actors=[obj(2, 20.0, 0.0)])
    m = kr.lane_map(ap, p)
    assert m['blocked_by'].get(str(list(L0))) == 2
