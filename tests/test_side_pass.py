"""
묶음 B 작업2 — side 루프 게이트 재구성 (B-1 ~ B-5).

  B-1  span_into_zone 여유 = zone_gate_margin_m (기본 0, sup_m 과 분리)
  B-2  BREAKOUT 단계 연동: lvl ≥ zone_gate_relax_level 이면 zone 건너뜀,
       lvl ≥ geom_relax_level 이면 전이 시작 여유 shift_ahead_l3_m → need 15
  B-3  solid 두 바퀴: 1바퀴 solid 강제, solid 기각으로 양쪽 실패 시에만 2바퀴
  B-4  standoff = shift_latest_m 25
  B-5  standoff 대상 = 신호 무관 정지 카운터(obj_stop_ticks ≥ standoff_stop_s)

여기서 지키는 불변:
  · center_line·kappa·lc_overlap 은 어느 단계·어느 바퀴에서도 풀지 않는다.
  · occupied 의 lvl<1 은 2바퀴에서도 그대로다 (별개 축).
  · 킬 스위치: zone_gate_margin_m 30 / relax_level 99 / solid_second_pass_enable
    false / standoff_stop_s 0 → 각각 이전 동작.
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
from test_avoid import (HZ, STATIC_TICKS, Ap, Box, GeomPlanner,     # noqa: E402
                        LgOne, Planner, _geom_rig, make)

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
TRANS = OT['transition_m']
AHEAD = OT['shift_ahead_m']
AHEAD_L3 = OT['shift_ahead_l3_m']
MARGIN = OT['shift_geom_margin_m']
SPAN0 = 2 * TRANS + OT['extra_before_m'] + OT['extra_after_m']   # 정지 시 span 39
ZONE_M = OT['zone_gate_margin_m']
ZONE_LVL = OT['zone_gate_relax_level']
GEOM_LVL = OT['geom_relax_level']
STANDOFF = OT['shift_latest_m']
A_STOP = CFG['speed']['stop_profile_a']
FAR = 52.3                                       # geom·zone 을 넉넉히 통과하는 거리


def test_params_present():
    assert ZONE_M == 0.0 and ZONE_LVL == 2 and GEOM_LVL == 3 and AHEAD_L3 == 1.0
    assert OT['solid_second_pass_enable'] is True
    assert STANDOFF == 25.0 and OT['standoff_stop_s'] == 1.5
    assert TRANS + AHEAD_L3 + MARGIN == pytest.approx(15.0)      # 정지 시 L3 need


def rig(cfg=CFG, obj_x=FAR, lg=None, stopline=None, lvl=0):
    kr, p, ap = _geom_rig(cfg, obj_x)
    if lg is not None:
        p.lg = lg
    if stopline is not None:
        kr._sl_all = [float(stopline)]
    if lvl:
        kr.bo_state, kr.bo_level = 'BREAKOUT', lvl
    return kr, p, ap


# ── B-1: zone 여유 경계 ───────────────────────────────────────────────────
def test_zone_gate_boundary_at_margin_zero():
    """span 끝 == 경계면 통과, 조금이라도 넘으면 기각 (여유 0 → 정지선 자체)."""
    kr, p, ap = rig(stopline=SPAN0 + ZONE_M)                      # span_end == zone_lo
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is not None and kr.last_overtake == 'right'
    kr, p, ap = rig(stopline=SPAN0 + ZONE_M - 0.1)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is None and kr.last_overtake == 'right:span_into_zone@p1'
    a = kr.last_avoid
    assert a['reject'] == 'right:span_into_zone' and a['pass'] == 1
    assert a['zone_lo'] == pytest.approx(SPAN0 - 0.1, abs=0.05)


def test_zone_gate_margin_param_shifts_boundary():
    """zone_gate_margin_m 만큼 경계가 정지선 앞으로 온다 (30 = B-1 이전)."""
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['zone_gate_margin_m'] = 5.0
    kr, p, ap = rig(cfg, stopline=SPAN0 + 5.0)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is not None
    kr, p, ap = rig(cfg, stopline=SPAN0 + 4.9)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is None and 'span_into_zone' in kr.last_overtake


def test_signal_zone_still_uses_suppress_m():
    """_signal_zone 은 그대로 sup_m — zone 여유와 결합이 없다."""
    kr, p = make(d_tl=OT['stopline_suppress_m'] - 1.0)
    ap = Ap(p)
    assert kr._signal_zone(p, ap) is not None
    kr2, p2 = make(d_tl=OT['stopline_suppress_m'] + 1.0)
    assert kr2._signal_zone(p2, Ap(p2)) is None


# ── B-2: 단계 연동 ────────────────────────────────────────────────────────
def test_zone_gate_skipped_at_relax_level():
    """lvl ≥ zone_gate_relax_level 이면 zone 게이트를 건너뛴다. 그 아래는 유지."""
    # span(39)이 경계를 넘되 _signal_zone 억제창(sup_m 30) 밖인 정지선 — 35 m
    near = SPAN0 - 4.0
    assert near > OT['stopline_suppress_m']
    kr, p, ap = rig(stopline=near, lvl=ZONE_LVL - 1)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is None and 'span_into_zone' in kr.last_overtake
    kr, p, ap = rig(stopline=near, lvl=ZONE_LVL)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is not None and kr.last_overtake == 'right'


def test_zone_relax_kill_switch():
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['zone_gate_relax_level'] = 99
    kr, p, ap = rig(cfg, stopline=SPAN0 - 4.0, lvl=4)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is None and 'span_into_zone' in kr.last_overtake


def test_geom_need_is_15_at_L3():
    """정지 시 L3 이상은 need = 12 + 1 + 2 = 15. 그 아래는 19."""
    need0 = TRANS + AHEAD + MARGIN
    need3 = TRANS + AHEAD_L3 + MARGIN
    assert need3 == pytest.approx(15.0)
    x = (need0 + need3) / 2.0                                      # 15 < x < 19
    kr, p, ap = rig(obj_x=x, lvl=GEOM_LVL - 1)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is None and kr.last_overtake == 'right:geom@p1'
    assert kr.last_avoid['need_geom'] == pytest.approx(need0, abs=0.05)
    kr, p, ap = rig(obj_x=x, lvl=GEOM_LVL)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is not None
    assert kr.last_avoid['ahead_m'] == pytest.approx(AHEAD_L3)
    assert kr.ot_span[0] == p.route_index + int(AHEAD_L3 * 10)     # 실제 시프트도 1.0 m 앞
    kr, p, ap = rig(obj_x=need3 - 0.2, lvl=GEOM_LVL)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is None and kr.last_avoid['need_geom'] == pytest.approx(need3, abs=0.05)


def test_geom_transition_floor_stays_12_at_L3():
    """L3 완화는 전이 시작 여유만 줄인다 — trans_m 하한 12 는 그대로."""
    kr, p, ap = rig(obj_x=FAR, lvl=GEOM_LVL)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.last_avoid['trans_m'] == pytest.approx(TRANS)


def test_geom_relax_kill_switch():
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['geom_relax_level'] = 99
    kr, p, ap = rig(cfg, obj_x=17.0, lvl=4)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is None and 'geom' in kr.last_overtake
    assert kr.last_avoid['need_geom'] == pytest.approx(TRANS + AHEAD + MARGIN, abs=0.05)


def test_relax_label_reflects_levels():
    kr = KrRules(CFG)
    kr.bo_level = 1
    assert kr._relax_label() is None
    kr.bo_level = ZONE_LVL
    assert 'zone_gate' in kr._relax_label()
    kr.bo_level = GEOM_LVL
    assert 'shift_ahead' in kr._relax_label()
    kr.bo_level = kr.BO_CREEP
    assert 'creep' in kr._relax_label()


# ── B-3: solid 두 바퀴 ────────────────────────────────────────────────────
class LgSolid(LgOne):
    """이웃은 있으나 실선뿐인 레인그래프 목."""

    def dashed_runs(self, key, side):
        return []


class Match:
    def __init__(self, lane, s):
        self.lane, self.s = lane, s


class LgSolidOccupied(LgSolid):
    """실선 + 목표 차로에 차가 있는 배치 (locate 가 옆 차로 객체를 매칭한다)."""

    def locate(self, x, y):
        return Match((1, 0, -2), x) if abs(y) > 1.5 else Match((1, 0, -1), x)


def test_dashed_corridor_shifts_in_first_pass():
    kr, p, ap = rig()
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is not None and kr.last_overtake == 'right'
    assert kr.last_avoid['pass'] == 1 and kr.last_avoid['solid_relaxed'] is False


def test_solid_rejects_first_pass_then_second_pass_shifts():
    """1바퀴 solid 기각 → 2바퀴에서 solid 게이트를 건너뛰고 시프트."""
    kr, p, ap = rig(lg=LgSolid())
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is not None and kr.last_overtake == 'right'
    assert kr.last_avoid['pass'] == 2 and kr.last_avoid['solid_relaxed'] is True
    assert kr.ot_pass_solid is True


def test_solid_is_forced_in_first_pass_even_at_high_level():
    """옛 lvl<2 조건 제거 — L2 이상이어도 1바퀴는 solid 를 본다 (2바퀴가 처리)."""
    kr, p, ap = rig(lg=LgSolid(), lvl=2)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_pass_solid is True                                # 1바퀴에서 기각됐다
    assert kr.last_avoid['pass'] == 2                              # 2바퀴에서 통과


def test_second_pass_disabled_keeps_old_behaviour():
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['solid_second_pass_enable'] = False
    kr, p, ap = rig(cfg, lg=LgSolid())
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is None and kr.last_overtake == 'right:solid@p1'
    assert kr.last_avoid['reject'] == 'right:solid' and kr.last_avoid['pass'] == 1


def test_second_pass_not_run_without_solid_reject():
    """1바퀴 기각이 solid 이외(geom)뿐이면 2바퀴는 돌지 않는다 (결과 동일, 비용 0)."""
    kr, p, ap = rig(obj_x=5.3)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.last_overtake == 'right:geom@p1' and kr.ot_pass_solid is False


def test_occupied_gate_survives_second_pass():
    """occupied(lvl<1)는 2바퀴에서도 그대로 — L1 이 되어야 풀린다."""
    kr, p, ap = rig(lg=LgSolidOccupied())
    ap._world._a.append(Box(9, 10.0, -3.5))                        # 목표 차로 위 (반경 30 안)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is None and kr.last_overtake == 'right:occupied@p2'
    kr, p, ap = rig(lg=LgSolidOccupied(), lvl=1)
    ap._world._a.append(Box(9, 10.0, -3.5))
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is not None and kr.last_avoid['pass'] == 2


def test_center_line_blocks_both_passes_any_level():
    class LgYellow(LgSolid):
        def mark_at(self, key, s, side):
            return ('solid', 'yellow', False)
    for lvl in (0, 2, 4):
        kr, p, ap = rig(lg=LgYellow(), lvl=lvl)
        kr._try_overtake(ap, p, ego_speed=0.0)
        assert kr.ot_span is None and kr.last_overtake == 'right:center_line@p1'


# ── B-4: standoff 25 ──────────────────────────────────────────────────────
def test_standoff_profile_uses_shift_latest_25():
    kr = KrRules(CFG)
    kr.wait_target_d = 50.0
    v = kr._standoff_profile(0.0)
    assert v == pytest.approx(math.sqrt(2.0 * A_STOP * (50.0 - STANDOFF)))
    kr.wait_target_d = STANDOFF                                    # 정지 위치에서 0
    assert kr._standoff_profile(0.0) == pytest.approx(0.0)


def test_need_m_in_avoid_diag_is_25():
    kr, p, ap = rig()
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.last_avoid['need_m'] == pytest.approx(STANDOFF)


# ── B-5: 신호 무관 정지 카운터 ─────────────────────────────────────────────
def test_obj_stop_ticks_count_through_pause():
    kr, p = make()
    ap = Ap(p, actors=[Box(1, 20.0, 0.0)])
    for _ in range(10):
        kr._update_obj_timers(ap, paused=True)
    assert kr.obj_ticks.get(1, 0) == 0                             # 신호 중 멈춘다
    assert kr.obj_stop_ticks[1] == 10                              # 이건 센다
    for _ in range(5):
        kr._update_obj_timers(ap, paused=False)
    assert kr.obj_ticks[1] == 5 and kr.obj_stop_ticks[1] == 15


def test_obj_stop_ticks_reset_when_moving_and_pop_when_lost():
    kr, p = make()
    b = Box(1, 20.0, 0.0)
    ap = Ap(p, actors=[b])
    for _ in range(8):
        kr._update_obj_timers(ap)
    b.speed = OT['blocker_speed_max'] + 1.0
    kr._update_obj_timers(ap)
    assert kr.obj_stop_ticks[1] == 0
    ap._world._a.remove(b)
    for _ in range(kr.obj_grace + 2):
        kr._update_obj_timers(ap)
    assert 1 not in kr.obj_stop_ticks and 1 not in kr.obj_ticks


def _two_object_rig(cfg, near_ticks):
    """A: 40 m 앞 정지 관찰 완료(정적). B: 20 m 앞, near_ticks 동안만 정지 관찰."""
    kr, p = make(cfg)
    a = Box(1, 40.0, 0.0, half_w=0.9)
    ap = Ap(p, actors=[a])
    for _ in range(STATIC_TICKS + 2):
        kr._update_obj_timers(ap)
    b = Box(2, 20.0, 0.0, half_w=0.9)
    ap._world._a.append(b)
    for _ in range(near_ticks):
        kr._update_obj_timers(ap)
    return kr, p, ap


def test_standoff_target_is_nearest_stopped_not_static():
    """정지 1.6 s 인 20 m 객체가 standoff 대상, PREEMPT/WAIT 판정은 40 m 정적 객체."""
    stop_ticks = int(round(OT['standoff_stop_s'] * HZ))
    kr, p, ap = _two_object_rig(CFG, stop_ticks + 2)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.wait_target_d == pytest.approx(20.0)
    assert kr.standoff_id == 2
    assert kr.last_avoid['blocker'] == 1                           # 판정 대상은 A
    assert kr.last_avoid['state'] in ('PREEMPT', 'WAIT', 'WAIT_EXPIRED')


def test_standoff_target_ignores_object_stopped_too_briefly():
    stop_ticks = int(round(OT['standoff_stop_s'] * HZ))
    kr, p, ap = _two_object_rig(CFG, stop_ticks - 2)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.wait_target_d == pytest.approx(40.0)                 # B 는 아직 아니다


def test_standoff_kill_switch_uses_static_corridor():
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['standoff_stop_s'] = 0.0
    stop_ticks = int(round(OT['standoff_stop_s'] * HZ))
    kr, p, ap = _two_object_rig(cfg, stop_ticks + 2)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.wait_target_d == pytest.approx(40.0)                 # 이전 동작: corridor[0]
    assert kr.standoff_id is None


def test_standoff_target_without_static_corridor():
    """정적 객체가 없어도(관찰 3 s 미만) 1.5 s 정지 객체로 standoff 감속은 건다."""
    kr, p = make()
    ap = Ap(p, actors=[Box(2, 30.0, 0.0, half_w=0.9)])
    for _ in range(int(round(OT['standoff_stop_s'] * HZ)) + 1):
        kr._update_obj_timers(ap)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.wait_target_d == pytest.approx(30.0)
    assert kr._standoff_profile(0.0) == pytest.approx(math.sqrt(2.0 * A_STOP * (30.0 - STANDOFF)))
