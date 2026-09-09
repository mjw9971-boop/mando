"""[1] 시프트가 **대상을 잃으면** 즉시 원복 (overtake.span_lost_restore_enable).

지금까지 원복 조건은 `route_index > span[1]` 하나였다. span 끝이 경로 끝
근처면 그 지점을 밟을 일이 없어 시프트가 영원히 남는다 — 실측
20260909_093415/실경로_02: span 끝 = rs 776.0 인데 종료선이 749.2 라,
rs 722.6 부터 blocker 가 None 인데도 시프트를 쥔 채 종료 차로 (418,2,-1)
대신 -2 로 끝났다 (횡오프셋 −3.07 m, 차로유지 −3, 지시등 3건).

콘과 무관한 일반 버그다. 다만 조건은 **좁아야** 한다:
  · "회랑이 비었다" 로 판정하면 안 된다 — 시프트 중에는 **밀린 경로 기준**
    회랑에 피한 물체가 안 들어오므로 항상 참이고, 모든 시프트가 즉시 원복돼
    왕복이 된다. 이 파일이 그 반례를 고정한다.
  · 판정은 **시프트를 만든 그 id 들**이 전부 사라졌는가 하나다
    (월드에서 소실 / 종료 게이트로 빠짐 / 스스로 움직여 감).
  · 스위치 off = 이전 동작.
"""
import copy
import pathlib
import sys

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                        # noqa: E402
from test_avoid import HZ, ActorList, Ap, Box, LgOne, try_overtake  # noqa: E402
from test_span_extend import Car, ExtPlanner                        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
V_STATIC = CFG['overtake']['blocker_speed_max']


def cfg_on(**over):
    c = copy.deepcopy(CFG)
    c['overtake']['span_lost_restore_enable'] = True
    c['overtake'].update(over)
    return c


def cfg_off():
    c = copy.deepcopy(CFG)
    c['overtake']['span_lost_restore_enable'] = False
    return c


def rig(cfg, objs):
    p = ExtPlanner(d_tl=float('inf'))
    p.lg = LgOne()
    kr = KrRules(cfg)
    kr._sl_all = []
    ap = Ap(p, actors=list(objs))
    ap._kr_ego_lane = (1, 0, -1)
    for _ in range(int(cfg['overtake']['wait_before_shift_s'] * HZ) + 5):
        kr._update_obj_timers(ap)
    return kr, p, ap


def make_shift(kr, p, ap):
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is not None
    return kr.ot_span


def test_ids_are_recorded_on_creation():
    kr, p, ap = rig(cfg_on(), [Car(7, 22.4, 0.0)])
    make_shift(kr, p, ap)
    assert kr.ot_ids == [7]


def test_active_shift_is_not_restored_while_the_target_is_there():
    """가장 중요한 반례 — 시프트 중 회랑은 비어 보이지만 원복하면 안 된다."""
    kr, p, ap = rig(cfg_on(), [Car(7, 22.4, 0.0)])
    span = make_shift(kr, p, ap)
    assert kr._corridor_blockers(ap, p) == []      # 밀린 경로 기준 회랑은 비었다
    assert kr._span_targets_lost(ap, p) is False
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span == span                      # 그대로 쥐고 있다


def test_restores_when_the_target_disappears():
    kr, p, ap = rig(cfg_on(), [Car(7, 22.4, 0.0)])
    make_shift(kr, p, ap)
    ap._world._a = ActorList(a for a in ap._world.get_actors() if a.id != 7)
    assert kr._span_targets_lost(ap, p) is True
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span is None
    assert kr.ot_ids == []
    assert kr.last_avoid.get('why') == 'targets_lost'


def test_restores_when_the_target_starts_moving():
    kr, p, ap = rig(cfg_on(), [Car(7, 22.4, 0.0)])
    make_shift(kr, p, ap)
    ap._world.get_actors()[0].speed = V_STATIC + 1.0
    assert kr._span_targets_lost(ap, p) is True


def test_one_remaining_target_blocks_the_restore():
    kr, p, ap = rig(cfg_on(), [Car(7, 22.4, 0.0), Car(8, 30.0, 0.0)])
    make_shift(kr, p, ap)
    assert len(kr.ot_ids) >= 1
    keep = kr.ot_ids[0]
    ap._world._a = ActorList(a for a in ap._world.get_actors() if a.id == keep)
    assert kr._span_targets_lost(ap, p) is False


def test_switch_off_keeps_the_old_behaviour():
    kr, p, ap = rig(cfg_off(), [Car(7, 22.4, 0.0)])
    span = make_shift(kr, p, ap)
    ap._world._a = ActorList()
    try_overtake(kr, ap, p, ego_speed=0.0)
    assert kr.ot_span == span                      # 이전 동작 — 안 푼다


def test_unknown_ids_fall_back_to_the_old_rule():
    """id 를 모르면(상태 유실) 거짓 — `route_index > span[1]` 에 맡긴다."""
    kr, p, ap = rig(cfg_on(), [Car(7, 22.4, 0.0)])
    make_shift(kr, p, ap)
    kr.ot_ids = []
    assert kr._span_targets_lost(ap, p) is False


def test_restore_resets_the_id_list():
    kr, p, ap = rig(cfg_on(), [Car(7, 22.4, 0.0)])
    make_shift(kr, p, ap)
    kr._restore_span(p)
    assert kr.ot_ids == []


# ── [3] courseRespawn — on_reset 이 회피·큐·BREAKOUT 상태를 전부 버린다 ──
def test_on_reset_restores_and_drops_the_span():
    """`ot_span` 은 **경로점 인덱스**다. `planner.reset_index()` 로 자차가 경로
    위 다른 곳으로 옮겨간 뒤에도 남으면 `_restore_span` 이 엉뚱한 구간의
    `original_route_points` 를 되돌린다. 그래서 먼저 원복하고 비운다.
    """
    import numpy as np
    kr, p, ap = rig(cfg_on(), [Car(7, 22.4, 0.0)])
    span = make_shift(kr, p, ap)
    a, b = span
    assert not np.allclose(p.route_points[a:b], p.original_route_points[a:b])
    kr.on_reset(p)
    assert kr.ot_span is None and kr.ot_ids == []
    # 경로가 원상 복구됐다 — 남겨 두면 리스폰 뒤 발밑 경로가 옆으로 밀린 채다
    assert np.allclose(p.route_points[a:b], p.original_route_points[a:b])


def test_on_reset_clears_every_avoid_state():
    """19개 상태 전부 — 하나라도 남으면 리스폰 뒤 옛 문맥으로 판단한다."""
    kr, p, ap = rig(cfg_on(), [Car(7, 22.4, 0.0)])
    make_shift(kr, p, ap)
    kr.ot_blocked_ticks = 5
    kr.ot_reject_ticks = 5
    kr.preempt_latch_id = 7
    kr.wait_target_d = 12.0
    kr.standoff_id = 7
    kr.standoff_half_len = 2.0
    kr.q_ticks = 30
    kr.q_info = {'x': 1}
    kr.bo_state = 'BREAKOUT'
    kr.bo_level = 3
    kr.bo_stop_ticks = 40
    kr.fg_dropped = 2
    kr.last_lane_plan = {'pick': 'x'}
    kr.lm_hop_n = 2
    kr.on_reset(p)
    assert (kr.ot_span, kr.ot_side, kr.ot_ids, kr.lm_hop_n) == (None, None, [], 0)
    assert (kr.ot_blocked_ticks, kr.ot_reject_ticks, kr.preempt_latch_id) == (0, 0, None)
    assert (kr.wait_target_d, kr.standoff_id, kr.standoff_half_len) == (None, None, None)
    assert (kr.q_ticks, kr.q_info, kr._tick_queue, kr._tick_corridor) == (0, None, False, [])
    assert (kr.bo_state, kr.bo_level, kr.bo_stop_ticks) == (None, 0, 0)
    assert (kr.fg_dropped, kr.last_lane_plan, kr.last_avoid) == (0, None, None)
    assert not kr.obj_ticks


def test_on_reset_works_without_a_planner():
    """planner 는 선택 인자다 — 안 줘도 나머지는 지운다 (목 플래너 테스트용)."""
    kr, p, ap = rig(cfg_on(), [Car(7, 22.4, 0.0)])
    make_shift(kr, p, ap)
    kr.on_reset()
    assert kr.ot_span is None and kr.q_ticks == 0


def test_run_agent_passes_the_planner():
    """접합부가 planner 를 넘기지 않으면 span 원복이 안 된다."""
    src = (ROOT / 'run_agent.py').read_text(encoding='utf-8')
    assert 'self.kr.on_reset(self.planner)' in src
