"""[1] 종료 구간 게이트를 **객체 위치**로 판정한다 (2026-09-09).

왜 축을 바꿨나 — 자차 기준은 판정 시점이 늦다. 회피 시프트는 자차가 종료
구간에 들어오기 **훨씬 전에** 결정된다. 실측 20260909_093415/실경로_02:

    종료선 finish_s = 749.2
    콘 id 2·3 의 **객체 route_s = 749.2** (종료선과 같은 자리)
    자차 rs 690.9 (s_rel 58.3) 에서 시프트 생성 → rs 723~751 을 0.9 m/s 로 25 초
    왕복, 종료 차로 (418,2,-1) 이 아니라 -2 로 끝남 (횡오프셋 −3.07 m)

자차 기준 게이트를 30 → 58.3 초과로 넓혀도 그 순간을 못 덮는다. 객체 기준이면
같은 콘이 감지창(detect_max_m 80 m)에 들어오는 **첫 틱**부터 빠진다 — replay
확인: 자차 rs 669.5 부터 `finish_gate.dropped=2`, 시프트 생성 **0회**.

계약:
  · 판정 기준은 **객체의 route_s** 다. 자차 위치는 안 본다.
  · 임계는 `_finish_gate_s()` 한 곳이다 (회랑·PDM 접합이 같은 값을 쓴다).
  · 객체 투영은 전·후 범퍼까지 세 점을 보고 **가장 가까운** s 를 쓴다 (보수적).
  · `finish_gate_by_object_enable=false` 면 이전(자차 기준) 판정 그대로.
  · 여전히 **정지 객체만** 뺀다 — 속도 임계는 blocker_speed_max 단일 출처.
"""
import copy
import pathlib
import sys

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402
from test_avoid import Ap, Box, GeomPlanner, HZ, LgOne             # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
GATE_M = CFG['speed']['finish_gate_m']
V_STATIC = CFG['overtake']['blocker_speed_max']
FINISH_S = 100.0


def obj_cfg(**over):
    """객체 기준 축. 기본값을 읽지 않고 사본에서 명시한다."""
    c = copy.deepcopy(CFG)
    c['speed']['finish_gate_ignore_enable'] = True
    c['speed']['finish_gate_by_object_enable'] = True
    c['speed'].update(over)
    return c


def ego_cfg(**over):
    """이전(자차 기준) 축."""
    c = obj_cfg(**over)
    c['speed']['finish_gate_by_object_enable'] = False
    return c


def rig(cfg, obj_s=FINISH_S, route_s=0.0, obj_speed=0.0):
    """객체를 **경로 위 절대 s = obj_s** 에 놓고 자차를 route_s 에 둔다.

    (기존 test_finish_gate_ignore 의 rig 는 객체를 자차에 붙여 옮기므로 객체
     기준 축을 볼 수 없다 — 그래서 여기서 절대 위치로 다시 짠다.)
    """
    p = GeomPlanner(d_tl=float('inf'))
    p.lg = LgOne()
    p.route_index = int(route_s * p.points_per_meter)
    kr = KrRules(cfg)
    kr._sl_all = []
    kr.finish_s = FINISH_S
    ap = Ap(p, actors=[Box(2, obj_s, 0.0, obj_speed, half_w=0.9)])
    ap._kr_ego_lane = (1, 0, -1)
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    kr._ap = ap
    for _ in range(int(cfg['overtake']['obj_static_s'] * HZ) + 5):
        kr._update_obj_timers(ap)
    return kr, p, ap


def drops(kr, p, ap):
    return [a for a in ap._world.get_actors() if kr.finish_gate_drop(p, a)]


# ── 축 자체 ───────────────────────────────────────────────────────────────
def test_gate_threshold_is_one_place():
    kr, _, _ = rig(obj_cfg())
    assert kr._finish_gate_s() == FINISH_S - GATE_M


def test_gate_is_none_when_switch_off():
    c = obj_cfg()
    c['speed']['finish_gate_ignore_enable'] = False
    kr, _, _ = rig(c)
    assert kr._finish_gate_s() is None


def test_gate_is_none_without_finish_s():
    kr, p, ap = rig(obj_cfg())
    kr.finish_s = None
    assert kr._finish_gate_s() is None
    assert drops(kr, p, ap) == []


# ── 객체 기준 판정 ────────────────────────────────────────────────────────
def test_cone_on_the_finish_line_is_dropped_from_far_away():
    """실측 재현 — 콘이 종료선 위에 있으면 자차가 한참 뒤여도 빠진다."""
    kr, p, ap = rig(obj_cfg(), obj_s=FINISH_S, route_s=FINISH_S - 60.0)
    assert len(drops(kr, p, ap)) == 1


def test_same_case_is_not_dropped_on_the_ego_axis():
    """같은 배치에서 자차 기준 축은 아직 안 뺀다 — 이것이 실측 버그다."""
    kr, p, ap = rig(ego_cfg(), obj_s=FINISH_S, route_s=FINISH_S - 60.0)
    assert drops(kr, p, ap) == []


def test_object_before_the_gate_is_kept_even_when_ego_is_inside():
    """게이트 **앞**의 물체는 자차가 종료 구간에 있어도 계속 장애물이다.

    자차 기준 축이었다면 반대로 빠졌다 — 그게 실제 장애물을 놓치는 쪽 오류다.
    """
    kr, p, ap = rig(obj_cfg(), obj_s=FINISH_S - GATE_M - 5.0,
                    route_s=FINISH_S - GATE_M - 12.0)
    assert drops(kr, p, ap) == []


def test_boundary_is_inclusive():
    kr, p, ap = rig(obj_cfg(), obj_s=FINISH_S - GATE_M, route_s=FINISH_S - GATE_M - 20.0)
    assert len(drops(kr, p, ap)) == 1
    kr2, p2, ap2 = rig(obj_cfg(), obj_s=FINISH_S - GATE_M - 0.5,
                       route_s=FINISH_S - GATE_M - 20.0)
    assert drops(kr2, p2, ap2) == []


def test_moving_object_is_kept():
    kr, p, ap = rig(obj_cfg(), obj_s=FINISH_S, route_s=FINISH_S - 20.0,
                    obj_speed=V_STATIC + 1.0)
    assert drops(kr, p, ap) == []


def test_speed_threshold_stays_the_single_source():
    kr, p, ap = rig(obj_cfg(), obj_s=FINISH_S, route_s=FINISH_S - 20.0,
                    obj_speed=V_STATIC - 0.01)
    assert len(drops(kr, p, ap)) == 1
    kr2, p2, ap2 = rig(obj_cfg(), obj_s=FINISH_S, route_s=FINISH_S - 20.0,
                       obj_speed=V_STATIC + 0.01)
    assert drops(kr2, p2, ap2) == []


# ── 투영 ─────────────────────────────────────────────────────────────────
def test_actor_route_s_matches_the_absolute_position():
    kr, p, ap = rig(obj_cfg(), obj_s=FINISH_S - 20.0, route_s=FINISH_S - 60.0)
    a = ap._world.get_actors()[0]
    assert abs(kr._actor_route_s(p, a) - (FINISH_S - 20.0)) < 0.5


def test_projection_uses_the_nearest_of_three_points():
    """세 점 중 **가장 가까운** s 를 쓴다 — 게이트에 걸친 물체는 안 뺀다."""
    kr, p, ap = rig(obj_cfg(), obj_s=FINISH_S - GATE_M, route_s=FINISH_S - 40.0)
    a = ap._world.get_actors()[0]
    a.length = 8.0                                   # 앞뒤로 4 m 씩
    s = kr._actor_route_s(p, a)
    assert s < FINISH_S - GATE_M                     # 뒤쪽 범퍼가 기준이다
    assert drops(kr, p, ap) == []                    # 그래서 아직 안 뺀다


# ── 회랑과 같은 답 ───────────────────────────────────────────────────────
def test_corridor_and_seam_agree():
    """회랑에서 빠지는 집합과 접합부가 빼는 집합이 같아야 한다 (조건 두 벌 금지)."""
    for rs in (FINISH_S - 60.0, FINISH_S - GATE_M, FINISH_S - 2.0):
        for os_ in (FINISH_S - GATE_M - 5.0, FINISH_S - GATE_M, FINISH_S):
            kr, p, ap = rig(obj_cfg(), obj_s=os_, route_s=rs)
            in_corridor = kr._corridor_blockers(ap, p, static_ok=kr._stop_ok)
            visible = [a for a in ap._world.get_actors()
                       if 0.5 < (kr._actor_route_s(p, a) or -1) - rs <= kr.detect_max_m]
            if not visible:
                continue
            assert bool(drops(kr, p, ap)) == (in_corridor == [])
