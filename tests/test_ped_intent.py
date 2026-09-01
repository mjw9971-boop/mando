"""
보행자 의도 감지 (P4) + 비상 제동 우회.

실측 근거 (2026-09-01, logs/batch/20260901_174724/실전주행_교통류_01_좌회전24):
  · id7 은 t=424.5 에 79 m 밖에서 잡히고 **v=0.00 으로 5.5 s 서 있다가**
    t=430.02 에 걸어나왔다 (횡 6.35 m, 자차 12.31 m/s, 종 22.91 m).
  · PDM walker 후보는 t=430.52 에야 생겼다 — forecast_walkers 가 등속 2 s
    예측이고 속도 하한이 min_walker_speed(0.5) 라, 예측 도달거리가
    v_ped·2 + pedestrian_minimum_extent(1.5) 다. 서 있으면 2.5 m 밖에 못 뻗어
    횡 6.35 m 를 못 덮는다. v_ped 가 1.70 이 되어서야 4.9 m 로 닿았다.
  · 그 0.5 s 사이 필요 감속이 3.31 → 4.29 m/s² 로 올라 a_dec_max(4.0)를 넘겼고,
    거기에 jerk 램프가 −4.0 도달까지 1.0 s 를 더 먹어 실제 평균 감속은
    1.95 m/s² (여력의 49%) 였다 → 접촉 (min_gap 0.0, 30.5 kph, 항목10 중대).

여기서 지키는 불변:
  · 의도 래치는 **정지 관찰을 마친** 보행자가 **경로 쪽으로** 움직일 때만 걸린다.
    멀어짐·평행·정지 관찰 없음은 전부 미발동이다.
  · 비상 우회는 **보행자 후보가 최종 목표를 구속하는 틱 한정**이다.
    선행차·신호·종점 후보가 이긴 틱에서는 절대 발동하지 않는다.
  · 두 기능 모두 params 로 끌 수 있다 (ped_intent_v / ped_emergency_ratio = 0).
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                   # noqa: E402
from test_avoid import Ap, Box, Planner                        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
HZ = CFG['comm']['send_hz']
OT = CFG['overtake']
STATIC_TICKS = int(round(OT['obj_static_s'] * HZ))
INTENT_V = CFG['speed']['ped_intent_v']
S0_PED = 4.0                                    # GlobalConfig.idm_pedestrian_minimum_distance
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']


class Walker(Box):
    """9910 보행자 — world.classify 가 붙이는 type_id 를 그대로 쓴다."""

    def __init__(self, oid, x, y, speed=0.0):
        super().__init__(oid, x, y, speed, half_w=0.35)
        self.type_id = 'walker.vtd.pedestrian'


def make(cfg=CFG):
    return KrRules(cfg), Planner()


def observe_static(kr, ap, n=STATIC_TICKS + 2, p=None, v=0.0):
    """정지 관찰을 쌓는다 (타이머 + 직전 lat 기록)."""
    for _ in range(n):
        kr._update_obj_timers(ap)
        kr._ped_intent(p, ap, v)


def walk(kr, ap, p, w, dy, ticks=1, v_ego=12.0):
    """보행자를 dy [m/틱] 만큼 옮기며 틱을 돌린다. 마지막 반환값을 준다."""
    out = None
    for _ in range(ticks):
        w._y += dy
        w.speed = abs(dy) * HZ
        kr._update_obj_timers(ap)
        out = kr._ped_intent(p, ap, v_ego)
    return out


# ── 의도 래치 ────────────────────────────────────────────────────────────
def test_static_pedestrian_stepping_out_creates_candidate_immediately():
    """실측 재현: 서 있던 보행자가 **걸어나오는 첫 틱**에 후보가 생긴다."""
    kr, p = make()
    w = Walker(7, 22.91, -6.35)
    ap = Ap(p, actors=[w])
    observe_static(kr, ap, p=p)
    assert kr._ped_intent(p, ap, 12.31) is None            # 서 있는 동안은 미발동
    # v_ped 0.50 m/s 로 경로 쪽 이동 시작 (실측 첫 틱과 같은 값)
    got = walk(kr, ap, p, w, dy=+0.50 / HZ, ticks=1, v_ego=12.31)
    assert got is not None
    v_allow, a_req, wid = got
    assert wid == 7
    d_eff = 22.91 - FRONT - S0_PED
    assert v_allow == pytest.approx((2.0 * CFG['speed']['stop_profile_a'] * d_eff) ** 0.5,
                                    abs=0.05)
    assert a_req == pytest.approx(12.31 ** 2 / (2.0 * d_eff), abs=0.05)


def test_latched_candidate_survives_speed_dip():
    """한 번 래치되면 보행자가 잠시 멈춰도 후보가 유지된다 (다시 걸어나온다)."""
    kr, p = make()
    w = Walker(7, 20.0, -5.0)
    ap = Ap(p, actors=[w])
    observe_static(kr, ap, p=p)
    assert walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1) is not None
    w.speed = 0.0
    assert kr._ped_intent(p, ap, 12.0) is not None


# ── 부정 테스트 ──────────────────────────────────────────────────────────
def test_far_standing_pedestrian_never_latches():
    """원거리 정지 보행자 — 움직이지 않으면 몇 초를 봐도 후보가 없다."""
    kr, p = make()
    w = Walker(7, 60.0, -8.0)
    ap = Ap(p, actors=[w])
    observe_static(kr, ap, n=STATIC_TICKS * 3, p=p)
    assert kr._ped_intent(p, ap, 12.0) is None


def test_parallel_walking_pedestrian_never_latches():
    """평행 보행 — 횡거리가 안 줄면 자기 속도가 커도 미발동."""
    kr, p = make()
    w = Walker(7, 25.0, -5.0)
    ap = Ap(p, actors=[w])
    observe_static(kr, ap, p=p)
    for _ in range(40):
        w._x += 2.5 / HZ                                   # 경로와 나란히
        w.speed = 2.5
        kr._update_obj_timers(ap)
        assert kr._ped_intent(p, ap, 12.0) is None


def test_receding_pedestrian_never_latches():
    """경로에서 멀어지는 보행자 — 부호가 반대라 미발동."""
    kr, p = make()
    w = Walker(7, 25.0, -5.0)
    ap = Ap(p, actors=[w])
    observe_static(kr, ap, p=p)
    for _ in range(40):
        assert walk(kr, ap, p, w, dy=-2.5 / HZ, ticks=1) is None


def test_never_observed_static_does_not_latch():
    """정지 관찰이 없으면(이미 걷고 있던 보행자) 이 경로로는 안 잡는다 —
    그런 보행자는 PDM 예측이 정상적으로 덮는다 (과검출 방지)."""
    kr, p = make()
    w = Walker(7, 25.0, -5.0, speed=2.5)
    ap = Ap(p, actors=[w])
    for _ in range(40):
        assert walk(kr, ap, p, w, dy=+2.5 / HZ, ticks=1) is None


def test_pedestrian_behind_is_ignored():
    """뒤에 있는 보행자는 대상이 아니다."""
    kr, p = make()
    w = Walker(7, -5.0, -5.0)
    ap = Ap(p, actors=[w])
    observe_static(kr, ap, p=p)
    assert walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1) is None


def test_switch_off_disables_intent():
    """ped_intent_v = 0 이면 완전 비활성 = 이전 동작."""
    cfg = copy.deepcopy(CFG)
    cfg['speed']['ped_intent_v'] = 0.0
    kr, p = make(cfg)
    w = Walker(7, 22.0, -6.0)
    ap = Ap(p, actors=[w])
    observe_static(kr, ap, p=p)
    assert walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1) is None


def test_vehicle_class_actor_is_not_a_walker():
    """type_id 가 walker 가 아니면(차량·장애물) 이 경로로 안 들어온다."""
    kr, p = make()
    v = Box(9, 22.0, -6.0)                                 # vehicle.vtd.object
    ap = Ap(p, actors=[v])
    observe_static(kr, ap, p=p)
    assert walk(kr, ap, p, v, dy=+0.5 / HZ, ticks=1) is None


# ── 비상 제동 우회 ───────────────────────────────────────────────────────
from vtd_adapter.carla_types import VehicleControl            # noqa: E402
from vtd_adapter.control import VtdLongitudinalController     # noqa: E402
from autopilot import AutoPilot                                # noqa: E402
from config import GlobalConfig                                # noqa: E402
from test_avoid import World                                   # noqa: E402

A_DEC = abs(CFG['control']['a_dec_max'])
A_EMG = CFG['speed']['a_emergency']
RATIO = CFG['speed']['ped_emergency_ratio']


def make_ap(planner, actors=(), cfg=CFG):
    """apply() 를 태울 수 있는 최소 AutoPilot (world 포함)."""
    ego = Box(0, 0.0, 0.0, 0.0, 0.94)
    a = AutoPilot()
    a.setup(world=World(list(actors) + [ego]), world_map=None,
            waypoint_planner=planner,
            longitudinal_controller=VtdLongitudinalController(cfg),
            ego_vehicle=ego, config=GlobalConfig())
    a.kr_rules = KrRules(cfg)
    return a


def run_apply(ap, p, v, target=12.5):
    ap._vehicle.speed = float(v)          # apply() 는 자차 속도를 목에서 읽는다
    ap._longitudinal_controller.get_throttle_and_brake(False, target, v)
    return ap.kr_rules.apply(VehicleControl(), target, ap)


def test_emergency_fires_and_skips_jerk_ramp():
    """필요 감속이 |a_dec_max|·ratio 를 넘으면 램프 없이 a_emergency 를 낸다."""
    p = Planner()
    w = Walker(7, 12.0, -3.0)                              # 가깝다 → a_req 가 크다
    ap = make_ap(p, [w])
    kr = ap.kr_rules
    observe_static(kr, ap, p=p)
    walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1)              # 래치
    ctrl, _ts = run_apply(ap, p, 12.0)
    assert kr.last_ped['a_req'] > RATIO * A_DEC
    assert kr.ped_emergency is True
    assert ctrl.accel == pytest.approx(A_EMG)              # 한 틱에 도달 (램프 없음)


def test_emergency_not_fired_when_required_decel_is_mild():
    """멀리서 걸어나오면 후보는 생기되 비상은 아니다 — 정상 램프."""
    p = Planner()
    w = Walker(7, 70.0, -3.0)
    ap = make_ap(p, [w])
    kr = ap.kr_rules
    observe_static(kr, ap, p=p)
    walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1, v_ego=6.0)
    ctrl, _ts = run_apply(ap, p, 6.0)
    assert kr.last_ped is not None                          # 후보는 있다
    assert kr.last_ped['a_req'] <= RATIO * A_DEC
    assert kr.ped_emergency is False
    assert ctrl.accel > A_EMG


def test_no_emergency_for_leading_vehicle():
    """선행차(차량 클래스)에서는 절대 발동하지 않는다."""
    p = Planner()
    v = Box(9, 6.0, 0.0)                                    # 코앞의 정지 차량
    ap = make_ap(p, [v])
    kr = ap.kr_rules
    for _ in range(STATIC_TICKS + 2):
        kr._update_obj_timers(ap)
    run_apply(ap, p, 12.0)
    assert kr.last_ped is None
    assert kr.ped_emergency is False


def test_no_emergency_for_red_light_stop():
    """신호 정지(적색 정지선 코앞)에서는 절대 발동하지 않는다."""
    from vtd_adapter.carla_types import TrafficLightState
    p = Planner(d_tl=5.0)
    tl = type('TL', (), {'state': TrafficLightState.Red})()
    p.next_traffic_lights = [tl] * len(p.route_s)
    ap = make_ap(p)
    kr = ap.kr_rules
    _ctrl, ts = run_apply(ap, p, 8.0)
    assert ts < 8.0                                          # 신호 후보가 이겼다
    assert kr.last_ped is None
    assert kr.ped_emergency is False


def test_emergency_switch_off():
    """ped_emergency_ratio = 0 이면 후보만 남고 비상 우회는 없다."""
    cfg = copy.deepcopy(CFG)
    cfg['speed']['ped_emergency_ratio'] = 0.0
    p = Planner()
    w = Walker(7, 12.0, -3.0)
    ap = make_ap(p, [w], cfg=cfg)
    kr = ap.kr_rules
    observe_static(kr, ap, p=p)
    walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1)
    ctrl, _ts = run_apply(ap, p, 12.0)
    assert kr.last_ped is not None
    assert kr.ped_emergency is False
    assert ctrl.accel > A_EMG
