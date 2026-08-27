"""
kr_rules.route_end — 경로 종점 정지 (phase4 작업1).

계약:
  · 남은 거리가 줄수록 후보 목표속도 단조 감소 (IDM — PDM 원문 재사용)
  · 종점(계획 정지점) 도달 시 0
  · 정지 래치: 한 번 서면 재출발하지 않는다
  · d_end 가 다시 커지면(courseRespawn) 래치 해제
  · 계획 정지점이 batch 완주 임계 안쪽 (params.yaml 한 곳에서 연동)
"""
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.control import VtdLongitudinalController

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from autopilot import AutoPilot            # noqa: E402
from config import GlobalConfig            # noqa: E402
from kr_rules import KrRules               # noqa: E402
from vtd_adapter.carla_types import VehicleControl  # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
TOTAL = 821.1
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
GAP = CFG['speed']['stop_gap_route_end_m']


class FakePlanner:
    """kr_rules 가 쓰는 표면만: route['total_length'], route_s[route_index]."""

    def __init__(self, total=TOTAL):
        self.route = {'total_length': total}
        self.route_s = np.arange(0.0, total + 60.0, 0.1)
        self.route_index = 0

    def set_route_s(self, rs):
        self.route_index = int(round(rs / 0.1))


class FakeEgo:
    def __init__(self):
        self.speed = 0.0

    def get_velocity(self):
        ego = self

        class V:
            def length(self):
                return ego.speed
        return V()


@pytest.fixture()
def ap():
    a = AutoPilot()
    a.setup(world=None, world_map=None, waypoint_planner=FakePlanner(),
            longitudinal_controller=VtdLongitudinalController(CFG),
            ego_vehicle=FakeEgo(), config=GlobalConfig())
    a.kr_rules = KrRules(CFG)
    return a


def _apply(ap, d_end, v, target=12.5):
    ap._waypoint_planner.set_route_s(TOTAL - d_end)
    ap._vehicle.speed = v
    control = VehicleControl(steer=0.0, accel=1.0)
    return ap.kr_rules.apply(control, target, ap)


def _closed_loop(ap, d0, v0, target=12.5, dt=0.05, max_ticks=4000):
    """실제 종방향 컨트롤러로 도는 폐루프 — 실행 경로 그대로: 매 틱 본류가
    get_throttle_and_brake(원 목표)를 부르고, kr_rules.apply 가 되감고 재계산
    (rewind_last). v 는 control.accel 적분."""
    d, v = float(d0), float(v0)
    vs = []
    for _ in range(max_ticks):
        ap._vehicle.speed = v
        # 본류 역할 (autopilot._get_control 의 원 목표 호출)
        base_accel, _b = ap._longitudinal_controller.get_throttle_and_brake(
            False, target, v)
        control, _ts = _apply(ap, d, v, target)
        if ap.kr_rules.last_candidate is None or ap.kr_rules.last_candidate >= target:
            control.accel = base_accel             # kr 미개입 틱은 본류 값
        v = max(0.0, v + control.accel * dt)
        d -= v * dt
        vs.append(v)
        if ap.kr_rules.latched and v < 0.01:
            break
    return d, vs


def test_closed_loop_decelerates_monotonically_and_stops(ap):
    """제동 개시 후 속도가 단조 감소해 0 — 진동/재가속 없다."""
    _d, vs = _closed_loop(ap, d0=60.0, v0=8.0)
    assert vs[-1] < 0.01 and ap.kr_rules.latched
    peak_i = vs.index(max(vs))                 # 멀 때는 desired 로 가속해도 된다
    braking = vs[peak_i:]
    assert all(b <= a + 1e-6 for a, b in zip(braking, braking[1:])), \
        '제동 구간에서 속도가 되튀었다'


def test_stops_at_planned_point_inside_done_threshold(ap):
    """정지 위치 본질 스펙 둘: (a) 종점(뒷축 기준)을 넘지 않는다 — 경로 밖
    이탈 방지, (b) batch 완주 임계(8.8) 안쪽 — 정지 후 완주 판정이 난다.
    계획 정지점(7.8) 대비 지나침은 컨트롤러 동역학상 수 m 허용 (시뮬 2~4 m)."""
    d_stop, _vs = _closed_loop(ap, d0=60.0, v0=8.0)
    assert d_stop > 0.0, f'종점을 {-d_stop:.2f} m 넘어 정지'
    from summarize_run import end_margin_m
    assert d_stop < end_margin_m(CFG), \
        f'정지 d_end {d_stop:.2f} 가 완주 임계 {end_margin_m(CFG):.2f} 밖 — 오판정'


def test_candidate_monotonic_in_distance_at_fixed_speed(ap):
    """저속 고정 스냅샷에서는 남은 거리가 줄수록 후보 비증가."""
    prev = None
    for d in range(60, 4, -4):
        ap.kr_rules.latched = False
        _c, ts = _apply(ap, float(d), v=2.0)
        if prev is not None:
            assert ts <= prev + 1e-6, f'd_end={d}: {prev:.2f} → {ts:.2f} 로 증가'
        prev = ts


def test_far_away_keeps_target(ap):
    """active_m 밖에서는 개입하지 않는다."""
    _c, ts = _apply(ap, CFG['route_end']['active_m'] + 50.0, v=12.0)
    assert ts == pytest.approx(12.5)
    assert ap.kr_rules.last_candidate is None


def test_latch_holds_stop(ap):
    """종점 근처 저속 → 래치. 이후 어떤 목표가 와도 0 유지 + a_hold."""
    _apply(ap, 8.0, v=0.3)
    assert ap.kr_rules.latched
    for _ in range(5):
        control, ts = _apply(ap, 8.0, v=0.05, target=12.5)
        assert ts == 0.0
    assert control.accel == pytest.approx(CFG['speed']['a_hold'])
    assert control.brake == 1.0


def test_latch_released_when_far_again(ap):
    """courseRespawn 으로 종점에서 멀어지면 래치 해제 — 고착 방지."""
    _apply(ap, 8.0, v=0.3)
    assert ap.kr_rules.latched
    _c, ts = _apply(ap, CFG['route_end']['unlatch_m'] + 20.0, v=0.3)
    assert not ap.kr_rules.latched
    assert ts > 0.3                            # 0 고정이 풀려 재가속 방향 (IDM 후보)


def test_no_latch_while_fast_near_end(ap):
    """종점 근처라도 아직 달리는 중이면 래치하지 않는다 (감속은 IDM 후보가)."""
    _c, _ts = _apply(ap, 10.0, v=6.0)
    assert not ap.kr_rules.latched
    # 그 지점부터 폐루프를 이어 가면 정지에 도달한다
    d_stop, _vs = _closed_loop(ap, d0=10.0, v0=6.0)
    assert ap.kr_rules.latched and d_stop < 10.0


def test_planned_stop_inside_batch_done_threshold():
    """작업3: 계획 정지점 route_s(= total − stop_gap − 앞범퍼)가 batch 완주
    임계(total − end_margin) **안쪽**이어야 정지 후 완주 판정이 난다.
    둘 다 params.yaml 의 같은 키에서 유도 — end_slack_m 이 그 여유다."""
    from summarize_run import end_margin_m
    margin = end_margin_m(CFG)
    planned_stop_short = GAP + FRONT           # 종점에서 뒷축까지
    assert margin - planned_stop_short == pytest.approx(CFG['batch']['end_slack_m'])
    assert margin > planned_stop_short         # end_slack > 0 ⇒ 임계 통과 후 정지
