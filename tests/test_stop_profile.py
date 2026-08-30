"""
정지선 정지 프로파일 (④′, kr_rules._stopline_profile, params speed.stop_profile_a).

PDM 의 적신호 IDM 은 차간모형이라 정지 컨트롤러가 아니다 — 평형 s* 보다 남은
거리가 조금만 커도 가속을 요구한다 (실측 2026-08-30: 접근 92틱 중 41틱,
err/dt 최대 +10.5 m/s²). 종방향이 브레이크를 풀면 타행으로 정지선을 넘는다
(앞범퍼 −1.50 목표에 −0.12 착지).

    v_allow = √(2 · a_stop · (정지선거리 − s0))     ← min() 후보

여기서 지키는 불변:
  · **정지선 한정** — route_end 는 대상이 아니다 (별건)
  · **적색에서만** — 녹색·황색은 후보를 만들지 않는다 (황색은 PDM 원문 소관)
  · 단조 감소 — 감속 커맨드에 부호 반전이 없다 (jerk 리미터 되감기 제거)
  · d_stop → 0 에서 v_allow → 0, 종방향 a_hold 분기로 자연 접속
  · s0 는 PDM 에 주입된 idm_red_light_minimum_distance 단일 출처
  · stop_profile_a = 0 이면 완전 비활성 (되돌리기 스위치)
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

from autopilot import AutoPilot                                  # noqa: E402
from kr_rules import KrRules                                     # noqa: E402
from run_agent import build_pdm_config                           # noqa: E402
from test_route_end import FakeEgo, FakePlanner, TOTAL, _apply    # noqa: E402
from test_stopline_stop import FakePlannerTL                     # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
A_STOP = CFG['speed']['stop_profile_a']
PDM = build_pdm_config(CFG)
S0 = PDM.idm_red_light_minimum_distance          # 5.299 = stop_gap_stopline_m + 앞범퍼
D_FAR = TOTAL - 200.0                            # route_end 비활성 거리


def make_ap(planner, cfg=CFG):
    """실제 주입 설정(build_pdm_config)으로 조립 — s0 단일 출처를 그대로 탄다."""
    a = AutoPilot()
    a.setup(world=None, world_map=None, waypoint_planner=planner,
            longitudinal_controller=VtdLongitudinalController(cfg),
            ego_vehicle=FakeEgo(), config=build_pdm_config(cfg))
    a.kr_rules = KrRules(cfg)
    return a


def v_allow(d_line):
    return math.sqrt(2.0 * A_STOP * max(0.0, d_line - S0))


# ── 수식 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize('d_line', [S0 + 40.0, S0 + 12.0, S0 + 3.0, S0 + 0.5, S0, S0 - 2.0])
def test_profile_matches_formula(d_line):
    p = FakePlannerTL(d_tl=d_line)
    ap = make_ap(p)
    assert ap.kr_rules._stopline_profile(p, ap) == pytest.approx(v_allow(d_line))


def test_profile_uses_pdm_injected_s0_not_a_local_copy():
    """s0 는 PDM 주입값이 단일 출처 — 그 값이 바뀌면 프로파일도 따라간다."""
    p = FakePlannerTL(d_tl=S0 + 10.0)
    ap = make_ap(p)
    ap.config.idm_red_light_minimum_distance = S0 + 5.0     # 주입값만 변경
    assert ap.kr_rules._stopline_profile(p, ap) == pytest.approx(
        math.sqrt(2.0 * A_STOP * 5.0))


def test_profile_is_monotone_decreasing_as_it_approaches():
    """단조 감소여야 감속 커맨드에 부호 반전이 없다 (jerk 되감기 방지)."""
    ap = make_ap(FakePlannerTL(d_tl=S0))
    prev = None
    for d in np.arange(S0 + 30.0, S0 - 0.01, -0.25):
        p = FakePlannerTL(d_tl=float(d))
        cur = ap.kr_rules._stopline_profile(p, ap)
        if prev is not None:
            assert cur <= prev + 1e-9
        prev = cur
    assert prev == pytest.approx(0.0)


# ── 적색에서만 ───────────────────────────────────────────────────────────
@pytest.mark.parametrize('state', [TrafficLightState.Green, TrafficLightState.Yellow,
                                   TrafficLightState.Off, TrafficLightState.Unknown])
def test_no_candidate_unless_red(state):
    """녹색·황색(현행 취급)에서는 후보를 만들지 않는다 — 황색은 PDM 원문 소관."""
    p = FakePlannerTL(d_tl=S0 + 5.0)
    p.tl.state = state
    ap = make_ap(p)
    assert ap.kr_rules._stopline_profile(p, ap) is None
    assert _apply(ap, D_FAR, v=8.0)[1] == pytest.approx(12.5)   # 무감속


def test_green_pass_through_is_unchanged():
    """역회귀: 녹색 정지선을 지나갈 때 목표속도가 손대지지 않는다."""
    p = FakePlannerTL(d_tl=S0 + 2.0)
    p.tl.state = TrafficLightState.Green
    ap = make_ap(p)
    for v in (0.0, 3.0, 8.0, 12.5):
        assert _apply(ap, D_FAR, v=v)[1] == pytest.approx(12.5)


def test_no_candidate_without_traffic_light_info():
    """신호 정보가 없는 환경(목 플래너)에서는 개입하지 않는다."""
    p = FakePlanner()
    ap = make_ap(p)
    assert ap.kr_rules._stopline_profile(p, ap) is None


# ── min() 합류 ──────────────────────────────────────────────────────────
def test_candidate_joins_min_and_never_raises_target():
    """min() 후보다 — 목표를 올리는 일은 없어야 한다 (외부 오버라이드 금지)."""
    for d in (S0 + 60.0, S0 + 20.0, S0 + 5.0, S0 + 1.0):
        ap = make_ap(FakePlannerTL(d_tl=d))
        assert _apply(ap, D_FAR, v=8.0)[1] <= 12.5 + 1e-9


def test_far_red_light_does_not_bind():
    """멀면 v_allow 가 제한속도보다 커서 스스로 비활성 (발동 거리 상수 불필요)."""
    d = S0 + 12.5 ** 2 / (2 * A_STOP) + 5.0        # v_allow > 12.5 가 되는 거리
    ap = make_ap(FakePlannerTL(d_tl=d))
    assert ap.kr_rules._stopline_profile(FakePlannerTL(d_tl=d), ap) > 12.5
    assert _apply(ap, D_FAR, v=12.0)[1] == pytest.approx(12.5)


# ── 종단: a_hold 접속 ────────────────────────────────────────────────────
def test_terminal_reaches_zero_and_hands_over_to_a_hold():
    """d_stop → 0 에서 목표 0, 저속이면 종방향이 a_hold 로 받는다."""
    p = FakePlannerTL(d_tl=S0)                      # 계획 정지점에 정확히 도달
    ap = make_ap(p)
    assert ap.kr_rules._stopline_profile(p, ap) == pytest.approx(0.0)
    control, ts = _apply(ap, D_FAR, v=0.1)          # v < 0.2 → a_hold 분기
    assert ts == pytest.approx(0.0)
    assert control.accel == pytest.approx(CFG['speed']['a_hold'])


def test_terminal_still_braking_above_hold_speed():
    """v ≥ 0.2 면 a_hold 가 아니라 감속 명령이 나간다 (크립으로 넘지 않게)."""
    p = FakePlannerTL(d_tl=S0)
    ap = make_ap(p)
    control, ts = _apply(ap, D_FAR, v=1.0)
    assert ts == pytest.approx(0.0)
    assert control.accel < 0.0


def test_past_the_planned_point_still_commands_zero():
    """계획 정지점을 넘겼으면 (d < s0) 음수 제곱근이 아니라 0 이어야 한다."""
    p = FakePlannerTL(d_tl=S0 - 3.0)
    ap = make_ap(p)
    assert ap.kr_rules._stopline_profile(p, ap) == pytest.approx(0.0)


# ── 정지선 한정 (route_end 비대상) ───────────────────────────────────────
def test_route_end_is_not_covered():
    """route_end 는 이번 범위가 아니다 — 신호가 없으면 프로파일은 None 이고,
    종점 정지는 기존 유령차 후보가 그대로 낸다."""
    p = FakePlanner()                                # 신호 정보 없음
    ap = make_ap(p)
    assert ap.kr_rules._stopline_profile(p, ap) is None
    _c, ts = _apply(ap, d_end=5.0, v=0.3)            # 종점 근처 → 기존 래치
    assert ts == pytest.approx(0.0)


def test_route_end_ghost_gap_constant_untouched():
    """route_end 유령차 s0 는 stop_gap_route_end_m 그대로 (프로파일과 무관)."""
    assert KrRules(CFG).stop_gap == pytest.approx(CFG['speed']['stop_gap_route_end_m'])


# ── 홀드와의 상호작용 ────────────────────────────────────────────────────
def test_hold_and_profile_are_both_zero_at_the_stop_point():
    """계획 정지점에서 둘 다 0 — min() 이라 서로를 약화시키지 않는다."""
    p = FakePlannerTL(d_tl=S0)
    ap = make_ap(p)
    assert ap.kr_rules._stopline_profile(p, ap) == pytest.approx(0.0)
    assert ap.kr_rules._stopline_hold(p, 0.1) == pytest.approx(0.0)


def test_hold_still_wins_after_green_while_profile_releases():
    """홀드 잔여가 있으면 녹색 전환 뒤에도 0 이 유지된다 (기존 동작 회귀).

    프로파일은 녹색에서 즉시 None 이 되므로, 녹색 후 0 을 유지하는 주체는
    홀드뿐이다 — 이 경계가 리필 수정 때 바뀔 지점이라 여기서 고정해 둔다.
    """
    p = FakePlannerTL(d_tl=S0)
    ap = make_ap(p)
    assert _apply(ap, D_FAR, v=0.1)[1] == pytest.approx(0.0)     # 적색: 홀드 무장
    p.tl.state = TrafficLightState.Green
    assert ap.kr_rules._stopline_profile(p, ap) is None          # 프로파일은 즉시 해제
    assert _apply(ap, D_FAR, v=0.0)[1] == pytest.approx(0.0)     # 홀드 잔여가 0 유지


# ── 스위치 ──────────────────────────────────────────────────────────────
def test_disabled_switch_is_fully_inert():
    cfg = copy.deepcopy(CFG)
    cfg['speed']['stop_profile_a'] = 0.0
    p = FakePlannerTL(d_tl=S0 + 5.0)
    ap = make_ap(p, cfg)
    assert ap.kr_rules._stopline_profile(p, ap) is None
    assert _apply(ap, D_FAR, v=8.0)[1] == pytest.approx(12.5)
