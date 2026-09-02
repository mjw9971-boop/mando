"""
정지선(적신호) 정지 정합 (대회 7번: 범퍼가 선 앞 2 m 이내 & 0.5 s 이상 정지).

  · run_agent.build_pdm_config — 앞범퍼가 정지선 앞 speed.stop_gap_stopline_m 에
    서도록 PDM idm_red_light_minimum_distance = stopline_gap + 앞범퍼 주입
    (PDM 원문 무수정).
  · route_end(stop_gap_route_end_m)·완주 임계는 영향 없다 (상수 분리).
  · kr_rules stopline hold — 저속 진입 시 stopline_hold_s 동안 목표 0 유지.
    **B-1(2026-09-01)**: 연속 정지 중에는 재무장하지 않고, 녹색 전환 시 잔여를
    버리되 규정 최소(stopline_hold_min_s)를 못 채운 만큼만 남긴다.
"""
import copy
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import end_margin_m, load_params_yaml
from vtd_adapter.control import VtdLongitudinalController

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from autopilot import AutoPilot                     # noqa: E402
from config import GlobalConfig                     # noqa: E402
from kr_rules import KrRules                        # noqa: E402
from run_agent import build_pdm_config              # noqa: E402
from test_route_end import FakeEgo, FakePlanner, TOTAL, _apply  # noqa: E402
from vtd_adapter.carla_types import VehicleControl  # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
GAP_SL = CFG['speed']['stop_gap_stopline_m']
HOLD_TICKS = round(CFG['speed']['stopline_hold_s'] * CFG['comm']['send_hz'])
MIN_TICKS = round(CFG['speed']['stopline_hold_min_s'] * CFG['comm']['send_hz'])


# ── 상수 분리: 주입과 불변 ───────────────────────────────────────────────
def test_pdm_config_red_light_distance_derived_from_params():
    """계획 정지: IDM 평형에서 뒷축 gap = s0 → 앞범퍼 = s0 − front = stopline_gap."""
    gc = build_pdm_config(CFG)
    assert gc.idm_red_light_minimum_distance == pytest.approx(GAP_SL + FRONT)
    assert gc.idm_red_light_minimum_distance - FRONT == pytest.approx(GAP_SL)


def test_route_end_and_done_threshold_unaffected():
    """route_end 유령차 s0 와 완주 임계는 stop_gap_route_end_m(4.0) 그대로."""
    assert KrRules(CFG).stop_gap == pytest.approx(CFG['speed']['stop_gap_route_end_m'])
    assert end_margin_m(CFG) == pytest.approx(
        CFG['speed']['stop_gap_route_end_m'] + FRONT + CFG['batch']['end_slack_m'])


def test_pdm_default_was_wider_than_rule_window():
    """이 주입이 필요한 이유 고정: PDM 원문 기본(6.0)은 앞범퍼 −2.2 m 계획이라
    규칙 창(2 m 이내) 밖이다."""
    assert GlobalConfig().idm_red_light_minimum_distance - FRONT > 2.0
    assert 0.0 < GAP_SL <= 2.0


# ── stopline hold ────────────────────────────────────────────────────────
class FakeTL:
    def __init__(self, state=TrafficLightState.Red):
        self.state = state


class FakePlannerTL(FakePlanner):
    """다음 신호 정지선이 d_tl(뒷축 기준) 앞에 있는 플래너."""

    def __init__(self, d_tl=5.0):
        super().__init__()
        n = len(self.route_s)
        self.tl = FakeTL()
        self.next_traffic_lights = [self.tl] * n
        self.distances_to_next_traffic_lights = np.full(n, float(d_tl))


def make_ap(planner):
    a = AutoPilot()
    a.setup(world=None, world_map=None, waypoint_planner=planner,
            longitudinal_controller=VtdLongitudinalController(CFG),
            ego_vehicle=FakeEgo(), config=GlobalConfig())
    a.kr_rules = KrRules(CFG)
    return a


def test_hold_starts_and_keeps_zero_for_hold_duration():
    """B-1 ①: 적색 연속 정지 중 홀드는 **1회 무장**이고 리필하지 않는다.

    홀드가 만료돼도 목표 0 이 유지되는 것은 ④′ 프로파일의 몫이므로, 여기서는
    홀드 자체(_stopline_hold 의 반환)를 본다.
    """
    planner = FakePlannerTL(d_tl=GAP_SL + FRONT)       # 계획 정지 위치에 서 있다
    kr = make_ap(planner).kr_rules
    assert kr._stopline_hold(planner, 0.3) == 0.0      # 무장 + 이번 틱 소모
    for _ in range(HOLD_TICKS - 1):
        assert kr._stopline_hold(planner, 0.0) == 0.0
    # 리필이 없으므로 정확히 HOLD_TICKS 만에 만료된다 (옛 동작은 영구 0)
    assert kr._stopline_hold(planner, 0.0) is None
    assert kr._stopline_hold(planner, 0.0) is None


def test_hold_rearms_only_when_stop_continuity_breaks():
    """B-1 ②: 굴러갔다 다시 서면 새 정지 — 그때만 재무장한다."""
    planner = FakePlannerTL(d_tl=GAP_SL + FRONT)
    kr = make_ap(planner).kr_rules
    for _ in range(HOLD_TICKS):
        kr._stopline_hold(planner, 0.0)
    assert kr._stopline_hold(planner, 0.0) is None          # 만료
    kr._stopline_hold(planner, 3.0)                          # 연속성 파괴 (주행)
    assert kr._stopline_hold(planner, 0.0) == 0.0            # 새 정지 → 재무장
    for _ in range(HOLD_TICKS - 1):
        assert kr._stopline_hold(planner, 0.0) == 0.0
    assert kr._stopline_hold(planner, 0.0) is None


def test_hold_cleared_on_green_when_minimum_already_met():
    """B-1 ③: 녹색 전환 시 최소 시간을 이미 채웠으면 잔여를 즉시 버린다."""
    planner = FakePlannerTL(d_tl=GAP_SL + FRONT)
    kr = make_ap(planner).kr_rules
    for _ in range(MIN_TICKS):                               # 규정 최소만큼 정지
        assert kr._stopline_hold(planner, 0.0) == 0.0
    planner.tl.state = TrafficLightState.Green
    assert kr._stopline_hold(planner, 0.0) is None            # 잔여 클리어


def test_hold_keeps_only_the_shortfall_on_green():
    """B-1 ④: 최소 시간을 못 채운 채 녹색이면 **모자란 만큼만** 남긴다."""
    planner = FakePlannerTL(d_tl=GAP_SL + FRONT)
    kr = make_ap(planner).kr_rules
    stopped = 2
    for _ in range(stopped):
        assert kr._stopline_hold(planner, 0.0) == 0.0
    planner.tl.state = TrafficLightState.Green
    for _ in range(MIN_TICKS - stopped):                     # 부족분만
        assert kr._stopline_hold(planner, 0.0) == 0.0
    assert kr._stopline_hold(planner, 0.0) is None
    assert MIN_TICKS < HOLD_TICKS                            # 스펙 전제 (0.5 < 1.0)


def test_hold_zero_is_kept_by_profile_while_red():
    """적색이 지속되는 동안의 목표 0 유지는 홀드가 아니라 ④′ 프로파일이 한다.

    B-1 이 리필을 없앤 뒤에도 '적색 대기 중 재출발' 이 생기지 않음을 고정한다
    (BACKLOG B-1 의 주의 항목).
    """
    planner = FakePlannerTL(d_tl=GAP_SL + FRONT)
    ap = make_ap(planner)
    d_far = TOTAL - 200.0
    for _ in range(HOLD_TICKS + 20):
        _c, ts = _apply(ap, d_far, v=0.0)
        assert ts == 0.0


def test_no_hold_when_far_or_fast_or_green():
    """홀드 자체를 단언한다 — 최종 목표속도로 보면 다른 후보(정지선 정지
    프로파일)와 섞인다. 멀거나/빠르거나/녹색이면 홀드는 걸리지 않는다."""
    # 멀다 (앞범퍼 기준 near 창 밖)
    p = FakePlannerTL(d_tl=CFG['speed']['stopline_hold_near_m'] + FRONT + 1.0)
    assert make_ap(p).kr_rules._stopline_hold(p, 0.3) is None
    # 빠르다
    p = FakePlannerTL(d_tl=GAP_SL + FRONT)
    assert make_ap(p).kr_rules._stopline_hold(p, 3.0) is None
    # 녹색
    p = FakePlannerTL(d_tl=GAP_SL + FRONT)
    p.tl.state = TrafficLightState.Green
    assert make_ap(p).kr_rules._stopline_hold(p, 0.3) is None


def test_no_hold_without_traffic_light_info():
    """신호 정보가 없는 플래너(기존 FakePlanner)는 미개입 — route_end 회귀 보장."""
    ap = make_ap(FakePlanner())
    assert _apply(ap, TOTAL - 200.0, v=0.3)[1] == pytest.approx(12.5)
