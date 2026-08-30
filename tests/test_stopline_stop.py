"""
정지선(적신호) 정지 정합 (대회 7번: 범퍼가 선 앞 2 m 이내 & 0.5 s 이상 정지).

  · run_agent.build_pdm_config — 앞범퍼가 정지선 앞 speed.stop_gap_stopline_m 에
    서도록 PDM idm_red_light_minimum_distance = stopline_gap + 앞범퍼 주입
    (PDM 원문 무수정).
  · route_end(stop_gap_route_end_m)·완주 임계는 영향 없다 (상수 분리).
  · kr_rules stopline hold — 저속 진입 시 stopline_hold_s 동안 목표 0 유지,
    홀드 중 녹색 전환에도 잔여를 채운다 (실측 0.4 s 재출발 사례 방어).
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
    planner = FakePlannerTL(d_tl=GAP_SL + FRONT)       # 계획 정지 위치에 서 있다
    ap = make_ap(planner)
    d_far = TOTAL - 200.0                              # route_end 비활성 거리
    _c, ts = _apply(ap, d_far, v=0.3)
    assert ts == 0.0                                   # 홀드 시작
    # 홀드 중 신호가 녹색으로 바뀌어도 잔여 틱 동안 0 유지 (규정 0.5 s 채움)
    planner.tl.state = TrafficLightState.Green
    for _ in range(HOLD_TICKS - 1):
        _c, ts = _apply(ap, d_far, v=0.0)
        assert ts == 0.0
    # 만료 후엔 개입하지 않는다 (녹색이므로 재홀드도 없다)
    _c, ts = _apply(ap, d_far, v=0.0)
    assert ts == pytest.approx(12.5)


def test_hold_rearms_while_red():
    """만료 후에도 적색이 계속이면 재홀드 — 적신호 동안 0 유지가 이어진다."""
    planner = FakePlannerTL(d_tl=GAP_SL + FRONT)
    ap = make_ap(planner)
    d_far = TOTAL - 200.0
    for _ in range(HOLD_TICKS + 3):
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
