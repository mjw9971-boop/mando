"""
kr_rules 정지 목표 stop_s (완주 기준 정합 — 2026-08-27).

규칙: 뒷축이 종료 지점(finish_xy → finish_s) 통과 = 시험 종료. 정지 목표는
stop_s = min(finish_s + clearance + stop_gap + 앞범퍼, total − end_slack) 로,
정지 시 뒷축 = finish_s + clearance 가 되게 한다 (stop_gap 누락 시 뒷축이
finish_s − 2.0 에 서는 결함을 잡은 정정 반영). 채점(score.py)은 같은
plan_stop_s 를 쓰되 통과 기준은 finish_s 그 자체다.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.control import VtdLongitudinalController

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from autopilot import AutoPilot                     # noqa: E402
from config import GlobalConfig                     # noqa: E402
from kr_rules import KrRules, plan_stop_s           # noqa: E402
from vtd_adapter.carla_types import VehicleControl  # noqa: E402
from test_route_end import (FakeEgo, FakePlanner, TOTAL,   # noqa: E402
                            _apply, _closed_loop)

CFG = load_params_yaml(PARAMS_YAML)
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
GAP = CFG['speed']['stop_gap_route_end_m']
CLEAR = CFG['scoring']['finish_clearance_m']
SLACK = CFG['batch']['end_slack_m']


def make_ap(cfg=CFG):
    a = AutoPilot()
    a.setup(world=None, world_map=None, waypoint_planner=FakePlanner(),
            longitudinal_controller=VtdLongitudinalController(cfg),
            ego_vehicle=FakeEgo(), config=GlobalConfig())
    a.kr_rules = KrRules(cfg)
    return a


# ── plan_stop_s 단위 ─────────────────────────────────────────────────────
def test_plan_stop_s_rear_axle_lands_past_finish():
    """항등식: 정지 시 뒷축(= stop_s − stop_gap − front) = finish_s + clearance."""
    finish_s = 200.0
    stop_s, clipped = plan_stop_s(CFG, total=300.0, finish_s=finish_s)
    assert not clipped
    assert stop_s == pytest.approx(finish_s + CLEAR + GAP + FRONT)
    assert stop_s - GAP - FRONT == pytest.approx(finish_s + CLEAR)


def test_plan_stop_s_without_finish_keeps_total():
    assert plan_stop_s(CFG, total=300.0, finish_s=None) == (300.0, False)


def test_plan_stop_s_clips_when_route_too_short():
    """종료선이 경로 끝에 붙어 있으면(기본 경로 total−0.2) 클립 + 플래그."""
    stop_s, clipped = plan_stop_s(CFG, total=300.0, finish_s=300.0 - 0.2)
    assert clipped and stop_s == pytest.approx(300.0 - SLACK)


# ── KrRules stop_s 해석 (투영·경고·폴백) ─────────────────────────────────
class FakeLg:
    def project(self, key, x, y):
        return float(x), 0.0, 0.0, 0        # finish_s = x 좌표로 매핑


class FakePlannerLg(FakePlanner):
    def __init__(self, total=TOTAL):
        super().__init__(total)
        self.route = {'total_length': total, 'lanes': [(1, 0, -1)], 'cum_s': [0.0]}
        self.lg = FakeLg()


def cfg_with_finish(x):
    cfg = copy.deepcopy(CFG)
    cfg['scoring']['finish_xy'] = [x, 0.0]
    return cfg


def test_resolve_uses_finish_projection():
    cfg = cfg_with_finish(TOTAL - 40.0)
    kr = KrRules(cfg)
    stop_s = kr._resolve_stop_s(FakePlannerLg())
    assert stop_s == pytest.approx(TOTAL - 40.0 + CLEAR + GAP + FRONT)


def test_resolve_warns_and_clips_on_short_tail(capsys):
    cfg = cfg_with_finish(TOTAL - 0.2)
    kr = KrRules(cfg)
    stop_s = kr._resolve_stop_s(FakePlannerLg())
    assert stop_s == pytest.approx(TOTAL - SLACK)
    assert '못 넘는다' in capsys.readouterr().out


def test_resolve_falls_back_without_finish_xy(capsys):
    kr = KrRules(CFG)                      # finish_xy: null
    assert kr._resolve_stop_s(FakePlanner()) == pytest.approx(TOTAL)
    assert '미설정' in capsys.readouterr().out


# ── 래치·폐루프가 stop_s 기준으로 도는지 ─────────────────────────────────
def test_latch_uses_stop_s_not_total():
    """stop_s 가 종점보다 30 m 앞이면, total 기준 38 m 남았어도 래치가 걸린다."""
    ap = make_ap()
    stop_s = TOTAL - 30.0
    ap.kr_rules.stop_s = stop_s
    d_total = TOTAL - (stop_s - 8.0)       # 뒷축을 stop_s − 8 에 놓는다
    _apply(ap, d_total, v=0.3)
    assert ap.kr_rules.latched
    assert ap.kr_rules.last_d_end == pytest.approx(8.0)


def test_closed_loop_rear_axle_clears_finish_line():
    """finish_s = total−40: 정지 완료 시 뒷축 route_s ≥ finish_s + clearance."""
    finish_s = TOTAL - 40.0
    ap = make_ap()
    stop_s, clipped = plan_stop_s(CFG, TOTAL, finish_s)
    assert not clipped
    ap.kr_rules.stop_s = stop_s
    d_stop, vs = _closed_loop(ap, d0=70.0, v0=8.0)
    assert ap.kr_rules.latched and vs[-1] < 0.01
    rear = TOTAL - d_stop
    assert rear >= finish_s + CLEAR, \
        f'뒷축 {rear:.2f} < 종료선+여유 {finish_s + CLEAR:.2f}'
    # 본질 스펙(경로 밖 이탈 방지): 종점을 넘지 않는다 — 오버런(수 m 허용)은
    # stop_s 가 종점보다 30 m 앞이라 종점 여유가 충분하다
    assert d_stop > 0.0, f'경로 종점을 {-d_stop:.2f} m 넘어 정지'
