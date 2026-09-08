"""
B1 / 실주행 1차 [2] — 연결로 곡률 감속 (speed.curvature_cap_enable).

실측 logs/batch/20260908_130919/실경로_01_PathShape03 rs 89~107: 연결로 797
(경로 곡률 R 6.5 m)에 6.8 m/s 로 진입해 조향이 −0.480 에 **9 m 포화**하고
t_off 가 1.36 m 까지 벌어진 뒤 +0.480 으로 되튀었다 (항목 3 차로유지, 중대).
그 구간 reasons.curvature 는 **전 틱 None** — 곡률을 보는 속도 후보가 없었다.

여기서 지키는 불변:
  · κ 는 **실제로 따라갈 경로점**(planner.route_points)에서 잰다. 차로 R 이
    아니다 — 회피 시프트·차선변경 블렌드가 경로를 밀면 따라가는 선이 더 굽는다.
  · min() **상한** 후보다. 신호·보행자·standoff 가 더 낮으면 그쪽이 이긴다.
  · 직선(κ < curvature_min_kappa)에서는 후보를 만들지 않는다 — 좌표 잡음이
    상한을 만들면 직선에서도 느려진다.
"""
import copy
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
SP = CFG['speed']
A_LAT = SP['curvature_a_lat_max']


def on_cfg(**over):
    c = copy.deepcopy(CFG)
    c['speed']['curvature_cap_enable'] = True
    c['speed'].update(over)
    return c


def off_cfg(**over):
    """이전 동작 사본 — 기본값을 읽지 않는다 (2026-09-07 드리프트 원칙)."""
    c = copy.deepcopy(CFG)
    c['speed']['curvature_cap_enable'] = False
    c['speed'].update(over)
    return c


class ArcPlanner:
    """반지름 R 원호(또는 직선) 경로. 기하가 손으로 검산된다."""

    def __init__(self, R=None, n=4000, ppm=10):
        self.points_per_meter = ppm
        self.route_index = 0
        step = 1.0 / ppm
        if R is None:
            xy = [(i * step, 0.0) for i in range(n)]
        else:
            xy = [(R * math.sin(i * step / R), R * (1 - math.cos(i * step / R)))
                  for i in range(n)]
        self.route_points = np.array([(x, y, 0.0) for x, y in xy])
        self.route_s = np.arange(n) * step


def cap(cfg, R, **kw):
    kr = KrRules(cfg)
    return kr._curvature_cap(ArcPlanner(R, **kw)), kr


def test_off_makes_no_candidate():
    v, kr = cap(off_cfg(), 6.5)
    assert v is None and kr.last_curv_info is None


def test_straight_makes_no_candidate():
    """직선에서는 후보가 없다 — 좌표 잡음이 상한을 만들면 안 된다."""
    assert cap(on_cfg(), None)[0] is None


@pytest.mark.parametrize('R', [6.5, 10.0, 13.0, 25.0])
def test_cap_is_sqrt_alat_over_kappa(R):
    v, kr = cap(on_cfg(), R)
    assert v == pytest.approx(math.sqrt(A_LAT * R), rel=0.02)
    assert kr.last_curv_info['R_m'] == pytest.approx(R, rel=0.02)


def test_measured_connector_matches_the_log():
    """실측 연결로 797 (R 6.5 m) → 4.0 m/s. 진입 6.8 m/s 가 이 값으로 눌린다."""
    v, _ = cap(on_cfg(), 6.5)
    assert v == pytest.approx(4.03, abs=0.1)
    assert v < 4.5                                     # 검증 기준


def test_a_lat_max_scales_the_cap():
    a, _ = cap(on_cfg(curvature_a_lat_max=2.5), 10.0)
    b, _ = cap(on_cfg(curvature_a_lat_max=4.0), 10.0)
    assert b / a == pytest.approx(math.sqrt(4.0 / 2.5), rel=0.02)


def test_min_kappa_filters_gentle_curves():
    """R 이 아주 크면(≈직선) 후보를 안 만든다 — 임계는 curvature_min_kappa."""
    R = 1.0 / SP['curvature_min_kappa'] * 2.0          # 임계의 절반 곡률
    assert cap(on_cfg(), R)[0] is None
    assert cap(on_cfg(), R / 4.0)[0] is not None       # 임계보다 굽으면 산다


def test_lookahead_sees_the_curve_before_entering_it():
    """직선 뒤에 곡선이 오면 **들어가기 전에** 상한이 걸려야 한다."""
    ppm, R, straight_m = 10, 6.5, 10.0
    step = 1.0 / ppm
    pts = [(i * step, 0.0) for i in range(int(straight_m * ppm))]
    x0, y0 = pts[-1]
    pts += [(x0 + R * math.sin(i * step / R), y0 + R * (1 - math.cos(i * step / R)))
            for i in range(2000)]
    p = ArcPlanner(None)
    p.route_points = np.array([(x, y, 0.0) for x, y in pts])
    p.route_s = np.arange(len(pts)) * step
    kr = KrRules(on_cfg())
    assert kr._curvature_cap(p) == pytest.approx(math.sqrt(A_LAT * R), rel=0.05)


def test_beyond_lookahead_is_not_seen():
    """창(curvature_lookahead_m) 밖의 곡선은 아직 안 본다."""
    ppm, R = 10, 6.5
    step = 1.0 / ppm
    straight_m = SP['curvature_lookahead_m'] + 20.0
    pts = [(i * step, 0.0) for i in range(int(straight_m * ppm))]
    x0, y0 = pts[-1]
    pts += [(x0 + R * math.sin(i * step / R), y0 + R * (1 - math.cos(i * step / R)))
            for i in range(2000)]
    p = ArcPlanner(None)
    p.route_points = np.array([(x, y, 0.0) for x, y in pts])
    p.route_s = np.arange(len(pts)) * step
    assert KrRules(on_cfg())._curvature_cap(p) is None
