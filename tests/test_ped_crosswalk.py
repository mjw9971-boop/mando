"""
A-2 + A-3 — 인도 정지 보행자 무한 정지 (2026-09-03 실전주행_교통류_02_직진11).

실측: 우측 인도(차로 중심 |lat| 3.17 m, 차로폭 3.33)에 도로를 향해 **서 있는** 보행자
id3 앞에서 winner=walker, PDM pedestrian 후보 0.0, ped.wins None 으로 121 s 정지
(no_progress). kr_rules 래치는 걷는 보행자만 잡으므로 무관 — PDM forecast_walkers 가
min_walker_speed 0.5 로 2 s 를 도로 쪽으로 밀고 반폭 1.5 상자를 씌운 결과다.

  A-2 pedestrian_minimum_extent 1.5 → 0.5: 정지 보행자 겹침 임계 0.943 + 0.5 = 1.443 m.
  A-3 횡단보도 앞 서행 (기본 false): 정지 보행자 앞 정지 프로파일 → 정지 중 3 s 대기
      → creep 2.0 상한으로 통과. 래치 우선, 회랑 안 제외, 킬 스위치.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import VehicleControl
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from config import GlobalConfig                                      # noqa: E402
from kr_rules import KrRules                                         # noqa: E402
from test_avoid import HZ, Ap, Planner                               # noqa: E402
from test_ped_intent import Walker, make_ap, observe_static, walk    # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
SP = CFG['speed']
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
WAIT_TICKS = int(round(SP['ped_crosswalk_wait_s'] * HZ))
CREEP_V = SP['ped_crosswalk_creep_v']
LAT_MIN, LAT_MAX = SP['ped_release_lat_m'], SP['ped_crosswalk_lat_m']


def cw_cfg(**kw):
    c = copy.deepcopy(CFG)
    c['speed']['ped_crosswalk_creep_enable'] = True
    c['speed'].update(kw)
    return c


# ── A-2 ──────────────────────────────────────────────────────────────────
def test_pedestrian_extent_threshold_is_1p443():
    """자차 반폭 0.943 + 0.5 = 1.443 m — 인도(3.17 m) 정지 보행자는 겹치지 않는다."""
    gc = GlobalConfig()
    assert gc.pedestrian_minimum_extent == 0.5
    assert CFG['pdm']['pedestrian_minimum_extent'] == 0.5
    thr = CFG['vehicle']['width'] / 2.0 + gc.pedestrian_minimum_extent
    assert thr == pytest.approx(1.443, abs=0.001)
    # 정지 보행자 2 s 예측 이동(min_walker_speed 0.5) 을 더해도 인도 3.17 m 는 밖
    assert 3.17 - gc.min_walker_speed * 2.0 > thr


def test_pdm_config_injects_extent_from_params():
    from run_agent import build_pdm_config
    c = copy.deepcopy(CFG)
    c['pdm']['pedestrian_minimum_extent'] = 1.5
    assert build_pdm_config(c).pedestrian_minimum_extent == 1.5           # 원복 스위치
    assert build_pdm_config(CFG).pedestrian_minimum_extent == 0.5


# ── A-3 ──────────────────────────────────────────────────────────────────
def rig(cfg=None, lat=-3.2, x=20.0, zones=((15.0, 25.0),), ped_speed=0.0):
    """x 앞, 횡거리 lat 에 서 있는 보행자. 횡단보도 구간은 직접 주입 (목 플래너)."""
    cfg = cw_cfg() if cfg is None else cfg
    p = Planner()
    w = Walker(7, x, lat, speed=ped_speed)
    ap = make_ap(p, [w], cfg=cfg)
    kr = ap.kr_rules
    kr._sl_all = []
    kr._cw_zones = [tuple(z) for z in zones]
    return kr, p, ap, w


def step(kr, p, ap, v, n=1):
    out = None
    for _ in range(n):
        kr._update_obj_timers(ap)
        kr._ped_intent(p, ap, v)
        out = kr._ped_crosswalk(p, ap, v)
    return out


def test_wait_counts_only_while_stopped_at_plan_point_then_creeps():
    kr, p, ap, w = rig()
    v0, info = step(kr, p, ap, 8.0)
    d_eff = 20.0 - FRONT - 4.0
    assert info['phase'] == 'wait' and v0 == pytest.approx((2 * SP['stop_profile_a'] * d_eff) ** 0.5)
    step(kr, p, ap, 8.0, 3 * WAIT_TICKS)                     # 주행 중엔 대기가 쌓이지 않는다
    assert kr.cw_wait[7] == 0
    p.route_index = int((20.0 - FRONT - 4.0) * 10) + 2       # 계획 정지점(살짝 지남) 정지
    v1, info = step(kr, p, ap, 0.0, WAIT_TICKS - 1)
    assert info['phase'] == 'wait' and v1 == 0.0
    v2, info = step(kr, p, ap, 0.0, 1)
    assert info['phase'] == 'creep' and v2 == pytest.approx(CREEP_V)
    v3, info = step(kr, p, ap, 1.5, 5)                        # 출발 후에도 상한 유지
    assert v3 == pytest.approx(CREEP_V) and info['wait_s'] == pytest.approx(SP['ped_crosswalk_wait_s'])


def test_wait_counts_when_stopped_a_few_m_before_plan_point_not_far():
    """PDM 이 계획 정지점 몇 m 앞에 먼저 세워도 대기가 찬다. 30 m 밖 정지는 아니다."""
    kr, p, ap, w = rig(x=50.0, zones=((45.0, 55.0),))
    p.route_index = int((50.0 - FRONT - 4.0 - 3.0) * 10)     # 계획 정지점 3 m 앞
    _v, info = step(kr, p, ap, 0.0, WAIT_TICKS)
    assert info['phase'] == 'creep'
    kr, p, ap, w = rig(x=50.0, zones=((45.0, 55.0),))
    p.route_index = 0                                          # 계획 정지점 42 m 앞 (먼 정지)
    _v, info = step(kr, p, ap, 0.0, WAIT_TICKS)
    assert info['phase'] == 'wait' and kr.cw_wait[7] == 0


def test_released_after_passing_and_wait_restarts():
    kr, p, ap, w = rig()
    p.route_index = int((20.0 - FRONT - 4.0) * 10) + 2
    step(kr, p, ap, 0.0, WAIT_TICKS)
    assert kr.cw_wait[7] == WAIT_TICKS
    p.route_index = 210                                       # 보행자를 지났다
    assert step(kr, p, ap, 2.0) is None and 7 not in kr.cw_wait


def test_apply_joins_as_min_candidate_and_logs():
    kr, p, ap, w = rig()
    p.route_index = int((20.0 - FRONT - 4.0) * 10) + 2
    ap._vehicle.speed = 0.0
    for _ in range(WAIT_TICKS - 1):
        ap._longitudinal_controller.get_throttle_and_brake(False, 12.5, 0.0)
        _c, t = kr.apply(VehicleControl(), 12.5, ap)
    assert t == 0.0 and kr.last_ped['crosswalk']['phase'] == 'wait'
    ap._longitudinal_controller.get_throttle_and_brake(False, 12.5, 0.0)
    _c, t = kr.apply(VehicleControl(), 12.5, ap)
    assert t == pytest.approx(CREEP_V) and kr.last_ped['crosswalk']['phase'] == 'creep'
    assert kr.last_ped['wins'] is True and kr.ped_emergency is False


def test_intent_latch_takes_priority():
    """서 있다가 걸어나오면 _ped_intent 래치가 서고 A-3 는 그 id 를 건드리지 않는다."""
    kr, p, ap, w = rig(x=30.0)
    observe_static(kr, ap, p=p)
    walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=2)                 # 경로 쪽으로 걷기 시작 → 래치
    assert 7 in kr.ped_intent
    assert kr._ped_crosswalk(p, ap, 5.0) is None


@pytest.mark.parametrize('lat', [LAT_MIN - 0.3, -(LAT_MAX + 0.3)])
def test_corridor_inside_or_too_far_is_not_target(lat):
    kr, p, ap, w = rig(lat=lat)
    assert step(kr, p, ap, 3.0) is None


def test_outside_crosswalk_zone_or_walking_is_not_target():
    kr, p, ap, w = rig(zones=((60.0, 70.0),))                  # 20 m 보행자, 구간 60~70 (±10 밖)
    assert step(kr, p, ap, 3.0) is None
    kr, p, ap, w = rig(ped_speed=1.0)                           # 걷는 중
    assert step(kr, p, ap, 3.0) is None


def test_stoplines_fallback_when_route_has_no_crosswalk():
    kr, p, ap, w = rig()
    kr._cw_zones = None
    kr._sl_all = [22.0]
    assert kr._crosswalk_zones(p) == [(22.0, 22.0)]
    assert step(kr, p, ap, 3.0) is not None


def test_kill_switch_default_off():
    kr, p, ap, w = rig(cfg=CFG)
    assert kr.cw_enable is False and step(kr, p, ap, 0.0) is None
    assert CFG['speed']['ped_crosswalk_creep_enable'] is False
