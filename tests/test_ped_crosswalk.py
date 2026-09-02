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

