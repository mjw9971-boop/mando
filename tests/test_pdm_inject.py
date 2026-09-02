"""
params.pdm.* → PDM GlobalConfig 주입 (B-6). config.py 원문은 무수정이고
run_agent.build_pdm_config 가 idm_red_light_minimum_distance 와 같은 방식으로
덮어쓴다. 키가 없거나 원문값이면 동작 불변.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'team_code'))

from config import GlobalConfig                 # noqa: E402
from run_agent import build_pdm_config          # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)


def test_default_equals_original():
    """기본 params(0.25)는 원문값과 같다 — 주입해도 동작 불변."""
    assert CFG['pdm']['idm_leading_vehicle_time_headway'] == pytest.approx(
        GlobalConfig().idm_leading_vehicle_time_headway)
    gc = build_pdm_config(CFG)
    assert gc.idm_leading_vehicle_time_headway == pytest.approx(
        GlobalConfig().idm_leading_vehicle_time_headway)


@pytest.mark.parametrize('t', [0.5, 1.0])
def test_override_is_injected(t):
    cfg = copy.deepcopy(CFG)
    cfg['pdm']['idm_leading_vehicle_time_headway'] = t
    assert build_pdm_config(cfg).idm_leading_vehicle_time_headway == pytest.approx(t)


def test_missing_section_keeps_original():
    cfg = copy.deepcopy(CFG)
    cfg.pop('pdm')
    assert build_pdm_config(cfg).idm_leading_vehicle_time_headway == pytest.approx(
        GlobalConfig().idm_leading_vehicle_time_headway)


def test_red_light_injection_unchanged():
    """기존 주입(정지선 gap)은 그대로다."""
    vh = CFG['vehicle']
    front = vh['wheelbase'] + vh['front_overhang_m']
    assert build_pdm_config(CFG).idm_red_light_minimum_distance == pytest.approx(
        CFG['speed']['stop_gap_stopline_m'] + front)
