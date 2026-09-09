"""
ctrl24 커밋 5 — run_agent 접점 · 종점 패드(D) · 로그 스키마.

여기서 지키는 불변:
  · ctrl24.enable=false 면 Runner 는 kr_rules + LoggingAutoPilot 을 조립한다 (won 동일).
  · enable=true 면 Ctrl24 + Ctrl24AutoPilot (PDM 보행자 후보 없음).
  · 종점 패드: 경로 배열 끝에 end_pad_m 직선. off 면 패드 0 = 이전 배열 그대로.
  · 로그 reasons: bicycle/pedestrian/route_end/red_zone 없음, kr 슬롯 있음, winner 어휘 유지.
"""
import argparse
import copy
import json
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.route import VtdRoutePlanner

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'team_code'))

import run_agent                                               # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route.pkl'
BATCH = ROOT / 'logs' / 'batch' / '20260906_020738'
LOG = BATCH / '실전주행_교통류_12_우회전11.jsonl'
LOG_ROUTE = BATCH / 'routes' / 'route_실전주행_교통류_12_우회전11.pkl'

needs_data = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='data/lane_graph.pkl · route.pkl 필요')
needs_log = pytest.mark.skipif(not (LOG.exists() and LOG_ROUTE.exists() and GRAPH.exists()),
                               reason='배치 로그 필요')


def cfg_off():
    c = copy.deepcopy(CFG)
    c['ctrl24']['enable'] = False
    return c


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH), cfg=CFG)


@needs_data
def test_end_pad_extends_straight_along_last_heading(lg):
    route = run_agent.load_route(str(ROUTE))
    pad_m = float(CFG['ctrl24']['end_pad_m'])
    p_on = VtdRoutePlanner(lg, route, CFG, config=run_agent.build_pdm_config(CFG))
    p_off = VtdRoutePlanner(lg, route, cfg_off(), config=run_agent.build_pdm_config(CFG))
    n_pad = int(round(pad_m * p_on.points_per_meter))
    assert p_on.end_pad_pts == n_pad and p_off.end_pad_pts == 0
    assert len(p_on.route_points) == len(p_off.route_points) + n_pad
    assert np.allclose(p_on.route_points[:len(p_off.route_points)], p_off.route_points)
    pad = p_on.route_points[-n_pad:, :2]
    d = np.diff(pad, axis=0)
    hdg = np.arctan2(d[:, 1], d[:, 0])
    assert np.ptp(hdg) < 1e-6                                  # 직선
    assert np.linalg.norm(pad[-1] - pad[0]) == pytest.approx(pad_m - 0.1, abs=0.05)
    assert p_on.route_s[-1] - p_on.route_s[-n_pad - 1] == pytest.approx(pad_m, abs=0.01)
    assert len(p_on.route_waypoints) == len(p_on.route_points)
    assert len(p_on.speed_limits) == len(p_on.route_points)
    assert p_on.next_traffic_lights[-1] is None


@needs_data
def test_ctrl24_forces_current_lane_target_and_local_search(lg):
    route = run_agent.load_route(str(ROUTE))
    p = VtdRoutePlanner(lg, route, CFG, config=run_agent.build_pdm_config(CFG))
    assert p.shift_target_current is True and p.span_search_local is True
    assert p.shift_target_max_steps == CFG['ctrl24']['shift_target_max_steps']


def _args(cfg_path, log_out, max_ticks):
    return argparse.Namespace(host=None, port=None, graph=str(GRAPH), route=str(LOG_ROUTE),
                              csv=None, route_out=None, allow_route_warnings=False,
                              config=str(cfg_path), replay=str(LOG), log=str(log_out),
                              max_ticks=max_ticks)


@needs_log
def test_runner_selects_controller_by_switch(tmp_path):
    import yaml
    off = tmp_path / 'off.yaml'
    yaml.safe_dump(cfg_off(), open(off, 'w'), allow_unicode=True, sort_keys=False)
    r_off = run_agent.Runner(_args(off, tmp_path / 'off.jsonl', 5))
    assert r_off.ctrl24 is False and type(r_off.kr).__name__ == 'KrRules'
    assert type(r_off.agent).__name__ == 'LoggingAutoPilot'
    r_off.run()
    r_on = run_agent.Runner(_args(PARAMS_YAML, tmp_path / 'on.jsonl', 60))
    assert r_on.ctrl24 is True and type(r_on.kr).__name__ == 'Ctrl24'
    assert type(r_on.agent).__name__ == 'Ctrl24AutoPilot'
    r_on.run()
    ticks = [json.loads(l) for l in open(tmp_path / 'on.jsonl', encoding='utf-8') if '"raw"' in l]
    assert len(ticks) == 60
    r = ticks[-1]['decision']['reasons']
    # route_end 는 logger.SPEED_CANDIDATES 의 옛 이름이라 null 로 깔린다 (값 없음이 계약)
    assert 'bicycle' not in r and 'pedestrian' not in r and 'red_zone' not in r
    assert r['route_end'] is None
    assert set(r['kr']) == {'stop_profile', 'stop_hold', 'rtor_cap', 'ped_intent',
                            'crosswalk', 'red_zone', 'shift_cap',
                            'span_v_req'}
    assert r['winner'] in ('none', 'lead', 'vehicle', 'light', 'walker', 'rtor', 'red_zone')
    assert 'red_zone_detail' in r
    assert 'prepass_ms' in r and 'signal' in r
    assert ticks[-1]['decision']['turn_signal'] in (0, 1, 2)
