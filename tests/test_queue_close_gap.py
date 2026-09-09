"""큐 뒤 정지 간격 — overtake.queue_close_gap_enable.

standoff(기본 22 m)는 **시프트 전이가 들어갈 공간**이다. 큐 뒤에서는 비켜갈
것이 아니라 줄을 서므로 그 공간이 필요 없다. 실측 2026-09-09:
18_연속교차로14 rs 78.4 standoff_d 20.6 m, run_20260909_232350 rs 485 20.5 m.

큐 판정은 새로 만들지 않고 이번 틱 플래그 `_tick_queue` 를 그대로 쓴다 —
`lane_map` 의 큐 제외와 같은 축이라야 억제 기준이 두 벌이 되지 않는다.
"""
import copy
import math
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)


def kr_at(on, d, queue, ego_speed=0.0):
    c = copy.deepcopy(CFG)
    c['overtake']['queue_close_gap_enable'] = bool(on)
    kr = KrRules(c)
    kr.wait_target_d = d
    kr._tick_queue = bool(queue)
    kr.standoff_id = 3
    kr.standoff_half_len = 2.2
    return kr, kr._standoff_profile(ego_speed)


def test_off_keeps_the_22m_baseline():
    """이전 동작 — 20.5 m 는 기준선(22) 안이라 프로파일이 0 이다."""
    kr, v = kr_at(False, 20.5, queue=True)
    assert v is not None and v <= 0.05


def test_on_moves_the_baseline_to_the_queue_gap():
    """켜면 기준선이 queue_stop_gap_m(5) 이라 20.5 m 에서는 아직 달린다."""
    kr, v = kr_at(True, 20.5, queue=True)
    expect = math.sqrt(2.0 * kr.stop_profile_a * (20.5 - kr.q_stop_gap_m))
    assert v == pytest.approx(expect, abs=1e-6)


def test_on_still_stops_at_the_queue_gap():
    """5 m 에 닿으면 선다 — 더 파고들지 않는다."""
    kr, v = kr_at(True, kr_at(True, 20.5, True)[0].q_stop_gap_m, queue=True)
    assert v is not None and v <= 0.05


def test_queue_flag_is_the_only_trigger():
    """큐가 아니면 켜져 있어도 기준선은 그대로다 (추월 대상은 공간이 필요하다)."""
    on = kr_at(True, 20.5, queue=False)[1]
    off = kr_at(False, 20.5, queue=False)[1]
    assert on == off


def test_switch_default_is_off():
    assert CFG['overtake'].get('queue_close_gap_enable') is False
    assert CFG['overtake'].get('queue_stop_gap_m') == 5.0
