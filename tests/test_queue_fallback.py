"""
(5) 큐 차로로의 폴백 금지 — `avoid_map.lane_map_fallback_no_queue_enable`.

계획한 쪽이 게이트에서 떨어지면 반대쪽이 최후 수단인데, 그 반대쪽이 신호
대기열이면 **비어 있는 차로에서 나와 막힌 차로로 들어간다**.

실측 2026-09-10 run_20260910_144656 t 65.3~66.5, 자차 (2533,0,5) free_run **80.0**:
    plan   pick (2533,0,3) side left hops 2    ← 왼쪽도 free 80.0
    기각   ['left:occupied_mid@p1']            ← 중간 차로 4 가 id 9 로 막힘
    폴백   side_pick picked 'right' → lane 6   ← queue_lanes 에 있다
    결과   t 70.6 자차가 lane 6 (free 21.0, id 12 대기열) 안에 있다

로그에 `fb_side` 키는 **없다** — `_lm_hops`/`_lm_fb_hops`(40c8ffe 계열) 경로가
아니라 `_side_pass` 의 side 폴백 순서(`_order`)다.
"""
import copy
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

EGO = (2533, 0, 5)
L4, L3, L6 = (2533, 0, 4), (2533, 0, 3), (2533, 0, 6)
ORDER = [L3, L4, EGO, L6]


class Lg:
    def __init__(self):
        s = np.linspace(0.0, 1000.0, 51)
        self.lanes = {k: {'junction': -1, 'dir': -1, 's': s,
                          'width': np.full_like(s, 3.0), 'length': 1000.0}
                      for k in ORDER}

    def neighbor(self, key, side):
        if key not in ORDER:
            return None
        i = ORDER.index(key) + (1 if side == 'right' else -1)
        return ORDER[i] if 0 <= i < len(ORDER) else None


class P:
    pass


def kr(**over):
    c = copy.deepcopy(CFG)
    c['avoid_map'].update(over)
    return KrRules(c)


# ── (5) ──────────────────────────────────────────────────────────────────
def test_fallback_no_queue_default_is_off():
    assert CFG['avoid_map']['lane_map_fallback_no_queue_enable'] is False


def test_side_lane_helper_reads_the_map():
    k = kr()
    p = P(); p.lg = Lg()
    k._tick_ego_lane = EGO
    assert k._lm_side_lane(p, 'left') == L4
    assert k._lm_side_lane(p, 'right') == L6
    k._tick_ego_lane = None
    assert k._lm_side_lane(p, 'right') is None


def test_right_is_the_queue_lane_in_the_log_case():
    """전제 — 폴백이 향한 쪽이 실제로 queue_lanes 에 있었다."""
    k = kr(lane_map_fallback_no_queue_enable=True)
    k.last_lane_map = {'queue_lanes': [str(list(L6))]}
    p = P(); p.lg = Lg()
    k._tick_ego_lane = EGO
    q = set(k.last_lane_map['queue_lanes'])
    assert str(list(k._lm_side_lane(p, 'right'))) in q
    assert str(list(k._lm_side_lane(p, 'left'))) not in q


def test_planned_side_is_never_dropped():
    """계획한 쪽은 남긴다 — 지도가 고른 목표는 이미 큐가 아니다."""
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    i = src.index('if self.lm_fallback_no_queue and len(_order) > 1:')
    blk = src[i:i + 700]
    assert '_keep = [_order[0]]' in blk
    assert 'for _sd in _order[1:]:' in blk


def test_lane_plan_already_excludes_queue_from_candidates():
    """폴백만 막으면 되는 이유 — 후보 산정은 이미 큐를 뺀다."""
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    assert 'if k in qlanes:' in src
    assert '# 신호 대기 줄은 후보가 아니다' in src


