"""
A-4 — 걷는 채로 등장한 보행자 래치 (2026-09-03).

_ped_intent 는 정지 관찰(ped_static)을 마친 보행자만 래치했다. GT 범위 안으로 걸어
들어오거나 걷는 상태로 스폰된 보행자는 래치가 없었고, A-2(PDM 상자 0.5) 이후 PDM 검출도
3.44 → 2.44 m 로 늦어진다. 정지 관찰 없이도 연속 ped_walkin_s(0.5 s) 동안
  보행자 속도 ≥ ped_intent_v ∧ 경로 쪽 횡속도 ≥ ped_walkin_v(0.5) ∧ |lat| < ped_walkin_lat_m(8)
이면 래치. 멀어지는 방향·평행 이동·|lat| 8 밖은 미래치. 래치 후 동작·해제는 A-1 그대로.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                   # noqa: E402
from test_avoid import HZ, Ap, Planner                         # noqa: E402
from test_ped_intent import Walker, walk                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
SP = CFG['speed']
WALKIN_TICKS = int(round(SP['ped_walkin_s'] * HZ))
STEP = 0.6 / HZ                                  # 경로 쪽 횡속도 0.6 m/s (> ped_walkin_v 0.5)


def test_params_present():
    assert SP['ped_walkin_enable'] is True
    assert SP['ped_walkin_v'] == 0.5 and SP['ped_walkin_s'] == 0.5 and SP['ped_walkin_lat_m'] == 8.0
    assert SP['ped_walkin_v'] >= SP['ped_intent_v']          # 정지 관찰 경로보다 보수적


def rig(cfg=CFG, y=-6.0, x=30.0):
    """정지 관찰 없이 곧바로 걷는 보행자 (스폰 직후 / GT 진입)."""
    p = Planner()
    kr = KrRules(cfg)
    w = Walker(7, x, y)
    ap = Ap(p, [w])
    return kr, p, ap, w


def test_walking_on_arrival_latches_after_walkin_s():
    kr, p, ap, w = rig()
    walk(kr, ap, p, w, dy=+STEP, ticks=1)                     # 첫 틱: 직전 lat 없음 → 차분 불가
    assert 7 not in kr.ped_intent and 7 not in kr.ped_static
    out = walk(kr, ap, p, w, dy=+STEP, ticks=WALKIN_TICKS - 1)
    assert 7 not in kr.ped_intent and kr.ped_walkin[7] == WALKIN_TICKS - 1
    out = walk(kr, ap, p, w, dy=+STEP, ticks=1)
    assert 7 in kr.ped_intent and out is not None and out[2] == 7
    assert 7 not in kr.ped_walkin                              # 래치 후 카운터 정리


def test_walking_away_never_latches():
    kr, p, ap, w = rig()
    walk(kr, ap, p, w, dy=-STEP, ticks=WALKIN_TICKS + 5)       # |lat| 이 커진다
    assert 7 not in kr.ped_intent and kr.ped_walkin.get(7, 0) == 0


def test_walking_parallel_never_latches():
    kr, p, ap, w = rig()
    for _ in range(WALKIN_TICKS + 5):                          # 경로와 나란히 (x 방향)
        w._x -= STEP
        w.speed = STEP * HZ
        kr._update_obj_timers(ap)
        kr._ped_intent(p, ap, 12.0)
    assert 7 not in kr.ped_intent and kr.ped_walkin.get(7, 0) == 0


def test_beyond_walkin_lat_does_not_count_until_inside():
    kr, p, ap, w = rig(y=-(SP['ped_walkin_lat_m'] + 1.0))
    walk(kr, ap, p, w, dy=+STEP, ticks=WALKIN_TICKS + 2)       # 아직 8 m 밖 (0.36 m 이동)
    assert 7 not in kr.ped_intent and kr.ped_walkin.get(7, 0) == 0
    w._y = -(SP['ped_walkin_lat_m'] - 0.1)
    walk(kr, ap, p, w, dy=+STEP, ticks=WALKIN_TICKS + 1)       # 안으로 들어오면 센다
    assert 7 in kr.ped_intent


def test_one_broken_tick_restarts_count():
    kr, p, ap, w = rig()
    walk(kr, ap, p, w, dy=+STEP, ticks=WALKIN_TICKS)           # 첫 틱 제외 → WALKIN−1
    assert kr.ped_walkin[7] == WALKIN_TICKS - 1
    walk(kr, ap, p, w, dy=0.0, ticks=1)                         # 멈춤 → 처음부터
    assert kr.ped_walkin[7] == 0 and 7 not in kr.ped_intent


def test_kill_switch_keeps_static_prerequisite():
    cfg = copy.deepcopy(CFG)
    cfg['speed']['ped_walkin_enable'] = False
    kr, p, ap, w = rig(cfg)
    walk(kr, ap, p, w, dy=+STEP, ticks=WALKIN_TICKS * 4)
    assert 7 not in kr.ped_intent and not kr.ped_walkin
