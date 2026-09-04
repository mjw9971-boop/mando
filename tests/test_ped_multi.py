"""
P4-M — 다중 보행자 회랑 홀드 (2026-09-04, logs/run_20260903_101405).

실측: id4·id5 가 12 m 앞 회랑(|lat| < 2.5)을 횡단하는 동안(k205~266) 의도 후보는
정지 프로파일 √(2·a·(s_rel − 앞범퍼 − s0)) ≈ 5.1 m/s 였고 적신호 IDM 후보(1.2 → 2.3)
가 더 낮아 그쪽이 winner 가 되며 정지(0.01 m/s)에서 재가속했다 (k238~266, accel
+0.7 → +0.95). 판정은 보행자 전부를 보지만(최저 v_allow) 회랑 안이라도 0 이 아니었다.

여기서 지키는 불변 (ped_multi_enable):
  · 회랑 안 보행자가 한 명이라도 있으면 후보는 0 — 몇 명이든, 어느 id 가 best 든.
  · 홀드 래치 on 이면 v_allow 는 0 뿐이다. 해제(A-1 규칙·지나감·coast 만료)가 먼저,
    그 다음 틱부터 후보가 사라진다.
  · 드롭아웃 ≤ ped_hold_coast_s 는 이탈이 아니다 (직전 기여 유지). 넘기면 해제.
  · 회랑 밖 보행자(인도 정지)는 홀드하지 않는다 — 101248 재현.
  · false 면 이전 동작과 동일 (홀드 집합이 비고 후보는 프로파일 그대로).
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
from test_avoid import Ap, Planner                             # noqa: E402
from test_ped_intent import (HZ, Walker, make_ap, observe_static,  # noqa: E402
                             run_apply, walk)

CFG = load_params_yaml(PARAMS_YAML)
SP = CFG['speed']
REL_LAT = SP['ped_release_lat_m']
REL_TICKS = int(round(SP['ped_release_s'] * HZ))
COAST_TICKS = int(round(SP['ped_hold_coast_s'] * HZ))
S0_PED = 4.0                                    # GlobalConfig.idm_pedestrian_minimum_distance
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
A_STOP = SP['stop_profile_a']
DY = 1.0 / HZ                                   # 보행 1.0 m/s 의 틱당 이동


def on_cfg(cfg=CFG):
    c = copy.deepcopy(cfg)
    c['speed']['ped_multi_enable'] = True
    return c


def profile(s_rel):
    return (2.0 * A_STOP * max(0.0, s_rel - FRONT - S0_PED)) ** 0.5


def step(kr, ap, p, w, y, speed=1.0, v_ego=0.0):
    w._y = float(y)
    w.speed = float(speed)
    kr._update_obj_timers(ap)
    return kr._ped_intent(p, ap, v_ego)


def rig(cfg, walkers):
    p = Planner()
    kr = KrRules(cfg)
    ap = Ap(p, actors=list(walkers))
    return kr, p, ap


class Vanishing(Walker):
    """드롭아웃 모사 — hidden 이면 액터 목록에서 빠진다."""


class World2:
    def __init__(self, actors):
        self._a = actors

    def get_actors(self):
        from test_avoid import ActorList
        return ActorList(a for a in self._a if not getattr(a, 'hidden', False))


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_params_present_and_default_off():
    assert SP['ped_multi_enable'] is False
    assert SP['ped_hold_coast_s'] == 0.5
    kr = KrRules(CFG)
    assert kr.ped_multi is False and kr.ped_coast_ticks == COAST_TICKS


def test_off_never_populates_hold_and_keeps_profile():
    """스위치 off: 회랑 안 보행자라도 홀드 집합은 비고 후보는 프로파일 그대로."""
    kr, p, ap = rig(CFG, [Walker(4, 12.0, -5.0)])
    w = ap._world.get_actors()[0]
    observe_static(kr, ap, p=p)
    out = walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1)
    assert out is not None and 4 in kr.ped_intent
    out = step(kr, ap, p, w, 0.0)                         # 경로 한복판
    assert not kr.ped_hold_ids
    assert out[0] == pytest.approx(profile(12.0), abs=1e-6) and out[0] > 0.0
    assert kr.ped_all == {}


# ── 홀드: 회랑 안 = 0 ─────────────────────────────────────────────────────
def test_pedestrian_inside_corridor_forces_zero():
    """회랑 진입 틱부터 v_allow = 0. 회랑 밖(접근 중)은 이전 프로파일."""
    kr, p, ap = rig(on_cfg(), [Walker(4, 12.0, -5.0)])
    w = ap._world.get_actors()[0]
    observe_static(kr, ap, p=p)
    out = walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1)      # 래치, 아직 회랑 밖
    assert out is not None and out[0] == pytest.approx(profile(12.0), abs=1e-6)
    assert 4 not in kr.ped_hold_ids
    out = step(kr, ap, p, w, -(REL_LAT - 0.1))            # 회랑 진입
    assert 4 in kr.ped_hold_ids
    assert out == (0.0, pytest.approx(0.0), 4)
    assert kr.ped_all[4]['hold'] is True and kr.ped_all[4]['v_allow'] == 0.0


def test_unlatched_pedestrian_standing_in_corridor_is_held():
    """의도 래치가 없어도(정지, 걸어나오지 않음) 회랑 안이면 홀드한다."""
    kr, p, ap = rig(on_cfg(), [Walker(9, 20.0, 0.5)])
    w = ap._world.get_actors()[0]
    out = None
    for _ in range(3):
        out = step(kr, ap, p, w, 0.5, speed=0.0)
    assert 9 in kr.ped_hold_ids and 9 not in kr.ped_intent
    assert out == (0.0, pytest.approx(0.0), 9)


def test_second_pedestrian_in_corridor_wins_regardless_of_best_id():
    """101405 재현: id4 는 회랑을 나갔고(프로파일 5.1) id5 가 회랑 안 — 후보는 0."""
    kr, p, ap = rig(on_cfg(), [Walker(4, 12.0, -5.0), Walker(5, 12.2, -6.0)])
    w4, w5 = ap._world.get_actors()
    observe_static(kr, ap, p=p)
    for w in (w4, w5):
        w._y += 0.5 / HZ
        w.speed = 0.5
    kr._update_obj_timers(ap)
    kr._ped_intent(p, ap, 12.0)
    assert {4, 5} <= kr.ped_intent
    # id4 가 먼저 회랑을 지나(0.0) 좌측 3 m 로 나감, id5 는 그 뒤 회랑 한복판
    w4._y, w4.speed = 0.0, 1.0
    w5._y, w5.speed = -3.0, 1.0
    kr._update_obj_timers(ap)
    out = kr._ped_intent(p, ap, 0.0)
    assert out[0] == 0.0 and out[2] == 4 and kr.ped_hold_ids == {4}
    w4._y, w4.speed = +3.0, 1.0
    w5._y, w5.speed = -0.5, 1.0
    kr._update_obj_timers(ap)
    out = kr._ped_intent(p, ap, 0.0)
    assert out[0] == 0.0 and out[2] == 4 and kr.ped_hold_ids == {4, 5}
    assert kr.ped_all[4]['hold'] is True and kr.ped_all[4]['v_allow'] == 0.0   # 회랑을 지났어도 해제 전엔 홀드
    assert kr.ped_all[5]['hold'] is True and kr.ped_all[5]['v_allow'] == 0.0
    # 반대 순서(id5 가 리스트 앞)라도 결과는 같다
    ap._world._a.reverse()
    kr._update_obj_timers(ap)
    out = kr._ped_intent(p, ap, 0.0)
    assert out[0] == 0.0


def test_hold_never_yields_positive_v_allow_before_release():
    """홀드 래치가 살아 있는 동안 어느 틱도 v_allow > 0 이 없다. 해제 후에야 후보가 사라진다."""
    kr, p, ap = rig(on_cfg(), [Walker(4, 12.0, -5.0)])
    w = ap._world.get_actors()[0]
    observe_static(kr, ap, p=p)
    walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1)
    y = -REL_LAT + 0.1
    released_at = None
    for i in range(2000):
        out = step(kr, ap, p, w, y, speed=1.0)
        if 4 in kr.ped_hold_ids:
            assert out is not None and out[0] == 0.0, f'tick {i}: 홀드 중 v_allow {out}'
        elif kr.ped_released.get(4):
            released_at = i
            assert out is None                     # 해제 틱 = 후보 없음 (순서 보장)
            break
        if y < SP['ped_offroad_lat_m'] + 1.0:
            y += DY
    assert released_at is not None and kr.ped_released[4] == 'clear'
    # 해제 이후 틱: 후보 없음, 진단에도 홀드 없음
    out = step(kr, ap, p, w, y, speed=1.0)
    assert out is None and not kr.ped_hold_ids


# ── coast ────────────────────────────────────────────────────────────────
def _held_rig():
    cfg = on_cfg()
    p = Planner()
    kr = KrRules(cfg)
    w = Vanishing(4, 12.0, 0.3)
    ap = Ap(p, actors=[w])
    ap._world = World2([w])
    for _ in range(3):
        step(kr, ap, p, w, 0.3, speed=0.0)
    assert 4 in kr.ped_hold_ids
    return kr, p, ap, w


def test_dropout_within_coast_keeps_hold_at_zero():
    kr, p, ap, w = _held_rig()
    w.hidden = True
    for i in range(COAST_TICKS):
        kr._update_obj_timers(ap)
        out = kr._ped_intent(p, ap, 0.0)
        assert 4 in kr.ped_hold_ids and out == (0.0, pytest.approx(0.0), 4), f'coast tick {i}'
        assert kr.ped_all[4]['coast'] == i + 1
    w.hidden = False                                       # 재관측 → 카운터 0
    out = step(kr, ap, p, w, 0.3, speed=0.0)
    assert out[0] == 0.0 and kr.ped_miss[4] == 0


def test_dropout_beyond_coast_releases():
    kr, p, ap, w = _held_rig()
    w.hidden = True
    out = None
    for _ in range(COAST_TICKS + 1):
        kr._update_obj_timers(ap)
        out = kr._ped_intent(p, ap, 0.0)
    assert out is None and 4 not in kr.ped_hold_ids and 4 not in kr.ped_miss


def test_off_dropout_releases_immediately():
    """스위치 off: 이전 동작 — 첫 미관측 틱에 즉시 해제."""
    kr, p = KrRules(CFG), Planner()
    w = Vanishing(4, 12.0, -5.0)
    ap = Ap(p, actors=[w])
    ap._world = World2([w])
    observe_static(kr, ap, p=p)
    walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1)
    assert 4 in kr.ped_intent
    w.hidden = True
    kr._update_obj_timers(ap)
    assert kr._ped_intent(p, ap, 0.0) is None and 4 not in kr.ped_intent


# ── 회귀: 인도 정지 보행자 ────────────────────────────────────────────────
@pytest.mark.parametrize('lat', [-8.75, -8.16, 8.5])
def test_sidewalk_standing_pedestrian_is_not_held(lat):
    """101248 재현: 회랑 밖 8.2~8.8 m 에 서 있는 보행자 — 홀드도 후보도 없다."""
    kr, p, ap = rig(on_cfg(), [Walker(4, 40.0, lat), Walker(5, 45.0, lat + 0.5)])
    out = None
    for _ in range(60):
        kr._update_obj_timers(ap)
        out = kr._ped_intent(p, ap, 10.0)
    assert out is None and not kr.ped_hold_ids and not kr.ped_intent
    assert all(v['hold'] is False and v['latched'] is False for v in kr.ped_all.values())


# ── apply 통합: min() 후보로서 최종 목표를 0 으로 ─────────────────────────
def test_apply_yields_zero_target_and_no_positive_accel():
    p = Planner()
    w = Walker(4, 12.0, -5.0)
    ap = make_ap(p, [w], cfg=on_cfg())
    kr = ap.kr_rules
    observe_static(kr, ap, p=p)
    walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1)
    w._y, w.speed = 0.0, 1.0
    for _ in range(5):
        kr._update_obj_timers(ap)
        control, target = run_apply(ap, p, 12.5)
        assert target == 0.0 and control.accel <= 0.0
        assert kr.last_ped['wins'] is True and kr.last_ped['v_allow'] == 0.0
        assert kr.last_ped['hold'] == [4] and kr.last_ped['all'][4]['hold'] is True


def test_reasons_ped_lists_every_evaluated_pedestrian():
    p = Planner()
    w4, w5, w9 = Walker(4, 12.0, -5.0), Walker(5, 12.2, -6.0), Walker(9, 30.0, 8.5)
    ap = make_ap(p, [w4, w5, w9], cfg=on_cfg())
    kr = ap.kr_rules
    observe_static(kr, ap, p=p)
    w4._y, w4.speed = 0.0, 1.0
    w5._y, w5.speed = -5.5, 0.5
    kr._update_obj_timers(ap)
    run_apply(ap, p, 12.5)
    allp = kr.last_ped['all']
    assert set(allp) == {4, 5, 9}
    assert allp[4]['hold'] is True and allp[4]['v_allow'] == 0.0
    assert allp[5]['hold'] is False and allp[5]['latched'] is True and allp[5]['v_allow'] > 0.0
    assert allp[9] == {'s_rel': 30.0, 'lat': 8.5, 'latched': False, 'hold': False}
    assert {'s_rel', 'lat'} <= set(allp[4]) and {'s_rel', 'lat'} <= set(allp[5])
