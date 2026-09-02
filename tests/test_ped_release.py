"""
보행자 의도 래치 **위치 기반 해제** (A-1).

실측 근거 (2026-09-02, logs/batch/20260902_212135/실전주행_교통류_02_좌회전8):
  · id4 가 t≈97.4 에 래치돼 자차가 t=103.5 에 섰다. 보행자는 횡단을 마치고
    t≈109 에 |lat| 8.8 m 길가에 멈췄는데, 래치 해제가 '지나감(뒤로 감)·관측
    끊김' 뿐이라 둘 다 아닌 이 보행자는 로그 끝(t=222)까지 래치를 잡았다 —
    118 s 정지.

여기서 지키는 불변:
  · 해제(clear)는 회랑 밖 ∧ 경로 쪽으로 오지 않음 ∧ (멈춤 ∨ 차도 밖) 이
    ped_release_s **연속**일 때만. 도로 위를 걷는 동안은 유지한다.
  · 되돌아오면(v_toward > 0) 카운터는 처음부터.
  · 투영 불가 틱은 clear 카운터만 리셋, 래치·hold 는 유지.
  · 관측 끊김·지나감 해제는 그대로이고 카운터도 같이 정리된다.
  · backstop 은 회랑 밖에서만 — 경로 위에 서 있는 보행자는 절대 안 푼다.
  · ped_release_lat_m = 0 이면 완전 비활성 = 이전 동작.
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
STOP_V = SP['ped_stop_v']
OFF_LAT = SP['ped_offroad_lat_m']
BACK_TICKS = int(round(SP['ped_backstop_s'] * HZ))
DY = 1.0 / HZ                                   # 보행 1.0 m/s 의 틱당 이동


def latched(cfg=CFG, y0=-5.0):
    """정지 관찰 → 걸어나옴 → 래치까지 만든 (kr, p, ap, w)."""
    kr, p = KrRules(cfg), Planner()
    w = Walker(4, 20.0, y0)
    ap = Ap(p, actors=[w])
    observe_static(kr, ap, p=p)
    assert walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1) is not None
    assert 4 in kr.ped_intent
    return kr, p, ap, w


def step(kr, ap, p, w, y, speed, v_ego=0.0):
    """보행자를 y 로 놓고(속도 speed) 한 틱."""
    w._y = float(y)
    w.speed = float(speed)
    kr._update_obj_timers(ap)
    return kr._ped_intent(p, ap, v_ego)


def cross_to(kr, ap, p, w, y_end, speed=1.0):
    """보행자를 경로 건너 y_end 까지 걸린다 (틱마다 유지 확인). 최종 y 를 준다."""
    y = w._y
    while y < y_end:
        y += DY
        assert step(kr, ap, p, w, y, speed) is not None, f'y={y:.2f} 에서 조기 해제'
    return y


def test_params_present():
    """(f) params 키가 있고 값이 스펙과 같다."""
    assert REL_LAT == 2.5 and SP['ped_release_s'] == 1.5
    assert STOP_V == 0.2 and OFF_LAT == 12.5 and SP['ped_backstop_s'] == 30.0


# ── 해제 (clear) ─────────────────────────────────────────────────────────
def test_crossed_and_stopped_offroad_releases_after_release_s():
    """실측 재현: 횡단을 마치고 길가(|lat| 8.8)에 멈추면 ped_release_s 뒤 해제."""
    kr, p, ap, w = latched()
    y = cross_to(kr, ap, p, w, 8.8)                           # 도로 위 — 유지
    assert kr.ped_clear[4] == 0
    got = None
    for i in range(REL_TICKS):
        got = step(kr, ap, p, w, y, 0.0)
        if i < REL_TICKS - 1:
            assert got is not None, f'{i}번째 틱에 조기 해제'
    assert got is None                                        # REL_TICKS 째 해제
    assert 4 not in kr.ped_intent
    assert kr.ped_released == {4: 'clear'}
    assert kr.ped_diag[4]['clear_s'] == pytest.approx(REL_TICKS / HZ, abs=0.11)


def test_walking_on_road_keeps_latch_until_offroad():
    """도로 위를 걷는 동안(|lat| 2.5~6, 속도 1.0)은 해제하지 않는다 — 되돌아올 수
    있다. 차도 밖(> ped_offroad_lat_m)으로 나가면 걷고 있어도 해제."""
    kr, p, ap, w = latched()
    y = w._y
    while y + DY <= OFF_LAT:
        y += DY
        assert step(kr, ap, p, w, y, 1.0) is not None
        assert kr.ped_clear[4] == 0
    n = 0
    while True:                                               # 차도 밖에서 계속 걷는다
        y += DY
        n += 1
        if step(kr, ap, p, w, y, 1.0) is None:
            break
        assert n <= REL_TICKS
    assert n == REL_TICKS
    assert kr.ped_released == {4: 'clear'}


def test_stopped_on_road_outside_corridor_releases():
    """회랑 밖(|lat| 3)이지만 도로 위 — 멈춰 서면 해제된다 (속도 < ped_stop_v)."""
    kr, p, ap, w = latched()
    y = cross_to(kr, ap, p, w, 3.0)
    for _ in range(REL_TICKS - 1):
        assert step(kr, ap, p, w, y, STOP_V * 0.5) is not None
    assert step(kr, ap, p, w, y, STOP_V * 0.5) is None
    assert kr.ped_released == {4: 'clear'}


def test_inside_corridor_never_releases():
    """회랑 안(|lat| ≤ ped_release_lat_m)에 멈춰 서면 절대 안 푼다."""
    kr, p, ap, w = latched(y0=-3.0)
    for _ in range(BACK_TICKS + 20):
        assert step(kr, ap, p, w, -1.0, 0.0) is not None
    assert 4 in kr.ped_intent and kr.ped_released == {}


def test_returning_pedestrian_resets_counter():
    """해제 조건이 한 틱이라도 깨지면(v_toward > 0) 카운터는 처음부터."""
    kr, p, ap, w = latched()
    y = cross_to(kr, ap, p, w, 4.0)
    for _ in range(REL_TICKS - 1):
        step(kr, ap, p, w, y, 0.0)
    assert kr.ped_clear[4] == REL_TICKS - 1
    y -= 0.1
    assert step(kr, ap, p, w, y, 1.0) is not None            # 경로 쪽으로 한 걸음
    assert kr.ped_clear[4] == 0
    for _ in range(REL_TICKS - 1):
        assert step(kr, ap, p, w, y, 0.0) is not None        # 다시 처음부터
    assert step(kr, ap, p, w, y, 0.0) is None


def test_v_toward_is_measured_every_tick():
    """(a) v_toward 는 래치 후에도 매 틱 진단에 실린다 (멀어짐 = 음수)."""
    kr, p, ap, w = latched()
    step(kr, ap, p, w, w._y + DY, 1.0)                        # 경로 쪽 (+)
    assert kr.ped_diag[4]['v_toward'] == pytest.approx(+1.0, abs=0.05)
    step(kr, ap, p, w, w._y - DY, 1.0)                        # 멀어짐 (−)
    d = kr.ped_diag[4]
    assert d['v_toward'] == pytest.approx(-1.0, abs=0.05)
    assert set(d) >= {'lat', 'v_toward', 'clear_s', 'hold_s'}


# ── 투영 불가 / 끊김 ───────────────────────────────────────────────────────
def test_unprojectable_tick_resets_clear_keeps_hold(monkeypatch):
    """(c) _project None 틱: clear 카운터 리셋, 래치·hold 는 유지."""
    kr, p, ap, w = latched()
    y = cross_to(kr, ap, p, w, OFF_LAT - 0.2)
    for _ in range(REL_TICKS - 1):
        step(kr, ap, p, w, y, 0.0)
    hold = kr.ped_hold[4]
    assert kr.ped_clear[4] == REL_TICKS - 1
    monkeypatch.setattr(kr, '_project', lambda *_a: None)
    assert step(kr, ap, p, w, y, 0.0) is None                # 후보는 없지만
    assert 4 in kr.ped_intent                                  # 래치 유지
    assert kr.ped_hold[4] == hold                              # hold 유지
    assert 4 not in kr.ped_clear                               # clear 리셋
    assert kr.ped_released == {}
    monkeypatch.undo()
    # 투영이 돌아오면 다시 REL_TICKS 를 채워야 한다 (첫 틱은 prev 가 없어 v_toward None)
    for _ in range(REL_TICKS):
        assert step(kr, ap, p, w, y, 0.0) is not None
    assert step(kr, ap, p, w, y, 0.0) is None


def test_lost_observation_pops_counters():
    """(d) 관측이 끊기면 기존대로 즉시 해제 + 카운터 pop."""
    kr, p, ap, w = latched()
    step(kr, ap, p, w, w._y + DY, 1.0)
    assert 4 in kr.ped_hold
    ap._world._a.remove(w)
    kr._update_obj_timers(ap)
    assert kr._ped_intent(p, ap, 0.0) is None
    assert 4 not in kr.ped_intent and 4 not in kr.ped_hold and 4 not in kr.ped_clear


def test_passed_behind_pops_counters():
    """(d) 지나감(뒤로 감) 해제도 그대로이고 카운터가 남지 않는다."""
    kr, p, ap, w = latched()
    w._x = -3.0
    assert step(kr, ap, p, w, w._y, 1.0) is None
    assert 4 not in kr.ped_intent and 4 not in kr.ped_hold and 4 not in kr.ped_clear


# ── backstop ──────────────────────────────────────────────────────────────
def test_backstop_releases_pacing_pedestrian_outside_corridor():
    """회랑 밖 도로 위에서 서성이며(v_toward 부호가 계속 바뀜) clear 가 못 차는
    보행자 — ped_backstop_s 뒤 조건 무관 해제."""
    kr, p, ap, w = latched()
    y = cross_to(kr, ap, p, w, 4.0)
    n = 0
    while True:
        n += 1
        got = step(kr, ap, p, w, y + (0.3 if n % 2 else 0.0), 1.0)   # 매 틱 왕복
        if got is None:
            break
        assert kr.ped_clear[4] == 0
        assert n <= BACK_TICKS + 2
    assert kr.ped_released == {4: 'backstop'}
    assert kr.ped_diag[4]['hold_s'] == pytest.approx(BACK_TICKS / HZ, abs=0.11)


def test_backstop_does_not_fire_inside_corridor():
    """회랑 안이면 backstop 시간이 지나도 유지 — 나가는 순간에만 푼다."""
    kr, p, ap, w = latched(y0=-3.0)
    for _ in range(BACK_TICKS + 5):
        assert step(kr, ap, p, w, -1.5, 0.0) is not None
    assert kr.ped_hold[4] >= BACK_TICKS
    # 회랑 밖으로 나가면(경로에서 멀어짐) 즉시 backstop
    assert step(kr, ap, p, w, -(REL_LAT + 0.1), 1.0) is None
    assert kr.ped_released == {4: 'backstop'}


# ── 스위치 ────────────────────────────────────────────────────────────────
def test_switch_off_restores_previous_behavior():
    """ped_release_lat_m = 0 → 위치 기반 해제 없음 = 이전 동작 (끝까지 래치)."""
    cfg = copy.deepcopy(CFG)
    cfg['speed']['ped_release_lat_m'] = 0.0
    kr, p, ap, w = latched(cfg)
    y = cross_to(kr, ap, p, w, 8.8)
    for _ in range(BACK_TICKS + 20):
        assert step(kr, ap, p, w, y, 0.0) is not None
    assert 4 in kr.ped_intent and kr.ped_released == {}


# ── reasons.ped 진단 (apply 경유) ───────────────────────────────────────────
def test_reasons_ped_carries_diag_and_release():
    """(e) reasons.ped 에 lat / v_toward / clear_s / hold_s 가 실리고,
    해제 틱에는 후보 없이 release 사유가 남는다."""
    p = Planner()
    w = Walker(4, 20.0, -5.0)
    ap = make_ap(p, [w])
    kr = ap.kr_rules
    observe_static(kr, ap, p=p)
    walk(kr, ap, p, w, dy=+0.5 / HZ, ticks=1)
    run_apply(ap, p, 0.0)
    assert 4 in kr.ped_intent
    assert set(kr.last_ped) >= {'id', 'v_allow', 'a_req', 'wins',
                                'lat', 'v_toward', 'clear_s', 'hold_s'}
    y = w._y
    last = None
    for _ in range(600):
        if y < 8.8:
            y += DY
            w._y, w.speed = y, 1.0
        else:
            w.speed = 0.0
        kr._update_obj_timers(ap)
        run_apply(ap, p, 0.0)
        if kr.last_ped and kr.last_ped.get('release'):
            last = kr.last_ped
            break
    assert last is not None
    assert last['release'] == 'clear' and last['wins'] is False
    assert last['lat'] == pytest.approx(8.8, abs=0.06)        # 길가에 멈춘 뒤 해제
    assert last['clear_s'] == pytest.approx(REL_TICKS / HZ, abs=0.11)
    assert 'v_allow' not in last
    # 다음 틱부터 후보도 진단도 없다 (PDM 예측에 맡긴다)
    kr._update_obj_timers(ap)
    run_apply(ap, p, 0.0)
    assert kr.last_ped is None
