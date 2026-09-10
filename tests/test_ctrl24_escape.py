"""
ctrl24 K9 — 탈출 바닥 (무한정지 방지).

규칙은 하나다:
  경로 진행이 escape_stuck_s 동안 escape_progress_m 미만이고 정당한 정지 원인이
  없으면, 전방 간격이 escape_clear_m 을 넘는 한 정당하지 않은 모든 min() 후보에
  바닥 escape_v 를 깔고 escape_release_m 진행할 때까지 유지한다.

여기서 지키는 불변:
  · 정당한 정지 원인(적신호 정지선·정지선 홀드·보행자·PDM 적신호)은 **바닥을 안 받는다.**
    그중 하나가 escape_v 보다 낮으면 그쪽이 이긴다 — 규칙이 신호·보행자를 못 뚫는다.
  · 정당한 원인이 살아 있으면 고착 판정 자체가 서지 않는다.
  · 전방 범퍼 간격이 escape_clear_m 이하면 바닥은 0 이다 (원래 후보 그대로).
  · 걸린 뒤 escape_release_m 진행할 때까지 유지한다 (진동 방지 래치).
  · 기본 off — off 면 이전 동작과 같다.
"""
import copy
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from test_ctrl24_avoid import LANE, apply, car, rig as avoid_rig     # noqa: E402
from test_ctrl24_long import TL, Walker, rig as long_rig, set_signal  # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
HZ = CFG['comm']['send_hz']
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']


def on_cfg(**kw):
    c = copy.deepcopy(CFG)
    c['ctrl24']['escape_enable'] = True
    c['ctrl24'].update(kw)
    return c


def move(ap, p, m):
    """자차를 경로를 따라 m 미터 앞으로 — 고착 판정은 **자차 이동거리**로 잰다."""
    ap._vehicle._x += float(m)
    p.route_index = min(len(p.route_s) - 1,
                        p.route_index + int(round(m * p.points_per_meter)))


def hold(kr, ap, p, n, v=0.0, target=12.5, advance=0.0):
    """n 틱 동안 같은 자리에서(또는 advance m/틱 진행하며) 돌린다."""
    out = None
    for _ in range(n):
        out = apply(kr, ap, v=v, target=target)
        if advance:
            move(ap, p, advance)
    return out


# ── 스위치·기본값 ────────────────────────────────────────────────────────
def test_params_defaults_and_off_by_default():
    assert C['escape_enable'] is False
    assert C['escape_stuck_s'] == 3.0 and C['escape_progress_m'] == 0.1
    assert C['escape_v'] == 1.0 and C['escape_release_m'] == 5.0
    assert C['escape_clear_m'] == 2.0
    assert C['escape_rearm_shift_enable'] is False
    kr, _p, _ap = avoid_rig()
    assert kr.esc_on_cfg is False and kr._esc_engaged is False


def test_off_never_engages_and_keeps_the_stop():
    kr, p, ap = avoid_rig(actors=[car(2, 40.0)])
    _c, t = hold(kr, ap, p, int(10 * HZ), v=0.0, target=0.0)
    assert kr._esc_engaged is False and kr.last_escape is None and t == 0.0


# ── 고착 판정 ────────────────────────────────────────────────────────────
def test_engages_after_stuck_window_and_raises_the_target():
    kr, p, ap = avoid_rig(cfg=on_cfg(), actors=[car(2, 40.0)])
    n = int(C['escape_stuck_s'] * HZ)
    for i in range(n):                                   # 창이 찰 때까지는 안 걸린다
        _c, t = apply(kr, ap, v=0.0, target=0.0)
        assert kr._esc_engaged is False, i
        assert t == 0.0
    _c, t = apply(kr, ap, v=0.0, target=0.0)
    assert kr._esc_engaged is True
    assert t == pytest.approx(C['escape_v'])
    e = kr.last_escape
    assert e['state'] == 'FLOOR' and e['raised'] is True
    assert e['before'] == 0.0 and e['after'] == pytest.approx(C['escape_v'])


def test_progress_prevents_the_latch():
    """매 틱 조금씩이라도 진행하면 고착이 아니다."""
    kr, p, ap = avoid_rig(cfg=on_cfg(), actors=[car(2, 40.0)])
    hold(kr, ap, p, int(10 * HZ), v=0.0, target=0.0, advance=0.2)
    assert kr._esc_engaged is False and kr.last_escape is None


def test_latch_holds_until_release_distance():
    kr, p, ap = avoid_rig(cfg=on_cfg(), actors=[car(2, 40.0)])
    hold(kr, ap, p, int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)
    assert kr._esc_engaged is True
    move(ap, p, C['escape_release_m'] - 1.0)
    apply(kr, ap, v=1.0, target=0.0)
    assert kr._esc_engaged is True                        # 아직 5 m 를 못 갔다
    move(ap, p, 1.5)
    apply(kr, ap, v=1.0, target=0.0)
    assert kr._esc_engaged is False and kr.last_escape is None


def test_progress_is_measured_by_travelled_distance_not_route_s():
    """route_s 가 멈춰도(종점 패드) 자차가 움직이면 고착이 아니다."""
    kr, p, ap = avoid_rig(cfg=on_cfg(), actors=[car(2, 40.0)])
    p.route_index = len(p.route_s) - 1                    # route_s 가 더 안 는다
    for _ in range(int(10 * HZ)):
        apply(kr, ap, v=1.0, target=0.0)
        ap._vehicle._x += 0.2                             # 자차는 계속 간다
    assert kr._esc_engaged is False and kr.last_escape is None
    assert kr._esc_odo == pytest.approx(0.2 * 10 * HZ, rel=0.02)


def test_teleport_is_not_counted_as_progress():
    """courseRespawn — on_reset 이 직전 위치를 지워 순간이동 거리가 안 섞인다."""
    kr, p, ap = avoid_rig(cfg=on_cfg(), actors=[car(2, 40.0)])
    hold(kr, ap, p, 5, v=0.0, target=0.0)
    before = kr._esc_odo
    kr.on_reset()
    ap._vehicle._x += 500.0
    apply(kr, ap, v=0.0, target=0.0)
    assert kr._esc_odo == pytest.approx(before)           # 500 m 가 안 실린다


# ── 정당한 정지 원인은 못 뚫는다 ──────────────────────────────────────────
def test_red_light_stop_is_never_broken():
    """적신호 정지선 앞 정지 — K1 이 0 이라 고착 판정 자체가 안 선다."""
    kr, p, ap = long_rig(cfg=on_cfg(), d_tl=FRONT + 0.2, state=TrafficLightState.Red)
    for _ in range(int(20 * HZ)):
        _c, t = apply(kr, ap, v=0.0)
        assert t == 0.0
    assert kr._esc_engaged is False and kr.last_escape is None


def test_pedestrian_stop_is_never_broken():
    w = Walker(9, 12.0, 3.0, speed=1.5)
    kr, p, ap = long_rig(cfg=on_cfg(), actors=[w])
    apply(kr, ap, v=0.0)
    w.move(12.0, 0.2)                                     # 경로 위로 들어왔다
    for _ in range(int(20 * HZ)):
        _c, t = apply(kr, ap, v=0.0)
        assert t == 0.0
    assert kr._esc_engaged is False


def test_pdm_red_light_hazard_blocks_the_latch():
    """ctrl24 후보가 아니라 PDM 적신호가 세운 경우 — hazard 플래그로 본다."""
    kr, p, ap = avoid_rig(cfg=on_cfg())
    ap.traffic_light_hazard = True
    hold(kr, ap, p, int(10 * HZ), v=0.0, target=0.0)
    assert kr._esc_engaged is False
    ap.traffic_light_hazard = False
    hold(kr, ap, p, int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)
    assert kr._esc_engaged is True


def test_legit_candidate_below_floor_still_wins_while_engaged():
    """걸린 뒤에도 정당한 후보가 바닥보다 낮으면 그쪽이 이긴다."""
    kr, p, ap = avoid_rig(cfg=on_cfg(), actors=[car(2, 40.0)])
    hold(kr, ap, p, int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)
    assert kr._esc_engaged is True
    kr._stopline_hold = lambda *_a, **_k: 0.0             # 정당한 원인 하나를 0 으로
    _c, t = apply(kr, ap, v=0.0, target=0.0)
    assert t == 0.0                                       # 바닥이 못 올린다
    assert kr._esc_engaged is False                       # 정당한 정지라 래치도 풀린다


# ── 간격 조건 ────────────────────────────────────────────────────────────
def test_floor_is_zero_when_the_front_gap_is_too_small():
    close = car(2, FRONT + 2.2 + 1.0)                     # 범퍼 간격 ≈ 1 m < 2.0
    kr, p, ap = avoid_rig(cfg=on_cfg(), actors=[close])
    hold(kr, ap, p, int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)
    assert kr._esc_engaged is True
    e = kr.last_escape
    assert e['state'] == 'BLOCKED_GAP' and e['raised'] is False
    assert 0.0 < e['gap_m'] < C['escape_clear_m']
    assert kr.last_target == 0.0                          # 원래 후보 그대로


def test_floor_applies_once_the_gap_opens():
    far = car(2, FRONT + 2.2 + 4.0)                       # 범퍼 간격 ≈ 4 m > 2.0
    kr, p, ap = avoid_rig(cfg=on_cfg(), actors=[far])
    hold(kr, ap, p, int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)
    e = kr.last_escape
    assert e['state'] == 'FLOOR' and e['gap_m'] > C['escape_clear_m']
    assert kr.last_target == pytest.approx(C['escape_v'])


# ── 재무장 ───────────────────────────────────────────────────────────────
def _stuck_with_handled_blocker(cfg):
    """회랑에 남아 있는 객체를 이미 시프트한 상태 (avoid.state = HANDLED) 로 만든다.
    양쪽에 이웃이 없어 시프트는 NOOP 이므로 객체가 회랑에 그대로 남는다."""
    kr, p, ap = avoid_rig(cfg=cfg, actors=[car(2, 40.0)], left=False, right=False)
    apply(kr, ap, v=8.0)
    kr._shifted_for.add(2)                                # 앞서 한 번 시프트했다고 친다
    return kr, p, ap


def test_rearm_off_by_default_keeps_the_shifted_set():
    kr, p, ap = _stuck_with_handled_blocker(on_cfg())
    hold(kr, ap, p, int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)
    assert kr._esc_engaged is True and 2 in kr._shifted_for
    assert 'rearmed' not in (kr.last_escape or {})


def test_rearm_fires_while_engaged_not_only_at_the_latch_tick():
    """래치가 선 **뒤에** 시프트한 객체도 잡는다 (2026-09-10 트리거 이동)."""
    kr, p, ap = avoid_rig(cfg=on_cfg(escape_rearm_shift_enable=True),
                          actors=[car(2, 40.0)], left=False, right=False)
    hold(kr, ap, p, int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)
    assert kr._esc_engaged is True and not kr._shifted_for
    kr._shifted_for.add(2)                                # 래치가 선 뒤에 생긴 HANDLED
    apply(kr, ap, v=0.0, target=0.0)
    assert 2 not in kr._shifted_for
    assert kr.last_escape.get('rearmed') == [2]


def test_rearm_on_clears_the_corridor_blocker_from_the_set():
    kr, p, ap = _stuck_with_handled_blocker(on_cfg(escape_rearm_shift_enable=True))
    hold(kr, ap, p, int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)
    assert kr._esc_engaged is True
    assert 2 not in kr._shifted_for                       # 중첩 시프트 재시도 가능
    assert kr.last_escape.get('rearmed') == [2]


# ── 리셋 ─────────────────────────────────────────────────────────────────
def test_reset_clears_the_clock_and_latch():
    kr, p, ap = avoid_rig(cfg=on_cfg(), actors=[car(2, 40.0)])
    hold(kr, ap, p, int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)
    assert kr._esc_engaged is True
    kr.on_reset()
    assert kr._esc_engaged is False and kr._esc_mark is None and not kr._esc_hist
