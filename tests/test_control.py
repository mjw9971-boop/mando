"""Pure Pursuit 부호/거동 검증 (SPEC §3.6)."""
import math

import pytest
from conftest import PARAMS_YAML
from hlfma.core.control import Control
from hlfma.core.types import Decision, EgoState, WorldState
from hlfma.nodes.params import load_params_yaml

CFG = load_params_yaml(PARAMS_YAML)


def make_world(x=0.0, y=0.0, yaw=0.0, speed=5.0, t=0.0) -> WorldState:
    ego = EgoState(x=x, y=y, z=0.0, yaw=yaw, pitch=0.0, roll=0.0,
                   speed=speed, accel=0.0, lane=(1, 0, -1), s=0.0,
                   route_s=0.0, t_off=0.0, heading_err=0.0)
    return WorldState(t=t, ego=ego, objects=[], light=None, ahead=[], summ={},
                      speed_limit=13.9, school_zone=False, left_solid=False,
                      right_solid=False, left_is_center=False, valid=True, flags={})


def make_decision(path, v_target=5.0, state='FOLLOW') -> Decision:
    return Decision(v_target=v_target, path=path, turn_signal=0, state=state, reasons={})


def straight_path(n=100, step=0.5):
    """+x 방향 직선."""
    return [(i * step, 0.0) for i in range(1, n + 1)]


def curved_path(n=100, step=0.5, curvature=0.02):
    """좌측(+y)으로 휘는 호."""
    pts = []
    for i in range(1, n + 1):
        s = i * step
        th = curvature * s
        pts.append((math.sin(th) / curvature, (1 - math.cos(th)) / curvature))
    return pts


def steer_once(path, **world_kw) -> float:
    """변화율 제한의 영향을 배제하려고 충분히 여러 틱 돌린 뒤 정상값을 본다."""
    c = Control(CFG)
    st = 0.0
    for k in range(200):
        w = make_world(t=k * 0.05, **world_kw)
        st = c.compute(w, make_decision(path)).steering
    return st


# ── 직선 ──────────────────────────────────────────────────────────────────
def test_straight_path_gives_zero_steering():
    """alpha ≈ 0 → steering ≈ 0"""
    assert steer_once(straight_path()) == pytest.approx(0.0, abs=1e-6)


def test_straight_path_offset_left_steers_right():
    """경로가 오른쪽에 있으면(차가 왼쪽으로 밀렸으면) 우조향(음수)."""
    assert steer_once(straight_path(), y=1.0) < 0.0


# ── 곡선 ──────────────────────────────────────────────────────────────────
def test_left_curve_gives_left_steering():
    """좌측으로 휜 경로 → 내부 조향 부호는 좌회전(+)."""
    assert steer_once(curved_path(curvature=0.02)) > 0.0


def test_right_curve_gives_right_steering():
    assert steer_once(curved_path(curvature=-0.02)) < 0.0


def test_sharper_curve_gives_larger_steering():
    gentle = abs(steer_once(curved_path(curvature=0.01)))
    sharp = abs(steer_once(curved_path(curvature=0.05)))
    assert sharp > gentle


def test_steering_is_clamped_to_max_steer():
    max_steer = float(CFG['vehicle']['max_steer'])
    st = steer_once(curved_path(curvature=0.5, step=0.2), speed=1.0)
    assert abs(st) <= max_steer + 1e-9


# ── 기하 ──────────────────────────────────────────────────────────────────
def test_lookahead_follows_k_ld_and_clamps():
    c = Control(CFG)
    cfg = CFG['control']
    for v, expect in [(0.0, cfg['ld_min']),
                      (100.0, cfg['ld_max']),
                      (10.0, min(max(cfg['k_ld'] * 10.0, cfg['ld_min']), cfg['ld_max']))]:
        c.compute(make_world(speed=v), make_decision(straight_path()))
        assert c.last_ld == pytest.approx(expect)


def test_empty_path_returns_zero_steering():
    c = Control(CFG)
    assert c.compute(make_world(), make_decision([])).steering == pytest.approx(0.0)


def test_steer_rate_is_limited():
    """한 틱에 steer_rate_max * dt 이상 못 움직인다."""
    c = Control(CFG)
    dt = 1.0 / float(CFG['comm']['send_hz'])
    c.compute(make_world(t=0.0), make_decision(straight_path()))
    st = c.compute(make_world(t=dt), make_decision(curved_path(curvature=0.5, step=0.2))).steering
    assert abs(st) <= float(CFG['control']['steer_rate_max']) * dt * 2 + 1e-9


# ── 종방향 ────────────────────────────────────────────────────────────────
def test_accel_positive_when_below_target():
    c = Control(CFG)
    cmd = c.compute(make_world(speed=0.0), make_decision(straight_path(), v_target=5.56))
    assert cmd.accel > 0.0


def test_accel_negative_when_above_target():
    c = Control(CFG)
    cmd = c.compute(make_world(speed=10.0), make_decision(straight_path(), v_target=5.56))
    assert cmd.accel < 0.0


def test_hold_brake_when_stopped_and_target_zero():
    c = Control(CFG)
    cmd = c.compute(make_world(speed=0.1), make_decision(straight_path(), v_target=0.0))
    assert cmd.accel == pytest.approx(float(CFG['speed']['a_hold']))


def test_accel_clamped_to_limits():
    c = Control(CFG)
    a_max, a_min = float(CFG['speed']['a_max']), float(CFG['speed']['a_min'])
    for k in range(100):
        cmd = c.compute(make_world(speed=0.0, t=k * 0.05),
                        make_decision(straight_path(), v_target=50.0))
    assert cmd.accel <= a_max + 1e-9
    c.reset()
    for k in range(100):
        cmd = c.compute(make_world(speed=50.0, t=k * 0.05),
                        make_decision(straight_path(), v_target=0.0))
    assert cmd.accel >= a_min - 1e-9
