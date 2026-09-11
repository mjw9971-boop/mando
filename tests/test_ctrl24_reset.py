"""
ctrl24 — 리스폰 뒤 시프트 상태 원복 (2026-09-11).

여기서 지키는 불변:
  · on_reset 은 밀린 route_points 를 _restore_span 으로 원복하고 시프트 래치
    (ot_span · ot_side · nested · _shifted_for · 가상 span)를 전부 비운다.
  · 전이 도중(SHIFT_ACTIVE)에 불러도 안전하다 — 시프트는 route_points[a:b] 안에서만
    쓰고 ot_span 은 합집합이라, 그 구간을 원본으로 덮으면 경계에 계단이 없다.
  · 기본 off — off 면 on_reset 이 시프트 상태를 건드리지 않는다 (이전 동작).
  · 이미 지우던 _esc_* · 신호 래치는 스위치와 무관하게 그대로 지운다.
  · 원복은 경로 변경이므로 _shift_seq 가 오른다 (예측 캐시 무효).

params 값은 팀원 소관이라 리터럴을 박지 않는다 (docs/BACKLOG.md B-25). 킬스위치
기본값과 관계만 보고, 산술이 필요한 곳은 사본에서 명시적으로 켠다.
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

from test_ctrl24_avoid import apply, car, rig                      # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']


def on_cfg(**kw):
    """사본에서 리셋 원복을 켠다 — 기본값이 또 바뀌어도 안 깨지게."""
    c = copy.deepcopy(CFG)
    c['ctrl24']['reset_restore_span_enable'] = True
    c['ctrl24'].update(kw)
    return c


def off_cfg(**kw):
    c = copy.deepcopy(CFG)
    c['ctrl24']['reset_restore_span_enable'] = False
    c['ctrl24'].update(kw)
    return c


def shifted(cfg):
    """시프트가 살아 있는(SHIFT_ACTIVE) 상태의 리그를 만든다."""
    kr, p, ap = rig(cfg=cfg, actors=[car(2, 78.0)])
    apply(kr, ap, v=12.5)
    assert kr.ot_span is not None                                  # 전제
    return kr, p, ap


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_kill_switch_defaults_off():
    assert C['reset_restore_span_enable'] is False


def test_off_leaves_shift_state_and_route_untouched():
    kr, p, ap = shifted(off_cfg())
    span = kr.ot_span
    before = p.route_points.copy()
    kr.on_reset()
    assert kr.ot_span == span and kr.ot_side is not None
    assert 2 in kr._shifted_for
    assert np.array_equal(p.route_points, before)                  # 경로는 밀린 그대로


def test_off_still_clears_the_latches_it_always_cleared():
    """스위치는 시프트 상태에만 붙는다 — 기존 리셋 동작은 그대로다."""
    kr, p, ap = shifted(off_cfg())
    kr._esc_engaged = True
    kr._esc_defer_ticks = 7
    kr._sig_go = True
    kr.cross_guard = True
    kr.on_reset()
    assert kr._esc_engaged is False and kr._esc_defer_ticks == 0
    assert kr._sig_go is False and kr.cross_guard is False


# ── on: 경로 원복 ────────────────────────────────────────────────────────
def test_on_restores_the_pushed_route_to_the_original():
    kr, p, ap = shifted(on_cfg())
    a, b = kr.ot_span
    assert float(np.abs(p.route_points[a:b, 1]).max()) > 1.0       # 전제: 실제로 밀려 있다
    kr.on_reset()
    assert np.allclose(p.route_points, p.original_route_points)


def test_on_restores_mid_transition_without_leaving_a_step():
    """자차가 전이 한가운데 있어도 경계에 계단이 없다."""
    kr, p, ap = shifted(on_cfg())
    a, b = kr.ot_span
    p.route_index = (a + b) // 2                                   # 플래토 한복판 = 전이 통과 뒤
    kr.on_reset()
    d = np.abs(np.diff(p.route_points[:, 1]))
    assert float(d.max()) == pytest.approx(0.0, abs=1e-9)          # 원 경로는 직선이다
    assert np.allclose(p.route_points, p.original_route_points)


def test_on_restores_commands_and_lat_shift_too():
    kr, p, ap = shifted(on_cfg())
    kr.on_reset()
    assert np.array_equal(p.commands, p.commands_orig)
    assert np.allclose(p.lat_shift, p._lat_build)


# ── on: 래치 비움 ────────────────────────────────────────────────────────
def test_on_clears_every_shift_latch():
    kr, p, ap = shifted(on_cfg())
    kr.nested = 2
    kr._virt_span = (10, 20)
    kr._virt_wait = 5
    kr.last_span_plan = (1, 2, True)
    kr.span_v_req = 3.0
    kr.on_reset()
    assert kr.ot_span is None and kr.ot_side is None and kr.nested == 0
    assert not kr._shifted_for
    assert kr._virt_span is None and kr._virt_wait == 0
    assert kr.last_span_plan is None and kr.span_v_req is None


def test_on_clears_latches_even_when_no_span_is_active():
    """원복할 경로가 없어도 남은 래치는 비운다 (요동 방지 집합만 남는 경우)."""
    kr, p, ap = rig(cfg=on_cfg(), actors=[car(2, 78.0)])
    kr._shifted_for.add(9)
    kr.ot_side = 'left'
    kr.nested = 1
    assert kr.ot_span is None                                      # 전제
    kr.on_reset()
    assert not kr._shifted_for and kr.ot_side is None and kr.nested == 0


def test_on_bumps_shift_seq_because_the_route_changed():
    kr, p, ap = shifted(on_cfg())
    seq = kr._shift_seq
    kr.on_reset()
    assert kr._shift_seq > seq


# ── 리셋 뒤 재시도 ───────────────────────────────────────────────────────
def test_shift_can_be_made_again_after_reset():
    """요동 방지 집합이 비었으므로 같은 객체에 다시 시프트를 만들 수 있다."""
    kr, p, ap = shifted(on_cfg())
    kr.on_reset()
    apply(kr, ap, v=12.5)
    assert kr.ot_span is not None and 2 in kr._shifted_for
    assert kr.last_avoid['state'] == 'PREEMPT'                     # 중첩이 아니라 새 시프트


def test_reset_without_a_planner_does_not_raise():
    """첫 틱 전에 리셋이 오면 _ap 가 없다 — 래치만 비우고 넘어간다."""
    kr, p, ap = rig(cfg=on_cfg(), actors=[])
    kr._ap = None
    kr._shifted_for.add(3)
    kr.on_reset()
    assert not kr._shifted_for


def test_restore_is_idempotent():
    kr, p, ap = shifted(on_cfg())
    kr.on_reset()
    first = p.route_points.copy()
    kr.on_reset()
    assert np.array_equal(p.route_points, first)
    assert kr.ot_span is None
