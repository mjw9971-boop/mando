"""
ctrl24 — 적색 유예 해제 틱에 고착 래치까지 푼다 (2026-09-11).

여기서 지키는 불변:
  · 동작하는 틱은 C 가 **참 → 거짓으로 바뀌는 그 틱** 하나뿐이다.
    C = (정지선 거리 ≤ escape_red_defer_stopline_m) ∧ (그 정지선 신호 state == Red).
  · 그 틱에 _esc_hist·_esc_engaged·_esc_mark·_esc_xy 를 푼다 — on_reset 의 K9 집합과
    같은 넷이다. 그래서 다음 틱부터 진행 시계가 처음부터 다시 차고,
    최소 escape_stuck_s 동안 바닥이 깔리지 않는다.
  · 거짓 → 거짓, 참 → 참 인 틱에서는 아무것도 하지 않는다 (매 틱 리셋 금지).
  · 정지선 이탈로 C 가 거짓이 된 경우도 같은 처리다 (사유를 가리지 않는다).
  · 기본 off — off 면 전이 검출조차 하지 않는다 (_esc_prev_c 가 안 움직인다).
  · 새 상태는 직전 틱 C 하나(_esc_prev_c)뿐이다. 타이머를 만들지 않는다.

params 값은 팀원 소관이라 리터럴을 박지 않는다 (docs/BACKLOG.md B-25).
킬스위치 기본값과 관계만 보고, 산술이 필요한 곳은 사본에서 명시적으로 켠다.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from test_ctrl24_avoid import apply, car, rig as avoid_rig        # noqa: E402
from test_ctrl24_escape import hold, on_cfg                        # noqa: E402
from test_ctrl24_long import TL, rig as long_rig, set_signal       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']
HZ = CFG['comm']['send_hz']
LATCH = ('_esc_hist', '_esc_engaged', '_esc_mark', '_esc_xy')


def rearm_cfg(**kw):
    """사본에서 유예와 해제 재무장을 함께 켠다."""
    c = on_cfg(escape_red_defer_enable=True, escape_defer_release_rearm_enable=True)
    c['ctrl24'].update(kw)
    return c


def off_cfg(**kw):
    c = on_cfg(escape_red_defer_enable=True, escape_defer_release_rearm_enable=False)
    c['ctrl24'].update(kw)
    return c


def rig_at(d_tl, state, cfg=None):
    return long_rig(cfg or rearm_cfg(), d_tl=d_tl, state=state)


def stick(kr, ap, p, n=None):
    hold(kr, ap, p, n or int(C['escape_stuck_s'] * HZ) + 1, v=0.0, target=0.0)


def latched(kr):
    return bool(kr._esc_engaged)


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_kill_switch_defaults_off():
    assert C['escape_defer_release_rearm_enable'] is False


def test_off_keeps_the_latch_and_never_tracks_c():
    """off 면 전이 검출조차 하지 않는다 — 직전 틱 C 가 안 움직인다."""
    kr, p, ap = rig_at(10.0, TrafficLightState.Red, cfg=off_cfg())
    stick(kr, ap, p)
    assert latched(kr) and kr._esc_deferring is True
    assert kr._esc_prev_c is False                      # 켜지 않았으므로 기록하지 않는다
    set_signal(p, TrafficLightState.Green, 10.0)
    apply(kr, ap, v=0.0, target=0.0)
    assert latched(kr)                                  # 래치 그대로
    assert kr.last_escape['state'] == 'FLOOR'           # 이전 동작: 즉시 바닥


# ── 전이에서만 푼다 ───────────────────────────────────────────────────────
def test_green_transition_releases_the_latch():
    kr, p, ap = rig_at(10.0, TrafficLightState.Red)
    stick(kr, ap, p)
    assert latched(kr) and kr._esc_deferring is True and kr._esc_prev_c is True
    set_signal(p, TrafficLightState.Green, 10.0)
    apply(kr, ap, v=0.0, target=0.0)
    assert not latched(kr) and kr._esc_mark is None and kr._esc_xy is None
    assert len(kr._esc_hist) == 0
    assert kr._esc_defer_ticks == 0 and kr._esc_prev_c is False
    assert kr.last_escape is None                       # 바닥도 BLOCKED_GAP 도 아니다


def test_no_floor_for_escape_stuck_s_after_the_release():
    """다음 틱부터 진행 시계가 처음부터 찬다 — 그 동안 바닥이 없다."""
    kr, p, ap = rig_at(10.0, TrafficLightState.Red)
    stick(kr, ap, p)
    set_signal(p, TrafficLightState.Green, 10.0)
    apply(kr, ap, v=0.0, target=0.0)                    # 전이 틱
    n = int(C['escape_stuck_s'] * HZ)
    for _ in range(n):                                  # escape_stuck_s 동안은 바닥이 없다
        apply(kr, ap, v=0.0, target=0.0)
        assert not latched(kr)
        assert kr._escape_floor(ap) is None
    for _ in range(5):                                  # 시계가 다시 차면 평소대로 걸린다
        apply(kr, ap, v=0.0, target=0.0)
        if latched(kr):
            break
    assert latched(kr)


def test_leaving_the_stopline_releases_too():
    """사유를 가리지 않는다 — 거리 이탈로 C 가 거짓이 되어도 같은 처리."""
    kr, p, ap = rig_at(10.0, TrafficLightState.Red)
    stick(kr, ap, p)
    assert latched(kr) and kr._esc_prev_c is True
    set_signal(p, TrafficLightState.Red, C['escape_red_defer_stopline_m'] + 20.0)
    apply(kr, ap, v=0.0, target=0.0)
    assert not latched(kr) and len(kr._esc_hist) == 0


def test_signal_mapping_loss_releases_too():
    kr, p, ap = rig_at(10.0, TrafficLightState.Red)
    stick(kr, ap, p)
    assert kr._esc_prev_c is True
    p.next_traffic_lights = [None] * len(p.route_s)
    apply(kr, ap, v=0.0, target=0.0)
    assert not latched(kr) and len(kr._esc_hist) == 0


# ── 전이가 아닌 틱은 건드리지 않는다 ──────────────────────────────────────
def test_c_stays_false_keeps_the_latch():
    """녹색 유지 — C 가 계속 거짓이면 래치도 시계도 그대로다."""
    kr, p, ap = rig_at(10.0, TrafficLightState.Green)
    stick(kr, ap, p)
    assert latched(kr) and kr._esc_prev_c is False
    mark, n = kr._esc_mark, len(kr._esc_hist)
    for _ in range(int(2 * HZ)):
        apply(kr, ap, v=0.0, target=0.0)
    assert latched(kr) and kr._esc_mark == mark and len(kr._esc_hist) >= n
    assert kr.last_escape['state'] == 'FLOOR'           # 바닥은 그대로 깔린다


def test_c_stays_true_keeps_the_latch():
    """적색 유지 — C 가 계속 참이면 아무것도 하지 않는다."""
    kr, p, ap = rig_at(10.0, TrafficLightState.Red)
    stick(kr, ap, p)
    assert latched(kr) and kr._esc_prev_c is True
    mark = kr._esc_mark
    for _ in range(int(2 * HZ)):
        apply(kr, ap, v=0.0, target=0.0)
    assert latched(kr) and kr._esc_mark == mark and kr._esc_prev_c is True
    assert kr._esc_deferring is True and kr._esc_defer_ticks > 0


def test_far_red_never_sets_prev_c_so_no_release():
    """정지선 밖 적색 — C 가 처음부터 거짓이라 전이가 없다 (174328 @672 의 자리)."""
    kr, p, ap = rig_at(C['escape_red_defer_stopline_m'] + 20.0, TrafficLightState.Red)
    stick(kr, ap, p)
    assert latched(kr) and kr._esc_prev_c is False
    assert kr.last_escape['state'] == 'FLOOR'
    set_signal(p, TrafficLightState.Green, C['escape_red_defer_stopline_m'] + 20.0)
    apply(kr, ap, v=0.0, target=0.0)
    assert latched(kr)                                  # 전이가 아니므로 그대로
    assert kr.last_escape['state'] == 'FLOOR'


def test_green_all_along_keeps_the_floor():
    """C 가 처음부터 끝까지 거짓 — 바닥이 유지된다."""
    kr, p, ap = rig_at(10.0, TrafficLightState.Green)
    stick(kr, ap, p)
    for _ in range(int(3 * HZ)):
        out = apply(kr, ap, v=0.0, target=0.0)
        assert kr.last_escape['state'] == 'FLOOR'
    assert out[1] == pytest.approx(C['escape_v'])


def test_no_signal_mapping_never_releases():
    kr, p, ap = avoid_rig(rearm_cfg(), actors=[car(2, 40.0)])
    stick(kr, ap, p)
    assert latched(kr) and kr._esc_prev_c is False
    for _ in range(int(2 * HZ)):
        apply(kr, ap, v=0.0, target=0.0)
    assert latched(kr)


# ── 집합 일치 ────────────────────────────────────────────────────────────
def test_release_clears_the_same_set_as_on_reset():
    """푸는 네 항목이 on_reset 의 K9 집합과 같다 — 한쪽만 고치면 이 검사가 깨진다."""
    def snap(kr):
        return (len(kr._esc_hist), kr._esc_engaged, kr._esc_mark, kr._esc_xy)

    kr1, p1, ap1 = rig_at(10.0, TrafficLightState.Red)
    stick(kr1, ap1, p1)
    kr1.on_reset()
    kr2, p2, ap2 = rig_at(10.0, TrafficLightState.Red)
    stick(kr2, ap2, p2)
    set_signal(p2, TrafficLightState.Green, 10.0)
    apply(kr2, ap2, v=0.0, target=0.0)
    assert snap(kr1) == snap(kr2) == (0, False, None, None)


def test_no_new_timer_is_introduced():
    """새 상태는 직전 틱 C 하나뿐이다."""
    kr, _p, _ap = rig_at(10.0, TrafficLightState.Red)
    assert isinstance(kr._esc_prev_c, bool)
    assert not [a for a in vars(kr) if a.startswith('_esc_') and a not in (
        '_esc_hist', '_esc_engaged', '_esc_mark', '_esc_xy', '_esc_odo',
        '_esc_defer_ticks', '_esc_deferring', '_esc_rearmed', '_esc_prev_c')]
