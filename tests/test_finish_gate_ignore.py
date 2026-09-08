"""
[1](b) 종료 구간 정적 장애물 무시 (speed.finish_gate_ignore_enable).

실측: logs/batch/20260908_130919/실경로_01_PathShape03 — 종료선 17 m 앞 영구
정지(미완주). 마지막 좌회전 연결로 (1502,0,-1) 폭 2.4 m 안에서 종료점 콘
(객체 id 2·3, cls obstacle, speed 0, 0.3×0.3×0.32 m)이 회랑(reach 1.403) 안에
들어와 blocker 가 됐다: rs 843.5 부터 standoff_v 0, rs 848.1 에서 정지.

생성기 쪽은 콘을 도로 끝으로 옮겨 고쳤지만(gen_placement.cone_at_road_edge)
**주최측 콘 위치는 우리가 못 정한다** — 대회에서 같은 일이 그대로 난다.

하필 이 구간은 다른 안전망이 전부 죽어 있다: route_end.active_m(150 m) 안이라
_obstacle_cause 가 거짓이고 BREAKOUT·크립·never_stall 이 하나도 안 선다.

여기서 지키는 불변:
  · 빼는 것은 **정지 객체뿐**이다. 움직이는 객체는 계속 본다.
  · 속도 임계는 overtake.blocker_speed_max 를 그대로 읽는다 (단일 출처).
  · 관문은 _corridor_blockers **하나**다 — 회피·standoff·큐가 같이 빠진다.
  · PDM 의 IDM 추종·보행자 정지는 여기와 무관하게 계속 산다.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402
from test_avoid import Ap, Box, GeomPlanner, HZ, LgOne             # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
GATE_M = CFG['speed']['finish_gate_m']
V_STATIC = CFG['overtake']['blocker_speed_max']
FINISH_S = 100.0


def on_cfg(**over):
    c = copy.deepcopy(CFG)
    c['speed']['finish_gate_ignore_enable'] = True
    c['speed'].update(over)
    return c


def off_cfg(**over):
    """이전 동작 사본. 기본값을 읽지 않는다 (2026-09-07 드리프트 원칙)."""
    c = copy.deepcopy(CFG)
    c['speed']['finish_gate_ignore_enable'] = False
    c['speed'].update(over)
    return c


def rig(cfg, obj_x=8.0, obj_speed=0.0, route_s=0.0):
    """장애물이 obj_x 앞. finish_s 는 100.0 으로 주입하고 자차를 route_s 에 둔다."""
    p = GeomPlanner(d_tl=float('inf'))
    p.lg = LgOne()
    p.route_index = int(route_s * p.points_per_meter)
    kr = KrRules(cfg)
    kr._sl_all = []
    kr.finish_s = FINISH_S
    ap = Ap(p, actors=[Box(2, obj_x + route_s, 0.0, obj_speed, half_w=0.9)])
    ap._kr_ego_lane = (1, 0, -1)
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    kr._ap = ap
    for _ in range(int(cfg['overtake']['obj_static_s'] * HZ) + 5):
        kr._update_obj_timers(ap)
    return kr, p, ap


def corridor(kr, p, ap):
    return kr._corridor_blockers(ap, p, static_ok=kr._stop_ok)


# ── 게이트 경계 ────────────────────────────────────────────────────────────
def test_gate_is_off_before_the_window():
    kr, p, ap = rig(on_cfg(), route_s=FINISH_S - GATE_M - 5.0)
    assert kr._in_finish_gate(p) is False
    assert len(corridor(kr, p, ap)) == 1                # 아직 장애물로 본다


def test_gate_opens_at_finish_minus_gate_m():
    kr, p, ap = rig(on_cfg(), route_s=FINISH_S - GATE_M)
    assert kr._in_finish_gate(p) is True
    assert corridor(kr, p, ap) == []


def test_gate_stays_open_past_the_finish_line():
    """꼬리 끝까지 — 종료선을 넘어선 뒤에도 유지된다."""
    kr, p, ap = rig(on_cfg(), route_s=FINISH_S + 20.0)
    assert kr._in_finish_gate(p) is True
    assert corridor(kr, p, ap) == []


def test_switch_off_keeps_the_blocker():
    kr, p, ap = rig(off_cfg(), route_s=FINISH_S)
    assert kr._in_finish_gate(p) is False
    assert len(corridor(kr, p, ap)) == 1


def test_no_finish_s_means_no_gate():
    """finish_xy 를 투영 못 한 경로에서는 게이트가 아예 없다 (종점 기준 없음)."""
    kr, p, ap = rig(on_cfg(), route_s=FINISH_S)
    kr.finish_s = None
    assert kr._in_finish_gate(p) is False
    assert len(corridor(kr, p, ap)) == 1


# ── 정지 객체만 뺀다 ───────────────────────────────────────────────────────
def test_moving_object_is_still_seen_inside_the_gate():
    """움직이는 객체는 계속 본다 — 빼는 것은 정지 객체뿐이다."""
    kr, p, ap = rig(on_cfg(), obj_speed=V_STATIC + 1.0, route_s=FINISH_S)
    kr.obj_stop_ticks[2] = 10 ** 6                      # standoff 대상 조건은 통과시킨다
    kr.obj_ticks[2] = 10 ** 6
    assert len(corridor(kr, p, ap)) == 1


def test_static_threshold_reads_blocker_speed_max():
    """임계는 overtake.blocker_speed_max 하나다 — 값을 두 곳에 적지 않는다."""
    kr, p, ap = rig(on_cfg(), obj_speed=V_STATIC - 0.01, route_s=FINISH_S)
    assert corridor(kr, p, ap) == []
    kr2, p2, ap2 = rig(on_cfg(), obj_speed=V_STATIC + 0.01, route_s=FINISH_S)
    kr2.obj_stop_ticks[2] = 10 ** 6
    kr2.obj_ticks[2] = 10 ** 6
    assert len(corridor(kr2, p2, ap2)) == 1


# ── standoff 가 실제로 안 걸린다 (이 작업의 목적) ──────────────────────────
def test_standoff_disappears_inside_the_gate():
    """실측 재현 축소판 — 콘이 회랑 안이어도 종료 구간에서는 안 세운다."""
    kr, p, ap = rig(off_cfg(), obj_x=17.3, route_s=FINISH_S - 5.0)
    kr._tick_cache(ap, p)
    kr._standoff_target(ap, p, kr._tick_corridor)
    assert kr.wait_target_d is not None                 # 이전 동작: standoff 대상이 선다

    kr2, p2, ap2 = rig(on_cfg(), obj_x=17.3, route_s=FINISH_S - 5.0)
    kr2._tick_cache(ap2, p2)
    kr2._standoff_target(ap2, p2, kr2._tick_corridor)
    assert kr2.wait_target_d is None
    assert kr2.fg_dropped > 0                           # 진단이 남는다


def test_off_leaves_no_diagnostic():
    kr, p, ap = rig(off_cfg(), route_s=FINISH_S)
    corridor(kr, p, ap)
    assert kr.fg_dropped == 0
