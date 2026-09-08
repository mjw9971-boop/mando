"""차로 지도 (정적 장애물 회피 개편, 커밋 A) — avoid_map.lane_map_avoid_enable.

**커밋 A 는 읽기 전용이다.** 자차 앞 lane_map_ahead_m 를 차로별로 재 두기만
하고, 후보 선택에는 쓰지 않는다 (그건 커밋 B). 그래서 이 파일이 지키는 것은
"지도가 맞나" 와 "off 면 아무것도 안 한다" 둘이다.

지금 로직의 증상 중 이 지도가 겨냥하는 것:
  · 회랑(~22 m)에 들어와야 발견 → 80 m 앞부터 본다
  · 좌우 한 칸만 본다 → ±lane_map_max_hops (기본 2)
  · **걸쳐 선 차를 중심만 보고 한 차로만 막힌 것으로 본다** → OBB 세 점 투영
  · 신호 대기 줄을 추월 후보로 삼는다 → _tick_queue 인 틱은 회랑 객체를 뺀다
"""
import copy
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402
from test_avoid import Ap, Box, Planner, HZ                        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
L0, L1, L2 = (1, 0, -1), (1, 0, -2), (1, 0, -3)


class Lg3:
    """3차로 직선 — 오른쪽으로 L0 → L1 → L2. 폭·방향·junction 을 갖춘 최소 목."""

    def __init__(self, width=3.0, w2=None, junction2=-1):
        s = np.linspace(0.0, 1000.0, 51)
        def rec(w, j):
            return {'junction': j, 'dir': -1, 's': s,
                    'width': np.full_like(s, w), 'length': 1000.0}
        self.lanes = {L0: rec(width, -1), L1: rec(width, -1),
                      L2: rec(width if w2 is None else w2, junction2)}

    def neighbor(self, key, side):
        order = [L0, L1, L2]
        if key not in order:
            return None
        i = order.index(key) + (1 if side == 'right' else -1)
        return order[i] if 0 <= i < len(order) else None

    def length(self, key):
        return 1000.0

    def locate(self, x, y, prefer=None, **kw):
        """y 로 차로를 가른다: 0±1.5 → L0, −3±1.5 → L1, −6±1.5 → L2.

        `_ego_local_s` 가 `.s` 를 읽으므로 x 를 그대로 s 로 준다 (직선 경로).
        """
        order = [L0, L1, L2]
        i = int(round(-y / 3.0))
        if not (0 <= i < len(order)):
            return None
        return type('M', (), {'key': order[i], 's': float(x), 't': 0.0})()


def on_cfg(**over):
    c = copy.deepcopy(CFG)
    c['avoid_map']['lane_map_avoid_enable'] = True
    c['avoid_map'].update(over)
    return c


def off_cfg():
    c = copy.deepcopy(CFG)
    c['avoid_map']['lane_map_avoid_enable'] = False
    return c


def rig(cfg, actors=(), lg=None, queue=False):
    p = Planner(d_tl=float('inf'))
    p.lg = lg or Lg3()
    kr = KrRules(cfg)
    kr._sl_all = []
    ap = Ap(p, actors=list(actors))
    ap._kr_ego_lane = L0
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    kr._ap = ap
    kr._tick_ego_lane = L0
    kr._tick_queue = queue
    kr._tick_corridor = [(10.0, 0.0, 0.9, a) for a in actors] if queue else []
    return kr, p, ap


def M(kr, p, ap):
    return kr.lane_map(ap, p)


def fr(m, key):
    return m['free_run'][str(list(key))]


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_off_returns_none():
    kr, p, ap = rig(off_cfg())
    assert kr.lane_map(ap, p) is None


def test_off_leaves_no_diagnostic_key():
    """off 는 로그까지 이전과 동일해야 지문 비교가 성립한다."""
    kr, p, ap = rig(off_cfg())
    kr.last_lane_map = kr.lane_map(ap, p)
    assert kr.last_lane_map is None


# ── 이웃 ─────────────────────────────────────────────────────────────────
def test_hops_cover_max_hops_each_side():
    kr, p, ap = rig(on_cfg(lane_map_max_hops=2))
    m = M(kr, p, ap)
    assert m['ego_lane'] == list(L0)
    assert set(m['hops']) == {str(list(k)) for k in (L0, L1, L2)}
    assert m['hops'][str(list(L0))] == 0
    assert m['hops'][str(list(L1))] == 1
    assert m['hops'][str(list(L2))] == 2


def test_hops_respects_max_hops_one():
    kr, p, ap = rig(on_cfg(lane_map_max_hops=1))
    m = M(kr, p, ap)
    assert set(m['hops']) == {str(list(L0)), str(list(L1))}


# ── free_run ─────────────────────────────────────────────────────────────
def test_empty_lanes_are_full_horizon():
    kr, p, ap = rig(on_cfg(lane_map_ahead_m=80.0))
    m = M(kr, p, ap)
    assert all(v == 80.0 for v in m['free_run'].values())


def test_static_object_shortens_its_own_lane_only():
    kr, p, ap = rig(on_cfg(), actors=[Box(2, 30.0, -3.0, 0.0)])
    m = M(kr, p, ap)
    assert fr(m, L1) == pytest.approx(30.0, abs=0.6)
    assert fr(m, L0) == 80.0 and fr(m, L2) == 80.0
    assert m['blocked_by'][str(list(L1))] == 2


def test_moving_object_does_not_shorten():
    """움직이는 것은 여유를 안 깎는다 — 정적 장애물 회피의 대상이 아니다."""
    v = CFG['overtake']['blocker_speed_max'] + 1.0
    kr, p, ap = rig(on_cfg(), actors=[Box(2, 30.0, -3.0, v)])
    assert fr(M(kr, p, ap), L1) == 80.0


def test_nearest_object_wins():
    kr, p, ap = rig(on_cfg(), actors=[Box(2, 50.0, -3.0, 0.0), Box(3, 20.0, -3.0, 0.0)])
    m = M(kr, p, ap)
    assert fr(m, L1) == pytest.approx(20.0, abs=0.6)
    assert m['blocked_by'][str(list(L1))] == 3


def test_object_beyond_horizon_is_ignored():
    kr, p, ap = rig(on_cfg(lane_map_ahead_m=40.0), actors=[Box(2, 60.0, -3.0, 0.0)])
    assert fr(M(kr, p, ap), L1) == 40.0


# ── 걸쳐 선 차 (증상 3) ──────────────────────────────────────────────────
def test_straddling_object_blocks_both_lanes():
    """차로 경계에 걸쳐 선 차는 **두 차로 다** 막는다 — 중심만 보면 놓친다."""
    kr, p, ap = rig(on_cfg(), actors=[Box(2, 25.0, -1.5, 0.0, half_w=1.2)])
    m = M(kr, p, ap)
    assert fr(m, L0) == pytest.approx(25.0, abs=0.6)
    assert fr(m, L1) == pytest.approx(25.0, abs=0.6)


# ── passable ─────────────────────────────────────────────────────────────
def test_narrow_lane_is_not_passable():
    kr, p, ap = rig(on_cfg(), lg=Lg3(w2=1.5))
    m = M(kr, p, ap)
    assert m['passable'][str(list(L2))] is False
    assert m['passable'][str(list(L1))] is True


def test_junction_lane_is_not_passable():
    kr, p, ap = rig(on_cfg(), lg=Lg3(junction2=7))
    assert M(kr, p, ap)['passable'][str(list(L2))] is False


def test_min_width_is_configurable():
    kr, p, ap = rig(on_cfg(lane_map_min_width_m=3.5))
    assert all(v is False for v in M(kr, p, ap)['passable'].values())


# ── 큐 제외 ──────────────────────────────────────────────────────────────
def test_queue_objects_are_dropped():
    """신호 대기 줄은 추월 후보가 아니다 — 기존 큐 판정을 그대로 쓴다."""
    a = Box(2, 20.0, -3.0, 0.0)
    kr, p, ap = rig(on_cfg(), actors=[a], queue=True)
    m = M(kr, p, ap)
    assert m['queue_dropped'] == 1
    assert fr(m, L1) == 80.0                    # 큐라서 여유를 안 깎는다


def test_no_queue_keeps_objects():
    a = Box(2, 20.0, -3.0, 0.0)
    kr, p, ap = rig(on_cfg(), actors=[a], queue=False)
    m = M(kr, p, ap)
    assert m['queue_dropped'] == 0
    assert fr(m, L1) == pytest.approx(20.0, abs=0.6)
