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
        """VTD y 로 차로를 가른다: 0±1.5 → L0, **+3**±1.5 → L1, +6±1.5 → L2.

        CARLA 는 y 미러라(`frame.from_carla_xy` = (x, −y)) 목의 오른쪽 이웃이
        VTD 에서는 +y 다. 픽스처가 액터를 CARLA y = −3 에 놓으므로 여기서는 +3 이다.

        **입력은 VTD 프레임이다** — `_ego_lane`·`_ego_local_s`·`_actor_lanes` 가
        전부 `frame.from_carla_xy` 를 거쳐 부른다. 이 목이 예전에 CARLA y 를
        그대로 받는 것처럼 굴어서, `_actor_lanes` 가 변환을 빠뜨린 버그를 오히려
        **고정하고 있었다** (2026-09-09 avoid_sim 으로 발견 — 실지도에서는
        blocked_by 가 늘 비어 free_run 이 항상 ahead_m 이었다).

        반환 필드도 실제 `LaneMatch` 와 같은 이름이어야 한다 — `.key` 가 아니라
        **`.lane`** 이다 (같은 버그의 다른 절반).
        """
        order = [L0, L1, L2]
        i = int(round(y / 3.0))
        if not (0 <= i < len(order)):
            return None
        return type('M', (), {'lane': order[i], 's': float(x), 't': 0.0})()


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


# ── 섹션 경계 (lane_map_span_sections_enable) ────────────────────────────
#
# `_lane_hops` 는 `lg.neighbor` 로 **자차 섹션 안**만 훑는데, 창(80 m)은 섹션을
# 여러 개 건넌다 — xodr 은 폭이 변할 때마다 섹션을 쪼개므로 11~13 m 짜리가 흔하다.
# 그러면 창 안의 정지 객체가 `(도로, 다른 섹션, 차로)` 로 투영돼 hop 밖으로 빠지고,
# free_run 이 전 차로 ahead_m 로 남아 blocked_by 가 통째로 빈다.
#
# 실측 2026-09-09 run_20260909_232350 t 55.6 rs 450.2: 자차 (2756,3,5), 정지
# 차량 3 대가 55.7~57.7 m 앞인데 lg.locate 는 (2756,0,3/4/5) — 섹션 3 개 앞이라
# 전부 빠졌다. blocked_by {}, lane_plan null → 방향을 옛 로직이 정보 없이 골랐다.
S0, S1 = (1, 0, -1), (1, 1, -1)


class LgSec(Lg3):
    """Lg3 에 **다음 섹션**을 붙인 목. 각 차로 30 m, x>30 은 섹션 1 로 잡힌다."""

    def __init__(self):
        super().__init__()
        s = np.linspace(0.0, 30.0, 7)
        for k in (L0, L1, L2):
            self.lanes[k].update({'s': s, 'width': np.full_like(s, 3.0),
                                  'length': 30.0,
                                  'next': [(1, 1, k[2])], 'prev': []})
        for k in (L0, L1, L2):
            nk = (1, 1, k[2])
            self.lanes[nk] = {'junction': -1, 'dir': -1, 's': s,
                              'width': np.full_like(s, 3.0), 'length': 30.0,
                              'next': [], 'prev': [k]}

    def locate(self, x, y, prefer=None, **kw):
        m = super().locate(x, y, prefer=prefer, **kw)
        if m is None or x <= 30.0:
            return m
        return type('M', (), {'lane': (1, 1, m.lane[2]),
                              's': float(x) - 30.0, 't': 0.0})()


def test_object_in_the_next_section_still_blocks_its_lane():
    """섹션이 달라도 승계로 이어진 같은 차로면 그 hop 이 막힌 것이다."""
    a = Box(2, 50.0, -3.0, 0.0)                 # 옆 차로 50 m 앞 = 섹션 1
    kr, p, ap = rig(on_cfg(), actors=[a], lg=LgSec())
    m = M(kr, p, ap)
    assert m['span_alias'] > 0
    assert m['blocked_by'][str(list(L1))] == 2
    assert fr(m, L1) == pytest.approx(50.0, abs=0.6)
    assert fr(m, L0) == 80.0                    # 내 차로는 그대로 비어 있다


def test_span_sections_off_reproduces_the_old_blind_map():
    """false = 이전 동작 — 다음 섹션의 객체를 못 보고 지도가 빈다."""
    a = Box(2, 50.0, -3.0, 0.0)
    kr, p, ap = rig(on_cfg(lane_map_span_sections_enable=False),
                    actors=[a], lg=LgSec())
    m = M(kr, p, ap)
    assert m['span_alias'] == 0
    assert m['blocked_by'] == {}
    assert fr(m, L1) == 80.0


def test_same_section_object_is_unchanged_by_the_alias():
    """섹션 안 객체는 별칭과 무관하게 예전 그대로 잡힌다."""
    a = Box(2, 20.0, -3.0, 0.0)
    kr, p, ap = rig(on_cfg(), actors=[a], lg=LgSec())
    m = M(kr, p, ap)
    assert m['blocked_by'][str(list(L1))] == 2
    assert fr(m, L1) == pytest.approx(20.0, abs=0.6)


def test_alias_stops_at_a_branch():
    """`next` 가 둘 이상이면 어느 쪽이 내 경로인지 모른다 — 따라가지 않는다."""
    lg = LgSec()
    lg.lanes[L1]['next'] = [(1, 1, -2), (1, 2, -2)]
    a = Box(2, 50.0, -3.0, 0.0)
    kr, p, ap = rig(on_cfg(), actors=[a], lg=lg)
    m = M(kr, p, ap)
    assert str(list(L1)) not in m['blocked_by']
    assert fr(m, L1) == 80.0


def test_alias_is_a_set_so_a_merge_blocks_both_lanes():
    """두 hop 차로가 같은 차로로 합류하면 **양쪽 다** 막힌 것이 맞다."""
    lg = LgSec()
    lg.lanes[L1]['next'] = [(1, 1, -3)]         # L1·L2 가 같은 차로로 합류
    a = Box(2, 50.0, -6.0, 0.0)                 # 섹션 1 의 그 합류 차로
    kr, p, ap = rig(on_cfg(), actors=[a], lg=lg)
    m = M(kr, p, ap)
    assert m['blocked_by'][str(list(L1))] == 2
    assert m['blocked_by'][str(list(L2))] == 2
