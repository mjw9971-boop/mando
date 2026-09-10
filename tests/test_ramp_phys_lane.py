"""
(4) 램프 한복판의 제자리 감속 — `lane_map_ramp_by_phys_lane_enable`.

실측 2026-09-10 run_20260910_121651, 2칸 좌 시프트를 마친 직후 t 63.6~65.7:

    winner=route_end 로 찍혔지만 실제 승자는 `lane_plan.v_cap` 1.12 였다
    (로그의 `route_end` 슬롯은 kr 후보 **전체의 min** 하나다 — 그래서 같은
    커밋에서 `reasons.kr_cands`·`kr_win` 을 남긴다).

    물리 차로 (2756,1,3) free_run 80.0   ← 앞이 완전히 비었다
    복귀 차로 (2756,1,5) free_run  8.7   ← 방금 비켜 온 정지차 id 3
    → first 8.7 → late → avail 3.7 → v_cap 1.12 m/s
    그 8 틱 동안 v 4.33 → 1.93.

`lane_map_shift_ref_enable` 의 목적은 **트리거**를 살리는 것이다 (밀린 경로
위에서는 물리 차로가 비어 보여 트리거가 영영 안 걸린다). 그 값이 램프 길이와
v_cap 까지 정하면, 이미 비켜 나온 차로의 잔여 거리로 램프를 재게 되어
아무것도 사지 못하는 감속이 나온다. 트리거는 그대로 두고 램프만 가른다.
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

from kr_rules import KrRules                                       # noqa: E402
from test_avoid import Ap, Planner                                 # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)

# 로그 그대로: road 2756 섹션 1, 자차는 lane 3, 복귀 차로는 lane 5 (hop +2)
EGO   = (2756, 1, 3)
LEFT  = (2756, 1, 2)      # hop -1  — 목표 후보
RIGHT = (2756, 1, 4)      # hop +1
RET   = (2756, 1, 5)      # hop +2  — 비켜 나온 차로
ORDER = [LEFT, EGO, RIGHT, RET]
LANE_W = 3.3

# 로그 실값 (run_20260910_121651 t 65.72 avoid.lane_map)
FREE = {str(list(EGO)): 80.0, str(list(LEFT)): 80.0,
        str(list(RIGHT)): 6.6, str(list(RET)): 8.7}
HOPS = {str(list(EGO)): 0, str(list(LEFT)): -1,
        str(list(RIGHT)): 1, str(list(RET)): 2}


class Lg4:
    def __init__(self):
        s = np.linspace(0.0, 1000.0, 51)
        self.lanes = {k: {'junction': -1, 'dir': -1, 's': s,
                          'width': np.full_like(s, LANE_W), 'length': 1000.0}
                      for k in ORDER}

    def neighbor(self, key, side):
        if key not in ORDER:
            return None
        i = ORDER.index(key) + (1 if side == 'right' else -1)
        return ORDER[i] if 0 <= i < len(ORDER) else None

    def length(self, key):
        return 1000.0

    def locate(self, x, y, prefer=None, **kw):
        return type('M', (), {'lane': EGO, 's': float(x), 't': 0.0})()


def rig(phys):
    c = copy.deepcopy(CFG)
    c['avoid_map']['lane_map_avoid_enable'] = True
    c['avoid_map']['lane_map_shift_ref_enable'] = True
    c['avoid_map']['lane_map_ramp_by_phys_lane_enable'] = phys
    kr = KrRules(c)
    kr._sl_all = []
    p = Planner(d_tl=float('inf'))
    p.lg = Lg4()
    ap = Ap(p, actors=[])
    ap._kr_ego_lane = EGO
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    kr._ap = ap
    kr._tick_ego_lane = EGO
    kr._tick_queue = False
    kr._tick_corridor = []
    # 시프트가 진행 중이고, 비켜 나온 쪽은 오른쪽 2칸 (좌 2칸 시프트)
    kr.ot_span = (4716, 5492)
    kr.ot_from_hop = 2
    kr.lane_map = lambda _ap, _p: {
        'ego_lane': list(EGO), 'hops': dict(HOPS), 'free_run': dict(FREE),
        'passable': {k: True for k in FREE}, 'blocked_by': {}, 'queue_lanes': [],
    }
    return kr, p, ap


def plan(phys):
    kr, p, ap = rig(phys)
    return kr.lane_plan(ap, p)


def test_default_is_previous_behaviour():
    assert CFG['avoid_map']['lane_map_ramp_by_phys_lane_enable'] is False


def test_trigger_still_uses_the_return_lane_both_ways():
    """트리거는 안 건드린다 — 물리 차로 80.0 만 보면 계획 자체가 안 선다."""
    for phys in (False, True):
        pl = plan(phys)
        assert pl is not None, phys
        assert pl['trigger_free_m'] == pytest.approx(8.7, abs=0.05), phys


def test_off_reproduces_the_1_12_crawl():
    """이전 동작 — 로그의 v_cap 1.12 · late · ramp 3.7 이 그대로 나온다."""
    pl = plan(False)
    assert pl['pick'] == str(list(LEFT))
    assert pl['late'] is True
    assert pl['ramp_m'] == pytest.approx(3.7, abs=0.05)
    assert pl['v_cap'] == pytest.approx(1.12, abs=0.02)


def test_on_keeps_the_avoid_cap_and_full_ramp():
    """수정 후 — 물리 차로가 80.0 이므로 늦지 않고 상한은 회피 속도 그대로."""
    pl = plan(True)
    assert pl['pick'] == str(list(LEFT))
    assert pl['late'] is False
    assert pl['v_cap'] == pytest.approx(CFG['avoid_map']['lane_map_avoid_speed_kph'] / 3.6,
                                        abs=0.02)
    assert pl['ramp_m'] > 3.7


def test_on_does_not_change_anything_when_not_shifting():
    """시프트 중이 아니면 복귀 차로 참조 자체가 없어 두 설정이 같다."""
    out = []
    for phys in (False, True):
        kr, p, ap = rig(phys)
        kr.ot_span = None
        kr.ot_from_hop = None
        out.append(kr.lane_plan(ap, p))
    assert out[0] == out[1]


def test_on_does_not_change_anything_when_shift_ref_is_off():
    """shift_ref 가 꺼져 있으면 mine 이 애초에 물리 차로다 — 두 설정이 같다."""
    out = []
    for phys in (False, True):
        c = copy.deepcopy(CFG)
        c['avoid_map']['lane_map_avoid_enable'] = True
        c['avoid_map']['lane_map_shift_ref_enable'] = False
        c['avoid_map']['lane_map_ramp_by_phys_lane_enable'] = phys
        kr, p, ap = rig(phys)
        kr.lm_shift_ref = False
        out.append(kr.lane_plan(ap, p))
    assert out[0] == out[1]


# ── (4)(a) 후보별 진단 ────────────────────────────────────────────────────
# 로그의 `route_end` 슬롯 하나로는 승자를 가릴 수 없었다. `reasons.kr_cands` 가
# 후보별 값을, `kr_win` 이 그중 최종 승자 이름을 남긴다 (관측 전용 — 중재 불변).

from vtd_adapter.carla_types import VehicleControl                 # noqa: E402
from test_ped_crosswalk import rig as cw_rig, FRONT, WAIT_TICKS    # noqa: E402


def test_kr_cands_names_the_winner():
    kr, p, ap, _w = cw_rig()
    p.route_index = int((20.0 - FRONT - 4.0) * 10) + 2
    ap._vehicle.speed = 0.0
    for _ in range(WAIT_TICKS):
        ap._longitudinal_controller.get_throttle_and_brake(False, 12.5, 0.0)
        _c, t = kr.apply(VehicleControl(), 12.5, ap)
    assert kr.last_kr_cands, '후보가 하나도 기록되지 않았다'
    # 승자 이름이 실제 최저 후보와 같은 값을 가리켜야 한다
    assert kr.last_kr_win in kr.last_kr_cands
    assert kr.last_kr_cands[kr.last_kr_win] == min(kr.last_kr_cands.values())
    assert kr.last_kr_cands[kr.last_kr_win] == pytest.approx(round(float(kr.last_candidate), 2))


def test_kr_cands_is_cleared_when_no_candidate_lives():
    """후보가 하나도 없는 틱은 키가 안 생긴다 (로그에서 '미구현' 과 구분)."""
    kr, p, ap, _w = cw_rig()
    kr.last_kr_cands = {'stale': 1.0}
    kr.last_kr_win = 'stale'
    p.route_index = 210                                   # 보행자를 지났다
    ap._longitudinal_controller.get_throttle_and_brake(False, 12.5, 0.0)
    kr.apply(VehicleControl(), 12.5, ap)
    assert kr.last_kr_cands is None or 'stale' not in kr.last_kr_cands
