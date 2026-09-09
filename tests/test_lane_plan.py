"""[6] 차로 지도 커밋 B — 후보 선택·시점·감속 (`avoid_map.lane_map_*`).

커밋 A 의 `lane_map()` 이 재료고, 여기서 **어느 차로로 언제 옮길지**를 정한다.
개입은 둘뿐이다 — `min()` 에 덧대는 **상한형** 속도 후보와, 시프트 **방향 힌트**.
기존 게이트 7개는 그대로 통과해야 한다 (외부 오버라이드 금지).

왜 필요한가 — 실측 `20260909_093415` 에서 quick 7건 중 **4건이 같은 모양으로
죽었다**: 20~22 m 앞 정지 차량, `v_target = 0`, 30 s 무전진.
`tools/avoid_sim` 케이스 2·4 가 같은 상황을 재현하고, 현행 코드는 옆 차로로
한 칸만 시프트한 뒤 **23~25 초** 서 있는다 (3차로는 비어 있는데 못 간다).

지키는 것:
  · 트리거는 **내 차로**의 free_run 뿐이다. 다른 차로가 막힌 것은 트리거가 아니다.
  · 램프 길이는 상수가 아니라 전이 기하에서 나온다 (L = v·π·√(Y/2a)),
    a 는 `overtake.a_lat_max` 를 그대로 읽는다 (_shift_speed_cap 과 같은 축).
  · 후보 = passable ∧ free_run > 램프 + shift_ahead_m. 선택 = free_run 최대,
    동률이면 가까운 쪽.
  · 늦었으면 감속해 램프를 줄이고, 크립보다 느려야 하면 후보 없음(standoff).
  · 두 칸은 **한 번에** 연다 — 한 칸씩 가면 첫 전이가 끝나기 전에 서 버린다.
  · 스위치 off = 이전 동작 (진단 키도 안 생긴다).
"""
import copy
import math
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                        # noqa: E402
from test_avoid import Ap, Box, Planner                             # noqa: E402
from test_lane_map import L0, L1, L2, Lg3, off_cfg, on_cfg, rig     # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
A_LAT = CFG['overtake']['a_lat_max']
AHEAD = CFG['overtake']['shift_ahead_m']
DECIDE = CFG['avoid_map']['lane_map_decide_m']
V_CAP = CFG['avoid_map']['lane_map_avoid_speed_kph'] / 3.6


def obj(oid, x, y):
    """CARLA 좌표에 정지 객체 (목의 오른쪽 이웃은 CARLA y = −3·−6)."""
    return Box(oid, x, y, 0.0, half_w=0.9)


def plan(cfg, actors, **kw):
    kr, p, ap = rig(cfg, actors=actors, **kw)
    return kr, kr.lane_plan(ap, p)


# ── 램프 길이 ─────────────────────────────────────────────────────────────
def test_ramp_length_follows_the_transition_geometry():
    """L = v·π·√(Y / (2·a_lat_max)) — shift_route_smoothly 의 형상에서 나온다."""
    kr, _ = plan(on_cfg(), [])
    for hops, w, v in ((1, 3.0, 5.56), (2, 3.0, 5.56), (2, 3.5, 8.0)):
        Y = hops * w
        assert kr._ramp_len_m(hops, w, v) == pytest.approx(
            v * math.pi * math.sqrt(Y / (2.0 * A_LAT)))


def test_ramp_uses_the_shared_lateral_limit():
    """새 상수를 만들지 않는다 — overtake.a_lat_max 를 그대로 읽는다."""
    kr, _ = plan(on_cfg(), [])
    assert kr.a_lat_max == A_LAT


def test_ramp_grows_with_hops_and_speed():
    kr, _ = plan(on_cfg(), [])
    assert kr._ramp_len_m(2, 3.0, 5.0) > kr._ramp_len_m(1, 3.0, 5.0)
    assert kr._ramp_len_m(1, 3.0, 10.0) > kr._ramp_len_m(1, 3.0, 5.0)
    assert kr._ramp_len_m(0, 3.0, 5.0) == 0.0


# ── 트리거 ────────────────────────────────────────────────────────────────
def test_switch_off_makes_no_plan():
    kr, pl = plan(off_cfg(), [obj(2, 20.0, 0.0)])
    assert pl is None
    assert kr.last_lane_plan is None


def test_far_obstacle_is_not_a_trigger():
    """내 차로가 decide_m 밖까지 뚫려 있으면 개입하지 않는다."""
    _kr, pl = plan(on_cfg(), [obj(2, DECIDE + 20.0, 0.0)])
    assert pl is None


def test_other_lane_blocked_is_not_a_trigger():
    """다른 차로만 막힌 것은 트리거가 아니다 (avoid_sim 케이스 10)."""
    _kr, pl = plan(on_cfg(), [obj(2, 20.0, -6.0)])
    assert pl is None


def test_my_lane_blocked_triggers():
    _kr, pl = plan(on_cfg(), [obj(2, 25.0, 0.0)])
    assert pl is not None
    assert pl['trigger_free_m'] == pytest.approx(25.0, abs=1.0)


# ── 후보 선택 ─────────────────────────────────────────────────────────────
def test_picks_the_free_lane_one_hop_away():
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0)])
    assert pl['pick'] == str(list(L1))
    assert pl['hops'] == 1
    assert pl['side'] == 'right'


def test_skips_a_blocked_neighbour_and_takes_two_hops():
    """avoid_sim 케이스 2 의 배치 — 1·2차로가 막히고 3차로가 비었다."""
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0), obj(3, 42.0, -3.0)])
    assert pl['pick'] == str(list(L2))
    assert pl['hops'] == 2


def test_no_candidate_when_everything_is_blocked():
    """옆 차로들이 **램프도 못 넣을 만큼** 가까이 막혔으면 후보가 없다.

    (옆 차로가 내 차로보다 조금이라도 더 뚫려 있고 램프가 들어가면 그건
     정당한 후보다 — 40/42/42 는 후보가 있는 배치다.)
    """
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0), obj(3, 8.0, -3.0),
                              obj(4, 8.0, -6.0)])
    assert pl['pick'] is None
    assert pl['why'] == 'no_candidate'


def test_impassable_lane_is_not_a_candidate():
    """소멸 차로(폭 < min_width)는 후보가 아니다 — 옆 칸으로 밀린다."""
    narrow = Lg3(w2=0.5)
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0), obj(3, 8.0, -3.0)], lg=narrow)
    assert pl['pick'] is None                          # L1 은 가깝게 막히고 L2 는 좁다
    kr2, pl2 = plan(on_cfg(), [obj(2, 40.0, 0.0)], lg=narrow)
    assert pl2['pick'] == str(list(L1))                # L1 이 뚫렸으면 그쪽
    assert str(list(L2)) not in (pl2.get('cands') or {})


def test_candidate_needs_room_for_the_ramp():
    """free_run 이 램프 + shift_ahead_m 보다 짧으면 후보가 아니다."""
    kr, _ = plan(on_cfg(), [])
    need = kr._ramp_len_m(1, 3.0, V_CAP) + AHEAD
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0), obj(3, need - 2.0, -3.0)])
    assert pl['pick'] != str(list(L1))


# ── 시점·감속 ─────────────────────────────────────────────────────────────
def test_speed_cap_is_the_avoid_speed():
    _kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0)])
    assert pl['v_cap'] == pytest.approx(V_CAP, abs=0.01)


def test_late_decision_lowers_the_cap_instead_of_giving_up():
    """램프 시작점을 지났으면 **더 감속해** 램프를 줄인다."""
    kr, pl = plan(on_cfg(), [obj(2, 18.0, 0.0)])
    assert pl['pick'] is not None
    assert pl['late'] is True
    assert pl['v_cap'] < V_CAP


def test_too_late_gives_up_to_standoff():
    """크립보다 느려야 겨우 되는 램프면 후보 없음 — standoff 에 맡긴다."""
    _kr, pl = plan(on_cfg(), [obj(2, AHEAD + 0.5, 0.0)])
    assert pl['pick'] is None
    assert pl['why'] == 'ramp_too_late'


def test_start_s_is_before_the_first_obstacle():
    kr, pl = plan(on_cfg(), [obj(2, 45.0, 0.0)])
    ramp = pl['ramp_m']
    assert pl['start_s_rel'] == pytest.approx(45.0 - AHEAD - ramp, abs=1.0)


# ── 다단 시프트 ───────────────────────────────────────────────────────────
def test_two_hops_are_opened_at_once():
    """한 칸씩 두 번은 못 쓴다 — 첫 전이가 끝나기 전에 서 버린다."""
    kr, pl = plan(on_cfg(), [obj(2, 40.0, 0.0), obj(3, 42.0, -3.0)])
    assert kr._lm_hops('right') == 2
    assert kr._lm_hops('left') is None                 # 지도가 고른 쪽만


def test_one_hop_passes_no_step_override():
    """한 칸이면 route.py 에 n_steps 를 넘기지 않는다 (이전 서명 그대로)."""
    kr, _pl = plan(on_cfg(), [obj(2, 40.0, 0.0)])
    assert kr._lm_hops('right') is None


def test_hops_are_none_when_off():
    kr, _pl = plan(off_cfg(), [obj(2, 40.0, 0.0), obj(3, 42.0, -3.0)])
    assert kr._lm_hops('right') is None


# ── 큐는 그대로 억제된다 ─────────────────────────────────────────────────
def test_queue_is_still_suppressed():
    """신호 대기 줄은 지도에서 빠지므로 트리거되지 않는다 (억제 기준 단일 출처)."""
    _kr, pl = plan(on_cfg(), [obj(2, 25.0, 0.0)], queue=True)
    assert pl is None
