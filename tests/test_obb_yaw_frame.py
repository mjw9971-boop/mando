"""OBB 각의 프레임 (avoid_map.lane_map_obb_yaw_vtd_enable).

`frame` 규약은 `yaw_c = −heading_v` 다. 그런데 `_actor_lanes` 는 위치만
`from_carla_xy` 로 VTD 로 되돌리고 **yaw 는 CARLA 값을 그대로** 썼다. 그러면
OBB 가 `+heading_v` 가 아니라 `−heading_v` 로 놓인다 = 도로축 기준
**2·heading 만큼 돌아간 직사각형**이다.

옛 주석은 "둘레 전체의 합집합을 쓰므로 결과는 같다" 고 했다. **틀렸다** —
그 대칭이 성립하는 것은 yaw ≈ 0 · ±π/2 뿐이고, 하필 **도로와 나란한 객체**
(일반 교통류의 대부분)에서 오차가 가장 크다.

실측 run_20260910_230148 틱 1808 (t 105.0, rs 250.9), 차량 id 8
(4.394 × 1.808, heading_v 5.3829 ≡ −0.9003, 도로 −0.8599):
    올바른 Δ  −2.3°  → 횡폭 1.99 m → 막힌 차로 1개  (940,2,-2)
    옛   Δ  +100.9°  → 횡폭 4.65 m → 막힌 차로 3개  (940,2,-1)(-2)(-3)
셋이 다 막히자 후보가 −4 하나만 남았고, 목표를 갈아타는 사이 여유를 다 써
`ramp_too_late`(avail 1.7 < need 25.5)로 rs 256.2 에서 13 s 정지했다.
"""
import copy
import math
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter import frame
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)

# 틱 1808 실측
HEAD_V = 5.382895469665527
L, W = 4.394000053405762, 1.8079999685287476
ROAD = -0.8599451780319214


def cfg(on):
    c = copy.deepcopy(CFG)
    c['avoid_map']['lane_map_obb_yaw_vtd_enable'] = on
    return c


def obb_yaw(kr, yaw_deg):
    """`_actor_lanes` 가 OBB 각으로 쓰는 값 — 그 분기만 그대로 재현한다."""
    return (frame.from_carla_yaw_deg(yaw_deg) if kr.obb_yaw_vtd
            else math.radians(yaw_deg))


def lat_width(yaw, road, L=L, W=W):
    """OBB 를 도로 횡축에 투영한 **전체 폭** [m]."""
    d = yaw - road
    return abs(L * math.sin(d)) + abs(W * math.cos(d))


def test_frame_round_trip_recovers_the_vtd_heading():
    """CARLA 로 나갔다 온 yaw 가 원래 VTD heading 으로 돌아온다."""
    yaw_deg = frame.to_carla_yaw_deg(HEAD_V)
    got = frame.from_carla_yaw_deg(yaw_deg)
    assert math.cos(got - HEAD_V) == pytest.approx(1.0, abs=1e-9)
    assert math.sin(got - HEAD_V) == pytest.approx(0.0, abs=1e-9)


def test_off_reproduces_the_sign_flip():
    """off = 이전 동작. 부호가 뒤집힌 각을 쓴다 (결함을 고정한다)."""
    kr = KrRules(cfg(False))
    yaw_deg = frame.to_carla_yaw_deg(HEAD_V)
    assert math.cos(obb_yaw(kr, yaw_deg) + HEAD_V) == pytest.approx(1.0, abs=1e-9)


def test_on_uses_the_vtd_heading():
    kr = KrRules(cfg(True))
    yaw_deg = frame.to_carla_yaw_deg(HEAD_V)
    assert math.cos(obb_yaw(kr, yaw_deg) - HEAD_V) == pytest.approx(1.0, abs=1e-9)


def test_aligned_vehicle_is_one_lane_wide_when_fixed():
    """이 건의 알맹이 — 도로와 나란한 승용차는 한 차로 폭 안이어야 한다."""
    kr = KrRules(cfg(True))
    yaw_deg = frame.to_carla_yaw_deg(HEAD_V)
    w = lat_width(obb_yaw(kr, yaw_deg), ROAD)
    assert w == pytest.approx(1.99, abs=0.05)
    assert w < 3.2                                     # 차로폭


def test_off_spreads_the_same_vehicle_over_three_lanes():
    kr = KrRules(cfg(False))
    yaw_deg = frame.to_carla_yaw_deg(HEAD_V)
    w = lat_width(obb_yaw(kr, yaw_deg), ROAD)
    assert w == pytest.approx(4.65, abs=0.05)
    assert w > 3.2                                     # 두 경계를 넘는다 = 3차로


@pytest.mark.parametrize('road', [-2.5, -0.86, 0.0, 0.7, 2.0, 3.0])
def test_fixed_width_never_exceeds_the_diagonal(road):
    """수정본은 어느 도로각에서도 나란한 차를 대각선 이상으로 부풀리지 않는다."""
    kr = KrRules(cfg(True))
    yaw_deg = frame.to_carla_yaw_deg(road)             # 도로와 나란한 객체
    assert lat_width(obb_yaw(kr, yaw_deg), road) == pytest.approx(W, abs=1e-6)


def test_error_is_largest_for_aligned_objects():
    """오차가 **나란할 때 최대**라는 것 — 일반 교통류가 그대로 밟는다."""
    kr_on, kr_off = KrRules(cfg(True)), KrRules(cfg(False))
    worst = None
    for deg in range(0, 180, 5):
        head = ROAD + math.radians(deg)
        yd = frame.to_carla_yaw_deg(head)
        gap = (lat_width(obb_yaw(kr_off, yd), ROAD)
               - lat_width(obb_yaw(kr_on, yd), ROAD))
        if worst is None or gap > worst[1]:
            worst = (deg, gap)
    assert worst[0] in (0, 5, 175)                     # 나란한 쪽


# ── pad 는 이 건과 무관하다 (지시안의 전제 반증) ────────────────────────
@pytest.mark.parametrize('pad', [0.0, 0.2, 0.5])
def test_pad_does_not_touch_a_normal_vehicle(pad):
    """`hw = max(width/2, pad)` 라 폭 1.0 m 이상 객체에는 pad 가 안 닿는다.

    id 8 은 폭 1.808 → hw = max(0.904, pad) = 0.904 로 pad 값과 무관하다.
    """
    c = cfg(True)
    c['avoid_map']['lane_map_obj_pad_m'] = pad
    kr = KrRules(c)
    pts = kr._obb_samples(0.0, 0.0, 0.0, L, W)
    assert max(abs(y) for _x, y in pts) == pytest.approx(W / 2.0, abs=1e-9)


def test_pad_still_protects_a_thin_object():
    """얇은 물체(자전거 0.2 m)에는 그대로 걸린다 — 원래 목적."""
    c = cfg(True)
    c['avoid_map']['lane_map_obj_pad_m'] = 0.5
    kr = KrRules(c)
    pts = kr._obb_samples(0.0, 0.0, 0.0, 2.22, 0.20)
    assert max(abs(y) for _x, y in pts) == pytest.approx(0.5, abs=1e-9)
