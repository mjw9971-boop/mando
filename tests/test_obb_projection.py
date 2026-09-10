"""
가로누운 물체를 못 보던 문제 — 객체를 **한 점 + 폭**으로만 봤다.

실측 2026-09-10 run_20260910_185406 rs 170.0, 32 s 정지:
  자전거 id 21, 2.22 × 0.20 m, heading 0.00 rad. 도로는 1.7253 rad —
  **98.9°, 거의 직각으로 누워 있다.**

두 곳이 같은 이유로 눈이 멀었다:

  ① `_actor_lanes` — 중심 + 좌우 폭 **3점**만 투영했다. 폭이 0.2 m 라 세 점이
     0.2 m 안에 몰려 옆 차로에만 붙었고, 자차 차로 free_run 이 80.0 으로 남아
     `lane_plan` 트리거(< decide_m)가 영영 안 걸렸다. **길이축이 표본에 없었다.**
  ② `_corridor_blockers` — `hw = bounding_box.extent.y`(= 폭/2)로 회랑 침범을
     쟀다. 그건 객체가 경로와 나란할 때만 맞다. 가로누우면 횡으로 길이/2 를
     차지하는데 폭/2 로 재니 회랑 밖으로 나간다 → standoff·PREEMPT 가 한 번도
     안 섰다 (①만 고치면 pick 은 뜨지만 `_try_overtake` 가 대상을 못 찾는다).

정렬된 객체(Δ≈0)에서는 새 식이 **정확히 옛 값으로 환원된다**:
    |L/2·sin 0| + |W/2·cos 0| = W/2
그래서 일반 교통류에서는 동작이 그대로다 (replay 42,673틱 불일치 0으로 확인).
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

from kr_rules import KrRules                                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)

# run_20260910_185406 rs 170.0 실값 (VTD 프레임)
BIKE = dict(vx=1527.567626953125, vy=7.239490985870361, yaw=0.0, L=2.22, W=0.20)
ROAD_YAW = 1.7252850532531738


class BB:
    def __init__(self, hl, hw):
        self.extent = type('E', (), {'x': hl, 'y': hw, 'z': 0.6})()


class A:
    def __init__(self, L, W, yaw_deg):
        self.bounding_box = BB(L / 2.0, W / 2.0)
        self.yaw_deg = yaw_deg


def kr(**over):
    c = copy.deepcopy(CFG)
    c['avoid_map'].update(over)
    return KrRules(c)


# ── ① OBB 둘레 표본 ──────────────────────────────────────────────────────
def test_defaults():
    assert float(CFG['avoid_map']['lane_map_obb_step_m']) == 1.0
    assert float(CFG['avoid_map']['lane_map_obj_pad_m']) == 0.5


def test_samples_cover_the_length_axis():
    """옛 3점에는 길이축이 아예 없었다 — 이제 들어간다."""
    k = kr()
    pts = k._obb_samples(0.0, 0.0, 0.0, 4.4, 1.8)
    xs = [p[0] for p in pts]
    assert min(xs) == pytest.approx(-2.2) and max(xs) == pytest.approx(2.2)


def test_sample_count_is_small():
    """비용이 무시할 수준인지 — 자전거 ≈ 9점, 승용차 ≈ 14점."""
    k = kr()
    bike = k._obb_samples(0, 0, 0, 2.22, 0.20)
    car = k._obb_samples(0, 0, 0, 4.4, 1.8)
    assert 6 <= len(bike) <= 12, len(bike)
    assert 10 <= len(car) <= 18, len(car)


def test_step_controls_density_and_never_skips_the_middle():
    """모서리만 찍으면 긴 물체가 가운데 차로를 건너뛴다 — 변을 샘플한다."""
    k = kr()
    long_obj = k._obb_samples(0.0, 0.0, 0.0, 7.0, 0.3)
    xs = sorted({round(p[0], 3) for p in long_obj})
    gaps = [b - a for a, b in zip(xs, xs[1:])]
    assert gaps and max(gaps) <= 1.0 + 1e-6, gaps


def test_pad_floors_a_very_narrow_object():
    k0, k1 = kr(lane_map_obj_pad_m=0.0), kr(lane_map_obj_pad_m=0.5)
    ys0 = [p[1] for p in k0._obb_samples(0, 0, 0, 2.22, 0.20)]
    ys1 = [p[1] for p in k1._obb_samples(0, 0, 0, 2.22, 0.20)]
    assert max(ys0) == pytest.approx(0.10)
    assert max(ys1) == pytest.approx(0.50)


def test_pad_does_not_shrink_a_wide_object():
    k = kr(lane_map_obj_pad_m=0.5)
    ys = [p[1] for p in k._obb_samples(0, 0, 0, 4.4, 1.8)]
    assert max(ys) == pytest.approx(0.9)          # 폭/2 가 더 크면 그쪽


# ── ② 경로 횡방향 반폭 ───────────────────────────────────────────────────
def test_aligned_object_reduces_exactly_to_half_width():
    """정렬(Δ=0)이면 옛 값과 **정확히** 같다 — 그래서 일반 교통류가 안 바뀐다."""
    k = kr()
    for L, W in ((4.4, 1.8), (2.22, 0.20), (12.0, 2.5)):
        a = A(L, W, 0.0)
        assert k._lat_half_extent(a, 0.0) == pytest.approx(W / 2.0)


def test_perpendicular_object_uses_half_length():
    k = kr()
    a = A(2.22, 0.20, 90.0)
    assert k._lat_half_extent(a, 0.0) == pytest.approx(2.22 / 2.0)


def test_the_log_case_becomes_detectable():
    """로그 실값 — 옛 식은 회랑 밖, 새 식은 안."""
    k = kr()
    a = A(BIKE['L'], BIKE['W'], math.degrees(BIKE['yaw']))
    hw_new = k._lat_half_extent(a, ROAD_YAW)
    hw_old = BIKE['W'] / 2.0
    half_ego = float(CFG['vehicle']['width']) / 2.0
    clr = float(CFG['percep']['obstacle_clearance_m'])
    lat = 1.68                                    # 로그의 자차 차로 중심 대비 횡거리
    assert hw_old + half_ego + clr < lat, '옛 식은 못 잡아야 이 회귀가 성립한다'
    assert hw_new + half_ego + clr > lat, (hw_new, half_ego, clr)
    assert hw_new == pytest.approx(1.10, abs=0.02)


def test_missing_bounding_box_falls_back():
    k = kr()
    assert k._lat_half_extent(object(), 0.0) == pytest.approx(0.9)


# ── ③ PDM hazard 동기화 ──────────────────────────────────────────────────
def test_pdm_hazard_reads_the_final_tuple():
    k = kr()
    ap = type('AP', (), {'final': (False, 0.0, [3.2, 'vehicle.vtd.object', 21, 6.7])})()
    assert k._pdm_hazard(ap) == (21, 6.7)


def test_pdm_hazard_ignores_lights_and_signs():
    k = kr()
    for kind in ('traffic.traffic_light', 'traffic.stop_sign'):
        ap = type('AP', (), {'final': (False, 0.0, [3.2, kind, 9, 30.0])})()
        assert k._pdm_hazard(ap) is None


def test_pdm_hazard_is_none_without_the_hook():
    """훅이 없는 환경(기본 AutoPilot·테스트 목)에서는 동작이 그대로다."""
    k = kr()
    assert k._pdm_hazard(object()) is None
    assert k._pdm_hazard(type('AP', (), {'final': None})()) is None
    assert k._pdm_hazard(type('AP', (), {'final': (False, 0.0, None)})()) is None


def test_run_agent_clears_final_every_tick():
    """`keep_driving` 분기에서 직전 틱 값이 얼어붙으면 지나간 객체로 차로를 막는다."""
    src = (ROOT / 'run_agent.py').read_text()
    i = src.index('def _manage_route_obstacle_scenarios')
    blk = src[i:i + 1200]
    assert 'self.final = None' in blk
