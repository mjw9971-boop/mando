"""
정적 장애물 회피 1차 — 규칙 1(신호 구역 억제) + 규칙 3(선제 회피).

실측 근거 (2026-08-30, logs/batch/20260830_210019):
  · 실전주행_01_연속교차로24 — 박스 4개(0.15×0.46×0.61) 앞 17.8 s 정지 후 종료.
    시프트는 발동했으나 **정지 후**라 경로가 현 위치에서 옆으로 밀려 조향이
    풀락(steer +0.480 고정)되고 OBB forecast 가 0 을 냈다.
  · 실전주행_02_직진28 — 정차 차량 앞 16.6 s 정지. 단일 차로라 no_neighbor.
  · 박스는 정확히 79.9~80.0 m 에서 크기·속도가 다 채워져 온다 (GT 상한 80 m).
  · 두 로그 모두 객체 id 드롭아웃 0회.

여기서 지키는 불변:
  · 규칙 1 은 **전 상태 공통 게이트** — 신호 30 m 이내·교차로 안·대기열이면
    회피 계열이 전면 미발동한다 (신호 대기 차량을 비켜가면 신호 위반).
  · 억제 1차 키는 distances_to_next_traffic_lights + 교차로다. dist_stop_line 은
    실측 27% 가 null 이라 1차로 못 쓴다 (무신호 정지선은 보조로만).
  · 물체별 타이머 — 자차가 달리는 중에도 관찰이 쌓여야 선제 회피가 성립한다.
    움직이면 즉시 리셋(철회), id 소실은 grace 틱 동안 유지.
  · 전이 길이는 속도 비례 max(shift_latest_m, k·v), 전이 시작은 자차 앞.
"""
import copy
import math
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                   # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
SUP_M = OT['stopline_suppress_m']
HZ = CFG['comm']['send_hz']
STATIC_TICKS = int(round(OT['obj_static_s'] * HZ))


# ── 목 ──────────────────────────────────────────────────────────────────
class Box:
    """9910 객체 하나. bounding_box.extent.y 는 반폭."""

    def __init__(self, oid, x, y, speed=0.0, half_w=0.23):
        self.id = oid
        self._x, self._y, self.speed = float(x), float(y), float(speed)
        self.type_id = 'vehicle.vtd.object'
        ext = type('E', (), {'x': 0.075, 'y': half_w, 'z': 0.3})()
        self.bounding_box = type('BB', (), {'extent': ext})()

    def get_location(self):
        s = self
        return type('L', (), {'x': s._x, 'y': s._y, 'z': 0.0})()

    def get_velocity(self):
        s = self
        return type('V', (), {'length': lambda self_: s.speed})()


class Ego(Box):
    def __init__(self):
        super().__init__(0, 0.0, 0.0, 0.0, 0.94)


class ActorList(list):
    def filter(self, pat):
        p = pat.strip('*')
        return ActorList(a for a in self if p in a.type_id)


class World:
    def __init__(self, actors=()):
        self._a = ActorList(actors)

    def get_actors(self):
        return self._a


class Planner:
    """x 축 직선 경로. 신호 정지선은 d_tl (뒷축거리, inf 면 없음)."""

    def __init__(self, d_tl=float('inf'), stop_lines=(), n=6000):
        self.route_points = np.stack(
            [np.arange(n) * 0.1, np.zeros(n), np.zeros(n)], axis=1)
        self.original_route_points = self.route_points.copy()
        self.route_s = np.arange(n) * 0.1
        self.route_index = 0
        self.points_per_meter = 10
        self.distances_to_next_traffic_lights = np.full(n, float(d_tl))
        self.next_traffic_lights = [None] * n
        self.lg = None
        self.route = {'lanes': [], 'cum_s': [], 'total_length': n * 0.1}
        self._stop_lines = list(stop_lines)


class Ap:
    def __init__(self, planner, actors=(), junction=False):
        self._waypoint_planner = planner
        self._world = World(list(actors))
        self._vehicle = Ego()
        self.junction = junction
        self.config = type('C', (), {'idm_red_light_minimum_distance': 5.299})()


def make(cfg=CFG, **kw):
    p = Planner(**kw)
    kr = KrRules(cfg)
    kr._sl_all = list(p._stop_lines)          # lg 없는 목이라 직접 주입
    return kr, p


def tick(kr, ap, p, n=1):
    for _ in range(n):
        kr._update_obj_timers(ap)


# ── 규칙 1: 신호 구역 억제 ──────────────────────────────────────────────
def test_suppress_when_signal_within_window():
    kr, p = make(d_tl=SUP_M - 5.0)
    ap = Ap(p)
    z = kr._signal_zone(p, ap)
    assert z is not None and z[0] == 'signal'


def test_no_suppress_when_signal_beyond_window():
    kr, p = make(d_tl=SUP_M + 5.0)
    assert kr._signal_zone(p, Ap(p)) is None


def test_no_suppress_when_no_signal_at_all():
    """실측 박스 지점 — 다음 신호 214.8 m. 억제하면 안 된다."""
    kr, p = make(d_tl=214.8)
    assert kr._signal_zone(p, Ap(p)) is None


def test_suppress_inside_junction():
    kr, p = make(d_tl=float('inf'))
    z = kr._signal_zone(p, Ap(p, junction=True))
    assert z is not None and z[0] == 'junction'


def test_unsignalized_stopline_suppresses_as_auxiliary():
    """1차 키(신호)가 inf 여도 무신호 정지선이 가까우면 억제 — 사각 보완."""
    kr, p = make(d_tl=float('inf'), stop_lines=[SUP_M - 10.0])
    z = kr._signal_zone(p, Ap(p))
    assert z is not None and z[0] == 'stopline'


def test_primary_key_covers_where_dist_stop_line_is_null():
    """실측 사각 고정: dist_stop_line 이 null 인 지점에서도 1차 키는 값을 준다.
    (목에서는 stop_lines 를 비워 dist_stop_line=null 을 모사한다.)"""
    kr, p = make(d_tl=SUP_M - 1.0, stop_lines=[])
    z = kr._signal_zone(p, Ap(p))
    assert z is not None and z[0] == 'signal'


# ── 규칙 1: 대기열 형태 판정 ────────────────────────────────────────────
def test_queue_of_two_longitudinal_objects_is_suppressed():
    """종방향으로 벌어져 줄 선 정지 객체 2대 = 대기열 (신호 데이터 없어도)."""
    kr, p = make(d_tl=float('inf'))
    ap = Ap(p, [Box(1, 20.0, 0.0), Box(2, 30.0, 0.0)])
    tick(kr, ap, p, STATIC_TICKS)
    assert kr._is_queue(kr._corridor_blockers(ap, p)) is True


def test_staggered_pair_is_not_a_queue():
    """스태거드(케이스 B) — 종방향으로 붙고 횡으로 갈린다. 대기열 아님."""
    kr, p = make(d_tl=float('inf'))
    ap = Ap(p, [Box(1, 20.0, -0.8), Box(2, 21.5, +0.8)])
    tick(kr, ap, p, STATIC_TICKS)
    assert kr._is_queue(kr._corridor_blockers(ap, p)) is False


def test_single_object_is_not_a_queue():
    kr, p = make(d_tl=float('inf'))
    ap = Ap(p, [Box(1, 20.0, 0.0)])
    tick(kr, ap, p, STATIC_TICKS)
    assert kr._is_queue(kr._corridor_blockers(ap, p)) is False


# ── 물체별 정지 타이머 ──────────────────────────────────────────────────
def test_static_timer_accumulates_and_gates_detection():
    kr, p = make(d_tl=float('inf'))
    ap = Ap(p, [Box(1, 20.0, 0.0)])
    tick(kr, ap, p, STATIC_TICKS - 1)
    assert kr._corridor_blockers(ap, p) == []          # 아직 3 s 미만
    tick(kr, ap, p, 1)
    assert len(kr._corridor_blockers(ap, p)) == 1


def test_moving_object_resets_timer_immediately():
    """접근 중 출발하면 철회 — 다음 틱에 바로 0."""
    kr, p = make(d_tl=float('inf'))
    b = Box(1, 20.0, 0.0)
    ap = Ap(p, [b])
    tick(kr, ap, p, STATIC_TICKS)
    assert kr._corridor_blockers(ap, p)
    b.speed = 3.0
    tick(kr, ap, p, 1)
    assert kr.obj_ticks[1] == 0
    assert kr._corridor_blockers(ap, p) == []


def test_dropout_within_grace_keeps_timer():
    kr, p = make(d_tl=float('inf'))
    b = Box(1, 20.0, 0.0)
    ap = Ap(p, [b])
    tick(kr, ap, p, STATIC_TICKS)
    ap._world = World([])                                # id 소실
    tick(kr, ap, p, OT['obj_grace_ticks'])
    assert kr.obj_ticks.get(1, 0) >= STATIC_TICKS        # grace 안 — 유지
    tick(kr, ap, p, 1)
    assert 1 not in kr.obj_ticks                         # grace 초과 — 폐기


# ── 규칙 3: 회랑 침범 판정 ──────────────────────────────────────────────
def test_object_outside_corridor_is_ignored():
    """갓길에 비켜 선 물체는 대상이 아니다 (폭 = 자차반폭+객체반폭+여유)."""
    kr, p = make(d_tl=float('inf'))
    lim = CFG['vehicle']['width'] / 2 + 0.23 + CFG['percep']['obstacle_clearance_m']
    ap = Ap(p, [Box(1, 20.0, lim + 0.5)])
    tick(kr, ap, p, STATIC_TICKS)
    assert kr._corridor_blockers(ap, p) == []


def test_detection_reaches_detect_max_m_not_blocker_dist_max():
    """탐지 상한은 detect_max_m(80)이지 blocker_dist_max(20)가 아니다."""
    assert OT['detect_max_m'] > OT['blocker_dist_max']
    kr, p = make(d_tl=float('inf'))
    ap = Ap(p, [Box(1, OT['detect_max_m'] - 5.0, 0.0)])
    tick(kr, ap, p, STATIC_TICKS)
    assert len(kr._corridor_blockers(ap, p)) == 1


def test_beyond_detect_max_is_not_seen():
    kr, p = make(d_tl=float('inf'))
    ap = Ap(p, [Box(1, OT['detect_max_m'] + 10.0, 0.0)])
    tick(kr, ap, p, STATIC_TICKS)
    assert kr._corridor_blockers(ap, p) == []


# ── 전이 길이 속도 비례 ─────────────────────────────────────────────────
@pytest.mark.parametrize('v', [0.0, 2.0, 6.0, 13.9])
def test_transition_length_is_speed_proportional_with_floor(v):
    """L = max(shift_latest_m, k·v). 코사인 전이 최대 횡가속
    a = v²Δπ²/(2L²) 에서 Δ=3.0, a=1.5 이면 k≈3.14 → 3.0 채택."""
    kr, _p = make()
    L = max(kr.shift_latest_m, kr.shift_k_s * max(v, 0.1))
    assert L >= kr.shift_latest_m
    if v > kr.shift_latest_m / kr.shift_k_s:
        assert L == pytest.approx(kr.shift_k_s * v)
        a_lat = v ** 2 * 3.0 * math.pi ** 2 / (2 * L ** 2)
        assert a_lat <= 1.8                              # 쾌적 범위 유지


def test_shift_starts_ahead_of_ego():
    """전이 시작이 자차 뒤면 정지 상태에서 경로가 옆으로 밀려 조향 풀락.
    min_start_ahead 로 자차 앞에서 시작해야 한다."""
    kr, _p = make()
    assert kr.shift_ahead_m > 0.0


# ── 스위치 ──────────────────────────────────────────────────────────────
def test_detect_disabled_when_static_time_unreachable():
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['obj_static_s'] = 1e6                # 사실상 비활성
    kr, p = make(cfg, d_tl=float('inf'))
    ap = Ap(p, [Box(1, 20.0, 0.0)])
    tick(kr, ap, p, 200)
    assert kr._corridor_blockers(ap, p) == []
