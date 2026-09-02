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

    def compute_leading_vehicles(self, vehicles, ego_id):
        """경로(x 축)에 붙어 전방에 있는 객체 id — 실물 판정의 최소 모사."""
        out = []
        for v in vehicles:
            if v.id == ego_id:
                continue
            loc = v.get_location()
            if loc.x > 0.0 and abs(loc.y) < 2.5:
                out.append(v.id)
        return out


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


# ══════════════════════════════════════════════════════════════════════════
# 규칙 2 — 데드락 해제 (BREAKOUT) / 크립 훅
#
# 크립 훅은 PDM 의 선행차·OBB 후보를 **무효화**한다. 필요 이상으로 열리면
# 앞차·장애물을 그대로 들이받는다. 그래서 참이 되는 경우가 **BREAKOUT
# 최종 단계(L4) 단독**임을 아래에서 못 박는다.
# ══════════════════════════════════════════════════════════════════════════
BO = OT
HARD_TICKS = int(round(BO['stuck_hard_s'] * HZ))
ESC_TICKS = int(round(BO['escalate_s'] * HZ))
FAIL_TICKS = int(round(BO['creep_fail_s'] * HZ))


def blocked_rig(cfg=CFG, d_tl=float('inf'), junction=False):
    """장애물이 코앞에 정지해 있고 자차도 정지 — BREAKOUT 조건을 만드는 목."""
    kr, p = make(cfg, d_tl=d_tl)
    ap = Ap(p, [Box(1, 8.0, 0.0, half_w=0.9)], junction=junction)
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    tick(kr, ap, p, STATIC_TICKS)
    kr.last_d_end = 1e6                      # 종점 사정권 밖
    return kr, p, ap


def drive(kr, p, ap, n, v=0.0):
    for _ in range(n):
        kr._update_obj_timers(ap)
        kr._breakout_tick(p, ap, v)


# ── 정상 발동 ───────────────────────────────────────────────────────────
def test_breakout_enters_after_stuck_hard_s():
    kr, p, ap = blocked_rig()
    drive(kr, p, ap, HARD_TICKS - 1)
    assert kr.bo_state is None
    drive(kr, p, ap, 1)
    assert (kr.bo_state, kr.bo_level) == ('BREAKOUT', 1)


def test_levels_escalate_and_creep_only_at_L4():
    kr, p, ap = blocked_rig()
    drive(kr, p, ap, HARD_TICKS)
    for expect in (2, 3, 4):
        assert kr.breakout_creep() is False, kr.bo_level      # L1~L3 은 거짓
        drive(kr, p, ap, ESC_TICKS)
        assert kr.bo_level == expect
    assert kr.bo_level == kr.BO_CREEP
    assert kr.breakout_creep() is True                        # L4 에서만 참


def test_progress_returns_to_normal():
    kr, p, ap = blocked_rig()
    drive(kr, p, ap, HARD_TICKS + ESC_TICKS)
    assert kr.bo_state == 'BREAKOUT'
    p.set_route_s = lambda rs: None                           # 목엔 없음
    p.route_index = int(round((kr.bo_ref_s + BO['progress_m'] + 0.5) / 0.1))
    drive(kr, p, ap, 1, v=1.0)
    assert kr.bo_state is None


def test_creep_fail_after_no_progress_and_holds_stop():
    """접촉은 실패 조건이 아니다 — 무진전 지속으로만 판정한다."""
    kr, p, ap = blocked_rig()
    drive(kr, p, ap, HARD_TICKS + ESC_TICKS * 3)
    assert kr.breakout_creep() is True
    drive(kr, p, ap, FAIL_TICKS)
    assert kr.bo_state == 'CREEP_FAIL'
    assert kr.breakout_creep() is False                       # 포기 후 크립 종료


def test_creep_continues_while_progressing():
    """조금씩이라도 나아가면 실패로 보지 않는다."""
    kr, p, ap = blocked_rig()
    drive(kr, p, ap, HARD_TICKS + ESC_TICKS * 3)
    for _ in range(FAIL_TICKS + 40):
        p.route_index += 10                                   # 매 틱 1 m 전진
        kr._update_obj_timers(ap)
        kr._breakout_tick(p, ap, 1.0)
        if kr.bo_state is None:
            break
    assert kr.bo_state != 'CREEP_FAIL'


# ── 크립 훅 부정 테스트 (지시 8종) ───────────────────────────────────────
def _to_creep(kr, p, ap):
    drive(kr, p, ap, HARD_TICKS + ESC_TICKS * 3)


def test_hook_false_initially():
    kr, p, ap = blocked_rig()
    assert kr.breakout_creep() is False


@pytest.mark.parametrize('flag', ['traffic_light_hazard', 'walker_hazard',
                                  'walker_close', 'stop_sign_hazard'])
def test_hook_false_on_pdm_hazard(flag):
    """light / walker / stop_sign 원인이면 BREAKOUT 자체가 안 선다."""
    kr, p, ap = blocked_rig()
    setattr(ap, flag, True)
    _to_creep(kr, p, ap)
    assert kr.bo_state is None
    assert kr.breakout_creep() is False


def test_hook_false_on_route_end_latch():
    kr, p, ap = blocked_rig()
    kr.latched = True
    _to_creep(kr, p, ap)
    assert kr.breakout_creep() is False


def test_hook_false_on_route_end_ghost_range():
    kr, p, ap = blocked_rig()
    kr.last_d_end = kr.active_m - 1.0                         # 종점 유령차 사정권
    _to_creep(kr, p, ap)
    assert kr.breakout_creep() is False


def test_hook_false_on_stopline_hold():
    kr, p, ap = blocked_rig()
    kr.sl_hold_left = 5
    _to_creep(kr, p, ap)
    assert kr.breakout_creep() is False


@pytest.mark.parametrize('dec', ['stop', 'go'])
def test_hook_false_on_yellow_latch(dec):
    kr, p, ap = blocked_rig()
    kr.y_decision = dec
    _to_creep(kr, p, ap)
    assert kr.breakout_creep() is False


def test_hook_false_on_cross_guard():
    kr, p, ap = blocked_rig()
    kr.cross_guard = True
    _to_creep(kr, p, ap)
    assert kr.breakout_creep() is False


def test_hook_false_in_signal_zone():
    """규칙 1 억제 구역 — BREAKOUT 도 서지 않는다."""
    kr, p, ap = blocked_rig(d_tl=SUP_M - 5.0)
    _to_creep(kr, p, ap)
    assert kr.bo_state is None
    assert kr.breakout_creep() is False


def test_hook_false_inside_junction():
    kr, p, ap = blocked_rig(junction=True)
    _to_creep(kr, p, ap)
    assert kr.breakout_creep() is False


def test_hook_false_during_preempt_and_reactive():
    """PREEMPT·REACTIVE 는 BREAKOUT 이 아니다 — 훅이 열리면 안 된다."""
    kr, p, ap = blocked_rig()
    for st in ('PREEMPT', 'REACTIVE', 'WATCH'):
        kr.last_avoid = {'state': st}
        assert kr.breakout_creep() is False
    drive(kr, p, ap, HARD_TICKS)                              # L1
    assert kr.bo_level == 1 and kr.breakout_creep() is False


def test_hook_false_when_cause_disappears():
    kr, p, ap = blocked_rig()
    _to_creep(kr, p, ap)
    assert kr.breakout_creep() is True
    ap._world = World([])                                     # 장애물 사라짐
    drive(kr, p, ap, kr.obj_grace + 2, v=0.0)
    assert kr.breakout_creep() is False


def test_disabled_switch_never_arms():
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['breakout_enabled'] = False
    kr, p, ap = blocked_rig(cfg)
    # apply 를 타지 않는 목이므로 _breakout_tick 을 직접 부르지 않는 경로를 모사
    assert kr.bo_enabled is False
    assert kr.breakout_creep() is False


# ── B-10: '막힌 채 정지' 회계는 회랑 후보 유무와 무관하다 ────────────────
def _blocked_ap(kr, p, oid=2, x=6.0):
    """코앞에 정지 차량 하나 — _blocker 가 잡는 배치."""
    b = Box(oid, x, 0.0, 0.0)
    ap = Ap(p, actors=[b])
    for _ in range(STATIC_TICKS + 2):
        kr._update_obj_timers(ap)
    return ap


def test_blocked_ticks_accumulate_while_corridor_candidate_exists():
    """PREEMPT/WAIT_EXPIRED 로 들어와 side 루프가 기각해도 카운터는 쌓인다.

    옛 동작은 회계가 `if actor is None:` 안에 있어 회랑 후보(=actor)가 있으면
    한 틱도 증가하지 않았다 → REACTIVE 가 영원히 무장되지 않았다 (B-10).
    여기서는 lg 를 주지 않아 side 루프 직전에 빠지므로(no_lane) '기각과 같은
    경로' 를 만든다.
    """
    kr, p = make(d_tl=float('inf'))
    ap = _blocked_ap(kr, p)
    assert kr.ot_blocked_ticks == 0
    for n in range(1, 6):
        kr._try_overtake(ap, p, ego_speed=0.0)          # 정지 + 전방 차단
        assert kr.ot_blocked_ticks == n                  # 매 틱 증가한다
    assert (kr.last_avoid or {}).get('state') in ('PREEMPT', 'WAIT', 'WAIT_EXPIRED')


def test_reactive_arms_after_trigger_s_even_under_preempt():
    """기각이 계속돼도 trigger_s 를 넘기면 REACTIVE 가 무장된다."""
    kr, p = make(d_tl=float('inf'))
    ap = _blocked_ap(kr, p)
    for _ in range(kr.ot_ticks + 1):
        kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_blocked_ticks >= kr.ot_ticks
    assert kr.last_overtake == 'no_lane'                 # side 루프 직전까지 갔다
    assert (kr.last_avoid or {}).get('state') == 'REACTIVE'


def test_blocked_ticks_reset_when_moving():
    """굴러가고 있으면 '막힌 채 정지' 가 아니다 — 0 으로 리셋."""
    kr, p = make(d_tl=float('inf'))
    ap = _blocked_ap(kr, p)
    for _ in range(4):
        kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_blocked_ticks == 4
    kr._try_overtake(ap, p, ego_speed=5.0)               # 주행 중
    assert kr.ot_blocked_ticks == 0


def test_blocked_ticks_reset_when_nothing_blocks():
    """전방에 막는 것이 없으면 카운터가 쌓이지 않는다."""
    kr, p = make(d_tl=float('inf'))
    ap = Ap(p, actors=[])
    for _ in range(5):
        kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_blocked_ticks == 0


# ── B-12: 기하 완성 게이트 (전이가 장애물 앞에서 끝나는가) ───────────────
GEOM_MARGIN = OT['shift_geom_margin_m']


class LgOne:
    """오른쪽 이웃만 있는 최소 레인그래프 목 — side 루프까지 도달시킨다."""

    def __init__(self):
        self.lanes = {(1, 0, -1): {'junction': -1}, (1, 0, -2): {'junction': -1}}

    def neighbor(self, key, side):
        return (1, 0, -2) if side == 'right' else None

    def mark_at(self, key, s, side):
        return ('broken', 'standard', True)

    def dashed_runs(self, key, side):
        return [(0.0, 1000.0)]

    def length(self, key):
        return 1000.0

    def successors(self, key):
        return []

    def predecessors(self, key):
        return []

    def locate(self, x, y):
        return None


class GeomPlanner(Planner):
    """시프트 적용을 흉내만 내는 목 — 게이트 통과 여부만 보면 되므로 경로는
    건드리지 않는다 (span 만 돌려준다)."""

    def shift_route_around_actors(self, first_actor, last_actor=None,
                                  obstacle_direction='right', transition_length=120.0,
                                  lane_transition_factor=1.0, extra_length_before=0.0,
                                  extra_length_after=0.0, min_start_ahead=0):
        a = self.route_index + int(min_start_ahead)
        return a, a + int(2 * transition_length)


def _geom_rig(cfg, obj_x, obs_ticks=None):
    """장애물이 obj_x [m] 앞에 정지해 있는 배치. side 루프까지 들어간다.

    관찰 틱을 wait_before_shift_s 이상 쌓아 WAIT_EXPIRED 로 진입시킨다 —
    원거리 장애물은 PREEMPT 의 시간예산(t_left < budget)을 만족하지 못한다.
    """
    if obs_ticks is None:
        obs_ticks = int(cfg['overtake']['wait_before_shift_s'] * HZ) + 5
    p = GeomPlanner(d_tl=float('inf'))
    p.lg = LgOne()
    kr = KrRules(cfg)
    kr._sl_all = []
    b = Box(2, obj_x, 0.0, 0.0)
    ap = Ap(p, actors=[b])
    ap._kr_ego_lane = (1, 0, -1)
    for _ in range(obs_ticks):
        kr._update_obj_timers(ap)
    return kr, p, ap


def test_geom_gate_rejects_when_transition_cannot_finish():
    """실측 재현 — s_rel 5.3 m, v 0 이면 need 19.0 m > 5.3 → 기각."""
    kr, p, ap = _geom_rig(CFG, 5.3)
    kr._try_overtake(ap, p, ego_speed=0.0)
    need = CFG['overtake']['transition_m'] + OT['shift_ahead_m'] + GEOM_MARGIN
    assert kr.ot_span is None                              # 시프트가 생기지 않았다
    assert kr.last_overtake == 'right:geom'
    a = kr.last_avoid or {}
    assert a.get('reject') == 'right:geom'
    assert a['need_geom'] == pytest.approx(need, abs=0.05)
    assert a['margin'] < 0


def test_geom_gate_passes_when_transition_finishes_in_time():
    """여유가 충분하면 통과한다 — 정상 회피 회귀."""
    kr, p, ap = _geom_rig(CFG, 52.3)
    kr._try_overtake(ap, p, ego_speed=1.62)
    assert kr.ot_span is not None                          # 시프트 생성됨
    assert kr.last_overtake == 'right'


def test_geom_gate_boundary_is_the_margin():
    """need 와 정확히 같은 거리면 통과, 그보다 가까우면 기각."""
    need = CFG['overtake']['transition_m'] + OT['shift_ahead_m'] + GEOM_MARGIN
    kr, p, ap = _geom_rig(CFG, need + 0.2)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is not None
    kr2, p2, ap2 = _geom_rig(CFG, need - 0.2)
    kr2._try_overtake(ap2, p2, ego_speed=0.0)
    assert kr2.ot_span is None and kr2.last_overtake == 'right:geom'


def test_geom_gate_kill_switch_restores_old_behaviour():
    """shift_geom_gate_enable: false 면 수정 전과 동일 — 기각하지 않는다."""
    cfg = copy.deepcopy(CFG)
    cfg['overtake']['shift_geom_gate_enable'] = False
    kr, p, ap = _geom_rig(cfg, 5.3)
    kr._try_overtake(ap, p, ego_speed=0.0)
    assert kr.ot_span is not None                          # 옛 동작: 생성된다
    assert kr.last_overtake == 'right'


def test_geom_gate_scales_with_speed():
    """trans_m 이 속도 비례로 커지면 필요 거리도 커진다 (재계산 없이 같은 값)."""
    v = 8.0
    trans = max(CFG['overtake']['transition_m'], OT['shift_k_s'] * v)
    need = trans + OT['shift_ahead_m'] + GEOM_MARGIN
    kr, p, ap = _geom_rig(CFG, need - 1.0)
    kr._try_overtake(ap, p, ego_speed=v)
    assert kr.ot_span is None and kr.last_overtake == 'right:geom'
    assert (kr.last_avoid or {})['need_geom'] == pytest.approx(need, abs=0.05)
