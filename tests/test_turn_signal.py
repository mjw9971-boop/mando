"""
kr_rules 방향지시등 — 채점 동적항목 "방향지시등 n초 전" (SPEC §3.3).

규칙은 둘로 나뉜다.
  · 회전 : route['events'] 의 turn_* 구간 (연결로 중심을 따라가 기하로는 안 잡힌다)
  · 차로 이동 : planner.lat_shift(경로점의 차로 중심 대비 횡오프셋)를 앞 창에서 훑는다
    — 계획된 차선변경과 **런타임 회피 시프트**가 같은 규칙 하나로 잡힌다.

계약:
  · 앞 창에서 lat_shift 가 lat_shift_on_m 이상 달라지면 그 방향으로 점등
  · 창 전체를 훑는다 (끝점만 보면 이동을 마친 뒤라 0 이 나와 놓친다)
  · 굽은 길은 차로 중심을 따라가므로 lat_shift 가 0 → 안 켜진다
  · 겹치면 남은거리 짧은 쪽, 동률이면 회전 우선
  · 깜빡임 방지는 최소 점등 시간뿐 — 끄는 임계·래치 없음 (고착 불가)
"""
import pathlib
import pickle
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.control import VtdLongitudinalController

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from autopilot import AutoPilot                            # noqa: E402
from config import GlobalConfig                            # noqa: E402
from kr_rules import (SIG_LEFT, SIG_OFF, SIG_RIGHT,        # noqa: E402
                      KrRules, turn_intervals)
from vtd_adapter.carla_types import VehicleControl         # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
SIG = CFG['signal']
PPM = 10
TOTAL = 400.0
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE_PKL = ROOT / 'data' / 'route.pkl'


class FakeLaneGraph:
    def __init__(self, junctions: dict):
        self.lanes = {k: {'junction': j} for k, j in junctions.items()}


class FakePlanner:
    """kr_rules 표면: route / route_s / lat_shift / route_index / points_per_meter."""

    points_per_meter = PPM

    def __init__(self, events=(), lanes=(), cum=(), lens=(), junctions=None,
                 lat=None):
        self.route = {'total_length': TOTAL, 'events': list(events),
                      'lanes': list(lanes), 'cum_s': list(cum), 'lengths': list(lens)}
        self.route_s = np.arange(0.0, TOTAL + 60.0, 1.0 / PPM)
        self.lat_shift = (np.zeros(len(self.route_s)) if lat is None
                          else np.asarray(lat, dtype=float))
        self.route_index = 0
        self.lg = FakeLaneGraph(junctions or {})

    def set_route_s(self, rs):
        self.route_index = int(round(rs * PPM))


class FakeEgo:
    def __init__(self):
        self.speed = 0.0

    def get_velocity(self):
        ego = self

        class V:
            def length(self):
                return ego.speed
        return V()


def make_ap(planner):
    a = AutoPilot()
    a.setup(world=None, world_map=None, waypoint_planner=planner,
            longitudinal_controller=VtdLongitudinalController(CFG),
            ego_vehicle=FakeEgo(), config=GlobalConfig())
    a.kr_rules = KrRules(CFG)
    return a


def sig_at(ap, route_s, v):
    ap._waypoint_planner.set_route_s(route_s)
    ap._vehicle.speed = v
    ap.kr_rules.apply(VehicleControl(steer=0.0, accel=1.0), 12.5, ap)
    return ap.kr_rules.last_turn_signal


def settle(ap, route_s, v):
    """최소 점등 유지가 풀릴 때까지 틱을 돌린 뒤의 값 (실주행처럼 연속 호출)."""
    out = None
    for _ in range(ap.kr_rules.sig_min_on_ticks + 2):
        out = sig_at(ap, route_s, v)
    return out


def ramp(start_m, end_m, peak, n=None):
    """[start_m, end_m] 구간에서 0 → peak 로 오르는 lat_shift 배열."""
    n = n or int((TOTAL + 60.0) * PPM) + 1
    a = np.zeros(n)
    i0, i1 = int(start_m * PPM), int(end_m * PPM)
    a[i0:i1] = np.linspace(0.0, peak, i1 - i0)
    a[i1:] = peak
    return a


# ── 회전 (이벤트 기반) ────────────────────────────────────────────────────
TURN_LANES = [(10, 0, -1), (20, 0, -1), (21, 0, -1), (30, 0, -1)]
TURN_CUM = [0.0, 100.0, 110.0, 125.0]
TURN_LENS = [100.0, 10.0, 15.0, 200.0]
TURN_JUNC = {(10, 0, -1): -1, (20, 0, -1): 7, (21, 0, -1): 7, (30, 0, -1): -1}
TURN_EV = {'kind': 'turn_left', 's': 100.0, 'lane': (20, 0, -1), 'junction': 7}


@pytest.fixture()
def turn_ap():
    return make_ap(FakePlanner([TURN_EV], TURN_LANES, TURN_CUM, TURN_LENS, TURN_JUNC))


def test_turn_lead_and_hold(turn_ap):
    v = 10.0
    lead = max(v * SIG['turn_lead_s'], SIG['lead_min_m'])
    assert sig_at(turn_ap, 100.0 - lead - 5.0, v) == SIG_OFF
    assert sig_at(turn_ap, 100.0 - lead + 1.0, v) == SIG_LEFT
    assert sig_at(turn_ap, 120.0, v) == SIG_LEFT          # 연결로 안 — 유지
    assert settle(turn_ap, 126.0, v) == SIG_OFF           # 연결로 끝 통과


def test_turn_end_spans_chained_connectors(turn_ap):
    ivs = turn_intervals(turn_ap._waypoint_planner)
    assert len(ivs) == 1
    assert ivs[0]['end_s'] == pytest.approx(125.0)


def test_lane_change_events_are_not_intervals():
    """차로 이동은 이벤트 목록이 아니라 기하로 본다 — 구간을 만들지 않는다."""
    ev = {'kind': 'lane_change_right', 's': 200.0, 'window_s0': 200.0, 'window_s1': 260.0}
    assert turn_intervals(FakePlanner([ev])) == []


def test_lead_min_m_covers_standstill(turn_ap):
    """정지(v=0)에서도 거리 하한 안이면 켜진다 — 대기 중 유지는 의도된 동작."""
    assert sig_at(turn_ap, 100.0 - SIG['lead_min_m'] + 1.0, 0.0) == SIG_LEFT


# ── 차로 이동 (기하 기반) ─────────────────────────────────────────────────
def test_lane_shift_turns_signal_on_before_movement():
    """블렌드가 시작되기 lead 만큼 전에 켜지고, 이동이 끝나면 꺼진다."""
    lat = ramp(200.0, 220.0, 3.0)          # 200~220 m 에서 좌로 3 m
    lat[2400:] = 0.0                       # 240 m 이후 새 차로 기준 → 0
    ap = make_ap(FakePlanner(lat=lat))
    v = 10.0
    assert sig_at(ap, 150.0, v) == SIG_OFF          # 창(30 m) 밖
    assert sig_at(ap, 190.0, v) == SIG_LEFT         # 창 안 — 점등
    assert settle(ap, 260.0, v) == SIG_OFF          # 이동 끝 — 소등


def test_lane_shift_direction_sign():
    """부호가 방향이다 (+좌 / −우)."""
    ap = make_ap(FakePlanner(lat=-ramp(200.0, 220.0, 3.0)))
    assert sig_at(ap, 195.0, 10.0) == SIG_RIGHT


def test_straight_and_curve_stay_off():
    """굽은 길은 차로 중심을 따라가므로 lat_shift 가 0 → 안 켜진다."""
    ap = make_ap(FakePlanner())            # lat 전부 0
    assert sig_at(ap, 100.0, 12.0) == SIG_OFF
    assert ap.kr_rules.last_sig_src is None


def test_scans_whole_window_not_just_endpoint():
    """창 끝이 이미 이동을 마친 뒤여도 중간 이동을 잡는다."""
    lat = ramp(100.0, 110.0, 3.0)
    lat[1200:] = 0.0                       # 120 m 이후 0 으로 복귀 (나갔다 돌아옴)
    ap = make_ap(FakePlanner(lat=lat))
    # v=10 → 창 30 m. s=95 에서 끝점(125 m)은 0 이지만 중간(100~120)이 3 m
    assert sig_at(ap, 95.0, 10.0) == SIG_LEFT


def test_below_threshold_stays_off():
    """임계(lat_shift_on_m) 미만 이동은 무시 — 테이퍼 같은 소소한 보정."""
    ap = make_ap(FakePlanner(lat=ramp(200.0, 210.0, SIG['lat_shift_on_m'] * 0.8)))
    assert sig_at(ap, 195.0, 10.0) == SIG_OFF


# ── 우선순위 / 깜빡임 ─────────────────────────────────────────────────────
def test_overlap_prefers_nearer_then_turn():
    """겹치면 남은거리 짧은 쪽, 동률이면 회전 우선."""
    lat = ramp(100.0, 120.0, 3.0)                  # 좌 이동이 100 m 부터
    turn = {'kind': 'turn_right', 's': 110.0, 'lane': (20, 0, -1), 'junction': 7}
    ap = make_ap(FakePlanner([turn], TURN_LANES, TURN_CUM, TURN_LENS, TURN_JUNC, lat))
    assert sig_at(ap, 95.0, 10.0) == SIG_LEFT      # 이동이 더 가깝다

    lat2 = ramp(110.0, 130.0, 3.0)                 # 이동과 회전이 같은 지점
    ap2 = make_ap(FakePlanner([turn], TURN_LANES, TURN_CUM, TURN_LENS, TURN_JUNC, lat2))
    assert sig_at(ap2, 100.0, 10.0) == SIG_RIGHT   # 동률 → 회전(우) 우선


def test_off_delay_keeps_signal_after_maneuver_then_cancels():
    """회전을 마친 뒤 off_delay_s 만큼 더 켜 두고 자동 소등한다 (실차 자동소등).

    min_on 과 기준 시점이 다르다 — min_on 은 켜진 시점부터, off_delay 는 조건이
    끝난 시점부터다. 그래서 긴 회전 뒤 꼬리는 min_on 이 아니라 off_delay 만큼이다.
    """
    ap = make_ap(FakePlanner([TURN_EV], TURN_LANES, TURN_CUM, TURN_LENS, TURN_JUNC))
    kr = ap.kr_rules
    for _ in range(200):                                   # 충분히 긴 회전
        assert sig_at(ap, 110.0, 10.0) == SIG_LEFT
    tail = 0
    while sig_at(ap, 200.0, 10.0) != SIG_OFF:              # 구간을 완전히 지났다
        tail += 1
        assert tail < 200, '꺼지지 않는다 — 고착'
    assert tail == kr.sig_off_delay_ticks, (tail, kr.sig_off_delay_ticks)
    assert kr.sig_off_delay_ticks < kr.sig_min_on_ticks, 'min_on 이 꼬리를 지배하면 안 된다'


def test_min_on_time_absorbs_flicker_and_releases():
    """0.1 s 깜빡임은 최소 점등으로 흡수되고, 그 시간이 지나면 반드시 꺼진다."""
    lat = ramp(200.0, 220.0, 3.0)
    lat[2400:] = 0.0
    ap = make_ap(FakePlanner(lat=lat))
    assert sig_at(ap, 190.0, 10.0) == SIG_LEFT
    # 조건이 거짓인 틱이 잠깐 끼어도 유지된다
    assert sig_at(ap, 100.0, 10.0) == SIG_LEFT
    hold = ap.kr_rules.sig_min_on_ticks
    for _ in range(hold + 2):
        last = sig_at(ap, 100.0, 10.0)
    assert last == SIG_OFF, '최소 점등이 끝나면 꺼져야 한다 (고착 금지)'


@pytest.mark.skipif(not (GRAPH.exists() and ROUTE_PKL.exists()),
                    reason='lane_graph.pkl / route.pkl 없음')
def test_real_route_lat_shift_marks_only_lane_changes():
    """실제 경로: lat_shift 는 차선변경 구간에서만 변한다 (그 밖은 상수).

    누적값이라 절대값이 0 으로 돌아오지 않는다 — 되돌리면 앞을 볼 때 "반대로
    돌아온다" 로 읽혀 지시등이 역방향으로 켜진다 (2026-08-28 재생에서 확인).
    """
    from vtd_adapter.lanegraph import LaneGraph
    from vtd_adapter.route import VtdRoutePlanner

    with open(ROUTE_PKL, 'rb') as f:
        route = pickle.load(f)
    pl = VtdRoutePlanner(LaneGraph(str(GRAPH)), route, CFG, config=GlobalConfig())
    lat = pl.lat_shift
    n_lc = sum(1 for e in route['events'] if e['kind'].startswith('lane_change'))
    moving = np.abs(np.diff(lat)) > 1e-9                 # 값이 변하는 지점
    runs = np.diff(np.concatenate([[0], moving.view(np.int8), [0]]))
    assert int((runs == 1).sum()) == n_lc, '차선변경 개수만큼만 이동 구간이 있어야 한다'
    # 각 이동의 크기가 차로 폭 수준 (테이퍼 같은 소소한 보정이 아니다)
    starts = np.flatnonzero(runs == 1)
    ends = np.flatnonzero(runs == -1)
    for a, b in zip(starts, ends):
        assert abs(lat[b] - lat[a]) > CFG['vehicle']['width'], (a, b)


# ── 차선변경 시점: 도로 진입로부터 3초 안에 ──────────────────────────────
@pytest.mark.skipif(not (GRAPH.exists() and ROUTE_PKL.exists()),
                    reason='lane_graph.pkl / route.pkl 없음')
def test_lc_ramp_starts_at_road_entry_and_lasts_move_time():
    """진출로가 회전이면 회전 차로에 **미리** 붙어 있어야 한다.

    램프 시작 = max(도로 진입로, 창 시작)  — 점선이 허용하는 가장 이른 지점
    램프 길이 = 계획 속도 x route.lc_move_s (하한·상한 클램프)
    그리고 회전 전에 **여유를 두고 끝난다** (종전엔 회전 직전 17 m 에 몰렸다).
    """
    from vtd_adapter.lanegraph import LaneGraph
    from vtd_adapter.route import VtdRoutePlanner

    with open(ROUTE_PKL, 'rb') as f:
        route = pickle.load(f)
    pl = VtdRoutePlanner(LaneGraph(str(GRAPH)), route, CFG, config=GlobalConfig())
    lanes, cum = route['lanes'], route['cum_s']
    lens = [float(v) for v in route['lengths']]
    lc = [dict(e) for e in route['events'] if e['kind'].startswith('lane_change')]
    turns = [e['s'] for e in route['events'] if e['kind'].startswith('turn_')]
    ramps = pl._build_lc_ramps(lanes, cum, lens, lc)
    assert len(ramps) == len(lc)

    for i_hop, r in ramps.items():
        frm = tuple(lanes[i_hop])
        ev = next(e for e in lc if tuple(e['from_lane']) == frm)
        want0 = max(float(ev['window_s0']), pl._road_entry_s(lanes, cum, i_hop))
        assert r['w0'] == pytest.approx(want0), (frm, r['w0'], want0)
        assert r['w1'] - r['w0'] == pytest.approx(pl._lc_move_len(frm), abs=0.2)
        nxt = min([t for t in turns if t >= r['w1']], default=None)
        if nxt is not None:
            assert nxt - r['w1'] > 20.0, f'회전 {nxt:.1f} 직전에 몰렸다 (램프 끝 {r["w1"]:.1f})'


def test_lc_move_length_scales_with_speed_limit():
    """길이가 시간 기준이다 — 제한속도가 높으면 길고, 낮으면 짧다 (클램프 안)."""
    from vtd_adapter.route import VtdRoutePlanner

    pl = object.__new__(VtdRoutePlanner)
    pl.cfg = CFG
    pl.lc_move_s = CFG['route']['lc_move_s']
    pl.lc_move_min_m = CFG['route']['lc_move_min_m']
    pl.lc_move_max_m = CFG['route']['lc_move_max_m']

    class LG:
        def __init__(self, kph):
            self.kph = kph

        def speed_limit_at(self, key):
            return self.kph, False

    margin = CFG['speed']['margin_kph']
    pl.lg = LG(50.0)
    fast = pl._lc_move_len((1, 0, -1))
    pl.lg = LG(30.0)
    slow = pl._lc_move_len((1, 0, -1))
    assert fast > slow
    assert fast == pytest.approx(min(pl.lc_move_max_m,
                                     (50.0 - margin) / 3.6 * pl.lc_move_s), abs=0.1)
    pl.lg = LG(5.0)
    assert pl._lc_move_len((1, 0, -1)) == pytest.approx(pl.lc_move_min_m)
