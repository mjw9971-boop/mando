"""
정차 차량 추월 (blocked 판정 + 합성 차선변경).

2026-08-23 18:33 런: 정차 차량 뒤 605 m 에서 갇혀 완주 불가.

계약 (전부 만족해야 blocked):
  · 선행차 속도 < 0.5 m/s
  · 전방에 **활성 신호 정지 후보 없음** — 신호 대기 중 추월은 즉시 실격
  · 교차로 내부 아님
  · 앞범퍼 기준 전방 정지선이 lead.blocked_ignore_stopline_m 보다 멀거나 없음
  · 위가 lead.blocked_dwell_s 이상 연속
"""
import pathlib
import pickle

import pytest

from conftest import PARAMS_YAML
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.planner import Planner
from hlfma.core.types import TrackedObject
from hlfma.nodes.params import load_params_yaml
from test_turn_signal import make_world

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route_example.pkl'
CFG = load_params_yaml(PARAMS_YAML)
DWELL = CFG['lead']['blocked_dwell_s']
IGN = CFG['lead']['blocked_ignore_stopline_m']
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
RED, GREEN = 1, 3

pytestmark = pytest.mark.skipif(not (GRAPH.exists() and ROUTE.exists()),
                                reason='lane_graph.pkl / route_example.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def route():
    return pickle.load(open(ROUTE, 'rb'))


def stopped(oid=2, s_rel=12.0, speed=0.0, lane=None):
    return TrackedObject(id=oid, x=0.0, y=0.0, heading=0.0, speed=speed, length=4.4, width=1.8,
                         height=1.4, cls='vehicle', lane=lane, on_route=True, s_rel=s_rel,
                         lat_off=0.0, v_rel=0.0, ttc=float('inf'), will_enter_lane=False,
                         age=0.0, coasting=False)


def feed(pl, lg, route, s, t, objects, light=None, summ_over=None):
    w = make_world(lg, route, s, speed=0.0, t=t, light=light, objects=objects)
    if summ_over:
        w.summ = dict(w.summ, **summ_over)
    return pl.plan(w)


def far_from_stopline(lg, route):
    """전방 정지선이 blocked_ignore_stopline_m 보다 먼 route_s 를 찾는다."""
    for s in range(0, 40):
        w = make_world(lg, route, float(s), speed=0.0, t=0.0)
        d = w.summ.get('dist_stop_line')
        if d is None or (d - FRONT) > IGN:
            return float(s)
    pytest.skip('정지선에서 충분히 먼 지점이 없다')


def test_blocked_after_dwell(lg, route):
    """정차 선행차 + 신호 무관 + 정지선 멀면 dwell 후 blocked."""
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    d0 = feed(pl, lg, route, s, 0.0, [stopped()])
    assert d0.reasons['blocked'] is False, 'dwell 전에는 아니다'
    assert 'dwell' in d0.reasons['blocked_why']
    d1 = feed(pl, lg, route, s, DWELL + 0.1, [stopped()])
    assert d1.reasons['blocked'] is True


def test_moving_lead_is_not_blocked(lg, route):
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    feed(pl, lg, route, s, 0.0, [stopped(speed=3.0)])
    d = feed(pl, lg, route, s, DWELL + 0.1, [stopped(speed=3.0)])
    assert d.reasons['blocked'] is False
    assert 'lead 주행' in d.reasons['blocked_why']


def test_signal_stop_blocks_overtake(lg, route):
    """★ 신호 정지 후보가 있으면 절대 blocked 가 아니다 (신호 대기 추월 = 실격)."""
    pl = Planner(lg, route, CFG)
    # 정지선 앞 + 적색 → stop_line 후보 활성
    s = 20.0
    for t in (0.0, DWELL + 0.1, DWELL + 5.0):
        d = feed(pl, lg, route, s, t, [stopped()], light=(3, RED))
        assert 'stop_line' in d.reasons
        assert d.reasons['blocked'] is False
        assert '신호' in d.reasons['blocked_why']


def test_near_stopline_is_not_blocked(lg, route):
    """정지선이 가까우면(신호 대기 행렬 가능) 추월하지 않는다."""
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    over = {'dist_stop_line': FRONT + IGN - 5.0, 'stop_signal_ids': []}
    for t in (0.0, DWELL + 0.1):
        d = feed(pl, lg, route, s, t, [stopped()], summ_over=over)
    assert d.reasons['blocked'] is False
    assert '정지선' in d.reasons['blocked_why']


def test_in_junction_is_not_blocked(lg, route):
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    for t in (0.0, DWELL + 0.1):
        d = feed(pl, lg, route, s, t, [stopped()], summ_over={'in_junction': True})
    assert d.reasons['blocked'] is False
    assert '교차로' in d.reasons['blocked_why']


def test_dwell_resets_on_condition_break(lg, route):
    """조건이 한 번 깨지면 dwell 이 처음부터 다시 센다."""
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    feed(pl, lg, route, s, 0.0, [stopped()])
    feed(pl, lg, route, s, DWELL - 0.2, [stopped(speed=3.0)])      # 조건 깨짐
    d = feed(pl, lg, route, s, DWELL + 0.1, [stopped()])
    assert d.reasons['blocked'] is False


def test_blocked_gap_is_wider(lg, route):
    """blocked 이면 조향 여유를 위해 lead.blocked_gap_m 만큼 더 멀리 선다."""
    pl = Planner(lg, route, CFG)
    s = far_from_stopline(lg, route)
    feed(pl, lg, route, s, 0.0, [stopped(s_rel=40.0)])
    d = feed(pl, lg, route, s, DWELL + 0.1, [stopped(s_rel=40.0)])
    assert d.reasons['blocked'] is True
    pl2 = Planner(lg, route, CFG)
    d2 = feed(pl2, lg, route, s, 0.0, [stopped(s_rel=40.0)])       # blocked 아님
    assert d.reasons['lead'] < d2.reasons['lead'], '더 넓은 간격 = 더 낮은 목표속도'


# ── 정지 출발 추월 (2026-08-24 13:10 런: v_min 게이트 교착) ─────────────────
def drive_to_blocked_standstill(lg, route, s=90.0, oid=7):
    """정차 lead 뒤에 v=0 으로 선 상태에서 dwell 을 채워 blocked 로 만든다."""
    pl = Planner(lg, route, CFG)
    d = None
    for t in (0.0, 1.0, DWELL + 0.1):
        d = pl.plan(make_world(lg, route, s, speed=0.0, t=t, objects=[stopped(oid, s_rel=14.0)]))
    assert d.reasons['blocked'] is True
    return pl


def test_standstill_overtake_starts_despite_v_min(lg, route):
    """★ 교착 해소: blocked 추월은 v=0 에서도 지시등 3 s 후 시작된다."""
    s = 90.0
    pl = drive_to_blocked_standstill(lg, route, s)
    lead_min = CFG['signal']['lc_lead_min_s']
    started = None
    t = DWELL + 0.2
    while t < DWELL + lead_min + 3.0:
        d = pl.plan(make_world(lg, route, s, speed=0.0, t=t, objects=[stopped(7, s_rel=14.0)]))
        assert d.turn_signal != 0, '추월 지시등이 꺼졌다'
        if d.state == 'LANE_CHANGE':
            started = t
            break
        # 시작 전에는 v_min 이 사유가 아니어야 한다 (면제)
        assert d.reasons.get('lc_too_slow') is None
        assert d.reasons.get('lc_vmin_exempt') == 'overtake'
        t += 0.2
    assert started is not None, 'v=0 에서 추월이 끝내 시작되지 않았다 (교착 재현)'
    assert d.reasons.get('lc_vmin_exempt') == 'overtake'


def test_standstill_overtake_stall_watchdog_aborts(lg, route):
    """전이 중 v≈0 이 stall_abort_s 이상 지속되면 중단·복귀한다."""
    s = 90.0
    pl = drive_to_blocked_standstill(lg, route, s)
    t = DWELL + 0.2
    d = None
    while t < DWELL + 8.0:
        d = pl.plan(make_world(lg, route, s, speed=0.0, t=t, objects=[stopped(7, s_rel=14.0)]))
        if d.state == 'LANE_CHANGE':
            break
        t += 0.2
    assert d.state == 'LANE_CHANGE'
    # 차가 전혀 움직이지 않는다 (v=0 유지) → 워치독이 끊어야 한다
    hold = CFG['lane_change']['stall_abort_s']
    aborted = None
    for k in range(int(hold / 0.2) + 6):
        t += 0.2
        d = pl.plan(make_world(lg, route, s, speed=0.0, t=t, objects=[stopped(7, s_rel=14.0)]))
        if d.reasons.get('lc_stall_abort') is not None:
            aborted = t
            break
    assert aborted is not None, '정체 워치독이 발동하지 않았다'
    assert pl._lc is None


def test_creep_releases_idm_stop_when_blocked(lg, route):
    """blocked 확정이면 IDM 정지 목표가 풀리고 크립 속도 이상이 나온다."""
    s = 90.0
    pl = drive_to_blocked_standstill(lg, route, s)
    d = pl.plan(make_world(lg, route, s, speed=0.0, t=DWELL + 0.3, objects=[stopped(7, s_rel=14.0)]))
    creep = CFG['lane_change']['creep_kph'] / 3.6
    assert d.reasons.get('lead_creep') is True
    assert d.reasons['lead'] >= creep - 1e-6
    # blocked 가 아니면(주행 이력 있는 차) 크립이 없어야 한다
    pl2 = Planner(lg, route, CFG)
    mov = stopped(8, s_rel=14.0, speed=3.0)
    pl2.plan(make_world(lg, route, s, speed=0.0, t=0.0, objects=[mov]))
    d2 = pl2.plan(make_world(lg, route, s, speed=0.0, t=DWELL + 0.3, objects=[stopped(8, s_rel=14.0)]))
    assert d2.reasons.get('lead_creep') is None
    assert d2.reasons['blocked'] is False


# ── 원거리 신호가 blocked 를 죽이지 않는다 (조건 2 통합) ─────────────────────
def test_far_red_light_does_not_kill_blocked(lg, route):
    """멀리(>30 m) 있는 적신호의 비구속 stop_line 후보는 추월을 막지 않는다."""
    s = 170.0                                 # 다음 정지선(305 m)까지 135 m — 지평(200 m) 안
    pl = Planner(lg, route, CFG)
    d = None
    for t in (0.0, 1.0, DWELL + 0.1, DWELL + 0.2):
        w = make_world(lg, route, s, speed=0.0, t=t, light=(5, 1),
                       objects=[stopped(7, s_rel=14.0)])
        d = pl.plan(w)
    assert w.summ.get('dist_stop_line') is not None and         (w.summ['dist_stop_line'] - 3.8) > CFG['lead']['blocked_ignore_stopline_m'],         '전제: 정지선이 보이되 30 m 보다 멀어야 한다'
    assert 'stop_line' in d.reasons and d.reasons['stop_line'] > d.reasons['lead'], \
        '전제: 정지 후보는 있으나 구속하지 않는다'
    assert d.reasons['blocked'] is True, '비구속 원거리 신호가 blocked 를 죽였다'


def test_binding_stop_candidate_still_blocks(lg, route):
    """stop_line 후보가 실제로 min 을 결정하면(구속) 추월 금지가 유지된다."""
    pl = Planner(lg, route, CFG)
    s = 20.0                                  # 정지선 50.03 — 30 m 전, 적색
    for t in (0.0, DWELL + 0.5, DWELL + 5.0):
        d = pl.plan(make_world(lg, route, s, speed=0.0, t=t, light=(3, 1),
                               objects=[stopped(9, s_rel=14.0)]))
        assert d.reasons['blocked'] is False
        assert '신호' in d.reasons['blocked_why'] or '정지선' in d.reasons['blocked_why']
