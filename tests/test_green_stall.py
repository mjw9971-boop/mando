"""
detect_green_stall — 대회 항목 8 녹색신호 통과 (합성 틱 경계값).

정지선 앞 green_dist_m 이내 + 녹색 + 정차가 green_minor_s 이상이면 검출,
severity 는 dur 로 (green_major_s 이상 중대). 면책: 보행자 / 선행차 / winner.
"""
import score as sc_mod
from conftest import mk_tick

STOP_SPEED = 0.5
SC = {'green_dist_m': 30.0, 'green_minor_s': 5.0, 'green_major_s': 10.0,
      'green_lead_m': 15.0, 'green_lead_lat_m': 2.0}
GREEN, RED = 3, 1
CTRL = [27]


def stall_ticks(dur_s, front_m=-10.0, state=GREEN, dt=0.1, objects=None,
                winner=None, reset_at=None):
    n = int(round(dur_s / dt)) + 1
    return [mk_tick(t=i * dt, speed=0.0, front_m=front_m, ctrl=CTRL,
                    lights=[(27, state)], objects=objects,
                    reasons={'winner': winner} if winner else None,
                    reset=(i == reset_at))
            for i in range(n)]


def run(ticks, merge_gap_s=0.0):
    return sc_mod.detect_green_stall(ticks, ticks[0]['t'], STOP_SPEED, SC, merge_gap_s)


def sev(ev):
    return sc_mod._severity('green_stall', ev, SC)


# ── 지속시간 경계 → 검출/severity ────────────────────────────────────────
def test_just_under_minor_ignored():
    assert run(stall_ticks(4.9)) == []


def test_at_minor_is_minor():
    evs = run(stall_ticks(5.0))
    assert len(evs) == 1 and sev(evs[0]) == 'minor'


def test_just_under_major_is_minor():
    evs = run(stall_ticks(9.9))
    assert len(evs) == 1 and sev(evs[0]) == 'minor'


def test_at_major_is_major():
    evs = run(stall_ticks(10.0))
    assert len(evs) == 1 and sev(evs[0]) == 'major'


# ── 조건 한정 ─────────────────────────────────────────────────────────────
def test_red_light_not_counted():
    assert run(stall_ticks(6.0, state=RED)) == []


def test_beyond_green_dist_not_counted():
    assert run(stall_ticks(6.0, front_m=-31.0)) == []


def test_past_stop_line_not_counted():
    assert run(stall_ticks(6.0, front_m=0.5)) == []


def test_reset_tick_breaks_episode():
    assert run(stall_ticks(6.0, reset_at=30)) == []        # 양쪽 각 3 s < 5


# ── 면책 3종 ─────────────────────────────────────────────────────────────
def test_pedestrian_exempts():
    ped = [{'id': 1, 'cls': 'pedestrian', 'x': 5.0, 'y': 3.0}]
    assert run(stall_ticks(6.0, objects=ped)) == []


def test_lead_vehicle_exempts():
    lead = [{'id': 2, 'cls': 'vehicle', 'x': 10.0, 'y': 0.0}]   # 전방 10 m
    assert run(stall_ticks(6.0, objects=lead)) == []


def test_far_vehicle_does_not_exempt():
    far = [{'id': 2, 'cls': 'vehicle', 'x': 20.0, 'y': 0.0}]    # 15 m 초과
    assert len(run(stall_ticks(6.0, objects=far))) == 1


def test_side_vehicle_does_not_exempt():
    side = [{'id': 2, 'cls': 'vehicle', 'x': 10.0, 'y': 3.0}]   # 횡 2 m 초과
    assert len(run(stall_ticks(6.0, objects=side))) == 1


def test_recorded_winner_exempts():
    assert run(stall_ticks(6.0, winner='walker')) == []


def test_light_winner_does_not_exempt():
    # 녹색인데 planner 가 'light' 로 정차 = 신호 오인 — 바로 항목 8 대상
    evs = run(stall_ticks(6.0, winner='light'))
    assert len(evs) == 1 and evs[0]['winner'] == 'light'
