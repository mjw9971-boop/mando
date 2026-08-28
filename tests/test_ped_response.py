"""
detect_pedestrian_response — 대회 항목 10 보행자 대응 (합성 틱).

직선 차로(폭 3.0, +x 방향, route_s = x) 가짜 그래프에서 보행자 횡단
시나리오를 합성한다. 횡단 완료 임계 = 1.5 + ped_clear_m(0.5) = 2.0 m.
"""
import math

import score as sc_mod
from conftest import mk_tick

STOP_SPEED = 0.5
SC = {'ped_near_m': 2.0, 'ped_cross_v': 0.3, 'ped_engage_m': 40.0,
      'ped_restart_v': 1.0, 'ped_clear_m': 0.5}
K = (10, 0, -1)
ROUTE = {'lanes': [K], 'cum_s': [0.0]}
PED = 7
DT = 0.1


class LineLG:
    """+x 방향 직선 차로 하나 (길이 100, 폭 3.0). project/point_at/width_at 만."""

    def project(self, key, x, y):
        if key != K:
            raise KeyError(key)
        s = min(max(x, 0.0), 100.0)
        return s, y, math.hypot(x - s, y) if (x < 0 or x > 100) else abs(y), 0

    def width_at(self, key, s):
        if key != K:
            raise KeyError(key)
        return 3.0

    def point_at(self, key, s):
        return s, 0.0, 0.0, 0.0


def tick(i, ego_s, v, ped_xy=None, ped_heading=0.0, ped_v=0.0, cls='pedestrian',
         reset=False):
    raw = []
    objs = []
    if ped_xy is not None:
        raw = [[PED, ped_xy[0], ped_xy[1], 0.0, ped_heading, ped_v, 0.6, 0.7, 1.8]]
        objs = [{'id': PED, 'cls': cls, 'x': ped_xy[0], 'y': ped_xy[1]}]
    return mk_tick(t=i * DT, speed=v, x=ego_s, route_s=ego_s, lane=K,
                   raw_objects=raw, objects=objs, reset=reset)


def run(ticks, merge_gap_s=1.0):
    return sc_mod.detect_pedestrian_response(ticks, ticks[0]['t'], LineLG(), ROUTE,
                                             SC, STOP_SPEED, merge_gap_s)


def drive_through(ped_y, v=8.0, n=60, **kw):
    """정지 없이 s 0→… 주행, 보행자는 (30, ped_y) 고정."""
    return [tick(i, ego_s=i * DT * v, v=v, ped_xy=(30.0, ped_y), **kw)
            for i in range(n)]


# ── 무정차 통과 ───────────────────────────────────────────────────────────
def test_no_stop_past_standing_ped_on_lane():
    evs = run(drive_through(ped_y=0.5))                    # 차로 위, 횡속도 0
    assert len(evs) == 1
    assert evs[0]['kind'] == 'no_stop' and evs[0]['obj_id'] == PED
    assert evs[0]['min_v_kph'] >= 28.0                     # 8 m/s 유지


def test_no_stop_on_approaching_ped_from_sidewalk():
    # 보도(y=3.2)에서 차로 쪽(-y)으로 1.2 m/s 접근 — 완료 전 통과 = 위반
    ticks = [tick(i, ego_s=i * DT * 8.0, v=8.0,
                  ped_xy=(30.0, 3.2 - 1.2 * i * DT), ped_heading=-math.pi / 2,
                  ped_v=1.2) for i in range(60)]
    evs = run(ticks)
    assert len(evs) == 1 and evs[0]['kind'] == 'no_stop'


def test_unknown_cls_is_protected():
    evs = run(drive_through(ped_y=0.5, cls='unknown'))
    assert len(evs) == 1 and evs[0]['obj_cls'] == 'unknown'


# ── 비대상 ────────────────────────────────────────────────────────────────
def test_parallel_walker_on_sidewalk_not_engaged():
    # 도로와 나란히(+x) 걷는 보행자, 경로에서 4 m — 대상 아님
    ticks = [tick(i, ego_s=i * DT * 8.0, v=8.0,
                  ped_xy=(30.0 + 1.5 * i * DT, 4.0), ped_heading=0.0, ped_v=1.5)
             for i in range(60)]
    assert run(ticks) == []


def test_near_boundary_just_outside_not_engaged():
    assert run(drive_through(ped_y=2.05)) == []            # d=2.05 > near_m, 정지 상태


def test_near_boundary_just_inside_engaged():
    assert len(run(drive_through(ped_y=1.95))) == 1


def test_reset_ticks_excluded():
    assert run(drive_through(ped_y=0.5, reset=True)) == []


# ── 정지 후 시나리오 ──────────────────────────────────────────────────────
def stop_and_go(restart_i, ped_clear_i, n=120, v_go=5.0):
    """s=20 에서 정지, restart_i 틱부터 v_go 재출발. 보행자는 ped_clear_i
    틱부터 차로 밖(y=2.5)으로 나간다."""
    ticks = []
    s = 0.0
    for i in range(n):
        if i < 25:
            v = 8.0
        elif i < restart_i:
            v = 0.0
        else:
            v = v_go
        s += v * DT
        y = 0.5 if i < ped_clear_i else 2.5
        ticks.append(tick(i, ego_s=s, v=v, ped_xy=(30.0, y)))
    return ticks


def test_normal_stop_wait_then_go_is_clean():
    # 보행자가 60틱에 비켜난 뒤 80틱에 출발 — 정상
    assert run(stop_and_go(restart_i=80, ped_clear_i=60)) == []


def test_early_restart_while_ped_on_lane():
    # 보행자가 아직 y=0.5(차로 위)인데 40틱에 재출발 — 조기 출발
    evs = run(stop_and_go(restart_i=40, ped_clear_i=90))
    assert len(evs) == 1 and evs[0]['kind'] == 'early_start'


def test_creep_below_restart_v_not_early():
    # 재출발 판정 미만(0.8 m/s < ped_restart_v)으로 기는 것은 조기 출발 아님
    evs = run(stop_and_go(restart_i=40, ped_clear_i=90, v_go=0.8))
    assert all(e['kind'] != 'early_start' for e in evs)


def test_one_event_per_ped_with_split_engagement():
    # 보행자가 잠깐 사라져 구간이 쪼개져도 merge_gap 병합 → 1건
    ticks = drive_through(ped_y=0.5, n=60)
    for i in (20, 21):                                     # 0.2 s 소멸
        ticks[i]['raw']['objects'] = []
        ticks[i]['objects'] = []
    evs = run(ticks, merge_gap_s=1.0)
    assert len(evs) == 1
