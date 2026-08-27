"""
detect_red_stop — 대회 항목 7 적색신호 정지 품질 (합성 틱 경계값).

분류: 앞범퍼 최전방 stop_line_front_m 이 [-stop_ok_m, 0] 정상(정보) /
< -stop_ok_m 경미(red_stop_far) / > 0 침범(stop_line_encroach).
정지 = 적색 + 전방 정지선 + v<stop_speed 가 stop_hold_s 이상 지속.
"""
import score as sc_mod
from conftest import mk_tick

STOP_SPEED = 0.5
SC = {'stop_ok_m': 2.0, 'stop_hold_s': 0.5}
RED, GREEN = 1, 3
CTRL = [27]


def stopped_run(front_m, n=8, dt=0.1, state=RED, speed=0.1, reset_at=None, ctrl=CTRL):
    """적색에서 front_m 위치에 n틱(dt 간격) 정지해 있는 로그 + 앞뒤 주행 틱."""
    ticks = [mk_tick(t=0.0, speed=5.0, front_m=front_m - 3.0, ctrl=ctrl,
                     lights=[(27, state)])]
    for i in range(n):
        ticks.append(mk_tick(t=1.0 + i * dt, speed=speed, front_m=front_m, ctrl=ctrl,
                             lights=[(27, state)], reset=(i == reset_at)))
    ticks.append(mk_tick(t=1.0 + n * dt + 0.05, speed=5.0, front_m=front_m + 3.0,
                         ctrl=ctrl, lights=[(27, state)]))
    return ticks


def run(ticks, merge_gap_s=0.0):
    return sc_mod.detect_red_stop(ticks, ticks[0]['t'], STOP_SPEED, SC, merge_gap_s)


# ── 위치 3분류 (거리 경계) ────────────────────────────────────────────────
def test_stop_within_ok_window_is_ok():
    ok, far, enc = run(stopped_run(front_m=-1.0))
    assert (len(ok), len(far), len(enc)) == (1, 0, 0)
    assert ok[0]['front_m'] == -1.0


def test_stop_exactly_at_ok_boundary_is_ok():
    ok, far, enc = run(stopped_run(front_m=-2.0))        # -stop_ok_m 포함
    assert (len(ok), len(far), len(enc)) == (1, 0, 0)


def test_stop_just_beyond_ok_boundary_is_far():
    ok, far, enc = run(stopped_run(front_m=-2.01))
    assert (len(ok), len(far), len(enc)) == (0, 1, 0)
    assert far[0]['front_m'] == -2.01


def test_stop_past_line_is_encroach():
    ok, far, enc = run(stopped_run(front_m=0.3))
    assert (len(ok), len(far), len(enc)) == (0, 0, 1)
    assert enc[0]['encroach_m'] == 0.3


# ── 지속시간 경계 ─────────────────────────────────────────────────────────
def test_hold_just_under_threshold_ignored():
    # 5틱 × 0.1 s → span 0.4 s < 0.5 — 일시 감속으로 본다
    assert run(stopped_run(front_m=-1.0, n=5)) == ([], [], [])


def test_hold_at_threshold_counts():
    # 6틱 × 0.1 s → span 0.5 s
    ok, _, _ = run(stopped_run(front_m=-1.0, n=6))
    assert len(ok) == 1


# ── 조건 밖: 녹색 / 신호 없음 / reset ────────────────────────────────────
def test_green_light_not_counted():
    assert run(stopped_run(front_m=-1.0, state=GREEN)) == ([], [], [])


def test_no_ctrl_ids_not_counted():
    assert run(stopped_run(front_m=-1.0, ctrl=[])) == ([], [], [])


def test_reset_tick_breaks_episode():
    # 가운데 reset 틱 → 에피소드가 둘로 쪼개져 각각 hold 미달
    ticks = stopped_run(front_m=-1.0, n=8, reset_at=4)
    assert run(ticks) == ([], [], [])


def test_merge_gap_rejoins_speed_blip():
    # 속도 블립으로 마스크가 끊겨도 merge_gap 이내면 한 에피소드
    ticks = stopped_run(front_m=-2.5, n=8)
    ticks[4]['ego']['speed'] = 1.0                       # 1틱 블립
    ok, far, enc = run(ticks, merge_gap_s=1.0)
    assert (len(ok), len(far), len(enc)) == (0, 1, 0)


# ── 좌회전 화살표 규칙 (detect_red_light 와 동일) ─────────────────────────
def test_left_arrow_on_left_turn_route_is_not_red():
    summ = {'next_turn': 'turn_left', 'dist_next_turn': 10.0}
    ticks = stopped_run(front_m=-1.0, state=4)
    for t in ticks:
        t['world']['summ'] = dict(summ)
    assert run(ticks) == ([], [], [])
