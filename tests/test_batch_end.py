"""batch_run 종료 판정(EndJudge) + 완주 임계(end_margin_m) 검증.

2026-08-25 실사고: speed.stop_gap_m 1→4 튜닝 후 고정 임계 5 m 에 계획 정지점
(total − stop_gap − 앞범퍼 ≈ total − 7.8)이 못 미쳐 정상 완주가 timeout 처리.
임계는 params.yaml 의 정지 정책에서 유도해야 한다.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

from batch_run import EndJudge, stop_excused                  # noqa: E402
from summarize_run import end_margin_m, load_cfg              # noqa: E402

BATCH = load_cfg()['batch']


def tk(route_s, v, v_target, *, winner='none', light=None, red_light=None, lead_type=None):
    """EndJudge.feed 가 받는 로그 틱 (2026-08-30: 시그니처가 틱 전체로 넓어졌다 —
    정차 사유를 봐야 신호·보행자 대기와 '앞이 막힘'을 가를 수 있다)."""
    reasons = {'winner': winner, 'red_light': red_light}
    if lead_type is not None:
        reasons['speed_reduced_by'] = {'type': lead_type, 'id': 1, 'dist': 5.0}
    return {'ego': {'route_s': route_s, 'speed': v},
            'decision': {'v_target': v_target, 'reasons': reasons},
            'world': {'light': light}}


def RED(route_s, v=0.0):
    """우리를 세우고 있는 적신호 정차 (예외 대상)."""
    return tk(route_s, v, 0.0, winner='light', light=[147, 1], red_light=0.0)


def judge(**over):
    """params 기본값에 필요한 것만 덮어쓴 EndJudge (상수 이중화 금지)."""
    return EndJudge(total=over.pop('total', 1000.0), margin=over.pop('margin', 8.8),
                    cfg_batch={**BATCH, **over})


def test_end_margin_tracks_stop_policy():
    cfg = load_cfg()
    m = end_margin_m(cfg)
    sp, vh = cfg['speed'], cfg['vehicle']
    planned_stop_short = sp['stop_gap_route_end_m'] + vh['wheelbase'] + vh['front_overhang_m']
    # 임계는 계획 정지점 미달량보다 커야 하고(안 그러면 사고 재발),
    # 과거 통과 기준(5.0)보다 좁아지면 안 된다(회귀).
    assert m > planned_stop_short
    assert m >= 5.0
    # 사고 케이스 수치: total 368.80, 정지 360.93 → 새 임계로 완주여야 한다
    assert 360.93 >= 368.80 - m


def test_judge_done_at_planned_stop():
    """계획 정지점(총길이 − stop_gap − 앞범퍼)에서 정지하면 즉시 완주."""
    m = end_margin_m(load_cfg())
    j = judge(total=368.80, margin=m)
    assert j.feed(0.0, tk(350.0, 8.0, 8.0)) is None
    assert j.feed(10.0, tk(360.93, 0.0, 0.0)) == '완주'


def test_judge_done_grace_when_still_rolling():
    """임계 도달 후 감속 중이면 유예, v<0.5 되면 완주, 유예 초과도 완주."""
    j = judge(total=100.0, stop_grace_s=10.0)
    assert j.feed(0.0, tk(95.0, 3.0, 0.0)) is None       # 도달했지만 아직 굴러감
    assert j.feed(2.0, tk(96.0, 0.2, 0.0)) == '완주'      # 정지 → 완주
    j2 = judge(total=100.0, stop_grace_s=10.0)
    assert j2.feed(0.0, tk(95.0, 3.0, 0.0)) is None
    assert j2.feed(11.0, tk(96.0, 3.0, 0.0)) == '완주'    # 유예 초과 → 완주


def test_judge_stall():
    """정지(v<0.1)인데 계획은 진행(v_target≥0.5)이 지속되면 stall."""
    j = judge(stall_end_s=20.0, no_progress_end_s=120.0)
    assert j.feed(0.0, tk(100.0, 0.0, 5.0)) is None
    assert j.feed(10.0, tk(100.0, 0.0, 5.0)) is None
    assert j.feed(21.0, tk(100.0, 0.0, 5.0)) == 'stall'


def test_judge_stall_resets_when_moving_or_planned_stop():
    """움직이거나(v↑) 계획 정지(v_target=0)면 stall 타이머가 풀린다."""
    j = judge(stall_end_s=20.0, no_progress_end_s=120.0)
    assert j.feed(0.0, tk(100.0, 0.0, 5.0)) is None
    assert j.feed(15.0, RED(100.0)) is None               # 적신호 정차 → 리셋
    assert j.feed(30.0, tk(100.0, 0.0, 5.0)) is None      # 다시 시작 — 아직 20s 안 됨
    assert j.feed(45.0, tk(100.0, 0.0, 5.0)) is None
    assert j.feed(51.0, tk(100.0, 0.0, 5.0)) == 'stall'


def test_judge_no_progress():
    """route_s 무전진이 임계를 넘으면 no_progress — 신호 대기(60 s)는 견딘다."""
    j = judge(stall_end_s=20.0, no_progress_end_s=120.0)
    assert j.feed(0.0, RED(100.0)) is None
    assert j.feed(60.0, RED(100.0)) is None               # 적60 대기 수준 — 아직 아님
    assert j.feed(119.0, RED(100.0)) is None
    assert j.feed(121.0, RED(100.0)) == 'no_progress'     # 신호 고장도 결국 끝난다
    # 전진이 재개되면 타이머 리셋
    j2 = judge(no_progress_end_s=120.0)
    assert j2.feed(0.0, RED(100.0)) is None
    assert j2.feed(100.0, tk(101.0, 2.0, 2.0)) is None    # +1 m 전진 → 리셋
    assert j2.feed(219.0, RED(101.0)) is None
    assert j2.feed(221.0, RED(101.0)) == 'no_progress'


def test_judge_normal_driving_never_ends():
    """정상 주행(전진 중)에는 어떤 판정도 나오면 안 된다."""
    j = judge()
    t, s = 0.0, 0.0
    while s < 900.0:
        assert j.feed(t, tk(s, 8.0, 8.0)) is None
        t += 1.0
        s += 8.0


# ── blocked: 앞이 막혀 못 감 (2026-08-30 정지차량 2대 사이 끼임) ──────────
def test_judge_blocked_by_stopped_lead():
    """선행차 정지는 예외가 아니다 — 앞차가 안 가면 우리도 영원히 못 간다."""
    j = judge(blocked_end_s=15.0)
    lead = tk(100.0, 0.0, 0.0, winner='lead', lead_type='vehicle.vtd.object')
    assert j.feed(0.0, lead) is None
    assert j.feed(14.0, lead) is None
    assert j.feed(16.0, lead) == 'blocked'


def test_judge_blocked_and_stall_are_exclusive():
    """v_target 으로 갈린다 — 계획이 진행이면 stall, 계획도 정지면 blocked."""
    j = judge(stall_end_s=20.0, blocked_end_s=15.0)
    assert j.feed(0.0, tk(100.0, 0.0, 5.0)) is None       # 계획 진행 → stall 타이머
    assert j.feed(16.0, tk(100.0, 0.0, 5.0)) is None      # blocked(15 s) 는 안 걸린다
    assert j.feed(21.0, tk(100.0, 0.0, 5.0)) == 'stall'


def test_judge_red_light_wait_is_not_blocked():
    """적신호 대기는 blocked 로 죽이지 않는다 (no_progress 가 안전망)."""
    j = judge(blocked_end_s=15.0, no_progress_end_s=120.0)
    for now in (0.0, 20.0, 50.0, 90.0, 119.0):
        assert j.feed(now, RED(100.0)) is None


def test_judge_yellow_and_flashing_also_excused():
    j = judge(blocked_end_s=15.0)
    for state in (1, 2, 6):                                # 적·황·점멸
        jj = judge(blocked_end_s=15.0)
        hold = tk(100.0, 0.0, 0.0, winner='light', light=[147, state], red_light=0.0)
        assert jj.feed(0.0, hold) is None and jj.feed(30.0, hold) is None


def test_judge_distant_red_light_does_not_excuse():
    """멀리 있는 적신호는 우리를 세우고 있지 않다 — red_light 후보가 낮지 않다."""
    j = judge(blocked_end_s=15.0)
    far = tk(100.0, 0.0, 0.0, winner='lead', light=[147, 1], red_light=6.9,
             lead_type='vehicle.vtd.object')
    assert j.feed(0.0, far) is None
    assert j.feed(16.0, far) == 'blocked'


def test_judge_green_ends_the_signal_excuse():
    """녹색이 되면 예외가 끝난다 — 녹색인데 안 가는 건 정상 정차가 아니다.

    실측(완주속도_01_좌회전0): t+373.1 녹색 전환 후 37.9 s 정지 → t+388.2 blocked.
    """
    j = judge(blocked_end_s=15.0)
    assert j.feed(0.0, RED(100.0)) is None
    green = tk(100.0, 0.0, 0.0, winner='none', light=[147, 5], red_light=6.9)
    assert j.feed(10.0, green) is None
    assert j.feed(26.0, green) == 'blocked'


def test_judge_pedestrian_crossing_is_excused():
    j = judge(blocked_end_s=15.0)
    for winner in ('walker', 'bicycle'):
        jj = judge(blocked_end_s=15.0)
        ped = tk(100.0, 0.0, 0.0, winner=winner)
        assert jj.feed(0.0, ped) is None and jj.feed(30.0, ped) is None


def test_judge_lead_that_is_a_pedestrian_is_excused():
    """winner 가 lead 여도 그 lead 가 보행자면 예외 (승인 규칙)."""
    j = judge(blocked_end_s=15.0)
    ped_lead = tk(100.0, 0.0, 0.0, winner='lead', lead_type='walker.vtd.pedestrian')
    assert j.feed(0.0, ped_lead) is None
    assert j.feed(30.0, ped_lead) is None


def test_judge_blocked_resets_when_moving():
    j = judge(blocked_end_s=15.0)
    lead = tk(100.0, 0.0, 0.0, winner='lead', lead_type='vehicle.vtd.object')
    assert j.feed(0.0, lead) is None
    assert j.feed(10.0, tk(101.0, 3.0, 3.0)) is None       # 다시 굴러감 → 리셋
    assert j.feed(20.0, lead) is None                      # 처음부터 다시
    assert j.feed(36.0, lead) == 'blocked'


# ── 임계는 전부 params 에서 (하드코딩 금지) ──────────────────────────────
def test_thresholds_come_from_params():
    for k in ('stop_grace_s', 'stall_end_s', 'blocked_end_s', 'no_progress_end_s',
              'stall_speed_mps', 'stall_intent_mps', 'progress_eps_m'):
        assert k in BATCH, k
    j = EndJudge(total=1000.0, margin=8.8)                 # cfg 생략 = params 그대로
    assert j.blocked_s == float(BATCH['blocked_end_s'])
    assert j.grace == float(BATCH['stop_grace_s'])
    assert j.v_stop == float(BATCH['stall_speed_mps'])


def test_stop_excused_direct():
    intent = float(BATCH['stall_intent_mps'])
    assert stop_excused(RED(100.0), intent)
    assert not stop_excused(tk(100.0, 0.0, 0.0, winner='lead',
                               lead_type='vehicle.vtd.object'), intent)
