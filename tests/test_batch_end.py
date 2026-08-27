"""batch_run 종료 판정(EndJudge) + 완주 임계(end_margin_m) 검증.

2026-08-25 실사고: speed.stop_gap_m 1→4 튜닝 후 고정 임계 5 m 에 계획 정지점
(total − stop_gap − 앞범퍼 ≈ total − 7.8)이 못 미쳐 정상 완주가 timeout 처리.
임계는 params.yaml 의 정지 정책에서 유도해야 한다.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

from batch_run import EndJudge                                # noqa: E402
from summarize_run import end_margin_m, load_cfg              # noqa: E402


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
    j = EndJudge(total=368.80, margin=m)
    assert j.feed(0.0, 350.0, 8.0, 8.0) is None
    assert j.feed(10.0, 360.93, 0.0, 0.0) == '완주'


def test_judge_done_grace_when_still_rolling():
    """임계 도달 후 감속 중이면 유예, v<0.5 되면 완주, 유예 초과도 완주."""
    j = EndJudge(total=100.0, margin=8.8, grace_s=10.0)
    assert j.feed(0.0, 95.0, 3.0, 0.0) is None       # 도달했지만 아직 굴러감
    assert j.feed(2.0, 96.0, 0.2, 0.0) == '완주'      # 정지 → 완주
    j2 = EndJudge(total=100.0, margin=8.8, grace_s=10.0)
    assert j2.feed(0.0, 95.0, 3.0, 0.0) is None
    assert j2.feed(11.0, 96.0, 3.0, 0.0) == '완주'    # 유예 초과 → 완주


def test_judge_stall():
    """정지(v<0.1)인데 계획은 진행(v_target≥0.5)이 지속되면 stall."""
    j = EndJudge(total=1000.0, margin=8.8, stall_end_s=20.0, no_progress_end_s=120.0)
    assert j.feed(0.0, 100.0, 0.0, 5.0) is None
    assert j.feed(10.0, 100.0, 0.0, 5.0) is None
    assert j.feed(21.0, 100.0, 0.0, 5.0) == 'stall'


def test_judge_stall_resets_when_moving_or_planned_stop():
    """움직이거나(v↑) 계획 정지(v_target=0)면 stall 타이머가 풀린다."""
    j = EndJudge(total=1000.0, margin=8.8, stall_end_s=20.0, no_progress_end_s=120.0)
    assert j.feed(0.0, 100.0, 0.0, 5.0) is None
    assert j.feed(15.0, 100.0, 0.0, 0.0) is None      # 계획 정지(신호 등) → 리셋
    assert j.feed(30.0, 100.0, 0.0, 5.0) is None      # 다시 시작 — 아직 20s 안 됨
    assert j.feed(45.0, 100.0, 0.0, 5.0) is None
    assert j.feed(51.0, 100.0, 0.0, 5.0) == 'stall'


def test_judge_no_progress():
    """route_s 무전진이 임계를 넘으면 no_progress — 신호 대기(60 s)는 견딘다."""
    j = EndJudge(total=1000.0, margin=8.8, stall_end_s=20.0, no_progress_end_s=120.0)
    assert j.feed(0.0, 100.0, 0.0, 0.0) is None
    assert j.feed(60.0, 100.0, 0.0, 0.0) is None      # 적60 대기 수준 — 아직 아님
    assert j.feed(119.0, 100.0, 0.0, 0.0) is None
    assert j.feed(121.0, 100.0, 0.0, 0.0) == 'no_progress'
    # 전진이 재개되면 타이머 리셋
    j2 = EndJudge(total=1000.0, margin=8.8, no_progress_end_s=120.0)
    assert j2.feed(0.0, 100.0, 0.0, 0.0) is None
    assert j2.feed(100.0, 101.0, 2.0, 2.0) is None    # +1 m 전진 → 리셋
    assert j2.feed(219.0, 101.0, 0.0, 0.0) is None
    assert j2.feed(221.0, 101.0, 0.0, 0.0) == 'no_progress'


def test_judge_normal_driving_never_ends():
    """정상 주행(전진 중)에는 어떤 판정도 나오면 안 된다."""
    j = EndJudge(total=1000.0, margin=8.8)
    t, s = 0.0, 0.0
    while s < 900.0:
        assert j.feed(t, s, 8.0, 8.0) is None
        t += 1.0
        s += 8.0
