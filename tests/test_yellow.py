"""
황색 딜레마 원샷 판정 (C, kr_rules._yellow_latch, params speed.a_yellow).

    v ≤ √(2·a_yellow·(정지선거리 − s0))  → STOP (적색과 동일 취급)
    그 외                                → GO   (신호·정지선 유래 후보 미생성)

**STOP 우선**이 채점표에서 나온 편향이다 — 황색 정지를 감점하는 항목이 없고
(항목8 5 s 카운트는 녹색 틱만 센다), 적색 통과·걸침은 항목7 중대다. 게다가
score.detect_red_light 는 **통과 순간의 신호**로 판정하므로 GO 로 나갔다 적색에
걸리면 그대로 중대다. 그래서 a_yellow 를 확실히 실행 가능한 최대(a_dec_max 와
같은 4.0)로 두어 STOP 영역을 최대화한다.

실측 근거 (2026-08-30, logs/batch/20260830_181706):
  실전주행_02 ctrl 162 — 황색 onset v=6.75 d=22.93 → v_allow(4.0)=11.88 이라
  STOP 이 여유 있게 가능(필요감속 1.29)했는데, ④′가 적색 한정이라 개입하지
  못했다. 황색 3 s 동안 PDM IDM 이 오히려 가속을 요구해(rl>v) 타행했고,
  적색 전환 시점에 이미 d=6.01 → 걸침 정지 slf=+1.35.

여기서 지키는 불변:
  · 판정은 **접근당 1회**. GO 중 적색 전환에도 번복하지 않는다 (교차로 한복판
    급제동 금지).
  · STOP 시 판정과 실행이 **같은 상수**(a_yellow) — 방안 B. 판정 시
    v ≤ v_allow 가 보장되므로 프로파일 위에서 시작하는 전이가 없다.
  · 래치 해제는 ① 다른 신호 ② 녹색 복귀 ③ 교차로 통과 가드 종료 ④ courseRespawn.
  · a_yellow = 0 이면 판정하지 않는다 (황색을 PDM 원문에만 맡기는 기존 동작).
"""
import copy
import math
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.carla_types import TrafficLightState
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from run_agent import build_pdm_config                              # noqa: E402
from test_route_end import FakePlanner, TOTAL, _apply               # noqa: E402
from test_stop_profile import FRONT, S0, make_ap                    # noqa: E402
from test_stopline_stop import FakePlannerTL                        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
A_Y = CFG['speed']['a_yellow']
A_STOP = CFG['speed']['stop_profile_a']
D_FAR = TOTAL - 200.0


def v_allow(d_line, a):
    return math.sqrt(2.0 * a * max(0.0, d_line - S0))


def yellow(d_tl, tl_id=7):
    p = FakePlannerTL(d_tl=d_tl)
    p.tl.state = TrafficLightState.Yellow
    p.tl.id = tl_id            # FakeTL 에는 id 가 없다 — 래치는 신호 id 로 구분한다
    return p


# ── 판정 경계 ───────────────────────────────────────────────────────────
def test_judgement_uses_a_yellow_not_stop_profile_a():
    """판정 감속은 a_yellow(4.0)이지 프로파일의 a_stop(3.0)이 아니다 —
    STOP 영역을 넓히는 것이 설계 의도다."""
    assert A_Y > A_STOP
    d = S0 + 20.0
    p, ap = yellow(d), None
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, v_allow(d, A_STOP) + 0.5, ap)   # a_stop 기준으론 GO
    assert ap.kr_rules.y_decision == 'stop'                       # a_yellow 기준이면 STOP


@pytest.mark.parametrize('margin,expect', [(-0.5, 'stop'), (-0.01, 'stop'),
                                           (0.0, 'stop'), (0.01, 'go'), (2.0, 'go')])
def test_decision_boundary_at_v_equals_v_allow(margin, expect):
    """경계 v = v_allow 는 STOP 쪽에 포함된다 (STOP 우선)."""
    d = S0 + 15.0
    p = yellow(d)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, v_allow(d, A_Y) + margin, ap)
    assert ap.kr_rules.y_decision == expect


def test_stop_executes_with_stop_profile_a_not_a_yellow():
    """판정은 a_yellow(4.0), **실행은 stop_profile_a(3.0)** — 상수가 다른 것이
    의도다. 같게 두면 진입 시 프로파일이 느슨해 늦게 구속되고, 그때는 최대
    감속을 여유 0 으로 요구해 jerk 램프인에 진다 (폐루프 걸침 6/12)."""
    d = S0 + 15.0
    p = yellow(d)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, 1.0, ap)
    assert ap.kr_rules.y_decision == 'stop'
    assert ap.kr_rules._stopline_profile(p, ap) == pytest.approx(v_allow(d, A_STOP))
    assert v_allow(d, A_STOP) < v_allow(d, A_Y)


def test_yellow_stop_is_identical_to_red():
    """황색 STOP 은 적색과 완전히 같은 후보를 낸다 (해석 단일 출처)."""
    d = S0 + 15.0
    py = yellow(d)
    apy = make_ap(py)
    apy.kr_rules._yellow_latch(py, 1.0, apy)
    pr = FakePlannerTL(d_tl=d)                                     # 기본 Red
    apr = make_ap(pr)
    assert apy.kr_rules._stopline_profile(py, apy) == pytest.approx(
        apr.kr_rules._stopline_profile(pr, apr))


# ── GO ──────────────────────────────────────────────────────────────────
def test_go_generates_no_signal_candidate():
    """GO 면 신호·정지선 유래 후보를 만들지 않는다 — 무제동 통과."""
    d = S0 + 8.0
    p = yellow(d)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, v_allow(d, A_Y) + 3.0, ap)
    assert ap.kr_rules.y_decision == 'go'
    assert ap.kr_rules._stopline_profile(p, ap) is None
    assert ap.kr_rules._stopline_hold(p, 0.1) is None
    assert _apply(ap, D_FAR, v=10.0)[1] == pytest.approx(12.5)     # 목표 손대지 않음


def test_go_latch_survives_red_transition_without_rebraking():
    """GO 중 적색으로 바뀌어도 번복하지 않는다 — 교차로 한복판 급제동 금지."""
    d = S0 + 8.0
    p = yellow(d)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, v_allow(d, A_Y) + 3.0, ap)
    assert ap.kr_rules.y_decision == 'go'
    p.tl.state = TrafficLightState.Red
    ap.kr_rules._yellow_latch(p, 10.0, ap)
    assert ap.kr_rules.y_decision == 'go'                          # 유지
    assert ap.kr_rules._stopline_profile(p, ap) is None             # 후보 없음
    assert _apply(ap, D_FAR, v=10.0)[1] == pytest.approx(12.5)


# ── 래치 유지·해제 ──────────────────────────────────────────────────────
def test_decision_is_one_shot_per_approach():
    """접근 중 v 가 바뀌어도 판정은 다시 하지 않는다."""
    d = S0 + 15.0
    p = yellow(d)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, 1.0, ap)
    assert ap.kr_rules.y_decision == 'stop'
    for v in (20.0, 0.0, 30.0):
        ap.kr_rules._yellow_latch(p, v, ap)
        assert ap.kr_rules.y_decision == 'stop'


def test_latch_cleared_on_green_return():
    p = yellow(S0 + 15.0)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, 1.0, ap)
    assert ap.kr_rules.y_decision == 'stop'
    p.tl.state = TrafficLightState.Green
    ap.kr_rules._yellow_latch(p, 1.0, ap)
    assert ap.kr_rules.y_decision is None
    assert ap.kr_rules._stopline_profile(p, ap) is None             # 녹색 무감속


def test_latch_cleared_on_next_signal():
    """다음 교차로로 넘어가면 래치를 버리고 새로 판정한다."""
    d = S0 + 8.0
    p = yellow(d, tl_id=7)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, v_allow(d, A_Y) + 3.0, ap)
    assert (ap.kr_rules.y_decision, ap.kr_rules.y_ctrl) == ('go', 7)
    p.tl.id = 9                                                    # 다음 신호
    ap.kr_rules._yellow_latch(p, 1.0, ap)                          # 저속 → STOP
    assert (ap.kr_rules.y_decision, ap.kr_rules.y_ctrl) == ('stop', 9)


def test_on_reset_discards_go_latch():
    """courseRespawn — GO 가 살아 있으면 정지선 뒤로 되돌아가서도 통과한다."""
    d = S0 + 8.0
    p = yellow(d)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, v_allow(d, A_Y) + 3.0, ap)
    assert ap.kr_rules.y_decision == 'go'
    ap.kr_rules.on_reset()
    assert ap.kr_rules.y_decision is None
    assert ap.kr_rules.cross_guard is False
    ap.kr_rules._yellow_latch(p, 1.0, ap)                           # 다시 판정 가능
    assert ap.kr_rules.y_decision == 'stop'


# ── 회귀: 적색·녹색은 그대로 ─────────────────────────────────────────────
def test_red_still_uses_stop_profile_a():
    """적색 경로는 손대지 않았다 — 여전히 a_stop(3.0)."""
    d = S0 + 15.0
    p = FakePlannerTL(d_tl=d)                                       # 기본 Red
    ap = make_ap(p)
    assert ap.kr_rules._stopline_profile(p, ap) == pytest.approx(v_allow(d, A_STOP))


def test_green_unchanged():
    p = FakePlannerTL(d_tl=S0 + 5.0)
    p.tl.state = TrafficLightState.Green
    ap = make_ap(p)
    assert ap.kr_rules._stopline_profile(p, ap) is None
    assert _apply(ap, D_FAR, v=10.0)[1] == pytest.approx(12.5)


# ── 스위치 ──────────────────────────────────────────────────────────────
def test_disabled_switch_leaves_yellow_to_pdm():
    cfg = copy.deepcopy(CFG)
    cfg['speed']['a_yellow'] = 0.0
    p = yellow(S0 + 15.0)
    ap = make_ap(p, cfg)
    ap.kr_rules._yellow_latch(p, 1.0, ap)
    assert ap.kr_rules.y_decision is None
    assert ap.kr_rules._stopline_profile(p, ap) is None              # 황색 후보 없음


# ── 조기 반환 훅 (signal_release) — 발동 조건이 둘뿐임을 고정 ────────────
#
# kr_rules 는 min() 에 후보를 덧대기만 하므로, PDM 이 스스로 만드는 적신호
# 감속을 없애려면 autopilot 의 조기 반환에 항을 붙이는 수밖에 없다. 그 훅이
# 필요 이상으로 열리면 **적신호를 그냥 통과한다(항목7 중대)**. 그래서 참이 되는
# 경우가 ① 황색 GO 래치 ② 교차로 통과 가드 **둘뿐**임을 여기서 못 박는다.

def test_hook_true_only_for_yellow_go():
    d = S0 + 8.0
    p = yellow(d)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, v_allow(d, A_Y) + 3.0, ap)
    assert ap.kr_rules.y_decision == 'go'
    assert ap.kr_rules.signal_release(ap) is True


def test_hook_true_only_for_cross_guard():
    p = FakePlannerTL(d_tl=FRONT - 0.5)              # 앞범퍼가 정지선을 넘었다
    ap = make_ap(p)
    ap.kr_rules._stopline_profile(p, ap)             # 가드 진입
    assert ap.kr_rules.cross_guard is True
    assert ap.kr_rules.signal_release(ap) is True


def test_hook_false_on_red_approach_and_stop():
    """적신호 접근·정지 — 훅이 열리면 그냥 통과한다. 전 구간 거짓이어야 한다."""
    for d in (S0 + 60.0, S0 + 20.0, S0 + 5.0, S0, FRONT + 0.2):
        p = FakePlannerTL(d_tl=d)                    # 기본 Red
        ap = make_ap(p)
        for v in (12.0, 5.0, 0.5, 0.0):
            _apply(ap, D_FAR, v=v)
            assert ap.kr_rules.signal_release(ap) is False, (d, v)


def test_hook_false_on_yellow_stop():
    """황색 STOP 은 적색과 동일 취급 — 훅을 열면 안 된다."""
    d = S0 + 15.0
    p = yellow(d)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, 1.0, ap)
    assert ap.kr_rules.y_decision == 'stop'
    assert ap.kr_rules.signal_release(ap) is False


def test_hook_false_on_green():
    p = FakePlannerTL(d_tl=S0 + 10.0)
    p.tl.state = TrafficLightState.Green
    ap = make_ap(p)
    _apply(ap, D_FAR, v=10.0)
    assert ap.kr_rules.signal_release(ap) is False


def test_hook_false_during_route_end_stop():
    """route_end 유령차 정지 중 — 신호와 무관한 후보다. 훅은 거짓."""
    p = FakePlannerTL(d_tl=S0 + 200.0)               # 신호는 멀다
    ap = make_ap(p)
    _c, ts = _apply(ap, d_end=5.0, v=0.3)            # 종점 래치
    assert ts == pytest.approx(0.0)
    assert ap.kr_rules.signal_release(ap) is False


def test_hook_false_when_other_candidates_win():
    """보행자·선행차처럼 kr_rules 밖에서 온 후보가 이겨도 훅은 거짓이다.

    (kr_rules 는 그 후보들을 만들지 않는다 — PDM 의 min() 갈래다. 여기서는
    'kr_rules 상태가 어떻든 훅은 래치 둘에만 반응한다'를 고정한다.)
    """
    p = FakePlannerTL(d_tl=S0 + 30.0)
    ap = make_ap(p)
    for target in (0.0, 1.0, 5.0):                   # 보행자·선행차가 낮춘 목표를 모사
        _apply(ap, D_FAR, v=3.0, target=target)
        assert ap.kr_rules.signal_release(ap) is False


def test_hook_false_without_traffic_light_info():
    p = FakePlanner()
    ap = make_ap(p)
    assert ap.kr_rules.signal_release(ap) is False


def test_hook_false_after_go_latch_reset():
    """GO 가 풀리면 훅도 닫힌다 (녹색 복귀 / courseRespawn)."""
    d = S0 + 8.0
    p = yellow(d)
    ap = make_ap(p)
    ap.kr_rules._yellow_latch(p, v_allow(d, A_Y) + 3.0, ap)
    assert ap.kr_rules.signal_release(ap) is True
    ap.kr_rules.on_reset()
    assert ap.kr_rules.signal_release(ap) is False
