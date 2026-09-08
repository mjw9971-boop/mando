"""
ctrl24 커밋 6 — 도구 정합: batch_run EndJudge 완주 래치 · 정차 사유 'avoid'.

  · 임계에 한 번 닿은 뒤 route_s 가 떨어져도(경로 차로 이탈) 유예 만료 시 완주다.
  · 옛 동작(정지 시 즉시 완주, 유예 전 굴러가면 대기)은 그대로다 (test_batch_end 가 본다).
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

from batch_run import EndJudge, STOP_CAUSES, stop_cause          # noqa: E402
from test_batch_end import tk, judge                              # noqa: E402


def test_finish_latched_after_route_s_drops_below_threshold():
    j = judge()                                        # thr = total − margin
    thr = j.thr
    assert j.feed(0.0, tk(thr - 5.0, 8.0, 8.0)) is None
    assert j.feed(0.5, tk(thr + 1.0, 8.0, 8.0)) is None      # 임계 도달, 굴러가는 중
    assert j.reached_at == 0.5
    for k in range(1, 40):                             # 경로 차로를 벗어나 route_s 가 떨어진다
        out = j.feed(0.5 + 0.5 * k, tk(3.0 + k, 8.0, 8.0))
        if out is not None:
            break
    assert out == '완주' and 0.5 + 0.5 * k - 0.5 > j.grace


def test_finish_immediate_on_stop_unchanged():
    j = judge()
    assert j.feed(0.0, tk(j.thr + 0.5, 0.2, 0.0)) == '완주'


def test_stop_cause_avoid_label():
    assert 'avoid' in STOP_CAUSES
    t = tk(100.0, 0.0, 0.0, lead_type='vehicle.vtd.object')
    t['decision']['reasons']['avoid'] = {'state': 'NOOP', 'blocker': 3}
    t['decision']['reasons']['speed_reduced_by'] = {'type': 'vehicle.vtd.object', 'id': 3, 'dist': 5.0}
    t['objects'] = [{'id': 3, 'speed': 0.0}]
    assert stop_cause(t, 0.5) == 'avoid'
    t['decision']['reasons']['avoid'] = None
    assert stop_cause(t, 0.5) == 'stopped_lead'
