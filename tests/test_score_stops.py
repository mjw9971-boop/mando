"""채점기 소항목 (S, 2026-09-06): 정지 지표 '전환' 과 적신호 원거리 정지 원인 태그.

둘 다 판정을 바꾸지 않는다 — 스위치 off 면 이전 출력과 같다.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

import score as SC                                              # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402

CFG = load_params_yaml()


def _tk(t, v, accel, *, front=-0.7, ctrl=(9, 10), state=1, winner='light', avoid=None,
        reduced=None, walker_lead=False):
    reasons = {'winner': winner, 'avoid': avoid}
    if reduced:
        reasons['speed_reduced_by'] = reduced
    return {'t': t, 'raw': {'lights': [[c, state] for c in ctrl]},
            'ego': {'speed': v, 'route_s': 100.0 + t, 'lane': [1, 0, -1], 's': 1.0, 't_off': 0.0},
            'cmd': {'accel': accel},
            'world': {'valid': True, 'flags': {'stop_ctrl_ids': list(ctrl)},
                      'stop_line_front_m': front, 'summ': {}},
            'decision': {'v_target': 0.0, 'state': winner, 'reasons': reasons}}


def _approach_then_stop(accels):
    """15 s 안의 최고속 지점부터 정지까지 accel 열이 accels 인 틱 열 + 정지 2 s."""
    # stop_metrics 는 정지 시작 인덱스 i > 20 을 요구한다 — 순항 틱을 앞에 깐다.
    ticks = [_tk(0.1 * j, 10.0, 0.0) for j in range(30)]   # 순항 (최고속 = 첫 틱)
    n = len(accels)
    for i, a in enumerate(accels):
        v = 10.0 * (1 - (i + 1) / n)
        ticks.append(_tk(3.0 + 0.1 * (i + 1), max(v, 0.6), a))
    t = 3.0 + 0.1 * (n + 1)
    for j in range(25):                                   # v < 0.5 정지 2.5 s
        ticks.append(_tk(t + 0.1 * j, 0.0, -1.0))
    return ticks


def test_params_present_default_off():
    assert CFG['score']['cmd_reversals_accel_sign'] is False
    assert float(CFG['score']['cmd_reversals_deadband_mps2']) > 0
    assert CFG['scoring']['red_stop_far_cause_tag'] is False


def test_reversal_metric_jerk_vs_accel_sign():
    # 미세 떨림: -0.24, -0.04, +0.07, +0.06, -0.24, +0.11, -0.19, +0.05 … (실측 rs 693.2 꼴)
    ripple = [-0.24, -0.04, 0.07, 0.06, -0.24, 0.11, -0.19, 0.05, -0.2, 0.08, -4.0, -4.0, -4.0]
    ticks = _approach_then_stop(ripple)
    off = SC.stop_metrics(ticks, 0.0, 0.5)
    on = SC.stop_metrics(ticks, 0.0, 0.5, accel_sign=True, deadband=0.1)
    assert len(off) == 1 and len(on) == 1
    assert off[0]['cmd_reversals'] == off[0]['jerk_flips']
    assert on[0]['cmd_reversals'] == on[0]['accel_flips']
    assert off[0]['jerk_flips'] == on[0]['jerk_flips']            # 옛 지표는 그대로 남는다
    # 데드밴드 0.1 안의 ±0.04~0.08 은 안 센다: 부호열 - 0 0 0 - + - 0 - 0 - - -  → 전환 2
    assert on[0]['accel_flips'] == 2
    assert off[0]['jerk_flips'] > on[0]['accel_flips']


def test_reversal_metric_monotonic_brake_is_zero():
    ticks = _approach_then_stop([-1.0, -2.0, -3.0, -4.0, -4.0, -4.0])
    on = SC.stop_metrics(ticks, 0.0, 0.5, accel_sign=True)
    assert on[0]['accel_flips'] == 0


def _far_stop(**kw):
    """정지선 12.57 m 앞 적색 정지 에피소드 (02_직진11 rs 295.1 실측 꼴)."""
    return [_tk(0.1 * i, 0.0, -1.0, front=-12.57, **kw) for i in range(12)]


def test_red_stop_far_cause_tag_off_and_on():
    sc = CFG['scoring']
    ticks = _far_stop(winner='walker')
    _ok, far, _enc = SC.detect_red_stop(ticks, 0.0, 0.5, sc)
    assert len(far) == 1 and 'cause' not in far[0]
    _ok, far, _enc = SC.detect_red_stop(ticks, 0.0, 0.5, sc, cause_tag=True)
    assert far[0]['cause'] == 'pedestrian'


def test_red_stop_far_cause_queue_and_pure():
    sc = CFG['scoring']
    q = _far_stop(avoid={'state': 'SUPPRESS', 'suppress': 'queue', 'standoff_id': 15})
    assert SC.detect_red_stop(q, 0.0, 0.5, sc, cause_tag=True)[1][0]['cause'] == 'queue'
    so = _far_stop(avoid={'state': 'STANDOFF', 'standoff_id': 11})
    assert SC.detect_red_stop(so, 0.0, 0.5, sc, cause_tag=True)[1][0]['cause'] == 'queue'
    p = _far_stop()
    assert SC.detect_red_stop(p, 0.0, 0.5, sc, cause_tag=True)[1][0]['cause'] == 'pure'
    lead = _far_stop(winner='lead', reduced={'type': 'walker.pedestrian', 'id': 4, 'dist': 3.0})
    assert SC.detect_red_stop(lead, 0.0, 0.5, sc, cause_tag=True)[1][0]['cause'] == 'pedestrian'


def test_cause_tag_does_not_change_classification():
    sc = CFG['scoring']
    ticks = _far_stop(winner='walker')
    a = SC.detect_red_stop(ticks, 0.0, 0.5, sc)
    b = SC.detect_red_stop(ticks, 0.0, 0.5, sc, cause_tag=True)
    assert [len(x) for x in a] == [len(x) for x in b]
    assert a[1][0]['front_m'] == b[1][0]['front_m']
