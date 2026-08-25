"""
속도 추정 (슬라이딩 창 Σ변위/Σ벽시계 dt).

2026-08-23 실측: 9910 송신 간격이 40/80 ms 로 불규칙한데 dt 하한을 50 ms 로
잡은 옛 추정기는 −15 % 편향 → v_target 45 km/h 에 실속도 53 km/h (S1.1.01 감점).
"""
import math
import pathlib

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.ego import EgoTracker
from vtd_adapter.lanegraph import LaneGraph
from vtd_adapter.types import RawPacket

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml(PARAMS_YAML)
X, Y, YAW = 4.933, -24.564, 1.70421

pytestmark = pytest.mark.skipif(not GRAPH.exists(), reason='lane_graph.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


def drive(per, v, dts, t0=100.0):
    """v [m/s] 등속으로 dts 간격의 틱을 먹인다. 마지막 WorldState 를 돌려준다.
    이미 틱을 먹은 perception 이면 마지막 위치/시각에서 이어 간다."""
    if per._prev_xy is not None:
        (x, y), t = per._prev_xy, per._prev_t
    else:
        x, y, t = X, Y, t0
    ws = None
    for dt in dts:
        t += dt
        x += v * dt * math.cos(YAW)
        y += v * dt * math.sin(YAW)
        ws = per.update(RawPacket(t_recv=t, ego=(x, y, 0.0, YAW, 0.0, 0.0), objects=[], lights=[]))
    return ws


def test_irregular_40_80ms_ticks_are_unbiased(lg):
    """40/80 ms 교대 도착(실측 패턴)에서 등속 14.7 m/s 가 ±1 % 로 나와야 한다."""
    per = EgoTracker(lg, None, CFG)
    v = 14.7
    ws = drive(per, v, [0.04, 0.04, 0.08] * 20)
    assert ws.ego.speed == pytest.approx(v, rel=0.01)
    assert 'reset' not in ws.flags


def test_constant_speed_has_no_lag(lg):
    """등속이면 창이 차는 즉시 정확한 값 (LPF 지연 없음)."""
    per = EgoTracker(lg, None, CFG)
    ws = drive(per, 5.0, [0.05] * 12)
    assert ws.ego.speed == pytest.approx(5.0, rel=1e-3)


def test_stall_then_burst_tick_does_not_spike(lg):
    """2 s 스톨(정상 주행 변위) 뒤 10 ms 간격 틱이 와도 속도가 튀지 않는다."""
    per = EgoTracker(lg, None, CFG)
    v = 9.0
    drive(per, v, [0.05] * 10)
    ws = drive(per, v, [2.0, 0.01, 0.01, 0.04])
    assert ws.flags.get('reset') is None
    assert ws.ego.speed == pytest.approx(v, rel=0.05)


def test_reset_clears_window(lg):
    per = EgoTracker(lg, None, CFG)
    drive(per, 5.0, [0.05] * 10)
    t = per._prev_t + 0.05
    ws = per.update(RawPacket(t_recv=t, ego=(X - 3.3, Y - 0.4, 0.0, YAW, 0.0, 0.0), objects=[], lights=[]))
    assert ws.flags.get('reset') is True
    assert ws.ego.speed == 0.0
    assert per._win == []
