"""
소멸 차로 → successor 인계 첫 틱의 t_off 클램프 산물 (ego.handover_tangent_enable).

2026-09-06 배치 01 (실전주행_교통류_01_좌회전24): 자차가 successor 시작점 1.3 m 앞에
있는데 매칭이 successor s=0 으로 붙으면서 t_off 가 −1.34 m 로 찍혔다 (실제 ±0.06).
project() 가 u 를 [0,1] 로 클램프해 종방향 부족분이 t 에 섞인 것. 스위치를 켜면
끝점 밖의 t 를 끝 세그먼트 접선 연장에 대한 수직 거리로 잰다 — s·dist·차로 선택은 불변.
"""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from vtd_adapter.config import load_params_yaml                 # noqa: E402
from vtd_adapter.ego import EgoTracker                          # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                     # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml()
SUCC = (126, 0, -3)            # (1154,0,-2) 소멸 연결로의 successor

# 9/6 배치 01 로그의 인계 첫 틱 raw ego (x, y, yaw) 와 그때 매칭된 차로
HANDOVER_TICKS = [
    ((712.4290, -259.0815, -2.590826), (126, 0, -3), -1.336),
    ((666.8979, -298.5976, -2.297193), (126, 3, -2), -1.219),
    ((673.5133, -862.5588, -0.468302), (1818, 0, -1), -1.340),
]


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH), cfg=CFG)


def _start_frame(lg, key):
    """차로 시작점과 첫 세그먼트 단위 접선 (tx, ty), 왼쪽 법선 (nx, ny)"""
    P = lg.lanes[key]['pts']
    tx, ty = P[1, 0] - P[0, 0], P[1, 1] - P[0, 1]
    n = math.hypot(tx, ty)
    tx, ty = tx / n, ty / n
    return (float(P[0, 0]), float(P[0, 1])), (tx, ty), (-ty, tx)


def test_params_default_off():
    assert CFG['ego']['handover_tangent_enable'] is False


def test_project_before_start_off_mixes_longitudinal(lg):
    (x0, y0), (tx, ty), (nx, ny) = _start_frame(lg, SUCC)
    x, y = x0 - 1.3 * tx + 0.1 * nx, y0 - 1.3 * ty + 0.1 * ny
    s, t, dist, _ = lg.project(SUCC, x, y)
    assert s == 0.0
    assert abs(abs(t) - math.hypot(1.3, 0.1)) < 1e-6      # 끝점까지 거리 = 산물
    assert t > 0                                            # 왼쪽 = +


def test_project_before_start_on_uses_tangent(lg):
    (x0, y0), (tx, ty), (nx, ny) = _start_frame(lg, SUCC)
    x, y = x0 - 1.3 * tx + 0.1 * nx, y0 - 1.3 * ty + 0.1 * ny
    s_off, _t_off, d_off, j_off = lg.project(SUCC, x, y)
    s_on, t_on, d_on, j_on = lg.project(SUCC, x, y, tangent_ends=True)
    assert abs(t_on - 0.1) < 1e-6
    # s · dist · 세그먼트는 그대로 — 차로 선택 점수는 바뀌지 않는다
    assert (s_on, d_on, j_on) == (s_off, d_off, j_off)


def test_project_inside_lane_identical(lg):
    (x0, y0), (tx, ty), (nx, ny) = _start_frame(lg, SUCC)
    for along in (0.5, 5.0, 20.0):
        x, y = x0 + along * tx - 0.3 * nx, y0 + along * ty - 0.3 * ny
        assert lg.project(SUCC, x, y) == lg.project(SUCC, x, y, tangent_ends=True)


def test_project_past_end_on_uses_tangent(lg):
    P = lg.lanes[SUCC]['pts']
    tx, ty = P[-1, 0] - P[-2, 0], P[-1, 1] - P[-2, 1]
    n = math.hypot(tx, ty)
    tx, ty = tx / n, ty / n
    nx, ny = -ty, tx
    x, y = P[-1, 0] + 2.0 * tx - 0.2 * nx, P[-1, 1] + 2.0 * ty - 0.2 * ny
    s_off, t_off, d_off, _ = lg.project(SUCC, x, y)
    s_on, t_on, d_on, _ = lg.project(SUCC, x, y, tangent_ends=True)
    assert abs(abs(t_off) - math.hypot(2.0, 0.2)) < 1e-6
    assert abs(t_on + 0.2) < 1e-6
    assert (s_on, d_on) == (s_off, d_off)


@pytest.mark.parametrize('pose,lane,log_t', HANDOVER_TICKS)
def test_locate_at_batch_0906_handover_ticks(lg, pose, lane, log_t):
    """로그 t_off 가 off 로 재현되고, on 이면 같은 차로에서 |t| < 0.2 로 준다."""
    x, y, yaw = pose
    m_off = lg.locate(x, y, yaw, prefer=[lane])
    m_on = lg.locate(x, y, yaw, prefer=[lane], tangent_ends=True)
    assert m_off.lane == lane and m_on.lane == lane
    assert m_off.s == 0.0 and m_on.s == 0.0
    assert abs(m_off.t - log_t) < 0.01                       # 로그값 재현 (클램프 산물)
    assert abs(m_on.t) < 0.2
    assert m_on.dist == m_off.dist and m_on.heading_err == m_off.heading_err


def test_tracker_switch_default_off_and_on(lg):
    cfg_off = dict(CFG)
    cfg_off.pop('ego', None)
    assert EgoTracker(lg, None, cfg_off)._tangent_ends is False
    assert EgoTracker(lg, None, CFG)._tangent_ends is False
    cfg_on = dict(CFG)
    cfg_on['ego'] = {'handover_tangent_enable': True}
    assert EgoTracker(lg, None, cfg_on)._tangent_ends is True
