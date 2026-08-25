"""
VTD courseRespawn(리셋) 감지.

대회 브릿지는 도로/경로를 벗어나면 차를 차로로 되돌린다
(hl_vtd_config.json: respawnEnabled=true, 이탈 0.3~0.5 s, 허용오차 1.5 m).
리셋을 못 잡으면 3 m 순간이동을 실제 주행으로 세서 속도 추정이 튀고,
제어기의 적분항이 리셋 전 오차를 그대로 안고 간다.
"""
import math
import pathlib

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.comm import build_frame, parse
from vtd_adapter.config import load_params_yaml
from vtd_adapter.ego import EgoTracker
from vtd_adapter.lanegraph import LaneGraph

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml(PARAMS_YAML)

# route_example 시작 지점 (차로 (30,0,-1))
X, Y, YAW = 4.933, -24.564, 1.70421

pytestmark = pytest.mark.skipif(not GRAPH.exists(), reason='lane_graph.pkl 없음')


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


def feed(per, x, y, yaw, t):
    return per.update(parse(build_frame((x, y, 42.0, yaw, 0.0, 0.0), [], [(0, 0)]), t_recv=t))


def test_param_threshold_matches_respawn_tolerance():
    """리셋 이동량이 3 m 남짓이라 문턱은 그보다 낮아야 한다."""
    assert CFG['percep']['jump_m'] < 3.0


def test_position_jump_is_detected_as_reset(lg):
    per = EgoTracker(lg, None, CFG)
    feed(per, X, Y, YAW, 100.0)
    feed(per, X + 0.25, Y + 0.05, YAW, 100.05)
    ws = feed(per, X - 3.3, Y - 0.4, YAW, 100.10)      # 리스폰
    assert ws.flags.get('reset') is True
    assert ws.flags.get('reset_count') == 1
    assert ws.flags['reset_jump_m'] == pytest.approx(3.3, abs=0.3)
    assert per.reset_count == 1


def test_normal_motion_is_not_a_reset(lg):
    """20 Hz 에서 5 m/s 면 한 틱에 0.25 m — 리셋이 아니다."""
    per = EgoTracker(lg, None, CFG)
    t = 100.0
    x, y = X, Y
    for _ in range(20):
        ws = feed(per, x, y, YAW, t)
        x += 0.25 * math.cos(YAW); y += 0.25 * math.sin(YAW); t += 0.05
    assert per.reset_count == 0
    assert 'reset' not in ws.flags


def test_speed_estimate_is_cleared_on_reset(lg):
    """순간이동을 주행으로 세면 속도가 60 m/s 로 튄다."""
    per = EgoTracker(lg, None, CFG)
    t = 100.0
    x, y = X, Y
    for _ in range(10):
        feed(per, x, y, YAW, t)
        x += 0.25 * math.cos(YAW); y += 0.25 * math.sin(YAW); t += 0.05
    ws = feed(per, x - 3.3, y - 0.4, YAW, t)
    assert ws.flags.get('reset') is True
    assert ws.ego.speed == pytest.approx(0.0, abs=1e-9)


def test_reset_count_accumulates(lg):
    """점프마다 1회씩 센다 (되돌아오는 것도 점프이므로 한 방향으로만 이동시킨다)."""
    per = EgoTracker(lg, None, CFG)
    t = 100.0
    feed(per, X, Y, YAW, t); t += 0.05
    for k in range(1, 4):
        feed(per, X + 4.0 * k, Y, YAW, t); t += 0.05
    assert per.reset_count == 3


def test_route_s_drop_is_detected(lg):
    """위치 점프가 작아도 경로상 위치가 크게 뒤로 가면 리셋이다."""
    import pickle
    rp = ROOT / 'data' / 'route_example.pkl'
    if not rp.exists():
        pytest.skip('route_example.pkl 없음')
    route = pickle.load(open(rp, 'rb'))
    per = EgoTracker(lg, route, CFG)
    # 경로를 따라 전진하다가 시작점 근처로 되돌려놓는다
    lane0 = route['lanes'][0]
    far_x, far_y, _z, far_h = lg.point_at(lane0, min(40.0, lg.length(lane0)))
    near_x, near_y, _z2, near_h = lg.point_at(lane0, 1.0)
    t = 100.0
    feed(per, far_x, far_y, far_h, t); t += 0.05
    ws = feed(per, near_x, near_y, near_h, t)
    assert ws.flags.get('reset') is True, ws.flags
