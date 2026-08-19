"""
노드 체인 통합 테스트.

/gt_state 를 하나 넣으면 perception → planner → control 콜백이 도미노처럼
돌아 /cmd 가 나오는지 본다. 타이머 없이 콜백만으로 한 틱이 완결되는지 확인하는 게
목적이다 (요구사항 1).
"""
import json
import math
import pathlib
import threading
import time

import pytest

pytest.importorskip('rclpy', reason='ROS 2 환경 필요')
pytest.importorskip('hlfma_msgs', reason='colcon build + source install 필요')

import rclpy  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

from hlfma_msgs.msg import Cmd, GtState, Decision, WorldState  # noqa: E402

from hlfma.nodes.control_node import ControlNode  # noqa: E402
from hlfma.nodes.perception_node import PerceptionNode  # noqa: E402
from hlfma.nodes.planner_node import PlannerNode  # noqa: E402
from hlfma.nodes.qos import LATEST  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'
ROUTE = ROOT / 'data' / 'route.pkl'

# SPEC §1.1 검증 완료 초기 위치
START = (508.79968, -168.28766, 42.0, 0.52727822, 0.0, 0.0)

pytestmark = pytest.mark.skipif(
    not (GRAPH.exists() and ROUTE.exists()),
    reason='data/lane_graph.pkl 또는 data/route.pkl 없음 (gitignore 대상)')


def _gt(x, y, yaw, t):
    m = GtState()
    m.t_recv = t
    m.ego_x, m.ego_y, m.ego_z = x, y, START[2]
    m.ego_heading, m.ego_pitch, m.ego_roll = yaw, 0.0, 0.0
    return m


class _Harness:
    """perception/planner/control 3노드를 띄우고 /gt_state 를 주입한다."""

    def __init__(self):
        params = [{'graph_path': str(GRAPH), 'route_path': str(ROUTE)}]
        self.perception = PerceptionNode()
        self.planner = PlannerNode()
        self.control = ControlNode()
        self.probe = Node('probe')

        self.pub = self.probe.create_publisher(GtState, '/gt_state', LATEST)
        self.cmds, self.worlds, self.decisions = [], [], []
        self.probe.create_subscription(Cmd, '/cmd', self.cmds.append, LATEST)
        self.probe.create_subscription(WorldState, '/world_state', self.worlds.append, LATEST)
        self.probe.create_subscription(Decision, '/decision', self.decisions.append, LATEST)

        self.ex = MultiThreadedExecutor(num_threads=4)
        for n in (self.perception, self.planner, self.control, self.probe):
            self.ex.add_node(n)
        self._t = threading.Thread(target=self.ex.spin, daemon=True)
        self._t.start()
        time.sleep(1.0)          # 디스커버리

    def tick(self, x, y, yaw, t):
        self.pub.publish(_gt(x, y, yaw, t))

    def shutdown(self):
        self.ex.shutdown()
        for n in (self.perception, self.planner, self.control, self.probe):
            n.destroy_node()


@pytest.fixture(scope='module')
def harness():
    rclpy.init()
    h = _Harness()
    yield h
    h.shutdown()
    rclpy.shutdown()


def test_one_packet_produces_one_cmd(harness):
    """패킷 하나 = 한 틱. 타이머 없이 콜백 체인만으로 /cmd 까지 간다."""
    harness.cmds.clear()
    harness.tick(*START[:2], START[3], 100.0)
    deadline = time.time() + 3.0
    while not harness.cmds and time.time() < deadline:
        time.sleep(0.02)
    assert harness.cmds, '/cmd 가 나오지 않았다'


def test_world_state_matches_known_lane(harness):
    """초기위치는 road 465 / lane -1 로 매칭돼야 한다 (SPEC §1.1)."""
    harness.worlds.clear()
    harness.tick(*START[:2], START[3], 200.0)
    deadline = time.time() + 3.0
    while not harness.worlds and time.time() < deadline:
        time.sleep(0.02)
    assert harness.worlds
    w = harness.worlds[-1]
    assert w.valid is True
    assert w.has_lane is True
    assert (w.lane[0], w.lane[2]) == (465, -1)
    assert abs(abs(w.t_off) - 0.052) < 0.05


def test_speed_estimated_from_successive_packets(harness):
    """패킷에 속도가 없으므로 위치 차분으로 올라와야 한다."""
    harness.worlds.clear()
    x, y, yaw = START[0], START[1], START[3]
    for k in range(1, 25):
        d = 0.25 * k                      # 0.05 s 마다 0.25 m -> 5 m/s
        harness.tick(x + d * math.cos(yaw), y + d * math.sin(yaw), yaw, 300.0 + 0.05 * k)
        time.sleep(0.05)
    deadline = time.time() + 2.0
    while len(harness.worlds) < 5 and time.time() < deadline:
        time.sleep(0.02)
    assert harness.worlds[-1].ego_speed > 1.0


def test_decision_is_follow_with_candidate_speed(harness):
    """기본 경로는 상수속도가 아니라 _speed_candidates 중재 결과를 쓴다.

    debug.enabled 가 꺼져 있으면 debug_const 후보가 아예 생기지 않아야 하고,
    v_target 은 유한한 양수이면서 법정 제한속도를 넘지 않아야 한다.
    """
    harness.decisions.clear()
    harness.tick(*START[:2], START[3], 400.0)
    deadline = time.time() + 3.0
    while not harness.decisions and time.time() < deadline:
        time.sleep(0.02)
    assert harness.decisions
    d = harness.decisions[-1]
    assert d.state == 'FOLLOW'
    reasons = json.loads(d.reasons_json) if d.reasons_json else {}
    assert 'debug_const' not in reasons
    assert 'const_cap' not in reasons.get('shield', {})
    assert 0.0 < d.v_target < math.inf
    assert d.v_target <= float(reasons['limit']) + 1e-6
    assert reasons.get('winner') not in (None, 'none')
    assert len(d.path_x) == len(d.path_y) > 0


def test_cmd_steering_within_limits(harness):
    harness.cmds.clear()
    harness.tick(*START[:2], START[3], 500.0)
    deadline = time.time() + 3.0
    while not harness.cmds and time.time() < deadline:
        time.sleep(0.02)
    assert harness.cmds
    c = harness.cmds[-1]
    assert abs(c.steering) <= 0.48 + 1e-9
    assert -6.0 - 1e-9 <= c.accel <= 2.0 + 1e-9
