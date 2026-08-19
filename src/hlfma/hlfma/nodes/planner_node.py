"""
/world_state → core.Planner → core.Shield → /decision

Planner 와 Shield 는 같은 노드에 둔다. 둘 사이에는 토픽을 두지 않는다 —
Shield 는 Planner 출력을 **반드시** 통과해야 하는 가드라, 토픽으로 갈라두면
Shield 를 건너뛴 Decision 이 존재할 수 있게 된다.
"""
from __future__ import annotations

import pickle

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from hlfma_msgs.msg import Decision, WorldState

from hlfma.convert import decision_to_msg, msg_to_world
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.planner import Planner
from hlfma.core.shield import Shield
from hlfma.core.timing import Stage, Timer
from hlfma.nodes.params import collect_core_config
from hlfma.nodes.qos import LATEST


class PlannerNode(Node):
    def __init__(self) -> None:
        super().__init__('planner')

        self.declare_parameter('graph_path', 'data/lane_graph.pkl')
        self.declare_parameter('route_path', 'data/route.pkl')
        cfg = collect_core_config(self)

        lg = LaneGraph(self.get_parameter('graph_path').value)
        route = None
        try:
            with open(self.get_parameter('route_path').value, 'rb') as f:
                route = pickle.load(f)
        except (OSError, pickle.UnpicklingError) as e:
            self.get_logger().warn(f'route 로드 실패 ({e}) — 경로 없이 진행')

        self.planner = Planner(lg, route, cfg)
        # shield 가 차선변경을 중단시키려면 planner 상태를 되돌려야 한다
        self.shield = Shield(lg, cfg, planner=self.planner)

        self.pub = self.create_publisher(Decision, '/decision', LATEST)
        self.create_subscription(WorldState, '/world_state', self._on_world, LATEST)

        self.st_all = Stage('planner 콜백')
        self.st_conv = Stage('  msg→world')
        self.st_plan = Stage('  planner.plan')
        self.st_shield = Stage('  shield.apply')
        self.st_pub = Stage('  decision→msg+publish')

    def _on_world(self, msg: WorldState) -> None:
        with Timer(self.st_all, msg.t):
            with Timer(self.st_conv, msg.t):
                ws = msg_to_world(msg)
            with Timer(self.st_plan, msg.t):
                d = self.planner.plan(ws)
            with Timer(self.st_shield, msg.t):
                d = self.shield.apply(ws, d)
            with Timer(self.st_pub, msg.t):
                self.pub.publish(decision_to_msg(d, msg.header.stamp))

    def destroy_node(self) -> None:
        for st in (self.st_all, self.st_conv, self.st_plan, self.st_shield, self.st_pub):
            self.get_logger().info(st.summary())
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
