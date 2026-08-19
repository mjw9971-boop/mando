"""
노드 4개를 한 프로세스에 올린다 (기본 실행 방식).

노드 간 지연을 없애는 게 목적이다. vtd_bridge 의 recv 루프는 자기 스레드에서 돌고,
나머지 세 노드의 콜백 체인은 MultiThreadedExecutor 위에서 돈다.

콜백 체인이 순서대로 흐르도록 perception/planner/control 은 각각
MutuallyExclusiveCallbackGroup 에 둔다 — 같은 노드의 콜백이 겹쳐 실행되면
core 로직의 내부 상태(속도 추정, PI 적분)가 깨진다.
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor

from hlfma.nodes.control_node import ControlNode
from hlfma.nodes.logger_node import LoggerNode
from hlfma.nodes.perception_node import PerceptionNode
from hlfma.nodes.planner_node import PlannerNode
from hlfma.nodes.vtd_bridge_node import VtdBridgeNode


def main(args=None) -> None:
    rclpy.init(args=args)

    nodes = [VtdBridgeNode(), PerceptionNode(), PlannerNode(), ControlNode(), LoggerNode()]
    ex = MultiThreadedExecutor(num_threads=5)
    for n in nodes:
        ex.add_node(n)

    nodes[0].get_logger().info(f'단일 프로세스로 {len(nodes)}개 노드 기동')
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for n in nodes:
            n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
