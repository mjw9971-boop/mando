"""
/decision (+ /world_state) → core.Control → /cmd

Control 은 목표 속도뿐 아니라 현재 속도/자세가 필요하다. Decision 에는 그게 없으므로
/world_state 를 같이 구독해 **가장 최근 것**을 들고 있다가 /decision 이 오면 계산한다.
(콜백 순서상 같은 틱의 world 가 항상 먼저 도착한다.)
"""
from __future__ import annotations

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from hlfma_msgs.msg import Cmd, Decision, WorldState

from hlfma.convert import command_to_msg, msg_to_decision, msg_to_world
from hlfma.core.control import Control
from hlfma.core.timing import Stage, Timer
from hlfma.nodes.params import collect_core_config
from hlfma.nodes.qos import LATEST


class ControlNode(Node):
    def __init__(self) -> None:
        super().__init__('control')

        cfg = collect_core_config(self)
        self.core = Control(cfg)
        self._world = None
        self._sign_reported = False
        self.st_all = Stage('control 콜백')

        self.pub = self.create_publisher(Cmd, '/cmd', LATEST)
        self.create_subscription(WorldState, '/world_state', self._on_world, LATEST)
        self.create_subscription(Decision, '/decision', self._on_decision, LATEST)

    def _on_world(self, msg: WorldState) -> None:
        self._world = msg_to_world(msg)

    def _on_decision(self, msg: Decision) -> None:
        if self._world is None:
            self.get_logger().warn('world_state 아직 없음 — /cmd 보류',
                                   throttle_duration_sec=2.0)
            return
        with Timer(self.st_all):
            cmd = self.core.compute(self._world, msg_to_decision(msg))
            self.pub.publish(command_to_msg(cmd, msg.header.stamp))
        self._report_steer_sign()

    def destroy_node(self) -> None:
        self.get_logger().info(self.st_all.summary())
        super().destroy_node()

    def _report_steer_sign(self) -> None:
        """조향 부호 판정이 나오면 한 번만 알린다."""
        if self._sign_reported:
            return
        m = self.core.sign_monitor
        if m.verdict is None:
            return
        self._sign_reported = True
        if m.verdict == 'inverted':
            self.get_logger().error(
                f'steer_sign 이 반대일 가능성 — 조향과 실제 회전이 역상관 '
                f'(corr={m.corr:+.3f}, 표본 {m.samples}개). '
                f'params.yaml 의 comm.steer_sign 부호를 뒤집을 것. '
                f'이대로면 직선에서도 발산해 리셋이 반복된다.')
        elif m.verdict == 'ok':
            self.get_logger().info(
                f'조향 부호 정상 (corr={m.corr:+.3f}, 표본 {m.samples}개)')
        else:
            self.get_logger().warn(
                f'조향 부호 판정 보류 — 표본 부족 ({m.samples}개). '
                f'직선 저속 구간만 주행했을 수 있다')


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlNode()
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
