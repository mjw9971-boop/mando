"""
9910 소켓 <-> ROS 경계.

**이 노드만 소켓 recv 루프를 돈다.** 패킷 하나 = /gt_state 하나 = 한 틱.
나머지 노드는 전부 구독 콜백에서 계산한다(타이머 없음).

  recv 루프 ── /gt_state ──> perception ─> planner ─> control ── /cmd ──> 여기서 송신
"""
from __future__ import annotations

import select
import threading

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from hlfma_msgs.msg import Cmd, GtState

from hlfma.convert import msg_to_command, packet_to_msg
from hlfma.core.comm import SAFE_STOP, Comm
from hlfma.nodes.params import collect_core_config
from hlfma.nodes.qos import LATEST


class VtdBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('vtd_bridge')

        # 다른 노드와 **같은** 파라미터 이름(comm.*)을 쓴다. 여기서만 host/port 로
        # 따로 선언하면 params.yaml 이나 launch 인자가 이 노드에 닿지 않는다.
        cfg = collect_core_config(self)['comm']
        self.comm = Comm(
            host=str(cfg['host']),
            port=int(cfg['port']),
            watchdog_s=float(cfg['watchdog_s']),
            steer_sign=float(cfg['steer_sign']),
            connect_retry_s=float(cfg['connect_retry_s']),
            recv_bufsize=int(cfg['recv_bufsize']),
            logger=_RosLog(self.get_logger()),
        )

        self.pub = self.create_publisher(GtState, '/gt_state', LATEST)
        self.create_subscription(Cmd, '/cmd', self._on_cmd, LATEST)

        self._last_cmd = SAFE_STOP
        self._select_timeout = float(cfg['watchdog_s']) / 4.0
        self._lock = threading.Lock()
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

        self.get_logger().info(
            f"vtd_bridge 시작 {cfg['host']}:{cfg['port']} steer_sign={cfg['steer_sign']}")

    # ── /cmd → 소켓 ───────────────────────────────────────────────────────
    def _on_cmd(self, msg: Cmd) -> None:
        """
        콜백 체인이 한 바퀴 돈 결과. **여기서 바로 송신한다.**

        패킷 하나 = /gt_state 하나 = /cmd 하나 이므로 송신도 정확히 20 Hz 가 된다.
        recv 루프에서 매 반복 보내면 소켓 송신 버퍼가 차서 EAGAIN 이 나고,
        Comm.send 는 그걸 치명적 오류로 보고 연결을 끊어버린다.
        """
        cmd = msg_to_command(msg)
        with self._lock:
            self._last_cmd = cmd

        if not self.comm.send(cmd):
            # EAGAIN 으로 이번 틱을 건너뛴 것일 수 있다(치명적이지 않다).
            # Comm 이 자체적으로 재시도/포기/재접속을 판단하므로 여기서는
            # 카운터만 주기적으로 남긴다.
            self.get_logger().warn(f'송신 보류 — {self.comm.send_stats()}',
                                   throttle_duration_sec=1.0)

    # ── 소켓 → /gt_state ──────────────────────────────────────────────────
    def _recv_loop(self) -> None:
        """
        유일한 블로킹 루프. 프레임이 오면 즉시 퍼블리시한다.

        소켓이 논블로킹이라 그냥 돌면 busy-spin 이 된다. select 로 데이터가
        올 때까지 기다려 CPU 를 태우지 않는다.
        """
        self.comm.connect()
        while self._running and rclpy.ok():
            sock = self.comm.sock
            if sock is None:
                self.get_logger().warn('연결 없음 — 재접속')
                self.comm.reconnect()
                continue

            try:
                select.select([sock], [], [], self._select_timeout)
            except OSError:
                continue

            pkt = self.comm.recv()
            if pkt is not None:
                self.pub.publish(packet_to_msg(pkt, self.get_clock().now().to_msg()))
                continue

            if self.comm.stale():
                # SPEC §3.2: 패킷 수신 여부로만 판단한다. 정지 중이어도 재시작 금지.
                with self._lock:
                    self._last_cmd = SAFE_STOP
                self.comm.send(SAFE_STOP)
                if not self.comm.connected:
                    self.get_logger().warn('패킷 끊김 — 재접속')
                    self.comm.reconnect()

    def destroy_node(self) -> None:
        self._running = False
        self.comm.close()
        super().destroy_node()


class _RosLog:
    """core.Comm 이 기대하는 info/warn 인터페이스를 rclpy 로거로 연결."""

    def __init__(self, logger) -> None:
        self._l = logger

    def info(self, msg: str) -> None:
        self._l.info(msg)

    def warn(self, msg: str) -> None:
        self._l.warn(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VtdBridgeNode()
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
