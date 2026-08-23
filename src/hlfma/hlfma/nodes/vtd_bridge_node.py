"""
9910 소켓 <-> ROS 경계.

**이 노드만 소켓 recv 루프를 돈다.** 패킷 하나 = /gt_state 하나 = 한 틱.
나머지 노드는 전부 구독 콜백에서 계산한다(타이머 없음).

  recv 루프 ── /gt_state ──> perception ─> planner ─> control ── /cmd ──> 여기서 송신
"""
from __future__ import annotations

import select
import threading
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from hlfma_msgs.msg import Cmd, GtState

from hlfma.convert import msg_to_command, packet_to_msg
from hlfma.core.comm import SAFE_STOP, Comm
from hlfma.core.types import Command
from hlfma.nodes.params import collect_core_config
from hlfma.nodes.qos import LATEST


class VtdBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__('vtd_bridge')

        # 다른 노드와 **같은** 파라미터 이름(comm.*)을 쓴다. 여기서만 host/port 로
        # 따로 선언하면 params.yaml 이나 launch 인자가 이 노드에 닿지 않는다.
        cfg_all = collect_core_config(self)
        cfg = cfg_all['comm']
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
        # ── 명령 홀드 방지 (조향만) ────────────────────────────────────────
        # 파이프라인이 멈추면 마지막 /cmd 가 VTD 에 그대로 남는다. 2026-08-21
        # 주행에서 조향 -0.480(풀락)이 0.32 s 유지된 채 3.5 m 를 가 도로이탈의
        # 직접 원인이 됐다. 새 /cmd 가 hold_decay_s 이상 안 오면 조향을
        # steer_rate_max 로 0 을 향해 감쇠시켜 재송신한다. **가속은 유지** —
        # 여기서 세우기 시작하면 적신호 대기(§3.2)와 구분이 안 된다. 완전 단절은
        # 기존대로 watchdog_s(1 s)의 SAFE_STOP 이 맡는다.
        self._hold_decay_s = float(cfg.get('hold_decay_s', 0.15))
        self._decay_rate = float(cfg_all['control']['steer_rate_max'])   # [rad/s]
        self._last_cmd_t: float | None = None    # 마지막 /cmd 수신 시각 (monotonic)
        self._last_decay_t: float | None = None
        self._decay_sends = 0
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
            self._last_cmd_t = time.monotonic()
            self._last_decay_t = None

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
            self._maybe_decay_steering()
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

    # ── 명령 홀드 방지 ────────────────────────────────────────────────────
    def _maybe_decay_steering(self) -> None:
        """recv 루프가 도는 한(프레임 수신 또는 select 타임아웃마다) 호출된다.

        /cmd 가 hold_decay_s 이상 끊겼으면 마지막 조향을 0 으로 서서히 줄여
        재송신한다. 조향만 건드린다 — 가속/지시등은 마지막 값 유지.
        """
        now = time.monotonic()
        with self._lock:
            if self._last_cmd_t is None or now - self._last_cmd_t < self._hold_decay_s:
                return
            cur = self._last_cmd
            if cur.steering == 0.0:
                return
            dt = now - (self._last_decay_t if self._last_decay_t is not None
                        else self._last_cmd_t + self._hold_decay_s)
            if dt <= 0.0:
                return
            step = self._decay_rate * dt
            s = cur.steering
            s = max(0.0, s - step) if s > 0 else min(0.0, s + step)
            cmd = Command(steering=s, accel=cur.accel, turn_signal=cur.turn_signal)
            self._last_cmd = cmd
            self._last_decay_t = now
            self._decay_sends += 1
        self.comm.send(cmd)
        self.get_logger().warn(
            f'/cmd 끊김 {now - self._last_cmd_t:.2f}s — 조향 감쇠 재송신 '
            f'{cur.steering:+.3f} → {s:+.3f} (#{self._decay_sends})',
            throttle_duration_sec=1.0)

    def destroy_node(self) -> None:
        self._running = False
        if self._decay_sends:
            self.get_logger().info(f'조향 감쇠 재송신 {self._decay_sends}회')
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
