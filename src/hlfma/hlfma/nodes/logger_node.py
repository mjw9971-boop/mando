"""
매 틱 jsonl 기록 (ROS 실행 경로).

core.Logger 는 원래 tools/run_standalone.py 에만 연결돼 있어서 `ros2 launch` 로
주행하면 로그가 한 줄도 안 남았다. 주행 후 원인 분석이 불가능해지므로 노드로 뺐다.

/gt_state /world_state /decision /cmd 를 모두 구독하고 **header.stamp 를 틱 키**로
묶는다. 네 메시지가 같은 stamp 를 물고 흐르기 때문에(perception→planner→control 이
받은 stamp 를 그대로 전달) 한 틱을 정확히 재구성할 수 있다.
체인의 마지막인 /cmd 가 오면 그 틱을 한 줄로 쓴다.

QoS: 다른 노드는 depth=1(최신 우선)이지만 로거는 기록이 목적이라 큐를 깊게 잡는다.
"""
from __future__ import annotations

import pathlib
import queue
import threading
import time
from collections import OrderedDict

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from hlfma_msgs.msg import Cmd, Decision, GtState, WorldState

from hlfma.convert import msg_to_command, msg_to_decision, msg_to_packet, msg_to_world
from hlfma.core.logger import Logger
from hlfma.core.scoring import SpeedMonitor
from hlfma.core.timing import Stage, Timer
from hlfma.nodes.params import collect_core_config

LOG_QOS = QoSProfile(depth=50, history=HistoryPolicy.KEEP_LAST,
                     reliability=ReliabilityPolicy.RELIABLE)

MAX_PENDING = 64          # stamp 버퍼 상한 (체인이 끊겨도 메모리가 안 는다)


def resolve_log_path(cfg: dict) -> str:
    """log.path 가 비어 있으면 log.dir/run_<타임스탬프>.jsonl 로 만든다."""
    log = cfg.get('log', {})
    path = (log.get('path') or '').strip()
    if path:
        return path
    d = log.get('dir') or 'logs'
    return str(pathlib.Path(d) / f"run_{time.strftime('%Y%m%d_%H%M%S')}.jsonl")


def _key(stamp) -> tuple:
    return (int(stamp.sec), int(stamp.nanosec))


class LoggerNode(Node):
    def __init__(self) -> None:
        super().__init__('logger')
        cfg = collect_core_config(self)

        self.enabled = bool(cfg['log'].get('enabled', True))
        self.path = resolve_log_path(cfg)
        self.core = Logger(self.path if self.enabled else None, cfg)

        self._gt: OrderedDict = OrderedDict()
        self._ws: OrderedDict = OrderedDict()
        self._dc: OrderedDict = OrderedDict()
        self._written = 0
        self._orphan = 0
        self._dropped = 0

        # 주행 요약(제한속도 준수는 채점 항목이라 매 주행 찍는다)
        self.speed_mon = SpeedMonitor(
            margin_kph=float(cfg['speed']['margin_kph']),
            school_cap_kph=float(cfg['caps_kph']['school_zone']))

        self.st_cb = Stage('logger 콜백')
        self.st_write = Stage('  파일 쓰기(백그라운드)')

        # 파일 I/O 를 콜백에서 하면 그 시간만큼 실행기 스레드가 묶인다.
        # 20 Hz 제어 루프에서 flush 가 수십 ms 걸리면 그대로 지연이 된다.
        # 큐에 넣고 별도 스레드에서 쓴다.
        self._q: queue.Queue = queue.Queue(maxsize=2000)
        self._stop = threading.Event()
        self._writer = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer.start()

        self.create_subscription(GtState, '/gt_state', self._on_gt, LOG_QOS)
        self.create_subscription(WorldState, '/world_state', self._on_ws, LOG_QOS)
        self.create_subscription(Decision, '/decision', self._on_dc, LOG_QOS)
        self.create_subscription(Cmd, '/cmd', self._on_cmd, LOG_QOS)

        if self.enabled:
            self.get_logger().info(f'틱 로그 기록: {self.path}')
        else:
            self.get_logger().warn('log.enabled=false — 주행 로그를 남기지 않는다')

    @staticmethod
    def _put(store: OrderedDict, stamp, value) -> None:
        store[_key(stamp)] = value
        while len(store) > MAX_PENDING:
            store.popitem(last=False)

    def _on_gt(self, msg: GtState) -> None:
        self._put(self._gt, msg.header.stamp, msg)

    def _on_ws(self, msg: WorldState) -> None:
        self._put(self._ws, msg.header.stamp, msg)

    def _on_dc(self, msg: Decision) -> None:
        self._put(self._dc, msg.header.stamp, msg)

    def _on_cmd(self, msg: Cmd) -> None:
        """체인의 마지막. 같은 stamp 의 나머지를 모아 한 줄 쓴다."""
        if not self.enabled:
            return
        k = _key(msg.header.stamp)
        gt, ws, dc = self._gt.pop(k, None), self._ws.pop(k, None), self._dc.pop(k, None)
        if gt is None or ws is None or dc is None:
            # 앞단 메시지를 못 받았다 (QoS 유실 등). 이 틱은 건너뛴다.
            self._orphan += 1
            if self._orphan % 50 == 1:
                self.get_logger().warn(
                    f'틱 조각 누락으로 {self._orphan}건 미기록 '
                    f'(gt={gt is not None} ws={ws is not None} dec={dc is not None})',
                    throttle_duration_sec=5.0)
            return
        with Timer(self.st_cb):
            try:
                item = (msg_to_packet(gt), msg_to_world(ws),
                        msg_to_decision(dc), msg_to_command(msg))
            except Exception as e:                                # noqa: BLE001
                self.get_logger().error(f'로그 변환 실패: {e}', throttle_duration_sec=5.0)
                return
            w = item[1]
            self.speed_mon.feed(w.ego.speed, w.speed_limit, w.school_zone,
                                w.ego.route_s, w.t)
            try:
                self._q.put_nowait(item)
            except queue.Full:
                # 기록보다 주행이 우선이다. 밀리면 버린다.
                self._dropped += 1
                if self._dropped % 100 == 1:
                    self.get_logger().warn(
                        f'로그 큐 포화 — {self._dropped}건 버림', throttle_duration_sec=5.0)

    def _writer_loop(self) -> None:
        """파일 쓰기 전용 스레드. 콜백을 막지 않는다."""
        while not self._stop.is_set() or not self._q.empty():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            with Timer(self.st_write):
                try:
                    self.core.write(*item)
                    self._written += 1
                except Exception:                                 # noqa: BLE001
                    pass

    def destroy_node(self) -> None:
        self._stop.set()
        self._writer.join(timeout=3.0)
        for line in self.speed_mon.lines():
            self.get_logger().info(line)
        if self.enabled:
            self.get_logger().info(
                f'틱 로그 {self._written}줄 기록 '
                f'(조각누락 {self._orphan}, 큐포화 폐기 {self._dropped}) → {self.path}')
            self.get_logger().info(self.st_cb.summary())
            self.get_logger().info(self.st_write.summary())
        self.core.close()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LoggerNode()
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
