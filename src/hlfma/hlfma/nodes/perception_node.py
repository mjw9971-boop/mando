"""
/gt_state → core.Perception → /world_state. 구독 콜백에서 계산 후 즉시 퍼블리시.

주행 중 눈으로 보라고 신호등 상태를 콘솔에 찍는다.
  - debug.print_hz 주기(기본 1초)로 한 줄
  - light id / state 가 **바뀌는 순간에는 즉시** 한 줄

SPEC §7-1 이 "신호등 id 가 교차로마다 바뀌는지, state 전이가 실제로 오는지"를
미확인으로 남겨뒀다. 변화 감지 로그가 그 관측 기록이 된다.
출력은 타이머가 아니라 구독 콜백 안에서 경과시간을 보고 낸다(타이머 금지).
"""
from __future__ import annotations

import pickle
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from hlfma_msgs.msg import GtState, WorldState

from hlfma.convert import msg_to_packet, world_to_msg
from hlfma.core.lanegraph import LaneGraph
from hlfma.core.perception import Perception
from hlfma.core.timing import Stage, Timer
from hlfma.nodes.params import collect_core_config
from hlfma.nodes.qos import LATEST

# SPEC §1.1 trafficLights[].state
STATE_NAME = {0: '미할당', 1: '적색', 2: '황색', 3: '녹색',
              4: '좌회전', 5: '녹색+좌회전', 6: '점멸'}


def _state_str(state: int) -> str:
    return f'{STATE_NAME.get(state, "?")}({state})'


def _dist_str(d) -> str:
    return '  --  ' if d is None else f'{float(d):6.1f}m'


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__('perception')

        self.declare_parameter('graph_path', 'data/lane_graph.pkl')
        self.declare_parameter('route_path', 'data/route.pkl')
        cfg = collect_core_config(self)

        graph = self.get_parameter('graph_path').value
        route_path = self.get_parameter('route_path').value

        self.get_logger().info(f'lane_graph 로드: {graph}')
        lg = LaneGraph(graph)

        route = None
        try:
            with open(route_path, 'rb') as f:
                route = pickle.load(f)
            self.get_logger().info(
                f"route 로드: {route_path} ({len(route['lanes'])} lanes, "
                f"{route['total_length']:.0f} m)")
        except (OSError, pickle.UnpicklingError) as e:
            self.get_logger().warn(f'route 로드 실패 ({e}) — 경로 없이 진행')

        self.core = Perception(lg, route, cfg)
        self.pub = self.create_publisher(WorldState, '/world_state', LATEST)
        self.create_subscription(GtState, '/gt_state', self._on_gt, LATEST)

        # 콘솔 출력 상태
        hz = float(cfg['debug']['print_hz'])
        self._print_dt = (1.0 / hz) if hz > 0 else None   # None 이면 주기 출력 끔
        self._next_print = 0.0
        self._last_light: tuple[int, int] | None = None
        self._t0 = time.monotonic()

        # 단계별 소요시간
        self.st_all = Stage('perception 콜백')
        self.st_core = Stage('  core.update')
        self.st_pub = Stage('  world→msg+publish')
        self.st_gap = Stage('/gt_state 수신 간격')
        self._prev_rx = None
        self._t_first = None

    def _on_gt(self, msg: GtState) -> None:
        at = msg.t_recv - self._t_first if self._t_first is not None else 0.0
        if self._t_first is None:
            self._t_first = msg.t_recv
        # 패킷 간격 — 스톨이 우리 콜백 안인지 밖인지 가르는 기준
        if self._prev_rx is not None:
            self.st_gap.record(msg.t_recv - self._prev_rx, at)
        self._prev_rx = msg.t_recv

        cb_t0 = time.monotonic()
        with Timer(self.st_all, at):
            with Timer(self.st_core, at):
                ws = self.core.update(msg_to_packet(msg))
            # 콜백 시작/끝 벽시계(monotonic) — 스톨이 어느 구간(수신/계산/전달)에서
            # 나는지 틱 로그에서 재구성하기 위한 계측 (flags 는 그대로 로그에 남는다)
            ws.flags['cb_perc'] = [cb_t0, time.monotonic()]
            with Timer(self.st_pub, at):
                self.pub.publish(world_to_msg(ws, msg.header.stamp))
        self._report_reset(ws)
        self._report_light(ws)

    # ── 리셋 보고 ─────────────────────────────────────────────────────────
    def _report_reset(self, ws) -> None:
        """
        VTD 가 차를 도로로 되돌린 순간을 남긴다.

        리셋 횟수 = 도로/경로를 이탈한 횟수이므로 그대로 주행 품질 지표다.
        대회 채점에서도 이탈은 감점이니 0 이 목표다.
        """
        f = ws.flags
        if f.get('stall'):
            # 우리 쪽 파이프라인이 멈춘 것. 리스폰과 섞어 세면 품질 지표가 망가진다.
            self.get_logger().warn(
                f"[스톨] #{f.get('stall_count')} — 틱 간격 {f['stall_dt_s']:.2f}s "
                f"(그 사이 {f['stall_jump_m']:.1f} m 이동). 제어 명령이 그동안 갱신되지 않았다",
                throttle_duration_sec=1.0)
        if not f.get('reset'):
            return
        e = ws.ego
        why = []
        if 'reset_jump_m' in f:
            why.append(f"위치 점프 {f['reset_jump_m']} m")
        if 'reset_route_s_drop' in f:
            why.append(f"route_s {f['reset_route_s_drop']} m 되돌아감")
        if 'reset_implied_mps' in f:
            why.append(f"환산 {f['reset_implied_mps']} m/s (dt {f.get('reset_dt_s')}s)")
        self.get_logger().warn(
            f"[리셋] #{f.get('reset_count')} — {' / '.join(why) or '원인 미상'}  "
            f"pos=({e.x:.2f},{e.y:.2f}) route_s={e.route_s:.1f}m  "
            f"속도추정·적분항 초기화")

    # ── 콘솔 출력 ─────────────────────────────────────────────────────────
    def _report_light(self, ws) -> None:
        """신호등 상태를 주기적으로, 그리고 바뀌는 순간에 찍는다."""
        light = ws.light                      # (id, state) 또는 None
        cur = (int(light[0]), int(light[1])) if light else (0, 0)

        summ = ws.summ or {}
        dist = summ.get('dist_stop_line')
        ids = summ.get('stop_signal_ids') or []
        # 9910 light_id 는 controller id, signal_ids 는 xodr signal id 라 계층이 다르다.
        # 대조 결과는 검증용으로만 표시한다 (주행 판단에는 안 쓴다).
        match = ws.flags.get('light_ctrl_match')
        cids = ws.flags.get('stop_ctrl_ids')
        mark = '' if match is None else ('  ctrl=OK' if match else
                                         f"  ctrl=불일치({ws.flags.get('light_ctrl_mismatch')})")

        now = time.monotonic()
        changed = self._last_light is None or cur != self._last_light

        if changed:
            prev = self._last_light
            self._last_light = cur
            if prev is None:
                self.get_logger().info(
                    f'[신호] 최초 수신  id={cur[0]} state={_state_str(cur[1])}  '
                    f'stop_line={_dist_str(dist)}  signals={ids}  ctrl={cids}{mark}')
            else:
                # id 와 state 중 실제로 바뀐 쪽을 드러낸다 (SPEC §7-1 관측 기록)
                parts = []
                if prev[0] != cur[0]:
                    parts.append(f'id {prev[0]}→{cur[0]}')
                if prev[1] != cur[1]:
                    parts.append(f'state {_state_str(prev[1])}→{_state_str(cur[1])}')
                self.get_logger().info(
                    f'[신호] 변화  {"  ".join(parts)}  '
                    f'stop_line={_dist_str(dist)}  signals={ids}  ctrl={cids}{mark}')
            if match is False:
                self.get_logger().warn(
                    f'[신호] light_id {cur[0]} 가 전방 정지선 controller {cids} 에 없다 '
                    f'— 주행은 계속(가장 가까운 전방 정지선 + 현재 state 사용)',
                    throttle_duration_sec=5.0)
            self._next_print = now + (self._print_dt or 0.0)
            return

        if self._print_dt is None or now < self._next_print:
            return
        self._next_print = now + self._print_dt

        e = ws.ego
        lane = f'{e.lane[0]}/{e.lane[2]}' if e.lane else '----/--'
        self.get_logger().info(
            f'[신호] t={now - self._t0:6.1f}s  road/lane={lane:>9}  '
            f'route_s={e.route_s:7.1f}m  |  '
            f'light id={cur[0]:<4d} state={_state_str(cur[1]):<12s} '
            f'stop_line={_dist_str(dist)}  signals={ids}{mark}'
            + ('' if ws.valid else '  [INVALID]'))


    def destroy_node(self) -> None:
        for st in (self.st_gap, self.st_all, self.st_core, self.st_pub):
            self.get_logger().info(st.summary())
            h = st.histogram()
            if h:
                self.get_logger().info(h)
        n = getattr(self.core, 'reset_count', 0)
        st = getattr(self.core, 'stall_count', 0)
        if n:
            self.get_logger().warn(
                f'VTD 리셋 {n}회 — 그만큼 도로/경로를 이탈했다는 뜻이다')
        else:
            self.get_logger().info('VTD 리셋 0회')
        if st:
            logs = getattr(self.core, 'stall_log', [])
            worst = max((x[1] for x in logs), default=0.0)
            self.get_logger().warn(
                f'파이프라인 스톨 {st}회 (최대 {worst:.2f}s) — 그동안 제어가 갱신되지 않았다. '
                f'리셋과는 별개 문제다')
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
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
