"""
틱 루프 — **ROS 없이 도는 백업 실행 경로**.

기본 실행은 ROS 2 launch (`ros2 launch hlfma drive.launch.py`) 다. 이 파일은
ROS 가 없거나 노드 계층을 배제하고 core 로직만 확인하고 싶을 때 쓴다.
같은 core/ 를 그대로 호출하므로 거동은 동일하다.

한 패킷 = 한 틱, 단일 프로세스 / 단일 스레드 (SPEC §2).

    9910 → Comm.recv → Perception → Planner → Shield → Control → Comm.send → 9910
                             ↘________ Logger (매 틱 전부 기록) ________↙

실행:
    python3 tools/run_standalone.py --graph data/lane_graph.pkl --route data/route.pkl
    python3 tools/run_standalone.py --replay logs/sample.jsonl
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src' / 'hlfma'))

from hlfma.core.comm import SAFE_STOP, Comm                # noqa: E402
from hlfma.core.control import Control                     # noqa: E402
from hlfma.core.lanegraph import LaneGraph                 # noqa: E402
from hlfma.core.logger import Logger                       # noqa: E402
from hlfma.core.perception import Perception               # noqa: E402
from hlfma.core.planner import Planner                     # noqa: E402
from hlfma.core.shield import Shield                       # noqa: E402
from hlfma.core.types import Command                       # noqa: E402


def load_config(path: str) -> dict:
    """
    params.yaml 로드. ROS 노드와 **같은 파일**을 읽는다 (설정 이중화 방지).
    모든 튜닝 상수는 여기서만 온다 (SPEC §4).
    """
    from hlfma.nodes.params import load_params_yaml
    return load_params_yaml(path)


def load_route(path: str) -> dict:
    """build_route.py 산출물. keys: lanes, cum_s, lengths, total_length,
    start_s_in_lane, waypoints, waypoint_s, events"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='HL FMA 2026 컨트롤러')
    ap.add_argument('--host', default=None, help='VTD 호스트 (기본: config)')
    ap.add_argument('--port', type=int, default=None, help='9910')
    ap.add_argument('--graph', default='data/lane_graph.pkl')
    ap.add_argument('--route', default='data/route.pkl')
    ap.add_argument('--config', default=str(_ROOT / 'src' / 'hlfma' / 'config' / 'params.yaml'))
    ap.add_argument('--replay', default=None,
                    help='jsonl 로그를 Comm 대신 소스로 사용 (VTD 불필요)')
    ap.add_argument('--log', default=None, help='틱 로그를 쓸 jsonl 경로')
    ap.add_argument('--max-ticks', type=int, default=0, help='0 = 무제한 (디버깅용)')
    return ap


class Runner:
    """파이프라인을 조립하고 틱을 돌린다."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.cfg = load_config(args.config)
        self.args = args

        self.lg = LaneGraph(args.graph)
        self.route = load_route(args.route) if pathlib.Path(args.route).exists() else None
        if self.route is None:
            print(f'[main] 경고: route 파일 없음 ({args.route}). '
                  f'tools/build_route.py 로 만들어야 한다.', flush=True)

        self.perception = Perception(self.lg, self.route, self.cfg)
        self.planner = Planner(self.lg, self.route, self.cfg)
        self.shield = Shield(self.lg, self.cfg)
        self.control = Control(self.cfg)
        self.logger = Logger(args.log, self.cfg)

        c = self.cfg['comm']
        self.send_dt = 1.0 / float(c['send_hz'])

        self.source = None      # Comm 또는 replay 소스
        if args.replay:
            from replay import ReplaySource
            self.source = ReplaySource(args.replay)
        else:
            self.source = Comm(
                host=args.host or c['host'],
                port=args.port or int(c['port']),
                watchdog_s=float(c['watchdog_s']),
                steer_sign=float(c['steer_sign']),
                connect_retry_s=float(c['connect_retry_s']),
                recv_bufsize=int(c['recv_bufsize']),
            )

        self.last_cmd = Command(0.0, 0.0, 0)
        self.error_streak = 0
        self.ticks = 0

        # 주행 중 눈으로 보려고 1초에 한 줄 (config debug.print_hz)
        self._print_dt = 1.0 / max(1e-6, float(self.cfg['debug']['print_hz']))
        self._next_print = 0.0
        self._t0 = time.monotonic()

    # ── 한 틱 ─────────────────────────────────────────────────────────────
    def tick(self, pkt) -> Command:
        """RawPacket → Command. 예외는 호출자가 잡는다."""
        world = self.perception.update(pkt)
        decision = self.planner.plan(world)
        decision = self.shield.apply(world, decision)
        cmd = self.control.compute(world, decision)
        self.logger.write(pkt, world, decision, cmd)
        self._maybe_print(world, decision, cmd)
        return cmd

    def _maybe_print(self, world, decision, cmd) -> None:
        """1초에 한 줄. 차가 어디서 뭘 하고 있는지 한눈에."""
        now = time.monotonic()
        if now < self._next_print:
            return
        self._next_print = now + self._print_dt

        e = world.ego
        lane = f'{e.lane[0]}/{e.lane[2]}' if e.lane else '----/--'
        tgt = self.control.last_target
        d_tgt = (((tgt[0] - e.x) ** 2 + (tgt[1] - e.y) ** 2) ** 0.5) if tgt else float('nan')
        print(
            f't={now - self._t0:6.1f}  road/lane={lane:>9}  s_route={e.route_s:7.1f}m  '
            f'v={e.speed:5.2f}→{decision.v_target:5.2f} m/s  '
            f'steer={cmd.steering:+.3f}  accel={cmd.accel:+.2f}  '
            f'tgt={d_tgt:5.1f}m  {decision.state}'
            + ('' if world.valid else '  [INVALID]'),
            flush=True,
        )

    # ── 루프 ──────────────────────────────────────────────────────────────
    def run(self) -> int:
        if hasattr(self.source, 'connect') and not self.args.replay:
            self.source.connect()

        next_send = time.monotonic()
        try:
            while True:
                if self.args.max_ticks and self.ticks >= self.args.max_ticks:
                    break

                pkt = self.source.recv()

                if pkt is not None:
                    try:
                        self.last_cmd = self.tick(pkt)
                        self.error_streak = 0
                    except Exception as e:                    # noqa: BLE001
                        # SPEC §4: 틱 예외로 루프가 죽지 않게. 직전 명령 유지.
                        self.error_streak += 1
                        self.logger.error(self.ticks, e, self.error_streak)
                        if self.error_streak >= 3:
                            self.last_cmd = SAFE_STOP
                elif self.args.replay:
                    break                                     # 로그 끝
                elif self.source.stale():
                    # 패킷 수신 여부로만 판단한다 (정지 중이어도 재시작 금지)
                    self.last_cmd = SAFE_STOP
                    if not self.source.connected:
                        self.source.reconnect()

                if not self.args.replay:
                    self.source.send(self.last_cmd)

                self.ticks += 1
                next_send += self.send_dt
                sleep = next_send - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_send = time.monotonic()              # 밀렸으면 리셋
        except KeyboardInterrupt:
            print('\n[main] 중단', flush=True)
        finally:
            if hasattr(self.source, 'close'):
                self.source.close()
            self.logger.close()
            print(f'[main] 틱 {self.ticks}회 종료', flush=True)
        return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return Runner(args).run()


if __name__ == '__main__':
    raise SystemExit(main())
