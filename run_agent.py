"""
실행 진입점 — 파이썬 단일 프로세스 틱 루프 (ROS 없음).

한 패킷 = 한 틱, 단일 프로세스 / 단일 스레드:

    9910 → Comm.recv → EgoTracker → [판단: phase3 에서 PDM-Lite] → Comm.send → 9910
                              ↘________ Logger (매 틱 전부 기록) ________↙

phase1 현재: 판단 자리는 **정속 주행 stub** (v_target = speed_limit, steering = 0)
이다 — comm→log 루프와 로그 스키마가 도는지 확인하는 골격. phase3 에서
team_code/autopilot.py (PDM-Lite) 로 교체된다.

실행:
    python3 run_agent.py --route data/route.pkl
    python3 run_agent.py --csv waypoints.csv            # route 를 빌드해 쓰고 시작
    python3 run_agent.py --replay logs/run_x.jsonl      # VTD 없이 로그 재생
"""
from __future__ import annotations

import argparse
import pathlib
import pickle
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from vtd_adapter.comm import SAFE_STOP, Comm               # noqa: E402
from vtd_adapter.config import load_params_yaml            # noqa: E402
from vtd_adapter.ego import EgoTracker                     # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                # noqa: E402
from vtd_adapter.logger import Logger                      # noqa: E402
from vtd_adapter.types import Command, Decision            # noqa: E402


def load_route(path: str) -> dict:
    """build_route.py 산출물. keys: lanes, cum_s, lengths, total_length,
    start_s_in_lane, waypoints, waypoint_s, events"""
    with open(path, 'rb') as f:
        return pickle.load(f)


def build_route_from_csv(csv_path: str, graph_path: str) -> str:
    """--csv: tools/build_route.py 를 돌려 data/route_<csv이름>.pkl 을 만들고 경로를 돌려준다."""
    stem = pathlib.Path(csv_path).stem
    out = _ROOT / 'data' / f'route_{stem}.pkl'
    cmd = [sys.executable, str(_ROOT / 'tools' / 'build_route.py'),
           graph_path, csv_path, '-o', str(out)]
    print(f'[main] route 빌드: {" ".join(cmd)}', flush=True)
    cp = subprocess.run(cmd, text=True)
    if cp.returncode not in (0, 1) or not out.exists():   # 1 = 경고 있음(진행 가능)
        raise SystemExit(f'build_route 실패 (rc={cp.returncode})')
    return str(out)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description='HL FMA 2026 에이전트 (PDM-Lite/VTD)')
    ap.add_argument('--host', default=None, help='VTD 호스트 (기본: config)')
    ap.add_argument('--port', type=int, default=None, help='9910')
    ap.add_argument('--graph', default='data/lane_graph.pkl')
    ap.add_argument('--route', default=None, help='build_route.py 산출 pkl')
    ap.add_argument('--csv', default=None,
                    help='대회 배포 waypoints.csv — 내부에서 route 를 빌드해 쓴다')
    ap.add_argument('--config', default=str(_ROOT / 'config' / 'params.yaml'))
    ap.add_argument('--replay', default=None,
                    help='jsonl 로그를 Comm 대신 소스로 사용 (VTD 불필요)')
    ap.add_argument('--log', default=None, help='틱 로그를 쓸 jsonl 경로')
    ap.add_argument('--max-ticks', type=int, default=0, help='0 = 무제한 (디버깅용)')
    return ap


class StubAgent:
    """phase1 임시 판단: 정속 주행 (v_target = speed_limit, steering = 0).

    phase3 에서 team_code/autopilot.py 로 교체된다. Decision/Command 는
    로그 스키마 확인용으로만 채운다.
    """

    def __init__(self, cfg: dict) -> None:
        s, c = cfg['speed'], cfg['control']
        self.kp = float(c['kp'])
        self.a_min = float(s['a_min'])
        self.a_max = float(s['a_max'])
        self.a_hold = float(s['a_hold'])

    def step(self, world) -> tuple[Decision, Command]:
        v_t = float(world.speed_limit)
        decision = Decision(v_target=v_t, path=[], turn_signal=0,
                            state='STUB', reasons={'limit': v_t})
        if v_t <= 1e-6 and world.ego.speed < 0.2:
            accel = self.a_hold
        else:
            accel = min(self.a_max, max(self.a_min, self.kp * (v_t - world.ego.speed)))
        return decision, Command(steering=0.0, accel=accel, turn_signal=0)


class Runner:
    """파이프라인을 조립하고 틱을 돌린다."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.cfg = load_params_yaml(args.config)
        self.args = args

        route_path = args.route
        if args.csv:
            route_path = build_route_from_csv(args.csv, args.graph)
        if not route_path:
            raise SystemExit('--route 또는 --csv 를 줘야 한다')

        self.lg = LaneGraph(args.graph)
        self.route = load_route(route_path) if pathlib.Path(route_path).exists() else None
        if self.route is None:
            raise SystemExit(f'route 파일 없음: {route_path}')

        self.tracker = EgoTracker(self.lg, self.route, self.cfg)
        self.agent = StubAgent(self.cfg)          # phase3: PDM-Lite autopilot 으로 교체
        log_path = args.log
        if log_path is None and bool(self.cfg['log'].get('enabled', True)):
            ts = time.strftime('%Y%m%d_%H%M%S')
            log_path = str(_ROOT / self.cfg['log'].get('dir', 'logs') / f'run_{ts}.jsonl')
        self.logger = Logger(log_path, self.cfg)

        c = self.cfg['comm']
        self.send_dt = 1.0 / float(c['send_hz'])

        self.source = None      # Comm 또는 replay 소스
        if args.replay:
            sys.path.insert(0, str(_ROOT / 'tools'))
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
        world = self.tracker.update(pkt)
        decision, cmd = self.agent.step(world)
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
        print(
            f't={now - self._t0:6.1f}  road/lane={lane:>9}  s_route={e.route_s:7.1f}m  '
            f'v={e.speed:5.2f}→{decision.v_target:5.2f} m/s  '
            f'steer={cmd.steering:+.3f}  accel={cmd.accel:+.2f}  {decision.state}'
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
