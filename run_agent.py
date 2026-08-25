"""
실행 진입점 — 파이썬 단일 프로세스 틱 루프 (ROS 없음).

한 패킷 = 한 틱, 단일 프로세스 / 단일 스레드:

    9910 → Comm.recv → EgoTracker/VtdWorld → PDM-Lite autopilot.run_step
         → command_from_control → Comm.send → 9910
                    ↘________ Logger (매 틱 전부 기록) ________↙

판단은 team_code/autopilot.py (PDM-Lite, Beißwenger 2024) 원문이 한다.
어댑터(vtd_adapter)가 CARLA 표면을 흉내내고, 여기서는 조립·로깅만 한다.

실행:
    python3 run_agent.py --route data/route.pkl
    python3 run_agent.py --csv waypoints.csv            # route 를 빌드해 쓰고 시작
    python3 run_agent.py --replay logs/run_x.jsonl      # VTD 없이 로그 재생
"""
from __future__ import annotations

import argparse
import math
import pathlib
import pickle
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'team_code'))    # PDM-Lite 는 평면 import 를 쓴다

from vtd_adapter import frame                              # noqa: E402
from vtd_adapter.comm import SAFE_STOP, Comm               # noqa: E402
from vtd_adapter.config import load_params_yaml            # noqa: E402
from vtd_adapter.control import (VtdLongitudinalController,  # noqa: E402
                                 command_from_control)
from vtd_adapter.ego import EgoTracker                     # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                # noqa: E402
from vtd_adapter.logger import Logger                      # noqa: E402
from vtd_adapter.map import VtdMap                         # noqa: E402
from vtd_adapter.route import VtdRoutePlanner              # noqa: E402
from vtd_adapter.types import Command, Decision, TrackedObject  # noqa: E402
from vtd_adapter.world import VtdWorld                     # noqa: E402

from autopilot import AutoPilot                            # noqa: E402  (team_code)
from config import GlobalConfig                            # noqa: E402  (team_code)
from kr_rules import KrRules                               # noqa: E402  (team_code)


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


class LoggingAutoPilot(AutoPilot):
    """autopilot 원문은 무수정(diff 0) 유지하고, 원인별 목표속도를 로그로 뽑기
    위해 개별 원인 함수의 **반환값만** 가로챈다 — 판단에는 관여하지 않는다.

    candidates: leading / vehicle / bicycle / pedestrian / red_light [m/s]
    final: (brake, target_speed, speed_reduced_by_obj), initial: 중재 전 목표.
    """

    def get_brake_and_target_speed(self, plant, route_points, dist_tl, next_tl,
                                   dist_ss, next_ss, vehicle_list, actor_list,
                                   initial_target_speed, speed_reduced_by_obj):
        self.candidates = {}
        self.initial_target = float(initial_target_speed)
        out = super().get_brake_and_target_speed(
            plant, route_points, dist_tl, next_tl, dist_ss, next_ss,
            vehicle_list, actor_list, initial_target_speed, speed_reduced_by_obj)
        self.final = out
        return out

    def compute_target_speed_wrt_leading_vehicle(self, *a, **kw):
        ts, obj = super().compute_target_speed_wrt_leading_vehicle(*a, **kw)
        self.candidates['leading'] = float(ts)
        return ts, obj

    def compute_target_speeds_wrt_all_actors(self, *a, **kw):
        bike, ped, veh, obj = super().compute_target_speeds_wrt_all_actors(*a, **kw)
        self.candidates.update(bicycle=float(bike), pedestrian=float(ped),
                               vehicle=float(veh))
        return bike, ped, veh, obj

    def ego_agent_affected_by_red_light(self, *a, **kw):
        ts = super().ego_agent_affected_by_red_light(*a, **kw)
        self.candidates['red_light'] = float(ts)
        return ts


# decision.state 에 쓰는 hazard 명 (이긴 원인). 지시등 판단(kr_rules)은 phase4.
_HAZARD_NAME = {'pedestrian': 'walker', 'red_light': 'light', 'leading': 'lead',
                'vehicle': 'vehicle', 'bicycle': 'bicycle', 'route_end': 'route_end'}


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

        # ── 어댑터 + PDM-Lite 조립 ────────────────────────────────────────
        self.pdm_config = GlobalConfig()
        self.tracker = EgoTracker(self.lg, self.route, self.cfg)
        self.world = VtdWorld(self.cfg)
        self.vmap = VtdMap(self.lg)
        self.planner = VtdRoutePlanner(self.lg, self.route, self.cfg,
                                       config=self.pdm_config)
        self.longc = VtdLongitudinalController(self.cfg)
        self.max_steer = float(self.cfg['vehicle']['max_steer'])

        self.agent = LoggingAutoPilot()
        self.agent.setup(self.world, self.vmap, self.planner, self.longc,
                         self.world.ego, config=self.pdm_config)
        # 한국 대회 규칙 계층 — autopilot._get_control 끝의 한 줄이 호출한다
        self.kr = KrRules(self.cfg)
        self.agent.kr_rules = self.kr

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
        world_state = self.tracker.update(pkt)

        # courseRespawn: 순간이동 전의 상태(트랙·경로 인덱스·제어 이력)는 전부 무효
        if world_state.flags.get('reset'):
            self.world.clear()
            cx, cy = frame.to_carla_xy(world_state.ego.x, world_state.ego.y)
            self.planner.reset_index([cx, cy])
            self.agent._turn_controller.error_history = []
            self.longc._prev_accel = 0.0

        self.world.update(pkt, world_state.ego)
        self.planner.update_lights(pkt.lights)

        control = self.agent.run_step(world_state, pkt.t_recv)
        cmd = command_from_control(control, self.max_steer, turn_signal=0)

        decision = self._build_decision()
        world_state.objects = self._log_objects()
        self.logger.write(pkt, world_state, decision, cmd)
        self._maybe_print(world_state, decision, cmd)
        return cmd

    def _build_decision(self) -> Decision:
        """autopilot 부수 상태 → 로그 스키마의 decision.

        state = 이긴 hazard 명 (walker/light/lead/vehicle/bicycle/none),
        reasons = 원인별 목표속도 + winner + 최저 원인 객체. turn_signal 은 0
        고정 (phase4 kr_rules 몫).
        """
        a = self.agent
        cand = dict(getattr(a, 'candidates', {}))
        initial = getattr(a, 'initial_target', None)
        brake, target, reduced = getattr(a, 'final', (False, 0.0, None))

        # kr_rules 의 route_end 후보를 같은 중재 축에 합친다 (작업2)
        if self.kr.last_candidate is not None:
            cand['route_end'] = float(self.kr.last_candidate)
        if self.kr.last_target is not None:
            target = float(self.kr.last_target)

        winner = 'none'
        if cand:
            low = min(cand, key=cand.get)
            if initial is None or cand[low] < initial - 1e-6:
                winner = _HAZARD_NAME[low]

        reasons = {'initial': initial, 'winner': winner, **cand}
        if reduced is not None and reduced[1] is not None:
            reasons['speed_reduced_by'] = {
                'type': reduced[1], 'id': reduced[2],
                'dist': None if reduced[3] is None else round(float(reduced[3]), 1)}
        return Decision(v_target=float(target), path=[], turn_signal=0,
                        state=winner, reasons=reasons)

    def _log_objects(self) -> list:
        """VtdWorld 액터 → 로그 objects[] (스키마 유지 — 좌표는 VTD 원 프레임).

        s_rel/lat_off/v_rel/ttc/will_enter_lane 은 옛 perception 의 판단 산물이라
        이제 없다 → None (지시 3-2). PDM 의 충돌 판단은 OBB forecast 가 한다.
        """
        out = []
        for actor in self.world.get_actors():
            vx, vy = frame.from_carla_xy(actor.x, actor.y)
            out.append(TrackedObject(
                id=actor.id, x=vx, y=vy,
                heading=frame.from_carla_yaw_deg(actor.yaw_deg),
                speed=actor.speed, length=actor.length, width=actor.width,
                height=actor.height, cls=actor.cls, lane=None, on_route=False,
                s_rel=None, lat_off=None, v_rel=None, ttc=None,
                will_enter_lane=False, age=actor.age, coasting=actor.coasting))
        return out

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
                if self.args.replay:
                    continue                                  # 재생은 페이싱 없이 최대 속도로
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
