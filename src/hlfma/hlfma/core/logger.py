"""
매 틱 jsonl 기록 + 리플레이 소스  (SPEC §3.7)

한 줄에 t / raw(ego·objects·lights) / WorldState 요약 / Decision(reasons 포함) /
Command 를 전부 남긴다. `tools/replay.py` 가 이걸 읽어 VTD 없이 재실행한다.
판단 로직을 고친 뒤 회귀 테스트하는 기반이므로 **필드를 함부로 빼지 말 것.**
"""
from __future__ import annotations

import json
import math
import pathlib
import queue
import threading
import time

from .types import Command, Decision, RawPacket, WorldState


def _num(v):
    """numpy 스칼라/NaN/inf 를 json 이 먹을 수 있는 형태로."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    if math.isinf(f):
        return 1e30 if f > 0 else -1e30
    return f


class Logger:
    """
    jsonl 틱 로그.

    파일 쓰기는 **전용 스레드**에서 한다. write() 는 직렬화까지만 하고 큐에
    넣는다 — 틱 콜백에서 동기 파일 I/O(플러시 수십 ms)를 하면 20 Hz 루프가
    그대로 밀린다. (ROS 경로의 logger_node 도 자체 큐를 한 겹 더 두는데,
    이중 버퍼일 뿐 해롭지 않다. standalone 경로는 이 큐가 유일한 방벽이다.)
    큐가 차면 버린다 — 기록보다 주행이 우선이다.
    """

    def __init__(self, path: str | None, cfg: dict) -> None:
        self.path = path
        self.cfg = cfg
        self.flush_every = int(cfg.get('log', {}).get('flush_every', 20))
        self._f = None
        self._n = 0
        self.dropped = 0
        # sim_dt 환산용 프레임 주기 (9910 은 프레임당 sim 1/send_hz 진행)
        self._frame_s = 1.0 / float(cfg.get('comm', {}).get('send_hz', 20))
        self._prev_wall: float | None = None
        self._prev_frames: int | None = None
        self._q: queue.Queue | None = None
        self._stop = threading.Event()
        self._writer: threading.Thread | None = None
        if path:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._f = open(path, 'w', encoding='utf-8')
            self._q = queue.Queue(maxsize=2000)
            self._writer = threading.Thread(target=self._writer_loop, daemon=True)
            self._writer.start()

    def _writer_loop(self) -> None:
        while not self._stop.is_set() or not self._q.empty():
            try:
                line = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._f.write(line)
                self._n += 1
                if self._n % self.flush_every == 0:
                    self._f.flush()
            except OSError:
                pass

    def _put(self, line: str) -> None:
        if self._q is None:
            return
        try:
            self._q.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    # ── 틱 기록 ───────────────────────────────────────────────────────────
    def write(self, pkt: RawPacket, world: WorldState, decision: Decision,
              cmd: Command, timing: dict | None = None) -> None:
        """한 틱을 jsonl 한 줄로. replay 가 재구성할 수 있을 만큼 전부 담는다.

        timing: 호출자(노드)가 잰 도착 시각 등 부가 계측. 아래의 wall/sim 차분과
        합쳐 'timing' 필드로 남는다.
        """
        if self._f is None:
            return
        e = world.ego
        rec = {
            't': _num(pkt.t_recv),
            # raw — replay 가 RawPacket 을 그대로 복원할 수 있어야 한다
            'raw': {
                'ego': [_num(v) for v in pkt.ego],
                'objects': [[_num(v) for v in o] for o in pkt.objects],
                'lights': [[int(i), int(s)] for i, s in pkt.lights],
            },
            'ego': {
                'x': _num(e.x), 'y': _num(e.y), 'yaw': _num(e.yaw),
                'speed': _num(e.speed), 'accel': _num(e.accel),
                'lane': list(e.lane) if e.lane else None,
                's': _num(e.s), 'route_s': _num(e.route_s),
                't_off': _num(e.t_off), 'heading_err': _num(e.heading_err),
            },
            'world': {
                'valid': bool(world.valid),
                'speed_limit': _num(world.speed_limit),
                'school_zone': bool(world.school_zone),
                'left_solid': bool(world.left_solid),
                'right_solid': bool(world.right_solid),
                'left_is_center': bool(world.left_is_center),
                'light': list(world.light) if world.light else None,
                'n_obj': len(world.objects),
                # 리스트/불리언이 문자열로 뭉개지지 않게 _jsonable 을 쓴다
                'flags': _jsonable(world.flags),
                # ahead 원본은 너무 크다 → kind/dist 만
                'ahead': [{'kind': a.kind, 'dist': _num(a.dist)} for a in world.ahead[:20]],
                'summ': _jsonable({k: v for k, v in world.summ.items()
                                   if k != 'speed_changes'}),
            },
            'objects': [
                {'id': int(o.id), 'cls': o.cls,
                 'x': _num(o.x), 'y': _num(o.y), 'speed': _num(o.speed),
                 'ttc': _num(o.ttc)}
                for o in world.objects
            ],
            'decision': {
                'state': decision.state,
                'v_target': _num(decision.v_target),
                'turn_signal': int(decision.turn_signal),
                'n_path': len(decision.path),
                'reasons': _jsonable(decision.reasons),
            },
            'cmd': {
                'steering': _num(cmd.steering),
                'accel': _num(cmd.accel),
                'turn_signal': int(cmd.turn_signal),
            },
        }
        # ── 시간 계측: RTF(sim/wall) 판별용 ──────────────────────────────
        # t(=t_recv) 는 소켓 수신 시각 time.monotonic() = **벽시계**다.
        # 9910 프레임에는 sim 타임스탬프 필드가 없어(§1.1) sim 경과시간은
        # 프레임 카운터 차분 x 프레임주기(50 ms)로 계산한다.
        tm = dict(timing or {})
        wall = float(pkt.t_recv)
        frames = int(getattr(pkt, 'frames_total', 0) or 0)
        if self._prev_wall is not None:
            tm['wall_dt'] = round(wall - self._prev_wall, 4)
        if frames > 0:
            tm['frames_total'] = frames
            if self._prev_frames:
                tm['sim_dt'] = round((frames - self._prev_frames) * self._frame_s, 4)
        self._prev_wall = wall
        self._prev_frames = frames if frames > 0 else self._prev_frames
        rec['timing'] = tm
        self._put(json.dumps(rec, ensure_ascii=False) + '\n')

    # ── 부가 로그 ─────────────────────────────────────────────────────────
    def error(self, tick: int, exc: BaseException, streak: int) -> None:
        """틱 내부 예외 기록 (SPEC §4: 루프는 죽지 않는다)."""
        msg = f'{type(exc).__name__}: {exc}'
        print(f'[tick {tick}] 예외({streak}연속) {msg}', flush=True)
        self._event('error', tick=tick, exc=msg, streak=streak)

    def info(self, msg: str) -> None:
        print(f'[info] {msg}', flush=True)
        self._event('info', msg=msg)

    def warn(self, msg: str) -> None:
        print(f'[warn] {msg}', flush=True)
        self._event('warn', msg=msg)

    def _event(self, kind: str, **kw) -> None:
        if self._f is None:
            return
        self._put(json.dumps({'t': time.monotonic(), 'event': kind, **kw},
                             ensure_ascii=False) + '\n')

    def close(self) -> None:
        if self._writer is not None:
            self._stop.set()
            self._writer.join(timeout=3.0)
            self._writer = None
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None
        if self.dropped:
            print(f'[logger] 큐 포화로 {self.dropped}줄 버림', flush=True)


def _jsonable(d):
    # bool 은 int 의 하위형이라 먼저 걸러야 한다 (True 가 1.0 으로 새면 분석이 꼬인다).
    # int 는 그대로 둔다 — 신호 id 같은 값이 431.0 으로 남으면 대조가 번거롭다.
    if isinstance(d, bool) or isinstance(d, str) or d is None:
        return d
    if isinstance(d, dict):
        return {str(k): _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    if isinstance(d, int):
        return d
    if isinstance(d, float):
        return _num(d)
    return str(d)
