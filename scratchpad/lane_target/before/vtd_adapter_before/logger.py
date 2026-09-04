"""
매 틱 jsonl 기록 + 리플레이 소스  (SPEC §3.7)

한 줄에 t / raw(ego·objects·lights) / WorldState 요약 / Decision(reasons 포함) /
Command 를 전부 남긴다. `tools/replay.py` 가 이걸 읽어 VTD 없이 재실행한다.
판단 로직을 고친 뒤 회귀 테스트하는 기반이므로 **필드를 함부로 빼지 말 것.**
"""
from __future__ import annotations

import gc
import json
import math
import pathlib
import queue
import threading
import time

try:
    import resource          # POSIX 전용 — 없는 환경에선 nivcsw/majflt 만 빠진다
except ImportError:          # pragma: no cover
    resource = None

from .types import Command, Decision, RawPacket, WorldState


# v_target 을 정하는 속도 후보 이름 (planner._speed_candidates). 이번 틱에 나오지
# 않은 후보는 **null 로 명시**한다 — "키가 없다" 와 "후보가 미구현이다" 를 로그에서
# 구분할 수 없어 선행차 무반응 원인 규명이 늦어졌다 (2026-08-23 16:31 런).
SPEED_CANDIDATES = ('limit', 'limit_ahead', 'school_zone', 'curvature', 'junction',
                    'stop_line', 'crosswalk_ped', 'route_end', 'visibility', 'lead')


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


class Probe:
    """틱 루프 프리즈 진단 계측 (params log.probe_*). 관측만 — 동작을 바꾸지 않는다.

    2026-08-30 실사고: 틱 로그에 14.95 s 공백이 났는데 그 사이 줄이 하나도 없어
    "우리 루프가 멈췄나 / VTD 가 안 보냈나"를 사후에 가릴 수 없었다. 여기서 재는
    값 중 **loop_iters** 가 그 둘을 가른다 (공백 동안 늘었으면 루프는 살아 있었다).

    비용 실측(2026-08-30): process_time 0.72 µs + getrusage 0.94 µs = 틱당 1.7 µs.
    틱 예산 50 ms 의 0.003% 라 매 틱 호출해도 루프에 영향이 없다. 그래도 주기를
    설정으로 열어 둔다 (probe_rusage_every).

    GC 는 gc.callbacks 로 재되, **콜백 안에서는 I/O 도 할당도 하지 않는다** —
    카운터만 올리고 다음 sample() 에서 꺼낸다 (수집 중 재진입 회피).
    """

    def __init__(self, cfg: dict) -> None:
        lg = cfg.get('log', {})
        self.enabled = bool(lg.get('probe_enabled', True))
        self.rusage_every = max(1, int(lg.get('probe_rusage_every', 1)))
        self.gc_min_s = float(lg.get('probe_gc_min_s', 0.05))
        self._n = 0
        self._prev_cpu: float | None = None
        self._prev_ru: tuple | None = None
        self._gc_t0: float | None = None
        self._gc_n = 0
        self._gc_s = 0.0
        self._gc_max = 0.0
        if self.enabled:
            gc.callbacks.append(self._on_gc)

    def _on_gc(self, phase, _info) -> None:
        # 콜백은 GC 수집 중에 불린다 — 카운터만 만진다 (할당·I/O 금지)
        now = time.monotonic()
        if phase == 'start':
            self._gc_t0 = now
        elif self._gc_t0 is not None:
            dur = now - self._gc_t0
            self._gc_t0 = None
            self._gc_n += 1
            self._gc_s += dur
            if dur > self._gc_max:
                self._gc_max = dur

    def sample(self, loop_iters: int) -> dict:
        """이번 틱의 계측 dict. 직전 sample 이후의 델타를 낸다."""
        if not self.enabled:
            return {}
        out: dict = {'loop_iters': int(loop_iters)}
        cpu = time.process_time()
        if self._prev_cpu is not None:
            out['cpu_dt'] = round(cpu - self._prev_cpu, 4)
        self._prev_cpu = cpu
        self._n += 1
        if resource is not None and self._n % self.rusage_every == 0:
            r = resource.getrusage(resource.RUSAGE_SELF)
            cur = (r.ru_nivcsw, r.ru_majflt)
            if self._prev_ru is not None:
                out['nivcsw'] = cur[0] - self._prev_ru[0]   # 비자발적 컨텍스트 스위치
                out['majflt'] = cur[1] - self._prev_ru[1]   # 메이저 페이지 폴트(스왑)
            self._prev_ru = cur
        if self._gc_n:
            out['gc_n'] = self._gc_n
            out['gc_s'] = round(self._gc_s, 4)
            out['gc_max_s'] = round(self._gc_max, 4)
        return out

    def take_long_gc(self) -> float | None:
        """임계를 넘는 GC 일시정지가 있었으면 그 최대값을 돌려주고 카운터를 비운다."""
        mx = self._gc_max
        self._gc_n = 0
        self._gc_s = 0.0
        self._gc_max = 0.0
        return mx if mx >= self.gc_min_s else None

    def close(self) -> None:
        if self.enabled and self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)


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

    def _front_gap(self, world: WorldState) -> float | None:
        """앞범퍼 위치 − 정지선 위치 [m]. 음수 = 못 미침(정상), 양수 = 침범."""
        d = (world.summ or {}).get('dist_stop_line')
        if d is None:
            return None
        vh = self.cfg.get('vehicle', {})
        front = float(vh.get('wheelbase', 2.944)) + float(vh.get('front_overhang_m', 0.855))
        return front - float(d)

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
                # ego:     [x, y, z, heading, pitch, roll]            (m, rad)
                # objects: [id, x, y, z, heading, speed, length, width, height]
                # lights:  [[id, state], ...]   state 0=미할당 1=적 2=황 3=녹 4=좌 5=녹+좌 6=점멸
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
                # 정지 위치 실측용: 앞범퍼 위치 − 정지선 위치.
                # **음수 = 앞범퍼가 정지선에 못 미침(정상), 양수 = 침범.**
                # ego 좌표가 뒷바퀴축이라 dist_stop_line 만 보면 전장만큼 착시가 난다.
                'stop_line_front_m': _num(self._front_gap(world)),
            },
            # 객체: perception 이 계산한 **경로 기준 상대량**까지 남긴다. 예전에는
            # x/y/speed/ttc 만 있어서 "선행차가 내 차선 몇 m 앞인지" 를 로그만으로
            # 알 수 없었다 (리플레이로 복원해야 했다).
            'objects': [
                {'id': int(o.id), 'cls': o.cls,
                 'x': _num(o.x), 'y': _num(o.y), 'speed': _num(o.speed),
                 'lane': list(o.lane) if o.lane else None,
                 'on_route': bool(o.on_route),
                 's_rel': _num(o.s_rel), 'lat_off': _num(o.lat_off),
                 'v_rel': _num(o.v_rel), 'ttc': _num(o.ttc),
                 'will_enter_lane': bool(o.will_enter_lane),
                 'age': _num(o.age), 'coasting': bool(o.coasting)}
                for o in world.objects
            ],
            'decision': {
                'state': decision.state,
                'v_target': _num(decision.v_target),
                'turn_signal': int(decision.turn_signal),
                'n_path': len(decision.path),
                # 이번 틱에 없던 후보는 null 로 남긴다 (위 SPEC_CANDIDATES 주석 참고)
                'reasons': {**{k: None for k in SPEED_CANDIDATES},
                            **_jsonable(decision.reasons)},
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
    def gap(self, kind: str, **kw) -> None:
        """무수신(수신 공백) 구간 기록.

        지금까지는 **패킷이 온 틱만** 로그에 남아, 수신이 끊긴 구간이 통째로
        비어 있었다 — 그래서 14.95 s 공백의 원인을 사후에 못 가렸다.
        kind: 'gap_open'(무수신 시작 감지) / 'gap'(지속 중) / 'gap_close'(복구).
        """
        self._event(kind, **kw)

    def gc_pause(self, seconds: float, **kw) -> None:
        """임계를 넘는 GC 일시정지. Probe 가 카운터로 모은 것을 여기서 줄로 남긴다."""
        self._event('gc_pause', seconds=round(float(seconds), 4), **kw)

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
