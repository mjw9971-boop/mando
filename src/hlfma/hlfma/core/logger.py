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
    def __init__(self, path: str | None, cfg: dict) -> None:
        self.path = path
        self.cfg = cfg
        self.flush_every = int(cfg.get('log', {}).get('flush_every', 20))
        self._f = None
        self._n = 0
        if path:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._f = open(path, 'w', encoding='utf-8')

    # ── 틱 기록 ───────────────────────────────────────────────────────────
    def write(self, pkt: RawPacket, world: WorldState, decision: Decision,
              cmd: Command) -> None:
        """한 틱을 jsonl 한 줄로. replay 가 재구성할 수 있을 만큼 전부 담는다."""
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
        self._f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        self._n += 1
        if self._n % self.flush_every == 0:
            self._f.flush()

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
        self._f.write(json.dumps({'t': time.monotonic(), 'event': kind, **kw},
                                 ensure_ascii=False) + '\n')

    def close(self) -> None:
        if self._f is not None:
            self._f.flush()
            self._f.close()
            self._f = None


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
