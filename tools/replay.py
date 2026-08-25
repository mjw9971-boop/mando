"""
로그 jsonl → Comm 대체 소스  (SPEC §3.7)

VTD 없이 로그를 다시 파이프라인에 흘린다:

    python3 run_agent.py --replay logs/run_xxx.jsonl    # 파이프라인 재생 (개루프)
    python3 tools/replay.py logs/run_xxx.jsonl          # 지시등 시퀀스 분석

**개루프**다: 자차 궤적은 원본 런 그대로이고 새 판단이 궤적을 바꾸지는
않는다. "같은 입력에 대한 판단" 의 회귀 비교에 쓴다.

(자체 planner 재생·비교 기능은 PDM-Lite 이식으로 제거 — git 이력 phase1 이전 참조.
 phase3 이후 필요해지면 autopilot 재생으로 다시 만든다.)
"""
from __future__ import annotations

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from vtd_adapter.types import Command, RawPacket  # noqa: E402


class ReplaySource:
    """Comm 과 같은 인터페이스(recv/send/stale/connected/close)를 흉내낸다."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._f = open(path, encoding='utf-8')
        self.records: list[dict] = []      # recv 가 돌려준 틱의 원본 레코드
        self.sent: list[Command] = []

    def recv(self) -> RawPacket | None:
        """다음 줄의 raw 필드를 RawPacket 으로 복원. 끝나면 None."""
        while True:
            line = self._f.readline()
            if not line:
                return None
            if '"raw"' not in line:
                continue                      # event 줄 (info/warn/error)
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            raw = rec.get('raw')
            if not raw:
                continue
            self.records.append(rec)
            frames = int((rec.get('timing') or {}).get('frames_total', 0) or 0)
            return RawPacket(
                t_recv=float(rec['t']),
                ego=tuple(float(v) for v in raw['ego']),
                objects=[tuple(o) for o in raw.get('objects', [])],
                lights=[(int(i), int(s)) for i, s in raw.get('lights', [])],
                frames_total=frames,
            )

    def send(self, cmd: Command) -> bool:
        """리플레이에서는 실제로 보내지 않는다. 원본 Command 와 비교하면 회귀 검출."""
        self.sent.append(cmd)
        return True

    def stale(self, now: float | None = None) -> bool:
        return False

    @property
    def connected(self) -> bool:
        return True

    def close(self) -> None:
        if self._f is not None:
            self._f.close()
            self._f = None


# ── 지시등 시퀀스 비교 ─────────────────────────────────────────────────────
def segments(ticks: list[tuple[float, int, str]]) -> list[dict]:
    """(route_s, signal, state) 열 → 연속 점등 구간 목록."""
    out = []
    cur = None
    for s, sig, st in ticks:
        if cur is None or sig != cur['sig']:
            if cur is not None:
                out.append(cur)
            cur = {'sig': sig, 's0': s, 's1': s, 'n': 1, 'states': {st}}
        else:
            cur['s1'] = s
            cur['n'] += 1
            cur['states'].add(st)
    if cur is not None:
        out.append(cur)
    return out


def flicker_count(segs: list[dict], max_ticks: int = 3) -> int:
    """max_ticks 틱 이하로 켜졌다 꺼진(또는 꺼졌다 켜진) 토막 수 = 깜빡임."""
    return sum(1 for g in segs[1:-1] if g['n'] <= max_ticks)


def fmt_segments(segs: list[dict]) -> str:
    name = {0: 'OFF', 1: 'LEFT', 2: 'RIGHT'}
    rows = []
    for g in segs:
        if g['sig'] == 0:
            continue
        rows.append(f"  {name[g['sig']]:<5} {g['s0']:7.1f} → {g['s1']:7.1f} m  "
                    f"({g['n']:4d}틱, {'/'.join(sorted(g['states']))})")
    return '\n'.join(rows) if rows else '  (점등 없음)'


def load_signal_ticks(path: str) -> list[tuple[float, int, str]]:
    """로그에서 (route_s, turn_signal, state) 열을 읽는다."""
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '"raw"' not in line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append((rec['ego']['route_s'], rec['decision']['turn_signal'],
                        rec['decision']['state']))
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description='로그 분석 — 지시등 시퀀스')
    ap.add_argument('log')
    a = ap.parse_args(argv)

    so = segments(load_signal_ticks(a.log))
    print(f'{pathlib.Path(a.log).name} — 깜빡임 {flicker_count(so)}회, 전환 {len(so) - 1}회')
    print(fmt_segments(so))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
