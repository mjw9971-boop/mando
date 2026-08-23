"""
로그 jsonl → Comm 대체 소스  (SPEC §3.7)

VTD 없이 Perception/Planner/Shield/Control 을 다시 돌린다.
판단 로직을 고친 뒤 회귀 테스트하는 기반.

    python3 tools/run_standalone.py --replay logs/run_xxx.jsonl
    python3 tools/replay.py logs/run_xxx.jsonl          # 지시등 시퀀스 원본 vs 재생 비교

**개루프**다: 자차 궤적은 원본 런 그대로이고 새 판단이 궤적을 바꾸지는
않는다. 지시등·상태·속도후보처럼 "같은 입력에 대한 판단" 의 회귀 비교에 쓴다.
"""
from __future__ import annotations

import json
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src' / 'hlfma'))

from hlfma.core.types import Command, RawPacket  # noqa: E402


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


def replay_signals(path: str, config: str | None = None,
                   graph: str = 'data/lane_graph.pkl', route: str = 'data/route.pkl'):
    """로그를 Perception→Planner→Shield 로 재생. (원본 열, 재생 열) 을 돌려준다."""
    from hlfma.core.lanegraph import LaneGraph
    from hlfma.core.perception import Perception
    from hlfma.core.planner import Planner
    from hlfma.core.shield import Shield
    from hlfma.nodes.params import load_params_yaml
    import pickle

    cfg = load_params_yaml(config or str(_ROOT / 'src' / 'hlfma' / 'config' / 'params.yaml'))
    lg = LaneGraph(str(_ROOT / graph))
    with open(_ROOT / route, 'rb') as f:
        rt = pickle.load(f)
    per = Perception(lg, rt, cfg)
    pl = Planner(lg, rt, cfg)
    sh = Shield(lg, cfg, planner=pl)

    src = ReplaySource(path)
    orig, new = [], []
    while True:
        pkt = src.recv()
        if pkt is None:
            break
        rec = src.records[-1]
        w = per.update(pkt)
        d = sh.apply(w, pl.plan(w))
        orig.append((rec['ego']['route_s'], rec['decision']['turn_signal'], rec['decision']['state']))
        new.append((w.ego.route_s, d.turn_signal, d.state))
    src.close()
    return orig, new, pl


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description='로그 재생 — 지시등 시퀀스 비교')
    ap.add_argument('log')
    ap.add_argument('--config', default=None)
    a = ap.parse_args(argv)

    orig, new, pl = replay_signals(a.log, a.config)
    so, sn = segments(orig), segments(new)
    print(f'원본 ({pathlib.Path(a.log).name}) — 깜빡임 {flicker_count(so)}회, 전환 {len(so) - 1}회')
    print(fmt_segments(so))
    print(f'재생 — 깜빡임 {flicker_count(sn)}회, 전환 {len(sn) - 1}회  '
          f'(LC 완료 {pl.lc_done}, 중단 {pl.lc_aborted})')
    print(fmt_segments(sn))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
