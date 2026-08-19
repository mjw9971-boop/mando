"""
로그 → 채점 (법규 10항목 자동 판정)  ※ SPEC §3 기준 뼈대만.

    python3 tools/score.py logs/run_xxx.jsonl

각 판정은 로그의 WorldState/Command 만으로 재현 가능해야 한다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / 'src' / 'hlfma'))

from hlfma.core.scoring import SpeedMonitor  # noqa: E402

# SPEC §1.4 법규 10개
CHECKS = [
    'S1.1.01 제한속도 준수',
    'S1.1.02 보호구역(스쿨존) 속도 준수',
    'S2.1.01 차로 유지',
    'S2.1.02 중앙선 침범·우측통행',
    'S2.1.03 보도 침범 금지',
    'S2.2.05 실선 차로변경 금지',
    'S5.1.01 적색신호 정지',
    'S5.1.03 녹색신호 통과',
    'S6.1.01 도로 파손·장애물 대응',
    'S6.3.03 횡단보도 정차 금지',
]


def load(path: str) -> list[dict]:
    """jsonl 로드 (틱 레코드만)."""
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '"ego"' not in line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def driving_span(ticks: list[dict]) -> list[dict]:
    """
    실제 주행 구간만 남긴다.

    경로를 완주한 뒤에는 route_s 가 다른 구간에 다시 붙으며 요동쳐서,
    그대로 채점하면 없는 위반이 잡힌다. route_s 가 최고점에 처음 닿을 때까지만 센다.
    """
    if not ticks:
        return ticks
    peak = max(t['ego']['route_s'] for t in ticks)
    for i, t in enumerate(ticks):
        if t['ego']['route_s'] >= peak - 0.5:
            return ticks[:i + 1]
    return ticks


def speed_limit_check(ticks: list[dict], margin_kph: float,
                      school_cap_kph: float = 28.0) -> dict:
    """
    S1.1.01 제한속도 / S1.1.02 스쿨존 속도.

    허용치 = min(제한 - margin, 스쿨존이면 school_cap).
    **초과는 0 이어야 한다** — 채점 항목이다.
    """
    groups: dict = {}
    for t in ticks:
        lim = round(t['world']['speed_limit'] * 3.6)
        sz = bool(t['world']['school_zone'])
        g = groups.setdefault((lim, sz), {'v': [], 'over': [], 'tight': []})
        v = t['ego']['speed'] * 3.6
        # **위반 기준은 법규 제한속도 자체**다. margin 은 우리가 스스로 둔 여유이므로
        # 그걸 넘었다고 감점되지 않는다. 두 값을 따로 센다.
        legal = lim
        target = min(lim - margin_kph, school_cap_kph) if sz else lim - margin_kph
        g['v'].append(v)
        g['legal'] = legal
        g['target'] = target
        if v > legal:
            g['over'].append((v - legal, t['ego']['route_s']))
        if v > target:
            g['tight'].append(v - target)
    n_over = sum(len(g['over']) for g in groups.values())
    worst = max((max((o[0] for o in g['over']), default=0.0) for g in groups.values()),
                default=0.0)
    n_tight = sum(len(g['tight']) for g in groups.values())
    return {'groups': groups, 'n_ticks': len(ticks), 'n_over': n_over,
            'worst': worst, 'n_tight': n_tight}


def print_speed_report(res: dict, margin_kph: float) -> None:
    print(f"\n[제한속도] S1.1.01 / S1.1.02 — 주행 {res['n_ticks']}틱, margin {margin_kph:.0f} km/h")
    print(f"  {'제한':>5} {'스쿨존':>7} {'틱':>6} {'평균':>7} {'최대':>7} "
          f"{'법규위반':>9} {'최대초과':>9} {'목표(-m)':>9} {'목표초과':>9}")
    for (lim, sz) in sorted(res['groups']):
        g = res['groups'][(lim, sz)]
        v = g['v']
        ov = [o[0] for o in g['over']]
        print(f"  {lim:5d} {str(sz):>7} {len(v):6d} {sum(v)/len(v):7.1f} {max(v):7.1f} "
              f"{len(ov):9d} {('+%.2f' % max(ov)) if ov else '        —':>9} "
              f"{g['target']:9.1f} {len(g['tight']):9d}")
    verdict = '통과 (법규 위반 없음)' if res['n_over'] == 0 else f"**위반 {res['n_over']}틱**"
    print(f"  → {verdict}   최대 초과 +{res['worst']:.2f} km/h")
    print(f"     (목표 초과 {res['n_tight']}틱 = 우리 여유를 쓴 것일 뿐, 감점 아님)")


def evaluate(ticks: list[dict]) -> dict[str, dict]:
    """항목별 {위반 횟수, 최악값, 발생 시각} 산출."""
    # TODO: 제한속도 — speed > speed_limit 인 구간 길이/최대 초과량
    # TODO: 스쿨존 — school_zone 구간에서의 속도
    # TODO: 차로 유지 / 보도 침범 — |t_off| 와 lane width 비교
    # TODO: 중앙선 침범 — left_is_center 인데 t_off > 0
    # TODO: 실선 차로변경 — lane_change_ok False 인데 lane 이 바뀐 틱
    # TODO: 적색신호 정지 / 녹색신호 통과 — 정지선 통과 시점의 light state
    # TODO: 장애물 대응 — 장애물 존재 시 감속 여부
    # TODO: 횡단보도 정차 — 횡단보도 구간에서 speed≈0 이 유지된 시간
    raise NotImplementedError('score.evaluate')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='로그 자동 채점')
    ap.add_argument('log')
    ap.add_argument('--margin-kph', type=float, default=5.0,
                    help='제한속도에서 빼고 판정할 여유 (params 의 speed.margin_kph)')
    ap.add_argument('--all-ticks', action='store_true',
                    help='완주 이후 구간도 포함 (기본: 주행 구간만)')
    args = ap.parse_args(argv)

    ticks = load(args.log)
    if not ticks:
        print('틱 레코드가 없다', file=sys.stderr)
        return 1
    span = ticks if args.all_ticks else driving_span(ticks)

    t0 = span[0]['t']
    dist = max(t['ego']['route_s'] for t in span)
    dur = span[-1]['t'] - t0
    print(f'주행 {dist:.1f} m / {dur:.1f} s / 평균 {dist/max(dur,1e-9)*3.6:.1f} km/h')

    res = speed_limit_check(span, args.margin_kph)
    print_speed_report(res, args.margin_kph)

    n_reset = sum(1 for t in span if t['world']['flags'].get('reset'))
    n_stall = sum(1 for t in span if t['world']['flags'].get('stall'))
    print(f'\n[안정성] 리셋 {n_reset}회 / 스톨 {n_stall}회')

    # TODO: 나머지 채점 항목 (차로유지·중앙선·신호 등) — evaluate() 참고
    return 0 if res['n_over'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
