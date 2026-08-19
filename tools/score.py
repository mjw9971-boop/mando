"""
로그 → 채점 (법규 10항목 자동 판정)  ※ SPEC §3 기준 뼈대만.

    python3 tools/score.py logs/run_xxx.jsonl

각 판정은 로그의 WorldState/Command 만으로 재현 가능해야 한다.
"""
from __future__ import annotations

import argparse

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
    """jsonl 로드."""
    # TODO: 구현
    raise NotImplementedError('score.load')


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
    args = ap.parse_args(argv)
    result = evaluate(load(args.log))
    for name in CHECKS:
        print(f'{name:34s} {result.get(name, {})}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
