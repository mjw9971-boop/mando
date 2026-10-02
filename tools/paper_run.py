#!/usr/bin/env python3
"""논문 Ⅳ장 실험 실행기 — 구현(arm) × 유형 계획을 짜서 batch_run 을 차례로 부른다.

    python3 tools/paper_run.py --dry-run          # 계획(런 수·예상 시간)만 출력하고 끝
    python3 tools/paper_run.py                    # 실행. 끝난 런은 건너뛴다
    python3 tools/paper_run.py --arms A3 --types 급정거집중

계획의 단일 출처는 config/arms/arms.yaml 이다 (구현 목록 · 목록 파일 · 출력 위치 · host).
구현별 설정은 tools/make_arms.py 가 만든 config/arms/A?.yaml 이고, batch_run 에
--config 로 넘겨 **종료 판정(batch.*)과 run_agent 가 같은 파일**을 읽게 한다.
결과는 <out_dir>/<구현>/<유형>/ 에 batch_run 의 산출물 그대로 쌓인다
(report.json · report.txt · <이름>.jsonl · score/events 전문 · routes/).

이어하기: batch_run --resume 이 report.json 에서 끝난 런(완주·stall·blocked·
no_progress·timeout)을 건너뛴다. 기반 실패(SCP·VTD 무응답·no-data·launch 사망)는
끝난 런이 아니므로 다음 실행에서 다시 돈다. Ctrl+C 로 멈추면 남은 배치로 넘어가지 않는다.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import signal
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import batch_run          # noqa: E402
import make_arms          # noqa: E402

DONE = batch_run.Runner.DONE_STATUS


def type_of(list_name: str) -> str:
    """batch_<유형>.json → <유형>."""
    stem = pathlib.Path(list_name).stem
    return stem[len('batch_'):] if stem.startswith('batch_') else stem


def build_plan(defs: dict, arms=None, types=None) -> list[dict]:
    """arms.yaml 의 plan → [{arm, type, list, config, out_dir, items, done}]."""
    sc_dir = ROOT / defs['scenario_dir']
    out = []
    for block in defs['plan']:
        for arm in block['arms']:
            if arm not in defs['arms']:
                raise SystemExit(f'arms.yaml plan: 모르는 구현 "{arm}"')
            if arms and arm not in arms:
                continue
            for ln in block['lists']:
                typ = type_of(ln)
                if types and typ not in types:
                    continue
                lp = sc_dir / ln
                if not lp.exists():
                    raise SystemExit(f'목록 파일이 없다: {lp}')
                items = json.loads(lp.read_text(encoding='utf-8'))
                od = ROOT / defs['out_dir'] / arm / typ
                done = {}
                rp = od / 'report.json'
                if rp.exists():
                    done = {r['name']: r['status'] for r in json.loads(rp.read_text(encoding='utf-8'))
                            if r.get('status') in DONE}
                out.append({'arm': arm, 'type': typ, 'list': lp,
                            'config': make_arms.ARMS_DIR / f'{arm}.yaml',
                            'out_dir': od, 'items': items, 'done': done})
    return out


def est_s(item: dict) -> float | None:
    """생성기가 정의에 남긴 예상 주행 시간 [s] (route_csv 옆 yaml 의 est_s)."""
    y = (ROOT / item['route_csv']).with_suffix('.yaml')
    if not y.exists():
        return None
    return float((yaml.safe_load(y.read_text(encoding='utf-8')) or {}).get('est_s') or 0.0) or None


def hms(sec: float) -> str:
    return f'{sec / 3600:5.2f} h'


def print_plan(plan: list[dict], defs: dict, host: str) -> None:
    bp = batch_run.build_parser()
    # 런당 고정 비용 = SCP 대기 3회(Load·Init·Start) + 시나리오 간 대기. batch_run 기본값을 읽는다.
    overhead = 3 * float(bp.get_default('settle_s')) + float(bp.get_default('pause_s'))
    print(f'논문 실험 계획  (host {host}, 출력 {defs["out_dir"]}/<구현>/<유형>/)')
    for name, spec in defs['arms'].items():
        print(f'  {name}: {spec["label_ko"]}')
    print()
    print(f'{"구현":4s} {"유형":14s} {"런":>4s} {"끝남":>4s} {"남음":>4s} {"예상":>8s} {"상한":>8s}  설정')
    tot = {'n': 0, 'done': 0, 'left': 0, 'est': 0.0, 'cap': 0.0}
    no_est = 0
    for p in plan:
        left = [i for i in p['items'] if i['name'] not in p['done']]
        est = cap = 0.0
        for i in left:
            e = est_s(i)
            if e is None:
                no_est += 1
                e = float(i['timeout_s'])
            est += e + overhead
            cap += float(i['timeout_s']) + overhead
        print(f'{p["arm"]:4s} {p["type"]:14s} {len(p["items"]):4d} {len(p["done"]):4d} {len(left):4d} '
              f'{hms(est)} {hms(cap)}  {p["config"].relative_to(ROOT)}')
        tot['n'] += len(p['items']); tot['done'] += len(p['done']); tot['left'] += len(left)
        tot['est'] += est; tot['cap'] += cap
    print(f'{"합계":4s} {"":14s} {tot["n"]:4d} {tot["done"]:4d} {tot["left"]:4d} '
          f'{hms(tot["est"])} {hms(tot["cap"])}')
    print()
    print(f'예상 = 생성기 추정 주행 시간(est_s) + 런당 고정 비용 {overhead:.0f} s. '
          f'상한 = 시나리오 timeout_s + 같은 비용.')
    print('  · est_s 는 완주를 가정한 값이다. 60 s 정지로 조기 종료되는 런은 그보다 짧고,')
    print('    신호 대기가 길면 길어진다. VTD 시나리오 로드 시간은 들어 있지 않다.')
    if no_est:
        print(f'  · est_s 가 없는 런 {no_est}개는 timeout_s 로 대신 셌다.')
    b = yaml.safe_load((make_arms.ARMS_DIR / f'{plan[0]["arm"]}.yaml').read_text(encoding='utf-8'))['batch'] if plan else {}
    if b:
        print(f'종료 판정(구현 사본 기준): stall {b["stall_end_s"]:g} s · blocked {b["blocked_end_s"]:g} s · '
              f'no_progress {b["no_progress_end_s"]:g} s · 큐 면책 {b["stall_excuse_queue"]}')


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='논문 실험: 구현 × 유형을 차례로 실행 (이어하기 지원)')
    ap.add_argument('--dry-run', action='store_true', help='계획만 출력하고 끝낸다')
    ap.add_argument('--arms', default=None, help='쉼표로 구분한 구현 이름만 (예: A1,A3)')
    ap.add_argument('--types', default=None, help='쉼표로 구분한 유형만 (예: 급정거집중)')
    ap.add_argument('--host', default=None, help='VTD 주소 (기본: arms.yaml 의 host)')
    a = ap.parse_args(argv)

    defs = make_arms.load_defs()
    host = a.host or defs['host']
    plan = build_plan(defs, a.arms.split(',') if a.arms else None,
                      a.types.split(',') if a.types else None)
    if not plan:
        raise SystemExit('계획이 비었다 (--arms/--types 필터 확인)')
    # 구현 사본이 지금의 params·overlay 와 맞는가 — 낡은 사본으로 돌리지 않는다.
    stale = [n for n in sorted({p['arm'] for p in plan})
             if not (make_arms.ARMS_DIR / f'{n}.yaml').exists()
             or (make_arms.ARMS_DIR / f'{n}.yaml').read_text(encoding='utf-8')
             != make_arms.build(n, defs)[0]]
    print_plan(plan, defs, host)
    if stale:
        print(f'\n[중단] 구현 사본이 낡았다: {stale} — python3 tools/make_arms.py 를 다시 돌릴 것')
        return 2
    if a.dry_run:
        print('\n실행 순서:')
        for p in plan:
            left = len(p['items']) - len(p['done'])
            if left:
                print(f'  python3 tools/batch_run.py {p["list"].relative_to(ROOT)} '
                      f'--config {p["config"].relative_to(ROOT)} '
                      f'--out-dir {p["out_dir"].relative_to(ROOT)} --resume --host {host}')
        print('\n[dry-run] 여기서 멈춘다 — 실제 실행은 --dry-run 없이.')
        return 0

    stop = {'int': False}

    def on_int(_sig, _frm):
        stop['int'] = True
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, on_int)
    for n, p in enumerate(plan, 1):
        left = len(p['items']) - len(p['done'])
        print(f'\n{"#" * 78}\n# [{n}/{len(plan)}] 구현 {p["arm"]} · {p["type"]} — 남은 런 {left}/{len(p["items"])}\n{"#" * 78}')
        if not left:
            continue
        batch_run.main([str(p['list']), '--config', str(p['config']),
                        '--out-dir', str(p['out_dir']), '--resume', '--host', host])
        if stop['int']:
            print('\n[중단] 남은 배치는 돌리지 않는다. 다시 실행하면 끝난 런을 건너뛰고 잇는다.')
            return 130
    print()
    print_plan(build_plan(defs, a.arms.split(',') if a.arms else None,
                          a.types.split(',') if a.types else None), defs, host)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
