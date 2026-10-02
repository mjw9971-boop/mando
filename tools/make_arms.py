#!/usr/bin/env python3
"""구현(arm)별 설정 사본 생성기 — params.yaml + overlay → config/arms/A?.yaml

    python3 tools/make_arms.py            # 전부 다시 만든다
    python3 tools/make_arms.py --check    # 만들어 둔 사본이 지금의 params·overlay 와 맞는지만 본다

왜 사본인가 (2026-10-01, 논문 Ⅳ장 실험): 구현 간 차이를 config/params.yaml 을 고쳐서
내면 "어느 런이 어느 값이었나" 가 git 이력에만 남고, 제어기 섹션을 실험 때문에 건드리게
된다. 그래서 params.yaml 은 그대로 두고 **바뀌는 키만** config/arms/*.overlay.yaml 에
적은 뒤, 덮어쓴 **전체 사본**을 만들어 run_agent.py --config 와 batch_run.py --config 에
같은 파일을 준다.

정의의 단일 출처는 config/arms/arms.yaml 이다 (구현 목록·overlay 파일·이름).
사본에는 최상위 `arm:` 이 하나 더 붙는다 — 이름·overlay 키 전체·기준 params 의 해시.
batch_run.py 가 이걸 report.json 에 옮겨 적는다. 제어기는 모르는 최상위 키를 읽지 않는다.

안전장치:
  · arms.yaml 의 allowed_sections 밖의 섹션을 건드리는 overlay 는 거부한다 (현재 batch 만 —
    주행 파라미터는 사본에서도 바꾸지 않는다).
  · overlay 의 키는 **기준 params 에 이미 있는 키**여야 한다 (오타가 새 키로 조용히
    들어가면 스위치가 안 먹은 채 200런이 돈다).
  · 사본을 다시 읽었을 때 기준과 다른 키가 overlay 키와 정확히 같아야 한다.
  · --check 는 params 나 overlay 가 바뀐 뒤 사본을 안 다시 만든 것을 잡는다
    (paper_run.py 가 실행 전에 부른다).
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
ARMS_DIR = ROOT / 'config' / 'arms'
ARMS_YAML = ARMS_DIR / 'arms.yaml'
HEADER = ('# 생성물 — 손으로 고치지 말 것. tools/make_arms.py 가 {base} 에\n'
          '# {overlays} 를 덮어 만든다. 바뀐 키는 맨 아래 arm.overlay 에 있다.\n')


class ArmError(SystemExit):
    """구현 설정이 틀렸다 — 무엇이 왜 틀렸는지 메시지에 담는다."""


def load_yaml(path: pathlib.Path) -> dict:
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def load_defs(arms_yaml: pathlib.Path = ARMS_YAML) -> dict:
    """config/arms/arms.yaml — 구현 정의와 실행 계획."""
    d = load_yaml(arms_yaml)
    for k in ('base', 'common', 'arms'):
        if k not in d:
            raise ArmError(f'{arms_yaml}: `{k}` 가 없다')
    return d


def flatten(d: dict, prefix: str = '') -> dict:
    """중첩 dict → {'a.b.c': 값}. 값이 dict 가 아닌 잎만 남긴다."""
    out = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(flatten(v, key + '.'))
        else:
            out[key] = v
    return out


def apply_overlay(cfg: dict, overlay: dict, where: str, allowed=None) -> None:
    """overlay 의 잎 키를 cfg 에 덮는다 (제자리). 기준에 없는 키면 죽는다.

    allowed: 건드려도 되는 최상위 섹션 목록 (arms.yaml allowed_sections). 주면 그 밖의
    섹션을 건드리는 overlay 를 거부한다 — 주행 파라미터를 실험 사본에서 바꾸지 않기 위한
    못이다 (2026-10-02: 사본에서 바뀌는 것은 배치 판정값뿐).
    """
    for dotted, val in flatten(overlay).items():
        cur = cfg
        parts = dotted.split('.')
        if allowed is not None and parts[0] not in allowed:
            raise ArmError(f'{where}: `{dotted}` — `{parts[0]}` 섹션은 사본에서 바꿀 수 없다 '
                           f'(arms.yaml allowed_sections = {list(allowed)})')
        for q in parts[:-1]:
            if not isinstance(cur.get(q), dict):
                raise ArmError(f'{where}: `{dotted}` — 기준 params 에 `{q}` 섹션이 없다')
            cur = cur[q]
        if parts[-1] not in cur:
            raise ArmError(f'{where}: `{dotted}` 는 기준 params 에 없는 키다 (오타?)')
        cur[parts[-1]] = val


def build(name: str, defs: dict, arms_dir: pathlib.Path = ARMS_DIR) -> tuple[str, dict]:
    """구현 하나의 사본 → (파일 본문, arm 메타)."""
    base_path = ROOT / defs['base']
    base_bytes = base_path.read_bytes()
    base = yaml.safe_load(base_bytes) or {}
    if 'arm' in base:
        raise ArmError(f'{base_path}: 최상위 `arm` 키가 이미 있다 — 사본 꼬리표와 겹친다')
    spec = defs['arms'][name]
    files = [defs['common'], spec['overlay']]
    cfg = yaml.safe_load(base_bytes) or {}
    merged: dict = {}
    for fn in files:
        ov = load_yaml(arms_dir / fn)
        apply_overlay(cfg, ov, fn, defs.get('allowed_sections'))
        merged.update(flatten(ov))
    before = flatten(base)
    changed = {k: [before[k], v] for k, v in merged.items() if before[k] != v}
    arm = {'name': name, 'label_ko': spec.get('label_ko'), 'label_en': spec.get('label_en'),
           'base': defs['base'], 'base_sha256': hashlib.sha256(base_bytes).hexdigest(),
           'overlay_files': files,
           # overlay 가 지정한 키 전체 (기준과 값이 같아도 남긴다 — "지정했다" 가 기록이다)
           'overlay': merged,
           # 그중 기준과 실제로 달라진 것 [기준값, 새 값]
           'changed': changed}
    cfg['arm'] = arm
    text = (HEADER.format(base=defs['base'], overlays=' + '.join(files))
            + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    # 되읽어서 확인 — 기준과 다른 잎이 changed 와 정확히 같아야 한다.
    back = yaml.safe_load(text)
    back.pop('arm')
    after = flatten(back)
    diff = {k for k in set(before) | set(after) if before.get(k, object()) != after.get(k, object())}
    if diff != set(changed):
        raise ArmError(f'{name}: 사본이 기준과 다른 키 {sorted(diff)} ≠ overlay 로 바뀐 키 '
                       f'{sorted(changed)} (yaml 왕복에서 값이 변했다)')
    return text, arm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='params.yaml + overlay → config/arms/A?.yaml')
    ap.add_argument('arms', nargs='*', help='구현 이름 (기본: arms.yaml 의 전부)')
    ap.add_argument('--check', action='store_true',
                    help='쓰지 않고, 있는 사본이 지금의 params·overlay 와 같은지만 본다')
    a = ap.parse_args(argv)
    defs = load_defs()
    names = a.arms or list(defs['arms'])
    stale = []
    for name in names:
        if name not in defs['arms']:
            raise ArmError(f'모르는 구현 "{name}" — 가능: {", ".join(defs["arms"])}')
        text, arm = build(name, defs)
        out = ARMS_DIR / f'{name}.yaml'
        if a.check:
            ok = out.exists() and out.read_text(encoding='utf-8') == text
            print(f'  {name}: {"최신" if ok else "낡음 — make_arms.py 를 다시 돌릴 것"}  ({out.relative_to(ROOT)})')
            if not ok:
                stale.append(name)
            continue
        out.write_text(text, encoding='utf-8')
        print(f'  {name}: {out.relative_to(ROOT)}  — {arm["label_ko"]}')
        for k, (old, new) in sorted(arm['changed'].items()):
            print(f'      {k}: {old} → {new}')
        same = sorted(set(arm['overlay']) - set(arm['changed']))
        if same:
            print(f'      (기준과 같은 값으로 지정: {", ".join(same)})')
    if stale:
        print(f'[중단] 낡은 사본 {stale} — params.yaml 또는 overlay 가 바뀌었다', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
