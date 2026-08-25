"""
config/params.yaml 로더.

단일출처 규칙 (phase1 에서 확정):
  · VTD·어댑터 상수(comm.*, vehicle.*, percep.*, batch.*, log.* 등) → config/params.yaml
  · PDM-Lite 판단 상수(IDM, forecast, lateral PID 등)              → team_code/config.py
코드 안의 DEFAULTS dict 이중화는 폐지했다 — yaml 이 없거나 키가 빠지면
조용히 기본값으로 도는 게 아니라 즉시 죽어야 설정 두 벌이 어긋나는 사고가 없다.
"""
from __future__ import annotations

import pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_PARAMS = ROOT / 'config' / 'params.yaml'


def load_params_yaml(path: str | None = None) -> dict[str, Any]:
    """config/params.yaml → 중첩 dict. 파일이 없으면 FileNotFoundError 를 그대로 낸다."""
    import yaml

    p = pathlib.Path(path) if path else DEFAULT_PARAMS
    with open(p, 'r', encoding='utf-8') as f:
        doc = yaml.safe_load(f) or {}

    # 구 ROS 형식(`/**: ros__parameters: {flat}`)도 받아 준다 — 옛 로그 재현용.
    if '/**' in doc and isinstance(doc['/**'], dict):
        flat = doc['/**'].get('ros__parameters', {})
        out: dict[str, Any] = {}
        for key, v in flat.items():
            parts = key.split('.')
            cur = out
            for q in parts[:-1]:
                cur = cur.setdefault(q, {})
            cur[parts[-1]] = v
        return out
    return doc
