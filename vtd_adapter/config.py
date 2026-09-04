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


def end_margin_m(cfg: dict[str, Any]) -> float:
    """완주 임계 [m]: route_s(뒷축) ≥ total − 이 값이면 완주. **폴백 경로 전용.**

    계획 정지점이 total − stop_gap − (wheelbase + front_overhang) 이므로
    임계 = stop_gap_route_end + 앞범퍼거리 + 여유(end_slack). stop_gap 튜닝을 자동으로
    따라간다 — 2026-08-25: stop_gap 1→4 후 고정 임계 5 m 로 완주가 timeout 처리.
    batch_run / summarize_run / score 가 전부 이 함수를 본다 (단일 출처).

    **이 유도식은 route_end.target_mode 가 'route_total' 일 때만 맞다.** 현재
    기본값은 'finish' 이고 계획 정지점은 finish_s + finish_clearance 라, 경로
    꼬리(route.finish_tail_m, 기본 12 m)만큼 이 임계보다 앞에 선다. 그래서
    score.detect_finish 와 batch_run 은 finish_xy 를 경로에 투영한 finish_s 를
    먼저 쓰고, 그게 없을 때만 이 값으로 폴백한다
    (batch.finish_judge_use_finish_s). 2026-09-04 실측: 11개 CSV 전부 계획
    정지점이 이 임계에 −4.2 ~ −78.2 m 못 미친다.
    """
    sp, vh = cfg['speed'], cfg['vehicle']
    return (float(sp['stop_gap_route_end_m']) + float(vh['wheelbase'])
            + float(vh.get('front_overhang_m', 0.855))
            + float(cfg.get('batch', {}).get('end_slack_m', 1.0)))
