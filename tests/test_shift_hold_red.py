"""
(1) 적색 홀드의 거리 상한 — `avoid_map.shift_hold_red_bounded_enable`.

`red_hold` 는 `_red_ahead` 를 쓴다 — **거리 무관**이다 (legacy 억제의 설계).
큐 판정·BREAKOUT pause 는 이미 `overtake.red_pause_max_m`(100) 상한을 씌운
`_red_pause` 를 쓰는데, 홀드만 옛 축에 남아 있었다.

실측 2026-09-10 run_20260910_144656:
  SHIFT_HOLD/red_ahead **597틱 (29.9 s / 90.7 s)**
    dist_stop_line 이 null (정지선이 전방 창 밖)  262틱 (13.1 s)
    100 m 초과                                    39틱  (1.9 s)
    100 m 이내 (상한을 켜도 그대로 유지)          296틱 (14.8 s)
  → 상한을 적용하면 최대 301틱(15.1 s)이 풀린다. 나머지 절반은 진짜 접근이라
    홀드가 유지되는 것이 맞다.
"""
import copy
import pathlib
import sys

import numpy as np
import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)

EGO = (2533, 0, 5)
L4, L3, L6 = (2533, 0, 4), (2533, 0, 3), (2533, 0, 6)
ORDER = [L3, L4, EGO, L6]


class Lg:
    def __init__(self):
        s = np.linspace(0.0, 1000.0, 51)
        self.lanes = {k: {'junction': -1, 'dir': -1, 's': s,
                          'width': np.full_like(s, 3.0), 'length': 1000.0}
                      for k in ORDER}

    def neighbor(self, key, side):
        if key not in ORDER:
            return None
        i = ORDER.index(key) + (1 if side == 'right' else -1)
        return ORDER[i] if 0 <= i < len(ORDER) else None


class P:
    pass


def kr(**over):
    c = copy.deepcopy(CFG)
    c['avoid_map'].update(over)
    return KrRules(c)


# ── (1) ──────────────────────────────────────────────────────────────────
def test_hold_bound_default_is_off():
    assert CFG['avoid_map']['shift_hold_red_bounded_enable'] is False


def test_red_pause_max_m_is_the_shared_constant():
    """새 상수를 만들지 않는다 — 큐 판정이 쓰는 값 그대로."""
    assert float(CFG['overtake']['red_pause_max_m']) == 100.0


def test_hold_uses_red_ahead_when_off_and_red_pause_when_on():
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text()
    i = src.index('red_hold = self.ot_span is not None')
    blk = src[max(0, i - 900):i + 120]
    assert 'self._red_pause(planner) if self.hold_red_bounded' in blk
    assert 'else self._red_ahead(planner)' in blk


def test_bounded_hold_ignores_a_far_red():
    """상한 밖 적색은 홀드를 걸지 않는다 (_red_pause 의 정의)."""
    k = kr(shift_hold_red_bounded_enable=True)
    k._red_ahead = lambda _p: 456.0
    assert k._red_pause(None) is None
    k._red_ahead = lambda _p: 84.3          # 로그의 t 47.2 값 — 상한 안이다
    assert k._red_pause(None) == pytest.approx(84.3)


def test_unbounded_hold_keeps_the_far_red():
    k = kr(shift_hold_red_bounded_enable=False)
    k._red_ahead = lambda _p: 456.0
    assert k.hold_red_bounded is False


