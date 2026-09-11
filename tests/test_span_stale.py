"""목표는 있는데 못 가고 있는 상태의 시한 (avoid_map.span_stale_max_s).

`never_stall`(overtake.deadlock_max_s)과 **축이 다르다**. 그쪽 시계는
`route_s` 가 `bo_progress_m` 만큼 나아가면 리셋되므로 — [kr_rules.py] 의
`_never_stall` 주석 "조금씩이라도 가고 있으면 stall 이 아니다" — **달리는
중에는 원리적으로 안 걸린다**. 종방향 진전을 보는 시계이기 때문이다.

실측 run_20260911_001250 t 130~150 이 정확히 그 사각이다: v 3~5.5 m/s 로
계속 달리면서 목표 −1 까지 **한 칸도 줄이지 못한 채** 25 s 를 썼고, 그 사이
여유가 69.5 → 20.8 m 로 사라져 `ramp_too_late` 로 끝났다 (총 정지 60 s).
`never_stall` 은 그 구간에서 한 번도 무장하지 않았다 (계속 나아가고 있었으므로).

이 시계는 **잔여 칸수**로 잰다 — `lane_map.hops[pick]` 은 자차가 지금 밟고
있는 차로 기준 오프셋이라, 줄면 시프트가 제 일을 하는 것이고 안 줄면 아무
일도 안 일어나고 있는 것이다.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                        # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
HOPS = {'[2011, 5, -2]': 0, '[2011, 5, -1]': -1,
        '[2011, 5, -3]': 1, '[2011, 5, -4]': 2}


def kr(stale_s=8.0):
    c = copy.deepcopy(CFG)
    c['avoid_map']['span_stale_max_s'] = stale_s
    k = KrRules(c)
    k.last_lane_map = {'hops': HOPS}
    k.last_lane_plan = {'pick': '[2011, 5, -1]'}
    return k


def run(k, n):
    """n 틱 돌려 발동 틱 번호를 돌려준다 (없으면 None)."""
    for i in range(n):
        if k._span_stale_tick():
            return i
    return None


def test_default_is_off():
    assert KrRules(CFG).span_stale_max_ticks == 0


def test_off_never_fires():
    k = kr(0.0)
    assert run(k, 1000) is None
    assert k._span_stale_ticks == 0


def test_fires_after_the_deadline():
    """첫 틱은 기준만 잡고 안 센다 — 그래서 시한 + 1틱에 발동한다."""
    k = kr(8.0)
    assert run(k, 400) == int(round(8.0 * k.hz))


def test_progress_resets_the_clock():
    """한 칸 줄면 시계를 새로 — 조금씩이라도 가고 있으면 버리지 않는다."""
    k = kr(8.0)
    for _ in range(100):
        assert k._span_stale_tick() is False
    k.last_lane_map = {'hops': dict(HOPS, **{'[2011, 5, -1]': 0})}   # 도착
    assert k._span_stale_tick() is False
    assert k._span_stale_ticks == 0


def test_two_hop_shift_counts_down_step_by_step():
    """2칸이면 1칸으로 줄 때마다 시계가 새로 선다 (긴 시프트를 안 버린다)."""
    k = kr(8.0)
    k.last_lane_plan = {'pick': '[2011, 5, -4]'}                     # 2칸
    for _ in range(100):
        k._span_stale_tick()
    assert k._span_stale_ticks == 99                                 # 첫 틱은 기준
    k.last_lane_map = {'hops': dict(HOPS, **{'[2011, 5, -4]': 1})}   # 1칸으로
    assert k._span_stale_tick() is False
    assert k._span_stale_ticks == 0


def test_growing_gap_keeps_counting():
    """멀어지는 것은 진전이 아니다 — 시계를 되돌리지 않는다."""
    k = kr(8.0)
    for _ in range(50):
        k._span_stale_tick()
    k.last_lane_map = {'hops': dict(HOPS, **{'[2011, 5, -1]': -2})}  # 더 멀어짐
    k._span_stale_tick()
    assert k._span_stale_ticks == 50                                 # 첫 틱은 기준


def test_arrival_clears_the_reference():
    k = kr(8.0)
    run(k, 10)
    k.last_lane_map = {'hops': dict(HOPS, **{'[2011, 5, -1]': 0})}
    k._span_stale_tick()
    assert k._span_stale_ref is None


@pytest.mark.parametrize('plan', [None, {}, {'pick': None},
                                  {'pick': '[9999, 0, -1]'}])
def test_unmeasurable_does_not_count(plan):
    """잴 수 없으면 세지 않는다 — 모르는 것을 근거로 span 을 버리지 않는다."""
    k = kr(8.0)
    k.last_lane_plan = plan
    assert run(k, 400) is None
    assert k._span_stale_ticks == 0


def test_no_lane_map_does_not_count():
    k = kr(8.0)
    k.last_lane_map = None
    assert run(k, 400) is None


# ── never_stall 과 겹치지 않는다 ────────────────────────────────────────
def test_never_stall_resets_on_longitudinal_progress():
    """겹치지 않는 근거를 소스로 고정한다 — 그쪽은 route_s 축이다."""
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text(encoding='utf-8')
    i = src.index('elif route_s - self.ns_ref_s >= self.bo_progress_m:')
    assert 'self.ns_ticks = 0' in src[i:i + 200]
    # 이쪽 시계는 route_s 를 아예 안 본다
    j = src.index('def _span_stale_tick(self)')
    body = src[j:src.index('def _span_targets_lost', j)]
    code = body[body.index('"""', body.index('"""') + 3) + 3:]       # docstring 제외
    assert 'route_s' not in code                                     # 종방향을 안 본다
    assert 'hops' in code


def test_new_target_resets_the_clock_in_apply_shift():
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text(encoding='utf-8')
    i = src.index('self.ot_target = target')
    assert '_span_stale_ticks = 0' in src[i:i + 500]
    assert '_span_stale_ref = None' in src[i:i + 500]


def test_caller_restores_the_span_and_labels_it():
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text(encoding='utf-8')
    i = src.index('if self._span_stale_tick():')
    blk = src[i:i + 500]
    assert '_restore_span(planner, ego_speed)' in blk
    assert "'why': 'span_stale'" in blk
