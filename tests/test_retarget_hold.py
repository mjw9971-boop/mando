"""재타겟 최소 유지 시간 + 마진 게이트의 섹션 구멍
(avoid_map.retarget_min_hold_s · `_free_of`).

실측 run_20260910_230148 t 49~108: `lane_plan.pick` 이
  −2 → −1 → −2 → −1 → None(ramp_too_late) → −1 → None(no_candidate)
  → −4 → None(ramp_too_late) → −4 → None(ramp_too_late)
로 흔들리는 동안 여유가 **76.7 → 9.5 m** 로 줄었고, 결국
`ramp_too_late`(avail 4.5 < need 25.5)로 rs 256.2 에서 13 s 정지했다.
74 m 있을 때 하나를 정해 끝냈으면 됐다.

**마진(lane_switch_margin_m 20)은 이걸 못 막는다. 두 가지 이유가 있다.**

① 마진은 `_lm_retarget` 에만 걸린다 — `lane_plan.pick` 자체는 매 틱 free_run
   최대값으로 자유롭게 다시 뽑힌다.
② 그나마도 조회가 **섹션 인덱스를 그대로 써서** 자차가 섹션을 넘는 순간
   `cur_f is None` 이 되어 조용히 통과했다. 실측에서 free_run 키의 섹션이
   t 79.4 에 0 → 1, t 82.1 에 1 → 2 로 바뀌는데 시프트는 t 54.0(섹션 0)에
   만들어졌다 — 그 뒤 **모든** 재타겟에서 마진이 한 번도 안 걸렸다.
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


def cfg(hold=None):
    c = copy.deepcopy(CFG)
    if hold is not None:
        c['avoid_map']['retarget_min_hold_s'] = hold
    return c


# ── `_free_of` — 섹션을 무시하고 찾는다 ─────────────────────────────────
FREE = {'[940, 2, -1]': 44.1, '[940, 2, -2]': 33.7, '[940, 2, -3]': 13.7}


def test_exact_key_still_works():
    assert KrRules(cfg())._free_of(FREE, (940, 2, -2)) == pytest.approx(33.7)


@pytest.mark.parametrize('sec', [0, 1, 3, 7])
def test_other_section_of_the_same_lane_is_found(sec):
    """이 구멍이 실측에서 마진을 통째로 죽였다."""
    assert KrRules(cfg())._free_of(FREE, (940, sec, -2)) == pytest.approx(33.7)


def test_unknown_lane_is_still_none():
    assert KrRules(cfg())._free_of(FREE, (940, 0, -9)) is None


def test_other_road_is_not_confused():
    assert KrRules(cfg())._free_of(FREE, (2820, 2, -2)) is None


def test_string_key_form_is_accepted():
    assert KrRules(cfg())._free_of(FREE, '[940, 0, -1]') == pytest.approx(44.1)


def test_malformed_key_falls_back_to_none():
    """파싱 실패는 조용히 옛 동작(조회 실패)으로 — 예외를 던지지 않는다."""
    kr = KrRules(cfg())
    assert kr._free_of(FREE, 'nonsense') is None


def test_section_match_reuses_the_existing_helper():
    """섹션 제거는 `_key_road_lane` 하나뿐이다 — 같은 판정을 두 벌 만들지 않는다."""
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text(encoding='utf-8')
    i = src.index('def _free_of(self')
    assert '_key_road_lane' in src[i:i + 900]
    assert 'def _road_lane(' not in src


def test_margin_would_now_block_the_logged_switch():
    """t 79.4 실측값 — −2(42.5) 에서 −1(54.2) 로 갈아타기: 이득 11.7 < 20."""
    kr = KrRules(cfg())
    free = {'[940, 1, -1]': 54.2, '[940, 1, -2]': 42.5}
    new_f = kr._free_of(free, (940, 1, -1))
    cur_f = kr._free_of(free, (940, 0, -2))            # 섹션 0 에서 만든 목표
    assert cur_f is not None                           # 예전에는 None 이었다
    assert new_f - cur_f < kr.lm_switch_margin


# ── 최소 유지 시간 ──────────────────────────────────────────────────────
def test_default_is_off():
    """커밋본 기본값 = 이전 동작 (params 가 정본, 배치 오버레이에서 켠다)."""
    assert KrRules(cfg()).retarget_hold_ticks == 0


def test_hold_seconds_convert_to_ticks():
    kr = KrRules(cfg(3.0))
    assert kr.retarget_hold_ticks == int(round(3.0 * kr.hz))


def test_lock_starts_when_a_target_is_chosen():
    """최초 시프트도 재타겟과 같다 — '한 번 정하면 그동안 안 바꾼다'."""
    kr = KrRules(cfg(3.0))
    assert kr._retarget_lock == 0
    kr._retarget_lock = kr.retarget_hold_ticks         # _apply_shift 가 하는 일
    assert kr._retarget_lock == int(round(3.0 * kr.hz))


def test_lock_counts_down_and_blocks(monkeypatch):
    """잠금이 살아 있는 동안 `_lm_retarget` 은 매 틱 False 를 돌려주며 센다."""
    kr = KrRules(cfg(3.0))
    kr.lm_retarget = True
    kr.lane_map_on = True
    kr.last_lane_plan = {'pick': '[940, 2, -4]', 'side': 'right'}
    kr.ot_target = (940, 2, -1)
    kr._retarget_lock = 3
    for left in (3, 2, 1):
        assert kr._retarget_lock == left
        assert kr._lm_retarget(None, None, 0.0) is False
    assert kr._retarget_lock == 0


def test_off_never_locks():
    """0 이면 잠금 분기가 아예 안 선다 = 이전 동작."""
    kr = KrRules(cfg(0.0))
    kr.lm_retarget = True
    kr.lane_map_on = True
    kr.last_lane_plan = {'pick': '[940, 2, -4]'}
    kr.ot_target = (940, 2, -1)
    kr._retarget_lock = 5                              # 남아 있어도 무시된다
    kr._lm_retarget(None, None, 0.0)                   # 마진/회랑에서 걸려 False
    assert kr._retarget_lock == 5                      # 세지 않는다


def test_restore_clears_the_lock():
    """원복 뒤 새 시프트를 잠금이 막으면 안 된다."""
    kr = KrRules(cfg(3.0))
    kr._retarget_lock = 40
    kr.ot_span = (0, 10)
    kr.ot_ids = [1]

    class P:
        route_points = __import__('numpy').zeros((20, 3))
        original_route_points = __import__('numpy').zeros((20, 3))
        commands = __import__('numpy').zeros(20, dtype=int)
        commands_orig = __import__('numpy').zeros(20, dtype=int)
        lat_shift = __import__('numpy').zeros(20)
        _lat_build = __import__('numpy').zeros(20)
        route_index = 15                               # span 을 지났다
        points_per_meter = 10
    kr._restore_span(P())
    assert kr._retarget_lock == 0
