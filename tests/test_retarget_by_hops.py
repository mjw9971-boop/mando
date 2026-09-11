"""재타겟 판정을 **자차 물리 차로 기준 잔여 칸수**로
(avoid_map.retarget_by_hops_enable).

옛 판정 축은 `ot_target` — 시프트를 만들 때 적어 둔 **의도**였다. 의도와 기하는
어긋날 수 있다: 시프트는 그때 연 칸 수만큼만 움직이고, 도중에 차로가 생기거나
없어지면 같은 횡변위가 다른 차로에 떨어진다.

실측 run_20260911_001250 — road 2011 의 차로 구성이 섹션마다 **재배열**된다:

    sec 0·1  [-3, -2, -1]
    sec 2    [-4, -3, -2, -1]      ← 안쪽에 한 칸이 생겨 번호가 밀린다
    sec 5    [-5, -4, -3, -2, -1]

자차는 t 142.0 에 `(2011,1,-2)` → t 142.1 에 `(2011,2,-3)` 으로 바뀌는데
`t_off` 는 0.00, 조향은 0.000 이다 — **물리적으로 안 움직였다**. 번호만 밀렸다.

그래서 t 129.9 재타겟이 연 한 칸은 −1 이 아니라 새로 생긴 차로에 떨어졌고,
−1 은 여전히 한 칸 밖인데 게이트는 `pick == ot_target` 이라 t 130~150 의
**25 s 를 전부 막았다** (`retarget_why` 가 그 구간 내내 `margin:+0.0`).
그동안 여유가 69.5 → 20.8 m 로 줄어 `ramp_too_late` 로 끝났다 (총 정지 60 s).

`lane_map.hops` 는 **자차가 지금 밟고 있는 차로 기준** 오프셋이라 의도도 섹션
번호도 타지 않는다. 0 이면 도착, 그 밖이면 아직 그만큼 남은 것이다.
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

# 실측 t 147.2 의 지도 (자차 (2011,5,-2), 목표 (2011,5,-1))
HOPS = {'[2011, 5, -2]': 0, '[2011, 5, -1]': -1,
        '[2011, 5, -3]': 1, '[2011, 5, -4]': 2}
FREE = {'[2011, 5, -2]': 24.4, '[2011, 5, -1]': 80.0,
        '[2011, 5, -3]': 10.5, '[2011, 5, -4]': 80.0}
EGO = [2011, 5, -2]


def cfg(on, **over):
    c = copy.deepcopy(CFG)
    c['avoid_map']['retarget_by_hops_enable'] = on
    c['avoid_map'].update(over)
    return c


def rig(on, pick, ot_target, **over):
    kr = KrRules(cfg(on, **over))
    kr.lm_retarget = True
    kr.lane_map_on = True
    kr.last_lane_map = {'hops': HOPS, 'free_run': FREE, 'ego_lane': EGO}
    kr.last_lane_plan = {'pick': pick, 'side': 'left'}
    kr.ot_target = ot_target
    kr._retarget_lock = 0
    return kr


def test_default_is_off():
    assert KrRules(CFG).retarget_by_hops is False


# ── 도착 판정 ───────────────────────────────────────────────────────────
def test_on_stops_when_already_in_the_picked_lane():
    """hops == 0 = 그 차로에 있다. 여기서만 멈춘다."""
    kr = rig(True, '[2011, 5, -2]', (2011, 0, -9))
    assert kr._lm_retarget(None, None, 0.0) is False
    assert kr._retarget_why == 'arrived'


def test_on_does_not_stop_when_a_hop_is_still_left():
    """**이 건의 알맹이** — 의도가 같아도 한 칸 남았으면 멈추지 않는다.

    옛 축이면 `ot_target == pick` 이라 `same_target` 으로 끝났을 자리다.
    """
    kr = rig(True, '[2011, 5, -1]', (2011, 0, -1))
    kr._lm_retarget(None, None, 0.0)
    assert kr._retarget_why != 'arrived'
    assert kr._retarget_why != 'same_target'


def test_off_stops_on_intent_even_with_a_hop_left():
    """off = 이전 동작 (결함을 고정한다)."""
    kr = rig(False, '[2011, 5, -1]', (2011, 5, -1))
    assert kr._lm_retarget(None, None, 0.0) is False
    assert kr._retarget_why == 'same_target'


def test_on_is_immune_to_section_renumbering():
    """섹션이 바뀌어 키가 달라져도 hops 는 그대로 0 을 낸다."""
    kr = rig(True, '[2011, 5, -2]', (2011, 0, -1))
    kr._lm_retarget(None, None, 0.0)
    assert kr._retarget_why == 'arrived'


def test_unknown_pick_falls_through():
    """지도에 없는 차로면 hop 을 모른다 — 막지 않고 다음 게이트로 보낸다."""
    kr = rig(True, '[9999, 0, -1]', (2011, 0, -1))
    kr._lm_retarget(None, None, 0.0)
    assert kr._retarget_why != 'arrived'


# ── 최소 이득의 기준 ────────────────────────────────────────────────────
def test_margin_is_measured_against_the_current_lane_when_on():
    """자차 차로(24.4) 대비 목표(80.0) = +55.6 → 20 을 넘으므로 통과."""
    kr = rig(True, '[2011, 5, -1]', (2011, 0, -1))
    kr._lm_retarget(None, None, 0.0)
    assert kr._retarget_why != 'margin:+55.6'          # 통과했으므로 기록 안 됨
    assert not str(kr._retarget_why).startswith('margin')


def test_margin_still_blocks_a_pointless_switch_when_on():
    """자차 차로보다 나을 게 없으면 여전히 막는다 — 게이트가 죽은 게 아니다."""
    kr = rig(True, '[2011, 5, -4]', (2011, 0, -1))
    kr.last_lane_map = {'hops': HOPS, 'ego_lane': EGO,
                        'free_run': dict(FREE, **{'[2011, 5, -4]': 30.0})}
    assert kr._lm_retarget(None, None, 0.0) is False
    assert kr._retarget_why == 'margin:+5.6'


def test_off_measures_against_the_intent():
    """off 는 옛 기준(ot_target) 그대로 — 같은 차로면 이득 0 이라 막힌다."""
    kr = rig(False, '[2011, 5, -1]', (2011, 0, -1))
    assert kr._lm_retarget(None, None, 0.0) is False
    assert kr._retarget_why == 'margin:+0.0'


def test_margin_zero_disables_the_gate_either_way():
    for on in (True, False):
        kr = rig(on, '[2011, 5, -1]', (2011, 0, -1),
                 lane_switch_margin_m=0.0)
        kr._lm_retarget(None, None, 0.0)
        assert not str(kr._retarget_why).startswith('margin')


# ── 유지 시간은 여전히 흔들림을 막는다 ──────────────────────────────────
def test_hold_still_applies_under_the_hop_axis():
    kr = rig(True, '[2011, 5, -1]', (2011, 0, -1), retarget_min_hold_s=3.0)
    kr._retarget_lock = 2
    assert kr._lm_retarget(None, None, 0.0) is False
    assert kr._retarget_why == 'hold:1'


# ── '새 장애물' 요구 완화 ───────────────────────────────────────────────
def test_source_relaxes_the_new_blocker_requirement_only_when_on():
    src = (ROOT / 'team_code' / 'kr_rules.py').read_text(encoding='utf-8')
    i = src.index('pool = new if new else')
    assert 'corridor if self.retarget_by_hops else []' in src[i:i + 120]
    assert 'self._chain(corridor, pool[0][3])' in src
