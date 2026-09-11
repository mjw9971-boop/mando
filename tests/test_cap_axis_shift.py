"""`shift_cap`·`gap_fit` 을 상한 축으로 (control.cap_axis_shift_enable).

두 후보는 **상한형**이다 — "이 속도를 넘지 마라" 이지 "1틱 뒤에 이 속도가
되어라" 가 아니다:
    shift_cap  v ≤ √(a_lat_max / κ)      (진행 중인 전이의 횡가속 상한)
    gap_fit    v ≤ trans_m / shift_k_s   (고른 전이가 실제로 만들어질 속도)

그런데 후보표에 `'cap'` 표시 없이 등록돼 IDM `err/dt` 축(1/dt = 20배 증폭)에
실려 있었다. 바로 옆의 `curvature` · `lc_cap` · `lane_map` 은 전부 `'cap'` 이다.

실측 run_20260911_001250 t 136.2: v 4.44, 목표 3.74 — 초과가 **0.70 m/s** 뿐인데
`accel −4.00`(a_dec_max 바닥)이 나가 1.5 s 만에 **완전 정지**했다. 앞이 36 m
비어 있었다. 그 정지가 램프를 멈춰 앞선 시프트가 안 끝났고, 그래서 새 목표를
못 만들어 거리를 다 쓰는 연쇄로 갔다 (총 정지 60 s).

CLAUDE.md 「확정 사실」의 `_raw_accel` err/dt 항목이 말하는 바로 그 축 오배치다.
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
SRC = (ROOT / 'team_code' / 'kr_rules.py').read_text(encoding='utf-8')


def cfg(on):
    c = copy.deepcopy(CFG)
    c['control']['cap_axis_shift_enable'] = on
    return c


def test_default_is_off():
    """커밋본 기본값 = 이전 동작 (배치 오버레이에서 켠다)."""
    assert KrRules(CFG).cap_axis_shift is False


@pytest.mark.parametrize('on', [True, False])
def test_switch_is_read(on):
    assert KrRules(cfg(on)).cap_axis_shift is on


def test_both_candidates_share_the_axis_variable():
    """`shift_cap` 과 `gap_fit` 은 같은 축이다 — 한쪽만 옮기면 반만 고친 것이다."""
    i = SRC.index("add('shift_cap'")
    j = SRC.index("add('gap_fit'")
    assert "_ax = 'cap' if self.cap_axis_shift else 'stop'" in SRC[i - 1200:i]
    assert "add('shift_cap', cap, _ax)" in SRC[i - 40:i + 60]
    assert "add('gap_fit', self.gap_v_req, _ax)" in SRC[j - 40:j + 60]


def test_the_sibling_caps_are_still_on_the_cap_axis():
    """같은 표의 상한형 셋은 그대로 'cap' 이어야 한다 (회귀 고정)."""
    for name in ("add('curvature', cv, 'cap')",
                 "add('lc_cap', self._lc_speed_cap(planner), 'cap')",
                 "add('lane_map', float(lmv), 'cap')"):
        assert name in SRC


def test_stop_profiles_are_not_on_the_cap_axis():
    """정지 프로파일 계열은 err/dt 축이 맞다 — 같이 옮기면 안 된다."""
    for name in ("add('standoff', so)", "add('stopline_profile', prof)",
                 "add('stopline_hold', self._stopline_hold(planner, ego_speed))"):
        assert name in SRC


# ── 축이 명령에 어떻게 나타나는가 (control 쪽 계약) ────────────────────
def test_cap_axis_clamps_small_overshoot():
    """0.70 m/s 초과 — err/dt 축이면 −4.0 포화, 상한 축이면 a_cap_dec_max 안."""
    import vtd_adapter.control as ctl
    cls = next(v for k, v in vars(ctl).items()
               if k.endswith('LongitudinalController') and isinstance(v, type))
    lon = cls(CFG)
    err = 3.74 - 4.44                                  # 실측 t 136.2
    a_err_dt = max(err / 0.05, CFG['control']['a_dec_max'])
    a_cap = max(lon.kp_cap_dec * err, lon.a_cap_dec_max)
    assert a_err_dt == pytest.approx(CFG['control']['a_dec_max'])   # −4.0 포화
    assert a_cap > CFG['control']['a_dec_max']                      # 훨씬 완만
    assert a_cap >= lon.a_cap_dec_max
