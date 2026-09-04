"""standoff 바닥을 shift_latest_m 에서 분리한 것 (2026-09-05).

왜 나눴나 — `shift_latest_m` 은 소비처가 셋이다:
  ① `_standoff_profile`       : standoff 정지 프로파일의 바닥      ← 여기만 분리
  ② `_shift_speed_cap`        : 횡가속 상한 미리보기 창 `look`
  ③ `_try_overtake_inner`     : WAIT/PREEMPT 시간 예산 `t_left`
정지 거리만 조정하려고 ① 을 내리면 ②③ 이 함께 움직여, 검증된 PREEMPT 창과
shift_cap 발동 시점이 같이 바뀐다. 그래서 ① 전용 상수 `standoff_floor_m` 을
두고 기본값을 `shift_latest_m` 과 같게(25.0) 잡아 **분리 자체는 무변화**로 둔다.
"""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'team_code'))

from conftest import PARAMS_YAML                                   # noqa: E402
from test_avoid import make                                        # noqa: E402
from vtd_adapter.config import load_params_yaml                    # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
A_STOP = CFG['speed']['stop_profile_a']


def test_standoff_floor_key_exists():
    assert 'standoff_floor_m' in OT


def test_shift_latest_m_untouched():
    """② ③ 이 읽는 값은 건드리지 않는다 — 분리의 전제."""
    assert OT['shift_latest_m'] == 25.0


def test_floor_not_above_shift_latest():
    """바닥은 shift_latest_m 을 넘지 않는다 (넘으면 PREEMPT 예산보다 늦게 서서
    t_left 가 음수인 채로 geom 이 통과해 버린다)."""
    assert OT['standoff_floor_m'] <= OT['shift_latest_m']


def test_floor_meets_geom_gate_need_at_standstill():
    """정지 상태 geom 게이트 need = transition_m + shift_ahead_m + margin.

    바닥이 이보다 낮으면, 서고 나서 시프트를 만들려 해도 ⑥ 기하 완성 게이트가
    매 틱 기각한다 (BREAKOUT L3 완화에 의존하게 되는데, 그 사다리는 신호 주기에
    리셋된다 — 2026-09-04 실측 02_직진3).
    """
    need = OT['transition_m'] + OT['shift_ahead_m'] + OT['shift_geom_margin_m']
    assert need == pytest.approx(19.0)
    assert OT['standoff_floor_m'] >= need


def test_profile_uses_floor_not_shift_latest():
    """_standoff_profile 이 읽는 것은 standoff_floor_m 이다."""
    kr, _p = make()
    kr.standoff_floor_m = 20.0
    kr.shift_latest_m = 25.0                    # 이 값은 프로파일에 영향이 없어야
    kr.wait_target_d = 30.0
    assert kr._standoff_profile(0.0) == pytest.approx(
        math.sqrt(2.0 * A_STOP * (30.0 - 20.0)))


def test_profile_floor_default_equals_shift_latest():
    """기본값에서는 분리 전과 완전히 같은 값을 낸다."""
    kr, _p = make()
    kr.wait_target_d = 40.0
    assert kr._standoff_profile(0.0) == pytest.approx(
        math.sqrt(2.0 * A_STOP * (40.0 - OT['shift_latest_m'])))
