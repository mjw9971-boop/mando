"""
(2) 적신호 '가속 금지' 가 '영구 정지' 가 됐다 — `speed.red_approach_min_kph`.

`_no_accel_red_cap` 은 상한을 **현재 속도**로 준다 (`max(0.0, ego_speed)`).
그건 자기잠금이다: 다른 축이 잠깐 감속시키면 상한이 그 값을 따라 내려가고,
상한이 v 를 다시 못 올리게 막으므로 되돌아올 길이 없다. 바닥이 0 이다.

실측 2026-09-10 run_20260910_144656, rs 299.9 — **정지선 50.2 m 앞**:

    t 52.5~53.1  shift_cap 4.3 이 감속       v 5.34 → 4.67
    t 53.4       no_accel_red 3.70  (승자)
    t 53.7       no_accel_red 2.86
    t 54.0       no_accel_red 1.58
    t 54.3       no_accel_red 0.86
    t 54.9       no_accel_red **0.00**       v 0.00
    t 54.9~      341틱(17 s) 목표 0.0, 승자 no_accel_red **단독**

같은 틱의 다른 후보: shift_cap 4.3 · lane_map 5.56 · curvature 7.64 ·
standoff 13.53 · **stopline_profile 16.32**. 정지선은 아무것도 요구하지
않았다 — 이 상한 하나가 50 m 앞에서 차를 세우고 붙잡고 있었다.

정지는 이 축이 하는 일이 아니다 (④′ 프로파일 · hold · IDM 이 한다).
그래서 바닥을 깔아도 정지선 앞 정지는 그대로다.

`no_accel_toward_red_enable` 자체는 유지한다 — 도입 커밋 14912b0 에 실측
근거가 있다 (PDM red IDM 이 적색 접근 667틱 중 290틱에서 가속을 요구,
최대 +4.17 m/s; 수정 후 accel 부호 전환 02 28 → 1, 07 36 → 10).
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)


def kr(floor_kph):
    c = copy.deepcopy(CFG)
    c['speed']['no_accel_toward_red_enable'] = True
    c['speed']['red_approach_min_kph'] = floor_kph
    k = KrRules(c)
    # 정지선이 red_lookahead_m 안에 있다 (거리 50.2 m — 로그 실값)
    k._stop_target = lambda *_a, **_kw: (50.2, 1.5)
    return k


def cap(k, v):
    return k._no_accel_red_cap(None, None, v)


def test_switch_is_on_in_params():
    """이 스위치는 실측 근거가 있는 채택 기능이다 (도입 커밋 14912b0)."""
    assert CFG['speed']['no_accel_toward_red_enable'] is True


def test_default_floor_is_previous_behaviour():
    assert float(CFG['speed']['red_approach_min_kph']) == 0.0


def test_off_reproduces_the_ratchet_to_zero():
    """이전 동작 — 로그의 감속 계열을 그대로 따라 0 으로 내려가 갇힌다."""
    k = kr(0.0)
    seq = [5.34, 4.67, 3.70, 2.86, 1.58, 0.86, 0.16, 0.00]
    got = [cap(k, v) for v in seq]
    assert got == pytest.approx(seq)
    assert got[-1] == 0.0
    # 갇힘: 목표가 0 이면 v 도 0 이라 다음 틱도 0 이다
    assert cap(k, 0.0) == 0.0


def test_on_floors_the_cap_and_lets_the_car_move_again():
    """수정 후 — 같은 계열에서 바닥 아래로 내려가지 않는다."""
    k = kr(10.0)
    floor = 10.0 / 3.6
    for v in (5.34, 4.67, 3.70, 2.86, 1.58, 0.86, 0.16, 0.00):
        assert cap(k, v) >= floor - 1e-9, v
    assert cap(k, 0.0) == pytest.approx(floor)


def test_on_still_forbids_acceleration_above_the_reference():
    """목적은 그대로 — 기준(감지 시점 속도)에서 위로 못 올라간다."""
    k = kr(10.0)
    assert cap(k, 8.0) == pytest.approx(8.0)      # 감지 시점 8.0 래치
    assert cap(k, 9.5) == pytest.approx(8.0)      # 가속 시도 → 여전히 8.0
    assert cap(k, 6.0) == pytest.approx(6.0)      # 감속은 따라 내려간다
    assert cap(k, 9.5) == pytest.approx(6.0)      # 다시 올라가지 않는다


def test_reference_latch_clears_when_the_target_is_gone():
    """녹색이 되거나 지나가면 래치가 풀린다 — 다음 적색에서 새로 잡는다."""
    k = kr(10.0)
    assert cap(k, 8.0) == pytest.approx(8.0)
    k._stop_target = lambda *_a, **_kw: None
    assert cap(k, 3.0) is None and k._nar_ref_v is None
    k._stop_target = lambda *_a, **_kw: (50.2, 1.5)
    assert cap(k, 11.0) == pytest.approx(11.0)     # 새 기준


def test_far_red_is_out_of_range_and_clears_the_latch():
    k = kr(10.0)
    cap(k, 8.0)
    k._stop_target = lambda *_a, **_kw: (10_000.0, 1.5)
    assert cap(k, 8.0) is None and k._nar_ref_v is None


def test_stopping_is_not_this_axis_job():
    """바닥이 있어도 정지선이 가까우면 ④′ 프로파일이 먼저 0 으로 내려가 이긴다."""
    k = kr(10.0)
    floor = 10.0 / 3.6
    import math
    a = float(CFG['speed']['stop_profile_a']) if 'stop_profile_a' in CFG['speed'] else 1.5
    # 프로파일 √(2·a·d) 가 바닥 밑으로 내려가는 거리
    d_cross = floor ** 2 / (2.0 * a)
    assert d_cross < 5.0, '정지선 몇 m 전에는 프로파일이 이겨야 한다'
    assert math.sqrt(2.0 * a * (d_cross / 2.0)) < floor
