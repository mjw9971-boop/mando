"""과속 / 실선 차선변경 / TTC 입력 시 clamp 되는지 (SPEC §3.5)."""
import pytest

pytestmark = pytest.mark.skip(reason='TODO: Shield 구현 후 활성화 (현재는 시그니처만)')


def test_speed_over_limit_is_clamped():
    """v_target > speed_limit - margin 이면 잘려야 한다."""


def test_lane_change_on_solid_mark_is_reverted():
    """lane_change_ok == False 인데 옆차로 path 면 현재 차로로 되돌린다."""


def test_center_line_crossing_is_reverted():
    """left_is_center 인데 좌측으로 벗어나는 path 는 교체."""


def test_ttc_below_emergency_forces_estop():
    """min TTC < ttc.emergency_s → v_target 0, accel a_emergency, state E_STOP."""


def test_stop_point_inside_crosswalk_is_pulled_back():
    """정지점이 횡단보도 안이면 구간 앞으로 당긴다."""


def test_lateral_offset_beyond_lane_edge_triggers_return():
    """|t_off| > width/2 - edge_margin 이면 복귀 우선."""
