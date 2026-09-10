"""(3) 목표 차로 점유 판정을 차로 지도로 — overtake.side_clear_by_map_enable.

옛 `_side_is_clear` 는 축이 틀렸다:
    near = {target} ∪ successors(target) ∪ **predecessors(target)**
    반경  = **자차 중심** clear_radius_m (30 m)
그래서 (a) 목표 차로 **뒤**의 차가 기각하고, (b) 목표 차로 **앞 50 m** 는
안 보이고, (c) 정지 차량 무리 안에서는 반경이 늘 차 있어 항상 occupied 다.

차로 지도는 정반대 축이다 — 앞 `lane_map_ahead_m`(80) 을 차로별로 잰다.
판정은 `free_run[target] > 램프 + shift_ahead_m`, 후방은 `rear_clear_m` 안에
**자차보다 빠른** 차가 있는지만 따로 본다 (채점 항목 14).
"""
import copy
import pathlib
import sys

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
TGT = (1, 0, -2)


def kr(map_on=True, side_map=True, rear_m=30.0, free=None, ramp=15.0):
    c = copy.deepcopy(CFG)
    c['avoid_map']['lane_map_avoid_enable'] = map_on
    c['overtake']['side_clear_by_map_enable'] = side_map
    c['overtake']['rear_clear_m'] = rear_m
    k = KrRules(c)
    k.last_lane_map = {'free_run': dict(free or {str(list(TGT)): 80.0})}
    k.last_lane_plan = {'ramp_m': ramp}
    return k


class _Ego:
    id = 1
    speed = 0.0


class _W:
    def __init__(self, actors=()):
        self._a = list(actors)

    def get_actors(self):
        return self._a


class _Ap:
    def __init__(self, actors=()):
        self._vehicle = _Ego()
        self._world = _W(actors)


def test_map_passes_when_the_lane_runs_free_ahead():
    """앞 80 m 가 뚫려 있으면 통과 — 옛 판정은 자차 반경만 봤다."""
    k = kr()
    assert k._side_clear_by_map(None, None, _Ap(), TGT, 1) is True


def test_map_rejects_when_free_run_is_shorter_than_the_ramp():
    """램프 + shift_ahead_m 가 안 들어가면 기각."""
    k = kr(free={str(list(TGT)): 15.0}, ramp=15.0)      # 15 <= 15 + 5
    assert k._side_clear_by_map(None, None, _Ap(), TGT, 1) is False


def test_returns_none_when_the_map_is_off():
    """지도가 없으면 판단하지 않는다 — 호출처가 옛 판정으로 간다."""
    assert kr(map_on=False)._side_clear_by_map(None, None, _Ap(), TGT, 1) is None


def test_returns_none_for_a_lane_outside_the_map():
    k = kr(free={'[9, 9, 9]': 80.0})
    assert k._side_clear_by_map(None, None, _Ap(), TGT, 1) is None


def test_rear_window_ignores_stopped_traffic():
    """정지 차량은 후방 위험이 아니다 — 무리 안에서 영영 못 나가면 안 된다."""
    k = kr(rear_m=0.0)                                   # 0 = 후방 안 봄
    assert k._rear_clear(None, None, _Ap(), TGT) is True


def test_switch_default_is_off():
    assert CFG['overtake'].get('side_clear_by_map_enable') is False
    assert CFG['overtake'].get('rear_clear_m') == 30.0
