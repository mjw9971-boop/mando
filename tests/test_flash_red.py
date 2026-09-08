"""B3 적색 점멸 일시정지 (채점 항목 9, 중대 −6) — speed.signal_flash_stop_enable.

규정: **범퍼 기준 정지선 2 m 이내에서 0.5 s 이상 정지 1회**, 그 뒤 통과.
무정차 통과는 중대다.

확정 사실 (재조사 금지):
  · 9910 state 6 = 점멸. 이 맵의 점멸 컨트롤러는 117 하나뿐이고 **적색**이다
    (배포본 7개 전부 blink=117, Signal 373 type 1000020=적, 실기 육안 09-01).
    황색 점멸은 이 맵에 없다.
  · 6 은 **지속 플래그**다 — 램프 on/off 위상이 9910 에 안 실린다 (629틱
    35.7 s 동안 6 고정, 전이 0회). 그래서 주기 토글을 찾지 않는다.

설계:
  · state 6 → `TrafficLightState.FlashRed`. PDM 에는 "녹색이 아닌 것" 이라
    적신호 IDM 이 걸린다. **세우는 것은 kr 의 ④′ 정지 프로파일**이다 —
    PDM 의 적신호 IDM 은 차간모형이라 정지 컨트롤러가 아니고, 그것만으로는
    점멸 정지선을 5.4 m/s 로 지난다 (replay 실측 2026-09-09).
  · 유지시간을 채우면 래치가 붙어 `_stop_target_raw` 가 None 이 되고 같은 틱에
    `signal_release` 가 PDM 적신호 IDM 도 건너뛴다 = 재출발.
  · 판정 임계는 **채점기와 같은 출처**(scoring.stop_ok_m / score.stop_speed_mps).
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
from test_avoid import Ap, Planner, HZ                             # noqa: E402
from vtd_adapter.carla_types import TrafficLightState              # noqa: E402
from vtd_adapter.route import LIGHT_STATE_MAP, LIGHT_STATE_MAP_FLASH  # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
OK_M = float(CFG['scoring']['stop_ok_m'])
STOP_V = float(CFG['score']['stop_speed_mps'])
FRONT = CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']


def on_cfg(**over):
    c = copy.deepcopy(CFG)
    c['speed']['signal_flash_stop_enable'] = True
    c['speed'].update(over)
    return c


def off_cfg():
    c = copy.deepcopy(CFG)
    c['speed']['signal_flash_stop_enable'] = False
    return c


class TL:
    def __init__(self, tl_id, state):
        self.id = tl_id
        self.state = state


def rig(cfg, front_m=-1.0, state=TrafficLightState.FlashRed, tl_id=116):
    """앞범퍼가 정지선에서 front_m (음수 = 선 앞) 인 배치."""
    d_line = FRONT - front_m                 # 뒷축 기준 남은 거리
    p = Planner(d_tl=d_line)
    p.next_traffic_lights = [TL(tl_id, state)] * len(p.route_s)
    kr = KrRules(cfg)
    kr._sl_all = []
    ap = Ap(p)
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    kr._ap = ap
    return kr, p, ap


# ── 매핑 ──────────────────────────────────────────────────────────────────
def test_map_default_is_previous_behaviour():
    assert LIGHT_STATE_MAP[6] is TrafficLightState.Green


def test_map_flash_differs_only_at_6():
    assert LIGHT_STATE_MAP_FLASH[6] is TrafficLightState.FlashRed
    for k in LIGHT_STATE_MAP:
        if k != 6:
            assert LIGHT_STATE_MAP_FLASH[k] is LIGHT_STATE_MAP[k]


def test_flashred_is_not_green_for_pdm():
    """PDM 원문 비교문이 전부 거짓이어야 한다 — 그래야 적신호 IDM 이 걸린다."""
    fr = TrafficLightState.FlashRed
    for other in (TrafficLightState.Red, TrafficLightState.Yellow,
                  TrafficLightState.Green, TrafficLightState.Off,
                  TrafficLightState.Unknown):
        assert fr != other


# ── 스위치 off = 이전 동작 ────────────────────────────────────────────────
def test_switch_off_makes_everything_inert():
    kr, p, _ap = rig(off_cfg())
    kr._flash_tick(p, 0.0)
    assert kr.last_flash is None
    assert kr._flash_release(p) is False


def test_switch_off_never_stops_for_flash():
    """off 면 6 이 녹색이라 애초에 FlashRed 가 안 온다 = 정지 후보 없음."""
    kr, p, ap = rig(off_cfg(), state=TrafficLightState.Green)
    assert kr._stop_target_raw(p, ap) is None


# ── 정지 → 유지 → 재출발 ─────────────────────────────────────────────────
def test_stop_candidate_before_hold():
    """유지시간을 채우기 전에는 적색과 같은 축(stop_profile_a)으로 세운다."""
    kr, p, ap = rig(on_cfg())
    got = kr._stop_target_raw(p, ap)
    assert got is not None
    assert got[1] == pytest.approx(kr.stop_profile_a)


def test_hold_then_release():
    """2 m 안에서 정지 0.8 s → 래치 → 정지 후보 사라짐 + signal_release 참."""
    kr, p, ap = rig(on_cfg(), front_m=-1.0)
    need = int(round(0.8 * HZ))
    for _ in range(need):
        assert kr._flash_release(p) is False
        kr._flash_tick(p, 0.0)
    assert kr._flash_release(p) is True
    assert kr._stop_target_raw(p, ap) is None
    assert kr.signal_release(ap) is True


def test_moving_does_not_accumulate_hold():
    kr, p, _ap = rig(on_cfg())
    for _ in range(int(round(3.0 * HZ))):
        kr._flash_tick(p, STOP_V + 0.5)
    assert kr._flash_release(p) is False


def test_stopping_past_the_line_does_not_count():
    """선을 넘어 선 것은 일시정지가 아니다 (채점기와 같은 규약)."""
    kr, p, _ap = rig(on_cfg(), front_m=+1.0)
    for _ in range(int(round(3.0 * HZ))):
        kr._flash_tick(p, 0.0)
    assert kr._flash_release(p) is False


def test_stopping_far_from_the_line_does_not_count():
    kr, p, _ap = rig(on_cfg(), front_m=-(OK_M + 1.0))
    for _ in range(int(round(3.0 * HZ))):
        kr._flash_tick(p, 0.0)
    assert kr._flash_release(p) is False


def test_threshold_is_the_scorer_source():
    """임계를 새로 만들지 않았다 — 채점기와 같은 값을 읽는다."""
    kr, _p, _ap = rig(on_cfg())
    assert kr.flash_ok_m == pytest.approx(OK_M)
    assert kr.flash_stop_v == pytest.approx(STOP_V)


# ── 래치 ──────────────────────────────────────────────────────────────────
def test_latch_survives_moving_off_the_line():
    """허가가 나면 그 정지선을 지날 때까지 유지 — 같은 선에 다시 안 선다."""
    kr, p, ap = rig(on_cfg(), front_m=-0.5)
    for _ in range(int(round(1.0 * HZ))):
        kr._flash_tick(p, 0.0)
    assert kr._flash_release(p) is True
    # 다시 굴러가기 시작해도 (속도 회복) 허가는 유지된다
    for fm in (-0.2, +0.5, +1.5):
        kr2_d = FRONT - fm
        p.distances_to_next_traffic_lights[:] = kr2_d
        kr._flash_tick(p, 3.0)
        assert kr._flash_release(p) is True
        assert kr._stop_target_raw(p, ap) is None


def test_latch_does_not_carry_to_the_next_signal():
    kr, p, ap = rig(on_cfg(), front_m=-0.5)
    for _ in range(int(round(1.0 * HZ))):
        kr._flash_tick(p, 0.0)
    assert kr._flash_release(p) is True
    p.next_traffic_lights = [TL(999, TrafficLightState.FlashRed)] * len(p.route_s)
    kr._flash_tick(p, 3.0)
    assert kr._flash_release(p) is False
    assert kr._stop_target_raw(p, ap) is not None      # 새 점멸 = 다시 선다


def test_non_flash_signal_is_untouched():
    """적색·황색·녹색 해석은 그대로다."""
    kr, p, ap = rig(on_cfg(), state=TrafficLightState.Red)
    assert kr._stop_target_raw(p, ap) is not None
    kr2, p2, ap2 = rig(on_cfg(), state=TrafficLightState.Green)
    assert kr2._stop_target_raw(p2, ap2) is None
