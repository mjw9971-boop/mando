"""[3] 회전 방향 차로 선호 (`route.turn_lane_bias_m`) + 소멸 연결로 벌점.

지도에 **노면 화살표**가 있다 (`lanes[k]['arrows']`, 비교차로 1773개 중 696개).
그것으로 보면 현재 경로의 회전 차로 위반이 명확하다 — 실측(경로 34건, 2026-09-09)
**19건**:

    실전주행_교통류_07 s477.8  진입 (146,0,1) 화살표 **L**(좌회전 전용)에서 우회전
    실전주행_교통류_11·15·16·08 진입 (2814,0,3) 화살표 **S**(직진)에서 좌회전
    실전주행_교통류_02·14      진입 (2813,0,2) 화살표 **S** 에서 우회전

이 파일이 고정하는 것:
  · 기본 0 = 이전 동작 (비용에 아무것도 안 더한다).
  · 벌점은 **회전 방향 쪽에 남은 차로 수**에 비례한다.
  · 연결로가 직진이면 안 붙는다 (회전만 본다).
  · 소멸 연결로 벌점(taper_penalty)과는 **다른 축**이다 — 실측에서 taper 를
    켜도 위반 수는 19건 그대로였다.
"""
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

from conftest import PARAMS_YAML                                    # noqa: E402
from vtd_adapter.config import load_params_yaml                     # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                         # noqa: E402

import build_route as br                                            # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
GRAPH = ROOT / 'data' / 'lane_graph.pkl'


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH), cfg=CFG)


def test_default_is_off():
    """기본 0 — 켜는 것은 사람이 정한다 (표는 docs/NIGHT_2026-09-09.md [3])."""
    assert CFG['route']['turn_lane_bias_m'] == 0.0


def test_connector_turn_reads_geometry_not_arrows(lg):
    """연결로에는 화살표가 없다 — 회전 방향은 시작·끝 헤딩 차이로 본다."""
    br._TURN_KIND = {}
    assert lg.lanes[(1154, 0, -2)]['arrows'] == []
    assert br.connector_turn(lg, (1154, 0, -2)) == 'right'
    assert br.connector_turn(lg, (146, 0, 1)) is None      # 교차로 차로가 아니다


def test_lanes_on_side_counts_neighbours(lg):
    br._SIDE_N = {}
    assert br.lanes_on_side(lg, (146, 0, 1), 'right') == 1
    assert br.lanes_on_side(lg, (146, 0, 2), 'right') == 0
    assert br.lanes_on_side(lg, (146, 1, 1), 'right') == 0


def test_the_measured_violation_is_a_left_only_lane(lg):
    """07 s477.8 — 진입 차로 화살표가 **L**(좌회전 전용)인데 우회전한다.

    지도가 직접 말해 주는 사실이라 고정해 둔다. 다만 이 사례는 bias 로 못
    고친다 (아래 테스트 참조).
    """
    arrows = {t for _, t in lg.lanes[(146, 0, 1)]['arrows']}
    assert arrows == {'L'}
    assert {t for _, t in lg.lanes[(146, 0, 2)]['arrows']} == {'SR'}


def test_the_alternative_lane_is_physically_unreachable(lg):
    """(146,0,2) 는 s 27.5 에서야 차폭이 되고 연결로까지 7 m 뿐이다.

    LC_MIN_CORRIDOR_M(25) 부족분이 18 m 라 bias 로 이기게 만들면 **불가능한
    경로**가 나온다. 실측: bias 400 에서야 뒤집힌다.
    """
    import numpy as np
    w = np.asarray(lg.lanes[(146, 0, 2)]['width'])
    s = np.asarray(lg.lanes[(146, 0, 2)]['s'])
    i = int(np.argmax(w >= CFG['vehicle']['width']))
    assert s[i] > 25.0
    assert s[-1] - s[i] < br.LC_MIN_CORRIDOR_M / 2.0


# ── 소멸 연결로 벌점 (같은 [3] 이지만 다른 축) ────────────────────────────
def test_taper_penalty_is_on_and_the_target_lane_vanishes(lg):
    """(1821,0,-2) 는 끝 폭 0.00 m — 차로가 완전히 사라진다."""
    assert CFG['route']['taper_penalty_enable'] is True
    assert lg.width_at((1821, 0, -2), lg.length((1821, 0, -2))) < 0.1
    assert br.is_taper_lane(lg, (1821, 0, -2), CFG['vehicle']['width'])
    assert not br.is_taper_lane(lg, (1821, 0, -3), CFG['vehicle']['width'])


def test_taper_alternative_needs_no_lane_change(lg):
    """대안이 successor 로 이어져야 벌점이 실제로 먹힌다 (차선변경 축과 분리)."""
    assert (1821, 0, -3) in lg.lanes
    assert lg.neighbor((1821, 0, -2), 'right') == (1821, 0, -3)
