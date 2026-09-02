"""
finish_xy 자동화 — CSV 마지막 행 → route.pkl → 제어·채점 (2026-09-03).

배경: scoring.finish_xy 를 사람이 CSV 마지막 행에서 옮겨 적어야 했다 — 대회 당일
오타 위험. 미설정이면 kr_rules 가 route_total 폴백으로 돌아 종료선 정확도가
떨어진다.

계약:
  · build_route 는 pkl 에 finish_xy(CSV 마지막 행 **원본** 좌표)를 넣는다
  · 마지막 경유점 매칭 차로의 잔여(꼬리)가 finish_tail_m 미만이면 successor 를
    따라 연장해 plan_stop_s 클립을 없앤다. finish_tail_m=0 이면 기존 그대로
  · kr_rules._resolve_stop_s 우선순위: params(scoring.finish_xy) →
    route['finish_xy'] → 폴백. route_end.finish_xy_from_route_enable=false 면
    route 값 무시 = 기존 동작
  · score.resolve_finish_xy 도 같은 우선순위 (제어·채점이 같은 값을 본다)
"""
import copy
import math
import pathlib
import pickle
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.lanegraph import LaneGraph

import build_route as br
from score import resolve_finish_xy

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules, plan_stop_s   # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
# plan_stop_s 가 요구하는 종료선 뒤 꼬리 [m] (kr_rules.plan_stop_s 와 같은 합)
NEED = (CFG['scoring']['finish_clearance_m'] + CFG['speed']['stop_gap_route_end_m']
        + CFG['vehicle']['wheelbase'] + CFG['vehicle']['front_overhang_m']
        + CFG['batch']['end_slack_m'])
FIXTURE_CSV = ROOT / 'tests' / 'fixtures' / 'waypoints.csv'
FIXTURE_PKL = ROOT / 'tests' / 'fixtures' / 'route.pkl'


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(ROOT / 'data' / 'lane_graph.pkl'))


@pytest.fixture(scope='module')
def wps():
    rows = br.read_waypoints_csv(str(FIXTURE_CSV))
    return [(x, y) for _seq, x, y in rows]


def _start_yaw(wps):
    """build_route.main 의 자동 추정과 같은 식 (seq1→첫 2 m 이상 떨어진 점)."""
    for j in range(1, len(wps)):
        if math.hypot(wps[j][0] - wps[0][0], wps[j][1] - wps[0][1]) >= 2.0:
            return math.atan2(wps[j][1] - wps[0][1], wps[j][0] - wps[0][0])
    return 0.0


def _build(lg, wps, tail):
    return br.build_route(lg, wps, start_yaw=_start_yaw(wps),
                          junction_segs=br.junction_segments(len(wps)),
                          finish_tail_m=tail)


# ── build_route: finish_xy 기록 ─────────────────────────────────────────
def test_pkl_finish_xy_equals_csv_last_row(lg, wps):
    rt = _build(lg, wps, tail=0.0)
    assert rt['finish_xy'] == [wps[-1][0], wps[-1][1]]
    # 원본 waypoint 도 그대로 보존돼 있다 (리샘플 없음)
    assert rt['waypoints'][-1] == wps[-1]


# ── build_route: 꼬리 연장 ──────────────────────────────────────────────
def test_tail_off_reproduces_fixture(lg, wps):
    """finish_tail_m=0 (스위치 off) = 기존 픽스처와 동일한 경로."""
    rt = _build(lg, wps, tail=0.0)
    with open(FIXTURE_PKL, 'rb') as f:
        base = pickle.load(f)
    assert rt['lanes'] == base['lanes']
    assert rt['total_length'] == pytest.approx(base['total_length'], abs=0.01)


def test_tail_extension_removes_clip(lg, wps, capsys):
    """픽스처 경로는 꼬리 0.2 m — 연장하면 plan_stop_s 클립이 사라진다."""
    rt0 = _build(lg, wps, tail=0.0)
    finish_s0 = rt0['waypoint_s'][-1]
    tail0 = rt0['total_length'] - finish_s0
    assert tail0 < NEED                                  # 전제: 원래 부족 (0.2 m)
    _s, clipped = plan_stop_s(CFG, rt0['total_length'], finish_s0)
    assert clipped

    rt = _build(lg, wps, tail=12.0)
    assert '경로 꼬리 연장' in capsys.readouterr().out
    finish_s = rt['waypoint_s'][-1]
    assert finish_s == pytest.approx(finish_s0)          # 종료선 위치는 불변
    assert rt['finish_xy'] == rt0['finish_xy']
    assert rt['total_length'] - finish_s >= 12.0
    stop_s, clipped = plan_stop_s(CFG, rt['total_length'], finish_s)
    assert not clipped
    # 정지 시 뒷축 = finish_s + clearance — 종료선을 넘는다
    rear = stop_s - CFG['speed']['stop_gap_route_end_m'] \
        - CFG['vehicle']['wheelbase'] - CFG['vehicle']['front_overhang_m']
    assert rear == pytest.approx(finish_s + CFG['scoring']['finish_clearance_m'])


def test_tail_extension_keeps_events(lg, wps):
    """연장 차로에서 가짜 turn/lane_change 이벤트가 생기지 않는다."""
    ev0 = [(e['kind'], round(e['s'], 3)) for e in _build(lg, wps, tail=0.0)['events']]
    ev1 = [(e['kind'], round(e['s'], 3)) for e in _build(lg, wps, tail=12.0)['events']]
    assert ev0 == ev1


# ── kr_rules: params → route → 폴백 ─────────────────────────────────────
class _FakeLg:
    def project(self, key, x, y):
        return float(x), 0.0, 0.0, 0        # finish_s = x 좌표로 매핑


class _FakePlanner:
    def __init__(self, total=800.0, finish_xy=None):
        self.route = {'total_length': total, 'lanes': [(1, 0, -1)], 'cum_s': [0.0]}
        if finish_xy is not None:
            self.route['finish_xy'] = finish_xy
        self.lg = _FakeLg()


def test_kr_rules_uses_route_finish_xy(capsys):
    cfg = copy.deepcopy(CFG)                # scoring.finish_xy: null
    kr = KrRules(cfg)
    stop_s = kr._resolve_stop_s(_FakePlanner(finish_xy=[760.0, 0.0]))
    assert stop_s == pytest.approx(plan_stop_s(cfg, 800.0, 760.0)[0])
    assert 'route.pkl' in capsys.readouterr().out


def test_kr_rules_params_wins_over_route(capsys):
    cfg = copy.deepcopy(CFG)
    cfg['scoring']['finish_xy'] = [700.0, 0.0]
    kr = KrRules(cfg)
    stop_s = kr._resolve_stop_s(_FakePlanner(finish_xy=[760.0, 0.0]))
    assert stop_s == pytest.approx(plan_stop_s(cfg, 800.0, 700.0)[0])
    assert 'params' in capsys.readouterr().out


def test_kr_rules_switch_off_ignores_route(capsys):
    cfg = copy.deepcopy(CFG)
    cfg['route_end']['finish_xy_from_route_enable'] = False
    kr = KrRules(cfg)
    stop_s = kr._resolve_stop_s(_FakePlanner(total=800.0, finish_xy=[760.0, 0.0]))
    assert stop_s == pytest.approx(800.0)   # 기존 route_total 폴백
    assert '미설정' in capsys.readouterr().out


# ── score: 같은 우선순위 ────────────────────────────────────────────────
def test_score_resolve_priority():
    cfg = copy.deepcopy(CFG)
    route = {'finish_xy': [760.0, 0.0]}
    assert resolve_finish_xy(cfg, route) == [760.0, 0.0]          # route 자동
    cfg['scoring']['finish_xy'] = [700.0, 0.0]
    assert resolve_finish_xy(cfg, route) == [700.0, 0.0]          # params 우선
    cfg['scoring']['finish_xy'] = None
    cfg['route_end']['finish_xy_from_route_enable'] = False
    assert resolve_finish_xy(cfg, route) is None                  # 스위치 off
    cfg['route_end']['finish_xy_from_route_enable'] = True
    assert resolve_finish_xy(cfg, None) is None                   # route 없음
    assert resolve_finish_xy(cfg, {}) is None                     # 구버전 pkl
