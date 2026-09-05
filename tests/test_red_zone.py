"""붉은 노면 = 감속(보호) 구간 — lane_graph red_spans 와 채점기 독립 판정.

대회 규칙상 **붉은 노면만** 보호구역(항목 2)이고, 붉지 않은 도로는 30 표시가
있어도 도로 기본 제한(50)이 적용된다. 붉은 포장은 xodr 에 없고 osgb 텍스처로만
있어 월드 정점(data/red_surface_verts.json)을 뽑아 두었다.

여기서 지키는 계약:
  1. red_spans 는 **차로 단위**다 — road 2312 는 dir −1 만 붉다. road 단위로
     묶으면 통행방향 하나가 통째로 잘못 감속한다.
  2. school_zone bool 은 bool(red_spans) 로 남는다 (하위호환).
  3. 30 노면표시는 speed.roadmark_30_as_limit 이 false 면 제한속도가 아니다.
     RM_517_50 같은 다른 값은 그대로다.
  4. 채점기는 lane_graph 를 **직접** 읽는다 — 제어기 로그의 world.speed_limit /
     school_zone 을 판정에 쓰면 제어기가 구간을 놓칠 때 같이 놓친다.
  5. 항목 1 과 항목 2 의 severity 임계가 다르다.
"""
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML, mk_tick
from vtd_adapter.config import load_params_yaml
from vtd_adapter.lanegraph import LaneGraph

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import score as SC                                             # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
GRAPH = ROOT / 'data' / 'lane_graph.pkl'


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


# ── params ──────────────────────────────────────────────────────────────
def test_params_present():
    rz = CFG['red_zone']
    assert rz['verts_json'] == 'data/red_surface_verts.json'
    assert float(rz['limit_kph']) == 30.0
    assert float(rz['exit_margin_m']) == 2.0
    assert rz['roadmark_30_as_limit'] is False
    # 이탈 여유는 제어기 몫이다 — speed.* 로 흩어져 있으면 안 된다
    assert 'roadmark_30_as_limit' not in CFG['speed']


def test_scorer_uses_exact_span_boundary(lg):
    """채점기는 구간을 넓히지 않는다 — 실제 경계(마진 0)로 판정한다.

    exit_margin_m 은 제어기가 캡 해제를 늦추는 값이다. 채점기가 같이 넓히면
    규칙보다 엄해져 우리 리포트가 실채점과 어긋난다.
    """
    mar = float(CFG['red_zone']['exit_margin_m'])
    # 구간 끝 뒤에 여유가 남는 차로를 고른다 (차로 끝에서 끝나는 구간이 다수다)
    pick = None
    for kk, r in sorted(lg.lanes.items()):
        for a, b in r['red_spans']:
            if b + mar + 1.0 < r['length']:
                pick = (kk, b); break
        if pick:
            break
    if pick is None:
        pytest.skip('구간 끝 뒤에 여유가 남는 차로가 없다')
    k, s1 = pick
    out_s = s1 + 0.5 * mar          # 구간 끝을 살짝 지난 지점
    x, y, _z, h = lg.point_at(k, out_s)
    ctx = SC.speed_context([mk_tick(x=float(x), y=float(y), yaw=float(h))], lg, CFG)
    assert ctx[0]['red'] is False


def test_scoring_thresholds_split_by_item():
    """항목 1(일반)과 항목 2(보호구역)의 임계가 분리돼 있다."""
    sc = CFG['scoring']
    assert (float(sc['speed_allow_kph']), float(sc['speed_major_kph'])) == (1.0, 20.0)
    assert (float(sc['speed_school_allow_kph']), float(sc['speed_school_major_kph'])) == (1.0, 5.0)


def test_time_limit_is_13_minutes():
    assert float(CFG['score']['time_limit_s']) == 780.0


# ── lane_graph ──────────────────────────────────────────────────────────
def test_red_spans_field_exists(lg):
    assert all('red_spans' in r for r in lg.lanes.values())


def test_school_zone_is_derived_from_red_spans(lg):
    for r in lg.lanes.values():
        assert r['school_zone'] == bool(r['red_spans'])


def test_red_spans_are_within_lane(lg):
    for k, r in lg.lanes.items():
        for s0, s1 in r['red_spans']:
            assert 0.0 <= s0 < s1 <= r['length'] + 1e-6, (k, s0, s1, r['length'])


def test_road_2312_is_one_direction_only(lg):
    """편도만 붉은 실측 — dir −1 (lane 2·3·4) 만 붉고 반대는 아니다."""
    plus = [k for k in lg.lanes if k[0] == 2312 and lg.lanes[k]['dir'] == 1]
    minus_red = [k for k in lg.lanes if k[0] == 2312 and lg.lanes[k]['dir'] == -1
                 and lg.lanes[k]['red_spans']]
    assert plus, '도로 2312 에 dir=+1 차로가 있어야 이 테스트가 유효하다'
    assert not [k for k in plus if lg.lanes[k]['red_spans']], \
        '반대 방향(dir=+1)에 red_spans 가 붙었다 — road 단위로 묶인 것이다'
    assert minus_red, 'dir=−1 쪽은 붉어야 한다'
    assert all(k[2] > 0 for k in minus_red)


def test_roadmark_30_is_not_a_speed_limit(lg):
    """30 표시는 제한속도로 쓰지 않는다. 50 표시는 그대로다.

    호환 모드가 켜져 있으면 **붉은 차로만** 30 이 남는다 — 노면표시에서 온 30 이
    아니라 red_compat 에서 온 30 이다 (speed_src 로 구분한다).
    """
    from_mark = [k for k, r in lg.lanes.items()
                 if r['speed_limit'] == 30 and r['speed_src'] != 'red_compat']
    assert not from_mark, from_mark[:5]
    assert [k for k, r in lg.lanes.items() if r['speed_limit'] == 50]


# ── 호환 모드 (2a-2) ────────────────────────────────────────────────────
def test_lane_level_compat_default_on():
    assert CFG['red_zone']['lane_level_compat'] is True


def test_compat_sets_limit_on_red_lanes_only(lg):
    """red_spans 가 있는 차로만 차로 단위 30 을 갖는다.

    red_spans 를 읽지 않는 제어기(팀원 브랜치·구 배포)가 붉은 차로에서 최소한
    30 을 읽게 하는 안전망이다. 붉지 않은 차로에는 절대 붙으면 안 된다 —
    그러면 이 작업이 푼 문제가 그대로 돌아온다.
    """
    if not CFG['red_zone']['lane_level_compat']:
        pytest.skip('호환 모드 off')
    red = {k for k, r in lg.lanes.items() if r['red_spans']}
    compat = {k for k, r in lg.lanes.items() if r['speed_src'] == 'red_compat'}
    assert compat == red
    assert all(lg.lanes[k]['speed_limit'] == float(CFG['red_zone']['limit_kph']) for k in compat)
    # 붉지 않은 차로에 30 이 남아 있지 않다
    assert not [k for k, r in lg.lanes.items() if k not in red and r['speed_limit'] == 30]


def test_no_red_lane_falls_back_to_carry(lg):
    """붉은 차로가 carry(앞 도로 값)로 새지 않는다.

    옛 그래프에는 **붉은데 speed_limit 이 None** 인 차로가 23개(742 m) 있었고,
    그 구간은 carry 로 앞 도로 값(대개 50)을 물고 45 km/h 로 지나갔다 — 항목 2
    중대 후보다. 지금은 두 겹으로 막힌다:
      · red_spans 가 그 구간을 표시하고 (제어기 2b·채점기가 읽는다)
      · 호환 모드가 차로 단위 speed_limit 도 세운다 (red_spans 를 안 읽는 제어기용)
    """
    red = [k for k, r in lg.lanes.items() if r['red_spans']]
    assert red
    assert all(lg.lanes[k]['school_zone'] for k in red)
    if CFG['red_zone']['lane_level_compat']:
        assert not [k for k in red if lg.lanes[k]['speed_limit'] is None]


# ── 채점기 독립 판정 ────────────────────────────────────────────────────
def _red_sample(lg):
    """붉은 구간 하나를 골라 (lane_key, s0, s1, 중간점 xy, yaw) 를 준다."""
    import numpy as np
    for k, r in sorted(lg.lanes.items()):
        for s0, s1 in r['red_spans']:
            if s1 - s0 < 20.0:
                continue
            sm = 0.5 * (s0 + s1)
            x, y, _z, h = lg.point_at(k, sm)
            return k, s0, s1, float(x), float(y), float(h)
    pytest.skip('20 m 넘는 붉은 구간이 없다')


def test_speed_context_reads_red_span_from_graph(lg):
    """붉은 구간 한복판에서 limit 이 red_zone.limit_kph 로 잡힌다.

    로그에는 **일부러 틀린 값**(50 / school False)을 넣는다 — 채점기가 로그를
    안 보고 lane_graph 를 본다는 것이 이 테스트의 요지다.
    """
    k, _s0, _s1, x, y, h = _red_sample(lg)
    ticks = [mk_tick(t=0.0, x=x, y=y, yaw=h, speed=12.0, speed_limit=50 / 3.6, school_zone=False)]
    ctx = SC.speed_context(ticks, lg, CFG)
    assert ctx[0]['red'] is True
    assert ctx[0]['limit_kph'] == round(float(CFG['red_zone']['limit_kph']))
    assert ctx[0]['src'] == 'graph'
    # 로그값은 진단으로 남는다
    assert ctx[0]['log_limit_kph'] == 50
    assert ctx[0]['log_school_zone'] is False


def test_speed_context_falls_back_to_log_without_graph(lg):
    ticks = [mk_tick(speed=10.0, speed_limit=30 / 3.6, school_zone=True)]
    ctx = SC.speed_context(ticks, None, CFG)
    assert ctx[0] == {'limit_kph': 30, 'red': True, 'lane': None, 's': None,
                      'log_limit_kph': 30, 'log_school_zone': True, 'src': 'log'}


def test_speed_context_carries_previous_limit(lg):
    """speed_limit 이 None 인 차로는 직전 값을 유지한다 (ego.py 와 같은 규칙)."""
    none_lane = next((k for k, r in lg.lanes.items()
                      if r['speed_limit'] is None and not r['red_spans']), None)
    lim_lane = next((k for k, r in lg.lanes.items() if r['speed_limit'] == 50), None)
    assert none_lane and lim_lane
    def xy(k, s):
        x, y, _z, h = lg.point_at(k, s)
        return float(x), float(y), float(h)
    x1, y1, h1 = xy(lim_lane, 0.5 * lg.length(lim_lane))
    x2, y2, h2 = xy(none_lane, 0.5 * lg.length(none_lane))
    ctx = SC.speed_context([mk_tick(x=x1, y=y1, yaw=h1), mk_tick(x=x2, y=y2, yaw=h2)], lg, CFG)
    assert ctx[0]['limit_kph'] == 50
    assert ctx[1]['limit_kph'] == 50, 'carry 가 끊겼다'


# ── severity ────────────────────────────────────────────────────────────
def _sev(over, school):
    return SC._severity('speed', {'max_over_kph': over, 'school_zone': school}, CFG['scoring'])


def test_severity_item1_thresholds():
    assert _sev(1.0, False) == 'none'
    assert _sev(1.01, False) == 'minor'
    assert _sev(20.0, False) == 'minor'
    assert _sev(20.01, False) == 'major'


def test_severity_item2_is_stricter():
    """보호구역은 1 초과 경미 / 5 초과 중대."""
    assert _sev(1.0, True) == 'none'
    assert _sev(1.01, True) == 'minor'
    assert _sev(5.0, True) == 'minor'
    assert _sev(5.01, True) == 'major'
    # 같은 초과량이 항목 1 에서는 경미, 항목 2 에서는 중대
    assert _sev(11.0, False) == 'minor'
    assert _sev(11.0, True) == 'major'


# ── 2b-A: 차로 s 로 붉은 구간 조회 ──────────────────────────────────────
def test_red_span_switch_default_on():
    """기본 on 이다 — 못 읽으면 항목 2 중대라 조용히 꺼지는 쪽이 더 위험하다."""
    assert CFG['speed']['red_span_enable'] is True


def test_speed_limit_at_without_s_is_unchanged(lg):
    """s 를 안 주면 예전 그대로 — 기존 호출부가 하나도 안 바뀐다."""
    for k, r in list(lg.lanes.items())[:200]:
        assert lg.speed_limit_at(k) == (r['speed_limit'], r['school_zone'])


def _partial_red_lane(lg):
    """구간이 차로 일부만 덮는 차로 (구간 뒤에 여유가 남는 것)."""
    mar = float(CFG['red_zone']['exit_margin_m'])
    for k, r in sorted(lg.lanes.items()):
        for _s0, s1 in r['red_spans']:
            if s1 + mar + 5.0 < r['length']:
                return k, s1
    pytest.skip('부분만 붉은 차로가 없다')


def test_inside_span_returns_zone_limit(lg):
    k, s1 = _partial_red_lane(lg)
    v, sc = lg.speed_limit_at(k, 0.5 * s1)
    assert (v, sc) == (float(CFG['red_zone']['limit_kph']), True)


def test_exit_margin_delays_release(lg):
    """구간 끝 + exit_margin_m 까지는 아직 보호구역이다."""
    k, s1 = _partial_red_lane(lg)
    mar = float(CFG['red_zone']['exit_margin_m'])
    assert lg.speed_limit_at(k, s1 + 0.5 * mar)[1] is True
    assert lg.speed_limit_at(k, s1 + mar + 0.5)[1] is False


def test_outside_span_ignores_compat_value(lg):
    """부분만 붉은 차로의 구간 밖은 호환 30 을 무시하고 None → carry.

    이게 없으면 (173,0,-1) 처럼 414.7 m 중 236.7 m 만 붉은 차로가 전 구간
    30 으로 묶인다.
    """
    if not CFG['red_zone']['lane_level_compat']:
        pytest.skip('호환 모드 off')
    k, s1 = _partial_red_lane(lg)
    mar = float(CFG['red_zone']['exit_margin_m'])
    assert lg.lanes[k]['speed_src'] == 'red_compat'
    assert lg.lanes[k]['speed_limit'] == float(CFG['red_zone']['limit_kph'])
    assert lg.speed_limit_at(k, s1 + mar + 1.0) == (None, False)


def test_switch_off_uses_lane_level_value(lg):
    """킬 스위치 off 면 s 를 줘도 차로 단위 값 그대로 = 이전 동작."""
    import copy
    from vtd_adapter.lanegraph import LaneGraph as LG
    off = copy.deepcopy(CFG)
    off['speed']['red_span_enable'] = False
    lg_off = LG(str(GRAPH), cfg=off)
    k, s1 = _partial_red_lane(lg)
    mar = float(CFG['red_zone']['exit_margin_m'])
    assert lg_off.speed_limit_at(k, s1 + mar + 1.0) == (lg.lanes[k]['speed_limit'],
                                                        lg.lanes[k]['school_zone'])


def test_cfg_follows_explicit_params(lg):
    """LaneGraph(cfg=...) 가 --config override 를 따라간다.

    2026-09-05 실측: 이게 없어서 replay 의 red_span_enable=false 가 무시됐고
    킬 스위치 off 검증이 통과하지 않았다 (모듈 폴백이 저장소 params 를 읽었다).
    """
    import copy
    from vtd_adapter.lanegraph import LaneGraph as LG
    alt = copy.deepcopy(CFG)
    alt['red_zone']['limit_kph'] = 22.0
    lg_alt = LG(str(GRAPH), cfg=alt)
    k, s1 = _partial_red_lane(lg)
    assert lg_alt.speed_limit_at(k, 0.5 * s1)[0] == 22.0


def test_plain_lane_is_untouched(lg):
    """붉은 구간이 없는 차로는 s 를 줘도 값이 그대로다."""
    k = next(k for k, r in lg.lanes.items() if not r['red_spans'] and r['speed_limit'] == 50)
    assert lg.speed_limit_at(k, 0.5 * lg.length(k)) == (50, False)


# ── 2b-B: 진입 전 감속 ──────────────────────────────────────────────────
def _kr(**over):
    import copy
    import sys as _s
    _s.path.insert(0, str(ROOT / 'team_code'))
    from kr_rules import KrRules
    cfg = copy.deepcopy(CFG)
    cfg['speed'].update(over)
    return KrRules(cfg)


class _FakePlanner:
    """route_s 축만 있는 최소 플래너 — 후보 공식만 검증한다."""
    def __init__(self, ivals, route_s=0.0):
        self._ivals = ivals
        self.route_index = 0
        self.route_s = [route_s]


def test_approach_params_present():
    sp = CFG['speed']
    assert sp['red_approach_enable'] is True
    assert float(sp['red_zone_target_kph']) == 27.0
    assert float(sp['approach_decel_mps2']) == 2.0
    assert float(sp['red_lookahead_m']) == 60.0


def test_approach_formula(monkeypatch):
    """v_ceiling = sqrt(v_zone^2 + 2 a d). 25 m 앞이면 45 km/h 쯤이 나온다."""
    import math
    k = _kr()
    pl = _FakePlanner([(100.0, 130.0)], route_s=75.0)
    k.red_ivals = pl._ivals
    v = k._red_approach_profile(pl)
    vz = float(CFG['speed']['red_zone_target_kph']) / 3.6
    a = float(CFG['speed']['approach_decel_mps2'])
    assert v == pytest.approx(math.sqrt(vz * vz + 2 * a * 25.0))
    assert 44.0 < v * 3.6 < 46.0


def test_no_candidate_inside_span():
    """구간 안에서는 후보를 안 낸다 — 거기선 제한속도가 상한이다."""
    k = _kr()
    pl = _FakePlanner([(100.0, 130.0)], route_s=110.0)
    k.red_ivals = pl._ivals
    assert k._red_approach_profile(pl) is None


def test_no_candidate_beyond_lookahead():
    k = _kr()
    pl = _FakePlanner([(100.0, 130.0)], route_s=0.0)   # 100 m 앞 > 60 m
    k.red_ivals = pl._ivals
    assert k._red_approach_profile(pl) is None


def test_switch_off_gives_no_candidate():
    k = _kr(red_approach_enable=False)
    pl = _FakePlanner([(100.0, 130.0)], route_s=90.0)
    k.red_ivals = pl._ivals
    assert k._red_approach_profile(pl) is None


def test_ceiling_at_entry_equals_zone_target():
    """진입점(d=0)에서 천장이 정확히 v_zone 이다 — 임계 31 아래다."""
    k = _kr()
    pl = _FakePlanner([(100.0, 130.0)], route_s=100.0 - 1e-9)
    k.red_ivals = pl._ivals
    v = k._red_approach_profile(pl)
    assert v * 3.6 == pytest.approx(float(CFG['speed']['red_zone_target_kph']), abs=1e-3)
    assert v * 3.6 < 31.0


def test_short_gap_between_spans_is_capped():
    """실측 3건 — 앞 구간 끝과 다음 진입 사이 틈이 1.0 / 9.0 m 뿐이다.

    그 틈에서 50(45 km/h)까지 가속하면 25 m 가 없어 되돌릴 수 없다.
    이 후보가 천장을 눌러 애초에 못 올라가게 하는지 본다.
    """
    import math
    k = _kr()
    vz = float(CFG['speed']['red_zone_target_kph']) / 3.6
    a = float(CFG['speed']['approach_decel_mps2'])
    for gap, want_kph in ((1.0, 27.9), (9.0, 34.6)):
        pl = _FakePlanner([(100.0, 130.0)], route_s=100.0 - gap)
        k.red_ivals = pl._ivals
        v = k._red_approach_profile(pl)
        assert v == pytest.approx(math.sqrt(vz * vz + 2 * a * gap))
        assert v * 3.6 == pytest.approx(want_kph, abs=0.1)
        assert v * 3.6 < 45.0, '틈에서 50 도로 속도까지 올라간다'


def test_intervals_use_exit_margin(lg):
    """구간 간격은 speed_limit_at 이 캡을 푸는 지점과 같은 축이어야 한다."""
    k = _kr()
    k.red_ivals = None
    class PL:
        pass
    pl = PL()
    pl.lg = lg
    pl.route_index = 0
    # 붉은 구간 하나를 그대로 경로로 삼는다
    kk, s1 = _partial_red_lane(lg)
    span = next(sp for sp in lg.lanes[kk]['red_spans'] if sp[1] == s1)
    import numpy as np
    ss = np.arange(0.0, lg.length(kk), 0.1)
    pl.route_s = ss
    class WP:
        def __init__(self, key, s):
            self.key, self.s = key, s
    pl.route_waypoints = [WP(kk, float(v)) for v in ss]
    ivals = k._red_intervals(pl)
    assert ivals, ivals
    mar = float(CFG['red_zone']['exit_margin_m'])
    assert ivals[0][0] == pytest.approx(span[0], abs=0.2)
    assert ivals[0][1] == pytest.approx(span[1] + mar, abs=0.3)
