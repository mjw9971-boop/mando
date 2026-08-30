"""
보행자 이벤트 트리거 거리 역산 (gen_scenarios.ped_trigger) + 조우 성립 판정
(event_check).

2026-08-30 실사고: 고정 트리거거리 25 m 로는 무단횡단 보행자가 ego 가 지나간
48 m 뒤(4 s 후)에 차로에 진입했다. 필요값이 도로 폭에 따라 27~87 m 로 변해
상수로는 원리적으로 못 맞춘다. 로그·리포트 어디에도 신호가 없어 육안으로만
알 수 있었고, 그마저 "제어기 미반응"으로 오진했다.
"""
import json
import math

import pytest

import event_check as ec        # noqa: E402 (conftest 가 tools 경로 추가)
import gen_scenarios as gs      # noqa: E402
from vtd_adapter.config import load_params_yaml


class FakeLG:
    """ped_trigger 가 보는 최소 LaneGraph — 차로폭·제한속도만."""

    def __init__(self, lane_w, limit_kph):
        self.lanes = {('r', 0, -1): {'speed_limit': limit_kph}}
        self._w = lane_w

    def width_at(self, _k, _s):
        return self._w


class FakeCtx:
    def __init__(self, lane_w=3.0, limit_kph=50, road_right=6.0, road_left=6.0):
        self.lg = FakeLG(lane_w, limit_kph)
        self.route = type('R', (), {'rt': {}})()
        self._edges = (road_left, road_right)
        self.occupied = []

    claim = gs.Ctx.claim


@pytest.fixture
def patched(monkeypatch):
    """road_width_at / lane_at 을 FakeCtx 값으로 — LaneGraph 없이 식만 검증한다."""
    def _apply(ctx):
        monkeypatch.setattr(gs, 'road_width_at', lambda lg, rt, s: ctx._edges)
        monkeypatch.setattr(gs, 'lane_at', lambda rt, s: (0, ('r', 0, -1), s))
        return ctx
    return _apply


def et():
    return load_params_yaml()['event_trigger']


# ── 역산 식 ──────────────────────────────────────────────────────────────
def test_trigger_matches_hand_computed_formula(patched):
    """trig_d = v_exp × (t_near + lead_s) − radius_m."""
    c = et()
    ctx = patched(FakeCtx(lane_w=3.0, limit_kph=50, road_right=7.9))
    trg = gs.ped_trigger(ctx, 500.0, 2.5, '우측', 'jaywalk')
    lat = 7.9 + float(c['ped_start_margin_m'])
    v_exp = 50 / 3.6 * float(c['speed_factor'])
    t_near = (lat - 1.5) / 2.5
    want = v_exp * (t_near + float(c['lead_s'])) - float(c['radius_m'])
    assert trg['trigger_d'] == pytest.approx(want, abs=0.01)
    assert trg['lat_start_m'] == pytest.approx(lat, abs=0.01)
    assert trg['t_near_s'] == pytest.approx(t_near, abs=0.01)


def test_trigger_scales_with_road_width(patched):
    """상수로는 못 맞추는 이유 — 같은 조건에서 폭만 바꿔도 필요값이 배로 뛴다."""
    narrow = gs.ped_trigger(patched(FakeCtx(road_right=3.0)), 500.0, 2.0, '우측', 'jaywalk')
    wide = gs.ped_trigger(patched(FakeCtx(road_right=8.0)), 500.0, 2.0, '우측', 'jaywalk')
    assert wide['trigger_d'] > narrow['trigger_d'] * 1.8


def test_left_side_uses_left_edge(patched):
    ctx = patched(FakeCtx(road_left=9.0, road_right=3.0))
    left = gs.ped_trigger(ctx, 500.0, 2.5, '좌측', 'jaywalk')
    assert left['lat_start_m'] == pytest.approx(9.0 + float(et()['ped_start_margin_m']))


def test_meet_lat_lands_inside_lane(patched):
    """조우 시점 보행자 횡위치는 차로 안이어야 한다 (설계 목표)."""
    trg = gs.ped_trigger(patched(FakeCtx(lane_w=3.2, road_right=7.0)), 500.0, 2.5,
                         '우측', 'jaywalk')
    assert abs(trg['meet_lat_m']) <= trg['lane_w_m'] / 2


# ── 보행속도 자동 조정 (승인안 (b)) ──────────────────────────────────────
def test_walk_speed_raised_when_trigger_would_exceed_max(patched):
    """넓은 도로 — 느린 보행이면 trig_d 가 상한을 넘으므로 속도를 올려 성립시킨다."""
    c = et()
    trg = gs.ped_trigger(patched(FakeCtx(road_right=9.0)), 500.0, 1.0, '우측', 'jaywalk')
    assert trg['walk_speed'] > 1.0
    assert trg['trigger_d'] <= float(c['trig_max_m']) + 1e-6


def test_walk_speed_lowered_when_trigger_would_undershoot_min(patched):
    """좁은 도로 — 빠른 보행이면 trig_d 가 하한 미만이라 속도를 낮춘다."""
    c = et()
    trg = gs.ped_trigger(patched(FakeCtx(road_right=2.5)), 500.0, 3.0, '우측', 'jaywalk')
    assert trg['walk_speed'] < 3.0
    assert trg['trigger_d'] >= float(c['trig_min_m']) - 1e-6


def test_dropped_when_needed_speed_out_of_range(patched):
    """(a) 조정 범위 밖이면 이 이벤트만 폐기 — 시나리오째 죽이지 않는다."""
    ctx = patched(FakeCtx(road_right=1.6, lane_w=3.0, limit_kph=30))
    with pytest.raises(gs.EventUnfeasible):
        gs.ped_trigger(ctx, 500.0, 2.0, '우측', 'jaywalk')


def test_unfeasible_is_gen_error_subclass():
    """resolve_events 의 기존 GenError 처리 경로를 그대로 탄다."""
    assert issubclass(gs.EventUnfeasible, gs.GenError)


def test_dropped_when_trigger_point_before_route_start(patched):
    """트리거점이 경로 앞이면 스폰 즉시 발동한다 — 옛 max(1.0, …) 클램프의 조용한 실패."""
    with pytest.raises(gs.EventUnfeasible):
        gs.ped_trigger(patched(FakeCtx(road_right=8.0)), 20.0, 1.5, '우측', 'jaywalk')


def test_dropped_when_start_already_inside_lane(patched):
    with pytest.raises(gs.EventUnfeasible):
        gs.ped_trigger(patched(FakeCtx(lane_w=8.0, road_right=1.0)), 500.0, 2.0,
                       '우측', 'jaywalk')


# ── 이벤트별 기본 보행속도 ───────────────────────────────────────────────
def test_walk_speed_default_per_kind():
    d = et()['walk_speed_default']
    assert gs.ped_walk_speed({}, 'jaywalk') == float(d['jaywalk'])
    assert gs.ped_walk_speed({}, 'pedestrian') == float(d['pedestrian'])
    assert float(d['jaywalk']) > float(d['pedestrian'])      # 무단횡단은 뛴다


def test_axis_overrides_default():
    assert gs.ped_walk_speed({'보행속도': 1.2}, 'jaywalk') == 1.2


# ── 점유 구간이 트리거 지점까지 ──────────────────────────────────────────
def test_claim_covers_trigger_point(patched):
    ctx = patched(FakeCtx(road_right=7.0))
    trg = gs.ped_trigger(ctx, 500.0, 2.5, '우측', 'jaywalk')
    gs.ped_claim(ctx, 500.0, trg, 'jaywalk')
    (a, b), = ctx.occupied
    assert a <= 500.0 - trg['trigger_d']          # 트리거점이 점유 구간 안
    assert b >= 500.0


# ── 조우 성립 판정 (event_check) ─────────────────────────────────────────
def tick(route_s, ped_lat=None, yaw=0.0, ex=0.0, ey=0.0, side='우측'):
    """보행자를 **출발측 기준** ped_lat m 지점에 놓는다 (+ = 아직 안 건넜다).

    ego 프레임 횡은 좌가 +. 우측 출발이면 출발측 = 오른쪽이므로 부호를 뒤집는다.
    """
    objs = []
    if ped_lat is not None:
        lat = -ped_lat if side == '우측' else ped_lat
        objs = [{'id': 6, 'cls': 'pedestrian',
                 'x': ex - lat * math.sin(yaw), 'y': ey + lat * math.cos(yaw)}]
    return {'ego': {'x': ex, 'y': ey, 'yaw': yaw, 'route_s': route_s}, 'objects': objs}


EV = {'kind': 'jaywalk', 'route_s': 100.0, 'lane_w_m': 3.0, 'from': '우측',
      'meet_lat_m': 0.5}
# 판정에 쓰는 기대값 — 허용대는 lane_w_m 에서 나오고 나머지는 표시용이다.
EXP = {'lane_w_m': 3.0, 'meet_lat_m': 0.5,
       't_near_s': 3.2, 't_far_s': 5.2, 't_meet_s': 3.6}


def test_verdict_ok_when_pedestrian_in_lane():
    ts = [tick(90.0, 3.0), tick(100.0, 0.5), tick(110.0, -2.0)]
    assert ec.check_event(EV, ts, 1.0, EXP)['verdict'] == ec.OK


def test_verdict_late_reproduces_the_real_failure():
    """실사고 재현: ego 통과 시 보행자가 아직 5.8 m 밖 (옛 고정 25 m 의 양상)."""
    r = ec.check_event(EV, [tick(100.0, 5.82)], 1.0, EXP)
    assert r['verdict'] == ec.LATE and r['lat_m'] == 5.82


def test_verdict_early_when_already_crossed():
    assert ec.check_event(EV, [tick(100.0, -6.0)], 1.0, EXP)['verdict'] == ec.EARLY


def test_verdict_no_pedestrian_object():
    assert ec.check_event(EV, [tick(100.0, None)], 1.0, EXP)['verdict'] == ec.NOSEE


def test_verdict_not_reached():
    assert ec.check_event(EV, [tick(40.0, 1.0)], 1.0, EXP)['verdict'] == ec.NOREACH


def test_band_is_half_lane_plus_tolerance():
    r = ec.check_event(EV, [tick(100.0, 2.4)], 1.0, EXP)
    assert r['band_m'] == pytest.approx(2.5) and r['verdict'] == ec.OK
    assert ec.check_event(EV, [tick(100.0, 2.6)], 1.0, EXP)['verdict'] == ec.LATE


def test_left_side_sign_convention():
    ev = dict(EV, **{'from': '좌측'})
    assert ec.check_event(ev, [tick(100.0, 0.5, side='좌측')], 1.0, EXP)['verdict'] == ec.OK
    assert ec.check_event(ev, [tick(100.0, 5.8, side='좌측')], 1.0, EXP)['verdict'] == ec.LATE


def test_verdict_holds_under_rotated_heading():
    """판정이 ego 헤딩과 무관해야 한다 (좌표 변환 검증)."""
    for yaw in (0.0, 1.0, -2.5, math.pi):
        assert ec.check_event(EV, [tick(100.0, 5.82, yaw=yaw)], 1.0, EXP)['verdict'] == ec.LATE
        assert ec.check_event(EV, [tick(100.0, 0.5, yaw=yaw)], 1.0, EXP)['verdict'] == ec.OK


def test_check_scenario_counts_unreached(tmp_path):
    """미도달은 total 에 남되 따로 세어, 성립 실패와 구분되게 한다."""
    import yaml as _yaml
    y = tmp_path / 's.yaml'
    y.write_text(_yaml.safe_dump({'events': [
        dict(EV, route_s=100.0), dict(EV, route_s=900.0), dict(EV, route_s=950.0)]},
        allow_unicode=True), encoding='utf-8')
    log = tmp_path / 'r.jsonl'
    log.write_text(''.join(json.dumps({**tick(s, 0.5), 'raw': {}}) + '\n'
                           for s in (50.0, 100.0, 150.0)), encoding='utf-8')
    rep = ec.check_scenario(str(y), str(log))
    assert (rep['total'], rep['ok'], rep['unreached']) == (3, 1, 2)
    assert '미도달2' in ec.render(rep)


# ── 기대값 산출 (expectation) ────────────────────────────────────────────
# 2026-08-30: 구 생성기 산출물(트리거 고정 25 m)의 보행자 4건이 전부 '기대없음'
# 으로 나와 판정이 통째로 비었다. yaml 에 기대값이 없으면 지도에서 다시 잰다.
LEGACY_EV = {'kind': 'jaywalk', 'route_s': 473.04, 'from': '우측',
             'walk_speed': 1.5, 'trigger_d': 25}      # lane_w_m·meet_lat_m 없음


class EdgeLG(FakeLG):
    """expectation 이 보는 최소 LaneGraph — 차로폭·제한속도·차도 가장자리."""

    def __init__(self, lane_w, limit_kph, left, right):
        super().__init__(lane_w, limit_kph)
        self._edges = (left, right)

    def roadway_edges(self, _k, _s):
        return self._edges


KEY = ('r', 0, -1)


def test_expectation_none_without_map_or_yaml_values():
    """지도도 yaml 값도 없으면 판정 불가 — 조용히 통과시키지 않는다."""
    assert ec.expectation(LEGACY_EV, et()) is None


def test_expectation_measured_from_map_when_yaml_lacks_it():
    """실사고 재현 — 지도로 재면 옛 고정 25 m 가 왜 못 만나는지가 수치로 나온다.

    ego 도착(2.24 s)이 보행자의 조우 창(5.21~7.28 s)보다 훨씬 이르다.
    """
    c = et()
    exp = ec.expectation(LEGACY_EV, c, EdgeLG(3.101, None, 11.26, 7.863), KEY, 48.5)
    v_exp = float(c['default_limit_kph']) / 3.6 * float(c['speed_factor'])
    lat0 = 7.863 + float(c['ped_start_margin_m'])
    t_meet = (25 + float(c['radius_m'])) / v_exp
    assert exp['lane_w_m'] == 3.1
    assert exp['t_meet_s'] == pytest.approx(t_meet, abs=0.01)
    assert exp['meet_lat_m'] == pytest.approx(lat0 - 1.5 * t_meet, abs=0.01)
    assert exp['t_near_s'] < exp['t_far_s']
    assert exp['t_meet_s'] < exp['t_near_s']          # 보행자가 차로에 닿기 전에 도착


def test_expectation_reproduces_generator_for_new_scenarios(patched):
    """같은 공식 하나 — 역산이 남긴 yaml 을 그대로 읽으면 생성기 값과 일치한다."""
    ctx = patched(FakeCtx(lane_w=3.3, limit_kph=30, road_right=5.0))
    trg = gs.ped_trigger(ctx, 500.0, 1.5, '우측', 'pedestrian')
    ev = {'kind': 'pedestrian', 'route_s': 500.0, 'from': '우측', **trg}
    exp = ec.expectation(ev, et())                     # 지도 없이 yaml 만으로
    assert exp['lane_w_m'] == trg['lane_w_m']
    assert exp['meet_lat_m'] == trg['meet_lat_m']
    assert (exp['t_near_s'], exp['t_far_s']) == (trg['t_near_s'], trg['t_far_s'])


def test_expectation_prefers_yaml_over_recomputation():
    """yaml 값이 곧 xml 이 만들어진 값 — 반올림된 입력으로 다시 계산하지 않는다."""
    ev = dict(LEGACY_EV, lane_w_m=3.0, lat_start_m=6.48, v_exp_mps=7.5,
              meet_lat_m=0.11, t_near_s=0.22, t_far_s=0.33)
    exp = ec.expectation(ev, et())
    assert (exp['meet_lat_m'], exp['t_near_s'], exp['t_far_s']) == (0.11, 0.22, 0.33)


def test_legacy_scenario_gets_real_verdict_not_placeholder(tmp_path):
    """구 시나리오도 지도만 있으면 성립/늦음/… 로 판정된다 (옛 '기대없음' 제거)."""
    import yaml as _yaml
    y = tmp_path / 's.yaml'
    y.write_text(_yaml.safe_dump({'events': [dict(LEGACY_EV, route_s=100.0)]},
                                 allow_unicode=True), encoding='utf-8')
    log = tmp_path / 'r.jsonl'
    log.write_text(''.join(
        json.dumps({**tick(s, 5.82), 'raw': {}, 'ego': dict(tick(s, 5.82)['ego'],
                                                            lane=list(KEY), s=48.5)}) + '\n'
        for s in (50.0, 100.0, 150.0)), encoding='utf-8')
    rep = ec.check_scenario(str(y), str(log), None,
                            EdgeLG(3.101, None, 11.26, 7.863))
    assert rep['events'][0]['verdict'] == ec.LATE
    assert ec.LEGACY not in ec.render(rep)


def test_unreached_events_still_counted_before_expectation(tmp_path):
    """미도달 판정이 기대값 유무보다 앞선다 — 옛 코드는 '기대없음'이 이걸 가렸다."""
    import yaml as _yaml
    y = tmp_path / 's.yaml'
    y.write_text(_yaml.safe_dump({'events': [dict(LEGACY_EV, route_s=900.0)]},
                                 allow_unicode=True), encoding='utf-8')
    log = tmp_path / 'r.jsonl'
    log.write_text(json.dumps({**tick(100.0, 1.0), 'raw': {},
                               'ego': dict(tick(100.0, 1.0)['ego'],
                                           lane=list(KEY), s=48.5)}) + '\n',
                   encoding='utf-8')
    rep = ec.check_scenario(str(y), str(log), None,
                            EdgeLG(3.101, None, 11.26, 7.863),
                            route={'lanes': [KEY], 'cum_s': [0.0], 'lengths': [1000.0],
                                   'total_length': 1000.0})
    assert (rep['unreached'], rep['events'][0]['verdict']) == (1, ec.NOREACH)
    assert rep['events'][0]['expect_lat_m'] is not None      # 못 갔어도 기대값은 낸다
