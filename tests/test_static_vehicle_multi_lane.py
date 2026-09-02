"""static_vehicle 은 **동일 방향 주행차로가 2개 이상인 자리**에만 놓는다 (2026-09-02).

결함: ev_static_vehicle 은 pick_s 로 가용축 비율 지점만 골랐고 차로 수를 보지
않았다. 자차로 정중앙 정차 차량은 1차로 구간에서 회피가 **원천 불가**다 —
차폭 1.9 m / 이 맵 차로 폭 2.6~3.2 m 라 좌우 여유가 0.35~0.65 m 이고,
kr_rules 는 황색 중앙선을 어느 단계에서도 넘지 않으며 이웃 차로가 없으면
no_neighbor 로 기각한다. ego 가 영구히 막혀 런이 blocked 로 끝난다.

실측(2026-09-02, seed 0 전수 생성): static_vehicle 배치 113건 중 24건(21%)이
1차로 구간이었다. 그중 6건은 '좌측차로' 요청이 좌측 이웃이 없어 조용히
t=0(자차로 정중앙)으로 떨어진 것이다 — 요청과 정반대 시나리오다.
blocked 로 끝난 기존 런 실전주행_02_직진28 이 이 원인이었다.

수정: params gen_placement.static_vehicle.require_multi_lane (기본 true) 가
켜져 있으면 _multi_lane_spans 로 후보 구간을 좁히고, 좌측 이웃이 없는
'좌측차로' 는 GenError 로 올려 백필로 넘긴다. false 면 종전 동작 그대로.
"""
import pathlib
import random
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import gen_scenarios as gs                                      # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'


# ── 가짜 지도: _same_dir_lane_count 가 쓰는 표면만 ────────────────────────
class FakeLG:
    """lanes[k] = (dir, type, width). nbs[(k, side)] = 이웃 키."""

    def __init__(self, lanes, nbs):
        self.lanes = {k: {'dir': d, 'type': t, 'length': 100.0, 'w': w}
                      for k, (d, t, w) in lanes.items()}
        self._nbs = dict(nbs)

    def neighbor(self, k, side):
        return self._nbs.get((k, side))

    def width_at(self, k, _s):
        return self.lanes[k]['w']

    def length(self, k):
        return self.lanes[k]['length']


MINW = 2.5


def test_lone_lane_counts_one():
    lg = FakeLG({'a': (1, 'driving', 3.0)}, {})
    assert gs._same_dir_lane_count(lg, 'a', 0.0, MINW) == 1


def test_same_direction_neighbour_counts():
    lg = FakeLG({'a': (1, 'driving', 3.0), 'b': (1, 'driving', 3.0)},
                {('a', 'right'): 'b'})
    assert gs._same_dir_lane_count(lg, 'a', 0.0, MINW) == 2


def test_zero_width_taper_is_not_a_lane():
    """이 맵에는 폭 0 인 테이퍼 차로가 실재한다 ((173,3,-1) w=0.0).
    개수만 세면 '이웃이 있다' 가 참이 되지만 비켜 설 자리는 없다."""
    lg = FakeLG({'a': (1, 'driving', 3.0), 'b': (1, 'driving', 0.0)},
                {('a', 'left'): 'b'})
    assert gs._same_dir_lane_count(lg, 'a', 0.0, MINW) == 1


def test_opposite_direction_neighbour_is_not_counted():
    """중앙선 너머는 회피 공간이 아니다 — kr_rules 가 중앙선을 넘지 않는다."""
    lg = FakeLG({'a': (1, 'driving', 3.0), 'b': (-1, 'driving', 3.0)},
                {('a', 'left'): 'b'})
    assert gs._same_dir_lane_count(lg, 'a', 0.0, MINW) == 1


def test_non_driving_neighbour_is_not_counted():
    lg = FakeLG({'a': (1, 'driving', 3.0), 'b': (1, 'shoulder', 3.0)},
                {('a', 'right'): 'b'})
    assert gs._same_dir_lane_count(lg, 'a', 0.0, MINW) == 1


def test_counts_both_sides_and_chains():
    lg = FakeLG({'a': (1, 'driving', 3.0), 'b': (1, 'driving', 3.0),
                 'c': (1, 'driving', 3.0), 'd': (1, 'driving', 3.0)},
                {('a', 'left'): 'b', ('b', 'left'): 'c', ('a', 'right'): 'd'})
    assert gs._same_dir_lane_count(lg, 'a', 0.0, MINW) == 4


def test_chain_stops_at_the_first_taper():
    """0 폭 차로 **너머**의 넓은 차로는 셈에 넣지 않는다 — 건너갈 수 없다."""
    lg = FakeLG({'a': (1, 'driving', 3.0), 'b': (1, 'driving', 0.0),
                 'c': (1, 'driving', 3.0)},
                {('a', 'left'): 'b', ('b', 'left'): 'c'})
    assert gs._same_dir_lane_count(lg, 'a', 0.0, MINW) == 1


# ── _multi_lane_spans: 가용구간을 2차로 부분구간으로 쪼갠다 ───────────────
class FakeCtx:
    """_multi_lane_spans / _place_in_one_span 이 쓰는 표면만.

    lane_at 이 route_s → 차로를 되짚으므로 합성 경로(rt)를 만든다: 길이
    seg_m 짜리 차로를 늘어놓고, one_lane 에 든 인덱스만 이웃이 없게 한다.
    """

    def __init__(self, spans, n_lanes, seg_m=10.0, occupied=()):
        keys = [f'L{i}' for i in range(len(n_lanes))]
        lanes, nbs = {}, {}
        for i, (k, n) in enumerate(zip(keys, n_lanes)):
            lanes[k] = (1, 'driving', 3.0)
            if n >= 2:
                nb = f'N{i}'
                lanes[nb] = (1, 'driving', 3.0)
                nbs[(k, 'right')] = nb
        self.lg = FakeLG(lanes, nbs)
        cum = [i * seg_m for i in range(len(keys))]
        rt = {'lanes': keys, 'cum_s': cum, 'lengths': [seg_m] * len(keys),
              'total_length': seg_m * len(keys)}
        self.route = type('R', (), {'rt': rt})()
        self._spans = list(spans)
        self.occupied = list(occupied)

    @property
    def spans(self):
        return list(self._spans)

    claim = gs.Ctx.claim


def test_all_multi_lane_span_survives_whole():
    ctx = FakeCtx([(0.0, 100.0)], [2] * 11)
    assert gs._multi_lane_spans(ctx, MINW, 30.0) == [(0.0, 100.0)]


def test_one_lane_stretch_splits_the_span():
    """가운데 40~60 m 가 1차로면 앞뒤 두 조각으로 갈린다."""
    n = [2] * 11
    n[4] = n[5] = 1                                   # route_s 40~60 m
    out = gs._multi_lane_spans(FakeCtx([(0.0, 100.0)], n), MINW, 30.0)
    assert len(out) == 2
    assert out[0][1] <= 40.0 and out[1][0] >= 60.0
    for a, b in out:
        assert b - a >= 30.0


def test_fragments_shorter_than_min_span_are_dropped():
    """짧은 조각은 옆 차로가 있어도 시프트가 성립하지 않는다
    (overtake.min_corridor_m = 25 m 점선 회랑이 필요하다)."""
    n = [2] * 11
    n[2] = 1                                          # 앞 조각이 20 m 로 잘린다
    out = gs._multi_lane_spans(FakeCtx([(0.0, 100.0)], n), MINW, 30.0)
    assert all(b - a >= 30.0 for a, b in out)
    assert all(a >= 30.0 for a, b in out)


def test_span_endpoints_are_verified_samples():
    """구간 양끝이 둘 다 검사를 통과한 지점이어야 한다 — 끝점이 1차로면
    그 자리에 놓을 수 있게 된다."""
    n = [2] * 11
    n[7] = 1
    ctx = FakeCtx([(0.0, 100.0)], n)
    out = gs._multi_lane_spans(ctx, MINW, 20.0)
    assert out
    for a, b in out:
        for s in (a, b):
            _i, k, sl = gs.lane_at(ctx.route.rt, s)
            assert gs._same_dir_lane_count(ctx.lg, k, sl, MINW) >= 2


def test_no_multi_lane_span_returns_empty():
    assert gs._multi_lane_spans(FakeCtx([(0.0, 100.0)], [1] * 11), MINW, 30.0) == []


# ── _place_in_one_span(spans=) — 기존 호출부 무변경 ───────────────────────
def test_place_defaults_to_ctx_spans():
    """spans 를 안 주면 종전과 같이 ctx.spans 를 본다 (기존 산출물 불변)."""
    ctx = FakeCtx([(0.0, 100.0)], [2] * 11)
    other = FakeCtx([(0.0, 100.0)], [2] * 11)
    assert gs._place_in_one_span(ctx, 0.5, 0.0, 30.0, (-40.0, 40.0), 'x') == \
        gs._place_in_one_span(other, 0.5, 0.0, 30.0, (-40.0, 40.0), 'x', spans=None)


def test_place_honours_narrowed_spans():
    ctx = FakeCtx([(0.0, 100.0)], [2] * 11)
    s = gs._place_in_one_span(ctx, 0.0, 0.0, 30.0, (-40.0, 40.0), 'x',
                              spans=[(60.0, 100.0)])
    assert s is not None and 60.0 <= s <= 100.0


# ── 실제 맵 통합 ─────────────────────────────────────────────────────────
def real_map():
    from vtd_adapter.lanegraph import LaneGraph
    lg = LaneGraph(str(GRAPH))
    route_defs, _themes, gen_cfg = gs.load_themes()
    return lg, gs.RoutePool(lg, route_defs, 7, gen_cfg), gs.junction_ctrl_map(lg)


def gate_cfg(monkeypatch, **over):
    """plc_cfg 캐시를 건드리지 않고 스위치만 바꾼다."""
    cfg = dict(gs.plc_cfg()['static_vehicle'])
    cfg.update(over)
    monkeypatch.setattr(gs, '_PLC_CFG', {'static_vehicle': cfg})


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_gate_on_never_places_in_a_single_lane_stretch(monkeypatch):
    """수정 전에는 여기서 깨졌다 — 실측 정적회피집중 5/23, 추월집중 7/30."""
    gate_cfg(monkeypatch, require_multi_lane=True)
    minw = float(gs.plc_cfg()['static_vehicle']['min_neighbor_width_m'])
    lg, pool, ctrl_map = real_map()
    checked = 0
    for i in range(12):
        name = ('직진', '우회전', '좌회전')[i % 3]
        try:
            route = pool.get(name, i, '정차차차로회귀', min_length_m=400.0)
        except (gs.GenError, gs.RouteError):
            continue
        for frac in (0.1, 0.3, 0.5, 0.75, 0.9):
            ctx = gs.Ctx(lg, route, random.Random(0), ctrl_map)
            try:
                out = gs.ev_static_vehicle(ctx, {'위치': frac})
            except gs.GenError:
                continue
            checked += 1
            _i, k, sl = gs.lane_at(route.rt, out['route_s'])
            assert gs._same_dir_lane_count(lg, k, sl, minw) >= 2, (
                f'{name}{i} frac={frac}: s={out["route_s"]} 가 1차로 구간이다 '
                f'(lane {k}) — 회피가 원천 불가다')
    assert checked > 0, '표본이 하나도 안 만들어졌다 — 테스트가 아무것도 안 봤다'


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_gate_off_reproduces_the_old_placement(monkeypatch):
    """off 는 종전 동작 그대로여야 한다 — 대회에 1차선 정적차가 나올지 아직
    확인 못 했으므로 그 케이스를 만드는 수단을 남긴다."""
    gate_cfg(monkeypatch, require_multi_lane=False)
    lg, pool, ctrl_map = real_map()
    checked = 0
    for i in range(9):
        name = ('직진', '우회전', '좌회전')[i % 3]
        try:
            route = pool.get(name, i, '정차차차로회귀', min_length_m=400.0)
        except (gs.GenError, gs.RouteError):
            continue
        for frac in (0.1, 0.5, 0.9):
            ctx = gs.Ctx(lg, route, random.Random(0), ctrl_map)
            try:
                out = gs.ev_static_vehicle(ctx, {'위치': frac})
            except gs.GenError:
                continue
            checked += 1
            want = gs.pick_s(gs.usable_spans(lg, route.rt), frac, need=30.0)
            assert out['route_s'] == pytest.approx(round(want, 2))
    assert checked > 0


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_left_lane_request_never_falls_back_to_blocking_the_ego_lane(monkeypatch):
    """'좌측차로' 는 자차로를 막지 않는다 — 조용한 폴백이 완전차단 24건 중
    6건을 만들었다. 이제는 GenError 로 올려 백필한다."""
    gate_cfg(monkeypatch, require_multi_lane=True)
    lg, pool, ctrl_map = real_map()
    checked = 0
    for i in range(12):
        name = ('직진', '우회전', '좌회전')[i % 3]
        try:
            route = pool.get(name, i, '정차차차로회귀', min_length_m=400.0)
        except (gs.GenError, gs.RouteError):
            continue
        for frac in (0.1, 0.3, 0.5, 0.75, 0.9):
            ctx = gs.Ctx(lg, route, random.Random(0), ctrl_map)
            try:
                out = gs.ev_static_vehicle(ctx, {'위치': frac, '차선': '좌측차로'})
            except gs.GenError:
                continue
            checked += 1
            assert out['lane'] == '좌측차로'
            # 실제로 옆 차로에 놓였는지 — 자차로 중앙이면 check 태그가 ego_lane 이다
            assert ('side_lane' in [tag for _x, _y, tag in ctx.checks]), (
                f'{name}{i} frac={frac}: 좌측차로 요청이 자차로 배치로 떨어졌다')
    assert checked > 0
