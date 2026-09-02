"""
여러 물체를 놓는 이벤트가 **한 가용구간 안에** 통째로 놓이는지 (2026-09-02 회귀).

결함: pick_s 는 구간들을 이어붙인 길이축에서 비율 지점을 고를 뿐인데, 배치는
그 지점부터 raw route_s 로 뻗어 나갔다. 그래서 뒷물체가 구간 밖 — 교차로
안으로 걸어 나갔다.
  · obstacle_chain 실측(표본 144): 개수 6 → 30% / 4 → 14~18% / 3 → 5% 가
    교차로 내부에 장애물을 놓았다.
  · narrow 실측: 실전주행_01_연속교차로24 의 뒤차 s=1713.0 이 구간 밖이었다.
교차로 안 장애물은 회피 연습이 아니라 채점 왜곡이다.

수정: _place_in_one_span 이 배치 전체가 들어가는 구간만 후보로 삼고, 요청
위치를 품은 구간 → 가까운 구간 순으로 옮겨 본다. 두 이벤트가 이 함수를
공유한다 (obstacle_chain 은 전부 실패해야 개수를 줄인다).
"""
import pathlib
import random
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import gen_scenarios as gs                                      # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
SPACING = 18.0


class FakeCtx:
    """_place_in_one_span 이 쓰는 표면만 — 가용구간과 점유 구간."""

    def __init__(self, spans, occupied=()):
        self._spans = list(spans)
        self.occupied = list(occupied)

    @property
    def spans(self):
        return list(self._spans)

    claim = gs.Ctx.claim


def place_chain(ctx, n, frac):
    """테스트 편의 래퍼 — ev_obstacle_chain 이 넘기는 인자와 같게 묶는다."""
    return gs._place_in_one_span(ctx, frac, reach=(n - 1) * SPACING,
                                 need=n * SPACING + 20.0,
                                 claim=(-20.0, n * SPACING + 20.0),
                                 what='obstacle_chain')


def place_narrow(ctx, frac):
    """ev_narrow 가 넘기는 인자와 같게 묶는다."""
    reach = max(ds for ds, _ in gs.NARROW_OFFSETS) - min(ds for ds, _ in gs.NARROW_OFFSETS)
    return gs._place_in_one_span(ctx, frac, reach=reach, need=40.0,
                                 claim=(-30.0, 50.0), what='narrow')


def chain_end(s0, n):
    return s0 + (n - 1) * SPACING


def in_one_span(spans, s0, n):
    return any(a - 1e-6 <= s0 and chain_end(s0, n) <= b + 1e-6 for a, b in spans)


# ── 단위: 구간 경계를 넘지 않는다 ────────────────────────────────────────
def test_chain_never_straddles_a_span_boundary():
    """구간 (100,200) 뒤에 교차로가 있고 요청 위치가 구간 끝 쪽이면, 예전에는
    체인이 200 을 넘어 교차로로 걸어 나갔다. 이제는 구간 안으로 당긴다."""
    spans = [(100.0, 200.0), (260.0, 300.0)]
    s0 = place_chain(FakeCtx(spans), 4, 0.9)
    assert s0 is not None
    assert in_one_span(spans, s0, 4), f's0={s0} end={chain_end(s0, 4)}'


@pytest.mark.parametrize('frac', [0.0, 0.15, 0.3, 0.5, 0.65, 0.8, 0.95, 1.0])
@pytest.mark.parametrize('n', [2, 3, 4, 6])
def test_all_fracs_and_counts_stay_inside(frac, n):
    spans = [(60.0, 130.0), (180.0, 320.0), (400.0, 455.0)]
    s0 = place_chain(FakeCtx(spans), n, frac)
    if s0 is None:                      # 어느 구간에도 안 들어가면 개수를 줄이는 게 맞다
        assert all(b - a < (n - 1) * SPACING for a, b in spans)
        return
    assert in_one_span(spans, s0, n)


def test_falls_back_to_another_span_before_shrinking():
    """요청 위치의 구간이 점유로 막혀도, 개수를 줄이기 전에 다른 구간을 쓴다."""
    spans = [(60.0, 200.0), (300.0, 440.0)]
    ctx = FakeCtx(spans, occupied=[(40.0, 220.0)])      # 첫 구간이 통째로 막혔다
    s0 = place_chain(ctx, 4, 0.1)          # 요청은 첫 구간 쪽
    assert s0 is not None and s0 >= 300.0
    assert in_one_span(spans, s0, 4)


def test_returns_none_when_no_span_can_hold_the_chain():
    """전부 짧으면 None — 호출자(ev_obstacle_chain)가 개수를 줄인다."""
    assert place_chain(FakeCtx([(60.0, 100.0), (200.0, 240.0)]), 6, 0.5) is None


def test_shrinks_only_after_every_span_failed():
    """개수 축소는 최후 수단이다 — 6은 못 들어가고 3은 들어가는 구간."""
    spans = [(60.0, 125.0)]                             # 65 m: n=4(54) 가능, n=6(90) 불가
    assert place_chain(FakeCtx(spans), 6, 0.5) is None
    s0 = place_chain(FakeCtx(spans), 4, 0.5)
    assert s0 is not None and in_one_span(spans, s0, 4)


def test_keeps_requested_position_when_it_already_fits():
    """이미 한 구간에 들어가던 체인은 시작점이 그대로여야 한다 (기존 산출물 불변)."""
    spans = [(100.0, 400.0)]
    want = gs.pick_s(spans, 0.5, need=4 * SPACING + 20.0)
    s0 = place_chain(FakeCtx(spans), 4, 0.5)
    assert s0 == pytest.approx(want)


def test_claim_width_unchanged():
    """점유 폭도 종전과 같아야 한다 — 다중이벤트 배치가 이 폭에 맞춰져 있다."""
    ctx = FakeCtx([(100.0, 400.0)])
    s0 = place_chain(ctx, 4, 0.5)
    assert ctx.occupied == [(s0 - 20, s0 + 4 * SPACING + 20)]


# ── 통합: 실제 맵에서 교차로 내부 배치가 0 이어야 한다 ────────────────────
@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
@pytest.mark.parametrize('n', [2, 3, 4, 6])
def test_no_obstacle_lands_in_a_junction_on_the_real_map(n):
    from vtd_adapter.lanegraph import LaneGraph
    lg = LaneGraph(str(GRAPH))
    route_defs, _themes, gen_cfg = gs.load_themes()
    pool = gs.RoutePool(lg, route_defs, 7, gen_cfg)
    checked = 0
    for i in range(9):
        name = ('직진', '우회전', '좌회전')[i % 3]
        try:
            route = pool.get(name, i, '체인구간회귀', min_length_m=400.0)
        except (gs.GenError, gs.RouteError):
            continue
        spans = gs.usable_spans(lg, route.rt)
        for frac in (0.3, 0.5, 0.75):
            s0 = place_chain(FakeCtx(spans), n, frac)
            if s0 is None:
                continue
            checked += 1
            for k in range(n):
                s = s0 + k * SPACING
                _i, lane, _sl = gs.lane_at(route.rt, s)
                assert lg.lanes[lane]['junction'] == -1, (
                    f'{name}{i} frac={frac} n={n}: s={s:.1f} 가 교차로 '
                    f'{lg.lanes[lane]["junction"]} 안이다')
    assert checked > 0, '표본이 하나도 안 만들어졌다 — 테스트가 아무것도 안 봤다'


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_ev_obstacle_chain_places_nothing_in_a_junction():
    """공개 표면(ev_obstacle_chain)으로도 같은 보장 — 수정 전에는 여기서 깨졌다.

    실측(2026-09-02, 수정 전): 연쇄장애물 --count 12 --seed 0 에서 교차로 내부
    배치 3건 (연쇄장애물_01_좌회전0 s=409.6·427.6 → junction 78,
    연쇄장애물_08_직진7 s=133.6 → junction 62).
    """
    from vtd_adapter.lanegraph import LaneGraph
    lg = LaneGraph(str(GRAPH))
    route_defs, _themes, gen_cfg = gs.load_themes()
    ctrl_map = gs.junction_ctrl_map(lg)
    pool = gs.RoutePool(lg, route_defs, 7, gen_cfg)
    checked = 0
    for i in range(9):
        name = ('직진', '우회전', '좌회전')[i % 3]
        try:
            route = pool.get(name, i, '체인구간회귀', min_length_m=400.0)
        except (gs.GenError, gs.RouteError):
            continue
        for n in (3, 4, 6):
            for frac in (0.3, 0.5, 0.75):
                ctx = gs.Ctx(lg, route, random.Random(0), ctrl_map)
                try:
                    out = gs.ev_obstacle_chain(ctx, {'위치': frac, '개수': n})
                except gs.GenError:
                    continue
                checked += 1
                for s in out['route_s']:
                    _i, lane, _sl = gs.lane_at(route.rt, s)
                    assert lg.lanes[lane]['junction'] == -1, (
                        f'{name}{i} frac={frac} 개수={n}: s={s} 가 교차로 '
                        f'{lg.lanes[lane]["junction"]} 안이다')
    assert checked > 0


# ── narrow: 양측 정차 차량 두 대도 한 구간 안이어야 한다 ──────────────────
def narrow_reach():
    ds = [d for d, _ in gs.NARROW_OFFSETS]
    return max(ds) - min(ds)


def test_narrow_offsets_are_the_single_source_of_reach():
    """배치와 구간 판정이 같은 상수를 봐야 한다 — 값을 두 곳에 적지 않는다."""
    assert narrow_reach() == 14.0
    assert [sgn for _, sgn in gs.NARROW_OFFSETS] == [-1.0, 1.0]   # 우측 먼저, 좌측 뒤


# 실측 회귀 (2026-09-02): 실전주행_01_연속교차로24 — 뒤차 s=1713.0 이
# 가용구간 (1661.03, 1712.53) 을 0.47 m 넘어 교차로 여유구간으로 나갔다.
# 이벤트 5개 중 narrow 가 3번째라 scale_events 슬롯은 (2+0.5)/5 = 0.5.
REAL_SPANS_연속교차로24 = [
    (60.0, 141.47), (215.9, 275.76), (532.67, 743.33), (807.93, 862.42),
    (963.07, 1180.24), (1275.38, 1331.2), (1431.17, 1566.23),
    (1661.03, 1712.53), (1834.2, 2049.24), (2197.5, 2319.29),
    (2418.12, 2636.5), (2725.22, 2868.19), (3060.9, 3241.69),
]


def test_real_case_연속교차로24_narrow_no_longer_leaves_its_span():
    """수정 전 이 배치가 s=1699.0 → 뒤차 1713.0 으로 구간을 넘었다."""
    assert gs.pick_s(REAL_SPANS_연속교차로24, 0.5, need=40.0) == pytest.approx(1699.0)
    s = place_narrow(FakeCtx(REAL_SPANS_연속교차로24), 0.5)
    assert s is not None
    assert s + narrow_reach() <= 1712.53 + 1e-6      # 뒤차가 구간 안
    assert s == pytest.approx(1698.53, abs=0.01)     # 구간 안으로 0.47 m 당겼다


@pytest.mark.parametrize('frac', [0.0, 0.15, 0.3, 0.5, 0.65, 0.8, 0.95, 1.0])
def test_narrow_both_vehicles_stay_in_one_span(frac):
    spans = [(60.0, 78.0), (120.0, 141.0), (200.0, 320.0)]   # 첫 구간은 14 m 초과지만 짧다
    s = place_narrow(FakeCtx(spans), frac)
    assert s is not None
    assert any(a - 1e-6 <= s and s + narrow_reach() <= b + 1e-6 for a, b in spans), \
        f's={s} 뒤차={s + narrow_reach()}'


def test_narrow_skips_spans_too_short_for_two_vehicles():
    """14 m 가 안 되는 구간은 후보에서 빠진다."""
    spans = [(60.0, 70.0), (100.0, 110.0), (200.0, 260.0)]
    s = place_narrow(FakeCtx(spans), 0.1)
    assert s is not None and 200.0 <= s <= 260.0 - narrow_reach()


def test_narrow_moves_to_another_span_before_giving_up():
    spans = [(60.0, 200.0), (300.0, 440.0)]
    ctx = FakeCtx(spans, occupied=[(20.0, 260.0)])           # 첫 구간이 막혔다
    s = place_narrow(ctx, 0.1)
    assert s is not None and s >= 300.0


def test_narrow_returns_none_when_no_span_fits():
    assert place_narrow(FakeCtx([(60.0, 70.0), (100.0, 112.0)]), 0.5) is None


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_ev_narrow_places_both_vehicles_inside_one_span_on_the_real_map():
    from vtd_adapter.lanegraph import LaneGraph
    lg = LaneGraph(str(GRAPH))
    route_defs, _themes, gen_cfg = gs.load_themes()
    ctrl_map = gs.junction_ctrl_map(lg)
    pool = gs.RoutePool(lg, route_defs, 7, gen_cfg)
    checked = 0
    for i in range(9):
        name = ('직진', '우회전', '좌회전')[i % 3]
        try:
            route = pool.get(name, i, '체인구간회귀', min_length_m=400.0)
        except (gs.GenError, gs.RouteError):
            continue
        spans = gs.usable_spans(lg, route.rt)
        for frac in (0.3, 0.5, 0.75):
            ctx = gs.Ctx(lg, route, random.Random(0), ctrl_map)
            try:
                out = gs.ev_narrow(ctx, {'위치': frac})
            except gs.GenError:
                continue
            checked += 1
            for veh in out['vehicles']:
                s = veh['route_s']
                assert any(a - 0.01 <= s <= b + 0.01 for a, b in spans), \
                    f'{name}{i} frac={frac}: {veh["name"]} s={s} 가 가용구간 밖'
                _i, lane, _sl = gs.lane_at(route.rt, s)
                assert lg.lanes[lane]['junction'] == -1
    assert checked > 0


# ── 겹칠 때는 구간을 옮기기 전에 같은 구간 안에서 민다 ────────────────────
def test_slides_within_the_span_before_moving_to_another():
    """옮기면 요청 위치에서 멀어진다 — 밀어서 되면 미는 게 먼저다.

    실측 회귀 (2026-09-02): 밀기가 없으면 정적회피집중_02_우회전 의 narrow 가
    186.9 → 296.8 로 110 m 튀었다. 수정 전에도 186.9 는 구간 안이었으므로
    그 이동은 개선이 아니라 회귀다.
    """
    spans = [(60.0, 223.0), (297.0, 343.0)]
    ctx = FakeCtx(spans, occupied=[(73.77, 153.77)])     # static_vehicle 이 앞을 점유
    s = place_narrow(ctx, 0.65)
    assert s is not None
    assert s <= 223.0 - narrow_reach()                   # 첫 구간에 남았다
    assert s + (-30.0) >= 153.77 - 1e-6                  # 점유 뒤로 밀렸을 뿐


def test_slide_lands_flush_against_the_occupied_block():
    """민 결과는 점유 경계에 딱 붙는다 — 요청 위치에서 가장 가까운 빈 자리다."""
    spans = [(0.0, 400.0)]
    ctx = FakeCtx(spans, occupied=[(100.0, 200.0)])
    # 요청 위치를 점유 한복판(150)으로: pick_s 는 frac*(400-40) 이므로 150/360
    s = place_narrow(ctx, 150.0 / 360.0)
    # 금지 s 구간은 (100-50, 200+30) = (50, 230). 150 에서 가까운 쪽은 230.
    assert s == pytest.approx(230.0)


def test_moves_span_only_when_the_whole_span_is_blocked():
    spans = [(60.0, 200.0), (300.0, 440.0)]
    ctx = FakeCtx(spans, occupied=[(20.0, 260.0)])       # 첫 구간이 통째로 막혔다
    s = place_narrow(ctx, 0.1)
    assert s is not None and s >= 300.0


def test_chain_also_slides_before_moving():
    """공용 함수라 obstacle_chain 도 같은 규칙을 따른다."""
    spans = [(0.0, 500.0)]
    ctx = FakeCtx(spans, occupied=[(100.0, 200.0)])
    s0 = place_chain(ctx, 4, 150.0 / (500.0 - 92.0))     # 요청 위치 = 점유 한복판
    # 금지 s 구간은 (100-92, 200+20) = (8, 220). 150 에서 가까운 쪽은 220.
    assert s0 == pytest.approx(220.0)
    assert in_one_span(spans, s0, 4)
