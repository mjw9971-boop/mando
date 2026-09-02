"""obstacle_chain 도 동일 방향 주행차로 2개 이상인 자리에만 놓는다 (2026-09-02).

결함: static_vehicle 게이트(작업2) 때 obstacle_chain 은 "가장자리 엇갈림
슬라럼이라 원천 불가 아님"으로 제외했으나 실측이 반증했다 —
logs/batch/20260902_231132 정적회피집중_01_좌회전0: 1차로 (2818,6,-1) 에서
체인(Fuelcan)이 자차로를 직접 막아 **충돌 1 + blocked**. 재생성 전수에서도
1차로 배치 7/58 이 남아 있었다.

수정: gen_placement.obstacle_chain.require_multi_lane (기본 true) 이 켜져
있으면 static_vehicle 과 같은 _multi_lane_spans 경로로 후보 구간을 좁힌다.
체인 전체(reach)가 한 2차로 부분구간 안에 들어가야 한다. 폭·최소구간 임계는
static_vehicle 값을 그대로 읽는다 (값을 두 곳에 적지 않는다).
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


def gate_cfg(monkeypatch, enable):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in gs.plc_cfg().items()}
    cfg['obstacle_chain']['require_multi_lane'] = enable
    monkeypatch.setattr(gs, '_PLC_CFG', cfg)


# ── 가짜 지도/경로: _multi_lane_spans 가 쓰는 표면만 (sv 테스트와 동일 골격) ──
class FakeLG:
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


class FakeCtx:
    def __init__(self, spans, n_lanes, seg_m=10.0):
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
        self.occupied = []

    @property
    def spans(self):
        return list(self._spans)

    claim = gs.Ctx.claim


def test_gate_raises_unfeasible_when_route_has_no_multi_lane_span(monkeypatch):
    """전 구간 1차로면 배치 전에 EventUnfeasible — resolve_events 가 위치를
    옮겨 재시도하고, scale_events 테마는 그 이벤트만 버린다."""
    gate_cfg(monkeypatch, True)
    ctx = FakeCtx([(0.0, 200.0)], [1] * 21)
    with pytest.raises(gs.EventUnfeasible, match='2개 이상'):
        gs.ev_obstacle_chain(ctx, {'위치': 0.5, '개수': 3})


# ── 실제 맵 ──────────────────────────────────────────────────────────────
MINW_OF = lambda: float(gs.plc_cfg()['static_vehicle']['min_neighbor_width_m'])  # noqa: E731


def real_map():
    from vtd_adapter.lanegraph import LaneGraph
    lg = LaneGraph(str(GRAPH))
    route_defs, _themes, gen_cfg = gs.load_themes()
    return lg, gs.RoutePool(lg, route_defs, 7, gen_cfg), gs.junction_ctrl_map(lg)


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_gate_on_no_can_lands_in_a_single_lane_stretch(monkeypatch):
    """수정 전에는 깨졌다 — 재생성 전수에서 1차로 체인 7/58 (2026-09-02)."""
    gate_cfg(monkeypatch, True)
    lg, pool, ctrl_map = real_map()
    minw = MINW_OF()
    checked = 0
    for i in range(9):
        name = ('직진', '우회전', '좌회전')[i % 3]
        try:
            route = pool.get(name, i, '체인차로회귀', min_length_m=400.0)
        except (gs.GenError, gs.RouteError):
            continue
        for n in (2, 3, 4, 6):
            for frac in (0.1, 0.4, 0.7, 0.9):
                ctx = gs.Ctx(lg, route, random.Random(0), ctrl_map)
                try:
                    out = gs.ev_obstacle_chain(ctx, {'위치': frac, '개수': n})
                except gs.GenError:
                    continue
                checked += 1
                for s in out['route_s']:
                    _i, k, sl = gs.lane_at(route.rt, s)
                    cnt = gs._same_dir_lane_count(lg, k, sl, minw)
                    assert cnt >= 2, (
                        f'{name}{i} frac={frac} n={n}: s={s} 가 1차로 구간이다 '
                        f'(lane {k}) — 체인이 자차로를 차단한다')
    assert checked > 0, '표본이 하나도 안 만들어졌다 — 테스트가 아무것도 안 봤다'


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_gate_off_reproduces_the_old_placement(monkeypatch):
    """off 는 종전 동작 그대로 — spans=None 이 옛 코드와 같은 자리를 고른다."""
    gate_cfg(monkeypatch, False)
    lg, pool, ctrl_map = real_map()
    checked = 0
    for i in range(6):
        name = ('직진', '우회전', '좌회전')[i % 3]
        try:
            route = pool.get(name, i, '체인차로회귀', min_length_m=400.0)
        except (gs.GenError, gs.RouteError):
            continue
        for n in (3, 4):
            for frac in (0.3, 0.7):
                ctx = gs.Ctx(lg, route, random.Random(0), ctrl_map)
                try:
                    out = gs.ev_obstacle_chain(ctx, {'위치': frac, '개수': n})
                except gs.GenError:
                    continue
                checked += 1
                # 종전 로직 재현: 같은 인자에 spans 미지정 (개수 축소 루프 포함)
                ref = gs.Ctx(lg, route, random.Random(0), ctrl_map)
                nn, s0 = n, None
                while nn >= 2:
                    s0 = gs._place_in_one_span(ref, frac, reach=(nn - 1) * SPACING,
                                               need=nn * SPACING + 20.0,
                                               claim=(-20.0, nn * SPACING + 20.0),
                                               what='obstacle_chain')
                    if s0 is not None:
                        break
                    nn -= 1
                assert out['route_s'][0] == pytest.approx(round(s0, 2))
                assert out['count'] == nn
    assert checked > 0


@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음 (gitignore 대상)')
def test_whole_chain_stays_inside_one_multi_lane_span(monkeypatch):
    """첫 캔만이 아니라 **체인 전체**가 2차로 부분구간 안이어야 한다 —
    reach 를 spans 에 같이 넘기는 이유."""
    gate_cfg(monkeypatch, True)
    lg, pool, ctrl_map = real_map()
    minw = MINW_OF()
    min_span = float(gs.plc_cfg()['static_vehicle']['min_span_m'])
    checked = 0
    for i in range(6):
        name = ('직진', '우회전', '좌회전')[i % 3]
        try:
            route = pool.get(name, i, '체인차로회귀', min_length_m=400.0)
        except (gs.GenError, gs.RouteError):
            continue
        ctx = gs.Ctx(lg, route, random.Random(0), ctrl_map)
        mspans = gs._multi_lane_spans(ctx, minw, min_span)
        for frac in (0.2, 0.6):
            c2 = gs.Ctx(lg, route, random.Random(0), ctrl_map)
            try:
                out = gs.ev_obstacle_chain(c2, {'위치': frac, '개수': 4})
            except gs.GenError:
                continue
            checked += 1
            lo, hi = out['route_s'][0], out['route_s'][-1]
            # 기록값은 round(s, 2) 라 경계 배치가 최대 0.005 m 위로 반올림된다
            tol = 0.006
            assert any(a - tol <= lo and hi <= b + tol for a, b in mspans), (
                f'{name}{i} frac={frac}: 체인 [{lo},{hi}] 가 2차로 부분구간 밖이다')
    assert checked > 0
