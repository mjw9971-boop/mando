"""
접근로 정지선 합성 (build_lane_graph.assign_objects, synth_stopline).

2026-09-02 조사: 신호가 붙었는데 같은 방향 정지선이 없는 접근로가 4개 남았다
(road 1928 dir-1 · 2575 · 2806 · 3142). 정지선이 없으면 route.collect_stops 가
traffic_lights 를 만들지 않아 next_traffic_light 이 None 이 되고, 제어기의
적신호 IDM 이 호출조차 되지 않는다 — road 2819 와 같은 실패다.

합성 위치의 근거: 정상 매칭된 신호 625개에서 (정지선 s - 신호 s) 는 최빈
+0.15 / -0.15, p1 -0.158 · p50 +0.146 · p99 +0.531 이고, |offset| > 1 m 은
3건뿐이다. 신호 자신은 중앙값 기준 도로 끝에 정확히 선다. 이 지도에서
"신호 s ± 0.15" 는 규약이므로 합성 오차는 최대 0.5 m 다.

넷 다 하류 연결로에 폭이 맞는 정지선이 14~29 m 떨어져 하나씩 있지만, 그걸
끌어오는 안(B2)은 채택하지 않았다 — 규약의 40~190배 밖이라 접근로 정지선이
아니고, 별개 지물일 가능성을 배제하지 못했다.
"""
import pytest

XODR = 'data/HL_FMA_VTD_LivingLab.xodr'
SYNTH_ROADS = {1928, 2575, 2806, 3142}
# 하류 연결로의 정지선 — B2 를 채택하지 않았으므로 그대로 있어야 한다
DOWNSTREAM = {3372, 2584, 3450, 3167}


def _mod():
    import sys
    sys.path.insert(0, 'tools')
    import build_lane_graph as B
    return B


@pytest.fixture(scope='module')
def on():
    return _mod().build(XODR, ds=0.5)


@pytest.fixture(scope='module')
def off():
    return _mod().build(XODR, ds=0.5, synth_stopline=False)


# ── 합성 결과 ────────────────────────────────────────────────────────────
def test_four_stoplines_synthesized(on):
    syn = on['meta']['stop_synthesized']
    assert len(syn) == 4
    assert {x['road'] for x in syn} == SYNTH_ROADS


def test_position_follows_the_map_convention(on):
    """s = 신호 s - 0.15*dir. 지도 자신의 규약이다."""
    for x in on['meta']['stop_synthesized']:
        assert x['s'] == pytest.approx(x['signal_s'] - 0.15 * x['dir'], abs=1e-6)


def test_synthesized_at_the_road_end(on):
    """대상 신호는 전부 그 방향 진행 끝에 서 있다 — 중간에 만들지 않았다."""
    for x in on['meta']['stop_synthesized']:
        assert x['road_end_gap_m'] < 0.05


def test_never_inside_a_junction(on):
    """교차로 내부(junction != -1)에는 만들지 않는다 — 정지선을 이미 지난 지점이다."""
    for x in on['meta']['stop_synthesized']:
        assert on['roads'][x['road']]['junction'] == -1


def test_direction_with_existing_stopline_is_untouched(on):
    """그 방향에 정지선이 있으면 건드리지 않는다.

    road 1928 은 dir+1 에 정지선이 있고 dir-1 에만 없다 — dir-1 만 합성된다.
    road 2575 는 그 반대다.
    """
    syn = {x['road']: x['dir'] for x in on['meta']['stop_synthesized']}
    assert syn[1928] == -1
    assert syn[2575] == 1
    for rid, d in ((1928, 1), (2575, -1)):
        cl = [c for c in on['roads'][rid]['stop_clusters'] if c['dir'] == d]
        assert cl, f'road {rid} dir {d} 의 기존 정지선이 사라졌다'


def test_synthesized_stoplines_carry_controllers(on):
    """합성 정지선에 그 접근로의 controller 가 붙는다."""
    want = {1928: 89, 2575: 213, 2806: 215, 3142: 221}
    for x in on['meta']['stop_synthesized']:
        cid = want[x['road']]
        for k in (tuple(y) for y in x['lanes']):
            hit = [s for s in on['lanes'][k]['stop_lines']
                   if cid in (s.get('controller_ids') or [])]
            assert len(hit) == 1, f'{k} 에 controller {cid} 정지선이 {len(hit)}개'
            # 합성 위치(도로 s)가 그 차로의 주행 s 로 옳게 변환됐는지
            assert 0.0 <= hit[0]['s'] <= on['lanes'][k]['length'] + 1e-6


def test_downstream_connecting_stoplines_kept(on):
    """B2 를 채택하지 않았다 — 하류 연결로 정지선은 그대로 둔다."""
    for rid in DOWNSTREAM:
        assert on['roads'][rid]['stop_clusters'], f'road {rid} 의 정지선이 사라졌다'


# ── 목표 지표 ────────────────────────────────────────────────────────────
def test_metrics_reach_zero(on):
    assert on['meta']['lanes_signal_no_stopline'] == 0
    assert on['meta']['sig_no_stopline'] == 0


def test_switch_restores_previous_behaviour(off):
    """--no-synth-stopline 으로 끄면 이전 동작으로 돌아간다."""
    assert off['meta']['stop_synthesized'] == []
    assert off['meta']['stop_synth_enabled'] is False
    assert off['meta']['lanes_signal_no_stopline'] == 11
    assert off['meta']['sig_no_stopline'] == 15
    # 재귀속(작업 A)은 스위치와 무관하게 살아 있다
    assert len(off['meta']['signal_reassigned']) == 6


def test_switch_changes_nothing_else(on, off):
    """합성 말고는 건드리지 않는다 — 기존 정지선 클러스터 수가 같다."""
    assert on['meta']['stop_repaired'] == off['meta']['stop_repaired'] == 24
    for rid in DOWNSTREAM | {1928, 2575}:
        a = [(c['dir'], round(c['s'], 3)) for c in off['roads'][rid]['stop_clusters']]
        b = [(c['dir'], round(c['s'], 3)) for c in on['roads'][rid]['stop_clusters']]
        assert set(a) <= set(b), f'road {rid} 의 기존 클러스터가 바뀌었다'


# ── 상류 가시성 (route.collect_stops 재현) ───────────────────────────────
def _collect_stops(g, chain):
    """route.py collect_stops 와 같은 규칙 — controller/signal 이 붙은 정지선만."""
    lanes, cum, acc = g['lanes'], [], 0.0
    for k in chain:
        cum.append(acc)
        acc += lanes[k]['length']
    return [(round(cum[i] + sl['s'], 2), tuple(sl['controller_ids']))
            for i, k in enumerate(chain)
            for sl in lanes[k]['stop_lines']
            if sl.get('controller_ids') or sl.get('signal_ids')]


def test_2575_section2_visible_from_upstream(on):
    """2575 section 2 는 4.07 m 뿐이다 — 상류에서 정지선이 보여야 한다.

    (2575,2,-2) 는 xodr 자체에 predecessor 가 없는 우회전 포켓이라 상류가 10 m
    밖에 안 된다. 실제 경로는 이웃 (2575,2,-3) 으로 들어오고, 그 계통은 상류
    100 m 를 확보한다. 같은 합성 정지선이 두 차로 모두에 붙는다.
    """
    lanes = on['lanes']
    chain = [(2575, 2, -3)]
    total = lanes[chain[0]]['length']
    seen = set(chain)
    while total < 250:
        prv = [k for k in lanes[chain[0]]['prev'] if k not in seen]
        if not prv:
            break
        k = sorted(prv)[0]
        chain.insert(0, k)
        seen.add(k)
        total += lanes[k]['length']
    got = _collect_stops(on, chain)
    assert any(213 in c for _, c in got), '합성 정지선이 collect_stops 에 안 잡힌다'
    rs = [s for s, c in got if 213 in c][0]
    assert rs > 90.0, f'상류 여유가 {rs:.1f} m 뿐이다'
    # 포켓 차로에도 같은 정지선이 붙어 있다
    assert any(213 in (sl.get('controller_ids') or [])
               for sl in lanes[(2575, 2, -2)]['stop_lines'])
