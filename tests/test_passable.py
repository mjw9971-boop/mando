"""장애물 배치의 통과 가능성 보장 (gen_placement.require_passable, 2026-09-07).

배치가 "자차가 **합법적으로** 지나갈 경로가 최소 하나 존재" 를 만족하는지
놓은 뒤 검증한다. 합법 = 실선·중앙선을 넘지 않고, 같은 방향 주행차로가 실재
하며 폭이 되고, 연쇄 장애물이면 그 구간 **내내** 같은 쪽이 살아 있는 것.

기존 게이트(_same_dir_lane_count)는 차로 개수·폭만 봤다 — 차선이 실선인지,
회피 측이 구간 끝까지 유지되는지는 아무도 안 봤다.

실측 근거 (2026-09-07, 21개 세트 · 장애물 119개):
  · 점 단위로는 불가 0건. 구간으로 묶어 보면 1건 —
    실전주행_교통류_18_연속교차로14 의 둘째 체인이 "구간 내내 유지되는 회피
    측 없음" (좌측은 실선으로 끊기고 우측은 이웃이 없다).
  · 검사를 켠 재생성에서 그 체인만 옮겨 갔고 나머지 20/21 은 바이트 동일,
    장애물 총수는 119 로 불변이다.
"""
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import gen_scenarios as gs                                      # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
SC = ROOT / 'scenarios' / '실전주행_교통류'
CFG = load_params_yaml()
PLC = CFG['gen_placement']


# ── 스위치·상수 ──────────────────────────────────────────────────────────
def test_params_present_and_defaults():
    assert PLC['require_passable'] is True          # 통과 보장은 기본 on
    assert PLC['pass_margin_enable'] is False       # 잔여폭 검사는 기본 off
    assert float(PLC['pass_margin_m']) == 0.5
    assert float(PLC['chain_gap_m']) == 18.0


def test_chain_gap_is_not_hardcoded():
    """체인 간격은 params 가 단일 출처다 (2026-09-07 이전 18.0 하드코딩)."""
    src = (ROOT / 'tools' / 'gen_scenarios.py').read_text(encoding='utf-8')
    body = src[src.index('def ev_obstacle_chain'):src.index('def ev_narrow')]
    assert 'spacing = 18.0' not in body
    assert 'pass_cfg()' in body                     # spacing 을 params 에서 받는다
    assert gs.pass_cfg()[1] == float(PLC['chain_gap_m'])


def test_pass_cfg_reads_neighbour_width_from_existing_key():
    """이웃 폭 임계는 static_vehicle.min_neighbor_width_m 재사용 (값 이중화 금지)."""
    assert gs.pass_cfg()[4] == float(PLC['static_vehicle']['min_neighbor_width_m'])


def test_pass_margin_need_is_vehicle_width_plus_margin():
    need = gs.pass_cfg()[3]
    assert need == pytest.approx(CFG['vehicle']['width'] + PLC['pass_margin_m'])


# ── 합법성 판정 (대역 lane_graph) ────────────────────────────────────────
LANE = (1, 0, -1)
NB = (1, 0, -2)


class FakeLG:
    """자기 차로 + 우측 이웃 하나. 마크/폭/이웃을 시험마다 갈아 끼운다."""

    def __init__(self, mark='broken', nb_w=3.0, has_nb=True,
                 left_is_center=False, nb_dir=1, nb_type='driving'):
        self.mark, self.nb_w, self.has_nb = mark, nb_w, has_nb
        self.lanes = {
            LANE: {'dir': 1, 'type': 'driving', 'length': 200.0,
                   'left_is_center': left_is_center},
            NB: {'dir': nb_dir, 'type': nb_type, 'length': 200.0,
                 'left_is_center': False},
        }

    def neighbor(self, key, side):
        return NB if (side == 'right' and self.has_nb) else None

    def mark_at(self, key, s, side):
        m = self.mark(s) if callable(self.mark) else self.mark
        return (m, 'standard', m == 'broken')

    def width_at(self, key, s):
        return self.nb_w if key == NB else 3.0


def rt(total=200.0):
    return {'lanes': [LANE], 'cum_s': [0.0], 'lengths': [total],
            'total_length': total}


def ctx(lg, total=200.0):
    return types.SimpleNamespace(lg=lg, route=types.SimpleNamespace(rt=rt(total)))


def test_solid_line_is_not_a_legal_escape():
    ok, why = gs._lc_legal_at(FakeLG(mark='solid'), rt(), 50.0, 'right', 2.5)
    assert ok is False and why == '실선'


def test_dashed_line_is_legal():
    ok, why = gs._lc_legal_at(FakeLG(mark='broken'), rt(), 50.0, 'right', 2.5)
    assert ok is True and why == ''


def test_missing_neighbour_is_not_legal():
    ok, why = gs._lc_legal_at(FakeLG(has_nb=False), rt(), 50.0, 'right', 2.5)
    assert ok is False and why == '이웃없음'


def test_centre_line_is_not_legal():
    """중앙선은 점선이어도 넘지 않는다 — 좌측만 해당."""
    lg = FakeLG(left_is_center=True)
    lg.neighbor = lambda key, side: NB          # 좌측에도 이웃이 있다고 두고
    ok, why = gs._lc_legal_at(lg, rt(), 50.0, 'left', 2.5)
    assert ok is False and why == '중앙선'


def test_narrow_neighbour_is_not_legal():
    ok, why = gs._lc_legal_at(FakeLG(nb_w=2.0), rt(), 50.0, 'right', 2.5)
    assert ok is False and why == '이웃폭부족'


def test_opposite_direction_neighbour_is_not_legal():
    ok, why = gs._lc_legal_at(FakeLG(nb_dir=-1), rt(), 50.0, 'right', 2.5)
    assert ok is False and why == '반대방향/비주행'


# ── 구간 유지 검사 ───────────────────────────────────────────────────────
def test_escape_side_holds_over_whole_span():
    assert gs._escape_side_over(ctx(FakeLG()), 10.0, 100.0, 2.5) == 'right'


def test_escape_side_none_when_broken_midway():
    """구간 중간에 실선이 끼면 그 측은 못 쓴다 — 18_연속교차로14 가 이 모양이다."""
    lg = FakeLG(mark=lambda s: 'solid' if 40.0 <= s <= 60.0 else 'broken')
    assert gs._escape_side_over(ctx(lg), 10.0, 100.0, 2.5) is None
    # 끊긴 구간을 피하면 다시 성립한다
    assert gs._escape_side_over(ctx(lg), 65.0, 100.0, 2.5) == 'right'


def test_escape_side_samples_finely_enough():
    """표본 간격 2 m — 이 맵의 최단 차로 섹션(1.6~1.8 m)을 건너뛰지 않는다."""
    lg = FakeLG(mark=lambda s: 'solid' if 49.0 <= s <= 51.0 else 'broken')
    assert gs._escape_side_over(ctx(lg), 10.0, 100.0, 2.5) is None


def test_escape_side_clamps_to_route_end():
    """구간이 경로 밖으로 넘어가도 lane_at 이 터지지 않는다."""
    assert gs._escape_side_over(ctx(FakeLG(), 80.0), 60.0, 500.0, 2.5) == 'right'


# ── 폐기 집계 ────────────────────────────────────────────────────────────
def test_reject_is_counted_and_typed():
    gs.PASS_STATS['reject'] = 0
    gs.PASS_STATS['why'].clear()
    e = gs._pass_reject('obstacle_chain', '사유')
    assert isinstance(e, gs.EventUnfeasible)        # 위치 재시도가 받는 예외
    assert gs.PASS_STATS['reject'] == 1
    assert gs.PASS_STATS['why']['obstacle_chain: 사유'] == 1


def test_inlane_free_edge_formula():
    """가장자리 장애물의 반대쪽 잔여폭 = w/2 + (w/2 − inset) − 반폭."""
    lg = FakeLG()
    got = gs._inlane_free_edge(lg, rt(), 50.0, 0.7, 0.16)
    assert got == pytest.approx(3.0 / 2 + (3.0 / 2 - 0.7) - 0.16)


# ── 실제 경로 ────────────────────────────────────────────────────────────
@pytest.mark.skipif(not GRAPH.exists(), reason='data/lane_graph.pkl 없음')
@pytest.mark.skipif(not SC.exists(), reason='scenarios/ 없음 (gitignore 대상)')
def test_generated_chains_have_a_side_that_survives_the_span():
    """21개 세트의 모든 체인이 구간 내내 유지되는 회피 측을 갖는다."""
    import glob
    import io
    import contextlib
    import yaml
    from vtd_adapter.lanegraph import LaneGraph
    lg = LaneGraph(str(GRAPH))
    need = gs.pass_cfg()[4]
    ot = CFG['overtake']
    bad = []
    for f in sorted(glob.glob(str(SC / '*.yaml'))):
        d = yaml.safe_load(open(f, encoding='utf-8'))
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            route = gs._build_from_rows(lg, 'x', [tuple(r) for r in d['route']['rows']])
        c = types.SimpleNamespace(lg=lg, route=route)
        for ev in d.get('events') or []:
            if ev.get('kind') != 'obstacle_chain':
                continue
            ss = sorted(ev.get('route_s') or [])
            if len(ss) < 2:
                continue
            lo = ss[0] - float(ot['shift_ahead_m'])
            hi = ss[-1] + float(ot['extra_after_m']) + float(CFG['vehicle']['length'])
            if gs._escape_side_over(c, lo, hi, need) is None:
                bad.append((d['name'], ss[0], ss[-1]))
    assert bad == [], f'구간 내내 유지되는 회피 측이 없는 체인: {bad}'
