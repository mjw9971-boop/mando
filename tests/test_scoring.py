"""
score.py 채점 층 (2026 HL FMA 안내문) 단위 테스트 — 합성 rep 로 규칙만 검증.

규칙: 구간별 독립 100점 / 경미 -3·중대 -6 / (항목,구간)당 1회 / 경미+중대
공존 → 중대 하나 / 차로유지·실선 2회↑ → 중대 / 리스폰 구간당 1회 무료·초과
-6/회 누적 / 미완주 시 도달 구간까지만 집계 / 음수 허용.
"""
import copy

import pytest

import score as sc_mod   # noqa: E402 (conftest 가 tools 경로 추가)
from vtd_adapter.config import load_params_yaml


def base_cfg(**scoring_over):
    cfg = copy.deepcopy(load_params_yaml())
    # 구간 규칙 회귀 테스트가 기본 — 단일 구간 모드는 명시적으로 켠다
    cfg['scoring']['single_section'] = False
    cfg['scoring'].update(scoring_over)
    return cfg


def ev(s0, i0=0, **extra):
    return {'i0': i0, 'i1': i0, 'ticks': 1, 't0': 0.0, 't1': 0.0, 'dur_s': 0.0,
            's0': s0, 's1': s0, **extra}


def make_rep(events_by_cat, peak, total, done):
    return {'finish': {'peak_route_s': peak, 'route_total': total, 'done': done},
            'violations': {cat: {'count': len(evs), 'events': evs}
                           for cat, evs in events_by_cat.items()}}


def scored(events_by_cat, cfg, peak=None, total=300.0, done=True):
    rep = make_rep(events_by_cat, peak if peak is not None else total, total, done)
    sc_mod.annotate_scoring(rep, cfg)
    return sc_mod.score_run(rep, cfg)


# ── (항목, 구간) 1회 규칙 / 심화 ─────────────────────────────────────────
def test_same_item_twice_deducts_once():
    cfg = base_cfg(section_bounds_s=[])
    s = scored({'speed': [ev(10, 0, max_over_kph=5.0), ev(20, 1, max_over_kph=5.0)]}, cfg)
    assert s['sections'][0]['score'] == 97          # 경미 2건 → -3 한 번

def test_minor_plus_major_counts_as_one_major():
    cfg = base_cfg(section_bounds_s=[])
    s = scored({'speed': [ev(10, 0, max_over_kph=5.0), ev(20, 1, max_over_kph=25.0)]}, cfg)
    assert s['sections'][0]['score'] == 94          # -6 하나

def test_lane_departure_escalates_on_second():
    cfg = base_cfg(section_bounds_s=[])
    assert scored({'lane_departure': [ev(10)]}, cfg)['sections'][0]['score'] == 97
    s = scored({'lane_departure': [ev(10, 0), ev(20, 1)]}, cfg)
    assert s['sections'][0]['score'] == 94          # 경미 심화 → 중대 -6

def test_sections_are_independent():
    cfg = base_cfg(section_bounds_s=[100.0])
    s = scored({'lane_departure': [ev(10, 0), ev(150, 1)]}, cfg)
    assert [x['score'] for x in s['sections']] == [97, 97]   # 구간별 각 1회 경미

# ── 리스폰 특칙 ──────────────────────────────────────────────────────────
def test_respawn_one_free_then_minus6_each():
    cfg = base_cfg(section_bounds_s=[])
    s = scored({'reset': [ev(10, 0, why={}), ev(20, 1, why={}), ev(30, 2, why={})]}, cfg)
    assert s['sections'][0]['score'] == 88          # 3회 - 무료 1 = 2회 × -6

def test_negative_section_score_allowed():
    cfg = base_cfg(section_bounds_s=[])
    resets = [ev(10 + i, i, why={}) for i in range(20)]
    s = scored({'reset': resets}, cfg)
    assert s['sections'][0]['score'] == 100 - 19 * 6   # -14

# ── 미완주 집계 ──────────────────────────────────────────────────────────
def test_unreached_sections_excluded():
    cfg = base_cfg(section_bounds_s=[100.0, 200.0])
    s = scored({}, cfg, peak=150.0, total=300.0, done=False)
    assert s['reached_sections'] == 2
    assert s['max_possible'] == 200
    assert [x['score'] for x in s['sections']] == [100, 100, None]
    assert s['total'] == 200

# ── 안내문 예시: 3구간 80 / 80 / -150 → 총점 10 ─────────────────────────
def test_notice_example_three_sections_total_10():
    cfg = base_cfg(section_bounds_s=[100.0, 200.0],
                   minor_penalty=10, major_penalty=50)
    events = {
        'speed': [ev(10, 0, max_over_kph=5.0)],           # 구간0 경미 -10
        'lane_departure': [ev(20, 1), ev(150, 2)],        # 구간0 -10, 구간1 -10
        'solid_lane_change': [ev(160, 3, side='left', from_lane=[], to_lane=[],
                                 n_crossings=1)],         # 구간1 -10
        'reset': [ev(210 + i, 4 + i, why={}) for i in range(6)],  # 구간2: 5회 × -50
    }
    s = scored(events, cfg)
    assert [x['score'] for x in s['sections']] == [80, 80, -150]
    assert s['total'] == 10
    assert s['max_possible'] == 300

# ── severity 매핑 ────────────────────────────────────────────────────────
def test_speed_severity_thresholds():
    sc = load_params_yaml()['scoring']
    f = sc_mod._severity
    assert f('speed', {'max_over_kph': 0.5}, sc) == 'none'     # ≤1 허용
    assert f('speed', {'max_over_kph': 5.0}, sc) == 'minor'
    assert f('speed', {'max_over_kph': 25.0}, sc) == 'major'   # >20
    assert f('red_light', {}, sc) == 'major'
    assert f('stop_line_encroach', {}, sc) == 'minor'
    assert f('collision', {}, sc) == 'major'
    assert f('off_route', {}, sc) == 'none'                    # 매핑 미확정 항목

def test_section_idx_by_event_s0():
    cfg = base_cfg(section_bounds_s=[100.0])
    rep = make_rep({'speed': [ev(50, 0, max_over_kph=5.0),
                              ev(150, 1, max_over_kph=5.0)]}, 300.0, 300.0, True)
    sc_mod.annotate_scoring(rep, cfg)
    evs = rep['violations']['speed']['events']
    assert [e['section_idx'] for e in evs] == [0, 1]
    assert all(e['severity'] == 'minor' for e in evs)


# ── 단일 구간 모드 (scoring.single_section — 대회 CSV 확정 전 기본) ──────
def test_single_section_ignores_bounds():
    cfg = base_cfg(single_section=True, section_bounds_s=[100.0, 200.0])
    s = scored({'speed': [ev(10, 0, max_over_kph=5.0), ev(250, 1, max_over_kph=5.0)]}, cfg)
    assert len(s['sections']) == 1                     # 경계 무시, 전체 1구간
    assert s['sections'][0]['score'] == 97             # (항목)당 1회 — 경미 한 번


def test_single_section_escalates_across_whole_course():
    # 구간 모드에서는 구간별 각 1회 경미(-3×2)였던 배치가 전체 2회 → 중대 -6
    cfg = base_cfg(single_section=True, section_bounds_s=[100.0])
    s = scored({'lane_departure': [ev(10, 0), ev(150, 1)]}, cfg)
    assert s['sections'][0]['score'] == 94


def test_single_section_respawn_one_free_total():
    cfg = base_cfg(single_section=True, section_bounds_s=[100.0, 200.0])
    s = scored({'reset': [ev(10, 0, why={}), ev(150, 1, why={}), ev(250, 2, why={})]}, cfg)
    assert s['sections'][0]['score'] == 88             # 3회 − 전체 무료 1 = 2 × -6


def test_single_section_not_finished_aggregates_full_course():
    # 미완주여도 전체 1구간으로 집계 — peak 너머의 위반도 감점에 들어간다
    cfg = base_cfg(single_section=True)
    s = scored({'speed': [ev(250, 0, max_over_kph=5.0)]}, cfg,
               peak=150.0, total=300.0, done=False)
    assert s['reached_sections'] == 1 and s['max_possible'] == 100
    assert s['sections'][0]['score'] == 97


def test_single_section_hand_computed_total():
    # 손계산: 속도 경미 -3, 충돌 중대 -6, 차선이탈 2회 심화 -6,
    # 리스폰 2회(무료 1) -6 → 100 - 21 = 79
    cfg = base_cfg(single_section=True)
    s = scored({'speed': [ev(10, 0, max_over_kph=5.0)],
                'collision': [ev(50, 1, obj_id=1, min_gap_m=0.0)],
                'lane_departure': [ev(100, 2), ev(200, 3)],
                'reset': [ev(120, 4, why={}), ev(220, 5, why={})]}, cfg)
    assert s['total'] == 79 and s['max_possible'] == 100
