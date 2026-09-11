"""
ctrl24 — 이웃 차로 스캔 + side 선택 (2026-09-10).

여기서 지키는 불변:
  · _neighbor_free_m: 목표 없음 → None / 빈 차로 → scan_len / 앞 10 m 정지차 → 10.0 / walker 무시.
  · 스캔은 side 선택에만 쓴다. min() 후보를 추가하지 않는다.
  · free < need 인 side 는 nb_short 로 빠지고, 남은 쪽을 free 내림차순으로 시도한다.
    동률(0.5 m 이내)은 좌측 우선 = 기존 순서.
  · 기본 off. off 면 스캔을 아예 안 부르고 ('left','right') 순서 그대로다.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml
from vtd_adapter.control import VtdLongitudinalController

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from ctrl24 import Ctrl24                                        # noqa: E402
from test_avoid import Ap                                        # noqa: E402
from test_ctrl24_avoid import LANE, ShiftPlanner, apply, car     # noqa: E402
from test_ctrl24_long import Walker                              # noqa: E402

CFG = load_params_yaml(PARAMS_YAML)
C = CFG['ctrl24']


class Loc:
    def __init__(self, x, y):
        self.x, self.y, self.z = float(x), float(y), 0.0


class WP:
    def __init__(self, x, y):
        self.transform = type('T', (), {'location': Loc(x, y)})()


class NbPlanner(ShiftPlanner):
    """이웃 차로 웨이포인트를 좌표로 돌려주는 목 — 좌 +y·우 −y, 한 칸 LANE.
    gaps[side] 구간(인덱스)에는 목표가 없다."""

    def __init__(self, gaps=None, **kw):
        super().__init__(**kw)
        self.gaps = {'left': [], 'right': [], **(gaps or {})}

    def _shift_target_wp(self, idx, left, n_steps):
        side = 'left' if left else 'right'
        if not self.has[side] or any(a <= idx < b for a, b in self.gaps[side]):
            return None
        return WP(idx / 10.0, (LANE if left else -LANE) * max(1, int(n_steps)))


def on_cfg(**kw):
    c = copy.deepcopy(CFG)
    c['ctrl24']['neighbor_scan_enable'] = True
    c['ctrl24'].update(kw)
    return c


def rig(cfg=CFG, actors=(), gaps=None, left=True, right=True):
    p = NbPlanner(gaps=gaps, left=left, right=right, d_tl=float('inf'))
    ap = Ap(p, list(actors))
    ap._longitudinal_controller = VtdLongitudinalController(cfg)
    kr = Ctrl24(cfg)
    kr._sl_all = []
    kr._ap = ap
    return kr, p, ap


def scan_len(v):
    return min(C['neighbor_scan_max_m'], max(C['neighbor_scan_min_m'], C['neighbor_scan_k'] * v))


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_params_defaults():
    """킬스위치만 불변이다 — k·min·max 는 튜닝 대상이라 값을 고정하지 않는다
    (params 가 정본, CLAUDE.md B-25 관례). 관계식만 본다."""
    assert C['neighbor_scan_enable'] is False
    assert C['neighbor_scan_k'] > 0.0
    assert 0.0 < C['neighbor_scan_min_m'] <= C['neighbor_scan_max_m']
    assert C['neighbor_scan_max_m'] <= C['detect_max_m']            # GT 범위 안


# ── _neighbor_free_m 단위 ─────────────────────────────────────────────────
def test_free_none_without_target():
    kr, p, ap = rig(on_cfg(), left=False)
    assert kr._neighbor_free_m(p, True, 8.0, ap=ap) is None
    assert kr._neighbor_free_m(p, False, 8.0, ap=ap) == pytest.approx(scan_len(8.0))


def test_free_is_scan_len_when_empty():
    """clip(k·v, min, max) — 산술을 보려고 사본에 값을 명시로 주입한다 (B-25 관례)."""
    cfg = on_cfg(neighbor_scan_k=4.0, neighbor_scan_min_m=20.0, neighbor_scan_max_m=80.0)
    kr, p, ap = rig(cfg)
    assert kr._neighbor_free_m(p, True, 0.0, ap=ap) == pytest.approx(20.0)      # 하한
    assert kr._neighbor_free_m(p, True, 8.0, ap=ap) == pytest.approx(32.0)      # 4·8
    assert kr._neighbor_free_m(p, True, 30.0, ap=ap) == pytest.approx(80.0)     # 상한


def test_free_is_distance_to_first_stopped_car():
    kr, p, ap = rig(on_cfg(), actors=[car(2, 10.0, LANE), car(3, 25.0, LANE)])
    assert kr._neighbor_free_m(p, True, 8.0, ap=ap) == pytest.approx(10.0, abs=0.15)
    assert kr._neighbor_free_m(p, False, 8.0, ap=ap) == pytest.approx(scan_len(8.0))  # 우측은 비었다


def test_free_ignores_walkers_and_moving_cars():
    kr, p, ap = rig(on_cfg(), actors=[Walker(9, 10.0, LANE), car(3, 12.0, LANE, speed=5.0)])
    assert kr._neighbor_free_m(p, True, 8.0, ap=ap) == pytest.approx(scan_len(8.0))


def test_free_ignores_objects_outside_scan_width():
    kr, p, ap = rig(on_cfg(), actors=[car(2, 10.0, LANE + 3.0)])        # 목표 차로에서 3 m 옆
    assert kr._neighbor_free_m(p, True, 8.0, ap=ap) == pytest.approx(scan_len(8.0))


# ── side 선택 ────────────────────────────────────────────────────────────
def test_off_keeps_left_first_and_never_scans():
    kr, p, ap = rig(actors=[car(2, 60.0), car(3, 10.0, LANE)])
    calls = []
    kr._neighbor_free_m = lambda *a, **k: calls.append(1)
    apply(kr, ap, v=8.0)
    assert not calls and kr.last_avoid['shift'] == 'left' and 'nb_free' not in kr.last_avoid


def test_short_left_is_rejected_and_right_is_chosen():
    """좌측 이웃에 10 m 앞 정지차 → free 10 < need → nb_short → 우측."""
    kr, p, ap = rig(on_cfg(), actors=[car(2, 60.0), car(3, 10.0, LANE)])
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['shift'] == 'right'
    assert any(r.startswith('left:nb_short(') for r in a['rejects'])
    assert a['nb_free']['order'] == ['right']
    assert a['nb_free']['need'] == pytest.approx(kr._trans_m(8.0) + C['shift_ahead_m'] + C['extra_before_m'], abs=0.1)


def test_longer_right_goes_first_but_tie_keeps_left():
    kr, p, ap = rig(on_cfg(), actors=[car(2, 60.0), car(3, 30.0, LANE)])     # 좌 30 m ≥ need, 우 비어 32 m
    apply(kr, ap, v=8.0)
    assert kr.last_avoid['nb_free']['order'] == ['right', 'left'] and kr.last_avoid['shift'] == 'right'
    kr2, p2, ap2 = rig(on_cfg(), actors=[car(2, 60.0)])                       # 양쪽 동률 → 좌
    apply(kr2, ap2, v=8.0)
    assert kr2.last_avoid['nb_free']['order'] == ['left', 'right'] and kr2.last_avoid['shift'] == 'left'


def test_both_short_is_noop():
    kr, p, ap = rig(on_cfg(), actors=[car(2, 60.0), car(3, 10.0, LANE), car(4, 12.0, -LANE)])
    before = p.route_points.copy()
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['state'] == 'NOOP' and kr.ot_span is None
    assert all(':nb_short(' in r for r in a['rejects']) and len(a['rejects']) == 2
    assert (p.route_points == before).all()


def test_no_target_side_is_labelled_and_other_side_used():
    kr, p, ap = rig(on_cfg(), actors=[car(2, 60.0)], left=False)
    apply(kr, ap, v=8.0)
    a = kr.last_avoid
    assert a['shift'] == 'right' and 'left:no_target' in a['rejects']


def test_scan_does_not_add_a_candidate():
    kr, p, ap = rig(on_cfg(), actors=[car(2, 60.0), car(3, 10.0, LANE)])
    apply(kr, ap, v=8.0)
    # 2026-09-11 K10 curvature 추가로 후보 키가 9 → 10 개 (값은 여기서 null).
    assert set(kr.last_kr) == {'stop_profile', 'stop_hold', 'rtor_cap', 'ped_intent',
                               'crosswalk', 'red_zone', 'shift_cap', 'span_v_req',
                               'virtual_cap', 'curvature'}
