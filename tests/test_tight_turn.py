"""연결로 회전 금지 완화 (2026-09-08) — route.banned_r_min_m.

명세:
  · 통행 금지는 R_min < route.banned_r_min_m (기본 3.0 m) 인 junction 연결로뿐.
    지도가 실도로 기반이라 연결로는 실차가 도는 길이고, 옛 임계(최소회전반경
    × vehicle.min_turn_margin = 5.65 m)의 근거였던 실측 이탈 4/4 는 커브 감속
    없이 25 km/h 로 진입한 결과였다. speed.curvature_cap 이 그 전제를 없앴다.
    규정상 "지정 경로 이탈 시 감점" 이라 **우회가 급회전보다 나쁘다**.
  · 3.0 ~ 5.65 는 '급회전': 통행하되 rt['tight_turns'] 에 기록하고 리포트 [5]
    가 WARN 을 낸다 (ERROR 아님 — rc 에 반영되지 않는다).
  · 금지 임계는 dijkstra · 짝 대조 채널 · route_check [5] 가 **같은 함수**
    (infeasible_connectors)에서 받는다. 두 벌 금지.
"""
import io
import contextlib
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

import build_route as BR                                        # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                     # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CSV = 'waypoints.csv'          # 10점 — 짝 4개, seq 4→5 가 (2506,0,-1) R 5.01


@pytest.fixture(scope='module')
def lg():
    return LaneGraph(str(GRAPH))


@pytest.fixture(autouse=True)
def _fresh_cfg():
    """route_cfg 캐시를 매 테스트 앞뒤로 되돌린다 (임계를 갈아끼우는 테스트가 있다)."""
    BR.route_cfg(reload=True)
    yield
    BR.route_cfg(reload=True)


def _build(lg, csv, radius=8.0):
    rows = BR.read_waypoints_csv(str(ROOT / csv))
    seqs = [r[0] for r in rows]
    wps = [(r[1], r[2]) for r in rows]
    yaw = BR.math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    jsegs = BR.junction_segments(len(wps), 1)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        return BR.build_route(lg, wps, radius, yaw, junction_segs=jsegs, seqs=seqs,
                              finish_tail_m=BR.finish_tail_cfg())


# ── 임계 ────────────────────────────────────────────────────────────────
def test_default_threshold_is_3m():
    """params 값이 정본이다. 기본 3.0."""
    assert load_params_yaml()['route']['banned_r_min_m'] == pytest.approx(3.0)
    assert BR.banned_r_min_m() == pytest.approx(3.0)


def test_tight_thr_is_geometric_min_radius():
    """급회전 상한은 여전히 기하 최소회전반경 × min_turn_margin (≈5.65 m)."""
    r_need, margin = BR.min_turn_radius_m()
    assert BR.tight_turn_r_m() == pytest.approx(r_need * margin)
    assert BR.tight_turn_r_m() > BR.banned_r_min_m()


def test_banned_set_shrinks(lg):
    """금지 38개(옛 5.65) → 4개(새 3.0). 남는 넷은 전부 R < 3.0 이다."""
    banned, thr = BR.infeasible_connectors(lg)
    assert thr == pytest.approx(3.0)
    assert all(r < 3.0 for r in banned.values())
    # 9_school_route 실측으로 off_route 를 낸 연결로는 계속 금지된다
    assert (1576, 0, -1) in banned


def test_banned_threshold_is_configurable(lg):
    """route.banned_r_min_m 을 5.65 로 올리면 옛 동작으로 돌아간다 (롤백 경로)."""
    small, _ = BR.infeasible_connectors(lg)
    BR._ROUTE_CFG['banned_r_min_m'] = 5.65
    big, thr = BR.infeasible_connectors(lg)
    assert thr == pytest.approx(5.65)
    assert set(small) < set(big)


# ── 급회전 기록 ──────────────────────────────────────────────────────────
def test_tight_turns_recorded(lg):
    """경로가 급회전 연결로를 타면 rt['tight_turns'] 에 남는다."""
    rt = _build(lg, CSV)
    tt = rt['tight_turns']
    assert [tuple(t['lane']) for t in tt] == [(2506, 0, -1)]
    assert tt[0]['r_min_m'] == pytest.approx(5.01, abs=0.05)
    # 기록된 s 는 **경로 누적거리** — 그 자리 차로가 실제로 그 연결로여야 한다
    i = min(range(len(rt['cum_s'])), key=lambda j: abs(rt['cum_s'][j] - tt[0]['s_m']))
    assert rt['lanes'][i] == (2506, 0, -1)
    assert rt['tight_turn_thr_m'] == pytest.approx(BR.tight_turn_r_m())


def test_tight_turns_band_is_half_open(lg):
    """기록 구간은 [banned_thr, tight_thr) — 금지된 것도, 여유 있는 것도 안 들어간다."""
    rt = _build(lg, CSV)
    lo, hi = BR.banned_r_min_m(), rt['tight_turn_thr_m']
    for t in rt['tight_turns']:
        assert lo <= t['r_min_m'] < hi


def test_no_detour_with_relaxed_threshold(lg):
    """완화의 값어치 — 옛 임계는 같은 CSV 를 988 m 우회시킨다.

    (2506,0,-1) R 5.01 하나를 피하려고 교차로 3개를 더 돌았다. 주최측 규정은
    "지정 경로 이탈 시 감점" 이라 이쪽이 더 나쁘다.
    """
    new = _build(lg, CSV)
    BR._ROUTE_CFG['banned_r_min_m'] = 5.65
    old = _build(lg, CSV)
    assert new['total_length'] < old['total_length'] - 500.0
    assert new['total_length'] == pytest.approx(2296.1, abs=5.0)


# ── 리포트 ───────────────────────────────────────────────────────────────
def test_report_warns_not_errors(lg, capsys):
    """급회전은 WARN 이다 — rc 를 올리지 않는다 (경로가 폐기되면 안 된다)."""
    rt = _build(lg, CSV)
    assert rt['tight_turns'], '이 CSV 는 급회전을 하나 타야 한다'
    rc = BR.report(lg, rt, 8.0, None)
    out = capsys.readouterr().out
    assert '급회전 연결로' in out and '감속 진입' in out
    assert '⚠ 회전 불가 기하' not in out
    assert rc == 0


def test_report_header_shows_both_thresholds(lg, capsys):
    """[5] 머리가 금지 임계와 급회전 대역을 둘 다 밝힌다 (현장에서 눈으로 읽는다)."""
    rt = _build(lg, CSV)
    BR.report(lg, rt, 8.0, None)
    out = capsys.readouterr().out
    assert 'route.banned_r_min_m' in out
    assert '급회전' in out


def test_report_counts_tight_turns_once(lg, capsys):
    """한 연결로에 회전 이벤트가 여럿 걸려도 급회전 집계는 rt 기록 개수만큼이다."""
    rt = _build(lg, CSV)
    BR.report(lg, rt, 8.0, None)
    out = capsys.readouterr().out
    n = sum(1 for ln in out.splitlines() if ln.lstrip().startswith('[경고]')
            and '급회전 연결로' in ln)
    assert n == len(rt['tight_turns'])


# ── 커브 감속과의 연동 ───────────────────────────────────────────────────
def test_curvature_cap_covers_tight_band():
    """급회전 대역 전체에서 커브 감속 상한이 살아 있어야 한다.

    v_cap = √(a_lat·R) 이고, κ = 1/R 이 curvature_min_kappa 를 넘어야 후보가
    생긴다. 대역 상한 5.65 m 에서도 κ 0.177 ≫ 0.005 이라 항상 잡힌다 —
    "금지를 풀었는데 감속이 안 걸리는" 구멍이 없다는 확인이다.
    """
    sp = load_params_yaml()['speed']
    assert sp['curvature_cap_enable'] is True
    a_lat = BR.curvature_a_lat_max_m_s2()
    assert a_lat == pytest.approx(sp['curvature_a_lat_max'])
    for r in (BR.banned_r_min_m(), 4.0, 5.0, BR.tight_turn_r_m()):
        assert 1.0 / r > sp['curvature_min_kappa']
    # R 5.0 → 3.5 m/s (12.6 km/h) 급
    assert BR.math.sqrt(a_lat * 5.0) == pytest.approx(3.54, abs=0.05)
