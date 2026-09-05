"""차선변경 여유 적합 — 검출(작업19-2)과 비용 모델(작업19-3).

hop 은 평행 차로로 옮겨 타는 것이라 advance() 가 진행거리를 0 으로 둔다.
그래서 탐색 비용은 "차선변경 3회를 100 m 에 나눠 함"과 "29.8 m 에 몰아넣음"을
구분하지 못했다. 대회장 실배포 CSV(venue_20260903)에서 실제로 후자가 총비용
0.54 m(0.26 %) 차이로 뽑혔고, 램프가 hop 하나만 블렌드하는 탓에(BACKLOG B-14)
경로 점열에 3.402 m · 6.800 m 계단이 남았다.

작업19-3 이 dijkstra 의 LC 분기에서 전이 하나가 목표 차로 안의 진행거리를
min_hop_gap_m 만큼 먹게 했다(s_enter 누적 + 부족분 페널티). venue 는 이제
섹션 0 의 70.5 m 에 3회를 나눠 하는 안을 고르고, 최대 계단이 2.083 m 로 준다.

검출 축도 같이 바뀌었다. cum 축(19-2 최초안)은 advance()=0 탓에 "29.8 m 에
3회"와 "70.5 m 에 3회"가 똑같이 0.0 으로 보여 **개선된 경로까지 버렸다**.
지금은 탐색과 같은 축(누적 필요거리 vs 차로 길이)으로 재고, 연쇄의 첫 hop 은
창 검사에 맡긴다.
"""
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

import build_route as BR                                        # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                     # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
VENUE = 'tests/fixtures/venue_20260903_waypoints.csv'
CLEAN = ['data/official_route.csv',
         'tests/fixtures/pair_lane_offset_waypoints.csv',
         'data/test_route_waypoints.csv',
         'waypoints.csv']


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


def _build(lg, path, cost_on=True):
    """CLI main() 과 같은 인자로 짓는다 (BACKLOG B-15 조사 규약).

    짝 해석 가드가 main() 에만 있어, 홀수 경유점 CSV 를 라이브러리로 그냥
    부르면 batch 와 다른 경로가 나온다.
    """
    rows = BR.read_waypoints_csv(str(ROOT / path))
    wps = [(r[1], r[2]) for r in rows]
    n = len(wps)
    ref = next((j for j in range(1, n)
                if math.hypot(wps[j][0] - wps[0][0], wps[j][1] - wps[0][1]) >= 2.0), 1)
    yaw = math.atan2(wps[ref][1] - wps[0][1], wps[ref][0] - wps[0][0])
    jsegs = BR.junction_segments(n) if (n % 2 == 0 and n > 2) else set()
    rt = BR.build_route(lg, wps, 8.0, yaw, junction_segs=jsegs,
                        seqs=[r[0] for r in rows],
                        finish_tail_m=BR.finish_tail_cfg())
    return rt[0] if isinstance(rt, tuple) else rt


def _violations(lg, rt):
    """연쇄 hop(nth>0) 중 차로 안에서 전이를 못 끝내는 것."""
    return [h for h in BR.hop_room(lg, rt) if h[6] > 0 and h[4] > h[5] + 1e-9]


# ── 상수 단일 출처 ────────────────────────────────────────────────────
def test_thresholds_come_from_params():
    from vtd_adapter.config import load_params_yaml
    rc = load_params_yaml()['route']
    assert BR.min_hop_gap_m() == pytest.approx(float(rc['min_hop_gap_m']))
    assert BR.hop_spacing_cost_enable() is bool(rc['hop_spacing_cost_enable'])


def test_untouched_constants():
    """작업19-3 은 기존 차선변경 비용 축을 건드리지 않는다."""
    assert BR.LC_PENALTY == 25.0
    assert BR.LC_MIN_CORRIDOR_M == 25.0
    assert BR.LC_SHORT_PENALTY_PER_M == 20.0


def test_advance_still_zero_for_hops(lg):
    """advance() 불변 — route_s 는 실주행거리와 일치해야 한다."""
    a, b = (418, 2, -2), (418, 2, -1)
    assert BR.is_lane_change_hop(lg, a, b)
    assert BR.advance(lg, a, b, lg.length(a)) == 0.0
    c = lg.successors(a)[0]
    assert BR.advance(lg, a, c, lg.length(a)) == pytest.approx(lg.length(a))


# ── 검출 축 ───────────────────────────────────────────────────────────
def test_hop_room_cursor_resets_on_successor(lg):
    """successor 전이에서 커서가 0 으로 돌아간다 — 새 차로에서 다시 시작."""
    rt = _build(lg, VENUE)
    rooms = BR.hop_room(lg, rt)
    firsts = [h for h in rooms if h[6] == 0]
    assert firsts, rooms
    for h in firsts:
        assert h[4] == pytest.approx(BR.min_hop_gap_m())


def test_chain_accumulates(lg):
    """연쇄 hop 은 누적 필요거리가 gap 씩 늘어난다."""
    rt = _build(lg, VENUE)
    chain = [h for h in BR.hop_room(lg, rt) if h[2][0] == 418 and h[2][1] == 0]
    assert len(chain) == 3, chain
    gap = BR.min_hop_gap_m()
    assert [h[6] for h in chain] == [0, 1, 2]
    assert [round(h[4], 3) for h in chain] == [gap, 2 * gap, 3 * gap]


def test_single_hop_into_short_lane_not_flagged(lg):
    """연쇄의 첫 hop 은 여기서 판정하지 않는다 (창 검사 담당).

    (2801,0,3)->(2801,0,2) 는 차로 17.07 m 로 gap 20 m 보다 짧지만 회랑이
    20.8 m 라 후행 차로에서 전이가 끝난다 — 점열 계단이 0 이다. 차로 길이만
    보고 첫 hop 까지 재면 이런 정상 경로를 버린다.

    **저장소 루트의 `waypoints.csv` 를 쓰지 않는다** (2026-09-05). 커밋
    883382d "루트 생성" 이 그 파일을 통째로 교체하면서 위 hop 이 사라져
    `assert short` 에서 실패했다 — 픽스처가 움직인 것이고 테스트도 코드도
    틀리지 않았다. 옛 내용을 보존한
    `tests/fixtures/waypoints_pair_banned.csv` 가 같은 hop 을 그대로 가진다
    (test_valid_entry_lanes 와 같은 파일 — 하나가 두 전제를 다 만족하므로
    나누지 않는다. 나누면 한쪽만 갱신되는 드리프트가 또 생긴다).
    """
    rt = _build(lg, 'tests/fixtures/waypoints_pair_banned.csv')
    short = [h for h in BR.hop_room(lg, rt) if h[4] > h[5] + 1e-9]
    assert short, '이 CSV 에 짧은 차로 hop 이 있어야 이 테스트가 유효하다'
    assert all(h[6] == 0 for h in short), short          # 전부 단발
    assert not _violations(lg, rt)


# ── 비용 모델 효과 ────────────────────────────────────────────────────
def test_venue_picks_spread_plan(lg):
    """venue 는 섹션 0(70.5 m)에 3회를 나눠 하는 안을 고른다."""
    rt = _build(lg, VENUE)
    lanes = [tuple(k) for k in rt['lanes']]
    assert (418, 0, -1) in lanes and (418, 2, -4) not in lanes
    assert not _violations(lg, rt)


def test_cost_switch_restores_old_plan(lg, monkeypatch):
    """route.hop_spacing_cost_enable=false 면 예전 안(29.8 m 에 3회)으로 돌아간다."""
    monkeypatch.setattr(BR, 'hop_spacing_cost_enable', lambda: False)
    rt = _build(lg, VENUE)
    lanes = [tuple(k) for k in rt['lanes']]
    assert (418, 2, -4) in lanes and (418, 0, -1) not in lanes
    bad = _violations(lg, rt)
    assert len(bad) == 2, bad
    for h in bad:
        assert h[2][0] == 418 and h[2][1] == 2       # 섹션 2, 29.8 m


def test_report_rc_follows_gate(lg, monkeypatch, capsys):
    """검사 게이트가 rc 를 가른다 — 예전 안에서만 유효한 비교다."""
    monkeypatch.setattr(BR, 'hop_spacing_cost_enable', lambda: False)
    rt = _build(lg, VENUE)
    base = BR.route_check_cfg()
    monkeypatch.setattr(BR, 'route_check_cfg', lambda: dict(base, hop_gap_enable=True))
    on = BR.report(lg, rt, 8.0)
    out_on = capsys.readouterr().out
    monkeypatch.setattr(BR, 'route_check_cfg', lambda: dict(base, hop_gap_enable=False))
    off = BR.report(lg, rt, 8.0)
    out_off = capsys.readouterr().out
    assert '앞 전이가 끝나기 전에 다음이 시작된다' in out_on
    assert '검사 꺼짐' in out_off
    assert on - off == 2


def test_report_prints_room_table(lg, capsys):
    """리포트 [4] 에 여유 적합 표가 나온다 (단발/연쇄 구분 포함)."""
    rt = _build(lg, VENUE)
    BR.report(lg, rt, 8.0)
    out = capsys.readouterr().out
    assert '차선변경 여유 적합' in out
    assert '단발' in out and '연쇄' in out


@pytest.mark.parametrize('csv', CLEAN)
def test_clean_routes_pass(lg, csv):
    """정상 기준 경로는 걸리지 않는다 (오탐 0)."""
    if not (ROOT / csv).exists():
        pytest.skip(f'{csv} 없음')
    assert not _violations(lg, _build(lg, csv))


def test_official_route_untouched(lg, monkeypatch):
    """대회 배포 경로는 비용 모델 on/off 로 달라지지 않는다."""
    a = _build(lg, 'data/official_route.csv')
    monkeypatch.setattr(BR, 'hop_spacing_cost_enable', lambda: False)
    b = _build(lg, 'data/official_route.csv')
    assert [tuple(k) for k in a['lanes']] == [tuple(k) for k in b['lanes']]
    assert a['total_length'] == pytest.approx(b['total_length'])
