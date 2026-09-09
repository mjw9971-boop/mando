"""DP 종점 차로 고정 (route.dp_finish_lane_lock, 2026-09-07).

왜 완주가 걸린 문제인가: 대회 규칙은 "뒷축이 **두 콘 사이** 종료선 통과" 이고
콘은 종료 좌표에서 그 차로 t 방향으로 선다. 경로가 종료 좌표의 옆 차로로
끝나면 종료선을 넘고도 콘 밖이라 완주로 안 쳐질 수 있다.

실측(2026-09-07, 고유 경로 28개 · finish_tail 12 m 적용):
  차로불일치 10 → 2, 콘의 회랑 침범 6 → 4.
남은 2건은 종료 좌표가 **어느 주행 차로 경계 밖**이라 고정 대상이 아니다.
"""
import contextlib
import csv
import io
import math
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))

import build_route as BR                                        # noqa: E402
from finish_cone import finish_gate                             # noqa: E402
from vtd_adapter.config import load_params_yaml                 # noqa: E402
from vtd_adapter.lanegraph import LaneGraph                     # noqa: E402

GRAPH = ROOT / 'data' / 'lane_graph.pkl'
CFG = load_params_yaml()
SC = ROOT / 'scenarios' / '실전주행_교통류'


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음 (gitignore 대상)')
    return LaneGraph(str(GRAPH), cfg=CFG)


@contextlib.contextmanager
def lock(on):
    old = BR._DP_CFG
    BR._DP_CFG = BR.dp_cfg(reload=True)._replace(finish_lock=on)
    try:
        yield
    finally:
        BR._DP_CFG = old


def build(lg, csv_path):
    rows = [(int(r[0]), float(r[1]), float(r[2]))
            for r in csv.reader(open(csv_path, encoding='utf-8-sig'))
            if r and r[0].strip() != 'seq']
    wps = [(x, y) for _s, x, y in rows]
    yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
        return BR.build_route(lg, wps, 8.0, yaw,
                              junction_segs=BR.junction_segments(len(wps)),
                              seqs=[r[0] for r in rows],
                              finish_tail_m=BR.finish_tail_cfg())


# ── 스위치 ───────────────────────────────────────────────────────────────
def test_default_is_on():
    """완주가 걸린 문제라 기본 true 다."""
    assert CFG['route']['dp_finish_lane_lock'] is True
    assert BR.dp_cfg(reload=True).finish_lock is True


def test_dpcfg_still_builds_from_eleven_positional_args():
    """밖에서 11개 위치인자로 DPCfg 를 짓는 자리가 있다 (test_global_dp)."""
    c = BR.DPCfg(True, 8.0, 3.0, 400.0, 10.0, 1.0, 45.0, True, False, 16.0, 1.0)
    # 뒤쪽 필드는 전부 기본값이 있어야 한다 (finish_lock / retry_ratio /
    # retry_junction). 기본값은 params 기본과 같은 값이다 — 위치인자로 지은
    # 설정이 조용히 새 기능을 켜면 안 된다.
    assert c.finish_lock is True and c.retry_ratio == 0.0 and c.retry_junction is False
    assert len(c) == 14


# ── 실제 경로 ────────────────────────────────────────────────────────────
CASES = ['실전주행_교통류_02_직진11', '실전주행_교통류_07_연속교차로8',
         '실전주행_교통류_08_우회전16', '실전주행_교통류_11_직진10']


@pytest.mark.parametrize('name', CASES)
def test_lock_ends_on_the_finish_coordinate_lane(lg, name):
    """고정 전에는 옆 차로로 끝나던 경로가 종료 좌표 차로로 끝난다."""
    f = SC / f'{name}.csv'
    if not f.exists():
        pytest.skip(f'{f} 없음 (scenarios/ 는 gitignore 대상)')
    with lock(False):
        g_off = finish_gate(lg, build(lg, f), CFG)
    with lock(True):
        g_on = finish_gate(lg, build(lg, f), CFG)
    if not g_off['lane_mismatch']:
        # 고정 없이도 이미 종료 좌표 차로로 끝난다 = 이 CSV 는 더 이상 양성 사례가
        # 아니다. 경로 비용이 바뀌면 생긴다 (2026-09-09 route.turn_lane_bias_m 60
        # 으로 실전주행_교통류_11_직진10 이 그렇게 됐다). 고정 자체의 계약은
        # 나머지 케이스가 계속 본다 — 여기서 실패로 보면 경로를 못 고친다.
        pytest.skip(f'{name}: 고정 없이도 일치 — 양성 사례가 아니다')
    assert g_on['lane_mismatch'] is False
    assert abs(g_on['t_finish']) < abs(g_off['t_finish'])


@pytest.mark.parametrize('name', CASES)
def test_lock_pulls_the_cones_out_of_the_corridor(lg, name):
    """차로 정합이 맞으면 콘이 회랑 밖으로 나온다 (여유 > 0)."""
    f = SC / f'{name}.csv'
    if not f.exists():
        pytest.skip(f'{f} 없음')
    with lock(True):
        g = finish_gate(lg, build(lg, f), CFG)
    assert g['clear_min'] > 0.0


def test_lock_records_why_it_did_or_did_not_apply(lg):
    f = SC / '실전주행_교통류_02_직진11.csv'
    if not f.exists():
        pytest.skip(f'{f} 없음')
    with lock(True):
        rt = build(lg, f)
    rec = rt['dp']['finish_lock']
    assert rec['applied'] is True and rec['on_lane'] is True
    assert rec['lane'] != rec['was'] and '고정' in rec['why']


def test_lock_declines_when_finish_is_outside_every_lane(lg):
    """종료 좌표가 어느 주행 차로 반폭 안에도 없으면 고정하지 않는다.

    실측 실전주행_교통류_16_직진20: 종료 좌표가 폭 0.75 m 테이퍼 차로 위라
    DP 후보(폭 ≥ 2.0 m)에 없고, 가장 가까운 후보와도 1.95 m 떨어져 있다.
    """
    f = SC / '실전주행_교통류_16_직진20.csv'
    if not f.exists():
        pytest.skip(f'{f} 없음')
    with lock(True):
        rec = build(lg, f)['dp']['finish_lock']
    assert rec['applied'] is False and rec['on_lane'] is False
    assert '경계 밖' in rec['why']


def test_off_reproduces_the_previous_route(lg):
    """스위치를 끄면 최소비용 후보를 쓰던 이전 동작이 그대로 재현된다."""
    f = SC / '실전주행_교통류_02_직진11.csv'
    if not f.exists():
        pytest.skip(f'{f} 없음')
    with lock(False):
        rt = build(lg, f)
    assert rt['dp']['finish_lock'] is None          # 기록 자체가 안 남는다
    with lock(True):
        rt_on = build(lg, f)
    assert rt['lanes'] != rt_on['lanes']            # 이 경로는 실제로 갈린다


def test_official_route_is_unchanged_by_the_lock(lg):
    """대회 경로는 꼬리 연장이 이미 종료 차로를 붙여 줘서 on/off 가 같다.

    finish_tail_m 을 안 넘기고 재면(생성기 경로) 달라 보인다 — 판정 산출물
    (route.pkl)과 같은 조건으로 재야 한다.
    """
    f = ROOT / 'data' / 'official_route.csv'
    if not f.exists():
        pytest.skip('data/official_route.csv 없음')
    with lock(False):
        a = build(lg, f)
    with lock(True):
        b = build(lg, f)
    assert a['lanes'] == b['lanes']
    assert a['total_length'] == pytest.approx(b['total_length'])
