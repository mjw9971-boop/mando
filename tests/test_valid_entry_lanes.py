"""직선 구간 유효 차로 집합 (작업10) — 정보 필드.

쓰임: 회피 시프트로 옆 차로에 나간 뒤 **원래 차로로 복귀해야 하는가**를
제어기가 판단할 재료다. 복귀 공간이 부족하면 곡률이 커져 속도가 하한에
붙거나(2026-09-03 정적회피집중_01, 27.7 s 크립) span 게이트가 회피를 기각해
장애물 앞에 선다. 직진 구간이면 복귀 없이 그 차로로 다음 교차로에 들어가도
되는 경우가 많다.

여기서 지키는 계약:
  1. 판정은 작업7-2 의 turn_connect 재사용 — 새 탐색을 만들지 않는다.
  2. **정보 필드다.** 아무도 안 읽으면 영향 0 — 다른 필드는 하나도 안 바뀐다.
  3. 킬 스위치 route.valid_entry_lanes_enable 이 필드 유무를 가른다 — on/off
     양쪽 다 검증한다.
  4. 마지막 세그먼트는 target='finish' + 빈 집합 = "차로 제약 없음".
     "유효 차로가 없다"(target='pair' + 빈 집합)와 구분된다.

**필드 내용을 보는 테스트는 `field_on` 픽스처로 스위치를 강제로 켠다.**
params 의 기본값은 팀 사정에 따라 꺼질 수 있고(2026-09-04 fe92b70 에서 off),
그때 로직 검증까지 같이 사라지면 나중에 다시 켤 때 믿을 근거가 없다.
스위치 자체의 계약은 `test_switch_default_and_source` ·
`test_kill_switch_both_ways` · `test_field_follows_params_default` 가
별도로 지킨다.
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
CSVS = ['data/official_route.csv', 'tests/fixtures/venue_20260903_waypoints.csv',
        'tests/fixtures/pair_lane_offset_waypoints.csv',
        'scenarios/정적회피집중/정적회피집중_01_좌회전2.csv']


@pytest.fixture(scope='module')
def lg():
    if not GRAPH.exists():
        pytest.skip('data/lane_graph.pkl 없음')
    return LaneGraph(str(GRAPH))


@pytest.fixture
def field_on(monkeypatch):
    """params 킬 스위치와 무관하게 valid_entry_lanes 필드를 켠다.

    이 픽스처를 쓰는 테스트가 검증하는 것은 **필드 로직**이지 스위치의 기본값이
    아니다. 기본값이 off 로 바뀌었다고 로직이 검증되지 않는 상태로 두면,
    다시 켜는 시점에 그 필드를 믿을 근거가 없다.
    """
    monkeypatch.setattr(BR, 'valid_entry_lanes_enable', lambda: True)


def _build(lg, path):
    rows = BR.read_waypoints_csv(str(ROOT / path))
    wps = [(r[1], r[2]) for r in rows]
    n = len(wps)
    ref = next((j for j in range(1, n)
                if math.hypot(wps[j][0] - wps[0][0], wps[j][1] - wps[0][1]) >= 2.0), 1)
    yaw = math.atan2(wps[ref][1] - wps[0][1], wps[ref][0] - wps[0][0])
    js = BR.junction_segments(n, 1) if (n % 2 == 0 and n > 2) else set()
    return BR.build_route(lg, wps, 8.0, yaw, junction_segs=js,
                          seqs=[r[0] for r in rows],
                          finish_tail_m=BR.finish_tail_cfg())


def test_switch_default_and_source():
    from vtd_adapter.config import load_params_yaml
    assert BR.valid_entry_lanes_enable() is bool(
        load_params_yaml()['route']['valid_entry_lanes_enable'])


def test_kill_switch_both_ways(lg, monkeypatch):
    """스위치가 필드 유무를 가르고, 나머지 필드는 어느 쪽에서도 같다.

    params 의 현재 기본값에 의존하지 않는다 — on/off 를 둘 다 강제해서 잰다.
    """
    monkeypatch.setattr(BR, 'valid_entry_lanes_enable', lambda: True)
    on = _build(lg, 'data/official_route.csv')
    monkeypatch.setattr(BR, 'valid_entry_lanes_enable', lambda: False)
    off = _build(lg, 'data/official_route.csv')
    assert 'valid_entry_lanes' in on and 'valid_entry_lanes' not in off
    for k in off:
        assert on[k] == off[k], k                      # 다른 필드는 하나도 안 바뀐다


def test_field_follows_params_default(lg):
    """기본 경로(픽스처 없이)는 params 값을 그대로 따른다 — 스위치가 실제로 먹는다."""
    from vtd_adapter.config import load_params_yaml
    want = bool(load_params_yaml()['route']['valid_entry_lanes_enable'])
    rt = _build(lg, 'data/official_route.csv')
    assert ('valid_entry_lanes' in rt) is want


@pytest.mark.parametrize('csv', CSVS)
def test_covers_every_non_pair_segment(lg, csv, field_on):
    """짝이 아닌 세그먼트 전부에 항목이 하나씩 있다 (짝 구간에는 없다)."""
    if not (ROOT / csv).exists():
        pytest.skip(f'{csv} 없음')
    rt = _build(lg, csv)
    js = set(rt['junction_segments'])
    segs = {e['seg'] for e in rt['valid_entry_lanes']}
    assert segs == set(range(len(rt['waypoints']) - 1)) - js


@pytest.mark.parametrize('csv', CSVS)
def test_last_segment_is_finish_with_empty_set(lg, csv, field_on):
    """마지막 세그먼트 = target 'finish' + 빈 집합 (차로 제약 없음)."""
    if not (ROOT / csv).exists():
        pytest.skip(f'{csv} 없음')
    rt = _build(lg, csv)
    last = max(rt['valid_entry_lanes'], key=lambda e: e['seg'])
    assert last['seg'] == len(rt['waypoints']) - 2
    assert last['target'] == 'finish'
    assert last['lanes'] == [] and last['in_set'] is None


@pytest.mark.parametrize('csv', CSVS)
def test_entry_pool_excludes_junction_lanes(lg, csv, field_on):
    """집합에 교차로 연결로가 들어가면 안 된다.

    진입 경유점은 교차로 어귀에 찍히므로 후보집합에 접근 차로와 그 successor
    연결로가 같이 들어온다. 연결로는 "어느 평행 차로에 있을 것인가" 의 선택지가
    아니다 — 넣어 두면 집합이 두 배로 부풀어 좌회전인데 2개로 보인다.
    """
    if not (ROOT / csv).exists():
        pytest.skip(f'{csv} 없음')
    rt = _build(lg, csv)
    for e in rt['valid_entry_lanes']:
        for k in e['lanes']:
            assert lg.lanes[tuple(k)]['junction'] == -1, (e['seq'], k)


def test_turns_narrow_the_set(lg, field_on):
    """좌·우회전은 집합을 좁힌다 — 그게 곧 복귀 강제다.

    "반드시 1개" 는 아니다. 병렬 좌회전 차로가 실재한다
    (official_route seq 6→7: (1222,1,-1)/(1222,1,-2) 가 각각
    (2823,0,-1)/(2823,0,-2) 로 이어진다). 계약은 "직진보다 좁다" 이다.
    """
    sizes = {'left': [], 'right': [], 'straight': []}
    for csv in CSVS:
        if not (ROOT / csv).exists():
            continue
        for e in _build(lg, csv)['valid_entry_lanes']:
            if e['target'] == 'pair':
                sizes[e['turn']].append(len(e['lanes']))
    turn = sizes['left'] + sizes['right']
    assert turn and sizes['straight']
    assert max(turn) < max(sizes['straight'])


def test_official_route_sets(lg, field_on):
    """대회 배포 경로의 값을 고정한다 — 바뀌면 눈에 띄어야 한다."""
    rt = _build(lg, 'data/official_route.csv')
    got = [(e['seg'], e['target'], e['turn'], len(e['lanes']), e['in_set'])
           for e in rt['valid_entry_lanes']]
    assert got == [(0, 'pair', 'left', 1, True),
                   (2, 'pair', 'straight', 1, True),
                   (4, 'pair', 'left', 2, True),
                   (6, 'finish', None, 0, None)]


def test_empty_pair_set_means_banned_connector(lg, field_on):
    """target='pair' 인데 빈 집합 = 통행 금지 연결로뿐이라는 뜻이다.

    waypoints.csv 는 R 5.43 m / R 4.52 m 연결로를 지난다(회전 불가 기하).
    turn_connect 에 banned 가 들어가므로 유효 진입 차로가 0 이 되고, 경로는
    대안이 없어 그대로 통과한다 — 그 사실은 infeasible_forced 에 남는다.
    """
    csv = 'waypoints.csv'
    if not (ROOT / csv).exists():
        pytest.skip('없음')
    rt = _build(lg, csv)
    empty = [e for e in rt['valid_entry_lanes'] if e['target'] == 'pair' and not e['lanes']]
    forced = {tuple(k) for _wi, k, _r in rt.get('infeasible_forced') or []}
    assert empty, '이 CSV 에 빈 집합이 있어야 이 테스트가 유효하다'
    assert forced, '빈 집합의 근거인 infeasible_forced 가 비어 있다'
    for e in empty:
        assert e['in_set'] is False


def test_reuses_turn_connect(lg, monkeypatch, field_on):
    """새 탐색을 만들지 않는다 — 판정은 turn_connect 를 통해서만 나간다."""
    rt = _build(lg, 'data/official_route.csv')
    calls = []
    orig = BR.turn_connect
    monkeypatch.setattr(BR, 'turn_connect',
                        lambda *a, **k: (calls.append(1), orig(*a, **k))[1])
    out = BR.valid_entry_lanes(lg, rt)
    assert calls, 'turn_connect 을 안 불렀다'
    assert [e['lanes'] for e in out] == [e['lanes'] for e in rt['valid_entry_lanes']]
