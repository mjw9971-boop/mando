"""[3] 종료 구간 정지 객체를 **PDM actor 목록에서도** 뺀다 (실주행 2차, 2026-09-08).

kr 후보만 빼는 것으로는 안 멈춘다. 실측 20260908_222954/실경로_02 rs 745.5
(종료선 749.2 — **3.7 m 앞**): `finish_gate.dropped=4` 로 kr 은 콘을 이미
뺐는데 PDM 의 `compute_target_speeds_wrt_all_actors` 가 콘을 보고 vehicle 후보
0.0 을 내 483틱(로그 끝까지) 정지했다 — 미완주다.

계약:
  · 판정은 `kr_rules.finish_gate_drop` **한 곳**이다 (조건 두 벌 금지).
    회랑 쪽 `_corridor_blockers` 와 같은 재료를 쓴다.
  · 빼는 것은 종료 구간의 **정지 객체뿐**. 움직이는 것은 그대로 보인다.
  · `speed.finish_gate_ignore_enable=false` 면 아무것도 안 뺀다 (이전 동작).
  · 로그의 objects[] 는 어댑터에서 그대로 나온다 — 진단이 사라지면 안 된다.
"""
import copy
import pathlib
import sys

import pytest

from conftest import PARAMS_YAML
from vtd_adapter.config import load_params_yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'team_code'))

from kr_rules import KrRules                                       # noqa: E402
from test_avoid import Ap, Box, GeomPlanner, HZ, LgOne             # noqa: E402
from test_finish_gate_ignore import (FINISH_S, GATE_M, V_STATIC,   # noqa: E402
                                     off_cfg, on_cfg, rig)


def drops(kr, p, ap):
    return [a for a in ap._world.get_actors() if kr.finish_gate_drop(p, a)]


def kept(kr, p, ap):
    return [a for a in ap._world.get_actors() if not kr.finish_gate_drop(p, a)]


# ── 판정 ──────────────────────────────────────────────────────────────────
def test_static_object_in_gate_is_dropped():
    kr, p, ap = rig(on_cfg(), route_s=FINISH_S - 2.0)
    assert len(drops(kr, p, ap)) == 1
    assert kept(kr, p, ap) == []


def test_before_the_gate_nothing_is_dropped():
    kr, p, ap = rig(on_cfg(), route_s=FINISH_S - GATE_M - 5.0)
    assert drops(kr, p, ap) == []


def test_moving_object_is_kept_inside_the_gate():
    """움직이는 것은 종료 구간에서도 그대로 본다 — 콘만 빼는 기능이다."""
    kr, p, ap = rig(on_cfg(), obj_speed=V_STATIC + 1.0, route_s=FINISH_S - 2.0)
    assert drops(kr, p, ap) == []
    assert len(kept(kr, p, ap)) == 1


def test_switch_off_drops_nothing():
    kr, p, ap = rig(off_cfg(), route_s=FINISH_S - 2.0)
    assert kr._in_finish_gate(p) is False
    assert drops(kr, p, ap) == []


def test_speed_threshold_is_the_single_source():
    """속도 임계는 overtake.blocker_speed_max 하나다 (별도 상수 금지)."""
    kr, p, ap = rig(on_cfg(), obj_speed=V_STATIC - 0.01, route_s=FINISH_S - 2.0)
    assert len(drops(kr, p, ap)) == 1
    kr2, p2, ap2 = rig(on_cfg(), obj_speed=V_STATIC + 0.01, route_s=FINISH_S - 2.0)
    assert drops(kr2, p2, ap2) == []


def test_same_verdict_as_the_corridor_gate():
    """회랑에서 빠지는 객체와 **같은 집합**이어야 한다 (조건이 두 벌이 아니다)."""
    for rs in (FINISH_S - GATE_M - 5.0, FINISH_S - GATE_M, FINISH_S, FINISH_S + 20.0):
        kr, p, ap = rig(on_cfg(), route_s=rs)
        corridor_empty = kr._corridor_blockers(ap, p, static_ok=kr._stop_ok) == []
        assert bool(drops(kr, p, ap)) == corridor_empty


# ── 접합부가 실제로 걸러내는가 ────────────────────────────────────────────
def test_autopilot_seam_filters_both_lists():
    """autopilot 의 `# VTD:` 접합부가 actors 와 vehicles 를 같이 거른다."""
    src = (ROOT / 'team_code' / 'autopilot.py').read_text(encoding='utf-8')
    # 표식 뒤 **그 줄의 나머지**(주석 본문)는 버리고 이어지는 블록만 본다
    seam = src.split('# VTD: 종료 구간')[1].split('\n', 1)[1].split('\n\n')[0]
    # 주석은 빼고 **코드 줄만** 본다 (주석에는 근거를 적으므로 낱말이 겹친다)
    code = '\n'.join(ln for ln in seam.splitlines()
                     if ln.strip() and not ln.strip().startswith('#'))
    assert 'finish_gate_drop' in code
    assert 'actors = type(actors)' in code and 'vehicles = [' in code
    # 조건을 여기서 새로 적지 않는다 — kr_rules 호출뿐이어야 한다
    for banned in ('finish_gate_m', 'blocker_speed_max', '.speed', 'finish_s'):
        assert banned not in code


def test_seam_is_inert_without_kr_rules():
    """kr_rules 가 없는 컨텍스트(순수 PDM)에서는 아무것도 안 뺀다."""
    src = (ROOT / 'team_code' / 'autopilot.py').read_text(encoding='utf-8')
    assert 'getattr(self, "kr_rules", None)' in src


# ── 대조 픽스처: 종료점이 연결로 밖 직선 위 ──────────────────────────────
# 대회 형식상 종료점은 직선 위다. 원본 PathShape03 은 종료점이 폭 2.70 m 짜리
# 좌회전 연결로 안이라 콘이 회랑에 들어오는데, 그 전제를 뺀 대조군을 둔다
# (tests/fixtures/README.md 참조).
def _build_route(csv):
    import io, contextlib
    sys.path.insert(0, str(ROOT / 'tools'))
    import build_route as BR
    from vtd_adapter.lanegraph import LaneGraph
    lg = LaneGraph(str(ROOT / 'data' / 'lane_graph.pkl'))
    rows = BR.read_waypoints_csv(str(ROOT / csv))
    wps = [(r[1], r[2]) for r in rows]
    seqs = [r[0] for r in rows]
    import math
    yaw = math.atan2(wps[1][1] - wps[0][1], wps[1][0] - wps[0][0])
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rt = BR.build_route(lg, wps, 8.0, yaw,
                            junction_segs=BR.junction_segments(len(wps), 1),
                            seqs=seqs, finish_tail_m=BR.finish_tail_cfg())
    return lg, rt


def _finish_lane(lg, rt):
    fs = rt['waypoint_s'][-1]
    for i, k in enumerate(rt['lanes']):
        a = rt['cum_s'][i]
        if a <= fs <= a + rt['lengths'][i]:
            return k, fs
    raise AssertionError('종료점이 경로 위에 없다')


@pytest.mark.skipif(not (ROOT / 'data' / 'lane_graph.pkl').exists(),
                    reason='data/lane_graph.pkl 없음')
def test_original_finish_is_inside_a_connector():
    """원본은 종료점이 연결로 안이다 — 이 픽스처가 필요한 이유 그 자체."""
    lg, rt = _build_route('tests/fixtures/pathshape03_waypoints.csv')
    k, fs = _finish_lane(lg, rt)
    assert lg.lanes[k]['junction'] != -1
    assert fs == pytest.approx(865.4, abs=0.5)


@pytest.mark.skipif(not (ROOT / 'data' / 'lane_graph.pkl').exists(),
                    reason='data/lane_graph.pkl 없음')
def test_variant_finish_is_on_a_straight():
    """대조군은 종료점이 junction 밖 직선 위고, 경로·꼬리는 그대로 성립한다."""
    lg, rt = _build_route('tests/fixtures/pathshape03_finish_straight_waypoints.csv')
    k, fs = _finish_lane(lg, rt)
    assert lg.lanes[k]['junction'] == -1
    assert rt['total_length'] == pytest.approx(896.7, abs=1.0)
    tail = rt['total_length'] - fs
    from vtd_adapter.config import load_params_yaml as _lp
    assert tail >= float(_lp(PARAMS_YAML)['route']['finish_tail_m'])


@pytest.mark.skipif(not (ROOT / 'data' / 'lane_graph.pkl').exists(),
                    reason='data/lane_graph.pkl 없음')
def test_variant_only_moves_the_finish_point():
    """종료점 말고는 원본과 같은 경유점이다 (코스를 바꾼 게 아니다)."""
    sys.path.insert(0, str(ROOT / 'tools'))
    import build_route as BR
    a = BR.read_waypoints_csv(str(ROOT / 'tests/fixtures/pathshape03_waypoints.csv'))
    b = BR.read_waypoints_csv(
        str(ROOT / 'tests/fixtures/pathshape03_finish_straight_waypoints.csv'))
    assert [r[:3] for r in a[:-1]] == [r[:3] for r in b[:-1]]
    assert a[-1][1:] != b[-1][1:]
