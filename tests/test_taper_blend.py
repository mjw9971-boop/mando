"""
경로 폴리라인 테이퍼 킹크 (2026-08-26 실기 사고 회귀).

소멸(테이퍼) 차로 (2192,3,4)는 끝 폭이 0 — 중심선이 이웃 경계로 수렴하다
successor (2192,2,3) 중심선(반폭 ≈1.5 m 옆)으로 순간이동했고, 룩어헤드 2.5 m
lateral PID 가 한 틱 만에 풀포화 → ±1.5 m 진동 → 차선이탈 3건.

이중 방어를 각각 검증한다:
  · route.py taper_blend — 재샘플 폴리라인이 연속(간격 ≤ 0.3 m)이어야 한다
  · gen_scenarios.polyline_gate — blend 를 꺼서 킹크를 재현하면 게이트가 잡아야
    한다 (route.py 가 퇴행해도 생성 시점에 막히는지)

lane_graph.pkl 은 .gitignore 대상이라 없을 수 있다 → 그때는 skip.
"""
import copy
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
GRAPH = ROOT / 'data' / 'lane_graph.pkl'

pytestmark = pytest.mark.skipif(not GRAPH.exists(),
                                reason='data/lane_graph.pkl 없음 (gitignore 대상)')

# 실기 사고 경로의 실제 앞부분 — (2192,3,4)가 끝 폭 0 테이퍼, successor (2192,2,3)
CHAIN = [(2192, 4, 4), (2192, 3, 4), (2192, 2, 3)]


@pytest.fixture(scope='module')
def lg():
    from vtd_adapter.lanegraph import LaneGraph
    return LaneGraph(str(GRAPH))


@pytest.fixture(scope='module')
def cfg():
    from vtd_adapter.config import load_params_yaml
    return load_params_yaml()


def synth_rt(lg, chain, start_s=1.0):
    """VtdRoutePlanner._build 가 보는 필드만 있는 합성 route (build_route 관례)."""
    for a, b in zip(chain, chain[1:]):
        assert b in lg.successors(a), f'{a} → {b} 가 successor 가 아니다 (체인 오기)'
    lengths = [lg.length(k) for k in chain]
    cum = [-start_s]
    for i in range(1, len(chain)):
        cum.append(cum[i - 1] + lengths[i - 1])
    return {'lanes': chain, 'cum_s': cum, 'lengths': lengths,
            'total_length': cum[-1] + lengths[-1], 'start_s_in_lane': start_s,
            'events': []}


def _max_step(lg, rt, cfg):
    from vtd_adapter.route import VtdRoutePlanner
    pts = VtdRoutePlanner(lg, rt, cfg).route_points
    return float(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])).max())


def test_chain_has_taper(lg, cfg):
    """전제 자체를 고정: (2192,3,4)는 끝 폭이 차폭 미만인 소멸 차로다."""
    assert lg.width_at((2192, 3, 4), lg.length((2192, 3, 4))) \
        < float(cfg['vehicle']['width'])


def test_taper_blend_removes_kink(lg, cfg):
    rt = synth_rt(lg, CHAIN)
    assert _max_step(lg, rt, cfg) <= 0.3, '테이퍼 블렌드 후에도 폴리라인이 불연속이다'


def test_kink_reappears_without_blend(lg, cfg):
    """blend 를 끄면(길이 0) 1.5 m 급 킹크가 재현된다 — 사고의 재현 조건."""
    cfg2 = copy.deepcopy(cfg)
    cfg2['route']['taper_blend_m'] = 0.0
    assert _max_step(lg, synth_rt(lg, CHAIN), cfg2) > 1.0


def test_polyline_gate_catches_kink(lg, cfg):
    """route.py 가 퇴행해도(blend 무력화) 생성 게이트가 독립적으로 잡는다."""
    import gen_scenarios as gs
    _, _, gen_cfg = gs.load_themes()
    rt = synth_rt(lg, CHAIN)
    saved = gs._PARAMS_CFG
    try:
        gs._PARAMS_CFG = copy.deepcopy(cfg)
        gs._PARAMS_CFG['route']['taper_blend_m'] = 0.0
        why = gs.polyline_gate(lg, rt, gen_cfg)
        assert why is not None and '불연속' in why
        gs._PARAMS_CFG = cfg                    # 정상 blend 면 통과해야 한다
        assert gs.polyline_gate(lg, rt, gen_cfg) is None
    finally:
        gs._PARAMS_CFG = saved
