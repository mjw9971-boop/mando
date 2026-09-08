"""
A1 — 교차로 안 자기잠금 해제 (junction_creep_release_enable).

실측 근거: logs/batch/20260906_012222/실전주행_교통류_02_직진11.jsonl
  t=104.7 좌회전 연결로 진입 → reject='junction' 이 로그 끝(t=138.2)까지 31.2 s
  연속. 그동안 BREAKOUT level 2 / paused=True 로 얼어 있고 standoff_v 는
  t=108.0 부터 0.0 고정, route_s 522.3 에서 30 s 무진전 (완주 실패).

왜 어떤 탈출도 안 걸리나 — 교차로 lane 에서 사다리 둘이 **상한 없이** 죽는다:
  · _try_overtake_inner  교차로 lane 이면 side 루프 전에 reject='junction'
  · _breakout_tick       교차로 lane 이면 맨 앞에서 return → bo_level 이 안 오르니
                         _creep_gate ②(L4)가 영영 안 열린다
남는 _creep_gate ③(지연)은 사다리가 도는 상황을 전제로 잡은 안전망이다.

여기서 지키는 불변:
  · 무장 조건은 **reject='junction' 전용 시계**다. 다른 사유의 기각
    (right:no_neighbor·occupied·geom)은 정상적으로 풀릴 수 있으므로 세지 않는다.
  · 무장돼도 **차로 시프트는 여전히 금지**다 (연결로에서 옆 차로로 밀지 않는다).
  · 무장돼도 **breakout_creep() 훅은 열지 않는다** — PDM 의 IDM·OBB 가 최후
    안전망으로 남아야 한다. 해제가 푸는 것은 크립 **상한**뿐이다.
  · 스위치 off 는 로그까지 이전과 동일하다 (j_* 키를 아예 안 남긴다).

모드 두 가지를 모두 막는다:
  A  d 가 standoff_floor_m(22.0)에 정확히 얹혀 √(2a·0)=0 — _standoff_creep 이
     호출조차 안 돼 크립 게이트에 도달하지 못한다 (시뮬 재현: 장애물 24 m).
  B  오버슛으로 기준선 안쪽(20.1) — 실주행 02 가 이쪽이다.
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

CFG = load_params_yaml(PARAMS_YAML)
OT = CFG['overtake']
REL_TICKS = int(round(OT['junction_release_s'] * HZ))
FLOOR_MPS = OT['junction_creep_floor_kph'] / 3.6
CAP_MPS = OT['junction_creep_kph'] / 3.6
STANDOFF = OT['standoff_floor_m']                                  # 22.0


def on_cfg(**over):
    """해제 스위치를 켠 사본. params 값이 정본이고 여기서는 켜기만 한다."""
    c = copy.deepcopy(CFG)
    c['overtake']['junction_creep_release_enable'] = True
    c['overtake'].update(over)
    return c


def off_cfg(**over):
    """이전 동작 사본. **기본값을 읽지 않는다** — params 가 true 로 바뀌어도
    off 경로 커버리지를 잃지 않기 위해서다 (CLAUDE.md 2026-09-07 드리프트 원칙)."""
    c = copy.deepcopy(CFG)
    c['overtake']['junction_creep_release_enable'] = False
    c['overtake']['never_stall_enable'] = False
    c['overtake'].update(over)
    return c


class LgJunction(LgOne):
    """자차 lane 이 교차로 연결로 — _try_overtake_inner 가 reject='junction'."""

    def __init__(self):
        super().__init__()
        self.lanes = {k: {'junction': 8} for k in self.lanes}


def rig(obj_x, cfg=CFG, junction=True):
    """장애물이 obj_x [m] 앞에 정지. 관찰을 쌓아 회랑 후보까지 들어간 상태."""
    p = GeomPlanner(d_tl=float('inf'))
    p.lg = LgJunction() if junction else LgOne()
    kr = KrRules(cfg)
    kr._sl_all = []
    ap = Ap(p, actors=[Box(2, obj_x, 0.0, 0.0, half_w=0.9)])
    ap._kr_ego_lane = (1, 0, -1)
    ap.traffic_light_hazard = ap.walker_hazard = ap.walker_close = False
    ap.stop_sign_hazard = False
    kr.last_d_end = 1e6                       # 종점 사정권 밖
    kr._ap = ap                               # apply() 가 하는 주입
    for _ in range(int(cfg['overtake']['wait_before_shift_s'] * HZ) + 5):
        kr._update_obj_timers(ap)
    return kr, p, ap


def step(kr, p, ap, n=1, v=0.0):
    """apply() 와 같은 순서: 캐시 → 사다리 → 회피 → standoff. 마지막 standoff 값."""
    so = None
    for _ in range(n):
        kr._update_obj_timers(ap)
        kr._tick_cache(ap, p)
        kr._breakout_tick(p, ap, v)
        kr._try_overtake(ap, p, v)
        so = kr._standoff_profile(v)
    return so


# ── 시계: reject='junction' 만 센다 ────────────────────────────────────────
def test_counter_rises_only_on_junction_reject():
    kr, p, ap = rig(24.0, on_cfg())
    step(kr, p, ap, 5)
    assert kr.last_overtake == 'junction'
    assert kr.j_reject_ticks == 5 and kr.ot_reject_ticks == 5


def test_counter_ignores_other_reject_reasons():
    """교차로 밖 기각(right:no_neighbor 등)은 교차로 해제를 무장시키면 안 된다."""
    kr, p, ap = rig(24.0, on_cfg(), junction=False)
    step(kr, p, ap, REL_TICKS + 20)
    assert kr.last_overtake != 'junction'
    assert kr.j_reject_ticks == 0
    assert kr.ot_reject_ticks > 0 or kr.ot_span is not None   # 다른 시계는 산다
    assert kr._junction_release() is False


def test_release_arms_exactly_at_threshold():
    kr, p, ap = rig(24.0, on_cfg())
    step(kr, p, ap, REL_TICKS - 1)
    assert kr._junction_release() is False
    step(kr, p, ap, 1)
    assert kr._junction_release() is True


def test_release_disarms_when_leaving_junction():
    """연결로를 벗어나면 그 틱부터 거짓 — 시계가 0 이 되기 전에도."""
    kr, p, ap = rig(24.0, on_cfg())
    step(kr, p, ap, REL_TICKS + 5)
    assert kr._junction_release() is True
    p.lg = LgOne()                                   # 연결로 밖으로
    kr._tick_cache(ap, p)
    assert kr.j_reject_ticks > REL_TICKS             # 시계는 아직 안 지워졌다
    assert kr._junction_release() is False


def test_on_reset_clears_counter():
    kr, p, ap = rig(24.0, on_cfg())
    step(kr, p, ap, REL_TICKS + 5)
    kr.on_reset()
    assert kr.j_reject_ticks == 0 and kr._junction_release() is False


# ── 모드 A: 기준선 **밖**(d ≥ standoff)에서 0 으로 얼어붙는 것 ─────────────
def test_mode_a_locks_at_baseline_when_off():
    """이전 동작 — d 가 22.0 에 얹히면 v_allow 0 이고 크립 진단조차 없다."""
    kr, p, ap = rig(STANDOFF, off_cfg())                   # d = standoff 정확히
    so = step(kr, p, ap, REL_TICKS + 40)
    assert so == pytest.approx(0.0)
    assert kr._creep_diag is None                    # _standoff_creep 미호출
    assert kr.bo_paused is True and kr.bo_state is None


def test_mode_a_creeps_after_release():
    kr, p, ap = rig(STANDOFF, on_cfg())
    assert step(kr, p, ap, REL_TICKS - 1) == pytest.approx(0.0)   # 아직 이전 동작
    so = step(kr, p, ap, 1)
    assert so >= FLOOR_MPS                           # 바닥이 걸렸다
    assert (kr._creep_diag or {}).get('creep_junction') is True


def test_mode_a_creep_is_capped():
    """기준선 밖 프로파일이 아무리 커도 교차로 안에서는 상한을 넘지 않는다."""
    kr, p, ap = rig(STANDOFF + 20.0, on_cfg())       # 프로파일만 보면 √(2·3·20)=11 m/s
    so = step(kr, p, ap, REL_TICKS + 1)
    assert so == pytest.approx(CAP_MPS)


# ── 모드 B: 기준선 **안쪽** — 크립 게이트가 'need' 로 보류하던 것 ──────────
def test_mode_b_gate_holds_on_need_when_off():
    kr, p, ap = rig(STANDOFF - 1.0, off_cfg())             # d = 21.0 < standoff
    step(kr, p, ap, REL_TICKS + 1)
    d = kr._creep_diag or {}
    assert d.get('creep_hold_why') == 'need' and d.get('so_creep') is False


def test_mode_b_gate_opens_with_junction_reason():
    kr, p, ap = rig(STANDOFF - 1.0, on_cfg())
    step(kr, p, ap, REL_TICKS - 1)
    assert (kr._creep_diag or {}).get('creep_hold_why') == 'need'
    so = step(kr, p, ap, 1)
    d = kr._creep_diag or {}
    assert d.get('creep_open_why') == 'junction' and d.get('creep_junction') is True
    assert so >= FLOOR_MPS


def test_mode_b_stop_gap_still_stops():
    """③ A2 로 넘긴 부분 — 진짜 정지 거리 안에서는 해제돼도 0 이다 (접촉 방지)."""
    kr, p, ap = rig(4.0, on_cfg())                   # d_stop(≈7 m) 안쪽
    so = step(kr, p, ap, REL_TICKS + 5)
    assert so == pytest.approx(0.0)
    assert (kr._creep_diag or {}).get('creep_block') == 'stop_gap'


# ── 사다리: 무장 후에는 돌지만 크립 훅은 닫힌 채다 ─────────────────────────
def test_ladder_stays_paused_when_off():
    kr, p, ap = rig(STANDOFF - 1.0, off_cfg())
    step(kr, p, ap, REL_TICKS + 60)
    assert kr.bo_paused is True and kr.bo_state is None and kr.bo_level == 0


def test_ladder_runs_after_release():
    kr, p, ap = rig(STANDOFF - 1.0, on_cfg())
    step(kr, p, ap, REL_TICKS)
    assert kr.bo_paused is True and kr.bo_state is None        # 무장 직전까지 정지
    step(kr, p, ap, int(round(OT['stuck_hard_s'] * HZ)) + 2)
    assert kr.bo_paused is False and kr.bo_state == 'BREAKOUT'


def test_breakout_creep_hook_never_opens_inside_junction():
    """L4 에 닿아도 거짓 — 연결로에서 선행차·OBB 를 지우면 그대로 들이받는다."""
    kr, p, ap = rig(STANDOFF - 1.0, on_cfg())
    step(kr, p, ap, REL_TICKS + int(round(
        (OT['stuck_hard_s'] + 3 * OT['escalate_s']) * HZ)) + 10)
    assert kr.bo_level >= kr.BO_CREEP                          # 사다리는 끝까지 갔고
    assert kr.breakout_creep() is False                        # 훅은 닫혀 있다


def test_shift_stays_forbidden_inside_junction_after_release():
    """해제는 종방향만 푼다 — 시프트는 어떤 단계에서도 교차로에서 금지."""
    kr, p, ap = rig(STANDOFF - 1.0, on_cfg())
    step(kr, p, ap, REL_TICKS + 200)
    assert kr.last_overtake == 'junction' and kr.ot_span is None


# ── off 지문 ──────────────────────────────────────────────────────────────
def test_off_leaves_no_new_log_keys():
    """54 지문 회귀의 근거 — off 는 로그 키까지 이전과 동일해야 한다."""
    kr, p, ap = rig(STANDOFF - 1.0, off_cfg())
    step(kr, p, ap, REL_TICKS + 5)
    a = kr.last_avoid or {}
    assert 'j_reject_s' not in a and 'j_release' not in a
    assert 'creep_junction' not in (kr._creep_diag or {})


def test_on_records_release_diagnostics():
    kr, p, ap = rig(STANDOFF - 1.0, on_cfg())
    step(kr, p, ap, REL_TICKS + 1)
    a = kr.last_avoid or {}
    assert a['j_reject_s'] == pytest.approx((REL_TICKS + 1) / HZ, abs=0.05)
    assert a['j_release'] is True
