"""
detect_blink_stop — 대회 항목 9 적색점멸 일시정지 (합성 틱 경계값).

점멸은 "정지 후 통과" 가 규정이라 통과 자체는 위반이 아니다 — 통과 **전에**
섰는가만 본다. 그래서 항목 7(적신호)의 무정차 통과와 판정 구조가 다르다.

state 6 은 **지속 플래그**다: 램프 on/off 위상은 9910 에 실리지 않는다
(실측 2026-09-01 signal_blink_obs: 629틱 35.7 s 동안 6 고정, 전이 0회).
그래서 주기 토글을 찾지 않는다 — 찾으면 오히려 못 잡는다.

이 맵의 점멸 컨트롤러는 117 하나뿐이고 적색으로 확정돼(대회 배포본 7개 전부
blink=117, Signal 373 type=1000020) 색 판정 없이 6 을 적색점멸로 본다.
"""
import score as sc_mod
from conftest import mk_tick

STOP_SPEED = 0.5
SC = {'stop_ok_m': 2.0, 'stop_hold_s': 0.5}
BLINK, RED, GREEN = 6, 1, 3
CTRL = [116, 117]


def approach(state=BLINK, stop_n=0, stop_front=-1.0, dt=0.05, speed=6.7,
             ctrl=CTRL, lights_id=117, reset_at=None, reach=True):
    """정지선으로 접근 → (선택) 정지 → 통과. front_m 이 음수→양수로 간다.

    stop_n=0 이면 무정차 통과. reach=False 면 정지선 2 m 안에 못 든다.
    """
    ticks, t = [], 0.0
    fronts = [-8.0, -5.0, -3.0] + ([] if reach else [-2.5])
    if reach:
        fronts += [-1.5, -1.0, -0.5]
    for f in fronts:
        ticks.append(mk_tick(t=t, speed=speed, front_m=f, ctrl=ctrl,
                             lights=[(lights_id, state)], route_s=100.0 + f))
        t += dt
    for i in range(stop_n):                       # 정지선 앞 정지
        ticks.append(mk_tick(t=t, speed=0.1, front_m=stop_front, ctrl=ctrl,
                             lights=[(lights_id, state)], route_s=100.0 + stop_front,
                             reset=(i == reset_at)))
        t += dt
    if reach:
        for f in (0.2, 1.0, 3.0):                 # 통과
            ticks.append(mk_tick(t=t, speed=speed, front_m=f, ctrl=ctrl,
                                 lights=[(lights_id, state)], route_s=100.0 + f))
            t += dt
    return ticks


def run(ticks, merge_gap_s=0.0):
    return sc_mod.detect_blink_stop(ticks, ticks[0]['t'], STOP_SPEED, SC, merge_gap_s)


# ── 핵심 판정 ─────────────────────────────────────────────────────────────
def test_no_stop_through_blink_is_violation():
    """실측 재현: 점멸 정지선을 무정차 통과 (signal_blink_obs 24 km/h)."""
    ev = run(approach(stop_n=0))
    assert len(ev) == 1
    assert ev[0]['ctrl_ids'] == CTRL
    assert ev[0]['min_v_kph'] > 0.0


def test_stop_long_enough_is_not_violation():
    """0.5 s 이상 정지하면 규정 준수 — 통과 자체는 위반이 아니다."""
    assert run(approach(stop_n=12)) == []          # 12틱 × 0.05 s = 0.55 s


def test_hold_just_under_threshold_is_violation():
    # 10틱 × 0.05 = 0.45 s < stop_hold_s 0.5 — 일시 감속은 정지가 아니다
    assert len(run(approach(stop_n=10))) == 1


def test_hold_exactly_at_threshold_is_not_violation():
    # 11틱 → span 0.50 s (>= 0.5)
    assert run(approach(stop_n=11)) == []


# ── 점멸이 아닌 신호는 대상이 아니다 ─────────────────────────────────────
def test_red_light_is_not_item9():
    """적색(1)은 항목 7 소관 — 항목 9 가 중복으로 잡으면 안 된다."""
    assert run(approach(state=RED, stop_n=0)) == []


def test_green_is_not_item9():
    assert run(approach(state=GREEN, stop_n=0)) == []


def test_state_six_is_sustained_flag_not_toggle():
    """6 이 내내 고정이어도 한 건으로 잡힌다 (토글 검출에 의존하지 않는다)."""
    ev = run(approach(stop_n=0))
    assert len(ev) == 1 and ev[0]['ticks'] > 1


# ── 대조·범위 조건 ────────────────────────────────────────────────────────
def test_blink_on_other_controller_not_governing_stopline_ignored():
    """9910 이 준 id 가 전방 정지선 controller 에 없으면 대상 아님 —
    _ctrl_states 가 걸러서 id 화이트리스트가 필요 없다."""
    assert run(approach(stop_n=0, lights_id=999)) == []


def test_no_stopline_ctrl_ids_ignored():
    ticks = approach(stop_n=0, ctrl=[])
    assert run(ticks) == []


def test_never_reaching_stopline_is_not_judged():
    """정지선 2 m 안까지 못 갔으면 판정 대상이 아니다 (멀리서 점멸만 본 경우)."""
    assert run(approach(stop_n=0, reach=False)) == []


def test_stop_after_crossing_does_not_count():
    """선을 넘어 선 것은 일시정지가 아니다 (그건 항목 7 침범 소관)."""
    ticks, t = [], 0.0
    for f in (-5.0, -2.0, -0.5):
        ticks.append(mk_tick(t=t, speed=6.7, front_m=f, ctrl=CTRL, lights=[(117, BLINK)]))
        t += 0.05
    for _ in range(15):                            # 선을 넘은 뒤 정지
        ticks.append(mk_tick(t=t, speed=0.1, front_m=1.0, ctrl=CTRL, lights=[(117, BLINK)]))
        t += 0.05
    assert len(run(ticks)) == 1


def test_reset_ticks_excluded():
    """courseRespawn 순간이동은 주행이 아니다 — 마스크에서 빠진다."""
    ev = run(approach(stop_n=0))
    assert len(ev) == 1
    ticks = approach(stop_n=0)
    for t in ticks:
        t['world']['flags']['reset'] = True
    assert run(ticks) == []


# ── 등록·배선 ─────────────────────────────────────────────────────────────
def test_item9_registered_with_real_key():
    """score.py:88 의 key=None(미구현)이 실제 key 로 교체됐는가."""
    item9 = next(i for i in sc_mod.ITEMS if i[0] == 9)
    assert item9[1] == 'blink_stop', '항목 9 가 아직 미구현(key=None)이다'
    assert sc_mod.ITEM_LABEL['blink_stop'].startswith('9 ')


def test_severity_is_major():
    """항목 7 의 적신호 무정차 통과와 같은 중대 등급."""
    assert sc_mod._severity('blink_stop', {}, SC) == 'major'


def test_not_an_info_key():
    """정보성이 아니라 위반 총계에 들어가야 한다."""
    assert 'blink_stop' not in sc_mod.INFO_KEYS


def test_blink_constant_is_six():
    assert sc_mod.BLINK == 6


def test_render_has_label():
    assert sc_mod.LABEL['blink_stop']
