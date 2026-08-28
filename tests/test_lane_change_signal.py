"""
detect_lane_change_signal — 대회 항목 13 차선변경 방향지시등 (합성 틱).

차선변경 시작 = 차로 경계 통과 틱(직전 틱 기준). cmd.turn_signal 이 해당
방향(좌=1/우=2)으로 signal_lead_s 이상 연속 점등돼 있어야 한다.
"""
import score as sc_mod
from conftest import mk_tick

SC = {'signal_lead_s': 3.0}
A, B = (1, 0, -1), (1, 0, -2)          # A 의 오른쪽 이웃이 B


class FakeLG:
    """detect 가 쓰는 neighbor/successors 만 흉내낸다."""

    def __init__(self, nbs=None, succs=None):
        self.nbs = nbs or {}
        self.succs = succs or {}

    def neighbor(self, k, side):
        return self.nbs.get((k, side))

    def successors(self, k):
        return self.succs.get(k, [])


LG = FakeLG(nbs={(A, 'right'): B, (B, 'left'): A})


def change_run(signal_from_s, dt=0.1, cross_t=5.0, n_before=60, signal=2):
    """cross_t 에 A→B 우측 차선변경. signal_from_s 이후 틱은 signal 점등."""
    ticks = []
    for i in range(n_before):
        t = cross_t - (n_before - i) * dt
        ticks.append(mk_tick(t=t, speed=5.0, lane=A,
                             turn_signal=signal if (signal_from_s is not None and t >= signal_from_s) else 0))
    ticks.append(mk_tick(t=cross_t, speed=5.0, lane=B, turn_signal=signal))
    return ticks


def run(ticks, merge_gap_s=0.0, lg=LG):
    return sc_mod.detect_lane_change_signal(ticks, ticks[0]['t'], lg, SC, merge_gap_s)


# ── 점등 시간 경계 ────────────────────────────────────────────────────────
def test_signal_on_long_enough_is_ok():
    evs = run(change_run(signal_from_s=1.5))               # 통과 직전 틱까지 3.4 s 점등
    assert len(evs) == 1 and evs[0]['signal_ok']
    assert evs[0]['on_s'] >= 3.0


def test_signal_on_just_under_lead_is_violation():
    # 직전 틱(t=4.9) 기준 2.9 s 점등 → 미달
    evs = run(change_run(signal_from_s=2.0))
    assert len(evs) == 1 and not evs[0]['signal_ok']
    assert evs[0]['on_s'] == 2.9


def test_signal_on_exactly_lead_is_ok():
    evs = run(change_run(signal_from_s=1.9))               # 4.9 − 1.9 = 3.0 s
    assert len(evs) == 1 and evs[0]['signal_ok']


def test_no_signal_is_violation_with_none_on_s():
    evs = run(change_run(signal_from_s=None))
    assert len(evs) == 1 and not evs[0]['signal_ok'] and evs[0]['on_s'] is None


def test_wrong_direction_signal_is_violation():
    evs = run(change_run(signal_from_s=0.0, signal=1))     # 우측 변경인데 좌측 점등
    assert len(evs) == 1 and not evs[0]['signal_ok']


# ── 차선변경 검출 관례 (solid_lane_change 와 동일) ────────────────────────
def test_successor_progress_is_not_change():
    lg = FakeLG(succs={A: [B]})
    assert run(change_run(None), lg=lg) == []


def test_reset_tick_excluded():
    ticks = change_run(None)
    ticks[-1]['world']['flags']['reset'] = True
    assert run(ticks) == []


def test_flicker_merges_to_one_change():
    ticks = change_run(None)
    tl = ticks[-1]['t']
    ticks.append(mk_tick(t=tl + 0.1, speed=5.0, lane=A))   # 다시 A (플리커)
    ticks.append(mk_tick(t=tl + 0.2, speed=5.0, lane=B))
    evs = run(ticks, merge_gap_s=1.0)
    assert len(evs) == 1 and evs[0]['n_crossings'] == 3
