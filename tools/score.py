"""
로그 → 위반 검출 + 채점.

검출 층은 "무엇을 몇 번 위반했는가"를 잡고, 그 위에 2026 HL FMA 안내문
(2026-08-27 공개) 채점 층(score_run)이 구간별 점수를 얹는다: 구간 100점,
(항목,구간)당 1회 감점(경미 -3/중대 -6, params scoring.*), 리스폰 구간당 1회
무료, 미완주 시 도달 구간까지만 집계. 검출과 채점은 분리 — 검출 건수는
severity 와 무관하게 유지된다.

    python3 tools/score.py logs/run_xxx.jsonl
    python3 tools/score.py LOG --route data/route_X.pkl --json

run_*.jsonl 을 **직접** 읽는다 (summarize_run 출력에 의존하지 않는다 — 두 도구는
독립). lane_graph/route/params 는 차로폭·완주 판정·임계에 쓴다.

검출 항목 (각각 위반 횟수 + 구간(route_s·틱 범위) + 정도):
  speed              법규 제한속도 초과 (구간별 50/30/스쿨존 구분, 연속 틱 = 1건)
  speed_margin       [정보] params speed.margin_kph 여유 침범 — 위반 아님
  lane_departure     |t_off| > 차로폭/2 − 차체폭/2 (계획된 차선변경 창·테이퍼 차로 제외)
  solid_lane_change  실선(left_solid/right_solid) 구간에서의 차로 변경
  red_light          적신호에 정지선 통과 (stop_line_front_m 부호 전환 시점 판정)
  red_right_turn     [정보] 적신호 우회전 통과 — 직전 일시정지 여부를 남긴다
  stop_line_encroach 정지 상태인데 앞범퍼가 정지선 너머 (침범 거리, 발진 크리프 제외)
  off_route          경로 이탈 구간 (route 있으면 최대 이탈 거리)
  reset              courseRespawn 리셋
  collision / near_miss  객체 외곽 간 최소 간격 ≤ params score.* 임계
  not_finished / overtime  미완주 / 제한시간(score.time_limit_s) 초과
  stall              정지 고착 — 계획은 진행(v_target≥0.5)인데 정지(v<0.1)가
                     batch.stall_end_s 이상 지속 (EndJudge 와 같은 의미)

각 판정은 로그의 WorldState/Command 만으로 재현 가능해야 한다.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import pathlib
import pickle
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / 'team_code'))

from kr_rules import plan_stop_s                               # noqa: E402 — 제어와 공용 (단일 출처)
from vtd_adapter.config import end_margin_m, load_params_yaml  # noqa: E402

# 정보성 집계 — 위반 총계(n_violations)에 넣지 않는다.
# overtime: 안내문 채점 규칙에 없음 (20분은 세팅 포함 운영 시간, 완주 시간은
# 동점 타이브레이커) — 검출은 유지하되 정보로 강등 (2026-08-27).
INFO_KEYS = ('speed_margin', 'red_right_turn', 'near_miss', 'overtime')

# ── 채점 매핑 (2026 HL FMA 안내문, params scoring.* 이 감점값의 단일 출처) ──
SCORING_ITEM = {'stop_line_encroach': 'red_light'}   # 정지선 침범은 "적색신호 정지" 항목으로 흡수
COUNT_ESCALATE = ('lane_departure', 'solid_lane_change')   # 구간 내 2회 이상 → 중대

RED = 1                    # 9910 신호 state (logger.py 주석: 1=적 4=좌 5=녹+좌)
LEFT_ARROW = 4


# ══════════════════════════════════════════════════════════════════════════
# 로드 / 공통
# ══════════════════════════════════════════════════════════════════════════
def load_ticks(path: str) -> list[dict]:
    """jsonl 에서 틱 레코드만."""
    out = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '"raw"' not in line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue                                   # 잘린 마지막 줄
    return out


def driving_span(ticks: list[dict]) -> list[dict]:
    """
    실제 주행 구간만 남긴다.

    경로를 완주한 뒤에는 route_s 가 다른 구간에 다시 붙으며 요동쳐서,
    그대로 보면 없는 위반이 잡힌다. route_s 가 최고점에 처음 닿을 때까지만 센다.
    """
    if not ticks:
        return ticks
    peak = max(t['ego']['route_s'] for t in ticks)
    for i, t in enumerate(ticks):
        if t['ego']['route_s'] >= peak - 0.5:
            return ticks[:i + 1]
    return ticks


def _runs(mask: list) -> list[tuple[int, int, object]]:
    """연속 동일값 구간 → [(i0, i1, value)]. value 가 falsy 인 구간은 버린다."""
    out, s0 = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[s0]:
            if mask[s0]:
                out.append((s0, i - 1, mask[s0]))
            s0 = i
    return out


def _ev(ticks: list[dict], t0: float, i0: int, i1: int, **extra) -> dict:
    """공통 이벤트 골격: 틱 범위 + 시각 + route_s 범위."""
    a, b = ticks[i0], ticks[i1]
    return {'i0': i0, 'i1': i1, 'ticks': i1 - i0 + 1,
            't0': round(a['t'] - t0, 1), 't1': round(b['t'] - t0, 1),
            'dur_s': round(b['t'] - a['t'], 2),
            's0': round(a['ego']['route_s'], 1), 's1': round(b['ego']['route_s'], 1),
            **extra}


# ══════════════════════════════════════════════════════════════════════════
# 1. 속도 초과 (+ 여유 침범 정보)
# ══════════════════════════════════════════════════════════════════════════
def detect_speed(ticks: list[dict], t0: float, margin_kph: float,
                 merge_gap_s: float = 0.0) -> tuple[list, list, dict]:
    """
    위반 기준은 **법규 제한속도 자체**다. params 의 speed.margin_kph 는 우리가
    스스로 둔 여유이므로 그걸 넘어도 위반이 아니다 — '여유 침범' 정보로만 센다.
    연속 초과 틱은 1건. 임계 바로 밑에서 진동하면 1틱짜리로 쪼개지므로,
    merge_gap_s 이하로 잠깐 내려갔다 다시 넘는 것도 같은 1건으로 병합한다.
    제한속도 구간(50/30/스쿨존)이 바뀌면 건을 끊는다.
    """
    def key(t):
        return (round((t['world']['speed_limit'] or 0) * 3.6), bool(t['world']['school_zone']))

    over_mask, tight_mask, groups = [], [], {}
    for t in ticks:
        lim, sz = key(t)
        v = t['ego']['speed'] * 3.6
        g = groups.setdefault((lim, sz), {'ticks': 0, 'over_ticks': 0, 'max_over': 0.0, 'v_max': 0.0})
        g['ticks'] += 1
        g['v_max'] = max(g['v_max'], v)
        over = lim > 0 and v > lim + 1e-6
        tight = lim > 0 and v > lim - margin_kph + 1e-6
        if over:
            g['over_ticks'] += 1
            g['max_over'] = max(g['max_over'], v - lim)
        over_mask.append((lim, sz) if over else None)
        tight_mask.append((lim, sz) if tight else None)

    def events(mask):
        runs = _runs(mask)
        merged = []
        for r in runs:
            if (merged and r[2] == merged[-1][2]
                    and ticks[r[0]]['t'] - ticks[merged[-1][1]]['t'] <= merge_gap_s):
                merged[-1] = (merged[-1][0], r[1], r[2])
            else:
                merged.append(r)
        out = []
        for i0, i1, (lim, sz) in merged:
            worst = max(range(i0, i1 + 1), key=lambda i: ticks[i]['ego']['speed'])
            v = ticks[worst]['ego']['speed'] * 3.6
            out.append(_ev(ticks, t0, i0, i1, limit_kph=lim, school_zone=sz,
                           max_kph=round(v, 1), max_over_kph=round(v - lim, 2)))
        return out

    return events(over_mask), events(tight_mask), groups


# ══════════════════════════════════════════════════════════════════════════
# 2. 차선 이탈
# ══════════════════════════════════════════════════════════════════════════
def detect_lane_departure(ticks: list[dict], t0: float, lg, veh_width: float,
                          route=None, merge_gap_s: float = 0.0) -> list:
    """|t_off| > 차로폭/2 − 차체폭/2. 차로폭은 lanegraph 의 해당 s 값.

    제외 3종 (2026-08-26 오탐 분석):
      · 계획된 차선변경 — 어댑터(PDM-lite)는 LANE_CHANGE 상태를 내지 않으므로
        state 대신 route 의 lane_change window(route_s 구간)로 판정한다
      · 테이퍼(소멸) 차로 — 끝 폭 < 차폭인 차로(route.py taper_blend 와 같은
        기준). 계획 경로가 중심선을 떠나 successor 로 붙는 구간이라 이 차로
        기준 t_off 판정은 무의미하다 (임계도 0 근처로 붕괴해 cm 단위 오탐)
      · reset 틱
    진동이 경계를 여러 번 넘으면 잘게 쪼개지므로 merge_gap_s 이하 끊김은
    같은 1건으로 병합한다 (speed/off_route 와 같은 규칙)."""
    windows = []
    if route:
        windows = [(float(e['window_s0']), float(e['window_s1']))
                   for e in route.get('events', [])
                   if e['kind'].startswith('lane_change') and 'window_s0' in e]
    taper_cache: dict = {}      # lane -> 끝 폭 < 차폭 (소멸 차로)

    def is_taper(key) -> bool:
        if key not in taper_cache:
            try:
                taper_cache[key] = lg.width_at(key, lg.length(key)) < veh_width
            except KeyError:
                taper_cache[key] = False
        return taper_cache[key]

    mask = []
    for t in ticks:
        e = t['ego']
        rs = e['route_s']
        ok = (t['world']['valid'] and e['lane']
              and t['decision']['state'] != 'LANE_CHANGE'
              and not any(a - 1e-6 <= rs <= b + 1e-6 for a, b in windows)
              and not t['world']['flags'].get('reset')
              and not is_taper(tuple(e['lane'])))
        thr = None
        if ok:
            try:
                thr = lg.width_at(tuple(e['lane']), e['s']) / 2.0 - veh_width / 2.0
            except KeyError:
                thr = None
        mask.append(thr is not None and thr > 0 and abs(e['t_off']) > thr)
    runs = _runs(mask)
    merged = []
    for r in runs:
        if merged and ticks[r[0]]['t'] - ticks[merged[-1][1]]['t'] <= merge_gap_s:
            merged[-1] = (merged[-1][0], r[1], True)
        else:
            merged.append(r)
    out = []
    for i0, i1, _ in merged:
        # 병합 구간에는 임계 미달(gap) 틱도 섞인다 — 극값은 판정된 틱 중에서만
        worst = max((i for i in range(i0, i1 + 1) if mask[i]),
                    key=lambda i: abs(ticks[i]['ego']['t_off']))
        e = ticks[worst]['ego']
        thr = lg.width_at(tuple(e['lane']), e['s']) / 2.0 - veh_width / 2.0
        out.append(_ev(ticks, t0, i0, i1, lane=e['lane'],
                       max_t_off=round(e['t_off'], 2),
                       exceed_m=round(abs(e['t_off']) - thr, 2)))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 3. 실선 차선변경
# ══════════════════════════════════════════════════════════════════════════
def detect_solid_lane_change(ticks: list[dict], t0: float, lg,
                             merge_gap_s: float = 0.0) -> list:
    """left_solid/right_solid 인 구간에서 lane 이 옆 차로로 바뀌면 위반.
    방향은 lanegraph 이웃 관계로 판정 (successor 진행은 차로변경이 아니다).
    실선에 걸친 채 lane 배정이 좌↔우로 플리커하면 같은 차로쌍의 전환이
    merge_gap_s 안에 연달아 잡힌다 — 1건으로 병합하고 횟수만 남긴다."""
    out = []
    for i in range(1, len(ticks)):
        a, b = ticks[i - 1], ticks[i]
        la, lb = a['ego']['lane'], b['ego']['lane']
        if not la or not lb or tuple(la) == tuple(lb):
            continue
        if b['world']['flags'].get('reset') or a['world']['flags'].get('reset'):
            continue                                       # 리스폰 텔레포트는 제외
        ka, kb = tuple(la), tuple(lb)
        side = None
        try:
            if lg.neighbor(ka, 'left') == kb:
                side = 'left'
            elif lg.neighbor(ka, 'right') == kb:
                side = 'right'
            elif kb in lg.successors(ka):
                side = None                                # 정상 진행
            else:
                # 세그먼트 이음매에서 대각으로 잡히는 경우: successor 의 이웃인지
                for s in lg.successors(ka):
                    if lg.neighbor(s, 'left') == kb:
                        side = 'left'
                    elif lg.neighbor(s, 'right') == kb:
                        side = 'right'
        except KeyError:
            continue
        if side is None:
            continue
        solid = a['world']['left_solid'] if side == 'left' else a['world']['right_solid']
        if not solid:
            continue
        pair = frozenset((ka, kb))
        if (out and out[-1]['_pair'] == pair
                and b['t'] - ticks[out[-1]['i1']]['t'] <= merge_gap_s):
            out[-1]['i1'] = i
            out[-1]['t1'] = round(b['t'] - t0, 1)
            out[-1]['s1'] = round(b['ego']['route_s'], 1)
            out[-1]['n_crossings'] += 1
            continue
        ev = _ev(ticks, t0, i - 1, i, side=side, from_lane=la, to_lane=lb, n_crossings=1)
        ev['_pair'] = pair
        out.append(ev)
    for ev in out:
        del ev['_pair']
    return out


# ══════════════════════════════════════════════════════════════════════════
# 4. 적신호 통과 / 5. 정지선 침범
# ══════════════════════════════════════════════════════════════════════════
def _crossings(ticks: list[dict]) -> list[int]:
    """정지선 통과 틱: stop_line_front_m 이 음수→비음수로 전환되는 i."""
    out, prev = [], None
    for i, t in enumerate(ticks):
        f = t['world'].get('stop_line_front_m')
        if prev is not None and f is not None and prev < 0 <= f:
            out.append(i)
        prev = f
    return out


def detect_red_light(ticks: list[dict], t0: float, stop_speed: float) -> tuple[list, list]:
    """
    정지선 통과 시점에 해당 신호가 적색이면 위반.

    9910 light id 는 xodr <controller> id 다 — 직전 틱 flags.stop_ctrl_ids
    (전방 정지선의 controller 목록)와 raw.lights 를 대조한다 (route.py 관례).
    state 4(좌회전 화살표)는 좌회전 경로가 아니면 적색과 동등.
    우회전(next_turn=turn_right)은 적신호에도 통과 관행이 있어 별도 집계하고,
    직전 10 s 내 일시정지 여부(stopped_before)를 남긴다.
    front_m 이 0 근처에서 진동하면 같은 정지선이 여러 번 잡힌다 — 같은
    controller 를 5 s 안에 다시 넘는 건 무시한다.
    """
    viol, right = [], []
    last = None                                            # (frozenset(ctrl), t)
    for i in _crossings(ticks):
        prev = ticks[i - 1]
        ctrl = prev['world']['flags'].get('stop_ctrl_ids') or []
        if not ctrl:
            continue                                       # 신호 없는 정지선
        if last is not None and last[0] == frozenset(ctrl) and ticks[i]['t'] - last[1] < 5.0:
            continue                                       # 같은 정지선 재통과(진동)
        last = (frozenset(ctrl), ticks[i]['t'])
        states = dict((int(a), int(b)) for a, b in ticks[i]['raw']['lights'])
        states.update({int(a): int(b) for a, b in prev['raw']['lights']
                       if int(a) not in states})
        got = [states[c] for c in ctrl if c in states]
        if not got:
            continue
        summ = prev['world'].get('summ') or {}
        turn = summ.get('next_turn')
        d_turn = summ.get('dist_next_turn')
        near_turn = d_turn is not None and d_turn < 30.0
        red = any(s == RED for s in got) or \
            (any(s == LEFT_ARROW for s in got) and not (turn == 'turn_left' and near_turn))
        if not red:
            continue
        ev = _ev(ticks, t0, i, i, ctrl_ids=ctrl, states=got,
                 v_kph=round(ticks[i]['ego']['speed'] * 3.6, 1), next_turn=turn)
        if turn == 'turn_right' and near_turn:
            j = i
            stopped = False
            while j > 0 and ticks[i]['t'] - ticks[j]['t'] < 10.0:
                if ticks[j]['ego']['speed'] < stop_speed:
                    stopped = True
                    break
                j -= 1
            ev['stopped_before'] = stopped
            right.append(ev)
        else:
            viol.append(ev)
    return viol, right


def detect_stop_line_encroach(ticks: list[dict], t0: float, stop_speed: float) -> list:
    """정지 상태(v<stop_speed)인데 앞범퍼가 정지선 너머(stop_line_front_m>0).
    발진 크리프 제외: v_target>0 이면 이미 출발 계획이 난 구간이라 침범으로
    세지 않는다 (녹색 전환 후 v<stop_speed 인 첫 몇 틱이 잡히던 오탐)."""
    mask = [t['ego']['speed'] < stop_speed
            and (t['world'].get('stop_line_front_m') or 0) > 0
            and (t['decision'].get('v_target') or 0.0) <= 1e-3 for t in ticks]
    out = []
    for i0, i1, _ in _runs(mask):
        worst = max(range(i0, i1 + 1), key=lambda i: ticks[i]['world']['stop_line_front_m'])
        out.append(_ev(ticks, t0, i0, i1,
                       encroach_m=round(ticks[worst]['world']['stop_line_front_m'], 2)))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 6. 경로 이탈 / 7. 리셋
# ══════════════════════════════════════════════════════════════════════════
def _route_polyline(lg, route):
    import numpy as np
    pts = []
    for k in route['lanes']:
        r = lg.lanes.get(tuple(k))
        if r is not None:
            pts.append(np.asarray(r['pts'])[:, :2])
    return np.vstack(pts) if pts else None


def detect_off_route(ticks: list[dict], t0: float, lg=None, route=None,
                     merge_gap_s: float = 0.0) -> list:
    """flags.off_route 연속 구간 = 1건 (beside_route 와의 틱 단위 플리커는
    merge_gap_s 로 병합). route 가 있으면 경로 중심선까지의 최대 거리도 낸다
    (off 틱에서만 계산 — 전체 틱이면 느리다)."""
    poly = _route_polyline(lg, route) if (lg is not None and route is not None) else None
    mask = [bool(t['world']['flags'].get('off_route')) for t in ticks]
    runs = _runs(mask)
    merged = []
    for r in runs:
        if merged and ticks[r[0]]['t'] - ticks[merged[-1][1]]['t'] <= merge_gap_s:
            merged[-1] = (merged[-1][0], r[1], True)
        else:
            merged.append(r)
    out = []
    for i0, i1, _ in merged:
        dmax = None
        if poly is not None:
            import numpy as np
            dmax = 0.0
            for i in range(i0, i1 + 1):
                e = ticks[i]['ego']
                dmax = max(dmax, float(np.min(np.hypot(poly[:, 0] - e['x'],
                                                       poly[:, 1] - e['y']))))
            dmax = round(dmax, 1)
        out.append(_ev(ticks, t0, i0, i1, max_dist_m=dmax))
    return out


def detect_reset(ticks: list[dict], t0: float) -> list:
    """courseRespawn — flags.reset 틱마다 1건, 사유 플래그를 같이 남긴다."""
    out = []
    for i, t in enumerate(ticks):
        fl = t['world']['flags']
        if not fl.get('reset'):
            continue
        why = {k: fl[k] for k in fl if k.startswith('reset_') or k in ('jump_m', 'toff_jump')}
        out.append(_ev(ticks, t0, i, i, why=why))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 8. 충돌·근접 — 회전 사각형 외곽 간 최소 간격
# ══════════════════════════════════════════════════════════════════════════
def _corners(cx, cy, yaw, length, width):
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length / 2.0, width / 2.0
    return [(cx + c * dx - s * dy, cy + s * dx + c * dy)
            for dx, dy in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def _seg_dist(p, q, a, b):
    """선분 pq ↔ 선분 ab 최소거리."""
    def pt_seg(px, py, ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        L2 = vx * vx + vy * vy
        u = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * vx + (py - ay) * vy) / L2))
        return math.hypot(px - (ax + u * vx), py - (ay + u * vy))
    return min(pt_seg(*p, *a, *b), pt_seg(*q, *a, *b),
               pt_seg(*a, *p, *q), pt_seg(*b, *p, *q))


def _rect_gap(A, B):
    """두 볼록 사각형 외곽 간 거리. 겹치면 0 (SAT 판정)."""
    def axes(P):
        out = []
        for i in range(4):
            x1, y1 = P[i]
            x2, y2 = P[(i + 1) % 4]
            n = (-(y2 - y1), x2 - x1)
            L = math.hypot(*n)
            if L > 1e-9:
                out.append((n[0] / L, n[1] / L))
        return out
    overlap = True
    for ax, ay in axes(A) + axes(B):
        pa = [x * ax + y * ay for x, y in A]
        pb = [x * ax + y * ay for x, y in B]
        if max(pa) < min(pb) or max(pb) < min(pa):
            overlap = False
            break
    if overlap:
        return 0.0
    return min(_seg_dist(A[i], A[(i + 1) % 4], B[j], B[(j + 1) % 4])
               for i in range(4) for j in range(4))


def detect_proximity(ticks: list[dict], t0: float, cfg: dict) -> tuple[list, list]:
    """객체와의 외곽 간 최소 간격이 임계 이하인 구간.
    간격 ≤ score.collision_dist_m → 충돌, ≤ score.near_dist_m → 근접 경고."""
    sc, vh = cfg['score'], cfg['vehicle']
    col_thr = float(sc['collision_dist_m'])
    near_thr = float(sc['near_dist_m'])
    e_len, e_wid = float(vh['length']), float(vh['width'])
    # ego x/y 는 뒷축 — 차체 중심은 (전장/2 − 뒤 오버행)만큼 전방
    rear_ov = e_len - float(vh['wheelbase']) - float(vh['front_overhang_m'])
    c_off = e_len / 2.0 - rear_ov

    gaps = []                      # (gap, obj_id) or None
    for t in ticks:
        e = t['ego']
        best = None
        if t['raw']['objects']:
            ego_rect = _corners(e['x'] + c_off * math.cos(e['yaw']),
                                e['y'] + c_off * math.sin(e['yaw']),
                                e['yaw'], e_len, e_wid)
            for o in t['raw']['objects']:
                oid, ox, oy, oh = int(o[0]), o[1], o[2], o[4]
                ol, ow = (o[6] or 1.0), (o[7] or 1.0)
                cd = math.hypot(ox - e['x'], oy - e['y'])
                if cd > near_thr + (e_len + ol) / 2.0 + 3.0:
                    continue                               # 프리필터
                gap = _rect_gap(ego_rect, _corners(ox, oy, oh, ol, ow))
                if best is None or gap < best[0]:
                    best = (gap, oid)
        gaps.append(best if best is not None and best[0] <= near_thr else None)

    cls_by_id = {}
    for t in ticks:
        for o in t['objects']:
            cls_by_id.setdefault(o['id'], o.get('cls'))

    collisions, nears = [], []
    mask = [bool(g) for g in gaps]
    for i0, i1, _ in _runs(mask):
        worst = min(range(i0, i1 + 1), key=lambda i: gaps[i][0])
        gap, oid = gaps[worst]
        ev = _ev(ticks, t0, i0, i1, obj_id=oid, obj_cls=cls_by_id.get(oid),
                 min_gap_m=round(gap, 2),
                 v_kph=round(ticks[worst]['ego']['speed'] * 3.6, 1))
        (collisions if gap <= col_thr else nears).append(ev)
    return collisions, nears


# ══════════════════════════════════════════════════════════════════════════
# 9. 미완주·시간 초과 / 10. 정지 고착
# ══════════════════════════════════════════════════════════════════════════
def detect_finish(ticks: list[dict], t0: float, route, cfg: dict,
                  finish_s: float | None = None) -> dict:
    """완주 판정.

    규칙(안내문): **뒷바퀴 축이 종료 지점 좌표 통과** — ego x/y 가 뒷축 기준이라
    (AGENT_SPEC §1.3) scoring.finish_xy 를 경로에 투영한 route_s(finish_s) 도달로
    판정한다. finish_s 가 없으면 기존 route_s 임계(end_margin — batch_run 의
    운영 종료 판정과 같은 식)로 폴백한다.
    """
    limit_s = float(cfg['score']['time_limit_s'])
    s = [t['ego']['route_s'] for t in ticks]
    peak_i = max(range(len(s)), key=lambda i: s[i])
    res = {'peak_route_s': round(s[peak_i], 1), 'route_total': None,
           'done': None, 'finish_time_s': None, 'time_limit_s': limit_s,
           'finish_basis': 'finish_xy' if finish_s is not None else 'route_s_margin',
           'wall_s': round(ticks[-1]['t'] - t0, 1)}
    not_finished, overtime = [], []
    if route is not None:
        total = float(route['total_length'])
        thr = float(finish_s) if finish_s is not None else total - end_margin_m(cfg)
        res['route_total'] = round(total, 1)
        hit = next((i for i in range(len(s)) if s[i] >= thr), None)
        res['done'] = hit is not None
        if hit is not None:
            res['finish_time_s'] = round(ticks[hit]['t'] - t0, 1)
            if res['finish_time_s'] > limit_s:
                overtime.append(_ev(ticks, t0, hit, hit,
                                    finish_time_s=res['finish_time_s'], limit_s=limit_s))
        else:
            not_finished.append(_ev(ticks, t0, peak_i, peak_i,
                                    peak_route_s=res['peak_route_s'],
                                    route_total=res['route_total']))
            if res['wall_s'] > limit_s:
                overtime.append(_ev(ticks, t0, len(ticks) - 1, len(ticks) - 1,
                                    finish_time_s=None, limit_s=limit_s))
    return {'summary': res, 'not_finished': not_finished, 'overtime': overtime}


def detect_stall(ticks: list[dict], t0: float, cfg: dict) -> list:
    """정지 고착: 계획은 진행(v_target≥0.5)인데 정지(v<0.1)가 batch.stall_end_s
    이상 지속. batch_run 의 EndJudge stall 판정과 같은 의미·같은 임계."""
    hold_s = float(cfg['batch']['stall_end_s'])
    mask = [t['ego']['speed'] < 0.1 and (t['decision'].get('v_target') or 0.0) >= 0.5
            for t in ticks]
    out = []
    for i0, i1, _ in _runs(mask):
        dur = ticks[i1]['t'] - ticks[i0]['t']
        if dur >= hold_s:
            out.append(_ev(ticks, t0, i0, i1))
    return out


# ══════════════════════════════════════════════════════════════════════════
# 채점 층 (2026 HL FMA 안내문) — 검출 결과 위에 얹는다. 검출 로직은 불변.
# ══════════════════════════════════════════════════════════════════════════
def project_route_s(lg, route, x: float, y: float) -> float | None:
    """좌표 → 경로 누적거리 (종료 지점 통과 판정용). 경로 차로들에 투영해 최근접."""
    best = None
    for i, k in enumerate(route['lanes']):
        try:
            s_p, _t, d_p, _ = lg.project(tuple(k), x, y)
        except KeyError:
            continue
        if best is None or d_p < best[0]:
            best = (d_p, float(route['cum_s'][i]) + float(s_p))
    return best[1] if best else None


def _severity(cat: str, ev: dict, sc: dict) -> str:
    """이벤트 → 'minor' | 'major' | 'none' (안내문 매핑, 작업3).

    · speed: 초과 ≤ speed_allow_kph 감점 없음 / ~speed_major_kph 경미 / 초과 중대
    · lane_departure / solid_lane_change: 건당 경미 — 구간 내 2회 이상이면
      score_run 이 중대로 심화 (COUNT_ESCALATE)
    · red_light(정지선 통과·무정차): 중대.  stop_line_encroach(정지는 했으나
      앞범퍼가 선을 넘음): 같은 항목의 경미로 흡수 — "2 m 이내 정지" 품질
      판정(멀리 정지 = 경미)은 별도 검출기 필요라 이번 범위 밖
    · collision: 중대.  reset 은 score_run 특칙.  그 외(off_route·stall 등)는
      안내문 매핑 미확정 — 'none' (검출·집계는 유지, 감점 없음)
    """
    if cat == 'speed':
        over = float(ev.get('max_over_kph', 0.0))
        if over <= float(sc['speed_allow_kph']):
            return 'none'
        return 'major' if over > float(sc['speed_major_kph']) else 'minor'
    if cat in COUNT_ESCALATE:
        return 'minor'
    if cat == 'red_light':
        return 'major'
    if cat == 'stop_line_encroach':
        return 'minor'
    if cat == 'collision':
        return 'major'
    if cat == 'reset':
        return 'major'                  # 초과분에만 적용 (score_run 특칙)
    return 'none'


def _section_edges(sc: dict, total: float) -> list[float]:
    bounds = sorted(float(b) for b in (sc.get('section_bounds_s') or []))
    return [0.0] + [b for b in bounds if 0.0 < b < total] + [max(total, 1e-6)]


def annotate_scoring(rep: dict, cfg: dict) -> None:
    """모든 위반 이벤트에 severity 와 section_idx(이벤트 s0 기준)를 붙인다."""
    sc = cfg['scoring']
    total = rep['finish'].get('route_total') or rep['finish']['peak_route_s']
    edges = _section_edges(sc, float(total))
    for cat, d in rep['violations'].items():
        for ev in d['events']:
            ev['severity'] = _severity(cat, ev, sc) if cat not in INFO_KEYS else 'none'
            ev['section_idx'] = min(bisect.bisect_right(edges, ev['s0']) - 1,
                                    len(edges) - 2)


def score_run(rep: dict, cfg: dict) -> dict:
    """구간별 독립 채점 (2026 HL FMA 안내문).

    구간 100점 시작 · (항목,구간)당 1회 감점(경미 minor_penalty / 중대
    major_penalty) · 경미+중대 공존이면 중대 하나 · 차로유지·실선은 구간 내
    2회 이상이면 중대(경미 심화 +3 = 총 -6) · 리스폰은 구간당
    respawn_free_per_section 회 무감점, 초과분 1회당 -major 누적(1회 규칙
    미적용) · 미완주면 뒷축이 진입한 구간까지만 만점·감점 집계 · 음수 허용.
    """
    sc = cfg['scoring']
    minor, major = int(sc['minor_penalty']), int(sc['major_penalty'])
    free = int(sc['respawn_free_per_section'])
    total = float(rep['finish'].get('route_total') or rep['finish']['peak_route_s'])
    edges = _section_edges(sc, total)
    n_sec = len(edges) - 1
    peak = float(rep['finish']['peak_route_s'])
    reached = n_sec if rep['finish'].get('done') else \
        max(1, sum(1 for i in range(n_sec) if peak > edges[i] + 1e-6))

    buckets: dict = {}                  # (item, sec) -> [(severity, i0)]
    resets: dict = {}                   # sec -> [i0, ...]
    for cat, d in rep['violations'].items():
        if cat in INFO_KEYS:
            continue
        for ev in d['events']:
            sec = int(ev.get('section_idx', 0))
            if cat == 'reset':
                resets.setdefault(sec, []).append(ev['i0'])
                continue
            if ev.get('severity', 'none') == 'none':
                continue
            item = SCORING_ITEM.get(cat, cat)
            buckets.setdefault((item, sec), []).append((ev['severity'], ev['i0']))

    sections = []
    for i in range(n_sec):
        deds = []
        for (item, sec), evs in sorted(buckets.items()):
            if sec != i:
                continue
            sev = 'major' if (any(s == 'major' for s, _ in evs)
                              or (item in COUNT_ESCALATE and len(evs) >= 2)) else 'minor'
            deds.append({'item': item, 'severity': sev, 'event_i0': evs[0][1]})
        for i0 in sorted(resets.get(i, []))[free:]:
            deds.append({'item': 'reset', 'severity': 'major', 'event_i0': i0})
        entered = i < reached
        score = (100 - sum(major if d['severity'] == 'major' else minor
                           for d in deds)) if entered else None
        sections.append({'idx': i, 's0': round(edges[i], 1), 's1': round(edges[i + 1], 1),
                         'score': score, 'deductions': deds})
    return {'sections': sections,
            'total': sum(s['score'] for s in sections if s['score'] is not None),
            'max_possible': 100 * reached,
            'reached_sections': reached}


# ══════════════════════════════════════════════════════════════════════════
# 종합
# ══════════════════════════════════════════════════════════════════════════
def analyze(log_path: str, cfg: dict, lg=None, route=None,
            all_ticks: bool = False) -> dict:
    """한 로그의 위반 목록. batch_run 이 이 dict 를 표에 얹는다."""
    ticks = load_ticks(log_path)
    rep: dict = {'file': pathlib.Path(log_path).name, 'ticks': len(ticks),
                 'warnings': [], 'violations': {}}
    if not ticks:
        rep['warnings'].append('틱 레코드가 없다')
        rep['n_violations'] = 0
        return rep
    span = ticks if all_ticks else driving_span(ticks)
    rep['span_ticks'] = len(span)
    t0 = span[0]['t']
    stop_speed = float(cfg['score']['stop_speed_mps'])
    margin_kph = float(cfg['speed']['margin_kph'])
    V = rep['violations']

    over, tight, groups = detect_speed(span, t0, margin_kph,
                                       float(cfg['score'].get('merge_gap_s', 0.0)))
    V['speed'] = {'count': len(over), 'events': over}
    V['speed_margin'] = {'count': len(tight), 'events': tight}
    rep['speed_groups'] = {f'{lim}{"/스쿨존" if sz else ""}': g
                           for (lim, sz), g in sorted(groups.items())}

    if lg is not None:
        V['lane_departure'] = {'count': 0, 'events': detect_lane_departure(
            span, t0, lg, float(cfg['vehicle']['width']), route,
            float(cfg['score'].get('merge_gap_s', 0.0)))}
        V['solid_lane_change'] = {'count': 0, 'events': detect_solid_lane_change(
            span, t0, lg, float(cfg['score'].get('merge_gap_s', 0.0)))}
    else:
        rep['warnings'].append('lane_graph 없음 — 차선 이탈·실선 차선변경 생략')
        V['lane_departure'] = {'count': 0, 'events': []}
        V['solid_lane_change'] = {'count': 0, 'events': []}

    red, red_right = detect_red_light(span, t0, stop_speed)
    V['red_light'] = {'count': 0, 'events': red}
    V['red_right_turn'] = {'count': 0, 'events': red_right}
    V['stop_line_encroach'] = {'count': 0, 'events': detect_stop_line_encroach(span, t0, stop_speed)}
    V['off_route'] = {'count': 0, 'events': detect_off_route(
        span, t0, lg, route, float(cfg['score'].get('merge_gap_s', 0.0)))}

    # 완주: 안내문 규칙은 "뒷축이 종료 지점 좌표 통과" — finish_xy 를 경로에
    # 투영한 route_s 로 판정. 미설정이면 기존 route_s 임계(end_margin) + 경고.
    finish_s = None
    fxy = cfg['scoring'].get('finish_xy')
    if fxy and lg is not None and route is not None:
        finish_s = project_route_s(lg, route, float(fxy[0]), float(fxy[1]))
        if finish_s is None:
            rep['warnings'].append('finish_xy 를 경로에 투영하지 못함 — route_s 임계로 폴백')
    elif not fxy:
        rep['warnings'].append('scoring.finish_xy 미설정 — 완주를 route_s 임계(end_margin)로 판정')
    fin = detect_finish(span, t0, route, cfg, finish_s)
    rep['finish'] = fin['summary']
    if route is None:
        rep['warnings'].append('route 없음 — 완주/시간초과 판정·이탈거리 생략')
    else:
        # 계획 정지점 vs 종료선 정합 — 제어(kr_rules.plan_stop_s)와 같은 식(단일 출처).
        # 채점 통과 기준은 finish_s 그 자체 (clearance 는 제어 여유일 뿐).
        total = float(route['total_length'])
        stop_s, _clipped = plan_stop_s(cfg, total, finish_s)
        front = float(cfg['vehicle']['wheelbase']) + float(cfg['vehicle']['front_overhang_m'])
        planned_stop = stop_s - float(cfg['speed']['stop_gap_m']) - front   # 계획 정지 시 뒷축
        rep['finish']['finish_s'] = None if finish_s is None else round(float(finish_s), 1)
        rep['finish']['planned_stop_s'] = round(planned_stop, 1)
        rep['finish']['margin_m'] = (None if finish_s is None
                                     else round(planned_stop - float(finish_s), 2))
        if finish_s is not None and planned_stop - float(finish_s) <= 0:
            rep['warnings'].append(
                f'계획 정지점(뒷축 {planned_stop:.1f} m)이 종료선(finish_s '
                f'{float(finish_s):.1f} m)을 못 넘는다 — 정상 정지해도 미완주 채점')
    V['not_finished'] = {'count': 0, 'events': fin['not_finished']}
    V['overtime'] = {'count': 0, 'events': fin['overtime']}

    # 리셋·충돌·정지 고착은 물리적 사실 — 미완주 런은 주행 구간(peak 컷) 밖에서도
    # 센다 (courseRespawn 은 route_s 를 뒤로 되돌려 peak 직후에 오는 경우가 많다).
    # 완주한 런의 peak 이후는 VTD 시나리오 재시작 아티팩트라 제외한다
    # (실측: 완주 정차 후 5 m 간격 리스폰 9회가 로그에 남았다).
    tail = span if fin['summary']['done'] else ticks
    V['reset'] = {'count': 0, 'events': detect_reset(tail, t0)}
    col, near = detect_proximity(tail, t0, cfg)
    V['collision'] = {'count': 0, 'events': col}
    V['near_miss'] = {'count': 0, 'events': near}
    V['stall'] = {'count': 0, 'events': detect_stall(tail, t0, cfg)}

    for k, d in V.items():
        d['count'] = len(d['events'])
    rep['n_violations'] = sum(d['count'] for k, d in V.items() if k not in INFO_KEYS)

    # ── 채점 층 (검출 결과는 위에서 확정 — 여기서는 severity·구간·점수만 얹는다)
    annotate_scoring(rep, cfg)
    rep['scoring'] = score_run(rep, cfg)
    return rep


# ══════════════════════════════════════════════════════════════════════════
# 출력
# ══════════════════════════════════════════════════════════════════════════
LABEL = {
    'speed': '속도 초과(법규)', 'speed_margin': '여유 침범(정보)',
    'lane_departure': '차선 이탈', 'solid_lane_change': '실선 차선변경',
    'red_light': '적신호 통과', 'red_right_turn': '적신호 우회전(정보)',
    'stop_line_encroach': '정지선 침범', 'off_route': '경로 이탈',
    'reset': '리셋(courseRespawn)', 'collision': '충돌', 'near_miss': '근접 경고(정보)',
    'not_finished': '미완주', 'overtime': '시간 초과(정보)', 'stall': '정지 고착',
}

_EXTRA = {          # 이벤트 상세에 덧붙일 항목별 필드
    'speed': ('limit_kph', 'school_zone', 'max_kph', 'max_over_kph'),
    'speed_margin': ('limit_kph', 'school_zone', 'max_kph'),
    'lane_departure': ('lane', 'max_t_off', 'exceed_m'),
    'solid_lane_change': ('side', 'from_lane', 'to_lane', 'n_crossings'),
    'red_light': ('ctrl_ids', 'states', 'v_kph', 'next_turn'),
    'red_right_turn': ('states', 'v_kph', 'stopped_before'),
    'stop_line_encroach': ('encroach_m',),
    'off_route': ('max_dist_m',),
    'reset': ('why',),
    'collision': ('obj_id', 'obj_cls', 'min_gap_m', 'v_kph'),
    'near_miss': ('obj_id', 'obj_cls', 'min_gap_m', 'v_kph'),
    'not_finished': ('peak_route_s', 'route_total'),
    'overtime': ('finish_time_s', 'limit_s'),
    'stall': (),
}


def render(rep: dict) -> str:
    L = [f"{'═' * 72}", f" {rep['file']}   틱 {rep['ticks']}"
         + (f" (주행 구간 {rep['span_ticks']})" if 'span_ticks' in rep else ''), '═' * 72]
    for w in rep['warnings']:
        L.append(f'⚠ {w}')
    if 'finish' in rep:
        f = rep['finish']
        done = {True: '완주', False: '**미완주**', None: '판정불가(route 없음)'}[f['done']]
        ft = f"{f['finish_time_s']} s" if f['finish_time_s'] is not None else '—'
        tot = f"/{f['route_total']}" if f['route_total'] else ''
        L.append(f"완주: {done}  route_s {f['peak_route_s']}{tot} m  "
                 f"완주시간 {ft} (제한 {f['time_limit_s']:.0f} s, 벽시계 {f['wall_s']} s)")
        if f.get('finish_s') is not None:
            L.append(f"종료선: finish_s {f['finish_s']} m  계획 정지 뒷축 "
                     f"{f['planned_stop_s']} m  여유 {f['margin_m']:+.2f} m")
    if 'speed_groups' in rep:
        L.append('구간별 속도: ' + '  '.join(
            f"[{k}] {g['ticks']}틱 v_max {g['v_max']:.1f}"
            + (f" 위반 {g['over_ticks']}틱 +{g['max_over']:.2f}" if g['over_ticks'] else '')
            for k, g in rep['speed_groups'].items()))
    L.append('')
    L.append(f"{'항목':<22} {'건수':>4}")
    for k in LABEL:
        if k not in rep['violations']:
            continue
        d = rep['violations'][k]
        mark = '' if d['count'] == 0 else ('  ←' if k not in INFO_KEYS else '')
        L.append(f"{LABEL[k]:<22} {d['count']:>4}{mark}")
    L.append(f"\n위반 합계(정보 제외): {rep.get('n_violations', 0)}")
    if 'scoring' in rep:
        s = rep['scoring']
        L.append(f"\n[채점] 총점 {s['total']} / {s['max_possible']}"
                 f"  (구간 {len(s['sections'])}개 중 도달 {s['reached_sections']})")
        for sec in s['sections']:
            if sec['score'] is None:
                L.append(f"  구간{sec['idx']} s {sec['s0']}~{sec['s1']}: 미도달 (집계 제외)")
                continue
            ded = '  '.join(f"{LABEL.get(d['item'], d['item'])}"
                            f"[{'중대' if d['severity'] == 'major' else '경미'}]"
                            for d in sec['deductions']) or '감점 없음'
            L.append(f"  구간{sec['idx']} s {sec['s0']}~{sec['s1']}: {sec['score']}점  {ded}")
    for k in LABEL:
        d = rep['violations'].get(k)
        if not d or not d['events']:
            continue
        L.append(f"\n── {LABEL[k]} ({d['count']}건) " + '─' * 40)
        cap = 15 if k in INFO_KEYS else None       # 정보성 항목은 상세를 줄인다 (--json 은 전부)
        for ev in d['events'][:cap]:
            base = (f"  s {ev['s0']}→{ev['s1']} m  t {ev['t0']}→{ev['t1']} s "
                    f"({ev['dur_s']} s, {ev['ticks']}틱)")
            extra = '  '.join(f'{f}={ev[f]}' for f in _EXTRA.get(k, ()) if f in ev)
            L.append(base + ('  ' + extra if extra else ''))
        if cap is not None and d['count'] > cap:
            L.append(f'  … 외 {d["count"] - cap}건 (--json 으로 전부)')
    return '\n'.join(L) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='주행 로그 위반 검출 (배점 없음)')
    ap.add_argument('log')
    ap.add_argument('--route', default='data/route.pkl', help='route pkl (완주·이탈거리 판정)')
    ap.add_argument('--graph', default='data/lane_graph.pkl', help='lane_graph pkl (차로폭)')
    ap.add_argument('--params', default=None, help='params.yaml 경로 (기본: config/params.yaml)')
    ap.add_argument('--json', action='store_true', help='JSON 으로 출력')
    ap.add_argument('--all-ticks', action='store_true',
                    help='완주 이후 구간도 포함 (기본: 주행 구간만)')
    args = ap.parse_args(argv)

    cfg = load_params_yaml(args.params)
    lg = route = None
    try:
        from vtd_adapter.lanegraph import LaneGraph
        lg = LaneGraph(args.graph)
    except Exception as e:                                 # noqa: BLE001 — 없으면 항목 생략
        print(f'⚠ lane_graph 로드 실패: {e}', file=sys.stderr)
    try:
        with open(args.route, 'rb') as f:
            route = pickle.load(f)
    except Exception as e:                                 # noqa: BLE001
        print(f'⚠ route 로드 실패: {e}', file=sys.stderr)

    rep = analyze(args.log, cfg, lg, route, all_ticks=args.all_ticks)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    else:
        print(render(rep))
    return 0 if rep.get('n_violations', 0) == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
