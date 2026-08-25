"""
주행 로그 한 줄 요약 — **매 주행 후 가장 먼저 돌린다.**

    python3 tools/summarize_run.py logs/run_xxx.jsonl
    python3 tools/summarize_run.py logs/run_xxx.jsonl --json   # 기계용

로그 jsonl 만 읽는다 (lane_graph/route 는 있으면 차로 폭·이벤트 보강에 쓴다).
여기서 이상으로 뜬 항목만 깊이 본다 (replay.py / ana 스크립트).

항목: 완주, 거리/시간/평균속도, 리스폰/스톨/드롭, 제한속도 초과, |t_off| 통계와
반폭 초과 구간, LC 별 성공/실패, 회전별 지시등 타이밍, 조향 부호반전 상위 구간,
winner 분포, wall_dt 통계.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import pickle
import statistics as st
import sys
from collections import Counter

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'src' / 'hlfma'))

SIG = {0: 'OFF', 1: 'LEFT', 2: 'RIGHT'}


def load(path: str):
    ticks, events = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if '"raw"' not in line:
                if '"event"' in line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                continue
            try:
                ticks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return ticks, events


def load_map():
    """lane_graph / route 가 있으면 (lg, route) — 없으면 (None, None)."""
    try:
        from hlfma.core.lanegraph import LaneGraph
        lg = LaneGraph(str(_ROOT / 'data' / 'lane_graph.pkl'))
        with open(_ROOT / 'data' / 'route.pkl', 'rb') as f:
            rt = pickle.load(f)
        return lg, rt
    except Exception:                                   # noqa: BLE001
        return None, None


def load_cfg() -> dict:
    """params.yaml (없으면 DEFAULTS) — 완주 임계가 컨트롤러 정지 정책과 같은 값을 보게."""
    from hlfma.nodes.params import DEFAULTS, load_params_yaml
    yaml_path = _ROOT / 'src' / 'hlfma' / 'config' / 'params.yaml'
    try:
        return load_params_yaml(str(yaml_path))
    except Exception:                                   # noqa: BLE001
        return DEFAULTS


def end_margin_m(cfg: dict) -> float:
    """완주 임계 [m]: route_s(뒷축) ≥ total − 이 값이면 완주.

    계획 정지점이 total − stop_gap − (wheelbase + front_overhang) 이므로
    임계 = stop_gap + 앞범퍼거리 + 여유(end_slack). stop_gap 튜닝을 자동으로
    따라간다 — 2026-08-25: stop_gap 1→4 후 고정 임계 5 m 로 완주가 timeout 처리.
    batch_run 과 여기(완주 표기)가 같은 함수를 쓴다.
    """
    sp, vh = cfg['speed'], cfg['vehicle']
    return (float(sp['stop_gap_m']) + float(vh['wheelbase'])
            + float(vh.get('front_overhang_m', 0.855))
            + float(cfg.get('batch', {}).get('end_slack_m', 1.0)))


def segments(vals):
    """연속 같은 값 구간 → [(start_idx, end_idx, value)]."""
    out, s0 = [], 0
    for i in range(1, len(vals) + 1):
        if i == len(vals) or vals[i] != vals[s0]:
            out.append((s0, i - 1, vals[s0]))
            s0 = i
    return out


def fmt_row(cols, widths):
    return '  '.join(str(c).ljust(w) for c, w in zip(cols, widths))


def table(title, header, rows):
    print(f'\n── {title} ' + '─' * max(0, 70 - len(title)))
    if not rows:
        print('  (없음)')
        return
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(header)]
    print('  ' + fmt_row(header, widths))
    for r in rows:
        print('  ' + fmt_row(r, widths))


# ══════════════════════════════════════════════════════════════════════════
def summarize(path: str, lg=None, route=None) -> dict:
    ticks, events = load(path)
    out: dict = {'file': pathlib.Path(path).name, 'ticks': len(ticks)}
    if not ticks:
        return out
    t0, t1 = ticks[0]['t'], ticks[-1]['t']
    s = [t['ego']['route_s'] for t in ticks]
    v = [t['ego']['speed'] * 3.6 for t in ticks]
    lim = [(t['world']['speed_limit'] or 0) * 3.6 for t in ticks]
    total = float(route['total_length']) if route else None

    # ── 완주 / 거리 / 시간 ───────────────────────────────────────────────
    peak_i = max(range(len(s)), key=lambda i: s[i])
    dist = s[peak_i] - s[0]
    dur = ticks[peak_i]['t'] - t0
    end_stopped = v[-1] < 2.0
    margin = end_margin_m(load_cfg())
    done = (total is not None and s[peak_i] >= total - margin)
    out['finish'] = {'done': done, 'peak_route_s': round(s[peak_i], 1), 'route_total': total,
                     'dist_m': round(dist, 1), 'time_s': round(dur, 1),
                     'avg_kph': round(dist / dur * 3.6, 1) if dur > 0 else None,
                     'end_speed_kph': round(v[-1], 1), 'end_stopped': end_stopped,
                     'wall_s': round(t1 - t0, 1)}

    # ── 리스폰 / 스톨 / 드롭 ─────────────────────────────────────────────
    resets = [(i, t) for i, t in enumerate(ticks) if t['world']['flags'].get('reset')]
    stalls = [(i, t) for i, t in enumerate(ticks) if t['world']['flags'].get('stall')]
    drops = 0
    prev_f = None
    for t in ticks:
        f = (t.get('timing') or {}).get('frames_total')
        if f is not None and prev_f is not None and f - prev_f > 1:
            drops += f - prev_f - 1
        prev_f = f if f is not None else prev_f
    out['resets'] = [{'t': round(t['t'] - t0, 1), 'route_s': round(t['ego']['route_s'], 1),
                      'why': {k: t['world']['flags'][k] for k in t['world']['flags']
                              if k.startswith('reset') or k in ('jump_m', 'toff_jump')}}
                     for _i, t in resets]
    out['stalls'] = [{'t': round(t['t'] - t0, 1), 'dt': t['world']['flags'].get('stall_dt_s')}
                     for _i, t in stalls]
    out['frames_dropped'] = drops
    out['errors'] = [e for e in events if e.get('event') == 'error']

    # ── 제한속도 초과 ────────────────────────────────────────────────────
    over = [(i, v[i] - lim[i]) for i in range(len(v)) if lim[i] > 0 and v[i] > lim[i] + 1e-6]
    over_rows = []
    for a, b, _ in segments([lim[i] > 0 and v[i] > lim[i] + 1e-6 for i in range(len(v))]):
        if not (lim[a] > 0 and v[a] > lim[a]):
            continue
        worst = max(range(a, b + 1), key=lambda i: v[i] - lim[i])
        over_rows.append({'t': round(ticks[a]['t'] - t0, 1), 's0': round(s[a], 1), 's1': round(s[b], 1),
                          'ticks': b - a + 1, 'lane': ticks[worst]['ego']['lane'],
                          'limit': round(lim[worst]), 'v_max': round(v[worst], 1),
                          'v_target': round((ticks[worst]['decision']['v_target'] or 0) * 3.6, 1),
                          'winner': ticks[worst]['decision']['reasons'].get('winner')})
    out['overspeed'] = {'ticks': len(over), 'max_over_kph': round(max((o[1] for o in over), default=0.0), 1),
                        'v_max_kph': round(max(v), 1), 'segments': over_rows}

    # ── 실속도 추정 (위치 변위 / 벽시계) ─────────────────────────────────
    # 로그의 ego.speed 는 perception 의 LPF 추정치다. 9910 송신 간격이 40/80 ms 로
    # 불규칙한데 dt 하한을 50 ms 로 잡으면 40 ms 틱에서 20 % 낮게 나온다
    # (2026-08-23 실측: 로그 45 km/h 일 때 실속도 53 km/h). 0.5 s 창의
    # Σ변위/Σwall_dt 로 독립 추정해 편향을 드러낸다.
    vt_true = []
    win_d, win_dt = [], []
    for i in range(len(ticks)):
        if i == 0:
            vt_true.append(v[0])
            continue
        a, b = ticks[i - 1]['raw']['ego'], ticks[i]['raw']['ego']
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        dt = ticks[i]['t'] - ticks[i - 1]['t']
        if dt <= 0 or dt > 1.0 or d > 20.0:          # 리셋/스톨은 창을 비운다
            win_d, win_dt = [], []
            vt_true.append(v[i])
            continue
        win_d.append(d)
        win_dt.append(dt)
        while sum(win_dt) > 0.5 and len(win_dt) > 1:
            win_d.pop(0)
            win_dt.pop(0)
        vt_true.append(sum(win_d) / sum(win_dt) * 3.6)
    steady = [i for i in range(len(v)) if v[i] > 20.0]
    bias = (st.mean(vt_true[i] / v[i] for i in steady) - 1.0) * 100.0 if steady else 0.0
    over_true = [i for i in range(len(v)) if lim[i] > 0 and vt_true[i] > lim[i]]
    out['speed_estimate'] = {'v_true_max_kph': round(max(vt_true), 1),
                             'bias_pct': round(bias, 1),
                             'over_limit_ticks_true': len(over_true),
                             'max_over_true_kph': round(max((vt_true[i] - lim[i] for i in over_true), default=0.0), 1)}

    # ── |t_off| ──────────────────────────────────────────────────────────
    drive = [i for i in range(peak_i + 1) if ticks[i]['world']['valid'] and ticks[i]['ego']['lane']
             and ticks[i]['decision']['state'] != 'LANE_CHANGE']
    toff = [abs(ticks[i]['ego']['t_off']) for i in drive]
    half = []
    for i in drive:
        w = None
        if lg is not None:
            rec = lg.lanes.get(tuple(ticks[i]['ego']['lane']))
            if rec is not None and rec.get('width') is not None:
                ws = rec['width']
                try:
                    w = float(st.median(float(x) for x in ws))
                except TypeError:
                    w = float(ws)
        half.append((w if w else 3.5) / 2.0)
    exceed = [i for i, h in zip(drive, half) if abs(ticks[i]['ego']['t_off']) > h - 0.3]
    ex_rows = []
    for a, b, val in segments([i in set(exceed) for i in drive]):
        if not val:
            continue
        ia, ib = drive[a], drive[b]
        worst = max(drive[a:b + 1], key=lambda i: abs(ticks[i]['ego']['t_off']))
        ex_rows.append({'t': round(ticks[ia]['t'] - t0, 1), 's0': round(s[ia], 1), 's1': round(s[ib], 1),
                        'ticks': b - a + 1, 'lane': ticks[worst]['ego']['lane'],
                        't_off_max': round(ticks[worst]['ego']['t_off'], 2),
                        'state': ticks[worst]['decision']['state']})
    out['t_off'] = {'mean': round(st.mean(toff), 3) if toff else None,
                    'p95': round(sorted(toff)[int(0.95 * (len(toff) - 1))], 3) if toff else None,
                    'max': round(max(toff), 3) if toff else None,
                    'exceed_ticks': len(exceed), 'segments': ex_rows}

    # ── 차선변경 ─────────────────────────────────────────────────────────
    lc_rows = []
    lc_events = [e for e in (route or {}).get('events', []) if e['kind'].startswith('lane_change')]
    states = [t['decision']['state'] for t in ticks]
    lc_segs = [(a, b) for a, b, val in segments(states) if val == 'LANE_CHANGE']
    for e in lc_events:
        segs = [(a, b) for a, b in lc_segs if e['window_s0'] - 1 <= s[a] <= e['window_s1'] + 1]
        if not segs:
            lc_rows.append({'kind': e['kind'], 'window': f"{e['window_s0']:.0f}-{e['window_s1']:.0f}",
                            'result': 'MISSED', 'start_s': None, 'end_s': None, 'ticks': 0,
                            'aborts': 0, 'signal_lead_s': None})
            continue
        a, b = segs[0][0], segs[-1][1]
        after = ticks[min(b + 1, len(ticks) - 1)]['ego']['lane']
        ok = tuple(after) == tuple(e['to_lane']) if after else False
        aborts = sum(1 for i in range(a, b + 1)
                     if any(k.startswith('lc_abort') or k == 'solid_line_lane_change'
                            for k in (ticks[i]['decision']['reasons'].get('shield') or {})))
        sig_want = 1 if e['kind'].endswith('left') else 2
        on_i = a
        while on_i > 0 and ticks[on_i - 1]['decision']['turn_signal'] == sig_want:
            on_i -= 1
        lc_rows.append({'kind': e['kind'], 'window': f"{e['window_s0']:.0f}-{e['window_s1']:.0f}",
                        'result': 'OK' if ok and not aborts else ('ABORT' if aborts else 'WRONG_LANE'),
                        'start_s': round(s[a], 1), 'end_s': round(s[b], 1), 'ticks': b - a + 1,
                        'aborts': aborts,
                        'signal_lead_s': round(ticks[a]['t'] - ticks[on_i]['t'], 1)})
    if not lc_events:
        for a, b in lc_segs:
            lc_rows.append({'kind': 'LANE_CHANGE', 'window': '?', 'result': '?', 'start_s': round(s[a], 1),
                            'end_s': round(s[b], 1), 'ticks': b - a + 1, 'aborts': 0, 'signal_lead_s': None})
    out['lane_changes'] = lc_rows

    # ── 회전 지시등 ──────────────────────────────────────────────────────
    turn_rows = []
    turn_events = [e for e in (route or {}).get('events', []) if e['kind'].startswith('turn_')]
    sig = [t['decision']['turn_signal'] for t in ticks]
    for e in turn_events:
        want = 1 if e['kind'].endswith('left') else 2
        # 이벤트 지점에 도달한 틱
        at = next((i for i in range(len(s)) if s[i] >= e['s'] and i <= peak_i), None)
        if at is None:
            turn_rows.append({'kind': e['kind'], 's': round(e['s'], 1), 'reached': False})
            continue
        on_i = at
        while on_i > 0 and sig[on_i - 1] == want:
            on_i -= 1
        off_i = at
        while off_i + 1 < len(sig) and sig[off_i + 1] == want:
            off_i += 1
        lead_s = ticks[at]['t'] - ticks[on_i]['t'] if sig[at] == want else 0.0
        srcs = Counter(ticks[i]['decision']['reasons'].get('sig_src') for i in range(on_i, off_i + 1))
        turn_rows.append({'kind': e['kind'], 's': round(e['s'], 1), 'reached': True,
                          'lit_at_event': sig[at] == want, 'signal_at_event': SIG.get(sig[at], sig[at]),
                          'lead_s': round(lead_s, 1), 'lead_m': round(s[at] - s[on_i], 1),
                          'on_s': round(s[on_i], 1), 'off_s': round(s[off_i], 1),
                          'src': dict(srcs), 'turn_dist_at_on': ticks[on_i]['decision']['reasons'].get('turn_dist')})
    out['turns'] = turn_rows
    # 깜빡임: 3틱 이하로 켜졌다 꺼진 토막
    sig_segs = segments(sig)
    out['signal_flicker'] = sum(1 for a, b, val in sig_segs[1:-1] if b - a + 1 <= 3)
    out['signal_transitions'] = len(sig_segs) - 1

    # ── 조향 부호 반전 (4 s 창) ──────────────────────────────────────────
    steer = [t['cmd']['steering'] for t in ticks]
    flips = [i for i in range(1, len(steer))
             if steer[i] * steer[i - 1] < 0 and max(abs(steer[i]), abs(steer[i - 1])) > 0.02]
    win = 80                                       # 틱 (≈4 s)
    flip_rows = []
    if flips:
        counts = []
        for c in range(0, len(ticks), win // 2):
            n = sum(1 for i in flips if c <= i < c + win)
            if n >= 2:
                counts.append((n, c))
        counts.sort(reverse=True)
        used = set()
        for n, c in counts:
            if any(abs(c - u) < win for u in used):
                continue
            used.add(c)
            seg = ticks[c:c + win]
            flip_rows.append({'t': round(ticks[c]['t'] - t0, 1), 's0': round(s[c], 1),
                              's1': round(s[min(c + win, len(s) - 1)], 1), 'flips': n,
                              'steer_max': round(max(abs(x['cmd']['steering']) for x in seg), 3),
                              'v_kph': round(st.mean(x['ego']['speed'] * 3.6 for x in seg), 1),
                              'states': '/'.join(sorted({x['decision']['state'] for x in seg}))})
            if len(flip_rows) >= 5:
                break
    out['steer_flips'] = {'total': len(flips), 'top': flip_rows}

    # ── winner 분포 ──────────────────────────────────────────────────────
    wc = Counter(t['decision']['reasons'].get('winner') for t in ticks[:peak_i + 1])
    out['winner'] = {k: round(100.0 * n / (peak_i + 1), 1) for k, n in wc.most_common()}

    # ── wall_dt ──────────────────────────────────────────────────────────
    wd = [(t.get('timing') or {}).get('wall_dt') for t in ticks]
    wd = [x for x in wd if x is not None]
    sd = [(t.get('timing') or {}).get('sim_dt') for t in ticks]
    sd = [x for x in sd if x is not None]
    if wd:
        srt = sorted(wd)
        out['wall_dt'] = {'mean_ms': round(1000 * st.mean(wd), 1), 'median_ms': round(1000 * srt[len(srt) // 2], 1),
                          'p99_ms': round(1000 * srt[int(0.99 * (len(srt) - 1))], 1),
                          'max_ms': round(1000 * max(wd), 1), 'over_100ms': sum(1 for x in wd if x > 0.1),
                          'rtf': round(sum(sd) / sum(wd[-len(sd):]), 3) if sd and len(wd) >= len(sd) else None}
    return out


# ══════════════════════════════════════════════════════════════════════════
def flag(cond, txt_bad, txt_ok='OK'):
    return f'⚠ {txt_bad}' if cond else txt_ok


def print_report(r: dict) -> None:
    print(f"\n{'═' * 72}\n {r['file']}   틱 {r['ticks']}\n{'═' * 72}")
    if 'finish' not in r:
        print('틱 레코드 없음')
        return
    f = r['finish']
    tot = f"/{f['route_total']:.0f}" if f['route_total'] else ''
    rows = [
        ['완주', flag(not f['done'], f"미완주 (route_s 최대 {f['peak_route_s']}{tot} m)",
                     f"완주 (route_s {f['peak_route_s']}{tot} m)")],
        ['주행거리 / 시간 / 평균', f"{f['dist_m']} m / {f['time_s']} s / {f['avg_kph']} km/h  (벽시계 {f['wall_s']} s)"],
        ['종료 시 속도', flag(not f['end_stopped'], f"{f['end_speed_kph']} km/h — 정지 안 함", f"{f['end_speed_kph']} km/h 정지")],
        ['리스폰 / 스톨 / 드롭', flag(bool(r['resets'] or r['stalls'] or r['frames_dropped']),
                              f"리스폰 {len(r['resets'])}  스톨 {len(r['stalls'])}  드롭 {r['frames_dropped']}",
                              '0 / 0 / 0')],
        ['틱 예외', flag(bool(r['errors']), f"{len(r['errors'])}건", '0')],
    ]
    o = r['overspeed']
    rows.append(['제한속도 초과', flag(o['ticks'] > 0, f"{o['ticks']}틱, 최대 +{o['max_over_kph']} km/h (v_max {o['v_max_kph']})",
                                 f"0틱 (v_max {o['v_max_kph']} km/h)")])
    se = r['speed_estimate']
    rows.append(['실속도(변위/벽시계) 추정', flag(abs(se['bias_pct']) > 5 or se['over_limit_ticks_true'] > 0,
                                      f"로그 대비 {se['bias_pct']:+.1f}%  v_true_max {se['v_true_max_kph']} km/h  "
                                      f"실속도 기준 초과 {se['over_limit_ticks_true']}틱 (최대 +{se['max_over_true_kph']})",
                                      f"로그 대비 {se['bias_pct']:+.1f}%  v_true_max {se['v_true_max_kph']} km/h")])
    t = r['t_off']
    rows.append(['|t_off| mean/p95/max', f"{t['mean']} / {t['p95']} / {t['max']} m" +
                 ('' if not t['exceed_ticks'] else f"  ⚠ 반폭-0.3 초과 {t['exceed_ticks']}틱")])
    lcs = r['lane_changes']
    n_ok = sum(1 for x in lcs if x['result'] == 'OK')
    rows.append(['차선변경', flag(n_ok != len(lcs), f"{n_ok}/{len(lcs)} 성공", f"{n_ok}/{len(lcs)} 성공")])
    tr = [x for x in r['turns'] if x.get('reached')]
    n_lit = sum(1 for x in tr if x.get('lit_at_event'))
    rows.append(['회전 지시등', flag(n_lit != len(tr), f"{n_lit}/{len(tr)} 회전에서 점등", f"{n_lit}/{len(tr)} 점등")])
    rows.append(['지시등 깜빡임 / 전환', flag(r['signal_flicker'] > 0, f"{r['signal_flicker']}회 / {r['signal_transitions']}회",
                                     f"0회 / {r['signal_transitions']}회")])
    rows.append(['조향 부호 반전', f"{r['steer_flips']['total']}회 (상위 구간 아래)"])
    if 'wall_dt' in r:
        w = r['wall_dt']
        rows.append(['wall_dt mean/med/p99/max', f"{w['mean_ms']} / {w['median_ms']} / {w['p99_ms']} / {w['max_ms']} ms"
                     + (f"  >100ms {w['over_100ms']}회" if w['over_100ms'] else '')
                     + (f"  RTF {w['rtf']}" if w['rtf'] is not None else '')])
    rows.append(['winner 분포 (%)', '  '.join(f'{k}:{p}' for k, p in r['winner'].items())])
    table('요약', ['항목', '값'], rows)

    if r['resets']:
        table('리스폰', ['t', 'route_s', '사유'], [[x['t'], x['route_s'], x['why']] for x in r['resets']])
    if o['segments']:
        table('제한속도 초과 구간', ['t', 's0', 's1', '틱', '차로', 'limit', 'v_max', 'v_target', 'winner'],
              [[x['t'], x['s0'], x['s1'], x['ticks'], x['lane'], x['limit'], x['v_max'], x['v_target'], x['winner']]
               for x in o['segments']])
    if t['segments']:
        table('|t_off| 반폭 초과 구간 (LC 제외)', ['t', 's0', 's1', '틱', '차로', 't_off', 'state'],
              [[x['t'], x['s0'], x['s1'], x['ticks'], x['lane'], x['t_off_max'], x['state']] for x in t['segments']])
    table('차선변경', ['kind', 'window', '결과', 'start_s', 'end_s', '틱', 'abort', '점등 lead[s]'],
          [[x['kind'], x['window'], x['result'], x['start_s'], x['end_s'], x['ticks'], x['aborts'], x['signal_lead_s']]
           for x in lcs])
    table('회전 지시등', ['kind', 's', '이벤트 시 점등', 'lead[s]', 'lead[m]', 'on_s', 'off_s', 'src', 'turn_dist@on'],
          [[x['kind'], x['s'], (SIG_OK(x)), x.get('lead_s'), x.get('lead_m'), x.get('on_s'), x.get('off_s'),
            x.get('src'), x.get('turn_dist_at_on')] for x in r['turns']])
    table('조향 부호 반전 상위 구간 (4 s 창)', ['t', 's0', 's1', '반전', '|steer|max', 'v', 'state'],
          [[x['t'], x['s0'], x['s1'], x['flips'], x['steer_max'], x['v_kph'], x['states']] for x in r['steer_flips']['top']])


def SIG_OK(x):
    if not x.get('reached'):
        return '미도달'
    return '예' if x['lit_at_event'] else f"아니오({x['signal_at_event']})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='주행 로그 요약')
    ap.add_argument('log')
    ap.add_argument('--json', action='store_true', help='JSON 으로 출력')
    ap.add_argument('--no-map', action='store_true', help='lane_graph/route 를 읽지 않는다')
    a = ap.parse_args(argv)
    lg, rt = (None, None) if a.no_map else load_map()
    r = summarize(a.log, lg, rt)
    if a.json:
        print(json.dumps(r, ensure_ascii=False, indent=1, default=str))
    else:
        print_report(r)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
