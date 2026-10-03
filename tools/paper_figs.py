#!/usr/bin/env python3
"""논문 Ⅳ장 표·그림 — paper_report 의 집계를 다시 돌려 논문 판형으로 낸다.

    python3 tools/paper_figs.py --arm WON24=logs/paper/WON24
    python3 tools/paper_figs.py --arm WON24=logs/paper/WON24 --arm WON=logs/paper/WON   # 두 구현 비교판

수치는 전부 로그 · report.json 에서 읽는다 (paper_report.main 을 <out>/report/ 로 돌리고 그
CSV 를 읽는다). 이 파일에는 결과 수치를 적지 않는다. 집계 기준은 paper_report 의 기본값 —
지나감 횡거리 8 m · 종료선 이후 사건 제외 (2026-10-03) — 이고 meta.json 에 남는다.

산출물 (--out, 기본 logs/paper/paper_figs/):
  report/                    paper_report 전체 산출물 (원자료 · meta.json)
  table2_paper.csv / .md     표 2 — 유형 × 구현: 런 · 성공률(95% Wilson) · 무한 정지 · 충돌 · 오탐
  fig_A_success_by_type      유형별 회피 성공률 + 95% 구간 (구현별 나란히 막대)
  fig_B_near_stop_trace      근접 정지 대표 시나리오 — 시간–속도 · 장애물까지 거리 (구현별 겹침)
  fig_C_failure_causes       실패 런 원인 분류 (제어기 판단 / 오탐 / 시나리오 / 시뮬레이터)
                             그림마다 .png (600 dpi) · .pdf · 원천 .csv
  IV2_draft.md               Ⅳ.2 문장 초안 — 수치마다 출처 파일·열

판형: 2단 편집의 한 단 폭 8.5 cm, 글자 8 pt, 회색조 + 무늬 (흑백 인쇄에서 구분).
한글 폰트(나눔고딕 · Noto Sans CJK)가 없으면 표까지만 쓰고 설치 명령을 알려 주고 멈춘다.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'tools'))
sys.path.insert(0, str(ROOT))

import paper_report as pr  # noqa: E402

CM = 1 / 2.54
COL_W = 8.5 * CM                      # 한 단 폭 [in]
FONT_PT = 8
FONT_CANDIDATES = ('NanumGothic', 'Noto Sans CJK KR', 'Noto Sans KR', 'Noto Sans CJK JP')
FONT_INSTALL = ('sudo apt install fonts-nanum fonts-noto-cjk && rm -rf ~/.cache/matplotlib')
# 구현 순서대로: (면 색, 무늬) · (선 색, 선 모양, 굵기) — 회색조만으로도, 무늬만으로도 구분된다.
BAR_STYLE = [('0.6', ''), ('1.0', '////'), ('0.85', '....'), ('0.35', 'xxxx')]   # 검은 오차막대가 보이게 밝게
LINE_STYLE = [('0.0', '-', 1.2), ('0.45', '--', 1.2), ('0.2', ':', 1.4), ('0.6', '-.', 1.0)]
CAUSES = ('controller', 'false_positive', 'scenario', 'simulator')
FP_CLASSES = ('brake', 'queue')       # 주 지표가 오탐 런인 유형 분류


def read_csv(p: pathlib.Path) -> list[dict]:
    if not p.exists():
        return []
    with open(p, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def num(v, cast=float):
    return None if v in (None, '') else cast(v)


def type_label(t: str, labels: dict) -> str:
    """유형 표기 — arms.yaml report.type_labels_ko 가 있으면 그것, 없으면 '논문N_' 을 뗀 이름."""
    if t in labels:
        return labels[t]
    head, _, rest = t.partition('_')
    return (rest if head.startswith('논문') and rest else t).replace('_', ' ')


# ════════════════════════════════════════════════════════════════════════
# 표 2
# ════════════════════════════════════════════════════════════════════════
def table2(rep: pathlib.Path, arms: list, lab_ko, tlab) -> list[dict]:
    """by_type.csv (유형 × 구현) + table2.csv (회피 전체) → 논문 표 2 행 (긴 형식)."""
    rows = []
    bt = read_csv(rep / 'by_type.csv')
    types = list(dict.fromkeys(r['type'] for r in bt))
    for t in types:
        for arm in arms:
            r = next((x for x in bt if x['type'] == t and x['arm'] == arm), None)
            if r is None:
                continue
            fp_applies = r['class'] in ('brake', 'queue')
            rows.append({'type': t, 'type_label': tlab(t), 'class': r['class'],
                         'class_ko': pr.CLASS_KO.get(r['class'], r['class']), 'arm': arm, 'label': lab_ko(arm),
                         'n_expected': num(r['n_expected'], int), 'n_valid': num(r['n_valid'], int),
                         'n_invalid': num(r['n_invalid'], int), 'n_judged': num(r['n_judged'], int),
                         'n_excluded': num(r['n_excluded'], int), 'success': num(r['success'], int),
                         'success_pct': num(r['success_pct']), 'lo_pct': num(r['success_lo_pct']),
                         'hi_pct': num(r['success_hi_pct']), 'inf_stop': num(r['inf_stop'], int),
                         'timeout': num(r['timeout'], int),
                         'collision_runs': num(r['runs_with_collision'], int),
                         'collisions': num(r['collisions'], int),
                         'false_positive_runs': num(r['false_positive_runs'], int) if fp_applies else None,
                         'source': 'report/by_type.csv'})
    # 회피 유형 전체 (table2.csv — 같은 판정의 합계)
    for r in read_csv(rep / 'table2.csv'):
        if r['arm'] not in arms:
            continue
        rows.append({'type': '회피 전체', 'type_label': '회피 전체', 'class': 'avoid', 'class_ko': '회피',
                     'arm': r['arm'], 'label': lab_ko(r['arm']),
                     'n_expected': num(r['n_expected'], int), 'n_valid': num(r['n_valid'], int),
                     'n_invalid': num(r['n_invalid'], int), 'n_judged': None, 'n_excluded': None,
                     'success': num(r['success'], int), 'success_pct': num(r['success_pct']),
                     'lo_pct': num(r['success_lo_pct']), 'hi_pct': num(r['success_hi_pct']),
                     'inf_stop': num(r['inf_stop'], int), 'timeout': num(r['timeout'], int),
                     'collision_runs': num(r['runs_with_collision'], int), 'collisions': num(r['collisions'], int),
                     'false_positive_runs': None, 'source': 'report/table2.csv'})
    for r in rows:
        if r['n_judged'] is None:                       # 회피 전체 — 판정 대상 = 성공률 분모
            r['n_judged'] = sum(x['n_judged'] or 0 for x in rows
                                if x['class'] == 'avoid' and x['arm'] == r['arm'] and x['type'] != '회피 전체')
        # 주 지표 (2026-10-03 결정): 오탐 유형(급정거 · 신호줄)은 성공률보다 오탐 런 수가 주 지표다 —
        # 그 유형이 재려는 것이 "다시 움직일 대상에 시프트를 만드는가" 이기 때문이다.
        fp = r['class'] in FP_CLASSES
        k, n = (r['false_positive_runs'], r['n_judged']) if fp else (r['success'], r['n_judged'])
        lo, hi = pr.wilson(k or 0, n or 0)
        r.update({'primary_metric': 'false_positive_runs' if fp else 'success', 'primary_k': k, 'primary_n': n,
                  'primary_pct': round(100.0 * k / n, 1) if n else None,
                  'primary_lo_pct': None if math.isnan(lo) else round(100.0 * lo, 1),
                  'primary_hi_pct': None if math.isnan(hi) else round(100.0 * hi, 1)})
    return rows


def table2_md(rows: list, arms: list, lab_ko) -> str:
    """붙여넣기용 — 행 = 유형, 열 묶음 = 구현. 주 지표는 회피·일반 = 성공률, 오탐 유형 = 오탐 런."""
    def cell(r):
        if r is None:
            return ['—'] * 5
        n = f'{r["n_judged"]}/{r["n_expected"]}' if r['n_expected'] is not None else f'{r["n_judged"]}'
        ci = (f' ({r["primary_lo_pct"]:.1f}–{r["primary_hi_pct"]:.1f})'
              if r['primary_lo_pct'] is not None else '')
        if r['primary_metric'] == 'false_positive_runs':
            main = f'오탐 {r["primary_k"]}/{r["primary_n"]} 런 = {r["primary_pct"]:.1f}%{ci}'
            ref = f'성공률 {r["success_pct"]:.1f}%' if r['success_pct'] is not None else '—'
        else:
            main = f'성공 {r["primary_pct"]:.1f}%{ci}' if r['primary_pct'] is not None else '—'
            ref = '—'
        return [n, main, str(r['inf_stop']), str(r['collision_runs']), ref]
    heads = ['유형', '분류']
    for arm in arms:
        a = lab_ko(arm)
        heads += [f'{a} 런', f'{a} 주 지표 (95% CI)', f'{a} 무한 정지', f'{a} 충돌 런', f'{a} 참고']
    line = lambda cells: '| ' + ' | '.join(cells) + ' |'          # noqa: E731
    out = [line(heads), line(['---'] * len(heads))]
    for t in dict.fromkeys(r['type'] for r in rows):
        rr = [r for r in rows if r['type'] == t]
        cells = [('**' + rr[0]['type_label'] + '**') if t == '회피 전체' else rr[0]['type_label'], rr[0]['class_ko']]
        for arm in arms:
            cells += cell(next((r for r in rr if r['arm'] == arm), None))
        out.append(line(cells))
    return '\n'.join(out) + '\n'


# ════════════════════════════════════════════════════════════════════════
# 그림
# ════════════════════════════════════════════════════════════════════════
def setup_font():
    """한글 글리프가 있는 폰트 이름 → rcParams. 없으면 None."""
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import font_manager as fm
    from matplotlib.ft2font import FT2Font
    by_name = {}
    for f in fm.fontManager.ttflist:
        by_name.setdefault(f.name, []).append(f)
    for name in FONT_CANDIDATES:
        for f in sorted(by_name.get(name, []), key=lambda f: abs(int(f.weight) - 400)):
            try:
                if FT2Font(f.fname).get_char_index(ord('한')):
                    _rc(name, _is_cff(f.fname))
                    return name
            except Exception:                           # noqa: BLE001 — 다음 후보
                continue
    return None


def _is_cff(path: str) -> bool:
    """OpenType-CFF 글꼴(Noto CJK 의 .otf/.ttc)인가. matplotlib 은 이것을 PDF Type 42 로 넣을 때
    TrueType 이라 적고 CFF 를 심어 '글꼴 형식 불일치' PDF 를 만든다 (poppler 는 경고만 내지만
    Acrobat 은 못 그릴 수 있다). 그럴 때는 Type 3 으로 넣는다 — 나눔고딕(.ttf)이면 Type 42."""
    with open(path, 'rb') as f:
        head = f.read(16)
    if head[:4] == b'ttcf':                             # 모음 — 첫 글꼴의 머리를 본다
        off = int.from_bytes(head[12:16], 'big')
        with open(path, 'rb') as f:
            f.seek(off)
            head = f.read(4)
    return head[:4] == b'OTTO'


def _rc(font: str, cff: bool = False) -> None:
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.family': font, 'font.size': FONT_PT, 'axes.titlesize': FONT_PT, 'axes.labelsize': FONT_PT,
        'xtick.labelsize': FONT_PT, 'ytick.labelsize': FONT_PT, 'legend.fontsize': FONT_PT,
        'axes.unicode_minus': False,                    # CJK 폰트의 U+2212 누락 대비
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'lines.linewidth': 1.0, 'hatch.linewidth': 0.5, 'legend.frameon': False,
        # TrueType 이면 42 (글자를 글자로 — 편집기에서 고칠 수 있게), CFF 면 3 (위 _is_cff)
        'pdf.fonttype': 3 if cff else 42, 'ps.fonttype': 3 if cff else 42,
        # 폭을 정확히 한 단(8.5 cm)으로 — bbox 'tight' 는 여백을 잘라 폭이 줄어든다
        'savefig.bbox': None, 'figure.constrained_layout.use': True,
        'figure.constrained_layout.h_pad': 0.02, 'figure.constrained_layout.w_pad': 0.02})


def save(fig, out: pathlib.Path, stem: str) -> list[pathlib.Path]:
    import matplotlib.pyplot as plt
    paths = [out / f'{stem}.png', out / f'{stem}.pdf']
    fig.savefig(paths[0], dpi=600)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def fig_a(rows: list, arms: list, lab_ko, out: pathlib.Path) -> tuple[list, list]:
    """유형별 회피 성공률 + 95% Wilson 구간 — 구현별 나란히 막대."""
    import matplotlib.pyplot as plt
    av = [r for r in rows if r['class'] == 'avoid' and r['n_judged']]
    types = list(dict.fromkeys(r['type'] for r in av))
    src = [{'type': r['type'], 'type_label': r['type_label'], 'arm': r['arm'], 'label': r['label'],
            'success': r['success'], 'n_judged': r['n_judged'], 'success_pct': r['success_pct'],
            'lo_pct': r['lo_pct'], 'hi_pct': r['hi_pct'], 'source': r['source']} for r in av]
    pr.write_csv(out / 'fig_A_success_by_type.csv', src)
    fig, ax = plt.subplots(figsize=(COL_W, 5.6 * CM))
    n = len(arms)
    w = 0.8 / n
    for j, arm in enumerate(arms):
        xs, ys, lo, hi = [], [], [], []
        for i, t in enumerate(types):
            r = next((x for x in av if x['type'] == t and x['arm'] == arm), None)
            if r is None or r['success_pct'] is None:
                continue
            xs.append(i + (j - (n - 1) / 2) * w)
            ys.append(r['success_pct'])
            lo.append(r['success_pct'] - r['lo_pct'])
            hi.append(r['hi_pct'] - r['success_pct'])
        face, hatch = BAR_STYLE[j % len(BAR_STYLE)]
        ax.bar(xs, ys, w, color=face, hatch=hatch, edgecolor='0', linewidth=0.5, label=lab_ko(arm), zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt='none', ecolor='0', elinewidth=0.6, capsize=1.6, capthick=0.6,
                    zorder=3)
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([next(r['type_label'] for r in av if r['type'] == t) for t in types],
                       rotation=30, ha='right', rotation_mode='anchor')
    ax.set_ylim(0, 105)
    ax.set_ylabel('성공률 [%]')
    ax.yaxis.grid(True, linewidth=0.3, color='0.8', zorder=0)
    ax.set_axisbelow(True)
    if '회피 전체' in types:                           # 합계 막대를 유형 막대와 가른다
        ax.axvline(types.index('회피 전체') - 0.5, color='0.5', linewidth=0.5, linestyle='--')
    # 범례는 축 위 바깥 — 100 % 막대와 겹치지 않게
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=max(1, n), handlelength=1.6,
              borderaxespad=0.2, columnspacing=1.0)
    return save(fig, out, 'fig_A_success_by_type'), src


def pick_trace(rep: pathlib.Path, arms: list, ref: str, ttype: str, axis: str, forced: str | None):
    """근접 정지 대표 시나리오 — 모든 구현이 끝까지 간 런(로그 있음) 중 기준 구현이 성공한 것,
    그중 조건 축 값(시작 거리)이 가장 작은 것 (가장 어려운 조건), 같으면 이름순."""
    runs = read_csv(rep / 'runs.csv')
    cand = {}
    for r in runs:
        if r['type'] != ttype:
            continue
        cand.setdefault(r['scenario'], {})[r['arm']] = r
    ok = []
    for name, by in cand.items():
        if forced and name != forced:
            continue
        if not all(a in by and by[a]['valid'] == 'True' and by[a]['log'] for a in arms):
            continue
        if not forced and by.get(ref, {}).get('success') != 'True':
            continue
        cond = json.loads(by[ref]['conditions'] or '{}')
        v = cond.get(axis)
        ok.append((v if isinstance(v, (int, float)) else math.inf, name, by))
    if not ok:
        return None
    ok.sort(key=lambda x: (x[0], x[1]))
    return ok[0]


def trace_series(run_row: dict, scen_dir: pathlib.Path, match_m: float) -> dict:
    """로그 → 시간·속도·앞범퍼~장애물 뒤끝 거리 (자차 진행 방향 성분, 지나가면 음수)."""
    log = pathlib.Path(run_row['log'])
    scen = pr.parse_scenario(scen_dir / run_row['type'] / f'{run_row["scenario"]}.yaml')
    obj = scen['encounters'][0]['objects'][0]
    cfg = pr.load_params_yaml(run_row['config'] if run_row.get('config') and pathlib.Path(run_row['config']).exists()
                              else None)
    veh = cfg['vehicle']
    front = float(veh['wheelbase']) + float(veh['front_overhang_m'])
    ticks = pr.score.load_ticks(str(log))
    m = pr.match_objects(ticks, [obj['xy']], match_m)
    obj_len = m[0][1] if 0 in m else 0.0
    t0 = ticks[0]['t']
    t, v, gap = [], [], []
    for tk in ticks:
        lon, _lat = pr.ego_frame(tk['ego'], obj['xy'][0], obj['xy'][1])
        t.append(round(tk['t'] - t0, 2))
        v.append(round(float(tk['ego']['speed']) * 3.6, 2))
        gap.append(round(lon - front - obj_len / 2.0, 2))
    # 다 지나간 시각 (뒤끝까지 자차 길이만큼 더 간 때)
    passed = next((t[i] for i, g in enumerate(gap) if g <= -(obj_len + float(veh['length']))), t[-1])
    return {'t': t, 'v_kph': v, 'gap_m': gap, 't_passed': passed, 'obj_len_m': obj_len,
            'obstacle': obj['name'], 'start_gap_m': gap[0]}


def fig_b(pick, arms: list, lab_ko, scen_dir: pathlib.Path, match_m: float, axis: str,
          out: pathlib.Path) -> tuple[list, list, dict]:
    import matplotlib.pyplot as plt
    axv, name, by = pick
    ser = {a: trace_series(by[a], scen_dir, match_m) for a in arms}
    t_end = min(max(s['t_passed'] for s in ser.values()) + 3.0, max(s['t'][-1] for s in ser.values()))
    src = []
    for a in arms:
        s = ser[a]
        for i, tt in enumerate(s['t']):
            if tt <= t_end:
                src.append({'scenario': name, 'arm': a, 'label': lab_ko(a), 't_s': tt,
                            'v_kph': s['v_kph'][i], 'gap_m': s['gap_m'][i]})
    pr.write_csv(out / 'fig_B_near_stop_trace.csv', src)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, 6.8 * CM), sharex=True)
    for j, a in enumerate(arms):
        c, ls, lw = LINE_STYLE[j % len(LINE_STYLE)]
        s = ser[a]
        k = [i for i, tt in enumerate(s['t']) if tt <= t_end]
        # 범례에 시작 간격 로그 실측값 (조건값과 기준점이 달라 조금 다르다 — 2026-10-03 결정 4)
        a1.plot([s['t'][i] for i in k], [s['v_kph'][i] for i in k], color=c, linestyle=ls, linewidth=lw,
                label=f'{lab_ko(a)} · 시작 {s["start_gap_m"]:.1f} m')
        a2.plot([s['t'][i] for i in k], [s['gap_m'][i] for i in k], color=c, linestyle=ls, linewidth=lw)
    a1.set_ylabel('속도 [km/h]')
    a1.set_ylim(bottom=0)
    a2.axhline(0, color='0.5', linewidth=0.5, linestyle='--')
    a2.set_ylabel('장애물까지 [m]')
    a2.set_xlabel('시간 [s]')
    a2.set_xlim(0, t_end)
    for ax in (a1, a2):
        ax.yaxis.grid(True, linewidth=0.3, color='0.85')
    a1.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=len(arms), handlelength=2.2,
              borderaxespad=0.2)
    fig.align_ylabels((a1, a2))
    if not math.isinf(axv):              # 조건값 — 그림 안에 한 줄
        a2.text(0.99, 0.95, f'조건 {axis} {axv:g} m', transform=a2.transAxes, ha='right', va='top')
    info = {'scenario': name, 'axis': axis, 'axis_value': None if math.isinf(axv) else axv,
            'start_gap_m': {a: ser[a]['start_gap_m'] for a in arms},
            'obstacle': ser[arms[0]]['obstacle'], 'obstacle_len_m': ser[arms[0]]['obj_len_m'],
            't_end_s': round(t_end, 1),
            'gap_definition': '앞범퍼 ~ 장애물 뒤끝, 자차 진행 방향 성분 (지나가면 음수)'}
    return save(fig, out, 'fig_B_near_stop_trace'), src, info


def fig_c(rep: pathlib.Path, arms: list, lab_ko, out: pathlib.Path) -> tuple[list, list]:
    import matplotlib.pyplot as plt
    causes = read_csv(rep / 'causes.csv')
    src = [{'arm': a, 'label': lab_ko(a), 'cause': c, 'cause_ko': pr.CAUSE_KO[c],
            'runs': sum(1 for r in causes if r['arm'] == a and r['cause'] == c)} for a in arms for c in CAUSES]
    pr.write_csv(out / 'fig_C_failure_causes.csv', src)
    fig, ax = plt.subplots(figsize=(COL_W, 4.6 * CM))
    n = len(arms)
    w = 0.8 / n
    top = max([s['runs'] for s in src] + [1])
    for j, a in enumerate(arms):
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(CAUSES))]
        ys = [next(s['runs'] for s in src if s['arm'] == a and s['cause'] == c) for c in CAUSES]
        face, hatch = BAR_STYLE[j % len(BAR_STYLE)]
        ax.bar(xs, ys, w, color=face, hatch=hatch, edgecolor='0', linewidth=0.5, label=lab_ko(a), zorder=2)
        for x, y in zip(xs, ys):
            ax.text(x, y + top * 0.02, str(y), ha='center', va='bottom')
    ax.set_xticks(range(len(CAUSES)))
    ax.set_xticklabels([pr.CAUSE_KO[c] for c in CAUSES])
    ax.set_ylabel('실패 런 수')
    ax.set_ylim(0, top * 1.25)
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.yaxis.grid(True, linewidth=0.3, color='0.85', zorder=0)
    ax.set_axisbelow(True)
    if n > 1:
        ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=n, handlelength=1.6, borderaxespad=0.2)
    return save(fig, out, 'fig_C_failure_causes'), src


# ════════════════════════════════════════════════════════════════════════
# Ⅳ.2 문장 초안
# ════════════════════════════════════════════════════════════════════════
CLASS_OBJ = {'queue': '신호 대기 줄', 'avoid': '옆 차로 교통', 'brake': '앞차', 'plain': '교통'}


def judge_text(meta: dict) -> str:
    """meta.json judge (런마다 판정값 JSON 목록) → 'stall 60 s · blocked 60 s'. 둘 이상이면 모두."""
    out = []
    for j in meta.get('judge') or []:
        d = json.loads(j)
        out.append(f'stall {d.get("stall_end_s"):g} s · blocked {d.get("blocked_end_s"):g} s')
    return ' / '.join(dict.fromkeys(out)) or '판정값 기록 없음'


def draft(rows: list, causes_src: list, trace: dict | None, arms: list, ref: str, lab_ko, meta: dict,
          won_arm: str, causes_rows: list = ()) -> str:
    """수치는 표·그림 원천에서 꺼내 끼운다. 비교 구현이 없으면 그 칸은 '—'."""
    src_notes = []

    def ref_of(key, file, where, col):
        src_notes.append(f'- [{key}] `{file}` — {where}, 열 `{col}`')
        return f'[{key}]'

    def row(t, a):
        return next((r for r in rows if r['type'] == t and r['arm'] == a), None)

    def pct(r):
        return '—' if r is None or r['success_pct'] is None else \
            f'{r["success"]}/{r["n_judged"]} = {r["success_pct"]:.1f}% (95% CI {r["lo_pct"]:.1f}–{r["hi_pct"]:.1f}%)'

    tot = {a: row('회피 전체', a) for a in arms}
    other = won_arm if won_arm in arms else None
    lines = ['# Ⅳ.2 결과 — 문장 초안 (자동 생성, 수치는 출처 파일에서)', '',
             f'집계 기준: 지나감 횡거리 {meta["thresholds"]["pass_lat_m"]} m · 종료선 이후 사건 '
             f'{"제외" if meta["thresholds"].get("after_finish") == "exclude" else "포함"} · '
             f'무한 정지 = stall + blocked ({judge_text(meta)}). '
             f'커밋 {meta.get("commit")}.', '']
    r = tot.get(ref)
    k1 = ref_of('1', 'table2_paper.csv', f'type=회피 전체, arm={ref}', 'success · n_judged · success_pct · lo_pct · hi_pct')
    k2 = ref_of('2', 'table2_paper.csv', f'type=회피 전체, arm={ref}', 'inf_stop · collision_runs · n_valid')
    lines.append(
        f'{lab_ko(ref)}은 회피 {sum(1 for x in rows if x["class"] == "avoid" and x["arm"] == ref and x["type"] != "회피 전체")}'
        f'개 유형 {r["n_valid"] if r else "—"}런에서 모든 장애물을 충돌 없이 지난 런의 비율이 {pct(r)}{k1}였다. '
        f'무한 정지는 {r["inf_stop"] if r else "—"}회, 충돌이 있었던 런은 {r["collision_runs"] if r else "—"}런이었다{k2}.')
    ro = tot.get(other) if other else None
    k3 = ref_of('3', 'table2_paper.csv', f'type=회피 전체, arm={won_arm}', 'success_pct · inf_stop')
    lines.append(f'같은 시나리오에서 {lab_ko(won_arm)}의 성공률은 {pct(ro)}, 무한 정지는 '
                 f'{ro["inf_stop"] if ro else "—"}회였다{k3}.')
    lines.append('')
    # 유형별 — 가장 낮은 유형
    av = [x for x in rows if x['class'] == 'avoid' and x['arm'] == ref and x['type'] != '회피 전체'
          and x['success_pct'] is not None]
    if av:
        worst = min(av, key=lambda x: (x['success_pct'], x['type']))
        k4 = ref_of('4', 'table2_paper.csv · fig_A_success_by_type.csv', f'arm={ref}, 회피 유형 행 전부', 'success_pct')
        full = [x['type_label'] for x in av if x['success'] == x['n_judged']]
        lines.append(f'유형별로는 {", ".join(full) or "—"} 유형에서 전 런 성공했고, 가장 낮은 유형은 '
                     f'{worst["type_label"]} 유형으로 {pct(worst)}였다{k4} (그림 A).')
    # 오탐
    for cls, what in (('brake', '앞차 급정거'), ('queue', '적신호 대기 줄')):
        rr = [x for x in rows if x['class'] == cls and x['arm'] == ref]
        for x in rr:
            k = ref_of(str(len(src_notes) + 1), 'table2_paper.csv', f'type={x["type"]}, arm={ref}',
                       'false_positive_runs · n_judged · primary_lo_pct · primary_hi_pct')
            xo = row(x['type'], won_arm) if other else None
            ci = (f' (95% CI {x["primary_lo_pct"]:.1f}–{x["primary_hi_pct"]:.1f}%)'
                  if x['primary_lo_pct'] is not None else '')
            lines.append(f'{what} 유형 {x["n_judged"]}런 중 다시 움직일 대상에 회피 시프트를 만든 오탐 런은 '
                         f'{x["false_positive_runs"]}런{ci}이었다{k} ({lab_ko(won_arm)}: '
                         f'{"—" if xo is None else xo["false_positive_runs"]}런).')
    lines.append('')
    # 실패 원인
    cs = {c['cause']: c['runs'] for c in causes_src if c['arm'] == ref}
    # 시나리오 원인의 내용 — causes.csv 근거에서 (대기 줄 · 교통 · 콘 …)
    sc = [r for r in causes_rows if r['arm'] == ref and r['cause'] == 'scenario']
    kinds = sorted({('종료선 콘 접촉' if '콘' in r['evidence'] else
                     f'반응하지 않는 스크립트 차량({CLASS_OBJ.get(r["class"], r["class"])})이 거의 선 자차 쪽으로 들어온 경우')
                    for r in sc})
    scen_text = ' · '.join(kinds)
    kc = ref_of(str(len(src_notes) + 1), 'fig_C_failure_causes.csv · report/causes.csv', f'arm={ref}', 'runs (cause별)')
    lines.append(f'{lab_ko(ref)}의 실패 런과 기반 실패 런을 원인별로 나누면 제어기 판단 {cs.get("controller", 0)}런, '
                 f'오탐 {cs.get("false_positive", 0)}런, 시나리오 {cs.get("scenario", 0)}런, 시뮬레이터 '
                 f'{cs.get("simulator", 0)}런이었다{kc} (그림 C).'
                 + (f' 시나리오로 분류한 런은 모두 {scen_text}이며, 분류만 하고 결과에서 빼지 않았다.' if scen_text else ''))
    if trace:
        kt = ref_of(str(len(src_notes) + 1), 'fig_B_near_stop_trace.csv', f'scenario={trace["scenario"]}', 't_s · v_kph · gap_m')
        meas = ', '.join(f'{lab_ko(a)} {g:.1f} m' for a, g in trace['start_gap_m'].items())
        lines.append(f'그림 B 는 근접 정지 유형 중 시작 {trace["axis"]}가 가장 짧은 {trace["scenario"]}'
                     f'(조건 {trace["axis_value"]} m, 로그 실측 시작 간격 {meas})에서 시간에 따른 속도와 '
                     f'장애물까지 거리를 보인다{kt}.')
    lines += ['', '시뮬레이션 결과이며 실주행 성능의 예측이 아니다.', '', '## 출처', ''] + src_notes
    lines += ['', '`table2_paper.csv` 는 `report/by_type.csv`(유형별) · `report/table2.csv`(회피 전체)에서 옮긴 것이고, '
              '그 둘은 `paper_report.py` 가 로그와 `report.json` 에서 계산한다.']
    return '\n'.join(lines) + '\n'


# ════════════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='논문 표 2 · 그림 A–C · Ⅳ.2 초안')
    ap.add_argument('--arm', action='append', required=True, metavar='이름=폴더')
    ap.add_argument('--scenarios', default=str(ROOT / 'scenarios'))
    ap.add_argument('--out', default=str(ROOT / 'logs' / 'paper' / 'paper_figs'))
    ap.add_argument('--arms-yaml', default=str(ROOT / 'config' / 'arms' / 'arms.yaml'))
    ap.add_argument('--ref', default=None, help='기준 구현 (기본: arms.yaml report.reference)')
    ap.add_argument('--trace-scenario', default=None, help='그림 B 시나리오 (기본: 규칙으로 고른다)')
    ap.add_argument('--report-arg', action='append', default=[], metavar='ARG',
                    help='paper_report 에 그대로 넘길 인자 (예: --report-arg=--pass-lat-m=6)')
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = out / 'report'
    rc = pr.main(sum((['--arm', x] for x in a.arm), []) +
                 ['--scenarios', a.scenarios, '--out', str(rep), '--arms-yaml', a.arms_yaml, '--no-figures']
                 + (['--ref', a.ref] if a.ref else []) + a.report_arg)
    if rc:
        return rc
    meta = json.loads((rep / 'meta.json').read_text(encoding='utf-8'))
    arms = list(meta['arms'])
    ref = meta['reference']
    lab = {k: v['label_ko'] for k, v in meta['arms'].items()}
    lab_ko = lambda k: lab.get(k) or k                              # noqa: E731
    p = pathlib.Path(a.arms_yaml)
    ycfg = (pr.yaml.safe_load(p.read_text(encoding='utf-8')) or {}) if p.exists() else {}
    rcfg = (ycfg.get('report') or {})
    fcfg = rcfg.get('paper_figs') or {}
    for k, v in (rcfg.get('labels') or {}).items():
        lab.setdefault(k, v.get('label_ko'))
    tl = rcfg.get('type_labels_ko') or {}
    tlab = lambda t: type_label(t, tl)                              # noqa: E731
    won_arm = fcfg.get('compare_arm', 'WON')

    # ── 표 2 ────────────────────────────────────────────────────────────
    rows = table2(rep, arms, lab_ko, tlab)
    pr.write_csv(out / 'table2_paper.csv', rows)
    md = ('표 2. 시나리오 유형별 결과 (런 = 판정 대상/계획. 주 지표 — 회피·일반 주행 유형은 성공률(유형별 '
          '성공 기준), 오탐 유형(앞차 급정거 · 신호 대기 줄)은 오탐 런 = 다시 움직일 대상에 회피 시프트를 만든 '
          '런 (낮을수록 좋다), 오탐 유형의 성공률은 참고 열. 95% CI = Wilson. 무한 정지 = stall + blocked, '
          '충돌 런 = score 항목 11 접촉이 있었던 런)\n\n' + table2_md(rows, arms, lab_ko)
          + f'\n집계 기준: 지나감 횡거리 {meta["thresholds"]["pass_lat_m"]} m · 종료선 이후 사건 '
            f'{"제외" if meta["thresholds"].get("after_finish") == "exclude" else "포함"}. '
            '기반 실패 런(시뮬레이터)은 분모에서 뺐다 — 그림 C 에 따로 센다.\n'
          + (f'모든 구현에서 뺀 시나리오 {len(meta["excluded_scenarios"])}개 (빌더 차로열 다름 · --exclude, '
             f'report/meta.json excluded_scenarios): '
             f'{", ".join(meta["excluded_scenarios"])}\n' if meta.get('excluded_scenarios') else ''))
    (out / 'table2_paper.md').write_text(md, encoding='utf-8')
    print(md)

    # ── 그림 ────────────────────────────────────────────────────────────
    font = setup_font()
    if font is None:
        print(f'[중단] 한글 폰트가 없다 ({", ".join(FONT_CANDIDATES)}). 설치 후 다시 돌린다:\n  {FONT_INSTALL}',
              file=sys.stderr)
        return 3
    made = {}
    made['A'], _ = fig_a(rows, arms, lab_ko, out)
    ttype = fcfg.get('trace_type')
    axis = fcfg.get('trace_axis', '거리')
    pick = pick_trace(rep, arms, ref, ttype, axis, a.trace_scenario) if ttype else None
    trace = None
    if pick is None:
        print(f'  그림 B: 조건에 맞는 시나리오 없음 (유형 {ttype}, 모든 구현이 끝까지 간 런 · 기준 구현 성공)')
    else:
        made['B'], _, trace = fig_b(pick, arms, lab_ko, pathlib.Path(a.scenarios), 1.5, axis, out)
    made['C'], csrc = fig_c(rep, arms, lab_ko, out)
    info = {'font': font, 'column_width_cm': 8.5, 'font_pt': FONT_PT, 'png_dpi': 600, 'arms': arms,
            'reference': ref, 'trace': trace, 'report_meta': 'report/meta.json',
            'files': {k: [p.name for p in v] for k, v in made.items()}}
    (out / 'paper_figs_meta.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    excl = meta.get('auto_excluded_builders_differ') or []
    text = draft(rows, csrc, trace, arms, ref, lab_ko, meta, won_arm, read_csv(rep / 'causes.csv'))
    if excl:
        text += (f'\n※ 구현이 둘 이상이라 두 브랜치 빌더의 차로열이 다른 시나리오 {len(excl)}개를 모든 구현에서 '
                 f'뺐다 (report/meta.json auto_excluded_builders_differ): {", ".join(excl)}\n')
    (out / 'IV2_draft.md').write_text(text, encoding='utf-8')
    print(f'폰트: {font}   그림 B: {trace["scenario"] if trace else "—"}')
    print(f'산출물: {out}/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
