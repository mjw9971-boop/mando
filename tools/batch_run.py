"""
배치 회귀 실행기 — 시나리오 여러 개를 사람 손 없이 순차 실행하고 한 표로 정리.

    python3 tools/batch_run.py scenarios.json [--host 192.168.10.1] [--dry-run]
    python3 tools/batch_run.py scenarios/batch_보행자집중.json scenarios/batch_급정거집중.json
    python3 tools/batch_run.py 'scenarios/batch_*집중.json'      # glob 도 받는다 (합쳐 실행)

시나리오 목록(JSON, PyYAML 이 있으면 YAML 도):
    [{"name": "2_lead_brake",
      "vtd_xml_path": "/home/vtd/.../2_lead_brake.xml",   # VTD PC 기준 절대경로
      "route_csv": "waypoints/2_lead_brake.csv",          # 제어기 노트북 기준
      "timeout_s": 180}, ...]

한 사이클:
  1. build_route.py 로 logs/batch/<ts>/routes/route_{name}.pkl 생성
     (data/ 에는 안 남긴다 — 손으로 만든 pkl 만 두는 곳이다. 배치 산출 route 는
      로그와 같은 디렉터리에 묶여 사후 분석·replay 후 통째로 지운다)
  2. SCP(48179) 로 VTD 에 시나리오 로드 → Init → 팔로워 카메라(<Camera>,
     IG 표시 전용 — params camera.*) → Start   (실측 확인 2026-08-24)
  3. run_agent.py 를 서브프로세스로 (--route <routes>/route_{name}.pkl)
  4. 로그 꼬리를 감시해 종료 판정 (EndJudge — 완주 임계는 params.yaml 연동):
       완주        : route_s ≥ total − (stop_gap+앞범퍼+end_slack), 이후 v<0.5 또는 유예
       stall       : 정지인데 계획은 진행(v_target≥0.5)이 batch.stall_end_s 지속
       no_progress : route_s 무전진이 batch.no_progress_end_s 지속
       실패        : timeout_s 초과 / launch 프로세스 사망 / NO_DATA_S 로그 무변화
  5. launch SIGINT 종료, SCP Stop, 로그를 logs/batch/<ts>/{name}.jsonl 로 복사

실패해도 다음 시나리오로 진행하고 표에 사유를 남긴다 (조용히 넘기지 않는다).
Ctrl+C 는 launch 와 SCP 를 정리하고 부분 표를 출력한다.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
import unicodedata

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / 'tools'))
sys.path.insert(0, str(_ROOT))

from scp_client import ScpClient                       # noqa: E402

NO_DATA_S = 20.0          # 로그가 이 시간 동안 안 자라면 no-data 실패
POLL_S = 2.0

# 완주 임계는 하드코딩하지 않는다 — vtd_adapter.config.end_margin_m(params.yaml) 과
# 공용 (summarize_run·score 도 같은 함수를 본다).
# (2026-08-25: stop_gap 1→4 튜닝 후 고정 임계 5 m 에 계획 정지점이 7.9 m 못 미쳐
#  정상 완주가 전부 timeout 처리된 사고)
from vtd_adapter.config import end_margin_m, load_params_yaml as load_cfg  # noqa: E402

# {grace}/{stall}/{blocked}/{noprog} 는 params batch.* 에서 채운다 (상수 이중화 금지)
DONE_RULE = ('완주 판정: 로그 ego.route_s ≥ route_{name}.pkl 총길이 − {margin:.1f} m '
             '(stop_gap+앞범퍼+여유, params.yaml 연동), 도달 후 v < 0.5 m/s 또는 '
             '{grace:.0f} s 유예.  조기 종료: stall(계획은 진행인데 못 감) {stall:.0f} s / '
             'blocked(앞이 막힘 — 신호·보행자 대기 제외) {blocked:.0f} s / '
             'no_progress(무전진, 최종 안전망) {noprog:.0f} s')


def done_rule(margin: float, b: dict) -> str:
    return DONE_RULE.format(name='<name>', margin=margin,
                            grace=float(b['stop_grace_s']), stall=float(b['stall_end_s']),
                            blocked=float(b['blocked_end_s']),
                            noprog=float(b['no_progress_end_s']))


# 정차 사유 판정 어휘.
# world.light = [id, state] — SPEC §1.1: 0=비알람 1=적 2=황 3=녹 4=좌 5=녹+좌 6=점멸
HOLD_LIGHT_STATES = (1, 2, 6)             # 적·황·점멸 = 서 있어야 하는 신호
# decision.reasons.winner (run_agent._HAZARD_NAME): 'walker' 보행자 / 'bicycle' 자전거
CROSSING_WINNERS = ('walker', 'bicycle')
# winner 가 'lead' 여도 그 lead 가 보행자면 예외다 (speed_reduced_by.type 로 가른다).
CROSSING_TYPES = ('walker', 'pedestrian')


def stop_excused(tick: dict, intent_mps: float) -> bool:
    """이 정차가 **곧 스스로 풀리는 정상 정차**인가 — 종료 타이머를 돌리지 않는다.

    예외는 둘뿐이다.
      · 적(황·점멸)신호 대기 — 단 그 신호가 실제로 우리를 세우고 있을 때만.
        멀리 있는 적신호는 red_light 후보가 낮지 않으므로 걸러진다. **녹색이 되면
        예외가 끝난다** — 녹색인데 안 가는 건 정상 정차가 아니다(score green_stall).
      · 보행자·자전거가 경로를 가로지르는 중.

    선행차는 예외가 **아니다** — 앞차가 영원히 안 움직이면 우리도 못 간다
    (2026-08-30: 정지 차량 두 대 사이에 끼어 배치가 다음으로 못 넘어감).

    winner 만 보지 않는 이유: 정지 중 winner 가 route_end 등 허위값으로 튄다
    (실측 완주속도_01: route_s 2346 m 인데 route_end 후보 0.0).
    """
    reasons = (tick.get('decision') or {}).get('reasons') or {}
    light = (tick.get('world') or {}).get('light')
    if light and int(light[1]) in HOLD_LIGHT_STATES:
        red = reasons.get('red_light')
        if red is not None and float(red) < intent_mps:      # 그 신호가 우리를 세운다
            return True
    winner = str(reasons.get('winner') or 'none')
    if winner in CROSSING_WINNERS:
        return True
    src = (reasons.get('speed_reduced_by') or {}).get('type') or ''
    return any(k in str(src) for k in CROSSING_TYPES)


class EndJudge:
    """틱 스트림 → 종료 판정 (완주 / stall / blocked / no_progress).

    순수 로직 — pytest 대상. 임계는 전부 params batch.* 에서 온다.

    · 완주       : route_s ≥ total − margin, 이후 v<0.5 즉시 또는 stop_grace_s 유예
    · stall      : 정지인데 **계획은 진행**(v_target ≥ stall_intent_mps) — 전방에
                   정지 사유가 없다는 뜻(제어기가 못 감) — stall_end_s 지속
    · blocked    : 정지이고 **계획도 정지**인데 사유가 신호·보행자가 아님 — 앞이
                   막혀 못 감 — blocked_end_s 지속. stall 과 v_target 으로 배타적이다.
    · no_progress: route_s 가 progress_eps_m 도 안 늘어남 — 위 예외를 전부 뚫는
                   최종 안전망 (신호 고장으로 영원히 적색이어도 끝난다)

    stall 과 blocked 를 나누는 이유: 아침에 표를 볼 때 "제어기가 못 감"과 "앞이
    막혀서 못 감"은 원인도 대응도 다르다 (전자는 코드, 후자는 시나리오 배치).
    """

    def __init__(self, total: float, margin: float, cfg_batch: dict | None = None,
                 **over) -> None:
        b = dict(cfg_batch if cfg_batch is not None else load_cfg()['batch'])
        b.update({k: v for k, v in over.items() if v is not None})
        self.total, self.margin = total, margin
        self.grace = float(b['stop_grace_s'])
        self.stall_s = float(b['stall_end_s'])
        self.blocked_s = float(b['blocked_end_s'])
        self.no_prog_s = float(b['no_progress_end_s'])
        self.v_stop = float(b['stall_speed_mps'])
        self.v_intent = float(b['stall_intent_mps'])
        self.eps = float(b['progress_eps_m'])
        self.reached_at = None
        self.best_rs = -math.inf
        self.progress_t = None
        self.stall_t = None
        self.blocked_t = None

    def feed(self, now: float, tick: dict) -> str | None:
        """로그 틱 한 줄 → 종료 사유 또는 None.

        틱 전체를 받는다 — 정차 사유(decision.reasons)를 봐야 신호·보행자 대기와
        '앞이 막힘'을 가를 수 있다 (2026-08-30 시그니처 확대).
        """
        ego = tick['ego']
        dec = tick.get('decision') or {}
        route_s = float(ego['route_s'])
        v = float(ego['speed'])
        v_target = float(dec.get('v_target') or 0.0)

        if self.progress_t is None:
            self.progress_t = now
        # 완주
        if route_s >= self.total - self.margin:
            if self.reached_at is None:        # `or now` 는 now=0.0 을 falsy 로 오판
                self.reached_at = now
            if v < 0.5 or now - self.reached_at > self.grace:
                return '완주'
        # 전진 감시 — 최종 안전망 (예외 없음)
        if route_s > self.best_rs + self.eps:
            self.best_rs = route_s
            self.progress_t = now
        elif now - self.progress_t > self.no_prog_s:
            return 'no_progress'

        moving = v >= self.v_stop
        if moving:
            self.stall_t = self.blocked_t = None
            return None
        if v_target >= self.v_intent:          # 계획은 가려는데 못 간다
            self.blocked_t = None
            self.stall_t = now if self.stall_t is None else self.stall_t
            return 'stall' if now - self.stall_t > self.stall_s else None
        # 계획도 정지 — 사유가 신호·보행자면 정상 정차라 타이머를 돌리지 않는다
        self.stall_t = None
        if stop_excused(tick, self.v_intent):
            self.blocked_t = None
            return None
        self.blocked_t = now if self.blocked_t is None else self.blocked_t
        return 'blocked' if now - self.blocked_t > self.blocked_s else None


def _load_one(path: str) -> list[dict]:
    text = open(path, encoding='utf-8').read()
    if path.endswith(('.yml', '.yaml')):
        import yaml                                    # 없으면 그대로 ImportError
        items = yaml.safe_load(text)
    else:
        items = json.loads(text)
    out = []
    for i, it in enumerate(items):
        for k in ('name', 'vtd_xml_path', 'route_csv'):
            if k not in it:
                raise SystemExit(f'{path} 시나리오 [{i}] 에 {k} 가 없다: {it}')
        it.setdefault('timeout_s', 180.0)
        out.append(it)
    return out


def load_scenarios(paths: list[str]) -> list[dict]:
    """목록 파일 여러 개(glob 포함)를 합쳐 하나의 실행 목록으로.

    이름 중복 검사는 **통합 후** 기준이다 — 같은 시나리오가 두 목록에 들어 있으면
    (batch_all.json 과 주제별 json 을 같이 주는 실수 등) 여기서 잡힌다.
    """
    import glob as _glob
    files: list[str] = []
    for p in paths:
        hits = sorted(_glob.glob(p))
        if not hits:
            raise SystemExit(f'시나리오 목록이 없다: {p}')
        files += hits
    seen_files = set()
    out: list[dict] = []
    for f in files:
        if f in seen_files:                            # 같은 파일이 두 패턴에 걸린 경우
            continue
        seen_files.add(f)
        out += _load_one(f)
    names = [it['name'] for it in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit(f'통합 목록에서 시나리오 이름이 중복된다: {dup}\n'
                         f'  (입력 파일: {", ".join(files)})')
    return out


def tail_tick(path: pathlib.Path) -> dict | None:
    """jsonl 꼬리에서 마지막 **완전한** 틱 레코드 하나."""
    try:
        size = path.stat().st_size
        with open(path, 'rb') as f:
            f.seek(max(0, size - 65536))
            lines = f.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if '"raw"' not in line:
            continue
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue                                   # 잘린 마지막 줄
    return None


# ── IG 팔로워 뷰 카메라 (표시 전용 — GT·제어·채점 무관) ───────────────────
def build_camera_scp(cfg: dict) -> str | None:
    """IG 팔로워 뷰 <Camera> SCP 문자열. camera.enabled=false 면 None.

    GUI 프리셋 "Relative - follower view" 와 동일 뷰 (VTD 2025.2 Doc/SCP_HTML
    실기 확인 2026-08-27): PosRelative 는 자차 로컬 [m], ViewRelative 각도는
    [rad], Show Owner 는 <Camera showOwner="true|false"> 속성, <Set/> 이 적용.
    값은 params.yaml camera.* 가 단일 출처 — 키가 없으면 KeyError 로 죽는 게 맞다.
    """
    cam = cfg['camera']
    if not cam['enabled']:
        return None
    return (f'<Camera name="followCam" showOwner="{str(bool(cam["show_owner"])).lower()}">'
            f'<PosRelative player="{cam["player"]}" dx="{float(cam["dx"]):.2f}" '
            f'dy="{float(cam["dy"]):.2f}" dz="{float(cam["dz"]):.2f}"/>'
            f'<ViewRelative dh="{float(cam["dh"]):.4f}" dp="{float(cam["dp"]):.4f}" '
            f'dr="{float(cam["dr"]):.4f}"/>'
            f'<Set/></Camera>')


# ── 표 렌더 (콘솔과 report.txt 가 같은 함수를 쓴다 — 중복 구현 금지) ──────
def disp_w(s) -> int:
    """표시 폭 — 한글·전각(east_asian_width W/F)은 2칸.
    str.ljust 는 한글을 1칸으로 세므로 그대로 쓰면 표가 어긋난다."""
    return sum(2 if unicodedata.east_asian_width(ch) in 'WF' else 1 for ch in str(s))


def _pad(s, w: int) -> str:
    return str(s) + ' ' * max(0, w - disp_w(s))


def render_table(hdr: list, rows: list) -> str:
    """폭 정렬 텍스트 표. 컬럼 폭은 실제 내용의 최대 표시폭에 동적으로 맞춘다.

    셀은 자르지 않는다 — 폭 상한이 필요한 열은 호출자가 clip() 으로 미리
    줄여서 넘긴다 (표가 총폭 상한을 넘지 않도록 예산을 배분하는 쪽이 안다).
    """
    widths = [max(disp_w(h), *(disp_w(r[i]) for r in rows)) if rows else disp_w(h)
              for i, h in enumerate(hdr)]
    lines = ['  '.join(_pad(c, widths[i]) for i, c in enumerate(hdr)).rstrip()]
    lines += ['  '.join(_pad(c, widths[i]) for i, c in enumerate(row)).rstrip()
              for row in rows]
    return '\n'.join(lines)


# ── report.txt 요약 표 ────────────────────────────────────────────────────
# 밤샘 배치를 아침에 한 화면에서 훑기 위한 표다. 상세(항목별 건수·감점 내역)는
# <시나리오>.score.txt 와 report.json 에 그대로 남으므로 여기서는 잘라도 된다.
# 폭·개수·제외 키는 전부 params.yaml report.* 에서 온다 (하드코딩 금지).
from score import LABEL as DETECT_LABEL           # noqa: E402 — 한국어 표기 단일 출처

REPORT_HDR = ['시나리오', '상태', '거리[m]', '시간', '평균[km/h]', '감점',
              '이벤트', '주요위반']


def clip(s, w: int) -> str:
    """표시폭 w 로 자르고, 잘렸으면 … 를 붙인다 (… 은 1칸)."""
    s = str(s)
    if disp_w(s) <= w:
        return s
    out, used = '', 0
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in 'WF' else 1
        if used + cw > w - 1:
            break
        out, used = out + ch, used + cw
    return out + '…'


def short_label(key: str, w: int) -> str:
    """score.LABEL 의 한국어 표기를 기계적으로 축약 — 괄호·공백 제거 후 폭 상한.

    별도 축약 테이블을 두지 않는다: 표기가 바뀌어도 score.py 한 곳만 고치면 된다.
    '속도 초과(법규)' → '속도초과', '적신호 통과' → '적신호통과'.
    """
    return clip(re.sub(r'\(.*?\)', '', DETECT_LABEL.get(key, key)).replace(' ', ''), w)


def top_violations(vio: dict | None, rcfg: dict) -> str:
    """주요위반 칸 — 건수 상위 n 개를 '차선이탈2, 적신호통과2, 속도초과1' 로.

    n 개를 넘으면 뒤에 ' 외N'. 위반이 없으면 '-'. exclude_violations 키(미완주·
    정지 고착·경로 이탈)는 상태 칸과 중복이라 빼고 센다.
    """
    excl = set(rcfg['exclude_violations'])
    items = sorted(((k, v) for k, v in (vio or {}).items() if k not in excl),
                   key=lambda kv: (-kv[1], kv[0]))
    if not items:
        return '-'
    n, lw = int(rcfg['top_violations']), int(rcfg['violation_label_w'])
    head = ', '.join(f'{short_label(k, lw)}{v}' for k, v in items[:n])
    return head + (f' 외{len(items) - n}' if len(items) > n else '')


def mmss(t) -> str:
    """초 → 분:초 (522.7 → 8:43). 값이 없으면 '-'."""
    if t is None:
        return '-'
    m, s = divmod(int(round(float(t))), 60)
    return f'{m}:{s:02d}'


def events_cell(r: dict) -> str:
    """이벤트 조우 성립 칸 — '3/5 (미도달1)'. 없으면 '-', 판정 실패면 '?'.

    미도달(ego 가 그 지점까지 못 감)을 괄호로 따로 보인다 — 미완주 런에서
    '0/4' 만 보이면 도달조차 못 한 이벤트가 성립 실패로 읽힌다. 0 건이면 생략.
    """
    ok, tot = r.get('events_ok'), r.get('events_total')
    if ok is None or tot is None:
        return '?' if 'events_total' in r else '-'
    if tot == 0:
        return '-'
    un = int(r.get('events_unreached') or 0)
    return f'{ok}/{tot}' + (f' (미도달{un})' if un else '')


def _ded(r: dict) -> int:
    """시나리오 총 감점 (≤0). 채점이 안 돌았으면 0 으로 취급해 정렬만 시킨다."""
    return int(r.get('deduction') or 0)


def sort_key(r: dict):
    """(1) 미완주 → (2) 완주·감점 큰 순 → (3) 무감점 완주. 동률은 시나리오명."""
    grp = 0 if r.get('status') != '완주' else (1 if _ded(r) < 0 else 2)
    return (grp, _ded(r), str(r.get('name', '')))


def summary_line(results: list, rcfg: dict) -> str:
    """표 위 한 줄 — '20건 · 완주 18 / stall 1 · 평균 감점 -12.4 · 최악 이름(-31) · 수집실패 2'.

    수집실패 = 채점이 못 돌아 deduction 이 없는 시나리오 수 (표에서는 감점 칸 '-').
    0 이면 아예 표시하지 않는다 — 평시 요약을 늘리지 않기 위해서다.
    """
    if not results:
        return '0건'
    by_status: dict = {}
    for r in results:
        st = str(r.get('status', '?'))
        by_status[st] = by_status.get(st, 0) + 1
    done = by_status.pop('완주', 0)
    nw = int(rcfg['name_w'])
    parts = ([f'완주 {done}'] if done else []) + [
        f'{clip(k, nw)} {v}' for k, v in sorted(by_status.items(),
                                                key=lambda kv: (-kv[1], kv[0]))]
    out = f'{len(results)}건 · ' + ' / '.join(parts)
    scored = [r for r in results if r.get('deduction') is not None]
    if scored:
        out += f' · 평균 감점 {sum(_ded(r) for r in scored) / len(scored):.1f}'
        worst = min(scored, key=_ded)
        if _ded(worst) < 0:
            out += f' · 최악 {clip(worst.get("name", "?"), nw)}({_ded(worst)})'
    scored_ev = [r for r in results if r.get('events_total')]
    tot = sum(r['events_total'] for r in scored_ev)
    if tot:
        got = sum(r['events_ok'] for r in scored_ev)
        un = sum(int(r.get('events_unreached') or 0) for r in scored_ev)
        out += f' · 이벤트 {got}/{tot} 성립' + (f' (미도달{un})' if un else '')
    if len(results) - len(scored):
        out += f' · 수집실패 {len(results) - len(scored)}'
    return out


def render_report(results: list, rcfg: dict) -> str:
    """요약 한 줄 + 표. 콘솔과 report.txt 가 같은 문자열을 쓴다 (중복 구현 금지)."""
    nw, width = int(rcfg['name_w']), int(rcfg['table_width'])
    rows = [[clip(r.get('name', '?'), nw), r.get('status', '?'),
             r.get('dist_m', '-'), mmss(r.get('time_s')), r.get('avg_kph', '-'),
             '-' if r.get('deduction') is None else _ded(r), events_cell(r),
             top_violations(r.get('violations'), rcfg)] for r in sorted(results, key=sort_key)]
    if rows:
        # 마지막 열(주요위반)이 남는 폭을 먹는다 — 앞 열은 내용대로 두고 총폭만 지킨다.
        last = len(REPORT_HDR) - 1
        fixed = sum(max(disp_w(REPORT_HDR[i]), *(disp_w(row[i]) for row in rows))
                    for i in range(last))
        budget = max(width - fixed - 2 * last, disp_w(REPORT_HDR[last]))
        for row in rows:
            row[last] = clip(row[last], budget)
    return summary_line(results, rcfg) + '\n' + render_table(REPORT_HDR, rows)


class Runner:
    def __init__(self, args) -> None:
        self.args = args
        self.launch: subprocess.Popen | None = None
        self.scp: ScpClient | None = None
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.out_dir = _ROOT / 'logs' / 'batch' / ts
        self.results: list[dict] = []
        # 종료 판정 값은 컨트롤러와 같은 params.yaml 에서 (하드코딩 금지)
        self.cfg = cfg = load_cfg()
        self.margin = end_margin_m(cfg)
        self.batch_cfg = cfg['batch']       # 종료 판정 임계 전부 (EndJudge 가 읽는다)

    # ── 정리 (Ctrl+C 포함 모든 경로에서 호출) ─────────────────────────────
    def cleanup(self) -> None:
        if self.launch is not None and self.launch.poll() is None:
            try:
                # SIGINT(정상 종료) → SIGTERM → SIGKILL 순으로 격상.
                # launch 가 SIGINT/SIGTERM 에 무반응인 경우가 있어 최종 KILL 보장.
                os.killpg(self.launch.pid, signal.SIGINT)
                try:
                    self.launch.wait(timeout=8.0)
                except subprocess.TimeoutExpired:
                    os.killpg(self.launch.pid, signal.SIGTERM)
                    try:
                        self.launch.wait(timeout=4.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.launch.pid, signal.SIGKILL)
                        self.launch.wait(timeout=5.0)
            except (ProcessLookupError, PermissionError):
                pass
        self.launch = None
        if self.scp is not None:
            try:
                self.scp.stop()
                self.scp.close()
            except OSError:
                pass
            self.scp = None

    # ── IG 카메라 ─────────────────────────────────────────────────────────
    def _apply_camera(self):
        """팔로워 뷰 SCP 1회 전송. 표시 전용 — 실패해도 배치는 계속 (경고만).

        반환값이 run 메타(res['camera_sent'])에 남는다:
        True 전송 / False 실패(무시) / None 비활성(camera.enabled=false).
        """
        xml = build_camera_scp(self.cfg)
        if xml is None:
            return None
        try:
            self.scp.send(xml)
            return True
        except (OSError, RuntimeError) as e:
            print(f'  ⚠ 카메라 뷰 SCP 실패 — 무시하고 진행: {e}')
            return False

    # ── VTD 9910 예열 확인 ────────────────────────────────────────────────
    def _wait_9910(self, warmup_s: float) -> float | None:
        """9910 에서 1 프레임이라도 받을 때까지 대기 → 걸린 시간 [s], 초과 시 None.

        확인 후 즉시 닫는다 (vtd_bridge 가 붙을 자리를 오래 점유하지 않는다).
        """
        import socket
        t0 = time.monotonic()
        while time.monotonic() - t0 < warmup_s:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(2.0)
                s.connect((self.args.host, 9910))
                data = s.recv(4096)
                s.close()
                if data:
                    time.sleep(0.5)      # 프로브가 소켓을 놓은 뒤 컨트롤러가 붙게
                    return time.monotonic() - t0
            except OSError:
                try:
                    s.close()
                except OSError:
                    pass
            time.sleep(2.0)
        return None

    # ── 한 시나리오 ───────────────────────────────────────────────────────
    def run_one(self, sc: dict) -> dict:
        name = sc['name']
        # scenario_yaml: 이벤트 기대값(트리거 역산 결과)이 든 파일 — collect 의
        # 조우 성립 판정 입력. route_csv 와 같은 자리에 같은 이름으로 있다.
        res = {'name': name, 'status': '?', 'log': None,
               'scenario_yaml': str(pathlib.Path(sc['route_csv']).with_suffix('.yaml'))}
        # route pkl 은 이 배치의 로그 디렉터리에 — data/ 를 오염시키지 않는다
        route_pkl = self.out_dir / 'routes' / f'route_{name}.pkl'

        # 1) route 빌드 — 공용 data/route.pkl 은 건드리지 않는다
        build_cmd = [sys.executable, str(_ROOT / 'tools' / 'build_route.py'),
                     str(_ROOT / 'data' / 'lane_graph.pkl'), sc['route_csv'],
                     '-o', str(route_pkl)]
        agent_log = self.out_dir / f'{name}.jsonl'
        launch_cmd = [sys.executable, str(_ROOT / 'run_agent.py'),
                      '--route', str(route_pkl),
                      '--graph', str(_ROOT / 'data' / 'lane_graph.pkl'),
                      '--host', self.args.host,
                      '--log', str(agent_log)]
        if self.args.dry_run:
            print(f'\n[dry-run] {name}')
            print('  route :', ' '.join(build_cmd))
            print('  scp   : Load(%r) → Init → Camera → Start @ %s:48179'
                  % (sc['vtd_xml_path'], self.args.host))
            print('  camera:', build_camera_scp(self.cfg) or '비활성 (camera.enabled=false)')
            print('  launch:', ' '.join(launch_cmd))
            print(f'  종료   : {done_rule(self.margin, self.batch_cfg)} '
                  f'/ timeout {sc["timeout_s"]}s')
            res['status'] = 'dry-run'
            return res

        route_pkl.parent.mkdir(parents=True, exist_ok=True)
        cp = subprocess.run(build_cmd, capture_output=True, text=True, timeout=120)
        (self.out_dir / f'{name}.build_route.txt').write_text(cp.stdout + cp.stderr)
        if cp.returncode != 0 or not route_pkl.exists():
            res['status'] = 'route 실패'
            # ⚠ 요약 한 줄을 표 비고에 노출 — 파일을 열지 않아도 원인이 보이게
            warn = next((line.strip() for line in (cp.stdout + cp.stderr).splitlines()
                         if '⚠' in line), '')
            if warn:
                res['collect_error'] = warn[:140]
            print(f'  ✗ build_route 실패 (rc={cp.returncode}) — {warn or f"{name}.build_route.txt 참고"}')
            return res
        import pickle
        total = float(pickle.load(open(route_pkl, 'rb'))['total_length'])
        res['route_total_m'] = round(total, 1)

        # 2) VTD 로드→Init→(카메라)→Start — scp_client.load_and_run 을 단계로
        #    분해했다: IG 팔로워 뷰는 시나리오 Load 마다 초기화되므로 Init 완료
        #    후·Start 직전에 시나리오당 1회 다시 적용해야 한다 (표시 전용).
        try:
            self.scp = ScpClient(self.args.host, verbose=False).connect()
            settle = self.args.settle_s
            self.scp.load_scenario(sc['vtd_xml_path'])
            self.scp.poll(settle)
            self.scp.init()
            self.scp.poll(settle)                  # Init 완료 대기 (기존 settle 관례)
            res['camera_sent'] = self._apply_camera()
            self.scp.start()
            self.scp.poll(settle)
        except OSError as e:
            res['status'] = f'SCP 실패: {e}'
            self.cleanup()
            return res

        # 2.5) VTD 예열 대기 — 특히 기동 직후 첫 시나리오는 로드/Init 에 수십 초가
        # 걸린다. 9910 이 실제로 프레임을 흘릴 때까지 기다린 뒤에 컨트롤러를 띄우고
        # timeout 시계를 돌린다 (안 기다리면 no-data 로 오판).
        waited = self._wait_9910(float(self.args.vtd_warmup_s))
        if waited is None:
            res['status'] = f'VTD 9910 무응답 ({self.args.vtd_warmup_s:.0f}s 예열 대기 초과)'
            self.cleanup()
            return res
        if waited > 1.0:
            print(f'    VTD 예열 {waited:.0f}s 후 9910 수신 시작')

        # 3) 컨트롤러 실행 (새 세션 = 프로세스 그룹 → SIGINT 로 통째로 정리)
        launch_out = open(self.out_dir / f'{name}.launch.txt', 'w')
        self.launch = subprocess.Popen(launch_cmd, cwd=str(_ROOT), start_new_session=True,
                                       stdout=launch_out, stderr=subprocess.STDOUT)

        # 4) 로그 감시 — --log 로 지정한 파일이 이 시나리오의 로그다
        log_path = None
        t0 = time.monotonic()
        deadline = t0 + float(sc['timeout_s'])
        last_size, last_grow = -1, time.monotonic()
        judge = EndJudge(total, self.margin, self.batch_cfg)
        status = 'timeout'
        while time.monotonic() < deadline:
            time.sleep(POLL_S)
            if self.launch.poll() is not None:
                status = f'launch 사망 (rc={self.launch.returncode})'
                break
            if log_path is None:
                if agent_log.exists():
                    log_path = agent_log
                    print(f'    로그: {log_path.name}')
                elif time.monotonic() - t0 > NO_DATA_S:
                    status = 'no-data (로그 파일이 생기지 않음)'
                    break
                continue
            size = log_path.stat().st_size
            if size != last_size:
                last_size, last_grow = size, time.monotonic()
            elif time.monotonic() - last_grow > NO_DATA_S:
                status = f'no-data ({NO_DATA_S:.0f}s 로그 정체)'
                break
            tick = tail_tick(log_path)
            if tick is None:
                continue
            verdict = judge.feed(time.monotonic(), tick)
            if verdict is not None:
                status = verdict
                break

        # 5) 정리 + 로그 회수 (--log 로 이미 out_dir 에 쓰였다)
        self.cleanup()
        launch_out.close()
        res['status'] = status
        if agent_log.exists():
            res['log'] = str(agent_log)
        return res

    # ── 결과 수집 ─────────────────────────────────────────────────────────
    def collect(self, res: dict) -> None:
        if not res.get('log'):
            return
        import pickle
        lg = route = None
        try:
            from summarize_run import load_map
            lg, _ = load_map()
            route = pickle.load(open(self.out_dir / 'routes' / f"route_{res['name']}.pkl", 'rb'))
        except Exception as e:                          # noqa: BLE001
            res['collect_error'] = f'map: {type(e).__name__}: {e}'
        try:
            from summarize_run import summarize
            s = summarize(res['log'], lg, route)
            f = s.get('finish', {})
            res.update(done=f.get('done'), dist_m=f.get('dist_m'), time_s=f.get('time_s'),
                       avg_kph=f.get('avg_kph'), end_stopped=f.get('end_stopped'),
                       overspeed_ticks=s.get('overspeed', {}).get('ticks'),
                       resets=len(s.get('resets', [])))
            ticks = [json.loads(l) for l in open(res['log']) if '"raw"' in l]
            res['estop_ticks'] = sum(1 for t in ticks if t['decision']['state'] == 'E_STOP')
            fired: dict = {}
            for t in ticks:
                for k in (t['decision']['reasons'].get('shield') or {}):
                    fired[k] = fired.get(k, 0) + 1
            res['shield'] = fired
            dmin = math.inf
            for t in ticks:
                ex, ey = t['ego']['x'], t['ego']['y']
                for o in t['objects']:
                    dmin = min(dmin, math.hypot(o['x'] - ex, o['y'] - ey))
            res['min_obj_dist_m'] = None if math.isinf(dmin) else round(dmin, 2)
        except Exception as e:                          # noqa: BLE001 — 수집 실패도 표에 남긴다
            res['collect_error'] = res.get('collect_error', '') + f' {type(e).__name__}: {e}'
        # 위반 검출 (tools/score.py — 배점 없음, 건수만). 전문은 {name}.score.txt 로.
        try:
            import score as score_tool
            rep = score_tool.analyze(res['log'], load_cfg(), lg, route)
            (self.out_dir / f"{res['name']}.score.txt").write_text(
                score_tool.render(rep), encoding='utf-8')
            res['n_violations'] = rep['n_violations']
            res['violations'] = {k: d['count'] for k, d in rep['violations'].items()
                                 if d['count'] and k not in score_tool.INFO_KEYS}
            # 총 감점 (≤0). score_run 이 이미 도달 구간만 집계한 값을 그대로 쓴다 —
            # deductions 를 직접 합산하면 미도달 구간 몫까지 더해진다 (score 만 None).
            s = rep.get('scoring') or {}
            res['deduction'] = -(int(s['max_possible']) - int(s['total'])) if s else None
        except Exception as e:                          # noqa: BLE001
            res['n_violations'] = None
            res['deduction'] = None
            res['collect_error'] = res.get('collect_error', '') + f' score:{e}'
        # 이벤트 조우 성립 (tools/event_check.py — 시나리오 기대값 vs 실주행 횡위치).
        # 밤샘 배치에서 "보행자가 지나간 뒤 건넜다" 류를 아침에 눈으로 못 잡는다.
        try:
            import event_check
            ev = event_check.check_scenario(res['scenario_yaml'], res['log'], load_cfg())
            (self.out_dir / f"{res['name']}.events.txt").write_text(
                event_check.render(ev, res['name']), encoding='utf-8')
            res['events_ok'], res['events_total'] = ev['ok'], ev['total']
            res['events_unreached'] = ev['unreached']
            res['events'] = ev['events']
        except Exception as e:                          # noqa: BLE001
            res['events_ok'] = res['events_total'] = res['events_unreached'] = None
            res['collect_error'] = res.get('collect_error', '') + f' events:{e}'

    # ── 표 ────────────────────────────────────────────────────────────────
    def report(self) -> str:
        # 삭제한 열(완주·E_STOP·shield·최소객체거리·비고)은 report.json 에 그대로 있다.
        return render_report(self.results, self.cfg['report'])

    # ── 전체 ──────────────────────────────────────────────────────────────
    # ── 실행 전 검사: VTD PC 에 시나리오 xml 이 실제로 있는가 ────────────
    def precheck_vtd_paths(self, scenarios: list[dict]) -> None:
        """첫 시나리오의 vtd_xml_path 를 ssh ls 로 실존 확인.

        과거 사고: /home/vtd/... 프리픽스로 생성된 목록이 실기에 존재하지 않아
        5개 중 2개가 로드 실패(no-data)로 시간만 태웠다. 없으면 즉시 중단한다.
        """
        path = scenarios[0]['vtd_xml_path']
        user = path.split('/')[2] if path.startswith('/home/') else 'mjw'
        target = self.args.ssh or f'{user}@{self.args.host}'
        cp = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5',
                             target, 'ls', path],
                            capture_output=True, text=True, timeout=15)
        if cp.returncode != 0:
            raise SystemExit(
                f'[중단] VTD PC({target})에 {path} 가 없다 — 복사 안 됨 또는 경로 불일치.\n'
                f'       scenarios/ 를 VTD PC 로 복사했는지, gen_scenarios --vtd-dir 프리픽스가\n'
                f'       실제 복사 위치와 같은지 확인할 것. (ssh 출력: {cp.stderr.strip() or cp.stdout.strip()})')
        print(f'VTD 경로 확인: {target}:{path} OK')

    def run(self) -> int:
        scenarios = load_scenarios(self.args.scenarios)
        if not self.args.dry_run:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.precheck_vtd_paths(scenarios)
        print(f'배치 {len(scenarios)}건, host={self.args.host}')
        print(done_rule(self.margin, self.batch_cfg))
        try:
            for i, sc in enumerate(scenarios, 1):
                print(f'\n[{i}/{len(scenarios)}] {sc["name"]}  (timeout {sc["timeout_s"]}s)')
                try:
                    res = self.run_one(sc)
                except KeyboardInterrupt:
                    raise
                except Exception as e:                  # noqa: BLE001 — 실패해도 다음으로
                    res = {'name': sc['name'], 'status': f'예외: {type(e).__name__}: {e}', 'log': None}
                    self.cleanup()
                self.results.append(res)
                if not self.args.dry_run:
                    self.collect(res)
                print(f'  → {res["status"]}')
                time.sleep(self.args.pause_s)
        except KeyboardInterrupt:
            print('\n[중단] 정리 중…')
            self.cleanup()
        table = self.report()
        rule = done_rule(self.margin, self.batch_cfg)
        # 판정 기준 설명문은 표 아래로 — 표가 첫 화면에 오도록.
        print('\n' + '=' * int(self.cfg['report']['table_width']))
        print(table)
        print('\n' + rule)
        if not self.args.dry_run:
            (self.out_dir / 'report.txt').write_text(table + '\n\n' + rule + '\n', encoding='utf-8')
            (self.out_dir / 'report.json').write_text(
                json.dumps(self.results, ensure_ascii=False, indent=1), encoding='utf-8')
            print(f'\n리포트: {self.out_dir}/report.txt (.json, 개별 로그·score 전문 포함)')
        return 0 if all(r['status'] in ('완주', 'dry-run') for r in self.results) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='배치 회귀 실행기 (SCP 원격 제어)')
    ap.add_argument('scenarios', nargs='+',
                    help='시나리오 목록 json/yaml — 여러 개·glob 가능, 합쳐서 순차 실행')
    ap.add_argument('--host', default='192.168.10.1', help='VTD 주소 (SCP 48179 + 9910)')
    ap.add_argument('--ssh', default=None,
                    help='실행 전 시나리오 실존 확인용 ssh 대상 (기본: <vtd_xml_path 의 홈 사용자>@host)')
    ap.add_argument('--settle-s', type=float, default=3.0, help='SCP 명령 간 대기')
    ap.add_argument('--vtd-warmup-s', type=float, default=120.0,
                    help='시나리오 Start 후 9910 첫 프레임까지 최대 대기 (VTD 기동 직후 대비)')
    ap.add_argument('--pause-s', type=float, default=5.0,
                    help='시나리오 간 대기 (9910 소켓 TIME_WAIT 정리 여유)')
    ap.add_argument('--dry-run', action='store_true', help='실행 없이 계획만 출력')
    a = ap.parse_args(argv)
    return Runner(a).run()


if __name__ == '__main__':
    raise SystemExit(main())
