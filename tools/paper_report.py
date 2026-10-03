#!/usr/bin/env python3
"""논문 Ⅳ장 집계기 — 구현(arm)별 로그 폴더 → 표·CSV·그림.

    python3 tools/paper_report.py --arm A1=logs/paper/A1 --arm A3=logs/paper/A3 \\
                                  --arm A4=logs/paper/A4 [--arm A2=/경로/초기구현로그]

입력은 "구현 이름 = 로그 폴더" 목록이다. 폴더 구조는 batch_run 산출물 그대로:
    <폴더>/<유형>/report.json     결과 목록 (status · violations · arm 꼬리표 …)
    <폴더>/<유형>/<이름>.jsonl    틱 로그
다른 브랜치에서 돌린 구현도 이 구조만 맞으면 그대로 합쳐진다 (--label-ko/--label-en 으로 이름).

**수치는 전부 report.json 과 로그에서 계산한다.** 시나리오의 장애물 위치는 시나리오 XML
에서, 판정 임계는 그 런이 쓴 설정 사본(report.json 의 config)에서 읽는다.

산출물 (--out, 기본 logs/paper/report/):
  table2.csv / table2.md   구현 × (무한 정지 · timeout · 충돌 · 완주율 · 95% Wilson 구간)
  runs.csv                 런 단위 원자료 (아래 표들의 근거)
  stall_review.csv         무한 정지마다 신호 상태 · 정지선 거리 · 앞 물체 — 사람이 걸러 볼 자료
  per_encounter.csv        장애물 조우 × 구현 (회피 성공/충돌/정지/미도달), per_encounter_long.csv
  by_type.csv              유형 × 구현
  paired.csv               같은 시나리오 짝 비교 (McNemar 정확 검정) — 기준 구현 vs 나머지
  collisions.csv           충돌 이벤트 전부 (상대 물체·최소 간격)
  collisions_after_finish.csv  종료선 이후라 세지 않은 충돌 (--after-finish exclude, 기본)
  causes.csv               실패 런 · 기반 실패 런의 원인 분류 (제어기 판단 / 오탐 / 시나리오 / 시뮬레이터)
  brake_fp.csv             앞차 급정거 유형에서 만든 시프트와 그 대상이 다시 움직였는가
  brake_fp_summary.csv
  figures/                 fig_*.png 과 원천 fig_*.csv (글자는 영문)
  meta.json                입력 · 임계 · 포함/제외 런 수
                           + 재현 정보: git(커밋 해시 · 미커밋 변경) · base_params(기준 params 해시) ·
                           scenario_manifest(MANIFEST.sha256 의 해시와 대조 결과) ·
                           arms.<구현>.settings(런이 실제로 쓴 overlay 키 전체 · 기준 params 해시 · 코드)

정의:
  무한 정지 = 종료 사유 stall + blocked (적신호·보행자 면책 없이 판정 시간만큼 정지)
  timeout   = 시나리오 제한 시간까지 끝나지 않음 — 무한 정지와 따로 센다
  집계 기준 (2026-10-03) = 지나감 횡거리 8 m (--pass-lat-m) · 종료선 이후 사건 제외
              (--after-finish exclude). 6.0 / include 를 주면 이전 집계.
  실패 원인 = failure_cause() — 로그에서 읽는 규칙 (meta.json definitions.cause_rules).
  충돌      = score.py 항목 11 (객체 외곽 간격 ≤ score.collision_dist_m). **이 저장소의
              검출기로 로그를 다시 센다** — 다른 브랜치 로그도 같은 자로 잰다. report.json 의
              값과 다르면 meta.json 의 warnings 에 남는다.
  완주율    = 완주 / 유효 런. 유효 런 = 주행이 끝까지 간 런(완주·stall·blocked·no_progress·
              timeout). 기반 실패(SCP·VTD 무응답·no-data …)는 분모에서 빼고 따로 센다.
  회피 유형 = 시나리오에 자차로 정지 장애물(static_vehicle · obstacle_chain)이 있는 유형.
              표 2·by_type·paired·per_encounter 는 회피 유형만 센다.
  급정거 유형 = lead_brake 이벤트가 있는 유형 — brake_fp 만 센다.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import json
import math
import pathlib
import statistics
import sys
import xml.etree.ElementTree as ET

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))

import batch_run          # noqa: E402
import score              # noqa: E402
from vtd_adapter.config import load_params_yaml   # noqa: E402

DONE = batch_run.Runner.DONE_STATUS                 # 주행이 끝까지 간 런
INF_STOP = ('stall', 'blocked')                     # 무한 정지
HOLD = batch_run.HOLD_LIGHT_STATES                  # 서 있어야 하는 신호 state
LIGHT_NAME = {0: 'none', 1: 'red', 2: 'yellow', 3: 'green', 4: 'left', 5: 'green+left', 6: 'blink'}


# ════════════════════════════════════════════════════════════════════════
# 통계
# ════════════════════════════════════════════════════════════════════════
def wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """이항 비율의 Wilson 점수 구간 → (하한, 상한). n = 0 이면 (nan, nan)."""
    if n <= 0:
        return math.nan, math.nan
    z = statistics.NormalDist().inv_cdf(0.5 + conf / 2.0)
    p = k / n
    den = 1.0 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    # k = 0 / k = n 의 끝은 식으로도 정확히 0 / 1 이다 — 부동소수 찌꺼기(0.99999…)를 남기지 않는다.
    return (0.0 if k == 0 else max(0.0, mid - half)), (1.0 if k == n else min(1.0, mid + half))


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 정확 검정(양측) p 값. b·c = 서로 엇갈린 짝의 수. 엇갈린 짝이 없으면 1."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n
    return min(1.0, 2.0 * tail)


# ════════════════════════════════════════════════════════════════════════
# 시나리오 정의 (yaml + XML) — 장애물 위치의 출처
# ════════════════════════════════════════════════════════════════════════
def parse_scenario(yaml_path: pathlib.Path) -> dict | None:
    """정의 yaml 과 같은 이름의 XML → 이벤트와 물체 좌표.

    조우 = 자차로를 막는 static_vehicle 1대 또는 obstacle_chain 한 줄. 체인 물체는
    XML 의 Obstacle* 를 이벤트 순서대로 count 개씩 가져온다 (생성기가 그 순서로 쓴다).
    """
    if not yaml_path.exists():
        return None
    d = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    pos: dict = {}
    chain_objs: list = []
    cones: list = []
    xml = yaml_path.with_suffix('.xml')
    if xml.exists():
        root = ET.parse(xml).getroot()
        # PathShape 로 시작하는 차량(옆 차로 교통 · 2026-10-03 부터 신호 대기 줄)은 첫 웨이포인트가 시작점
        shapes = {}
        for ps in root.iter('PathShape'):
            wp = ps.find('Waypoint')
            if wp is not None:
                shapes[ps.get('ShapeId')] = (float(wp.get('X')), float(wp.get('Y')))
        for pl in root.iter('Player'):
            desc, init = pl.find('Description'), pl.find('Init')
            pa = init.find('PosAbsolute') if init is not None else None
            psr = init.find('PathShapeRef') if init is not None else None
            if desc is not None and pa is not None:
                pos[desc.get('Name')] = (float(pa.get('X')), float(pa.get('Y')))
            elif desc is not None and psr is not None and psr.get('PathShapeId') in shapes:
                pos[desc.get('Name')] = shapes[psr.get('PathShapeId')]
        for ob in root.iter('Object'):
            sp = ob.find('StartPosAbs')
            if sp is None:
                continue
            xy = (float(sp.get('X')), float(sp.get('Y')))
            name = ob.get('Name') or ''
            if name.startswith('Obstacle'):
                chain_objs.append((name, xy))
            elif name.startswith('FinishCone'):
                cones.append((name, xy))
    encounters, kinds = [], []
    ci = 0
    for ei, e in enumerate(d.get('events') or []):
        kinds.append(e.get('kind'))
        if e.get('kind') == 'static_vehicle' and e.get('lane') != '좌측차로':
            xy = pos.get(e.get('name'))
            encounters.append({'event': ei, 'kind': 'static_vehicle', 'route_s': e.get('route_s'),
                               'objects': [{'name': e.get('name'), 'xy': xy}]})
        elif e.get('kind') == 'obstacle_chain':
            objs = chain_objs[ci:ci + int(e.get('count', 0))]
            ci += int(e.get('count', 0))
            rs = e.get('route_s') or []
            encounters.append({'event': ei, 'kind': 'obstacle_chain',
                               'route_s': rs[0] if rs else None,
                               'objects': [{'name': n, 'xy': xy} for n, xy in objs]})
        elif e.get('objects'):
            # 논문 세트의 이벤트 (parked_row · side_by_side · zigzag · side_traffic …) —
            # 정의가 "지나가야 할 물체" 이름을 직접 적어 둔다. 전부 지나야 그 조우가 성공이다.
            rs = e.get('route_s')
            encounters.append({'event': ei, 'kind': e.get('kind'),
                               'route_s': rs[0] if isinstance(rs, list) and rs else rs,
                               'objects': [{'name': n, 'xy': pos.get(n)} for n in e['objects']]})
    lead = [e for e in (d.get('events') or []) if e.get('kind') == 'lead_brake']
    queue = [e for e in (d.get('events') or []) if e.get('kind') == 'signal_queue']
    # 유형 분류는 이벤트에서 나온다 (테마 이름을 코드에 적지 않는다).
    cls = 'avoid' if encounters else 'brake' if lead else 'queue' if queue else 'plain'
    # 두 브랜치의 빌더가 같은 차로열을 만드는가 — 생성기가 정의 옆에 남긴 기록.
    same, builders = None, None
    pj = yaml_path.with_name(yaml_path.stem + '.placement.json')
    if pj.exists():
        rq = json.loads(pj.read_text(encoding='utf-8')).get('route_quality') or {}
        ob = rq.get('other_builder') or {}
        same = ob.get('same_lanes')
        if ob:
            # 비교 제외 판단의 원자료 (판단은 main — builders_differ)
            builders = {'other': ob.get('name'), 'other_rc': ob.get('rc'),
                        'off_lane_this': rq.get('builder_off_lane') or [],
                        'off_lane_other': ob.get('off_lane') or [],
                        'len_this_m': rq.get('builder_length_m'), 'len_other_m': ob.get('length_m')}
    return {'name': d.get('name'), 'theme': d.get('theme'), 'kinds': kinds,
            'class': cls, 'builders_same': same, 'builders': builders,
            'queue_objs': [(n, pos.get(n)) for e in queue for n in e.get('queue', [])],
            'encounters': encounters, 'cones': cones,
            'lead_brake': [e for e in (d.get('events') or []) if e.get('kind') == 'lead_brake'],
            'signal_queue': [dict(e, xy=[pos.get(n) for n in e.get('queue', [])])
                             for e in (d.get('events') or []) if e.get('kind') == 'signal_queue'],
            'axes': d.get('axes') or {}}


# ════════════════════════════════════════════════════════════════════════
# 입력: 구현 폴더 → 런 목록
# ════════════════════════════════════════════════════════════════════════
def load_arm(arm: str, folder: pathlib.Path, scen_dir: pathlib.Path) -> list[dict]:
    """<폴더>/<유형>/report.json (+ <폴더>/<유형>/*/report.json) → 런 목록.

    유형 폴더 안에 배치 폴더가 여러 개 있어도 된다 — --out-dir/--resume 이 없는 브랜치에서
    logs/batch/<시각>/ 을 통째로 옮겨 넣은 경우다 (끊겼다 이어 돌리면 배치가 여럿이 된다).
    같은 시나리오가 여러 번 있으면 **끝까지 간 런**(DONE)을, 그중에서는 나중 것을 쓴다.
    """
    runs = []
    for td in sorted(p for p in folder.iterdir() if p.is_dir()):
        reports = ([td / 'report.json'] if (td / 'report.json').exists() else []) \
            + sorted(td.glob('*/report.json'))
        best: dict = {}
        for rp in reports:
            for res in json.loads(rp.read_text(encoding='utf-8')):
                name = res.get('name')
                old = best.get(name)
                if old is None or res.get('status') in DONE or old[0].get('status') not in DONE:
                    best[name] = (res, rp.parent)
        for name, (res, rdir) in best.items():
            log = pathlib.Path(res['log']) if res.get('log') else None
            if log is None or not log.exists():          # 다른 PC·브랜치에서 온 절대경로
                cand = rdir / f'{name}.jsonl'
                log = cand if cand.exists() else None
            runs.append({'arm': arm, 'type': td.name, 'name': name, 'status': res.get('status'),
                         'res': res, 'log': log, 'dir': rdir,
                         'yaml': scen_dir / td.name / f'{name}.yaml'})
    return runs


def manifest_info(scen_dir: pathlib.Path) -> dict | None:
    """시나리오 세트의 지문 — <scenarios>/MANIFEST.sha256 자체의 해시와, 지금 파일이 그 목록과 맞는지.

    MANIFEST 는 gen_scenarios.py --manifest 가 쓴다 (`<sha256>  <상대경로>` 줄). 해시 하나만
    적어 두면 "어느 시나리오 세트로 돌렸나" 가 정해지고, mismatch 가 비어 있지 않으면 생성
    뒤에 파일이 바뀐 것이다. MANIFEST 가 없으면 None.
    """
    p = scen_dir / 'MANIFEST.sha256'
    if not p.exists():
        return None
    raw = p.read_bytes()
    bad, n = [], 0
    for ln in raw.decode('utf-8').splitlines():
        want, _, rel = ln.partition('  ')
        if not rel:
            continue
        n += 1
        f = scen_dir / rel
        if not f.exists() or hashlib.sha256(f.read_bytes()).hexdigest() != want:
            bad.append(rel)
    return {'path': str(p), 'sha256': hashlib.sha256(raw).hexdigest(), 'files': n, 'mismatch': bad}


def arm_settings(runs: list[dict], arm: str) -> dict:
    """그 구현의 런들이 실제로 쓴 설정·코드 — 런마다 report.json 에 남은 꼬리표를 모은다.

    값이 하나면 전 런이 같은 설정·같은 코드로 돈 것이다. 둘 이상이면 도중에 바뀐 것이라
    (params 수정 뒤 make_arms 재생성, 커밋·수정 뒤 이어 돌리기) 그대로 다 보여 준다.
    꼬리표가 없는 런(--config 없이 돈 다른 브랜치 결과)은 untagged_runs 로 센다.
    """
    mine = [r['res'] for r in runs if r['arm'] == arm]

    def distinct(key):
        seen = {json.dumps(x.get(key), sort_keys=True, ensure_ascii=False) for x in mine if x.get(key) is not None}
        return [json.loads(s) for s in sorted(seen)]
    codes: dict = {}
    for x in mine:
        g = x.get('git')
        if g:
            k = (g.get('commit'), g.get('branch'), g.get('dirty'), tuple(g.get('modified') or ()))
            codes.setdefault(k, {'commit': g.get('commit'), 'branch': g.get('branch'), 'dirty': g.get('dirty'),
                                 'modified': g.get('modified'), 'untracked': g.get('untracked'), 'runs': 0})
            codes[k]['runs'] += 1
    return {'config': distinct('config'), 'overlay': distinct('arm_overlay'),
            'base_params_sha256': distinct('arm_base_sha256'), 'judge': distinct('judge'),
            'code': list(codes.values()),
            'untagged_runs': sum(1 for x in mine if not x.get('config')),
            'runs_without_git': sum(1 for x in mine if not x.get('git'))}


_CFG_CACHE: dict = {}


def run_cfg(run: dict) -> dict:
    """그 런이 쓴 설정 (report.json 의 config). 없으면 기준 params."""
    p = run['res'].get('config')
    key = p if p and pathlib.Path(p).exists() else None
    if key not in _CFG_CACHE:
        _CFG_CACHE[key] = load_params_yaml(key)
    return _CFG_CACHE[key]


# ════════════════════════════════════════════════════════════════════════
# 런 하나 분석 (로그 1회 읽기)
# ════════════════════════════════════════════════════════════════════════
def ego_frame(e: dict, x: float, y: float) -> tuple[float, float]:
    """월드 (x,y) → 자차(뒷축) 기준 (종거리 앞 +, 횡거리 좌 +)."""
    dx, dy = x - e['x'], y - e['y']
    c, s = math.cos(e['yaw']), math.sin(e['yaw'])
    return dx * c + dy * s, -dx * s + dy * c


def match_objects(ticks: list, points: list, tol: float) -> dict:
    """기대 좌표 → 로그의 GT 객체 {점 인덱스: (id, 길이, 폭, heading)}.
    정지 물체라 처음 보인 틱의 좌표로 충분하다."""
    out: dict = {}
    seen: set = set()
    for t in ticks:
        for o in t['raw']['objects']:
            oid = int(o[0])
            if oid in seen:
                continue
            seen.add(oid)
            for i, xy in enumerate(points):
                if i in out or xy is None:
                    continue
                if math.hypot(o[1] - xy[0], o[2] - xy[1]) <= tol:
                    out[i] = (oid, float(o[6] or 0.0), float(o[7] or 0.0), float(o[4]))
        if len(out) == len(points):
            break
    return out


def analyze_run(run: dict, scen: dict | None, opt) -> dict:
    """런 하나의 파생값 — 충돌·조우·정지 검토·시프트. 로그가 없으면 빈 값."""
    out = {'collisions': [], 'collisions_after_finish': [], 'finish': None, 'encounters': [],
           'review': None, 'snapshot': None, 'shifts': [],
           'queue': None, 'n_ticks': 0, 'dur_s': None, 'warn': []}
    if run['log'] is None:
        out['warn'].append('로그 없음')
        return out
    ticks = score.load_ticks(str(run['log']))
    out['n_ticks'] = len(ticks)
    if not ticks:
        out['warn'].append('틱 없음')
        return out
    cfg = run_cfg(run)
    veh = cfg['vehicle']
    t0 = ticks[0]['t']
    if scen is None:
        out['warn'].append(f'시나리오 정의 없음 ({run["yaml"]}) — 조우·물체 구분을 못 한다')
    out['dur_s'] = round(ticks[-1]['t'] - t0, 1)
    rear_ov = float(veh['length']) - float(veh['wheelbase']) - float(veh['front_overhang_m'])
    front = float(veh['wheelbase']) + float(veh['front_overhang_m'])

    # ── 물체 식별: 조우 물체 · 종료선 콘 ────────────────────────────────
    enc = (scen or {}).get('encounters') or []
    pts, owner = [], []
    for k, e in enumerate(enc):
        for o in e['objects']:
            pts.append(o['xy']); owner.append(('event', k))
    for _n, xy in (scen or {}).get('cones') or []:
        pts.append(xy); owner.append(('finish_cone', None))
    for _n, xy in (scen or {}).get('queue_objs') or []:
        pts.append(xy); owner.append(('queue', None))
    matched = match_objects(ticks, pts, opt.match_m)
    kind_of = {oid: owner[i] for i, (oid, _l, _w, _h) in matched.items()}

    # ── 종료선 (2026-10-03 집계 기준: 종료선 이후 사건은 세지 않는다) ─────
    # 완주 판정 뒤의 몇 초는 채점 대상 주행이 아니다 — 그 사이 스크립트 교통이 자차 쪽으로
    # 꺾여 들어온 접촉(논문6_07)이 실패로 잡혔다. 기준은 report.json 의 finish_thr_m
    # (batch_run 이 채점과 같은 규칙으로 쓴 종료선 route_s — won 브랜치도 같은 키를 쓴다).
    fin = run['res'].get('finish_thr_m')
    cut = len(ticks)
    if opt.after_finish == 'exclude':
        if fin is None:
            out['warn'].append('report.json 에 finish_thr_m 이 없다 — 종료선 이후 사건을 거르지 못했다')
        else:
            cut = next((i for i, t in enumerate(ticks) if float(t['ego']['route_s']) >= float(fin)), len(ticks))
    out['finish'] = {'finish_thr_m': fin, 'cut_t_s': round(ticks[cut]['t'] - t0, 1) if cut < len(ticks) else None}
    out['collisions_after_finish'] = []

    # ── 충돌 (score.py 항목 11 과 같은 검출기·같은 구간) ───────────────
    tail = score.driving_span(ticks) if run['status'] == '완주' else ticks
    col, _near = score.detect_proximity(tail, t0, cfg)
    col_thr = float(cfg['score']['collision_dist_m'])
    for ev in col:
        kind, k = kind_of.get(int(ev['obj_id']), ('other', None))
        c = {'t': ev['t0'], 'route_s': ev['s0'], 'obj_id': int(ev['obj_id']),
             'obj_cls': ev.get('obj_cls'), 'min_gap_m': ev['min_gap_m'],
             'v_kph': ev.get('v_kph'), 'obj_kind': kind, 'encounter': k,
             **contact_onset(tail[ev['i0']:ev['i1'] + 1], int(ev['obj_id']), veh, col_thr, t0)}
        if ev['i0'] >= cut:
            out['collisions_after_finish'].append(c)
        else:
            out['collisions'].append(c)

    # ── 조우별 성패 ─────────────────────────────────────────────────────
    detect = float((cfg.get('ctrl24') or {}).get('detect_max_m', opt.detect_m))
    base = 0
    for k, e in enumerate(enc):
        n = len(e['objects'])
        idx = range(base, base + n)
        base += n
        dmin, passed, t_reach, t_pass = math.inf, [False] * n, None, None
        gap_min = math.inf
        for t in ticks:
            eg = t['ego']
            for j, i in enumerate(idx):
                xy = pts[i]
                if xy is None:
                    continue
                d = math.hypot(xy[0] - eg['x'], xy[1] - eg['y'])
                if j == 0:
                    dmin = min(dmin, d)
                    if t_reach is None and d <= detect:
                        t_reach = round(t['t'] - t0, 1)
                if passed[j] or d > opt.pass_near_m:
                    continue
                lon, lat = ego_frame(eg, xy[0], xy[1])
                half = matched[i][1] / 2.0 if i in matched else 0.0
                if lon < -(rear_ov + half) and abs(lat) <= opt.pass_lat_m:
                    passed[j] = True
                    if all(passed):
                        t_pass = round(t['t'] - t0, 1)
        ids = [matched[i][0] for i in idx if i in matched]
        # 최소 외곽 간격 — 조우 물체가 로그에 보인 틱만
        if ids:
            c_off = float(veh['length']) / 2.0 - rear_ov
            dim = {matched[i][0]: matched[i][1:] for i in idx if i in matched}
            for t in ticks:
                eg = t['ego']
                rect = None
                for o in t['raw']['objects']:
                    oid = int(o[0])
                    if oid not in dim or math.hypot(o[1] - eg['x'], o[2] - eg['y']) > opt.pass_near_m:
                        continue
                    if rect is None:
                        rect = score._corners(eg['x'] + c_off * math.cos(eg['yaw']),
                                              eg['y'] + c_off * math.sin(eg['yaw']),
                                              eg['yaw'], float(veh['length']), float(veh['width']))
                    g = score._rect_gap(rect, score._corners(o[1], o[2], o[4],
                                                             o[6] or 1.0, o[7] or 1.0))
                    gap_min = min(gap_min, g)
        hit = [c for c in out['collisions'] if c['encounter'] == k]
        reached = dmin <= detect
        if dmin <= opt.pass_near_m and not ids:
            # 코앞까지 갔는데 그 자리에 물체가 로그에 없다 — 다른 시나리오 파일로 돌렸을 수 있다.
            out['warn'].append(f'조우 {e["event"]}: {dmin:.1f} m 까지 갔으나 그 좌표에 객체가 없다 '
                               f'(시나리오 파일 불일치?)')
        if hit:
            outcome = 'collision'
        elif all(passed):
            outcome = 'success'
        elif reached:
            outcome = 'stopped'
        else:
            outcome = 'not_reached'
        out['encounters'].append({
            'event': e['event'], 'kind': e['kind'], 'n_objects': n, 'route_s': e['route_s'],
            'outcome': outcome, 'reached': reached, 'passed': all(passed),
            'n_passed': sum(passed), 'collisions': len(hit),
            'min_gap_m': None if math.isinf(gap_min) else round(gap_min, 2),
            'min_dist_m': None if math.isinf(dmin) else round(dmin, 1),
            't_reach_s': t_reach, 't_pass_s': t_pass, 'gt_ids': ids})

    # ── 끝 상태 (마지막 틱 + 마지막 정지 구간) — 무한 정지 검토·실패 사유의 근거 ──
    out['snapshot'] = stall_review(run, ticks, t0, cfg, enc, pts, kind_of, front, opt, matched)
    if run['status'] in INF_STOP:
        out['review'] = out['snapshot']

    # ── 시프트 생성 (오탐 판정) ─────────────────────────────────────────
    out['shifts'] = shift_events(ticks, t0, cfg, kind_of, upto=cut)

    # ── 신호 대기 줄: 대기 차량이 풀린 시각 · 자차가 줄 뒤에 닿았는가 ────
    qids = {oid for oid, (kind, _k) in kind_of.items() if kind == 'queue'}
    if (scen or {}).get('queue_objs'):
        thr = float((cfg.get('ctrl24') or {}).get('blocker_speed_max', 0.5))
        t_rel, met = None, False
        for t in ticks:
            for o in t['raw']['objects']:
                if int(o[0]) not in qids:
                    continue
                if t_rel is None and abs(float(o[5])) > thr:
                    t_rel = round(t['t'] - t0, 2)
                if t_rel is None and math.hypot(o[1] - t['ego']['x'], o[2] - t['ego']['y']) <= opt.ahead_m + front:
                    met = True
            if t_rel is not None:
                break
        out['queue'] = {'ids': sorted(qids), 't_release_s': t_rel, 'met_before_release': met,
                        'shifts_on_queue': [s for s in out['shifts'] if s['blocker_kind'] == 'queue'
                                            and (t_rel is None or s['t'] < t_rel)]}
    return out


def stall_review(run, ticks, t0, cfg, enc, pts, kind_of, front, opt, matched=None) -> dict:
    last = ticks[-1]
    eg = last['ego']
    v_stop = float(cfg['batch']['stall_speed_mps'])
    # 마지막 연속 정지 구간
    i = len(ticks) - 1
    while i > 0 and float(ticks[i - 1]['ego']['speed']) < v_stop:
        i -= 1
    win = ticks[i:]
    lights = [(t.get('world') or {}).get('light') for t in win]
    hold = sum(1 for l in lights if l and int(l[1]) in HOLD)
    world = last.get('world') or {}
    light = world.get('light')
    summ = world.get('summ') or {}
    # 앞 물체 — 자차 앞 ahead_m · 횡 ahead_lat_m 안에서 가장 가까운 것
    best = None
    for o in last.get('objects') or []:
        lon, lat = ego_frame(eg, o['x'], o['y'])
        if 0.0 < lon <= opt.ahead_m + front and abs(lat) <= opt.ahead_lat_m:
            if best is None or lon < best[0]:
                best = (lon, lat, o)
    ahead = {}
    if best is not None:
        lon, lat, o = best
        kind, k = kind_of.get(int(o['id']), ('other', None))
        ahead = {'ahead_id': int(o['id']), 'ahead_cls': o.get('cls'), 'ahead_kind': kind,
                 'ahead_encounter': k, 'ahead_lon_m': round(lon - front, 1),
                 'ahead_lat_m': round(lat, 2), 'ahead_speed_mps': round(float(o.get('speed') or 0.0), 2)}
    # 가장 가까운 조우 물체까지 (시나리오 장애물 뒤에 선 것인가)
    near = None
    b = 0
    for k, e in enumerate(enc):
        for j, _o in enumerate(e['objects']):
            xy = pts[b + j]
            if xy is not None:
                lon, lat = ego_frame(eg, xy[0], xy[1])
                d = math.hypot(xy[0] - eg['x'], xy[1] - eg['y'])
                if near is None or d < near[0]:
                    half = (matched or {}).get(b + j, (None, 0.0))[1] / 2.0
                    near = (d, lon, lat, k, lon - front - half)
        b += len(e['objects'])
    # 마지막 정지 구간에서 가장 오래 유지된 회피 진단 (상태, 기각 사유) — "왜 멈췄나"
    cnt = collections.Counter()
    for t in win:
        a = ((t.get('decision') or {}).get('reasons') or {}).get('avoid') or {}
        cnt[(a.get('state'), a.get('reject') or a.get('suppress'))] += 1
    stop_state, stop_reject = cnt.most_common(1)[0][0] if cnt else (None, None)
    reasons = (last.get('decision') or {}).get('reasons') or {}
    avoid = reasons.get('avoid') or {}
    esc = reasons.get('escape') or {}
    red = bool(light) and int(light[1]) in HOLD
    veh_ahead = bool(ahead) and ahead['ahead_kind'] != 'event' and ahead['ahead_cls'] == 'vehicle'
    return {
        't_end_s': round(last['t'] - t0, 1), 'route_s': round(float(eg['route_s']), 1),
        'stopped_s': round(win[-1]['t'] - win[0]['t'], 1),
        'v_target': round(float((last.get('decision') or {}).get('v_target') or 0.0), 2),
        'winner': reasons.get('winner'),
        'light_id': light[0] if light else None,
        'light_state': LIGHT_NAME.get(int(light[1]), light[1]) if light else None,
        'light_hold_frac': round(hold / len(win), 2),
        'dist_stop_line_m': None if summ.get('dist_stop_line') is None else round(float(summ['dist_stop_line']), 1),
        'in_junction': summ.get('in_junction'),
        'ahead_present': bool(ahead), **ahead,
        'nearest_encounter': None if near is None else near[3],
        'nearest_encounter_dist_m': None if near is None else round(near[0], 1),
        'nearest_encounter_lon_m': None if near is None else round(near[1], 1),
        # 자차 앞범퍼 ~ 그 장애물 뒤끝 (음수 = 이미 지나갔거나 겹침)
        'nearest_encounter_gap_m': None if near is None else round(near[4], 1),
        'stop_avoid_state': stop_state, 'stop_avoid_reject': stop_reject,
        'avoid_state': avoid.get('state'), 'avoid_reject': avoid.get('reject'),
        'avoid_rejects': ';'.join(avoid.get('rejects') or []) or None,
        'escape_state': esc.get('state'),
        'blocked_why': run['res'].get('blocked_why'),
        # 사람이 먼저 볼 줄: 정지 신호를 마주한 채 시나리오 장애물이 아닌 차량 뒤에 섰다.
        # 판정이 아니라 **검토 표시**다 (대기행렬은 면책 사유가 아니다 — 표 2 에는 그대로 센다).
        'queue_suspect': bool(red and veh_ahead)}


def contact_onset(span: list, oid: int, veh: dict, thr: float, t0: float) -> dict:
    """충돌 구간에서 처음 간격이 임계 이하가 된 틱의 상태 — 누가 누구에게 들어왔나.

    자차 속도 · 상대 속도 · 상대 진행각(자차 기준)을 남긴다. 실패 원인 분류
    (failure_cause) 가 "움직이는 스크립트 차량이 거의 선 자차로 꺾여 들어왔다" 를 가른다.
    """
    L, W = float(veh['length']), float(veh['width'])
    c_off = L / 2.0 - (L - float(veh['wheelbase']) - float(veh['front_overhang_m']))
    first = None
    for t in span:
        e = t['ego']
        rect = score._corners(e['x'] + c_off * math.cos(e['yaw']), e['y'] + c_off * math.sin(e['yaw']),
                              e['yaw'], L, W)
        for o in t['raw']['objects']:
            if int(o[0]) != oid:
                continue
            g = score._rect_gap(rect, score._corners(o[1], o[2], o[4], o[6] or 1.0, o[7] or 1.0))
            if first is None or (g <= thr and first[0] > thr):
                first = (g, t, o)
        if first is not None and first[0] <= thr:
            break
    if first is None:
        return {'onset_t': None, 'ego_v_mps': None, 'obj_v_mps': None, 'obj_dyaw_deg': None}
    _g, t, o = first
    dyaw = (float(o[4]) - float(t['ego']['yaw']) + math.pi) % (2 * math.pi) - math.pi
    return {'onset_t': round(t['t'] - t0, 1), 'ego_v_mps': round(float(t['ego']['speed']), 2),
            'obj_v_mps': round(abs(float(o[5])), 2), 'obj_dyaw_deg': round(math.degrees(dyaw), 1)}


def shift_events(ticks: list, t0: float, cfg: dict, kind_of: dict, upto: int | None = None) -> list:
    """회피 시프트를 **만든** 틱들 — 대상 객체가 그 뒤 다시 움직였는가를 붙인다.

    upto: 이 틱 앞에서 만든 시프트만 센다 (종료선 이후 제외). 대상이 다시 움직였는지는
    그 뒤 틱까지 본다.
    """
    thr = float((cfg.get('ctrl24') or {}).get('blocker_speed_max',
                                              (cfg.get('percep') or {}).get('obstacle_speed_max', 0.5)))
    made, seen = [], set()
    for i, t in enumerate(ticks[:upto]):
        a = ((t.get('decision') or {}).get('reasons') or {}).get('avoid') or {}
        if a.get('shift') not in ('left', 'right') or a.get('blocker') is None:
            continue
        span = tuple(a.get('span_new') or a.get('span') or ())
        key = (int(a['blocker']), span)
        if key in seen:                       # 같은 시프트가 여러 틱에 찍힌 로그 (구 스키마)
            continue
        seen.add(key)
        made.append((i, t, a))
    out = []
    for i, t, a in made:
        oid = int(a['blocker'])
        v_at, v_max, t_move = None, 0.0, None
        for u in ticks[i:]:
            for o in u['raw']['objects']:
                if int(o[0]) != oid:
                    continue
                v = abs(float(o[5]))
                if v_at is None:
                    v_at = v
                if v > v_max:
                    v_max = v
                if t_move is None and v > thr:
                    t_move = round(u['t'] - t0, 2)
        out.append({'tick': i, 't': round(t['t'] - t0, 2), 'route_s': round(float(t['ego']['route_s']), 1),
                    'state': a.get('state'), 'side': a.get('shift'), 'blocker_id': oid,
                    'chain_ids': ';'.join(str(x) for x in (a.get('chain') or [])) or None,
                    's_rel_m': a.get('s_rel'), 'trigger': a.get('trigger'),
                    'blocker_kind': kind_of.get(oid, ('other', None))[0],
                    'light_state': (lambda l: LIGHT_NAME.get(int(l[1]), l[1]) if l else None)(
                        (t.get('world') or {}).get('light')),
                    'blocker_speed_at_shift': None if v_at is None else round(v_at, 2),
                    'blocker_speed_max_after': round(v_max, 2), 'moved_threshold_mps': thr,
                    't_moved_s': t_move, 'false_positive': bool(v_max > thr)})
    return out


# ════════════════════════════════════════════════════════════════════════
# 유형별 성공 기준
# ════════════════════════════════════════════════════════════════════════
CLASS_KO = {'avoid': '회피', 'brake': '오탐·급정거', 'queue': '오탐·신호 대기 줄', 'plain': '일반 주행'}


def judge(run: dict) -> tuple:
    """런 하나의 성공 여부 → (True/False/None, 사유). None = 판정에서 뺀다 (따로 센다).

    avoid  모든 장애물을 지나감 + 충돌 없음 + 무한 정지(stall·blocked) 없음
    queue  적색 동안 대기 차량에 시프트를 만들지 않음 + 충돌 없음.
           대기 차량이 끝까지 안 풀려 생긴 정지, 자차가 줄에 닿기 전에 줄이 풀린 런은 뺀다.
    brake  다시 움직인 앞차에 시프트를 만들지 않음 + 충돌 없음 + 무한 정지 없음
    plain  완주
    """
    an, st = run['an'], run['status']
    cls = (run.get('scen') or {}).get('class')
    if st not in DONE or cls is None:
        return None, 'invalid'
    col = len(an['collisions'])
    if cls == 'avoid':
        if st in INF_STOP:
            return False, f'inf_stop({st})'
        if col:
            return False, 'collision'
        if not all(e['outcome'] == 'success' for e in an['encounters']):
            return False, f'not_passed({st})'
        return True, 'ok'
    if cls == 'queue':
        q = an.get('queue') or {}
        if q.get('shifts_on_queue'):
            return False, 'false_positive_shift'
        if col:
            return False, 'collision'
        if st in INF_STOP and q.get('t_release_s') is None:
            return None, 'queue_not_released'
        if not q.get('met_before_release'):
            return None, 'queue_not_met'
        if st in INF_STOP:
            return False, f'inf_stop({st})'
        return True, 'ok'
    if cls == 'brake':
        if any(s['false_positive'] for s in an['shifts']):
            return False, 'false_positive_shift'
        if col:
            return False, 'collision'
        if st in INF_STOP:
            return False, f'inf_stop({st})'
        return True, 'ok'
    return (True, 'ok') if st == '완주' else (False, st)


# 실패 원인 분류 (2026-10-03). 규칙은 로그에서만 읽는다 — 시나리오 이름을 적지 않는다.
CAUSE_KO = {'controller': '제어기 판단', 'false_positive': '오탐', 'scenario': '시나리오',
            'simulator': '시뮬레이터'}
# "움직이는 스크립트 차량이 거의 선 자차로 꺾여 들어왔다" — 접촉 시작 틱에서 상대가 이 속도
# 이상으로 이 각도 이상 틀어 움직이고, 자차는 이 속도 이하. VTD 스크립트 차량(대기 줄 · 옆 차로
# 교통)은 자차에 반응하지 않으므로, 그 경로가 자차로를 지나는 것은 시나리오 쪽 문제다.
# 실측 (WON24 2026-10-02): 신호줄 02·03·06 은 상대 3.7–7.4 m/s · 23–32° · 자차 0.4–1.2 m/s.
# 자차가 옆 차로로 들어가 붙은 경우(논문6_03)는 상대 진행각 −5~−8° 라 걸리지 않는다.
INTRUDE = {'obj_v_mps': 1.0, 'obj_dyaw_deg': 15.0, 'ego_v_mps': 1.5}


def builders_differ(scen: dict | None, tol: float) -> str | None:
    """두 브랜치 빌더의 경로가 다른 길인가 → 사유 문자열 (같으면 None, 기록이 없으면 None).

    다른 길 = 장애물 차로를 어느 한쪽 경로가 지나지 않음 · 비교 빌더 실패 · 길이 차 ≥ tol (비율).
    """
    b = (scen or {}).get('builders')
    if not b:
        return None
    if b['other_rc'] not in (0, None):
        return f'{b["other"]} 빌더 실패 (rc={b["other_rc"]})'
    if b['off_lane_this'] or b['off_lane_other']:
        return f'장애물 차로를 지나지 않음 (이 브랜치 {b["off_lane_this"]}, {b["other"]} {b["off_lane_other"]})'
    lt, lo = b['len_this_m'], b['len_other_m']
    if lt and lo and abs(lo - lt) / lt >= tol:
        return f'경로 길이 {lt:.1f} m vs {b["other"]} {lo:.1f} m ({100 * abs(lo - lt) / lt:.1f} %)'
    return None


def failure_cause(run: dict) -> tuple:
    """실패 런 (success False) 또는 기반 실패 런 → (원인, 근거). 그 밖의 런은 (None, None).

    순서대로 처음 맞는 것:
      simulator       기반 실패 (SCP · VTD 무응답 · no-data … — 제어기가 돌지 않았다)
      false_positive  오탐 시프트 (다시 움직일 대상 — 대기 줄 · 급정거 앞차 — 에 시프트)
      scenario        충돌이 전부 종료선 콘이거나, 전부 '움직이는 차량이 거의 선 자차로 들어옴'
      controller      나머지 (충돌 · 못 지나감 · 무한 정지)
    """
    st = run['status']
    if st not in DONE:
        return 'simulator', f'기반 실패: {st}'
    if run['success'] is not False:
        return None, None
    an = run['an']
    if run['why'] == 'false_positive_shift':
        q = (an.get('queue') or {}).get('shifts_on_queue') or [s for s in an['shifts'] if s['false_positive']]
        s = q[0] if q else {}
        return 'false_positive', (f'다시 움직인 {s.get("blocker_kind")} 객체에 시프트 '
                                  f'(t={s.get("t")} s, 앞 {s.get("s_rel_m")} m, 신호 {s.get("light_state")})')
    cols = an['collisions']
    if cols and all(c['obj_kind'] == 'finish_cone' for c in cols):
        return 'scenario', '종료선 콘 접촉 (콘이 회피 회랑 안)'
    if cols and all(c.get('obj_v_mps') is not None and c['obj_v_mps'] >= INTRUDE['obj_v_mps']
                    and abs(c['obj_dyaw_deg']) >= INTRUDE['obj_dyaw_deg']
                    and c['ego_v_mps'] <= INTRUDE['ego_v_mps'] for c in cols):
        c = cols[0]
        return 'scenario', (f'움직이는 {c["obj_kind"]} 차량({c["obj_v_mps"]} m/s, 진행각 {c["obj_dyaw_deg"]:+.0f}°)이 '
                            f'거의 선 자차({c["ego_v_mps"]} m/s)로 들어옴')
    if cols:
        c = cols[0]
        return 'controller', (f'{c["obj_kind"]} 와 충돌 (접촉 시 자차 {c["ego_v_mps"]} m/s, '
                              f'상대 {c["obj_v_mps"]} m/s, 최소 간격 {c["min_gap_m"]} m)')
    sn = an.get('snapshot') or {}
    if run['why'].startswith('not_passed'):
        return 'controller', f'장애물을 다 지나지 못함 (끝: {sn.get("stop_avoid_state")}/{sn.get("stop_avoid_reject")})'
    return 'controller', f'{run["why"]} (정지 중 회피 {sn.get("stop_avoid_state")}/{sn.get("stop_avoid_reject")})'


def success_row(runs: list) -> dict:
    """판정 대상 런 묶음 → 성공률과 Wilson 구간."""
    j = [r for r in runs if r['success'] is not None]
    n, k = len(j), sum(1 for r in j if r['success'])
    lo, hi = wilson(k, n)
    return {'n_judged': n, 'n_excluded': sum(1 for r in runs if r['status'] in DONE and r['success'] is None),
            'success': k, 'success_pct': round(100.0 * k / n, 1) if n else None,
            'success_lo_pct': None if math.isnan(lo) else round(100.0 * lo, 1),
            'success_hi_pct': None if math.isnan(hi) else round(100.0 * hi, 1)}


# ════════════════════════════════════════════════════════════════════════
# 집계
# ════════════════════════════════════════════════════════════════════════
def rate_row(runs: list) -> dict:
    """유효 런 묶음 → 완주율·무한 정지·timeout·충돌."""
    valid = [r for r in runs if r['status'] in DONE]
    n = len(valid)
    k = sum(1 for r in valid if r['status'] == '완주')
    lo, hi = wilson(k, n)
    cols = [c for r in valid for c in r['an']['collisions']]
    return {
        'n_valid': n, 'n_invalid': len(runs) - n,
        'completed': k,
        'completion_pct': round(100.0 * k / n, 1) if n else None,
        'wilson95_lo_pct': None if math.isnan(lo) else round(100.0 * lo, 1),
        'wilson95_hi_pct': None if math.isnan(hi) else round(100.0 * hi, 1),
        'inf_stop': sum(1 for r in valid if r['status'] in INF_STOP),
        'stall': sum(1 for r in valid if r['status'] == 'stall'),
        'blocked': sum(1 for r in valid if r['status'] == 'blocked'),
        'timeout': sum(1 for r in valid if r['status'] == 'timeout'),
        'no_progress': sum(1 for r in valid if r['status'] == 'no_progress'),
        # 충돌 = 채점기(score.py 항목 11)의 검출기로 로그를 다시 센 건수 (모든 구현에 같은 자).
        'collisions': sum(r['n_col'] for r in valid),
        'runs_with_collision': sum(1 for r in valid if r['n_col'] > 0),
        'collisions_event_obstacle': sum(1 for c in cols if c['obj_kind'] == 'event'),
        'collisions_finish_cone': sum(1 for c in cols if c['obj_kind'] == 'finish_cone'),
        'collisions_other': sum(1 for c in cols if c['obj_kind'] == 'other'),
        'min_gap_m': min((c['min_gap_m'] for c in cols), default=None)}


def write_csv(path: pathlib.Path, rows: list, fields: list | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r.get(k) is None else r.get(k)) for k in fields})


def md_table(rows: list, fields: list, heads: list) -> str:
    line = lambda cells: '| ' + ' | '.join(str(c) for c in cells) + ' |'      # noqa: E731
    out = [line(heads), line(['---'] * len(heads))]
    for r in rows:
        out.append(line(['' if r.get(k) is None else r.get(k) for k in fields]))
    return '\n'.join(out) + '\n'


# ════════════════════════════════════════════════════════════════════════
# 그림 (글자는 영문 — 원천 CSV 를 같은 이름으로 남긴다)
# ════════════════════════════════════════════════════════════════════════
def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def fig_table2(rows, lab, fdir):
    plt = _plt()
    src = [{'arm': r['arm'], 'label': lab(r['arm']), 'n': r['n_valid'],
            'completion_pct': r['completion_pct'], 'wilson95_lo_pct': r['wilson95_lo_pct'],
            'wilson95_hi_pct': r['wilson95_hi_pct'], 'inf_stop': r['inf_stop'],
            'timeout': r['timeout'], 'collisions': r['collisions']} for r in rows if r['n_valid']]
    write_csv(fdir / 'fig_table2.csv', src)
    if not src:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    x = range(len(src))
    y = [s['completion_pct'] for s in src]
    err = [[s['completion_pct'] - s['wilson95_lo_pct'] for s in src],
           [s['wilson95_hi_pct'] - s['completion_pct'] for s in src]]
    ax[0].bar(x, y, yerr=err, capsize=5, color='#4c72b0')
    ax[0].set_xticks(list(x)); ax[0].set_xticklabels([s['arm'] for s in src])
    ax[0].set_ylim(0, 105); ax[0].set_ylabel('Completion rate [%]')
    ax[0].set_title('Completion rate (95% Wilson interval)')
    w = 0.27
    for j, (key, name, colr) in enumerate((('inf_stop', 'Indefinite stop', '#c44e52'),
                                           ('timeout', 'Timeout', '#8172b2'),
                                           ('collisions', 'Collision', '#dd8452'))):
        ax[1].bar([i + (j - 1) * w for i in x], [s[key] for s in src], width=w, label=name, color=colr)
    ax[1].set_xticks(list(x)); ax[1].set_xticklabels([s['arm'] for s in src])
    ax[1].set_ylabel('Count'); ax[1].legend(); ax[1].set_title('Failure events')
    fig.text(0.5, 0.01, '   '.join(f'{s["arm"]}: {s["label"]} (n={s["n"]})' for s in src),
             ha='center', fontsize=7)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(fdir / 'fig_table2.png', dpi=200); plt.close(fig)


def fig_by_type(rows, lab, tlab, fdir):
    """유형 × 구현 성공률 (유형별 성공 기준, 95% Wilson)."""
    plt = _plt()
    src = [{'type': r['type'], 'type_label': tlab(r['type']), 'class': r['class'], 'arm': r['arm'],
            'label': lab(r['arm']), 'n': r['n_judged'], 'success': r['success'],
            'success_pct': r['success_pct'], 'wilson95_lo_pct': r['success_lo_pct'],
            'wilson95_hi_pct': r['success_hi_pct']} for r in rows if r['n_judged']]
    write_csv(fdir / 'fig_by_type.csv', src)
    if not src:
        return
    types = list(dict.fromkeys(s['type'] for s in src))
    arms = list(dict.fromkeys(s['arm'] for s in src))
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(types) + 2), 4.5))
    w = 0.8 / len(arms)
    for j, a in enumerate(arms):
        xs, ys, lo, hi = [], [], [], []
        for i, t in enumerate(types):
            s = next((s for s in src if s['type'] == t and s['arm'] == a), None)
            if s is None:
                continue
            xs.append(i + (j - (len(arms) - 1) / 2) * w); ys.append(s['success_pct'])
            lo.append(s['success_pct'] - s['wilson95_lo_pct']); hi.append(s['wilson95_hi_pct'] - s['success_pct'])
        ax.bar(xs, ys, width=w, yerr=[lo, hi], capsize=3, label=f'{a}: {lab(a)}')
    ax.set_xticks(range(len(types)))
    ax.set_xticklabels([tlab(t) for t in types], fontsize=7, rotation=20, ha='right')
    ax.set_ylim(0, 105); ax.set_ylabel('Success rate [%]'); ax.legend(fontsize=7)
    ax.set_title('Success rate by scenario type (95% Wilson interval)', fontsize=9)
    fig.tight_layout(); fig.savefig(fdir / 'fig_by_type.png', dpi=200); plt.close(fig)


def fig_condition(bc, spec, lab, tlab, fdir):
    """조건 축 값별 성공률 — arms.yaml report.condition_figures 의 한 항목.
    spec = {name, types: [...], axis, xlabel, value_labels_en: {값: 표기}}"""
    plt = _plt()
    vl = spec.get('value_labels_en') or {}
    src = [{'type': r['type'], 'type_label': tlab(r['type']), 'axis': r['axis'], 'value': r['value'],
            'value_label': vl.get(r['value'], r['value']), 'arm': r['arm'], 'label': lab(r['arm']),
            'n': r['n_judged'], 'success': r['success'], 'success_pct': r['success_pct'],
            'wilson95_lo_pct': r['success_lo_pct'], 'wilson95_hi_pct': r['success_hi_pct']}
           for r in bc if r['type'] in spec['types'] and r['axis'] == spec['axis'] and r['n_judged']]
    write_csv(fdir / f'{spec["name"]}.csv', src)
    if not src:
        return
    types = [t for t in spec['types'] if any(s['type'] == t for s in src)]
    fig, axes = plt.subplots(1, len(types), figsize=(4.6 * len(types), 3.8), squeeze=False)
    for ax, t in zip(axes[0], types):
        ss = [s for s in src if s['type'] == t]
        vals = list(dict.fromkeys(s['value'] for s in ss))
        try:
            vals = sorted(vals, key=float)
        except (TypeError, ValueError):
            pass
        arms = list(dict.fromkeys(s['arm'] for s in ss))
        w = 0.8 / len(arms)
        for j, a in enumerate(arms):
            xs, ys, lo, hi = [], [], [], []
            for i, v in enumerate(vals):
                s = next((s for s in ss if s['value'] == v and s['arm'] == a), None)
                if s is None:
                    continue
                xs.append(i + (j - (len(arms) - 1) / 2) * w); ys.append(s['success_pct'])
                lo.append(s['success_pct'] - s['wilson95_lo_pct']); hi.append(s['wilson95_hi_pct'] - s['success_pct'])
            ax.bar(xs, ys, width=w, yerr=[lo, hi], capsize=3, label=f'{a}: {lab(a)}')
        ax.set_xticks(range(len(vals))); ax.set_xticklabels([str(vl.get(v, v)) for v in vals])
        ax.set_ylim(0, 105); ax.set_xlabel(spec.get('xlabel', spec['axis']))
        ax.set_ylabel('Success rate [%]'); ax.set_title(tlab(t), fontsize=9)
        n = ', '.join(sorted({f'n={s["n"]}' for s in ss}))
        ax.text(0.99, 0.02, n, transform=ax.transAxes, ha='right', fontsize=6)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fdir / f'{spec["name"]}.png', dpi=200); plt.close(fig)


END_EN = {'완주': 'Completed', 'blocked': 'Stop (blocked)', 'stall': 'Stop (stall)',
          'timeout': 'Timeout', 'no_progress': 'No progress'}


def fig_end_reason(runs, arms, lab, fdir):
    plt = _plt()
    src = []
    for a in arms:
        cnt = collections.Counter(r['status'] if r['status'] in DONE else 'invalid'
                                  for r in runs if r['arm'] == a)
        for st in list(DONE) + ['invalid']:
            src.append({'arm': a, 'label': lab(a), 'end_reason': st,
                        'end_reason_en': END_EN.get(st, 'Invalid (infrastructure)'), 'count': cnt.get(st, 0)})
    write_csv(fdir / 'fig_end_reason.csv', src)
    if not any(s['count'] for s in src):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    bottom = [0] * len(arms)
    for st in list(DONE) + ['invalid']:
        ys = [next(s['count'] for s in src if s['arm'] == a and s['end_reason'] == st) for a in arms]
        if not any(ys):
            continue
        ax.bar(range(len(arms)), ys, bottom=bottom, label=END_EN.get(st, 'Invalid (infrastructure)'))
        bottom = [b + y for b, y in zip(bottom, ys)]
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms)
    ax.set_ylabel('Runs'); ax.set_title('End reason by implementation'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fdir / 'fig_end_reason.png', dpi=200); plt.close(fig)


def trace_rows(run: dict, scen: dict, near_m: float) -> list:
    """시간–속도·첫 장애물까지 거리. 거리는 직선거리이고, 장애물을 지난 뒤로는 음수다
    (near_m 안에서 장애물이 자차 뒤로 넘어간 틱부터)."""
    ticks = score.load_ticks(str(run['log']))
    xy = scen['encounters'][0]['objects'][0]['xy']
    t0 = ticks[0]['t']
    out, sign = [], 1.0
    for t in ticks:
        eg = t['ego']
        d = math.hypot(xy[0] - eg['x'], xy[1] - eg['y'])
        if sign > 0 and d <= near_m and ego_frame(eg, xy[0], xy[1])[0] < 0.0:
            sign = -1.0
        out.append({'arm': run['arm'], 't_s': round(t['t'] - t0, 2), 'speed_mps': round(float(eg['speed']), 3),
                    'dist_to_obstacle_m': round(sign * d, 2)})
    return out


def fig_speed_trace(pairs, lab, tlab, fdir, near_m):
    plt = _plt()
    for other, ref, scen, why in pairs:
        rows = trace_rows(other, scen, near_m) + trace_rows(ref, scen, near_m)
        stem = f'fig_speed_trace_{other["arm"]}_vs_{ref["arm"]}'
        for r in rows:
            r.update(scenario=other['name'], type=other['type'])
        write_csv(fdir / f'{stem}.csv', rows, ['scenario', 'type', 'arm', 't_s', 'speed_mps', 'dist_to_obstacle_m'])
        fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
        for run in (other, ref):
            rr = [r for r in rows if r['arm'] == run['arm']]
            ax[0].plot([r['t_s'] for r in rr], [r['speed_mps'] for r in rr], label=f'{run["arm"]}: {lab(run["arm"])}')
            ax[1].plot([r['t_s'] for r in rr], [r['dist_to_obstacle_m'] for r in rr], label=run['arm'])
        ax[0].set_ylabel('Speed [m/s]')
        ax[1].axhline(0, color='k', lw=0.5)
        ax[1].set_ylabel('Distance to obstacle [m]\n(+ ahead, − passed)'); ax[1].set_xlabel('Time [s]')
        ax[0].set_title(f'{tlab(other["type"])} — same scenario ({why})', fontsize=9)
        h, l = ax[0].get_legend_handles_labels()
        fig.legend(h, l, loc='lower center', ncol=1, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(fdir / f'{stem}.png', dpi=200); plt.close(fig)


def fig_brake_fp(summary, lab, fdir):
    plt = _plt()
    write_csv(fdir / 'fig_brake_fp.csv', summary)
    src = [s for s in summary if s['runs']]
    if not src:
        return
    fig, ax = plt.subplots(figsize=(5, 4))
    x = range(len(src)); w = 0.38
    ax.bar([i - w / 2 for i in x], [s['shifts'] for s in src], width=w, label='Avoidance shifts created')
    ax.bar([i + w / 2 for i in x], [s['false_positive_shifts'] for s in src], width=w,
           label='... on a lead vehicle that moved again')
    ax.set_xticks(list(x)); ax.set_xticklabels([f'{s["arm"]} (n={s["runs"]})' for s in src])
    ax.set_ylabel('Count'); ax.set_title('Lead hard-braking scenarios'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fdir / 'fig_brake_fp.png', dpi=200); plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════
def _kv(items, what):
    out = {}
    for it in items or []:
        if '=' not in it:
            raise SystemExit(f'{what}: "이름=값" 형식이어야 한다 — {it}')
        k, v = it.split('=', 1)
        out[k] = v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='논문 Ⅳ장 집계 — 구현별 로그 폴더 → 표·CSV·그림')
    ap.add_argument('--arm', action='append', required=True, metavar='이름=폴더',
                    help='구현 이름 = 로그 폴더 (<폴더>/<유형>/report.json). 여러 번 준다')
    ap.add_argument('--scenarios', default=str(ROOT / 'scenarios'), help='시나리오 정의 폴더')
    ap.add_argument('--out', default=str(ROOT / 'logs' / 'paper' / 'report'))
    ap.add_argument('--arms-yaml', default=str(ROOT / 'config' / 'arms' / 'arms.yaml'),
                    help='구현 이름·기준 구현·영문 유형 이름의 출처')
    ap.add_argument('--ref', default=None, help='짝 비교의 기준 구현 (기본: arms.yaml report.reference)')
    ap.add_argument('--label-ko', action='append', metavar='이름=표기', help='arms.yaml 에 없는 구현의 이름')
    ap.add_argument('--label-en', action='append', metavar='이름=표기')
    ap.add_argument('--trace', default=None, help='시간–속도 그래프에 쓸 시나리오 이름 (기본: 자동 선택)')
    ap.add_argument('--no-figures', action='store_true')
    ap.add_argument('--exclude', action='append', default=[], metavar='시나리오',
                    help='모든 구현에서 뺄 시나리오 이름 (쉼표·반복 가능). 구현끼리 조건이 같지 않은 '
                         '시나리오를 뺄 때 쓴다 — meta.json 에 남는다')
    # 분석 임계 — 결과 수치가 아니라 "무엇을 같은 물체·지나감으로 볼 것인가" 의 정의다. meta.json 에 남긴다.
    ap.add_argument('--match-m', type=float, default=1.5, help='시나리오 물체 ↔ 로그 객체 좌표 허용 오차 [m]')
    ap.add_argument('--pass-near-m', type=float, default=15.0, help='지나감 판정을 하는 거리 [m]')
    # 2026-10-03: 6 → 8 m. 나란히 유형은 열린 쪽으로 지나가면 먼 쪽 차가 두 차로 건너
    # (실측 6.2–6.6 m) 에 있어, 6 m 로는 실제로 지나간 런 4건이 "못 지나감" 으로 잡혔다.
    # 8 m 는 세 차로 건너(≈ 9.6 m) · 다른 길의 물체는 여전히 뺀다. 6.0 을 주면 이전 집계.
    ap.add_argument('--pass-lat-m', type=float, default=8.0, help='지나감 판정의 횡거리 한계 [m]')
    ap.add_argument('--builder-len-tol', type=float, default=0.05,
                    help='구현 2개 이상일 때 두 빌더 경로 길이 차가 이 비율 이상이면 비교에서 뺀다 (기본 0.05)')
    ap.add_argument('--after-finish', choices=('exclude', 'include'), default='exclude',
                    help='종료선(report.json finish_thr_m) 을 지난 뒤 시작한 충돌·시프트를 셀지. '
                         '기본 exclude (2026-10-03 집계 기준). include = 이전 집계')
    ap.add_argument('--detect-m', type=float, default=80.0, help='조우 도달 거리 — 설정에 ctrl24.detect_max_m 가 없을 때만')
    ap.add_argument('--ahead-m', type=float, default=30.0, help='정지 검토: 앞 물체를 찾는 범퍼 앞 거리 [m]')
    ap.add_argument('--ahead-lat-m', type=float, default=2.0, help='정지 검토: 앞 물체의 횡거리 한계 [m]')
    a = ap.parse_args(argv)

    arm_dirs = {k: pathlib.Path(v) for k, v in _kv(a.arm, '--arm').items()}
    scen_dir = pathlib.Path(a.scenarios)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    defs = {}
    if pathlib.Path(a.arms_yaml).exists():
        defs = yaml.safe_load(pathlib.Path(a.arms_yaml).read_text(encoding='utf-8')) or {}
    rep_cfg = defs.get('report') or {}
    # 이름: arms.yaml 의 arms(여기서 돌린 구현) + report.labels(다른 브랜치에서 온 구현)
    named = {**(rep_cfg.get('labels') or {}), **(defs.get('arms') or {})}
    ko = {k: v.get('label_ko') for k, v in named.items()}
    en = {k: v.get('label_en') for k, v in named.items()}
    ko.update(_kv(a.label_ko, '--label-ko')); en.update(_kv(a.label_en, '--label-en'))
    lab_ko = lambda n: ko.get(n) or n                     # noqa: E731
    lab_en = lambda n: en.get(n) or n                     # noqa: E731
    tlab = lambda t: (rep_cfg.get('type_labels_en') or {}).get(t, t)   # noqa: E731
    ref = a.ref or rep_cfg.get('reference')
    arms = list(arm_dirs)

    # ── 읽기 + 런 분석 ──────────────────────────────────────────────────
    runs, scen_cache = [], {}
    for arm, folder in arm_dirs.items():
        if not folder.is_dir():
            print(f'  [주의] {arm}: 폴더가 없다 — {folder}')
            continue
        runs += load_arm(arm, folder, scen_dir)
    excluded = sorted({n for it in a.exclude for n in it.split(',') if n})
    for r in runs:
        key = str(r['yaml'])
        if key not in scen_cache:
            scen_cache[key] = parse_scenario(r['yaml'])
        r['scen'] = scen_cache[key]
    # 브랜치 비교(구현 2개 이상)에서는 두 빌더의 경로가 **다른 길**인 시나리오를 자동으로 뺀다
    # (생성기가 남긴 route_quality 기록 기준, builders_differ). 2026-10-03 결정 — won 결과를 보기
    # 전에 확정한 규칙: 장애물 차로를 어느 한쪽 빌더가 지나지 않거나, 경로 길이 차가 5 % 이상.
    # 차로열이 조금 다른 것(길이 차 5–8 m, 장애물 차로는 같음)은 같은 조우를 달리므로 남긴다.
    auto_excl, auto_why = [], {}
    if len({r['arm'] for r in runs}) >= 2:
        for r in runs:
            why = builders_differ(r['scen'], a.builder_len_tol)
            if why and r['name'] not in excluded:
                auto_why[r['name']] = why
        auto_excl = sorted(auto_why)
        excluded = sorted(set(excluded) | set(auto_excl))
    n_before = len(runs)
    runs = [r for r in runs if r['name'] not in excluded]
    for r in runs:
        r['an'] = analyze_run(r, r['scen'], a) if r['status'] in DONE else \
            {'collisions': [], 'collisions_after_finish': [], 'finish': None, 'encounters': [],
             'review': None, 'snapshot': None, 'shifts': [],
             'queue': None, 'n_ticks': 0, 'dur_s': None, 'warn': ['기반 실패 — 분석 제외']}
        # 충돌 수의 정본은 **로그를 이 저장소의 채점기로 다시 센 값**이다 — 다른 브랜치에서
        # 돌린 구현은 그 브랜치의 score.py·임계로 report.json 을 썼을 수 있어, 구현끼리
        # 같은 자로 재려면 한 검출기로 다시 세야 한다. report.json 값은 대조용으로만 둔다
        # (batch_run 은 0 이 아닌 위반만 싣는다 — 키가 없으면 0 건).
        vio = r['res'].get('violations')
        r['n_col_report'] = None if vio is None else int(vio.get('collision') or 0)
        r['n_col'] = len(r['an']['collisions'])
        r['success'], r['why'] = judge(r)
        r['cause'], r['cause_evidence'] = failure_cause(r)
        # report.json 은 종료선 이후 접촉도 센다 — 대조는 뺀 것까지 더한 수로 한다.
        n_all = len(r['an']['collisions']) + len(r['an'].get('collisions_after_finish') or [])
        if r['n_col_report'] is not None and r['n_col_report'] != n_all \
                and r['status'] in DONE and r['log'] is not None:
            r['an']['warn'].append(f'충돌 수 불일치: report {r["n_col_report"]} / '
                                   f'로그 재계산 {n_all} (종료선 이후 포함)')

    # 유형 분류 — 시나리오 정의에서 (이름을 코드에 적지 않는다)
    types = list(dict.fromkeys(r['type'] for r in runs))
    type_class = {}
    for t in types:
        cs = collections.Counter(r['scen']['class'] for r in runs if r['type'] == t and r['scen'])
        type_class[t] = cs.most_common(1)[0][0] if cs else None
    avoid_types = [t for t in types if type_class[t] == 'avoid']
    brake_types = [t for t in types if type_class[t] == 'brake']
    queue_types = [t for t in types if type_class[t] == 'queue']
    plain_types = [t for t in types if type_class[t] == 'plain']
    arun = [r for r in runs if r['type'] in avoid_types]

    # 기대 런 수 (목록 파일) — 중간 집계에서 얼마나 찼는지 보이게
    def expected(t):
        p = scen_dir / f'batch_{t}.json'
        if not p.exists():
            return None
        return sum(1 for i in json.loads(p.read_text(encoding='utf-8')) if i.get('name') not in excluded)

    # ── runs.csv ────────────────────────────────────────────────────────
    run_rows = []
    for r in runs:
        res = r['res']
        run_rows.append({
            'arm': r['arm'], 'type': r['type'], 'scenario': r['name'], 'status': r['status'],
            'class': type_class.get(r['type']), 'success': r['success'], 'why': r['why'],
            'conditions': json.dumps({k: v for k, v in ((r['scen'] or {}).get('axes') or {}).items()
                                      if k not in ('route', 'event')}, ensure_ascii=False),
            'time_s': res.get('time_s'),
            'valid': r['status'] in DONE, 'ran_s': res.get('ran_s'), 'dist_m': res.get('dist_m'),
            'route_total_m': res.get('route_total_m'), 'collisions': r['n_col'],
            'collisions_report': r['n_col_report'],
            'min_gap_m': min((c['min_gap_m'] for c in r['an']['collisions']), default=None),
            'encounters': len(r['an']['encounters']),
            'encounters_success': sum(1 for e in r['an']['encounters'] if e['outcome'] == 'success'),
            'shifts': len(r['an']['shifts']), 'deduction': res.get('deduction'),
            'config': res.get('config'), 'arm_in_report': res.get('arm'),
            'log': str(r['log']) if r['log'] else None, 'warn': '; '.join(r['an']['warn']) or None})
    write_csv(out / 'runs.csv', run_rows)

    # ── table2 ──────────────────────────────────────────────────────────
    t2 = []
    for arm in arms:
        rr = [r for r in arun if r['arm'] == arm]
        exp = sum(expected(t) or 0 for t in avoid_types)
        enc = [e for r in rr if r['status'] in DONE for e in r['an']['encounters']]
        t2.append({'arm': arm, 'label': lab_ko(arm), 'n_expected': exp, **rate_row(rr),
                   **success_row(rr), 'encounters': len(enc),
                   'encounters_success': sum(1 for e in enc if e['outcome'] == 'success')})
    f2 = ['arm', 'label', 'n_expected', 'n_valid', 'n_invalid', 'success', 'success_pct',
          'success_lo_pct', 'success_hi_pct', 'inf_stop', 'stall', 'blocked',
          'timeout', 'no_progress', 'collisions', 'runs_with_collision', 'collisions_event_obstacle',
          'collisions_finish_cone', 'collisions_other', 'min_gap_m', 'completed', 'completion_pct',
          'wilson95_lo_pct', 'wilson95_hi_pct', 'encounters', 'encounters_success']
    write_csv(out / 'table2.csv', t2, f2)
    md_rows = [dict(r, wilson=(f'{r["wilson95_lo_pct"]}–{r["wilson95_hi_pct"]}'
                               if r['wilson95_lo_pct'] is not None else ''),
                    swilson=(f'{r["success_lo_pct"]}–{r["success_hi_pct"]}'
                             if r['success_lo_pct'] is not None else ''),
                    n=f'{r["n_valid"]}/{r["n_expected"]}') for r in t2]
    (out / 'table2.md').write_text(
        '표 2. 구현별 결과 — 회피 유형 ' + ', '.join(avoid_types) + '\n\n'
        + md_table(md_rows, ['arm', 'label', 'n', 'inf_stop', 'timeout', 'collisions', 'min_gap_m',
                             'success_pct', 'swilson', 'completion_pct', 'wilson'],
                   ['구현', '이름', '유효 런/계획', '무한 정지 [회]', 'timeout [회]', '충돌 [회]',
                    '최소 간격 [m]', '성공률 [%]', '95% Wilson [%]', '완주율 [%]', '95% Wilson [%]'])
        + '\n성공 = 모든 장애물을 지나감 + 충돌 없음 + 무한 정지 없음. '
          '무한 정지 = stall + blocked (적신호·보행자 면책 없이 판정 시간만큼 정지). '
          'timeout 은 따로 센다. 충돌 = score.py 항목 11.\n', encoding='utf-8')

    # ── by_type (전 유형 — 성공 기준은 유형 분류에 따른다) ────────────────
    bt = []
    for t in types:
        for arm in arms:
            rr = [r for r in runs if r['arm'] == arm and r['type'] == t]
            ok = [r for r in rr if r['status'] in DONE]
            enc = [e for r in ok for e in r['an']['encounters']]
            times = [r['res'].get('time_s') for r in ok if r['status'] == '완주' and r['res'].get('time_s')]
            ded = [r['res'].get('deduction') for r in ok if r['res'].get('deduction') is not None]
            bt.append({'type': t, 'class': type_class[t], 'arm': arm, 'label': lab_ko(arm),
                       'n_expected': expected(t), **success_row(rr), **rate_row(rr),
                       'false_positive_runs': sum(1 for r in ok if r['why'] == 'false_positive_shift'),
                       'excluded_reasons': ';'.join(f'{k}={v}' for k, v in collections.Counter(
                           r['why'] for r in ok if r['success'] is None).items()) or None,
                       'encounters': len(enc),
                       'encounters_success': sum(1 for e in enc if e['outcome'] == 'success'),
                       'encounters_collision': sum(1 for e in enc if e['outcome'] == 'collision'),
                       'encounters_stopped': sum(1 for e in enc if e['outcome'] == 'stopped'),
                       'encounters_not_reached': sum(1 for e in enc if e['outcome'] == 'not_reached'),
                       'mean_time_s': round(sum(times) / len(times), 1) if times else None,
                       'mean_deduction': round(sum(ded) / len(ded), 1) if ded else None})
    write_csv(out / 'by_type.csv', bt)

    # ── by_condition (조건 축 값별 성공률 — 거리·간격·열린 쪽 …) ──────────
    bc = []
    for t in types:
        keys = sorted({k for r in runs if r['type'] == t and r['scen']
                       for k in r['scen']['axes'] if k not in ('route', 'event')})
        for ax in keys:
            vals = sorted({json.dumps(r['scen']['axes'].get(ax), ensure_ascii=False)
                           for r in runs if r['type'] == t and r['scen']})
            for v in vals:
                for arm in arms:
                    rr = [r for r in runs if r['arm'] == arm and r['type'] == t and r['scen']
                          and json.dumps(r['scen']['axes'].get(ax), ensure_ascii=False) == v]
                    bc.append({'type': t, 'axis': ax, 'value': json.loads(v), 'arm': arm,
                               'n_valid': sum(1 for r in rr if r['status'] in DONE), **success_row(rr),
                               'inf_stop': sum(1 for r in rr if r['status'] in INF_STOP),
                               'collisions': sum(r['n_col'] for r in rr if r['status'] in DONE)})
    write_csv(out / 'by_condition.csv', bc)

    # ── failures (실패 런마다 멈춘 이유와 그때 장애물까지 거리) ───────────
    fl = []
    for r in runs:
        if r['status'] not in DONE or r['success'] is not False:
            continue
        sn = r['an']['snapshot'] or {}
        bad = next((e for e in r['an']['encounters'] if e['outcome'] != 'success'), None)
        c0 = r['an']['collisions'][0] if r['an']['collisions'] else {}
        fl.append({'arm': r['arm'], 'type': r['type'], 'scenario': r['name'],
                   'class': type_class[r['type']], 'status': r['status'], 'why': r['why'],
                   'conditions': json.dumps({k: v for k, v in r['scen']['axes'].items()
                                             if k not in ('route', 'event')}, ensure_ascii=False),
                   'failed_encounter': None if bad is None else bad['event'],
                   'failed_encounter_outcome': None if bad is None else bad['outcome'],
                   # 끝(멈춘 자리)에서: 가장 가까운 장애물까지 앞범퍼~뒤끝 거리, 멈춘 동안의 회피 진단
                   'obstacle_gap_m': sn.get('nearest_encounter_gap_m'),
                   'obstacle_dist_m': sn.get('nearest_encounter_dist_m'),
                   'stop_avoid_state': sn.get('stop_avoid_state'),
                   'stop_avoid_reject': sn.get('stop_avoid_reject'),
                   'end_avoid_state': sn.get('avoid_state'), 'end_avoid_rejects': sn.get('avoid_rejects')
                   or sn.get('avoid_reject'), 'escape_state': sn.get('escape_state'),
                   'winner': sn.get('winner'), 'stopped_s': sn.get('stopped_s'),
                   'light_state': sn.get('light_state'), 'ahead_kind': sn.get('ahead_kind'),
                   'ahead_lon_m': sn.get('ahead_lon_m'), 'route_s': sn.get('route_s'),
                   'collision_obj_kind': c0.get('obj_kind'), 'collision_min_gap_m': c0.get('min_gap_m'),
                   'collision_t_s': c0.get('t'), 'collision_ego_v_mps': c0.get('ego_v_mps'),
                   'collision_obj_v_mps': c0.get('obj_v_mps'), 'collision_obj_dyaw_deg': c0.get('obj_dyaw_deg'),
                   'cause': r['cause'], 'cause_ko': CAUSE_KO.get(r['cause']),
                   'cause_evidence': r['cause_evidence'], 'log': str(r['log'])})
    write_csv(out / 'failures.csv', fl)

    # ── causes (실패 런 + 기반 실패 런의 원인 분류 — 결과에서 빼지 않는다) ──
    write_csv(out / 'causes.csv', [
        {'arm': r['arm'], 'type': r['type'], 'scenario': r['name'], 'class': type_class.get(r['type']),
         'status': r['status'], 'why': r['why'], 'cause': r['cause'], 'cause_ko': CAUSE_KO.get(r['cause']),
         'evidence': r['cause_evidence']}
        for r in runs if r['cause']],
        ['arm', 'type', 'scenario', 'class', 'status', 'why', 'cause', 'cause_ko', 'evidence'])
    # 종료선 이후라 세지 않은 충돌 — 무엇을 뺐는지 남긴다
    write_csv(out / 'collisions_after_finish.csv', [
        {'arm': r['arm'], 'type': r['type'], 'scenario': r['name'],
         'finish_thr_m': (r['an'].get('finish') or {}).get('finish_thr_m'), **c}
        for r in runs for c in r['an'].get('collisions_after_finish') or []])

    # ── queue_fp (신호 대기 줄: 적색 동안 대기 차량에 만든 시프트) ─────────
    qrows = []
    for r in runs:
        if r['type'] not in queue_types or r['status'] not in DONE:
            continue
        q = r['an'].get('queue') or {}
        ev = (r['scen'].get('signal_queue') or [{}])[0]
        base = {'arm': r['arm'], 'type': r['type'], 'scenario': r['name'], 'status': r['status'],
                'success': r['success'], 'why': r['why'], 'queue_count': ev.get('count'),
                'green_at_s': ev.get('green_at_s'), 't_release_s': q.get('t_release_s'),
                'met_before_release': q.get('met_before_release'),
                'shifts_on_queue': len(q.get('shifts_on_queue') or []),
                'collisions': r['n_col'], 'log': str(r['log'])}
        hits = q.get('shifts_on_queue') or [{}]
        for h in hits:
            qrows.append({**base, 'shift_tick': h.get('tick'), 'shift_t_s': h.get('t'),
                          'shift_side': h.get('side'), 'blocker_id': h.get('blocker_id'),
                          's_rel_m': h.get('s_rel_m'), 'light_state_at_shift': h.get('light_state')})
    write_csv(out / 'queue_fp.csv', qrows,
              ['arm', 'type', 'scenario', 'status', 'success', 'why', 'queue_count', 'green_at_s',
               't_release_s', 'met_before_release', 'shifts_on_queue', 'shift_tick', 'shift_t_s',
               'shift_side', 'blocker_id', 's_rel_m', 'light_state_at_shift', 'collisions', 'log'])

    # ── plain (일반 주행: 완주 · 주행 시간 · 감점) ────────────────────────
    write_csv(out / 'plain.csv',
              [{'arm': r['arm'], 'type': r['type'], 'scenario': r['name'], 'status': r['status'],
                'success': r['success'], 'time_s': r['res'].get('time_s'), 'ran_s': r['res'].get('ran_s'),
                'dist_m': r['res'].get('dist_m'), 'avg_kph': r['res'].get('avg_kph'),
                'deduction': r['res'].get('deduction'),
                'violations': json.dumps(r['res'].get('violations') or {}, ensure_ascii=False),
                'collisions': r['n_col'],
                'collision_kinds': ';'.join(sorted({c['obj_kind'] for c in r['an']['collisions']})) or None,
                'builders_same_lanes': r['scen']['builders_same'] if r['scen'] else None,
                'log': str(r['log'])}
               for r in runs if r['type'] in plain_types and r['status'] in DONE])

    # ── collisions ──────────────────────────────────────────────────────
    write_csv(out / 'collisions.csv',
              [{'arm': r['arm'], 'type': r['type'], 'scenario': r['name'], 'status': r['status'], **c,
                'log': str(r['log'])} for r in runs for c in r['an']['collisions']],
              ['arm', 'type', 'scenario', 'status', 't', 'route_s', 'obj_id', 'obj_cls', 'obj_kind',
               'encounter', 'min_gap_m', 'v_kph', 'onset_t', 'ego_v_mps', 'obj_v_mps', 'obj_dyaw_deg', 'log'])

    # ── stall_review ────────────────────────────────────────────────────
    sr = [{'arm': r['arm'], 'type': r['type'], 'scenario': r['name'], 'status': r['status'],
           **r['an']['review'], 'log': str(r['log'])}
          for r in runs if r['an']['review'] is not None]
    write_csv(out / 'stall_review.csv', sr,
              ['arm', 'type', 'scenario', 'status', 't_end_s', 'route_s', 'stopped_s', 'v_target',
               'winner', 'light_id', 'light_state', 'light_hold_frac', 'dist_stop_line_m', 'in_junction',
               'ahead_present', 'ahead_kind', 'ahead_cls', 'ahead_id', 'ahead_encounter', 'ahead_lon_m',
               'ahead_lat_m', 'ahead_speed_mps', 'nearest_encounter', 'nearest_encounter_dist_m',
               'nearest_encounter_lon_m', 'avoid_state', 'avoid_reject', 'avoid_rejects', 'escape_state',
               'blocked_why', 'queue_suspect', 'log'])

    # ── per_encounter ───────────────────────────────────────────────────
    long_rows, enc_index = [], {}
    for r in arun:
        if r['status'] not in DONE:
            continue
        for e in r['an']['encounters']:
            long_rows.append({'arm': r['arm'], 'type': r['type'], 'scenario': r['name'],
                              'run_status': r['status'], **{k: v for k, v in e.items() if k != 'gt_ids'},
                              'gt_ids': ';'.join(map(str, e['gt_ids']))})
            enc_index.setdefault((r['type'], r['name'], e['event']), {'kind': e['kind'],
                                 'n_objects': e['n_objects'], 'route_s': e['route_s']})[r['arm']] = e
    write_csv(out / 'per_encounter_long.csv', long_rows)
    # 시나리오 정의 기준 전체 조우 목록 (아직 안 돈 것도 줄은 있다)
    for key, sc in scen_cache.items():
        if not sc or not sc['encounters'] or sc['name'] in excluded:
            continue
        p = pathlib.Path(key)
        for e in sc['encounters']:
            enc_index.setdefault((p.parent.name, sc['name'], e['event']),
                                 {'kind': e['kind'], 'n_objects': len(e['objects']), 'route_s': e['route_s']})
    wide = []
    for (t, name, ev), d in sorted(enc_index.items()):
        row = {'type': t, 'scenario': name, 'event': ev, 'kind': d['kind'],
               'n_objects': d['n_objects'], 'route_s': d['route_s']}
        for arm in arms:
            e = d.get(arm)
            row[f'{arm}_outcome'] = e['outcome'] if e else None
            row[f'{arm}_min_gap_m'] = e['min_gap_m'] if e else None
        wide.append(row)
    write_csv(out / 'per_encounter.csv', wide)

    # ── paired (McNemar) ────────────────────────────────────────────────
    paired = []
    status = {(r['arm'], r['type'], r['name']): r['status'] for r in arun if r['status'] in DONE}

    def mc(unit, pairs, other, note):
        a_only = sum(1 for x, y in pairs if x and not y)
        o_only = sum(1 for x, y in pairs if y and not x)
        paired.append({'unit': unit, 'ref': ref, 'other': other, 'n_pairs': len(pairs),
                       'both_success': sum(1 for x, y in pairs if x and y),
                       'ref_only_success': a_only, 'other_only_success': o_only,
                       'both_fail': sum(1 for x, y in pairs if not x and not y),
                       'mcnemar_exact_p': round(mcnemar_exact(a_only, o_only), 6), 'note': note})

    if ref in arms:
        for other in [x for x in arms if x != ref]:
            keys = sorted({(t, n) for (ar, t, n) in status if ar == ref}
                          & {(t, n) for (ar, t, n) in status if ar == other})
            succ = {(r['arm'], r['type'], r['name']): r['success'] for r in runs
                    if r['success'] is not None}
            for cls, tt in (('avoid', avoid_types), ('queue', queue_types), ('brake', brake_types),
                            ('plain', plain_types)):
                pk = sorted({(t, n) for (ar, t, n) in succ if ar == ref and t in tt}
                            & {(t, n) for (ar, t, n) in succ if ar == other and t in tt})
                mc(f'scenario(성공, {CLASS_KO[cls]})', [(succ[(ref, t, n)], succ[(other, t, n)]) for t, n in pk],
                   other, '같은 시나리오 짝 — 유형별 성공 기준')
            mc('scenario(완주)', [(status[(ref, t, n)] == '완주', status[(other, t, n)] == '완주')
                               for t, n in keys], other, '같은 시나리오 짝 — 완주 여부 (회피 유형)')
            ep = [(d[ref]['outcome'] == 'success', d[other]['outcome'] == 'success')
                  for d in enc_index.values() if ref in d and other in d]
            mc('encounter(회피 성공, 미도달=실패)', ep, other,
               '조우 단위 — 한 시나리오의 조우들은 서로 독립이 아니다 (참고용)')
            ep2 = [(d[ref]['outcome'] == 'success', d[other]['outcome'] == 'success')
                   for d in enc_index.values() if ref in d and other in d
                   and d[ref]['reached'] and d[other]['reached']]
            mc('encounter(회피 성공, 둘 다 도달한 조우만)', ep2, other, '위와 같음')
    write_csv(out / 'paired.csv', paired)

    # ── brake_fp ────────────────────────────────────────────────────────
    fp_rows, fp_sum = [], []
    for arm in arms:
        rr = [r for r in runs if r['arm'] == arm and r['type'] in brake_types and r['status'] in DONE]
        n_shift = n_fp = runs_fp = 0
        for r in rr:
            lb = (r['scen'] or {}).get('lead_brake') or [{}]
            hits = 0
            for s in r['an']['shifts']:
                fp_rows.append({'arm': arm, 'type': r['type'], 'scenario': r['name'],
                                'run_status': r['status'], 'lead_resume': lb[0].get('resume'),
                                'lead_decel': lb[0].get('decel'), **s, 'log': str(r['log'])})
                n_shift += 1
                hits += int(s['false_positive'])
            n_fp += hits
            runs_fp += int(hits > 0)
        fp_sum.append({'arm': arm, 'label': lab_en(arm), 'runs': len(rr),
                       'n_expected': sum(expected(t) or 0 for t in brake_types),
                       'shifts': n_shift, 'false_positive_shifts': n_fp, 'runs_with_false_positive': runs_fp,
                       'completed': sum(1 for r in rr if r['status'] == '완주'),
                       'inf_stop': sum(1 for r in rr if r['status'] in INF_STOP),
                       'collisions': sum(r['n_col'] for r in rr)})
    write_csv(out / 'brake_fp.csv', fp_rows,
              ['arm', 'type', 'scenario', 'run_status', 'lead_resume', 'lead_decel', 'tick', 't',
               'route_s', 'state', 'side', 'blocker_id', 'chain_ids', 's_rel_m', 'trigger', 'blocker_kind',
               'blocker_speed_at_shift', 'blocker_speed_max_after', 'moved_threshold_mps', 't_moved_s',
               'false_positive', 'log'])
    write_csv(out / 'brake_fp_summary.csv', fp_sum)

    # ── 그림 ────────────────────────────────────────────────────────────
    trace_note = []
    if not a.no_figures:
        fdir = out / 'figures'
        fdir.mkdir(exist_ok=True)
        fig_table2(t2, lab_en, fdir)
        fig_by_type(bt, lab_en, tlab, fdir)
        for fc in rep_cfg.get('condition_figures') or []:
            fig_condition(bc, fc, lab_en, tlab, fdir)
        fig_end_reason(arun, arms, lab_en, fdir)
        fig_brake_fp(fp_sum, lab_en, fdir)
        pairs = []
        byk = {(r['arm'], r['type'], r['name']): r for r in arun if r['status'] in DONE and r['log']}
        for other in [x for x in (rep_cfg.get('trace_arms') or []) if x in arms and x != ref]:
            cands = sorted((t, n) for (ar, t, n), r in byk.items()
                           if ar == other and r['status'] in INF_STOP
                           and (ref, t, n) in byk and byk[(ref, t, n)]['status'] == '완주'
                           and r['scen'] and r['scen']['encounters']
                           and r['scen']['encounters'][0]['objects'][0]['xy'] is not None)
            if a.trace:
                cands = [c for c in cands if c[1] == a.trace]
            if not cands:
                trace_note.append(f'{other} vs {ref}: 조건에 맞는 시나리오 없음 '
                                  f'({other} 무한 정지 · {ref} 완주)')
                continue
            t, n = cands[0]
            pairs.append((byk[(other, t, n)], byk[(ref, t, n)], byk[(ref, t, n)]['scen'],
                          f'{other} stopped, {ref} completed'))
            trace_note.append(f'{other} vs {ref}: {t}/{n} (후보 {len(cands)}개 중 이름순 첫 번째)')
        fig_speed_trace(pairs, lab_en, tlab, fdir, a.pass_near_m)

    # ── meta + 화면 요약 ────────────────────────────────────────────────
    cover = [{'arm': arm, 'type': t, 'expected': expected(t),
              'valid': sum(1 for r in runs if r['arm'] == arm and r['type'] == t and r['status'] in DONE),
              'invalid': sum(1 for r in runs if r['arm'] == arm and r['type'] == t and r['status'] not in DONE)}
             for arm in arms for t in types if any(r['arm'] == arm and r['type'] == t for r in runs)]
    # 재현 정보 (2026-10-02): 집계한 시점의 코드(git) · 런이 실제로 쓴 설정과 코드(arms.*.settings) ·
    # 지금의 기준 params 해시 · 시나리오 세트의 지문. 커밋 해시만으로는 미커밋 변경을 못 가린다.
    git = batch_run.git_state(ROOT)
    base_path = ROOT / 'config' / 'params.yaml'
    base_sha = hashlib.sha256(base_path.read_bytes()).hexdigest() if base_path.exists() else None
    manifest = manifest_info(scen_dir)
    settings = {k: arm_settings(runs, k) for k in arm_dirs}
    warns = [f'{r["arm"]}/{r["type"]}/{r["name"]}: {w}' for r in runs for w in r['an']['warn']
             if r['status'] in DONE]
    repro_warn = []
    if git['dirty']:
        repro_warn.append(f'집계 시점에 미커밋 변경 {len(git["modified"])}개 — 커밋 해시가 코드를 특정하지 못한다')
    if manifest is None:
        repro_warn.append(f'{scen_dir}/MANIFEST.sha256 없음 — 시나리오 세트를 해시로 특정할 수 없다')
    elif manifest['mismatch']:
        repro_warn.append(f'시나리오 파일 {len(manifest["mismatch"])}개가 MANIFEST 와 다르다 (생성 뒤 바뀜)')
    for k, s in settings.items():
        if len(s['code']) > 1 or len(s['base_params_sha256']) > 1 or len(s['overlay']) > 1:
            repro_warn.append(f'{k}: 런마다 코드·설정이 다르다 (코드 {len(s["code"])}종, '
                              f'기준 params {len(s["base_params_sha256"])}종, overlay {len(s["overlay"])}종)')
        if any(c['dirty'] for c in s['code']):
            repro_warn.append(f'{k}: 미커밋 변경이 있는 채로 돈 런 {sum(c["runs"] for c in s["code"] if c["dirty"])}건')
    meta = {'generated': datetime.datetime.now().isoformat(timespec='seconds'), 'commit': git['commit_short'],
            'git': git, 'base_params': {'path': str(base_path), 'sha256': base_sha},
            'scenario_manifest': manifest, 'repro_warnings': repro_warn,
            'arms': {k: {'dir': str(v), 'label_ko': lab_ko(k), 'label_en': lab_en(k), 'settings': settings[k]}
                     for k, v in arm_dirs.items()},
            'reference': ref, 'scenarios': str(scen_dir),
            'excluded_scenarios': excluded, 'excluded_runs': n_before - len(runs),
            'avoid_types': avoid_types, 'brake_types': brake_types,
            'queue_types': queue_types, 'plain_types': plain_types,
            'auto_excluded_builders_differ': auto_excl,
            'auto_excluded_why': auto_why,
            'builder_rule': {'off_lane': '어느 한쪽 빌더의 경로가 장애물 차로를 지나지 않음 (또는 빌드 실패)',
                             'len_tol': a.builder_len_tol},
            'definitions': {'inf_stop': list(INF_STOP), 'valid_status': list(DONE),
                            'collision': 'score.detect_proximity (항목 11)',
                            'after_finish': a.after_finish,
                            'cause_rules': ['simulator: 기반 실패', 'false_positive: 오탐 시프트',
                                            'scenario: 충돌이 전부 종료선 콘, 또는 전부 움직이는 차량이 거의 선 '
                                            '자차로 들어옴', 'controller: 나머지'],
                            'cause_intrude': INTRUDE},
            'thresholds': {k: getattr(a, k) for k in ('match_m', 'pass_near_m', 'pass_lat_m', 'detect_m',
                                                      'ahead_m', 'ahead_lat_m', 'after_finish')},
            'collisions_after_finish_excluded': sum(len(r['an'].get('collisions_after_finish') or [])
                                                    for r in runs),
            'judge': sorted({json.dumps(r['res'].get('judge'), sort_keys=True) for r in runs
                             if r['res'].get('judge')}),
            'coverage': cover, 'trace': trace_note, 'warnings': warns}
    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')

    print((out / 'table2.md').read_text(encoding='utf-8'))
    if excluded:
        print(f'제외한 시나리오 {len(excluded)}개 (런 {n_before - len(runs)}건): {", ".join(excluded)}'
              + (f'  — 그중 두 빌더의 경로가 달라 자동 제외 {len(auto_excl)}개' if auto_excl else ''))
    print('유형별 성공률 (판정 대상 런 기준):')
    for r in bt:
        if r['n_valid']:
            print(f'  {r["arm"]} {r["type"]:16s} [{CLASS_KO.get(r["class"], r["class"])}] '
                  f'{r["success"]}/{r["n_judged"]}'
                  + (f' = {r["success_pct"]}%' if r['n_judged'] else '')
                  + (f'  (판정 제외 {r["n_excluded"]}: {r["excluded_reasons"]})' if r['n_excluded'] else ''))
    print('진행 (유효/계획, 기반 실패):')
    for c in cover:
        print(f'  {c["arm"]} {c["type"]:14s} {c["valid"]}/{c["expected"]}' + (f'  기반 실패 {c["invalid"]}' if c['invalid'] else ''))
    if len(meta['judge']) > 1:
        print('  [주의] 런마다 판정값이 다르다 — meta.json 의 judge 확인')
    for w in trace_note:
        print('  그래프:', w)
    if warns:
        print(f'  경고 {len(warns)}건 — meta.json 의 warnings')
    print(f'재현 정보: 커밋 {git["commit_short"]} ({git["branch"]})'
          + (' +미커밋 변경' if git['dirty'] else '')
          + (f'  시나리오 MANIFEST {manifest["sha256"][:12]}… ({manifest["files"]}개)' if manifest else ''))
    for w in repro_warn:
        print(f'  [재현] {w}')
    print(f'\n산출물: {out}/')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
