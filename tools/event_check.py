"""
시나리오 이벤트가 실주행에서 **실제로 조우했는지** 판정한다.

생성기(tools/gen_scenarios.py)가 보행자 이벤트마다 트리거 거리를 역산하면서
기대값(t_near/t_far/meet_lat_m/lane_w_m)을 시나리오 yaml 에 남긴다. 여기서는
그 기대값과 주행 로그(run_*.jsonl)의 실제 보행자 횡위치를 대조해 이벤트별로
성립/미성립을 낸다. batch_run 이 이 결과를 표와 요약 한 줄에 얹는다.

    python3 tools/event_check.py <시나리오>.yaml <런>.jsonl

왜 필요한가 — 2026-08-30 실사고: 무단횡단 보행자가 ego 가 지나간 48 m 뒤에야
차로에 진입했는데, 로그·리포트 어디에도 "조우가 없었다"는 신호가 없었다.
육안(VTD 화면)으로만 알 수 있었고, 그마저 "제어기 미반응"으로 오진했다.
밤샘 배치 20건을 아침에 훑을 때 이걸 눈으로 잡을 수는 없다.

판정: ego 가 이벤트 지점(route_s)을 지나는 순간 보행자의 **출발측 기준 횡위치**가
차로 반폭 + params event_trigger.meet_tol_m 안이면 성립.
  · 미도달  — ego 가 그 지점까지 못 감 (미완주·조기 종료)
  · 미검출  — 그 순간 보행자 객체가 9910 에 없음 (스폰 실패·80 m 밖)
  · 늦음    — 아직 출발측에 있음 (트리거가 늦었다 = 옛 고정 25 m 의 실패 양상)
  · 이름    — 이미 건너감 (트리거가 일렀다)
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from vtd_adapter.config import load_params_yaml          # noqa: E402

# 기대값을 남기는 이벤트 = 보행자 3종 (정차차량·협착 등은 t=0 배치라 조우 타이밍이 없다)
PED_KINDS = ('pedestrian', 'jaywalk', 'ped_blind')

OK, LATE, EARLY, NOSEE, NOREACH = '성립', '늦음', '이름', '미검출', '미도달'
LEGACY = '기대없음'      # 역산 이전 생성기가 만든 시나리오 — 판정 근거가 yaml 에 없다


def load_ticks(log_path) -> list:
    with open(log_path, encoding='utf-8') as f:
        return [json.loads(ln) for ln in f if '"raw"' in ln]


def ped_lat_from_start(tick: dict, from_side: str):
    """이 틱에서 ego 에 가장 가까운 보행자의 **출발측 기준** 횡위치 [m].

    + 는 아직 출발측(안 건넘), − 는 건너간 뒤. 보행자가 없으면 None.
    """
    e = tick['ego']
    yaw = float(e['yaw'])
    best = None
    for o in tick.get('objects') or ():
        if o.get('cls') != 'pedestrian':
            continue
        dx, dy = float(o['x']) - float(e['x']), float(o['y']) - float(e['y'])
        d = math.hypot(dx, dy)
        # ego 프레임 횡(좌 +) → 출발측 부호로 뒤집는다
        lat = -dx * math.sin(yaw) + dy * math.cos(yaw)
        signed = -lat if from_side == '우측' else lat
        if best is None or d < best[0]:
            best = (d, signed)
    return None if best is None else best[1]


def check_event(ev: dict, ticks: list, tol_m: float) -> dict:
    """이벤트 1건 판정 → {kind, route_s, verdict, lat_m, expect_lat_m, band_m}."""
    s = float(ev['route_s'])
    band = 0.5 * float(ev['lane_w_m']) + tol_m
    out = {'kind': ev['kind'], 'route_s': round(s, 1), 'band_m': round(band, 2),
           'expect_lat_m': ev.get('meet_lat_m'), 'lat_m': None}
    if not ticks or max(float(t['ego']['route_s']) for t in ticks) < s:
        return {**out, 'verdict': NOREACH}
    # ego 가 이벤트 지점을 지나는 순간
    tick = min(ticks, key=lambda t: abs(float(t['ego']['route_s']) - s))
    lat = ped_lat_from_start(tick, ev.get('from', '우측'))
    if lat is None:
        return {**out, 'verdict': NOSEE}
    out['lat_m'] = round(lat, 2)
    if abs(lat) <= band:
        return {**out, 'verdict': OK}
    return {**out, 'verdict': LATE if lat > 0 else EARLY}


def check_scenario(yaml_path, log_path, cfg: dict | None = None) -> dict:
    """시나리오 yaml + 로그 → {total, ok, events[…]}. 보행자 이벤트가 없으면 total 0."""
    import yaml as _yaml
    cfg = cfg or load_params_yaml()
    tol = float(cfg['event_trigger']['meet_tol_m'])
    sdef = _yaml.safe_load(pathlib.Path(yaml_path).read_text(encoding='utf-8'))
    ticks = load_ticks(log_path)
    res = []
    for e in (sdef.get('events') or ()):
        if e.get('kind') not in PED_KINDS:
            continue
        if 'lane_w_m' not in e:          # 구 생성기 산출물 — 조용히 빼지 않고 드러낸다
            res.append({'kind': e['kind'], 'route_s': round(float(e['route_s']), 1),
                        'verdict': LEGACY, 'lat_m': None, 'expect_lat_m': None,
                        'band_m': None})
            continue
        res.append(check_event(e, ticks, tol))
    return {'total': len(res), 'ok': sum(1 for r in res if r['verdict'] == OK),
            # 미도달은 "실패"와 구분해야 한다 — 미완주 런에서 도달조차 못 한
            # 이벤트를 성립 실패로 읽으면 원인 판단이 어긋난다.
            'unreached': sum(1 for r in res if r['verdict'] == NOREACH),
            'events': res}


def render(rep: dict, name: str = '') -> str:
    un = rep.get('unreached') or 0
    L = [f"이벤트 조우 {rep['ok']}/{rep['total']} 성립"
         + (f' (미도달{un})' if un else '') + (f'  ({name})' if name else '')]
    for r in rep['events']:
        lat = '—' if r['lat_m'] is None else f"{r['lat_m']:+.2f}"
        band = '—' if r['band_m'] is None else f"±{r['band_m']:.2f}"
        exp = '—' if r['expect_lat_m'] is None else f"{r['expect_lat_m']:+.2f}"
        L.append(f"  {r['kind']:<11} s={r['route_s']:>7.1f} m  {r['verdict']:<5} "
                 f"횡 {lat:>6} / 허용 {band:>6} m  (예상 {exp})")
    return '\n'.join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='시나리오 이벤트 조우 성립 판정')
    ap.add_argument('scenario_yaml')
    ap.add_argument('log')
    a = ap.parse_args(argv)
    rep = check_scenario(a.scenario_yaml, a.log)
    print(render(rep, pathlib.Path(a.scenario_yaml).stem))
    return 0 if rep['ok'] == rep['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
