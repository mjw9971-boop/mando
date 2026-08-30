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
LEGACY = '기대없음'      # 지도가 없어 기대값을 못 만든 경우에만 남는다 (아래 expectation)


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


def expectation(ev: dict, et: dict, lg=None, key=None, s_in_lane: float = 0.0):
    """이벤트 1건의 조우 기대값 — 없으면 None.

    공식은 gen_scenarios.ped_trigger 의 역산과 같은 하나다. 입력(차로폭·보행자
    시작 횡거리·예상속도)만 두 곳에서 온다:
      · 신 생성기 산출물 — yaml 에 lane_w_m/lat_start_m/v_exp_mps 가 실려 있다.
      · 구 생성기 산출물(트리거 고정 25 m) — 그 키가 없다. **여기서 지도로 다시
        잰다** — 없다고 판정을 포기하면(옛 '기대없음') 조우가 왜 안 됐는지가
        리포트에서 사라진다. 2026-08-30 실전주행_01_좌회전16 이 그 사례다.

    상수(반경·예상속도계수·대체 제한속도·보행자 시작 여유)는 전부 params
    event_trigger.* 다 — 생성기와 같은 값을 봐야 기대값이 생성기와 일치한다.

        t_meet = (trigger_d + radius_m) / v_exp      # 트리거 → ego 도착
        meet_lat = 횡시작거리 − 보행속도 × t_meet    # 그 순간 보행자 횡위치
        조우 창 = [t_near, t_far] = (횡시작거리 ∓ 차로반폭) / 보행속도
    """
    lane_w, lat0 = ev.get('lane_w_m'), ev.get('lat_start_m')
    v_exp, ws = ev.get('v_exp_mps'), ev.get('walk_speed')
    if None in (lane_w, lat0, v_exp) and lg is not None and key is not None:
        left, right = lg.roadway_edges(key, s_in_lane)
        side = right if ev.get('from', '우측') == '우측' else left
        lim = lg.lanes[key]['speed_limit'] or float(et['default_limit_kph'])
        lane_w = float(lg.width_at(key, s_in_lane)) if lane_w is None else lane_w
        lat0 = side + float(et['ped_start_margin_m']) if lat0 is None else lat0
        v_exp = float(lim) / 3.6 * float(et['speed_factor']) if v_exp is None else v_exp
    if lane_w is None:              # 차로폭이 없으면 허용대를 못 만든다 = 판정 불가
        return None
    ws = float(et['walk_speed_default'][ev['kind']]) if ws is None else float(ws)
    half = 0.5 * float(lane_w)
    rad = float(ev.get('trigger_radius_m', et['radius_m']))
    t_meet = (None if ev.get('trigger_d') is None or v_exp is None else
              (float(ev['trigger_d']) + rad) / float(v_exp))
    # yaml 에 생성기가 남긴 값이 있으면 그것이 우선 — xml 이 그 값으로 만들어졌으므로
    # 반올림된 입력으로 다시 계산한 값보다 그쪽이 사실이다.
    out = {'lane_w_m': round(float(lane_w), 2),
           't_meet_s': None if t_meet is None else round(t_meet, 2),
           't_near_s': ev.get('t_near_s'), 't_far_s': ev.get('t_far_s'),
           'meet_lat_m': ev.get('meet_lat_m')}
    if lat0 is not None:
        if out['t_near_s'] is None:
            out['t_near_s'] = round((lat0 - half) / ws, 2)
        if out['t_far_s'] is None:
            out['t_far_s'] = round((lat0 + half) / ws, 2)
        if out['meet_lat_m'] is None and t_meet is not None:
            out['meet_lat_m'] = round(lat0 - ws * t_meet, 2)
    return out


def check_event(ev: dict, ticks: list, tol_m: float, exp: dict) -> dict:
    """이벤트 1건 판정 → {kind, route_s, verdict, lat_m, expect_lat_m, band_m, …}."""
    s = float(ev['route_s'])
    band = 0.5 * float(exp['lane_w_m']) + tol_m
    out = {'kind': ev['kind'], 'route_s': round(s, 1), 'band_m': round(band, 2),
           'expect_lat_m': exp['meet_lat_m'], 'lat_m': None,
           't_near_s': exp['t_near_s'], 't_far_s': exp['t_far_s'],
           't_meet_s': exp['t_meet_s']}
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


def _lane_at_event(s: float, route, ticks: list):
    """이벤트 지점의 (LaneKey, 차로내 s) — 기대값을 지도에서 잴 때 쓴다.

    경로(route pkl)가 있으면 거기서 — ego 가 못 간 이벤트도 잴 수 있다.
    없으면 그 지점을 지나는 로그 틱의 ego 차로로 대신한다 (미도달이면 못 잰다).
    """
    if route is not None:
        from gen_scenarios import lane_at         # 차로 탐색 중복 구현 금지
        _, k, sl = lane_at(route, s)
        return k, sl
    if not ticks or max(float(t['ego']['route_s']) for t in ticks) < s:
        return None, 0.0
    t = min(ticks, key=lambda t: abs(float(t['ego']['route_s']) - s))
    return tuple(t['ego']['lane']), float(t['ego']['s'])


def traffic_metrics(ticks: list, planned: int, tc: dict) -> dict:
    """동방향 교통류 **관측 지표** — 판정이 아니다.

    조우 성립(보행자)과 달리 교통류는 "언제 만나야 한다" 는 기대 시점이 없다.
    대신 배치한 대수가 실제로 로그에 나타났는지, 얼마나 가까웠는지를 남긴다.
    이 훅이 없으면 교통류가 XML 에만 있고 실주행에 안 나타나도 리포트가 조용하다
    — 2026-08-30 보행자 이벤트가 정확히 그렇게 실패했다.

    planned 는 PulkDef Count(영역 내 유지 대수)다. PulkTraffic 은 영역을 벗어난
    차량을 가장자리에 재배치하므로 observed 가 planned 를 넘을 수 있다 — 관측
    지표지 판정이 아니라서 상한을 두지 않는다. 0 대만 경고 대상이다.

    로그에는 객체 heading 이 없고 s_rel/on_route 는 PDM 이관 후 항상 None 이라
    (run_agent._log_objects) ego 프레임 종·횡거리를 여기서 직접 계산한다.
    '동방향 주행 차량' = cls 가 vehicle 이고, 한 번이라도 observe_moving_mps 이상
    움직였고, 그때 |횡| 이 observe_corridor_m 안인 id. 정차 차량(narrow ·
    static_vehicle 은 속도 0)과 회랑 밖 차량이 이 정의에서 빠진다.

    min_lon_m 은 **부호 있는 종거리의 절댓값 최소** — 가장 가깝게 스친 순간이다
    (+ 앞, − 뒤). 앞뒤를 각각 따로도 남긴다.
    """
    v_min = float(tc['observe_moving_mps'])
    corr = float(tc['observe_corridor_m'])
    seen, lon_min, ahead, behind = set(), None, None, None
    for t in ticks:
        e = t['ego']
        ex, ey, yaw = float(e['x']), float(e['y']), float(e['yaw'])
        cy, sy = math.cos(yaw), math.sin(yaw)
        for o in t.get('objects') or ():
            if o.get('cls') != 'vehicle' or float(o.get('speed') or 0.0) < v_min:
                continue
            dx, dy = float(o['x']) - ex, float(o['y']) - ey
            lon, lat = dx * cy + dy * sy, -dx * sy + dy * cy
            if abs(lat) > corr:
                continue
            seen.add(int(o['id']))
            if lon_min is None or abs(lon) < abs(lon_min):
                lon_min = lon
            if lon >= 0 and (ahead is None or lon < ahead):
                ahead = lon
            if lon < 0 and (behind is None or -lon < behind):
                behind = -lon
    return {'planned': planned, 'observed': len(seen),
            'min_lon_m': None if lon_min is None else round(lon_min, 2),
            'min_ahead_m': None if ahead is None else round(ahead, 2),
            'min_behind_m': None if behind is None else round(behind, 2),
            'corridor_m': corr, 'moving_mps': v_min}


def check_scenario(yaml_path, log_path, cfg: dict | None = None,
                   lg=None, route=None) -> dict:
    """시나리오 yaml + 로그 → {total, ok, events[…]}. 보행자 이벤트가 없으면 total 0.

    lg/route 를 주면 yaml 에 기대값이 없는 구 시나리오도 지도에서 다시 재서
    판정한다 (expectation 참고). 둘 다 없고 yaml 에도 없으면 그때만 '기대없음'.
    """
    import yaml as _yaml
    cfg = cfg or load_params_yaml()
    et = cfg['event_trigger']
    tol = float(et['meet_tol_m'])
    sdef = _yaml.safe_load(pathlib.Path(yaml_path).read_text(encoding='utf-8'))
    ticks = load_ticks(log_path)
    res = []
    for e in (sdef.get('events') or ()):
        if e.get('kind') not in PED_KINDS:
            continue
        key, sl = (None, 0.0) if 'lane_w_m' in e else _lane_at_event(
            float(e['route_s']), route, ticks)
        exp = expectation(e, et, lg, key, sl)
        if exp is None:                  # 지도도 경로도 없다 — 조용히 빼지 않고 드러낸다
            res.append({'kind': e['kind'], 'route_s': round(float(e['route_s']), 1),
                        'verdict': LEGACY, 'lat_m': None, 'expect_lat_m': None,
                        'band_m': None, 't_near_s': None, 't_far_s': None,
                        't_meet_s': None})
            continue
        res.append(check_event(e, ticks, tol, exp))
    # 교통류는 별도 키다 — events 에 섞으면 성립/전체 집계가 오염된다.
    # 배치 대수의 출처는 시나리오 전역 속성 pulk.Count 다 (PulkTraffic). 교통류는
    # 이벤트가 아니므로 events 목록에는 없다 (2026-08-30 ev_traffic 대체).
    planned = int((sdef.get('pulk') or {}).get('Count') or 0)
    traffic = traffic_metrics(ticks, planned, cfg['gen_traffic']) if planned else None
    return {'total': len(res), 'ok': sum(1 for r in res if r['verdict'] == OK),
            'traffic': traffic,
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
        t = (r.get('t_meet_s'), r.get('t_near_s'), r.get('t_far_s'))
        when = ('시점 —' if None in t else
                f'도착 {t[0]:.2f} s / 조우창 {t[1]:.2f}~{t[2]:.2f} s')
        L.append(f"  {r['kind']:<11} s={r['route_s']:>7.1f} m  {r['verdict']:<5} "
                 f"횡 {lat:>6} / 허용 {band:>6} m  (예상 {exp}, {when})")
    tr = rep.get('traffic')
    if tr:
        def _m(x):
            return '—' if x is None else f'{x:+.1f} m'
        L.append(f"  교통류(관측)  배치 {tr['planned']}대 / 관측 {tr['observed']}대  "
                 f"최근접 종거리 {_m(tr['min_lon_m'])} "
                 f"(앞 {_m(tr['min_ahead_m'])} / 뒤 {_m(None if tr['min_behind_m'] is None else -tr['min_behind_m'])})"
                 + ('   <= [경고] 배치했는데 한 대도 안 보였다' if tr['observed'] == 0 else ''))
    return '\n'.join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='시나리오 이벤트 조우 성립 판정')
    ap.add_argument('scenario_yaml')
    ap.add_argument('log')
    ap.add_argument('--route', default=None,
                    help='route_<name>.pkl — 주면 ego 가 못 간 이벤트의 기대값도 잰다')
    ap.add_argument('--no-map', action='store_true',
                    help='지도를 싣지 않는다 (yaml 에 기대값이 있는 시나리오 전용)')
    a = ap.parse_args(argv)
    lg = None
    if not a.no_map:
        from summarize_run import load_map
        lg, _ = load_map()
    route = None
    if a.route:
        import pickle
        with open(a.route, 'rb') as f:
            route = pickle.load(f)
    rep = check_scenario(a.scenario_yaml, a.log, None, lg, route)
    print(render(rep, pathlib.Path(a.scenario_yaml).stem))
    return 0 if rep['ok'] == rep['total'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
