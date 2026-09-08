"""종료선 콘 게이트 — 좌표 계산의 단일 출처.

주최측 공지(2건) 요약:
  · 종료 지점은 정지선 위 **라바콘 2개**로 표시되고, 뒷바퀴 축이 두 콘 사이
    종료선을 통과하면 완주다.
  · 콘은 **대략적 위치 표시**이고 실제 판정은 운영측이 제공하는 종료 지점
    좌표 기준이다. 콘은 그 좌표에서 도로 t 방향(횡방향)으로 생성된다.

그래서 이 모듈은 **판정에 관여하지 않는다**. 완주 판정은 종전대로
score.detect_finish 가 finish_s(= route.pkl 의 finish_xy 를 경로에 투영한
route_s) 도달로 내리고, 여기서 만드는 것은 두 가지뿐이다:

  · gen_scenarios — VTD XML 에 세울 콘 2개의 월드 좌표 (눈으로 확인용)
  · score        — 리포트에 "두 콘 사이로 지났는가" 표시 (판정 아님, 표시만)

두 도구가 **같은 함수**를 쓴다 — 콘을 세운 자리와 채점기가 그렸다고 믿는
자리가 어긋나면 눈으로 하는 확인 자체가 무의미해진다.

좌표계 두 개를 구분해서 쓴다 (섞으면 조용히 틀린다):
  · **배치 프레임** — 콘을 세우는 기준. 종료 좌표가 *실제로 놓인* 차로
    (lg.locate) 의 t 방향이다. 공지의 "그 좌표에서 도로 t 방향" 그대로.
  · **경로 프레임** — 회랑 여유·통과 표시의 기준. ego 가 달리는 경로
    중심선까지의 lat 이다 (kr_rules 회랑 판정과 같은 축).
둘은 보통 같지만, 마지막 경유점이 경로 차로의 **옆 차로**에 찍힌 경로에서는
차로 하나(약 3 m)만큼 어긋난다. 그래서 lat 은 콘 월드좌표를 경로에 **다시
투영해서** 낸다 — 배치 프레임 값을 경로 프레임 값으로 유용하지 않는다.
"""
from __future__ import annotations

import math
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def project_route(lg, route, x: float, y: float, tangent_ends: bool = False):
    """점 (x,y) → 경로 차로들 중 최근접 투영.

    반환 (dist, lane_key, s_in_lane, t_signed(좌 +), route_s) 또는 None.
    score._route_project 가 이 함수를 그대로 쓴다 — 콘 게이트와 완주 판정
    (finish_s)이 **같은 투영**을 봐야 표시와 판정이 어긋나지 않는다.

    tangent_ends: lanegraph.project 로 그대로 넘긴다. **기본 False 를 바꾸지
    말 것** — 완주(finish_s)·이탈·보행자 판정이 전부 이 값을 쓰고 있고,
    True 로 바꾸면 그 판정들이 같이 움직인다. 콘 기하만 True 로 부른다
    (아래 finish_gate 의 lat 계산 주석 참조).
    """
    best = None
    # 기본 경로에서는 인자를 **넘기지 않는다** — 호출 형태가 종전과 한 글자도
    # 달라지지 않아야 한다. score 의 검출기 테스트들이 project(key, x, y) 만
    # 받는 대역 LaneGraph 를 쓰고 있어서, 무조건 넘기면 그쪽이 TypeError 로
    # 깨진다 (2026-09-07 실제로 깨뜨렸다: test_ped_response 등 3파일).
    kw = {'tangent_ends': True} if tangent_ends else {}
    for i, k in enumerate(route['lanes']):
        try:
            s_p, t_p, d_p, _ = lg.project(tuple(k), x, y, **kw)
        except KeyError:
            continue
        if best is None or d_p < best[0]:
            best = (float(d_p), tuple(k), float(s_p), float(t_p),
                    float(route['cum_s'][i]) + float(s_p))
    return best


def cone_cfg(cfg: dict) -> dict:
    """params.yaml gen_placement.* 의 콘 관련 키만 뽑는다 (기본값 포함)."""
    plc = cfg.get('gen_placement') or {}
    return {'enable': bool(plc.get('finish_cone_enable', True)),
            'model': str(plc.get('cone_model', 'RdMiscPylon03-32cm')),
            'margin_m': float(plc.get('cone_margin_m', 0.3)),
            'half_width_m': float(plc.get('cone_half_width_m', 0.16)),
            'z_offset_m': float(plc.get('cone_z_offset_m', 0.0)),
            'min_gate_from_vehicle': bool(
                plc.get('cone_min_gate_from_vehicle', True)),
            'at_road_edge': bool(plc.get('cone_at_road_edge', True))}


def road_edge_dist(lg, key, x: float, y: float, side: str, cap: int = 6) -> float:
    """차로 중심선 → **같은 방향 차로들의 바깥 경계**까지 횡거리 [m].

    주최측 공지는 "도로 양 끝에 세운 콘" 이다. 차로 반폭으로 세우면 편도
    2차로 이상에서 콘이 도로 한가운데에 서고, 1차로여도 종료 좌표가 차로
    중앙에서 벗어나 있으면 한쪽 콘이 회랑 안으로 들어온다 (실측 아래).

    neighbor() 가 주는 같은 방향 주행 차로를 바깥쪽으로 따라가며 폭을 더한다.
    이웃 차로에서의 s 는 **같은 세계 좌표를 다시 투영해서** 얻는다 — 차로마다
    s 축 원점이 다르므로 s 를 그대로 물려주면 엉뚱한 지점의 폭을 읽는다.
    """
    s0 = lg.project(key, x, y, tangent_ends=True)[0]
    d = float(lg.width_at(key, s0)) / 2.0
    # neighbor 가 없는 대역 LaneGraph 도 있다 (score 검출기 테스트) — 그 경우는
    # 차로 하나짜리 도로로 본다. 있는 API 만 쓰고 조용히 틀리지 않게 한다.
    if not hasattr(lg, 'neighbor'):
        return d
    k = key
    seen = {tuple(key)}
    for _ in range(cap):
        nb = lg.neighbor(k, side)
        if nb is None or tuple(nb) in seen:
            break
        seen.add(tuple(nb))
        try:
            s_nb = lg.project(tuple(nb), x, y, tangent_ends=True)[0]
            d += float(lg.width_at(tuple(nb), s_nb))
        except (KeyError, IndexError):
            break
        k = tuple(nb)
    return d


def corridor_reach(cfg: dict, obj_half_w: float) -> float:
    """회피 회랑 침범 임계 [m] — kr_rules._corridor_blockers 와 **같은 식**.

    reach = 자차반폭 + 객체반폭 + percep.obstacle_clearance_m
    (team_code/kr_rules.py `reach = half_ego + hw + clr`). |lat| < reach 인
    정지 객체만 회랑 침범으로 잡히므로, 콘의 |lat| 이 이 값을 넘으면 회피
    로직은 콘을 장애물로 보지 않는다. 값을 두 곳에 적지 않으려고 상수는
    전부 params 에서 읽는다.
    """
    half_ego = float(cfg['vehicle']['width']) / 2.0
    clr = float((cfg.get('percep') or {}).get('obstacle_clearance_m', 0.3))
    return half_ego + obj_half_w + clr


def _place_lane(lg, route, fx: float, fy: float, route_proj):
    """콘 **배치 프레임** 차로 → (lane_key, s_in_lane).

    공지의 "그 좌표에서 도로 t 방향" 은 종료 좌표가 실제로 놓인 차로 기준이다.
    lg.locate 가 그 차로를 찾고, 못 찾으면(경로 밖 좌표 등) 경로 투영 차로로
    떨어진다 — 어느 쪽이든 콘은 선다.
    """
    try:
        m = lg.locate(fx, fy)
    except Exception:                                        # noqa: BLE001
        m = None
    if m is not None and m.dist <= route_proj[0]:
        return tuple(m.lane), float(m.s)
    return route_proj[1], route_proj[2]


def finish_gate(lg, route: dict, cfg: dict, finish_xy=None) -> dict | None:
    """종료 좌표 → 콘 2개의 위치와 회랑 여유. 투영 실패면 None.

    반환 키:
      finish_xy / finish_s     종료 좌표와 그 route_s (= 완주 판정 기준, 불변)
      place_lane / place_s     배치 프레임 차로와 그 지점
      route_lane / t_finish    경로 프레임 차로와 종료 좌표의 lat (좌 +, 판정 투영)
      t_geom                   같은 lat 을 접선연장 투영으로 잰 값 (기하용)
      lane_width / half_gate   배치 차로 폭, 종료좌표→콘 거리(= 반폭 + margin)
      left / right             콘 월드 좌표 (x, y, z)
      lat_left / lat_right     콘을 **통과 차로(lat_lane)에 투영한** lat (좌 +)
      lat_lane                 그 기준 차로 = 종료 좌표가 붙은 경로 차로
      cone_half_w / reach      콘 bbox 반폭(가정값), 회랑 침범 임계
      clear_left / clear_right |lat| − reach — 양수면 회랑 밖 (장애물 미검출)
      clear_min                둘 중 작은 쪽 (보고·경고 기준)
      lane_mismatch            배치 차로 ≠ 경로 차로 (경유점이 옆 차로에 찍힘)
      gate_floored             테이퍼 차로라 게이트 폭을 차폭으로 받쳤는가
    """
    fxy = finish_xy if finish_xy is not None else route.get('finish_xy')
    if not fxy:
        return None
    fx, fy = float(fxy[0]), float(fxy[1])
    pr = project_route(lg, route, fx, fy)
    if pr is None:
        return None
    _dist, k_route, sl_route, t_f, finish_s = pr
    k_pl, sl_pl = _place_lane(lg, route, fx, fy, pr)
    _x, _y, z, h = lg.point_at(k_pl, sl_pl)
    cc = cone_cfg(cfg)
    width = float(lg.width_at(k_pl, sl_pl))
    half_gate = width / 2.0 + cc['margin_m']
    # 소멸 직전 테이퍼 차로가 배치 차로로 뽑히면(이 맵에는 폭 0.11 m·0.0 m 차로가
    # 실재한다 — params gen_placement.static_vehicle 주석) 차폭보다 좁은 게이트가
    # 나온다: 실측 실전주행_교통류_16_직진20 은 폭 0.75 m → 콘 간격 0.27 m 로
    # 차(1.886 m)가 못 지나간다. 콘은 "통과를 눈으로 확인" 하는 물건이라 지나갈
    # 수 없는 게이트는 무의미하므로, 안쪽 폭이 차폭 밑으로 내려가지 않게 받친다.
    # false 로 두면 차로 반폭만 쓰는 이전 동작이 그대로 재현된다.
    gate_floored = False
    if cc['min_gate_from_vehicle']:
        floor = float(cfg['vehicle']['width']) / 2.0 + cc['half_width_m'] + cc['margin_m']
        if floor > half_gate:
            half_gate, gate_floored = floor, True
    # 좌(+t) 단위벡터 = (−sin h, cos h) — gen_scenarios.route_pt 와 같은 규약.
    dx, dy = -math.sin(h), math.cos(h)
    zc = z + cc['z_offset_m']
    # ── 콘 위치 (2026-09-08) ──────────────────────────────────────────────
    # 기본(at_road_edge): 주최측 공지대로 **도로 양 끝**에 세운다. 기준점이
    # 종료 좌표가 아니라 **배치 차로 중심선**이고, 좌우 각각 그 방향 마지막
    # 차로의 바깥 경계 + margin 이다. 좌우 거리가 다를 수 있어 비대칭이다.
    #
    # 왜 바꿨나 — 실측 실경로_01_PathShape03 (logs/batch/20260908_130919):
    # 마지막 좌회전 연결로 (1502,0,-1) 은 폭 2.4 m 이고 종료 좌표가 중심선에서
    # t=+0.406 m 치우쳐 있었다. 종료 좌표 기준 대칭 배치(half_gate 1.5)는
    # 오른쪽 콘을 lat −1.094 에 놓았는데 회랑 임계 reach 는 1.403 이라 여유가
    # **−0.309 m**, 즉 콘이 회랑 안이다. 자차는 그 콘을 정적 장애물로 잡아
    # 종료선 17 m 앞에서 영구 정지했다(미완주). 도로 끝 기준이면 양쪽 콘이
    # lat ±1.5 로 서서 여유 +0.097 로 바뀐다.
    # false = 이전 동작(종료 좌표 기준 대칭).
    if cc['at_road_edge']:
        d_l = road_edge_dist(lg, k_pl, fx, fy, 'left') + cc['margin_m']
        d_r = road_edge_dist(lg, k_pl, fx, fy, 'right') + cc['margin_m']
        if cc['min_gate_from_vehicle']:
            floor = float(cfg['vehicle']['width']) / 2.0 + cc['half_width_m'] \
                + cc['margin_m']
            if floor > d_l or floor > d_r:
                d_l, d_r, gate_floored = max(d_l, floor), max(d_r, floor), True
        half_gate = (d_l + d_r) / 2.0                # 보고용 대표값
        left = (_x + d_l * dx, _y + d_l * dy, zc)
        right = (_x - d_r * dx, _y - d_r * dy, zc)
    else:
        left = (fx + half_gate * dx, fy + half_gate * dy, zc)
        right = (fx - half_gate * dx, fy - half_gate * dy, zc)
    # lat 은 배치 프레임 값을 유용하지 않고 **다시 투영**해서 낸다 (배치 차로와
    # 경로 차로가 다를 수 있다 — 모듈 도입부 참조). 기준 차로는 종료 좌표가
    # 붙은 **경로 차로 하나**(k_route)로 고정한다.
    #
    # 경로 전체에서 최근접 차로를 고르면(project_route) 안 된다: 경로가 종료선
    # 직전에 차선변경으로 끝나면 두 차로가 route['lanes'] 에 **둘 다** 들어 있고
    # 횡으로 겹쳐서, 콘이 ego 가 통과 시점에 있지도 않은 옆 차로에 투영된다.
    # 실측 2026-09-07 실전주행_교통류_03_우회전11: 경로 끝이 (429,5,-4) →
    # (429,5,-3) 인데 우측 콘이 -4 에 붙어 lat +1.20 (여유 −0.20) 으로 나왔다.
    # 실제로는 통과 차로 -3 기준 −1.80 (여유 +0.40) 이다. kr_rules 의 회랑 판정도
    # ego 근방 경로 폴리라인에 투영하지 전체 차로에서 최근접을 고르지 않는다.
    #
    # tangent_ends=True: 종료 좌표가 그 차로 폴리라인 **끝점 밖**인 경로가 실재
    # 하고(실측 16_직진20 — 종료 좌표·콘 두 개가 전부 같은 s 로 클램프됐다),
    # 기본 클램프는 종방향 부족분을 t 에 섞어 lat 을 망가뜨린다: 세계좌표로
    # 2.806 m 떨어진 콘의 lat 차가 1.202 m 로 나왔다.
    # **판정용 finish_s 는 기본(False) 투영 그대로** 두고 여기 기하만 바꾼다.
    def _lat(px, py):
        return float(lg.project(k_route, px, py, tangent_ends=True)[1])

    lat_l, lat_r, t_geom = _lat(left[0], left[1]), _lat(right[0], right[1]), _lat(fx, fy)
    reach = corridor_reach(cfg, cc['half_width_m'])
    return {'finish_xy': (fx, fy), 'finish_s': finish_s,
            'place_lane': k_pl, 'place_s': sl_pl, 'hdg': h,
            'route_lane': k_route, 'route_s_in_lane': sl_route,
            't_finish': t_f, 't_geom': t_geom,
            'lane_width': width, 'half_gate': half_gate,
            'left': left, 'right': right,
            'lat_left': lat_l, 'lat_right': lat_r, 'lat_lane': k_route,
            'cone_half_w': cc['half_width_m'], 'reach': reach,
            'clear_left': abs(lat_l) - reach, 'clear_right': abs(lat_r) - reach,
            'clear_min': min(abs(lat_l), abs(lat_r)) - reach,
            'lane_mismatch': k_pl != k_route, 'gate_floored': gate_floored,
            'model': cc['model'], 'enable': cc['enable']}


def lat_of(lg, gate: dict, x: float, y: float) -> float:
    """점 (x,y) 의 lat — **콘 lat 과 같은 자**로 잰다 (통과 차로 tangent 투영).

    채점기가 통과 시점 ego 위치를 이 함수로 재야 gate_span 과 같은 축이 된다.
    """
    return float(lg.project(gate['lat_lane'], x, y, tangent_ends=True)[1])


def gate_span(gate: dict) -> tuple[float, float]:
    """두 콘 **안쪽 면** 사이의 lat 구간 (경로 프레임, 좌 +).

    콘 중심까지가 lat_left/lat_right 이고 콘에도 두께가 있으므로 반폭을 뺀다.
    구간이 뒤집히면(콘 간격 < 콘 두께) 빈 구간 대신 중심 한 점을 준다.
    """
    lo, hi = sorted((gate['lat_left'], gate['lat_right']))
    lo, hi = lo + gate['cone_half_w'], hi - gate['cone_half_w']
    if lo > hi:
        mid = (lo + hi) / 2.0
        return mid, mid
    return lo, hi


def between_cones(gate: dict, lat: float) -> bool:
    """경로 프레임 횡오프셋 lat 이 두 콘 사이인가.

    **판정이 아니라 표시**용이다 — 완주 여부는 종전대로 finish_s 로만 낸다.
    lat 은 뒷축 기준 점(ego x/y = 뒷축 중심, AGENT_SPEC §1.3)을 경로에 투영한
    값이라 차폭은 넣지 않는다 — "콘 사이를 지났는가" 이지 "차체가 닿았는가"
    가 아니다.
    """
    lo, hi = gate_span(gate)
    return lo <= lat <= hi
