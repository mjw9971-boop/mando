"""
경로·신호 대응 유틸 (phase1: 신호 대조만. phase2 에서 VtdRoutePlanner 가 여기 추가된다).

9910 light_id ↔ 정지선 controller 매핑은 기존 perception.py 에서 검증된 로직 이식.
"""
from __future__ import annotations

from .lanegraph import LaneGraph
from .types import RawPacket


def stop_line_controllers(lg: LaneGraph, item) -> list:
    """lookahead 의 stop_line 항목 → 그 정지선의 controller_ids."""
    rec = lg.lanes.get(item.lane)
    if rec is None:
        return []
    best, best_d = None, None
    for sl in rec.get('stop_lines', []):
        d = abs(float(sl['s']) - float(item.s_in_lane))
        if best_d is None or d < best_d:
            best, best_d = sl, d
    if best is None or best_d is None or best_d > 1.0:
        return []
    return list(best.get('controller_ids') or [])


def check_light_controller(lg: LaneGraph, pkt: RawPacket, ahead: list, flags: dict) -> None:
    """
    9910 의 light_id 가 전방 정지선의 controller 목록에 들어있는지 확인한다.

    9910 light_id 는 **xodr <controller> id** 다 (개별 signal id 가 아니다).
    실측: 정지선 signal_ids=[101..106] 인 곳에서 9910 이 id=27 을 줬고,
    ctrl027 이 제어하는 신호가 101/102/105 였다. 즉 계층이 다를 뿐 정상이다.

    **판단에는 쓰지 않는다.** 주행 로직은 "가장 가까운 전방 정지선 + 현재 state"
    를 그대로 쓰고, 여기서는 규약이 맞는지 관측만 해서 flags 에 남긴다.
    불일치해도 주행은 계속된다. (phase3: VtdTrafficLight.state 갱신이 같은
    매핑을 쓴다 — junction_ctrl_map.json 폴백 포함)
    """
    if not pkt.lights:
        return
    light_id = int(pkt.lights[0][0])
    flags['light_id'] = light_id

    item = next((a for a in ahead if a.kind == 'stop_line'), None)
    if item is None:
        flags['light_ctrl_match'] = None      # 대조할 전방 정지선이 없다
        return

    cids = stop_line_controllers(lg, item)
    flags['stop_ctrl_ids'] = cids
    if not cids:
        # 신호 없는 정지선(일단정지/양보)이거나 controller 매핑이 없는 경우
        flags['light_ctrl_match'] = None
        return

    ok = light_id in cids
    flags['light_ctrl_match'] = ok
    if not ok:
        flags['light_ctrl_mismatch'] = f'{light_id} not in {cids}'


# ═══════════════════════════════════════════════════════════════════════════
# VtdRoutePlanner — PDM-Lite privileged_route_planner 의 VTD 구현 (phase2)
# ═══════════════════════════════════════════════════════════════════════════
import math as _math

import numpy as np
from scipy.spatial import cKDTree

from . import frame
from .carla_types import RoadOption, TrafficLightState
from .map import VtdWaypoint

# 9910 신호 state → CARLA TrafficLightState (phase2 기본 매핑).
# 좌회전 화살표(4)의 경로 방향 조건부 처리는 phase3 §0-5 에서 얹는다.
LIGHT_STATE_MAP = {
    0: TrafficLightState.Unknown,   # 미할당
    1: TrafficLightState.Red,
    2: TrafficLightState.Yellow,
    3: TrafficLightState.Green,
    4: TrafficLightState.Red,       # 좌회전 화살표 — 직진/우회전엔 적색과 동등
    5: TrafficLightState.Green,     # 녹색+좌
    6: TrafficLightState.Green,     # 점멸 — 기존 flash_mode 'yield' 관례
}


def lg_neighbor_missing(lg, key, side) -> bool:
    return lg.neighbor(key, side) is None


class VtdTrafficLight:
    """carla.TrafficLight 의 PDM-Lite 사용 표면: id / state / type_id.

    실체는 "신호 걸린 정지선" 이다 — 거리 기준을 정지선으로 쓰기 위해서다
    (한국 교차로는 신호등이 정지선에서 25 m 이상 떨어져 있다: phase0 §0-5).
    state 는 매 틱 VtdRoutePlanner.update_lights() 가 9910 으로 갱신한다.
    """

    def __init__(self, controller_ids: list, signal_ids: list, route_s: float) -> None:
        self.controller_ids = list(controller_ids)
        self.signal_ids = list(signal_ids)
        self.route_s = float(route_s)                    # 정지선의 경로 누적거리
        self.id = int(controller_ids[0]) if controller_ids else -1
        self.type_id = 'traffic.traffic_light'
        self.state = TrafficLightState.Green             # 미수신 = 진행 (phase0 §0-5)

    def __repr__(self) -> str:
        return (f'VtdTrafficLight(ctrl={self.controller_ids}, '
                f'rs={self.route_s:.1f}, {self.state.name})')


class VtdRoutePlanner:
    """
    PDM-Lite PrivilegedRoutePlanner 와 같은 표면. run_step(pos) 반환 8-튜플의
    순서·타입은 autopilot._get_control 의 언패킹과 글자 단위로 같다:

        (route_points[idx:],                 np.ndarray [N,3]  (CARLA 프레임)
         route_waypoints[idx:],              list[VtdWaypoint]
         commands[idx:],                     np.ndarray [N]    (RoadOption 값)
         distances_to_next_traffic_lights[idx],   float  [m, 정지선까지]
         next_traffic_lights[idx],           VtdTrafficLight | None
         distances_to_next_stop_signs[idx],  float  (항상 inf — 코스에 정지표지 없음)
         next_stop_signs[idx],               None
         speed_limits[idx])                  float  [m/s, margin 반영]

    route dict(lanes/cum_s/events)를 lanegraph 로 1/points_per_meter 간격 재샘플.
    차선변경 이음매(나란한 이웃 차로, cum_s 동일)는 window_s0 부터 최대
    LC_TRANSITION_M 에 걸쳐 cos 블렌드로 옆 차로에 붙인다 — 그 구간의 command 가
    CHANGELANELEFT/RIGHT 다.
    """

    LC_TRANSITION_M = 25.0      # 차선변경 블렌드 길이 상한 (기존 transition_min 20 + 여유)

    def __init__(self, lg, route: dict, cfg: dict, config=None) -> None:
        self.lg = lg
        self.route = route
        self.cfg = cfg
        # PDM GlobalConfig 가 오면 그 값을, 아니면 PDM 원본 기본값을 쓴다 (phase3 주입)
        g = lambda name, default: getattr(config, name, default) if config is not None else default
        self.points_per_meter = int(g('points_per_meter', 10))
        ppm = self.points_per_meter
        self.ego_vehicles_route_point_search_distance = int(g('ego_vehicles_route_point_search_distance', 4 * ppm))
        self.leading_vehicles_max_route_distance = g('leading_vehicles_max_route_distance', 2.5)
        self.leading_vehicles_max_route_angle_distance = g('leading_vehicles_max_route_angle_distance', 35.0)
        self.leading_vehicles_maximum_detection_radius = int(g('leading_vehicles_maximum_detection_radius', 80 * ppm))
        self.trailing_vehicles_max_route_distance = g('trailing_vehicles_max_route_distance', 3.0)
        self.trailing_vehicles_max_route_distance_lane_change = g('trailing_vehicles_max_route_distance_lane_change', 6.0)
        self.tailing_vehicles_maximum_detection_radius = int(g('tailing_vehicles_maximum_detection_radius', 80 * ppm))
        self.max_distance_lane_change_trailing_vehicles = int(g('max_distance_lane_change_trailing_vehicles', 15 * ppm))
        self.transition_smoothness_distance = int(g('transition_smoothness_distance', 8 * ppm))
        self.extra_route_length = int(g('extra_route_length', 50))
        # 테이퍼 꼬리 블렌드 (params 가 단일 출처 — 없으면 KeyError 로 죽는 게 맞다)
        self.taper_blend_m = float(cfg['route']['taper_blend_m'])
        rt_cfg = cfg['route']
        self.lc_move_s = float(rt_cfg['lc_move_s'])
        self.lc_move_min_m = float(rt_cfg['lc_move_min_m'])
        self.lc_move_max_m = float(rt_cfg['lc_move_max_m'])
        self.veh_width = float(cfg['vehicle']['width'])
        # 정적 장애물 인식 (compute_leading_vehicles) — params 가 단일 출처
        self.obstacle_speed_max = float(cfg['percep']['obstacle_speed_max'])
        self.obstacle_clearance_m = float(cfg['percep']['obstacle_clearance_m'])
        # 시프트 목표 차로의 기준을 '원 경로 차로' 에서 '현재 경로가 놓인 차로' 로
        # 옮긴다 (_shift_target_steps 참조). false = PDM 원문 그대로(원 경로 기준).
        _ot = cfg.get('overtake', {})
        self.shift_target_current = bool(_ot.get('shift_target_current_lane_enable', False))
        self.shift_target_max_steps = int(_ot.get('shift_target_max_steps', 2))
        # plan_shift_span 의 최근접 경로점 탐색을 자차 앞 창으로 제한한다 (아래 참조).
        # false = PDM 원문 그대로(앞쪽 전 구간 탐색).
        self.span_search_local = bool(
            cfg.get('overtake', {}).get('span_search_local_enable', False))

        self.route_index = 0
        self.last_route_index = 0
        self.lc_solid_warnings: list = []   # 점선 없는 차선변경 (경로 결함)
        self.lc_clipped: list = []          # 점선 안으로 좁힌 차선변경 구간
        self._build()

    def _ramp_is_dashed(self, lanes, cum_s, lens, w0: float, w1: float,
                        side: str, step: float = 1.0) -> bool:
        """램프 [w0, w1] 전 구간이 점선인가 — 차로 경계를 넘어 확인한다.

        build_route 의 창은 이미 연속 점선을 보장하지만(lane_change_window), 손으로
        만든·옛 route.pkl 은 그 게이트를 안 탄다. 램프를 창 앞쪽으로 옮기면 확인
        범위가 여러 차로에 걸치므로 _clip_to_dashed(한 차로 기준)로는 부족하다.
        """
        rs = float(w0)
        while rs < w1 - 1e-6:
            j = None
            for k in range(len(lanes)):
                L = float(lens[k]) if k < len(lens) else 0.0
                if cum_s[k] - 1e-6 <= rs < cum_s[k] + L:
                    j = k
                    break
            if j is None:
                return False
            key = tuple(lanes[j])
            if lg_neighbor_missing(self.lg, key, side):
                return False
            if not self.lg.lane_change_ok(key, rs - cum_s[j], side):
                return False
            rs += step
        return True

    def _road_entry_s(self, lanes, cum_s, i_hop) -> float:
        """hop 차로가 속한 **도로에 route 가 진입한 지점** [route_s].

        laneSection 은 도로 안에서만 쪼개지므로 road_id 가 같은 동안 뒤로 거슬러
        올라가면 그 도로의 진입로가 나온다.
        """
        road = tuple(lanes[i_hop])[0]
        k = i_hop
        while k > 0 and tuple(lanes[k - 1])[0] == road:
            k -= 1
        return float(cum_s[k])

    def _lc_move_len(self, key) -> float:
        """차선변경에 쓸 거리 [m] = 계획 속도 x lc_move_s (하한·상한 클램프)."""
        kph, _sc = self.lg.speed_limit_at(key)
        kph = float(kph) if kph is not None else float(self.cfg.get('default_speed_kph', 50.0))
        v = max(0.0, kph - float(self.cfg.get('speed', {}).get('margin_kph', 0.0))) / 3.6
        return min(self.lc_move_max_m, max(self.lc_move_min_m, v * self.lc_move_s))

    def _build_lc_ramps(self, lanes, cum_s, lens, lc_events) -> dict:
        """차선변경 램프 → {hop_index: ramp}.

        **도로에 진입하면 곧바로, lc_move_s 안에 옮겨탄다.** 진출로가 회전이면
        회전 차로에 미리 붙어 있어야 하는데, 종전에는 hop 차로 안에서만 블렌드해
        회전 직전 17 m 에 몰렸다 (실측: 창 93 m 중 뒤 17 m 만 사용, 차선변경과
        회전이 겹침).

        시작 = max(도로 진입로, 창 시작) — 점선이 허용하는 가장 이른 지점.
        길이 = 계획 속도 x lc_move_s (지시등 선행 signal.lc_lead_s 와 같은 축).
        옮겨탄 뒤 남은 거리는 그 차로에서 그대로 주행한다.
        """
        ramps: dict = {}
        for ev in lc_events:
            frm, to = tuple(ev['from_lane']), tuple(ev['to_lane'])
            i_hop = next((k for k in range(len(lanes) - 1)
                          if tuple(lanes[k]) == frm and tuple(lanes[k + 1]) == to), None)
            if i_hop is None:
                continue
            left = to == self.lg.neighbor(frm, 'left')
            side = 'left' if left else 'right'
            w1_cap = float(ev.get('window_s1', cum_s[i_hop]))
            # 도로 진입로부터 — 단 점선이 허용하는 가장 이른 지점 이후여야 한다
            w0 = max(float(ev.get('window_s0', cum_s[i_hop])),
                     self._road_entry_s(lanes, cum_s, i_hop))
            w1 = min(w0 + self._lc_move_len(frm), w1_cap)
            if w1 - w0 > 1e-6 and self._ramp_is_dashed(lanes, cum_s, lens, w0, w1, side):
                ramps[i_hop] = {'w0': w0, 'w1': w1, 'side': side,
                                'sign': 1.0 if left else -1.0, 'i_hop': i_hop}
            else:
                # 확인 실패 → 기존 동작(hop 차로 안에서 블렌드 + 점선 클립)으로 폴백
                w0f = max(float(ev.get('window_s0', cum_s[i_hop])), cum_s[i_hop])
                w0f, w1f = self._clip_to_dashed(frm, cum_s[i_hop], w0f, w1_cap, side)
                ramps[i_hop] = {'w0': w0f, 'w1': min(w0f + self._lc_move_len(frm), w1f),
                                'side': side, 'sign': 1.0 if left else -1.0,
                                'i_hop': i_hop, 'fallback': True}
        return ramps

    def _clip_to_dashed(self, key, base: float, w0: float, w1: float,
                        side: str) -> tuple:
        """차선변경 블렌드 구간 [w0, w1](route_s)을 점선 구간 안으로 좁힌다.

        같은 차로쌍 안에서 **넘는 위치만** 옮긴다 — 출발/도착 차로는 그대로라
        경로가 달라지지 않는다. 전이거리(LC_TRANSITION_M)를 채우는 가장 이른
        점선 구간을 고르고, 그런 구간이 없으면 가장 긴 것을 쓴다.
        겹치는 점선이 아예 없으면(= 경로가 실선 횡단을 요구) 원본을 그대로 두고
        경고만 남긴다 — 임의로 창을 옮기면 목표 차로에 못 닿는다.
        """
        runs = [(base + a, base + b) for a, b in self.lg.dashed_runs(key, side)]
        overlaps = [(max(w0, a), min(w1, b)) for a, b in runs
                    if min(w1, b) - max(w0, a) > 1e-6]
        if not overlaps:
            msg = (f'[route] ⚠ S2.2.05 실선 차선변경 — lane {key} {side} 구간 '
                   f'{w0:.1f}~{w1:.1f} m 에 점선이 없다. 경로를 다시 만들 것 '
                   f'(build_route 가 이 hop 을 만들었을 리 없다).')
            print(msg, flush=True)
            self.lc_solid_warnings.append({'lane': key, 'side': side,
                                           'w0': w0, 'w1': w1})
            return w0, w1
        long_enough = [o for o in overlaps if o[1] - o[0] >= self.LC_TRANSITION_M]
        pick = long_enough[0] if long_enough else max(overlaps, key=lambda o: o[1] - o[0])
        if pick != (w0, w1):
            self.lc_clipped.append({'lane': key, 'side': side,
                                    'from': (w0, w1), 'to': pick})
        return pick

    # ── 경로 재샘플 ───────────────────────────────────────────────────────
    def _build(self) -> None:
        lg, route = self.lg, self.route
        lanes = route['lanes']
        cum_s = [float(v) for v in route['cum_s']]
        step = 1.0 / self.points_per_meter

        # 차선변경 이벤트 풀: (from, to) -> (window_s0, window_s1)
        lc_events = []
        for e in route.get('events', []):
            if e['kind'].startswith('lane_change') and e.get('to_lane'):
                lc_events.append(dict(e))

        pts: list[tuple] = []          # (x, y, z)  VTD 프레임
        keys: list[tuple] = []         # (lane_key, s_in_lane)
        cmds: list[int] = []
        rs_list: list[float] = []
        limits: list[float] = []
        lat: list[float] = []          # 누적 횡오프셋 (+좌 / −우)
        # 차선변경이 끝나면 기준 차로가 바뀐다. 여기서 0 으로 되돌리면 앞을 볼 때
        # "반대로 돌아온다" 로 읽혀 지시등이 역방향으로 켜진다(2026-08-28 재생에서
        # 확인). 그래서 **누적값**으로 둔다 — 판정은 차이만 보므로 절대값은 무의미하다.
        lat_acc = 0.0
        stops: list[tuple] = []        # (rs, controller_ids, signal_ids)

        lengths = [float(v) for v in route.get('lengths') or
                   [lg.length(tuple(k)) for k in lanes]]
        ramps = self._build_lc_ramps(lanes, cum_s, lengths, lc_events)

        margin_kph = float(self.cfg.get('speed', {}).get('margin_kph', 0.0))
        default_kph = float(self.cfg.get('default_speed_kph', 50.0))
        carry_kph: float | None = None

        red_no_carry = bool(self.cfg.get('speed', {})
                            .get('red_limit_no_carry', False))

        def limit_mps(key, s_in_lane=None) -> float:
            """경로점 하나의 목표 상한 [m/s].

            `s_in_lane` 을 주면 붉은 구간(red_spans)이 s 로 반영된다 — 같은
            차로 안에서도 구간 안팎이 갈린다. carry 규칙은 그대로다.

            `speed.red_limit_no_carry` 를 켜면 carry 를 `road_limit_at`
            (표시값, 구역 값 제외) 으로 채우고 구역 값은 그 위에 min 으로
            얹는다 — 구역의 30 이 붉지 않은 다음 도로까지 따라가지 않는다.
            끄면 아래 else 가 그대로 돌아 이전과 완전히 같다.
            """
            nonlocal carry_kph
            v, sc = lg.speed_limit_at(key, s_in_lane)
            if red_no_carry:
                rv, _rsc = lg.road_limit_at(key)
                if rv is not None:
                    carry_kph = float(rv)
                kph = carry_kph if carry_kph is not None else default_kph
                if sc and v is not None:
                    kph = min(kph, float(v))
            else:
                if v is not None:
                    carry_kph = float(v)
                kph = carry_kph if carry_kph is not None else default_kph
            return max(0.0, kph - margin_kph) / 3.6

        def collect_stops(key, i, s_from, s_to) -> None:
            for sl in lg.lanes[key].get('stop_lines', []):
                if s_from - 1e-6 <= sl['s'] <= s_to + 1e-6:
                    if sl.get('controller_ids') or sl.get('signal_ids'):
                        stops.append((cum_s[i] + float(sl['s']),
                                      sl.get('controller_ids') or [],
                                      sl.get('signal_ids') or []))

        i = 0
        s = float(route.get('start_s_in_lane', 0.0))
        rs = cum_s[0] + s
        while i < len(lanes):
            key = lanes[i]
            L = lg.length(key)
            nxt = lanes[i + 1] if i + 1 < len(lanes) else None
            hop = (nxt is not None and nxt not in lg.successors(key)
                   and nxt in (lg.neighbor(key, 'left'), lg.neighbor(key, 'right')))
            # 차선변경 램프: hop 차로 안이 아니라 **창 앞쪽**에서 시작한다.
            # 램프가 이 차로를 지나가면(hop 이전 차로 포함) 그 차로의 side 이웃으로
            # 민다 — 이웃들이 successor 로 이어져 목표 차로에 닿는 것은 창 계산
            # (lane_change_window)이 이미 확인했다.
            # 램프는 hop 까지 **살아 있다**. w1 이후에는 가중치 1 로 고정돼 목표
            # 차로 중심을 그대로 따라간다 — 여기서 램프를 끄면 원 차로로 되돌아가
            # 옮겨탄 게 취소된다 (2026-08-28: 횡이동 0.00 으로 확인).
            ramp = next((r for r in ramps.values()
                         if i <= r['i_hop'] and r['w0'] < cum_s[i] + L), None)
            blend = None
            if ramp is not None:
                tgt = lg.neighbor(key, ramp['side'])
                if tgt is not None:
                    side = (RoadOption.CHANGELANELEFT if ramp['side'] == 'left'
                            else RoadOption.CHANGELANERIGHT)
                    blend = (ramp['w0'], max(ramp['w1'], ramp['w0'] + 1e-6),
                             side, ramp['sign'], tgt)
            if hop and ramp is not None and ramp['w1'] > cum_s[i] + 1e-6:
                s_end = min(L, ramp['w1'] - cum_s[i])   # 램프가 이 차로 안에서 끝난다
            else:
                s_end = L                               # 이미 목표 차로 위 — 끝까지

            # 소멸(테이퍼) 차로 꼬리: 끝 폭이 차폭 미만이면 중심선이 폭과 함께
            # 이웃 경계로 수렴하다 successor 중심선(반폭 ≈1.5 m 옆)으로 순간이동
            # 한다 — 2026-08-26 실기: (2192,3,4)→(2192,2,3) 1.50 m 킹크에 룩어헤드
            # 2.5 m 짜리 lateral PID 가 한 틱 만에 풀포화, ±1.5 m 진동으로 차선이탈.
            # 마지막 taper_blend_m 구간에 successor 시작점과의 오프셋을 코사인
            # 가중(차선변경 blend 와 같은 방식)으로 얹어 폴리라인을 연속으로 만든다.
            taper = None
            if (not hop and nxt is not None and nxt in lg.successors(key)
                    and lg.width_at(key, L) < self.veh_width):
                p_end = np.array(lg.point_at(key, L)[:3], dtype=float)
                p_nxt = np.array(lg.point_at(nxt, 0.0)[:3], dtype=float)
                taper = (max(0.0, L - self.taper_blend_m), p_nxt - p_end)

            s_seg_start = s
            while s < s_end - 1e-9:
                x, y, z, _h = lg.point_at(key, s)
                if taper is not None and s >= taper[0]:
                    wt = -_math.cos(min(1.0, (s - taper[0]) / max(L - taper[0], 1e-6))
                                    * _math.pi) / 2.0 + 0.5
                    x += wt * taper[1][0]
                    y += wt * taper[1][1]
                    z += wt * taper[1][2]
                cmd = RoadOption.LANEFOLLOW
                cur_key, cur_s = key, s
                off = lat_acc
                if blend is not None and rs >= blend[0]:
                    tgt = blend[4]
                    u = min(1.0, max(0.0, (rs - blend[0]) / (blend[1] - blend[0])))
                    w = -_math.cos(u * _math.pi) / 2.0 + 0.5
                    x2, y2, z2, _h2 = lg.point_at(tgt, min(s, lg.length(tgt)))
                    # 지시등 입력: 원 차로 중심에서 옆으로 밀린 양 (테이퍼는 제외 —
                    # 차로를 옮기는 게 아니라 소멸 차로 기하 보정이라 지시등 대상이
                    # 아니다. 실측: 테이퍼 최대 2.36 m 로 LC 3.0 m 와 임계로는 못 가른다)
                    off = lat_acc + blend[3] * w * _math.hypot(x2 - x, y2 - y)
                    x = (1.0 - w) * x + w * x2
                    y = (1.0 - w) * y + w * y2
                    z = (1.0 - w) * z + w * z2
                    cmd = blend[2]
                    if w >= 0.5:
                        cur_key, cur_s = tgt, min(s, lg.length(tgt))
                pts.append((x, y, z))
                keys.append((cur_key, cur_s))
                cmds.append(int(cmd))
                rs_list.append(rs)
                limits.append(limit_mps(cur_key, cur_s))
                lat.append(off)
                s += step
                rs += step
            collect_stops(key, i, s_seg_start, s_end)

            if hop:
                lat_acc = lat[-1] if lat else lat_acc   # 옮겨탄 차로가 새 기준
                s = min(s, lg.length(nxt))     # 나란한 차로 — s 매개화 유지, rs 연속
            else:
                s -= L                          # 초과분 이월 → 간격 유지
            i += 1

        # ── 꼬리 연장 (extra_route_length) — 경로 끝에서도 checkpoint 소진 방지 ──
        # 본 루프가 마지막 차로를 끝까지 샘플하고 s -= L 를 이월했으므로,
        # s 는 이미 **successor 안의 오프셋**이다. successor 부터 이어 간다.
        key = lanes[-1]
        nx0 = lg.successors(key)
        key = nx0[0] if nx0 else None
        ext = 0.0
        while key is not None and ext < self.extra_route_length:
            nx = lg.successors(key)
            if s >= lg.length(key) - 1e-9:
                if not nx:
                    break
                s -= lg.length(key)
                key = nx[0]
                continue
            x, y, z, _h = lg.point_at(key, s)
            pts.append((x, y, z))
            keys.append((key, s))
            cmds.append(int(RoadOption.LANEFOLLOW))
            rs_list.append(rs)
            limits.append(limit_mps(key, s))
            lat.append(lat_acc)
            s += step
            rs += step
            ext += step

        # ── 배열화 (CARLA 프레임) ────────────────────────────────────────
        self.route_points = frame.to_carla_np(np.array(pts, dtype=float))
        self.original_route_points = np.copy(self.route_points)
        self.route_s = np.array(rs_list, dtype=float)
        self.commands = np.array(cmds, dtype=int)
        self.commands_orig = np.copy(self.commands)
        self.speed_limits = np.array(limits, dtype=float)
        # 지시등 판단 입력 (kr_rules). 런타임 시프트가 얹히면 갱신된다.
        self.lat_shift = np.array(lat, dtype=float)
        self._lat_build = np.copy(self.lat_shift)
        self.route_waypoints = [VtdWaypoint(lg, k, sv) for k, sv in keys]
        self.rotation_angles = self.compute_rotation_angles(self.route_points)
        self._kd = cKDTree(self.route_points[:, :2])

        # ── 신호 정지선 → 인덱스별 거리/객체 배열 ────────────────────────
        stops.sort(key=lambda t: t[0])
        dedup: list[tuple] = []
        for rs_stop, ctrl, sigs in stops:                 # 나란한 차로 중복 제거 (0.5 m)
            if dedup and rs_stop - dedup[-1][0] < 0.5:
                continue
            dedup.append((rs_stop, ctrl, sigs))
        self.traffic_lights = [VtdTrafficLight(c, sg, r) for r, c, sg in dedup]

        n = len(self.route_points)
        self.distances_to_next_traffic_lights = np.full(n, np.inf)
        self.next_traffic_lights: list = [None] * n
        self.distances_to_next_stop_signs = np.full(n, np.inf)
        self.next_stop_signs: list = [None] * n           # 코스에 정지표지 없음
        ti = 0
        for idx in range(n):
            while ti < len(self.traffic_lights) and \
                    self.traffic_lights[ti].route_s < self.route_s[idx] - 0.5:
                ti += 1
            if ti < len(self.traffic_lights):
                self.next_traffic_lights[idx] = self.traffic_lights[ti]
                self.distances_to_next_traffic_lights[idx] = \
                    self.traffic_lights[ti].route_s - self.route_s[idx]

    # ── 매 틱 ─────────────────────────────────────────────────────────────
    def run_step(self, agent_position):
        """PDM 원문과 동일 — 전방 창에서 최근접 점으로 인덱스 전진 후 슬라이스 반환."""
        till = self.ego_vehicles_route_point_search_distance
        search_range = min(self.route_index + till, self.route_points.shape[0])

        self.route_index += int(np.argmin(np.linalg.norm(
            np.asarray(agent_position)[None, :2]
            - self.route_points[self.route_index:search_range, :2], axis=1)))

        idx = self.route_index
        return (
            self.route_points[idx:],
            self.route_waypoints[idx:],
            self.commands[idx:],
            self.distances_to_next_traffic_lights[idx],
            self.next_traffic_lights[idx],
            self.distances_to_next_stop_signs[idx],
            self.next_stop_signs[idx],
            self.speed_limits[idx],
        )

    def save(self) -> None:
        self.last_route_index = self.route_index

    def load(self) -> None:
        self.route_index = self.last_route_index

    def reset_index(self, agent_position) -> None:
        """courseRespawn 뒤 — 경로상 위치가 뒤로 갈 수 있어 전역 재탐색한다."""
        _d, idx = self._kd.query(np.asarray(agent_position)[:2], k=1)
        self.route_index = int(idx)
        self.last_route_index = self.route_index

    def update_lights(self, lights: list) -> None:
        """9910 lights [(id, state)] → 매칭되는 정지선 신호의 state 갱신.

        9910 id 는 controller id 다 (junction_ctrl_map / controller_ids 로 검증됨).
        매칭 안 된 신호는 직전 state 유지 — 9910 은 '지금 볼 신호' 하나만 주므로
        멀어진 신호의 낡은 state 가 남지만, 판단은 next_traffic_light(전방 첫
        정지선)만 보므로 무해하다.
        """
        for lid, state in lights:
            mapped = LIGHT_STATE_MAP.get(int(state), TrafficLightState.Unknown)
            for tl in self.traffic_lights:
                if int(lid) in tl.controller_ids:
                    tl.state = mapped

    # ── 이하 PDM-Lite privileged_route_planner.py 원문 이식 ──────────────
    def compute_rotation_angles(self, route_points):
        """경로 점열의 yaw [deg, CARLA 프레임] — PDM 원문."""
        indices = np.arange(1, route_points.shape[0] - 1)
        differences = route_points[indices + 1] - route_points[indices - 1]
        yaws = np.arctan2(differences[:, 1], differences[:, 0]) * 180.0 / np.pi
        return np.concatenate([[yaws[0]], yaws, [yaws[-1]]])

    def get_closest_route_index(self, begin_idx, location):
        """PDM 원문 — 일정 그래디언트 하강으로 최근접 경로 인덱스."""
        index = begin_idx
        location_np = np.array([location.x, location.y])
        direction = 1
        if np.linalg.norm(location_np - self.original_route_points[index, :2]) \
                < np.linalg.norm(location_np - self.original_route_points[index + 1, :2]):
            direction = -1
        while True:
            if index + direction == 0 or index + direction == self.original_route_points.shape[0]:
                return index
            dist1 = np.linalg.norm(location_np - self.original_route_points[index, :2])
            dist2 = np.linalg.norm(location_np - self.original_route_points[index + direction, :2])
            if dist1 < dist2:
                return index
            index += direction

    def _smooth_transition(self, value):
        """PDM 원문 — 0..1 선형 전이를 cos 전이로."""
        return -np.cos(value * np.pi) / 2.0 + 0.5

    # ── VTD 추가: 시프트 목표 차로의 기준점 ──────────────────────────────
    # 원문은 목표를 항상 `route_waypoints[idx]`(= 경로 **생성 시점**의 차로)의
    # 이웃으로 잡는다. 이미 한 번 시프트해 옆 차로에 있는 상태에서 같은 쪽을
    # 다시 고르면 목표가 **지금 있는 그 차로**가 되어 경로가 1 mm 도 안 움직인다
    # (실측 2026-09-03 104648: 자차 (429,3,3), 원 경로 (429,3,2), 빈 차로는
    #  (429,3,4). BREAKOUT L1 이 ot_span 을 풀고 다시 시프트를 만들었으나
    #  좌측은 원 경로의 좌측 = 없음 → 폴백으로 원 경로에 되돌아갔다).
    #
    # 아래 두 헬퍼는 **적용(shift_route_smoothly)과 계측(planned_lateral_offsets)이
    # 같은 목표를 쓰도록** 한 곳에 둔 것이다. 둘이 어긋나면 shift_entry·span_extend
    # 의 사전 이격 계측이 실제로 갈 차로가 아닌 곳을 재게 된다.

    def _shift_target_steps(self, shift_to_left_lane) -> int:
        """목표까지 따라갈 이웃 차로 **단계 수**. 원문은 항상 1 이다.

        현재 경로가 원 경로에서 그 방향으로 몇 칸 밀려 있는지(k)를 재어 k+1 을
        돌려준다 — "지금 있는 차로에서 한 칸" 이다. 한 칸 제한은 그대로다.

        k 는 차로폭으로 나누지 않고 **1칸 이웃까지의 오프셋 D1 로 나눈다** —
        D1 은 planned_lateral_offsets 와 같은 외적 규약(CARLA 미러 프레임)이라
        부호가 저절로 맞고, 차로폭이 구간마다 달라도(429 구간 3.20/3.75/2.48 m)
        같은 축으로 잰다. 시프트가 없으면 t_cur = 0 → k = 0 → 1 단계이므로
        원문과 **같은 좌표**가 나온다 (스위치 on 에서도 비트 동일).
        반대쪽으로 밀려 있으면 k = −1 → 0 단계 = 목표가 원 경로 차로 자신
        (= 지금 있는 차로의 한 칸 반대쪽)이라 좌우 어느 쪽도 맞게 동작한다.
        """
        if not self.shift_target_current:
            return 1                                   # 원문 동작
        i0 = int(self.route_index)
        n = len(self.original_route_points)
        if i0 >= n - 1:
            return 1
        wp = self.route_waypoints[i0]
        nb = self.lg.neighbor(wp.key, 'left' if shift_to_left_lane else 'right')
        if nb is None:
            return 1                                   # 잴 기준이 없다 → 이전 동작
        orig = self.original_route_points
        tx, ty = orig[i0 + 1, :2] - orig[i0, :2]
        d = _math.hypot(tx, ty)
        if d < 1e-9:
            return 1
        loc = VtdWaypoint(self.lg, nb, min(wp.s, self.lg.length(nb))).transform.location
        bx, by = orig[i0, :2]
        d1 = (tx * (loc.y - by) - ty * (loc.x - bx)) / d          # 1칸 이웃 오프셋
        if abs(d1) < 1.0:                              # 차로폭이라 보기 어렵다
            return 1
        cx, cy = self.route_points[i0, :2]
        t_cur = (tx * (cy - by) - ty * (cx - bx)) / d             # 현재 경로 변위
        k = int(round(t_cur / d1))                     # |비율| < 0.5 면 0 = 이전 동작
        return int(np.clip(k + 1, 0, self.shift_target_max_steps))

    def _shift_target_wp(self, idx, shift_to_left_lane, n_steps):
        """목표 waypoint — 이웃을 n_steps 만큼 따라간다. 못 가면 None (원문 폴백).

        n_steps == 1 이면 `route_waypoints[idx].get_left_lane()/get_right_lane()`
        과 **같은 객체**다 (s 규약 min(self.s, length(nb)) 까지 동일).
        n_steps == 0 이면 원 경로 차로 자신을 돌려준다.
        """
        wp = self.route_waypoints[idx]
        side = 'left' if shift_to_left_lane else 'right'
        key = wp.key
        for _ in range(int(n_steps)):
            key = self.lg.neighbor(key, side)
            if key is None:
                return None
        if key == wp.key:
            return wp
        return VtdWaypoint(self.lg, key, min(wp.s, self.lg.length(key)))

    def shift_route_smoothly(self, start_index, end_index, shift_to_left_lane,
                             transition_length=120.0, lane_transition_factor=1.0):
        """PDM 원문 (visualize 제외) — 경로를 옆 차로로 부드럽게 시프트."""
        # VTD: 원문은 목표가 route_waypoints[idx].get_*_lane() 고정이다.
        # 기준점만 현재 경로 차로로 옮긴다 (_shift_target_steps). off 면 1 단계 = 원문.
        n_steps = self._shift_target_steps(shift_to_left_lane)
        for idx in range(start_index, end_index):
            wp_t = self._shift_target_wp(idx, shift_to_left_lane, n_steps)
            if wp_t is None and n_steps != 1:
                # VTD: 다단계 목표가 그 지점에서 끊겼다 (열 사슬이 짧아짐).
                # 원문 폴백(원 경로 차로 중심)으로 가면 이미 밀려 있는 경로가
                # 한 칸 되돌아가 계단이 생긴다 → 그 점은 그대로 둔다.
                loc = np.array(self.route_points[idx], dtype=float)
            else:
                loc = (self.route_waypoints[idx].transform.location
                       if wp_t is None else wp_t.transform.location)
                loc = np.array([loc.x, loc.y, loc.z])

            transition_factor = 1.0
            if idx <= start_index + transition_length and idx - start_index < end_index - idx:
                transition_factor = self._smooth_transition(
                    float(idx - start_index) / transition_length)
                self.commands[idx] = (RoadOption.CHANGELANELEFT if shift_to_left_lane
                                      else RoadOption.CHANGELANERIGHT)
            elif idx >= end_index - transition_length:
                transition_factor = self._smooth_transition(
                    float(end_index - idx) / transition_length)
                self.commands[idx] = (RoadOption.CHANGELANERIGHT if shift_to_left_lane
                                      else RoadOption.CHANGELANELEFT)

            self.route_points[idx] = (
                lane_transition_factor * transition_factor * loc
                + (1.0 - lane_transition_factor * transition_factor) * self.route_points[idx])
            # 지시등 입력 갱신 — 회피로 경로를 밀면 깜빡이가 따라온다 (kr_rules).
            # 빌드 시 LC 오프셋 + 런타임 시프트 변위. 부호는 +좌 / −우.
            if getattr(self, 'lat_shift', None) is not None and idx < len(self.lat_shift):
                d = float(np.linalg.norm(
                    self.route_points[idx][:2] - self.original_route_points[idx][:2]))
                self.lat_shift[idx] = self._lat_build[idx] + (d if shift_to_left_lane else -d)

    def plan_shift_span(self, first_actor, last_actor=None,
                        obstacle_direction='right', transition_length=120.0,
                        extra_length_before=0.0, extra_length_after=0.0,
                        min_start_ahead=0):
        """시프트 구간 인덱스만 계산한다 — **경로를 건드리지 않는다**.

        shift_route_around_actors 의 앞부분을 그대로 뗀 것이다. 적용 전에
        기하를 검사하려면(kr_rules 의 계단 불연속 계측) 같은 인덱스가 필요한데,
        계산을 복제하면 두 곳이 어긋난다. 반환은 (시작, 끝, 좌측시프트여부).
        """
        # VTD: 원문은 route_index 이후 **전 구간**에 cKDTree 를 세운다
        # (privileged_route_planner.py 447-453). 순환 코스에서 경로가 같은 자리를
        # 두 번 지나면 한 바퀴 뒤 경로점이 근소하게 더 가까워 그쪽이 잡히고, 자차
        # 앞이 아니라 4.5 km 뒤 구간이 밀린다 (실측 2026-09-03 100310/100458 14건,
        # span 시작 − route_index = 4550.1~4595.2 m). 탐색 창을 자차 앞
        # leading_vehicles_maximum_detection_radius(80 m — compute_leading_vehicles 가
        # 쓰는 같은 상수)로 제한한다. kr 계층이 넘기는 actor 는 전부 detect_max_m(80 m)
        # 안이라 참 최근접점이 창 안에 있고, 창 밖이면 창 끝으로 클램프된다(None·예외를
        # 내면 shift_route_around_actors 의 튜플 언패킹이 깨진다). span_gate 는 그대로
        # 안전망으로 남는다. 스위치가 꺼지면 아래 hi 가 None 이라 원문과 같은 슬라이스다.
        hi = (self.route_index + self.leading_vehicles_maximum_detection_radius
              if self.span_search_local else None)
        tree = cKDTree(self.original_route_points[self.route_index:hi, :2])
        first_actor_location = np.array(
            [first_actor.get_location().x, first_actor.get_location().y])
        _, closest_idx = tree.query(first_actor_location, k=1)
        first_idx = closest_idx + self.route_index

        first_actor_extent = first_actor.bounding_box.extent.x
        shift_start_index = first_idx - int(
            first_actor_extent * self.points_per_meter + transition_length + extra_length_before)

        if last_actor is None:
            shift_end_index = first_idx + int(
                first_actor_extent * self.points_per_meter + transition_length + extra_length_after)
        else:
            last_idx = self.get_closest_route_index(first_idx, last_actor.get_location())
            last_actor_extent = last_actor.bounding_box.extent.x
            shift_end_index = last_idx + int(
                last_actor_extent * self.points_per_meter + transition_length + extra_length_after)

        floor = self.route_index + int(min_start_ahead)
        if shift_start_index < floor:                  # 자차 앞에서 시작 (아래 참조)
            shift_start_index = min(floor, shift_end_index - 1)
        return shift_start_index, shift_end_index, obstacle_direction == 'right'

    def planned_lateral_offsets(self, start_index, end_index, shift_to_left_lane,
                                step_pts=10):
        """적용 전, 이웃 차로 중심까지의 **부호 있는 횡오프셋**을 성긴 간격으로.

        shift_route_smoothly 가 각 점에서 목표로 삼는 `get_left_lane()/
        get_right_lane()` 위치를, 경로를 건드리지 않고 step_pts 간격으로만 읽는다.
        전 해상도(0.1 m)로 읽으면 span 하나에 30 ms 가 넘어 틱 예산을 먹는다
        (실측 2026-09-02: 적용+원복 왕복 36.1 ms = 예산의 72%). 1 m 간격이면
        1.7~5.2 ms 이고, 찾으려는 **계단 불연속**(2.9 m/점)은 성기게 떠도 그대로
        보인다.

        전이 계수는 곱하지 않는다 — 계수는 코사인이라 매끄럽고, 여기서 보려는
        것은 **목표 차로가 튀는가** 뿐이다. 계수를 빼면 정상 전이의 곡률이
        빠지므로 판별이 오히려 깨끗해진다 (실측 정상 κ 0.0002~0.0095 대
        계단 2.977).
        """
        out = []
        n = len(self.original_route_points)
        # VTD: 원문 자리에서 목표 차로를 잡던 것을 shift_route_smoothly 와 **같은**
        # 헬퍼로 바꾼다. 둘이 어긋나면 여기(계측)가 재는 차로와 저기(적용)가 미는
        # 차로가 달라져, shift_entry·span_extend 의 사전 이격 계측이 무효가 된다.
        # off 면 n_steps == 1 이라 원문과 완전히 같다.
        n_steps = self._shift_target_steps(shift_to_left_lane)
        for i in range(int(start_index), min(int(end_index), n), max(1, int(step_pts))):
            wp_t = self._shift_target_wp(i, shift_to_left_lane, n_steps)
            if wp_t is None and n_steps == 1:
                out.append(0.0)                        # 원문 동작
                continue
            bx, by = self.original_route_points[i, :2]
            j = min(i + 1, n - 1)
            tx, ty = self.original_route_points[j, :2] - self.original_route_points[i, :2]
            d = _math.hypot(tx, ty) or 1.0
            if wp_t is None:
                # 사슬이 끊긴 지점 — 적용 쪽이 그 점을 그대로 두므로 계측도 현재
                # 경로점을 목표로 본다 (이동량 0 이 되도록).
                lx, ly = self.route_points[i, :2]
            else:
                loc = wp_t.transform.location
                lx, ly = loc.x, loc.y
            out.append((tx * (ly - by) - ty * (lx - bx)) / d)
        return np.asarray(out, dtype=float)

    def shift_route_around_actors(self, first_actor, last_actor=None,
                                  obstacle_direction='right', transition_length=120.0,
                                  lane_transition_factor=1.0,
                                  extra_length_before=0.0, extra_length_after=0.0,
                                  min_start_ahead=0):
        """PDM 원문 — 액터 주위로 경로 시프트. 발동은 phase4 에서.

        VTD 추가 min_start_ahead [경로점 수]: 전이 시작을 **자차보다 이만큼 앞**
        으로 강제한다. 원문은 시작점을 액터에서 거꾸로 재므로, 코앞의 장애물에
        정지한 상태에서 부르면 시작점이 자차 뒤로 가고 **현재 위치의 경로가 옆으로
        밀린다** → 정지 상태에서 횡오차가 생겨 조향이 풀락된다 (2026-08-30
        실전주행_01_연속교차로24 실측: 시프트 직후 steer +0.480 고정, 17.8 s 정지).

        구간 계산은 plan_shift_span 이 한다 (적용 전 검사와 같은 인덱스를 쓰려고
        뗐다). 아래 shift_route_smoothly 호출이 PDM 원문 그대로다.
        """
        shift_start_index, shift_end_index, shift_to_left_lane = self.plan_shift_span(
            first_actor, last_actor, obstacle_direction, transition_length,
            extra_length_before, extra_length_after, min_start_ahead)
        self.shift_route_smoothly(shift_start_index, shift_end_index, shift_to_left_lane,
                                  transition_length=transition_length,
                                  lane_transition_factor=lane_transition_factor)
        return shift_start_index, shift_end_index

    def compute_leading_vehicles(self, list_vehicles, ego_vehicle_id):
        """경로 전방 80 m 의 선행 객체 id.

        PDM 원문 조건: 경로에 붙어(2.5 m) 진행방향이 35° 이내인 차.
        **VTD 추가**: 정지한 객체는 방향을 묻지 않고 "경로를 막는가" 만 본다
        (percep.obstacle_*). 라바콘·공사 자재·비스듬히 선 차는 heading 이 임의라
        35° 조건에서 빠지는데, 그러면 IDM 감속이 안 걸려 OBB 충돌예측이 코앞에서
        급제동하는 것만 남는다. 막는 폭은 차폭/2 + 객체폭/2 + 여유로 크기에
        비례시킨다 — 갓길에 비켜 선 물체까지 세우지 않기 위해서다.
        """
        vehicle_ids = np.array(
            [vehicle.id for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])
        if len(vehicle_ids) and self.route_index != self.route_points.shape[0]:
            max_distance = self.leading_vehicles_maximum_detection_radius
            vehicle_yaws = np.array(
                [vehicle.get_transform().rotation.yaw
                 for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])
            vehicle_locations = np.array(
                [[v.get_location().x, v.get_location().y, v.get_location().z]
                 for v in list_vehicles if v.id != ego_vehicle_id])

            distances = (vehicle_locations[:, None, :2]
                         - self.route_points[None, self.route_index:self.route_index + max_distance, :2][:, ::self.points_per_meter, :])
            distances = np.linalg.norm(distances, axis=2)
            route_indices = distances.argmin(axis=1)
            distances = distances.min(axis=1)
            rotation_angles = self.rotation_angles[
                self.route_index:self.route_index + max_distance][::self.points_per_meter]
            route_yaws = rotation_angles[route_indices]
            yaw_differences = (route_yaws - vehicle_yaws) % 360
            yaw_differences = np.minimum(yaw_differences, 360 - yaw_differences)

            moving_ok = ((distances < self.leading_vehicles_max_route_distance)
                         & (yaw_differences < self.leading_vehicles_max_route_angle_distance))

            # VTD: 정지 객체는 heading 무관 — 경로를 막는지만 (크기 비례 폭)
            speeds = np.array([v.get_velocity().length() for v in list_vehicles
                               if v.id != ego_vehicle_id])
            half_widths = np.array([float(v.bounding_box.extent.y)
                                    for v in list_vehicles if v.id != ego_vehicle_id])
            blocking = ((speeds < self.obstacle_speed_max)
                        & (distances < self.veh_width / 2.0 + half_widths
                           + self.obstacle_clearance_m))

            return vehicle_ids[moving_ok | blocking].tolist()
        return []

    def compute_trailing_vehicles(self, list_vehicles, ego_vehicle_id):
        """PDM 원문 — 경로 후방에서 따라오는 차."""
        vehicle_ids = np.array(
            [vehicle.id for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])
        max_distance = self.trailing_vehicles_max_route_distance
        for i in range(max(0, self.route_index - self.max_distance_lane_change_trailing_vehicles),
                       self.route_index):
            if self.commands[i] in (RoadOption.CHANGELANELEFT, RoadOption.CHANGELANERIGHT):
                max_distance = self.trailing_vehicles_max_route_distance_lane_change
                break

        if len(vehicle_ids) and self.route_index != 0:
            vehicle_yaws = np.array(
                [vehicle.get_transform().rotation.yaw
                 for vehicle in list_vehicles if vehicle.id != ego_vehicle_id])
            vehicle_locations = np.array(
                [[v.get_location().x, v.get_location().y, v.get_location().z]
                 for v in list_vehicles if v.id != ego_vehicle_id])

            from_idx = max(0, self.route_index - self.tailing_vehicles_maximum_detection_radius)
            distances = (vehicle_locations[:, None, :2]
                         - self.route_points[None, from_idx:self.route_index, :2][:, ::self.points_per_meter, :])
            distances = np.linalg.norm(distances, axis=2)
            route_indices = distances.argmin(axis=1)
            distances = distances.min(axis=1)
            rotation_angles = self.rotation_angles[from_idx:self.route_index][::self.points_per_meter]
            route_yaws = rotation_angles[route_indices]
            yaw_differences = (route_yaws - vehicle_yaws) % 360
            yaw_differences = np.minimum(yaw_differences, 360 - yaw_differences)
            vehicles_behind_ids = vehicle_ids[(distances < max_distance) & (yaw_differences < 30)]
            return vehicles_behind_ids.tolist()
        return []
