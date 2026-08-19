"""
WorldState → Decision  (SPEC §3.4)

FSM + 종방향 **min() 중재**. 여러 근거가 각자 상한을 내고 가장 낮은 것이 이긴다.
어떤 후보가 이겼는지는 `Decision.reasons` 에 남겨 로그로 추적한다.
"""
from __future__ import annotations

import math

import numpy as np

from .lanegraph import LaneGraph
from .types import Decision, WorldState

# FSM 상태 (SPEC §3.4)
FOLLOW = 'FOLLOW'
STOP_LINE = 'STOP_LINE'
FOLLOW_LEAD = 'FOLLOW_LEAD'
YIELD_PED = 'YIELD_PED'
AVOID = 'AVOID'
RETURN = 'RETURN'
LANE_CHANGE = 'LANE_CHANGE'
E_STOP = 'E_STOP'

TURN_OFF, TURN_LEFT, TURN_RIGHT = 0, 1, 2


class Planner:
    def __init__(self, lg: LaneGraph, route: dict | None, cfg: dict) -> None:
        self.lg = lg
        self.route = route
        self.cfg = cfg
        self.state = FOLLOW

        # 차선변경 상태
        self._lc = None            # 진행/대기 중인 이벤트 dict
        self._lc_signal_on = False
        self._lc_warned = set()    # 창을 놓칠 뻔한 경고를 이벤트당 한 번만
        self.lc_done = 0           # 완료 횟수 (품질 지표)
        self.lc_aborted = 0
        # TODO: AVOID 진입 시점 등

    def plan(self, world: WorldState) -> Decision:
        """
        상수 속도 주행 + 차선변경.

        제한속도/정지선/신호/객체 중재는 아직 TODO (_speed_candidates).
        차선변경만 경로 이벤트를 근거로 실행한다.
        """
        v_const = float(self.cfg['debug']['const_speed_kph']) / 3.6
        v_target = 0.0 if not world.valid else v_const
        reasons: dict = {'const': v_const}
        turn_signal = TURN_OFF

        if not world.valid or world.ego.lane is None:
            self.state = FOLLOW
            return Decision(v_target=0.0, path=[], turn_signal=TURN_OFF,
                            state=self.state, reasons={'invalid': 0.0})

        d = self.cfg['debug']
        idx = self._route_idx(world)
        base = self.lg.points_ahead(
            world.ego.lane, world.ego.s,
            dist=float(d['path_dist_m']), step=float(d['path_step_m']),
            route=self.route, idx=idx)
        path = [(float(x), float(y)) for x, y in base]

        # ── 차선변경 ─────────────────────────────────────────────────────
        lc = self._update_lane_change(world, reasons)
        if lc is not None:
            turn_signal = lc['signal']
            if lc['active']:
                self.state = LANE_CHANGE
                blended = self._blend_path(world, lc, base)
                if blended is not None:
                    path = blended
                # 전이 중 감속하지 않는다 (녹색신호 통과 감점 방지)
            else:
                self.state = FOLLOW
        else:
            self.state = FOLLOW

        # ── 종방향: 후보들의 min() 중재 ──────────────────────────────────
        cand = self._speed_candidates(world)
        reasons.update(cand)
        v_target = min([v_target] + list(cand.values()))
        reasons['v_target'] = round(v_target, 3)
        winner = min(cand, key=cand.get) if cand else None
        reasons['winner'] = winner if (winner and cand[winner] <= v_target + 1e-6) else 'const'

        # 정지선 때문에 서는 중이면 FSM 상태로 드러낸다 (차선변경이 우선)
        if self.state != LANE_CHANGE and winner == 'stop_line' and v_target < 0.5:
            self.state = STOP_LINE

        return Decision(v_target=v_target, path=path, turn_signal=turn_signal,
                        state=self.state, reasons=reasons)

    # ══════════════════════════════════════════════════════════════════════
    # 종방향 속도 후보
    # ══════════════════════════════════════════════════════════════════════
    def _approach(self, v_at: float, dist: float) -> float:
        """
        dist [m] 앞에서 v_at [m/s] 가 되려면 지금 낼 수 있는 최대 속도.

        v^2 = v_at^2 + 2*a*d  (등감속). 이걸 후보로 쓰면 제한이 걸리는 지점에
        **도달했을 때 이미 그 속도**가 되어 있다.
        """
        a = float(self.cfg['speed']['a_comf'])
        return math.sqrt(max(0.0, v_at * v_at + 2.0 * a * max(0.0, dist)))

    def _speed_candidates(self, world: WorldState) -> dict:
        """
        속도 상한 후보들 [m/s]. 최종 v_target = min(values()).

        어떤 근거가 이겼는지 로그로 추적할 수 있도록 이름을 붙여 돌려준다.
        """
        sp = self.cfg['speed']
        caps = self.cfg['caps_kph']
        margin = float(sp['margin_kph']) / 3.6
        out: dict = {}

        # ── 1) 제한속도 + 스쿨존 ─────────────────────────────────────────
        out['limit'] = max(0.0, world.speed_limit - margin)
        if world.school_zone:
            out['school_zone'] = float(caps['school_zone']) / 3.6

        # 전방에서 제한속도가 낮아지면 미리 감속한다
        for a in world.ahead:
            if a.kind != 'speed':
                continue
            lim = a.data.get('limit')
            if lim is None:
                continue
            v_lim = max(0.0, float(lim) / 3.6 - margin)
            if a.data.get('school_zone'):
                v_lim = min(v_lim, float(caps['school_zone']) / 3.6)
            if v_lim < out['limit']:
                key = 'limit_ahead'
                out[key] = min(out.get(key, math.inf), self._approach(v_lim, a.dist))

        # ── 2) 곡률 ──────────────────────────────────────────────────────
        v_curv = self._curvature_speed(world)
        if v_curv is not None:
            out['curvature'] = v_curv

        # ── 3) 위험 구간 캡 ──────────────────────────────────────────────
        for a in world.ahead:
            if a.kind in ('crosswalk', 'crosswalk_warn'):
                cap = float(caps['crosswalk']) / 3.6
                out['crosswalk'] = min(out.get('crosswalk', math.inf),
                                       self._approach(cap, a.dist))
            elif a.kind == 'junction_in':
                cap = float(caps['junction']) / 3.6
                out['junction'] = min(out.get('junction', math.inf),
                                      self._approach(cap, a.dist))

        # ── 4) 정지선 + 신호 ─────────────────────────────────────────────
        v_stop = self._signal_speed(world, out)
        if v_stop is not None:
            out['stop_line'] = v_stop

        # ── 5) 가시거리 (GT 80 m 밖은 안 보이는 것과 같다) ───────────────
        out['visibility'] = self.visibility_limit()

        return {k: float(v) for k, v in out.items() if math.isfinite(v)}

    def _curvature_speed(self, world: WorldState) -> float | None:
        """
        전방 곡률에 맞춘 속도. sqrt(a_lat_max / |curv|).

        전방 각 지점의 곡률을 보고 "거기 도달했을 때 그 속도" 가 되도록 미리
        줄인다. 교차로 회전 연결로는 곡률이 커서 자동으로 감속된다.
        """
        if world.ego.lane is None or not self.route:
            return None
        a_lat = float(self.cfg['speed']['a_lat_max'])
        horizon = float(self.cfg['debug']['path_dist_m'])

        idx = self._route_idx(world)
        if idx is None:
            return None
        lanes = self.route['lanes']

        best = None
        acc = 0.0
        s0 = world.ego.s
        i = idx
        while acc < horizon and i < len(lanes):
            r = self.lg.lanes[lanes[i]]
            ss, cv = r['s'], r['curv']
            for j in range(len(ss)):
                if ss[j] < s0:
                    continue
                d = acc + (ss[j] - s0)
                if d > horizon:
                    break
                k = abs(float(cv[j]))
                if k < 1e-4:                     # 반경 1 km 이상 = 직선
                    continue
                v_here = math.sqrt(a_lat / k)
                v_now = self._approach(v_here, d)
                if best is None or v_now < best:
                    best = v_now
            acc += r['length'] - s0
            s0 = 0.0
            i += 1
        return best

    def _signal_speed(self, world: WorldState, out: dict) -> float | None:
        """
        전방 정지선에서 서야 하는가. 서야 하면 그때까지 허용 속도.

        - 적/황 이고 아직 여유가 있으면 정지선 - stop_gap_m 에 정지
        - 황색 딜레마존: d <= v * yellow_s 면 급정거가 더 위험하니 통과
        - 녹 / 녹+좌 는 통과 (**녹색신호 통과도 채점 항목이다. 불필요한 정지 금지**)
        - 좌회전(4) 은 경로가 좌회전일 때만 진행 근거로 쓴다
        - 신호 없는 정지선(signal_ids 비어 있음)은 지금은 무시한다
          (일시정지 규정 확인 전까지 — SPEC §7 미확인 항목)
        """
        summ = world.summ or {}
        d = summ.get('dist_stop_line')
        if d is None:
            return None
        if not (summ.get('stop_signal_ids') or []):
            return None                      # 비신호 정지선 — 아직 다루지 않는다
        if world.light is None:
            return None

        state = int(world.light[1])
        sp = self.cfg['speed']
        gap = float(sp['stop_gap_m'])
        yellow_s = float(self.cfg['signal']['yellow_s'])
        v = max(world.ego.speed, 0.0)

        GREEN, GREEN_LEFT, LEFT, YELLOW, RED, FLASH = 3, 5, 4, 2, 1, 6

        if state in (GREEN, GREEN_LEFT):
            return None                      # 통과
        if state == LEFT:
            nxt = summ.get('next_turn')
            return None if nxt == 'turn_left' else self._stop_at(d, gap)
        if state == YELLOW:
            if d <= v * yellow_s:
                return None                  # 딜레마존 — 통과가 안전
            return self._stop_at(d, gap)
        if state == RED:
            return self._stop_at(d, gap)
        if state == FLASH:
            mode = str(self.cfg['signal'].get('flash_mode', 'yield'))
            if mode == 'stop':
                return self._stop_at(d, gap)
            # yield/go: 서행만 (TODO: 대회 규정 확인 — SPEC §7-6)
            return self._approach(float(self.cfg['caps_kph']['junction']) / 3.6, d)
        return None                          # 0 = 미할당

    def _stop_at(self, dist: float, gap: float) -> float:
        """정지선 dist 앞, gap 여유를 두고 정지하기 위한 현재 허용 속도."""
        return self._approach(0.0, dist - gap)

    # ══════════════════════════════════════════════════════════════════════
    # 차선변경
    # ══════════════════════════════════════════════════════════════════════
    def _pending_lane_change(self, world: WorldState) -> dict | None:
        """아직 안 끝난 가장 가까운 lane_change 이벤트."""
        if not self.route:
            return None
        s_now = world.ego.route_s
        best = None
        for e in self.route.get('events', []):
            if not e['kind'].startswith('lane_change'):
                continue
            if e.get('to_lane') is None:
                continue
            if s_now > e['window_s1']:
                continue                      # 창을 이미 지났다
            if best is None or e['window_s0'] < best['window_s0']:
                best = e
        return best

    def _update_lane_change(self, world: WorldState, reasons: dict) -> dict | None:
        """
        지시등 점등 / 안전 확인 / 실행 여부 판단.

        반환 dict: {'signal', 'active', 'target', 'event'} 또는 None.
        active=True 면 목표 차로로 경로를 전이한다.
        """
        lc = self.cfg['lane_change']
        sig = self.cfg['signal']
        ev = self._lc or self._pending_lane_change(world)
        if ev is None:
            self._lc = None
            self._lc_signal_on = False
            return None

        s_now = world.ego.route_s
        v = max(world.ego.speed, 0.1)
        target = tuple(ev['to_lane'])
        side = 'left' if ev['kind'].endswith('left') else 'right'
        signal = TURN_LEFT if side == 'left' else TURN_RIGHT

        # 창(window)을 지나쳤다.
        # **시작**은 창 안에서만 한다. 다만 이미 전이 중이라면 차로 한복판에서
        # 경로를 되돌리는 편이 더 위험하므로, 실제 법규 기준인 "지금 밟고 있는
        # 차선이 점선인가" 로 계속 여부를 판단한다. 실선이 되면 즉시 중단하고
        # (shield 도 같은 규칙으로 독립 검사한다) 창을 크게 넘기면 포기한다.
        if s_now > ev['window_s1']:
            if self._lc is None:
                return None                       # 시작도 못 했다 — 조용히 넘긴다
            still_dashed = self.lg.lane_change_ok(world.ego.lane, world.ego.s, side)
            over = s_now - ev['window_s1']
            if not still_dashed or over > float(lc['transition_min_m']):
                self._finish_lane_change(
                    ev, ok=False,
                    why=f"창을 {over:.0f} m 넘겼다 (점선={still_dashed})")
                return None

        # 완료 판정 (이미 목표 차로에 안착했나)
        if self._lc is not None and self._lane_change_done(world, target):
            self._finish_lane_change(ev, ok=True)
            return None

        dist_to_window = ev['window_s0'] - s_now
        # 지시등: 창 시작 (lead_s + margin_s) 초 전부터
        lead_m = v * (float(sig['lead_s']) + float(sig['margin_s']))
        want_signal = dist_to_window <= lead_m
        if want_signal and not self._lc_signal_on:
            self._lc_signal_on = True

        in_window = ev['window_s0'] <= s_now <= ev['window_s1']
        # 지시등을 lead_s 만큼 켠 뒤에야 실행 (채점 항목)
        signaled_long_enough = dist_to_window <= v * float(sig['lead_s'])

        clear, why = self._target_lane_clear(world, target)
        reasons['lc_dist_to_window'] = round(dist_to_window, 1)
        reasons['lc_clear'] = clear

        active = False
        if self._lc is not None:
            # 이미 전이 중 — 점선인 동안에는 끝까지 간다
            active = True
        elif in_window and clear and signaled_long_enough:
            self._lc = dict(ev)               # 실행 시작
            # 전이 진행도의 기준점. 자차 위치에 매 틱 다시 고정하면
            # 목표의 일부 지점만 계속 쫓게 되어 차선을 끝까지 못 넘는다.
            self._lc['s_start'] = s_now
            active = True
        left = ev['window_s1'] - s_now
        if in_window and left < float(lc['min_window_m']) and ev['window_s0'] not in self._lc_warned:
            self._lc_warned.add(ev['window_s0'])
            reasons['lc_warn'] = (
                f'차선변경 창이 {left:.0f} m 남았는데 아직 못 끝냈다 — '
                + (why if not clear else '전이 진행 중')
                + '. 놓치면 다음 교차로에서 경로를 못 따라간다')

        return {'signal': signal if (self._lc_signal_on or want_signal) else TURN_OFF,
                'active': active, 'target': target, 'event': ev, 'side': side}

    def _target_lane_clear(self, world: WorldState, target) -> tuple[bool, str]:
        """목표 차로가 driving 이고, 점선이고, 뒤/앞 범위가 비었는가."""
        lc = self.cfg['lane_change']
        ego = world.ego
        rec = self.lg.lanes.get(target)
        if rec is None or rec['type'] != 'driving':
            return False, '목표 차로가 주행 차로가 아니다'

        side = 'left' if target[2] > ego.lane[2] else 'right'
        if not self.lg.lane_change_ok(ego.lane, ego.s, side):
            return False, f'{side} 차선이 점선이 아니다(실선)'

        back = -abs(float(lc['back_m']))
        front = abs(float(lc['front_m']))
        for o in world.objects:
            if o.lane is None or tuple(o.lane) != target:
                continue
            if back <= o.s_rel <= front:
                return False, f'목표 차로에 객체 id={o.id} ({o.s_rel:+.0f} m)'
        return True, ''

    def _lane_change_done(self, world: WorldState, target) -> bool:
        """목표 차로 중심선에 붙었는가."""
        lc = self.cfg['lane_change']
        if tuple(world.ego.lane) != tuple(target):
            return False
        return (abs(world.ego.t_off) < float(lc['done_t_off_m'])
                and abs(world.ego.heading_err) < math.radians(float(lc['done_heading_deg'])))

    def _finish_lane_change(self, ev: dict, ok: bool, why: str = '') -> None:
        if ok:
            self.lc_done += 1
        elif self._lc is not None:
            self.lc_aborted += 1
        self._lc = None
        self._lc_signal_on = False
        self.last_lc_note = why

    def abort_lane_change(self, why: str = '') -> None:
        """shield 가 위험을 감지했을 때 호출. 원래 차로로 되돌린다."""
        if self._lc is not None:
            self.lc_aborted += 1
        self._lc = None
        self._lc_signal_on = False
        self.last_lc_note = why

    def _blend_path(self, world: WorldState, lc: dict, base) -> list | None:
        """
        현재 차로 중심선 → 목표 차로 중심선으로 **부드럽게** 전이.

        두 중심선을 같은 간격으로 뽑아 가중 평균한다. 가중치는 경로를 따라
        0 → 1 로 올라가므로 자차 바로 앞은 현재 차로에 붙어 있고(점프 없음),
        전이 거리 끝에서 목표 차로에 도달한다.
        전이 거리 = max(transition_s * v, transition_min_m).
        """
        cfg = self.cfg['lane_change']
        d = self.cfg['debug']
        step = float(d['path_step_m'])
        target = lc['target']

        s_t, _t, _dd, _j = self.lg.project(target, world.ego.x, world.ego.y)
        tgt = self.lg.points_ahead(target, s_t, dist=float(d['path_dist_m']), step=step,
                                   route=self.route, idx=self._route_idx_of(target))
        if len(tgt) < 2 or len(base) < 2:
            return None

        n = min(len(base), len(tgt))
        L = max(float(cfg['transition_s']) * max(world.ego.speed, 1.0),
                float(cfg['transition_min_m']))

        # 진행도는 **경로상 고정 시작점** 기준이다.
        # 매 틱 자차 위치를 0 으로 두면 전방 점의 가중치가 늘 같은 값에 머물러
        # 차가 목표 차로의 일부 지점(≈0.5 m)만 쫓다가 넘어가지 못한다.
        s_start = self._lc.get('s_start', world.ego.route_s) if self._lc else world.ego.route_s
        prog = max(0.0, world.ego.route_s - s_start)

        out = []
        for i in range(n):
            w = min(1.0, max(0.0, (prog + i * step) / L))
            w = w * w * (3.0 - 2.0 * w)        # smoothstep — 시작/끝 기울기 0
            out.append((float(base[i][0]) * (1 - w) + float(tgt[i][0]) * w,
                        float(base[i][1]) * (1 - w) + float(tgt[i][1]) * w))
        return out

    def _route_idx_of(self, lane) -> int | None:
        if not self.route:
            return None
        try:
            return self.route['lanes'].index(tuple(lane))
        except ValueError:
            return None

    def _route_idx(self, world: WorldState) -> int | None:
        """route['lanes'] 상의 현재 인덱스. points_ahead 가 다음 차로로 이어붙일 때 쓴다."""
        if not self.route or world.ego.lane is None:
            return None
        lanes = self.route['lanes']
        try:
            return lanes.index(world.ego.lane)
        except ValueError:
            return None

    # ── 종방향 ────────────────────────────────────────────────────────────
    def visibility_limit(self) -> float:
        """
        가시거리 상한 속도 [m/s].

        GT 객체는 `percep.gt_range_m`(80 m) 안쪽만 온다. 그 밖은 안 보이는 것과
        같으므로, 80 m 안에서 멈출 수 있는 속도가 물리적 상한이다.
        80 m / a_comf 1.5 기준 약 15.4 m/s (55 km/h) — 제한속도 50 보다 높아
        평소에는 안 걸리지만, 상한이 존재한다는 사실 자체가 근거로 남아야 한다.
        """
        p = self.cfg['percep']
        sp = self.cfg['speed']
        sight = float(p['gt_range_m']) - float(sp['stop_gap_m'])
        return math.sqrt(max(0.0, 2.0 * float(sp['a_comf']) * sight))

    @staticmethod
    def _v_safe(dist: float, a_comf: float, stop_gap: float) -> float:
        """정지점까지 dist 일 때 허용 속도 sqrt(2*a*(d - gap)). 음수면 0."""
        # TODO: 구현
        raise NotImplementedError('planner._v_safe')

    def _idm(self, world: WorldState) -> float:
        """선행차 추종 속도 상한 (config `lead.time_headway_s`, `lead.min_gap_m`)."""
        # TODO: 구현
        raise NotImplementedError('planner._idm')

    # ── 횡방향 ────────────────────────────────────────────────────────────
    def _lateral_path(self, world: WorldState) -> list[tuple[float, float]]:
        """
        추종 목표 점열 (월드 좌표). 기본은 경로 차로 중심선 = `lg.points_ahead`.

        AVOID 진입 조건 (**전부** 만족해야 함):
          - 내 차로에 정지 장애물
          - 해당 방향 `lg.lane_change_ok` 가 True (실선이면 불가)
          - 옆차로 뒤 `lane_change.back_m` / 앞 `lane_change.front_m` 비었음
          - 지시등을 `signal.lead_s` 만큼 미리 켜둔 상태

        하나라도 불만족이면 **회피하지 않고 정지**한다.
        실선 차선변경·중앙선 침범이 감점이라 정지가 항상 이득이다.
        장애물을 지나고 조건이 맞으면 RETURN.
        """
        # TODO: 구현
        raise NotImplementedError('planner._lateral_path')

    # ── 교차로 게이트 ─────────────────────────────────────────────────────
    def _cross_gate_ok(self, world: WorldState) -> bool:
        """
        신호 위반 차량 대비 진입 게이트.

        교차 방향 접근 객체의 t_them = d / v 와 내 통과시간 t_me 를 비교해
        여유가 config `cross.margin_s` 미만이면 진입 보류.
        감속하지 않는 접근 차량은 **신호와 무관하게** 양보한다.
        """
        # TODO: 구현
        raise NotImplementedError('planner._cross_gate_ok')

    # ── 지시등 ────────────────────────────────────────────────────────────
    def _turn_signal(self, world: WorldState) -> int:
        """
        `route['events']` 의 turn_* / lane_change_* 지점까지 남은 거리를 현재
        속도로 나눈 시간이 `signal.lead_s + signal.margin_s` 이하면 점등.
        (채점 항목: 회전·차선변경 n초 전 점등)
        """
        # TODO: 구현
        raise NotImplementedError('planner._turn_signal')

    def _next_state(self, world: WorldState) -> str:
        """FSM 전이. 우선순위: E_STOP > YIELD_PED > STOP_LINE > AVOID/RETURN > FOLLOW_LEAD > FOLLOW"""
        # TODO: 구현
        raise NotImplementedError('planner._next_state')
