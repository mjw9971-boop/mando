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

# IDM 자유주행 지수. 무차원 형상 파라미터라 튜닝값이 아니다(표준 4).
IDM_DELTA = 4.0
# lead 후보 탐색 범위 [m]. GT 가 80 m 안쪽만 주므로(percep.gt_range_m) 그 이상은 의미 없다.
LEAD_MAX_S = 80.0


def crosswalk_blockers(objects, d_cw, cfg) -> list:
    """
    전방 횡단보도 폴리곤(근사) 안에 있거나 차로 진입이 예측되는 객체들.

    타입 구분 없이 **전부 보행자로 간주**한다(보수적) — 9910 패킷에 타입 필드가
    없어 크기로만 분류하므로(SPEC §1.1) 오분류 시 치는 쪽이 더 위험하다.
    폴리곤은 s_rel/lat_off 근사: 종방향 [d_cw - back, d_cw + fwd], 횡방향 |lat_off| < half_w.
    RTOR 안전 확인과 횡단보도 정지가 **같은 판정**을 쓰도록 여기 한 곳에 둔다.
    """
    if d_cw is None:
        return []
    p = cfg['percep']
    half_w = float(p.get('crosswalk_half_w_m', 8.0))
    back = float(p.get('crosswalk_back_m', 3.0))
    fwd = float(p.get('crosswalk_fwd_m', 7.0))
    out = []
    for o in objects:
        in_poly = abs(o.lat_off) < half_w and d_cw - back <= o.s_rel <= d_cw + fwd
        # 진입 예측(will_enter_lane)은 횡단보도 근처의 것만 — 멀리서 차로로 들어오는
        # 차량까지 여기서 세우면 안 된다(그건 선행차/TTC 담당).
        near = abs(o.lat_off) < half_w * 2 and d_cw - back <= o.s_rel <= d_cw + fwd
        if in_poly or (o.will_enter_lane and near):
            out.append(o)
    return out


class Planner:
    def __init__(self, lg: LaneGraph, route: dict | None, cfg: dict) -> None:
        self.lg = lg
        self.route = route
        self.cfg = cfg
        self.state = FOLLOW

        # 차선변경 상태
        self._lc = None            # 진행/대기 중인 이벤트 dict
        self._lc_signal_on = False
        # 점등 래치가 **어느 이벤트** 것인지. pending 이 다른 이벤트로 바뀌면
        # 래치를 풀고 새 이벤트의 lead 조건으로 다시 판단한다 — 이게 없으면
        # LC2 완료 직후 pending 이 LC3 로 바뀌는 순간 래치가 그대로 넘어가
        # 창까지 수십 초를 연속 점등한다 (2026-08-23 10:26 런: 30 s).
        self._lc_signal_ev = None
        # 완료한 LC 이벤트 (window_s0 키). 안착 직후 같은 이벤트를 다시 고르면
        # "선택 → 즉시 완료 → 재선택" 을 2틱 주기로 반복해 깜빡인다
        # (같은 런: 23회). 창 s1 을 지나면 _pending 이 자연히 거르므로 그 전까지만
        # 기억하면 된다. 리셋으로 창 앞으로 되돌아가면 다시 풀어 재실행한다.
        self._lc_finished: set = set()
        self._lc_warned = set()    # 창을 놓칠 뻔한 경고를 이벤트당 한 번만
        # 실제 출력 지시등의 연속 점등 추적: (방향, 켜진 시각). 계획 LC 는 해당
        # 방향이 signal.lc_lead_min_s 이상 **연속 점등**된 뒤에만 실행한다
        # (2026-08-23 11:57 런: 회전1 연결로 끝 = LC1 창 시작이라 회전 중 RIGHT 가
        #  켜져 있다가 창 진입과 동시에 LEFT 점등·즉시 실행 → lead 0.0 s).
        self._sig_dir = TURN_OFF
        self._sig_since: float | None = None
        # 창 확장 (window_s0 키 → 늘어난 window_s1). 대기로 창이 모자랄 때
        # 목표 차로가 s1 뒤로 이어지면 그만큼 연장한다.
        self._win_ext: dict = {}
        # shield 가 긴급 회피(장애물 대응)로 차선변경을 요구하면 True —
        # 점등 선행 조건을 건너뛰고 즉시 실행한다.
        self.emergency_avoid = False
        self._signal_fallback = False   # 이번 틱 정지선 판단이 9910 light 폴백이었나 (로그용)
        # RTOR(적신호 우회전) 상태: 정지선 앞 완전 정지 시각, dwell 충족 래치, 로그 메모
        self._rtor_stop_t: float | None = None
        self._rtor_dwell_done = False
        self._rtor_note = ''
        # 회전 이벤트: route 의 turn_* 에 연결로 끝(end_s)을 붙인 목록
        self._turns = self._build_turns()
        self._stop_latch = False   # 정지 확정. 정지선이 시야를 벗어나도 유지한다
        self._stop_line_s: float | None = None   # 래치 대상 정지선의 경로거리 (통과 판정용)
        self._v_now = 0.0
        self._cw_ped_latch = False   # 횡단보도 보행자 정지 확정 (보행자가 비면 풀린다)
        self._cw_note = ''
        self._lead = None                   # 이번 틱 추종 대상 (TrackedObject)
        # 정차 차량 추월(blocked): 판정 지속 시각, 판정 결과, 진행 중인 추월 상태
        self._blocked_since: float | None = None
        self._blocked = False
        self._blocked_why = ''
        self._lead_creep = False             # 이번 틱 크립 발동 (로그용)
        self._last_prog = None               # 블렌드 진행도 (로그용)
        self._obj_moved_t: dict = {}         # 객체 id → 마지막으로 '달리던' 시각
        self._overtake: dict | None = None   # {'blocker_id', 'orig_lane', 'returning'}
        self.lead_id: int | None = None     # 그 id (shield 회피·횡단보도 판정 제외용)
        self._route_end_latch = False   # 경로 끝 정지 확정 (끝을 지나쳐도 유지)
        self.lc_done = 0           # 완료 횟수 (품질 지표)
        self.lc_aborted = 0
        # TODO: AVOID 진입 시점 등

    def plan(self, world: WorldState) -> Decision:
        """
        상수 속도 주행 + 차선변경.

        제한속도/정지선/신호/객체 중재는 아직 TODO (_speed_candidates).
        차선변경만 경로 이벤트를 근거로 실행한다.
        """
        # 기본 주행은 _speed_candidates 만으로 속도를 정한다.
        # debug.enabled 일 때만 상수속도 상한을 얹는다 (초기 연동 확인용).
        dbg = self.cfg['debug']
        reasons: dict = {}
        if dbg.get('enabled') and float(dbg['const_speed_kph']) > 0:
            v_const = float(dbg['const_speed_kph']) / 3.6
            reasons['debug_const'] = v_const
            v_target = 0.0 if not world.valid else v_const
        else:
            v_target = 0.0 if not world.valid else math.inf
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
        self._last_prog = None               # 이번 틱 블렌드가 다시 채운다 (로그용)
        lc = self._update_lane_change(world, reasons)
        # ── 지시등: 차선변경 vs 교차로 회전 — **더 가까운 이벤트가 이긴다** ──
        # 둘 다 후보를 (방향, 남은거리) 로 내고 거리가 짧은 쪽을 켠다. 실행 중
        # (전이 중 / 연결로 안) 은 거리 0. 동률이면 회전 우선 — 창 끝과 연결로
        # 시작이 같은 지점인 경우(LC2→회전2, LC3→회전3) 회전이 법규상 더 무겁다.
        turn = self._turn_signal(world)
        lc_sig = None
        if lc is not None and lc['signal'] != TURN_OFF:
            lc_sig = (lc['signal'], 0.0 if lc['active'] else max(0.0, lc['dist']))
        if turn is not None:
            reasons['turn_dist'] = round(turn[1], 1)
        if lc_sig is not None and (turn is None or lc_sig[1] < turn[1]):
            turn_signal = lc_sig[0]
            reasons['sig_src'] = 'lc'
        elif turn is not None:
            turn_signal = turn[0]
            reasons['sig_src'] = 'turn'
        # 연속 점등 추적 (다음 틱의 LC 실행 조건이 이 값을 본다)
        if turn_signal != self._sig_dir:
            self._sig_dir = turn_signal
            self._sig_since = world.t if turn_signal != TURN_OFF else None
        if lc is not None:
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
        self._signal_fallback = False
        self._rtor_note = ''
        self._cw_note = ''
        self._lead_creep = False
        cand = self._speed_candidates(world)
        reasons.update(cand)
        if self._signal_fallback:
            reasons['signal_fallback'] = True
        if self._rtor_note:
            reasons['rtor'] = self._rtor_note
        if self._cw_note:
            reasons['crosswalk'] = self._cw_note
        reasons['blocked'] = self._blocked
        if self._blocked_why:
            reasons['blocked_why'] = self._blocked_why
        if self._lead_creep:
            reasons['lead_creep'] = True
        if self._last_prog is not None:
            reasons['lc_prog'] = round(self._last_prog, 2)
        if self._overtake is not None:
            reasons['overtake'] = ('return' if self._overtake.get('returning') else 'out')
        pool = dict(cand)
        if 'debug_const' in reasons:
            pool['debug_const'] = reasons['debug_const']
        v_target = min([v_target] + list(pool.values()))
        if not math.isfinite(v_target):
            v_target = 0.0                      # 후보가 하나도 없다 = 근거 없음 → 정지
            reasons['no_candidate'] = 0.0
        reasons['v_target'] = round(v_target, 3)
        winner = min(pool, key=pool.get) if pool else None
        reasons['winner'] = winner if winner else 'none'

        # 정지선 때문에 서는 중이면 FSM 상태로 드러낸다 (차선변경이 우선)
        if self.state != LANE_CHANGE and winner == 'stop_line' and (v_target < 0.5 or self._stop_latch):
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
        sp = self.cfg['speed']
        # 계획 감속도는 a_comf 에 여유를 둔다. 제어기가 프로파일을 즉시 따라오지
        # 못해서(램프 추종 지연) a_comf 를 그대로 쓰면 정지선을 넘어간다.
        a = float(sp['a_comf']) * float(sp.get('a_plan_factor', 1.0))
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

        # 추종 대상 선행차를 **먼저** 정한다. 횡단보도 정지(4c)와 IDM(6)이 같은
        # 판정을 쓰게 하려는 것이다 — 선행차는 서서 기다릴 대상이 아니라 따라갈
        # 대상이라 횡단보도 차단 객체에서 빼야 한다 (폐루프 시뮬: 선행차가 횡단보도
        # 위를 지날 때 crosswalk_ped 가 winner 가 되어 이중 제동).
        self._lead = self._select_lead(world)
        self.lead_id = self._lead.id if self._lead is not None else None

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
        # 횡단보도 캡은 없다 — 채점표의 속도 항목은 S1.1.01(제한속도)과
        # S1.1.02(스쿨존) 뿐이고, S6.3.03 은 "횡단보도 **정차** 금지"다.
        # 예전의 25 km/h 캡은 2026-08-21 주행에서 137틱을 눌러 평균 19 km/h 를
        # 만들었다 (완주에는 36 km/h 필요).
        for a in world.ahead:
            if a.kind == 'junction_in':
                cap = float(caps['junction']) / 3.6
                out['junction'] = min(out.get('junction', math.inf),
                                      self._approach(cap, a.dist))

        # ── 4) 정지선 + 신호 ─────────────────────────────────────────────
        v_stop = self._signal_speed(world, out)
        if v_stop is not None:
            out['stop_line'] = v_stop

        # ── 4b) 경로 끝 정지 ─────────────────────────────────────────────
        # 지정 경로 이탈은 감점이다. 경로 차로열이 끝나는 지점에 정지선처럼
        # 감속-정지하고, 충분히 느려지면 래치한다 — 끝을 지나치면 lookahead 의
        # route_end 가 시야에서 사라지므로(off_route + 낡은 idx) 래치 없이는
        # 브레이크를 놓아 그대로 계속 달린다 (2026-08-21: 완주 후 93 m 초과 주행).
        if self._route_end_latch:
            out['route_end'] = 0.0
        else:
            d_end = (world.summ or {}).get('dist_route_end')
            if d_end is not None:
                # 정지선과 같은 제어 지연 보상(v·stop_lag_s). 이게 없어 목표보다
                # 2.07 m 넘어 섰다 (2026-08-23 13:59 런: 앞범퍼가 경로 끝 +1.07 m).
                lag = float(sp.get('stop_lag_s', 0.0)) * max(world.ego.speed, 0.0)
                v_end = self._approach(
                    0.0, d_end - float(sp['stop_gap_m']) - self._front_m() - lag)
                if v_end < float(sp.get('stop_latch_v', 1.5)):
                    self._route_end_latch = True
                    v_end = 0.0
                out['route_end'] = v_end

        # ── 4c) 횡단보도 보행자 → 정지선 정지 ────────────────────────────
        v_cw = self._crosswalk_ped_speed(world)
        if v_cw is not None:
            out['crosswalk_ped'] = v_cw

        # ── 5) 가시거리 (GT 80 m 밖은 안 보이는 것과 같다) ───────────────
        out['visibility'] = self.visibility_limit()

        # ── 5b) 정차 차량에 갇혔는가 (추월 판정) ─────────────────────────
        self._update_blocked(world, out)

        # ── 6) 선행차 추종 (IDM) — **맨 마지막** ─────────────────────────
        # 자유주행 목표 v0 는 여기까지의 후보 min 이다. 제한속도·곡률·정지선이
        # 이미 깎아 놓은 상한을 IDM 이 되올리지 않게 한다.
        finite = [v for v in out.values() if math.isfinite(v)]
        v0 = min(finite) if finite else world.speed_limit
        v_lead = self._idm(world, v0)
        if v_lead is not None:
            out['lead'] = v_lead

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

        # 지도 곡률 이상치 방어. 빌드 단계(build_lane_graph.smooth_curvature)에서
        # 스파이크를 걸러내지만, 낡은 lane_graph.pkl 로 돌 수도 있고 걸러지지 않은
        # 것이 남을 수도 있다. **이웃 중앙값의 ratio 배를 넘는 고립 값은 건너뛴다.**
        # (2026-08-23: 직선 도로에 반경 6.5 m 가 한 점 찍혀 45 → 16 km/h 로 감속)
        ratio = float(self.cfg['speed'].get('curv_outlier_ratio', 5.0))

        best = None
        acc = 0.0
        s0 = world.ego.s
        i = idx
        while acc < horizon and i < len(lanes):
            r = self.lg.lanes[lanes[i]]
            ss, cv = r['s'], r['curv']
            n_pt = len(ss)
            for j in range(n_pt):
                if ss[j] < s0:
                    continue
                d = acc + (ss[j] - s0)
                if d > horizon:
                    break
                k = abs(float(cv[j]))
                if k < 1e-4:                     # 반경 1 km 이상 = 직선
                    continue
                if ratio > 0.0 and n_pt >= 5:
                    # 창은 ±3, 중앙값은 **아래쪽**을 고른다. 스파이크가 두 점
                    # 연속으로 찍히는 경우가 흔한데(섹션 끝 cv[-2], cv[-1]),
                    # 위쪽 중앙값을 쓰면 두 점이 서로를 가려 걸러지지 않는다.
                    lo, hi = max(0, j - 3), min(n_pt, j + 4)
                    nb = sorted(abs(float(cv[m])) for m in range(lo, hi) if m != j)
                    if nb and k > ratio * max(nb[(len(nb) - 1) // 2], 1e-4):
                        continue                 # 이웃 대비 고립 스파이크 — 무시
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
        sp = self.cfg['speed']
        # 정지 목표는 **앞범퍼**가 정지선 − stop_gap 에 오는 것이다. ego 좌표는
        # 뒷바퀴축(SPEC §1.3)이므로 dist_stop_line 에서 차체 앞부분(wheelbase +
        # front_overhang)을 더 빼야 한다. 이게 없어 3/3 정지 모두 앞범퍼가 정지선을
        # 3~5 m 넘었다 (2026-08-23 12:51 런).
        gap = float(sp['stop_gap_m']) + self._front_m()
        yellow_s = float(self.cfg['signal']['yellow_s'])
        v = max(world.ego.speed, 0.0)
        self._v_now = v
        s_now = world.ego.route_s

        UNASSIGNED, GREEN, GREEN_LEFT, LEFT, YELLOW, RED, FLASH = 0, 3, 5, 4, 2, 1, 6

        # ── 래치 무결성: 대상 정지선을 명백히 지났으면 **무조건** 해제한다 ──────
        # light 소멸 분기 안에만 있던 검사를 여기로 올렸다. 신호가 계속 오는 동안
        # 지나쳐 버린 경우에도 풀려야 하고, 무엇보다 _stop_line_s 가 래치 중에
        # 갱신되지 않아야(아래) 이 비교가 의미를 갖는다.
        if (self._stop_latch and self._stop_line_s is not None
                and s_now - self._stop_line_s > 5.0):
            self._stop_latch = False
            self._rtor_reset()

        if world.light is None:
            # 9910 light 필드는 ego 가 정지선 도로를 벗어나 연결로에 들어서는 순간
            # 통째로 사라진다. 예전엔 여기서 무조건 래치를 풀어, 감속 중 소멸과
            # 겹치면 완전정지 전에 재가속했다 (12:51 런 3/3 전부). **래치가 걸린 뒤
            # light 소멸은 해제 사유가 아니다.** 해제는 녹색류 수신 / RTOR go /
            # 명백한 통과(정지선을 5 m 이상 지남) 뿐이다.
            if self._stop_latch:
                if self._rtor_dwell_done:      # 통과 해제는 위에서 이미 처리했다
                    self._stop_latch = False
                    self._rtor_reset()
                    return None
                self._rtor_note = 'hold: light lost'
                return 0.0
            self._rtor_reset()
            return None
        state = int(world.light[1])
        nxt = summ.get('next_turn')
        if state != RED:
            self._rtor_reset()

        # 통과 신호에서는 래치를 푼다 — 래치 검사보다 **먼저** 매 틱 재평가한다.
        # 녹(3)/녹+좌(5), 그리고 미할당(0: 지금 따를 신호가 없다)과
        # 좌회전(4) 중 경로가 좌회전인 경우. 예전에는 3/5 에서만 풀어서,
        # 적신호에 래치된 채 state 가 0 이나 4(+좌회전 경로)로 바뀌면 영구
        # 정차했다 (19:43 로그에 id=30/state=0 전이 실측 — 발생 가능한 시나리오).
        # 해제 후에도 아래 분기에서 적/황이면 _stop_at 이 다시 래치한다.
        if state in (GREEN, GREEN_LEFT, UNASSIGNED) or (state == LEFT and nxt == 'turn_left'):
            self._stop_latch = False

        # 녹색/미할당이면 통과 (**녹색신호 통과도 채점 항목이다. 불필요한 정지 금지**)
        if state in (GREEN, GREEN_LEFT, UNASSIGNED):
            return None

        d = summ.get('dist_stop_line')
        # 래치 대상 정지선의 경로거리는 **래치 시점 값으로 고정(freeze)** 한다.
        # 매 틱 갱신하면 정지선을 지나는 순간 lookahead 가 **다음 교차로** 정지선을
        # 가리키면서 _stop_line_s 가 그쪽으로 덮어써지고, 해제 조건
        # (route_s > _stop_line_s + 5) 을 영영 만족하지 못해 그 자리에 영구 정지한다
        # (2026-08-23 17:49 런: 378 m 에서 23.8 s 정지, 기준이 494 m 로 밀림).
        # 차선변경이 blend base 를 시작 시점 차로로 고정하는 것과 같은 이유다 —
        # **판단의 기준점은 확정 시점에 얼어붙어야 한다.**
        if d is not None and not self._stop_latch:
            self._stop_line_s = s_now + d

        # ── 통과한 정지선으로는 새로 세우지 않는다 ─────────────────────────
        # 앞범퍼가 이미 정지선을 지났으면 그 교차로 신호로 감속할 이유가 없다.
        # 황색으로 딜레마존 통과 직후 적색으로 바뀌면 dist_stop_line 이 아직
        # 양수(뒷바퀴축 기준)라 정지 후보가 생겨 교차로 한복판에서 급정거했다
        # (같은 런: 앞범퍼 +1.0 m 통과 시점에 발동, -3.94 m/s²).
        # 이미 래치된 정지(멀리서부터 정상 감속 중)는 건드리지 않는다.
        passed_front = d is not None and d <= self._front_m()
        if d is not None and not (summ.get('stop_signal_ids') or []):
            # 정지선에 신호 id 가 안 붙어 있다. 두 경우다:
            #  · 진짜 비신호 정지선(일시정지/양보) — 아직 다루지 않는다
            #  · 빌드 시 연결 누락 — 신호등이 교차로 연결로(junction road) 상공에
            #    달려 있으면 "같은 도로 20 m 이내" 규칙에 안 걸린다 (도로 30 정지선,
            #    controller 3 = 도로 556 의 signal 30/31/34, 25 m 떨어짐).
            #    2026-08-23 11:57 런: 적색(3,1) 인데 비신호로 보고 통과 → S5.1.01.
            # 9910 의 light 는 VTD 가 **ego 접근로 기준으로 골라 주는** controller
            # 이므로(light_ctrl_match 계층으로 확인), light 가 유효(id != 0)하면 그
            # state 를 이 정지선에 적용한다(폴백). 단 light_ctrl_match 가 명시적으로
            # False(수신 id 가 이 정지선 controller 와 불일치 확인)면 폴백하지 않는다.
            if int(world.light[0]) == 0 or world.flags.get('light_ctrl_match') is False:
                d = None                     # 폴백 불가 → 비신호 정지선으로 본다
            else:
                self._signal_fallback = True

        # ── RTOR: 적색 + 다음 회전이 우회전 — 완전 정지 → dwell → 안전 확인 → 서행 ──
        if state == RED and nxt == 'turn_right' and (
                (d is not None and not passed_front) or self._stop_latch):
            r = self._rtor(world, d)
            if r is not None:
                return r

        # 정지 확정 상태면 정지선이 시야를 벗어나도 계속 선다.
        # 이게 없으면 정지선이 lookahead 를 벗어나는 순간 브레이크를 놓아
        # 적신호에 그대로 넘어간다 (실측: 정지선 0.13 m 통과).
        if self._stop_latch:
            if d is None or not bool(sp.get('stop_continuous', True)):
                return 0.0
            lag = float(sp.get('stop_lag_s', 0.0)) * v
            return self._approach(0.0, d - gap - lag)   # 연속 프로파일 (0 에서 자연히 0)
        if d is None or passed_front:
            return None
        if state == LEFT:
            return None if nxt == 'turn_left' else self._stop_at(d, gap)
        if state == YELLOW:
            if d - self._front_m() <= v * yellow_s:
                return None                  # 딜레마존(앞범퍼 기준) — 통과가 안전
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

    # ── RTOR (적신호 우회전) ──────────────────────────────────────────────
    def _rtor_reset(self) -> None:
        self._rtor_stop_t = None
        self._rtor_dwell_done = False

    def _rtor(self, world: WorldState, d: float | None) -> float | None:
        """
        적신호 우회전 상태기계. None 을 돌려주면 일반 정지 로직(_stop_at/래치)이 이어진다.

          1) 정지선 stop_gap 앞 완전 정지 (래치 + v < 0.2) 가 signal.stop_dwell_s 유지
          2) 그 뒤 매 틱 안전 확인: 통과 경로(우회전 연결로 + 횡단보도)에 객체 없음,
             접근 차량 TTC > ttc.warn_s, 진입 예측(will_enter_lane) 없음
          3) 안전하면 signal.rtor_speed_kph 로 서행 통과, 아니면 정지 유지(재평가)
        signal.rtor_enabled=false 면 전부 건너뛴다(녹색 대기). 직진·좌회전 경로는
        호출되지 않는다.
        """
        sig = self.cfg['signal']
        if not bool(sig.get('rtor_enabled', True)):
            return None
        t = world.t
        v = world.ego.speed
        if not self._rtor_dwell_done:
            if self._stop_latch and v < 0.2:
                if self._rtor_stop_t is None:
                    self._rtor_stop_t = t
                held = t - self._rtor_stop_t
                if held >= float(sig.get('stop_dwell_s', 1.0)):
                    self._rtor_dwell_done = True
                else:
                    self._rtor_note = f'dwell {held:.1f}s'
                    return None              # 정지 유지 (래치가 0 을 낸다)
            else:
                self._rtor_stop_t = None
                self._rtor_note = 'stopping'
                return None                  # 아직 정지 전 — 일반 정지 프로파일
        why = self._rtor_blocked(world)
        if why:
            self._rtor_note = f'hold: {why}'
            return 0.0
        self._rtor_note = 'go'
        return float(sig.get('rtor_speed_kph', 20.0)) / 3.6

    def _crosswalk_ped_speed(self, world: WorldState) -> float | None:
        """
        전방 횡단보도에 보행자(또는 진입 예측 객체)가 있으면 **그 앞 정지선**에
        정지하기 위한 속도 상한. 없으면 None.

        - **신호 상태와 무관**하게 활성이다. 녹색이어도 횡단보도에 사람이 있으면
          선다 (도로교통법 27조, 채점 "보행자 출현" 항목).
        - 목표는 그 횡단보도에 선행하는 정지선. 정지선이 없으면 횡단보도 경계.
          정지 프로파일은 정지선과 동일(전장 보정 + stop_lag 선행 보상).
        - 보행자가 폴리곤을 벗어나고 진입 예측도 아니면 래치를 풀어 재출발한다.
        - **이미 정지선을 지나쳤으면 활성화하지 않는다** — 횡단보도 안에 서는 것은
          그 자체가 감점이다(S6.3.03). 그 상황은 TTC 비상제동의 몫이다.
        """
        summ = world.summ or {}
        d_cw = summ.get('dist_crosswalk')
        # 추종 중인 선행차는 제외한다 (회피·대기 대상이 아니라 따라가는 대상).
        objs = [o for o in world.objects if o.id != self.lead_id]
        blockers = crosswalk_blockers(objs, d_cw, self.cfg)
        if not blockers:
            self._cw_ped_latch = False
            return None

        sp = self.cfg['speed']
        gap = float(sp['stop_gap_m']) + self._front_m()
        # 그 횡단보도에 선행하는 정지선(= 횡단보도보다 앞이거나 같은 지점). 없으면 횡단보도 경계.
        d_sl = summ.get('dist_stop_line')
        d_target = d_sl if (d_sl is not None and d_sl <= d_cw + 1.0) else d_cw

        if self._cw_ped_latch:
            self._cw_note = 'stop: %s' % ','.join('id%d' % o.id for o in blockers[:3])
            return 0.0
        lag = float(sp.get('stop_lag_s', 0.0)) * max(world.ego.speed, 0.0)
        room = d_target - gap - lag
        if room <= 0.0:
            # 이미 정지선/횡단보도를 지났다 — 서면 횡단보도 정차가 된다. 통과.
            self._cw_note = 'pass: 정지선 통과 후 감지'
            return None
        v = self._approach(0.0, room)
        if v < float(sp.get('stop_latch_v', 1.5)):
            self._cw_ped_latch = True
            v = 0.0
        self._cw_note = 'slow: %s (%.0f m)' % (
            ','.join('id%d' % o.id for o in blockers[:3]), d_target)
        return v

    def _rtor_blocked(self, world: WorldState) -> str:
        """우회전 통과 경로가 막혀 있으면 사유, 아니면 ''. 타입 구분 없이 전부 막는다(보수적)."""
        summ = world.summ or {}
        warn_s = float(self.cfg['ttc']['warn_s'])
        s_now = world.ego.route_s
        cur = next((tn for tn in self._turns if s_now < tn['end_s']), None)
        d_end = (cur['end_s'] - s_now) if cur else 30.0
        d_cw = summ.get('dist_crosswalk')
        for o in world.objects:
            if o.ttc < warn_s:
                return f'id={o.id} ttc={o.ttc:.1f}s'
            if o.will_enter_lane:
                return f'id={o.id} 진입 예측'
            if o.on_route and -2.0 <= o.s_rel <= d_end + 5.0:
                return f'id={o.id} 통과 경로 {o.s_rel:+.0f} m'
        for o in crosswalk_blockers(world.objects, d_cw, self.cfg):
            return f'id={o.id} 횡단보도 {o.s_rel:+.0f} m'
        return ''

    def _front_m(self) -> float:
        """뒷바퀴축 → 앞범퍼 거리 = wheelbase + front_overhang."""
        vh = self.cfg['vehicle']
        return float(vh['wheelbase']) + float(vh.get('front_overhang_m', 0.855))

    def _stop_at(self, dist: float, gap: float) -> float:
        """
        정지선 dist 앞, gap 여유(차체 앞부분 포함)를 두고 정지하기 위한 현재 허용 속도.
        충분히 느려지면 정지를 확정(래치)해 정지선이 시야를 벗어나도 유지한다.

        speed.stop_continuous=true 면 래치 뒤에도 0 으로 스냅하지 않고 _approach
        곡선을 v=0 까지 연속으로 낸다 — 스냅은 목표를 한 틱에 수 m/s 떨어뜨려
        P 제어 + 저크 제한이 못 따라가 정지선을 넘긴다.
        """
        sp = self.cfg['speed']
        # 제어 지연 보상: P 제어는 내려가는 목표를 a_plan/kp ≈ 1.3 m/s 늦게 따라가
        # 정지점을 ~2 m 넘긴다 (폐루프 시뮬, 로그와 ±0.1 m 일치). 목표를 v·stop_lag_s
        # 만큼 앞당긴다 — 속도에 비례하므로 저속에선 사라져 일찍 서지 않는다.
        lag = float(sp.get('stop_lag_s', 0.0)) * max(self._v_now, 0.0)
        v = self._approach(0.0, dist - gap - lag)
        if v < float(sp.get('stop_latch_v', 1.5)):
            self._stop_latch = True
            if not bool(sp.get('stop_continuous', True)):
                return 0.0
        return v if (self._stop_latch and bool(sp.get('stop_continuous', True))) else v

    # ══════════════════════════════════════════════════════════════════════
    # 차선변경
    # ══════════════════════════════════════════════════════════════════════
    def _pending_lane_change(self, world: WorldState) -> dict | None:
        """아직 안 끝난 가장 가까운 lane_change 이벤트 (추월 합성 이벤트 우선)."""
        ov = self._overtake_event(world)
        if ov is not None:
            return ov
        if not self.route:
            return None
        # 경로를 벗어났으면 route_s 가 의미 없다 — 차로가 바뀔 때마다 0 부근으로
        # 되돌아가며 요동친다. 그 값으로 창을 판정하면 엉뚱한 이벤트가 선택되고,
        # 한 번 켜진 _lc_signal_on 래치는 이벤트가 안 바뀌므로 계속 켜져 있다
        # (2026-08-23 14:31 런: 경로 불일치로 전 구간 off_route, 좌회전 지시등이
        #  sig_src='lc' 로 오점등). 경로 위가 아니면 계획 차선변경은 하지 않는다.
        if (world.flags or {}).get('off_route'):
            return None

        s_now = world.ego.route_s
        # 리셋(리스폰)으로 창 시작 앞까지 되돌아갔으면 완료 기록을 풀어 다시 한다
        # (완료는 창 안에서만 일어나므로 s_now < window_s0 는 되돌아간 것이다)
        self._lc_finished = {k for k in self._lc_finished if s_now >= k}
        best = None
        for e in self.route.get('events', []):
            if not e['kind'].startswith('lane_change'):
                continue
            if e.get('to_lane') is None:
                continue
            if s_now > e['window_s1']:
                continue                      # 창을 이미 지났다
            if e['window_s0'] in self._lc_finished:
                continue                      # 이미 끝냈다 — 재선택 금지
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
            self._lc_signal_ev = None
            return None
        # pending 이 다른 이벤트로 바뀌었다 → 래치 리셋, 새 이벤트 기준으로 재판단
        if self._lc_signal_ev != ev['window_s0']:
            self._lc_signal_on = False
            self._lc_signal_ev = ev['window_s0']

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
        w1 = self._window_s1(ev)
        if s_now > w1:
            if self._lc is None:
                return None                       # 시작도 못 했다 — 조용히 넘긴다
            still_dashed = self.lg.lane_change_ok(world.ego.lane, world.ego.s, side)
            over = s_now - w1
            if not still_dashed or over > float(lc['transition_min_m']):
                self._finish_lane_change(
                    ev, ok=False,
                    why=f"창을 {over:.0f} m 넘겼다 (점선={still_dashed})")
                return None

        # 완료 판정 (이미 목표 차로에 안착했나)
        if self._lc is not None and self._lane_change_done(world, target):
            self._finish_lane_change(ev, ok=True)
            return None

        # ── 최소 속도 게이트 ────────────────────────────────────────────
        # 블렌드 진행도가 **주행거리**(route_s − s_start)에 비례하므로, 너무 느리면
        # 전이가 진행되지 않는데 조향 목표만 목표 차로 쪽으로 고착돼 차가 그대로
        # 옆으로 밀려 나간다 (2026-08-23 19:56 런: v=0.84 m/s 에서 시작 → 도로
        # 이탈, 차로 id -3 → +5 로 반대편 차선 진입, shield 발동 0회).
        v_min = float(lc.get('v_min_mps', 2.0))
        # blocked 추월은 **정지 출발**이 정상이므로 면제한다 — IDM 이 blocked_gap 앞
        # 정지를 만들어 지시등 3 s 를 채울 때쯤 반드시 v≈0 이 되고, 게이트를 그대로
        # 두면 시작이 영구 거부된다 (2026-08-24 13:10 런: 교착 확정). 조향 고착은
        # 블렌드의 시간 항이, 새 교착은 아래 정체 워치독이 막는다.
        vmin_exempt = bool(ev.get('overtake'))
        if vmin_exempt:
            reasons['lc_vmin_exempt'] = 'overtake'
        if self._lc is not None and not vmin_exempt and world.ego.speed < v_min:
            self._finish_lane_change(
                ev, ok=False, why='속도 %.2f < v_min %.1f m/s — 전이 중단' % (world.ego.speed, v_min))
            reasons['lc_slow_abort'] = round(world.ego.speed, 2)
            return None

        # ── [작업 3] 전이 정체 워치독: v≈0 인 채 진행이 없으면 중단·복귀 ────
        if self._lc is not None:
            if world.ego.speed < 0.2:
                t0 = self._lc.setdefault('stall_t', world.t)
                if world.t - t0 > float(lc.get('stall_abort_s', 3.0)):
                    self._finish_lane_change(
                        ev, ok=False, why='전이 정체 %.1fs — 중단' % (world.t - t0))
                    reasons['lc_stall_abort'] = round(world.t - t0, 1)
                    return None
            else:
                self._lc.pop('stall_t', None)

        dist_to_window = ev['window_s0'] - s_now
        # 지시등: 창 시작 (lead_s + margin_s) 초 전부터
        lead_m = v * (float(sig['lead_s']) + float(sig['margin_s']))
        want_signal = dist_to_window <= lead_m
        if want_signal and not self._lc_signal_on:
            self._lc_signal_on = True

        in_window = ev['window_s0'] <= s_now <= w1

        # ── 점등 선행 조건: 해당 방향이 lc_lead_min_s 이상 **연속 점등**됐는가 ──
        # 거리 기준(dist ≤ v·lead_s)이 아니라 실제 출력 지시등 기준이다. 회전 중에는
        # 우선순위 규칙으로 반대 방향이 켜져 있으므로 회전이 끝나고 이 방향이 실제로
        # 켜진 시점부터 센다. shield 긴급 회피(emergency_avoid)는 예외 — 즉시 실행.
        lead_min = float(sig.get('lc_lead_min_s', 3.0))
        sig_lead_s = 0.0
        if self._sig_dir == signal and self._sig_since is not None:
            sig_lead_s = max(0.0, world.t - self._sig_since)
        exempt = bool(self.emergency_avoid)
        signaled_long_enough = exempt or sig_lead_s >= lead_min
        reasons['sig_lead_s'] = round(sig_lead_s, 2)
        if exempt:
            reasons['lc_lead_exempt'] = True

        clear, why = self._target_lane_clear(world, target, side)
        reasons['lc_dist_to_window'] = round(dist_to_window, 1)
        reasons['lc_clear'] = clear
        if clear and not vmin_exempt and world.ego.speed < v_min:
            reasons['lc_too_slow'] = round(world.ego.speed, 2)

        # ── 대기로 창이 모자라는 경우 ──────────────────────────────────────
        # 남은 창 < 전이거리면 (1) 창 확장 시도 → (2) 불가하면 대기 중단, 즉시 실행
        # (창 실패보다 lead 부족이 낫다). 둘 다 로그에 남긴다.
        if self._lc is None and in_window and clear and not signaled_long_enough:
            L = max(float(lc['transition_s']) * max(world.ego.speed, 1.0),
                    float(lc['transition_min_m']))
            if w1 - s_now < L:
                new_w1 = self._try_extend_window(ev, side, need=L - (w1 - s_now))
                if new_w1 is not None and new_w1 - s_now >= L:
                    self._win_ext[ev['window_s0']] = new_w1
                    w1 = new_w1
                    reasons['lc_window_ext'] = round(new_w1 - ev['window_s1'], 1)
                else:
                    signaled_long_enough = True
                    reasons['lead_short'] = round(sig_lead_s, 2)

        active = False
        if self._lc is not None:
            # 이미 전이 중 — 점선인 동안에는 끝까지 간다
            active = True
        elif in_window and clear and signaled_long_enough and (
                vmin_exempt or world.ego.speed >= v_min):
            self._lc = dict(ev)               # 실행 시작
            # 전이 진행도의 기준점. 자차 위치에 매 틱 다시 고정하면
            # 목표의 일부 지점만 계속 쫓게 되어 차선을 끝까지 못 넘는다.
            self._lc['s_start'] = s_now
            self._lc['t_start'] = world.t
            # 블렌드 base 의 기준 차로. 매칭 차로를 그대로 쓰면 차선을 넘는
            # 순간 base 가 원차로 → 목표차로 중심선으로 불연속 교체되어
            # 경계에서 정체 + 조향 요동이 난다 (2026-08-21 실측: 1.2 s 정체,
            # 조향 +0.36 → -0.24 요동, 순간 최대 0.359 rad).
            self._lc['src_lane'] = tuple(world.ego.lane)
            if ev.get('overtake') and self._overtake is None and self._lead is not None:
                self._overtake = {'blocker_id': self._lead.id, 'side': side,
                                  'returning': False}
                # 블렌드 전이거리 상한: 그 차 **뒷범퍼**까지의 거리 (위 _blend_path 참고)
                self._lc['ov_gap'] = max(
                    0.0, self._lead.s_rel - self._front_m()
                    - 0.5 * float(self._lead.length))
            active = True
        left = w1 - s_now
        if in_window and left < float(lc['min_window_m']) and ev['window_s0'] not in self._lc_warned:
            self._lc_warned.add(ev['window_s0'])
            reasons['lc_warn'] = (
                f'차선변경 창이 {left:.0f} m 남았는데 아직 못 끝냈다 — '
                + (why if not clear else '전이 진행 중')
                + '. 놓치면 다음 교차로에서 경로를 못 따라간다')

        return {'signal': signal if (self._lc_signal_on or want_signal) else TURN_OFF,
                'active': active, 'target': target, 'event': ev, 'side': side,
                'dist': dist_to_window}

    def _window_s1(self, ev: dict) -> float:
        return float(self._win_ext.get(ev['window_s0'], ev['window_s1']))

    def _try_extend_window(self, ev: dict, side: str, need: float) -> float | None:
        """
        창 끝(window_s1) 뒤로 차선변경이 계속 가능하면 연장된 s1 을 돌려준다.

        build_route.lane_change_window 와 같은 규칙의 전방 버전: 원차로의
        successor 체인을 따라가며 (교차로 연결로 아님) ∧ (그 방향 이웃 = 목표
        차로 체인의 successor) ∧ (그 방향 차선이 점선) 인 동안 길이를 더한다.
        need 만큼 확보되면 멈춘다. 한 구간도 못 늘리면 None.
        """
        from_lane = tuple(ev.get('from_lane') or ev['lane'])
        tgt = tuple(ev['to_lane'])
        w1 = float(ev['window_s1'])
        k, nb = from_lane, tgt
        ext = 0.0
        for _ in range(8):
            nxt = self.lg.successors(k)
            if len(nxt) != 1:
                break
            k = nxt[0]
            r = self.lg.lanes[k]
            if r['junction'] != -1 or r['type'] != 'driving':
                break
            nb_k = self.lg.neighbor(k, side)
            if nb_k is None or nb_k not in self.lg.successors(nb):
                break
            nb = nb_k
            L = float(r['length'])
            step = 1.0
            s = 0.0
            while s < L and self.lg.lane_change_ok(k, s, side):
                s += step
            ok_len = min(s, L)
            ext += ok_len
            if ok_len < L - 0.5 or ext >= need:
                break
        return (w1 + ext) if ext > 0.5 else None

    def _target_lane_clear(self, world: WorldState, target, side: str) -> tuple[bool, str]:
        """목표 차로가 driving 이고, 점선이고, 뒤/앞 범위가 비었는가.

        `side` 는 반드시 호출자가(= 경로 이벤트 kind 가) 준 값을 쓴다. 차로 id
        크기로 좌/우를 되짚으면 안 된다 — 좌측(양수 id) 차로는 주행방향이 도로
        s 와 반대라 id 순서가 뒤집힌다. 실제로 (128,3,2) -> (128,3,3) 우측
        차선변경이 id 비교로는 'left' 로 나와, 중앙선(황색 실선)을 보고
        "실선이라 불가" 로 판정해 창에 들어가도 시작하지 못했다.
        """
        lc = self.cfg['lane_change']
        ego = world.ego
        rec = self.lg.lanes.get(target)
        if rec is None or rec['type'] != 'driving':
            return False, '목표 차로가 주행 차로가 아니다'

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
            if not ev.get('overtake'):
                self._lc_finished.add(ev['window_s0'])
        elif self._lc is not None:
            self.lc_aborted += 1
        self._win_ext.pop(ev['window_s0'], None)
        self._lc = None
        self._lc_signal_on = False
        self._lc_signal_ev = None
        self.last_lc_note = why

    def abort_lane_change(self, why: str = '') -> None:
        """shield 가 위험을 감지했을 때 호출. 원래 차로로 되돌린다."""
        if self._lc is not None:
            self.lc_aborted += 1
        self._lc = None
        self._lc_signal_on = False
        self._lc_signal_ev = None
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

        # base 는 **LC 시작 시점의 원차로 체인**에 고정한다. 인자로 받은 base 는
        # 매칭 차로 기준이라, 자차가 차선을 넘어 목표 차로로 매칭되는 순간
        # 원차로 → 목표차로 중심선으로 불연속 교체된다(경계 정체 + 조향 요동).
        # 완료 판정(_lane_change_done: |t_off|<0.3, |herr|<5°) 후에는 _lc 가
        # 풀리므로 자연히 목표 차로 base 로 넘어간다 — 그 시점엔 이미 중심선에
        # 붙어 있어 불연속이 없다.
        #
        # 반드시 **체인**을 따라 재투영해야 한다. 섹션이 짧아(128 도로 12 m)
        # 자차가 원차로 구간을 지나치면 단일 차로 투영은 끝점에 클램프되고,
        # base 가 자차 뒤 수십 m 에서 시작해 경로가 퇴화한다 (mock 폐루프
        # 실측: 조향이 +0.011 에 눌린 채 차로를 흘러 나감).
        src = tuple(self._lc['src_lane']) if (self._lc and self._lc.get('src_lane')) else None
        if src is not None and tuple(world.ego.lane) != src:
            k = src
            for _ in range(8):                    # 자차 위치까지 체인을 따라간다
                s_b, _tb, _db, _jb = self.lg.project(k, world.ego.x, world.ego.y)
                if s_b < self.lg.length(k) - 0.25:
                    break
                nxt = self.lg.successors(k)
                if not nxt:
                    break
                k = nxt[0]
            if tuple(world.ego.lane) != k:        # 매칭 차로와 같으면 재계산 불필요
                nb = self.lg.points_ahead(k, s_b, dist=float(d['path_dist_m']), step=step,
                                          route=self.route, idx=self._route_idx_of(k))
                if len(nb) >= 2:
                    base = nb

        s_t, _t, _dd, _j = self.lg.project(target, world.ego.x, world.ego.y)
        tgt = self.lg.points_ahead(target, s_t, dist=float(d['path_dist_m']), step=step,
                                   route=self.route, idx=self._route_idx_of(target))
        if len(tgt) < 2 or len(base) < 2:
            return None

        n = min(len(base), len(tgt))
        L = max(float(cfg['transition_s']) * max(world.ego.speed, 1.0),
                float(cfg['transition_min_m']))
        # 정차 차량 추월은 **그 차 뒷범퍼에 도달하기 전에 횡이동이 끝나야** 한다.
        # 일반 전이거리(최소 20 m)를 그대로 쓰면 절반 지점(횡이동 최대 기울기)이
        # 그 차 **뒤**에 놓여 스치고 지나간다 (폐루프: 최소 접근 1.91 m 충돌).
        # 시작 시 저장한 뒷범퍼까지 거리로 상한을 걸되, 최소 회전 기하(S 커브
        # ≈ 10 m) 아래로는 줄이지 않는다 — 그보다 짧으면 물리적으로 못 넘는다.
        if self._lc.get('overtake') and self._lc.get('ov_gap') is not None:
            L = max(10.0, min(L, float(self._lc['ov_gap'])))

        # 진행도는 **경로상 고정 시작점** 기준이다.
        # 매 틱 자차 위치를 0 으로 두면 전방 점의 가중치가 늘 같은 값에 머물러
        # 차가 목표 차로의 일부 지점(≈0.5 m)만 쫓다가 넘어가지 못한다.
        # 진행도는 **거리와 시간 중 더 많이 진척된 쪽**을 쓴다.
        # 거리 단독이면 저속에서 전이가 멈춘 채 조향만 고착된다. 시간 항은
        # v_min 으로 달렸다면 갔을 거리로 환산해, 속도가 떨어져도 전이가 끝까지
        # 진행되어 차가 어느 한쪽 차로에는 반드시 안착하게 한다.
        s_start = self._lc.get('s_start', world.ego.route_s) if self._lc else world.ego.route_s
        prog = max(0.0, world.ego.route_s - s_start)
        if self._lc is not None:
            t_start = self._lc.get('t_start')
            if t_start is not None:
                v_min = float(cfg.get('v_min_mps', 2.0))
                prog = max(prog, v_min * max(0.0, world.t - t_start))
        self._last_prog = prog

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

    def _update_blocked(self, world: WorldState, out: dict) -> None:
        """
        선행차가 **서 있고, 신호 때문이 아니면** 'blocked' 로 판정한다 → 추월 대상.

        전부 만족해야 한다 (하나라도 아니면 즉시 해제):
          1) 선행차가 있고 속도 < 0.5 m/s
          2) **최근에 달리던 차가 아니다** — `lead.blocked_recent_move_v`(2.0 m/s)를
             넘겨 달린 이력이 `lead.blocked_recent_move_s`(20 s) 안에 있으면 "잠깐
             선 것"으로 보고 기다린다. 앞차 급제동은 정지 자체가 연속이라
             dwell 로는 절대 구분되지 않는다 (2026-08-23 19:56 런: v=0.000 이
             10.9 s 연속 → dwell 2 s 만에 추월 발동, 그 차는 15 s 뒤 재출발 예정이었다)
          3) 전방에 **활성 신호 정지 후보가 없다** (적/황으로 서는 중이면 추월 금지)
          4) 교차로 내부가 아니다
          5) 앞범퍼 기준 전방 정지선이 `lead.blocked_ignore_stopline_m` 보다 멀거나 없다
             — 가까우면 신호 대기 행렬일 수 있다
          6) 1~5 가 `lead.blocked_dwell_s` 이상 **연속** 유지 (센서 노이즈 방지)

        신호 대기 중 추월은 즉시 실격이므로 3·5 가, 급제동 오판 방지는 2 가 핵심이다.
        """
        ld = self.cfg['lead']
        summ = world.summ or {}
        lead = self._lead

        # 관측된 모든 객체의 '마지막으로 달리던 시각' 갱신 (사라진 것은 정리)
        mv = float(ld['blocked_recent_move_v'])
        alive = set()
        for o in world.objects:
            alive.add(o.id)
            if o.speed > mv:
                self._obj_moved_t[o.id] = world.t
        for oid in [k for k in self._obj_moved_t if k not in alive]:
            if world.t - self._obj_moved_t[oid] > 60.0:
                del self._obj_moved_t[oid]

        why = ''
        if lead is None:
            why = 'lead 없음'
        elif lead.speed >= 0.5:
            why = 'lead 주행 중(%.1f m/s)' % lead.speed
        elif (self._obj_moved_t.get(lead.id) is not None
              and world.t - self._obj_moved_t[lead.id] < float(ld['blocked_recent_move_s'])):
            why = '최근 주행 이력 %.1fs 전' % (world.t - self._obj_moved_t[lead.id])
        elif summ.get('in_junction'):
            why = '교차로 내부'
        else:
            # 신호 정지 판정 — **실제로 구속 중일 때만** 추월을 막는다.
            # 예전 조건("stop_line 후보 존재")은 114 m 앞 다음 교차로가 황색으로
            # 바뀌기만 해도(후보 53.9 km/h, winner 는 여전히 lead) 추월 의사를
            # 영구 소멸시켰다 (2026-08-24 13:10 런). 기준은 두 가지로 통합한다:
            #   · 후보가 현재 min() 을 실제로 결정하고 있다 (binding)
            #   · 또는 정지선이 blocked_ignore_stopline_m 이내다 (신호 대기 행렬 가능)
            sl = out.get('stop_line')
            binding = sl is not None and out and sl <= min(out.values()) + 1e-9
            d = summ.get('dist_stop_line')
            d_front = (d - self._front_m()) if d is not None else None
            near = d_front is not None and d_front <= float(ld['blocked_ignore_stopline_m'])
            if binding:
                why = '신호 정지 구속 중(%.1f m/s)' % sl
            elif near:
                why = '정지선 %.0f m (신호 대기 가능)' % d_front

        if why:
            self._blocked_since = None
            self._blocked = False
            self._blocked_why = why
            return
        if self._blocked_since is None:
            self._blocked_since = world.t
        held = world.t - self._blocked_since
        self._blocked = held >= float(ld['blocked_dwell_s'])
        self._blocked_why = 'blocked %.1fs' % held if self._blocked else 'dwell %.1fs' % held

    def _overtake_event(self, world: WorldState) -> dict | None:
        """
        정차 차량 추월용 **합성 차선변경 이벤트**. 경로 이벤트와 같은 형식이라
        기존 판정(_target_lane_clear / 지시등 lead / _blend_path)을 그대로 탄다.

        나갈 때: blocked 이면 좌측 우선(추월 관례), 없으면 우측 이웃으로.
        돌아올 때: 막고 있던 차를 지나쳤으면 원 차로로 되돌아간다.
        창은 지금부터 전이거리+여유까지다 (경로 이벤트처럼 미리 정해진 구간이 없다).
        """
        ego = world.ego
        if ego.lane is None:
            return None
        lc = self.cfg['lane_change']
        v = max(ego.speed, 1.0)
        span = max(float(lc['transition_s']) * v, float(lc['transition_min_m'])) + 10.0
        s_now = ego.route_s

        def mk(target, side):
            return {'kind': 'lane_change_%s' % side, 's': s_now,
                    'lane': tuple(ego.lane), 's_in_lane': ego.s,
                    'window_s0': s_now - 1.0, 'window_s1': s_now + span,
                    'to_lane': tuple(target), 'from_lane': tuple(ego.lane),
                    'overtake': True}

        # 경로 차로 집합. 원 차로 **키**를 기억하면 안 된다 — 키에 섹션 인덱스가
        # 들어 있어 전진만 해도 (2003,2,-2) → (2003,3,-1) 처럼 달라지고, 이웃 비교가
        # 전부 빗나가 복귀에 실패한다. "경로 위 차로인가" 와 "어느 쪽으로 나갔나"
        # 두 가지만 들고 간다.
        route_lanes = set(self.route['lanes']) if self.route else set()
        on_route_lane = tuple(ego.lane) in route_lanes
        OPP = {'left': 'right', 'right': 'left'}

        if self._overtake is not None:
            side = self._overtake['side']
            if self._overtake['returning']:
                if on_route_lane:
                    self._overtake = None        # 복귀 완료
                    return None
                back = self.lg.neighbor(ego.lane, OPP[side])
                if back is not None and tuple(back) in route_lanes:
                    return mk(back, OPP[side])   # 복귀 이벤트 재발행
                self._overtake = None            # 되돌아갈 경로 차로가 없다 — 포기
                return None

            if on_route_lane:
                # 아직 경로 차로다 = 전이 진행 중이거나 시작 전. **여기서 상태를
                # 지우면 안 된다** — 예전엔 '복귀 완료'로 오판해 매 틱 초기화되며
                # 차선변경이 시작·취소를 반복했다.
                nb = self.lg.neighbor(ego.lane, side)
                if nb is not None:
                    return mk(nb, side)
                self._overtake = None
                return None

            # 옆 차로에 나왔다 — 막던 차를 충분히 지나쳤으면 복귀
            blocker = next((o for o in world.objects
                            if o.id == self._overtake['blocker_id']), None)
            passed = blocker is None or blocker.s_rel < -float(self.cfg['vehicle']['length'])
            if not passed:
                return None                      # 아직 옆을 지나는 중 — 그대로 간다
            back = self.lg.neighbor(ego.lane, OPP[side])
            if back is not None and tuple(back) in route_lanes:
                self._overtake['returning'] = True
                return mk(back, OPP[side])
            self._overtake = None
            return None

        if not self._blocked or self._lead is None:
            return None
        for side in ('left', 'right'):
            nb = self.lg.neighbor(ego.lane, side)
            if nb is None:
                continue
            rec = self.lg.lanes.get(nb)
            if rec is None or rec['type'] != 'driving':
                continue
            return mk(nb, side)
        return None

    def _select_lead(self, world: WorldState):
        """
        추종할 선행차 하나. on_route ∧ 내 차로 폭 안 ∧ 앞 0~80 m 중 **최근접**.

        `lat_off` 는 내 경로 중심선 기준이므로(perception) 옆 차로 차는 폭 밖으로
        걸러진다. 범위 상한은 GT 가 주는 80 m 와 같다.
        """
        half_w = 1.75
        if world.ego.lane is not None:
            try:
                half_w = 0.5 * float(self.lg.width_at(world.ego.lane, world.ego.s))
            except Exception:                          # noqa: BLE001
                pass
        margin = float(self.cfg['shield'].get('lane_side_m', 1.0)) * 0.5
        # 추월 전이 중에는 **막고 있던 차를 추종 대상에서 뺀다**. 안 빼면 IDM 이
        # 계속 0 을 명령해 차가 서고, 전이 진행도(prog = route_s − s_start)가 늘지
        # 않아 차선변경이 영영 끝나지 않는다 (폐루프 시뮬: t_off 0.81 에서 멈춤).
        # 목표 차로는 _target_lane_clear 로 이미 비었음을 확인했고, 만일의 접근은
        # shield 의 TTC 비상제동이 최후 방어로 남는다.
        # 추월 기동 동안(나가기~복귀)은 그 대상을 추종 후보에서 뺀다 — shield 의
        # TTC 면제와 같은 범위다.
        skip = self._overtake['blocker_id'] if self._overtake is not None else None
        best = None
        for o in world.objects:
            if o.id == skip:
                continue
            if not o.on_route or not (0.0 < o.s_rel <= LEAD_MAX_S):
                continue
            if abs(o.lat_off) > half_w + margin:
                continue
            if best is None or o.s_rel < best.s_rel:
                best = o
        return best

    def _idm(self, world: WorldState, v0: float) -> float | None:
        """
        선행차 추종 속도 상한 [m/s]. 선행차가 없으면 None.

            s* = s0 + max(0, v·T + v·Δv / (2√(a_max·b)))
            a  = a_max · [ 1 − (v/v0)^δ − (s*/gap)² ]

        s0 = `lead.min_gap_m`, T = `lead.time_headway_s`, a_max = `speed.a_max`,
        b = `speed.a_comf`(쾌적 감속), δ = IDM_DELTA. **전부 기존 키 재사용**이다.
        (PDM-Lite §3.2.1 계수 확보 시 이 값들을 교체할 것.)

        gap 은 범퍼 간격이다(자차 앞부분 + 객체 반길이 차감) — perception 의 ttc 와
        같은 기준을 쓴다. 가속도는 **지평 τ** 를 곱해 목표속도 v + a·τ 로 바꿔 다른
        후보와 같은 단위(m/s)로 min() 중재에 넣는다. τ = 1 / `control.kp` 다:
        제어기가 accel = kp·(v_target − v) 를 내므로 τ = 1/kp 여야 IDM 이 지시한
        가속도가 **그대로** 재현된다. 한 틱 dt(0.05 s)를 쓰면 kp·a·dt = 0.04·a 로
        IDM 감속의 4 % 만 전달되어(실측) gap 7.8 m 에서도 12.0 m/s 를 유지했다.

        선행차가 서 있으면(v_lead < 0.5) 정지선과 같은 감속 프로파일로 s_rel − s0
        지점에 세우는 후보를 함께 내고 둘 중 낮은 값을 쓴다.
        """
        lead = self._lead
        if lead is None:
            return None

        ld = self.cfg['lead']
        sp = self.cfg['speed']
        vh = self.cfg['vehicle']
        # 추월 대상 뒤에서는 조향 여유를 두고 더 멀리 선다.
        s0 = float(ld['blocked_gap_m'] if self._blocked else ld['min_gap_m'])
        T = float(ld['time_headway_s'])
        a_max = float(sp['a_max'])
        b = float(sp['a_comf'])

        v = max(world.ego.speed, 0.0)
        front = float(vh['wheelbase']) + float(vh.get('front_overhang_m', 0.855))
        gap = max(lead.s_rel - front - 0.5 * float(lead.length), 0.1)
        dv = lead.v_rel                                 # + = 접근

        s_star = s0 + max(0.0, v * T + v * dv / (2.0 * math.sqrt(a_max * b)))
        # 자유주행 항은 **1 에서 자른다**. v > v0 일 때 원식은 추가 제동을 거는데,
        # 그 상한은 이미 다른 후보(limit/junction/curvature)가 min() 으로 걸고 있어
        # 이중 제동이 된다 — 목표가 v0 아래로 내려갔다가 되올라오며 진동했다
        # (폐루프 시뮬: ego 6.4~11.0 m/s 요동, 추종 gap 35 m). 자른 뒤에는
        # v >= v0 에서 상호작용(제동) 항만 남는다.
        free = min(1.0, (v / v0) ** IDM_DELTA) if v0 > 1e-6 else 1.0
        a = a_max * (1.0 - free - (s_star / gap) ** 2)
        a = max(float(sp['a_min']), min(a_max, a))

        tau = 1.0 / max(1e-6, float(self.cfg['control']['kp']))
        v_next = max(0.0, min(v0, v + a * tau))

        # 서 있는 선행차: 정지선과 같은 프로파일로 s_rel − s0 앞에 정지
        if lead.speed < 0.5:
            lag = float(sp.get('stop_lag_s', 0.0)) * v
            v_stop = self._approach(0.0, gap - s0 - lag)
            v_next = min(v_next, v_stop)
            # blocked 추월 확정이면 정지 목표를 풀고 크립으로 나간다 — 지시등을
            # 채우는 동안 완전 정지하면 v_min 교착이 된다 (2026-08-24 13:10 런).
            # 단 간격이 min_gap_m 아래로 좁아지면 크립도 멈춘다 (충돌 여유).
            if self._blocked and gap > float(ld['min_gap_m']):
                creep = float(self.cfg['lane_change'].get('creep_kph', 8.0)) / 3.6
                if v_next < creep:
                    v_next = creep
                    self._lead_creep = True
        return v_next

    # ── 횡방향 ────────────────────────────────────────────────────────────
    def _lateral_path(self, world: WorldState) -> list[tuple[float, float]]:
        """
        추종 목표 점열 (월드 좌표). 기본은 경로 차로 중심선 = `lg.points_ahead`.

        AVOID 진입 조건 (**전부** 만족해야 함):
          - 내 차로에 정지 장애물. **아래 둘은 제외** — 피해 가는 게 아니라
            서서 기다리거나 따라가는 대상이다:
              · 횡단보도 위 보행자 (`crosswalk_blockers`) → `_crosswalk_ped_speed`
              · 추종 중인 선행차 (`self.lead_id`) → `_idm` (`_select_lead`)
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

    # ── 지시등 (교차로 회전) ──────────────────────────────────────────────
    def _build_turns(self) -> list:
        """
        route 의 turn_* 이벤트에 **연결로 끝 경로거리(end_s)** 를 붙인다.

        lookahead 는 이벤트 지점을 지나면 그 turn 을 더 내지 않으므로 "회전이
        언제 끝나는가" 는 여기서 미리 계산해 둬야 한다. 끝 = 같은 junction 의
        연결로가 이어지는 마지막 차로의 끝 (= junction_out 지점).
        """
        out = []
        if not self.route:
            return out
        lanes, cum, lens = self.route['lanes'], self.route['cum_s'], self.route['lengths']
        for ev in self.route.get('events', []):
            if ev['kind'] not in ('turn_left', 'turn_right'):
                continue
            try:
                i = lanes.index(tuple(ev['lane']))
            except ValueError:
                continue
            j = i
            junc = ev.get('junction')
            if junc is not None and junc != -1:
                while (j + 1 < len(lanes)
                       and self.lg.lanes[lanes[j + 1]]['junction'] == junc):
                    j += 1
            out.append({'kind': ev['kind'], 's': float(ev['s']),
                        'end_s': float(cum[j]) + float(lens[j]),
                        'signal': TURN_LEFT if ev['kind'] == 'turn_left' else TURN_RIGHT})
        out.sort(key=lambda t: t['s'])
        return out

    def _turn_signal(self, world: WorldState) -> tuple[int, float] | None:
        """
        교차로 회전 지시등 후보 → (방향, 남은거리) 또는 None.

        - lead 단계: lookahead 의 turn 이벤트(summ.dist_next_turn / next_turn)까지
          남은 거리가 v·(signal.lead_s + margin_s) 이하면 점등. (채점 항목:
          회전 n초 전 점등) lookahead 가 없으면(경로 밖) route 이벤트 거리로 대체.
        - 회전 중: 이벤트 s 부터 연결로 끝(end_s)까지는 거리 0 으로 계속 점등.
          연결로를 벗어나면(route_s ≥ end_s) 끈다.
        """
        if not self._turns or world.ego.lane is None:
            return None
        sig = self.cfg['signal']
        s_now = world.ego.route_s
        v = max(world.ego.speed, 0.1)
        lead_m = v * (float(sig['lead_s']) + float(sig['margin_s']))

        # 아직 끝나지 않은 첫 회전
        cur = next((t for t in self._turns if s_now < t['end_s']), None)
        if cur is None:
            return None
        if s_now >= cur['s'] - 0.5:
            return cur['signal'], 0.0             # 연결로 안 — 회전 중

        # lead 단계: lookahead 가 같은 회전을 보고 있으면 그 거리를 쓴다
        summ = world.summ or {}
        d = summ.get('dist_next_turn')
        if d is None or summ.get('next_turn') != cur['kind']:
            d = cur['s'] - s_now
        d = float(d)
        if d <= lead_m:
            return cur['signal'], max(0.0, d)
        return None

    def _next_state(self, world: WorldState) -> str:
        """FSM 전이. 우선순위: E_STOP > YIELD_PED > STOP_LINE > AVOID/RETURN > FOLLOW_LEAD > FOLLOW"""
        # TODO: 구현
        raise NotImplementedError('planner._next_state')
