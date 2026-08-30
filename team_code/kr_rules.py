"""
한국 대회 규칙 계층 — PDM-Lite 판단 결과를 받아 규칙을 덮어쓴다.

PDM-Lite(autopilot.py) 원문은 건드리지 않는다. autopilot._get_control 맨 끝의
한 줄(`self.kr_rules.apply(...)`)이 유일한 접점이고, 여기서는 PDM 의 min()
중재에 **후보를 덧대는** 형태로만 개입한다 — 새 감속 프로파일을 만들지 않고
PDM 의 _compute_target_speed_idm / 종방향 컨트롤러를 그대로 재사용한다.

phase4 현재: route_end 정지 / 정지선 유지 홀드 / 방향지시등. (RTOR·황색 딜레마는 이후 단계.)

route_end — 경로 종점 정지:
  CARLA 리더보드는 결승선 통과로 시나리오가 끝나 PDM 에 종점 정지 개념이
  없다. 실기(2026-08-26 완주속도_01_기본): 종점 도달 후 v_target 6.9 로 계속
  주행 → 경로 밖 이탈 → courseRespawn 9회.

  구현: "종점에 정지해 있는 길이 0 유령 선행차" 를 IDM 에 넣는다. 유효거리를
  d_end − 앞범퍼, s0 를 speed.stop_gap_route_end_m 으로 주면 앞범퍼가 기준점 −
  stop_gap 에 선다. batch 완주 임계(total − end_margin, end_margin = stop_gap +
  앞범퍼 + end_slack)보다 end_slack_m 만큼 안쪽이라 완주 판정과 자동 정합한다
  (tests/test_route_end). ※ 정지선(적신호) 정지는 이 관례와 무관 — PDM
  red-light IDM 소관이고 run_agent.build_pdm_config 가 stop_gap_stopline_m 로
  주입한다. 여기서는 그 정지의 **0.5 s 유지**만 홀드로 보강한다 (아래).

stopline hold — 정지선 정지 유지:
  대회 7번: 정지선 앞 정지는 0.5 s 이상 유지해야 정상. 실측(2026-08-27,
  보행자집중_06) 0.4 s 만에 재출발한 사례가 감점 대상이라, 적신호 정지선
  근처(stopline_hold_near_m)에서 저속(latch_v — 기존 래치 관례 재사용)이 되면
  stopline_hold_s 동안 목표 0 을 유지한다. 홀드 중에는 신호가 녹색으로 바뀌어도
  잔여 시간을 채운다.

  래치: 종점 근처(latch_m)에서 저속(latch_v)이 되면 래치 — 재출발하지 않는다.
  d_end 가 unlatch_m 이상으로 다시 커지면(courseRespawn 으로 뒤로 간 경우)
  해제해 고착을 막는다.

  정지 목표 기준점(stop_s): 대회 규칙은 "뒷축이 종료 지점 통과 = 시험 종료"라
  route_end.target_mode='finish' 면 scoring.finish_xy 를 경로에 투영한 종료선
  (finish_s)을 뒷축이 finish_clearance_m 만큼 넘어 정지하도록 기준점을 잡는다
  (plan_stop_s — 채점 score.py 와 공용, 단일 출처). d_eff/s0 관례·래치·active_m
  판정 거리는 전부 stop_s 기준으로 그대로 동작한다.

turn signal — 방향지시등:
  채점 동적항목("방향지시등 n초 전"). PDM 은 CARLA 리더보드용이라 지시등 개념이
  없어 9910 turnSignal 이 계속 0 이었다.

  경로가 정적이므로 **점등 구간을 시작 시 1회 계산**한다 — route['events'] 의
  turn_left/right(연결로 시작 s, 끝은 같은 junction 차로가 이어지는 데까지)와
  lane_change_left/right(window_s0 ~ 블렌드 끝). 매 틱은 route_s 로 고르기만
  하므로 재선택 깜빡임(실사고 §6-9)이 구조적으로 생기지 않는다.

  점등 조건: 남은거리 ≤ max(v · lead_s, lead_min_m). 시간 기준만 쓰면 적신호
  대기(v→0)에서 선행거리가 0 이 돼 회전 지시등이 안 켜지므로 거리 하한을 둔다.
  겹치면 SPEC §3.3 대로 **남은거리가 짧은 쪽 우선, 동률이면 회전 우선**.

  결과는 last_turn_signal/last_sig_src/last_sig_lead_s 로 노출하고 run_agent 가
  Command.turn_signal 과 로그에 싣는다 (기존 last_candidate/last_target 관례).
"""
from __future__ import annotations

import math as _math

import numpy as np
from scipy.spatial import cKDTree as _cKDTree

from vtd_adapter import frame


def plan_stop_s(cfg: dict, total: float, finish_s: float | None) -> tuple[float, bool]:
    """정지 목표 기준점 stop_s [route_s] 와 클립 여부. 제어·채점 공용 (단일 출처).

    finish_s 있으면 stop_s = min(finish_s + finish_clearance_m + stop_gap + 앞범퍼,
    total − end_slack) — 유령차 기준점에서 앞범퍼가 stop_s − stop_gap, 뒷축이
    stop_s − stop_gap − 앞범퍼 = finish_s + clearance 에 서므로 뒷축이 종료선을
    여유를 두고 넘는다. stop_gap 을 빼먹으면 뒷축이 finish_s − 2.0 에 서서 여전히
    미달한다 (2026-08-27 검토에서 잡은 결함). 클립되면(경로 꼬리 부족) True 와
    함께 total − end_slack 을 돌려준다. finish_s 없으면 기존과 동일하게 total.
    """
    if finish_s is None:
        return float(total), False
    sp, vh = cfg['speed'], cfg['vehicle']
    want = (float(finish_s) + float(cfg['scoring']['finish_clearance_m'])
            + float(sp['stop_gap_route_end_m'])
            + float(vh['wheelbase']) + float(vh['front_overhang_m']))
    cap = float(total) - float(cfg.get('batch', {}).get('end_slack_m', 1.0))
    return (min(want, cap), want > cap)


SIG_OFF, SIG_LEFT, SIG_RIGHT = 0, 1, 2        # 9910 turnSignal (SPEC §1.2)


def _turn_end_s(lg, lanes, cum, lens, ev) -> float:
    """회전 이벤트의 소등 지점 [route_s] — 같은 junction 차로가 이어지는 끝까지.

    build_route 의 turn 이벤트는 시작 s 만 준다(연결로가 여러 개 이어질 수 있어
    끝은 경로에서 되짚어야 한다). lg 나 lanes 가 없으면(목 플래너) 시작점 반환.
    """
    s0 = float(ev['s'])
    if lg is None or not lanes or not cum:
        return s0
    i = min(range(len(cum)), key=lambda j: abs(float(cum[j]) - s0))
    rec = lg.lanes.get(tuple(lanes[i]))
    if rec is None:
        return s0
    end_of = lambda j: float(cum[j]) + (float(lens[j]) if j < len(lens) else 0.0)
    jid = rec.get('junction', -1)
    if jid == -1:
        return end_of(i)
    j = i
    while j + 1 < len(lanes):
        nxt = lg.lanes.get(tuple(lanes[j + 1]))
        if nxt is None or nxt.get('junction') != jid:
            break
        j += 1
    return end_of(j)


def turn_intervals(planner) -> list[dict]:
    """route['events'] 의 **회전만** → 점등 구간 [{sig, src, ev_s, end_s}]. 시작 시 1회.

    회전은 연결로 중심선을 따라가므로 "차로 중심에서 벗어남" 이 0 이라 기하로는
    잡히지 않는다 — 이벤트 목록이 유일한 근거다. 반대로 **차로를 옮기는 움직임**
    (계획된 차선변경 · 런타임 회피 시프트)은 목록이 아니라 경로 기하로 본다
    (_lane_shift). 목록에 없는 런타임 시프트도 자동으로 잡히게 하기 위해서다.
    """
    route = getattr(planner, 'route', None) or {}
    lg = getattr(planner, 'lg', None)
    lanes = route.get('lanes') or []
    cum = route.get('cum_s') or []
    lens = route.get('lengths') or []
    out: list[dict] = []
    for ev in route.get('events') or []:
        kind = str(ev.get('kind', ''))
        if kind.startswith('turn_'):
            out.append({'sig': SIG_LEFT if kind.endswith('left') else SIG_RIGHT,
                        'src': 'turn', 'ev_s': float(ev['s']),
                        'end_s': _turn_end_s(lg, lanes, cum, lens, ev)})
    out.sort(key=lambda d: d['ev_s'])
    return out


def _project_route_s(lg, route: dict, x: float, y: float) -> float | None:
    """좌표 → 경로 누적거리 (score.project_route_s 와 같은 정의 — 경로 차로 투영)."""
    best = None
    for i, k in enumerate(route.get('lanes') or []):
        try:
            s_p, _t, d_p, _ = lg.project(tuple(k), x, y)
        except KeyError:
            continue
        if best is None or d_p < best[0]:
            best = (d_p, float(route['cum_s'][i]) + float(s_p))
    return best[1] if best else None


class KrRules:
    def __init__(self, cfg: dict) -> None:
        re_cfg = cfg['route_end']
        sp, vh = cfg['speed'], cfg['vehicle']
        self.cfg = cfg
        self.stop_gap = float(sp['stop_gap_route_end_m'])
        self.front = float(vh['wheelbase']) + float(vh['front_overhang_m'])
        self.T = float(re_cfg['idm_time_headway'])
        self.active_m = float(re_cfg['active_m'])
        self.latch_v = float(re_cfg['latch_v'])
        self.latch_m = float(re_cfg['latch_m'])
        self.unlatch_m = float(re_cfg['unlatch_m'])
        self.target_mode = str(re_cfg['target_mode'])
        self.finish_xy = (cfg['scoring'] or {}).get('finish_xy')
        # 정지선 0.5 s 유지 홀드 (규정 + 여유는 params 가 단일 출처).
        # 틱 카운트로 잰다 — wall clock 은 리플레이/시뮬에서 흐름이 다르다.
        self.sl_hold_ticks = int(round(float(sp['stopline_hold_s'])
                                       * float(cfg['comm']['send_hz'])))
        self.sl_near_m = float(sp['stopline_hold_near_m'])
        # 정지선 정지 프로파일 (④′). 0 이면 완전 비활성 — 되돌리는 스위치다.
        self.stop_profile_a = float(sp.get('stop_profile_a', 0.0))
        # 황색 딜레마 원샷 판정 (C). 0 이면 비활성 = 황색을 PDM 원문에만 맡긴다.
        self.a_yellow = float(sp.get('a_yellow', 0.0))
        self.y_guard_max_m = float(sp.get('yellow_guard_max_m', 60.0))
        # ap.config 를 못 읽는 환경(목)에서만 쓰는 폴백 — 정상 경로는 PDM 값 사용
        self.stop_gap_sl_fallback = float(sp.get('stop_gap_stopline_m', 1.5)) + self.front
        # 방향지시등 (SPEC §3.3). lc_lead_s 는 규정 미확정 가정값 (§7-2).
        sig = cfg['signal']
        self.turn_lead_s = float(sig['turn_lead_s'])
        self.lc_lead_s = float(sig['lc_lead_s'])
        self.sig_lead_min_m = float(sig['lead_min_m'])
        self.lat_on_m = float(sig['lat_shift_on_m'])
        self.sig_min_on_ticks = int(round(float(sig['min_on_s'])
                                          * float(cfg['comm']['send_hz'])))
        self.sig_off_delay_ticks = int(round(float(sig['off_delay_s'])
                                             * float(cfg['comm']['send_hz'])))
        # 정적 장애물 회피 시프트 (SPEC §3.4 회피 — PDM 원문은 stub)
        ot = cfg['overtake']
        self.hz = float(cfg['comm']['send_hz'])
        # 규칙 1 — 신호 구역 억제
        self.sup_m = float(ot.get('stopline_suppress_m', 30.0))
        self.queue_gap_min_m = float(ot.get('queue_gap_min_m', 3.0))
        self.queue_lat_max_m = float(ot.get('queue_lat_max_m', 1.5))
        # 규칙 3 — 선제 회피
        self.detect_max_m = float(ot.get('detect_max_m', 80.0))
        self.shift_latest_m = float(ot.get('shift_latest_m', 10.0))
        self.shift_k_s = float(ot.get('shift_k_s', 3.0))
        self.shift_ahead_m = float(ot.get('shift_ahead_m', 5.0))
        self.obj_static_ticks = int(round(float(ot.get('obj_static_s', 3.0)) * self.hz))
        self.obj_grace = int(ot.get('obj_grace_ticks', 10))
        self.ot_enabled = bool(ot['enabled'])
        self.ot_v_max = float(ot['blocker_speed_max'])
        self.ot_d_max = float(ot['blocker_dist_max'])
        self.ot_ticks = int(round(float(ot['trigger_s']) * float(cfg['comm']['send_hz'])))
        self.ot_min_corridor = float(ot['min_corridor_m'])
        self.ot_clear_r = float(ot['clear_radius_m'])
        self.ot_trans_m = float(ot['transition_m'])
        self.ot_before_m = float(ot['extra_before_m'])
        self.ot_after_m = float(ot['extra_after_m'])

        self.latched = False
        self.stop_s: float | None = None           # 시작 시 1회 계산 캐시 (매 틱 투영 금지)
        self.sl_hold_left = 0                      # 정지선 홀드 잔여 틱
        self.last_candidate: float | None = None   # 이번 틱 route_end 후보 (로그용)
        self.last_target: float | None = None      # 이번 틱 최종 목표속도 (로그용)
        self.last_d_end: float | None = None
        self.last_stop_profile: float | None = None   # 이번 틱 정지 프로파일 상한 (로그용)
        # 황색 원샷 판정 래치 (접근당 1회, 번복 금지)
        self.y_decision: str | None = None            # None | 'stop' | 'go'
        self.y_ctrl: int | None = None                # 래치가 걸린 신호 id
        self.y_v_allow: float | None = None           # 판정 시 v_allow (로그용)
        self.last_yellow: dict | None = None          # 판정 순간 1틱만 채운다 (로그용)
        # 교차로 통과 가드
        self.cross_guard = False
        self.cross_s: float | None = None
        self.cross_junction_seen = False
        self._ap = None                               # 홀드가 _stop_target 을 부르려면 필요
        self.sig_plan: list[dict] | None = None    # 회전 구간 (시작 시 1회)
        self.sig_on_ticks = 0                      # 켜진 뒤 지난 틱 (min_on 기준)
        self.sig_off_left = 0                      # 소등 지연 잔여 틱 (상한 있음)
        self.sig_held: int = SIG_OFF               # 유지 중인 값
        self.ot_blocked_ticks = 0                  # 막힌 채 정지한 틱
        self.ot_span: tuple | None = None          # 시프트한 인덱스 구간
        self.last_overtake: str | None = None      # 로그용 ('left'|'right'|사유)
        self.last_avoid: dict | None = None        # 회피 진단 (reasons.avoid)
        self.obj_ticks: dict = {}                  # 객체별 정지 지속 틱
        self.obj_miss: dict = {}                   # 객체별 미관측 틱 (grace)
        self._sl_all: list | None = None           # 경로상 전 정지선 route_s (1회)
        self.last_turn_signal: int = SIG_OFF       # 이번 틱 지시등 (run_agent 가 읽는다)
        self.last_sig_src: str | None = None       # 'turn' | 'lc'
        self.last_sig_lead_s: float | None = None  # 이벤트까지 남은 시간 [s]

    def _resolve_stop_s(self, planner) -> float:
        """정지 목표 기준점 1회 산출. finish 모드 실패 시 경고 후 total 폴백."""
        total = float(planner.route['total_length'])
        if self.target_mode != 'finish':
            return total
        if not self.finish_xy:
            print('[kr_rules] scoring.finish_xy 미설정 — route_total 기준으로 정지 (기존 동작)',
                  flush=True)
            return total
        lg = getattr(planner, 'lg', None)
        finish_s = (_project_route_s(lg, planner.route,
                                     float(self.finish_xy[0]), float(self.finish_xy[1]))
                    if lg is not None else None)
        if finish_s is None:
            print('[kr_rules] finish_xy 를 경로에 투영하지 못함 — route_total 기준으로 정지',
                  flush=True)
            return total
        stop_s, clipped = plan_stop_s(self.cfg, total, finish_s)
        if clipped:
            print(f'[kr_rules] ⚠ 계획 정지점이 종료선을 못 넘는다 — finish_s {finish_s:.1f} '
                  f'+ 여유가 경로 종점을 초과 (경로 꼬리 부족). 종점까지 주행한다', flush=True)
        return stop_s

    def _lane_shift(self, planner, ego_speed: float):
        """앞 창에서 경로가 차로 중심 기준으로 옆으로 갈 예정인가 → (sig, 남은거리).

        planner.lat_shift 는 경로점마다 "기준 차로 중심에서 밀린 양"(+좌/−우)이다.
        계획된 차선변경 블렌드와 **런타임 회피 시프트**가 둘 다 여기에 반영되므로,
        경로를 옆으로 미는 어떤 동작이든 지시등이 따라온다. 테이퍼(소멸 차로 기하
        보정)는 차로를 옮기는 게 아니라 제외돼 있다.

        한 점만 비교하지 않고 **창 전체를 훑는다** — 창 끝점이 이미 이동을 마친
        뒤라면 차이가 0 으로 나와 놓친다.
        """
        lat = getattr(planner, 'lat_shift', None)
        if lat is None or len(lat) == 0:
            return None
        i = int(getattr(planner, 'route_index', 0))
        if i >= len(lat):
            return None
        ppm = float(getattr(planner, 'points_per_meter', 10))
        look = max(ego_speed * self.lc_lead_s, self.sig_lead_min_m)
        j = min(len(lat), i + int(look * ppm) + 1)
        seg = lat[i:j] - lat[i]
        if seg.size == 0:
            return None
        k = int(np.argmax(np.abs(seg)))
        if abs(float(seg[k])) < self.lat_on_m:
            return None
        # 남은거리 = 임계를 처음 넘는 지점까지 (우선순위 비교용)
        over = np.nonzero(np.abs(seg) >= self.lat_on_m)[0]
        remain = float(over[0]) / ppm if over.size else 0.0
        return (SIG_LEFT if seg[k] > 0 else SIG_RIGHT), remain

    def _turn_signal(self, planner, route_s: float, ego_speed: float) -> tuple:
        """이번 틱 지시등 → (sig, src, lead_s).

        후보는 둘 — 회전(이벤트 구간)과 차로 이동(경로 기하). 겹치면 SPEC §3.3
        대로 남은거리가 짧은 쪽, 동률이면 회전 우선.

        깜빡임 방지는 **최소 점등 시간뿐**이다. 끄는 임계도 래치도 두지 않는다 —
        조건이 거짓이 되면 그대로 꺼진다. 유지 구간에 상한이 있으니 고착되지 않는다.
        """
        if self.sig_plan is None:
            self.sig_plan = turn_intervals(planner)

        best = None                                  # (정렬키, sig, src, remain)
        for iv in self.sig_plan:
            if route_s > iv['end_s']:
                continue
            remain = iv['ev_s'] - route_s
            if remain > max(ego_speed * self.turn_lead_s, self.sig_lead_min_m):
                continue
            key = (max(0.0, remain), 0)              # 0 = 회전 우선
            if best is None or key < best[0]:
                best = (key, iv['sig'], 'turn', remain)

        shift = self._lane_shift(planner, ego_speed)
        if shift is not None:
            sig, remain = shift
            key = (max(0.0, remain), 1)
            if best is None or key < best[0]:
                best = (key, sig, 'lc', remain)

        if best is None:
            sig, src, remain = SIG_OFF, None, None
        else:
            _k, sig, src, remain = best

        # 유지 장치 둘. 기준 시점이 다르다 — 둘 다 상한이 있어 고착되지 않는다.
        #   min_on   : **켜진 시점부터** 최소 점등 시간. 켜자마자 꺼지는 깜빡임 방지
        #              (조건이 계속 참인 동안 갱신하지 않는다 — 갱신하면 이게 곧
        #               소등 지연이 돼 off_delay 가 무의미해진다)
        #   off_delay: **조건이 끝난 시점부터** 소등까지. 다 돌기 전에 꺼지는 것 방지
        #              (실차 자동소등 관례)
        if sig != SIG_OFF:
            if self.sig_held != sig:
                self.sig_on_ticks = 0                  # 새로 켜짐 / 방향 전환
            self.sig_held = sig
            self.sig_on_ticks += 1
            self.sig_off_left = self.sig_off_delay_ticks
        elif self.sig_held != SIG_OFF and (self.sig_off_left > 0
                                           or self.sig_on_ticks < self.sig_min_on_ticks):
            self.sig_off_left = max(0, self.sig_off_left - 1)
            self.sig_on_ticks += 1
            sig, src = self.sig_held, 'hold'
        else:
            self.sig_held = SIG_OFF
            self.sig_on_ticks = 0

        lead = (max(0.0, remain) / ego_speed
                if remain is not None and ego_speed > 0.1 else None)
        return sig, src, lead

    # ── 정적 장애물 회피 시프트 ──────────────────────────────────────────
    # ── 규칙 1: 신호 구역 억제 ───────────────────────────────────────────
    def _all_stopline_s(self, planner) -> list:
        """경로상 **모든** 정지선의 route_s (신호 유무 무관). 시작 시 1회.

        planner.traffic_lights 는 신호 매핑된 정지선만 담는다 (route.py
        collect_stops 게이트) — 지도 576개 중 245개가 미매핑이라 그것만 보면
        무신호 정지선 앞 대기열을 놓친다. 여기서는 lanegraph 원본을 직접 읽는다.
        """
        if self._sl_all is not None:
            return self._sl_all
        out = []
        lg = getattr(planner, 'lg', None)
        route = getattr(planner, 'route', None) or {}
        if lg is not None:
            for i, k in enumerate(route.get('lanes') or []):
                rec = lg.lanes.get(tuple(k))
                if not rec:
                    continue
                for sl in rec.get('stop_lines', []):
                    out.append(float(route['cum_s'][i]) + float(sl['s']))
        self._sl_all = sorted(out)
        return self._sl_all

    def _signal_zone(self, planner, ap) -> tuple | None:
        """신호 구역이면 (사유, 거리), 아니면 None — 회피 계열 전면 억제 게이트.

        **대기열은 신호 앞에만 선다**는 것이 근거다. 신호 대기 차량을 정적
        장애물로 오인해 비켜가면 신호 위반이 된다.

        1차 키는 planner.distances_to_next_traffic_lights + 교차로 판정이다.
        보조로 무신호 정지선까지 본다. 1차 키를 dist_stop_line(world.summ) 대신
        쓰는 이유 — 실측 2026-08-30: dist_stop_line 은 27% 가 null 이고 실제
        장애물 정지 지점에서도 null 이었다. 반면 이 배열은 inf 비율 1~2% 이고
        같은 지점에서 214.8 m 를 정상으로 준다.
        """
        if bool(getattr(ap, 'junction', False)):
            return ('junction', None)
        try:
            d = float(planner.distances_to_next_traffic_lights[planner.route_index])
        except Exception:                                  # noqa: BLE001
            d = float('inf')
        if d < self.sup_m:
            return ('signal', round(d, 1))
        route_s = float(planner.route_s[planner.route_index])
        for s_sl in self._all_stopline_s(planner):         # 보조: 무신호 정지선
            if 0.0 <= s_sl - route_s < self.sup_m:
                return ('stopline', round(s_sl - route_s, 1))
        return None

    # ── 물체별 정지 관찰 타이머 ──────────────────────────────────────────
    def _update_obj_timers(self, ap) -> None:
        """객체별 '정지 상태 지속 틱'. 매 틱 1회.

        자차 상태 카운터(ot_blocked_ticks)와 달리 **물체마다** 센다 — 자차가
        달리는 동안에도 관찰이 쌓여야 규칙 3(선제 회피)이 성립한다.
        id 가 잠깐 빠져도 obj_grace_ticks 동안은 타이머를 유지한다 (GT 는
        '가까운 순 30개' 제한이 있어 혼잡 시 밀릴 수 있다). 실측 2026-08-30
        두 로그에서는 드롭아웃이 0회였지만 방어로 둔다.
        """
        try:
            actors = list(ap._world.get_actors())
        except Exception:                                  # noqa: BLE001
            return
        ego_id = ap._vehicle.id
        seen = set()
        for a in actors:
            if a.id == ego_id:
                continue
            seen.add(a.id)
            self.obj_miss[a.id] = 0
            if float(getattr(a, 'speed', 0.0)) < self.ot_v_max:
                self.obj_ticks[a.id] = self.obj_ticks.get(a.id, 0) + 1
            else:
                self.obj_ticks[a.id] = 0                   # 움직이면 즉시 리셋 (철회)
        for oid in list(self.obj_ticks):
            if oid in seen:
                continue
            self.obj_miss[oid] = self.obj_miss.get(oid, 0) + 1
            if self.obj_miss[oid] > self.obj_grace:
                self.obj_ticks.pop(oid, None)
                self.obj_miss.pop(oid, None)

    def _static_ok(self, actor) -> bool:
        return self.obj_ticks.get(getattr(actor, 'id', None), 0) >= self.obj_static_ticks

    def _blocker(self, ap, planner):
        """앞을 막고 선 정적 장애물 → VtdActor. 없으면 None.

        PDM 은 타입 필드가 없는 9910 객체를 전부 vehicle 로 감싸므로(actor.py),
        정차 차량·공사 표지·파손 차량이 모두 여기 걸린다 — 대응이 같으니 무방하다.
        """
        try:
            vehicles = list(ap._world.get_actors().filter('*vehicle*'))
        except Exception:                                  # noqa: BLE001
            return None
        ego = ap._vehicle
        ids = set(planner.compute_leading_vehicles(vehicles, ego.id))
        if not ids:
            return None
        best, best_d = None, None
        ex, ey = ego.get_location().x, ego.get_location().y
        for a in vehicles:
            if a.id not in ids or float(getattr(a, 'speed', 0.0)) > self.ot_v_max:
                continue
            loc = a.get_location()
            d = _math.hypot(loc.x - ex, loc.y - ey)
            if d > self.ot_d_max:
                continue
            if best_d is None or d < best_d:
                best, best_d = a, d
        return best

    def _project(self, planner, x_carla, y_carla):
        """CARLA 좌표 → (route_s, 횡오프셋). 전방 창에서만 찾는다."""
        pts = planner.route_points
        i0 = planner.route_index
        ppm = int(getattr(planner, 'points_per_meter', 10))
        hi = min(len(pts), i0 + int(self.detect_max_m * ppm))
        seg = pts[i0:hi, :2]
        if len(seg) < 2:
            return None
        q = np.array([x_carla, y_carla])
        j = int(np.argmin(np.linalg.norm(seg - q, axis=1)))
        tan = seg[min(j + 1, len(seg) - 1)] - seg[j]
        n = float(np.linalg.norm(tan))
        if n < 1e-9:
            return None
        tan = tan / n
        dv = q - seg[j]
        lat = float(tan[0] * dv[1] - tan[1] * dv[0])
        return (float(planner.route_s[i0 + j]) - float(planner.route_s[i0]), lat)

    def _corridor_blockers(self, ap, planner):
        """전방 detect_max_m 안에서 **주행 회랑을 침범한 정지 객체** 목록.

        반환: [(s_rel, lat, half_w, actor)] — s_rel 은 자차 기준 전방거리.
        침범 판정은 선행차 판정([route.py] compute_leading_vehicles) 과 같은 축:
        |lat| < 자차반폭 + 객체반폭 + obstacle_clearance_m.
        """
        try:
            actors = list(ap._world.get_actors())
        except Exception:                                  # noqa: BLE001
            return []
        ego_id = ap._vehicle.id
        half_ego = float(self.cfg['vehicle']['width']) / 2.0
        clr = float(self.cfg['percep'].get('obstacle_clearance_m', 0.3))
        out = []
        for a in actors:
            if a.id == ego_id or not self._static_ok(a):
                continue
            loc = a.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is None:
                continue
            s_rel, lat = pr
            if not (0.5 < s_rel <= self.detect_max_m):
                continue
            hw = float(getattr(getattr(a, 'bounding_box', None), 'extent', None).y) \
                if getattr(a, 'bounding_box', None) is not None else 0.9
            if abs(lat) < half_ego + hw + clr:
                out.append((s_rel, lat, hw, a))
        out.sort(key=lambda z: z[0])
        return out

    def _is_queue(self, blockers) -> bool:
        """정지 객체가 **종방향으로** 2대 이상 줄지어 있으면 대기열로 본다.

        신호 구역 판정(_signal_zone)의 사각 보완이다 — 정지선 데이터가 없는
        도로에서도 "줄 서 있으면 신호 대기"로 걸러낸다. 케이스 B(스태거드)와
        구분되는 점: 대기열은 **횡 위치가 비슷하고 종방향으로 벌어져** 있다.
        스태거드는 종방향으로 붙어 있고 횡으로 갈린다.
        """
        if len(blockers) < 2:
            return False
        n = 0
        for (s1, l1, _h1, _a1), (s2, l2, _h2, _a2) in zip(blockers, blockers[1:]):
            if (s2 - s1) > self.queue_gap_min_m and abs(l2 - l1) < self.queue_lat_max_m:
                n += 1
        return n >= 1

    def _side_is_clear(self, lg, planner, ap, target) -> bool:
        """목표 차로에 차가 없는가 (lc_clear 대용 — 아직 후방 추종차는 안 본다)."""
        ego = ap._vehicle
        ex, ey = ego.get_location().x, ego.get_location().y
        try:
            actors = list(ap._world.get_actors())
        except Exception:                                  # noqa: BLE001
            return False
        near = {target}
        near |= set(lg.successors(target)) | set(lg.predecessors(target))
        for a in actors:
            if a.id == ego.id:
                continue
            loc = a.get_location()
            if _math.hypot(loc.x - ex, loc.y - ey) > self.ot_clear_r:
                continue
            vx, vy = frame.from_carla_xy(loc.x, loc.y)
            try:
                m = lg.locate(vx, vy)
            except Exception:                              # noqa: BLE001
                continue
            if m is not None and m.lane in near:
                return False
        return True

    def _try_overtake(self, ap, planner, ego_speed: float) -> None:
        """막힌 채 서 있으면 경로를 옆 차로로 밀어 비켜간다 (1회, 게이트 통과 시).

        게이트: 목표 차로 존재 · 교차로 아님 · 점선 회랑 충분(S2.2.05) · 측방 비어 있음.
        시프트는 나갔다 돌아오는 프로파일이라(양 끝 전이계수 0) 복귀는 자동이고,
        지나가면 경로를 원상 복구해 다음 장애물에 다시 쓸 수 있게 한다.
        """
        # 지나갔으면 원복 (다음 장애물용)
        if self.ot_span is not None and planner.route_index > self.ot_span[1]:
            a, b = self.ot_span
            planner.route_points[a:b] = planner.original_route_points[a:b]
            planner.commands[a:b] = planner.commands_orig[a:b]
            planner.lat_shift[a:b] = planner._lat_build[a:b]
            planner._kd = _cKDTree(planner.route_points[:, :2])
            self.ot_span = None
            self.last_overtake = 'restored'
            return
        if not self.ot_enabled or self.ot_span is not None:
            return

        # ── 규칙 1: 신호 구역 억제 (전 상태 공통 게이트) ──────────────────
        zone = self._signal_zone(planner, ap)
        if zone is not None:
            self.ot_blocked_ticks = 0
            self.last_avoid = {'state': 'SUPPRESS', 'suppress': zone[0],
                               'sup_d': zone[1]}
            return

        corridor = self._corridor_blockers(ap, planner)
        if self._is_queue(corridor):
            self.ot_blocked_ticks = 0
            self.last_avoid = {'state': 'SUPPRESS', 'suppress': 'queue',
                               'n': len(corridor)}
            return

        # ── 규칙 3: 선제 회피 — 정지 전에 비켜간다 ────────────────────────
        actor = None
        preempt = False
        if corridor:
            s_rel, lat, _hw, cand = corridor[0]
            # 전이 길이는 속도 비례: 코사인 전이의 최대 횡가속
            #   a_lat = v²·Δ·π²/(2L²)  →  L = v·√(Δπ²/(2·a_lat))
            # Δ=3.0 m, a_lat=1.5 m/s² 이면 L ≈ 3.14·v → shift_k_s 3.0 의 근거.
            need = max(self.shift_latest_m, self.shift_k_s * max(ego_speed, 0.1))
            if s_rel <= need + self.shift_latest_m:
                actor, preempt = cand, True
                self.last_avoid = {'state': 'PREEMPT', 'blocker': cand.id,
                                   's_rel': round(s_rel, 1), 'lat': round(lat, 2),
                                   'obj_s': round(self.obj_ticks.get(cand.id, 0) / self.hz, 1),
                                   'need_m': round(need, 1)}

        # ── REACTIVE: 막힌 채 정지가 지속되면 (기존 경로) ─────────────────
        if actor is None:
            blocked = ego_speed < self.latch_v and self._blocker(ap, planner) is not None
            self.ot_blocked_ticks = self.ot_blocked_ticks + 1 if blocked else 0
            if self.ot_blocked_ticks < self.ot_ticks:
                if corridor and self.last_avoid is None:
                    self.last_avoid = {'state': 'WATCH', 'blocker': corridor[0][3].id,
                                       's_rel': round(corridor[0][0], 1)}
                return
            actor = self._blocker(ap, planner)
            if actor is None:
                return
            self.last_avoid = {'state': 'REACTIVE', 'blocker': actor.id}
        lg = getattr(planner, 'lg', None)
        ego_lane = getattr(ap, '_kr_ego_lane', None) or self._ego_lane(lg, ap)
        if lg is None or ego_lane is None:
            self.last_overtake = 'no_lane'
            return
        if lg.lanes[ego_lane]['junction'] != -1:
            self.last_overtake = 'junction'
            return

        for side in ('left', 'right'):                     # 좌측 추월 우선
            target = lg.neighbor(ego_lane, side)
            if target is None:
                continue
            if lg.dashed_corridor_m(ego_lane, side) < self.ot_min_corridor:
                self.last_overtake = f'{side}:solid'
                continue
            if not self._side_is_clear(lg, planner, ap, target):
                self.last_overtake = f'{side}:occupied'
                continue
            ppm = float(getattr(planner, 'points_per_meter', 10))
            # 전이 길이 속도 비례 (위 유도) — 정지 상태면 하한이 지배한다
            trans_m = max(self.ot_trans_m, self.shift_k_s * max(ego_speed, 0.1))
            span = planner.shift_route_around_actors(
                actor,
                obstacle_direction='right' if side == 'left' else 'left',
                transition_length=trans_m * ppm,
                extra_length_before=self.ot_before_m * ppm,
                extra_length_after=self.ot_after_m * ppm,
                # 전이 시작을 자차 **앞**으로 — 뒤에서 시작하면 현재 위치의
                # 경로가 옆으로 밀려 정지 상태에서 조향이 풀락된다
                min_start_ahead=self.shift_ahead_m * ppm)
            planner._kd = _cKDTree(planner.route_points[:, :2])
            self.ot_span = span
            self.ot_blocked_ticks = 0
            self.last_overtake = side
            (self.last_avoid or {}).update(
                {'shift': side, 'span': list(span), 'trans_m': round(trans_m, 1),
                 'preempt': preempt})
            print(f'[kr_rules] 정적 장애물 회피 — {side} 로 경로 시프트 '
                  f'(id={actor.id}, 구간 {span[0]}~{span[1]})', flush=True)
            return
        if self.last_overtake is None:
            self.last_overtake = 'no_neighbor'

    @staticmethod
    def _ego_lane(lg, ap):
        if lg is None:
            return None
        loc = ap._vehicle.get_location()
        vx, vy = frame.from_carla_xy(loc.x, loc.y)
        try:
            m = lg.locate(vx, vy)
        except Exception:                                  # noqa: BLE001
            return None
        return m.lane if m is not None else None

    def _next_stopline(self, planner):
        """전방 신호 정지선 → (뒷축거리 [m], 상태명, 신호 id). 없으면 None.

        색을 가리지 않고 그대로 준다 — 색 해석은 호출자 한 곳
        (_stop_target)에서만 한다.
        """
        dists = getattr(planner, 'distances_to_next_traffic_lights', None)
        tls = getattr(planner, 'next_traffic_lights', None)
        if dists is None or tls is None:
            return None
        tl = tls[planner.route_index]
        if tl is None:
            return None
        return (float(dists[planner.route_index]),
                getattr(getattr(tl, 'state', None), 'name', None),
                getattr(tl, 'id', None))

    def _yellow_latch(self, planner, ego_speed: float, ap) -> None:
        """황색 원샷 판정 — 접근당 1회 STOP/GO 를 정하고 래치한다.

        대회 채점표 편향이 **STOP 우선**을 강제한다: 황색 정지를 감점하는 항목이
        없고(항목8 5 s 카운트는 녹색 틱만 센다), 반대로 적색 통과·걸침은 항목7
        중대다. 게다가 score.detect_red_light 는 **통과 순간의 신호**로 판정하므로
        GO 로 나갔다가 적색에 걸리면 그대로 중대가 된다 (실측 2026-08-30
        실전주행_02: 적신호 통과 2 + 정지선 침범 2 = 항목7 4건).

        그래서 판정 감속은 **확실히 실행 가능한 최대**(speed.a_yellow, 기본
        a_dec_max 와 같은 4.0)를 쓴다 — STOP 영역을 최대화하고 GO 는 물리적으로
        설 수 없는 영역에만 남긴다.

            v ≤ √(2·a_yellow·(d − s0))  → STOP (적색과 동일 취급)
            그 외                        → GO   (신호·정지선 유래 후보 미생성)

        래치는 접근당 유지한다. GO 중 적색으로 바뀌어도 번복하지 않는다 —
        번복하면 교차로 한복판 급제동이 된다. 해제는 ① 다른 신호로 넘어감
        ② 녹색 복귀 ③ 교차로 통과 가드 종료.
        """
        nxt = self._next_stopline(planner)
        if nxt is None:
            self._yellow_reset()
            return
        d_line, state, tl_id = nxt
        if self.y_ctrl is not None and tl_id != self.y_ctrl:
            self._yellow_reset()                      # 다음 교차로
        if state == 'Green':
            self._yellow_reset()                      # 녹색 복귀
            return
        if self.a_yellow <= 0.0 or self.y_decision is not None or state != 'Yellow':
            return                                    # 비활성 / 이미 래치 / 황색 아님
        v_allow = _math.sqrt(2.0 * self.a_yellow * max(0.0, d_line - self._s0(ap)))
        self.y_decision = 'stop' if ego_speed <= v_allow else 'go'
        self.y_ctrl = tl_id
        self.y_v_allow = v_allow
        # 판정 **순간**만 기록한다 — 사후 분류(어느 접근이 STOP/GO 였나)의 근거.
        # 매 틱 싣지 않는 이유: 판정은 접근당 1회고, 그 1틱이 조건을 다 담는다.
        self.last_yellow = {'decision': self.y_decision, 'ctrl': tl_id,
                            'v': round(float(ego_speed), 2),
                            'v_allow': round(float(v_allow), 2),
                            'd_line': round(float(d_line), 2),
                            'a_judge': self.a_yellow}

    def _yellow_reset(self) -> None:
        self.y_decision = None
        self.y_ctrl = None
        self.y_v_allow = None

    def on_reset(self) -> None:
        """courseRespawn — 순간이동 전 래치는 전부 무효 (run_agent 가 부른다).

        특히 GO 래치를 살려 두면, 리스폰으로 정지선 **뒤로** 되돌아간 뒤에도
        "이미 가기로 했다"가 유지돼 적신호를 그대로 통과한다 (항목7 중대).
        """
        self._yellow_reset()
        self.cross_guard = False
        self.cross_s = None
        self.cross_junction_seen = False
        self.sl_hold_left = 0
        self.latched = False

    def _s0(self, ap) -> float:
        """계획 정지점의 뒷축 gap — PDM 주입값이 단일 출처."""
        return float(getattr(ap.config, 'idm_red_light_minimum_distance',
                             self.stop_gap_sl_fallback))

    def _cross_guard(self, planner, ap, d_line) -> bool:
        """교차로 통과 가드 — 앞범퍼가 정지선을 넘은 뒤 교차로를 벗어날 때까지
        **신호·정지선 유래 정지 후보를 만들지 않는다** (보행자·선행차 후보는
        min() 의 다른 갈래라 그대로 산다).

        가드가 없으면 교차로 한복판에서 뒤쪽 정지선을 향해 제동하거나, 바로
        다음 정지선에 성급히 반응한다. 해제는 교차로를 벗어났을 때, 또는 진입이
        관측되지 않은 채 yellow_guard_max_m 를 지났을 때(상한 — 고착 방지).
        """
        route_s = float(planner.route_s[planner.route_index])
        if not self.cross_guard and d_line is not None and (d_line - self.front) <= 0.0:
            self.cross_guard = True                   # 앞범퍼가 정지선을 넘었다
            self.cross_s = route_s
            self.cross_junction_seen = False
        if not self.cross_guard:
            return False
        in_j = bool(getattr(ap, 'junction', False))
        if in_j:
            self.cross_junction_seen = True
        elif self.cross_junction_seen or (
                self.cross_s is not None       # 0.0 은 falsy — or 로 폴백하면 안 된다
                and route_s - self.cross_s > self.y_guard_max_m):
            self.cross_guard = False
            self.cross_s = None
            self._yellow_reset()
            return False
        return True

    def signal_release(self, ap, _distance_to_traffic_light=None) -> bool:
        """PDM 의 적신호 IDM 을 이번 틱 건너뛸 것인가 — autopilot 조기 반환 조건.

        kr_rules 는 min() 에 후보를 **덧대기만** 하므로 PDM 이 스스로 만드는
        적신호 감속을 없앨 수 없다. 황색 GO 와 교차로 통과 가드는 "감속하지
        말 것"이 요지라, 이 규칙만 예외적으로 판단(여기)과 소비(autopilot 한 줄)가
        분리된다. 녹색일 때 IDM 을 건너뛰는 것과 같은 메커니즘이고 IDM 본문은
        무수정이다.

        참이 되는 경우는 둘뿐이다:
          · 황색 GO 래치 — 접근당 1회 판정으로 "설 수 없다" 가 확정된 상태.
            여기서 PDM 이 감속하면 어중간히 늦춰 정지선에 걸친다 (실측 2026-08-30
            실전주행_02: 황색 3 s 동안 IDM 이 가속↔감속을 오가다 slf=+1.35).
          · 교차로 통과 가드 — 앞범퍼가 이미 정지선을 넘었다. 여기서 제동하면
            걸친 채로 선다.
        보행자·선행차 후보는 min() 의 다른 갈래라 그대로 살아 있다.
        """
        return bool(self.y_decision == 'go' or self.cross_guard)

    def _stop_target(self, planner, ap) -> tuple | None:
        """정지 후보를 만들 대상이면 (뒷축거리, 실행 감속 a_eff), 아니면 None.

        색 해석의 **단일 출처**다 — 프로파일과 홀드가 같은 판정을 본다.
          · 적색            → (d, stop_profile_a)
          · 황색 + STOP 래치 → (d, stop_profile_a)  적색과 **완전히 동일** 취급
          · 황색 + GO 래치   → None
          · 녹색 / 신호 없음 → None

        판정(a_yellow=4.0)과 실행(stop_profile_a=3.0)의 상수가 **다른 것이
        의도다**. 같게 두면(초기 설계안 B) STOP 판정의 정의상 진입 시
        v ≤ v_allow 라 프로파일이 느슨하고, v_allow 가 v 밑으로 내려올 때까지
        구속하지 못한다. 그 시점엔 최대 감속을 여유 0 으로 요구해 jerk 램프인에
        진다 — 폐루프 12조건에서 걸침 6건. 실행 상수를 작게 두면 진입 즉시
        구속되고 a_dec_max 까지 여유가 남는다 (같은 12조건에서 걸침 0건,
        전부 −1.52~−0.97). 2026-08-30 검증.
        교차로 통과 가드가 걸려 있으면 무조건 None.
        """
        nxt = self._next_stopline(planner)
        d_line = nxt[0] if nxt else None
        if self._cross_guard(planner, ap, d_line):
            return None
        if nxt is None:
            return None
        d_line, state, _tl_id = nxt
        if state == 'Red' and self.y_decision != 'go':
            return (d_line, self.stop_profile_a)
        if state == 'Yellow' and self.y_decision == 'stop':
            return (d_line, self.stop_profile_a)
        return None

    def _red_stopline_dist(self, planner) -> float | None:
        """구 인터페이스 — 적색이면 뒷축거리. 색 해석은 _stop_target 이 한다."""
        nxt = self._next_stopline(planner)
        if nxt is None or nxt[1] != 'Red':
            return None
        return nxt[0]

    def _stopline_profile(self, planner, ap) -> float | None:
        """적신호 정지선까지의 **정지 프로파일 속도 상한** — min() 후보.

        PDM 의 적신호 IDM 은 차간모형이라 정지 컨트롤러가 아니다: 평형이
        s* (= s0 + vT + v²/2√(ab)) 라, 남은 거리가 s* 보다 조금만 커도 **가속을
        요구한다** (실측 2026-08-30 실전주행: 접근 92틱 중 41틱이 v 보다 높은
        목표, err/dt 최대 +10.5 m/s²). 종방향이 그 요구에 브레이크를 풀면
        타행으로 2.5 m 를 먹고 정지선을 넘는다 (앞범퍼 −1.50 목표에 −0.12 착지).

        여기서는 남은 거리로부터 **단조 감소하는 속도 상한**을 만들어 덧댄다:

            d_stop  = 정지선거리 − s0        (s0 = PDM 과 동일값, 아래 참조)
            v_allow = √(2 · a_stop · d_stop)

        · 단조라 감속 커맨드에 부호 반전이 없다 → jerk 리미터가 되감을 일이
          없다 (재제동 지연이 구조적으로 사라진다).
        · d_stop → 0 에서 v_allow → 0 이라 점근 크립이 아니라 유한 시간 도달.
          목표 0 은 종방향의 a_hold 분기(target<1e-5 ∧ v<0.2)로 자연 접속된다.
        · 멀리서는 v_allow 가 제한속도보다 커서 min() 에 지므로 스스로 비활성
          이다 (별도 발동 거리 상수를 두지 않는 이유).

        s0 는 PDM 에 주입된 idm_red_light_minimum_distance 를 그대로 읽는다 —
        run_agent.build_pdm_config 가 params 의 stop_gap_stopline_m + 앞범퍼로
        채우는 값이라, 여기서 다시 계산하면 단일 출처가 깨진다.
        """
        tgt = self._stop_target(planner, ap)
        if tgt is None:
            return None
        d_line, a_eff = tgt
        if a_eff <= 0.0:
            return None
        return _math.sqrt(2.0 * a_eff * max(0.0, d_line - self._s0(ap)))

    def _stopline_hold(self, planner, ego_speed: float) -> float | None:
        """적신호 정지선 정지의 최소 유지 (speed.stopline_hold_s) — 목표 0 후보.

        다음 신호 정지선이 적색이고 앞범퍼가 stopline_hold_near_m 안에서
        저속(latch_v — 기존 래치 관례 재사용)이 되면 홀드 시작. 홀드 중에는
        신호가 녹색으로 바뀌어도 잔여 틱을 채운다 (규정 "0.5 s 이상 정지" —
        실측 0.4 s 재출발이 감점 대상). 신호 정보가 없는 환경(목 플래너 등)
        에서는 개입하지 않는다.
        """
        if self.sl_hold_left > 0:
            self.sl_hold_left -= 1
            return 0.0
        tgt = self._stop_target(planner, self._ap)
        if tgt is None:
            return None
        d_front = tgt[0] - self.front
        if d_front < self.sl_near_m and ego_speed < self.latch_v:
            self.sl_hold_left = max(0, self.sl_hold_ticks - 1)   # 이번 틱 포함
            return 0.0
        return None

    def apply(self, control, target_speed: float, ap):
        """(control, target_speed) → 규칙 반영 후 (control, target_speed).

        ap 는 AutoPilot 인스턴스 (판단 컨텍스트: _waypoint_planner /
        _compute_target_speed_idm / _longitudinal_controller / _vehicle).
        d_end 는 정지 기준점 stop_s 까지 남은 planner route_s — ego.route_s 와
        같은 축이고, courseRespawn 후 reset_index() 재탐색을 그대로 따라간다.
        래치(latch_m/unlatch_m)·active_m 판정도 이 d_end(stop_s 기준)를 쓴다.
        """
        planner = ap._waypoint_planner
        if self.stop_s is None:
            self.stop_s = self._resolve_stop_s(planner)
        route_s = float(planner.route_s[planner.route_index])
        d_end = self.stop_s - route_s
        ego_speed = ap._vehicle.get_velocity().length()
        self.last_candidate = None
        self.last_stop_profile = None
        self.last_d_end = d_end
        self._ap = ap
        self.last_yellow = None
        # 황색 원샷 판정 — 프로파일·홀드보다 먼저 정해야 같은 틱에 반영된다
        self._yellow_latch(planner, ego_speed, ap)

        # 정적 장애물 회피 — 경로를 밀면 PDM 의 선행차 판정에서 빠져 다시 달린다.
        self.last_avoid = None
        self._update_obj_timers(ap)
        # (지시등은 lat_shift 를 보므로 시프트를 자동으로 따라온다)
        self._try_overtake(ap, planner, ego_speed)

        # 방향지시등 — 속도 중재와 독립이다 (켜는 것이 감속을 만들지 않는다)
        (self.last_turn_signal, self.last_sig_src,
         self.last_sig_lead_s) = self._turn_signal(planner, route_s, ego_speed)

        # 래치 해제: 종점에서 다시 멀어졌다 = 리셋으로 뒤로 갔다 (고착 방지)
        if self.latched and d_end > self.unlatch_m:
            self.latched = False

        # 래치 진입: 종점 근처에서 사실상 정지 (latch_v 는 batch 완주 판정과 동일)
        if not self.latched and d_end <= self.latch_m and ego_speed < self.latch_v:
            self.latched = True

        candidate = None
        if self.latched:
            candidate = 0.0
        elif d_end <= self.active_m and target_speed > 0.1:
            # 종점의 유령 선행차 (정지, 길이 0). 유효거리는 앞범퍼 기준 —
            # IDM 이 net gap ≈ s0(stop_gap)에서 서므로 앞범퍼가 종점 − stop_gap.
            d_eff = max(0.1, d_end - self.front)
            candidate = float(ap._compute_target_speed_idm(
                desired_speed=target_speed,
                leading_actor_length=0.0,
                ego_speed=ego_speed,
                leading_actor_speed=0.0,
                distance_to_leading_actor=d_eff,
                s0=self.stop_gap,
                T=self.T,
            ))

        # 정지선 정지 프로파일 (④′) — 적색일 때만, min 으로 합류.
        # route_end 는 대상이 아니다 (검증 통과 후 별건).
        prof = self._stopline_profile(planner, ap)
        self.last_stop_profile = prof
        if prof is not None and (candidate is None or prof < candidate):
            candidate = prof

        # 정지선 0.5 s 유지 홀드 — route_end 후보와 min 으로 합류
        hold = self._stopline_hold(planner, ego_speed)
        if hold is not None and (candidate is None or hold < candidate):
            candidate = hold

        if candidate is not None:
            self.last_candidate = candidate
            if candidate < target_speed:
                target_speed = candidate
                # 종방향 재계산 — 본류가 이번 틱 이미 호출했으므로 되감고 다시
                # (되감지 않으면 두 호출이 jerk 창을 나눠 갖는 핑퐁 — rewind_last 참고)
                hazard = target_speed < 1e-5
                ap._longitudinal_controller.rewind_last()
                accel, brake = ap._longitudinal_controller.get_throttle_and_brake(
                    hazard, target_speed, ego_speed)
                control.accel = accel
                control.throttle = accel
                control.brake = float(brake)

        self.last_target = float(target_speed)
        return control, target_speed
