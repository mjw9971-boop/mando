"""
ctrl24 — won_24 제어기 (2026-09-08).

kr_rules 를 **대체**하는 한국 대회 규칙 계층. 접점은 kr_rules 와 같다:
  · autopilot._get_control 끝의 `apply(control, target_speed, ap)` — min() 후보 덧대기
  · autopilot 적신호 IDM 의 `signal_release(ap, d)` — 유일한 오버라이드 훅
  · run_agent 가 읽는 last_* (turn_signal / sig_src / sig_lead_s / kr / avoid / ped / yellow / signal)
  + `pre_pass(ap, route_np, vehicles, target_speed, ego_speed)` — PDM 후보 계산 **앞**에서
    정적 장애물 회피 시프트를 만드는 훅 (커밋 3).

kr_rules 와의 차이 (설계 결정 2026-09-08, 사용자 승인):
  유지  P1 leading / P2 vehicle / P3 red_light (PDM 원문) · K1 stop_profile · K2 stop_hold ·
        K3 ped_intent(walkin·multi hold·release) · K4 emergency · K5 crosswalk(기본 off) ·
        황색 원샷 STOP/GO · 교차로 통과 가드 · 보행자 래치 · timeout GO · RTOR · 지시등
  제거  PDM bicycle·pedestrian 후보(run_agent 가 forecast_walkers 를 비운다) · route_end ·
        BREAKOUT · standoff(프로파일·크립·delay) · shift_cap · gap_v_req · red_zone 접근 감속 ·
        큐 판정·SUPPRESS · WAIT/WAIT_EXPIRED/REACTIVE · 정지 관찰 시계 전부 · span_extend ·
        preempt_latch · side 게이트 8개(no_neighbor/center_line/geom/zone/solid/occupied/
        kappa·lc_overlap/entry_block) · side_pick · gap_fit · shift_entry
  변경  정적 장애물은 첫 틱에 PREEMPT (관찰·대기·예산 없음) · side 게이트는 span_too_far 하나 ·
        중첩 시프트 (활성 span 위에 같은 방향으로 한 칸 더, 횟수 제한 없음) ·
        종점 정지 없이 계속 주행 (route.py 의 종점 패드) · P2 정지 객체는 pre_pass 시프트로 넘긴다

상수는 config/params.yaml 의 `ctrl24:` 섹션이 단일 출처다. 차량 제원·종방향 한계
(vehicle.* / control.a_dec_max / speed.a_emergency)·회랑 여유(percep.obstacle_clearance_m —
P1 의 정지 객체 폭 판정과 같은 값이어야 한다)는 그 섹션을 읽는다.
"""
from __future__ import annotations

import math as _math

import numpy as np

from vtd_adapter import frame

SIG_OFF, SIG_LEFT, SIG_RIGHT = 0, 1, 2        # 9910 turnSignal (SPEC §1.2)


class Ctrl24:
    def __init__(self, cfg: dict) -> None:
        c = cfg['ctrl24']
        vh = cfg['vehicle']
        self.cfg = cfg
        self.c = c
        self.hz = float(cfg['comm']['send_hz'])
        self.front = float(vh['wheelbase']) + float(vh['front_overhang_m'])
        self.half_ego = float(vh['width']) / 2.0
        # ── K1·K2 정지선 (kr_rules ④′·B-1 확정 스펙 그대로) ────────────────
        self.stop_profile_a = float(c['stop_profile_a'])
        self.sl_hold_ticks = int(round(float(c['stopline_hold_s']) * self.hz))
        self.sl_min_ticks = int(round(float(c['stopline_hold_min_s']) * self.hz))
        self.sl_near_m = float(c['stopline_hold_near_m'])
        # ap.config 를 못 읽는 환경(목)에서만 쓰는 폴백 — 정상 경로는 PDM 주입값
        self.stop_gap_sl_fallback = float(cfg['speed']['stop_gap_stopline_m']) + self.front
        # 황색 원샷 판정·교차로 통과 가드
        self.a_yellow = float(c['a_yellow'])
        self.y_guard_max_m = float(c['yellow_guard_max_m'])
        # ── K3·K4·K5 보행자 ────────────────────────────────────────────────
        self.ped_intent_v = float(c['ped_intent_v'])
        self.ped_emg_ratio = float(c['ped_emergency_ratio'])
        self.ped_release_lat = float(c['ped_release_lat_m'])
        self.ped_release_ticks = int(round(float(c['ped_release_s']) * self.hz))
        self.ped_stop_v = float(c['ped_stop_v'])
        self.ped_offroad_lat = float(c['ped_offroad_lat_m'])
        self.ped_backstop_ticks = int(round(float(c['ped_backstop_s']) * self.hz))
        self.walkin_v = float(c['ped_walkin_v'])
        # 0.0 = 첫 틱 래치 (정지 관찰 시계가 없어 옛 ped_static 경로가 사라진 만큼 되돌린다).
        self.walkin_ticks = int(round(float(c['ped_walkin_s']) * self.hz))
        self.walkin_lat = float(c['ped_walkin_lat_m'])
        self.ped_multi = bool(c['ped_multi_enable'])
        self.ped_coast_ticks = int(round(float(c['ped_hold_coast_s']) * self.hz))
        self.cw_enable = bool(c['ped_crosswalk_creep_enable'])
        self.cw_zone_m = float(c['ped_crosswalk_zone_m'])
        self.cw_lat_m = float(c['ped_crosswalk_lat_m'])
        self.cw_wait_ticks = int(round(float(c['ped_crosswalk_wait_s']) * self.hz))
        self.cw_creep_v = float(c['ped_crosswalk_creep_v'])
        self.a_emergency = float(cfg['speed']['a_emergency'])
        self.a_dec_max = abs(float(cfg['control']['a_dec_max']))
        self.detect_max_m = float(c['detect_max_m'])

        # ── 상태 ──────────────────────────────────────────────────────────
        self._ap = None
        self.last_target: float | None = None
        self.last_kr: dict = {}                       # 이번 틱 kr 후보 (reasons.kr)
        self.last_kr_winner: str | None = None        # kr 후보 중 최저 (run_agent 가 winner 로)
        self.last_stop_profile: float | None = None
        self.last_avoid: dict | None = None
        self.last_ped: dict | None = None
        self.last_yellow: dict | None = None
        self.last_signal: dict | None = None
        self.ped_emergency = False
        self.last_turn_signal: int = SIG_OFF
        self.last_sig_src: str | None = None
        self.last_sig_lead_s: float | None = None
        # K2
        self.sl_hold_left = 0
        self.sl_stopped = False
        self.sl_stop_ticks = 0
        # 황색 래치 / 가드
        self.y_decision: str | None = None
        self.y_ctrl: int | None = None
        self.y_v_allow: float | None = None
        self.cross_guard = False
        self.cross_s: float | None = None
        self.cross_junction_seen = False
        # 보행자 (P4)
        self.ped_lat: dict = {}
        self.ped_intent: set = set()
        self.ped_clear: dict = {}
        self.ped_hold: dict = {}
        self.ped_diag: dict = {}
        self.ped_released: dict = {}
        self.ped_hold_ids: set = set()
        self.ped_miss: dict = {}
        self.ped_last: dict = {}
        self.ped_all: dict = {}
        self.ped_walkin: dict = {}
        self.cw_wait: dict = {}
        self._cw_zones: list | None = None
        self._sl_all: list | None = None
        # 훅 호환 — BREAKOUT 은 없다. autopilot 의 `if self.kr_rules.breakout_creep()` 은
        # kr_rules 와 파일을 공유하므로 남겨 두고 여기서는 항상 거짓이다 (오버라이드 아님).
        self._sig_go = False                          # 커밋 2 (timeout GO)
        self._rtor_go = False                         # 커밋 2 (RTOR)
        self.ot_span = None                           # 커밋 3 (회피)

    # ── 훅 호환 ───────────────────────────────────────────────────────────
    def breakout_creep(self) -> bool:
        return False

    # ── 공통 ──────────────────────────────────────────────────────────────
    def _s0(self, ap) -> float:
        """계획 정지점의 뒷축 gap — PDM 주입값이 단일 출처."""
        return float(getattr(getattr(ap, 'config', None), 'idm_red_light_minimum_distance',
                             self.stop_gap_sl_fallback))

    def _next_stopline(self, planner):
        """전방 신호 정지선 → (뒷축거리 [m], 상태명, 신호 id). 없으면 None."""
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

    def _project(self, planner, x_carla, y_carla):
        """CARLA 좌표 → (route_s 상대거리, 횡오프셋). 전방 창에서만 찾는다.
        경로점은 **현재** route_points 다 — 시프트가 얹히면 밀린 경로 기준이 된다."""
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

    def _all_stopline_s(self, planner) -> list:
        """경로상 모든 정지선 route_s (신호 유무 무관). 시작 시 1회. K5 의 폴백 구간."""
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

    # ── 황색 원샷 판정 / 교차로 통과 가드 / 해제 훅 (kr_rules 원문) ──────────
    def _yellow_latch(self, planner, ego_speed: float, ap) -> None:
        """접근당 1회 STOP/GO. v ≤ √(2·a_yellow·(d − s0)) → STOP, 그 외 GO. 번복 없음."""
        nxt = self._next_stopline(planner)
        if nxt is None:
            self._yellow_reset()
            return
        d_line, state, tl_id = nxt
        if self.y_ctrl is not None and tl_id != self.y_ctrl:
            self._yellow_reset()
        if state == 'Green':
            self._yellow_reset()
            return
        if self.a_yellow <= 0.0 or self.y_decision is not None or state != 'Yellow':
            return
        v_allow = _math.sqrt(2.0 * self.a_yellow * max(0.0, d_line - self._s0(ap)))
        self.y_decision = 'stop' if ego_speed <= v_allow else 'go'
        self.y_ctrl = tl_id
        self.y_v_allow = v_allow
        self.last_yellow = {'decision': self.y_decision, 'ctrl': tl_id,
                            'v': round(float(ego_speed), 2),
                            'v_allow': round(float(v_allow), 2),
                            'd_line': round(float(d_line), 2),
                            'a_judge': self.a_yellow}

    def _yellow_reset(self) -> None:
        self.y_decision = None
        self.y_ctrl = None
        self.y_v_allow = None

    def _cross_guard(self, planner, ap, d_line) -> bool:
        """앞범퍼가 정지선을 넘은 뒤 교차로를 벗어날 때까지 신호 정지 후보를 만들지 않는다.
        진입 미관측이면 yellow_guard_max_m 상한으로 푼다 (고착 방지)."""
        route_s = float(planner.route_s[planner.route_index])
        if not self.cross_guard and d_line is not None and (d_line - self.front) <= 0.0:
            self.cross_guard = True
            self.cross_s = route_s
            self.cross_junction_seen = False
        if not self.cross_guard:
            return False
        in_j = bool(getattr(ap, 'junction', False))
        if in_j:
            self.cross_junction_seen = True
        elif self.cross_junction_seen or (
                self.cross_s is not None
                and route_s - self.cross_s > self.y_guard_max_m):
            self.cross_guard = False
            self.cross_s = None
            self._yellow_reset()
            return False
        return True

    def signal_release(self, ap, _distance_to_traffic_light=None) -> bool:
        """PDM 적신호 IDM 을 이번 틱 건너뛸 것인가 — **유일한 오버라이드 훅**.
        황색 GO 래치 / 교차로 통과 가드 / timeout GO / RTOR 래치(붙은 신호 한정)."""
        return bool(self.y_decision == 'go' or self.cross_guard or self._sig_go
                    or self._rtor_active(getattr(ap, '_waypoint_planner', None)))

    def _rtor_active(self, planner) -> bool:          # 커밋 2 에서 채운다
        return False

    # ── K1·K2 ─────────────────────────────────────────────────────────────
    def _stop_target(self, planner, ap) -> tuple | None:
        """정지 후보 대상 — timeout GO / RTOR 래치가 살아 있으면 None."""
        if self._sig_go or self._rtor_active(planner):
            return None
        return self._stop_target_raw(planner, ap)

    def _stop_target_raw(self, planner, ap) -> tuple | None:
        """색 해석의 단일 출처. 적색 / 황색+STOP → (뒷축거리, a). 가드 중이면 None."""
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

    def _stopline_profile(self, planner, ap) -> float | None:
        """K1: v_allow = √(2·a·(d − s0)) — 단조 감소 상한. min() 후보."""
        tgt = self._stop_target(planner, ap)
        if tgt is None:
            return None
        d_line, a_eff = tgt
        if a_eff <= 0.0:
            return None
        return _math.sqrt(2.0 * a_eff * max(0.0, d_line - self._s0(ap)))

    def _stopline_hold(self, planner, ego_speed: float) -> float | None:
        """K2: 정지선 6 m 안 정지 시 1회 무장 stopline_hold_s. 녹색 전환·래치 해제 시
        최소 시간(stopline_hold_min_s) 미충족분만 남긴다 (B-1 확정 스펙 그대로)."""
        tgt = self._stop_target(planner, self._ap)
        stopped = ego_speed < 0.5
        near = tgt is not None and (tgt[0] - self.front) < self.sl_near_m
        if stopped:
            if near and not self.sl_stopped:
                self.sl_stopped = True
                self.sl_stop_ticks = 0
                self.sl_hold_left = self.sl_hold_ticks
        else:
            self.sl_stopped = False
            self.sl_stop_ticks = 0
        if tgt is None and self.sl_hold_left > 0:
            self.sl_hold_left = min(self.sl_hold_left,
                                    max(0, self.sl_min_ticks - self.sl_stop_ticks))
        if stopped:
            self.sl_stop_ticks += 1
        if self.sl_hold_left > 0:
            self.sl_hold_left -= 1
            return 0.0
        return None

    # ── K3·K5 보행자 (kr_rules 원문 — ped_static 경로만 없다) ──────────────
    @staticmethod
    def _walkers(ap) -> list:
        try:
            return list(ap._world.get_actors().filter('*walker*'))
        except Exception:                                  # noqa: BLE001
            return []

    def _ped_intent(self, planner, ap, ego_speed: float):
        """경로 쪽으로 걸어나오는 보행자의 정지 후보 → (v_allow, a_req, id) | None.

        정지 관찰 시계(obj_static_s)가 없으므로 래치 경로는 walkin 하나다 —
        연속 ped_walkin_s(기본 0.0 = 첫 틱) 동안 보행자 속도 ≥ ped_intent_v ∧ 경로 쪽
        횡속도 ≥ ped_walkin_v ∧ |lat| < ped_walkin_lat_m. 회랑(|lat| < ped_release_lat_m)
        안은 홀드 래치(v_allow 0). 해제 clear/backstop/coast 는 A-1·P4-M 그대로.
        """
        if self.ped_intent_v <= 0.0 or self.stop_profile_a <= 0.0:
            return None
        walkers = self._walkers(ap)
        live = set()
        best = None
        self.ped_diag = {}
        self.ped_released = {}
        self.ped_all = {}
        multi = self.ped_multi and self.ped_release_lat > 0.0
        s0 = float(getattr(getattr(ap, 'config', None),
                           'idm_pedestrian_minimum_distance', 4.0))
        for w in walkers:
            wid = getattr(w, 'id', None)
            if wid is None:
                continue
            live.add(wid)
            loc = w.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is None:
                self.ped_lat.pop(wid, None)
                self.ped_clear.pop(wid, None)
                if multi and wid in self.ped_hold_ids and wid in self.ped_last:
                    self.ped_miss[wid] = self.ped_miss.get(wid, 0) + 1
                    if self.ped_miss[wid] <= self.ped_coast_ticks:
                        v_allow, a_req = self.ped_last[wid]
                        self.ped_all[wid] = {'v_allow': round(float(v_allow), 2),
                                             'latched': wid in self.ped_intent,
                                             'hold': True, 'coast': self.ped_miss[wid],
                                             'unprojectable': True}
                        if best is None or v_allow < best[0]:
                            best = (v_allow, a_req, wid)
                    else:
                        self._ped_unlatch(wid)
                continue
            s_rel, lat = pr
            prev = self.ped_lat.get(wid)
            self.ped_lat[wid] = lat
            v_toward = None if prev is None else (abs(prev) - abs(lat)) * self.hz
            if not (0.0 < s_rel <= self.detect_max_m):
                self._ped_unlatch(wid)
                continue
            self.ped_miss[wid] = 0
            if (multi and abs(lat) < self.ped_release_lat
                    and wid not in self.ped_hold_ids):
                self.ped_hold_ids.add(wid)
                self.ped_clear.setdefault(wid, 0)
                self.ped_hold.setdefault(wid, 0)
            if wid not in self.ped_intent:
                latch_ok = v_toward is not None and self._ped_walkin(
                    wid, float(getattr(w, 'speed', 0.0)), v_toward, lat)
                if latch_ok:
                    self.ped_intent.add(wid)
                    self.ped_clear[wid] = 0
                    self.ped_hold[wid] = 0
                    self.ped_walkin.pop(wid, None)
                elif wid not in self.ped_hold_ids:
                    if multi:
                        self.ped_all[wid] = {'s_rel': round(float(s_rel), 2),
                                             'lat': round(float(lat), 2),
                                             'latched': False, 'hold': False}
                    continue
            why = self._ped_release_tick(wid, w, lat, v_toward)
            self.ped_diag[wid] = {
                'lat': round(float(lat), 2),
                'v_toward': None if v_toward is None else round(float(v_toward), 2),
                'clear_s': round(self.ped_clear.get(wid, 0) / self.hz, 1),
                'hold_s': round(self.ped_hold.get(wid, 0) / self.hz, 1)}
            if why is not None:
                self._ped_unlatch(wid)
                self.ped_released[wid] = why
                continue
            d_eff = s_rel - self.front - s0
            hold = multi and wid in self.ped_hold_ids
            v_allow = 0.0 if hold else _math.sqrt(2.0 * self.stop_profile_a * max(0.0, d_eff))
            a_req = ego_speed * ego_speed / (2.0 * max(d_eff, 0.5))
            if multi:
                self.ped_last[wid] = (v_allow, a_req)
                self.ped_all[wid] = {'s_rel': round(float(s_rel), 2),
                                     'lat': round(float(lat), 2),
                                     'v_allow': round(float(v_allow), 2),
                                     'latched': wid in self.ped_intent, 'hold': bool(hold)}
            if best is None or v_allow < best[0]:
                best = (v_allow, a_req, wid)
        for wid in list(self.ped_intent | self.ped_hold_ids):
            if wid in live:
                continue
            if multi and self.ped_coast_ticks > 0 and wid in self.ped_last:
                self.ped_miss[wid] = self.ped_miss.get(wid, 0) + 1
                if self.ped_miss[wid] <= self.ped_coast_ticks:
                    v_allow, a_req = self.ped_last[wid]
                    self.ped_all[wid] = {'v_allow': round(float(v_allow), 2),
                                         'latched': wid in self.ped_intent,
                                         'hold': wid in self.ped_hold_ids,
                                         'coast': self.ped_miss[wid]}
                    if best is None or v_allow < best[0]:
                        best = (v_allow, a_req, wid)
                    continue
            self._ped_unlatch(wid)
        for d in (self.ped_lat, self.ped_walkin):
            for wid in list(d):
                if wid not in live:
                    d.pop(wid, None)
        return best

    def _ped_walkin(self, wid, w_speed: float, v_toward: float, lat: float) -> bool:
        """래치 조건 (A-4 walkin). ped_walkin_s 0 이면 조건 성립 첫 틱에 래치.
        한 틱이라도 깨지면 처음부터."""
        ok = (w_speed >= self.ped_intent_v and v_toward >= self.walkin_v
              and abs(lat) < self.walkin_lat)
        if not ok:
            self.ped_walkin[wid] = 0
            return False
        self.ped_walkin[wid] = self.ped_walkin.get(wid, 0) + 1
        return self.ped_walkin[wid] >= max(1, self.walkin_ticks)

    def _ped_unlatch(self, wid) -> None:
        self.ped_intent.discard(wid)
        self.ped_hold_ids.discard(wid)
        self.ped_clear.pop(wid, None)
        self.ped_hold.pop(wid, None)
        self.ped_miss.pop(wid, None)
        self.ped_last.pop(wid, None)

    def _ped_release_tick(self, wid, w, lat: float, v_toward) -> str | None:
        """A-1 위치 기반 해제: clear(회랑 밖 ∧ 멀어짐 ∧ 멈춤/차도 밖 연속) / backstop."""
        if self.ped_release_lat <= 0.0:
            return None
        outside = abs(lat) > self.ped_release_lat
        self.ped_hold[wid] = self.ped_hold.get(wid, 0) + 1
        away = v_toward is not None and v_toward <= 0.0
        settled = (float(getattr(w, 'speed', 0.0)) < self.ped_stop_v
                   or abs(lat) > self.ped_offroad_lat)
        if outside and away and settled:
            self.ped_clear[wid] = self.ped_clear.get(wid, 0) + 1
        else:
            self.ped_clear[wid] = 0
        if self.ped_clear[wid] >= self.ped_release_ticks:
            return 'clear'
        if outside and self.ped_hold[wid] >= self.ped_backstop_ticks:
            return 'backstop'
        return None

    def _crosswalk_zones(self, planner) -> list:
        if self._cw_zones is not None:
            return self._cw_zones
        out = []
        lg = getattr(planner, 'lg', None)
        route = getattr(planner, 'route', None) or {}
        if lg is not None:
            for i, k in enumerate(route.get('lanes') or []):
                rec = lg.lanes.get(tuple(k))
                if not rec:
                    continue
                base = float(route['cum_s'][i])
                for a, b, _kind in rec.get('crosswalks', []):
                    out.append((base + float(a), base + float(b)))
        if not out:
            out = [(s, s) for s in self._all_stopline_s(planner)]
        self._cw_zones = sorted(out)
        return self._cw_zones

    def _ped_crosswalk(self, planner, ap, ego_speed: float):
        """K5 (A-3, 기본 off): 횡단보도 앞에 서 있는 보행자 — 정지 프로파일 → 대기 → creep."""
        if not self.cw_enable or self.stop_profile_a <= 0.0:
            return None
        walkers = self._walkers(ap)
        zones = self._crosswalk_zones(planner)
        if not walkers or not zones:
            self.cw_wait.clear()
            return None
        route_s = float(planner.route_s[planner.route_index])
        s0 = float(getattr(getattr(ap, 'config', None),
                           'idm_pedestrian_minimum_distance', 4.0))
        best, live = None, set()
        for w in walkers:
            wid = getattr(w, 'id', None)
            if wid is None or wid in self.ped_intent:
                continue
            if float(getattr(w, 'speed', 0.0)) >= self.ped_stop_v:
                continue
            loc = w.get_location()
            pr = self._project(planner, loc.x, loc.y)
            if pr is None:
                continue
            s_rel, lat = pr
            if not (0.0 < s_rel <= self.detect_max_m):
                continue
            if abs(lat) < self.ped_release_lat or abs(lat) >= self.cw_lat_m:
                continue
            ps = route_s + s_rel
            if not any(a - self.cw_zone_m <= ps <= b + self.cw_zone_m for a, b in zones):
                continue
            live.add(wid)
            d_eff = s_rel - self.front - s0
            v_prof = _math.sqrt(2.0 * self.stop_profile_a * max(0.0, d_eff))
            ticks = self.cw_wait.get(wid, 0)
            if (ticks < self.cw_wait_ticks and ego_speed < self.ped_stop_v
                    and d_eff <= self.cw_zone_m):
                ticks += 1
            self.cw_wait[wid] = ticks
            if ticks >= self.cw_wait_ticks:
                v_allow, phase = self.cw_creep_v, 'creep'
            else:
                v_allow, phase = v_prof, 'wait'
            if best is None or v_allow < best[0]:
                best = (v_allow, {'id': int(wid), 'phase': phase,
                                  'wait_s': round(ticks / self.hz, 1),
                                  'lat': round(float(lat), 2), 's_rel': round(float(s_rel), 1),
                                  'v_allow': round(float(v_allow), 2)})
        for wid in list(self.cw_wait):
            if wid not in live:
                self.cw_wait.pop(wid, None)
        return best

    # ── 리셋 ──────────────────────────────────────────────────────────────
    def on_reset(self) -> None:
        """courseRespawn — 순간이동 전 래치는 전부 무효 (run_agent 가 부른다)."""
        self._yellow_reset()
        self.cross_guard = False
        self.cross_s = None
        self.cross_junction_seen = False
        self.sl_hold_left = 0
        self.sl_stopped = False
        self.sl_stop_ticks = 0
        self.ped_lat.clear()
        self.ped_intent.clear()
        self.ped_clear.clear()
        self.ped_hold.clear()
        self.cw_wait.clear()
        self.ped_walkin.clear()
        self.ped_hold_ids.clear()
        self.ped_miss.clear()
        self.ped_last.clear()

    # ── 틱 ────────────────────────────────────────────────────────────────
    def _tick_head(self, ap, planner, ego_speed: float) -> None:
        """apply 머리 — 틱당 1회 상태·진단 리셋과 래치 갱신. 커밋 2·3 이 확장한다."""
        self._ap = ap
        self.last_stop_profile = None
        self.last_yellow = None
        self.last_ped = None
        self.last_kr = {}
        self.last_kr_winner = None
        self.ped_emergency = False
        self._yellow_latch(planner, ego_speed, ap)

    def _tick_signals(self, ap, planner, ego_speed: float, route_s: float) -> None:
        """timeout GO · RTOR · 지시등 — 커밋 2."""
        self.last_turn_signal, self.last_sig_src, self.last_sig_lead_s = SIG_OFF, None, None

    def _candidates(self, ap, planner, ego_speed: float, target_speed: float):
        """kr 후보 → (candidate, ped_bind, ped). 전부 min() 후보다."""
        kr: dict = {}
        candidate = None

        def add(name, v):
            nonlocal candidate
            if v is None:
                kr[name] = None
                return
            kr[name] = round(float(v), 3)
            if candidate is None or v < candidate:
                candidate = v

        prof = self._stopline_profile(planner, ap)
        self.last_stop_profile = prof
        add('stop_profile', prof)
        add('stop_hold', self._stopline_hold(planner, ego_speed))
        add('rtor_cap', self._rtor_cap())
        ped = self._ped_intent(planner, ap, ego_speed)
        ped_bind = False
        if ped is not None:
            v_allow, a_req, wid = ped
            ped_bind = candidate is None or v_allow <= candidate + 1e-9
            add('ped_intent', v_allow)
            self.last_ped = {'id': int(wid), 'v_allow': round(float(v_allow), 2),
                             'a_req': round(float(a_req), 2), 'wins': bool(ped_bind),
                             **self.ped_diag.get(wid, {})}
            if self.ped_released:
                self.last_ped['released'] = {int(k): v for k, v in self.ped_released.items()}
        else:
            add('ped_intent', None)
            if self.ped_released:
                wid, why = next(iter(self.ped_released.items()))
                self.last_ped = {'id': int(wid), 'wins': False, 'release': why,
                                 **self.ped_diag.get(wid, {})}
        if self.ped_multi and (self.ped_all or self.ped_hold_ids):
            if self.last_ped is None:
                self.last_ped = {'id': None, 'wins': False}
            self.last_ped['all'] = {int(k): v for k, v in self.ped_all.items()}
            self.last_ped['hold'] = sorted(int(k) for k in self.ped_hold_ids)
        cw = self._ped_crosswalk(planner, ap, ego_speed)
        if cw is not None:
            v_cw, info = cw
            add('crosswalk', v_cw)
            cw_wins = bool(v_cw <= min(target_speed, candidate) + 1e-9)
            if self.last_ped is None:
                self.last_ped = {'id': int(info['id']), 'wins': cw_wins, 'crosswalk': info}
            else:
                self.last_ped['crosswalk'] = info
        else:
            add('crosswalk', None)
        self.last_kr = kr
        if candidate is not None:
            self.last_kr_winner = min((k for k, v in kr.items() if v is not None),
                                      key=lambda k: kr[k])
        return candidate, ped_bind, ped

    def _rtor_cap(self):                              # 커밋 2
        return None

    def apply(self, control, target_speed: float, ap):
        """(control, target_speed) → 규칙 반영 후 (control, target_speed).
        속도는 전부 min() 후보다. 후보가 PDM 목표보다 낮으면 종방향을 되감아 재계산."""
        planner = ap._waypoint_planner
        route_s = float(planner.route_s[planner.route_index])
        ego_speed = ap._vehicle.get_velocity().length()
        self._tick_head(ap, planner, ego_speed)
        self._tick_signals(ap, planner, ego_speed, route_s)

        candidate, ped_bind, ped = self._candidates(ap, planner, ego_speed, target_speed)

        # K4 보행자 비상 우회 — 보행자 후보가 최종 목표를 구속하는 틱 한정.
        final_t = target_speed if candidate is None else min(target_speed, candidate)
        emg = bool(ped is not None and ped_bind and self.ped_emg_ratio > 0.0
                   and ped[1] > self.ped_emg_ratio * self.a_dec_max
                   and final_t <= ped[0] + 1e-9)
        if emg:
            self.ped_emergency = True
            if self.last_ped is not None:
                self.last_ped['emergency'] = True

        if (candidate is not None and candidate < target_speed) or emg:
            if candidate is not None and candidate < target_speed:
                target_speed = candidate
            hazard = target_speed < 1e-5
            ap._longitudinal_controller.rewind_last()
            if emg:
                accel, brake = ap._longitudinal_controller.emergency()
            else:
                accel, brake = ap._longitudinal_controller.get_throttle_and_brake(
                    hazard, target_speed, ego_speed)
            control.accel = accel
            control.throttle = accel
            control.brake = float(brake)
        self.last_target = float(target_speed)
        return control, target_speed
