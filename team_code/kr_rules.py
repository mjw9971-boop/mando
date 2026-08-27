"""
한국 대회 규칙 계층 — PDM-Lite 판단 결과를 받아 규칙을 덮어쓴다.

PDM-Lite(autopilot.py) 원문은 건드리지 않는다. autopilot._get_control 맨 끝의
한 줄(`self.kr_rules.apply(...)`)이 유일한 접점이고, 여기서는 PDM 의 min()
중재에 **후보를 덧대는** 형태로만 개입한다 — 새 감속 프로파일을 만들지 않고
PDM 의 _compute_target_speed_idm / 종방향 컨트롤러를 그대로 재사용한다.

phase4 현재: route_end 정지만 구현. (깜빡이·RTOR·황색 딜레마는 이후 단계.)

route_end — 경로 종점 정지:
  CARLA 리더보드는 결승선 통과로 시나리오가 끝나 PDM 에 종점 정지 개념이
  없다. 실기(2026-08-26 완주속도_01_기본): 종점 도달 후 v_target 6.9 로 계속
  주행 → 경로 밖 이탈 → courseRespawn 9회.

  구현: "종점에 정지해 있는 길이 0 유령 선행차" 를 IDM 에 넣는다. 유효거리를
  d_end − 앞범퍼, s0 를 speed.stop_gap_m 으로 주면 앞범퍼가 종점 −
  stop_gap 에 선다 — 기존 정지선 정지와 같은 관례고, batch 완주 임계
  (total − end_margin, end_margin = stop_gap + 앞범퍼 + end_slack)보다
  end_slack_m 만큼 안쪽이라 완주 판정과 자동 정합한다 (tests/test_route_end).

  래치: 종점 근처(latch_m)에서 저속(latch_v)이 되면 래치 — 재출발하지 않는다.
  d_end 가 unlatch_m 이상으로 다시 커지면(courseRespawn 으로 뒤로 간 경우)
  해제해 고착을 막는다.

  정지 목표 기준점(stop_s): 대회 규칙은 "뒷축이 종료 지점 통과 = 시험 종료"라
  route_end.target_mode='finish' 면 scoring.finish_xy 를 경로에 투영한 종료선
  (finish_s)을 뒷축이 finish_clearance_m 만큼 넘어 정지하도록 기준점을 잡는다
  (plan_stop_s — 채점 score.py 와 공용, 단일 출처). d_eff/s0 관례·래치·active_m
  판정 거리는 전부 stop_s 기준으로 그대로 동작한다.
"""
from __future__ import annotations


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
            + float(sp['stop_gap_m'])
            + float(vh['wheelbase']) + float(vh['front_overhang_m']))
    cap = float(total) - float(cfg.get('batch', {}).get('end_slack_m', 1.0))
    return (min(want, cap), want > cap)


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
        self.stop_gap = float(sp['stop_gap_m'])
        self.front = float(vh['wheelbase']) + float(vh['front_overhang_m'])
        self.T = float(re_cfg['idm_time_headway'])
        self.active_m = float(re_cfg['active_m'])
        self.latch_v = float(re_cfg['latch_v'])
        self.latch_m = float(re_cfg['latch_m'])
        self.unlatch_m = float(re_cfg['unlatch_m'])
        self.target_mode = str(re_cfg['target_mode'])
        self.finish_xy = (cfg['scoring'] or {}).get('finish_xy')

        self.latched = False
        self.stop_s: float | None = None           # 시작 시 1회 계산 캐시 (매 틱 투영 금지)
        self.last_candidate: float | None = None   # 이번 틱 route_end 후보 (로그용)
        self.last_target: float | None = None      # 이번 틱 최종 목표속도 (로그용)
        self.last_d_end: float | None = None

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
        d_end = self.stop_s - float(planner.route_s[planner.route_index])
        ego_speed = ap._vehicle.get_velocity().length()
        self.last_candidate = None
        self.last_d_end = d_end

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
