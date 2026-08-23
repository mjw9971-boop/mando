"""
Decision 하드 클램프  (SPEC §3.5)

**Shield 는 Planner 를 신뢰하지 않는다.** Planner 가 버그로 위법한 값을 내도
여기서 잘린다. 발동한 clamp 는 전부 `Decision.reasons['shield']` 에 기록한다.
"""
from __future__ import annotations

import math

from .lanegraph import LaneGraph
from .types import Decision, WorldState


class Shield:
    def __init__(self, lg: LaneGraph, cfg: dict, planner=None) -> None:
        self.lg = lg
        self.cfg = cfg
        # 차선변경을 중단시키려면 planner 상태를 되돌려야 한다.
        # (Decision 만 깎으면 planner 는 다음 틱에 또 옆 차로 경로를 낸다)
        self.planner = planner
        # 비상제동 래치. 발동 ttc.emergency_s, 해제 ttc.brake_s (히스테리시스) —
        # 문턱 하나로 하면 TTC 가 경계에서 떨리며 제동이 껌뻑인다.
        self._estop = False

    def _overtake_exempt(self) -> int | None:
        """
        추월 전이 중인 **그 대상 한 대**의 id. TTC 판정에서 뺀다.

        정차 차량을 추월하려면 그 차 옆을 지나야 하는데, 서 있는 차라 TTC 가
        당연히 낮다. 빼지 않으면 _abort_lane_change_on_ttc 가 매 틱 차선변경을
        취소해 **추월이 구조적으로 불가능**하다 (폐루프 시뮬: 시작→취소 무한 반복,
        전이 진행도 0). 목표 차로가 비었음은 planner 가 이미 확인했고, 나머지
        모든 객체에는 TTC 방어가 그대로 살아 있다.
        """
        pl = self.planner
        if pl is None:
            return None
        # 면제는 **추월 기동 전체**(나가기 → 옆을 지나기 → 복귀)에 걸린다.
        # 전이 중에만 걸면 차선을 넘은 직후 옆에 나란히 선 순간 다시 TTC 대상이
        # 되어 비상제동이 걸리고 그 자리에 멈춘다 (시뮬: 618 m 에서 E_STOP 반복).
        # planner 가 복귀를 마치면 _overtake 가 None 이 되어 면제도 끝난다.
        ov = getattr(pl, '_overtake', None)
        return ov.get('blocker_id') if ov is not None else None

    def apply(self, world: WorldState, decision: Decision) -> Decision:
        """
        **최소 주행 루프 단계**: 상수속도 상한만 건다.

        SPEC §3.5 의 6개 가드는 아직 붙이지 않았다. Planner 가 법규 로직을
        갖게 되는 시점에 아래 _clamp_* 들을 순서대로 활성화할 것.
        """
        fired: dict = {}

        # 디버그 상수속도 캡: debug.enabled 이고 값이 양수일 때만.
        # (기본 주행에서는 planner 의 _speed_candidates 가 v_target 을 정한다.
        #  이 캡이 무조건 걸리면 const_speed_kph=0 일 때 v_target 이 0 으로 눌린다.)
        dbg = self.cfg['debug']
        if dbg.get('enabled') and float(dbg['const_speed_kph']) > 0.0:
            v_cap = float(dbg['const_speed_kph']) / 3.6
            v_new = max(0.0, min(decision.v_target, v_cap))
            if v_new != decision.v_target:
                fired['const_cap'] = [decision.v_target, v_new]
                decision.v_target = v_new

        # ── 종방향 ──────────────────────────────────────────────────────
        self._clamp_speed_limit(world, decision, fired)
        self._no_stop_in_crosswalk(world, decision, fired)

        # ── 횡방향(path) ────────────────────────────────────────────────
        # 순서가 중요하다. 셋 다 _path_side / _revert_to_current_lane 을 공유하므로
        # **더 무거운 위반부터** 보고, 앞에서 이미 되돌렸으면 뒤는 건너뛴다
        # (같은 틱에 두 번 되돌리면 로그만 지저분해지고 결과는 같다).
        #   1) 중앙선 침범  — 반대 통행방향. 가장 무겁다.
        #   2) 실선 차로변경 — 같은 방향이지만 실선.
        #   3) 차로 이탈 복귀 — path 가 아니라 **자차 위치**(t_off) 기준. 별개 신호라
        #      앞의 둘과 겹치지 않지만, 전이 중에는 정상이므로 LANE_CHANGE 예외.
        self._forbid_center_crossing(world, decision, fired)
        if 'center_crossing' not in fired:
            self._forbid_illegal_lane_change(world, decision, fired)
        self._abort_lane_change_on_ttc(world, decision, fired)
        if 'center_crossing' not in fired and 'solid_line_lane_change' not in fired:
            self._pull_back_to_lane(world, decision, fired)

        # 비상제동은 **맨 마지막**이다 — 다른 가드가 v_target 을 되올리지 못하게.
        self._emergency_brake(world, decision, fired)

        decision.reasons['shield'] = fired
        return decision

    # ── 경로를 현재 차로로 되돌리는 공통 처리 ────────────────────────────
    def _revert_to_current_lane(self, world: WorldState, decision: Decision) -> bool:
        """path 를 현재 차로 중심선으로 교체. 교체했으면 True."""
        if world.ego.lane is None:
            return False
        d = self.cfg['debug']
        pts = self.lg.points_ahead(world.ego.lane, world.ego.s,
                                   dist=float(d['path_dist_m']),
                                   step=float(d['path_step_m']))
        if len(pts) < 2:
            return False
        decision.path = [(float(x), float(y)) for x, y in pts]
        return True

    def _path_side(self, world: WorldState, decision: Decision, ahead_m: float = 15.0):
        """
        path 가 **현재 차로 중심선** 기준 어느 쪽으로 벗어나 있는지 [m] (좌 +).
        차선변경 판정용이라 자차 바로 앞이 아니라 조금 앞을 본다.

        자차 헤딩 프레임으로 재면 안 된다 — 우측 차선변경 중에는 차체가 우로
        틀어져 전방 경로가 "왼쪽"으로 보이고(실측 2026-08-21 t=11993.7: 우측
        LC1 전이 중 +1.12 m "좌측 실선 침범" 오판 → LC 강제 중단/재시작 →
        경계 1.2 s 정체 + 조향 요동), 곡선로에서는 경로가 차로를 그대로
        따라가도 이탈 -10 m 로 보인다(1702 우회전 연결로에서 수십 틱 연속
        오발화). 중심선(successor 로 이어붙인 폴리라인)에 투영해서 잰다.
        """
        if not decision.path or world.ego.lane is None:
            return 0.0
        e = world.ego
        # ahead_m 앞의 경로 점
        acc = 0.0
        prev = (e.x, e.y)
        px, py = decision.path[-1]
        for qx, qy in decision.path:
            acc += math.hypot(qx - prev[0], qy - prev[1])
            prev = (qx, qy)
            if acc >= ahead_m:
                px, py = qx, qy
                break
        # 현재 차로 중심선 (섹션이 짧으므로 successor 로 이어붙인다)
        cl = self.lg.points_ahead(e.lane, e.s, dist=ahead_m + 10.0, step=1.0)
        if len(cl) < 2:
            return 0.0
        best = None
        for (ax, ay), (bx, by) in zip(cl[:-1], cl[1:]):
            vx, vy = bx - ax, by - ay
            L2 = vx * vx + vy * vy
            if L2 < 1e-12:
                continue
            u = min(1.0, max(0.0, ((px - ax) * vx + (py - ay) * vy) / L2))
            cx, cy = ax + u * vx, ay + u * vy
            dd = math.hypot(px - cx, py - cy)
            if best is None or dd < best[0]:
                cross = vx * (py - ay) - vy * (px - ax)   # >0 이면 왼쪽
                best = (dd, math.copysign(dd, cross))
        return best[1] if best else 0.0

    # 1 ────────────────────────────────────────────────────────────────────
    def _clamp_speed_limit(self, world: WorldState, d: Decision, fired: dict) -> None:
        """
        v_target 을 **법정 제한속도 − `speed.margin_kph`** 로 무조건 자른다.

        planner 의 `limit` 후보가 이미 같은 값을 내지만, Shield 는 planner 를
        신뢰하지 않는다 (SPEC §3.5). 여기서 발동한다는 것은 planner 쪽 버그다.
        """
        if world.speed_limit is None or world.speed_limit <= 0:
            return
        margin = float(self.cfg['speed']['margin_kph']) / 3.6
        cap = max(0.0, float(world.speed_limit) - margin)
        if d.v_target > cap + 1e-6:
            fired['speed_cap'] = [round(d.v_target, 2), round(cap, 2)]
            d.v_target = cap


    # 2 ────────────────────────────────────────────────────────────────────
    def _forbid_illegal_lane_change(self, world: WorldState, d: Decision,
                                    fired: dict) -> None:
        """
        현재 지점이 실선인데 path 가 옆 차로로 벗어나면 현재 차로 path 로 교체.
        (채점: 실선 차로변경 금지 S2.2.05)

        Planner 가 창(window)을 잘못 계산하거나 버그로 실선에서 넘어가려 해도
        여기서 잘린다. Shield 는 Planner 를 신뢰하지 않는다.
        """
        if world.ego.lane is None or not d.path:
            return
        off = self._path_side(world, d)
        thr = float(self.cfg['shield'].get('lane_side_m', 1.0))
        if abs(off) < thr:
            return                       # 차로 안이다

        side = 'left' if off > 0 else 'right'
        if self.lg.lane_change_ok(world.ego.lane, world.ego.s, side):
            return                       # 점선 — 허용

        if self._revert_to_current_lane(world, d):
            fired['solid_line_lane_change'] = f'{side} 실선, path 이탈 {off:+.2f} m'
            if self.planner is not None:
                self.planner.abort_lane_change('실선 구간')

    def _abort_lane_change_on_ttc(self, world: WorldState, d: Decision,
                                  fired: dict) -> None:
        """
        차선변경 중 TTC 위험이 잡히면 중단하고 원래 차로로 되돌린다.

        옆 차로로 넘어가는 도중이 가장 취약하다. 목표 차로 차량이 접근하면
        끝까지 밀고 가기보다 원래 차로로 돌아오는 편이 안전하고 감점도 적다.
        """
        if d.state != 'LANE_CHANGE':
            return
        warn_s = float(self.cfg['ttc']['warn_s'])
        exempt = self._overtake_exempt()
        risky = [o for o in world.objects if o.ttc < warn_s and o.id != exempt]
        if not risky:
            return
        worst = min(risky, key=lambda o: o.ttc)
        if self._revert_to_current_lane(world, d):
            fired['lc_abort_ttc'] = f'id={worst.id} ttc={worst.ttc:.1f}s'
            d.state = 'FOLLOW'
            d.turn_signal = 0
            if self.planner is not None:
                self.planner.abort_lane_change(f'TTC {worst.ttc:.1f}s')

    # 3 ────────────────────────────────────────────────────────────────────
    def _forbid_center_crossing(self, world: WorldState, d: Decision, fired: dict) -> None:
        """
        경로가 **반대 통행방향** 쪽으로 넘어가면 현재 차로로 되돌린다.

        판정 근거는 `world.left_is_center` 다 — 좌측에 같은 방향 주행차로가 없다는
        뜻이고(lanegraph `left_nb is None`), 그쪽으로 넘어가는 것은 중앙선 침범
        또는 도로 이탈이다. 노면표시 데이터가 없어도 성립하는 기하 근거라
        실선 판정(`lane_change_ok`)보다 강하다.

        2026-08-23 19:56 런: 저속 차선변경이 발산해 차로 id −3 → +5(반대편 차선)로
        넘어갔는데 shield 가 한 번도 발동하지 않았다 — 이 가드가 스텁이었다.
        """
        if world.ego.lane is None or not d.path or not world.left_is_center:
            return
        thr = float(self.cfg['shield'].get('lane_side_m', 1.0))
        off = self._path_side(world, d)
        if off <= thr:                       # 좌(+)로 넘어갈 때만
            return
        if self._revert_to_current_lane(world, d):
            fired['center_crossing'] = f'좌측이 중앙선인데 path {off:+.2f} m'
            if self.planner is not None:
                self.planner.abort_lane_change('중앙선 침범')


    # 4 ────────────────────────────────────────────────────────────────────
    def _emergency_brake(self, world: WorldState, d: Decision, fired: dict) -> None:
        """
        min TTC < `ttc.emergency_s` → v_target = 0, state = E_STOP.

        accel = `speed.a_emergency` 와 저크 제한 해제는 Control 이 state 를 보고
        건다 — Decision 에 accel 필드가 없기 때문이다(단방향 파이프라인).

        대상은 **내 경로 위 객체(on_route) + 차로 진입 예측(will_enter_lane)** 뿐이다.
        옆 차로를 스쳐 지나가는 객체까지 세면 멀쩡한 주행에서 급제동이 걸린다.

        해제는 TTC 가 `ttc.brake_s`(2.5) 를 넘겨 회복했을 때다. 발동 문턱(1.5)과
        다른 값을 쓰는 히스테리시스라, 경계에서 TTC 가 떨려도 제동이 껌뻑이지 않는다.

        이건 **최후 방어**다. 정상 감속(정지선/횡단보도 보행자/선행차)이 전부 실패해
        실제로 충돌이 임박했을 때만 걸린다.
        """
        t = self.cfg['ttc']
        emergency_s = float(t['emergency_s'])
        brake_s = float(t['brake_s'])

        exempt = self._overtake_exempt()
        cand = [o for o in world.objects
                if (o.on_route or o.will_enter_lane) and o.id != exempt]
        worst = min(cand, key=lambda o: o.ttc) if cand else None
        ttc = worst.ttc if worst is not None else math.inf

        if self._estop:
            if ttc > brake_s:
                self._estop = False          # 회복
        elif ttc < emergency_s:
            self._estop = True

        if not self._estop:
            return

        why = (f'id={worst.id} ttc={worst.ttc:.2f}s' if worst is not None
               else f'회복 대기 (ttc {ttc:.2f}s <= {brake_s})')
        fired['emergency_brake'] = why
        d.v_target = 0.0
        d.state = 'E_STOP'

    # 5 ────────────────────────────────────────────────────────────────────
    def _no_stop_in_crosswalk(self, world: WorldState, d: Decision, fired: dict) -> None:
        """
        **정지 예상 지점이 횡단보도 구간 안이면** 목표를 구간 앞으로 당긴다.
        (채점: S6.3.03 횡단보도 정차 금지)

        v_target 이 이미 0 근처(= 세우려는 중)일 때만 본다. 현재 속도로 a_comf
        감속했을 때 **앞범퍼**가 멈출 지점이 [횡단보도 시작, 끝] 안이면, 시작
        지점 앞에 서도록 상한을 더 낮춘다 — 늦게가 아니라 **일찍** 세운다.
        """
        if not world.ahead or d.v_target > 0.5:
            return
        cw = next((a for a in world.ahead if a.kind == 'crosswalk'), None)
        if cw is None:
            return
        s0 = float(cw.data.get('s0', cw.dist))
        s1 = float(cw.data.get('s1', cw.dist))
        if s1 <= s0:
            return
        sp = self.cfg['speed']
        vh = self.cfg['vehicle']
        front = float(vh['wheelbase']) + float(vh.get('front_overhang_m', 0.855))
        a = max(1e-3, float(sp['a_comf']))
        v = max(0.0, world.ego.speed)
        stop_at = v * v / (2.0 * a) + front           # 앞범퍼 기준 정지 예상 지점
        if not (s0 <= stop_at <= s1):
            return
        gap = float(sp['stop_gap_m']) + front
        room = s0 - gap
        v_new = math.sqrt(max(0.0, 2.0 * a * room)) if room > 0 else 0.0
        if v_new < d.v_target:
            fired['no_stop_in_crosswalk'] = [round(stop_at, 2), round(s0, 2), round(s1, 2)]
            d.v_target = v_new


    # 6 ────────────────────────────────────────────────────────────────────
    def _pull_back_to_lane(self, world: WorldState, d: Decision, fired: dict) -> None:
        """
        |t_off| > 차로폭/2 − `shield.edge_margin_m` 이면 복귀를 우선한다.
        (채점: S2.1.01 차로 유지 / S2.1.03 보도 침범 금지)

        위의 두 가드는 **경로**가 어디로 가는지를 보지만, 이건 **자차가 지금 어디
        있는지**(t_off)를 본다. 신호가 달라 중복이 아니다.
        **차선변경 전이 중에는 t_off 가 크게 나오는 것이 정상**이므로 제외한다.
        """
        if world.ego.lane is None or not d.path or d.state == 'LANE_CHANGE':
            return
        try:
            w = float(self.lg.width_at(world.ego.lane, world.ego.s))
        except Exception:                              # noqa: BLE001
            w = 3.5
        thr = max(0.2, 0.5 * w - float(self.cfg['shield']['edge_margin_m']))
        if abs(world.ego.t_off) <= thr:
            return
        if self._revert_to_current_lane(world, d):
            fired['pull_back'] = f't_off {world.ego.t_off:+.2f} m > {thr:.2f}'

