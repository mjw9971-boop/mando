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

        self._forbid_illegal_lane_change(world, decision, fired)
        self._abort_lane_change_on_ttc(world, decision, fired)

        # TODO: 1) _clamp_speed_limit   (제한속도 절대 초과 금지)
        # TODO: 3) _forbid_center_crossing
        # TODO: 4) _emergency_brake
        # TODO: 5) _no_stop_in_crosswalk
        # TODO: 6) _pull_back_to_lane

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

    @staticmethod
    def _path_side(world: WorldState, decision: Decision, ahead_m: float = 15.0):
        """
        path 가 현재 위치 기준 어느 쪽으로 벗어나 있는지 [m] (좌 +).
        차선변경 판정용이라 자차 바로 앞이 아니라 조금 앞을 본다.
        """
        if not decision.path:
            return 0.0
        e = world.ego
        acc = 0.0
        prev = (e.x, e.y)
        for px, py in decision.path:
            acc += math.hypot(px - prev[0], py - prev[1])
            prev = (px, py)
            if acc >= ahead_m:
                dx, dy = px - e.x, py - e.y
                return -dx * math.sin(e.yaw) + dy * math.cos(e.yaw)
        dx, dy = decision.path[-1][0] - e.x, decision.path[-1][1] - e.y
        return -dx * math.sin(e.yaw) + dy * math.cos(e.yaw)

    # 1 ────────────────────────────────────────────────────────────────────
    def _clamp_speed_limit(self, world: WorldState, d: Decision) -> None:
        """v_target = min(v_target, speed_limit - SPEED_MARGIN). 절대 초과 금지."""
        # TODO: 구현
        raise NotImplementedError('shield._clamp_speed_limit')

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
        risky = [o for o in world.objects if o.ttc < warn_s]
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
    def _forbid_center_crossing(self, world: WorldState, d: Decision) -> None:
        """`left_is_center` 인데 path 가 좌측으로 벗어나면 교체. (채점: 중앙선 침범)"""
        # TODO: 구현
        raise NotImplementedError('shield._forbid_center_crossing')

    # 4 ────────────────────────────────────────────────────────────────────
    def _emergency_brake(self, world: WorldState, d: Decision) -> None:
        """min TTC < `ttc.emergency_s` → v_target = 0, accel = `speed.a_emergency`.
        이때는 저크 제한을 풀어야 하므로 state 를 E_STOP 으로 바꾼다."""
        # TODO: 구현
        raise NotImplementedError('shield._emergency_brake')

    # 5 ────────────────────────────────────────────────────────────────────
    def _no_stop_in_crosswalk(self, world: WorldState, d: Decision) -> None:
        """정지점이 횡단보도 구간 안이면 구간 **앞으로** 당긴다. (채점: 횡단보도 정차 금지)"""
        # TODO: 구현
        raise NotImplementedError('shield._no_stop_in_crosswalk')

    # 6 ────────────────────────────────────────────────────────────────────
    def _pull_back_to_lane(self, world: WorldState, d: Decision) -> None:
        """|t_off| > lane_width/2 - `shield.edge_margin_m` 이면 복귀를 우선한다.
        (채점: 차로 유지 / 보도 침범 금지)"""
        # TODO: 구현
        raise NotImplementedError('shield._pull_back_to_lane')
