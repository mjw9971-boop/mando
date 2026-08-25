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
