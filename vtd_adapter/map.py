"""
carla.Map / carla.Waypoint 흉내 — lanegraph 위에 CARLA 관례 좌표로 얹는다.

VtdWaypoint 는 (LaneKey, s) 를 감싼다. 내주는 transform 은 CARLA 프레임
(frame.py), lanegraph 조회는 내부에서 VTD 프레임으로 되돌려서 한다.

get_left_lane/get_right_lane 은 **주행방향 상대** 개념이라 미러 변환과 무관하게
lanegraph 의 left/right neighbor 그대로다 (phase0 §0-3).
"""
from __future__ import annotations

from . import frame
from .carla_types import Location, Rotation, Transform
from .lanegraph import LaneGraph, LaneKey


class VtdWaypoint:
    """carla.Waypoint 의 PDM-Lite 사용 표면: transform / lane_width / is_junction /
    road_id / lane_id / next / previous / get_left_lane / get_right_lane."""

    __slots__ = ('_lg', 'key', 's', '_transform')

    def __init__(self, lg: LaneGraph, key: LaneKey, s: float) -> None:
        self._lg = lg
        self.key = key
        self.s = float(max(0.0, min(s, lg.length(key))))
        self._transform: Transform | None = None

    # ── 속성 ──────────────────────────────────────────────────────────────
    @property
    def transform(self) -> Transform:
        if self._transform is None:
            x, y, z, hdg = self._lg.point_at(self.key, self.s)
            cx, cy = frame.to_carla_xy(x, y)
            self._transform = Transform(Location(cx, cy, z),
                                        Rotation(yaw=frame.to_carla_yaw_deg(hdg)))
        return self._transform

    @property
    def lane_width(self) -> float:
        return float(self._lg.width_at(self.key, self.s))

    @property
    def is_junction(self) -> bool:
        return self._lg.lanes[self.key]['junction'] != -1

    @property
    def road_id(self) -> int:
        return int(self.key[0])

    @property
    def section_id(self) -> int:
        return int(self.key[1])

    @property
    def lane_id(self) -> int:
        return int(self.key[2])

    # ── 이동 ──────────────────────────────────────────────────────────────
    def next(self, distance: float) -> list['VtdWaypoint']:
        """distance[m] 앞의 waypoint 목록 (분기마다 하나). CARLA 와 같은 규약:
        차로 끝을 넘으면 successor 로 이어 간다."""
        out: list[VtdWaypoint] = []
        self._walk(self.key, self.s + float(distance), out, forward=True, depth=0)
        return out

    def previous(self, distance: float) -> list['VtdWaypoint']:
        out: list[VtdWaypoint] = []
        self._walk(self.key, self.s - float(distance), out, forward=False, depth=0)
        return out

    def _walk(self, key: LaneKey, s: float, out: list, forward: bool, depth: int) -> None:
        L = self._lg.length(key)
        if 0.0 <= s <= L:
            out.append(VtdWaypoint(self._lg, key, s))
            return
        if depth >= 8:                      # 순환 successor 방어
            return
        if forward:
            links = self._lg.successors(key)
            rem = s - L
            for nk in links:
                self._walk(nk, rem, out, forward, depth + 1)
        else:
            links = self._lg.predecessors(key)
            for nk in links:
                self._walk(nk, self._lg.length(nk) + s, out, forward, depth + 1)
        if not links:                       # 막다른 차로 — 끝점을 돌려준다
            out.append(VtdWaypoint(self._lg, key, max(0.0, min(s, L))))

    # ── 이웃 차로 (주행방향 상대) ─────────────────────────────────────────
    def get_left_lane(self) -> 'VtdWaypoint | None':
        nb = self._lg.neighbor(self.key, 'left')
        if nb is None:
            return None
        return VtdWaypoint(self._lg, nb, min(self.s, self._lg.length(nb)))

    def get_right_lane(self) -> 'VtdWaypoint | None':
        nb = self._lg.neighbor(self.key, 'right')
        if nb is None:
            return None
        return VtdWaypoint(self._lg, nb, min(self.s, self._lg.length(nb)))

    def __repr__(self) -> str:
        return f'VtdWaypoint({self.key}, s={self.s:.2f})'


class VtdMap:
    """carla.Map 의 PDM-Lite 사용 표면: get_waypoint(location)."""

    def __init__(self, lg: LaneGraph) -> None:
        self.lg = lg
        self.name = 'HL_FMA_VTD_LivingLab'

    def get_waypoint(self, location: Location) -> VtdWaypoint:
        """CARLA 관례 좌표 → 가장 가까운 driving 차로 waypoint.

        CARLA get_waypoint(project_to_road=True) 처럼 **항상** 무언가를 돌려준다 —
        locate 가 기본 반경(8 m)에서 실패하면 반경 제한 없이 다시 찾는다.
        """
        vx, vy = frame.from_carla_xy(location.x, location.y)
        m = self.lg.locate(vx, vy)
        if m is None:
            m = self.lg.locate(vx, vy, max_dist=1e9, k=1)
        return VtdWaypoint(self.lg, m.lane, m.s)
