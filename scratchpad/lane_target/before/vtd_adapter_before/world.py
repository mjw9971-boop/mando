"""
carla.World 흉내 — 9910 객체 목록을 VtdActor 로 유지·코스팅한다.

객체 유지 규칙은 기존 perception._track_objects 의 검증 로직 이식:
  공식 확인 — 객체는 수평거리 80 m 이내만, 가까운 순 최대 30개만 온다.
  즉 **목록에서 사라진 것이 소멸을 뜻하지 않는다.**
    · 마지막 거리가 80 m 근처였다  -> 범위 밖으로 나간 것. 즉시 버린다.
    · 목록이 30개로 꽉 차 있었다    -> 더 가까운 객체에 밀려난 것. 아직 있다.
    · 둘 다 아니다                  -> 진짜 소멸(가림 등). coast_s 동안 등속 외삽 유지.
"""
from __future__ import annotations

import math

from . import frame
from .actor import VtdActor, VtdEgo
from .comm import OBJ_COUNT
from .types import RawPacket


class DebugHelper:
    """world.debug.* — 렌더링 없음, 전부 no-op."""

    def draw_point(self, *a, **kw) -> None:
        pass

    def draw_box(self, *a, **kw) -> None:
        pass

    def draw_string(self, *a, **kw) -> None:
        pass


class ActorList(list):
    """carla.ActorList 의 filter 만 흉내낸다 (get_actors 반환용)."""

    def filter(self, pattern: str) -> 'ActorList':
        pat = pattern.strip('*')
        return ActorList(a for a in self if pat in a.type_id)


def classify(length: float, width: float, height: float, speed: float) -> str:
    """
    크기로 객체 종류 판정 (SPEC §1.1 — 타입 필드가 없다).
      보행자: width 0.5–0.8, height 1.5–2.0, length < 1.0
      차량  : length > 3.0
      그 외 + speed≈0: 정적 장애물
    """
    if length > 3.0:
        return 'vehicle'
    if 0.5 <= width <= 0.8 and 1.5 <= height <= 2.0 and length < 1.0:
        return 'pedestrian'
    if abs(speed) < 0.2:
        return 'obstacle'
    return 'unknown'


class VtdWorld:
    """매 틱 update(pkt, ego_state) 로 갱신되는 액터 컨테이너."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.debug = DebugHelper()
        self.ego = VtdEgo(cfg['vehicle'])
        # id -> (VtdActor, 마지막 수신 시각, 마지막 자차거리)
        self._tracks: dict[int, tuple] = {}
        self.flags: dict = {}                 # 관측 카운터 (로그 flags 에 합쳐 쓸 수 있음)

    def clear(self) -> None:
        """리셋(courseRespawn) 시 호출 — 자차가 순간이동하면 상대량이 전부 어긋난다."""
        self._tracks.clear()

    def update(self, pkt: RawPacket, ego_state) -> None:
        """9910 프레임 → 액터 갱신. ego_state 는 EgoTracker 의 EgoState (VTD 프레임)."""
        # 자차 (VTD → CARLA 변환은 여기 한 번뿐)
        ex, ey = frame.to_carla_xy(ego_state.x, ego_state.y)
        self.ego.update_from_ego(ego_state, frame.to_carla_yaw_deg(ego_state.yaw), ex, ey)

        p = self.cfg['percep']
        gt_range = float(p['gt_range_m'])
        margin = float(p['range_margin_m'])
        coast_s = float(p['coast_s'])
        t = pkt.t_recv

        flags: dict = {'obj_n': len(pkt.objects)}
        seen: set[int] = set()
        max_dist = 0.0

        for (oid, ox, oy, oz, ohead, ospeed, olen, owid, ohei) in pkt.objects:
            oid = int(oid)
            seen.add(oid)
            dist = math.hypot(ox - ego_state.x, oy - ego_state.y)
            max_dist = max(max_dist, dist)
            cls = classify(olen, owid, ohei, ospeed)
            actor = self._tracks[oid][0] if oid in self._tracks else VtdActor(oid, cls)
            actor.cls = cls
            actor.type_id = ('walker.vtd.pedestrian' if cls == 'pedestrian'
                             else 'vehicle.vtd.object')
            cx, cy = frame.to_carla_xy(ox, oy)
            actor.update(cx, cy, oz, frame.to_carla_yaw_deg(ohead), ospeed,
                         olen, owid, ohei)
            actor.age = 0.0
            actor.coasting = False
            self._tracks[oid] = (actor, t, dist)

        # ── 이번 틱에 안 온 객체 (모듈 docstring 의 3분법) ────────────────
        list_full = len(pkt.objects) >= OBJ_COUNT
        dropped_far = dropped_lost = coasted = 0
        for oid, (actor, last_t, last_d) in list(self._tracks.items()):
            if oid in seen:
                continue
            age = t - last_t
            near_range_edge = last_d >= gt_range - margin
            if age > coast_s or (near_range_edge and not list_full):
                del self._tracks[oid]
                if near_range_edge:
                    dropped_far += 1
                else:
                    dropped_lost += 1
                continue
            # 등속 외삽 유지 (CARLA 프레임에서 그대로 굴린다)
            yaw = math.radians(actor.yaw_deg)
            actor.x += actor.speed * math.cos(yaw) * (t - last_t - actor.age)
            actor.y += actor.speed * math.sin(yaw) * (t - last_t - actor.age)
            actor.age = age
            actor.coasting = True
            self._tracks[oid] = (actor, last_t, last_d)
            coasted += 1

        flags['obj_max_dist'] = round(max_dist, 1)
        if list_full:
            flags['obj_list_full'] = True
        if max_dist > gt_range + margin:
            flags['obj_beyond_gt_range'] = round(max_dist, 1)
        if coasted:
            flags['obj_coasting'] = coasted
        if dropped_far:
            flags['obj_left_range'] = dropped_far
        if dropped_lost:
            flags['obj_lost'] = dropped_lost
        self.flags = flags

    # ── CARLA 표면 ────────────────────────────────────────────────────────
    def get_actors(self) -> ActorList:
        """주변 객체 목록 (자차 제외 — CARLA 와 달리 ego 는 히어로로 따로 관리)."""
        return ActorList(a for a, _t, _d in self._tracks.values())

    def get_actor(self, actor_id: int):
        if actor_id == self.ego.id:
            return self.ego
        rec = self._tracks.get(int(actor_id))
        return rec[0] if rec else None
