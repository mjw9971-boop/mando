"""합성 1109 B 프레임 → 파싱 왕복 검증 (SPEC §1.1 / §1.2)."""
import struct

import pytest

from vtd_adapter.comm import (CTRL_SIZE, FRAME_SIZE, OBJ_BASE, OBJ_SIZE, TL_BASE,
                      build_frame, pack_command, parse)
from vtd_adapter.types import Command

EGO = (508.80, -168.29, 42.0, 0.52727822, 0.01, -0.02)


def test_frame_layout_matches_spec():
    """24 + 36*30 + 5 = 1109, 제어는 9 B."""
    assert OBJ_BASE == 24
    assert OBJ_SIZE == 36
    assert TL_BASE == 24 + 36 * 30 == 1104
    assert FRAME_SIZE == 1109
    assert CTRL_SIZE == 9


def test_roundtrip_ego_objects_lights():
    objs = [
        (7, 100.0, 20.0, 0.0, 0.1, 5.0, 4.5, 1.9, 1.5),      # 차량
        (12, 30.0, -3.0, 0.0, 1.5, 1.2, 0.6, 0.6, 1.7),      # 보행자
    ]
    pkt = parse(build_frame(EGO, objs, [(3, 1)]), t_recv=1.0)

    assert pkt.t_recv == 1.0
    assert pkt.ego == pytest.approx(EGO, rel=1e-6)
    assert [o[0] for o in pkt.objects] == [7, 12]
    assert pkt.objects[1][6] == pytest.approx(0.6)           # length
    assert pkt.lights == [(3, 1)]


def test_empty_object_slots_are_skipped():
    """빈 슬롯은 전 필드 0 → id == 0 이면 스킵 (SPEC §1.1)."""
    objs = [(0, 0, 0, 0, 0, 0, 0, 0, 0)] * 5 + [(9, 1.0, 2.0, 0, 0, 0, 4.0, 1.8, 1.4)]
    pkt = parse(build_frame(EGO, objs, [(1, 3)]))
    assert [o[0] for o in pkt.objects] == [9]


def test_absent_traffic_light_is_dropped():
    """신호등이 없으면 (0, 0) 이 온다 → lights 비어야 한다."""
    pkt = parse(build_frame(EGO, [], [(0, 0)]))
    assert pkt.lights == []


def test_all_30_object_slots_used():
    objs = [(i + 1, float(i), 0.0, 0.0, 0.0, 1.0, 4.0, 1.8, 1.4) for i in range(30)]
    pkt = parse(build_frame(EGO, objs, [(2, 2)]))
    assert len(pkt.objects) == 30


def test_short_frame_rejected():
    with pytest.raises(ValueError):
        parse(b'\x00' * (FRAME_SIZE - 1))


def test_pack_command_applies_steer_sign():
    """SPEC §1.2: 주최 예제가 steer_out = -steer 로 뒤집는다."""
    raw = pack_command(Command(steering=0.2, accel=-1.5, turn_signal=2), steer_sign=-1.0)
    assert len(raw) == CTRL_SIZE
    steer, accel, sig = struct.unpack('<ffB', raw)
    assert steer == pytest.approx(-0.2, rel=1e-6)
    assert accel == pytest.approx(-1.5, rel=1e-6)
    assert sig == 2

    raw_pos = pack_command(Command(0.2, -1.5, 2), steer_sign=1.0)
    assert struct.unpack('<ffB', raw_pos)[0] == pytest.approx(0.2, rel=1e-6)


def test_invalid_turn_signal_falls_back_to_off():
    assert struct.unpack('<ffB', pack_command(Command(0.0, 0.0, 7)))[2] == 0
