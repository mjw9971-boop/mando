"""테스트에서 vtd_adapter / tools 를 import 할 수 있게 경로를 잡는다."""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (str(ROOT), str(ROOT / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

PARAMS_YAML = str(ROOT / 'config' / 'params.yaml')


def mk_tick(t=0.0, speed=0.0, x=0.0, y=0.0, yaw=0.0, s=0.0, route_s=0.0, t_off=0.0,
            lane=None, front_m=None, ctrl=None, lights=(), left_is_center=False,
            reset=False, valid=True, summ=None, objects=None, raw_objects=None,
            v_target=0.0, turn_signal=0, reasons=None, flags=None,
            speed_limit=None, school_zone=False):
    """score.py 검출기용 합성 틱 (logger.write 스키마의 부분집합 — 검출기가
    읽는 필드만). 각 검출기 테스트가 공유한다."""
    fl = dict(flags or {})
    if ctrl is not None:
        fl['stop_ctrl_ids'] = list(ctrl)
    if reset:
        fl['reset'] = True
    return {
        't': t,
        'raw': {'ego': [x, y, 0.0, yaw, 0.0, 0.0],
                'objects': [list(o) for o in (raw_objects or [])],
                'lights': [[int(a), int(b)] for a, b in lights]},
        'ego': {'x': x, 'y': y, 'yaw': yaw, 'speed': speed, 'accel': 0.0,
                'lane': list(lane) if lane else None, 's': s, 'route_s': route_s,
                't_off': t_off, 'heading_err': 0.0},
        'world': {'valid': valid, 'speed_limit': speed_limit, 'school_zone': school_zone,
                  'left_solid': False, 'right_solid': False,
                  'left_is_center': left_is_center, 'light': None, 'n_obj': 0,
                  'flags': fl, 'ahead': [], 'summ': dict(summ or {}),
                  'stop_line_front_m': front_m},
        'objects': list(objects or []),
        'decision': {'state': 'none', 'v_target': v_target, 'turn_signal': turn_signal,
                     'n_path': 0, 'reasons': dict(reasons or {})},
        'cmd': {'steering': 0.0, 'accel': 0.0, 'turn_signal': turn_signal},
    }
