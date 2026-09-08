"""테스트에서 vtd_adapter / tools 를 import 할 수 있게 경로를 잡는다."""
import contextlib
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


# ── 전역 DP 의 짝 대조를 테스트에서는 끈다 ──────────────────────────────────
# route.dp_compare_enable 은 **운영 기본이 true** 다 (대회 당일 짝 대조 WARN 을
# 봐야 한다). 그런데 대조는 경로를 한 벌 더 짓고 폴리라인까지 재샘플하므로
# 테스트 전체가 두 배 가까이 느려진다 (실측 525 s → 303 s).
# 대조 자체를 검증하는 테스트는 BR._DP_CFG 를 직접 세팅하므로 이 기본값을 덮는다.
import pytest                                                    # noqa: E402


@pytest.fixture(autouse=True)
def _dp_compare_off():
    try:
        import build_route as BR
    except Exception:                                            # noqa: BLE001
        yield
        return
    old = BR._DP_CFG
    BR._DP_CFG = BR.dp_cfg()._replace(compare=False)
    try:
        yield
    finally:
        BR._DP_CFG = old


# ── 옛 금지 임계(5.65 m)를 명시적으로 고정하는 헬퍼 ─────────────────────────
# 2026-09-08 회전 금지 완화로 route.banned_r_min_m 기본이 3.0 이 됐고, 금지
# 연결로가 38개 → 4개로 줄었다 (docs/BACKLOG.md B-29). "금지 연결로가 있으면
# 어떻게 되나" 를 보는 테스트들은 그 **금지 자체가 전제**라, 기본값을 읽으면
# 전제가 사라져 깨진다.
#
# 기본값을 되돌리는 게 아니라 **사본에서 임계를 명시적으로 올려** 본다 —
# params 기본값이 또 움직여도 이 검사들은 안 깨진다 (CLAUDE.md 의 off_cfg() /
# a1_cfg() 와 같은 처리 원칙).
@contextlib.contextmanager
def banned_r_min(thr):
    """route.banned_r_min_m 을 이 블록 안에서만 thr 로 둔다."""
    import build_route as BR
    BR.route_cfg(reload=True)
    old = dict(BR._ROUTE_CFG)
    BR._ROUTE_CFG['banned_r_min_m'] = float(thr)
    try:
        yield
    finally:
        BR._ROUTE_CFG = old


# 옛 임계 = 기하 최소회전반경 × vehicle.min_turn_margin (2.944/tan(0.48) ≈ 5.65).
# 숫자를 박지 않고 계산해 둔다 — vehicle 제원이 바뀌면 같이 따라간다.
def legacy_banned_r_min():
    import build_route as BR
    return BR.tight_turn_r_m()
