"""
모든 튜닝 파라미터의 **단일 출처**.

여기 DEFAULTS 가 기본값이고 config/params.yaml 이 그 위를 덮는다. ROS 노드와
ROS 없이 도는 경로(tools/run_standalone.py, 단위 테스트)가 같은 파일을 읽는다.
설정 파일을 두 벌로 두면 반드시 어긋나므로 config.yaml 은 없앴다.

core 는 cfg['comm']['send_hz'] 같은 중첩 dict 를 받는다. ROS 파라미터는 평평하므로
`comm.send_hz` 로 선언했다가 읽을 때 다시 중첩으로 조립한다.
params.yaml 에서 일부만 덮어써도 나머지는 아래 기본값이 쓰인다.
"""
from __future__ import annotations

from typing import Any

# 근거는 AGENT_SPEC §5.
DEFAULTS: dict[str, Any] = {
    'comm': {'host': '127.0.0.1', 'port': 9910, 'send_hz': 20, 'watchdog_s': 1.0,
             'steer_sign': 1.0, 'connect_retry_s': 1.0, 'recv_bufsize': 65536},
    'vehicle': {'wheelbase': 2.944, 'max_steer': 0.48, 'length': 4.848,
                'width': 1.886, 'height': 1.507},
    'speed': {'margin_kph': 3.0, 'a_comf': 1.5, 'a_max': 2.0, 'a_min': -6.0,
              'a_emergency': -8.0, 'jerk_max': 2.0, 'a_lat_max': 2.0,
              'stop_gap_m': 1.0, 'a_hold': -1.0},
    'caps_kph': {'school_zone': 28.0, 'crosswalk': 25.0, 'junction': 30.0, 'blind': 25.0},
    'signal': {'yellow_s': 3.0, 'lead_s': 3.0, 'margin_s': 1.0, 'flash_mode': 'yield'},
    'lead': {'time_headway_s': 2.0, 'min_gap_m': 5.0},
    'ttc': {'warn_s': 4.0, 'brake_s': 2.5, 'emergency_s': 1.5},
    'lane_change': {'back_m': 30.0, 'front_m': 50.0, 'min_window_m': 20.0,
                    # 횡방향 전이 거리 = max(transition_s * v, transition_min_m)
                    'transition_s': 3.0, 'transition_min_m': 20.0,
                    # 완료 판정
                    'done_t_off_m': 0.3, 'done_heading_deg': 5.0},
    'cross': {'margin_s': 2.0},
    'percep': {'horizon_m': 200.0, 'coast_s': 1.5, 'jump_m': 2.0, 'speed_lpf': 0.3,
               # VTD 대회 브릿지의 courseRespawn 감지.
               # respawnPoseToleranceM=1.5 라 실제 리셋 이동량은 3 m 남짓이다.
               'reset_route_s_drop_m': 5.0,
               # 리셋 판정은 거리 단독으로 하지 않는다. 파이프라인이 멈췄다가
               # 재개되면 차는 정상 속도로 크게 움직이는데 그건 리스폰이 아니다.
               'stall_dt_s': 0.2,          # 이보다 긴 틱 간격은 '스톨'로 분류
               'reset_speed_factor': 3.0,  # 점프속도가 실제속도의 이 배를 넘어야
               'reset_abs_speed': 30.0,    # 그리고 이 절대상한도 넘어야 리스폰
               'ped_extrapolate_s': 2.0,
               # 공식 확인: 객체는 수평거리 80 m 이내만, 가까운 순 최대 30개
               'gt_range_m': 80.0, 'range_margin_m': 5.0},
    'control': {'kp': 0.8, 'ki': 0.15, 'k_ld': 0.8, 'ld_min': 5.0, 'ld_max': 20.0,
                'steer_rate_max': 1.0},
    # lane_side_m: path 가 이만큼 옆으로 벗어나면 차선변경 시도로 본다
    'shield': {'edge_margin_m': 0.3, 'lane_side_m': 1.0},
    # path 가 비어 있으면 dir/run_<타임스탬프>.jsonl 로 자동 생성한다.
    'log': {'enabled': True, 'dir': 'logs', 'path': '', 'flush_every': 20},
    'debug': {'const_speed_kph': 20.0, 'path_dist_m': 40.0, 'path_step_m': 0.5,
              'print_hz': 1.0},
    'default_speed_kph': 50.0,
}


# params.yaml 에 함께 써 넣을 키별 근거. 실측으로 확정된 값은 여기에 남긴다.
NOTES: dict[str, list[str]] = {
    'comm.steer_sign': [
        '실측 확정 (2026-08-19, VTD 2025.2): +1.0.',
        '-1.0 이면 직선 구간에서 발산하며 courseRespawn 리셋이 발생한다.',
        'corr(steering, yaw_rate) = -0.93 으로 확인 (logs/run_20260819_185936.jsonl).',
        '뒤집히면 control 의 SteerSignMonitor 가 주행 5초 안에 경고를 띄운다.',
    ],
    'percep.jump_m': [
        'courseRespawn 리셋 감지 문턱. 실제 리셋 이동량이 3.3 m 남짓이라 그보다 낮아야 한다.',
    ],
    'percep.gt_range_m': [
        '공식 확인: 객체는 수평거리 80 m 이내만, 가까운 순 최대 30개.',
    ],
}


def render_yaml() -> str:
    """params.yaml 본문을 만든다 (DEFAULTS 가 단일 출처, NOTES 를 주석으로 붙인다)."""
    out = [
        '# HL FMA 2026 컨트롤러 파라미터 (모든 튜닝 값의 단일 출처)',
        '# 단위: m, m/s, m/s^2, rad. km/h 는 이 파일과 로그 출력에서만.',
        '# 노드가 모두 같은 값을 봐야 하므로 와일드카드로 한 번에 준다.',
        '#',
        '# 이 파일은 nodes/params.py 의 DEFAULTS 로부터 생성한다:',
        '#   python3 -c "import sys;sys.path.insert(0,\'src/hlfma\');'
        'from hlfma.nodes.params import render_yaml;print(render_yaml())" \\',
        '#     > src/hlfma/config/params.yaml',
        '/**:',
        '  ros__parameters:',
        '    graph_path: data/lane_graph.pkl',
        '    route_path: data/route.pkl',
        '',
    ]
    section = None
    for key, val in _flatten(DEFAULTS).items():
        top = key.split('.')[0]
        if top != section:
            out.append(f'    # ── {top} ──')
            section = top
        for line in NOTES.get(key, []):
            out.append(f'    # {line}')
        if isinstance(val, bool):
            v = 'true' if val else 'false'
        elif isinstance(val, str):
            v = f'"{val}"'
        else:
            v = val
        out.append(f'    {key}: {v}')
    out.append('')
    return '\n'.join(out)


def _flatten(d: dict, prefix: str = '') -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f'{prefix}{k}'
        if isinstance(v, dict):
            out.update(_flatten(v, f'{key}.'))
        else:
            out[key] = v
    return out


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, v in flat.items():
        parts = key.split('.')
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


def collect_core_config(node) -> dict[str, Any]:
    """
    노드에 모든 설정 파라미터를 선언하고 core 용 중첩 dict 로 돌려준다.
    이미 선언된 파라미터(노드가 따로 declare 한 것)는 건너뛴다.
    """
    flat = _flatten(DEFAULTS)
    got: dict[str, Any] = {}
    for key, default in flat.items():
        if not node.has_parameter(key):
            node.declare_parameter(key, default)
        got[key] = node.get_parameter(key).value
    return _unflatten(got)


def load_params_yaml(path: str) -> dict[str, Any]:
    """
    params.yaml (ROS 형식) → core 용 중첩 dict.

    ROS 없이 도는 경로(tools/run_standalone.py, 단위 테스트)도 **같은 파일**을
    읽게 해서 설정이 두 벌로 갈라지지 않게 한다.
    """
    import yaml

    with open(path, 'r', encoding='utf-8') as f:
        doc = yaml.safe_load(f) or {}

    flat = dict(_flatten(DEFAULTS))
    for node_key, body in doc.items():
        if not isinstance(body, dict):
            continue
        params = body.get('ros__parameters', {})
        for k, v in params.items():
            if k in flat or k in ('graph_path', 'route_path'):
                flat[k] = v
    return _unflatten(flat)
