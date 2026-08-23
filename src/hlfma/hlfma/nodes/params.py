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
    'comm': {'host': '192.168.10.1', 'port': 9910, 'send_hz': 20, 'watchdog_s': 1.0,
             'steer_sign': 1.0, 'connect_retry_s': 1.0, 'recv_bufsize': 65536,
             # /cmd 가 이 시간 이상 끊기면 vtd_bridge 가 조향을 0 으로 감쇠 재송신
             # (가속은 유지). 2026-08-21: 풀락 조향이 0.32 s 유지돼 도로이탈.
             'hold_decay_s': 0.15},
    'vehicle': {'wheelbase': 2.944, 'max_steer': 0.48, 'length': 4.848,
                # 뒷바퀴축 → 앞범퍼. Ioniq 6 제원: 전장 4855 / 축거 2950 / 앞 오버행 855 /
                # 뒤 오버행 1050 mm (여기 length 4.848 − wheelbase 2.944 = 1.904 와 일치).
                'front_overhang_m': 0.855,
                'width': 1.886, 'height': 1.507},
    # margin_kph: 제한속도에서 항상 빼고 달릴 여유. 감점이 시간보다 비싸다.
    'speed': {'margin_kph': 5.0, 'a_comf': 1.5,
              # 계획용 감속도는 a_comf 보다 낮게 잡는다. a_comf 를 그대로 쓰면
              # 제어 지연분 여유가 없어 정지선을 넘어간다(실측 -1.1 m).
              'a_plan_factor': 0.7,
              # 이 속도 아래로 내려가면 정지 확정(래치). 정지선이 시야를 벗어나도 유지
              'stop_latch_v': 1.5, 'a_max': 2.0, 'a_min': -6.0,
              'a_emergency': -8.0, 'jerk_max': 2.0, 'a_lat_max': 2.0,
              # 정지 목표 선행 보상 [s]: P 제어가 내려가는 목표를 a_plan/kp ≈ 1.3 m/s 늦게
              # 따라가 정지점을 ~2 m 넘긴다. 목표를 v·stop_lag_s 만큼 앞당긴다 (폐루프
              # 시뮬: 0.6 이면 앞범퍼가 정지선 −1.0~−1.2 m, 완주 시간 손실 0).
              # 지도 곡률 이상치 방어: 이웃 중앙값의 이 배수를 넘는 고립 곡률은 무시.
              # 0 이면 방어 끔. (빌드 단계 필터가 1차 방어, 이건 2차 방어)
              'curv_outlier_ratio': 5.0,
              'stop_lag_s': 0.6,
              # 래치 뒤 0 스냅 대신 _approach 곡선을 v=0 까지 연속으로 (스냅은
              # 마지막에 감속도가 튄다 — 2026-08-23 15:48 런에서 -1.29 -> -1.87)
              'stop_continuous': True,
              'stop_gap_m': 1.0, 'a_hold': -1.0},
    # crosswalk 캡은 없다: 채점표의 속도 항목은 S1.1.01 제한속도 / S1.1.02 스쿨존
    # 뿐이고 S6.3.03 은 "횡단보도 정차 금지"(멈추지 말라)다. 25 km/h 캡이
    # 2026-08-21 주행 평균을 19 km/h 로 눌렀다 (완주 필요 평균 36 km/h).
    'caps_kph': {'school_zone': 28.0, 'junction': 30.0, 'blind': 25.0},
    'signal': {'yellow_s': 3.0, 'lead_s': 3.0, 'margin_s': 1.0, 'flash_mode': 'yield',
               # 계획 차선변경은 해당 방향 지시등이 이 시간 이상 연속 점등된 뒤에만 실행
               'lc_lead_min_s': 3.0,
               # RTOR(적신호 우회전): 완전 정지 유지 시간, 서행 통과 속도, 스위치(조직위 답변에 따라)
               'stop_dwell_s': 1.0, 'rtor_speed_kph': 20.0, 'rtor_enabled': True},
    'lead': {'time_headway_s': 2.0, 'min_gap_m': 5.0,
             # ── 정차 차량 추월 (blocked) ──
             # 정차 판정을 이 시간 이상 연속 만족해야 추월에 들어간다(센서 노이즈 방지).
             'blocked_dwell_s': 2.0,
             # 전방 정지선이 이보다 가까우면 신호 대기 행렬일 수 있으므로 추월하지 않는다.
             'blocked_ignore_stopline_m': 30.0,
             # 추월 대상 뒤에서는 조향 여유를 두고 더 멀리 선다.
             'blocked_gap_m': 8.0,
             # **최근에 달리던 차는 추월하지 않는다.** 이 속도를 넘겨 달린 이력이
             # blocked_recent_move_s 안에 있으면 "잠깐 선 것"으로 보고 기다린다.
             # 2_lead_brake(앞차가 15 s 뒤 재출발)에서 정차 2 s 만에 추월이
             # 발동했다 — 앞차는 10.9 s 동안 v=0.000 이라 dwell 은 정상이었고,
             # 구분 근거는 "직전에 달리고 있었는가" 뿐이다.
             # 값은 예상되는 일시정차 시간보다 길어야 한다.
             'blocked_recent_move_v': 2.0, 'blocked_recent_move_s': 20.0},
    'ttc': {'warn_s': 4.0, 'brake_s': 2.5, 'emergency_s': 1.5},
    'lane_change': {'back_m': 30.0, 'front_m': 50.0, 'min_window_m': 20.0,
                    # 횡방향 전이 거리 = max(transition_s * v, transition_min_m)
                    'transition_s': 3.0, 'transition_min_m': 20.0,
                    # 완료 판정
                    'done_t_off_m': 0.3, 'done_heading_deg': 5.0,
                    # 차선변경 최소 속도 [m/s]. 이보다 느리면 시작하지 않고,
                    # 전이 중 이 아래로 떨어지면 중단하고 원 차로로 되돌린다.
                    # 블렌드 진행도가 주행거리에 비례하므로 저속에서는 전이가
                    # 진행되지 않는데 조향만 한쪽으로 고착돼 차가 옆으로 밀린다
                    # (2026-08-23 19:56 런: v=0.84 m/s 에서 시작 → 도로 이탈,
                    #  차로 id -3 → +5 로 반대편 차선 진입).
                    'v_min_mps': 2.0},
    'cross': {'margin_s': 2.0},
    'percep': {'horizon_m': 200.0, 'coast_s': 1.5, 'jump_m': 2.0, 'speed_lpf': 0.3,
               # 속도 추정 슬라이딩 창 [s] (Σ변위/Σ벽시계 dt). 9910 송신 간격이
               # 40/80 ms 로 불규칙해 틱 단위 d/dt 는 못 쓴다 (2026-08-23 실측).
               'speed_win_s': 0.4,
               # VTD 대회 브릿지의 courseRespawn 감지.
               # respawnPoseToleranceM=1.5 라 실제 리셋 이동량은 3 m 남짓이다.
               'reset_route_s_drop_m': 5.0,
               # 리셋 판정은 거리 단독으로 하지 않는다. 파이프라인이 멈췄다가
               # 재개되면 차는 정상 속도로 크게 움직이는데 그건 리스폰이 아니다.
               'stall_dt_s': 0.2,          # 이보다 긴 틱 간격은 '스톨'로 분류
               # 스톨(긴 dt)이라도 이동량이 이보다 크면 텔레포트 = 리셋으로 처리
               # (2026-08-21: 25 s 갭 + 590 m 이동이 스톨로 분류돼 속도 오염)
               'stall_teleport_m': 50.0,
               'reset_speed_factor': 3.0,  # 점프속도가 실제속도의 이 배를 넘어야
               'reset_abs_speed': 30.0,    # 그리고 이 절대상한도 넘어야 리스폰
               # 같은 차로에서 t_off 가 한 틱에 이만큼 뛰면 리스폰 (고속 리스폰은
               # 환산속도 문턱을 통과한다 — 실측 33 m/s vs 문턱 33.4)
               'reset_toff_jump_m': 2.0,
               'ped_extrapolate_s': 2.0,
               # 횡단보도 폴리곤 근사 (s_rel/lat_off 기준). RTOR 안전 확인과
               # 횡단보도 보행자 정지가 같은 판정을 쓴다.
               'crosswalk_half_w_m': 8.0, 'crosswalk_back_m': 3.0, 'crosswalk_fwd_m': 7.0,
               # 공식 확인: 객체는 수평거리 80 m 이내만, 가까운 순 최대 30개
               'gt_range_m': 80.0, 'range_margin_m': 5.0},
    'control': {'kp': 0.8, 'ki': 0.15, 'k_ld': 0.8, 'ld_min': 5.0, 'ld_max': 20.0,
                # 적분은 목표 근처에서만 쌓는다. 큰 오차 구간에서 쌓으면
                # 목표에 닿은 뒤에도 가속이 남아 제한속도를 넘는다(와인드업).
                'ki_band_mps': 1.0,
                # 곡률이 크면 lookahead 를 줄인다 (코너 컷 방지):
                #   L_d = clamp(k_ld*v, ld_min, ld_max) / (1 + k_curv*|curv|)
                'k_curv': 12.0, 'ld_curve_min': 3.0,
                'steer_rate_max': 1.0},
    # lane_side_m: path 가 이만큼 옆으로 벗어나면 차선변경 시도로 본다
    'shield': {'edge_margin_m': 0.3, 'lane_side_m': 1.0},
    # path 가 비어 있으면 dir/run_<타임스탬프>.jsonl 로 자동 생성한다.
    'log': {'enabled': True, 'dir': 'logs', 'path': '', 'flush_every': 20},
    # enabled=true 일 때만 const_speed_kph 가 속도 상한으로 걸린다.
    # 기본 주행은 제한속도·곡률·정지선만으로 속도를 정한다.
    'debug': {'enabled': False, 'const_speed_kph': 20.0,
              'path_dist_m': 40.0, 'path_step_m': 0.5, 'print_hz': 1.0},
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
    'comm.hold_decay_s': [
        '/cmd 가 이 시간 이상 안 오면 vtd_bridge 가 조향만 0 으로 감쇠해 재송신.',
        '(2026-08-21 주행: 풀락 조향 -0.480 이 0.32 s 유지된 채 3.5 m → 도로이탈)',
        '가속은 유지 — 적신호 대기와 구분 불가. 완전 단절은 watchdog_s 의 SAFE_STOP.',
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
