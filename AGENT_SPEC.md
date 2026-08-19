# HL FMA 2026 자율주행 컨트롤러 — 리포 스캐폴딩 지시서

이 문서를 CLI 코딩 에이전트에 그대로 먹인다. 아래 "작업 지시" 절이 실제 요청이고,
그 위의 "확정 사실"은 추측하지 말고 그대로 따를 것.

---

## 0. 배경 (한 줄)

VTD 2025.2 시뮬레이터가 9910 TCP 로 GT(ego/객체/신호등)를 주고, 우리는 조향·가속도·지시등을 되돌려준다.
도로교통법 10개 항목 + 동적 이벤트 5개로 자동 채점된다. **순수 룰베이스**로 간다. 신경망 없음.

---

## 1. 확정 사실 (실측 완료 — 바꾸지 말 것)

### 1.1 9910 TCP 수신 프레임
- **1109 바이트 고정, 헤더 없음, little-endian, 20 Hz**
- 구조 (연속, 패딩 없음):

| 구간 | 크기 | 포맷 | 내용 |
|---|---|---|---|
| ego | 24 B | `<6f` | x, y, z, heading, pitch, roll (m, rad) |
| objects[30] | 36 B × 30 = 1080 B | `<I8f` × 30 | id(uint32), x, y, z, heading, speed, length, width, height |
| trafficLights[1] | 5 B | `<iB` | id(int32), state(uint8) |

- **빈 객체 슬롯은 전 필드 0** → `id == 0` 이면 스킵
- **trafficLights 는 1칸 고정.** 없으면 `(0, 0)` 이 온다. VTD 가 "지금 봐야 할 신호등" 하나를 골라서 준다.
- state: `0=미할당 1=적 2=황 3=녹 4=좌회전 5=녹+좌회전 6=점멸`
- **ego 에 속도 필드가 없다.** 위치 미분 + 저역통과로 직접 추정할 것.
- ego x, y 는 **xodr 월드 좌표와 동일** (검증 완료: 시나리오 초기위치 508.80 / -168.29 가 lane_graph 의 road 465 lane -1, 중심선 오차 0.05 m — VTD ModuleManager 의 `Road ID=465 Lane ID=-1 Off=0.052` 와 일치).
- 객체 타입 필드는 **없다.** 크기로 분류할 것:
  - 보행자: width 0.5–0.8, height 1.5–2.0, length < 1.0
  - 차량: length > 3.0
  - 그 외 + speed≈0: 정적 장애물
- 객체 id 는 작은 고정 정수(시나리오 플레이어 번호). 프레임 간 동일 id 유지 → 트래킹 용이.
- 객체는 가시성(occlusion) 필터 없이 온다 (60–70 m 밖 객체도 수신 확인). 단 **미확인**: 정확한 거리 컷오프. 77 m 객체가 목록에서 빠진 사례 있음 → 런타임에 관찰 로그 남길 것.

### 1.2 9910 TCP 송신 (제어)
- **9 바이트, 헤더 없음**: `struct.pack('<ffB', steering, targetAccel, turnSignal)`
- steering [rad] (좌 +), targetAccel [m/s²] (음수 = 감속), turnSignal `0=OFF 1=LEFT 2=RIGHT`
- 20 Hz 로 계속 보낼 것. 끊기면 VTD 가 멈출 수 있음.
- **부호 주의**: 주최 제공 예제(`tcp_manual_controller.py`)는 내부 steer 를 `steer_out = -steer` 로 뒤집어 보낸다. 실차 검증 전까지 조향 부호는 파라미터로 빼둘 것 (`STEER_SIGN = -1.0`).

### 1.3 차량 제원 (Ioniq 6)
- wheelbase 2.944 m, 전장 4.848, 전폭 1.886, 전고 1.507
- 최대 조향 0.48 rad, 회전반경(지름) 12.53 m
- **좌표 원점은 뒷바퀴 축 중심(지면)** — Pure Pursuit 에 그대로 쓰면 됨
- 센서(사용 안 함): LiDAR Velodyne HDL-32E 80 m @ (1.0, 0, 2.4), 카메라 FOV 80°×50.6° @ (1.5, 0, 1.6)

### 1.4 채점 항목
법규 10개: 제한속도, 스쿨존 속도, 차로 유지, 중앙선 침범·우측통행, 보도 침범, 실선 차선변경 금지,
적색신호 정지, 녹색신호 통과, 도로 파손·장애물 대응, 횡단보도 정차 금지.
동적 이벤트 5개: 보행자 출현(거리·감속·TTC·정지), 차량 출현/급정거(안전거리), 장애물(감속·정지·회피),
방향지시등(회전·차선변경 n초 전 점등), 충돌(감점).
경로 3 km 이내 / 약 5분.

### 1.5 이미 있는 자산 (재작성 금지, import 해서 쓸 것)
- `lane_graph/build_lane_graph.py` — xodr → `lane_graph.pkl` (오프라인 1회)
- `lane_graph/lanegraph.py` — 런타임 헬퍼. **이 API 를 그대로 사용**:
  - `LaneGraph(path)` / `.locate(x, y, yaw, prefer=route['lanes']) -> LaneMatch(lane, s, t, heading_err, dist, idx)`
  - `.lookahead(route, idx, s_in_lane, horizon) -> [Ahead(dist, kind, lane, s_in_lane, data)]`
    - kind: `stop_line`(data.signal_ids) / `crosswalk`(data.kind=pelican|inferred) / `crosswalk_warn` / `yield` /
      `speed`(data.limit, data.school_zone) / `mark_left`·`mark_right`(data.type, data.lane_change_ok, data.is_center) /
      `junction_in`·`junction_out` / `turn_left`·`turn_right` / `lane_change_left`·`lane_change_right`(window_s0/s1) /
      `dead_end` / `route_end`
  - `.summarize(ahead) -> dict`
  - `.points_ahead(lane, s, dist, step, route, idx) -> (N,2)` ← Pure Pursuit 목표점용
  - `.mark_at(lane, s, side)`, `.lane_change_ok(lane, s, side)`, `.speed_limit_at(lane)`, `.width_at(lane, s)`,
    `.point_at(lane, s)`, `.neighbor(lane, side)`, `.successors/.predecessors(lane)`
  - lane key = `(road_id, section_idx, lane_id)`. lane 레코드에 `left_is_center`(중앙선 여부), `dir`, `curv` 등 포함.
- `lane_graph/build_route.py` — 경유점 csv → `route.pkl`
  (`lanes`, `cum_s`, `lengths`, `total_length`, `start_s_in_lane`, `waypoints`, `waypoint_s`, `events`)
  경로 누적거리 = `cum_s[i] + s_in_lane`
- `lane_graph/plot_lane_graph.py` — 시각화
- `probe_9910.py` — 9910 덤프/디코드 도구 (참고용, `decode()` 재사용 가능)
- 맵 파싱 결과: 도로 651 / 주행차로 2480 / 신호등 646 / 정지선 277(방향별 클러스터) / 제한속도는
  **표준 `<speed>` 필드 없음** → 노면표시 객체(RM_517_50=50, roadmark_speed_30=30, RM_518=스쿨존 30)로 도로·방향 단위 부여,
  표시 없는 도로는 `None` → 런타임에서 직전 값 유지(carry).

---

## 2. 아키텍처 (한 틱 = 한 패킷, 순차 실행)

```
9910 → Comm.recv → Perception → Planner → Shield → Control → Comm.send → 9910
                          ↘________ Logger (매 틱 전부 기록) ________↙
```

- **단일 프로세스, 단일 스레드 루프.** ROS 안 씀. 노드 간 지연/시간동기화 문제 제거.
- 각 단계는 앞 단계 출력(dataclass)만 받는다. 역방향 참조 금지.
- Perception 은 판단하지 않는다. Control 은 교통법을 모른다. Shield 는 Planner 를 신뢰하지 않는다.

---

## 3. 작업 지시

아래 구조로 리포를 스캐폴딩하고, 각 파일에 **타입 시그니처와 docstring, TODO 주석**까지 작성한다.
로직 본문은 `# TODO:` 로 남기되, **Comm 파싱/송신과 dataclass 정의와 main loop 는 완전히 구현**한다.

```
hlfma/
├── README.md                  # 실행법, 대회날 체크리스트
├── requirements.txt           # numpy, scipy  (matplotlib 은 tools 전용)
├── run.sh                     # 환경변수 + main 실행 (대회날 이것만 실행)
├── config.yaml                # 모든 튜닝 파라미터 (아래 5절)
├── src/
│   ├── main.py                # 틱 루프. argparse(--host --port --route --graph --replay --log)
│   ├── types.py               # dataclass: RawPacket, EgoState, TrackedObject, WorldState, Decision, Command
│   ├── comm.py                # Comm: connect/recv/parse/send/watchdog/reconnect
│   ├── perception.py          # Perception: RawPacket + LaneGraph + route → WorldState
│   ├── planner.py             # Planner: WorldState → Decision (FSM + min() 속도중재)
│   ├── shield.py              # Shield: Decision clamp (법규 하드 가드)
│   ├── control.py             # Control: Decision → Command (Pure Pursuit + 종방향 PI)
│   ├── logger.py              # 매 틱 jsonl 기록 + 리플레이 소스
│   └── lanegraph.py           # (기존 파일 복사, 수정 금지)
├── tools/
│   ├── build_lane_graph.py    # (기존)
│   ├── build_route.py         # (기존)
│   ├── plot_lane_graph.py     # (기존)
│   ├── probe_9910.py          # (기존)
│   ├── replay.py              # 로그 jsonl → Comm 대체 소스로 재생 (VTD 없이 P/P/S/C 재실행)
│   └── score.py               # 로그 → 채점 (법규 10항목 자동 판정)  ※ 뼈대만
├── data/
│   ├── lane_graph.pkl
│   └── route.pkl
└── tests/
    ├── test_comm_parse.py     # 합성 1109B 프레임 → 파싱 왕복 검증
    ├── test_perception.py     # 시나리오 초기위치가 (465,2,-1), t≈0.05 로 매칭되는지
    └── test_shield.py         # 과속/실선 차선변경/TTC 입력 시 clamp 되는지
```

### 3.1 `types.py` — 이대로 정의할 것

```python
@dataclass
class RawPacket:
    t_recv: float                    # time.monotonic()
    ego: tuple[float, ...]           # x, y, z, heading, pitch, roll
    objects: list[tuple]             # id≠0 인 것만, (id, x, y, z, heading, speed, length, width, height)
    lights: list[tuple[int, int]]    # (id, state), id==0 이면 제외

@dataclass
class EgoState:
    x: float; y: float; z: float; yaw: float; pitch: float; roll: float
    speed: float                     # 위치 미분 + LPF 로 추정
    accel: float                     # speed 미분
    lane: tuple[int, int, int] | None
    s: float                         # 차로 내 s
    route_s: float                   # 경로 누적거리
    t_off: float                     # 중심선 횡오프셋 (좌 +)
    heading_err: float

@dataclass
class TrackedObject:
    id: int
    x: float; y: float; heading: float; speed: float
    length: float; width: float; height: float
    cls: str                         # 'vehicle' | 'pedestrian' | 'obstacle' | 'unknown'
    lane: tuple[int, int, int] | None
    on_route: bool                   # 경로 차로 위인지
    s_rel: float                     # 경로 기준 종방향 상대거리 (+ = 앞)
    lat_off: float                   # 내 경로 중심선 기준 횡거리
    v_rel: float                     # 접근 속도 (+ = 접근)
    ttc: float                       # inf 가능
    will_enter_lane: bool            # 1~2 s 등속 외삽 시 내 차로 진입
    age: float                       # 마지막 수신 후 경과 (coasting 용)
    coasting: bool                   # 이번 틱 수신 없어 외삽 중

@dataclass
class WorldState:
    t: float
    ego: EgoState
    objects: list[TrackedObject]
    light: tuple[int, int] | None    # (id, state)
    ahead: list                      # lanegraph.Ahead 리스트
    summ: dict                       # lanegraph.summarize 결과
    speed_limit: float               # 현재 유효 제한속도 [m/s] (carry 반영)
    school_zone: bool
    left_solid: bool; right_solid: bool; left_is_center: bool
    valid: bool                      # 패킷 신선도/좌표 점프 없음
    flags: dict

@dataclass
class Decision:
    v_target: float                  # [m/s]
    path: list[tuple[float, float]]  # 추종 목표 점열 (월드 좌표)
    turn_signal: int
    state: str                       # FSM 상태명
    reasons: dict                    # 각 속도 후보값 (로그/디버깅용)

@dataclass
class Command:
    steering: float
    accel: float
    turn_signal: int
```

### 3.2 `comm.py` — 완전 구현

- `connect()` 재시도 루프, `TCP_NODELAY`
- **스트림 재조립**: `recv` 가 1109 배수로 오지 않을 수 있으므로 버퍼에 쌓고 1109 씩 잘라 **가장 최신 프레임만** 사용 (밀린 프레임은 버림 — 지연 누적 방지)
- `parse(frame) -> RawPacket` (§1.1 그대로, id==0 스킵)
- `send(Command)` (§1.2)
- watchdog: `WATCHDOG_S` 동안 수신 없으면 `valid=False` + 안전정지 명령 송신 + 재접속.
  **주의: 적신호 대기가 13–18 s 이므로 "속도 0" 을 이유로 재시작하면 안 된다.** 오직 패킷 수신 여부로 판단.

### 3.3 `perception.py`

1. ego 속도/가속도 추정 (dt 는 `t_recv` 차분, LPF 계수 config)
2. `lg.locate(x, y, yaw, prefer=route['lanes'])` → lane/s/t/heading_err, `route_s = cum_s[idx] + s`
3. `lg.lookahead(route, idx, s, horizon)` + `summarize`
4. 제한속도 carry (표시 없는 도로는 직전 값 유지), 스쿨존 플래그
5. 객체 분류(§1.1 크기 규칙) → `lg.locate(obj.x, obj.y, obj.heading)` 로 lane 매칭 → `on_route`, `s_rel`, `lat_off`
6. `v_rel`, `ttc`, 보행자 1–2 s 등속 외삽 → `will_enter_lane`
7. 직전 틱에 있었으나 이번에 없는 id 는 `COAST_S` 동안 마지막 상태로 유지 (`coasting=True`)
8. 유효성: dt 이상, 좌표 점프(> `JUMP_M`), lane 매칭 실패 → `flags`

### 3.4 `planner.py`

FSM: `FOLLOW / STOP_LINE / FOLLOW_LEAD / YIELD_PED / AVOID / RETURN / E_STOP`

종방향은 **후보들의 min()**:
```
v_target = min(
  speed_limit - SPEED_MARGIN,
  school/crosswalk/junction/blind 구간 캡,
  곡률 속도  sqrt(A_LAT_MAX / |curv|),
  IDM(선행차),
  v_safe(정지점),                       # sqrt(2*A_COMF*(d - STOP_GAP))
  v_safe(장애물), v_safe(보행자)
)
```
신호 판단(정지선까지 d, 현재 v, 황색 잔여 t_y=3 s 가정):
- 적/황이고 `d > v*t_y` → 정지점 = 정지선 - `STOP_GAP`
- 황이고 `d <= v*t_y` → 통과
- 녹/녹+좌 → 통과 (**녹색신호 통과도 채점 항목. 불필요한 정지 금지**)
- 좌회전 신호(4/5)는 경로가 좌회전일 때만 진행 근거로 사용
- 점멸(6)은 config 플래그로 동작 선택 (기본: 서행 + 교차 차량 양보)

횡방향:
- 기본 = 현재 경로 차로 중심선 (`lg.points_ahead`)
- `AVOID` 진입 조건: 내 차로 정지 장애물 && 해당 방향 `lane_change_ok` && 옆차로 뒤 `LC_BACK_M`/앞 `LC_FRONT_M` 비었음 && 지시등 `SIGNAL_LEAD_S` 선행 점등 완료
- **위 조건 하나라도 불만족 → 회피하지 않고 정지** (실선 차선변경·중앙선 침범이 감점이므로 정지가 항상 이득)
- 장애물 통과 후 조건 만족 시 `RETURN`

교차로 진입 게이트 (신호 위반 차량 대비):
- 교차 방향 접근 객체의 `t_them = d/v`, 내 통과시간 `t_me` 비교 → 여유 `CROSS_MARGIN_S` 미만이면 진입 보류
- 감속 안 하는 접근 차량은 신호와 무관하게 양보

지시등: `route['events']` 의 `turn_*`, `lane_change_*` 지점까지 남은 거리를 현재 속도로 나눈 시간이
`SIGNAL_LEAD_S + SIGNAL_MARGIN_S` 이하가 되면 점등.

### 3.5 `shield.py` — Planner 를 신뢰하지 않는 하드 가드

```
1. v_target = min(v_target, speed_limit - SPEED_MARGIN)          # 절대 초과 금지
2. 현재 s 에서 해당 방향 mark.lane_change_ok == False 인데 path 가 옆차로면 → 현재 차로 path 로 교체
3. left_is_center 인데 path 가 좌측 이탈 → 교체
4. min TTC < TTC_EMERGENCY → v_target = 0, accel = A_EMERGENCY (저크 제한 해제)
5. 횡단보도 구간 내 정차 금지: 정지점이 crosswalk 구간 안이면 구간 앞으로 당김
6. |t_off| > lane_width/2 - EDGE_MARGIN → 복귀 우선
```
각 clamp 발동을 `Decision.reasons['shield']` 에 기록.

### 3.6 `control.py`

- 횡: Pure Pursuit. `L_d = clamp(K_LD * v, LD_MIN, LD_MAX)`, `delta = atan2(2*WHEELBASE*sin(alpha), L_d)`,
  `steering = STEER_SIGN * clamp(delta, -MAX_STEER, MAX_STEER)`, 조향 변화율 제한
- 종: `accel = KP*(v_target - v) + KI*∫ + FF`, `clamp(A_MIN, A_MAX)`, 저크 제한 `JERK_MAX`
  (단 `state == 'E_STOP'` 이면 저크 제한 해제)
- 정지 유지: `v_target == 0 && v < 0.2` → `accel = A_HOLD` (음수 유지)

### 3.7 `logger.py` / `tools/replay.py`

- 매 틱 1줄 jsonl: `t`, raw ego/objects/lights, WorldState 요약, Decision(reasons 포함), Command
- `replay.py` 는 이 jsonl 을 읽어 `Comm.recv` 를 대체 → **VTD 없이** Perception/Planner/Shield/Control 재실행.
  판단 로직 수정 후 회귀 테스트의 기반.

---

## 4. 코딩 규칙

- Python 3.10+, 표준 라이브러리 + numpy/scipy 만. 딥러닝 프레임워크 금지.
- 모든 튜닝 상수는 `config.yaml` 에서만. 코드에 매직넘버 금지.
- 단위 명시: 내부는 **m, m/s, m/s², rad** 통일. km/h 는 입출력에서만 변환.
- 각 모듈은 부작용 없는 순수 함수 형태 선호 (입력 dataclass → 출력 dataclass).
- 예외로 루프가 죽지 않게: 틱 내부 예외는 잡아서 로깅 + 직전 Command 유지 + `E_STOP` 카운터.
- 타입힌트 필수, docstring 은 한국어.

## 5. `config.yaml` 초기값 (근거 있는 값만; 나머지는 TODO 로 표시)

```yaml
comm: {host: 127.0.0.1, port: 9910, send_hz: 20, watchdog_s: 1.0, steer_sign: -1.0}
vehicle: {wheelbase: 2.944, max_steer: 0.48, length: 4.848, width: 1.886}
speed:   {margin_kph: 3.0, a_comf: 1.5, a_max: 2.0, a_min: -6.0, a_emergency: -8.0,
          jerk_max: 2.0, a_lat_max: 2.0, stop_gap_m: 1.0}
caps_kph: {school_zone: 28, crosswalk: 25, junction: 30, blind: 25}   # TODO: 튜닝
signal:  {yellow_s: 3.0, lead_s: 3.0, margin_s: 1.0, flash_mode: yield}
lead:    {time_headway_s: 2.0, min_gap_m: 5.0}
ttc:     {warn_s: 4.0, brake_s: 2.5, emergency_s: 1.5}
lane_change: {back_m: 30, front_m: 50, min_window_m: 20}
cross:   {margin_s: 2.0}
percep:  {horizon_m: 200, coast_s: 1.5, jump_m: 5.0, speed_lpf: 0.3,
          ped_extrapolate_s: 2.0}
control: {kp: 0.8, ki: 0.15, k_ld: 0.8, ld_min: 5.0, ld_max: 20.0}   # TODO: 튜닝
```

## 6. 산출물 확인 기준

- `python3 -m pytest tests/` 통과
- `python3 src/main.py --replay logs/sample.jsonl` 가 VTD 없이 끝까지 돈다
- `python3 src/main.py --host 127.0.0.1 --graph data/lane_graph.pkl --route data/route.pkl` 가
  연결 후 20 Hz 로 Command 를 보내고 매 틱 로그를 남긴다

## 7. 아직 미확인 (코드에 TODO 로 남기고 가정은 config 로 뺄 것)

1. 신호등이 교차로마다 id 가 바뀌는지, state 전이가 실제로 오는지 (관측 중)
2. GT 객체 거리 컷오프 (77 m 사례) — 런타임 관찰 로그 남길 것
3. 대회 경유점 파일 형식 (좌표 개수/판정 반경) → `build_route.py --radius`
4. 표시 없는 도로의 기본 제한속도 (주최 문의 중) → `config: default_speed_kph`
5. `RM_518` 이 어린이보호구역 30 이 맞는지 (텍스처 확인 필요)
6. 점멸 신호(state 6) 의 대회 규정 해석
