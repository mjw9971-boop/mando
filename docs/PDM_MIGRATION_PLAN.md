# PDM-Lite 이식 계획 — Phase 0 분석 결과

작성: 2026-08-25. 원본: DriveLM `DriveLM-CARLA` 브랜치 `pdm_lite/team_code/`
(스크래치패드에 sparse-clone 완료, 라이선스 **Apache 2.0** — `LICENSE` 를 team_code/ 에 같이 복사).

가져올 파일 (줄 수):

| 파일 | 줄 | 비고 |
|---|---|---|
| autopilot.py | 2844 | 판단 본체. 수정 지점은 §0-1/§0-2 |
| config.py | 434 | GlobalConfig. `import carla` 는 debug 색상 8줄뿐 → 제거 |
| privileged_route_planner.py | 1110 | VtdRoutePlanner 가 같은 시그니처로 대체. shift_route_* 원문은 보존해 이식 |
| kinematic_bicycle_model.py | 142 | forecast 용. 종방향 §0-4 결정에 따라 일부 수정 |
| longitudinal_controller.py | 272 | §0-4 (b) 채택 시 대체됨 (파일은 참고용으로 보관) |
| lateral_controller.py | 161 | 무수정 사용 가능 |
| nav_planner.py | — | **가져오지 않는다** (§0-2: `RoutePlanner`(_command_planner) 는 save() 데이터 수집 전용) |
| transfuser_utils.py | — | **가져오지 않는다** (§0-1: 쓰이는 3개 함수 전부 제거 경로에 있음) |

외부 의존: numpy, `scipy.integrate.RK45`(IDM), `scipy.interpolate.interp1d`·`scipy.spatial.cKDTree`(route planner)
— 전부 기존 requirements.txt 로 충족. ujson/gzip/carla/srunner/leaderboard/agents 의존은 제거 대상 코드에만 있다.

---

## 0-1. autopilot.py 의 CARLA API 호출 전수 목록

분류: **[어댑터]** = vtd_adapter 가 같은 시그니처로 대체, **[제거]** = 코드째 삭제, **[stub]** = 빈/고정 반환으로 무력화.

### 프레임워크·전역

| 위치 | 호출 | 분류 |
|---|---|---|
| import (L12–26) | `carla`, `RoadOption`, `CarlaDataProvider`, `autonomous_agent(_local)`, `nav_planner`, `transfuser_utils`, `scenario_logger` | [어댑터] carla 타입·RoadOption 은 vtd_adapter 제공(RoadOption 은 값 호환 enum 복제), 나머지 import 삭제 |
| setup() L51 | `autonomous_agent.Track.MAP` | [제거] |
| setup() L124 | `carla.VehicleLightState.*` | [제거] (전조등 — VTD 9910 에 없음) |
| setup() L129 | `CarlaDataProvider.get_map()` | [어댑터] 생성자 주입 `VtdMap` |
| setup() L132–150 | SAVE_PATH / ScenarioLogger / speed_histogram / tp_stats / datagen | [제거] |
| toggle_recording() | `CarlaDataProvider.get_client()`, `start/stop_recorder` | [제거] 메서드째 |
| _init() L193–194 | `CarlaDataProvider.get_hero_actor()`, `.get_world()` | [어댑터] `VtdEgo`, `VtdWorld` 주입 |
| _init() L197–202 | `org_dense_route_world_coord`, `location.distance` (parking exit 판정) | [제거] (한국 코스에 주차 출발 없음, `starts_with_parking_exit=False` 고정) |
| _init() L205–213 | `PrivilegedRoutePlanner.setup_route(...)` | [어댑터] `VtdRoutePlanner` (route dict 로 초기화) |
| _init() L219–223 | `RoutePlanner`(_command_planner) | [제거] — 소비처가 save()/commands 뿐 (아래 참고) |
| _init() L231–237 | `world.get_actors()` + `t_u.get_traffic_light_waypoints` → `list_traffic_lights` | [stub] `self.list_traffic_lights = []` — 소비처는 ego_agent_affected_by_red_light 의 close_traffic_lights(데이터 수집)·visualize 뿐. 판단은 `next_traffic_light.state` 만 쓴다 |
| _init() L241–245 | bugged 2-wheeler `actor.destroy()` | [제거] (CARLA 버그 대응) |
| sensors() | opendrive_map/imu/speedometer 스펙 | [제거] 메서드째 (leaderboard 전용) |

### tick / run_step / _get_control

| 위치 | 호출 | 분류 |
|---|---|---|
| tick_autopilot L288 | `self._vehicle.get_velocity().length()` | [어댑터] `EgoSpeedEstimator` (기존 perception `_estimate_motion` 이식 — 9910 에 자차 속도 필드가 없다) |
| tick_autopilot L291–294 | `input_data["imu"]` + `t_u.preprocess_compass` | [어댑터] 9910 heading 을 §0-3 관례로 변환한 rad 직접 공급 (compass −90° 보정 불필요) |
| tick_autopilot L297 | `self._vehicle.get_location()` | [어댑터] 9910 ego (x,y,z) |
| run_step L327–329 | `CarlaDataProvider.get_client().get_world().get_map()` | [제거] — `_init()` 을 주입 기반으로 교체 |
| _get_control L363 | `_waypoint_planner.run_step(pos)` 8-튜플 | [어댑터] `VtdRoutePlanner.run_step` 동일 시그니처 |
| _get_control L388–389 | `self._world.get_actors()`, `.filter("*vehicle*")` | [어댑터] `VtdWorld.get_actors()` + `ActorList.filter` |
| _get_control L392 | `_manage_route_obstacle_scenarios(...)` | [stub] `(target_speed, False, [target_speed,None,None,None])` 반환 (§0-2) |
| _get_control L419 | `world_map.get_waypoint(loc).is_junction` | [어댑터] `VtdMap.get_waypoint` → `VtdWaypoint.is_junction` (lanegraph `lanes[key]['junction'] != -1`) |
| _get_control L431–434 | `carla.VehicleControl()` | [어댑터] shim `VehicleControl` |
| _get_control L432 | `steer_noise * np.random.randn()` | [제거] (데이터 수집용 노이즈 — config 에서 0) |
| _get_control L437–441 | rolling-back 방지 (`throttle==0 & v<0.5 → brake=1`) | §0-4 (b) 채택 시 어댑터 a_hold(−1.0 유지)로 대체 |
| _get_control L444 | `CarlaDataProvider.get_velocity(self._vehicle)` | [어댑터] EgoSpeedEstimator |
| _get_control L445–452 | `max_blocked_ticks` 시 `throttle=1` | 원문 유지하되 **상수 무력화 필수** (§0-6 — 8.5 s 정지에서 풀스로틀은 적신호 대기(13–18 s)와 충돌) |
| _get_control L460–475 | `_command_planner.run_step` → target_point/commands | [제거] — 소비처가 save() 뿐 |
| _get_control L477 | `self.save(...)` | [제거] — 우리 Logger 가 대체 (§0-2) |

### 판단 함수 (원문 유지, 어댑터 타입 위에서 그대로 동작)

| 함수 | CARLA 의존 | 처리 |
|---|---|---|
| `_compute_target_speed_idm` | scipy RK45 만 | 원문 유지 |
| `is_near_lane_change` | `RoadOption.CHANGELANE*` (planner.commands) | 원문 유지 — VtdRoutePlanner 가 commands 배열 제공 |
| `predict_other_actors_bounding_boxes` | `actor.get_control()`(steer/throttle/brake), `get_velocity/location/transform`, `bounding_box.extent`, `carla.Location/Rotation/Vector3D/BoundingBox`, `world.debug.draw_box` | [어댑터] VtdActor: get_control() 은 (0,0,0) 고정 → forecast 가 등속·직진 외삽이 됨(9910 에 타차 조작량 없음, 허용 근사). debug 는 no-op |
| `compute_target_speed_wrt_leading_vehicle` | `world.get_actor(id)`, visualize 블록 | [어댑터] `VtdWorld.get_actor`; visualize 블록 [제거] |
| `compute_target_speeds_wrt_all_actors` | `blocking_actor.attributes["base_type"]=="bicycle"` | [어댑터] `VtdActor.attributes = {}` → bicycle 분기 자연 비활성 (9910 분류에 자전거 없음, 차량 취급 = 더 보수적) |
| `get_brake_and_target_speed` | `transform.transform(bb.location)`, `carla.BoundingBox` | [어댑터] shim Transform.transform (bb.location=(0,0,0) 이면 항등) |
| `forecast_ego_agent` | `_turn_controller.save/load`, `_waypoint_planner.save/load`, `get_throttle_extrapolation` | 원문 유지 — VtdRoutePlanner 도 save/load(route_index) 제공. 종방향은 §0-4 |
| `forecast_walkers` | `actors.filter("*walker*")`, `ped.get_control().direction` | [어댑터] walker 의 get_control().direction 을 heading 단위벡터로 합성 제공 |
| `ego_agent_affected_by_red_light` | `list_traffic_lights` 루프(close_traffic_lights + visualize), `carla.TrafficLightState.Green` 비교, IDM | 루프는 빈 리스트로 자연 무력화(코드 유지 가능), enum 은 어댑터 `TrafficLightState` 제공, IDM 원문 유지 |
| `ego_agent_affected_by_stop_sign` | trigger_volume, filter("*traffic.stop*") | [stub] `return target_speed` (§0-2 — 한국 코스에 정지표지 없음). `get_nearby_object` 도 같이 [제거] |
| OBB 계열 (`_dot_product`/`cross_product`/`get_separating_plane`/`check_obb_intersection`) | `carla.Vector3D` 산술, `rotation.get_forward/right/up_vector` | 원문 유지 — shim Vector3D(+,−,·스칼라)·Rotation(get_*_vector) 구현 |
| `_get_angle_to` | 순수 수학 | 원문 유지 |

### privileged_route_planner.py 쪽

`setup_route`/`compute_route_info`/`compute_distances_to_traffic_lights`/`compute_distances_to_stop_signs`/`compute_speed_limits`/`prevent_too_early_lane_changes` 는
CARLA map/world 전제라 **VtdRoutePlanner 가 route dict + lanegraph 로 재구현** (Phase 2-4). 다음은 원문 그대로 가져와 보존:
`run_step`(argmin 창 탐색), `save/load`, `_smooth_transition`, `shift_route_smoothly`, `shift_route_around_actors`(cKDTree), `get_closest_route_index`,
`compute_rotation_angles`, `compute_leading_vehicles`, `compute_trailing_vehicles`, `extend_lane_shift_transition_*`(Phase 4 판단 후).
`shift_route_for_invading_turn` 은 CARLA 시나리오 전용 → 제거.

---

## 0-2. 제거·stub 범위 확정

| 대상 | 처리 | 근거 |
|---|---|---|
| `_manage_route_obstacle_scenarios` (640줄) | [stub] `(target_speed, False, [target_speed,None,None,None])` | CarlaDataProvider.active_scenarios + CARLA 시나리오명 하드코딩. 단 내부가 쓰는 `shift_route_around_actors`/`is_overtaking_path_clear` 패턴은 Phase 4 의 blocked 추월 발동에 재사용하므로 route planner 쪽 원문은 살려 둔다 |
| `ego_agent_affected_by_stop_sign` | [stub] `return target_speed` | 한국 코스에 정지표지 없음. `close_stop_signs`/`cleared_stop_sign` 상태도 같이 죽음(무해) |
| `save()` / `destroy()` / `toggle_recording()` / `sensors()` / datagen·histogram·tp_stats | [제거] | 데이터 수집 전용. save() 제거로 `_command_planner`·`inverse_conversion_2d`·`ego_matrix`·commands deque 가 전부 불필요해짐 |
| visualize / `world.debug.draw_*` | [제거] (또는 `VtdWorld.debug` no-op — 어느 쪽이든 `# VTD:` 주석) | 렌더링 없음 |
| `tick_autopilot` 센서 입력 (imu/speedometer) | [교체] 어댑터 WorldState 로부터 `{gps, speed, compass}` 구성 | 센서 스택 없음 |
| `ScenarioLogger` / `lon_logger` | [제거] | 우리 Logger(run_*.jsonl) 가 유일한 기록 경로 |

### PDM-Lite 에 없어서 어댑터가 보충해야 하는 기능 (기존 시스템에 있었음)

1. **방향지시등** — PDM-Lite 는 turn signal 을 전혀 내지 않는다. 9910 송신 3번째 필드이자 채점 항목.
   → route dict 의 events(turn_left/right, lane_change_*) + 거리로 어댑터(run_agent 루프)에서 산출. 기존 planner 의 지시등 규칙을 독립 유틸로 이식(판단 로직 수정 아님).
2. **RTOR(적신호 우회전)** — PDM 은 적신호면 무조건 정지 유지. 기존 `signal.rtor_enabled=True` 거동이 사라진다. → **열린 질문 §열린-1**.
3. **황색 통과 판단** — PDM 은 Green 이 아니면 IDM 정지. 기존 yellow_s(3 s) 기반 통과/정지 판단 소멸. 정지선 직전 황색 전환 시 IDM 이 급감속을 시킬 수 있다 — 원문 유지 원칙에 따라 감수하고 실기에서 모니터.
4. **스쿨존 캡(28 kph)·교차로 캡** — PDM 의 speed_limits 배열/max_speed_in_junction 으로 흡수 (§0-6). VtdRoutePlanner 가 school_zone 차로의 limit 에 캡을 반영해서 준다 (autopilot 무수정).
5. **리셋(courseRespawn) 감지·이벤트 로그** — 어댑터(ego.py)가 유지. 리셋 시 EgoSpeedEstimator·VtdRoutePlanner.route_index·LateralPID error_history 를 초기화해야 한다 (기존 control.reset() 대응).
6. **route_s/t_off/heading_err 로깅** — PDM 은 모름. run_agent 가 lg.locate 로 매 틱 계산해 jsonl 스키마를 채운다 (§제약).

---

## 0-3. 좌표계·단위 결정

### 사실 정리

- **VTD/OpenDRIVE**: 우수계(z-up, y 좌측), heading rad 반시계(+ = 좌회전). 9910 ego/obj heading 도 rad.
- **CARLA**: 좌수계(y 가 ODR 의 −y), `rotation.yaw` **도(deg)**, 수치상 시계방향 증가. autopilot 내부에서 rad 로 쓸 때는 `deg2rad(yaw)` 그대로.
- PDM-Lite 조향 출력: `[-1, 1]` 정규화 (CARLA steer, + = CARLA 프레임에서 yaw 증가 방향). 우리 9910 송신은 **rad, 좌 +** (`comm.steer_sign +1.0` 실측 확정).

### 결정 (권장안 채택): 어댑터 경계에서 CARLA 관례로 변환

```
입력  (VTD → PDM):  x_c = x_v,  y_c = −y_v,  z_c = z_v,  yaw_c[deg] = −degrees(heading_v)
                    tick_autopilot 의 compass[rad] = −heading_v  (preprocess_compass 우회)
출력  (PDM → VTD):  steering_vtd[rad] = −steer_carla × vehicle.max_steer(0.48)
                    (Comm.pack_command 의 steer_sign +1.0 은 그대로 유지 — 변환은 vtd_adapter/control.py 한 곳에서만)
```

이렇게 하면 autopilot·lateral_controller·route planner 의 각도/외적/OBB 수식을 한 줄도 손대지 않는다.
lanegraph 좌표(VTD 프레임)는 VtdMap/VtdRoutePlanner **내부에서만** 쓰고, 밖으로 내주는 모든 Waypoint/route_np/actor 좌표는 CARLA 프레임이다.
Logger 에는 **VTD 원 좌표**를 남긴다(스키마 유지 — 어댑터가 원 패킷을 그대로 보관하므로 변환 불필요).

### handedness 민감 연산 목록 (전부 "일관된 프레임"만 요구 → 미러 변환으로 무수정 성립)

| 연산 | 판정 |
|---|---|
| `_get_angle_to` (회전행렬 투영 + atan2) | 프레임 일관 — 미러 프레임에서 그대로 성립 |
| `LateralPIDController.step` (arctan2 heading error) | 동일 |
| `compute_rotation_angles` (경로 yaw deg) | 동일 — actor yaw 와 같은 프레임이면 됨 |
| `compute_leading/trailing_vehicles` (yaw 차 mod 360) | 동일 |
| `forecast_ego_vehicle`/`forecast_other_vehicles` (heading+slip 적분) | 동일 |
| `check_obb_intersection` (SAT) | get_forward/right/up_vector 를 shim 에서 한 정의로 고정하면 성립. OBB 가 y 대칭이라 right 벡터 부호조차 결과 불변 |
| `get_left_lane`/`get_right_lane` (route shift) | **주행방향 상대** 개념 — CARLA 의 left = 운전자 좌측 = lanegraph `neighbor('left')` 그대로 매핑 (미러에서도 뒤집지 않는다. CARLA 자체가 ODR 의 미러이고 그 위에서 left 를 운전자 좌로 정의하기 때문) |
| `shift_route_for_invading_turn` 의 `[[0,-1],[1,0]]` 회전 | 프레임 종속이지만 [제거] 대상이라 무관 |

검증 테스트(Phase 2): 우회전 연결로에서 `VtdWaypoint.transform.rotation.yaw` 가 진행에 따라 **증가**(CARLA 시계방향 = 우회전)하는지, 직선에서 lateral PID 부호가 t_off 를 줄이는 방향인지.

### 단위

- 속도 m/s (양쪽 동일), lateral lookahead 는 "포인트 인덱스"(= 0.1 m 단위, points_per_meter=10) — route 재샘플 간격만 지키면 됨.
- steer 스케일: PID 는 CARLA Lincoln (유효 최대조향 ≈ 0.37 rad, bicycle steering_gain) 기준 튜닝. 우리 0.48 rad 곱이면 동일 오차에 ~30 % 강한 조향 → 첫 실기에서 진동 시 `lateral_pid_kp` 계열이 아니라 **스케일(0.37 고정)** 쪽을 먼저 검토 (원문·상수 원칙 유지). SteerSignMonitor(기존 control.py)를 어댑터로 이식해 부호 검증 자동화.

---

## 0-4. 종방향 출력 (throttle/brake ↔ targetAccel)

### 비교

**(a) 어댑터에서 throttle/brake → accel 매핑**
- LongitudinalLinearRegressionController 원문 유지 가능.
- 그러나 회귀식·상수(1.9 m/tick 등)는 **CARLA Lincoln 물리에 베이지안 최적화**한 것 — VTD Ioniq6 에는 근거 없는 이중 변환(v→throttle→accel 역매핑)이 되고, brake=True 를 몇 m/s² 로 칠지 임의 상수가 새로 필요하다.
- forecast(`get_throttle_extrapolation` + bicycle 다항식)도 CARLA 물리 그대로라 예측이 어긋난다.

**(b) accel 직접 출력 컨트롤러로 교체** ← **권고**
- 9910 이 애초에 targetAccel[m/s²] 를 받는다. IDM 이 이미 감속 프로파일을 반영한 target_speed 를 주므로 컨트롤러는 추종만 하면 된다.
- 기존 control.py 종방향(PI: kp 0.8 / ki 0.15 / ki_band 1.0, a_min −6 / a_max 2 / jerk_max 2, 정지 유지 a_hold −1.0)은 이 VTD·이 차량에서 **검증된** 값이다.
- 구현: `vtd_adapter` 에 `VtdLongitudinalController` — 인터페이스는 원본과 동일 형태 유지:
  - `get_throttle_and_brake(hazard_brake, target_speed, ego_speed)` → 내부적으로 accel 계산 (hazard_brake 또는 target<1e-5 → 정지 프로파일/a_hold)
  - `get_throttle_extrapolation(target_speed, speed)` → forecast 용 accel
  - autopilot 의 두 호출부와 `carla.VehicleControl` 조립부만 `# VTD:` 주석으로 수정 (shim VehicleControl 에 accel 필드 추가). rolling-back 방지(L437)·max_blocked throttle=1(L451) 은 각각 a_hold·a_max 로 치환.
- `KinematicBicycleModel.forecast_ego_vehicle` 의 속도 갱신(throttle/brake 다항식)을 `v' = clip(v + a·dt, 0, ∞)` 로 교체 (`# VTD:` 주석, 원 다항식 병기). `forecast_other_vehicles` 는 get_control()=(0,0,0) 입력이라 등속 — 원문 유지.
- 트레이드오프: forecast_ego 의 가감속 응답이 실차보다 이상적이 됨 — OBB 여유 계수(extent factor)가 흡수. Phase 4 에서 문제 시 상수로만 조정.

---

## 0-5. 신호등 — 가능함을 확인

PDM 이 필요로 하는 두 값과 우리 소스의 대응:

| PDM 입력 | 우리 소스 |
|---|---|
| `distance_to_next_traffic_light` | lanegraph `stop_lines`(signal_ids 있는 정지선)까지의 경로거리. VtdRoutePlanner 빌드 시 PDM 원본과 같은 방식(역방향 스캔)으로 **경로 인덱스별 배열** 사전 계산 — lookahead 의 `Ahead(kind='stop_line').dist` 와 동일 값 |
| `next_traffic_light` (`.state`, `.id`) | `VtdTrafficLight` — id 는 그 정지선의 controller_ids, state 는 매 틱 9910 lights(1개) 를 기존 `_check_light_controller` 매핑(`data/junction_ctrl_map.json` + controller_ids 폴백)으로 대조해 갱신 |

**핵심: 거리 기준은 정지선이다.** 한국 교차로는 신호등이 정지선에서 25 m 이상 떨어져 있으므로 CARLA 식 "신호등 액터까지 거리" 를 쓰면 정지 위치가 교차로 한복판이 된다. 위 구성은 정지선 거리라 자연 충족. CARLA 의 trigger volume 도 사실상 정지선 위치라 IDM 상수의 의미는 보존되지만, `idm_red_light_minimum_distance 6.0` 은 CARLA 정지선 세트백 기준이라 실기에서 정지 위치(목표: 앞범퍼 −1 m 근처)를 보고 조정 후보 (§0-6).

state 매핑 (9910 → TrafficLightState):

| 9910 | 매핑 |
|---|---|
| 1 적 | Red |
| 2 황 | Yellow (PDM 은 Green 외 전부 정지 — §0-2 공백 3) |
| 3 녹 | Green |
| 4 좌회전 | 경로가 그 교차로에서 좌회전(route events)이면 Green, 아니면 Red |
| 5 녹+좌 | Green |
| 6 점멸 | 기존 flash_mode 'yield' 관례 → Green 취급 (서행은 IDM/교차로 캡이 담당) |
| 0/미수신/컨트롤러 불일치 | next_traffic_light=None (Green 동등 — 기존 시스템도 자기 경로 신호일 때만 정지) |

---

## 0-6. config.py 상수 — Ioniq6·VTD 치환 목록

`# VTD:` 주석 + 원값 병기로 team_code/config.py 안에서 변경. params.yaml `vehicle.*` (wheelbase 2.944 / length 4.848 / width 1.886 / height 1.507 / front_overhang 0.855 / max_steer 0.48) 과 대조 완료.

| 상수 | CARLA 값 | 변경 | 근거 |
|---|---|---|---|
| `ego_extent_x/y/z` | 2.4508 / 1.0642 / 0.7554 | **2.424 / 0.943 / 0.7535** | Ioniq6 4.848/1.886/1.507 의 ½ |
| `bicycle_frame_rate`, `fps`, `time_step`, `carla_fps` | 20 / 20 / 0.05 / 20 | 유지 | 우리 루프도 20 Hz |
| `points_per_meter` | 10 | 유지 | 경로 ~수 km × 10 pt/m — run_step argmin 창 40 pt 라 비용 무관 |
| `ratio_target_speed_limit` | 0.72 (+72 kph 상한) | **폐지에 준하는 처리**: VtdRoutePlanner 의 speed_limits 에 `(limit − speed.margin_kph)` 를 반영하고 ratio=1.0, 72 kph 상한 제거 (`# VTD:`) | CARLA 는 "NPC 가 제한의 70 %로 달려서" 0.72; 우리는 감점 회피용 margin 5 kph 가 검증값 |
| `max_speed_in_junction` | 64/3.6 | **30/3.6 (≈8.33)** | caps_kph.junction 30 검증값 |
| `max_blocked_ticks` | 170 (8.5 s) | **무력화 (예: 10⁹)** | 8.5 s 정지 후 throttle=1 은 적신호 대기 13–18 s 와 정면 충돌 → 신호위반 폭주. CARLA AgentBlockedTest 전용 규칙 |
| `steer_noise` | 1e-3 | **0.0** | 데이터 수집용 노이즈 |
| `minimum_speed_to_prevent_rolling_back` | 0.5 | (b) 채택 시 미사용 — a_hold 가 대체 | |
| `longitudinal_linear_regression_*`, `longitudinal_pid_*` | — | (b) 채택 시 미사용 (VtdLongitudinalController 는 params.yaml speed/control 값 사용) | 단일출처 규칙 |
| bicycle `front/rear_wheel_base`, `steering_gain` | −0.0908 / 1.4178 / 0.3685 | **후보: rear=front=1.472 (축거 2.944 균등), steering_gain=0.48** — 1차는 원값 유지도 가능 | WoR 캘리브레이션(Lincoln). forecast 정확도용이라 실기 관찰 후 1회 결정 |
| bicycle `throttle_values`/`brake_values`/`throttle_threshold` | 다항식 | (b) 채택 시 ego forecast 에서 미사용 (`v+aΔt` 로 교체). other-vehicle 용 `brake_acceleration`/`throttle_acceleration` 은 get_control()=(0,0,0) 이라 사실상 미사용 | |
| `idm_*` 전체 | — | **유지** (판단 로직 불변 원칙). 실기 후 조정 1순위 후보만 기록: `idm_red_light_minimum_distance 6.0` (정지 위치), `idm_leading_vehicle_minimum_distance 4.0` (기존 min_gap 5.0) | |
| `lateral_pid_*` | — | 유지. 진동 시 §0-3 의 steer 스케일부터 | |
| `detection_radius` 50, `light_radius` 64 | — | 유지 (9910 은 80 m·30개 제공, light_radius 는 stub 경로라 무관) | |
| `default_forecast_length` 2.0 / `forecast_length_lane_change` 1.1 | — | 유지 | |
| `extra_route_length` 50 | — | 유지 — VtdRoutePlanner 가 route 끝을 lanegraph successor 로 50 m 연장 (완주 지점 통과 시 checkpoint 소진 방지). 완주 판정은 batch_run 의 route_s 기준이라 영향 없음 |
| `route_planner_min/max_distance`, DataAgent/Sensor/LiDAR/camera/Dataloader/Logger 섹션 | — | _command_planner·데이터수집 제거로 미사용 — config 정리 시 삭제 가능(잔류해도 무해) | |
| `carla.Color` 클래스 변수 8개 | — | **삭제** (import carla 제거를 위해 필수) | |

---

## 0-7. 삭제 대상과 영향 받는 tests/

### 삭제 파일

```
src/hlfma/hlfma/nodes/            전부 (params.py 의 DEFAULTS/load_params_yaml 만 vtd_adapter/config.py 로 이동)
src/hlfma/launch/                 전부
src/hlfma_msgs/                   전부
src/hlfma/package.xml, setup.py, setup.cfg, resource/
src/hlfma/hlfma/convert.py        (ROS msg 변환)
src/hlfma/hlfma/core/planner.py   (1475줄 — 판단 전체를 PDM-Lite 가 대체)
src/hlfma/hlfma/core/shield.py
src/hlfma/hlfma/core/perception.py  (§Phase1-2 의 이식 후)
src/hlfma/hlfma/core/timing.py    (nodes 전용)
build/, install/, log/            (colcon 산출물 — .gitignore 도 추가)
run.sh                            ros2 부분
```

이동(삭제 아님): core/{comm,lanegraph,logger,types}.py → vtd_adapter/, core/scoring.py → tools/score.py 가 쓰므로 함께 이동(import 경로 수정).

### tests/ 분류 (import 실측 기준)

| 처리 | 테스트 | 사유 |
|---|---|---|
| **삭제** | test_nodes, test_convert | ROS msg/노드 의존 |
| **삭제** | test_shield, test_emergency_brake, test_blocked_overtake, test_lane_change, test_lc_lead, test_turn_signal, test_crosswalk_ped, test_rtor, test_stop_line | planner/shield 의존 (test_stop_line 의 lanegraph 정지선 검증 부분은 lanegraph 단독 테스트로 발췌 가치 있음 — Phase 1 에서 판단) |
| **삭제(일부 이식)** | test_control | control.py 삭제. 단 §0-4(b)로 종방향 PI 가 VtdLongitudinalController 로 살아남으므로 해당 케이스는 이식 |
| **이식(임포트 수정)** | test_reset_detect, test_speed_estimate, test_perception_range | Perception → EgoSpeedEstimator/VtdWorld 로 옮긴 로직의 기대값 유지 (검증 이력 자산) |
| **유지(경로만 수정)** | test_comm_parse, test_comm_send, test_perception(lanegraph 만 사용), test_gen_scenarios, test_batch_end | hlfma.core → vtd_adapter 임포트 치환, `load_params_yaml` 경로 치환 |

공통 수정: 대부분 테스트가 `from hlfma.nodes.params import load_params_yaml` 을 쓴다 → `vtd_adapter.config` 로 일괄 치환.

---

## 열린 질문 (사용자 결정 필요)

1. **RTOR**: PDM-Lite 는 적신호면 우회전이라도 정지 유지. (i) 그대로 수용(시간 손실, 안전) / (ii) 어댑터의 신호 매핑에서 "우회전 경로 + 완전정지 stop_dwell_s 경과 + 횡단보도 clear" 시 state 를 Green 으로 바꿔 주는 것으로 기존 RTOR 재현(판단 로직 수정 없이 매핑 계층에서). 기존 `rtor_enabled: true` 였으므로 (ii) 를 기본 후보로 두되 Phase 3 이후 결정 권장.
2. **황색 정책**: §0-5 매핑대로 Yellow→정지 시도(안전 우선)로 갈지, 잔여시간 기반으로 Green 취급 조건을 둘지. 1차는 매핑대로 진행 권장.
3. **차선변경**: 기존 planner 의 계획 차선변경(blocked 추월 포함)은 PDM 에선 route 자체 시프트(`shift_route_around_actors`)로 표현된다. Phase 4 계획대로 "우리 blocked 조건으로 발동" 하되, **계획된 경유점 차선변경**(route events 의 lane_change)은 build_route 가 경로 점열에 이미 반영하므로 PDM 은 그냥 추종 — 별도 작업 불필요함을 확인했다.

## Phase 1 진입 전 요약

- ROS 제거·이동은 기계적 작업으로 확정 (§0-7).
- autopilot.py 실수정 지점은 **_init 재작성, tick_autopilot 재작성, 종방향 호출부 2곳, VehicleControl 조립부, stub 2개, 제거 블록들** — 판단 함수(IDM/OBB/forecast/steer)는 무수정.
- 어댑터가 새로 구현할 실질 로직: VtdRoutePlanner 의 경로 재샘플 + 신호등/제한속도 배열 (§0-5), VtdWaypoint 의 next/previous/is_junction, 좌표 변환 (§0-3), VtdLongitudinalController (§0-4b), 지시등 산출 (§0-2 공백 1).
