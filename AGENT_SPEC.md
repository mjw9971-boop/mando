# HL FMA 2026 자율주행 컨트롤러 — 규격 및 설계 근거

이 문서는 CLI 코딩 에이전트와 팀원이 공유하는 **단일 사실 소스**다.
"확정 사실"은 실측·공식 답변으로 검증된 것이니 추측으로 바꾸지 말 것.
초기 스캐폴딩 지시(구 §3)는 완료되어 삭제했다 — 현재 구조는 [README](README.md) /
[docs/OVERVIEW.md](docs/OVERVIEW.md) 참조.

> **이 문서는 "목표 규격"이다.** 규격은 지우지 않는다 — 아직 코드가 없는 항목도
> 그대로 두고 상태 태그만 붙인다. 태그는 2026-08-27 코드(HEAD `a5b5578`) 기준.
>
> | 태그 | 뜻 |
> |---|---|
> | **[구현됨]** | 규격대로 동작하는 코드가 있다 (파일:라인 근거 표기) |
> | **[부분]** | 일부만. 무엇이 빠졌는지 함께 적는다 |
> | **[미구현]** | 코드 없음. 규격은 유효, 착수 안 함 |
> | **[구조 변경됨]** | 규격이 전제한 구조가 PDM-Lite 이식으로 바뀜. 원문은 이력으로 존치 |
>
> §1(확정 사실)·§6(실사고 대장)은 관측·이력이라 태그를 붙이지 않는다.

---

## 0. 배경 (한 줄)

VTD 2025.2 시뮬레이터가 9910 TCP 로 GT(ego/객체/신호등)를 주고, 우리는 조향·가속도·지시등을 되돌려준다.
도로교통법 10개 항목 + 동적 이벤트 5개로 자동 채점된다. **순수 룰베이스**. 신경망 없음.

---

## 1. 확정 사실 (실측·공식 확인 완료 — 바꾸지 말 것)

### 1.1 9910 TCP 수신 프레임

- **1109 바이트 고정, 헤더 없음, little-endian, 20 Hz** (주기는 조직위 공식 답변 + RTF 1.000 실측 일치)
- timestamp·frame ID **없음** (공식 답변) → sim 시간은 프레임 카운트로만 유도 가능
- **도착 간격은 40/80 ms 교대** (TCP 전달 지터, 유실 아님 — 2000틱 프레임 증분 전부 1 확인).
  평균은 정확히 50 ms. **고정 dt 가정 금지** — 실사고: `dt_eff = max(dt, 1/send_hz)` 하한이
  −15 % 속도 편향을 만들어 53 km/h 과속 186틱이 로그에 은폐됐다. 속도는 벽시계
  슬라이딩 창(Σ변위/Σwall_dt, `percep.speed_win_s`)으로 추정한다.
- 구조 (연속, 패딩 없음):

| 구간 | 크기 | 포맷 | 내용 |
|---|---|---|---|
| ego | 24 B | `<6f` | x, y, z, heading, pitch, roll (m, rad) |
| objects[30] | 36 B × 30 = 1080 B | `<I8f` × 30 | id(uint32), x, y, z, heading, speed, length, width, height |
| trafficLights[1] | 5 B | `<iB` | id(int32), state(uint8) |

- **빈 객체 슬롯은 전 필드 0** → `id == 0` 스킵. 선택 규칙(공식): 수평거리 80 m 이내,
  가까운 순, 최대 30개. 시야각·occlusion 필터 없음(후방 포함).
- **trafficLights 는 1칸 고정.** 없으면 `(0, 0)`.
- **id 는 xodr signal id 가 아니라 SignalController id 다** (관측 id 3/5/11/19/30/38/80/90 이
  signal 테이블에 부재, controller 계층과 일치). VTD 는 **자차 접근로 기준** controller 를
  골라 준다 — 행동 상관 3건(적색 대기→녹색 통과 등)으로 검증. 자차가 도로를 옮기면
  id 가 다음 교차로 접근로 단위로 바뀌고, 교차로 연결로·비신호 구간에서는 `(0,0)` 또는
  **필드 소멸**이 온다. **소멸 ≠ 통과 허가** — 정지 래치는 소멸로 풀지 않는다(실사고 §6).
- state: `0=비알람 1=적 2=황 3=녹 4=좌회전 5=녹+좌회전 6=점멸`
  - 5(녹+좌)는 녹색 포함 → 직진·우회전 통과가 정상 (실주행 3곳 검증)
  - 0은 제약 없음으로 처리하되 **래치 해제 조건에는 포함하지 않음**
    - ⚠ 현재 코드는 state 0 을 `TrafficLightState.Unknown` 으로 매핑하고
      ([route.py:79](vtd_adapter/route.py#L79)), PDM 은 "Green 이 아니면 감속"이라
      **0 에서 감속한다** ([autopilot.py:1426-1443](team_code/autopilot.py#L1426-L1443)).
      규격과 다르다 — §8 질문 1.
- **ego 속도 필드 없음** (공식 답변: 직접 계산). 위치 차분 + 창 추정.
- ego x, y 는 **xodr 월드 좌표와 동일**, 원점은 **뒷바퀴 축 중심** — 정지·통과 판정에는
  **전장 보정 필수**: 앞범퍼 = 뒷축 + wheelbase(2.944) + front_overhang(0.855).
- 객체 타입 필드 없음 → 크기 분류: 보행자 width 0.5–0.8·height 1.5–2.0·length<1.0,
  차량 length>3.0, 그 외+speed≈0 = 정적 장애물. id 는 프레임 간 유지(트래킹 용이).

### 1.2 9910 TCP 송신 (제어)

- **9 바이트**: `struct.pack('<ffB', steering, targetAccel, turnSignal)`
- steering [rad] (좌 +), targetAccel [m/s²], turnSignal `0=OFF 1=LEFT 2=RIGHT`
- 20 Hz 지속 송신. **`comm.steer_sign = +1.0` 실측 확정** (2026-08-19,
  corr(steering, yaw_rate) = −0.93 사고로 규명; SteerSignMonitor 가 주행 5초 내 자동 검증).
  - ⚠ **SteerSignMonitor 는 현재 저장소에 없다** — 자동 검증 **[미구현]**.
    부호 자체는 `comm.steer_sign` 으로 적용된다 **[구현됨]** [comm.py:82-92](vtd_adapter/comm.py#L82-L92).
- VTD 는 targetAccel 을 **폐루프로 실현**한다 — 6.4 % 오르막에서 명령 0.0 에 속도 유지
  실측. 중력 보상은 VTD 내부가 하므로 **pitch feed-forward 를 넣지 말 것**
  (넣으면 오르막 +2.9 km/h 과속).

### 1.3 차량 제원 (Ioniq 6)

- wheelbase 2.944 / 전장 4.848 / 전폭 1.886 / 전고 1.507 / **front_overhang 0.855**
  (`config/params.yaml` `vehicle.*` 가 단일 출처)
- 최대 조향 0.48 rad → 최소회전반경 ≈ 5.65 m. **R < 5.65 m 호는 추종 불가**
  (속도 무관 — 예시 경로 LC4 6.09 m 창, 회전1 연결로 R=4.5 m 가 해당)
- 좌표 원점 뒷바퀴 축 중심 — Pure Pursuit 는 무보정, **정지 판정은 보정**(§1.1)

### 1.4 채점 항목 (공지 캡처 확보)

법규 10: S1.1.01 제한속도 / S1.1.02 스쿨존 속도 / S2.1.01 차로 유지 / S2.1.02 중앙선·우측통행 /
S2.1.03 보도 침범 / S2.2.05 실선 차선변경 금지 / S5.1.01 적색 정지 / S5.1.03 녹색 통과 /
S6.1.01 파손·장애물 대응 / **S6.3.03 횡단보도 "정차 금지"** (속도 캡 아님 — crosswalk 25 캡은
근거 없어 제거됨).
동적 5: 보행자 출현(거리·감속·TTC·정지) / 차량 출현·급정거 / 장애물 / 방향지시등 n초 전 /
충돌. 차로유지·중앙선·보도·실선·우측통행은 **상시 측정**.
경로 ~3 km / 5분 (→ 평균 36 km/h 필요. 현 평균 ~26–31, §7-6).

> 채점 항목의 **사후 검출·감점 계산**은 `tools/score.py` 가 구현한다 **[구현됨]**
> ([score.py:120-590](tools/score.py#L120-L590), 구간별 채점 [score.py:637](tools/score.py#L637)).
> 주행 중 이 항목들을 **회피하는 로직**의 상태는 §3 을 볼 것.

### 1.5 지도·경로 자산

- xodr: 도로 651 / 주행차로 2480 / 교차로 94 / 신호 646(controller 214) /
  정지선 클러스터 277(비신호 156) / 횡단보도 400+
- 제한속도는 표준 `<speed>` **0건** — 노면표시 객체로만: `RM_517_50`=50(도로 16),
  `RM_518`=스쿨존(도로 23, 차로 230; 30 해석은 §7-5 미검증), `roadmark_speed_30`=30.
  **50 초과 마킹 없음** → 맵 최고 제한 50. 무표시 도로는 `default_speed_kph 50`
  (한국 5030 정책과 부합, 채점도 동일 xodr 기준일 것).
- 경사: 5 % 초과 180곳, 최대 39.6 % — 언덕 지형 (속도 저하는 §1.2 로 VTD 가 처리)
- 신호 좌표(x/y)는 xodr 에 없음 — 신호 검증은 controller↔정지선 매핑으로 한다.
  **빌드 매핑 규칙(같은 도로 20 m)은 한국식 배치(신호가 건너편 25 m)를 놓친다** →
  런타임 폴백 ②(§3.4)로 보완. 스쿨존 비신호 정지선 46곳에서 폴백 오정지 위험(§7-4).
- 대회 경유점 CSV 형식 (공식): 첫 점=출발, 끝 점=도착, 중간은 **교차로 진입·진출 짝**,
  짝 사이 = 교차로 내부. 경로 이탈 감점.
- lane key = `(road_id, section_idx, lane_id)`. **좌측 차로는 id 순서가 반전**된다
  (실사고: 좌우 역추론 → 차선변경 10.5 m 지연).
- route_s = cum_s + s_in_lane. **차선변경 hop 은 길이 0** — 평행 차로 중복 가산이
  route_s 52 m / lookahead 12–55 m 과대를 만든 실사고 2건. lookahead 도 동일 규칙.
- 차선변경 창은 **같은 lane 연속 section 병합 + 시작점 최대 전진**으로 산정.
  창 < 전이거리 max(3·v, 20 m) 인 경로는 물리적 실패 — 빌드 리포트가 20 m 미만 경고.

---

## 2. 아키텍처  **[구조 변경됨]**

**현재 구조** — ROS 없음. 단일 프로세스 / 단일 스레드 틱 루프 ([run_agent.py](run_agent.py)):

```
9910 ─recv─> Comm ─RawPacket─> EgoTracker (속도·차로·리셋) ─WorldState─┐
                ↑              VtdWorld   (객체 트래킹)                │
                │                                                      ▼
                │        VtdRoutePlanner / VtdMap ──> AutoPilot.run_step (PDM-Lite)
                │        VtdLongitudinalController          │  + kr_rules (한국 규칙)
                └──9 B Command── command_from_control <─────┘
                         Logger (매 틱 raw+판단 전부 jsonl)
```

- **판단은 team_code/autopilot.py(PDM-Lite 원문)가 한다.** 어댑터(vtd_adapter)는 CARLA
  표면을 흉내낼 뿐 판단하지 않는다. kr_rules 는 PDM 원문을 고치지 않고 min 중재에
  후보를 덧대기만 한다 (접점은 [autopilot.py:288](team_code/autopilot.py#L288) 한 줄).
- core/ 가 ROS 를 모른다는 원칙은 그대로 유효하다 — vtd_adapter/team_code 가
  pytest·replay·mock 에서 같은 코드로 돈다.

<details>
<summary>구 규격 (ROS 2 4노드 구조 — 2026-08-23 까지. 이력으로 존치)</summary>

```
9910 ─recv─> vtd_bridge ─/gt_state─> perception ─/world_state─> planner(+shield)
                  ↑                                                    │
                  └────────────── /cmd ────────── control <──/decision─┘
                          Logger (매 틱 raw+판단 전부 jsonl)
```

- **ROS 2, 노드별 프로세스 분리가 기본.** 단일 프로세스는 GIL 경합으로 8.8 Hz
  (드롭 67 %, 지연 169 ms) — 분리로 20 Hz / 0.1 % / 4.4 ms (실차 실측).
- 한 패킷 = 한 틱, 타이머 없이 콜백 체인. QoS depth=1 최신 우선.
- Perception 은 판단하지 않고, Control 은 교통법을 모르며, Shield 는 Planner 를 신뢰하지 않는다.

이 측정은 **ROS 노드 간 IPC 비용**에 대한 것이라 현재의 함수 호출 체인에는 적용되지
않는다. 다만 실사고 §6-2 의 교훈(파이썬 GIL 경합을 실측 없이 낙관하지 말 것)은 유효하다.
</details>

## 3. 모듈 규격 (구현 완료분의 계약)

### 3.1 Comm
스트림 재조립 후 **최신 프레임만** 사용 **[구현됨]** [comm.py:166-203](vtd_adapter/comm.py#L166-L203).
watchdog 은 수신 유무로만(적신호 대기 13–18 s 동안 속도 0 은 정상) **[구현됨]**
[comm.py:283-294](vtd_adapter/comm.py#L283-L294) + [run_agent.py](run_agent.py) 루프의 SAFE_STOP.
`/cmd` 가 `hold_decay_s`(0.15) 이상 끊기면 vtd_bridge 가 조향만 0 으로 감쇠(가속 유지 —
실사고: 풀락 0.48 이 0.32 s 유지된 채 3.5 m → 이탈) **[미구현 — 구조상 소멸]**:
노드 간 토픽이 없어져 `/cmd` 가 끊기는 경로 자체가 없다. `hold_decay_s` 키도 없다.
대응물은 틱 예외 3연속 시 SAFE_STOP(accel −3.0, steer 0) [comm.py:313](vtd_adapter/comm.py#L313).

### 3.2 Perception  → 현재는 `vtd_adapter/ego.py` + `world.py`
- 속도: 벽시계 슬라이딩 창(§1.1) **[구현됨]** [ego.py:118-183](vtd_adapter/ego.py#L118-L183).
  창 pop 은 "빼도 ≥ win_s 남을 때만"(스톨 후 버스트 방어) **[구현됨]** [ego.py:133-134](vtd_adapter/ego.py#L133-L134).
- 리셋 감지: 점프 `max(jump_m, v·dt·f)` / 환산속도 / t_off 점프 / **차로중심 스냅**
  (리스폰은 중심 0.00 복귀 — 이 조건 추가로 미검출 5건 소급 검출). 스톨(dt>0.2 s)이라도
  이동 > `stall_teleport_m`(50) 이면 리셋 처리. 리셋 틱은 속도 창·적분항·조향 이력 전부 초기화.
  **[구현됨]** [ego.py:41-116](vtd_adapter/ego.py#L41-L116), 차로중심 스냅 [ego.py:261-263](vtd_adapter/ego.py#L261-L263),
  이력 초기화는 [run_agent.py:210-216](run_agent.py#L210-L216)(world·route_index·조향·accel).
- 객체: 분류(§1.1) **[구현됨]** [world.py:42-58](vtd_adapter/world.py#L42-L58) /
  80 m·30개 규칙에 따른 유지·코스팅 **[구현됨]** [world.py:73-142](vtd_adapter/world.py#L73-L142) /
  lane 매칭·on_route·s_rel·ttc·will_enter_lane(등속 외팔 2 s) **[미구현]** —
  로그에 `None` 으로 남는다 [run_agent.py:271-274](run_agent.py#L271-L274).
  PDM 은 이 상대량 대신 OBB forecast 로 충돌을 판정한다.

### 3.3 Planner — 종방향은 후보 min()  **[부분]**
```
v_target = min( limit − margin, school/junction/blind 캡, sqrt(a_lat_max/|curv|),
                limit_ahead 선행감속, IDM(선행차), 정지 후보들, route_end )
```
min 중재 구조 자체는 PDM 이 그대로 한다 **[구현됨]** [autopilot.py:911](team_code/autopilot.py#L911).
후보별 상태:

| 후보 | 상태 | 근거 / 비고 |
|---|---|---|
| `limit − margin` | **[구현됨]** | [route.py:177-188](vtd_adapter/route.py#L177-L188) — `speed.margin_kph` 반영해 `speed_limits` 배열로 |
| junction 캡 | **[구현됨]** | [autopilot.py:199-206](team_code/autopilot.py#L199-L206) + `max_speed_in_junction` 30 km/h [config.py:138](team_code/config.py#L138) |
| school 캡 | **[부분]** | 스쿨존 도로의 제한속도(30)가 `speed_limit` 에 들어 있으면 자동 반영. `school_zone` 플래그 자체는 [route.py:183](vtd_adapter/route.py#L183) 에서 **무시**된다 (별도 캡 없음) |
| curvature (`a_lat_max`) | **[미구현]** | `a_lat_max` 키 없음. 곡률 캡 코드 없음 |
| `limit_ahead` 선행감속 | **[미구현]** | 제한속도 하향 지점 앞 미리 감속하는 로직 없음 |
| blind 캡 (사각 서행) | **[미구현]** | — |
| IDM(선행차) | **[구현됨]** | [autopilot.py:640](team_code/autopilot.py#L640) |
| 보행자·타차·자전거 | **[구현됨]** | [autopilot.py:738](team_code/autopilot.py#L738) — OBB forecast 기반 |
| 정지 후보(적신호) | **[구현됨]** | [autopilot.py:1340](team_code/autopilot.py#L1340) — IDM 감속 |
| `route_end` | **[구현됨]** | [kr_rules.py:152-200](team_code/kr_rules.py#L152-L200) — 종점 유령 선행차 + 래치 |

- **정지 프로파일**: 목표 = 앞범퍼가 (정지점 − stop_gap). 유효거리에서 전장 보정(§1.1)
  차감 + **속도 비례 선행 보상 `stop_lag_s`(0.6)** — P 추종 지연(err≈a_plan/kp≈1.3 m/s)
  상쇄. route_end 에도 동일 lag 적용. 실측: 앞범퍼가 정지선 1.10 m 앞 v=0 (오차 0.10 m).
  - 전장 보정 **[구현됨]** — 정지선: `idm_red_light_minimum_distance = stop_gap_stopline_m
    + 앞범퍼` 주입 [run_agent.py:76-88](run_agent.py#L76-L88) (2026-08-27 커밋 `a5b5578`).
    route_end: [kr_rules.py:83](team_code/kr_rules.py#L83), `d_eff = d_end − front`.
  - `stop_lag_s` 선행 보상 **[미구현]** — 키 없음. PDM IDM 이 감속 프로파일을 내고
    `control.a_dec_max` 클램프가 실행을 맞추는 방식으로 대체됐다 [control.py:59-79](vtd_adapter/control.py#L59-L79).
    실측 오버런 산포 −2.3~+2.2 m 는 `speed.stop_gap_stopline_m` 1.5 로 흡수 중.
- **신호**: 적/황(딜레마존 밖) → 정지 래치. **해제는 ① 녹색류(3/5, 4+좌회전경로)
  ② RTOR go ③ 명백 통과(정지선 +5 m) 뿐** — light 소멸·state 0 으로 풀지 않음.
  - **[부분]** — 래치 자체는 없다. 매 틱 현재 state 로 IDM 감속만 한다
    ([autopilot.py:1426-1443](team_code/autopilot.py#L1426-L1443)). 신호가 소멸하면
    `VtdTrafficLight.state` 는 직전 값을 유지하므로([route.py:363-376](vtd_adapter/route.py#L363-L376))
    실사고 §6-6(소멸 시 재가속)은 우회되지만, 규격이 요구한 명시적 래치/해제 조건은 없다.
  - **최소 정지 유지(규정 0.5 s 이상)** 는 별도로 **[구현됨]** —
    `speed.stopline_hold_s`(1.0) 홀드 [kr_rules.py:127-150](team_code/kr_rules.py#L127-L150)
    (2026-08-27. 실측 0.4 s 재출발이 감점 대상이라 도입).
- **폴백 ②**: 전방 정지선의 signal_ids 가 비어도 9910 light 유효 시 그 state 적용
  (`light_ctrl_match is False` 확인 시에만 차단). 빌드 매핑 누락(도로 30) 보완.
  - **[미구현]** — 현재는 controller_ids 에 매칭될 때만 state 를 갱신한다
    ([route.py:370-376](vtd_adapter/route.py#L370-L376)). 매칭 실패 신호는 기본값
    Green 을 유지해 **그냥 통과**한다. `check_light_controller` 는 로그 플래그만 남긴다.
- **RTOR**: 적색+`next_turn==turn_right` → 완전정지(v<0.2) → dwell 1.0 s → 통과경로·
  횡단보도·좌측 TTC 안전 확인 → `rtor_speed_kph`(20) 서행. `rtor_enabled` 스위치
  (규정 미확정 §7-3). hold 사유는 reasons 에 기록.  **[미구현]** — 관련 키·코드 없음.
- **차선변경**: 창 진입 + `lc_clear` + **지시등 연속 점등 ≥ `lc_lead_min_s`(3.0)**.
  미달 시 점등 유지하며 원차로 직진; 남은 창 < 전이거리면 창 연장 시도 후 즉시 실행
  (`lead_short` 플래그). shield 긴급 회피는 면제. 완료 이벤트는 창 s1 통과까지
  재선택 금지(재선택 깜빡임 23회 실사고). blend base 는 **LC 시작 시점 원차로로 고정**
  — 매칭 전환 시 base 교체가 경계 요동(1.2 s, ±0.36) 실사고.
  - **[부분]** — 차선변경 **경로**는 빌드 시점에 확정된다: `VtdRoutePlanner._build` 가
    LC 이음매를 최대 25 m 코사인 블렌드로 이어 붙이고 그 구간 command 를
    CHANGELANELEFT/RIGHT 로 둔다 ([route.py:126-131](vtd_adapter/route.py#L126-L131)).
    base 고정 문제는 경로가 정적이라 구조적으로 사라졌다.
  - **[미구현]** — 런타임 판단: `lc_clear`(측방 안전 확인), 창 연장·`lead_short`.
    즉 **옆차로에 차가 있어도 그대로 진입한다.** (지시등 lead 는 아래 항목에서 구현됨)
- **지시등**: 회전 = lookahead turn 이벤트 lead 4 s 전 점등, 연결로 끝(end_s)까지 유지.
  LC 와 겹치면 **거리 짧은 쪽 우선, 동률 회전 우선**. `sig_src`/`sig_lead_s` 로그.
  - **[구현됨]** (2026-08-27) — `kr_rules` 가 판단하고 run_agent 가 싣는다:
    점등 구간은 `route['events']` 에서 시작 시 1회 계산
    ([kr_rules.signal_intervals](team_code/kr_rules.py#L109)), 매 틱 선택은
    [kr_rules._turn_signal](team_code/kr_rules.py#L188), 배관은
    [run_agent.py:236-237](run_agent.py#L236-L237)(Command)·[:277-279](run_agent.py#L277-L279)(Decision).
    회전은 연결로 끝까지 유지하고, 겹치면 남은거리 짧은 쪽·동률이면 회전 우선.
    `sig_src`/`sig_lead_s` 를 `decision.reasons` 에 남긴다.
  - 규격과 다른 점 둘: ① lead 판정을 시간이 아니라 **거리**로 한다
    (`max(v·lead_s, signal.lead_min_m)` — v→0 에서 시간 기준은 선행거리가 0 이 되어
    적신호 대기 중 회전 지시등이 안 켜진다) ② `lc_clear` 미구현이라 LC 점등은
    측방 안전과 무관하게 창 기준으로만 켜진다.
  - 실사고 §6-9(재선택 깜빡임)는 **구조적으로 해소** — 경로가 정적이라 점등 구간이
    불변이고, 지난 구간은 후보에서 빠진다 (`tests/test_turn_signal.py`).

### 3.4 Shield  **[미구현 — 모듈 없음]**
corridor·실선·중앙선 가드, TTC 비상제동(1.5 s, 저크 해제). §7-7: 횡단보도 보행자는
회피(AVOIDING) 대상이 아니라 **정지선 대기 대상** — 우선순위 정리 구현 중.

> 현재 저장소에 shield 에 해당하는 모듈이 없다. PDM 의 OBB forecast 충돌 회피가
> TTC 비상제동 역할을 일부 대신하지만, **Planner 를 불신하는 독립 감시 계층은 없다.**
> corridor·실선·중앙선 가드는 주행 중 강제하는 코드가 없고 `tools/score.py` 의
> 사후 검출만 있다 ([score.py:170](tools/score.py#L170), [score.py:237](tools/score.py#L237)).
> 규격은 유효하다 — 다시 만들 때 이 절을 계약으로 쓴다.

### 3.5 Control
- 횡: Pure Pursuit, `L_d = clamp(0.8·v, 5, 20) / (1 + k_curv·|curv|)` (곡률 축소 —
  코너 안쪽 잘라먹기 방지, `ld_curve_min` 3). 조향 변화율 1.0 rad/s.
  - **[구조 변경됨]** — PDM-Lite 의 `LateralPIDController`(속도 비례 lookahead + PID)로
    대체됐다 [lateral_controller.py](team_code/lateral_controller.py),
    [autopilot.py:308-355](team_code/autopilot.py#L308-L355). 상수는 `team_code/config.py`.
    조향 변화율 제한은 **[미구현]** (`max_steer` 클램프만: [control.py:106-118](vtd_adapter/control.py#L106-L118)).
- 종: PI + 저크 제한. **적분은 |err| ≤ `ki_band`(1 m/s) 에서만** (오버슛 28.4 실사고),
  포화 시 와인드업 되돌림, 정지 유지 `a_hold`(−1.0). E_STOP 은 저크 해제.
  - **[부분]** — `VtdLongitudinalController` 는 **P 만** (kp 0.8) + jerk 제한 +
    감속 클램프 `a_dec_max` + `a_hold` [control.py:26-104](vtd_adapter/control.py#L26-L104).
    적분항이 없으므로 `ki_band`·와인드업 되돌림은 **[미구현이자 불필요]**.
    9910 은 targetAccel 을 폐루프로 실현하므로(§1.2) throttle 회귀 대신 accel 직출력.
- 리셋 플래그 수신 시 전 이력 초기화 **[구현됨]** [run_agent.py:210-216](run_agent.py#L210-L216).

### 3.6 Logger  **[구현됨]**
매 틱 raw(ego·objects·lights) + WorldState 요약 + Decision(reasons) + Command 를
jsonl 한 줄로 [logger.py](vtd_adapter/logger.py). `tools/replay.py`·`summarize_run.py`·
`score.py` 의 계약이다.
- ⚠ `SPEED_CANDIDATES` [logger.py:23](vtd_adapter/logger.py#L23) 는 구 planner 후보명
  (limit/limit_ahead/school_zone/curvature/junction/stop_line/crosswalk_ped/route_end/
  visibility/lead)이라 대부분 항상 `null` 로 깔린다. 실제 값은 PDM 후보명
  (leading/vehicle/bicycle/pedestrian/red_light/route_end)으로 덮인다 —
  **로그에 두 세트가 공존한다** [run_agent.py:229-258](run_agent.py#L229-L258). §8 질문 3.

## 4. 코딩 규칙 (불변)

- Python 3.10+, numpy/scipy. 딥러닝 금지. **[유효]**
- **모든 튜닝 상수는 params.py DEFAULTS → params.yaml 생성, 단일 출처.**
  두 파일 수동 이원화 금지(steer_sign 회귀 실사고). yaml 수정 후 colcon build 필수.
  - **[변경됨]** 현재 규칙: 코드 안의 DEFAULTS dict 은 **폐지**됐고
    ([config.py:1-9](vtd_adapter/config.py#L1-L9)), yaml 키가 없으면 조용히 기본값으로
    도는 대신 즉시 죽는다. 빌드 단계가 없으므로 colcon build 도 불필요.
  - 출처는 **둘**이고, 경계가 규칙이다:
    `config/params.yaml` = VTD 연결·차량 제원·추정기·종방향 한계·경로·배치·채점 임계 /
    `team_code/config.py`(GlobalConfig) = PDM 판단 상수(IDM·forecast·OBB·lateral PID).
    같은 값을 양쪽에 두지 말 것 — 겹치는 것은 주입으로 잇는다
    (예: [run_agent.py:76-88](run_agent.py#L76-L88) 이 정지선 gap 을 PDM 에 넣는다).
- 내부 단위 m·m/s·m/s²·rad. km/h 는 입출력·로그만. **[유효]**
- 틱 내부 예외는 잡아서 로깅 + 직전 Command 유지 **[구현됨]** [run_agent.py:322-336](run_agent.py#L322-L336).
- 로그(logger.py)는 raw 포함 전 필드 유지 — replay·summarize 의 계약. 필드 삭제 금지. **[유효]**
- 검증 사다리: pytest → tools/replay.py(개루프) → mock/폐루프 시뮬 → 실차.
  수치 주장은 실측 로그 근거를 함께 남길 것. **[유효]** (`run_agent.py --replay`,
  `tools/mock_vtd.py`, `tools/batch_run.py`)
- **PDM-Lite 원문(team_code/autopilot.py)은 무수정 유지.** 수정한 줄에는 전부 `# VTD:`
  주석을 단다. 한국 규칙은 `kr_rules.apply` 한 접점으로만 개입한다. **[신규 규칙]**

## 5. 파라미터

`config/params.yaml` 이 VTD·어댑터 상수의 유일한 값 목록이다 (이 문서에 사본을 두지
않는다 — 이원화 방지). 근거가 있는 값에는 yaml 주석으로 실측 날짜·사고 번호를 남긴다.
PDM 판단 상수는 `team_code/config.py` 가 단일 출처다 (§4).

## 6. 실사고 대장 (같은 실수 반복 금지)

| # | 사고 | 교훈 |
|---|---|---|
| 1 | steer_sign 반전 → 직선 발산·리스폰 | 부호는 실측+자동 감시(SteerSignMonitor) ※ 감시는 현재 미구현 |
| 2 | 단일 프로세스 8.8 Hz | GIL — 분리가 기본 ※ ROS 노드 IPC 기준. 현 구조는 단일 프로세스이므로 재측정 필요 |
| 3 | route_s/lookahead 평행차로 중복 가산 | LC hop = 0, 두 곳 모두 |
| 4 | dt 하한 → 속도 −15 % 편향, 과속 은폐 | 벽시계 창 추정, 고정 주기 가정 금지 |
| 5 | 좌측 차로 id 역순 미인지 | lane id 부호·순서 규칙 명시(§1.5) |
| 6 | light 소멸 시 무조건 언래치 → 감속 중 재가속 3/3 | 소멸 ≠ 허가 |
| 7 | 뒷축 기준 + 보정 없음 → 앞범퍼 3–5 m 침범 | 정지 판정은 전장 보정 |
| 8 | P 추종 지연 → 스냅 시점 3.3 m/s 잔속 | stop_lag_s 선행 보상 ※ 현재는 a_dec_max 클램프로 대체 |
| 9 | LC 완료 미기억 → 지시등 깜빡임 23회·88 % 점등 | 완료 집합 + 래치 리셋 ※ 점등 구간을 경로에서 1회 확정하는 현 방식으로 구조적 해소 |
| 10 | blend base 매칭 전환 → 경계 요동 | base 를 원차로 고정 ※ 경로 정적 확정으로 구조적 해소 |
| 11 | 횡단보도 보행자를 장애물 회피 → 도로 중앙 급정지 | §7-7 수정 중 |
| 12 | 횡단 마친 보행자가 crosswalk 폴리곤(반폭 8 m)에 영원히 잔류 → 172 s 정지 고착·완주 실패 | ① 차도 밖+정지+진입예측 없음이면 blocker 제외(차도 폭은 lanegraph.roadway_edges — 반대차로 포함) ② 시나리오 생성 횡단 폭도 같은 함수 사용 (2026-08-25) |
| 13 | 종점 도달 후 v_target 6.9 로 계속 주행 → 경로 이탈·courseRespawn 9회 | CARLA 리더보드엔 종점 정지 개념이 없다 — kr_rules.route_end 로 보완 (2026-08-26) |
| 14 | 소멸(테이퍼) 차로 중심선 킹크 → 조향 풀포화·차선이탈 진동 | 꼬리 `route.taper_blend_m` 코사인 블렌드 (2026-08-26) |
| 15 | 적신호 정지 후 0.4 s 만에 재출발 | 규정 "0.5 s 이상 정지" — `speed.stopline_hold_s` 홀드 (2026-08-27) |

## 7. 미확인·미해결 (가정은 전부 params 로)

| # | 내용 | 관련 |
|---|---|---|
| 1 | 점멸(state 6) 황/적 구분 | `signal.flash_mode` — **키 없음**. 현재 [route.py:84](vtd_adapter/route.py#L84) 에서 Green 하드코딩. 조직위 문의 |
| 2 | 지시등 선행 "n초" 규정값 | `signal.lc_lead_s` 3.0 / `signal.turn_lead_s` 4.0 (가정값). 규정 확정되면 yaml 만 고치면 된다 |
| 3 | RTOR 허용 여부 | `signal.rtor_enabled` — **키 없음**. 미구현 |
| 4 | 스쿨존 비신호 횡단보도 일시정지 채점 여부 | 폴백 오정지 위험 46곳과 상충 — 규정 후 결정. 폴백②도 현재 미구현 |
| 5 | RM_518 = 스쿨존 30 | 텍스처 미검증. `build_lane_graph` [SPEED_MARKS](tools/build_lane_graph.py#L680) 가 30 으로 가정 |
| 6 | 5분 초과 처리 | `speed.margin_kph` 전략 직결 (현 평균 26–31 km/h). 검출은 `score.time_limit_s` |
| 7 | 횡단보도 보행자 → 정지선 정지 + shield 우선순위 | **미구현**. 현재는 PDM forecast 충돌 회피로만 처리 |
| 8 | 라우터: 연속 LC 이격 / waypoint 짝으로 진출차로 특정 | 미착수 |
| 9 | junction 6 신호가 대회날 켜지는지 | 연습 환경 state=0 고정 — 당일 첫 수 초 확인. state 0 처리는 §8 질문 1 |
| 10 | `scoring.finish_xy` 미확보 | null 이면 route_s 임계 폴백 + 경고 [kr_rules.py:104-125](team_code/kr_rules.py#L104-L125) |

## 8. 규격 vs 코드 — 판단 보류 (팀 확인 필요)

아래는 "규격이 바뀐 것"인지 "단순 미구현"인지 문서만으로 판정할 수 없어 **고치지 않고
남겨 둔** 항목이다.

1. **신호 state 0 의 의미.** §1.1 은 "0은 제약 없음으로 처리"라고 못박았는데, 코드는
   0 → `Unknown` → (Green 아님) → IDM 감속이다 ([route.py:79](vtd_adapter/route.py#L79)).
   연습 환경에서 junction 6 이 state 0 고정이었다는 §7-9 를 감안하면 대회날 0 이 오는
   구간마다 서게 될 수 있다. **의도적 보수 설계인가, 매핑 실수인가?**
2. **미매칭 신호의 기본값.** `VtdTrafficLight.state` 기본이 Green(= 통과)이다
   ([route.py:87](vtd_adapter/route.py#L87), "미수신 = 진행"). §1.1 의 "소멸 ≠ 통과 허가"
   와 방향이 반대로 보인다. 폴백②(§3.4) 미구현과 묶여 있어 함께 결정할 문제다.
3. **로그 `reasons` 후보명 이원화.** 구 planner 후보명을 계속 null 로 남길 것인가
   (옛 로그와의 호환), PDM 후보명으로 갈아탈 것인가. `summarize_run`·`score` 가
   어느 이름을 읽는지에 달려 있다.
4. **Pure Pursuit 규격(§3.5 횡)** 을 폐기 처리할 것인가, LateralPID 를 잠정으로 보고
   되돌릴 여지를 남길 것인가. 지금은 "구조 변경됨"으로만 표시했다.
