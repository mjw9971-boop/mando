# 논문 Ⅳ장 실험 — 시나리오 세트 · 실행 · 집계 · won 브랜치 체크리스트

2026-10-02 기준 (2026-10-04 A2 추가). 비교 대상은 둘이고, **둘 다 `paper_exp` 에서 돌린다**.

| 구현 | 무엇 | 이름 (표·그림 범례) |
|---|---|---|
| `WON24` | won_24 제어기 (ctrl24) — params 주행 값 그대로 | 최종 구현(won_24) / Final implementation (won_24) |
| `A2` | won 제어기 — 같은 브랜치에서 `ctrl24.enable: false`(kr_rules) + won 의 주행 값 3개. **won 과 틱 단위 동일** (§2a) | 초기 구현(won) / Initial implementation (won) |
| (`WON`) | won 브랜치(`origin/won` 9b05f1f)에서 직접 돌린 로그 — **A2 로 대체해 쓰지 않는다** (§3 은 참고로 남긴다) | 초기 구현(won 브랜치) |

`paper_exp` = won_24 `70160e9` + 실험 도구(생성기 · 집계기 · `config/arms` · 이 문서 ·
`data/paper_routes`). **제어기 파일(`team_code/` · `vtd_adapter/` · `run_agent.py`)은 won_24 와
같다** (`git diff won_24 paper_exp -- team_code vtd_adapter run_agent.py` 가 비어 있다).
won_24 브랜치 자체는 그대로 둔다.

**최종 구현의 주행 파라미터는 바꾸지 않는다** (WON24 사본은 배치 판정값만 다르다). 초기 구현 A2 는
won 의 주행 값 세 개를 overlay 로 되돌릴 뿐이고, 그 밖의 키는 `make_arms.py` 가 거부한다
(`arms.yaml` 의 `allowed_sections: [batch]` + `allowed_keys` 네 개).

## 1. 시나리오 세트 — 한 줄로 재현

```
python3 tools/gen_scenarios.py 논문1_기본 논문2_근접정지 논문3_앞뒤연속 논문4_나란히 \
    논문5_지그재그 논문6_옆차로교통 논문7_오탐_급정거 논문7_오탐_신호줄 논문8_일반주행 \
    --seed 0 --vtd-dir /home/mjw/mando/scenarios --coverage-report --manifest
```

- 출력은 `scenarios/` (비어 있어야 한다 — 남아 있는 `coverage_history.json` 이 경로 추첨을 바꾼다).
- **86개** · 목록·XML·CSV·정의·판정 기록 354개 파일. 확인: `cd scenarios && sha256sum -c MANIFEST.sha256 --quiet`
- `MANIFEST.sha256` 자체의 해시: `56599fe567196379e938e8d135ae1ae325f266d520d6a8c5090c5f9dc38ec1b2`
  (2026-10-03 신호줄 재생성 뒤). 이전 세트 `4ada8e2d…` 와는 **신호줄 10개와 `batch_all.json` 만
  다르고**, 나머지 8개 유형의 파일·목록은 바이트 동일함을 확인했다.
- 같은 명령을 다른 폴더에 다시 돌려 354개 파일이 전부 같음을 확인했다 (2026-10-02, 2026-10-03).

### 2026-10-03 신호줄 재생성

첫 본 실행에서 신호줄 10런 중 3런(02·03·06)이 시나리오 원인 접촉이었다 — 대기 줄이 경로의
차로변경 창 안 출발 차로에 놓여 자차가 줄 옆에 섰고, 풀린 대기 차량이 VTD 운전자 모델대로
목표 차로로 꺾여 들어왔다. 실패한 것만 고르지 않고 **생성 규칙을 고쳐 유형 전체를 다시 만들었다**:

- 대기 줄(뒤끝 − `queue_lc_clear_m` 10 m ~ 정지선)이 경로의 차로변경 창과 겹치는 신호 접근은 쓰지
  않는다 (`gen_placement.paper_events.queue_lc_clear_m`, null = 이전 동작).
- 대기 차량은 자기 차로를 따라 교차로를 가장 곧게 지나는 PathShape 로 `queue_release_path_m` 80 m
  간다 (0 = 이전 동작 — 경로 없이 속도만 풂).
- 새 10개 모두 `build_route.py` CLI 경로 기준으로 차로변경 창과 겹치지 않고, 대기 차량이 경로 차로
  위에 있음을 확인했다. 대수 분포(1대 4 · 2대 3 · 3대 3)는 이전과 같다.
- 이전 신호줄 로그(WON24 10런)와 그 시나리오 파일은 지우지 않고
  `logs/paper/archive/set_4ada8e2d_논문7_오탐_신호줄/` 로 옮겼다 (`WON24_논문7_오탐_신호줄/`,
  `scenarios/` — 이전 MANIFEST 포함). 집계에는 들어가지 않는다.
- **won 은 이 새 세트로 돌린다** (`MANIFEST.sha256` 해시 `56599fe5…`).
- 생성기 상수 2개가 `config/params.yaml` 에 들어가 기준 params 해시가 바뀌었고 `config/arms/WON24.yaml` 을
  다시 만들었다. 이전 사본과 다른 키는 `gen_placement.paper_events.queue_lc_clear_m` ·
  `queue_release_path_m` · `arm.base_sha256` 뿐이다 (주행 키 동일). 그래서 재주행 런(신호줄 10 ·
  일반주행 2)은 나머지 런과 커밋·기준 params 해시가 달라 집계에 `[재현]` 경고가 뜬다 — 예상된 것이다.
- 개수·조건은 `configs/themes.yaml` 의 `논문*` 테마가, 상수는 `config/params.yaml` 의
  `gen_placement.{avoid_geom, route_gates, paper_events, stopped_car}` 가 정한다.
  **이 둘과 `data/lane_graph.pkl` · `origin/won`(9b05f1f) 이 바뀌면 세트가 달라진다.**

### 유형

| 유형 | 조건 × 곳 | 시나리오 | 조우 | 장애물 | 장애물이 놓인 road |
|---|---|---|---|---|---|
| 논문1_기본 | 직진 5 · 회전 직후(출구 뒤 25~60 m) 5 | 10 | 10 | 10 | 10 |
| 논문2_근접정지 | 거리 8 m · 30 m × 5 | 10 | 10 | 10 | 9 |
| 논문3_앞뒤연속 | 간격 10 · 20 · 30 · 45 m × 3 (2~3대) | 12 | 12 | 28 | 9 |
| 논문4_나란히 | 열린 쪽 좌 · 우 × 4 | 8 | 8 | 16 | 6 |
| 논문5_지그재그 | 간격 30 · 45 · 60 m × 4 | 12 | 12 | 24 | 11 |
| 논문6_옆차로교통 | 차간 20 · 40 m × 5 | 10 | 10 | 10 | 7 |
| 논문7_오탐_급정거 | 앞차 급정거 | 10 | — | — | — |
| 논문7_오탐_신호줄 | 적색 정지선 앞 대기 1~3대 | 10 | — | — | — |
| 논문8_일반주행 | 실경로 | 4 | — | — | — |
| 합계 | | **86** | 62 | 98 | 33 (경로가 지나는 road 225) |

- 조건마다 **서로 다른 도로**에 놓는다 (한 조건 안에서 같은 road 금지). 조건이 달라지면 같은 road 가
  다시 나올 수 있다.
- 전부 목표 개수를 채웠다 (부족분 없음). 나란히는 자차가 가운데 차로인 자리만 썼다.
- 일반주행은 `route_example.csv` 가 저장소에 없어 **5개가 아니라 4개**다.

### 생성 단계 게이트 (걸리면 버리고 백필 — 손으로 교체하지 않는다)

| 코드 | 조건 | 스위치 |
|---|---|---|
| A | 이웃 차로가 회피 창 전체에 이어진다 | `avoid_geom.neighbor_run_enable` |
| E | 장애물이 출발점에서 81.56 m 이상 (근접 정지만 예외) | `avoid_geom.start_clear_enable` |
| S | 그 유형이 요구하는 차로 구조 (가운데 차로 · 유일한 옆 차로 · 같은 차로 일렬 …) | 유형 정의 |
| T | 종료선 뒤 꼬리 ≥ 12 m | `route_gates.tail_enable` |
| R | 기하 최소회전반경(5.65 m) 미만 연결로 없음 | `route_gates.sharp_turn_enable` |
| C | 종료선 콘이 회피 회랑 밖 | `route_gates.cone_enable` |
| B | **배치가 쓰는** `build_route.py` CLI 가 장애물 차로를 지난다 | `route_gates.builder_enable` |
| W | **won 의** `build_route.py`(9b05f1f) 도 장애물 차로를 지난다 | `route_gates.other_builder_enable` |

생성 실측 (시도 164 → 최종 86): S 38 · A 3 · E 1 · T 2 · R 6 · C 2 · W 1 · 기타 25
(기타 = 2차로 구간 없음 · 구간 부족 · 같은 조건에 이미 쓴 도로). D·F 는 꺼져 있다.

### 근접 정지 거리 — 정의와 won 기준 환산

거리 = **자차 앞범퍼 ~ 장애물 뒤끝**. 생성기는 장애물 중심을
`(축거 2.944 + 앞 오버행 0.855) + 거리 + 차량 길이 4.394/2` 에 놓는다 (경로의 route_s 0 = 자차 뒷축).

won 의 회피 기각식 (`kr_rules._try_overtake` 의 geom 게이트):
`필요 = 전이 max(overtake.transition_m 12.0, shift_k_s 3.0 × v) + 시작 여유 + shift_geom_margin_m 2.0`,
시작 여유는 `shift_ahead_m` 5.0 (BREAKOUT 3단계 이상이면 `shift_ahead_l3_m` 1.0).
이 거리는 **뒷축 ~ 장애물 중심**으로 잰다 — 앞범퍼 기준으로는 6.0 m (3.80 + 2.20) 를 뺀다.

| 기준 | 뒷축 ~ 장애물 중심 | 앞범퍼 ~ 장애물 뒤끝 |
|---|---|---|
| won 기각 (정지 상태) | 19.0 m | **13.0 m** |
| won 기각 (BREAKOUT 3단계 완화 뒤) | 15.0 m | 9.0 m |
| won_24 전이가 장애물 전에 끝나는 최소 (`shift_ahead 0.5 + trans_min 8.0 + extra_before 1.0`) | 11.7 m | 5.7 m |

그래서 **8 m** (won 은 어느 단계에서도 기각, won_24 최소보다 2.3 m 김) 와 **30 m** (기준 위) 다.
실제 출발 위치는 VTD 스폰에 따라 ±0.5 m 어긋날 수 있다 (2026-10-01 런: 첫 틱 route_s 0.50).

### 실경로 (유형 8) 의 출처

| 경로 이름 | 파일 | sha256 |
|---|---|---|
| 실경로_01 | `data/paper_routes/실경로_01_PathShape03.csv` | `9649c8866f5f4a66a5f6a1d93808130a05d7e20ce69453e3121753034ecf55ff` |
| 실경로_02 | `data/paper_routes/실경로_02_venue0903.csv` | `9bed1f07d921ddd25f9b2c5dbe83eee0c1ebd160ac65b78d87751568f6db3a58` |
| 공식경로 | `data/official_route.csv` | `2c7e6d0dace07829e25bb25ac5b1c75c339424ce9bdfe3688780a99bc3b1ecab` |
| 기본 | `waypoints.csv` | `119a9b99ee3cb73bdbf4c63aeff223e2d2c5e12bb8946e08075f1da5ac1a9881` |

실경로_01·02 는 휴지통의 2026-09-10 사본(`~/.local/share/Trash/files/실전주행_교통류.13/`, 세 사본이
해시 동일)을 `data/paper_routes/` 로 복원한 것이다. `route_example.csv` 는 없다 (pkl 만 있다).

실경로는 주어진 경로라 게이트로 버리지 않고 검사 결과만 기록한다 (`<이름>.placement.json`).

| 시나리오 | won_24 빌더 | won 빌더 | 같은 차로열 | 급회전 | 종료선 콘 여유 |
|---|---|---|---|---|---|
| 논문8_일반주행_01_기본 | 2470.4 m | **6531.4 m** | **아니오** | R 4.88 · 4.94 · 5.65 m | −0.003 m |
| 논문8_일반주행_02_실경로_02 | 793.1 m | 793.1 m | 예 | 없음 | **−0.701 m** |
| 논문8_일반주행_03_공식경로 | 1107.6 m | 1107.6 m | 예 | 없음 | 0.397 m |
| 논문8_일반주행_04_실경로_01 | 896.7 m | 896.7 m | 예 | 없음 | **−0.309 m** |

- **`기본`(waypoints.csv) 은 두 빌더가 다른 경로를 만든다** — 브랜치 간 비교에서 뺀다
  (자동 — §4 비교 제외 규칙, 길이 차 164 %).
- 콘 여유가 음수인 경로는 종료선 콘이 회피 회랑 안이다 — 콘을 장애물로 잡을 수 있다.
  일반 주행의 감점·주행 시간을 읽을 때 종료 직전 거동을 따로 볼 것.

### 신호 대기 줄 — 실주행 확인 1건 (2026-10-02)

대기 차량은 신호를 지키는 운전자 모델에 기대지 않는다: 적색 동안 강제 정지, **녹색 시각에
30 km/h 로 풀린다** (신호는 적색으로 시작해 그 시각에 녹색). 시험 1개를 VTD 에서 돌려
(`logs/paper/queue_test/run/`) 대기 차량이 녹색 전환과 같은 시각에 출발해 8.33 m/s 까지 가는 것을
확인했다. 그 런에서 최종 구현은 대기 차량 80 m 앞에서 시프트를 만들어 옆 차로로 비켜
정지선에 섰다 — **이 유형이 재려는 오탐이 실제로 나온다** (완주, 충돌 없음).

### 옆 차로 교통 · 실경로 종료선 콘 — 실주행 확인 3건 (2026-10-02, `logs/paper/check/run/`)

- **옆 차로 교통** (`논문6_옆차로교통_01_직진0`): 설계대로 맞는다. 줄(5대, 간격 24.4 m, 2.8 m/s)은
  자차가 장애물 159 m 전일 때 출발했고, 자차가 장애물 75 m 전부터 지나갈 때까지 옆 차로에
  4~5대가 장애물 앞뒤 −88 ~ +63 m 에 걸쳐 흐르고 있었다. 자차는 장애물 10 m 앞에서 0.5 m/s 까지
  줄였다가 두 차량 사이(24 m)로 들어가 지나갔다 — 완주, 충돌 없음, 최소 간격 0.69 m.
  `paper_events.traffic_*` 는 조정하지 않았다.
- **실경로_02** (콘 여유 −0.70 m): 우측 종료선 콘을 장애물로 잡는다 (`NOOP right:span_no_room`).
  콘 3.6 m 앞에서 **약 20 s 정지**한 뒤 탈출 바닥이 걸려 지나가며 **콘에 닿았다** (간격 0.0 m — 충돌 1,
  항목 11). 완주, 주행 157 s, 감점 −24.
- **실경로_01** (콘 여유 −0.31 m): 같은 양상 — 우측 콘을 장애물로 잡고(`NOOP right:no_target`)
  3.1 m 앞에서 **약 3 s 정지**, 탈출 바닥으로 지나가며 **콘에 닿았다** (충돌 1). 완주, 141 s, 감점 −18.
- 시나리오는 바꾸지 않았다. 일반 주행의 감점·충돌 수에는 이 종료선 콘 접촉이 들어 있다 —
  `collisions.csv` 의 `obj_kind = finish_cone`, `plain.csv` 의 `collision_kinds` 로 갈라 본다.

### 실주행으로 확인하지 못한 것

- **근접 정지(유형 2)** 의 실제 범퍼 거리 (스폰 오차 ±0.5 m).
- 유형 3·4·5 (앞뒤 연속 · 나란히 · 지그재그) 는 배치 좌표만 지도와 대조했다.

## 2. paper_exp 에서 실행·집계

```
python3 tools/make_arms.py            # params.yaml + overlay → config/arms/WON24.yaml · A2.yaml
python3 tools/paper_run.py --dry-run  # 계획: 9배치 × (WON24, A2)
python3 tools/paper_run.py            # 실행. logs/paper/<구현>/<유형>/, 끝난 런은 건너뛴다
python3 tools/paper_figs.py --arm WON24=logs/paper/WON24 --arm A2=logs/paper/A2   # 표·그림·초안
```

- 사본 `config/arms/WON24.yaml` 이 params 와 다른 키는 `batch.stall_end_s` · `batch.blocked_end_s`
  (99999 → 60.0) 둘뿐이다. `A2.yaml` 은 거기에 아래 §2a 의 네 키가 더 다르다. `make_arms.py` 는
  `batch` 밖에서는 `allowed_keys` 의 네 키만 허용하고 나머지는 거부한다.
- **params.yaml 을 바꾸면 사본을 다시 만든다** (`paper_run` 이 낡은 사본을 잡아 중단한다).
- `logs/paper/A1/` 은 계획 변경 전에 돌다 멈춘 것이다 (끝난 런 0건). 지우지 않고 두되 집계에 넣지 않는다.

### 2a. A2 — 초기 구현을 이 브랜치에서 돌리는 근거 (2026-10-04)

**코드.** `origin/won` 과 `paper_exp` 의 제어기 쪽 차이는 4개 파일이고, kr_rules 경로에서는 전부 동작이 같다.

| 파일 | 차이 | kr 경로 (`ctrl24.enable: false`) |
|---|---|---|
| `team_code/kr_rules.py` · `config.py` · `vtd_adapter/`(route.py 제외) · `data/lane_graph.pkl` | 없음 | — |
| `team_code/autopilot.py` | `pre_pass` 훅 8줄 | 건너뛴다 — `KrRules` 에 `pre_pass` 가 없다 |
| `vtd_adapter/route.py` | 종점 패드 · 시프트 목표 · 가상 시프트 | 전부 `ctrl24.enable` 로 막히거나 기본 인자일 때 원문과 같다 |
| `run_agent.py` | 리셋 때 `reset_index` 순서 · 로그용 결정 조립 순서 · 예외 메시지 | `KrRules.on_reset` 은 플래너를 안 건드린다 · 나머지는 로그 표기 |
| `team_code/ctrl24.py` | 신규 | 불리지 않는다 |

**params.** kr 경로가 읽는 키 중 won 과 값이 다른 것은 셋이다 — A2 overlay 가 won 값으로 되돌린다.

| 키 | won → won_24 | 읽는 곳 |
|---|---|---|
| `speed.margin_kph` | 5.0 → 6.0 | `vtd_adapter/route.py` 목표 속도 = 제한속도 − 여유 |
| `speed.approach_decel_mps2` | 2.0 → 1.5 | `kr_rules` 적신호 접근 감속도 |
| `overtake.a_lat_max` | 1.5 → 1.0 | `kr_rules` 시프트 중 곡률 속도 상한 |

그 밖의 차이(`ctrl24.*` 94 · `gen_placement.*` 33 · `batch.*` · `comm.host` · `route.*` 6)는 kr 경로가
읽지 않는다. `route.*` 는 경로 빌더(`tools/build_route.py`)만 읽는다.

**리플레이 (개루프, `run_agent.py --replay`).** WON24 로그 7개 — 기본_01 · 근접정지_01 · 나란히_03 ·
옆차로교통_01 · 급정거_01 · 실경로_02 · 신호줄_01(이전 세트) — 를 같은 경로 pkl 로 돌려 틱마다
`v_target` · `steering` · `accel` · winner · 지시등을 비교했다.

| 비교 | 결과 |
|---|---|
| won 코드 vs paper_exp 코드 (같은 won params) | 7로그 차이 0 — 코드 동일 |
| won vs paper_exp params + `ctrl24.enable: false` 만 | 7로그 전부 다름 (v_target 371–1237틱, steering 최대 243틱) |
| won vs 위 + `margin_kph` · `approach_decel_mps2` 만 won 값 | 2로그 다름 (`a_lat_max` 미반영) |
| won vs **`config/arms/A2.yaml`** | **7로그 · 12,007틱 차이 0** |

(로그 사이에 끼는 `gc_pause` 같은 이벤트 줄은 틱이 아니라 비교에서 뺐다.)

**경로.** 두 구현 모두 batch_run 이 이 브랜치의 `build_route.py` 로 경로를 짓는다 — 같은 시나리오면 같은
경로다. won 빌더와 경로가 달라 빠지던 5개(`논문8_일반주행_01_기본` · `논문3_앞뒤연속_03·06` ·
`논문4_나란히_04` · `논문7_오탐_급정거_04`)를 다시 지어 WON24 런의 경로와 차로열·길이가 같음을
확인했다. 그래서 `arms.yaml` 에 구현별 `builder: paper_exp` 를 적고, `paper_report` 는 **구현끼리 빌더가
다를 때만** 빌더 차이 제외를 한다 — WON24 · A2 비교에서는 86개 전부 남는다.

**한계.** A2 는 "won 제어기 + 최종 경로 빌더" 다. won 브랜치를 통째로 돌린 것과는 경로 생성만 다르다
(위 5개에서 차로열이 다르다). 비교는 제어기의 차이만 본다.

**논문에 쓸 설명 (제안).**

> 초기 구현은 최종 구현과 같은 코드베이스에서 회피 제어기를 끄고 초기 규칙 계층과 당시의 주행
> 파라미터 세 개(목표 속도 여유 5 km/h, 적신호 접근 감속도 2.0 m/s², 시프트 횡가속 상한 1.5 m/s²)로
> 실행하였다. 이 구성이 초기 구현 브랜치와 같은 명령을 내는지 개루프 리플레이 7개 로그(12,007틱)로
> 확인하였으며 조향·가감속 명령이 모든 틱에서 일치하였다. 두 구현은 같은 경로 생성기로 만든 같은 경로를 주행한다.

### 재현 정보 — 어디에 무엇이 남는가

| 무엇 | 런마다 (`logs/paper/WON24/<유형>/report.json`) | 집계 (`meta.json`) |
|---|---|---|
| 코드 | `git` — `commit`(40자) · `branch` · `dirty` · `modified` · `untracked`. **런을 띄우기 직전**의 값 | `git` (집계한 시점) · `arms.<구현>.settings.code` (런이 실제로 돈 코드, 종류별 런 수) |
| 구현 설정 | `config` · `arm` · `arm_overlay`(overlay 키 전체) · `arm_base_sha256` · `judge` | `arms.<구현>.settings.{config, overlay, base_params_sha256, judge}` · `base_params`(지금의 params 해시) |
| 시나리오 세트 | — | `scenario_manifest` — `MANIFEST.sha256` 의 해시 · 파일 수 · `mismatch`(목록과 다른 파일) |

- `dirty` 는 **추적 파일의 미커밋 변경**이다. 추적 안 된 파일은 `untracked` 에 따로 남는다
  (colcon 산출물 `build/` · `install/` · `log/` 가 늘 남아 있어, 합치면 항상 참이 된다).
- `settings` 의 각 값은 **서로 다른 값의 목록**이다. 길이가 1 이면 전 런이 같은 설정·같은 코드로
  돈 것이고, 2 이상이면 도중에 바뀐 것이다 — `repro_warnings` 와 화면의 `[재현]` 줄로 알린다.
  미커밋 변경이 있는 채로 돈 런, MANIFEST 와 다른 시나리오 파일도 같은 곳에 뜬다.
- 꼬리표가 없는 런(`--config` 없이 돈 won 브랜치 결과)은 `untagged_runs` · `runs_without_git`
  으로 센다. won 쪽 커밋은 체크리스트(§3)대로 손으로 적는다.

## 3. won 브랜치에서 같은 실험을 돌릴 때 — 체크리스트

> 2026-10-04 부터 초기 구현은 A2(§2a)로 돌린다. 아래는 won 브랜치를 직접 돌릴 일이 생길 때를 위한 참고다.

won 의 `batch_run.py` 에는 `--config` · `--out-dir` · `--resume` 이 **없다** (2026-09-07 판).
won 코드를 따로 풀어 확인한 것은 ✔ 로 표시했다.

### 준비

- [ ] **작업 트리를 따로 만든다.** `git worktree add ../mando_won origin/won`
      (won_24 작업 트리에서 브랜치를 바꾸지 않는다 — 커밋 안 된 변경과 `scenarios/` · `logs/` 가 섞인다.)
- [ ] **같은 시나리오 파일을 쓴다 — 다시 생성하지 않는다** (won 의 생성기는 이 세트를 못 만든다).
      목록 파일의 `vtd_xml_path` 는 절대경로라 VTD 는 같은 XML 을 읽는다. `route_csv` 는 상대경로라
      작업 트리에서 보이게 링크를 건다: `ln -s /home/mjw/mando/scenarios ../mando_won/scenarios`
- [ ] **파일이 같은지 확인.** `cd scenarios && sha256sum -c MANIFEST.sha256 --quiet`
- [ ] **판정값을 맞춘다.** won 의 batch_run 은 `config/params.yaml` 만 읽으므로 작업 트리에서
      `batch:` 의 세 줄을 고친다 (주행에는 영향 없다. 커밋하지 않는다).

      | 키 | 맞출 값 | won 현재 값 |
      |---|---|---|
      | `batch.stall_end_s` | 60.0 | 30.0 |
      | `batch.blocked_end_s` | 60.0 | 30.0 |
      | `batch.no_progress_end_s` | 99999.0 | 120.0 |
      | `batch.stall_excuse_queue` | false | false ✔ |
      | `batch.stall_speed_mps` / `stall_intent_mps` | 0.1 / 0.5 | 0.1 / 0.5 ✔ |

      `no_progress_end_s` 를 빼먹으면 적신호를 120 s 넘게 기다린 런이 won 에서만 `no_progress` 로 끊긴다.
      확인: `python3 -c "import yaml;print(yaml.safe_load(open('config/params.yaml'))['batch'])"`
- [ ] **`--host 127.0.0.1` 로 돌린다.** won 의 batch_run 은 host 가 로컬이어도 시나리오 실존 확인을
      `ssh mjw@127.0.0.1 ls …` 로 한다. 둘 중 하나: (a) 이 PC 에 sshd 와 키 로그인을 열어 둔다,
      (b) won_24 작업 트리의 `precheck_vtd_paths` 앞머리 5줄(host 가 로컬이면 `os.path.exists` 로 확인)을
      won 의 `tools/batch_run.py` 에 옮긴다 — 실행기만 고치는 것이고 제어기는 건드리지 않는다.
- [x] **경로** ✔ — 생성 게이트 W 가 won 의 `build_route.py` 로 지은 경로가 장애물 차로를 지나는
      것만 통과시켰다. `data/lane_graph.pkl` 은 두 브랜치가 바이트 동일.

### 실행

유형마다 한 번씩, 작업 트리 루트에서:

```
cd ../mando_won
python3 tools/batch_run.py scenarios/batch_논문1_기본.json --host 127.0.0.1
mkdir -p /home/mjw/mando/logs/paper/WON/논문1_기본
mv logs/batch/<방금 생긴 시각 폴더> /home/mjw/mando/logs/paper/WON/논문1_기본/
```

나머지 8개 목록도 같다. **유형 폴더 이름은 목록 파일의 `batch_<유형>.json` 과 같아야 한다.**

- [ ] 종료 정책은 기본(`--timeout`, 조기 종료). `--no` 를 붙이지 않는다.
- [ ] `--settle-s` · `--pause-s` · `--vtd-warmup-s` 는 기본값 그대로 (won_24 와 같다 ✔).
- [ ] **끊겼을 때**: `--resume` 이 없으므로 남은 것만 담은 목록을 만들어 다시 돌리고, 새 시각 폴더를
      **같은 유형 폴더 아래에 또 옮겨 넣는다.** `paper_report.py` 는 `<유형>/*/report.json` 을 전부 읽어
      같은 시나리오는 끝까지 간 런(완주·stall·blocked·no_progress·timeout)을, 그중 나중 것을 쓴다.
      ```
      python3 - <<'E'
      import json, glob
      typ = '논문1_기본'
      done = {r['name'] for f in glob.glob(f'/home/mjw/mando/logs/paper/WON/{typ}/*/report.json')
              for r in json.load(open(f)) if r['status'] in ('완주', 'stall', 'blocked', 'no_progress', 'timeout')}
      left = [i for i in json.load(open(f'scenarios/batch_{typ}.json')) if i['name'] not in done]
      json.dump(left, open(f'/tmp/left_{typ}.json', 'w'), ensure_ascii=False)
      print(len(left), '개 남음')
      E
      python3 tools/batch_run.py /tmp/left_논문1_기본.json --host 127.0.0.1
      ```
      Ctrl+C 로 멈추면 batch_run 이 그때까지의 report.json 을 쓴다. 프로세스가 죽으면 그 배치의
      report.json 이 없으므로 그 배치 전체를 다시 돌린다.

### 로그가 집계기로 읽히는지 — 첫 유형이 끝나면 바로 확인

```
cd /home/mjw/mando
python3 tools/paper_report.py --arm WON=logs/paper/WON --out /tmp/won_check
```

- [ ] 표의 "유효 런/계획" 이 돌린 수와 맞는다.
- [ ] `meta.json` 의 `warnings` 가 비어 있다. 특히 "시나리오 정의 없음" (유형 폴더 이름이 다르다) ·
      "그 좌표에 객체가 없다 (시나리오 파일 불일치?)" (다른 XML 로 돌렸다) ·
      "충돌 수 불일치" (won 의 채점기와 이 저장소 채점기가 다르게 셌다 — 집계는 이 저장소 것으로 센다).
- [ ] `per_encounter_long.csv` 의 `gt_ids` 가 채워져 있다 (장애물을 로그에서 찾았다).

집계기가 읽는 필드 — won(kr_rules) 로그에 전부 있다 ✔ (2026-09-03 kr_rules 배치 로그 3건으로 확인:
종료 사유·충돌 수·정지 검토·시프트 검출이 읽힌다. **won 브랜치로 이 세트를 실제 돌린 로그로는 아직
확인하지 못했다**):

| 어디 | 필드 |
|---|---|
| `report.json` (결과 목록) | `name` · `status` · `log` · `violations` · `ran_s` |
| 틱 `raw.objects[]` | `[id, x, y, z, heading, speed, length, width, height]` |
| 틱 `ego` | `x` · `y` · `yaw` · `speed` · `route_s` |
| 틱 `world` | `light [id, state]` · `summ.dist_stop_line` · `summ.in_junction` |
| 틱 `objects[]` | `id` · `x` · `y` · `speed` · `cls` |
| 틱 `decision` | `v_target` · `reasons.winner` · `reasons.avoid {shift, blocker, span}` |

## 4. 합쳐서 집계

```
python3 tools/paper_report.py --arm WON24=logs/paper/WON24 --arm A2=logs/paper/A2
```

- 구현 이름과 빌더는 `config/arms/arms.yaml` 에서 온다 (`arms.WON24` · `arms.A2`, 다른 브랜치 로그면 `report.labels`).
- **충돌 수는 두 구현 모두 이 저장소의 검출기로 로그를 다시 센다** (같은 자). 종료 사유는 각 브랜치의
  batch_run 이 그 자리에서 내린 판정 그대로다 — 그래서 판정값을 위 표대로 맞춰야 한다.
- 구현이 둘 이상이고 **구현끼리 빌더가 다르면**(예: won 브랜치 로그) 두 빌더의 경로가 다른 시나리오가
  자동으로 빠진다 (생성기가 남긴 `route_quality` 기록 기준). WON24 · A2 는 빌더가 같아 빠지는 것이 없다.
  빌더를 모르는 구현이 끼면 다르다고 본다. `meta.json` 의 `arm_builders` · `builder_rule_applied` 에 남는다.

### 비교 제외 규칙 (2026-10-03 확정 — won 결과를 보기 전에 정했고 양쪽에 똑같이 적용한다)

> 2026-10-04: 이 규칙은 **구현끼리 빌더가 다를 때만** 돈다. WON24 · A2 는 같은 빌더라 적용되지 않는다.

구현이 둘 이상일 때 다음 중 하나면 그 시나리오를 **모든 구현에서** 뺀다 (`paper_report.builders_differ`):

1. 장애물이 놓인 차로를 어느 한쪽 빌더의 경로가 지나지 않는다 (`builder_off_lane` · `other_builder.off_lane`),
   또는 비교 빌더가 경로를 짓지 못했다.
2. 두 빌더의 경로 길이 차가 **5 % 이상**이다 (`--builder-len-tol 0.05`).

차로열이 조금 다른 것만으로는 빼지 않는다 — 장애물 차로가 같으면 같은 조우를 달린다.

현재 세트(`56599fe5…`)에서 걸리는 것 — **1개**:

| 시나리오 | 사유 |
|---|---|
| `논문8_일반주행_01_기본` | 경로 길이 2470.4 m vs won 6531.4 m (164.4 %) |

빼지 않는 것 (차로열은 다르나 장애물 차로가 같고 길이 차 < 5 %): `논문3_앞뒤연속_03_좌회전5`
(634.3 / 639.2 m) · `논문3_앞뒤연속_06_좌회전8` (602.5 / 607.3) · `논문4_나란히_04_직진9` (676.7 / 681.2) ·
`논문7_오탐_급정거_04_좌회전3` (622.3 / 630.7). 집계할 때마다 `report/meta.json` 의
`auto_excluded_why` 에 사유가 남는다.

### 집계 기준 (2026-10-03)

- **지나감 횡거리 8 m** (`--pass-lat-m`, 이전 6 m). 나란히 유형은 열린 쪽으로 지나가면 먼 쪽 차가
  두 차로 건너(6.2–6.6 m)라 6 m 로는 실제로 지나간 런 4건이 "못 지나감" 으로 잡혔다.
- **종료선 이후 사건 제외** (`--after-finish exclude`). 기준은 `report.json` 의 `finish_thr_m`
  (won 의 batch_run 도 쓴다). 뺀 충돌은 `collisions_after_finish.csv` 에 남는다.
- **실패 원인 분류** (`causes.csv` · `failures.csv` 의 `cause`) — 로그에서 읽는 규칙, 순서대로:
  기반 실패 → 시뮬레이터 / 오탐 시프트 → 오탐 / 충돌이 전부 종료선 콘이거나 전부 "움직이는 차량
  (≥ 1 m/s, 진행각 ≥ 15°)이 거의 선 자차(≤ 1.5 m/s)로 들어옴" → 시나리오 / 나머지 → 제어기 판단.
  분류만 하고 결과에서 빼지 않는다.

### 논문 판형 표·그림

```
python3 tools/paper_figs.py --arm WON24=logs/paper/WON24                              # 지금
python3 tools/paper_figs.py --arm WON24=logs/paper/WON24 --arm WON=logs/paper/WON     # won 이 오면 같은 명령
```

출력 `logs/paper/paper_figs/`: `table2_paper.csv/.md` (표 2) · `fig_A_success_by_type` (유형별 회피
성공률 + 95% 구간) · `fig_B_near_stop_trace` (근접 정지 대표 시나리오의 시간–속도 · 장애물까지 거리) ·
`fig_C_failure_causes` (실패 원인) — 각각 `.png`(600 dpi) · `.pdf` · 원천 `.csv`, 한 단 폭 8.5 cm ·
8 pt · 회색조+무늬 — 그리고 `IV2_draft.md` (Ⅳ.2 문장 초안과 출처), `report/` (집계 원자료 · meta.json).
표 2 의 주 지표는 회피·일반 주행 유형이 성공률, **오탐 유형(앞차 급정거 · 신호 대기 줄)은 오탐 런 수**
(다시 움직일 대상에 시프트를 만든 런, 95% Wilson 구간)이다 — 오탐 유형의 성공률은 참고 열에 둔다.
그림 B 의 범례에는 구현마다 **로그로 잰 시작 간격**을, 아래 칸에는 시나리오의 조건값을 적는다
(로그 기준점과 생성 기준점이 달라 0.5 m 쯤 다르다 — 첫 실행에서 조건 8 m, 실측 7.5 m).
그림 B 의 시나리오는 `arms.yaml report.paper_figs` 의 규칙으로 고른다 (모든 구현이 끝까지 간 런 중
기준 구현이 성공한 것, 시작 거리가 가장 짧은 것). PDF 글꼴은 Noto CJK(CFF)면 Type 3 으로 넣는다 —
나눔고딕(`fonts-nanum`)을 깔면 Type 42(편집 가능한 글자)로 바뀐다.

### 유형별 성공 기준 (집계기가 시나리오 정의의 이벤트로 유형을 가른다)

| 유형 | 성공 | 실패·제외 사유 (`runs.csv` 의 `why`) |
|---|---|---|
| 회피 (유형 1~6) | 모든 장애물을 지나감 + 충돌 없음 + 무한 정지 없음 | `inf_stop(stall/blocked)` · `collision` · `not_passed(...)` |
| 신호 대기 줄 | 적색 동안 대기 차량에 시프트를 만들지 않음 + 충돌 없음 | `false_positive_shift` · `collision` · `inf_stop(...)` |
| | *판정 제외* — 대기 차량이 끝까지 안 풀려 생긴 정지, 자차가 닿기 전에 줄이 풀림 | `queue_not_released` · `queue_not_met` |
| 급정거 | 다시 움직인 앞차에 시프트를 만들지 않음 + 충돌 없음 + 무한 정지 없음 | `false_positive_shift` · `collision` · `inf_stop(...)` |
| 일반 주행 | 완주 (주행 시간·감점은 `plain.csv`) | 종료 사유 |

산출물 (`logs/paper/report/`): `table2` (회피 유형) · `by_type.csv` (전 유형 성공률) ·
`by_condition.csv` (거리·간격·열린 쪽·차간별) · `failures.csv` (실패 런마다 멈춘 이유 — 멈춘 동안의
회피 진단과 기각 사유 — 와 그때 장애물까지 앞범퍼 거리) · `per_encounter.csv` (장애물 조우 단위) ·
`queue_fp.csv` · `brake_fp.csv` · `plain.csv` · `stall_review.csv` · `collisions.csv` · `paired.csv` ·
`figures/` (`fig_by_type` 유형별 성공률, `fig_near_stop_distance` 근접 정지 거리별,
`fig_gap_success` 앞뒤 연속·지그재그 간격별, `fig_table2`, `fig_end_reason`, `fig_brake_fp`,
`fig_speed_trace_*`).

## 5. 읽을 때 주의

- 무한 정지 = `stall` + `blocked`. 면책은 적신호·보행자뿐이다. **대기행렬은 면책이 아니다** —
  `stall_review.csv` 의 `queue_suspect` 를 사람이 보고 걸러 낸다.
- `timeout` 은 무한 정지와 따로 센다.
- won_24 의 `ctrl24.escape_clear_m` 은 −5.0 이다 (CLAUDE.md·B-32 는 2.0 이라고 적고 있다 — 고치지 않고
  기록만). 장애물에 붙어 있어도 3 m/s 바닥이 깔리므로 WON24 의 충돌 수를 읽을 때 같이 본다.
- 종료선 콘도 로그에 장애물 객체로 잡힌다. 충돌의 상대는 `collisions.csv` 의 `obj_kind`.
