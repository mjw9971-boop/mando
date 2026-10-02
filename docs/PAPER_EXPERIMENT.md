# 논문 Ⅳ장 실험 — 시나리오 세트 · 실행 · 집계 · won 브랜치 체크리스트

2026-10-02 기준. 비교 대상은 둘이다.

| 구현 | 브랜치 | 이름 (표·그림 범례) |
|---|---|---|
| `WON24` | won_24 (실험은 `paper_exp` 에서 돌린다 — 아래) | 최종 구현(won_24) / Final implementation (won_24) |
| `WON` | won (`origin/won`, 9b05f1f 2026-09-07) | 초기 구현(won) / Initial implementation (won) |

`paper_exp` = won_24 `70160e9` + 실험 도구(생성기 · 집계기 · `config/arms` · 이 문서 ·
`data/paper_routes`). **제어기 파일(`team_code/` · `vtd_adapter/` · `run_agent.py`)은 won_24 와
같다** (`git diff won_24 paper_exp -- team_code vtd_adapter run_agent.py` 가 비어 있다).
won_24 브랜치 자체는 그대로 둔다.

**주행 파라미터는 어느 쪽도 바꾸지 않는다.** 두 브랜치 모두 그 브랜치의 `config/params.yaml`
주행 값을 그대로 쓴다. 맞추는 것은 배치 판정값뿐이다 (60 s 무한 정지 — 주행에 영향 없음).

## 1. 시나리오 세트 — 한 줄로 재현

```
python3 tools/gen_scenarios.py 논문1_기본 논문2_근접정지 논문3_앞뒤연속 논문4_나란히 \
    논문5_지그재그 논문6_옆차로교통 논문7_오탐_급정거 논문7_오탐_신호줄 논문8_일반주행 \
    --seed 0 --vtd-dir /home/mjw/mando/scenarios --coverage-report --manifest
```

- 출력은 `scenarios/` (비어 있어야 한다 — 남아 있는 `coverage_history.json` 이 경로 추첨을 바꾼다).
- **86개** · 목록·XML·CSV·정의·판정 기록 354개 파일. 확인: `cd scenarios && sha256sum -c MANIFEST.sha256 --quiet`
- `MANIFEST.sha256` 자체의 해시: `4ada8e2d6676d0e87f3020cbc8a18cf7724884ea8f6acfcd6de5e02a53673cdc`
- 같은 명령을 다른 폴더에 다시 돌려 354개 파일이 전부 같음을 확인했다 (2026-10-02).
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
  (`--exclude 논문8_일반주행_01_기본`).
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

## 2. won_24 에서 실행·집계

```
python3 tools/make_arms.py            # params.yaml + overlay → config/arms/WON24.yaml
python3 tools/paper_run.py --dry-run  # 계획: 9배치 86런
python3 tools/paper_run.py            # 실행. logs/paper/WON24/<유형>/, 끝난 런은 건너뛴다
python3 tools/paper_report.py --arm WON24=logs/paper/WON24          # → logs/paper/report/
```

- 사본 `config/arms/WON24.yaml` 이 params 와 다른 키는 `batch.stall_end_s` · `batch.blocked_end_s`
  (99999 → 60.0) 둘뿐이다. `make_arms.py` 는 `batch` 밖의 섹션을 건드리는 overlay 를 거부한다.
- **params.yaml 을 바꾸면 사본을 다시 만든다** (`paper_run` 이 낡은 사본을 잡아 중단한다).
- `logs/paper/A1/` 은 계획 변경 전에 돌다 멈춘 것이다 (끝난 런 0건). 지우지 않고 두되 집계에 넣지 않는다.

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
python3 tools/paper_report.py --arm WON24=logs/paper/WON24 --arm WON=logs/paper/WON
```

- 구현 이름은 `config/arms/arms.yaml` 에서 온다 (`arms.WON24`, `report.labels.WON`).
- **충돌 수는 두 구현 모두 이 저장소의 검출기로 로그를 다시 센다** (같은 자). 종료 사유는 각 브랜치의
  batch_run 이 그 자리에서 내린 판정 그대로다 — 그래서 판정값을 위 표대로 맞춰야 한다.
- 구현이 둘 이상이면 **두 빌더의 경로가 다른 시나리오는 자동으로 빠진다** (생성기가 남긴
  `route_quality.other_builder.same_lanes` 기준 — 지금은 `논문8_일반주행_01_기본` 하나).
  `--exclude` 를 따로 줄 필요가 없다.

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
