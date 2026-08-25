# HL FMA 2026 자율주행 컨트롤러

VTD 2025.2 가 9910 TCP 로 GT(ego/객체/신호등)를 주고, 우리는 조향·가속도·지시등을 되돌려준다.
**순수 룰베이스. 신경망 없음.** 상세 규격과 설계 근거는 [AGENT_SPEC.md](AGENT_SPEC.md).

## 구조

ROS 2 (rclpy) 워크스페이스. 검증된 로직은 `core/` 에 그대로 있고, 노드는 그걸 감싸기만 한다.

```
src/hlfma_msgs/     GtState Object TrafficLight WorldState TrackedObject Decision Cmd
src/hlfma/hlfma/
  nodes/            vtd_bridge  perception  planner  control  (+ single_process, params, qos)
  core/             comm perception planner shield control lanegraph types logger timing
                    ↑ ROS 를 모른다. 단위 테스트/replay/standalone 에 그대로 쓴다.
  convert.py        dataclass <-> msg 변환 (여기 한 곳에만)
src/hlfma/launch/   drive.launch.py  replay.launch.py
src/hlfma/config/   params.yaml      ← 모든 튜닝 파라미터의 단일 출처
tools/              build_lane_graph build_route plot_lane_graph make_test_route
                    probe_9910 mock_vtd run_standalone replay score summarize_run
                    scp_client set_ego_start gen_scenarios batch_run
data/               lane_graph.pkl route.pkl xodr   (레포에 포함 — 클론 즉시 사용 가능)
configs/            themes.yaml      ← 시나리오 생성 프리셋 (주제·경로 풀·변형 축)
templates/          9_clean_drive.xml 등 검증된 VTD 시나리오 원본 (블록 추출용)
```

한 패킷 = /gt_state 하나 = 한 틱. **타이머 없이 콜백 체인으로만** 흐른다:

```
9910 ─recv─> vtd_bridge ─/gt_state─> perception ─/world_state─> planner(+shield)
                  ↑                                                    │
                  └────────────── /cmd ────────── control <──/decision─┘
```

QoS 는 전 토픽 depth=1 / RELIABLE / 최신 우선 — 밀린 메시지는 버린다.
**기본 실행은 노드별 프로세스 분리**(`use_single_process:=false`)다. 단일 프로세스는
GIL 경합으로 8.8 Hz 까지 떨어지는 것이 실측됐다(아래 처리율 표).

각 단계는 앞 단계 출력(dataclass)만 받는다. 역방향 참조 금지.
Perception 은 판단하지 않고, Control 은 교통법을 모르며, Shield 는 Planner 를 신뢰하지 않는다.

## 설치

```bash
pip install -r requirements.txt
```

## 빌드 & 실행

```bash
colcon build --symlink-install
source install/setup.bash

./run.sh                              # 대회날에는 이것만
```

환경변수로 덮어쓴다: `VTD_HOST` `VTD_PORT` `GRAPH` `ROUTE` `RECORD`

```bash
# 직접 실행 (인자 생략 시 data/ 의 기본 pkl 사용)
ros2 launch hlfma drive.launch.py

# 전 토픽 녹화
ros2 launch hlfma drive.launch.py record:=true bag_path:=bags/drive

# VTD 없이 재생 — bag 의 /gt_state 만 틀고 나머지를 새로 계산한다
ros2 launch hlfma replay.launch.py bag:=bags/drive rate:=1.0

# jsonl 로그 개루프 재생 (코드 수정 후 회귀 확인의 기본 도구)
python3 tools/replay.py logs/run_xxx.jsonl

# 주행 직후 성적표 (0.5 s) — 완주·과속·t_off·LC·지시등·정지 이벤트
python3 tools/summarize_run.py logs/$(ls -t logs/ | head -1)

# VTD 원격 제어 (시나리오 로드 / 시작 / 정지 / 위치)
python3 tools/scp_client.py --host 192.168.10.1 load <PC-B상의 시나리오 경로>
python3 tools/scp_client.py --host 192.168.10.1 start

# VTD 없이 닫힌 루프 (자전거 모델 목 서버)
python3 tools/mock_vtd.py                        # 터미널 1
ros2 launch hlfma drive.launch.py               # 터미널 2

# 9910 프레임 눈으로 확인
python3 tools/probe_9910.py --host <VTD_IP>

# 그날치 테스트 시나리오 자동 생성 (주제는 configs/themes.yaml, --list 로 목록)
python3 tools/gen_scenarios.py 보행자집중 급정거집중 --count 3 --seed 1
python3 tools/gen_scenarios.py 교차로집중 --hours 2        # 시간 예산으로 개수 환산
#  → scenarios/<주제>/<이름>.{xml,csv,yaml} + batch_<주제>.json + batch_all.json(통합)
#    모든 경로는 생성 시점에 build_route 로 검증 — 경고(회전 불가 연결로 등) 있으면
#    다른 시작점으로 재시도, 소진 시 사유와 함께 목록에서 제외
#  → scenarios/ 를 VTD PC 의 /home/mjw/scenarios 로 복사(scp) 후:
#    python3 tools/batch_run.py scenarios/batch_all.json
#    (실행 전에 첫 시나리오 경로를 ssh ls 로 실존 확인 — 없으면 즉시 중단)
#    (batch_run 은 목록 여러 개·glob 도 받아 합쳐 실행한다 — 단 batch_all 과 주제별을
#     같이 주면 이름 중복으로 막힌다. 중복 검사는 통합 후 기준)

python3 -m pytest tests/
```

### 실행 모드와 처리율 (실차 실측, 2026-08-23)

| 모드 | 제어 주기 | 프레임 드롭 | gt→cmd 지연 p50 |
|---|---|---|---|
| 단일 프로세스 | 8.8 Hz | 67 % | 169 ms |
| **프로세스 분리 (기본)** | **20 Hz** | **0.1 %** | **4.4 ms** |

VTD 는 정확히 20 Hz / RTF 1.0 으로 보낸다(조직위 공식 확인 + 실측 일치).
도착 간격이 40/80 ms 로 교대하는 것은 TCP 전달 지터이며 유실이 아니다 —
**속도 추정은 이 때문에 벽시계 슬라이딩 창(`percep.speed_win_s`) 기반**이다.
(고정 dt 가정은 −15 % 편향 → 53 km/h 은폐 과속을 만들었던 실사고가 있다.)

### 파라미터

`src/hlfma/config/params.yaml` 한 파일이 단일 출처다. **수정 후 반드시
`colcon build`** — yaml 은 install/ 로 복사된다. launch 인자
(`graph:=` `route:=` `host:=` `port:=`)가 그 위에 덮인다.

## 지도 준비

`data/` 에 xodr·pkl 이 레포에 포함돼 있어 클론 직후 바로 주행 가능하다.
경로만 새로 받으면:

```bash
python3 tools/build_route.py data/lane_graph.pkl waypoints.csv -o data/route.pkl
```

지도 자체를 다시 만들 때 (xodr 이 바뀐 경우에만):

```bash
python3 tools/build_lane_graph.py data/HL_FMA_VTD_LivingLab.xodr -o data/lane_graph.pkl
python3 tools/plot_lane_graph.py data/lane_graph.pkl -o docs/images/map_full.png
```

빌드 리포트의 **차선변경 창 20 m 미만 경고**와 `total_length` 를 반드시 눈으로 확인.
자세한 내용은 [docs/lane_graph.md](docs/lane_graph.md).

## 대회날 체크리스트

**네트워크 (가장 먼저)**

0. 노트북 유선 IP 를 대회장 VTD PC 와 **동일 서브넷**으로 수동 설정
   → `ping <VTD_IP>` 응답 확인 → `params.yaml` 의 `comm.host` 수정 → **colcon build**

**출발 전**

1. `python3 -m pytest tests/` 통과
2. 주최측 경유점 CSV 수령 → `tools/build_route.py` 로 `data/route.pkl` 생성
   - CSV 형식 확정: 첫 점=출발, 끝 점=도착, 중간은 교차로 진입·진출 짝
   - 출력의 `total_length` / 창 경고 / `events` 개수를 눈으로 확인
3. `tools/plot_lane_graph.py` 로 경로를 그려서 **의도한 길인지 확인**
4. `python3 tools/probe_9910.py --host <VTD_IP>` 로 프레임 수신 확인

**연결 후**

5. `./run.sh` → 20 Hz 로 Command 가 나가고 매 틱 로그가 쌓이는지
6. **조향 부호** — `comm.steer_sign: 1.0` 실측 확정값 (2026-08-19).
   주행 5초 안에 `조향 부호 정상 (corr=+0.9x)` 로그 확인.
   반대 경고가 뜨면 params 뒤집고 재빌드·재기동.
7. junction 6 신호가 켜지는지 (연습 환경에서는 state=0 고정이었음) —
   첫 수 초 안에 light state 가 0 이 아닌 값으로 오는지 확인

**주행 후**

8. `python3 tools/summarize_run.py logs/<런>.jsonl` — 과속·이탈·지시등 즉시 확인
9. 이상 항목만 `tools/replay.py` 로 재현·수정·재검증

## 검증 완료 (실차, 2026-08-23 기준)

| 항목 | 결과 |
|---|---|
| 821 m 경로 완주 + 경로 끝 정지 | v=0 정지, 리스폰 0 |
| 제한속도 (S1.1.01) | 초과 0 틱, 실속도 편향 0.0 % |
| 적색 완전정지 (S5.1.01) | v=0 을 1.3 s 유지, 앞범퍼가 정지선 1.10 m 앞 (오차 0.10 m) |
| RTOR (적신호 우회전) | stopping→dwell 1.0 s→go 서행, 녹색 전환 해제 모두 확인 |
| 차선변경 3/3 + 지시등 lead | 1.4* / 4.0 / 4.8 s (*창 41 m 제약, lead_short 플래그) |
| 회전 지시등 3/3 | lead 최대 15 s, 깜빡임 0 |
| 신호 폴백 | 매핑 누락 정지선(도로 30)에서 적색 정지 재현 |

## 미확인·미해결

코드에 `TODO(SPEC §7-n)` 표시. 가정값은 전부 `params.yaml`.

| # | 내용 | 상태 |
|---|---|---|
| 1 | 점멸 신호(state 6) 황색/적색 구분 | 조직위 문의 대상, `signal.flash_mode` |
| 2 | 방향지시등 선행 점등 "n초"의 규정값 | `signal.lc_lead_min_s` (현재 3.0) |
| 3 | RTOR 허용 여부 | `signal.rtor_enabled` 로 즉시 on/off |
| 4 | 스쿨존 내 비신호 횡단보도 일시정지 채점 여부 | 폴백 오정지 위험 46곳과 상충 — 규정 확인 후 결정 |
| 5 | `RM_518` = 스쿨존 30 해석 | 텍스처 미검증 |
| 6 | 5분 제한 초과 시 처리 | 평균 속도 전략에 직결 |
| 7 | 횡단보도 보행자 → 정지선 정지 | **구현 중** (shield 급정지가 선점하는 갭 발견) |
| 8 | 라우터 연속 차선변경 이격 / waypoint 짝 활용 | 임의 대회 경로 강건화, 미착수 |

## 구현 상태

| 모듈 | 상태 |
|---|---|
| `core/comm` `control` `logger` `timing` `types` | 완전 구현 |
| `core/perception` | 창 기반 속도추정·차로매칭·lookahead·리셋/스톨 감지 완비. 객체 TTC 구현 |
| `core/planner` | 속도 중재(min)·신호 7상태·폴백·RTOR·정지(전장보정+lag)·LC·지시등 완비 |
| `core/shield` | corridor·TTC 비상제동. 횡단보도 보행자 우선순위 정리 중 |
| `core/lanegraph` | 창 병합·cum_s·lookahead 이중계상 수정 완료 |
| 노드 4종 + convert + launch | 완전 구현, 프로세스 분리 기본 |
| tools | replay·summarize_run·scp_client 실전 사용 중 |