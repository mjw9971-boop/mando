# HL FMA 2026 자율주행 컨트롤러

VTD 2025.2 가 9910 TCP 로 GT(ego/객체/신호등)를 주고, 우리는 조향·가속도·지시등을 되돌려준다.
**순수 룰베이스. 신경망 없음.** 상세 규격과 설계 근거는 [AGENT_SPEC.md](AGENT_SPEC.md).

## 구조

ROS 2 (rclpy) 워크스페이스. 검증된 로직은 `core/` 에 그대로 있고, 노드는 그걸 감싸기만 한다.

```
src/hlfma_msgs/     GtState Object TrafficLight WorldState TrackedObject Decision Cmd
src/hlfma/hlfma/
  nodes/            vtd_bridge  perception  planner  control  (+ single_process, params, qos)
  core/             comm perception planner shield control lanegraph types logger
                    ↑ ROS 를 모른다. 단위 테스트/standalone 에 그대로 쓴다.
  convert.py        dataclass <-> msg 변환 (여기 한 곳에만)
src/hlfma/launch/   drive.launch.py  replay.launch.py
src/hlfma/config/   params.yaml      ← 모든 튜닝 파라미터의 단일 출처
tools/              build_lane_graph build_route plot_lane_graph make_test_route
                    probe_9910 mock_vtd run_standalone replay score
data/               lane_graph.pkl route.pkl   (※ .gitignore 대상)
```

한 패킷 = /gt_state 하나 = 한 틱. **타이머 없이 콜백 체인으로만** 흐른다:

```
9910 ─recv─> vtd_bridge ─/gt_state─> perception ─/world_state─> planner(+shield)
                  ↑                                                    │
                  └────────────── /cmd ────────── control <──/decision─┘
```

QoS 는 전 토픽 depth=1 / RELIABLE / 최신 우선 — 밀린 메시지는 버린다.
기본 실행은 4개 노드를 **한 프로세스**(MultiThreadedExecutor)에 올린다.

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
# 직접 실행
ros2 launch hlfma drive.launch.py graph:=data/lane_graph.pkl route:=data/route.pkl

# 노드를 프로세스별로 분리 (디버깅)
ros2 launch hlfma drive.launch.py use_single_process:=false

# 전 토픽 녹화
ros2 launch hlfma drive.launch.py record:=true bag_path:=bags/drive

# VTD 없이 재생 — bag 의 /gt_state 만 틀고 나머지를 새로 계산한다
ros2 launch hlfma replay.launch.py bag:=bags/drive rate:=1.0

# VTD 없이 닫힌 루프 (자전거 모델 목 서버)
python3 tools/mock_vtd.py --steer-sign -1.0     # 터미널 1
ros2 launch hlfma drive.launch.py               # 터미널 2

# ROS 없이 core 만 (백업 실행 경로)
python3 tools/run_standalone.py --graph data/lane_graph.pkl --route data/route.pkl

# 9910 프레임 눈으로 확인
python3 tools/probe_9910.py --host 127.0.0.1

python3 -m pytest tests/
```

### 실행 모드와 처리율 (실측)

목 VTD 20 Hz 입력, 15 초 측정:

| 모드 | /gt_state | /world_state | /decision | /cmd |
|---|---|---|---|---|
| `use_single_process:=true` (기본) | 17.9 Hz | 16.4 Hz | 15.8 Hz | 15.8 Hz |
| `use_single_process:=false` | **20.5 Hz** | **20.5 Hz** | **20.5 Hz** | **20.5 Hz** |

소켓 계층만 따로 재면 19.97 Hz / 유실 0 이다. 즉 손실은 전부 ROS 계층에서 난다 —
한 프로세스 안에서 GIL 을 두고 recv 스레드와 3개 노드 콜백이 경쟁하고, QoS 가
depth=1 "최신 우선" 이라 밀린 메시지는 설계대로 버려진다.

분리 실행은 프로세스마다 GIL 이 따로라 유실이 사라진다(네 토픽 메시지 수가 정확히 일치).
**대회에서는 `use_single_process:=false` 를 권한다.** 기본값은 요구사항대로 true 로 두었다.

### 파라미터

`src/hlfma/config/params.yaml` 한 파일이 단일 출처다. ROS 노드와
`tools/run_standalone.py` 가 **같은 파일**을 읽는다. launch 인자
(`graph:=` `route:=` `host:=` `port:=`)가 그 위에 덮인다.

## 지도 준비

`data/*.pkl` 은 `.gitignore` 대상이라 **클론 직후에는 없다.** 다시 만들어야 한다:

```bash
python3 tools/build_lane_graph.py HL_FMA_VTD_LivingLab.xodr -o data/lane_graph.pkl
python3 tools/build_route.py data/lane_graph.pkl waypoints.csv -o data/route.pkl
python3 tools/plot_lane_graph.py data/lane_graph.pkl -o docs/images/map_full.png
```

자세한 내용은 [docs/lane_graph.md](docs/lane_graph.md).

## 대회날 체크리스트

**출발 전**

1. `python3 -m pytest tests/` 통과
2. 주최측 경유점 파일 수령 → `tools/build_route.py` 로 `data/route.pkl` 생성
   - 판정 반경은 `--radius` 로 조정 (SPEC §7-3: 파일 형식 미확인)
   - 출력의 `total_length` 와 `events` 개수가 상식적인지 눈으로 확인
3. `tools/plot_lane_graph.py` 로 경로를 그려서 **의도한 길인지 확인**
4. `python3 tools/probe_9910.py --host <VTD_IP>` 로 프레임이 들어오는지 확인
   - ego x,y 가 지도 좌표와 같은지 (SPEC §1.1 검증 완료 사항이지만 재확인)
   - 신호등 id/state 가 실제로 바뀌는지 (SPEC §7-1 미확인)

**연결 후**

5. `./run.sh` → 20 Hz 로 Command 가 나가고 매 틱 로그가 쌓이는지
6. **조향 부호** — `comm.steer_sign: 1.0` 이 실측 확정값이다 (2026-08-19, VTD 2025.2).
   주행 5초 안에 control 노드가 `조향 부호 정상 (corr=+0.9x)` 을 찍는지 본다.
   `steer_sign 이 반대일 가능성` 오류가 뜨면 `src/hlfma/config/params.yaml` 의
   `comm.steer_sign` 을 뒤집고 재기동 (그대로 두면 직선에서도 발산해 리셋이 반복된다)
7. 정지선 앞에서 실제로 서는지, 녹색에서 **불필요하게 서지 않는지**
   (녹색신호 통과도 채점 항목이다)

**주행 중 문제 생기면**

8. 로그는 계속 쌓인다. 끝나고 `tools/score.py` 로 어느 항목이 깎였는지 확인
9. `--replay` 로 같은 상황을 VTD 없이 재현하고 고친 뒤 다시 재생해 비교

## 미확인 사항

코드에 `TODO(SPEC §7-n)` 으로 표시돼 있고, 가정값은 전부 `src/hlfma/config/params.yaml` 에 있다.

| # | 내용 | 관련 설정 |
|---|---|---|
| 1 | 신호등 id 가 교차로마다 바뀌는지, state 전이가 실제로 오는지 | — (관측 중) |
| 2 | GT 객체 거리 컷오프 (77 m 사례) | `percep.observe_range_log_m` |
| 3 | 대회 경유점 파일 형식 | `build_route.py --radius` |
| 4 | 표시 없는 도로의 기본 제한속도 | `default_speed_kph` |
| 5 | `RM_518` 이 스쿨존 30 이 맞는지 | — |
| 6 | 점멸 신호(state 6) 규정 해석 | `signal.flash_mode` |

## 구현 상태

| 모듈 | 상태 |
|---|---|
| `core/types` `core/comm` `core/control` `core/logger` | 완전 구현 |
| `core/perception` | 최소 주행분(속도추정·차로매칭·lookahead)까지. 객체 TTC 등은 TODO |
| `core/planner` | 상수속도 + FOLLOW 고정. 법규 로직 TODO |
| `core/shield` | 상수속도 상한만. 6개 가드 TODO |
| `core/lanegraph` | 기존 검증 완료 파일 (수정 금지) |
| 노드 4종 + `convert.py` + launch | 완전 구현 |
| `tools/replay.py` `tools/score.py` | 뼈대 (ROS 경로는 replay.launch.py 가 대체) |
