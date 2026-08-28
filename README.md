# HL FMA 2026 자율주행 컨트롤러

VTD 2025.2 가 9910 TCP 로 GT(ego/객체/신호등)를 주고, 우리는 조향·가속도·지시등을 되돌려준다.
**순수 룰베이스. 신경망 없음.** 목표 규격과 설계 근거는 [AGENT_SPEC.md](AGENT_SPEC.md),
현재 구조 지도는 [docs/OVERVIEW.md](docs/OVERVIEW.md).

판단 로직은 직접 쓰지 않는다 — **PDM-Lite**(DriveLM `pdm_lite`, Apache 2.0)를 원문 그대로
가져와(`team_code/`) VTD 위에서 돌린다. `vtd_adapter/` 가 CARLA 표면을 흉내내 그 사이를 잇는다.

## 구조

ROS 없음. **단일 프로세스 / 단일 스레드 틱 루프**다 (로그 기록만 별도 스레드).

```
run_agent.py        진입점 — 어댑터+PDM 조립, 틱 루프, 예외 격리, 콘솔 1줄 출력

team_code/          판단 계층 (PDM-Lite 이식, 수정한 줄에는 전부 `# VTD:` 주석)
  autopilot.py        주행 판단 전부: IDM·OBB forecast·적색신호·조향
  config.py           GlobalConfig — PDM 하이퍼파라미터의 단일 출처
  lateral_controller.py     횡방향 PID
  kinematic_bicycle_model.py 자차·타차 미래 궤적 외삽
  kr_rules.py         한국 대회 규칙 계층 (종점 정지 + 적신호 최소 정지 유지 + 방향지시등)

vtd_adapter/        VTD ↔ CARLA 어댑터. **판단 없음**
  comm.py             9910 TCP (1109 B 수신 / 9 B 송신), watchdog·재접속
  types.py            RawPacket → WorldState → Decision → Command
  frame.py            좌표·각도 변환의 유일한 자리 (VTD ↔ CARLA)
  carla_types.py      Location/Rotation/BoundingBox/VehicleControl 흉내
  ego.py              자차 속도 추정 + courseRespawn 리셋 감지
  world.py actor.py   carla.World / carla.Actor 흉내 (객체 트래킹·코스팅)
  map.py              carla.Map / Waypoint 흉내 (lanegraph 위)
  lanegraph.py        lane_graph.pkl 런타임 조회
  route.py            VtdRoutePlanner (+실선 차선변경 가드) + 9910 light ↔ 정지선 대조
  control.py          종방향 P 제어 (throttle 자리에 accel 출력) + Command 변환
  config.py           config/params.yaml 로더 + end_margin_m
  logger.py           run_*.jsonl 틱 로그 (score/summarize/replay 의 계약)

tools/              build_lane_graph build_route plot_lane_graph probe_9910 mock_vtd
                    scp_client gen_scenarios batch_run replay summarize_run score
config/params.yaml  ← VTD 연결·차량 제원·추정기·배치·채점 임계
configs/themes.yaml ← 시나리오 생성 프리셋 (gen_scenarios 전용, 위와 별개 파일)
templates/          9_clean_drive.xml 등 검증된 VTD 시나리오 원본 (블록 추출용)
data/               lane_graph.pkl route.pkl xodr (레포에 포함 — 클론 즉시 사용 가능)
tests/              pytest 17개
```

한 패킷 = 한 틱. 타이머 없이 수신이 루프를 끈다:

```
9910 ──recv──> Comm ──RawPacket──> EgoTracker (속도·차로·리셋)
                 ↑                      └──> VtdWorld (객체 트래킹)
                 │                              │
                 │                    AutoPilot.run_step  ← VtdRoutePlanner / VtdMap
                 │                              │            VtdLongitudinalController
                 └──9 B Command──  command_from_control ←──┘   + kr_rules
                                        │
                                   Logger (매 틱 raw+판단 전부 jsonl)
```

어댑터는 판단하지 않고, PDM 은 VTD 를 모르며, kr_rules 는 PDM 원문을 고치지 않는다
(접점은 `autopilot._get_control` 끝의 한 줄뿐).

## 설치

```bash
pip install -r requirements.txt
```

빌드 단계 없음. 파이썬 3.10+ / numpy / scipy / PyYAML (도구용 matplotlib).

## 실행

```bash
# 기본 — 미리 만들어 둔 route.pkl 로 주행
python3 run_agent.py --route data/route.pkl

# 대회 배포 CSV 를 받은 경우 — 내부에서 build_route.py 를 돌려 쓰고 시작
python3 run_agent.py --csv waypoints.csv

# 호스트/포트/그래프/로그 경로는 인자로 덮는다 (기본은 config/params.yaml)
python3 run_agent.py --route data/route.pkl --host 192.168.10.1 --port 9910 \
                     --graph data/lane_graph.pkl --log logs/my_run.jsonl

# VTD 없이 로그 재생 (개루프 — 코드 수정 후 회귀 확인의 기본 도구)
python3 run_agent.py --replay logs/run_xxx.jsonl
python3 tools/replay.py logs/run_xxx.jsonl          # 지시등 시퀀스 분석

# 주행 직후 성적표 — 완주·과속·t_off·리스폰·winner 분포
python3 tools/summarize_run.py logs/$(ls -t logs/ | head -1)

# 위반 검출 + 구간별 채점 (2026 안내문 기준)
python3 tools/score.py logs/run_xxx.jsonl --route data/route.pkl

# VTD 없이 닫힌 루프 (자전거 모델 목 서버)
python3 tools/mock_vtd.py                                  # 터미널 1
python3 run_agent.py --route data/route.pkl --host 127.0.0.1   # 터미널 2

# 9910 프레임 눈으로 확인
python3 tools/probe_9910.py --host <VTD_IP>

# VTD 원격 제어 (시나리오 로드 / 시작 / 정지)
python3 tools/scp_client.py --host 192.168.10.1 load <PC-B상의 시나리오 경로>
python3 tools/scp_client.py --host 192.168.10.1 start

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

VTD 는 정확히 20 Hz / RTF 1.0 으로 보낸다(조직위 공식 확인 + 실측 일치).
도착 간격이 40/80 ms 로 교대하는 것은 TCP 전달 지터이며 유실이 아니다 —
**속도 추정은 이 때문에 벽시계 슬라이딩 창(`percep.speed_win_s`) 기반**이다.
(고정 dt 가정은 −15 % 편향 → 53 km/h 은폐 과속을 만들었던 실사고가 있다.)

### 파라미터 — 두 파일, 역할이 다르다

| 파일 | 담당 | 읽는 곳 |
|---|---|---|
| **`config/params.yaml`** | VTD 연결(comm)·카메라·차량 제원·속도 한계와 정지 gap(`stop_gap_route_end_m` / `stop_gap_stopline_m`)·추정기(percep)·종방향 control·경로·route_end·지시등(signal)·배치·채점 임계·로그 | `vtd_adapter/config.py` → run_agent, tools, tests |
| **`team_code/config.py`** | PDM-Lite 판단 상수 (IDM, forecast, OBB, lateral PID, junction 캡) | `GlobalConfig` — autopilot 이 직접. 대회 정합 오버라이드는 `run_agent.build_pdm_config` 가 params.yaml 값으로 주입 |

수정 후 빌드 불필요. yaml 키가 없으면 조용히 기본값으로 도는 대신 **즉시 죽는다**
(설정 두 벌이 어긋나는 사고 방지). `--config` 로 다른 파일을 지정할 수 있다.

`configs/themes.yaml` 은 위와 무관한 별개 파일로, `tools/gen_scenarios.py` 만 읽는다.

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
   → `ping <VTD_IP>` 응답 확인 → `config/params.yaml` 의 `comm.host` 수정
   (또는 `run_agent.py --host` 로 덮기)

**출발 전**

1. `python3 -m pytest tests/` 통과
2. 주최측 경유점 CSV 수령 → `tools/build_route.py` 로 `data/route.pkl` 생성
   - CSV 형식 확정: 첫 점=출발, 끝 점=도착, 중간은 교차로 진입·진출 짝
   - 출력의 `total_length` / 창 경고 / `events` 개수를 눈으로 확인
3. `tools/plot_lane_graph.py` 로 경로를 그려서 **의도한 길인지 확인**
4. `python3 tools/probe_9910.py --host <VTD_IP>` 로 프레임 수신 확인
5. 종료 지점 좌표를 받으면 `scoring.finish_xy` 에 기입 (완주 판정·종점 정지가 이 값을 쓴다)

**연결 후**

6. `python3 run_agent.py --route data/route.pkl` → 20 Hz 로 Command 가 나가고
   매 틱 로그가 쌓이는지, 콘솔 1초 1줄에 `road/lane`·`s_route` 가 전진하는지
7. **조향 부호** — `comm.steer_sign: 1.0` 실측 확정값 (2026-08-19).
   ⚠ 자동 감시(SteerSignMonitor)는 **현재 구현되어 있지 않다** — 첫 주행 몇 초를
   눈으로 보고, 직선에서 발산하면 params 뒤집고 재기동.
8. junction 6 신호가 켜지는지 (연습 환경에서는 state=0 고정이었음) —
   첫 수 초 안에 light state 가 0 이 아닌 값으로 오는지 확인

**주행 후**

9. `python3 tools/summarize_run.py logs/<런>.jsonl` — 과속·이탈 즉시 확인
10. `python3 tools/score.py logs/<런>.jsonl --route data/route.pkl` — 감점 추정
11. 이상 항목만 `run_agent.py --replay` 로 재현·수정·재검증

## 실차 검증 기록 (구 ROS 구현 기준, 2026-08-23)

⚠ 아래는 **PDM-Lite 이식 이전의 자체 planner 구현**으로 얻은 기록이다.
현재 코드는 판단 계층이 통째로 교체됐으므로 **재검증이 필요하다.**

| 항목 | 결과 | 현재 코드 |
|---|---|---|
| 821 m 경로 완주 + 경로 끝 정지 | v=0 정지, 리스폰 0 | kr_rules.route_end 로 재구현 (2026-08-26 실기 후) |
| 제한속도 (S1.1.01) | 초과 0 틱, 실속도 편향 0.0 % | VtdRoutePlanner.speed_limits (margin 반영) |
| 적색 완전정지 (S5.1.01) | 앞범퍼가 정지선 1.10 m 앞 | PDM IDM 감속 + 전장 보정 주입(`speed.stop_gap_stopline_m`) + 최소 1 s 홀드. **명시적 정지 래치는 없음** |
| RTOR (적신호 우회전) | stopping→dwell→go 확인 | **미구현** |
| 차선변경 3/3 + 지시등 lead | 1.4 / 4.0 / 4.8 s | 경로 블렌드는 VtdRoutePlanner, 지시등은 kr_rules 로 재구현 (실차 재검증 필요) |
| 회전 지시등 3/3 | lead 최대 15 s | kr_rules 로 재구현 — 같은 경로에서 회전 3 / LC 3 구간 확인 (실차 재검증 필요) |
| 신호 폴백 | 도로 30 적색 정지 재현 | **미구현** |

## 미확인·미해결

| # | 내용 | 상태 |
|---|---|---|
| 1 | 점멸 신호(state 6) 황색/적색 구분 | 조직위 문의 대상. 현재 코드는 **Green 하드코딩** (route.py:84) |
| 2 | 방향지시등 선행 점등 "n초"의 규정값 | `signal.turn_lead_s` 4.0 / `signal.lc_lead_s` 3.0 가정값 — 규정 확정 시 yaml 만 수정 |
| 3 | RTOR 허용 여부 | 미구현 |
| 4 | 스쿨존 내 비신호 횡단보도 일시정지 채점 여부 | 폴백 오정지 위험 46곳과 상충 — 규정 확인 후 결정 |
| 5 | `RM_518` = 스쿨존 30 해석 | 텍스처 미검증 (build_lane_graph 가 30 으로 가정) |
| 6 | 5분 제한 초과 시 처리 | 평균 속도 전략에 직결 (`score.time_limit_s`) |
| 7 | 횡단보도 보행자 → 정지선 정지 | 미구현 (PDM 은 forecast 충돌 회피로 처리) |
| 8 | 라우터 연속 차선변경 이격 / waypoint 짝 활용 | 미착수 |
| 9 | `scoring.finish_xy` 미확보 | null 이면 route_s 임계 방식으로 폴백 + 경고 |

## 구현 상태

| 모듈 | 상태 |
|---|---|
| `vtd_adapter/comm` `frame` `carla_types` `types` `config` `logger` | 완전 구현 |
| `vtd_adapter/ego` | 창 기반 속도추정·차로매칭·route_s·리셋/스톨 감지 완비 |
| `vtd_adapter/world` `actor` `map` | CARLA 표면 흉내 완비 (객체 80 m/30개 규칙·코스팅 포함) |
| `vtd_adapter/route` | 경로 재샘플·LC 블렌드·테이퍼 보정·신호 대조·실선 LC 가드 완비 |
| `vtd_adapter/control` | 종방향 P + 감속 클램프 + jerk 제한 |
| `vtd_adapter/lanegraph` | 창 병합·cum_s·lookahead·dashed_runs(점선 단일 출처) 완비 |
| `team_code/autopilot` | PDM-Lite 원문 (`_manage_route_obstacle_scenarios` 는 stub) |
| `team_code/kr_rules` | route_end 종점 정지 + 적신호 정지선 최소 유지 + 방향지시등. RTOR·황색 딜레마는 미착수 |
| shield (corridor·실선·중앙선·TTC 비상제동) | **없음** — 안전망은 SAFE_STOP·watchdog·리셋 초기화뿐 |
| tools | replay·summarize_run·score·scp_client·batch_run·gen_scenarios 실전 사용 중 |
