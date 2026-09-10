# 작업 파트 경계

이 저장소는 **두 파트**로 나눠 작업한다. 세션은 **기본이 제어기 파트**다.
채점·시나리오 파트는 사용자가 그 세션에서 **명시적으로 선언했을 때만** 활성이다.

## 제어기 파트 (기본)

| | 대상 |
|---|---|
| **수정 가능** | `team_code/`, `vtd_adapter/`, `run_agent.py`, `config/params.yaml` 의 `speed.*` · `control.*` · `vehicle.*` 키 |
| **수정 금지** | `tools/batch_run.py` · `tools/build_route.py` · `tools/gen_scenarios.py` · `tools/score.py` · `tools/scp_client.py` · `tools/summarize_run.py` · `tools/finish_cone.py`, `configs/themes.yaml`, `config/params.yaml` 의 `scoring.*` · `batch.*` · `gen_*` (`gen_placement.*` 포함) · `route.*` 중 **경로 조립 키** (`dp_*` · `candidates_*` · `finish_tail_*` · `waypoint_*`) |

## 채점·시나리오 파트 (사용자가 세션에서 선언할 때만)

수정 가능/금지가 **정확히 뒤집힌다**.

| | 대상 |
|---|---|
| **수정 가능** | `tools/batch_run.py` · `build_route.py` · `gen_scenarios.py` · `score.py` · `scp_client.py` · `summarize_run.py` · `finish_cone.py`, `configs/themes.yaml`, `params.yaml` 의 `scoring.*` · `batch.*` · `gen_*` (`gen_placement.*` 포함) · `route.*` 중 **경로 조립 키** (`dp_*` · `candidates_*` · `finish_tail_*` · `waypoint_*`) |
| **수정 금지** | `team_code/`, `vtd_adapter/`, `run_agent.py` |

`route.*` 는 **두 파트가 나눠 쓴다**: `taper_blend_m` · `lc_move_*` 는
`vtd_adapter/route.py`(제어기 플래너)가 읽으므로 **제어기 파트** 키다.
경로를 *짓는* 키(`dp_*` 등)만 채점·시나리오 파트 소관이다.

## 경계 밖이 필요할 때 — **고치지 말고 보고한다**

작업 도중 반대편 파트의 파일·키를 고쳐야 한다는 판단이 서면, **손대지 않고**
보고 말미에 다음 형식으로 남긴다. 사용자가 파트를 전환해 처리한다.

```
## 경계 요청
- 대상: <파일 또는 params 키>
- 필요한 변경: <한 줄>
- 이유: <왜 이번 작업이 이것 없이는 불완전한가>
- 우회 여부: <우회 가능하면 그 방법 / 불가능하면 명시>
```

경계 밖 파일을 **읽는 것은 자유다** — 분석·대조·근거 확보에 필요하다.
금지되는 것은 **쓰기**뿐이다.

## 경계에 걸리지 않는 것

- `tests/` — 양쪽 파트 모두 자기 변경에 대한 테스트를 쓴다.
- `docs/` — 보고·백로그·검증 시트.
- 새로 만드는 파일은 그 파일이 속할 파트의 규칙을 따른다.

---

# 이 저장소에 대해

한국 자율주행 대회(HL FMA 2026) 에이전트. VTD 시뮬레이터와 9910 포트로
통신하며, 판단은 PDM-Lite(`team_code/autopilot.py`) 이식본이 한다.

## 구조

```
9910 → Comm.recv → EgoTracker/VtdWorld → autopilot.run_step → Comm.send → 9910
                                              ↘ kr_rules.apply (한국 규칙 계층)
```

- `team_code/autopilot.py` — **PDM-Lite 원문. 무수정이 원칙**이다.
  VTD 접합은 `# VTD:` 주석이 달린 최소 지점뿐이다.
- `team_code/kr_rules.py` — 한국 대회 규칙 계층. `_get_control` 맨 끝의
  `kr_rules.apply(...)` 한 줄이 유일한 접점이고, **PDM 의 `min()` 중재에
  후보를 덧대는 형태로만** 개입한다. 외부 오버라이드 금지.
- `team_code/ctrl24.py` — **won_24 제어기 (2026-09-08, 기본 on).** kr_rules 와 같은
  접점(apply / signal_release) + `pre_pass` 훅(autopilot 이 PDM 후보 계산 앞에서
  부른다). `config/params.yaml` `ctrl24.enable` 로 고른다 — `false` 면 kr_rules 경로이고
  won 과 틱 단위 동일(리플레이 diff 0 확인). 상수는 `ctrl24:` 섹션이 단일 출처.
  정적 장애물은 첫 틱 PREEMPT(게이트 span_too_far 하나)·중첩 시프트, 종점에 서지
  않고 계속 주행(route.py 종점 패드 300 m, batch_run 완주 래치).
  geom 게이트를 삭제한 대가는 **기각이 아니라 속도로** 갚는다 — K7 shift_cap
  (`ctrl24.shift_cap_enable`, 상수는 `overtake.a_lat_max`·`shift_cap_min_v`·
  `shift_latest_m`)이 시프트 활성 구간에서만 곡률 상한을 min() 후보로 낸다.
  시프트 목표 이웃이 span 중간에서 끊기는 것도 기각하지 않는다 — K8 span_v_req
  (`ctrl24.span_v_req_enable`)이 장애물 지점에 목표가 없으면 반대편 side 로 넘기고,
  뒤가 끊기면 연속 창에 전이 2회가 들어가는 속도를 min() 후보로 낸다 (B-30).
  무한정지 방지는 규칙 하나다 — K9 탈출 바닥 (`ctrl24.escape_enable`, 기본 off):
  경로 진행이 멈추고 정당한 정지 원인(신호·보행자·정지선 홀드)이 없으면 정당하지
  않은 후보에만 바닥을 깐다. 정당한 원인은 바닥을 안 받아 구조적으로 못 뚫는다.
- `vtd_adapter/` — CARLA 표면을 흉내내는 어댑터 (플래너·월드·제어·로거).
- `config/params.yaml` — VTD·차량·판정 상수의 단일 출처.
  판단(IDM·forecast·lateral) 상수는 `team_code/config.py` 가 단일 출처다.

## 작업 관례

- **상수는 한 곳에서만** 정의한다. 값을 두 곳에 적지 말고 한쪽이 다른 쪽을
  읽게 한다 (예: `kr_rules` 가 PDM 주입값 `idm_red_light_minimum_distance` 를
  그대로 읽는다).
- 주석은 **왜**를 적는다. 실측 근거가 있으면 날짜·로그와 함께 남긴다.
- 기능 추가는 **끄는 스위치**를 함께 둔다 (`0` 또는 `false` 로 이전 동작).
- **경유점 위치는 "차로 안 s" 로만 넘긴다.** 경로 조립부(`build_route`)는
  마지막 경유점의 `(차로, s)` 로 종료선 뒤 꼬리를 계산한다. 여기에 차로에
  **진입한** s 를 넣으면 꼬리가 과대 계산돼 `finish_tail` 연장이 안 걸리고,
  계획대로 정지해도 **완주 판정 임계에 8~9 m 미달**한다 (2026-09-06 전역 DP
  도입 때 실제로 발생, `test_batch_finish_judge` 11건이 잡았다). 같은 축의
  실수로 `infeasible_forced` 에 구간 인덱스 대신 차로 인덱스를 넣은 것도 있었다
  — 리포트가 "구간 N" 으로 찍으므로 번호가 어긋난다.

## 검증

- `pytest` 는 이제 이 환경에 **설치되어 있다** (7.4.4). 예전의 "고정 실패
  37건" 은 shim 러너가 builtin fixture 를 지원하지 못해 생긴 것이었다
  (2026-09-02 확인, `docs/BACKLOG.md` B-4). 통과 수는 테스트가 늘면 같이
  오르므로, 회귀 판단은 개수가 아니라 **실패 목록이 늘었는지**로 한다.

  **현재 기준선 (2026-09-07 실측, `python3 -m pytest -q`, ~7.0 분)**:
  `1303 passed / 2 skipped / **0 failed**`. 전부 통과한다.

  실패가 하나라도 나오면 그건 회귀다.

  2026-09-07 이전의 "실패 17~24건(실행 순서에 따라 변동)" 은 정리됐다.
  원인은 전부 **params 기본값과 테스트의 드리프트** 하나였다 — 기능을
  `params.yaml` 에서 켜 놓고 "기본값이 꺼져 있는지" 보는 테스트를 같이 안
  고친 것이다. 플래그를 하나씩 꺼 보며 확인한 결과 실패 17건이 params 플래그
  5개에 **1:1 로 대응**했고, 독립적으로 깨진 테스트는 없었다. 실행 순서 의존
  (17~24 변동)도 같은 플래그가 만든 것이라 함께 사라졌다.

  처리 원칙 (다시 생기면 그대로 따를 것):

  · **키마다 "params 가 맞나 테스트가 맞나" 를 따로 판정한다.** 판정 기준은
    그 키를 넣은/바꾼 커밋의 의도다. 일괄 처리 금지.
  · params 가 틀린 사례가 실제로 있었다 — `red_zone.roadmark_30_as_limit` 은
    무제목 커밋 `c491357 "루트 생성"` 이 작업 2a 의 `false` 를 뒤집은 것이라
    되돌렸다 (그대로 두면 다음 `build_lane_graph` 실행에서 붉지 않은 30 표시
    도로 10,237 m 가 30 캡으로 돌아간다).
  · 나머지 5개(`speed.ped_multi_enable` · `overtake.shift_entry_enable` ·
    `side_pick_enable` · `signal_stale_queue_enable` · `standoff_creep_enable`)
    는 **제어기 파트 소관**이라 params 를 건드리지 않고 **테스트를 현재
    기본값에 맞췄다**. 각 테스트에 "params 값이 정본, 팀원 소관" 주석과
    `docs/BACKLOG.md` B-25 참조가 달려 있다.
  · 켜고 끄는 동작 자체의 커버리지는 잃지 않았다 — off 경로를 보는 검사는
    기본값을 읽는 대신 **사본에서 명시적으로 끄고**(`off_cfg()` / `a1_cfg()` /
    `OFF`) 검사하도록 바꿨다. 기본값이 또 바뀌어도 안 깨진다.

  회귀 판단은 전체 실행으로 충분하지만, 어느 파일인지 좁힐 때는 파일별
  격리 실행이 편하다:

  ```
  for f in tests/test_*.py; do echo "$f: $(python3 -m pytest $f -q 2>&1 | tail -1)"; done
  ```

  옛 기준선 `1069 passed / 1 failed` (2026-09-05) 의 알려진 실패였던
  `test_batch_finish_judge.py::…[정적회피집중_08_직진11.csv]` 는 지금 없다 —
  그 시나리오가 `scenarios/` 에 더는 없어서다 (`scenarios/` 는 `.gitignore`
  대상이라 재생성하면 목록이 바뀐다). `test_batch_finish_judge` 는 현재 전건
  통과한다.

- 실주행 배치는 VTD PC 에서 사용자가 돌린다. 이 환경에서는 **리플레이와
  폐루프 시뮬**까지만 가능하다. 시뮬 수치를 실주행 예측으로 제시하지 않는다.
- 배치 결과는 `logs/batch/<ts>/report.txt` 에 남고, 로그를 지워도 **정지 지표
  표는 남는다**.

## 확정 사실 (재조사 금지)

조사가 끝나 결론이 난 것들. **다시 파지 말 것** — 근거와 함께 여기 남긴다.

### 채점 범위
**채점은 주최측 안내문의 15개 항목이 전부다. 그 밖의 동작으로는 감점되지
않는다.** 안내문이 "각 평가항목의 위반 기준 및 경미/중대 구분은 아래와
같습니다" 로 15개를 열거한다 (부호화된 목록은 `tools/score.py` `ITEMS`).
무신호 교차로·정지선 일시정지는 **항목에 없다**. 그래서 AGENT_SPEC §7 의
"스쿨존 비신호 횡단보도 일시정지 채점 여부" 는 2026-09-07 종결·삭제했다.

### junction_ctrl_map 의 8개 교차로 "누락" 은 오류가 아니다
`xodr <junction><controller>` 보유 53개 vs 맵 키 45개의 차집합
`[11, 27, 47, 48, 50, 52, 71, 85]` 은 **정상**이다.

- 이웃 교차로와 7~20 m 토막 도로로 붙은 **쌍둥이 교차로의 반쪽**이고
  **자기 신호가 없다** (8개 전 접근로 신호 0개).
- controller 14개는 xodr 이 양쪽 교차로에 **중복 선언**했을 뿐, 실제 등은
  이웃 교차로 정지선(s≈L, ori=+) 또는 far-side(s≈0, ori=+)에 있다.
- 그래서 현재 맵 귀속(j11→j20, j27→j6, j47→j46, j48·j50→j49, j52→j53,
  j71→j72, j85→j86)이 **물리적으로 맞다**. xodr 귀속으로 바꾸면 정상 교차로
  7개의 신호를 잘못 옮긴다.
- **맵 귀속을 바꾸지 말 것. 재조사 금지** (2026-09-07 결론).

### 무신호 정지선은 제어기가 수집 단계에서 버린다
지도 전체 정지선 576개 중 **245개가 신호 미매핑**이다.

- [route.py:355](vtd_adapter/route.py#L355) `collect_stops` 가
  `controller_ids or signal_ids` 인 것만 모은다 → `VtdTrafficLight` 가 아예
  안 만들어진다. PDM 정지표지 경로도 stub (`next_stop_signs` 전부 None).
- 따라서 `kr_rules._stop_target_raw` 의 색 해석이 "신호 없음 → None" 으로
  끝난다. **UNKNOWN 으로 세우지 않는다 — 세울 객체가 없으므로 잠기지도,
  풀 조건이 필요하지도 않다.**
- **무한 정지의 원인이 될 수 없다.** 로그 실측(2026-09-07, 36개 로그):
  무신호 정지선 앞 통과 3,966틱에서 `reasons.stop_line` 이 **전부 null**,
  속도 중앙 22.6 km/h, 그 지점 **최장 연속 정지 2.0 s**.
- 무신호 정지선이 쓰이는 곳은 회피 억제(`_signal_zone`)와 큐 판정뿐이다.

## 진행 중 과제

`docs/BACKLOG.md` — 보류 결정된 과제와 재개 근거.
`docs/VALIDATION_stop_profile.md` — ④′ 정지 프로파일 실주행 검증 시트.
