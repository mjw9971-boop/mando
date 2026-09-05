# 작업 파트 경계

이 저장소는 **두 파트**로 나눠 작업한다. 세션은 **기본이 제어기 파트**다.
채점·시나리오 파트는 사용자가 그 세션에서 **명시적으로 선언했을 때만** 활성이다.

## 제어기 파트 (기본)

| | 대상 |
|---|---|
| **수정 가능** | `team_code/`, `vtd_adapter/`, `run_agent.py`, `config/params.yaml` 의 `speed.*` · `control.*` · `vehicle.*` 키 |
| **수정 금지** | `tools/batch_run.py` · `tools/build_route.py` · `tools/gen_scenarios.py` · `tools/score.py` · `tools/scp_client.py` · `tools/summarize_run.py`, `configs/themes.yaml`, `config/params.yaml` 의 `scoring.*` · `batch.*` 키 |

## 채점·시나리오 파트 (사용자가 세션에서 선언할 때만)

수정 가능/금지가 **정확히 뒤집힌다**.

| | 대상 |
|---|---|
| **수정 가능** | `tools/batch_run.py` · `build_route.py` · `gen_scenarios.py` · `score.py` · `scp_client.py` · `summarize_run.py`, `configs/themes.yaml`, `params.yaml` 의 `scoring.*` · `batch.*` |
| **수정 금지** | `team_code/`, `vtd_adapter/`, `run_agent.py` |

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
- `vtd_adapter/` — CARLA 표면을 흉내내는 어댑터 (플래너·월드·제어·로거).
- `config/params.yaml` — VTD·차량·판정 상수의 단일 출처.
  판단(IDM·forecast·lateral) 상수는 `team_code/config.py` 가 단일 출처다.

## 작업 관례

- **상수는 한 곳에서만** 정의한다. 값을 두 곳에 적지 말고 한쪽이 다른 쪽을
  읽게 한다 (예: `kr_rules` 가 PDM 주입값 `idm_red_light_minimum_distance` 를
  그대로 읽는다).
- 주석은 **왜**를 적는다. 실측 근거가 있으면 날짜·로그와 함께 남긴다.
- 기능 추가는 **끄는 스위치**를 함께 둔다 (`0` 또는 `false` 로 이전 동작).

## 검증

- `pytest` 는 이제 이 환경에 **설치되어 있다** (7.4.4). 예전의 "고정 실패
  37건" 은 shim 러너가 builtin fixture 를 지원하지 못해 생긴 것이었고,
  실제 pytest 로 돌리면 **전부 통과한다** (2026-09-02 확인, `docs/BACKLOG.md` B-4).
  통과 수는 테스트가 늘면 같이 오르므로, 회귀 판단은 개수가 아니라
  **실패 목록이 늘었는지**로 한다.

  **현재 기준선 (2026-09-05 실측, `python3 -m pytest -q`, ~165 s)**:
  `1069 passed / 2 skipped / 1 failed`.

  | 알려진 실패 | 원인 | 성격 |
  |---|---|---|
  | `test_batch_finish_judge.py::test_planned_stop_is_inside_threshold[scenarios/정적회피집중/정적회피집중_08_직진11.csv]` | 그 경로의 종료선 뒤 꼬리가 10.0 m 로 요구 12 m 에 못 미쳐 계획 정지점이 클립된다 (`[경고] 종료선 뒤 꼬리 …` 가 빌드 때 함께 뜬다) | **생성물 의존** — `scenarios/` 는 `.gitignore` 대상이고 `gen_scenarios` 를 다시 돌리면 사라지거나 다른 경로로 옮겨간다. 코드 결함이 아니다 |

  이 목록에 없는 실패가 나오면 그건 회귀다. `scenarios/` 를 재생성했다면
  목록을 실제 상태에 맞게 갱신한다.
- 실주행 배치는 VTD PC 에서 사용자가 돌린다. 이 환경에서는 **리플레이와
  폐루프 시뮬**까지만 가능하다. 시뮬 수치를 실주행 예측으로 제시하지 않는다.
- 배치 결과는 `logs/batch/<ts>/report.txt` 에 남고, 로그를 지워도 **정지 지표
  표는 남는다**.

## 진행 중 과제

`docs/BACKLOG.md` — 보류 결정된 과제와 재개 근거.
`docs/VALIDATION_stop_profile.md` — ④′ 정지 프로파일 실주행 검증 시트.
