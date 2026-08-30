# 실주행 검증 시트 — 정지선 정지 프로파일 (④′)

대상 커밋 `ebf2175` (`speed.stop_profile_a = 3.0`).
이 문서는 **VTD PC 에서 실행할 배치와 판정 기준**만 담는다. 분석 근거는 커밋 메시지에 있다.

## 0. 사전 조건

시나리오 XML 이 VTD PC 에 있어야 한다 (`vtd_xml_path` 프리픽스 `/home/mjw/scenarios`).

```bash
# 컨트롤러 PC → VTD PC (경로는 gen_scenarios --vtd-dir 기본값 기준)
scp -r scenarios/정지선접근 scenarios/실전주행 mjw@192.168.10.1:/home/mjw/scenarios/
```

배치 실행 전 스위치 확인:

```bash
grep -n "stop_profile_a" config/params.yaml     # 3.0 이어야 한다 (0 = 기능 비활성)
```

## 1. 실행

```bash
# 3건 한 번에 (실전주행 1 + 정지선저속 1 + 정지선고속 1)
python3 tools/batch_run.py scenarios/batch_all.json

# 나눠 돌릴 때
python3 tools/batch_run.py scenarios/batch_정지선접근.json     # 저속·고속 진입
python3 tools/batch_run.py scenarios/batch_실전주행.json       # 역회귀(혼합 이벤트)

# 계획만 확인
python3 tools/batch_run.py scenarios/batch_all.json --dry-run
```

산출: `logs/batch/<타임스탬프>/report.txt` 의 **정지선 정지 지표** 표.
로그를 지워도 이 표는 남는다 (그래서 지표를 report 에 넣었다).

## 2. 판정 기준

`report.txt` 정지 지표 표의 `판정:` 줄로 본다.

| # | 기준 | 표에서 보는 곳 |
|---|---|---|
| 1 | **정지선 침범 0건** | `침범(slf>0) 0건` |
| 2 | **조건 내 산포 < 0.3 m** | 같은 시나리오·같은 접근속도대 행끼리 `slf[m]` 편차 |
| 3 | **max slf ≤ −0.3 m** | `최대 −0.xx` |
| 4 | **A형 정상 착지 재현** | 접근 43~48 km/h 행이 `slf ≈ −1.5`, `전환` 한 자리 |

※ 2번은 **조건 내** 산포다. 저속·고속을 한 통에 넣은 전체 p-p 가 아니다
(폐루프 시뮬 기준 조건 내 0.06~0.52, 전체 1.31).

## 3. 관측 항목 (판정 아님)

**녹색후출발[s]** — 녹색 전환 → `v > 0.3 m/s` 까지의 시간.

- 참고 기준 **< 0.3 s**. 이 값을 넘어도 **이번 배치의 합격/불합격과 무관하다.**
- 용도: `_stopline_hold` 리필 수정(보류, [BACKLOG.md](BACKLOG.md) B-1) **재개 여부 판단**.
- 기존 실측(154015 런, ④′ 이전): **2.3 s / 1.9 s**. 이 수치가 그대로면 재개를 검토한다.

## 4. 실패 시 되돌리기

```bash
# 기능만 끄기 (코드 되돌림 없음)
#   config/params.yaml
speed:
  stop_profile_a: 0.0
```

`0.0` 이면 후보를 만들지 않는다 = ④′ 이전 동작. 커밋 되돌림은
`git revert ebf2175`.

## 5. 튜닝 여지

폐루프 시뮬 기준 `stop_profile_a` 는 **2.5~3.5 가 양호**, 4.0 이상은 악화한다
(프로파일이 고속을 오래 허용해 늦게 잡는다). 종단 급제동 체감이 문제면
2.5 쪽으로 내린다.

## 6. 알려진 한계

진입 시점에 이미 프로파일 위인 경우(`v > √(2·a_stop·(d−s0))`)는 `a_stop` 과
무관하게 `control.a_dec_max`(−4.0)와 jerk 램프인 시간이 지배한다 — ④′ 가
잡지 못한다. 늦은 적색 인지에서 발생하며, 별도 과제다
([BACKLOG.md](BACKLOG.md) B-2).
