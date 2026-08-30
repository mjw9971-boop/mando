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

---

# 황색 딜레마 (C) 검증 — 추가분

대상 커밋 `68118cc`. 상수 둘은 **역할이 다르다** — `speed.a_yellow = 4.0` 은 STOP/GO **판정 전용**, `speed.stop_profile_a = 3.0` 은 STOP 시 **실행** 프로파일이다.

## 7. 시나리오 생성 (사후 분류 방식)

`configs/themes.yaml` 에 황색 타이밍을 STOP/GO 로 **결정적으로** 가르는 축이
없다 (BACKLOG B-6, 대회 후). 그래서 시드를 바꿔 여러 판 만들고 **사후에**
분류한다.

```bash
python3 tools/gen_scenarios.py 정지선접근 --count 4 --seed 11
python3 tools/gen_scenarios.py 정지선접근 --count 4 --seed 23   # 부족하면 시드 추가
python3 tools/batch_run.py scenarios/batch_정지선접근.json
```

## 8. 사후 분류 절차 — **한 필드로 끝난다**

로그 `decision.reasons.yellow` 를 본다. 이 필드는 **황색 판정이 일어난 그
1틱에만** 채워지고 나머지는 `null` 이다 (판정은 접근당 1회이므로 그 1틱이
조건을 전부 담는다). 따라서 **`yellow` 가 non-null 인 틱만 뽑으면 그 런의 모든
황색 접근이 한 줄씩 나온다.**

```bash
python3 - <<'EOF'
import json, glob
for f in sorted(glob.glob('logs/batch/<ts>/*.jsonl')):
    for line in open(f):
        y = json.loads(line)['decision']['reasons'].get('yellow')
        if y:
            kind = '이른(STOP)' if y['decision'] == 'stop' else '늦은(GO)'
            print(f"{f.split('/')[-1]:28s} ctrl={y['ctrl']:4d} {kind}"
                  f"  v={y['v']:5.2f}  v_allow={y['v_allow']:6.2f}  d={y['d_line']:6.2f}")
EOF
```

각 필드의 뜻:

| 필드 | 뜻 |
|---|---|
| `decision` | `stop` / `go` — **이것이 곧 이른/늦은 황색이다** (정의상 동치) |
| `v` / `v_allow` | 판정 순간의 속도와 임계. `v ≤ v_allow` 면 STOP |
| `d_line` | 판정 순간 뒷축→정지선 [m] |
| `ctrl` | 신호 controller id — `report.txt` 정지 지표 표의 `ctrl` 열과 대조해 **같은 접근의 결과 slf** 를 찾는다 |

**이른/늦은을 따로 판별할 필요가 없다.** `v ≤ v_allow(4.0)` 가 곧 "쾌적 감속으로
설 수 있다" = 이른 황색이고, 그것이 그대로 STOP 판정이다.

**STOP 접근의 결과**는 `report.txt` 정지 지표 표에서 같은 `ctrl` 행의 `slf[m]`.
**GO 접근은 정지 이벤트가 없어 그 표에 안 나온다** — 대신 `<name>.score.txt` 의
`적신호 통과` 섹션에 그 `ctrl` 이 있으면 **위반**이다 (아래 §9).

## 9. 판정 기준

| # | 대상 | 기준 | 보는 곳 |
|---|---|---|---|
| 1 | STOP 전건 | **−2.0 ≤ slf ≤ −0.3**, 걸침 0 | `report.txt` 정지 지표 표 |
| 2 | GO 전건 | 통과 순간 신호가 **비적색** | `score.txt` `적신호 통과` 섹션에 해당 ctrl 이 **없어야** 한다 |
| 3 | 역회귀 | 적색·녹색·보행자 거동 무변화 | 기존 배치 재실행 |

**2번이 깨지면 즉시 보고**한다 — GO 로 나갔는데 적색에 통과한 것이고,
채점 항목7 **중대**다. `score.detect_red_light` 가 통과 순간의 신호로 판정하므로
딜레마존 정당성과 무관하게 위반으로 잡힌다.

## 10. 판정에서 제외 — 관측만

**적신호 원거리 정지** (`slf < −2.0`). 실측 −3.25 / −3.51 로 이미 발생 중이며
**황색과 무관한 별건**이다 (BACKLOG B-5). 정지 지표 표의 `slf[m]` 로 관측만 하고
황색 합격 판정에는 쓰지 않는다.

**녹색후출발[s]** — 기존대로 관측만 (§3, BACKLOG B-1).

## 11. 되돌리기

```yaml
speed:
  a_yellow: 0.0        # 황색 판정만 끈다 (황색을 PDM 원문에 맡기는 이전 동작)
  stop_profile_a: 0.0  # ④′까지 함께 끈다
```

