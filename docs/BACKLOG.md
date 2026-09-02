# 백로그

착수하지 않기로 **결정한** 과제만 둔다. 각 항목은 재개할 때 다시 조사하지
않아도 되도록 **스펙·손댈 파일·손댈 테스트·재개 판단 근거**까지 적는다.
(`[미구현]` 표시는 AGENT_SPEC.md 가 규격 단위로 따로 관리한다.)

---

## B-1. `_stopline_hold` 리필 — 녹색 후 출발 지연

**상태**: 보류 (2026-08-30). 스펙·수정 대상 식별 완료.

**증상**: 적신호 정지 후 신호가 녹색으로 바뀌어도 최대 `speed.stopline_hold_s`
(1.0 s) 만큼 더 서 있는다. 실측(154015 런) **녹색 전환 → 출발 2.3 s / 1.9 s**.

**원인**: [kr_rules.py `_stopline_hold`](../team_code/kr_rules.py) 가 적색 + 근접 +
저속인 동안 **매 틱 잔여 틱을 다시 채운다**.

```python
if d_front < self.sl_near_m and ego_speed < self.latch_v:
    self.sl_hold_left = max(0, self.sl_hold_ticks - 1)   # ← 매 틱 재무장
    return 0.0
```

그래서 녹색이 되는 순간 항상 최대 1.0 s 의 잔여 홀드가 남는다. 규정은
"0.5 s 이상 정지"이므로 **한 번만 채우면 된다.**

**주의 — 리필은 의도된 동작이기도 하다**: 적색이 지속되는 동안 목표 0 을
유지하는 것이 리필이다. 고칠 것은 *적색 지속 중 유지*가 아니라 *녹색 전환 후
잔여*다. 둘을 가르지 않고 리필만 제거하면 적색 대기 중 재출발이 생긴다.

**수정 대상**
- 코드: `team_code/kr_rules.py::_stopline_hold`
- 테스트: `tests/test_stopline_stop.py::test_hold_rearms_while_red`
  — 현재 리필을 정상으로 단언한다. 녹색 전환 꼬리 부분을 함께 손봐야 한다.
- 경계 고정: `tests/test_stop_profile.py::test_hold_still_wins_after_green_while_profile_releases`
  가 "녹색 후 0 을 유지하는 주체는 홀드뿐"임을 이미 고정해 두었다.

**재개 판단**: 배치 report 의 **녹색후출발[s]** 관측값
([VALIDATION_stop_profile.md](VALIDATION_stop_profile.md) §3). 참고 기준 0.3 s.
2 s 대가 유지되면 재개한다.

---

## B-2. 종방향 jerk 비대칭 · 속도 추정 잡음 증폭

**상태**: 보류. 진단 완료, 설계안 미확정.

**뿌리 하나에서 나온 두 증상**
1. **정지 접근 커맨드 전환 다발** — 실측 정상 정지 6회 vs 오버슛 97회.
   ④′ 적용 후에도 폐루프 시뮬에서 **8~39회** 남는다 (④′ 로는 안 없어진다).
2. **보호구역 속도 리밋사이클** — 30 km/h 순항 추적 p-p **2.47 km/h**
   (±1 km/h 밴드 이탈). 주기 3.00 s = jerk 램프 왕복 시간.

**원인** ([control.py `_raw_accel`](../vtd_adapter/control.py))
```python
if err < 0.0: return max(err / self.dt, self.a_dec_max)   # 게인 1/dt = 20
return min(self.kp * err, self.a_max)                     # 게인 0.8  → 25배 비대칭
```
- 속도는 9910 에 필드가 없어 **위치 차분 추정**이다 (σ ≈ 0.22~0.44 m/s).
  게인 20 을 통과하면 ±4~9 m/s² 로 증폭돼 개시 시 항상 `a_dec_max` 포화.
- `control.jerk_dec_mult = 3.0` 이 제동 −0.30/틱 대 회복 +0.10/틱 비대칭을
  만들어, `−4.0 → +2.0` 복귀에 **정확히 3.00 s** 가 걸린다.

**기각된 접근** (2026-08-30 설계 비교, 근거는 그 대화 기록):
- IDM `dvdt` 피드포워드 — IDM 이 접근 중 **가속을 요구**한다(`err/dt` 최대
  +10.5). 저속 정지 악화.
- 대칭 `err/dt` — 위와 동일 + 순항 추적 σ 0.45 → 0.98 악화.
- 가속 분기 금지 — jerk 리미터가 이미 흡수해 실측 효과 수 cm.

**남은 방향**: 잡음을 상류에서 줄이거나(`percep.speed_win_s`), 감속 분기에
데드밴드를 두거나, jerk 비대칭을 재검토. **④′ 실주행 검증 이후로 미룬다** —
정지 위치가 잡히면 이 항목의 우선순위가 달라질 수 있다.

---

## B-3. 정지 프로파일 route_end 확장

**상태**: 보류 — ④′ 정지선 검증 통과 후 착수.

현재 `_stopline_profile` 은 **정지선 한정**이다. 종점 정지는 여전히
"길이 0 유령 선행차 + IDM" 이라 같은 오버슛 기제를 안고 있다
(`route_end` 후보, `speed.stop_gap_route_end_m = 4.0`).

**착수 조건**: [VALIDATION_stop_profile.md](VALIDATION_stop_profile.md) §2 의 4개
판정 기준 통과.

**작업 크기**: `_stopline_profile` 을 기준점·s0 인자화해 `stop_s` 에도 적용.
`tests/test_route_end.py` 의 계획 정지점 단언이 함께 바뀐다.

---

## B-4. pytest 정식 설치 — 해소됨 (2026-09-02)

**상태**: 해소. `pytest 7.4.4` 가 설치돼 있고 `python3 -m pytest -q` 가
**591 passed / 1 skipped, 실패 0** 이다 (~90 s). 아래 37건은 전부 shim 러너의
한계였고 제품 코드 결함이 아니었음이 실제 실행으로 확인됐다. 기준선은 이제
"변경 전후 실패 건수가 같은지" 가 아니라 **실패 0** 이다.

<details><summary>경위 (해소 전 기록)</summary>

**상태**: 환경 문제. 코드 아님.

이 환경에는 `pytest` 가 없다 — 시스템 파이썬은 PEP 668 로 설치가 막히고
`python3 -m venv` 도 실패한다. 그래서 검증은 **최소 shim 러너**로 돌리고 있다.

**영향**: 전체 스위트에서 **37건이 고정 실패**한다. 전부 shim 이 지원하지 않는
builtin fixture 탓이며 **제품 코드 결함이 아니다**:
`tmp_path` / `tmp_path_factory` / `monkeypatch` / `capsys`, 인자 있는 fixture
(`vmap(lg)`, `rig(lg)`).

해당 파일: `test_adapter`(3) `test_batch_lists`(5) `test_camera_scp`(1)
`test_csv_warn_gate`(4) `test_finish_target`(2) `test_gen_scenarios`(13)
`test_overtake`(8) `test_solid_lc`(1).

**해소되면**: 이 37건이 실제로 통과하는지 처음으로 확인된다. 그 전까지
"239 passed / 37 failed" 의 37 은 **변경 전후 동일**함만 근거로 쓴다.

</details>

---

## B-5. `a_stop` 종단 — 적신호 조기 정지 (항목7 경미)

**상태**: 보류 (2026-08-30). 황색 검증 **판정에서 제외**, 관측만 한다.

**증상**: ④′ 적용 후 일부 적신호 정지가 **너무 멀리** 선다. 실측
(20260830_181706 실전주행_01) `slf = −3.25 / −3.51`. `score.stop_ok_m = 2.0`
이므로 둘 다 **항목7 `red_stop_far` 경미**로 잡힌다.

```
── 적신호 원거리 정지 (2건) ──
  s 304.7  t 44.0→48.9 s  front_m=-3.25
  s 688.4  t 89.9→102.8 s  front_m=-3.51
```

**원인 가설(미검증)**: `v_allow = √(2·a_stop·(d−s0))` 는 `d = s0` 에서 0 이
되지만, 종단에서 `err/dt`(게인 20)가 `a_dec_max` 로 포화한 뒤 `a_hold` 로
넘어가는 구간이 있어 계획보다 일찍 멈출 수 있다. `a_stop` 을 올리면 늦게
잡고, 내리면 더 일찍 선다 — 종단 처리와 함께 봐야 한다.

**황색과 무관하다**: 황색 STOP 폐루프 13조건은 −1.52 ~ −0.97 로 전부 밴드
안이다. 이 항목은 **적색 원거리 접근**에서만 나온다.

**관측 유지**: `report.txt` 정지 지표 표의 `slf[m]` 열이 그대로 근거다
(별도 카운터를 두려면 score.py 변경이 필요 — 채점 파트). 황색 배치 판정에는
이 값을 쓰지 않는다.

**재개 조건**: 황색 검증 통과 후, B-2(종방향 jerk·잡음)와 묶어서 본다 —
종단 거동이 같은 뿌리일 가능성이 있다.

---

## B-6. gen_scenarios 황색 타이밍 축 (경계 요청 1호)

**상태**: 보류 — **대회 후**. 우회안 채택으로 당장은 불필요.

**요청 내용**: `configs/themes.yaml` 의 `신호` 축(현재 `적60 / 딜레마 /
짧은녹색`)에 황색 onset 을 STOP/GO 로 **결정적으로** 가르는 변형이 없다.
`딜레마` 하나로는 어느 쪽이 나올지 시드에 달린다.

**우회안 (채택)**: `--seed` 를 바꿔 여러 판 생성한 뒤 **사후 분류**한다 —
로그 `decision.reasons.yellow` 가 판정 틱에만 채워지므로 그 필드로 STOP/GO 와
이른/늦은을 그대로 가른다 (`docs/VALIDATION_stop_profile.md` 참조).
재현성이 시드에 묶이는 것이 유일한 대가다.

**경계**: `configs/themes.yaml` · `tools/gen_scenarios.py` 는 **채점·시나리오
파트**다 (CLAUDE.md). 제어기 파트 세션에서 손대지 않는다.

