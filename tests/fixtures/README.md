# tests/fixtures — 테스트 전용 고정 입력

여기 있는 파일은 **테스트의 기대값이 매달린 기준(baseline)** 이다. 값을 바꾸면
그 값을 기준으로 쓴 테스트가 함께 깨진다. 사람이 의도적으로 갱신할 때 말고는
어떤 도구도 여기에 쓰지 않는다.

| 파일 | 무엇 | 쓰는 곳 |
|---|---|---|
| `route.pkl` | 821.1 m · 차로 26 · 경유점 8 짜리 고정 경로 | `test_overtake` `test_turn_signal` `test_solid_lc` `test_lc_corridor` `test_static_obstacle` |
| `waypoints.csv` | 위 `route.pkl` 을 만든 경유점 CSV (seq,x,y) | 재생성 근거 — 현재 테스트가 직접 읽지는 않는다 |
| `pair_lane_offset_waypoints.csv` | 짝 진입 차로가 경유점에서 한 차로 벗어나는 경로 (경유점 6) | `test_route_check_pair` `test_build_route_pair` |

`route.pkl` 은 `waypoints.csv` 로부터 재현 가능하다:

```bash
python3 tools/build_route.py data/lane_graph.pkl tests/fixtures/waypoints.csv \
        -o tests/fixtures/route.pkl
```

## 기준 vs 산출물

레포 루트의 `waypoints.csv` 와 `data/route.pkl` 은 **사용자 작업용 산출물**이다.
대회날 배포 CSV 를 넣거나 경로를 눈으로 확인하려고 자유롭게 덮어써도 된다 —
테스트는 그쪽을 보지 않는다.

과거에는 두 테스트 묶음이 `data/route.pkl` 을 직접 봤고, 경로 시각화를 한 번
돌려 그 파일이 바뀔 때마다 테스트 4개가 깨졌다. 9/3 대회 CSV 를 넣는 순간
같은 일이 반복되므로 기준을 여기로 분리했다.

`data/lane_graph.pkl` 은 여기로 옮기지 않았다 — xodr 에서만 파생되고 xodr 이
바뀔 때만 재생성하는 안정 자산이라, 주행·채점·테스트가 같은 것을 봐야 한다.

## `venue_20260903_waypoints.csv`

2026-09-03 대회장 배포 경유점 CSV (10점, 대회형식 = 시작 + 교차로 짝 4 + 종료).
`tests/test_build_route_candidates.py` 가 읽는다 — 개수 기반 후보 수집(k=40)이
seq 8→9 에서 "경로 없음" 으로 깨지고 반경 기반은 `(836,0,-1)` 로 통과하는,
이 저장소에서 유일하게 그 차이가 드러나는 CSV 다.

레포 루트의 `waypoints.csv`(32점)와는 다른 파일이다. 루트 쪽은
`configs/themes.yaml` `routes.기본` 이 "검증된 연습 경로" 로 읽으므로 교체하지
않는다.

## `pair_lane_offset_waypoints.csv`

**짝 진입 차로가 경유점에서 한 차로(~3 m) 벗어나는** 경로다. 작업11 이 세운
"차로 반폭 기준 폐기 · 짝 경유점은 [1] 이탈 판정에서 제외" 명세를 검증하는
기준이라, `test_route_check_pair` 6건과
`test_build_route_pair.test_switch_off_is_previous_behaviour` 가 이것에 매달려
있다 (짝 로직 on/off 로 짝 2·4 의 진입 차로가 실제로 갈리는 케이스).

출처: `logs/batch/20260903_012155/routes/route_정적회피집중_01_좌회전2.pkl` 의
`waypoints` + `waypoint_seq` 를 복원한 것. 같은 경유점이 5개 배치
(`20260903_012155` · `171320` · `171455` · `173432` · `191825`) 에 남아 있고
**전부 md5 가 같다** — 원본 경유점은 하나뿐이다.

왜 옮겼나 (2026-09-04, 작업17): 예전에는
`scenarios/정적회피집중/정적회피집중_01_좌회전2.csv` 를 직접 가리켰다.
`scenarios/` 는 `.gitignore` 대상 생성물이라 git 이 되돌려 주지 않고, 시드
스트림이 조금만 달라져도 **같은 이름에 다른 경로**가 들어온다. 실제로
추월집중 재생성이 막혀 시드가 밀린 것만으로 이 파일 관련 테스트 6건이
"파일은 있는데 내용이 달라서" 깨졌다.

옛 로그 pkl 을 그대로 두지 않고 CSV 만 반입한 이유: 테스트의 `_build()` 가
`read_waypoints_csv` → `build_route` 구조라 **경유점만 있으면 현재 코드로
재현**된다. 실제로 이 CSV 를 현재 코드로 빌드하면 515.9 m / 차로 9 로, 9/3
당시 로그 pkl(512.9 m / 차로 11)과 다르다 — 그 사이 build_route 가 바뀐
결과이고, 검증 대상은 "지금 코드가 이 경유점을 어떻게 푸는가" 이므로 이것이
맞다 (`vehicle.min_turn_margin` 0.7/1.0 양쪽에서 같은 결과다).
