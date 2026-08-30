# tests/fixtures — 테스트 전용 고정 입력

여기 있는 파일은 **테스트의 기대값이 매달린 기준(baseline)** 이다. 값을 바꾸면
그 값을 기준으로 쓴 테스트가 함께 깨진다. 사람이 의도적으로 갱신할 때 말고는
어떤 도구도 여기에 쓰지 않는다.

| 파일 | 무엇 | 쓰는 곳 |
|---|---|---|
| `route.pkl` | 821.1 m · 차로 26 · 경유점 8 짜리 고정 경로 | `test_overtake` `test_turn_signal` `test_solid_lc` `test_lc_corridor` `test_static_obstacle` |
| `waypoints.csv` | 위 `route.pkl` 을 만든 경유점 CSV (seq,x,y) | 재생성 근거 — 현재 테스트가 직접 읽지는 않는다 |

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
