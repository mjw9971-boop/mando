# lane_graph — HL_FMA_VTD_LivingLab.xodr → 인지용 지도

xodr(651 도로, 33.2 km)을 **주행 차로 2,480개 × 0.5 m 점(188,571개)** 으로 미리 풀어둔 것.
실시간에는 KD-tree 로 "내 위치가 어느 차로 어디냐"만 물으면 됨. 센서 안 씀.

```
build_lane_graph.py   xodr → lane_graph.pkl        (대회 전 1번, 2초)
lanegraph.py          런타임 헬퍼 (인지 노드에서 import)
build_route.py        경유점 csv → route.pkl        (대회날 경로 받으면 실행)
plot_lane_graph.py    그림으로 확인
lane_graph.pkl        지금 맵으로 이미 만들어둔 결과 (11 MB)
map_full.png          전체 맵 렌더 (초록 50 / 파랑 30 / 주황 스쿨존30 / 빨강 정지선)
```

## 실행

```bash
pip install numpy scipy matplotlib
python3 build_lane_graph.py HL_FMA_VTD_LivingLab.xodr -o lane_graph.pkl
python3 plot_lane_graph.py lane_graph.pkl -o map.png
python3 plot_lane_graph.py lane_graph.pkl -o zoom.png --center 15 42 --radius 120

# 대회날: 경유점 csv (x,y 한 줄씩) → route.pkl → 그림으로 눈 확인 후 출발
python3 build_route.py lane_graph.pkl waypoints.csv -o route.pkl --start-yaw 0.53
python3 plot_lane_graph.py lane_graph.pkl -o route.png --route route.pkl
```

## 인지 노드에서 쓰는 법

```python
from lanegraph import LaneGraph
lg = LaneGraph('lane_graph.pkl')
route = pickle.load(open('route.pkl','rb'))

# 매 틱
m = lg.locate(ego_x, ego_y, ego_yaw, prefer=route['lanes'])   # ego → (lane, s, t, heading_err)
idx = route['lanes'].index(m.lane)                              # 경로상 몇 번째 차로
ahead = lg.lookahead(route, idx, m.s, horizon=200)              # 전방 200 m 이벤트 (거리순)
summ  = lg.summarize(ahead)      # dist_stop_line, stop_signal_ids, dist_crosswalk, dist_next_turn, speed_changes ...

# 객체도 같은 함수: 헤딩 주면 반대 차로 배제, prefer 로 경로 차로 우선
om = lg.locate(obj_x, obj_y, obj_heading)
# om.lane == m.lane 이고 om.s > m.s → 선행 객체, 종거리 = (om.s - m.s) (같은 차로) / 경로 누적거리로 환산하면 다른 차로도 비교 가능
```

lookahead 항목(kind): `stop_line`(signal_ids 포함) / `crosswalk`(kind=pelican|inferred) / `crosswalk_warn` / `yield` /
`speed`(limit, school_zone) / `mark_left`·`mark_right`(type, lane_change_ok, is_center) / `junction_in`·`junction_out` /
`turn_left`·`turn_right` / `lane_change_left`·`lane_change_right`(window_s0~s1 = 점선 구간) / `dead_end` / `route_end`

## lane_graph.pkl 구조

```
g['lanes'][(road_id, section_idx, lane_id)] = {
  dir            +1 (lane_id<0, 도로 s 방향) / -1 (lane_id>0, 역방향)   ※ pts/s/hdg 는 이미 주행 방향으로 정렬됨
  pts (N,3)      x,y,z      s (N,)  차로 시작부터 누적거리      road_s (N,) 도로 s
  hdg, curv, width (N,)     length
  left_mark / right_mark    [(s0, s1, type, color, lane_change_ok)]   운전자 기준 좌/우 차선
  left_nb / right_nb        같은 방향 옆 차로 key (없으면 None)
  left_is_center            True 면 왼쪽이 중앙선/분리대 (넘으면 중앙선 침범)
  next / prev               주행 방향 다음/이전 차로들 (교차로 연결 포함)
  stop_lines                [{s, signal_ids}]
  signals                   [{id, stop_s, explicit, type, subtype}]
  crosswalks                [(s0, s1, 'pelican'|'inferred')]
  crosswalk_warn / yield_marks / arrows(s,'S'|'L'|'R'|'SL'|'SR'|'LU') / speed_marks(s,값,이름) / markings
  speed_limit (30/50/None)  school_zone (bool)  speed_src
  junction                  -1 이면 일반 도로, 아니면 교차로 연결 도로
}
g['roads'][road_id]  길이, 링크, 섹션, speed_by_dir, stop_clusters
g['signals'][signal_id] = {road, lanes, type, subtype}
g['kd_pts'/'kd_lane'/'kd_i'/'kd_hdg']  KD-tree 용 (lanegraph.py 가 씀)
g['meta']  통계 + assumptions + warnings
```

## 이 맵에서 확인된 인코딩 (코드가 이렇게 가정함)

- 기하: line / arc / spiral / poly3. laneSection 최대 12개, laneOffset 있음, 표준 `<speed>` 필드 0개.
- 신호등 646개 전부 dynamic. 어느 차로 신호인지: `validity` 태그(148개) 우선,
  없으면 `hOffset` (0 → 도로 s 방향 차로(lane_id<0), π → 역방향(lane_id>0)) — validity 있는 148개로 검증했을 때 100% 일치.
  hOffset 이 ±π/2 인 6개(556, 791 도로 = 교차로 내부/pelican)는 t 부호로 배정하고 warnings 에 기록.
  한 차로 그룹에 신호 3개(type 1000020/1000008/1000012)가 세트로 걸림 → 판단에서 id 별 state 를 합쳐서 해석 필요.
- 정지선: `Rm_StopLine_300cm_JPN_01` 객체 710개, 차로당 1개, hdg 0/π 로 방향. 3 m 이내 클러스터 → 방향별 정지선 277개.
- 제한속도: 노면표시 객체로만 존재. `RM_517_50`(50, 16도로 2.6 km) / `roadmark_speed_30`(30, 35도로 3.1 km) /
  `RM_518`(23도로 2.7 km — 한국 노면표시 518 = 어린이보호구역 속도제한으로 가정, **30 으로 처리**).
  도로+방향 단위로 적용, 없는 도로는 None → `lookahead` 가 이전 값을 그대로 유지(carry).
- 횡단보도: 실제 객체는 pelican 2개뿐. 교차로 횡단보도는 xodr 에 **없음**(osgb 시각용).
  `Rm_Warning_Crosswalk_JPN_03` 400개는 횡단보도 예고 다이아몬드(정지선 앞 중앙값 52 m).
  → 정지선 뒤 8 m 를 `inferred` 횡단보도 존으로 넣어둠. "정지선 앞에서 서라"가 횡단보도 정차 금지도 커버.
- 화살표 노면표시(RM_537/538/539)는 arrows 에, 양보(Rm_Give_Way)는 yield_marks 에.
- 막다른 차로 72개 (경계 24, 포켓 끝 33, 교차로 연결 없음 15). 경로 탐색은 자동으로 피함. 런타임에 `dead_end` 로 뜸.

## 대회 전에 반드시 확인할 것 (코드 가정 → 실측)

1. **9910 ego X,Y 가 xodr 좌표계와 같은지** — 시나리오 초기위치 (508.8, -168.3, yaw 0.527) 를 `lg.locate` 하면 (465,2,-1) 에 5 cm 로 붙음. 실제 패킷으로 다시 확인.
2. **9910 trafficLights[].id 가 xodr signal id 인지** (1~967, 646개). 정지선 앞에서 id 별 state 로그 찍어서 stop_lines[].signal_ids 와 대조.
3. **RM_518 텍스처가 정말 어린이보호구역 30 인지**, 표시 없는 도로의 기본 제한속도(50? 60?)는 redmine 에 질문.
4. 경유점 형식(좌표 몇 개, 반경 판정 기준) → `build_route.py --radius` 조정. 좌표에 z 가 오면 무시하면 됨.
5. 교차로 횡단보도 위치를 주최 평가 로직이 어떻게 정의하는지 — 알게 되면 `inferred` 존 8 m 를 그 값으로.
6. `meta['warnings']` (6개) 확인.

## 안 한 것 (필요해지면)
- 보도/연석 폴리곤 (보도 침범 판정용) — 차로 유지만 되면 자동 만족이라 뺌. 필요하면 sidewalk lane 도 같이 뽑으면 됨.
- 신호기 ↔ 컨트롤러 위상(녹 10~15/황 3/적 13~18) 매핑 — 시나리오 xml 에 Signal↔Controller 배정이 없어서 못 함. 9910 상태 전환 시점 트래킹으로 대체.
