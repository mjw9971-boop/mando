"""
vtd_adapter — VTD 9910 연결·지도·경로·로깅 계층.

PDM-Lite(team_code/)가 CARLA 에서 받던 것들을 VTD 로부터 만들어 주는 층이다.
판단 로직은 여기 없다 — 그건 전부 team_code/autopilot.py 의 몫이다.

  comm.py        9910 TCP 송수신 (1109 B 프레임 / 9 B 명령)
  lanegraph.py   lane_graph.pkl 런타임 조회 (xodr 기하·정지선·신호 매핑) — VTD 프레임
  types.py       파이프라인 dataclass (RawPacket → WorldState → Command)
  frame.py       좌표계 변환의 **유일한** 자리 (VTD ↔ CARLA 관례)
  carla_types.py CARLA 기하·제어 타입 흉내 (Location/Rotation/BoundingBox/…)
  map.py         VtdMap / VtdWaypoint  (carla.Map/Waypoint 표면)
  actor.py       VtdActor / VtdEgo     (carla.Actor 표면, CARLA 프레임)
  world.py       VtdWorld              (carla.World 표면 — 객체 유지·코스팅)
  route.py       VtdRoutePlanner       (PDM privileged_route_planner 표면)
                 + 9910 light_id ↔ 정지선 controller 대조
  control.py     VtdLongitudinalController (accel 직접 출력) + Command 변환
  ego.py         자차 속도 추정(9910 에 속도 필드 없음) + courseRespawn 리셋 감지
  config.py      config/params.yaml 로더 (VTD·어댑터 상수의 단일 출처)
  logger.py      run_*.jsonl 틱 로그 (채점 파이프라인이 읽는 스키마 — 변경 금지)
  scoring.py     제한속도 감점 판정 (tools/score.py 가 사용)
"""
