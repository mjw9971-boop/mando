# core.types.TrackedObject 와 1:1.
uint32 id
float32 x
float32 y
float32 heading
float32 speed
float32 length
float32 width
float32 height

string cls              # 'vehicle' | 'pedestrian' | 'obstacle' | 'unknown'

bool has_lane
int32[3] lane           # (road_id, section_idx, lane_id), has_lane 이 false 면 무의미

bool on_route
float32 s_rel           # 경로 기준 종방향 상대거리 (+ = 앞)
float32 lat_off
float32 v_rel           # 접근 속도 (+ = 접근)
float32 ttc             # inf 는 매우 큰 값으로 전달된다
bool will_enter_lane
float32 age
bool coasting
