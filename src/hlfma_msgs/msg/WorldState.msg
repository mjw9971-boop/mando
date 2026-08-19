# core.types.WorldState 와 1:1.
std_msgs/Header header

float64 t

# ── ego ───────────────────────────────────────────────────────────────
float64 ego_x
float64 ego_y
float64 ego_z
float64 ego_yaw
float64 ego_pitch
float64 ego_roll
float64 ego_speed        # 위치 미분 + LPF 추정치 (패킷에 속도 필드가 없다)
float64 ego_accel

bool has_lane
int32[3] lane
float64 s
float64 route_s
float64 t_off
float64 heading_err

# ── 주변 ──────────────────────────────────────────────────────────────
TrackedObject[] objects

bool has_light
TrafficLight light

# ── 차로 속성 ─────────────────────────────────────────────────────────
float64 speed_limit      # [m/s] carry 반영
bool school_zone
bool left_solid
bool right_solid
bool left_is_center

bool valid

# ── 전방 프로파일 ─────────────────────────────────────────────────────
# lanegraph.Ahead 의 data 는 항목마다 스키마가 달라(신호 id 배열, 제한속도,
# lane_change_ok ...) 메시지 필드로 펴기 어렵다. 무손실 전달을 위해 JSON 문자열로
# 싣는다. convert.py 가 왕복 변환을 책임진다.
string ahead_json        # [{kind, dist, lane, s_in_lane, data}, ...]
string summ_json         # lanegraph.summarize 결과
string flags_json
