# vtd_bridge 가 1109 B 프레임 하나를 푼 것. 한 패킷 = 한 틱.
# 해석/판단 없음 (core.types.RawPacket 과 1:1).
std_msgs/Header header

float64 t_recv          # time.monotonic() — core 로직이 dt 계산에 쓴다

float32 ego_x
float32 ego_y
float32 ego_z
float32 ego_heading     # [rad]
float32 ego_pitch
float32 ego_roll

Object[] objects        # id != 0 인 것만
TrafficLight[] lights   # id != 0 인 것만

# 접속 후 소켓에 도착한 9910 프레임 누적 수 (버린 백로그 포함).
# 프레임 = sim 50 ms. 차분 x 50 ms = sim_dt → RTF 판별용 (프레임에 sim 시각 필드가 없다).
uint32 frames_total
