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
