# core.types.Command 와 1:1. vtd_bridge 가 그대로 9 B 로 packing 한다.
# steer_sign 은 송신 직전에 곱하므로 여기 값은 내부 부호(좌 +).
std_msgs/Header header

float64 steering         # [rad]
float64 accel            # [m/s^2] 음수 = 감속
uint8 turn_signal
