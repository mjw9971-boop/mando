# core.types.Decision 과 1:1.
std_msgs/Header header

float64 v_target         # [m/s]

# 추종 목표 점열 (월드 좌표). path_x[i], path_y[i] 가 한 점.
float64[] path_x
float64[] path_y

uint8 turn_signal        # 0=OFF 1=LEFT 2=RIGHT
string state             # FSM 상태명
string reasons_json      # 속도 후보값 + shield 발동 내역 (로그/디버깅용)
