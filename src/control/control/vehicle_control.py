import math
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist

class VehicleControl(Node):

    def __init__(self):
        super().__init__('vehicle_control')
        self.get_logger().info('🚘 [Vehicle Control] Pure Pursuit & 속도 제어 노드 시작')

        # 1. 구독자(Subscriber) 설정
        self.sub_ego = self.create_subscription(
            Odometry, '/vtd/ego_state', self.ego_callback, 10
        )
        self.sub_local_path = self.create_subscription(
            Path, '/planning/local_path', self.path_callback, 10
        )

        # 2. 발행자(Publisher) 설정 (VTD/차량 제어 토픽)
        # linear.x: 목표 속도(m/s) 또는 가속도, angular.z: 핸들 조향각(rad)
        self.pub_cmd = self.create_publisher(
            Twist, '/control/cmd_drive', 10
        )

        # 차량 내부 변수
        self.ego_x = 0.0
        self.ego_y = 0.0
        self.ego_yaw = 0.0
        self.current_speed = 0.0
        self.local_path = None

        # 📌 차량 제어 파라미터 (차종 및 환경에 맞춰 조정 가능)
        self.WHEELBASE = 2.7       # 휠베이스 (축거, 단위: m)
        self.MIN_LOOKAHEAD = 4.0   # 최소 표적 거리 (단위: m)
        self.MAX_LOOKAHEAD = 10.0  # 최대 표적 거리 (단위: m)
        self.TARGET_SPEED = 8.33   # 목표 크루즈 속도 (약 30 km/h = 8.33 m/s)

        # 20Hz (0.05초) 제어 주기
        self.timer = self.create_timer(0.05, self.control_loop)

    def ego_callback(self, msg):
        self.ego_x = msg.pose.pose.position.x
        self.ego_y = msg.pose.pose.position.y

        # 쿼터니언(Quaternion) ➔ 쿼터니언을 Yaw(오일러각)로 변환
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.ego_yaw = math.atan2(siny_cosp, cosy_cosp)

        # 현재 속도 계산 (m/s)
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        self.current_speed = math.hypot(vx, vy)

    def path_callback(self, msg):
        self.local_path = msg.poses

    def control_loop(self):
        cmd_msg = Twist()

        # ----------------------------------------------------
        # 🛑 1. STOP 상태 처리 (경로가 비어있거나 수신 전)
        # ----------------------------------------------------
        if self.local_path is None or len(self.local_path) == 0:
            cmd_msg.linear.x = 0.0   # 속도 0
            cmd_msg.angular.z = 0.0  # 핸들 정중앙
            self.pub_cmd.publish(cmd_msg)
            return

        # ----------------------------------------------------
        # 🎯 2. 가변 Pure Pursuit (표적 거리 Ld 계산)
        # 속도가 빠를수록 멀리 보고, 느릴수록 가까이 봄
        # ----------------------------------------------------
        look_ahead_dist = self.MIN_LOOKAHEAD + (0.3 * self.current_speed)
        look_ahead_dist = min(self.MAX_LOOKAHEAD, max(self.MIN_LOOKAHEAD, look_ahead_dist))

        # 경로 상에서 Look-ahead 거리 근처의 Target Point 찾기
        target_point = None
        for pose in self.local_path:
            px = pose.pose.position.x
            py = pose.pose.position.y
            dist = math.hypot(px - self.ego_x, py - self.ego_y)

            if dist >= look_ahead_dist:
                target_point = (px, py)
                break

        # 만약 Look-ahead 거리에 도달하는 점이 없으면 경로의 맨 마지막 점 선택
        if target_point is None:
            last_pose = self.local_path[-1].pose.position
            target_point = (last_pose.x, last_pose.y)

        # ----------------------------------------------------
        # 📐 3. 좌표 변환 (Global ➔ Vehicle Local Coordinates)
        # ----------------------------------------------------
        dx = target_point[0] - self.ego_x
        dy = target_point[1] - self.ego_y

        # 차량의 현재 Yaw 기준 Local 좌표 계산
        local_x = dx * math.cos(-self.ego_yaw) - dy * math.sin(-self.ego_yaw)
        local_y = dx * math.sin(-self.ego_yaw) + dy * math.cos(-self.ego_yaw)

        # ----------------------------------------------------
        # 🎡 4. Pure Pursuit 조향각(Steering Angle) 산출
        # 공식: delta = arctan(2 * L * y_local / Ld^2)
        # ----------------------------------------------------
        steering_angle = math.atan2(2.0 * self.WHEELBASE * local_y, (look_ahead_dist ** 2))

        # 빗길/악천후 시 급조향 방지를 위한 조향각 제한 (약 ±30도)
        max_steer_rad = math.radians(30.0)
        steering_angle = max(-max_steer_rad, min(max_steer_rad, steering_angle))

        # ----------------------------------------------------
        # 🚀 5. 종방향 속도 제어
        # ----------------------------------------------------
        # 경로의 길이가 짧으면(SLOWDOWN 상태) 목표 속도를 낮춤
        if len(self.local_path) < 20:
            target_speed = 4.16  # 약 15 km/h
        else:
            target_speed = self.TARGET_SPEED  # 약 30 km/h

        # 제어 명령 발행
        cmd_msg.linear.x = target_speed
        cmd_msg.angular.z = steering_angle
        self.pub_cmd.publish(cmd_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VehicleControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()