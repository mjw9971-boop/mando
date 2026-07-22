import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry, Path

class VehicleControl(Node):

    def __init__(self):
        super().__init__('vehicle_control')
        self.get_logger().info('🚗 Vehicle Control Node Started (Pure Pursuit)')

        # 1. 수신(Subscriber): 차량 상태 및 PathPlanner의 국소 경로(Local Path) 받아오기
        self.sub_ego = self.create_subscription(
            Odometry, '/vtd/ego_state', self.ego_callback, 10
        )
        self.sub_path = self.create_subscription(
            Path, '/planning/local_path', self.path_callback, 10
        )

        # 2. 송신(Publisher): scp_sender로 보낼 제어 명령 토픽
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, '/control/cmd_drive', 10
        )

        # 변수 초기화
        self.current_pose = None
        self.local_path = None

        # ----------------------------------------------------
        # 📌 Pure Pursuit 파라미터 설정
        # ----------------------------------------------------
        self.wheelbase = 2.7        # 차량 축거 L (m)
        self.look_ahead_dist = 5.0  # 전방 표적 거리 L_d (m)

        # 3. 주기적 제어 루프 (20Hz = 0.05초)
        self.timer = self.create_timer(0.05, self.control_loop)

    def ego_callback(self, msg):
        self.current_pose = msg.pose.pose

    def path_callback(self, msg):
        self.local_path = msg.poses

    def get_yaw_from_pose(self, pose):
        """쿼터니언(Quaternion) 방위를 Yaw 각도(radians)로 변환"""
        q = pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        # 데이터가 수신되지 않았으면 대기
        if self.current_pose is None or self.local_path is None or len(self.local_path) == 0:
            return

        # 1. 현재 차량 위치 (x, y) 및 헤딩 각도(yaw) 계산
        ego_x = self.current_pose.position.x
        ego_y = self.current_pose.position.y
        yaw = self.get_yaw_from_pose(self.current_pose)

        # 2. Look-ahead distance (L_d) 떨어진 표적점(Target Point) 찾기
        target_pt = None
        for pose_stamped in self.local_path:
            wx = pose_stamped.pose.position.x
            wy = pose_stamped.pose.position.y
            dist = math.hypot(wx - ego_x, wy - ego_y)

            if dist >= self.look_ahead_dist:
                target_pt = (wx, wy)
                break

        # 만약 L_d 이상 떨어진 점이 없다면 가장 멀리 있는 점 지정
        if target_pt is None:
            target_pt = (
                self.local_path[-1].pose.position.x,
                self.local_path[-1].pose.position.y
            )

        # 3. 월드 좌표계 표적점을 차량 중심 상대 좌표계(Vehicle Frame)로 변환
        dx = target_pt[0] - ego_x
        dy = target_pt[1] - ego_y

        # 차량 좌우 횡방향 오차 (y_local)
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy

        # 4. Pure Pursuit 조향각(Steering Angle) 계산
        # 공식: delta = atan2(2 * L * y_local, L_d^2)
        l_d = self.look_ahead_dist
        steering = math.atan2(2.0 * self.wheelbase * local_y, l_d ** 2)

        # 5. 가속도 설정 (기본 주행 가속도 1.0 m/s^2)
        accel = 1.0

        # 6. 제어 명령 발행 (/control/cmd_drive -> scp_sender)
        cmd_msg = Float64MultiArray()
        cmd_msg.data = [accel, steering]
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