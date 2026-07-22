import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class PathPlanner(Node):

    def __init__(self):
        super().__init__('path_planner')
        self.get_logger().info('🧠 Path Planner Node Started')

        # 1. 기존 구독자 (차량 상태 및 전역 경로)
        self.sub_ego = self.create_subscription(
            Odometry, '/vtd/ego_state', self.ego_callback, 10
        )
        self.sub_global_path = self.create_subscription(
            Path, '/vtd/global_path', self.global_path_callback, 10
        )

        # 2. 📸 🚨 퍼셉션 토픽 구독자 추가!
        self.sub_traffic_light = self.create_subscription(
            String, '/perception/traffic_light', self.traffic_light_callback, 10
        )
        self.sub_lidar_dist = self.create_subscription(
            Float64, '/perception/lidar_dist', self.lidar_callback, 10
        )

        # 3. 발행자 (제어 노드로 보낼 Local Path)
        self.pub_local_path = self.create_publisher(
            Path, '/planning/local_path', 10
        )

        # 변수 초기화
        self.ego_pose = None
        self.global_path = None
        self.local_path_len = 30

        # 퍼셉션 상태 변수
        self.traffic_light_state = "GREEN"  # 기본값: 진행
        self.obstacle_distance = 999.0     # 기본값: 장애물 없음 (m)

        # 20Hz 실행
        self.timer = self.create_timer(0.05, self.plan_path)

    def ego_callback(self, msg):
        self.ego_pose = msg.pose.pose

    def global_path_callback(self, msg):
        self.global_path = msg.poses

    # 📸 퍼셉션 콜백 함수들
    def traffic_light_callback(self, msg):
        self.traffic_light_state = msg.data  # 예: "RED", "GREEN"

    def lidar_callback(self, msg):
        self.obstacle_distance = msg.data    # 전방 장애물 거리 (m)

    def plan_path(self):
        if self.ego_pose is None or self.global_path is None or len(self.global_path) == 0:
            return

        local_path_msg = Path()
        local_path_msg.header.frame_id = 'map'
        local_path_msg.header.stamp = self.get_clock().now().to_msg()

        # ----------------------------------------------------
        # 🧠 [판단 로직] 정지 조건 체크 (신호등 RED or 장애물 5m 이내)
        # ----------------------------------------------------
        if self.traffic_light_state == "RED" or self.obstacle_distance < 5.0:
            self.get_logger().warn('🛑 [STOP] 장애물 또는 신호등 감지! 차량을 정지합니다.')
            # 경로 비워 보내기 -> vehicle_control이 경로를 못 찾아 정지함 (또는 속도 0 명령)
            local_path_msg.poses = []
            self.pub_local_path.publish(local_path_msg)
            return

        # ----------------------------------------------------
        # 🚗 [정상 주행] 내 위치 기반 Local Path 생성
        # ----------------------------------------------------
        ego_x = self.ego_pose.position.x
        ego_y = self.ego_pose.position.y

        min_dist = float('inf')
        closest_idx = 0

        for i, pose_stamped in enumerate(self.global_path):
            wx = pose_stamped.pose.position.x
            wy = pose_stamped.pose.position.y
            dist = math.hypot(wx - ego_x, wy - ego_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        end_idx = min(closest_idx + self.local_path_len, len(self.global_path))
        local_path_msg.poses = self.global_path[closest_idx:end_idx]

        self.pub_local_path.publish(local_path_msg)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()