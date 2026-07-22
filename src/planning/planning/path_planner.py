import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, String
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class PathPlanner(Node):

    def __init__(self):
        super().__init__('path_planner')
        self.get_logger().info('🧠 [Path Planner] FSM 판단 및 경로 생성 노드 시작')

        # 1. 구독자(Subscriber) 설정
        self.sub_ego = self.create_subscription(
            Odometry, '/vtd/ego_state', self.ego_callback, 10
        )
        self.sub_global_path = self.create_subscription(
            Path, '/vtd/global_path', self.global_path_callback, 10
        )

        # 퍼셉션 데이터 구독자
        self.sub_traffic_light = self.create_subscription(
            String, '/perception/traffic_light', self.traffic_light_callback, 10
        )
        self.sub_lidar_dist = self.create_subscription(
            Float64, '/perception/lidar_dist', self.lidar_callback, 10
        )
        self.sub_radar_dist = self.create_subscription(
            Float64, '/perception/radar_dist', self.radar_callback, 10
        )

        # 2. 발행자(Publisher) 설정
        self.pub_local_path = self.create_publisher(
            Path, '/planning/local_path', 10
        )

        # 내부 변수 초기화
        self.ego_pose = None
        self.global_path = None

        # 퍼셉션 수신 변수 (기본값)
        self.traffic_light_state = "GREEN"
        self.lidar_dist = 999.0
        self.radar_dist = 999.0

        # FSM 상태 정보
        self.state = "CRUISE"

        # 20Hz 주기 실행 (0.05초마다 판단)
        self.timer = self.create_timer(0.05, self.plan_path)

    def ego_callback(self, msg):
        self.ego_pose = msg.pose.pose

    def global_path_callback(self, msg):
        self.global_path = msg.poses

    def traffic_light_callback(self, msg):
        self.traffic_light_state = msg.data

    def lidar_callback(self, msg):
        self.lidar_dist = msg.data

    def radar_callback(self, msg):
        self.radar_dist = msg.data

    def plan_path(self):
        # 필수 데이터 수신 전이면 대기
        if self.ego_pose is None or self.global_path is None or len(self.global_path) == 0:
            return

        # ----------------------------------------------------
        # 🧠 1. 센서 크로스체크 & FSM 상태 판단
        # ----------------------------------------------------
        # 라이다와 레이다 중 더 안전한(가까운) 장애물 거리 선택
        min_obstacle_dist = min(self.lidar_dist, self.radar_dist)

        # FSM 상태 결정
        if self.traffic_light_state == "RED" or min_obstacle_dist < 5.0:
            new_state = "STOP"
        elif 5.0 <= min_obstacle_dist < 10.0:
            new_state = "SLOWDOWN"
        else:
            new_state = "CRUISE"

        # 상태 변경 시 로그 출력
        if new_state != self.state:
            self.get_logger().info(f'🔄 [State Change] {self.state} ➔ {new_state}')
            self.state = new_state

        # ----------------------------------------------------
        # 🚗 2. FSM 상태별 Local Path 생성 및 행동 제어
        # ----------------------------------------------------
        local_path_msg = Path()
        local_path_msg.header.frame_id = 'map'
        local_path_msg.header.stamp = self.get_clock().now().to_msg()

        # A. [STOP 상태] 경로를 빈 상태로 발행하여 정지 명령
        if self.state == "STOP":
            local_path_msg.poses = []
            self.pub_local_path.publish(local_path_msg)
            return

        # B. [CRUISE / SLOWDOWN 상태] 경로 추출
        ego_x = self.ego_pose.position.x
        ego_y = self.ego_pose.position.y

        # 내 위치에서 가장 가까운 전역 경로 점(WayPoint) 찾아내기
        min_dist = float('inf')
        closest_idx = 0

        for i, pose_stamped in enumerate(self.global_path):
            wx = pose_stamped.pose.position.x
            wy = pose_stamped.pose.position.y
            dist = math.hypot(wx - ego_x, wy - ego_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        # 상태별 가시 거리(Waypoint 개수) 조절
        # SLOWDOWN일 때는 앞쪽을 짧게 잘라내어 제어기가 자연스럽게 감속하도록 유도
        path_length = 12 if self.state == "SLOWDOWN" else 30

        end_idx = min(closest_idx + path_length, len(self.global_path))
        local_path_msg.poses = self.global_path[closest_idx:end_idx]

        # 3. 국소 경로 발행
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