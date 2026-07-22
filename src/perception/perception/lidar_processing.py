import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float64

class LidarProcessing(Node):

    def __init__(self):
        super().__init__('lidar_processing')
        self.get_logger().info('🚨 Lidar Processing Node Started')

        # 1. VTD 라이다 토픽 구독
        self.sub_lidar = self.create_subscription(
            PointCloud2, '/vtd/lidar/points', self.lidar_callback, 10
        )

        # 2. PathPlanner로 보낼 전방 장애물 거리 발행
        self.pub_dist = self.create_publisher(
            Float64, '/perception/lidar_dist', 10
        )

    def lidar_callback(self, msg):
        # ----------------------------------------------------
        # 📌 [Point Cloud 필터링 & 전방 거리 계산 자리]
        # ----------------------------------------------------
        # ROI 조건 예시: x > 0 (전방), -1.0 < y < 1.0 (내 차선 폭), -0.5 < z < 1.0 (지면 제외)
        
        min_front_distance = 999.0  # 기본값: 장애물 없음 (m)

        # 3. 거리 발행
        dist_msg = Float64()
        dist_msg.data = min_front_distance
        self.pub_dist.publish(dist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()