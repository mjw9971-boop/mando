import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Float64
import math

class LidarProcessing(Node):

    def __init__(self):
        super().__init__('lidar_processing')
        self.get_logger().info('🚨 [LiDAR Processing] 지면/빗물 필터링 노드 시작')

        self.sub_lidar = self.create_subscription(
            PointCloud2, '/vtd/lidar/points', self.lidar_callback, 10
        )
        self.pub_dist = self.create_publisher(
            Float64, '/perception/lidar_dist', 10
        )

    def lidar_callback(self, msg):
        min_dist = 999.0

        # PointCloud2 메시지에서 x, y, z 좌표 추출
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = point[0], point[1], point[2]

            # ----------------------------------------------------
            # 📌 1. Height Crop (지면 및 빗물 웅덩이 제거)
            # 차체 바닥 기준 z < -0.3m 이하의 지면/빗물 반사 점군은 완전히 무시
            # ----------------------------------------------------
            if z < -0.3 or z > 1.5:
                continue

            # ----------------------------------------------------
            # 📌 2. Ego-lane ROI (내 차선 내부만 오려서 점검)
            # 전방 0.5m ~ 20.0m, 좌우 ±1.2m 폭 범위 안의 장애물만 측정
            # ----------------------------------------------------
            if 0.5 <= x <= 20.0 and -1.2 <= y <= 1.2:
                dist = math.hypot(x, y)
                if dist < min_dist:
                    min_dist = dist

        # 장애물 거리 발행 (장애물 없으면 999.0)
        dist_msg = Float64()
        dist_msg.data = min_dist
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