import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class RadarProcessing(Node):

    def __init__(self):
        super().__init__('radar_processing')
        self.get_logger().info('📡 Radar Processing Node Started')

        # 1. PathPlanner로 보낼 레이다 기반 장애물 거리 발행
        self.pub_dist = self.create_publisher(
            Float64, '/perception/radar_dist', 10
        )

    def process_radar(self):
        # 레이다 패킷을 수신해서 파싱하는 로직 작성
        radar_distance = 999.0

        dist_msg = Float64()
        dist_msg.data = radar_distance
        self.pub_dist.publish(dist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = RadarProcessing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()