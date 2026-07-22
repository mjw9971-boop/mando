import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64

class RadarProcessing(Node):

    def __init__(self):
        super().__init__('radar_processing')
        self.get_logger().info('📡 [Radar Processing] 악천후 보조 레이다 노드 시작')

        self.pub_dist = self.create_publisher(
            Float64, '/perception/radar_dist', 10
        )
        # 10Hz 주기로 실행
        self.timer = self.create_timer(0.1, self.process_radar)

    def process_radar(self):
        # ----------------------------------------------------
        # 📌 비/눈 날씨 영향을 받지 않는 레이다 패킷 파싱 자리
        # ----------------------------------------------------
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