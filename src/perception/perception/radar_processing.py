import rclpy
from rclpy.node import Node

class RadarProcessing(Node):
    def __init__(self):
        super().__init__('radar_processing')
        self.get_logger().info('📡 Radar Processing Node Started')

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
