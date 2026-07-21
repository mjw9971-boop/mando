import rclpy
from rclpy.node import Node

class LidarProcessing(Node):
    def __init__(self):
        super().__init__('lidar_processing')
        self.get_logger().info('🚨 Lidar Processing Node Started')

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
