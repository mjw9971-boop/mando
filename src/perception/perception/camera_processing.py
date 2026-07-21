import rclpy
from rclpy.node import Node

class CameraProcessing(Node):
    def __init__(self):
        super().__init__('camera_processing')
        self.get_logger().info('📷 Camera Processing Node Started')

def main(args=None):
    rclpy.init(args=args)
    node = CameraProcessing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
