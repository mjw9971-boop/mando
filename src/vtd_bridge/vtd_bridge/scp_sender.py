import rclpy
from rclpy.node import Node

class SCPSender(Node):
    def __init__(self):
        super().__init__('scp_sender')
        self.get_logger().info('🟢 SCP Sender Initialized')

def main(args=None):
    rclpy.init(args=args)
    node = SCPSender()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
