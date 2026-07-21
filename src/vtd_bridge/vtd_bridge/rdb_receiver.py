import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class RDBReceiver(Node):
    def __init__(self):
        super().__init__('rdb_receiver')
        self.publisher_ = self.create_publisher(Float64MultiArray, 'vtd/vehicle_state', 10)
        self.get_logger().info('🟢 RDB Receiver Initialized')

def main(args=None):
    rclpy.init(args=args)
    node = RDBReceiver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
