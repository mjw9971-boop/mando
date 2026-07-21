import rclpy
from rclpy.node import Node

class VehicleControl(Node):
    def __init__(self):
        super().__init__('vehicle_control')
        self.get_logger().info('🚘 Vehicle Control Node Started')

def main(args=None):
    rclpy.init(args=args)
    node = VehicleControl()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
