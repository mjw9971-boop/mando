import rclpy
from rclpy.node import Node

class PathPlanner(Node):
    def __init__(self):
        super().__init__('path_planner')
        self.get_logger().info('🧠 Path Planner Node Started')

def main(args=None):
    rclpy.init(args=args)
    node = PathPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
