import socket
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class SCPControlNode(Node):

    def __init__(self):
        super().__init__('scp_control_node')
        self.get_logger().info('🚗 SCP Control Node Started')

        # VTD SCP 연결 정보
        self.vtd_ip = '127.0.0.1'
        self.vtd_port = 8010
        self.sock = None

        # 최초 연결 시도
        self.connect_to_vtd()

        # 제어 토픽 구독
        self.sub_control = self.create_subscription(
            Float64MultiArray,
            'control/cmd_drive',
            self.control_cmd_callback,
            10,
        )

    def connect_to_vtd(self):
        """VTD SCP 서버 연결 시도"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.vtd_ip, self.vtd_port))
            self.get_logger().info(
                f'🟢 Connected to VTD SCP Server ({self.vtd_ip}:{self.vtd_port})'
            )
        except Exception as e:
            self.get_logger().error(f'🔴 SCP Connection Failed: {e}')
            self.sock = None

    def send_scp_command(self, xml_string: str):
        """SCP 규격 전송"""
        if self.sock is None:
            self.get_logger().warn('Socket not connected. Trying to reconnect...')
            self.connect_to_vtd()
            if self.sock is None:
                return

        try:
            # Null Character('\0') 포함 전송
            full_msg = xml_string + '\0'
            self.sock.sendall(full_msg.encode('utf-8'))
            self.get_logger().debug(f'Sent SCP: {xml_string}')
        except Exception as e:
            self.get_logger().error(f'Failed to send SCP command: {e}')
            self.sock = None  # 에러 발생 시 재연결을 위해 소켓 초기화

    def control_cmd_callback(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            self.get_logger().warn('Invalid msg format. Needs [accel, steering].')
            return

        # 1. 안전 한계값(Clipping) 적용
        accel = max(-8.0, min(5.0, float(msg.data[0])))       # 가속도 제한 (m/s^2)
        steering = max(-0.6, min(0.6, float(msg.data[1])))   # 조향각 제한 (rad)

        # 2. VTD SCP XML 생성 (차량이름: Ego)
        scp_xml = (
            f'<SCP>'
            f'<Player name="Ego">'
            f'<Control accel="{accel:.2f}" steering="{steering:.3f}"/>'
            f'</Player>'
            f'</SCP>'
        )

        self.send_scp_command(scp_xml)


def main(args=None):
    rclpy.init(args=args)
    node = SCPControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if hasattr(node, 'sock') and node.sock:
            node.sock.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()