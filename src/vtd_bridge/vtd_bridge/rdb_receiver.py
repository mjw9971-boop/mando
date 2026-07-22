import socket
import struct
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float64MultiArray


class RDBReceiver(Node):

    def __init__(self):
        super().__init__('rdb_receiver')

        # ----------------------------------------------------
        # 1. ROS2 Publisher 설정
        # ----------------------------------------------------
        # 차량 상태 (위치, 속도, 조향각 등)
        self.state_pub = self.create_publisher(
            Float64MultiArray, 'vtd/vehicle_state', 10
        )
        # 레이더 (타겟 물체 거리, 상대속도 등)
        self.radar_pub = self.create_publisher(
            Float64MultiArray, 'vtd/radar_data', 10
        )
        # 카메라 (영상)
        self.camera_pub = self.create_publisher(Image, 'vtd/camera_image', 10)
        # 라이다 (포인트클라우드)
        self.lidar_pub = self.create_publisher(
            PointCloud2, 'vtd/lidar_points', 10
        )

        # ----------------------------------------------------
        # 2. VTD RDB 포트 연결 설정
        # ----------------------------------------------------
        self.vtd_ip = '127.0.0.1'  # VTD IP 주소
        self.vtd_port = 8001  # VTD Main RDB 포트

        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((self.vtd_ip, self.vtd_port))
            self.get_logger().info(
                f'🟢 Connected to VTD RDB Server ({self.vtd_ip}:{self.vtd_port})'
            )
        except Exception as e:
            self.get_logger().error(f'🔴 RDB Connection Failed: {e}')
            self.sock = None

        # ----------------------------------------------------
        # 3. 센서별 타이머 주기 설정 (Hz 분리)
        # ----------------------------------------------------
        # 차량 상태: 100Hz (0.01초)
        self.state_timer = self.create_timer(0.01, self.receive_vehicle_state)

        # 레이더: 20Hz (0.05초)
        self.radar_timer = self.create_timer(0.05, self.receive_radar_data)

        # 카메라: 30Hz (0.033초)
        self.camera_timer = self.create_timer(0.033, self.receive_camera_data)

        # 라이다: 10Hz (0.10초)
        self.lidar_timer = self.create_timer(0.10, self.receive_lidar_data)

    # ----------------------------------------------------
    # 4. 센서별 데이터 수신 콜백 함수
    # ----------------------------------------------------
    def receive_vehicle_state(self):
        """차량 상태 데이터 수신 (100Hz)"""
        if self.sock is None:
            return

        try:
            # 소켓 수신 예시 (실무 개발 시 RDB Header 파싱 진행)
            # raw_data = self.sock.recv(4096)

            msg = Float64MultiArray()
            # [속도(m/s), 위치X(m), 위치Y(m), 헤딩각(rad), 조향각(rad)]
            msg.data = [15.0, 102.5, 45.2, 0.01, 0.05]
            self.state_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f'Vehicle State Receive Error: {e}')

    def receive_radar_data(self):
        """레이더 데이터 수신 (20Hz)"""
        if self.sock is None:
            return

        try:
            msg = Float64MultiArray()
            # [타겟1_거리, 타겟1_상대속도, 타겟2_거리, 타겟2_상대속도 ...]
            msg.data = [25.4, -2.1, 40.8, 1.2]
            self.radar_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f'Radar Receive Error: {e}')

    def receive_camera_data(self):
        """카메라 영상 데이터 수신 (30Hz)"""
        if self.sock is None:
            return

        try:
            msg = Image()
            # RDB 이미지 패킷/RTSP 버퍼를 Image 메시지 포맷으로 담아서 전송
            self.camera_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f'Camera Receive Error: {e}')

    def receive_lidar_data(self):
        """라이다 포인트클라우드 데이터 수신 (10Hz)"""
        if self.sock is None:
            return

        try:
            msg = PointCloud2()
            # RDB 센서 패킷을 PointCloud2 메시지 포맷으로 담아서 전송
            self.lidar_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f'Lidar Receive Error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = RDBReceiver()
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