import socket
import struct
import threading
import time
import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header, Float64MultiArray, Float64
import sensor_msgs_py.point_cloud2 as pc2

# ------------------------------------------------------------------------------
# VTD RDB (Runtime Data Bus) 프로토콜 상수 정의
# ------------------------------------------------------------------------------
RDB_MAGIC_NO = 0x1C2A       # VTD RDB 고유 매직 넘버
RDB_MSG_HDR_SIZE = 24       # RDB 메인 헤더 크기 (24 Bytes)
RDB_ENTRY_HDR_SIZE = 16     # RDB 패키지 엔트리 헤더 크기 (16 Bytes)

# VTD 패키지 ID
# ⚠️ 아래 ID 및 parse_* 함수들의 byte offset은 실제 VTD SDK의 viRDBIcd.h와
#    반드시 대조 확인이 필요합니다 (버전에 따라 구조체 크기가 다를 수 있음).
RDB_PKG_ID_ROAD_POS = 5      # 차선/도로 위치 데이터 (Ground Truth)
RDB_PKG_ID_OBJECT_STATE = 9  # 차량/장애물 상태 데이터
RDB_PKG_ID_IMAGE = 14        # 카메라 데이터
RDB_PKG_ID_SENSOR_STATE = 20 # 센서 상태/레이더/라이다 데이터


class RDBReceiver(Node):

    def __init__(self):
        super().__init__('rdb_receiver')

        # ----------------------------------------------------
        # 1. VTD 연결 설정
        # ----------------------------------------------------
        self.declare_parameter('vtd_ip', '127.0.0.1')
        self.declare_parameter('vtd_port', 8001)

        self.vtd_ip = self.get_parameter('vtd_ip').get_parameter_value().string_value
        self.vtd_port = self.get_parameter('vtd_port').get_parameter_value().integer_value

        # ----------------------------------------------------
        # 2. ROS 2 Publisher 설정 (퍼셉션 노드와 토픽명 연동 완료)
        # ----------------------------------------------------
        self.state_pub = self.create_publisher(Float64MultiArray, '/vtd/vehicle_state', 10)
        self.camera_pub = self.create_publisher(Image, '/vtd/camera_image', 10)
        self.lane_pub = self.create_publisher(Float64MultiArray, '/vtd/lane_info', 10)

        # ✅ [수정] sensor_fusion_node.py가 구독하는 '/vtd/ego_speed' 토픽이
        #    기존 코드에는 발행하는 곳이 없어 자차 속도가 항상 0으로 고정되던 문제 수정
        self.speed_pub = self.create_publisher(Float64, '/vtd/ego_speed', 10)

        # 라이다 & 레이다 데이터를 PointCloud2 타입으로 발행
        self.lidar_pub = self.create_publisher(PointCloud2, '/vtd/lidar_pointcloud', 10)
        self.radar_pub = self.create_publisher(PointCloud2, '/vtd/radar_pointcloud', 10)

        # ✅ [수정] 자차 속도 계산을 위한 이전 위치/시간 저장용 변수
        #    (RDB_OBJECT_STATE 페이로드 내 정확한 speed 필드 오프셋이 확인되지
        #     않아, 위치값을 시간差 미분하는 방식으로 안전하게 속도를 추정합니다.
        #     실제 RDB에 speed 필드가 별도 존재한다면 VTD SDK 문서 확인 후
        #     직접 파싱하는 편이 더 정확합니다.)
        self.prev_pos = None
        self.prev_pos_time = None

        # ----------------------------------------------------
        # 3. 소켓 수신 백그라운드 스레드 생성
        # ----------------------------------------------------
        self.sock = None
        self.is_running = True
        self.recv_thread = threading.Thread(target=self.rdb_worker_thread, daemon=True)
        self.recv_thread.start()

    def recv_exact(self, n_bytes):
        buffer = bytearray()
        while len(buffer) < n_bytes and self.is_running and rclpy.ok():
            try:
                packet = self.sock.recv(n_bytes - len(buffer))
                if not packet:
                    return None
                buffer.extend(packet)
            except socket.timeout:
                continue
            except Exception as e:
                self.get_logger().error(f'Socket Recv Error: {e}')
                return None
        return bytes(buffer)

    def rdb_worker_thread(self):
        while self.is_running and rclpy.ok():
            if self.sock is None:
                try:
                    self.get_logger().info(f'Connecting to VTD RDB ({self.vtd_ip}:{self.vtd_port})...')
                    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.sock.settimeout(1.0)
                    self.sock.connect((self.vtd_ip, self.vtd_port))
                    self.get_logger().info(f'🟢 Connected to VTD RDB Server ({self.vtd_ip}:{self.vtd_port})')
                except Exception as e:
                    self.get_logger().error(f'🔴 RDB Connection Failed: {e}. Retrying in 2 sec...')
                    self.sock = None
                    rclpy.sleep_for(2.0)
                    continue

            hdr_bytes = self.recv_exact(RDB_MSG_HDR_SIZE)
            if not hdr_bytes:
                self.get_logger().warn('VTD Disconnected. Reconnecting...')
                if self.sock:
                    self.sock.close()
                self.sock = None
                continue

            magic_no, version, header_size, data_size, frame_no, sim_time = struct.unpack('<HHIII d', hdr_bytes)

            if magic_no != RDB_MAGIC_NO:
                continue
            if data_size == 0:
                continue

            data_bytes = self.recv_exact(data_size)
            if not data_bytes:
                continue

            offset = 0
            while offset < data_size:
                if offset + RDB_ENTRY_HDR_SIZE > data_size:
                    break

                entry_hdr = data_bytes[offset:offset + RDB_ENTRY_HDR_SIZE]
                e_hdr_size, e_data_size, element_size, pkg_id, flags = struct.unpack('<IIIHH', entry_hdr)

                entry_payload_offset = offset + e_hdr_size
                entry_data = data_bytes[entry_payload_offset : entry_payload_offset + e_data_size]

                self.process_rdb_package(pkg_id, entry_data)
                offset += (e_hdr_size + e_data_size)

    def process_rdb_package(self, pkg_id, payload):
        try:
            if pkg_id == RDB_PKG_ID_ROAD_POS:
                self.parse_road_pos(payload)
            elif pkg_id == RDB_PKG_ID_OBJECT_STATE:
                self.parse_object_state(payload)
            elif pkg_id == RDB_PKG_ID_IMAGE:
                self.parse_image(payload)
            elif pkg_id == RDB_PKG_ID_SENSOR_STATE:
                # 라이다/레이다 데이터 처리로 연결
                self.parse_sensor_data(payload)
        except Exception as e:
            self.get_logger().warn(f'Package Parsing Error (pkg_id={pkg_id}): {e}')

    def parse_road_pos(self, payload):
        if len(payload) < 28:
            return
        try:
            player_id, road_id, lane_id = struct.unpack('<iih', payload[:10])
            offset, hdg_rel = struct.unpack('<ff', payload[16:24])
            if player_id == 1:
                msg = Float64MultiArray()
                msg.data = [float(lane_id), float(offset), float(hdg_rel)]
                self.lane_pub.publish(msg)
        except Exception:
            pass

    def parse_object_state(self, payload):
        if len(payload) < 88:
            return
        obj_id = struct.unpack('<I', payload[:4])[0]
        if obj_id == 1:
            pos_x, pos_y, pos_z = struct.unpack('<ddd', payload[48:72])
            heading, pitch, roll = struct.unpack('<fff', payload[72:84])
            msg = Float64MultiArray()
            msg.data = [pos_x, pos_y, pos_z, heading, pitch, roll]
            self.state_pub.publish(msg)

            # ✅ [수정] '/vtd/ego_speed' 발행: 위치값의 시간差 미분으로 자차 속도 추정
            now = time.time()
            if self.prev_pos is not None and self.prev_pos_time is not None:
                dt = now - self.prev_pos_time
                if dt > 1e-4:
                    dx = pos_x - self.prev_pos[0]
                    dy = pos_y - self.prev_pos[1]
                    speed = math.hypot(dx, dy) / dt  # m/s
                    speed_msg = Float64()
                    speed_msg.data = float(speed)
                    self.speed_pub.publish(speed_msg)
            self.prev_pos = (pos_x, pos_y, pos_z)
            self.prev_pos_time = now

    def parse_image(self, payload):
        if len(payload) < 24:
            return
        target_id, width, height, pixel_size, pixel_format = struct.unpack('<IHHBB', payload[:10])
        raw_pixels = payload[24:]
        if len(raw_pixels) > 0:
            msg = Image()
            msg.height = height
            msg.width = width
            msg.encoding = 'bgr8' if pixel_format == 1 else 'rgb8'
            msg.is_bigendian = False
            msg.step = width * pixel_size
            msg.data = raw_pixels
            self.camera_pub.publish(msg)

    def parse_sensor_data(self, payload):
        """
        VTD 센서(라이다/레이다) 데이터를 퍼셉션 노드가 처리할 수 있도록
        PointCloud2 형식으로 변환하여 퍼블리시합니다.
        (실제 RDB 바이너리 구조에 맞춰 파싱 부분은 조정될 수 있습니다 — 현재는 예시용 더미 데이터)
        """
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = 'sensor_frame'

        # --- 임시 데이터 생성 예시 (실제 payload 해독 로직으로 대체 필요) ---
        dummy_lidar_points = np.array([[5.0, 0.0, 0.0], [10.0, 1.5, 0.5]], dtype=np.float32)
        dummy_radar_points = np.array([[15.0, 0.0, 0.0, -2.5]], dtype=np.float32)

        lidar_msg = pc2.create_cloud_xyz32(header, dummy_lidar_points.tolist())
        self.lidar_pub.publish(lidar_msg)

        radar_fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='velocity', offset=12, datatype=PointField.FLOAT32, count=1),
        ]
        radar_msg = pc2.create_cloud(header, radar_fields, dummy_radar_points.tolist())
        self.radar_pub.publish(radar_msg)

    def destroy_node(self):
        self.is_running = False
        if self.sock:
            self.sock.close()
        super().destroy_node()


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