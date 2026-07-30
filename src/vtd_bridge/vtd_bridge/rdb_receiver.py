"""
HL FMA 2026 자율주행 시뮬레이션 대회 - VTD 연동 노드 (rdb_receiver)

=====================================================================
대회 통신 규격 ("연동 구성 및 통신 규격" 슬라이드 기준)
=====================================================================
  - TCP  9910 : 학생 PC <-> Host PC   | 제어기 <-> VTD 데이터 연동 (제어 명령 송신)
  - RTSP 8554 : 학생 PC -> Host PC 접속 | 카메라 영상 스트리밍 수신
  - UDP  9912 : Host PC -> 학생 PC    | 시뮬레이션 상태 전송 (속도/위치 등)

*** 중요: 아래는 "가정"을 포함한 뼈대 코드입니다 ***
9910(제어 명령 패킷 구조)과 9912(상태 패킷 구조)의 정확한 스펙은
2026-08-13 온라인 교육에서 공식 공개됩니다. 그 전까지는:

  1) UDP 9912 상태 채널은 기존에 확인했던 VIRES RDB 포맷(magic 0x2364 헤더 +
     엔트리 구조)을 그대로 UDP로 보낸다고 "가정"하고 파싱합니다.
     -> 만약 실제로 RDB 포맷이 아니라면(JSON, 커스텀 struct 등)
        _parse_state_payload() 함수만 교체하면 됩니다.

  2) TCP 9910 제어 채널은 (steer, throttle, brake) 3개 float32를
     그대로 보내는 placeholder 포맷입니다.
     -> 실제 포맷 확인되면 CONTROL_CMD_FMT와 _on_control_cmd()만 고치면 됩니다.

  3) RTSP 경로(rtsp_path)도 실제 값을 모르므로 파라미터로 빼뒀습니다.
     -> 8/13 교육 후 launch 파일이나 파라미터로 실제 경로를 넣어주세요.

  4) 이전 버전에서 카메라 이미지를 TCP RDB 스트림(RDB_PKG_ID_IMAGE)으로
     받던 로직은 제거했습니다. 대회 규격상 영상은 RTSP(8554)로 오기 때문입니다.
     LiDAR는 별도 포트 언급이 없어, 우선 상태 채널(UDP 9912)의 RDB 엔트리
     안에 같이 들어온다고 가정하고 처리합니다. 8/13 교육 후 다르면 조정 필요.
=====================================================================
"""
import socket
import struct
import threading
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge

# ---------------------------------------------------------------------------
# RDB 상수 / 구조체 (UDP 9912 상태 채널이 기존 VIRES RDB 포맷을 따른다고 가정)
# ---------------------------------------------------------------------------
RDB_MAGIC_NO = 0x2364

RDB_MSG_HDR_FMT = "<HHIIId"  # magicNo(u16) version(u16) headerSize(u32) dataSize(u32) frameNo(u32) simTime(d)
RDB_MSG_HDR_SIZE = struct.calcsize(RDB_MSG_HDR_FMT)

RDB_ENTRY_HDR_FMT = "<IIHHHH"  # headerSize(u32) dataSize(u32) elementSize(u16) pkgId(u16) flags(u16) reserved(u16)
RDB_ENTRY_HDR_SIZE = struct.calcsize(RDB_ENTRY_HDR_FMT)

RDB_PKG_ID_OBJECT_STATE = 9
RDB_PKG_FLAG_EXTENDED = 1

# RDB_OBJECT_STATE_BASE_t
RDB_OBJ_BASE_FMT = "<IBBH32s" + "6f" + "3d3fBBHI"
RDB_OBJ_BASE_SIZE = struct.calcsize(RDB_OBJ_BASE_FMT)

# RDB_COORD_t (속도 필드에 재사용)
RDB_COORD_FMT = "<3d3fBBHI"
RDB_COORD_SIZE = struct.calcsize(RDB_COORD_FMT)

# 가정한 포인트클라우드 헤더 포맷: numPoints(u32) sensorId(u16) reserved(u16)
RDB_POINTCLOUD_HDR_FMT = "<IHH"
RDB_POINTCLOUD_HDR_SIZE = struct.calcsize(RDB_POINTCLOUD_HDR_FMT)

# 제어 명령(TCP 9910) placeholder 포맷: steer(f32) throttle(f32) brake(f32)
CONTROL_CMD_FMT = "<3f"


# ---------------------------------------------------------------------------
# LiDAR point cloud -> BEV 격자 변환
# ---------------------------------------------------------------------------
class BevProjector:
    def __init__(self, bev_h=256, bev_w=256,
                 x_range=(-10.0, 60.0), y_range=(-35.0, 35.0), z_clip=(-3.0, 3.0)):
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_clip
        self.x_res = (self.x_max - self.x_min) / bev_h
        self.y_res = (self.y_max - self.y_min) / bev_w

    def project(self, points_xyz: np.ndarray) -> np.ndarray:
        """points_xyz: (N,3) 이상 [x,y,z,...] -> (bev_h, bev_w, 2) float32 [occupancy, norm_height]"""
        if points_xyz is None or points_xyz.size == 0:
            return np.zeros((self.bev_h, self.bev_w, 2), dtype=np.float32)

        x, y, z = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
        mask = (
            (x >= self.x_min) & (x < self.x_max) &
            (y >= self.y_min) & (y < self.y_max) &
            (z >= self.z_min) & (z < self.z_max)
        )
        x, y, z = x[mask], y[mask], z[mask]
        if x.size == 0:
            return np.zeros((self.bev_h, self.bev_w, 2), dtype=np.float32)

        row = np.clip(((x - self.x_min) / self.x_res).astype(np.int32), 0, self.bev_h - 1)
        col = np.clip(((y - self.y_min) / self.y_res).astype(np.int32), 0, self.bev_w - 1)

        occupancy = np.zeros((self.bev_h, self.bev_w), dtype=np.float32)
        height = np.zeros((self.bev_h, self.bev_w), dtype=np.float32)

        flat_idx = row * self.bev_w + col
        order = np.argsort(z)  # 오름차순 정렬 후 순서대로 기록 -> 마지막에 최댓값이 남음
        flat_idx_sorted = flat_idx[order]
        z_sorted = z[order]

        occ_flat = occupancy.reshape(-1)
        height_flat = height.reshape(-1)
        occ_flat[flat_idx_sorted] = 1.0
        height_flat[flat_idx_sorted] = z_sorted

        height_norm = (height_flat.reshape(self.bev_h, self.bev_w) - self.z_min) / (self.z_max - self.z_min)
        height_norm = np.clip(height_norm, 0.0, 1.0)
        height_norm = np.where(occupancy > 0, height_norm, 0.0)

        return np.stack([occupancy, height_norm], axis=-1).astype(np.float32)


def _recv_exact(sock: socket.socket, n: int):
    """소켓에서 정확히 n바이트를 읽는다. 연결 종료 시 None 반환."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


class RdbReceiverNode(Node):
    def __init__(self):
        super().__init__("rdb_receiver")

        # --- 연결 대상 (연습: 127.0.0.1 / 대회 당일: Host PC IP로 교체) ---
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("control_port", 9910)   # TCP - 제어 명령 송신
        self.declare_parameter("rtsp_port", 8554)       # RTSP - 카메라 영상
        self.declare_parameter("rtsp_path", "/live")    # *** 실제 경로 8/13 교육 후 확인 필요 ***
        self.declare_parameter("state_port", 9912)      # UDP - 시뮬레이션 상태

        self.declare_parameter("ego_player_id", 1)

        # --- LiDAR / BEV 관련 파라미터 ---
        self.declare_parameter("lidar_pkg_id", -1)      # 확인 전이면 -1 (파싱 안 하고 로그만)
        self.declare_parameter("point_stride", 4)        # x,y,z,intensity = 4
        self.declare_parameter("bev_h", 256)
        self.declare_parameter("bev_w", 256)
        self.declare_parameter("bev_x_min", -10.0)
        self.declare_parameter("bev_x_max", 60.0)
        self.declare_parameter("bev_y_min", -35.0)
        self.declare_parameter("bev_y_max", 35.0)
        self.declare_parameter("bev_z_min", -3.0)
        self.declare_parameter("bev_z_max", 3.0)
        self.declare_parameter("reconnect_interval", 2.0)

        self.host = self.get_parameter("host").value
        self.control_port = self.get_parameter("control_port").value
        self.rtsp_port = self.get_parameter("rtsp_port").value
        self.rtsp_path = self.get_parameter("rtsp_path").value
        self.state_port = self.get_parameter("state_port").value

        self.ego_id = self.get_parameter("ego_player_id").value
        self.lidar_pkg_id = self.get_parameter("lidar_pkg_id").value
        self.point_stride = self.get_parameter("point_stride").value
        self.reconnect_interval = self.get_parameter("reconnect_interval").value

        self.bev_projector = BevProjector(
            bev_h=self.get_parameter("bev_h").value,
            bev_w=self.get_parameter("bev_w").value,
            x_range=(self.get_parameter("bev_x_min").value, self.get_parameter("bev_x_max").value),
            y_range=(self.get_parameter("bev_y_min").value, self.get_parameter("bev_y_max").value),
            z_clip=(self.get_parameter("bev_z_min").value, self.get_parameter("bev_z_max").value),
        )

        self.bridge = CvBridge()
        self.pub_rgb = self.create_publisher(Image, "/rdb/rgb", 10)
        self.pub_bev = self.create_publisher(Image, "/rdb/bev", 10)
        self.pub_speed = self.create_publisher(Float32, "/rdb/speed", 10)

        # 제어 명령(steer, throttle, brake)을 이 토픽으로 구독해서 TCP 9910으로 송신
        # *** 실제 e2e_inference_node.py 가 publish 하는 토픽 이름/타입에 맞춰 조정 필요 ***
        self.sub_control = self.create_subscription(
            Float32MultiArray, "/cmd/control", self._on_control_cmd, 10
        )

        self._stop = False
        self._seen_unknown_ids = {}  # pkgId -> 마지막 로그 시각 (스팸 방지용)

        self._control_sock = None
        self._control_lock = threading.Lock()

        # 3개 채널을 각각 독립된 스레드로 동시에 실행
        self._control_thread = threading.Thread(target=self._control_connect_loop, daemon=True)
        self._video_thread = threading.Thread(target=self._video_loop, daemon=True)
        self._state_thread = threading.Thread(target=self._state_loop, daemon=True)

        self._control_thread.start()
        self._video_thread.start()
        self._state_thread.start()

        self.get_logger().info(
            f"RDB receiver 시작 | host={self.host} "
            f"control(TCP)={self.control_port} rtsp={self.rtsp_port}{self.rtsp_path} state(UDP)={self.state_port} "
            f"| ego_id={self.ego_id} lidar_pkg_id={self.lidar_pkg_id} "
            f"(-1이면 아직 확인 전 -> 미확인 패키지 로그를 보고 값을 찾아서 파라미터로 넣어주세요)"
        )

    # ==================================================================
    # 1) TCP 9910 - 제어 명령 송신 채널
    # ==================================================================
    def _control_connect_loop(self):
        while not self._stop:
            try:
                sock = socket.create_connection((self.host, self.control_port), timeout=5.0)
                with self._control_lock:
                    self._control_sock = sock
                self.get_logger().info(f"제어 채널(TCP {self.control_port}) 연결 성공")
                # 연결 유지. 필요하면 여기서 Host로부터의 ack/응답 수신 로직 추가 가능.
                while not self._stop:
                    time.sleep(1.0)
                    with self._control_lock:
                        if self._control_sock is None:
                            break
            except OSError as e:
                self.get_logger().warn(f"제어 채널 연결 실패 ({e}), {self.reconnect_interval}s 후 재시도")
                time.sleep(self.reconnect_interval)
            finally:
                with self._control_lock:
                    if self._control_sock is not None:
                        try:
                            self._control_sock.close()
                        except OSError:
                            pass
                        self._control_sock = None

    def _on_control_cmd(self, msg: Float32MultiArray):
        """
        /cmd/control (steer, throttle, brake) 를 받아 TCP 9910으로 송신.
        *** placeholder 포맷: CONTROL_CMD_FMT = "<3f" ***
        실제 패킷 스펙(헤더 유무, 순서, 추가 필드 등)은 8/13 교육에서 확인 후 이 함수만 고치면 됨.
        """
        if len(msg.data) < 3:
            self.get_logger().warn("제어 명령 데이터 길이 부족 (steer, throttle, brake 3개 필요)")
            return
        steer, throttle, brake = msg.data[0], msg.data[1], msg.data[2]
        packet = struct.pack(CONTROL_CMD_FMT, steer, throttle, brake)

        with self._control_lock:
            sock = self._control_sock
        if sock is None:
            self.get_logger().warn("제어 채널 미연결 상태 - 명령 전송 스킵")
            return
        try:
            sock.sendall(packet)
        except OSError as e:
            self.get_logger().warn(f"제어 명령 송신 실패: {e}")
            with self._control_lock:
                self._control_sock = None  # 다음 루프에서 재연결

    # ==================================================================
    # 2) RTSP 8554 - 카메라 영상 수신 채널
    # ==================================================================
    def _video_loop(self):
        rtsp_url = f"rtsp://{self.host}:{self.rtsp_port}{self.rtsp_path}"
        while not self._stop:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                self.get_logger().warn(f"RTSP 연결 실패 ({rtsp_url}), {self.reconnect_interval}s 후 재시도")
                cap.release()
                time.sleep(self.reconnect_interval)
                continue

            self.get_logger().info(f"RTSP 영상 수신 시작 ({rtsp_url})")
            while not self._stop:
                ret, frame_bgr = cap.read()
                if not ret:
                    self.get_logger().warn("RTSP 프레임 수신 실패 - 재연결 시도")
                    break
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "rgb_camera"
                self.pub_rgb.publish(msg)

            cap.release()
            time.sleep(self.reconnect_interval)

    # ==================================================================
    # 3) UDP 9912 - 시뮬레이션 상태 수신 채널
    # ==================================================================
    def _state_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind(("0.0.0.0", self.state_port))
        except OSError as e:
            self.get_logger().error(f"상태 채널(UDP {self.state_port}) bind 실패: {e}")
            return
        sock.settimeout(1.0)
        self.get_logger().info(f"상태 채널(UDP {self.state_port}) 수신 대기")

        while not self._stop:
            try:
                payload, _addr = sock.recvfrom(65536)
            except socket.timeout:
                continue
            except OSError as e:
                self.get_logger().warn(f"UDP 수신 오류: {e}")
                continue
            self._parse_state_payload(payload)

        sock.close()

    def _parse_state_payload(self, payload: bytes):
        """
        *** 가정: UDP 9912 상태 패킷이 기존 VIRES RDB 포맷(헤더+엔트리)을 그대로 따른다고 가정 ***
        실제로 다른 포맷(JSON/커스텀 struct 등)이면 이 함수만 교체하면 됨.
        """
        if len(payload) < RDB_MSG_HDR_SIZE:
            return
        magic, version, header_size, data_size, frame_no, sim_time = struct.unpack(
            RDB_MSG_HDR_FMT, payload[:RDB_MSG_HDR_SIZE]
        )
        if magic != RDB_MAGIC_NO:
            # RDB 포맷이 아닐 수 있음 -> 실제 포맷 파악용으로 로그만 남김
            self._log_unknown_pkg(-1, len(payload))
            return

        entries = payload[header_size: header_size + data_size]
        self._parse_entries(entries)

    # ------------------------------------------------------------------
    # RDB 엔트리 파싱 (object_state / point_cloud)
    # ------------------------------------------------------------------
    def _parse_entries(self, payload: bytes):
        offset = 0
        n = len(payload)
        while offset + RDB_ENTRY_HDR_SIZE <= n:
            (entry_hdr_size, entry_data_size, element_size,
             pkg_id, flags, _reserved) = struct.unpack(
                RDB_ENTRY_HDR_FMT, payload[offset: offset + RDB_ENTRY_HDR_SIZE]
            )
            data_start = offset + entry_hdr_size
            data_end = data_start + entry_data_size
            if data_end > n or entry_hdr_size == 0:
                break

            entry_data = payload[data_start:data_end]

            try:
                if pkg_id == RDB_PKG_ID_OBJECT_STATE:
                    extended = bool(flags & RDB_PKG_FLAG_EXTENDED)
                    self._handle_object_state(entry_data, extended)
                elif self.lidar_pkg_id != -1 and pkg_id == self.lidar_pkg_id:
                    self._handle_point_cloud(entry_data)
                else:
                    self._log_unknown_pkg(pkg_id, entry_data_size)
            except Exception as e:
                self.get_logger().warn(f"RDB 패키지(id={pkg_id}) 파싱 실패: {e}")

            offset = data_end

    def _log_unknown_pkg(self, pkg_id: int, data_size: int):
        """확인 안 된 pkgId를 5초에 한 번씩만 로그로 남긴다 (lidar_pkg_id 찾는 용도)."""
        now = time.time()
        last = self._seen_unknown_ids.get(pkg_id, 0.0)
        if now - last > 5.0:
            self._seen_unknown_ids[pkg_id] = now
            self.get_logger().info(f"[미확인 패키지] pkgId={pkg_id}  dataSize={data_size} bytes")

    def _handle_point_cloud(self, data: bytes):
        """
        가정한 포맷: [numPoints(u32) sensorId(u16) reserved(u16)] + numPoints*point_stride*float32
        실제 포맷이 다르면 이 함수만 고치면 된다.
        """
        if len(data) < RDB_POINTCLOUD_HDR_SIZE:
            return
        num_points, sensor_id, _reserved = struct.unpack(
            RDB_POINTCLOUD_HDR_FMT, data[:RDB_POINTCLOUD_HDR_SIZE]
        )
        raw = data[RDB_POINTCLOUD_HDR_SIZE:]

        expected_bytes = num_points * self.point_stride * 4
        if len(raw) < expected_bytes or num_points == 0:
            self.get_logger().warn(
                f"LiDAR 포인트 데이터 크기 불일치 (기대={expected_bytes}, 실제={len(raw)}) — "
                f"point_stride 파라미터를 실제 포맷에 맞게 조정 필요"
            )
            return

        points = np.frombuffer(raw[:expected_bytes], dtype=np.float32).reshape(
            (num_points, self.point_stride)
        )
        bev = self.bev_projector.project(points[:, :3])

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "lidar_bev"
        msg.height, msg.width = bev.shape[0], bev.shape[1]
        msg.encoding = "32FC2"
        msg.is_bigendian = 0
        msg.step = msg.width * 2 * 4  # channels(2) * float32(4bytes)
        msg.data = bev.tobytes()
        self.pub_bev.publish(msg)

    def _handle_object_state(self, data: bytes, extended: bool):
        if len(data) < RDB_OBJ_BASE_SIZE:
            return
        base = struct.unpack(RDB_OBJ_BASE_FMT, data[:RDB_OBJ_BASE_SIZE])
        player_id = base[0]
        if player_id != self.ego_id or not extended:
            return

        ext_data = data[RDB_OBJ_BASE_SIZE:]
        if len(ext_data) < RDB_COORD_SIZE:
            return
        speed_coord = struct.unpack(RDB_COORD_FMT, ext_data[:RDB_COORD_SIZE])
        vx, vy, vz = speed_coord[0], speed_coord[1], speed_coord[2]
        speed = float(np.sqrt(vx * vx + vy * vy + vz * vz))

        msg = Float32()
        msg.data = speed
        self.pub_speed.publish(msg)

    def destroy_node(self):
        self._stop = True
        with self._control_lock:
            if self._control_sock is not None:
                try:
                    self._control_sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RdbReceiverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()