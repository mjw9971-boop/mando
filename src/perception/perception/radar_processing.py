import struct
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Float64, String
import json


class VtdRadarProcessingNode(Node):

    def __init__(self):
        super().__init__('vtd_radar_processing_node')

        # -------------------------------------------------------------
        # 1. ROS 2 퍼블리셔 / 서브스크라이버 설정
        # -------------------------------------------------------------
        # VTD 레이더 센서 토픽 수신 (시뮬레이터 설정에 맞춰 토픽명 확인)
        self.sub_radar = self.create_subscription(
            PointCloud2, '/vtd/radar_pointcloud', self.radar_callback, 10
        )

        # 판단/퓨전 노드로 퍼블리시할 결과 토픽 (전방 타겟 상대속도, 거리, TTC)
        self.pub_radar_target = self.create_publisher(
            String, '/perception/radar_target', 10
        )

        # -------------------------------------------------------------
        # 2. 레이더 필터링 파라미터 (전방 관심 차선 영역)
        # -------------------------------------------------------------
        self.ROI_X_MIN = 1.0    # 전방 1m 이상
        self.ROI_X_MAX = 80.0   # 레이더는 먼 거리까지 탐지 가능 (80m)
        self.ROI_Y_MIN = -2.0   # 내 차선 내부 영역 (좌우 -2m)
        self.ROI_Y_MAX = 2.0

        self.get_logger().info('📻 VTD 레이더 퍼셉션 노드가 성공적으로 시작되었습니다.')

    def radar_callback(self, msg):
        # -------------------------------------------------------------
        # Step 1. PointCloud2 파싱 (x, y, z, velocity)
        # -------------------------------------------------------------
        # VTD 레이더가 속도 정보(vx 또는 velocity) 필드를 포함하는 경우
        field_names = [field.name for field in msg.fields]
        
        # velocity 필드가 존재하면 포함하여 파싱, 없으면 x, y, z만 기본 파싱
        has_velocity = 'velocity' in field_names or 'vx' in field_names
        vel_field = 'velocity' if 'velocity' in field_names else ('vx' if 'vx' in field_names else None)

        if has_velocity:
            gen = pc2.read_points(msg, field_names=('x', 'y', 'z', vel_field), skip_nans=True)
            raw_data = np.array(list(gen), dtype=np.float32)
        else:
            gen = pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True)
            raw_points = np.array(list(gen), dtype=np.float32)
            if len(raw_points) == 0:
                self.publish_empty_result()
                return
            # 속도 정보가 메타데이터에 없을 경우 0.0으로 임시 채움
            raw_data = np.hstack((raw_points, np.zeros((len(raw_points), 1), dtype=np.float32)))

        if len(raw_data) == 0:
            self.publish_empty_result()
            return

        points = raw_data[:, :3]
        velocities = raw_data[:, 3]

        # -------------------------------------------------------------
        # Step 2. 관심 영역(ROI) 필터링
        # -------------------------------------------------------------
        roi_mask = (
            (points[:, 0] >= self.ROI_X_MIN) & (points[:, 0] <= self.ROI_X_MAX) &
            (points[:, 1] >= self.ROI_Y_MIN) & (points[:, 1] <= self.ROI_Y_MAX)
        )

        valid_points = points[roi_mask]
        valid_vels = velocities[roi_mask]

        if len(valid_points) == 0:
            self.publish_empty_result()
            return

        # -------------------------------------------------------------
        # Step 3. 가장 가까운 전방 타겟 추출 및 상태 계산
        # -------------------------------------------------------------
        # X축(전방 거리) 기준 정렬
        closest_idx = np.argmin(valid_points[:, 0])
        target_point = valid_points[closest_idx]
        target_vel = float(valid_vels[closest_idx])  # m/s 단위 (음수: 접근 중, 양수: 멀어지는 중)

        target_dist = float(np.sqrt(target_point[0]**2 + target_point[1]**2))

        # ---------------------------------------------------------
        # [수정됨] TTC (충돌 임계 시간) 계산 로직 추가
        # ---------------------------------------------------------
        ttc = 999.0
        if target_vel < -0.1:  # 앞차가 내 차와 가까워지고 있을 때만 계산 (0으로 나누기 방지)
            ttc = target_dist / abs(target_vel)

        # 타겟의 상태 추정 (상대속도 기준)
        # target_vel < -0.5: 앞차가 감속 중이거나 접근 중
        # target_vel > 0.5: 앞차가 가속 중이거나 멀어지는 중
        status = "CRUISING"
        if target_vel < -0.5:
            status = "APPROACHING"  # 접근 중 (주의/감속 필요)
        elif target_vel > 0.5:
            status = "MOVING_AWAY"  # 멀어지는 중

        target_info = {
            'target_detected': True,
            'distance': round(target_dist, 2),       # m
            'relative_velocity': round(target_vel, 2), # m/s
            'ttc': round(ttc, 2),                    # [추가됨] 퓨전 노드로 보낼 TTC 값
            'status': status
        }

        # -------------------------------------------------------------
        # Step 4. ROS 2 토픽 발행
        # -------------------------------------------------------------
        msg_target = String()
        msg_target.data = json.dumps(target_info)
        self.pub_radar_target.publish(msg_target)

        self.get_logger().info(
            f'  [Radar] 전방 타겟 거리: {target_dist:.2f}m | 상대속도: {target_vel:.2f}m/s | TTC: {ttc:.2f}s ({status})'
        )

    def publish_empty_result(self):
        """타겟이 없을 때 기본값 퍼블리시"""
        target_info = {
            'target_detected': False,
            'distance': 999.0,
            'relative_velocity': 0.0,
            'ttc': 999.0,          # [추가됨] 타겟이 없으므로 충돌 시간 무한대
            'status': "CLEAR"
        }
        msg_target = String()
        msg_target.data = json.dumps(target_info)
        self.pub_radar_target.publish(msg_target)


def main(args=None):
    rclpy.init(args=args)
    node = VtdRadarProcessingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()