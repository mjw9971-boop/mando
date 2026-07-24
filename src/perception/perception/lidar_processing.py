import numpy as np
import open3d as o3d
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Float64

class SimpleLidarNode(Node):
    def __init__(self):
        super().__init__('simple_lidar_node')
        
        # 1. 라이다 데이터 수신 (VTD 센서 토픽 이름에 맞게 수정)
        self.sub_pc = self.create_subscription(
            PointCloud2, '/vtd/lidar_pointcloud', self.lidar_callback, 10
        )
        
        # 2. 가장 가까운 장애물 거리 퍼블리시
        self.pub_dist = self.create_publisher(Float64, '/perception/obstacle_distance', 10)
        
        # [여기만 수정하세요!] 내 차가 신경 쓸 '관심 영역(ROI)' 세팅 (단위: 미터)
        self.roi_x_min = 0.5   # 내 차 바로 앞 (0.5m) 부터
        self.roi_x_max = 20.0  # 전방 20m 까지만 감지
        self.roi_y_min = -1.5  # 우측 1.5m (차선 안쪽만)
        self.roi_y_max = 1.5   # 좌측 1.5m (차선 안쪽만)
        self.roi_z_min = -1.0  # 바닥 근처 높이
        self.roi_z_max = 1.0   # 내 차 높이 정도

        self.get_logger().info('🟢 라이다 퍼셉션 노드 실행 완료! (학습 필요 없음)')

    def lidar_callback(self, msg):
        # 1. ROS2 메시지를 파이썬 숫자 배열(X, Y, Z)로 변환
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(gen), dtype=np.float32)

        if len(points) == 0:
            self.publish_safe()
            return

        # 2. 내가 설정한 관심 영역(ROI) 안의 점들만 싹둑 잘라내기
        mask = (
            (points[:, 0] >= self.roi_x_min) & (points[:, 0] <= self.roi_x_max) &
            (points[:, 1] >= self.roi_y_min) & (points[:, 1] <= self.roi_y_max) &
            (points[:, 2] >= self.roi_z_min) & (points[:, 2] <= self.roi_z_max)
        )
        roi_points = points[mask]

        # ROI 안에 점이 거의 없다면? -> 앞에 아무것도 없음 (안전)
        if len(roi_points) < 10:
            self.publish_safe()
            return

        # 3. 바닥(아스팔트) 점들 지우기 (RANSAC 알고리즘 - Open3D가 다 해줌)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(roi_points)
        _, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=100)
        
        # 바닥이 아닌 '장애물' 점들만 남기기
        obstacle_pcd = pcd.select_by_index(inliers, invert=True)
        obstacle_points = np.asarray(obstacle_pcd.points)

        if len(obstacle_points) < 5:
            self.publish_safe()
            return

        # 4. 내 차(0, 0)에서 장애물 점들까지의 2D 평면 거리 구하기
        distances = np.sqrt(obstacle_points[:, 0]**2 + obstacle_points[:, 1]**2)
        
        # 가장 짧은 거리 뽑기
        min_distance = float(np.min(distances))

        # 5. 결과 쏘기
        msg_dist = Float64()
        msg_dist.data = min_distance
        self.pub_dist.publish(msg_dist)
        
        self.get_logger().info(f'🚨 앗! 전방 {min_distance:.2f}m 에 장애물 발견!')

    def publish_safe(self):
        # 전방에 아무것도 없을 때는 999.0m (아주 멀다 = 안전하다) 라고 쏴줌
        msg_dist = Float64()
        msg_dist.data = 999.0
        self.pub_dist.publish(msg_dist)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleLidarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()