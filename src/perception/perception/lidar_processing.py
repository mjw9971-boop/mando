import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import String
import numpy as np
import open3d as o3d
import json
import math

class LidarProcessing(Node):
    def __init__(self):
        super().__init__('lidar_processing')
        self.get_logger().info('🛸 [Lidar Processing] 카메라/센서퓨전 연동용 라이다 객체 인식 노드 시작')

        self.sub_pc = self.create_subscription(
            PointCloud2, '/vtd/lidar_pointcloud', self.lidar_callback, 10
        )
        self.pub_objects = self.create_publisher(String, '/perception/lidar_objects', 10)

        # ROI 설정 (VTD 라이다 기준)
        self.roi_x_min, self.roi_x_max = 0.5, 30.0   # 전방 (m)
        self.roi_y_min, self.roi_y_max = -5.0, 5.0   # 좌우 (m)
        self.roi_z_min, self.roi_z_max = -1.2, 1.0   # 높이 (m)

        self.dbscan_eps = 0.6
        self.dbscan_min_points = 8

    def lidar_callback(self, msg):
        gen = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
        points = np.array(list(gen), dtype=np.float32)

        if len(points) == 0:
            self.publish_empty()
            return

        # 1. ROI 필터링 (VTD 라이다: X가 전방, Y가 좌우)
        mask = (
            (points[:, 0] >= self.roi_x_min) & (points[:, 0] <= self.roi_x_max) &
            (points[:, 1] >= self.roi_y_min) & (points[:, 1] <= self.roi_y_max) &
            (points[:, 2] >= self.roi_z_min) & (points[:, 2] <= self.roi_z_max)
        )
        roi_points = points[mask]

        if len(roi_points) < self.dbscan_min_points:
            self.publish_empty()
            return

        # 2. 바닥 제거 (RANSAC)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(roi_points)

        try:
            _, inliers = pcd.segment_plane(distance_threshold=0.2, ransac_n=3, num_iterations=100)
            obstacle_pcd = pcd.select_by_index(inliers, invert=True)
        except Exception:
            self.publish_empty()
            return

        obstacle_points = np.asarray(obstacle_pcd.points)
        if len(obstacle_points) < self.dbscan_min_points:
            self.publish_empty()
            return

        # 3. 객체 군집화 (DBSCAN)
        labels = np.array(obstacle_pcd.cluster_dbscan(eps=self.dbscan_eps, min_points=self.dbscan_min_points))
        max_label = labels.max()

        if max_label < 0:
            self.publish_empty()
            return

        # 4. 중심점 추출 → 카메라 노드와 동일한 JSON 스키마로 변환
        detected_objects = []
        for i in range(max_label + 1):
            cluster_indices = np.where(labels == i)[0]
            cluster_points = obstacle_points[cluster_indices]

            centroid = np.mean(cluster_points, axis=0)

            # 📌 카메라 노드 좌표축과 맞춤:
            # camera_x (좌우) = -centroid[1] (라이다 Y축 반전: 우측 +, 좌측 -)
            # camera_y (전방) = centroid[0]  (라이다 X축: 전방 +)
            real_dist_x = float(-centroid[1])
            real_dist_y = float(centroid[0])
            straight_dist = math.hypot(real_dist_x, real_dist_y)

            # ✅ [수정] 라이다는 YOLO 같은 분류기가 없으므로 class는 "unknown"으로 표시.
            #    sensor_fusion_node에서는 class 매칭이 아니라 거리 기반으로만 교차검증해야 함.
            #    camera_processing.py와 동일한 키(class/distance/lateral/forward)로 통일.
            detected_objects.append({
                "class": "unknown",
                "distance": round(straight_dist, 2),
                "lateral": round(real_dist_x, 2),
                "forward": round(real_dist_y, 2),
                "num_points": int(len(cluster_points)),
            })

        msg_out = String()
        # ✅ [수정] 기존 "lidar:[x,y]" 파이프 문자열 → JSON 배열로 통일
        #    (sensor_fusion_node의 json.loads()와 호환되도록)
        msg_out.data = json.dumps(detected_objects)
        self.pub_objects.publish(msg_out)

    def publish_empty(self):
        msg_out = String()
        # ✅ [수정] "NONE" → 빈 JSON 배열 "[]" (json.loads 안전)
        msg_out.data = "[]"
        self.pub_objects.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = LidarProcessing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()