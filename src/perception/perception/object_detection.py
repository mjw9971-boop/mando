import json
import math
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


class ObjectDetection(Node):

  def __init__(self):
    super().__init__('object_detection')
    self.get_logger().info('📦 [Object Detection] YOLO 객체 탐지 노드 시작')

    self.bridge = CvBridge()

    # ROS 2 파라미터
    self.declare_parameter('yolo_model_path', 'yolov8n.engine')
    self.declare_parameter('conf_threshold', 0.5)
    self.declare_parameter('meters_per_pixel', 0.003)

    # YOLO 모델 로드
    model_path = self.get_parameter('yolo_model_path').value
    try:
      self.yolo_model = YOLO(model_path)
      self.get_logger().info(f'✅ YOLO 모델 로드 완료: {model_path}')
    except Exception as e:
      self.get_logger().error(f'❌ YOLO 모델 로드 실패: {e}')

    # 퍼블리셔 / 서브스크라이버
    self.sub_image = self.create_subscription(
        Image, 'vtd/camera_image', self.image_callback, 10
    )
    self.pub_objects = self.create_publisher(
        String, '/perception/camera_objects', 10
    )

  def process_yolo_detection(self, cv_image):
    h, w = cv_image.shape[:2]
    conf_thresh = self.get_parameter('conf_threshold').value
    m_per_pix = self.get_parameter('meters_per_pixel').value

    # BEV 시점 변환 행렬 생성
    src_points = np.float32(
        [[w * 0.4, h * 0.6], [w * 0.6, h * 0.6], [w * 0.1, h], [w * 0.9, h]]
    )
    dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    bev_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    results = self.yolo_model.predict(
        source=cv_image, conf=conf_thresh, verbose=False
    )

    detected_objects = []
    for r in results:
      boxes = r.boxes
      for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0])
        class_name = self.yolo_model.names[cls_id]

        # 바운딩 박스 하단 중앙점
        bottom_x = (x1 + x2) / 2.0
        bottom_y = y2

        # BEV 변환
        pt = np.array([[[bottom_x, bottom_y]]], dtype=np.float32)
        bev_pt = cv2.perspectiveTransform(pt, bev_matrix)[0][0]

        # 실제 거리(m) 환산
        real_dist_x = (bev_pt[0] - (w / 2.0)) * m_per_pix  # 좌우 거리
        real_dist_y = (h - bev_pt[1]) * m_per_pix  # 전방 거리
        straight_dist = float(math.hypot(real_dist_x, real_dist_y))  # 직선 거리

        # JSON용 스키마 생성
        detected_objects.append({
            'class': class_name,
            'distance': round(straight_dist, 2),
            'lateral': round(float(real_dist_x), 2),
            'forward': round(float(real_dist_y), 2),
        })

    return json.dumps(detected_objects)

  def image_callback(self, msg):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    except Exception as e:
      self.get_logger().error(f'CvBridge 변환 오류: {e}')
      return

    objects_json_str = self.process_yolo_detection(cv_image)

    msg_objects = String()
    msg_objects.data = objects_json_str
    self.pub_objects.publish(msg_objects)


def main(args=None):
  rclpy.init(args=args)
  node = ObjectDetection()
  try:
    rclpy.spin(node)
  except KeyboardInterrupt:
    pass
  finally:
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
  main()