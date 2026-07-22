import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np

class CameraProcessing(Node):

    def __init__(self):
        super().__init__('camera_processing')
        self.get_logger().info('📷 [Camera Processing] 우천/반사 대비 신호등 인식 노드 시작')

        self.bridge = CvBridge()
        self.sub_image = self.create_subscription(
            Image, '/vtd/camera/image_raw', self.image_callback, 10
        )
        self.pub_traffic_light = self.create_publisher(
            String, '/perception/traffic_light', 10
        )

        # ----------------------------------------------------
        # 📌 Temporal Filter (시간 검수) 변수 설정
        # ----------------------------------------------------
        self.red_frame_count = 0
        self.RED_THRESHOLD = 5  # 연속 5프레임 이상 RED일 때만 진짜 신호로 인정

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge 변환 오류: {e}')
            return

        h, w, _ = cv_image.shape

        # ----------------------------------------------------
        # 📌 1. 상단 ROI Crop (화면 상단 35%만 사용)
        # 노면 빗물에 비친 신호등 반사 불빛을 아예 잘라내어 무시함
        # ----------------------------------------------------
        roi = cv_image[0:int(h * 0.35), :]

        # 📌 2. HSV 색상 공간 변환 및 적색 마스크 검출
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # HSV에서 Red 색상 영역 (두 영역으로 나뉨)
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 | mask2

        red_pixel_count = cv2.countNonZero(red_mask)

        # ----------------------------------------------------
        # 📌 3. Temporal Filtering (연속 검증 로직)
        # 빗방울로 인해 1프레임 간발의 차로 튄 노이즈를 걸러냄
        # ----------------------------------------------------
        if red_pixel_count > 80:  # 적색 픽셀 임계값
            self.red_frame_count += 1
        else:
            self.red_frame_count = max(0, self.red_frame_count - 1)

        # 연속 검수 기준 달성 여부 확인
        status_msg = String()
        if self.red_frame_count >= self.RED_THRESHOLD:
            status_msg.data = "RED"
        else:
            status_msg.data = "GREEN"

        self.pub_traffic_light.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraProcessing()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()