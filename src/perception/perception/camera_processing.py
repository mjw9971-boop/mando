import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2

class CameraProcessing(Node):

    def __init__(self):
        super().__init__('camera_processing')
        self.get_logger().info('📷 Camera Processing Node Started')

        self.bridge = CvBridge()

        # 1. VTD 카메라 토픽 구독 (시뮬레이터 접속 시 토픽명 변경 가능)
        self.sub_image = self.create_subscription(
            Image, '/vtd/camera/image_raw', self.image_callback, 10
        )

        # 2. PathPlanner로 보낼 신호등 인식 결과 발행
        self.pub_traffic_light = self.create_publisher(
            String, '/perception/traffic_light', 10
        )

    def image_callback(self, msg):
        try:
            # ROS2 Image 메시지를 OpenCV 이미지(BGR)로 변환
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge 변환 실패: {e}')
            return

        # ----------------------------------------------------
        # 📌 [OpenCV / YOLO 신호등 인식 알고리즘 들어갈 자리]
        # ----------------------------------------------------
        # 예시: ROI 영역 지정 및 색상(HSV) 처리
        traffic_status = "GREEN"  # 기본값 (RED / GREEN / UNKNOWN)

        # 3. 결과 발행
        status_msg = String()
        status_msg.data = traffic_status
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