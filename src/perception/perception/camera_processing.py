import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import time
from ultralytics import YOLO

class _Stabilizer:
    """단순 EMA를 넘어선 강력한 차선 오차 안정화기"""
    def __init__(self, ema_tau_s=0.065, outlier_jump=0.5, outlier_relatch_s=0.16, lost_stop_s=0.26):
        self.ema = None
        self.missing_s = 0.0   
        self.rejects_s = 0.0   
        
        self.ema_tau_s = ema_tau_s
        self.outlier_jump = outlier_jump
        self.outlier_relatch_s = outlier_relatch_s
        self.lost_stop_s = lost_stop_s

    def _ema_alpha(self, dt_s):
        if self.ema_tau_s <= 0.0:
            return 1.0
        if dt_s <= 0.0:
            return 0.0
        return 1.0 - math.exp(-dt_s / self.ema_tau_s)

    def update(self, center_error, is_valid, dt_s):
        if center_error is None or not is_valid:
            self.missing_s += dt_s
            self.rejects_s = 0.0
            return self.ema, ('LOST' if self.missing_s >= self.lost_stop_s else 'HOLD')
            
        self.missing_s = 0.0
        
        if self.ema is not None and abs(center_error - self.ema) > self.outlier_jump:
            self.rejects_s += dt_s
            if self.rejects_s < self.outlier_relatch_s:
                return self.ema, 'OUTLIER'
            self.ema = None
            
        self.rejects_s = 0.0
        a = self._ema_alpha(dt_s)
        self.ema = center_error if self.ema is None else a * center_error + (1 - a) * self.ema
        return self.ema, 'OK'

class CameraProcessing(Node):
    def __init__(self):
        super().__init__('camera_processing')
        self.get_logger().info('📷 [Camera Processing] 차선 인식 & 객체 거리(m) 추정 노드 시작')

        self.bridge = CvBridge()
        
        # --- 1. ROS 2 파라미터 설정 ---
        self.declare_parameter('yolo_model_path', 'yolov8n.engine')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('meters_per_pixel', 0.003) # 1픽셀 당 실제 거리(m)
        
        # --- 2. 안정화기(Stabilizer) & 공유 변수 초기화 ---
        self.prev_time = time.time()
        self.stabilizer = _Stabilizer(
            ema_tau_s=0.065,
            outlier_jump=80.0,
            outlier_relatch_s=0.16,
            lost_stop_s=0.26
        )
        
        # BEV 변환 행렬을 클래스 변수로 공유 (YOLO에서도 사용)
        self.bev_matrix = None

        # --- 3. YOLO 모델 로드 ---
        model_path = self.get_parameter('yolo_model_path').value
        try:
            self.yolo_model = YOLO(model_path)
            self.get_logger().info(f'✅ YOLO 모델 로드 완료: {model_path}')
        except Exception as e:
            self.get_logger().error(f'❌ YOLO 모델 로드 실패: {e}')

        # --- 4. ROS 2 퍼블리셔 / 서브스크라이버 ---
        self.sub_image = self.create_subscription(Image, 'vtd/camera_image', self.image_callback, 10)
        self.pub_lane_offset = self.create_publisher(Float32, '/perception/lane_offset', 10)
        self.pub_objects = self.create_publisher(String, '/perception/camera_objects', 10)

    def process_lane_detection(self, cv_image, dt_s):
        h, w = cv_image.shape[:2]
        
        # [1단계] 색상 마스킹
        img_hls = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HLS)
        img_lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)

        yellow_mask = cv2.inRange(img_lab, np.array([0, 130, 140]), np.array([255, 170, 255]))
        white_mask = cv2.inRange(img_hls, np.array([0, 200, 0]), np.array([255, 255, 255]))
        color_mask = cv2.bitwise_or(yellow_mask, white_mask)

        # [2단계] BEV 시점 변환 (행렬을 self.bev_matrix에 저장하여 공유!)
        src_points = np.float32([[w*0.4, h*0.6], [w*0.6, h*0.6], [w*0.1, h], [w*0.9, h]])
        dst_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        
        self.bev_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        bev_img = cv2.warpPerspective(color_mask, self.bev_matrix, (w, h))

        # [3단계] 모폴로지
        kv = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
        bev_img = cv2.morphologyEx(bev_img, cv2.MORPH_CLOSE, kv)

        # [4단계] 히스토그램 & 슬라이딩 윈도우
        histogram = np.sum(bev_img[int(h/2):, :], axis=0)
        midpoint = int(histogram.shape[0] / 2)
        
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        nwindows = 9
        window_height = int(h / nwindows)
        margin, minpix = 80, 40
        sw_dir_ema, sw_max_miss = 0.6, 3

        nonzero = bev_img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        def find_lane_pixels(x_base):
            lane_inds = []
            cur_x = float(x_base)
            step = 0.0
            prev_cx = None
            miss, hits = 0, 0

            for window in range(nwindows):
                win_y_low = h - (window + 1) * window_height
                win_y_high = h - window * window_height
                win_x_low = int(cur_x) - margin
                win_x_high = int(cur_x) + margin

                good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                             (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
                
                if len(good_inds) > minpix:
                    lane_inds.append(good_inds)
                    mx = float(np.mean(nonzerox[good_inds]))
                    if prev_cx is not None:
                        obs = max(-margin, min(margin, mx - prev_cx))
                        step = sw_dir_ema * obs + (1 - sw_dir_ema) * step
                    prev_cx = mx
                    cur_x = mx + step
                    miss = 0
                    hits += 1
                else:
                    cur_x += step
                    if hits > 0:
                        miss += 1
                        if miss >= sw_max_miss:
                            break
                            
            if not lane_inds:
                return [], []
            inds = np.concatenate(lane_inds)
            return nonzerox[inds], nonzeroy[inds]

        leftx, lefty = find_lane_pixels(leftx_base)
        rightx, righty = find_lane_pixels(rightx_base)

        # [5단계] 차선 중심 도출
        y_eval = h 
        lane_center = None
        is_valid = False

        if len(leftx) > 100 and len(rightx) > 100:
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            left_x = left_fit[0] * (y_eval**2) + left_fit[1] * y_eval + left_fit[2]
            right_x = right_fit[0] * (y_eval**2) + right_fit[1] * y_eval + right_fit[2]
            lane_center = (left_x + right_x) / 2.0
            is_valid = True
        elif len(leftx) > 100:
            left_fit = np.polyfit(lefty, leftx, 2)
            left_x = left_fit[0] * (y_eval**2) + left_fit[1] * y_eval + left_fit[2]
            lane_center = left_x + (w * 0.25)
            is_valid = True
        elif len(rightx) > 100:
            right_fit = np.polyfit(righty, rightx, 2)
            right_x = right_fit[0] * (y_eval**2) + right_fit[1] * y_eval + right_fit[2]
            lane_center = right_x - (w * 0.25)
            is_valid = True

        car_center = w / 2.0
        pixel_offset = (lane_center - car_center) if is_valid else None
        
        ema_pixel_offset, state = self.stabilizer.update(pixel_offset, is_valid, dt_s)
        
        if ema_pixel_offset is not None:
            m_per_pix = self.get_parameter('meters_per_pixel').value
            return ema_pixel_offset * m_per_pix
        return 0.0

    def process_yolo_detection(self, cv_image):
        if self.bev_matrix is None:
            return "NONE"

        h, w = cv_image.shape[:2]
        conf_thresh = self.get_parameter('conf_threshold').value
        m_per_pix = self.get_parameter('meters_per_pixel').value
        
        results = self.yolo_model.predict(source=cv_image, conf=conf_thresh, verbose=False)
        
        detected_objects = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0])
                class_name = self.yolo_model.names[cls_id]
                
                # 📌 [핵심 반영 1] 바운딩 박스의 맨 밑 중앙 점(바닥점) 추출
                bottom_x = (x1 + x2) / 2.0
                bottom_y = y2
                
                # 📌 [핵심 반영 2] 바닥점 1개만 BEV 행렬로 좌표 변환
                pt = np.array([[[bottom_x, bottom_y]]], dtype=np.float32)
                bev_pt = cv2.perspectiveTransform(pt, self.bev_matrix)[0][0]
                
                # 📌 [핵심 반영 3] BEV 좌표를 차량 기준 실제 거리(m)로 환산
                # bev_pt[0]: BEV 상의 X 좌표, bev_pt[1]: BEV 상의 Y 좌표
                real_dist_x = (bev_pt[0] - (w / 2.0)) * m_per_pix  # 좌우 거리 (m, +:우, -:좌)
                real_dist_y = (h - bev_pt[1]) * m_per_pix          # 전방 거리 (m)
                
                # 포맷: 이름:[좌우m, 전방m] (예: car:[0.50m, 12.30m])
                detected_objects.append(f"{class_name}:[{real_dist_x:.2f}m,{real_dist_y:.2f}m]")
                
        return "|".join(detected_objects)

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CvBridge 변환 오류: {e}')
            return

        current_time = time.time()
        dt_s = current_time - self.prev_time
        self.prev_time = current_time

        # [1] 차선 인식 (이 과정에서 self.bev_matrix가 갱신됨)
        offset = self.process_lane_detection(cv_image, dt_s)
        msg_offset = Float32()
        msg_offset.data = float(offset)
        self.pub_lane_offset.publish(msg_offset)

        # [2] 객체 탐지 및 물리 거리(m) 퍼블리시
        objects_str = self.process_yolo_detection(cv_image)
        msg_objects = String()
        msg_objects.data = objects_str if objects_str else "NONE"
        self.pub_objects.publish(msg_objects)

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