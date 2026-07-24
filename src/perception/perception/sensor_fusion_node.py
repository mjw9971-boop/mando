import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64
 
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
 
        # 1. 자차 속도 수신 (m/s 단위)
        self.ego_speed = 0.0
        self.sub_ego_speed = self.create_subscription(
            Float64, '/vtd/ego_speed', self.ego_speed_callback, 10
        )
 
        # 2. 레이더 & 카메라 & 라이다 인지 결과 수신
        self.radar_target = None
        self.camera_objects = []
        self.lidar_objects = []  # ✅ [추가]
 
        self.sub_radar = self.create_subscription(
            String, '/perception/radar_target', self.radar_callback, 10
        )
        # 카메라 노드 결과: [{"class": "car", "distance": 30.5, ...}, ...]
        self.sub_camera = self.create_subscription(
            String, '/perception/camera_objects', self.camera_callback, 10
        )
        # ✅ [추가] 라이다 노드 결과: [{"class": "unknown", "distance": 30.5, ...}, ...]
        self.sub_lidar = self.create_subscription(
            String, '/perception/lidar_objects', self.lidar_callback, 10
        )
 
        # 3. 제어기(ACC/AEB)로 보낼 최종 퓨전 타겟 퍼블리셔
        self.pub_fusion_target = self.create_publisher(String, '/control/fusion_target', 10)
 
        # 20Hz 주기로 퓨전 로직 실행
        self.create_timer(0.05, self.fusion_process)
 
        self.get_logger().info('🧠 센서 퓨전 노드 실행 완료! (레이더 + 카메라 + 라이다 교차 검증 중)')
 
    def ego_speed_callback(self, msg):
        self.ego_speed = msg.data
 
    def camera_callback(self, msg):
        try:
            self.camera_objects = json.loads(msg.data)
        except Exception:
            self.camera_objects = []
 
    def lidar_callback(self, msg):
        # ✅ [추가]
        try:
            self.lidar_objects = json.loads(msg.data)
        except Exception:
            self.lidar_objects = []
 
    def radar_callback(self, msg):
        try:
            self.radar_target = json.loads(msg.data)
        except Exception:
            self.radar_target = None
 
    def fusion_process(self):
        if not self.radar_target or not self.radar_target.get('target_detected'):
            self.publish_empty()
            return
 
        r_dist = self.radar_target['distance']
        r_vel = self.radar_target['relative_velocity']
        ttc = self.radar_target.get('ttc', 999.0)
 
        # -------------------------------------------------------------
        # [핵심 로직] 타겟이 이동 중인가? 멈춰있는가?
        # -------------------------------------------------------------
        is_stationary = abs(r_vel - (-self.ego_speed)) < 1.5
 
        final_status = "UNKNOWN"
        is_valid_target = False
 
        if not is_stationary:
            # 케이스 1. 이동 중인 물체 (앞차) -> 레이더 정보 신뢰 (정상 추종)
            final_status = "MOVING_VEHICLE"
            is_valid_target = True
        else:
            # 케이스 2. 정지 물체 -> 카메라 및/또는 라이다와 교차 검증
            confirming_sources = []
 
            # 카메라 교차검증 (클래스가 차량류이고 거리가 5m 이내로 비슷할 때)
            for obj in self.camera_objects:
                if obj.get('class') in ['car', 'truck', 'bus']:
                    c_dist = obj.get('distance', 0)
                    if abs(c_dist - r_dist) < 5.0:
                        confirming_sources.append('camera')
                        break
 
            # ✅ [추가] 라이다 교차검증 (라이다는 클래스 분류가 없으므로 거리만으로 판단)
            for obj in self.lidar_objects:
                l_dist = obj.get('distance', 0)
                if abs(l_dist - r_dist) < 5.0:
                    confirming_sources.append('lidar')
                    break
 
            if confirming_sources:
                # 카메라 또는 라이다 중 하나라도 확인되면 정지 차량으로 확정
                # (두 센서 모두 확인되면 신뢰도가 더 높음 → status에 소스 개수 표기)
                final_status = "STATIONARY_VEHICLE"
                is_valid_target = True
            else:
                # 어떤 센서도 확인 못 함 -> 가드레일/표지판 등 노이즈로 판단
                final_status = "GUARDRAIL_IGNORED"
                is_valid_target = False
 
        # -------------------------------------------------------------
        # 퓨전 결과 퍼블리시
        # -------------------------------------------------------------
        if is_valid_target:
            fusion_result = {
                'target_valid': True,
                'distance': r_dist,
                'relative_velocity': r_vel,
                'ttc': ttc,
                'status': final_status,
                'confirmed_by': confirming_sources if is_stationary else ['radar'],  # ✅ [추가] 어떤 센서로 확정됐는지 기록
            }
            self.get_logger().info(f"🚨 [퓨전] {final_status} 확정! 거리: {r_dist}m, TTC: {ttc}초")
        else:
            fusion_result = {
                'target_valid': False,
                'status': final_status
            }
            self.get_logger().debug(f"🚧 [퓨전] {final_status} 무시됨. (거리: {r_dist}m)")
 
        msg_out = String()
        msg_out.data = json.dumps(fusion_result)
        self.pub_fusion_target.publish(msg_out)
 
    def publish_empty(self):
        msg_out = String()
        msg_out.data = json.dumps({'target_valid': False, 'status': "CLEAR"})
        self.pub_fusion_target.publish(msg_out)
 
def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
 
if __name__ == '__main__':
    main()
 