import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # 1. 자차 속도 수신 (m/s 단위라고 가정)
        self.ego_speed = 0.0
        self.sub_ego_speed = self.create_subscription(
            Float64, '/vtd/ego_speed', self.ego_speed_callback, 10
        )

        # 2. 레이더 & 카메라 인지 결과 수신
        self.radar_target = None
        self.camera_objects = []
        
        # 앞서 만든 레이더 노드의 결과
        self.sub_radar = self.create_subscription(
            String, '/perception/radar_target', self.radar_callback, 10
        )
        # 카메라 노드 결과 (예: [{"class": "car", "distance": 30.5}, ...])
        self.sub_camera = self.create_subscription(
            String, '/perception/camera_objects', self.camera_callback, 10
        )

        # 3. 제어기(ACC/AEB)로 보낼 최종 퓨전 타겟 퍼블리셔
        self.pub_fusion_target = self.create_publisher(String, '/control/fusion_target', 10)

        # 20Hz 주기로 퓨전 로직 실행
        self.create_timer(0.05, self.fusion_process)
        
        self.get_logger().info('🧠 센서 퓨전 노드 실행 완료! (카메라 + 레이더 교차 검증 중)')

    def ego_speed_callback(self, msg):
        self.ego_speed = msg.data

    def camera_callback(self, msg):
        # 카메라가 인식한 객체 리스트 파싱
        try:
            self.camera_objects = json.loads(msg.data)
        except Exception:
            self.camera_objects = []

    def radar_callback(self, msg):
        # 레이더 타겟 정보 파싱
        try:
            self.radar_target = json.loads(msg.data)
        except Exception:
            self.radar_target = None

    def fusion_process(self):
        # 레이더 타겟이 아예 없으면 퓨전할 것도 없으므로 종료
        if not self.radar_target or not self.radar_target.get('target_detected'):
            self.publish_empty()
            return

        r_dist = self.radar_target['distance']
        r_vel = self.radar_target['relative_velocity']
        ttc = self.radar_target.get('ttc', 999.0)

        # -------------------------------------------------------------
        # [핵심 로직] 타겟이 이동 중인가? 멈춰있는가?
        # 자차 속도의 역방향(-ego_speed)과 상대 속도가 오차 범위(예: 1.5m/s) 내로 비슷하면 정지 물체!
        # -------------------------------------------------------------
        is_stationary = abs(r_vel - (-self.ego_speed)) < 1.5 
        
        final_status = "UNKNOWN"
        is_valid_target = False

        if not is_stationary:
            # 케이스 1. 이동 중인 물체 (앞차) -> 레이더 정보 신뢰 (정상 추종)
            final_status = "MOVING_VEHICLE"
            is_valid_target = True
        else:
            # 케이스 2. 정지 물체 -> 카메라 데이터와 교차 검증 (퓨전)
            camera_confirmed = False
            
            for obj in self.camera_objects:
                # 카메라가 '차량류'로 인식했고
                if obj.get('class') in ['car', 'truck', 'bus']:
                    # 카메라 객체 거리와 레이더 거리가 5m 이내로 비슷하다면 동일 물체로 간주!
                    c_dist = obj.get('distance', 0)
                    if abs(c_dist - r_dist) < 5.0:
                        camera_confirmed = True
                        break
            
            if camera_confirmed:
                # 카메라에도 차로 보이고, 레이더에도 멈춰있다고 뜬다 -> 주정차 차량!
                final_status = "STATIONARY_VEHICLE" 
                is_valid_target = True
            else:
                # 카메라에는 차가 안 보이는데 레이더만 멈춰있다고 뜬다 -> 가드레일/표지판 노이즈!
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
                'status': final_status
            }
            self.get_logger().info(f"🚨 [퓨전] {final_status} 확정! 거리: {r_dist}m, TTC: {ttc}초")
        else:
            fusion_result = {
                'target_valid': False,
                'status': final_status
            }
            # 무시된 가드레일은 화면에 도배되지 않도록 debug 로그로만 출력
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