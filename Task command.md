



* `won` : 프로젝트 전체 한번에 빌드 (`cd ~/mando && colcon build --symlink-install ...`)

---

### 🚀 1. 전체 시스템 한 번에 실행 (런치 파일)
* `ros2 launch vtd_bridge vtd_bridge.launch.py` : VTD 브릿지 및 연결 노드 일괄 실행

---

### 🔧 2. 개별 노드 수동 실행 (디버깅용)
* `ros2 run perception radar_processing` : 레이더 인지 노드 실행
* `ros2 run perception camera_processing` : 카메라 인지 노드 실행
* `ros2 run perception lidar_processing` : 라이다 인지 노드 실행
* `ros2 run planning path_planner` : 판단/경로 노드 실행
* `ros2 run control vehicle_control` : 차량 제어 노드 실행