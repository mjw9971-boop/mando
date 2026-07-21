cat << 'EOF' > ~/mando/README.md
# 🚘 Mando Autonomous Driving ROS 2 Workspace

VTD(Virtual Test Drive) 시뮬레이터 연동 및 자율주행 모듈 프로젝트입니다.

---

## 🌳 Repository Structure

```text
mando/
├── Notice/             # 📢 대회 가이드, 규정집, 트랙 지도 등 보관 폴더
│   └── NOTICE_GUIDE.md
├── offline/            # 📦 오프라인 데이터 및 로그 보관 폴더
├── src/                # 📂 ROS 2 패키지 소스 코드
│   ├── vtd_bridge/     # 🌉 VTD <-> ROS 2 데이터 브릿지 (RDB Receiver, SCP Sender)
│   ├── perception/     # 👁️ 센서 인지 노드 (Camera, Lidar, Radar)
│   ├── planning/       # 🧠 경로 및 판단 노드 (Path Planner)
│   └── control/        # 🚘 차량 제어 노드 (Vehicle Control)
├── .gitignore          # Git 제외 설정 파일
├── NOTICE.md           # 팀 공지사항 및 규칙
├── README.md           # 프로젝트 메인 설명 문서
└── Task command.md     # 주요 빌드 및 실행 명령어 모음