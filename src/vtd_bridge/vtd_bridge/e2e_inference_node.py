"""
E2E 추론 ROS 노드
subscribe: /rdb/rgb, /rdb/bev, /rdb/speed
publish  : /scp/control (Twist), /scp/waypoints (Path)
"""
import sys, os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path, PoseStamped
from cv_bridge import CvBridge
import numpy as np
import torch
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from e2e_model.models.e2e_model import E2EModel
from e2e_model.config import CFG


class E2EInferenceNode(Node):
    def __init__(self):
        super().__init__("e2e_inference_node")
        self.declare_parameter("checkpoint", "checkpoints/best.pth")
        self.declare_parameter("device", "cuda")
        ckpt = self.get_parameter("checkpoint").value
        dev  = self.get_parameter("device").value
        self.device = torch.device(dev if torch.cuda.is_available() else "cpu")
        self.model = E2EModel.load(ckpt, str(self.device))
        self.get_logger().info(f"모델 로드 완료 | device={self.device}")

        self.bridge = CvBridge()
        self._rgb = self._bev = None
        self._speed = 0.0

        self.create_subscription(Image,   "/rdb/rgb",   self._rgb_cb, 10)
        self.create_subscription(Image,   "/rdb/bev",   self._bev_cb, 10)
        self.create_subscription(Float32, "/rdb/speed", self._spd_cb, 10)
        self.pub_ctrl = self.create_publisher(Twist, "/scp/control",   10)
        self.pub_path = self.create_publisher(Path,  "/scp/waypoints", 10)
        self.create_timer(0.05, self._infer)  # 20 Hz

    def _rgb_cb(self, msg): self._rgb = self.bridge.imgmsg_to_cv2(msg, "rgb8")
    def _bev_cb(self, msg): self._bev = self.bridge.imgmsg_to_cv2(msg, "passthrough")
    def _spd_cb(self, msg): self._speed = msg.data

    def _infer(self):
        if self._rgb is None or self._bev is None:
            return
        rgb_t = self._prep_rgb(self._rgb)
        bev_t = self._prep_bev(self._bev)
        spd_t = torch.tensor([[self._speed]], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            out = self.model(rgb_t, bev_t, spd_t)
        ctrl = out["controls"][0, 0].cpu().numpy()
        twist = Twist()
        twist.angular.z = float(ctrl[0])   # steer
        twist.linear.x  = float(ctrl[1])   # throttle
        twist.linear.y  = float(ctrl[2])   # brake
        self.pub_ctrl.publish(twist)

        path = Path()
        path.header.stamp    = self.get_clock().now().to_msg()
        path.header.frame_id = "base_link"
        for wp in out["waypoints"][0].cpu().numpy():
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(wp[0])
            ps.pose.position.y = float(wp[1])
            path.poses.append(ps)
        self.pub_path.publish(path)

    def _prep_rgb(self, img):
        img = cv2.resize(img, (CFG.img_w, CFG.img_h)).astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        return torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)

    def _prep_bev(self, img):
        if img.ndim == 2: img = img[:,:,np.newaxis]
        img = cv2.resize(img, (CFG.bev_w, CFG.bev_h)).astype(np.float32)
        if img.ndim == 2: img = img[:,:,np.newaxis]
        return torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(self.device)


def main(args=None):
    rclpy.init(args=args)
    node = E2EInferenceNode()
    try:    rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
