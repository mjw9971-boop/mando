import torch
import torch.nn as nn
from e2e_model.models.backbone import SharedBackbone
from e2e_model.models.waypoint_decoder import WaypointDecoder
from e2e_model.models.control_head import ControlHead


class E2EModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = SharedBackbone()
        self.wp_decoder = WaypointDecoder()
        self.ctrl_head = ControlHead()

    def forward(self, rgb, bev, speed):
        feat = self.backbone(rgb, bev)
        waypoints = self.wp_decoder(feat, speed)
        controls, situation = self.ctrl_head(feat, waypoints, speed)
        return {"waypoints": waypoints, "controls": controls, "situation": situation}

    @classmethod
    def load(cls, ckpt_path: str, device: str = "cpu"):
        model = cls().to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
        model.eval()
        return model
