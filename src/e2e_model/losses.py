import torch
import torch.nn as nn
import torch.nn.functional as F
from e2e_model.config import CFG


class E2ELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, gt_waypoints, gt_controls, gt_situation):
        wp_loss = F.l1_loss(pred["waypoints"], gt_waypoints)
        ctrl_loss = F.mse_loss(pred["controls"], gt_controls)
        sit_loss = self.ce(pred["situation"], gt_situation)
        total = (CFG.lambda_waypoint * wp_loss
                 + CFG.lambda_control * ctrl_loss
                 + CFG.lambda_situation * sit_loss)
        return {"total": total, "waypoint": wp_loss, "control": ctrl_loss, "situation": sit_loss}
