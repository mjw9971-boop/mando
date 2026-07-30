import torch
import torch.nn as nn
from e2e_model.config import CFG


class ControlHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.situation_head = nn.Linear(CFG.fusion_dim, CFG.situation_classes)
        self.wp_proj = nn.Linear(2, CFG.fusion_dim)
        self.query_proj = nn.Linear(CFG.fusion_dim, CFG.fusion_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=CFG.fusion_dim,
            num_heads=CFG.num_heads,
            batch_first=True,
        )
        self.control_mlp = nn.Sequential(
            nn.Linear(CFG.fusion_dim + CFG.speed_input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, CFG.control_dim * CFG.control_steps),
        )

    def forward(self, feat, waypoints, speed):
        B = feat.size(0)
        situation_logit = self.situation_head(feat)
        kv = self.wp_proj(waypoints)
        q = self.query_proj(feat).unsqueeze(1)
        ctx, _ = self.attn(q, kv, kv)
        ctx = ctx.squeeze(1)
        ctrl_in = torch.cat([ctx, speed], dim=-1)
        ctrl_flat = self.control_mlp(ctrl_in)
        controls = torch.sigmoid(ctrl_flat.view(B, CFG.control_steps, CFG.control_dim))
        steer = controls[..., 0:1] * 2.0 - 1.0
        throttle_brake = controls[..., 1:]
        controls = torch.cat([steer, throttle_brake], dim=-1)
        return controls, situation_logit
