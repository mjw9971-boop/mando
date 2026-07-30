import torch
import torch.nn as nn
from e2e_model.config import CFG


class WaypointDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_dim = CFG.fusion_dim + CFG.speed_input_dim
        self.input_proj = nn.Linear(in_dim, CFG.gru_hidden)
        self.gru = nn.GRUCell(CFG.gru_hidden, CFG.gru_hidden)
        self.out_head = nn.Linear(CFG.gru_hidden, 2)

    def forward(self, feat: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
        x = torch.cat([feat, speed], dim=-1)
        h = torch.relu(self.input_proj(x))
        waypoints = []
        for _ in range(CFG.waypoint_steps):
            h = self.gru(h, h)
            waypoints.append(self.out_head(h))
        return torch.stack(waypoints, dim=1)
