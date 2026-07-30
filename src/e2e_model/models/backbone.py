import torch
import torch.nn as nn
import torchvision.models as tvm
from e2e_model.config import CFG


def _make_resnet(name: str, pretrained: bool = False) -> nn.Module:
    weights = "IMAGENET1K_V1" if pretrained else None
    model = getattr(tvm, name)(weights=weights)
    return nn.Sequential(*list(model.children())[:-2])


class SharedBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        self.rgb_enc = _make_resnet(CFG.rgb_backbone, pretrained=True)
        rgb_feat_dim = 512 if any(x in CFG.rgb_backbone for x in ["18","34"]) else 2048

        self.bev_enc = _make_resnet(CFG.bev_backbone, pretrained=False)
        if CFG.bev_channels != 3:
            orig = self.bev_enc[0]
            self.bev_enc[0] = nn.Conv2d(
                CFG.bev_channels, orig.out_channels,
                kernel_size=orig.kernel_size,
                stride=orig.stride,
                padding=orig.padding,
                bias=False,
            )
        bev_feat_dim = 512

        self.rgb_proj = nn.Conv2d(rgb_feat_dim, CFG.fusion_dim // 2, 1)
        self.bev_proj = nn.Conv2d(bev_feat_dim, CFG.fusion_dim // 2, 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=CFG.fusion_dim // 2,
            nhead=CFG.num_heads // 2,
            dim_feedforward=CFG.fusion_dim * 2,
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=CFG.num_layers)
        self.out_proj = nn.Linear(CFG.fusion_dim // 2, CFG.fusion_dim)

    def forward(self, rgb: torch.Tensor, bev: torch.Tensor) -> torch.Tensor:
        f_rgb = self.rgb_proj(self.rgb_enc(rgb))
        f_bev = self.bev_proj(self.bev_enc(bev))
        B, D, h, w = f_rgb.shape
        t_rgb = f_rgb.flatten(2).permute(0, 2, 1)
        t_bev = f_bev.flatten(2).permute(0, 2, 1)
        tokens = torch.cat([t_rgb, t_bev], dim=1)
        fused = self.transformer(tokens)
        feat = fused.mean(dim=1)
        return self.out_proj(feat)
