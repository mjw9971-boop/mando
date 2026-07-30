from dataclasses import dataclass

@dataclass
class ModelConfig:
    # 입력
    img_h: int = 256
    img_w: int = 256
    bev_h: int = 256
    bev_w: int = 256
    bev_channels: int = 2

    # backbone
    rgb_backbone: str = "resnet34"
    bev_backbone: str = "resnet18"
    fusion_dim: int = 512
    num_heads: int = 8
    num_layers: int = 3

    # TransFuser: waypoint decoder
    waypoint_steps: int = 4
    gru_hidden: int = 256

    # TCP: control head
    control_steps: int = 4
    speed_input_dim: int = 1
    situation_classes: int = 4
    control_dim: int = 3

    # loss 가중치
    lambda_waypoint: float = 1.0
    lambda_control: float = 1.0
    lambda_situation: float = 0.5

    # 학습
    batch_size: int = 16
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 100
    num_workers: int = 4
    checkpoint_dir: str = "checkpoints"

    # 데이터
    data_root: str = "dataset_out"

CFG = ModelConfig()
