import os, glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from e2e_model.config import CFG


class E2EDataset(Dataset):
    def __init__(self, split: str = "train"):
        pattern = os.path.join(CFG.data_root, split, "*.npz")
        self.files = sorted(glob.glob(pattern))
        if not self.files:
            raise FileNotFoundError(f"데이터 없음: {pattern}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        rgb = torch.from_numpy(data["rgb"]).permute(2, 0, 1).float() / 255.0
        rgb = TF.resize(rgb, [CFG.img_h, CFG.img_w])
        rgb = TF.normalize(rgb, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        bev = torch.from_numpy(data["bev"]).permute(2, 0, 1).float()
        bev = TF.resize(bev, [CFG.bev_h, CFG.bev_w])
        return {
            "rgb": rgb,
            "bev": bev,
            "speed": torch.tensor([data["speed"]], dtype=torch.float32),
            "waypoints": torch.from_numpy(data["waypoints"]).float(),
            "controls": torch.from_numpy(data["controls"]).float(),
            "situation": torch.tensor(int(data["situation"]), dtype=torch.long),
        }
