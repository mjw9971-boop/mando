import os
import torch
from torch.utils.data import DataLoader
from e2e_model.config import CFG
from e2e_model.models.e2e_model import E2EModel
from e2e_model.data.dataset import E2EDataset
from e2e_model.losses import E2ELoss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_dl = DataLoader(E2EDataset("train"), batch_size=CFG.batch_size,
                          shuffle=True, num_workers=CFG.num_workers, pin_memory=True)
    val_dl = DataLoader(E2EDataset("val"), batch_size=CFG.batch_size,
                        shuffle=False, num_workers=CFG.num_workers)

    model = E2EModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
    criterion = E2ELoss().to(device)
    os.makedirs(CFG.checkpoint_dir, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, CFG.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            rgb = batch["rgb"].to(device)
            bev = batch["bev"].to(device)
            speed = batch["speed"].to(device)
            optimizer.zero_grad()
            pred = model(rgb, bev, speed)
            losses = criterion(pred, batch["waypoints"].to(device),
                               batch["controls"].to(device), batch["situation"].to(device))
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += losses["total"].item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_dl:
                pred = model(batch["rgb"].to(device), batch["bev"].to(device), batch["speed"].to(device))
                val_loss += criterion(pred, batch["waypoints"].to(device),
                                      batch["controls"].to(device), batch["situation"].to(device))["total"].item()

        train_loss /= len(train_dl)
        val_loss /= len(val_dl)
        scheduler.step()
        print(f"[{epoch:03d}/{CFG.epochs}] train={train_loss:.4f}  val={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       os.path.join(CFG.checkpoint_dir, "best.pth"))
        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model": model.state_dict()},
                       os.path.join(CFG.checkpoint_dir, f"epoch_{epoch:03d}.pth"))


if __name__ == "__main__":
    main()
