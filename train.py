import os
import math
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from dataset import RadarColorizationDataset
from model import UNetColorizer
from losses import L1PerceptualColorLoss


def seed_everything(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_finite_tensor(x: torch.Tensor) -> bool:
    return torch.isfinite(x).all().item()


@torch.no_grad()
def validate(model, loader, criterion, device, save_dir=None, max_save=8, use_amp=False):
    model.eval()

    total_loss = 0.0
    n = 0
    saved = 0

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    for batch in loader:
        x = batch["input"].to(device, non_blocking=True)
        y = batch["target"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            pred = model(x)

        # loss всегда считаем в float32
        pred_f = pred.float()
        y_f = y.float()
        loss, _ = criterion(pred_f, y_f)

        if not torch.isfinite(loss):
            print("[WARN] Non-finite loss detected in validation batch. Skipping batch.")
            continue

        total_loss += float(loss.item()) * x.size(0)
        n += x.size(0)

        if save_dir is not None and saved < max_save:
            pred_vis = torch.sigmoid(pred_f)
            y_vis = torch.clamp(y_f, 0.0, 1.0)
            x_vis = torch.clamp(x.float(), 0.0, 1.0)

            for i in range(x.size(0)):
                if saved >= max_save:
                    break

                radio_vis = x_vis[i].repeat(3, 1, 1)
                grid = torch.cat([radio_vis, pred_vis[i], y_vis[i]], dim=2)
                save_image(grid, os.path.join(save_dir, f"sample_{saved:03d}.png"))
                saved += 1

    if n == 0:
        return float("inf")

    return total_loss / n


def train(args):
    seed_everything(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = (device == "cuda") and (not args.no_amp)

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"AMP enabled: {use_amp}")

    train_ds = RadarColorizationDataset(
        root=args.train_dir,
        image_size=args.image_size,
        train=True,
    )
    val_ds = RadarColorizationDataset(
        root=args.val_dir,
        image_size=args.image_size,
        train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )

    model = UNetColorizer(
        in_channels=1,
        out_channels=3,
        base=args.base_channels,
    ).to(device)

    criterion = L1PerceptualColorLoss(
        l1_weight=args.l1_weight,
        perceptual_weight=args.perceptual_weight,
        color_weight=args.color_weight,
        resize_vgg=False,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    vis_dir = out_dir / "val_vis"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()

        running_loss = 0.0
        sample_count = 0
        skipped_batches = 0

        for batch_idx, batch in enumerate(train_loader, start=1):
            x = batch["input"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            if not is_finite_tensor(x) or not is_finite_tensor(y):
                print(f"[WARN] Non-finite input/target at batch {batch_idx}. Skipping.")
                skipped_batches += 1
                continue

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                pred = model(x)

            pred_f = pred.float()
            y_f = y.float()

            loss, metrics = criterion(pred_f, y_f)

            if not torch.isfinite(loss):
                print(
                    f"[WARN] Non-finite loss at epoch {epoch}, batch {batch_idx}. "
                    f"Skipping batch. "
                    f"L1={metrics.get('l1', 'nan')}, SSIM={metrics.get('ssim', 'nan')}"
                )
                skipped_batches += 1
                continue

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * x.size(0)
            sample_count += x.size(0)

        train_loss = running_loss / max(sample_count, 1)

        val_loss = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            save_dir=str(vis_dir / f"epoch_{epoch:03d}"),
            max_save=6,
            use_amp=use_amp,
        )

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"lr={current_lr:.7f} | "
            f"skipped={skipped_batches}"
        )

        last_ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
        }

        torch.save(last_ckpt, ckpt_dir / "last.pt")

        if math.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            torch.save(last_ckpt, ckpt_dir / "best.pt")
            print(f"Saved best checkpoint: val_loss={best_val:.5f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--out_dir", type=str, default="runs/exp1")

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--l1_weight", type=float, default=0.8)
    parser.add_argument("--perceptual_weight", type=float, default=0.3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--color_weight", type=float, default=0.7)

    args = parser.parse_args()
    train(args)