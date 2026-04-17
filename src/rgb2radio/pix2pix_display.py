from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import cosine_window, ensure_dir, read_rgb, resolve_existing_path, save_gray, save_json


@dataclass(frozen=True)
class Pix2PixConfig:
    patch_size: int = 128
    batch_size: int = 4
    epochs: int = 24
    learning_rate: float = 2e-4
    lambda_l1: float = 40.0
    lambda_edge: float = 10.0
    lambda_adv: float = 0.6
    seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True)
class Pix2PixResult:
    model_path: Path
    output_path: Path
    report_path: Path
    best_epoch: int
    best_masked_ncc: float
    best_masked_mae: float
    history: List[Dict[str, float]]


def _read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read grayscale image: {path}")
    return image


def _masked_metrics(prediction_u8: np.ndarray, reference_u8: np.ndarray) -> Dict[str, float]:
    mask = reference_u8 > max(12, int(np.quantile(reference_u8, 0.10)))
    if int(mask.sum()) == 0:
        return {"masked_mae": 0.0, "masked_ncc": 0.0}
    pred = prediction_u8[mask].astype(np.float32) / 255.0
    ref = reference_u8[mask].astype(np.float32) / 255.0
    mae = float(np.mean(np.abs(pred - ref)))
    pred_centered = pred - float(np.mean(pred))
    ref_centered = ref - float(np.mean(ref))
    denominator = float(np.linalg.norm(pred_centered) * np.linalg.norm(ref_centered))
    ncc = float(np.dot(pred_centered, ref_centered) / denominator) if denominator > 1e-8 else 0.0
    return {"masked_mae": mae, "masked_ncc": ncc}


def _edge_map(tensor: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=tensor.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], device=tensor.device).view(1, 1, 3, 3)
    grad_x = torch.nn.functional.conv2d(tensor, sobel_x, padding=1)
    grad_y = torch.nn.functional.conv2d(tensor, sobel_y, padding=1)
    return torch.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-6)


class PairedDisplayDataset(Dataset):
    def __init__(
        self,
        real_chunks_dir: Path,
        metadata_path: Path,
        split: str,
        synth_full_path: Path,
    ) -> None:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.entries = [entry for entry in metadata if entry["split"] == split]
        self.real_optical_dir = real_chunks_dir / split / "optical"
        self.real_radar_dir = real_chunks_dir / split / "radar"
        self.synth_full = _read_gray(synth_full_path).astype(np.float32) / 255.0
        self.full_height, self.full_width = self.synth_full.shape

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        file_name = f"{index:05d}.png"
        entry = self.entries[index]
        x = int(entry["x"])
        y = int(entry["y"])
        size = int(entry["size"])
        optical = read_rgb(self.real_optical_dir / file_name).astype(np.float32) / 255.0
        radar = _read_gray(self.real_radar_dir / file_name).astype(np.float32) / 255.0
        synth = self.synth_full[y : y + size, x : x + size]
        if synth.shape[0] != size or synth.shape[1] != size:
            synth = np.zeros((size, size), dtype=np.float32)
        yy, xx = np.mgrid[y : y + size, x : x + size].astype(np.float32)
        coord_x = xx / max(float(self.full_width - 1), 1.0)
        coord_y = yy / max(float(self.full_height - 1), 1.0)

        input_tensor = np.concatenate([optical, synth[:, :, None], coord_x[:, :, None], coord_y[:, :, None]], axis=2)
        input_tensor = np.transpose(input_tensor, (2, 0, 1)).astype(np.float32)
        target_tensor = radar[None, :, :].astype(np.float32)

        return {
            "input": torch.from_numpy(input_tensor),
            "target": torch.from_numpy(target_tensor),
        }


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, normalize: bool, activation: str) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        if activation == "leaky":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        else:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels: int = 6, base_channels: int = 32) -> None:
        super().__init__()
        self.down1 = ConvBlock(in_channels, base_channels, stride=2, normalize=False, activation="leaky")
        self.down2 = ConvBlock(base_channels, base_channels * 2, stride=2, normalize=True, activation="leaky")
        self.down3 = ConvBlock(base_channels * 2, base_channels * 4, stride=2, normalize=True, activation="leaky")
        self.down4 = ConvBlock(base_channels * 4, base_channels * 8, stride=2, normalize=True, activation="leaky")
        self.bottleneck = ConvBlock(base_channels * 8, base_channels * 8, stride=2, normalize=False, activation="leaky")

        self.up4 = UpBlock(base_channels * 8, base_channels * 8, dropout=0.2)
        self.up3 = UpBlock(base_channels * 16, base_channels * 4)
        self.up2 = UpBlock(base_channels * 8, base_channels * 2)
        self.up1 = UpBlock(base_channels * 4, base_channels)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        bottleneck = self.bottleneck(d4)

        u4 = self.up4(bottleneck)
        u3 = self.up3(torch.cat([u4, d4], dim=1))
        u2 = self.up2(torch.cat([u3, d3], dim=1))
        u1 = self.up1(torch.cat([u2, d2], dim=1))
        return self.final(torch.cat([u1, d1], dim=1))


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 7, base_channels: int = 32) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(base_channels * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def _reconstruct_full_display(
    generator: UNetGenerator,
    metadata_path: Path,
    optical_full_path: Path,
    synth_full_path: Path,
    device: torch.device,
) -> np.ndarray:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    optical_full = read_rgb(optical_full_path).astype(np.float32) / 255.0
    synth_full = _read_gray(synth_full_path).astype(np.float32) / 255.0
    height, width = synth_full.shape
    accum = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)

    generator.eval()
    with torch.no_grad():
        for entry in metadata:
            x = int(entry["x"])
            y = int(entry["y"])
            size = int(entry["size"])
            optical_patch = optical_full[y : y + size, x : x + size]
            synth_patch = synth_full[y : y + size, x : x + size]
            if optical_patch.shape[0] != size or optical_patch.shape[1] != size:
                continue
            yy, xx = np.mgrid[y : y + size, x : x + size].astype(np.float32)
            coord_x = xx / max(float(width - 1), 1.0)
            coord_y = yy / max(float(height - 1), 1.0)
            input_patch = np.concatenate(
                [optical_patch, synth_patch[:, :, None], coord_x[:, :, None], coord_y[:, :, None]],
                axis=2,
            )
            input_tensor = torch.from_numpy(np.transpose(input_patch, (2, 0, 1)).astype(np.float32))[None, :, :, :].to(device)
            prediction = generator(input_tensor).cpu().numpy()[0, 0]
            window = cosine_window(size)
            accum[y : y + size, x : x + size] += prediction.astype(np.float32) * window
            weights[y : y + size, x : x + size] += window

    reconstructed = accum / np.maximum(weights, 1e-6)
    return (np.clip(reconstructed, 0.0, 1.0) * 255.0).astype(np.uint8)


def train_pix2pix_display(
    output_dir: str | Path,
    config: Pix2PixConfig | None = None,
) -> Pix2PixResult:
    if config is None:
        config = Pix2PixConfig()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    output_dir = ensure_dir(output_dir)

    real_chunks_dir = resolve_existing_path(
        [
            Path("outputs/chunks/paired_display_real"),
        ]
    )
    metadata_path = real_chunks_dir / "metadata.json"
    optical_full_path = resolve_existing_path([Path("outputs/run/projected_display_optical_rgb.png")])
    synth_full_path = resolve_existing_path([Path("outputs/run/projected_display_gray.png")])
    reference_full_path = resolve_existing_path([Path("outputs/run/reference_clean_gray.png")])

    train_dataset = PairedDisplayDataset(
        real_chunks_dir=real_chunks_dir,
        metadata_path=metadata_path,
        split="train",
        synth_full_path=synth_full_path,
    )
    val_dataset = PairedDisplayDataset(
        real_chunks_dir=real_chunks_dir,
        metadata_path=metadata_path,
        split="val",
        synth_full_path=synth_full_path,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
    adversarial_loss = nn.BCEWithLogitsLoss()

    history: list[Dict[str, float]] = []
    best_prediction: np.ndarray | None = None
    best_epoch = 0
    best_masked_ncc = -1e9
    best_masked_mae = 1e9
    best_model_path = output_dir / "pix2pix_display_best.pt"

    reference_full = _read_gray(reference_full_path)

    for epoch in range(1, config.epochs + 1):
        generator.train()
        discriminator.train()
        running_g = 0.0
        running_d = 0.0

        for batch in train_loader:
            input_tensor = batch["input"].to(device)
            target_tensor = batch["target"].to(device)

            fake_tensor = generator(input_tensor)

            optimizer_d.zero_grad(set_to_none=True)
            real_logits = discriminator(torch.cat([input_tensor, target_tensor], dim=1))
            fake_logits = discriminator(torch.cat([input_tensor, fake_tensor.detach()], dim=1))
            loss_d = 0.5 * (
                adversarial_loss(real_logits, torch.ones_like(real_logits))
                + adversarial_loss(fake_logits, torch.zeros_like(fake_logits))
            )
            loss_d.backward()
            optimizer_d.step()

            optimizer_g.zero_grad(set_to_none=True)
            fake_logits = discriminator(torch.cat([input_tensor, fake_tensor], dim=1))
            loss_adv = adversarial_loss(fake_logits, torch.ones_like(fake_logits))
            weight_map = 1.0 + 4.0 * target_tensor
            loss_l1 = torch.mean(torch.abs(fake_tensor - target_tensor) * weight_map)
            loss_edge = torch.mean(torch.abs(_edge_map(fake_tensor) - _edge_map(target_tensor)))
            loss_g = config.lambda_adv * loss_adv + config.lambda_l1 * loss_l1 + config.lambda_edge * loss_edge
            loss_g.backward()
            optimizer_g.step()

            running_d += float(loss_d.item())
            running_g += float(loss_g.item())

        generator.eval()
        val_l1 = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input_tensor = batch["input"].to(device)
                target_tensor = batch["target"].to(device)
                fake_tensor = generator(input_tensor)
                val_l1 += float(torch.mean(torch.abs(fake_tensor - target_tensor)).item())
        val_l1 /= max(len(val_loader), 1)

        reconstructed = _reconstruct_full_display(
            generator=generator,
            metadata_path=metadata_path,
            optical_full_path=optical_full_path,
            synth_full_path=synth_full_path,
            device=device,
        )
        full_metrics = _masked_metrics(reconstructed, reference_full)
        history.append(
            {
                "epoch": float(epoch),
                "train_g_loss": running_g / max(len(train_loader), 1),
                "train_d_loss": running_d / max(len(train_loader), 1),
                "val_l1": val_l1,
                "masked_mae": float(full_metrics["masked_mae"]),
                "masked_ncc": float(full_metrics["masked_ncc"]),
            }
        )

        if full_metrics["masked_ncc"] > best_masked_ncc:
            best_masked_ncc = float(full_metrics["masked_ncc"])
            best_masked_mae = float(full_metrics["masked_mae"])
            best_epoch = epoch
            best_prediction = reconstructed.copy()
            torch.save(
                {
                    "generator": generator.state_dict(),
                    "config": config.__dict__,
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_masked_ncc": best_masked_ncc,
                    "best_masked_mae": best_masked_mae,
                },
                best_model_path,
            )

    if best_prediction is None:
        best_prediction = _reconstruct_full_display(
            generator=generator,
            metadata_path=metadata_path,
            optical_full_path=optical_full_path,
            synth_full_path=synth_full_path,
            device=device,
        )

    output_path = output_dir / "pix2pix_display_gray.png"
    report_path = output_dir / "pix2pix_report.json"
    save_gray(output_path, best_prediction)
    save_json(
        report_path,
        {
            "best_epoch": best_epoch,
            "best_masked_ncc": best_masked_ncc,
            "best_masked_mae": best_masked_mae,
            "history": history,
            "config": config.__dict__,
            "model_path": str(best_model_path),
            "output_path": str(output_path),
        },
    )
    return Pix2PixResult(
        model_path=best_model_path,
        output_path=output_path,
        report_path=report_path,
        best_epoch=best_epoch,
        best_masked_ncc=best_masked_ncc,
        best_masked_mae=best_masked_mae,
        history=history,
    )
