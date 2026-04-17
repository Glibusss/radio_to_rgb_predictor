from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import (
    corner_response,
    cosine_window,
    gradient_magnitude,
    gray_from_rgb,
    iter_windows,
    local_variance,
    normalize01,
    normalize_to_uint8,
)

TERRAIN_CLASS_NAMES: Sequence[str] = (
    "forest",
    "water",
    "rippling_water",
    "concrete_building",
    "metal_building",
    "dirt_road",
    "asphalt_road",
    "wood_building",
)

TERRAIN_CLASS_COLORS = np.array(
    [
        [54, 128, 58],
        [48, 86, 164],
        [92, 168, 210],
        [182, 182, 188],
        [230, 232, 238],
        [160, 116, 66],
        [68, 68, 74],
        [146, 106, 80],
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class TerrainTrainingResult:
    model_path: Path
    patch_size: int
    train_samples: int
    val_samples: int
    best_val_accuracy: float
    class_histogram: Dict[str, int]
    history: List[Dict[str, float]]


def _wrapped_hue_response(hue: np.ndarray, center: float, spread: float) -> np.ndarray:
    direct = np.abs(hue - center)
    wrapped = np.minimum(direct, 1.0 - direct)
    return np.exp(-0.5 * np.square(wrapped / max(spread, 1e-3)))


def _structure_anisotropy(gray_u8: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    j_xx = cv2.GaussianBlur(grad_x * grad_x, (0, 0), sigmaX=2.2, sigmaY=2.2)
    j_xy = cv2.GaussianBlur(grad_x * grad_y, (0, 0), sigmaX=2.2, sigmaY=2.2)
    j_yy = cv2.GaussianBlur(grad_y * grad_y, (0, 0), sigmaX=2.2, sigmaY=2.2)
    trace = j_xx + j_yy
    det = j_xx * j_yy - j_xy * j_xy
    discriminant = np.maximum(trace * trace - 4.0 * det, 0.0)
    lambda1 = 0.5 * (trace + np.sqrt(discriminant))
    lambda2 = 0.5 * (trace - np.sqrt(discriminant))
    return normalize01((lambda1 - lambda2) / np.maximum(lambda1 + lambda2, 1e-6))


def build_heuristic_score_maps(rgb_image: np.ndarray) -> Dict[str, np.ndarray]:
    rgb_f32 = rgb_image.astype(np.float32) / 255.0
    red, green, blue = cv2.split(rgb_f32)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue = hsv[:, :, 0] / 179.0
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    gray_u8 = gray_from_rgb(rgb_image)
    gray_f32 = gray_u8.astype(np.float32) / 255.0

    edges = normalize01(gradient_magnitude(gray_u8))
    corners = normalize01(corner_response(gray_u8))
    texture = normalize01(local_variance(gray_f32, sigma=3.0))
    anisotropy = _structure_anisotropy(gray_u8)
    smooth = 1.0 - texture
    dark = 1.0 - value
    bright = value
    neutral = 1.0 - saturation

    excess_green = np.maximum(2.0 * green - red - blue, 0.0)
    green_hue = _wrapped_hue_response(hue, 0.30, 0.10)
    blue_hue = _wrapped_hue_response(hue, 0.58, 0.13)
    warm_hue = normalize01(
        0.55 * _wrapped_hue_response(hue, 0.08, 0.08) + 0.45 * _wrapped_hue_response(hue, 0.98, 0.08)
    )
    brownness = normalize01(0.42 * red + 0.30 * green - 0.20 * blue + 0.08 * warm_hue)
    metallic_tone = normalize01(0.45 * neutral + 0.35 * bright + 0.20 * (1.0 - warm_hue))
    blue_bias = normalize01(blue - red)

    teal_green = normalize01(0.46 * green + 0.22 * blue - 0.18 * red)
    vegetation = normalize01(0.40 * excess_green + 0.24 * green_hue + 0.20 * teal_green + 0.16 * saturation)
    water_color = normalize01(0.34 * neutral + 0.22 * dark + 0.22 * smooth + 0.22 * blue_bias)
    shadow = normalize01(dark * (0.65 + 0.35 * neutral))
    hard_edges = normalize01(0.58 * edges + 0.42 * corners)
    canopy_shadow = normalize01(shadow * (0.58 * vegetation + 0.42 * teal_green) * (0.35 + 0.65 * texture))
    water_evidence = normalize01(0.42 * water_color + 0.22 * blue_bias + 0.20 * neutral + 0.16 * smooth)

    forest = normalize01(
        vegetation
        * (0.44 + 0.56 * texture)
        * (0.40 + 0.60 * teal_green)
        * (0.55 + 0.45 * (0.5 * saturation + 0.5 * smooth))
        * (0.70 + 0.30 * (1.0 - blue_bias))
        + 0.38 * canopy_shadow
    )
    water = normalize01(
        water_evidence
        * (0.60 + 0.40 * dark)
        * np.power(0.62 + 0.38 * smooth, 1.8)
        * (1.0 - 0.88 * hard_edges)
        * (1.0 - 0.92 * forest)
        * (1.0 - 0.55 * anisotropy)
        * (1.0 - 0.82 * teal_green)
        * (1.0 - 0.88 * canopy_shadow)
        * (0.35 + 0.65 * (1.0 - texture))
    )
    rippling_water = normalize01(
        water_evidence
        * (0.14 + 0.86 * texture)
        * (0.32 + 0.68 * bright)
        * (1.0 - 0.62 * forest)
        * (1.0 - 0.55 * canopy_shadow)
        * (0.48 + 0.52 * neutral)
        * (0.38 + 0.62 * blue_bias)
    )

    building_candidate = normalize01(
        (0.30 * bright + 0.26 * hard_edges + 0.18 * corners + 0.14 * neutral + 0.12 * (1.0 - smooth))
        * (1.0 - 0.65 * forest)
        * (1.0 - 0.75 * water)
        * (0.45 + 0.55 * (1.0 - canopy_shadow))
    )
    concrete_building = normalize01(
        building_candidate
        * (0.30 * bright + 0.26 * neutral + 0.24 * hard_edges + 0.20 * texture)
    )
    metal_building = normalize01(
        building_candidate
        * (0.48 * metallic_tone + 0.22 * bright + 0.18 * corners + 0.12 * hard_edges)
        * (1.0 - 0.55 * warm_hue)
    )
    wood_building = normalize01(
        building_candidate
        * (0.40 * warm_hue + 0.24 * brownness + 0.20 * saturation + 0.16 * hard_edges)
    )

    road_candidate = normalize01(
        (0.34 * neutral + 0.24 * smooth + 0.22 * anisotropy + 0.10 * dark + 0.10 * hard_edges)
        * (1.0 - 0.84 * forest)
        * (1.0 - 0.46 * building_candidate)
        * (1.0 - 0.78 * water)
        * (1.0 - 0.45 * canopy_shadow)
    )
    asphalt_road = normalize01(
        road_candidate
        * np.power(0.35 + 0.65 * neutral, 1.6)
        * (0.34 * dark + 0.30 * smooth + 0.20 * neutral + 0.16 * anisotropy)
        * (1.0 - 0.60 * texture)
    )
    dirt_road = normalize01(
        road_candidate
        * (0.34 * brownness + 0.24 * warm_hue + 0.22 * bright + 0.20 * texture)
        * (0.45 + 0.55 * smooth)
    )

    return {
        "forest": forest.astype(np.float32),
        "water": water.astype(np.float32),
        "rippling_water": rippling_water.astype(np.float32),
        "concrete_building": concrete_building.astype(np.float32),
        "metal_building": metal_building.astype(np.float32),
        "dirt_road": dirt_road.astype(np.float32),
        "asphalt_road": asphalt_road.astype(np.float32),
        "wood_building": wood_building.astype(np.float32),
        "building_group": building_candidate.astype(np.float32),
        "road_group": road_candidate.astype(np.float32),
        "water_group": normalize01(water + rippling_water).astype(np.float32),
        "forest_group": forest.astype(np.float32),
        "edges": edges.astype(np.float32),
        "corners": corners.astype(np.float32),
        "texture": texture.astype(np.float32),
        "anisotropy": anisotropy.astype(np.float32),
        "shadow": shadow.astype(np.float32),
        "canopy_shadow": canopy_shadow.astype(np.float32),
        "smooth": smooth.astype(np.float32),
        "dark": dark.astype(np.float32),
        "bright": bright.astype(np.float32),
        "neutral": neutral.astype(np.float32),
        "blue_bias": blue_bias.astype(np.float32),
        "vegetation": vegetation.astype(np.float32),
        "hard_edges": hard_edges.astype(np.float32),
        "metallic_tone": metallic_tone.astype(np.float32),
        "brownness": brownness.astype(np.float32),
        "warm_hue": warm_hue.astype(np.float32),
    }


def stack_class_scores(score_maps: Dict[str, np.ndarray]) -> np.ndarray:
    return np.stack([score_maps[name] for name in TERRAIN_CLASS_NAMES], axis=0).astype(np.float32)


def normalize_probabilities(probabilities: np.ndarray) -> np.ndarray:
    denominator = np.maximum(np.sum(probabilities, axis=0, keepdims=True), 1e-6)
    return probabilities / denominator


def build_pseudo_labels(score_maps: Dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stacked = stack_class_scores(score_maps)
    dominant = np.argmax(stacked, axis=0).astype(np.int32)
    top1 = np.max(stacked, axis=0)
    top2 = np.partition(stacked, -2, axis=0)[-2]
    margin = top1 - top2
    confidence = 0.72 * top1 + 0.28 * np.clip(margin, 0.0, 1.0)

    labels = dominant.copy()
    labels[top1 < 0.34] = -1
    labels[margin < 0.05] = -1
    labels = cv2.medianBlur((labels + 1).astype(np.uint8), 5).astype(np.int32) - 1
    confidence[labels < 0] = 0.0
    return labels, confidence.astype(np.float32)


def render_terrain_map(probabilities: np.ndarray) -> np.ndarray:
    labels = np.argmax(probabilities, axis=0)
    rgb = TERRAIN_CLASS_COLORS[labels]
    confidence = np.max(probabilities, axis=0)[:, :, None]
    return (rgb.astype(np.float32) * (0.45 + 0.55 * confidence)).clip(0, 255).astype(np.uint8)


def render_pseudo_label_map(labels: np.ndarray) -> np.ndarray:
    output = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    valid_mask = labels >= 0
    output[valid_mask] = TERRAIN_CLASS_COLORS[labels[valid_mask]]
    return output


def compute_image_stats(rgb_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = rgb_image.astype(np.float32) / 255.0
    mean = values.reshape(-1, 3).mean(axis=0)
    std = values.reshape(-1, 3).std(axis=0)
    std = np.clip(std, 1e-3, None)
    return mean.astype(np.float32), std.astype(np.float32)


def _spatial_split(x: int, y: int, patch_size: int) -> str:
    return "val" if ((x // patch_size) + (y // patch_size)) % 5 == 0 else "train"


def _extract_patch(image: np.ndarray, center_x: int, center_y: int, patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    return image[center_y - radius : center_y + radius, center_x - radius : center_x + radius].copy()


def collect_patch_catalog(
    rgb_image: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    patch_size: int,
    max_samples_per_class: int,
    seed: int,
) -> List[Dict[str, int | str | float]]:
    radius = patch_size // 2
    height, width = labels.shape
    rng = np.random.default_rng(seed)
    catalog: List[Dict[str, int | str | float]] = []

    for class_index, class_name in enumerate(TERRAIN_CLASS_NAMES):
        candidate_mask = labels == class_index
        candidate_mask[:radius, :] = False
        candidate_mask[-radius:, :] = False
        candidate_mask[:, :radius] = False
        candidate_mask[:, -radius:] = False
        ys, xs = np.where(candidate_mask)
        if len(xs) == 0:
            continue

        order = np.argsort(-confidence[ys, xs])
        xs = xs[order]
        ys = ys[order]
        selected: List[tuple[int, int]] = []
        taken_cells: set[tuple[int, int]] = set()
        min_cell = max(1, patch_size // 2)

        for x, y in zip(xs, ys):
            cell = (x // min_cell, y // min_cell)
            if cell in taken_cells:
                continue
            taken_cells.add(cell)
            selected.append((int(x), int(y)))
            if len(selected) >= max_samples_per_class:
                break

        rng.shuffle(selected)
        for x, y in selected:
            catalog.append(
                {
                    "x": x,
                    "y": y,
                    "label": class_index,
                    "class_name": class_name,
                    "split": _spatial_split(x, y, patch_size),
                    "confidence": float(confidence[y, x]),
                }
            )

    return catalog


def _augment_patch(patch: np.ndarray, rng: random.Random) -> np.ndarray:
    augmented = patch.copy()
    k = rng.randint(0, 3)
    if k:
        augmented = np.rot90(augmented, k).copy()
    if rng.random() < 0.5:
        augmented = np.fliplr(augmented).copy()
    if rng.random() < 0.25:
        augmented = np.flipud(augmented).copy()
    if rng.random() < 0.65:
        alpha = 0.85 + 0.30 * rng.random()
        beta = rng.uniform(-0.06, 0.06)
        augmented = np.clip(augmented.astype(np.float32) / 255.0 * alpha + beta, 0.0, 1.0)
        augmented = (augmented * 255.0).astype(np.uint8)
    if rng.random() < 0.25:
        sigma = 0.5 + 1.1 * rng.random()
        augmented = cv2.GaussianBlur(augmented, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return augmented


class TerrainPatchDataset(Dataset):
    def __init__(
        self,
        rgb_image: np.ndarray,
        samples: Sequence[Dict[str, int | str | float]],
        patch_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool,
        seed: int,
    ) -> None:
        self.rgb_image = rgb_image
        self.samples = list(samples)
        self.patch_size = patch_size
        self.mean = mean.reshape(1, 1, 3)
        self.std = std.reshape(1, 1, 3)
        self.augment = augment
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        patch = _extract_patch(self.rgb_image, int(sample["x"]), int(sample["y"]), self.patch_size)
        if self.augment:
            patch = _augment_patch(patch, random.Random(self.seed + index * 19))
        patch_f32 = patch.astype(np.float32) / 255.0
        patch_f32 = (patch_f32 - self.mean) / self.std
        patch_chw = np.transpose(patch_f32, (2, 0, 1))
        return torch.from_numpy(patch_chw), torch.tensor(int(sample["label"]), dtype=torch.long)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class TerrainResNet(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(256, num_classes)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers: List[nn.Module] = [BasicBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.head(x)


def _epoch_pass(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = loss_fn(logits, labels)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item()) * int(labels.numel())
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_items += int(labels.numel())

    if total_items == 0:
        return 0.0, 0.0
    return total_loss / total_items, total_correct / total_items


def train_terrain_resnet(
    rgb_image: np.ndarray,
    output_path: str | Path,
    patch_size: int = 64,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    max_samples_per_class: int = 320,
    seed: int = 42,
) -> TerrainTrainingResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    score_maps = build_heuristic_score_maps(rgb_image)
    labels, confidence = build_pseudo_labels(score_maps)
    samples = collect_patch_catalog(
        rgb_image=rgb_image,
        labels=labels,
        confidence=confidence,
        patch_size=patch_size,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
    )

    train_samples = [sample for sample in samples if sample["split"] == "train"]
    val_samples = [sample for sample in samples if sample["split"] == "val"]
    if not train_samples or not val_samples:
        raise RuntimeError("Not enough weak labels to train/validate the terrain ResNet.")

    mean, std = compute_image_stats(rgb_image)
    train_dataset = TerrainPatchDataset(rgb_image, train_samples, patch_size, mean, std, augment=True, seed=seed)
    val_dataset = TerrainPatchDataset(rgb_image, val_samples, patch_size, mean, std, augment=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_hist = np.zeros((len(TERRAIN_CLASS_NAMES),), dtype=np.int64)
    for sample in train_samples:
        class_hist[int(sample["label"])] += 1
    weights = 1.0 / np.maximum(class_hist.astype(np.float32), 1.0)
    weights = weights / float(weights.mean())

    model = TerrainResNet(num_classes=len(TERRAIN_CLASS_NAMES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.from_numpy(weights).to(device))

    history: List[Dict[str, float]] = []
    best_state = None
    best_val_accuracy = -math.inf

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _epoch_pass(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = _epoch_pass(model, val_loader, loss_fn, None, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "train_accuracy": float(train_acc),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_acc),
            }
        )
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Terrain ResNet training did not produce a valid checkpoint.")

    payload = {
        "state_dict": best_state,
        "class_names": list(TERRAIN_CLASS_NAMES),
        "patch_size": int(patch_size),
        "mean": mean,
        "std": std,
        "history": history,
        "class_histogram": {name: int(class_hist[idx]) for idx, name in enumerate(TERRAIN_CLASS_NAMES)},
        "train_samples": train_samples,
        "val_samples": val_samples,
        "pseudo_labels": labels,
        "confidence": confidence,
    }
    torch.save(payload, output_path)
    return TerrainTrainingResult(
        model_path=output_path,
        patch_size=patch_size,
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        best_val_accuracy=float(best_val_accuracy),
        class_histogram={name: int(class_hist[idx]) for idx, name in enumerate(TERRAIN_CLASS_NAMES)},
        history=history,
    )


def load_trained_model(path: str | Path) -> tuple[TerrainResNet, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = TerrainResNet(num_classes=len(payload["class_names"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def load_checkpoint_metadata(path: str | Path) -> dict:
    _, payload = load_trained_model(path)
    return payload


def predict_terrain_probabilities(
    rgb_image: np.ndarray,
    checkpoint_path: str | Path,
    stride: int = 24,
    batch_size: int = 48,
) -> np.ndarray:
    model, payload = load_trained_model(checkpoint_path)
    mean = np.asarray(payload["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(payload["std"], dtype=np.float32).reshape(1, 1, 3)
    patch_size = int(payload["patch_size"])
    window = cosine_window(patch_size)
    device = torch.device("cpu")

    height, width = rgb_image.shape[:2]
    accum = np.zeros((len(TERRAIN_CLASS_NAMES), height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32)
    pending_tensors: List[np.ndarray] = []
    pending_meta: List[tuple[int, int]] = []

    def flush() -> None:
        if not pending_tensors:
            return
        batch = np.stack(pending_tensors, axis=0)
        with torch.no_grad():
            logits = model(torch.from_numpy(batch).to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        for (x, y), prob in zip(pending_meta, probs):
            accum[:, y : y + patch_size, x : x + patch_size] += prob[:, None, None] * window[None, :, :]
            weight_sum[y : y + patch_size, x : x + patch_size] += window
        pending_tensors.clear()
        pending_meta.clear()

    for x, y in iter_windows(height, width, patch_size, stride):
        patch = rgb_image[y : y + patch_size, x : x + patch_size].astype(np.float32) / 255.0
        patch = (patch - mean) / std
        pending_tensors.append(np.transpose(patch, (2, 0, 1)).astype(np.float32))
        pending_meta.append((x, y))
        if len(pending_tensors) >= batch_size:
            flush()

    flush()
    weight_sum = np.maximum(weight_sum, 1e-6)
    return accum / weight_sum[None, :, :]


def _class_index(name: str) -> int:
    return int(TERRAIN_CLASS_NAMES.index(name))


def _support_mask(values: np.ndarray, threshold: float, kernel_size: int) -> np.ndarray:
    mask = (values > threshold).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return cv2.GaussianBlur(closed.astype(np.float32), (0, 0), sigmaX=1.4, sigmaY=1.4)


def _cleanup_small_components(probabilities: np.ndarray) -> np.ndarray:
    labels = np.argmax(probabilities, axis=0).astype(np.int32)
    min_sizes = {
        "forest": 180,
        "water": 700,
        "rippling_water": 260,
        "concrete_building": 18,
        "metal_building": 18,
        "dirt_road": 50,
        "asphalt_road": 90,
        "wood_building": 18,
    }

    for class_index, class_name in enumerate(TERRAIN_CLASS_NAMES):
        mask = (labels == class_index).astype(np.uint8)
        if int(mask.sum()) == 0:
            continue
        num_labels, component_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component_id in range(1, num_labels):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area >= min_sizes[class_name]:
                continue
            component_mask = component_labels == component_id
            probabilities[class_index, component_mask] *= 0.05
            labels[component_mask] = -1
    return normalize_probabilities(probabilities)


def _region_mask(values: np.ndarray, threshold: float, kernel_size: int) -> np.ndarray:
    mask = (values > threshold).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def derive_object_priors(score_maps: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    shape = score_maps["forest"].shape
    priors = {name: np.zeros(shape, dtype=np.float32) for name in TERRAIN_CLASS_NAMES}

    building_candidate = normalize01(score_maps["building_group"] * (1.0 - 0.55 * score_maps["canopy_shadow"]))
    road_candidate = normalize01(score_maps["road_group"] * (1.0 - 0.35 * score_maps["canopy_shadow"]))
    water_candidate = normalize01(
        score_maps["water_group"]
        * (0.45 + 0.55 * score_maps["smooth"])
        * (1.0 - 0.78 * score_maps["forest_group"])
        * (1.0 - 0.82 * score_maps["canopy_shadow"])
        * (0.35 + 0.65 * score_maps["blue_bias"])
        * (1.0 - 0.40 * score_maps["hard_edges"])
    )
    forest_candidate = normalize01(score_maps["forest_group"] + 0.30 * score_maps["canopy_shadow"])

    building_seed = _region_mask(building_candidate, threshold=0.20, kernel_size=5)
    road_seed = _region_mask(road_candidate, threshold=0.18, kernel_size=7)
    water_seed = _region_mask(water_candidate, threshold=0.30, kernel_size=9)
    forest_seed = _region_mask(forest_candidate, threshold=0.22, kernel_size=7)

    building_seed = cv2.bitwise_and(building_seed, cv2.bitwise_not(water_seed))
    road_seed = cv2.bitwise_and(road_seed, cv2.bitwise_not(building_seed))
    road_seed = cv2.bitwise_and(road_seed, cv2.bitwise_not(water_seed))
    forest_seed = cv2.bitwise_and(forest_seed, cv2.bitwise_not(road_seed))
    forest_seed = cv2.bitwise_and(forest_seed, cv2.bitwise_not(building_seed))
    forest_seed = cv2.bitwise_and(forest_seed, cv2.bitwise_not(water_seed))

    compound_seed = _region_mask(np.maximum(building_candidate, 0.82 * road_candidate), threshold=0.18, kernel_size=13)
    compound_seed = cv2.bitwise_and(compound_seed, cv2.bitwise_not(water_seed))

    num_compounds, compound_labels, compound_stats, _ = cv2.connectedComponentsWithStats(compound_seed, connectivity=8)
    for component_id in range(1, num_compounds):
        area = int(compound_stats[component_id, cv2.CC_STAT_AREA])
        if area < 160:
            continue
        component_mask = (compound_labels == component_id).astype(np.uint8)
        expanded = cv2.dilate(component_mask, np.ones((11, 11), dtype=np.uint8))
        expanded_mask = expanded > 0
        building_pixels = expanded_mask & (building_candidate > 0.08)
        road_pixels = expanded_mask & (road_candidate > 0.08)
        confidence = min(area / 2200.0, 1.0)

        if np.any(building_pixels):
            metallic = float(np.mean(score_maps["metallic_tone"][building_pixels]))
            warm = float(np.mean(score_maps["warm_hue"][building_pixels]))
            brown = float(np.mean(score_maps["brownness"][building_pixels]))
            texture = float(np.mean(score_maps["texture"][building_pixels]))
            if metallic > 0.55:
                priors["metal_building"][building_pixels] = np.maximum(
                    priors["metal_building"][building_pixels],
                    0.52 + 0.42 * confidence,
                )
            elif warm + brown > 0.95:
                priors["wood_building"][building_pixels] = np.maximum(
                    priors["wood_building"][building_pixels],
                    0.46 + 0.38 * confidence,
                )
            else:
                priors["concrete_building"][building_pixels] = np.maximum(
                    priors["concrete_building"][building_pixels],
                    0.48 + 0.36 * confidence + 0.10 * texture,
                )

        if np.any(road_pixels):
            dark = float(np.mean(score_maps["dark"][road_pixels]))
            neutral = float(np.mean(score_maps["neutral"][road_pixels]))
            smooth = float(np.mean(score_maps["smooth"][road_pixels]))
            brown = float(np.mean(score_maps["brownness"][road_pixels]))
            warm = float(np.mean(score_maps["warm_hue"][road_pixels]))
            if dark + neutral + smooth > brown + warm + 0.22:
                priors["asphalt_road"][road_pixels] = np.maximum(
                    priors["asphalt_road"][road_pixels],
                    0.46 + 0.34 * confidence,
                )
            else:
                priors["dirt_road"][road_pixels] = np.maximum(
                    priors["dirt_road"][road_pixels],
                    0.42 + 0.30 * confidence,
                )

    num_buildings, building_labels, building_stats, _ = cv2.connectedComponentsWithStats(building_seed, connectivity=8)
    for component_id in range(1, num_buildings):
        area = int(building_stats[component_id, cv2.CC_STAT_AREA])
        if area < 18:
            continue
        component_mask = building_labels == component_id
        metallic = float(np.mean(score_maps["metallic_tone"][component_mask]))
        warm = float(np.mean(score_maps["warm_hue"][component_mask]))
        brown = float(np.mean(score_maps["brownness"][component_mask]))
        texture = float(np.mean(score_maps["texture"][component_mask]))
        neutral = float(np.mean(score_maps["neutral"][component_mask]))
        confidence = min(area / 300.0, 1.0)

        if metallic > 0.55 and neutral > 0.45:
            priors["metal_building"][component_mask] = max(priors["metal_building"][component_mask].max(), 0.55 + 0.45 * confidence)
        elif warm + brown > 1.00:
            priors["wood_building"][component_mask] = max(priors["wood_building"][component_mask].max(), 0.50 + 0.40 * confidence)
        else:
            concrete_score = 0.45 + 0.35 * confidence + 0.20 * texture
            priors["concrete_building"][component_mask] = max(priors["concrete_building"][component_mask].max(), concrete_score)

    num_roads, road_labels, road_stats, _ = cv2.connectedComponentsWithStats(road_seed, connectivity=8)
    for component_id in range(1, num_roads):
        area = int(road_stats[component_id, cv2.CC_STAT_AREA])
        if area < 90:
            continue
        component_mask = road_labels == component_id
        dark = float(np.mean(score_maps["dark"][component_mask]))
        neutral = float(np.mean(score_maps["neutral"][component_mask]))
        smooth = float(np.mean(score_maps["smooth"][component_mask]))
        brown = float(np.mean(score_maps["brownness"][component_mask]))
        warm = float(np.mean(score_maps["warm_hue"][component_mask]))
        confidence = min(area / 1200.0, 1.0)

        if dark + neutral + smooth > brown + warm + 0.35:
            priors["asphalt_road"][component_mask] = max(priors["asphalt_road"][component_mask].max(), 0.52 + 0.38 * confidence)
        else:
            priors["dirt_road"][component_mask] = max(priors["dirt_road"][component_mask].max(), 0.48 + 0.36 * confidence)

    num_water, water_labels, water_stats, _ = cv2.connectedComponentsWithStats(water_seed, connectivity=8)
    for component_id in range(1, num_water):
        area = int(water_stats[component_id, cv2.CC_STAT_AREA])
        if area < 180:
            continue
        component_mask = water_labels == component_id
        smooth = float(np.mean(score_maps["smooth"][component_mask]))
        vegetation = float(np.mean(score_maps["vegetation"][component_mask]))
        blue = float(np.mean(score_maps["blue_bias"][component_mask]))
        hard_edges = float(np.mean(score_maps["hard_edges"][component_mask]))
        canopy_shadow = float(np.mean(score_maps["canopy_shadow"][component_mask]))
        texture = float(np.mean(score_maps["texture"][component_mask]))
        confidence = min(area / 6000.0, 1.0)
        if smooth < 0.54 or vegetation > 0.42 or hard_edges > 0.18 or canopy_shadow > 0.40:
            continue
        if texture > 0.26:
            priors["rippling_water"][component_mask] = np.maximum(
                priors["rippling_water"][component_mask],
                0.42 + 0.30 * confidence + 0.14 * blue,
            )
        else:
            priors["water"][component_mask] = np.maximum(
                priors["water"][component_mask],
                0.52 + 0.32 * confidence + 0.12 * blue,
            )

    priors["forest"][forest_seed > 0] = 0.72

    for class_name in priors:
        priors[class_name] = cv2.GaussianBlur(priors[class_name], (0, 0), sigmaX=1.2, sigmaY=1.2).astype(np.float32)
    return priors


def combine_terrain_probabilities(
    score_maps: Dict[str, np.ndarray],
    model_probabilities: np.ndarray | None = None,
    model_class_histogram: Dict[str, int] | None = None,
) -> np.ndarray:
    object_priors = derive_object_priors(score_maps)
    heuristic_scores = np.maximum(stack_class_scores(score_maps), 1e-5)
    heuristic_probabilities = normalize_probabilities(np.power(heuristic_scores, 1.35))

    if model_probabilities is None:
        combined = heuristic_probabilities.copy()
    else:
        if model_class_histogram is None:
            reliability = np.full((len(TERRAIN_CLASS_NAMES),), 0.18, dtype=np.float32)
        else:
            reliability = np.array(
                [min(float(model_class_histogram.get(name, 0)) / 120.0, 1.0) for name in TERRAIN_CLASS_NAMES],
                dtype=np.float32,
            )
            reliability = 0.02 + 0.28 * reliability
        combined = (
            heuristic_probabilities * (1.0 - reliability[:, None, None])
            + model_probabilities * reliability[:, None, None]
        ).astype(np.float32)

    forest_gate = normalize01(score_maps["forest_group"] + 0.30 * score_maps["canopy_shadow"])
    water_gate = normalize01(
        score_maps["water_group"]
        * (0.35 + 0.65 * score_maps["blue_bias"])
        * (1.0 - 0.60 * score_maps["vegetation"])
    )
    building_gate = normalize01(score_maps["building_group"])
    road_gate = normalize01(score_maps["road_group"])
    metallic_tone = normalize01(score_maps["metallic_tone"])
    brownness = normalize01(score_maps["brownness"])
    warm_hue = normalize01(score_maps["warm_hue"])
    smooth = normalize01(score_maps["smooth"])
    dark = normalize01(score_maps["dark"])
    bright = normalize01(score_maps["bright"])
    neutral = normalize01(score_maps["neutral"])
    blue_bias = normalize01(score_maps["blue_bias"])
    vegetation = normalize01(score_maps["vegetation"])
    canopy_shadow = normalize01(score_maps["canopy_shadow"])
    texture = normalize01(score_maps["texture"])
    anisotropy = normalize01(score_maps["anisotropy"])
    manmade_gate = np.maximum(building_gate, road_gate)

    combined[_class_index("forest")] *= 0.18 + 1.15 * forest_gate * (1.0 - 0.42 * manmade_gate) * (0.72 + 0.28 * canopy_shadow)
    combined[_class_index("water")] *= (
        0.01
        + 1.28
        * water_gate
        * (0.42 + 0.58 * smooth)
        * (0.24 + 0.76 * blue_bias)
        * (1.0 - 0.82 * manmade_gate)
        * (1.0 - 0.82 * canopy_shadow)
    )
    combined[_class_index("rippling_water")] *= (
        0.01
        + 1.20
        * water_gate
        * (0.28 + 0.72 * texture)
        * (0.28 + 0.72 * blue_bias)
        * (1.0 - 0.60 * building_gate)
        * (1.0 - 0.65 * canopy_shadow)
    )
    combined[_class_index("concrete_building")] *= (
        0.01 + 1.48 * building_gate * (0.45 + 0.55 * texture) * (0.42 + 0.58 * bright) * (1.0 - 0.58 * water_gate)
    )
    combined[_class_index("metal_building")] *= (
        0.01 + 1.62 * building_gate * (0.30 + 0.70 * metallic_tone) * (0.40 + 0.60 * bright) * (1.0 - 0.55 * warm_hue)
    )
    combined[_class_index("wood_building")] *= (
        0.01 + 1.35 * building_gate * (0.35 + 0.65 * brownness) * (0.40 + 0.60 * warm_hue)
    )
    combined[_class_index("dirt_road")] *= (
        0.01 + 1.40 * road_gate * (0.30 + 0.70 * brownness) * (1.0 - 0.40 * water_gate)
    )
    combined[_class_index("asphalt_road")] *= (
        0.01 + 1.55 * road_gate * (0.35 + 0.65 * neutral) * (0.35 + 0.65 * dark) * (1.0 - 0.55 * texture)
    )

    combined[_class_index("forest")] *= 1.0 - 0.48 * water_gate
    combined[_class_index("water")] *= 1.0 - 0.30 * forest_gate
    combined[_class_index("rippling_water")] *= 1.0 - 0.25 * forest_gate
    combined[_class_index("dirt_road")] *= 1.0 - 0.30 * forest_gate
    combined[_class_index("asphalt_road")] *= 1.0 - 0.25 * forest_gate

    for class_name, prior_map in object_priors.items():
        combined[_class_index(class_name)] *= 0.35 + 1.55 * prior_map

    combined = normalize_probabilities(np.maximum(combined, 1e-7))

    smoothed = np.stack(
        [
            cv2.GaussianBlur(combined[class_index], (0, 0), sigmaX=1.6, sigmaY=1.6)
            for class_index in range(combined.shape[0])
        ],
        axis=0,
    ).astype(np.float32)
    combined = normalize_probabilities(0.72 * combined + 0.28 * smoothed)

    support_maps = {
        "forest": _support_mask(np.maximum(forest_gate, 0.68 * canopy_shadow), threshold=0.20, kernel_size=5),
        "water": _support_mask(
            water_gate * smooth * blue_bias * (1.0 - canopy_shadow) * (1.0 - 0.80 * vegetation),
            threshold=0.15,
            kernel_size=9,
        ),
        "rippling_water": _support_mask(
            water_gate * texture * blue_bias * (1.0 - 0.75 * canopy_shadow) * (1.0 - 0.65 * vegetation),
            threshold=0.14,
            kernel_size=5,
        ),
        "concrete_building": _support_mask(building_gate * texture * (1.0 - 0.55 * canopy_shadow), threshold=0.14, kernel_size=3),
        "metal_building": _support_mask(building_gate * metallic_tone * (1.0 - 0.45 * canopy_shadow), threshold=0.12, kernel_size=3),
        "wood_building": _support_mask(building_gate * brownness * (1.0 - 0.35 * canopy_shadow), threshold=0.12, kernel_size=3),
        "dirt_road": _support_mask(road_gate * brownness * (1.0 - 0.35 * canopy_shadow), threshold=0.14, kernel_size=5),
        "asphalt_road": _support_mask(road_gate * neutral * dark * (1.0 - 0.45 * canopy_shadow), threshold=0.14, kernel_size=5),
    }

    for class_name, support in support_maps.items():
        class_index = _class_index(class_name)
        combined[class_index] *= 0.10 + 0.90 * support

    scene_presence = {
        class_name: max(float(object_priors[class_name].mean()), float(support_maps[class_name].mean()))
        for class_name in TERRAIN_CLASS_NAMES
    }
    scene_scales = {
        "forest": np.clip(0.65 + 1.45 * scene_presence["forest"], 0.65, 1.35),
        "water": np.clip(0.04 + 7.50 * scene_presence["water"], 0.04, 1.05),
        "rippling_water": np.clip(0.04 + 7.50 * scene_presence["rippling_water"], 0.04, 1.05),
        "concrete_building": np.clip(0.08 + 8.50 * scene_presence["concrete_building"], 0.08, 1.25),
        "metal_building": np.clip(0.08 + 8.50 * scene_presence["metal_building"], 0.08, 1.30),
        "dirt_road": np.clip(0.08 + 8.00 * scene_presence["dirt_road"], 0.08, 1.20),
        "asphalt_road": np.clip(0.08 + 8.00 * scene_presence["asphalt_road"], 0.08, 1.25),
        "wood_building": np.clip(0.06 + 8.50 * scene_presence["wood_building"], 0.06, 1.15),
    }
    for class_name, scale in scene_scales.items():
        combined[_class_index(class_name)] *= float(scale)

    water_index = _class_index("water")
    rippling_index = _class_index("rippling_water")
    forest_index = _class_index("forest")
    water_residual = combined[water_index] * (1.0 - support_maps["water"]) * (0.50 + 0.50 * forest_gate)
    rippling_residual = combined[rippling_index] * (1.0 - support_maps["rippling_water"]) * (0.45 + 0.55 * forest_gate)
    combined[water_index] *= 0.06 + 0.94 * support_maps["water"]
    combined[rippling_index] *= 0.08 + 0.92 * support_maps["rippling_water"]
    combined[forest_index] += 0.82 * water_residual + 0.68 * rippling_residual

    combined = _cleanup_small_components(combined)
    labels = np.argmax(combined, axis=0).astype(np.uint8)
    labels = cv2.medianBlur(labels + 1, 5) - 1
    refined = np.zeros_like(combined)
    for class_index in range(combined.shape[0]):
        refined[class_index] = cv2.GaussianBlur((labels == class_index).astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
    combined = normalize_probabilities(0.60 * combined + 0.40 * refined)
    return combined.astype(np.float32)


def export_training_debug_images(checkpoint_path: str | Path, output_dir: str | Path) -> None:
    _, payload = load_trained_model(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pseudo_labels = payload["pseudo_labels"].astype(np.int32)
    confidence = payload["confidence"].astype(np.float32)
    cv2.imwrite(str(output_dir / "pseudo_labels.png"), cv2.cvtColor(render_pseudo_label_map(pseudo_labels), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(output_dir / "pseudo_confidence.png"), normalize_to_uint8(confidence))
