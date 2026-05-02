from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .common import cosine_window, gray_from_rgb, iter_windows, local_variance, normalize01
from .territory_segmentation import TERRITORY_CLASS_NAMES, render_territory_map


RADAR_FEATURE_NAMES: Sequence[str] = (
    "intensity",
    "local_mean",
    "local_variance",
    "gradient",
    "laplacian_abs",
    "local_contrast",
    "top_hat",
    "radial_distance",
    "active_mask",
)


@dataclass(frozen=True)
class RadarObservation:
    visualization_rgb: np.ndarray
    intensity: np.ndarray
    active_mask: np.ndarray
    support_mask: np.ndarray
    origin_px: tuple[float, float]
    source_path: Path


@dataclass(frozen=True)
class RadarTrainingResult:
    model_path: Path
    patch_size: int
    train_samples: int
    val_samples: int
    best_val_accuracy: float
    class_histogram: Dict[str, int]
    history: List[Dict[str, float]]


@dataclass(frozen=True)
class RadarInferenceResult:
    class_names: Sequence[str]
    class_map: np.ndarray
    probabilities: np.ndarray
    color_map: np.ndarray
    overlay: np.ndarray
    active_mask: np.ndarray
    report: Mapping[str, object]


def _detect_edge_ring_mask(
    intensity: np.ndarray,
    active_mask: np.ndarray,
    origin_px: tuple[float, float],
) -> np.ndarray:
    if not np.any(active_mask):
        return np.zeros_like(active_mask, dtype=bool)

    height, width = intensity.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = xx - float(origin_px[0])
    dy = yy - float(origin_px[1])
    radius_map = np.sqrt(dx * dx + dy * dy)
    max_radius = float(radius_map[active_mask].max())
    start_radius = int(max(8, round(0.82 * max_radius)))
    end_radius = int(round(max_radius))
    angle_bins = (np.floor(((np.arctan2(dy, dx) + np.pi) / (2.0 * np.pi)) * 180.0).astype(np.int32) % 180)

    candidate_radii: list[int] = []
    for radius in range(start_radius, end_radius + 1):
        band = active_mask & (radius_map >= radius - 1.5) & (radius_map <= radius + 1.5)
        if int(np.count_nonzero(band)) < 120:
            continue

        mean_intensity = float(intensity[band].mean())
        if mean_intensity < 0.035:
            continue

        coverages: List[float] = []
        for angle_bin in range(180):
            angular_band = band & (angle_bins == angle_bin)
            if not np.any(angular_band):
                continue
            coverages.append(float(np.mean(intensity[angular_band] > 0.03)))
        if not coverages:
            continue

        circular_coverage = float(np.mean(np.asarray(coverages) > 0.5))
        if circular_coverage >= 0.55:
            candidate_radii.append(radius)

    if not candidate_radii:
        return np.zeros_like(active_mask, dtype=bool)

    ring_mask = np.zeros_like(active_mask, dtype=bool)
    segment_start = candidate_radii[0]
    previous = candidate_radii[0]
    for radius in candidate_radii[1:] + [candidate_radii[-1] + 10]:
        if radius <= previous + 2:
            previous = radius
            continue
        lower = max(0.0, float(segment_start) - 2.5)
        upper = min(max_radius, float(previous) + 2.5)
        ring_mask |= active_mask & (radius_map >= lower) & (radius_map <= upper)
        segment_start = radius
        previous = radius

    return ring_mask


def _estimate_signal_support_mask(intensity: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    if not np.any(active_mask):
        return active_mask.astype(bool)

    high = ((intensity > 0.10) & active_mask).astype(np.uint8)
    low = ((intensity > 0.05) & active_mask).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(low, connectivity=8)
    support = np.zeros_like(low, dtype=np.uint8)
    for label in range(1, num_labels):
        component = labels == label
        if not np.any(high[component]):
            continue
        if int(stats[label, cv2.CC_STAT_AREA]) < 150:
            continue
        support[component] = 1

    if int(np.count_nonzero(support)) == 0:
        return active_mask.astype(bool)

    support = cv2.morphologyEx(support, cv2.MORPH_CLOSE, _kernel(15))
    support = cv2.dilate(support, _kernel(9), iterations=1)
    support = cv2.morphologyEx(support, cv2.MORPH_OPEN, _kernel(5))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(support, connectivity=8)
    if num_labels <= 1:
        return support > 0

    best_label = 1
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label
    return labels == best_label


def align_label_map_to_radar(
    label_map: np.ndarray,
    target_shape: tuple[int, int],
    mode: str = "stretch",
) -> tuple[np.ndarray, Dict[str, object]]:
    source_h, source_w = label_map.shape
    target_h, target_w = target_shape
    resolved_mode = str(mode).lower()
    if resolved_mode == "auto":
        resolved_mode = "stretch"

    if (source_h, source_w) == (target_h, target_w):
        return label_map.astype(np.int32).copy(), {
            "mode": resolved_mode,
            "source_shape": [int(source_h), int(source_w)],
            "target_shape": [int(target_h), int(target_w)],
            "resized": False,
        }

    if resolved_mode == "stretch":
        aligned = cv2.resize(
            label_map.astype(np.uint8),
            (int(target_w), int(target_h)),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
        return aligned, {
            "mode": resolved_mode,
            "source_shape": [int(source_h), int(source_w)],
            "target_shape": [int(target_h), int(target_w)],
            "resized": True,
        }

    if resolved_mode not in {"contain", "cover"}:
        raise ValueError(f"Unsupported alignment mode: {mode}")

    scale_fn = min if resolved_mode == "contain" else max
    scale = float(scale_fn(target_w / max(source_w, 1), target_h / max(source_h, 1)))
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(
        label_map.astype(np.uint8),
        (int(resized_w), int(resized_h)),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)

    canvas = np.full((target_h, target_w), -1, dtype=np.int32)
    if resolved_mode == "contain":
        offset_x = max(0, (target_w - resized_w) // 2)
        offset_y = max(0, (target_h - resized_h) // 2)
        canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
        placement = {
            "x": int(offset_x),
            "y": int(offset_y),
            "width": int(resized_w),
            "height": int(resized_h),
        }
    else:
        crop_x = max(0, (resized_w - target_w) // 2)
        crop_y = max(0, (resized_h - target_h) // 2)
        cropped = resized[crop_y : crop_y + target_h, crop_x : crop_x + target_w]
        canvas[:, :] = cropped
        placement = {
            "crop_x": int(crop_x),
            "crop_y": int(crop_y),
            "width": int(target_w),
            "height": int(target_h),
        }

    return canvas, {
        "mode": resolved_mode,
        "source_shape": [int(source_h), int(source_w)],
        "target_shape": [int(target_h), int(target_w)],
        "resized": True,
        "scale": scale,
        "placement": placement,
        "valid_pixel_count": int(np.count_nonzero(canvas >= 0)),
    }


def _kernel(size: int) -> np.ndarray:
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _robust_normalize(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    array = values.astype(np.float32)
    if mask is None or not np.any(mask):
        samples = array.reshape(-1)
    else:
        samples = array[mask]
    finite_samples = samples[np.isfinite(samples)]
    if finite_samples.size == 0:
        return np.zeros_like(array, dtype=np.float32)
    low = float(np.percentile(finite_samples, 2.0))
    high = float(np.percentile(finite_samples, 98.0))
    if high - low < 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return ((array - low) / (high - low)).clip(0.0, 1.0).astype(np.float32)


def _largest_center_component(mask: np.ndarray) -> np.ndarray:
    work = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(work, connectivity=8)
    if num_labels <= 1:
        return mask.astype(bool)
    center_y = mask.shape[0] // 2
    center_x = mask.shape[1] // 2
    center_label = int(labels[center_y, center_x])
    if center_label > 0:
        return labels == center_label
    best_label = 1
    best_area = 0
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_label = label
    return labels == best_label


def _estimate_active_mask(rgb_image: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is not None and np.any(alpha > 0) and float(np.mean(alpha < 250)) > 0.01:
        base_mask = alpha > 0
    else:
        gray_u8 = gray_from_rgb(rgb_image)
        threshold = max(4, int(np.percentile(gray_u8, 80.0) * 0.18))
        base_mask = gray_u8 > threshold
    base_mask = cv2.morphologyEx(base_mask.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(11)) > 0
    base_mask = cv2.morphologyEx(base_mask.astype(np.uint8), cv2.MORPH_OPEN, _kernel(7)) > 0
    base_mask = _largest_center_component(base_mask)
    if float(np.mean(base_mask)) < 0.08:
        return np.ones(base_mask.shape, dtype=bool)
    return base_mask


def _visualization_from_scalar_map(values: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    normalized = (_robust_normalize(values, mask=active_mask) * 255.0).astype(np.uint8)
    return np.repeat(normalized[:, :, None], 3, axis=2)


def load_radar_observation(path: str | Path) -> RadarObservation:
    source_path = Path(path)
    if not source_path.exists():
        raise FileNotFoundError(f"Radar input does not exist: {source_path}")

    if source_path.suffix.lower() == ".npy":
        scalar_map = np.load(source_path).astype(np.float32)
        if scalar_map.ndim != 2:
            raise ValueError(f"Radar .npy input must be a 2D array, got shape {scalar_map.shape}.")
        finite = np.isfinite(scalar_map)
        if not np.any(finite):
            raise ValueError(f"Radar .npy input contains no finite values: {source_path}")
        fill_value = float(np.median(scalar_map[finite]))
        scalar_map = np.where(finite, scalar_map, fill_value).astype(np.float32)
        active_mask = np.ones(scalar_map.shape, dtype=bool)
        visualization_rgb = _visualization_from_scalar_map(scalar_map, active_mask=active_mask)
        intensity = _robust_normalize(scalar_map, mask=active_mask)
        height, width = scalar_map.shape
        return RadarObservation(
            visualization_rgb=visualization_rgb,
            intensity=intensity.astype(np.float32),
            active_mask=active_mask,
            support_mask=active_mask.copy(),
            origin_px=(width / 2.0, height / 2.0),
            source_path=source_path,
        )

    image = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Cannot read radar image: {source_path}")

    alpha = None
    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        bgr = image[:, :, :3]
    elif image.ndim == 3 and image.shape[2] == 3:
        bgr = image
    else:
        raise ValueError(f"Unsupported radar image shape: {image.shape}")

    rgb_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    active_mask = _estimate_active_mask(rgb_image, alpha)
    intensity = _robust_normalize(gray_from_rgb(rgb_image).astype(np.float32) / 255.0, mask=active_mask)
    ring_mask = _detect_edge_ring_mask(intensity=intensity, active_mask=active_mask, origin_px=(rgb_image.shape[1] / 2.0, rgb_image.shape[0] / 2.0))
    if np.any(ring_mask):
        active_mask = active_mask & ~ring_mask
        intensity = intensity.copy()
        intensity[ring_mask] = 0.0
        rgb_image = rgb_image.copy()
        rgb_image[ring_mask] = 0
    support_mask = _estimate_signal_support_mask(intensity=intensity, active_mask=active_mask)
    visualization_rgb = rgb_image.copy()
    if np.any(support_mask):
        visualization_rgb[~support_mask] = 0
    height, width = intensity.shape
    return RadarObservation(
        visualization_rgb=visualization_rgb,
        intensity=intensity.astype(np.float32),
        active_mask=active_mask,
        support_mask=support_mask.astype(bool),
        origin_px=(width / 2.0, height / 2.0),
        source_path=source_path,
    )


def build_radar_feature_cube(observation: RadarObservation) -> np.ndarray:
    intensity = observation.intensity.astype(np.float32)
    active_mask = observation.active_mask
    support_mask = observation.support_mask
    intensity_u8 = (intensity * 255.0).clip(0, 255).astype(np.uint8)

    local_mean = cv2.GaussianBlur(intensity, (0, 0), sigmaX=1.8, sigmaY=1.8)
    local_var = normalize01(local_variance(intensity, sigma=2.2))
    gradient_x = cv2.Sobel(intensity_u8, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(intensity_u8, cv2.CV_32F, 0, 1, ksize=3)
    gradient = normalize01(np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y))
    laplacian_abs = normalize01(np.abs(cv2.Laplacian(intensity_u8, cv2.CV_32F, ksize=3)))
    blurred_small = cv2.GaussianBlur(intensity, (0, 0), sigmaX=1.2, sigmaY=1.2)
    blurred_large = cv2.GaussianBlur(intensity, (0, 0), sigmaX=5.0, sigmaY=5.0)
    local_contrast = normalize01(np.abs(blurred_small - blurred_large))
    top_hat = normalize01(
        cv2.morphologyEx(intensity_u8, cv2.MORPH_TOPHAT, _kernel(9)).astype(np.float32) / 255.0
    )

    yy, xx = np.mgrid[0 : intensity.shape[0], 0 : intensity.shape[1]].astype(np.float32)
    dx = xx - float(observation.origin_px[0])
    dy = yy - float(observation.origin_px[1])
    radial_distance = np.sqrt(dx * dx + dy * dy)
    if np.any(active_mask):
        max_distance = float(np.max(radial_distance[active_mask]))
    else:
        max_distance = float(np.max(radial_distance))
    radial_distance = radial_distance / max(max_distance, 1e-6)

    feature_cube = np.stack(
        [
            intensity,
            local_mean.astype(np.float32),
            local_var.astype(np.float32),
            gradient.astype(np.float32),
            laplacian_abs.astype(np.float32),
            local_contrast.astype(np.float32),
            top_hat.astype(np.float32),
            radial_distance.astype(np.float32),
            support_mask.astype(np.float32),
        ],
        axis=-1,
    ).astype(np.float32)
    return feature_cube


def _spatial_split(x: int, y: int, patch_size: int) -> str:
    return "val" if ((x // patch_size) + (y // patch_size)) % 5 == 0 else "train"


def _extract_patch(image: np.ndarray, center_x: int, center_y: int, patch_size: int) -> np.ndarray:
    radius = patch_size // 2
    return image[center_y - radius : center_y + radius, center_x - radius : center_x + radius].copy()


def _valid_centers_mask(active_mask: np.ndarray, patch_size: int) -> np.ndarray:
    radius = max(1, patch_size // 2)
    valid = active_mask.astype(bool).copy()
    valid[:radius, :] = False
    valid[-radius:, :] = False
    valid[:, :radius] = False
    valid[:, -radius:] = False
    if not np.all(active_mask):
        kernel = np.ones((max(3, radius), max(3, radius)), dtype=np.uint8)
        valid = cv2.erode(valid.astype(np.uint8), kernel, iterations=1) > 0
    return valid


def collect_radar_patch_catalog(
    labels: np.ndarray,
    active_mask: np.ndarray,
    patch_size: int,
    max_samples_per_class: int,
    seed: int,
) -> List[Dict[str, int | str]]:
    if labels.shape != active_mask.shape:
        raise ValueError(f"Labels and active mask must match, got {labels.shape} vs {active_mask.shape}.")

    rng = np.random.default_rng(seed)
    valid_centers = _valid_centers_mask(active_mask, patch_size=patch_size)
    catalog: List[Dict[str, int | str]] = []

    for class_index, class_name in enumerate(TERRITORY_CLASS_NAMES):
        candidate_mask = (labels == class_index) & valid_centers
        ys, xs = np.where(candidate_mask)
        if len(xs) == 0:
            continue

        order = rng.permutation(len(xs))
        xs = xs[order]
        ys = ys[order]
        selected: List[tuple[int, int]] = []
        taken_cells: set[tuple[int, int]] = set()
        min_cell = max(1, patch_size // 2)

        for x, y in zip(xs.tolist(), ys.tolist()):
            cell = (int(x) // min_cell, int(y) // min_cell)
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
                }
            )

    return catalog


def compute_feature_stats(feature_cube: np.ndarray, active_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    feature_count = feature_cube.shape[-1]
    if np.any(active_mask):
        values = feature_cube[active_mask]
    else:
        values = feature_cube.reshape(-1, feature_count)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.clip(std, 1e-3, None)
    return mean.astype(np.float32), std.astype(np.float32)


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
        noise = rng.uniform(0.0, 0.025)
        augmented[:, :, :-1] = np.clip(
            augmented[:, :, :-1] + np.random.default_rng(rng.randint(0, 1_000_000)).normal(0.0, noise, augmented[:, :, :-1].shape),
            0.0,
            1.0,
        )
    if rng.random() < 0.20:
        sigma = 0.5 + 0.8 * rng.random()
        augmented[:, :, :-1] = np.stack(
            [
                cv2.GaussianBlur(
                    np.ascontiguousarray(augmented[:, :, channel]),
                    (0, 0),
                    sigmaX=sigma,
                    sigmaY=sigma,
                )
                for channel in range(augmented.shape[2] - 1)
            ],
            axis=-1,
        )
    return augmented.astype(np.float32)


class RadarPatchDataset(Dataset):
    def __init__(
        self,
        feature_cube: np.ndarray,
        samples: Sequence[Dict[str, int | str]],
        patch_size: int,
        mean: np.ndarray,
        std: np.ndarray,
        augment: bool,
        seed: int,
    ) -> None:
        self.feature_cube = feature_cube.astype(np.float32)
        self.samples = list(samples)
        self.patch_size = int(patch_size)
        self.mean = mean.reshape(1, 1, -1).astype(np.float32)
        self.std = std.reshape(1, 1, -1).astype(np.float32)
        self.augment = augment
        self.seed = int(seed)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[index]
        patch = _extract_patch(self.feature_cube, int(sample["x"]), int(sample["y"]), self.patch_size)
        if self.augment:
            patch = _augment_patch(patch, random.Random(self.seed + index * 37))
        normalized = (patch.astype(np.float32) - self.mean) / self.std
        patch_chw = np.transpose(normalized, (2, 0, 1)).astype(np.float32)
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


class RadarTerritoryResNet(nn.Module):
    def __init__(self, input_channels: int, num_classes: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(32, 32, blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 192, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(192, num_classes)

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


def train_radar_territory_classifier(
    feature_cube: np.ndarray,
    labels: np.ndarray,
    active_mask: np.ndarray,
    output_path: str | Path,
    patch_size: int = 48,
    epochs: int = 8,
    batch_size: int = 32,
    learning_rate: float = 3e-4,
    max_samples_per_class: int = 420,
    seed: int = 42,
    sample_mask: np.ndarray | None = None,
) -> RadarTrainingResult:
    if feature_cube.shape[:2] != labels.shape:
        raise ValueError(f"Feature cube and labels must match, got {feature_cube.shape[:2]} vs {labels.shape}.")
    if active_mask.shape != labels.shape:
        raise ValueError(f"Active mask and labels must match, got {active_mask.shape} vs {labels.shape}.")
    if sample_mask is not None and sample_mask.shape != labels.shape:
        raise ValueError(f"Sample mask and labels must match, got {sample_mask.shape} vs {labels.shape}.")

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    effective_sample_mask = active_mask.astype(bool)
    if sample_mask is not None:
        effective_sample_mask &= sample_mask.astype(bool)

    samples = collect_radar_patch_catalog(
        labels=labels,
        active_mask=effective_sample_mask,
        patch_size=patch_size,
        max_samples_per_class=max_samples_per_class,
        seed=seed,
    )
    train_samples = [sample for sample in samples if sample["split"] == "train"]
    val_samples = [sample for sample in samples if sample["split"] == "val"]
    if not train_samples or not val_samples:
        raise RuntimeError("Not enough radar-aligned samples to train/validate the radar classifier.")

    mean, std = compute_feature_stats(feature_cube, active_mask=active_mask)
    train_dataset = RadarPatchDataset(feature_cube, train_samples, patch_size, mean, std, augment=True, seed=seed)
    val_dataset = RadarPatchDataset(feature_cube, val_samples, patch_size, mean, std, augment=False, seed=seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    class_hist = np.zeros((len(TERRITORY_CLASS_NAMES),), dtype=np.int64)
    for sample in train_samples:
        class_hist[int(sample["label"])] += 1
    weights = 1.0 / np.maximum(class_hist.astype(np.float32), 1.0)
    weights = weights / float(weights.mean())

    model = RadarTerritoryResNet(
        input_channels=int(feature_cube.shape[-1]),
        num_classes=len(TERRITORY_CLASS_NAMES),
    ).to(device)
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
        raise RuntimeError("Radar classifier training did not produce a valid checkpoint.")

    payload = {
        "state_dict": best_state,
        "class_names": list(TERRITORY_CLASS_NAMES),
        "feature_names": list(RADAR_FEATURE_NAMES),
        "patch_size": int(patch_size),
        "mean": mean,
        "std": std,
        "history": history,
        "class_histogram": {name: int(class_hist[idx]) for idx, name in enumerate(TERRITORY_CLASS_NAMES)},
        "train_samples": train_samples,
        "val_samples": val_samples,
    }
    torch.save(payload, output_path)
    return RadarTrainingResult(
        model_path=output_path,
        patch_size=patch_size,
        train_samples=len(train_samples),
        val_samples=len(val_samples),
        best_val_accuracy=float(best_val_accuracy),
        class_histogram={name: int(class_hist[idx]) for idx, name in enumerate(TERRITORY_CLASS_NAMES)},
        history=history,
    )


def load_radar_classifier(path: str | Path) -> tuple[RadarTerritoryResNet, dict]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = RadarTerritoryResNet(
        input_channels=len(payload["feature_names"]),
        num_classes=len(payload["class_names"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload


def predict_radar_probabilities(
    feature_cube: np.ndarray,
    checkpoint_path: str | Path,
    stride: int = 20,
    batch_size: int = 64,
) -> np.ndarray:
    model, payload = load_radar_classifier(checkpoint_path)
    mean = np.asarray(payload["mean"], dtype=np.float32).reshape(1, 1, -1)
    std = np.asarray(payload["std"], dtype=np.float32).reshape(1, 1, -1)
    patch_size = int(payload["patch_size"])
    window = cosine_window(patch_size)
    device = torch.device("cpu")

    height, width = feature_cube.shape[:2]
    class_count = len(payload["class_names"])
    accum = np.zeros((class_count, height, width), dtype=np.float32)
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
        patch = feature_cube[y : y + patch_size, x : x + patch_size].astype(np.float32)
        patch = (patch - mean) / std
        pending_tensors.append(np.transpose(patch, (2, 0, 1)).astype(np.float32))
        pending_meta.append((x, y))
        if len(pending_tensors) >= batch_size:
            flush()

    flush()
    weight_sum = np.maximum(weight_sum, 1e-6)
    return accum / weight_sum[None, :, :]


def _class_distribution(labels: np.ndarray, active_mask: np.ndarray) -> Dict[str, Dict[str, float | int]]:
    active_pixels = labels[active_mask] if np.any(active_mask) else labels.reshape(-1)
    total = max(1, int(active_pixels.size))
    report: Dict[str, Dict[str, float | int]] = {}
    for class_index, class_name in enumerate(TERRITORY_CLASS_NAMES):
        count = int(np.count_nonzero(active_pixels == class_index))
        report[class_name] = {
            "pixel_count": count,
            "pixel_share": float(count / total),
        }
    return report


def _cleanup_radar_probabilities(probabilities: np.ndarray, active_mask: np.ndarray) -> np.ndarray:
    cleaned = probabilities.astype(np.float32).copy()
    if "vehicle" in TERRITORY_CLASS_NAMES:
        vehicle_index = int(TERRITORY_CLASS_NAMES.index("vehicle"))
        labels = np.argmax(cleaned, axis=0).astype(np.uint8)
        vehicle_mask = ((labels == vehicle_index) & active_mask).astype(np.uint8)
        if int(vehicle_mask.sum()) > 0:
            num_labels, component_labels, stats, _ = cv2.connectedComponentsWithStats(vehicle_mask, connectivity=8)
            for component_id in range(1, num_labels):
                area = int(stats[component_id, cv2.CC_STAT_AREA])
                if area <= 256:
                    continue
                component = component_labels == component_id
                cleaned[vehicle_index, component] *= 0.02
    denominator = np.maximum(np.sum(cleaned, axis=0, keepdims=True), 1e-6)
    return cleaned / denominator


def render_radar_overlay(base_rgb: np.ndarray, color_map: np.ndarray, active_mask: np.ndarray, alpha: float = 0.46) -> np.ndarray:
    blended = base_rgb.astype(np.float32) * (1.0 - alpha) + color_map.astype(np.float32) * alpha
    output = blended.clip(0, 255).astype(np.uint8)
    output[~active_mask] = base_rgb[~active_mask]
    return output


def classify_radar_observation(
    observation: RadarObservation,
    checkpoint_path: str | Path,
    stride: int = 20,
    batch_size: int = 64,
) -> RadarInferenceResult:
    feature_cube = build_radar_feature_cube(observation)
    support_mask = observation.support_mask.astype(bool)
    probabilities = predict_radar_probabilities(feature_cube, checkpoint_path=checkpoint_path, stride=stride, batch_size=batch_size)
    probabilities = _cleanup_radar_probabilities(probabilities, support_mask)
    probabilities[:, ~support_mask] = 0.0
    class_map = np.argmax(probabilities, axis=0).astype(np.uint8)
    class_map[~support_mask] = np.uint8(255)

    color_map = render_territory_map(probabilities)
    color_map[~support_mask] = 0
    overlay = render_radar_overlay(
        base_rgb=observation.visualization_rgb,
        color_map=color_map,
        active_mask=support_mask,
    )
    report: Dict[str, object] = {
        "input_path": str(observation.source_path),
        "model_path": str(checkpoint_path),
        "origin_px": [float(observation.origin_px[0]), float(observation.origin_px[1])],
        "active_pixel_count": int(np.count_nonzero(support_mask)),
        "class_distribution": _class_distribution(np.argmax(probabilities, axis=0).astype(np.uint8), support_mask),
    }
    return RadarInferenceResult(
        class_names=TERRITORY_CLASS_NAMES,
        class_map=class_map,
        probabilities=probabilities.astype(np.float32),
        color_map=color_map.astype(np.uint8),
        overlay=overlay.astype(np.uint8),
        active_mask=support_mask,
        report=report,
    )
