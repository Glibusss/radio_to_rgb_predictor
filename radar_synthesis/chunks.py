"""Утилиты для нарезки парных и непарных обучающих патчей."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Window:
    """Описывает квадратное окно в координатах исходного изображения."""

    x: int
    y: int
    size: int


def ensure_dir(path: str | Path) -> Path:
    """Создает каталог, если его еще нет, и возвращает путь к нему."""

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def sliding_windows(height: int, width: int, patch_size: int, stride: int) -> List[Window]:
    """Строит сетку квадратных окон с гарантированным покрытием границ."""

    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive")
    if patch_size > height or patch_size > width:
        return []

    def axis_positions(length: int) -> List[int]:
        """Возвращает позиции начала окна вдоль одной оси."""

        positions = list(range(0, max(length - patch_size + 1, 1), stride))
        last = length - patch_size
        if not positions or positions[-1] != last:
            positions.append(last)
        return sorted(set(positions))

    xs = axis_positions(width)
    ys = axis_positions(height)
    return [Window(x=x, y=y, size=patch_size) for y in ys for x in xs]


def patch_content_score(image: np.ndarray) -> float:
    """Оценивает информативность патча по контрасту и средней яркости."""

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    gray_f32 = gray.astype(np.float32)
    return float(gray_f32.std() + 0.35 * gray_f32.mean())


def extract_patch(image: np.ndarray, window: Window) -> np.ndarray:
    """Возвращает копию патча, вырезанного по заданному окну."""

    return image[window.y : window.y + window.size, window.x : window.x + window.size].copy()


def split_train_val(items: Sequence[Window], val_ratio: float, seed: int) -> Tuple[List[Window], List[Window]]:
    """Перемешивает окна и делит их на обучающую и валидационную выборки."""

    items_list = list(items)
    random.Random(seed).shuffle(items_list)
    if not items_list:
        return [], []
    val_count = max(1, int(round(len(items_list) * val_ratio)))
    val_count = min(val_count, len(items_list) - 1) if len(items_list) > 1 else 1
    val_items = items_list[:val_count]
    train_items = items_list[val_count:]
    return train_items, val_items


def save_patch(path: str | Path, image: np.ndarray) -> None:
    """Сохраняет патч на диск с учетом цветности изображения."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), image_bgr)
    else:
        cv2.imwrite(str(path), image)


def apply_augmentation(image: np.ndarray, mode: str) -> np.ndarray:
    """Применяет одно из поддерживаемых геометрических преобразований."""

    if mode == "orig":
        return image.copy()
    if mode == "rot90":
        return np.rot90(image, 1).copy()
    if mode == "rot180":
        return np.rot90(image, 2).copy()
    if mode == "flip_lr":
        return np.fliplr(image).copy()
    raise ValueError(f"Unsupported augmentation mode: {mode}")


def export_paired_chunks(
    optical_image: np.ndarray,
    radar_image: np.ndarray,
    output_dir: str | Path,
    patch_size: int = 256,
    stride: int = 128,
    val_ratio: float = 0.15,
    min_content_score: float = 22.0,
    seed: int = 42,
    train_augmentations: Sequence[str] = ("orig", "rot90", "rot180", "flip_lr"),
) -> Dict[str, int]:
    """Экспортирует пары оптических и радарных патчей для обучения."""

    if optical_image.shape[:2] != radar_image.shape[:2]:
        raise ValueError("optical_image and radar_image must have identical spatial size")

    output_dir = ensure_dir(output_dir)
    windows = sliding_windows(optical_image.shape[0], optical_image.shape[1], patch_size, stride)

    kept_windows: List[Window] = []
    for window in windows:
        optical_patch = extract_patch(optical_image, window)
        radar_patch = extract_patch(radar_image, window)
        score = 0.55 * patch_content_score(optical_patch) + 0.45 * patch_content_score(radar_patch)
        if score >= min_content_score:
            kept_windows.append(window)

    train_windows, val_windows = split_train_val(kept_windows, val_ratio=val_ratio, seed=seed)

    metadata: List[Dict[str, int | str | float]] = []
    for split_name, split_windows in (("train", train_windows), ("val", val_windows)):
        optical_dir = ensure_dir(output_dir / split_name / "optical")
        radar_dir = ensure_dir(output_dir / split_name / "radar")
        for index, window in enumerate(split_windows):
            optical_patch = extract_patch(optical_image, window)
            radar_patch = extract_patch(radar_image, window)
            augmentations = train_augmentations if split_name == "train" else ("orig",)
            for augmentation in augmentations:
                suffix = "" if augmentation == "orig" else f"_{augmentation}"
                name = f"{index:05d}{suffix}.png"
                save_patch(optical_dir / name, apply_augmentation(optical_patch, augmentation))
                save_patch(radar_dir / name, apply_augmentation(radar_patch, augmentation))
                metadata.append(
                    {
                        "split": split_name,
                        "name": name,
                        "augmentation": augmentation,
                        "x": window.x,
                        "y": window.y,
                        "size": window.size,
                    }
                )

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)

    return {
        "candidate_windows": len(windows),
        "kept_windows": len(kept_windows),
        "train_windows": len(train_windows),
        "train_samples": len(train_windows) * len(train_augmentations),
        "val_windows": len(val_windows),
        "val_samples": len(val_windows),
    }


def export_unpaired_radar_bank(
    radar_image: np.ndarray,
    output_dir: str | Path,
    patch_size: int = 256,
    stride: int = 128,
    min_content_score: float = 18.0,
) -> Dict[str, int]:
    """Сохраняет банк информативных радарных патчей без парной оптики."""

    output_dir = ensure_dir(output_dir)
    windows = sliding_windows(radar_image.shape[0], radar_image.shape[1], patch_size, stride)

    saved_count = 0
    for window in windows:
        patch = extract_patch(radar_image, window)
        score = patch_content_score(patch)
        nonzero_ratio = float(np.count_nonzero(patch > 8)) / float(patch.size)
        if score < min_content_score or nonzero_ratio < 0.08:
            continue
        save_patch(output_dir / f"{saved_count:05d}.png", patch)
        saved_count += 1

    return {
        "candidate_windows": len(windows),
        "saved_windows": saved_count,
    }
