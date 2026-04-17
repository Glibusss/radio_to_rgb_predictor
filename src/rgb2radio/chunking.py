from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .common import ensure_dir, iter_windows


def _save_patch(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.ndim == 3:
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(path), image)


def _content_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    gray_f32 = gray.astype(np.float32)
    return float(gray_f32.std() + 0.30 * gray_f32.mean())


def _split_name(x: int, y: int, patch_size: int) -> str:
    return "val" if ((x // patch_size) + (y // patch_size)) % 5 == 0 else "train"


def export_paired_chunks(
    optical_image: np.ndarray,
    radar_image: np.ndarray,
    output_dir: str | Path,
    patch_size: int = 128,
    stride: int = 64,
    min_content_score: float = 20.0,
) -> Dict[str, int]:
    if optical_image.shape[:2] != radar_image.shape[:2]:
        raise ValueError("optical_image and radar_image must have identical spatial sizes.")

    output_dir = ensure_dir(output_dir)
    saved = {"train": 0, "val": 0}
    metadata: List[dict] = []

    for x, y in iter_windows(optical_image.shape[0], optical_image.shape[1], patch_size, stride):
        optical_patch = optical_image[y : y + patch_size, x : x + patch_size].copy()
        radar_patch = radar_image[y : y + patch_size, x : x + patch_size].copy()
        score = 0.55 * _content_score(optical_patch) + 0.45 * _content_score(radar_patch)
        if score < min_content_score:
            continue

        split = _split_name(x, y, patch_size)
        patch_id = saved[split]
        _save_patch(output_dir / split / "optical" / f"{patch_id:05d}.png", optical_patch)
        _save_patch(output_dir / split / "radar" / f"{patch_id:05d}.png", radar_patch)
        saved[split] += 1
        metadata.append({"split": split, "x": x, "y": y, "size": patch_size, "score": score})

    with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=True, indent=2)

    return {"train_samples": saved["train"], "val_samples": saved["val"]}


def export_unpaired_radar_chunks(
    radar_image: np.ndarray,
    output_dir: str | Path,
    patch_size: int = 128,
    stride: int = 64,
    min_content_score: float = 14.0,
) -> Dict[str, int]:
    output_dir = ensure_dir(output_dir)
    saved_count = 0
    for x, y in iter_windows(radar_image.shape[0], radar_image.shape[1], patch_size, stride):
        patch = radar_image[y : y + patch_size, x : x + patch_size].copy()
        if _content_score(patch) < min_content_score:
            continue
        non_zero_ratio = float(np.count_nonzero(patch > 8)) / float(patch.size)
        if non_zero_ratio < 0.06:
            continue
        _save_patch(output_dir / f"{saved_count:05d}.png", patch)
        saved_count += 1
    return {"saved_samples": saved_count}
