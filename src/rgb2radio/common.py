from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List

import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_existing_path(candidates: Iterable[str | Path]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"None of the candidate paths exists: {list(candidates)}")


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def save_gray(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    return (normalize01(values) * 255.0).clip(0, 255).astype(np.uint8)


def gray_from_rgb(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def gradient_magnitude(gray_u8: np.ndarray) -> np.ndarray:
    grad_x = cv2.Sobel(gray_u8, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_u8, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(grad_x * grad_x + grad_y * grad_y)


def corner_response(gray_u8: np.ndarray) -> np.ndarray:
    corners = cv2.cornerHarris(gray_u8.astype(np.float32), blockSize=2, ksize=3, k=0.04)
    corners = cv2.GaussianBlur(corners, (5, 5), 0.0)
    return np.maximum(corners, 0.0)


def local_variance(gray_f32: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    mean = cv2.GaussianBlur(gray_f32, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mean_sq = cv2.GaussianBlur(gray_f32 * gray_f32, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.maximum(mean_sq - mean * mean, 0.0)


def histogram_match_u8(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    source_u8 = source.astype(np.uint8)
    reference_u8 = reference.astype(np.uint8)
    source_hist = np.bincount(source_u8.ravel(), minlength=256).astype(np.float64)
    reference_hist = np.bincount(reference_u8.ravel(), minlength=256).astype(np.float64)
    source_cdf = np.cumsum(source_hist)
    reference_cdf = np.cumsum(reference_hist)
    if source_cdf[-1] <= 0 or reference_cdf[-1] <= 0:
        return source_u8
    source_cdf /= source_cdf[-1]
    reference_cdf /= reference_cdf[-1]
    mapping = np.interp(source_cdf, reference_cdf, np.arange(256))
    return mapping[source_u8].clip(0, 255).astype(np.uint8)


def sliding_positions(length: int, patch_size: int, stride: int) -> List[int]:
    if patch_size > length:
        return [0]
    positions = list(range(0, max(length - patch_size + 1, 1), stride))
    last = length - patch_size
    if not positions or positions[-1] != last:
        positions.append(last)
    return sorted(set(positions))


def iter_windows(height: int, width: int, patch_size: int, stride: int) -> Iterator[tuple[int, int]]:
    for y in sliding_positions(height, patch_size, stride):
        for x in sliding_positions(width, patch_size, stride):
            yield x, y


def cosine_window(size: int) -> np.ndarray:
    base = np.hanning(size).astype(np.float32)
    window = np.outer(base, base).astype(np.float32)
    window = np.maximum(window, 1e-4)
    return window / float(window.max())
