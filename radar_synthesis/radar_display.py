from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class RadarDisplayResult:
    clean_image: np.ndarray
    center_px: Tuple[float, float]
    radius_px: float


def read_rgb_image(path: str | Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def save_gray_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum - minimum < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values - minimum) / (maximum - minimum)


def normalize_to_uint8(values: np.ndarray) -> np.ndarray:
    return (normalize01(values) * 255.0).clip(0, 255).astype(np.uint8)


def extract_green_response(rgb_image: np.ndarray) -> np.ndarray:
    rgb_f32 = rgb_image.astype(np.float32)
    response = rgb_f32[:, :, 1] - 0.50 * rgb_f32[:, :, 0] - 0.50 * rgb_f32[:, :, 2]
    return normalize01(np.maximum(response, 0.0))


def estimate_display_circle(response: np.ndarray) -> Tuple[Tuple[float, float], float]:
    threshold = max(0.08, float(np.quantile(response, 0.75)))
    mask = (response > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        height, width = response.shape
        return (width / 2.0, height / 2.0), min(height, width) * 0.45

    contour = max(contours, key=cv2.contourArea)
    center, radius = cv2.minEnclosingCircle(contour)
    return center, radius


def suppress_radial_rings(response: np.ndarray, center_px: Tuple[float, float], radius_px: float) -> np.ndarray:
    height, width = response.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    center_x, center_y = center_px
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)

    clipped_radius = max(int(radius_px), 1)
    radial_index = np.clip(dist.astype(np.int32), 0, clipped_radius)
    radial_profile = np.zeros((clipped_radius + 1,), dtype=np.float32)
    radial_count = np.zeros((clipped_radius + 1,), dtype=np.float32)

    np.add.at(radial_profile, radial_index, response)
    np.add.at(radial_count, radial_index, 1.0)
    radial_profile /= np.maximum(radial_count, 1.0)

    smooth_profile = cv2.GaussianBlur(radial_profile.reshape(-1, 1), (1, 0), sigmaX=0.0, sigmaY=4.0).reshape(-1)
    ring_component = np.maximum(radial_profile - smooth_profile, 0.0)
    ring_lut = ring_component[radial_index]

    cleaned = np.maximum(response - 0.85 * ring_lut, 0.0)
    cleaned[dist > radius_px * 0.96] = 0.0
    return cleaned


def extract_clean_radar_reference(rgb_image: np.ndarray) -> RadarDisplayResult:
    response = extract_green_response(rgb_image)
    center_px, radius_px = estimate_display_circle(response)
    cleaned = suppress_radial_rings(response, center_px, radius_px)
    cleaned = cv2.medianBlur(normalize_to_uint8(cleaned), 3)
    return RadarDisplayResult(clean_image=cleaned, center_px=center_px, radius_px=radius_px)
