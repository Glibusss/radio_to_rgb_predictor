from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .common import normalize01, normalize_to_uint8


@dataclass(frozen=True)
class RadarReference:
    clean_gray: np.ndarray
    raw_green_response: np.ndarray
    style_background: np.ndarray
    style_artifacts: np.ndarray
    center_px: Tuple[float, float]
    radius_px: float


def extract_green_response(rgb_image: np.ndarray) -> np.ndarray:
    rgb_f32 = rgb_image.astype(np.float32)
    response = rgb_f32[:, :, 1] - 0.50 * rgb_f32[:, :, 0] - 0.50 * rgb_f32[:, :, 2]
    return normalize01(np.maximum(response, 0.0))


def estimate_display_circle(green_response: np.ndarray) -> tuple[Tuple[float, float], float]:
    threshold = max(0.08, float(np.quantile(green_response, 0.75)))
    mask = (green_response > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        height, width = green_response.shape
        return (width / 2.0, height / 2.0), min(height, width) * 0.45
    contour = max(contours, key=cv2.contourArea)
    center, radius = cv2.minEnclosingCircle(contour)
    return center, radius


def suppress_rings(green_response: np.ndarray, center_px: Tuple[float, float], radius_px: float) -> np.ndarray:
    height, width = green_response.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dist = np.sqrt((xx - center_px[0]) ** 2 + (yy - center_px[1]) ** 2)
    clipped_radius = max(int(radius_px), 1)
    radial_index = np.clip(dist.astype(np.int32), 0, clipped_radius)

    radial_profile = np.zeros((clipped_radius + 1,), dtype=np.float32)
    radial_count = np.zeros((clipped_radius + 1,), dtype=np.float32)
    np.add.at(radial_profile, radial_index, green_response)
    np.add.at(radial_count, radial_index, 1.0)
    radial_profile /= np.maximum(radial_count, 1.0)

    smooth_profile = cv2.GaussianBlur(radial_profile.reshape(-1, 1), (1, 0), sigmaX=0.0, sigmaY=4.0).reshape(-1)
    ring_component = np.maximum(radial_profile - smooth_profile, 0.0)
    cleaned = np.maximum(green_response - 0.85 * ring_component[radial_index], 0.0)
    cleaned[dist > radius_px * 0.96] = 0.0
    return cleaned


def extract_radar_reference(rgb_image: np.ndarray) -> RadarReference:
    green_response = extract_green_response(rgb_image)
    center_px, radius_px = estimate_display_circle(green_response)
    clean = suppress_rings(green_response, center_px, radius_px)

    raw_u8 = normalize_to_uint8(green_response)
    clean_u8 = cv2.medianBlur(normalize_to_uint8(clean), 3)
    raw_f32 = raw_u8.astype(np.float32) / 255.0

    floor_level = float(np.quantile(raw_f32, 0.86))
    style_background = np.minimum(raw_f32, floor_level)
    style_background = cv2.GaussianBlur(style_background, (0, 0), sigmaX=2.2, sigmaY=2.2)

    smooth = cv2.GaussianBlur(raw_f32, (0, 0), sigmaX=3.4, sigmaY=3.4)
    style_artifacts = np.maximum(raw_f32 - smooth, 0.0)
    style_artifacts = cv2.GaussianBlur(style_artifacts, (0, 0), sigmaX=1.0, sigmaY=1.0)

    return RadarReference(
        clean_gray=clean_u8,
        raw_green_response=raw_u8,
        style_background=normalize_to_uint8(style_background),
        style_artifacts=normalize_to_uint8(style_artifacts),
        center_px=center_px,
        radius_px=radius_px,
    )
