from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import cv2
import numpy as np


SPEED_OF_LIGHT_M_S = 299_792_458.0
_MIN_LINEAR_POWER = 1e-18


@dataclass(frozen=True)
class RadarEquationResult:
    received_power_w: np.ndarray
    received_power_dbw: np.ndarray
    heatmap: np.ndarray
    overlay: np.ndarray
    debug_maps: Mapping[str, np.ndarray]
    report: Mapping[str, object]
    config: Mapping[str, object]
    origin_px: tuple[float, float]


def _load_radar_config(path: str | Path) -> Dict[str, object]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["config_path"] = str(config_path)
    return payload


def _require_float(config: Mapping[str, object], key: str) -> float:
    if key not in config:
        raise ValueError(f"Radar config is missing '{key}'.")
    return float(config[key])


def _db_to_linear(values_db: np.ndarray | float) -> np.ndarray:
    return np.power(10.0, np.asarray(values_db, dtype=np.float32) / 10.0).astype(np.float32)


def _linear_to_db(values_linear: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(values_linear.astype(np.float32), _MIN_LINEAR_POWER))).astype(np.float32)


def _estimate_radar_origin(probabilities: np.ndarray, class_names: Sequence[str]) -> tuple[float, float]:
    height, width = probabilities.shape[1:]
    class_index = {name: idx for idx, name in enumerate(class_names)}
    weights = np.zeros((height, width), dtype=np.float32)
    if "building" in class_index:
        weights += 0.90 * probabilities[class_index["building"]]
    if "asphalt_road" in class_index:
        weights += 0.45 * probabilities[class_index["asphalt_road"]]
    if "dirt_road" in class_index:
        weights += 0.20 * probabilities[class_index["dirt_road"]]
    if "vehicle" in class_index:
        weights += 0.35 * probabilities[class_index["vehicle"]]
    weights = cv2.GaussianBlur(weights, (0, 0), sigmaX=9.0, sigmaY=9.0)
    total = float(weights.sum())
    if total < 1e-8:
        return width / 2.0, height / 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return float((xx * weights).sum() / total), float((yy * weights).sum() / total)


def _geometry(
    shape: tuple[int, int],
    origin_px: tuple[float, float],
    meters_per_pixel: float,
    antenna_height_m: float,
    reference_range_m: float,
) -> Dict[str, np.ndarray]:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx_m = (xx - origin_px[0]) * meters_per_pixel
    dy_m = (yy - origin_px[1]) * meters_per_pixel
    ground_offset_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)
    line_of_sight_ground_m = reference_range_m + ground_offset_m
    slant_range_m = np.sqrt(line_of_sight_ground_m * line_of_sight_ground_m + antenna_height_m * antenna_height_m)
    return {
        "dx_m": dx_m.astype(np.float32),
        "dy_m": dy_m.astype(np.float32),
        "ground_offset_m": ground_offset_m.astype(np.float32),
        "line_of_sight_ground_m": line_of_sight_ground_m.astype(np.float32),
        "slant_range_m": slant_range_m.astype(np.float32),
    }


def _render_grayscale(values_dbw: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values_dbw)
    if not np.any(finite):
        return np.zeros(values_dbw.shape + (3,), dtype=np.uint8)
    samples = values_dbw[finite]
    lower = float(np.percentile(samples, 2.0))
    upper = float(np.percentile(samples, 98.0))
    if upper - lower < 1e-6:
        upper = lower + 1e-6
    scaled = ((values_dbw - lower) / (upper - lower)).clip(0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def map_radar_equation_to_pixels(
    rgb_image: np.ndarray,
    class_names: Sequence[str],
    probabilities: np.ndarray,
    pixel_rcs_map_m2: np.ndarray,
    config_path: str | Path,
) -> RadarEquationResult:
    config = _load_radar_config(config_path)
    frequency_ghz = _require_float(config, "frequency_ghz")
    transmit_power_w = _require_float(config, "transmit_power_w")
    antenna_gain_db = _require_float(config, "antenna_gain_db")
    system_loss_db = _require_float(config, "system_loss_db")
    antenna_height_m = _require_float(config, "antenna_height_m")
    reference_range_m = _require_float(config, "reference_range_m")
    meters_per_pixel = _require_float(config, "meters_per_pixel")

    origin_px = _estimate_radar_origin(probabilities=probabilities, class_names=class_names)
    geometry = _geometry(
        shape=rgb_image.shape[:2],
        origin_px=origin_px,
        meters_per_pixel=meters_per_pixel,
        antenna_height_m=antenna_height_m,
        reference_range_m=reference_range_m,
    )

    wavelength_m = SPEED_OF_LIGHT_M_S / (frequency_ghz * 1e9)
    antenna_gain_linear = float(_db_to_linear(antenna_gain_db))
    system_loss_linear = float(_db_to_linear(system_loss_db))

    radar_constant = (
        transmit_power_w
        * antenna_gain_linear
        * antenna_gain_linear
        * (wavelength_m ** 2)
        / (((4.0 * np.pi) ** 3) * max(system_loss_linear, 1e-12))
    )
    received_power_w = (
        radar_constant
        * np.maximum(pixel_rcs_map_m2.astype(np.float32), 0.0)
        / np.maximum(geometry["slant_range_m"], 1.0) ** 4
    ).astype(np.float32)
    received_power_dbw = _linear_to_db(received_power_w)
    heatmap = _render_grayscale(received_power_dbw)
    overlay = (
        rgb_image.astype(np.float32) * 0.56 + heatmap.astype(np.float32) * 0.44
    ).clip(0, 255).astype(np.uint8)

    summary = {
        "min_received_power_dbw": float(received_power_dbw.min()),
        "max_received_power_dbw": float(received_power_dbw.max()),
        "mean_received_power_dbw": float(received_power_dbw.mean()),
        "sum_received_power_w": float(received_power_w.sum()),
        "wavelength_m": float(wavelength_m),
        "radar_constant": float(radar_constant),
        "config_path": str(config["config_path"]),
        "origin_px": [float(origin_px[0]), float(origin_px[1])],
    }
    report: Dict[str, object] = {
        "equation": "Pr = Pt * Gt * Gr * lambda^2 * sigma / ((4*pi)^3 * R^4 * L)",
        "summary": summary,
        "config": {
            "frequency_ghz": frequency_ghz,
            "transmit_power_w": transmit_power_w,
            "antenna_gain_db": antenna_gain_db,
            "system_loss_db": system_loss_db,
            "antenna_height_m": antenna_height_m,
            "reference_range_m": reference_range_m,
            "meters_per_pixel": meters_per_pixel,
        },
    }
    debug_maps: Dict[str, np.ndarray] = {
        "radar_slant_range_m": geometry["slant_range_m"].astype(np.float32),
        "radar_ground_offset_m": geometry["ground_offset_m"].astype(np.float32),
        "radar_received_power_dbw": received_power_dbw.astype(np.float32),
        "radar_equation_map_rgb": heatmap,
        "radar_equation_overlay_rgb": overlay,
    }

    return RadarEquationResult(
        received_power_w=received_power_w,
        received_power_dbw=received_power_dbw,
        heatmap=heatmap,
        overlay=overlay,
        debug_maps=debug_maps,
        report=report,
        config=config,
        origin_px=origin_px,
    )
