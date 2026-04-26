from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import cv2
import numpy as np


SPEED_OF_LIGHT_M_S = 299_792_458.0
_MIN_LINEAR_POWER = 1e-18
_SMOOTH_SURFACE_CLASSES = frozenset({"asphalt_road", "water"})
_ROUGH_SURFACE_CLASSES = frozenset({"dirt_road"})
_STRUCTURED_SURFACE_CLASSES = frozenset({"building", "building_shadow"})
_VOLUME_SCATTER_CLASSES = frozenset({"forest", "forest_shadow", "shrub"})
_POINT_SCATTER_CLASSES = frozenset({"vehicle"})


@dataclass(frozen=True)
class RadarEquationResult:
    received_power_w: np.ndarray
    received_power_dbw: np.ndarray
    heatmap: np.ndarray
    overlay: np.ndarray
    blind_zones_map: np.ndarray
    glint_map: np.ndarray
    glint_overlay: np.ndarray
    boundary_map: np.ndarray
    boundary_overlay: np.ndarray
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


def _kernel(size: int) -> np.ndarray:
    size = max(1, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _resolve_radar_origin(
    config: Mapping[str, object],
    shape: tuple[int, int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
) -> tuple[float, float]:
    height, width = shape
    origin_mode = str(config.get("origin_mode", "image_center")).lower()
    if origin_mode == "image_center":
        return width / 2.0, height / 2.0
    if origin_mode == "auto_scene":
        return _estimate_radar_origin(probabilities=probabilities, class_names=class_names)
    if origin_mode == "fixed_px":
        x = float(config.get("origin_x_px", width / 2.0))
        y = float(config.get("origin_y_px", height / 2.0))
        return x, y
    raise ValueError(f"Unsupported radar origin_mode: {origin_mode}")


def _geometry(
    shape: tuple[int, int],
    origin_px: tuple[float, float],
    meters_per_pixel: float,
    antenna_height_m: float,
    range_bias_m: float,
) -> Dict[str, np.ndarray]:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx_m = (xx - origin_px[0]) * meters_per_pixel
    dy_m = (yy - origin_px[1]) * meters_per_pixel
    ground_offset_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)
    line_of_sight_ground_m = float(range_bias_m) + ground_offset_m
    slant_range_m = np.sqrt(line_of_sight_ground_m * line_of_sight_ground_m + antenna_height_m * antenna_height_m)
    return {
        "dx_m": dx_m.astype(np.float32),
        "dy_m": dy_m.astype(np.float32),
        "ground_offset_m": ground_offset_m.astype(np.float32),
        "line_of_sight_ground_m": line_of_sight_ground_m.astype(np.float32),
        "slant_range_m": slant_range_m.astype(np.float32),
    }


def _apply_boundary_scatter_cleanup(
    pixel_rcs_map_m2: np.ndarray,
    support_region: np.ndarray,
) -> tuple[np.ndarray, Dict[str, object]]:
    effective_rcs_map_m2 = pixel_rcs_map_m2.astype(np.float32).copy()
    support_mask = support_region.astype(bool)
    outside_mask = ~support_mask
    outside_values = effective_rcs_map_m2[outside_mask]
    positive_outside_values = outside_values[np.isfinite(outside_values) & (outside_values > 0.0)]
    if positive_outside_values.size > 0:
        background_rcs_m2 = float(np.median(positive_outside_values))
    else:
        all_values = effective_rcs_map_m2[np.isfinite(effective_rcs_map_m2) & (effective_rcs_map_m2 > 0.0)]
        background_rcs_m2 = float(np.median(all_values)) if all_values.size > 0 else 0.0

    replaced_pixel_count = int(np.count_nonzero(outside_mask))
    if replaced_pixel_count > 0:
        effective_rcs_map_m2[outside_mask] = background_rcs_m2

    cleanup_report: Dict[str, object] = {
        "applied": bool(replaced_pixel_count > 0),
        "background_rcs_m2": background_rcs_m2,
        "replaced_pixel_count": replaced_pixel_count,
        "preserved_pixel_count": int(np.count_nonzero(support_mask)),
    }
    return effective_rcs_map_m2, cleanup_report


def _apply_receiver_noise(
    signal_power_w: np.ndarray,
    noise_floor_dbw: float,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    receiver_floor_w = float(np.asarray(_db_to_linear(noise_floor_dbw)).reshape(-1)[0])
    rng = np.random.default_rng(seed)
    noise_power_w = rng.exponential(scale=max(receiver_floor_w, _MIN_LINEAR_POWER), size=signal_power_w.shape).astype(
        np.float32
    )
    observed_power_w = (signal_power_w.astype(np.float32) + noise_power_w).astype(np.float32)
    report: Dict[str, object] = {
        "applied": True,
        "model": "thermal_noise_exponential_power",
        "noise_floor_dbw": float(noise_floor_dbw),
        "noise_floor_w": receiver_floor_w,
        "seed": None if seed is None else int(seed),
        "realized_mean_noise_w": float(noise_power_w.mean()),
        "realized_std_noise_w": float(noise_power_w.std()),
    }
    return observed_power_w, noise_power_w, report


def _render_positive_grayscale(
    values_linear: np.ndarray,
    floor_linear: float | None = None,
    white_linear: float | None = None,
) -> np.ndarray:
    positive = np.isfinite(values_linear) & (values_linear > 0.0)
    if not np.any(positive):
        return np.zeros(values_linear.shape + (3,), dtype=np.uint8)
    positive_values = values_linear[positive].astype(np.float32)
    reference_value = float(floor_linear) if floor_linear is not None else float(np.percentile(positive_values, 70.0))
    max_value = float(white_linear) if white_linear is not None else float(np.percentile(positive_values, 99.8))
    reference_value = max(reference_value, _MIN_LINEAR_POWER)
    max_value = max(max_value, reference_value * 1.01)
    scaled = np.zeros_like(values_linear, dtype=np.float32)
    numerator = np.log1p(np.maximum(values_linear[positive] - reference_value, 0.0) / reference_value)
    denominator = np.log1p(max(max_value - reference_value, reference_value) / reference_value)
    scaled[positive] = numerator / max(denominator, 1e-12)
    scaled = scaled.clip(0.0, 1.0)
    gray = (np.power(scaled, 0.72) * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _render_grayscale(values_dbw: np.ndarray, lower_dbw: float, upper_dbw: float) -> np.ndarray:
    finite = np.isfinite(values_dbw)
    if not np.any(finite):
        return np.zeros(values_dbw.shape + (3,), dtype=np.uint8)
    if upper_dbw - lower_dbw < 1e-6:
        upper_dbw = lower_dbw + 1e-6
    scaled = ((values_dbw - lower_dbw) / (upper_dbw - lower_dbw)).clip(0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _render_red_mask(mask: np.ndarray) -> np.ndarray:
    overlay = np.zeros(mask.shape + (3,), dtype=np.uint8)
    overlay[mask.astype(bool)] = np.array([255, 0, 0], dtype=np.uint8)
    return overlay


def _polar_angles_deg(dx_m: np.ndarray, dy_m: np.ndarray) -> np.ndarray:
    return ((np.degrees(np.arctan2(dy_m.astype(np.float32), dx_m.astype(np.float32))) + 360.0) % 360.0).astype(
        np.float32
    )


def _polar_bilinear_coordinates(
    range_m: np.ndarray,
    angle_deg: np.ndarray,
    range_sample_m: float,
    angular_sample_deg: float,
    num_range_bins: int,
    num_angle_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    safe_range_sample_m = max(float(range_sample_m), 1e-6)
    safe_angular_sample_deg = max(float(angular_sample_deg), 1e-6)
    range_coord = (range_m.astype(np.float32) / safe_range_sample_m).astype(np.float32)
    angle_coord = (angle_deg.astype(np.float32) / safe_angular_sample_deg).astype(np.float32)
    range_floor = np.floor(range_coord).astype(np.int32)
    angle_floor = np.floor(angle_coord).astype(np.int32) % int(num_angle_bins)
    r0 = np.clip(range_floor, 0, int(num_range_bins) - 1)
    r1 = np.clip(range_floor + 1, 0, int(num_range_bins) - 1)
    a0 = angle_floor
    a1 = (a0 + 1) % int(num_angle_bins)
    wr1 = (range_coord - range_floor).astype(np.float32)
    wr0 = (1.0 - wr1).astype(np.float32)
    wa1 = (angle_coord - np.floor(angle_coord)).astype(np.float32)
    wa0 = (1.0 - wa1).astype(np.float32)
    return a0, a1, r0, r1, wa0, wa1, wr0, wr1


def _splat_to_polar(
    values: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    r0: np.ndarray,
    r1: np.ndarray,
    wa0: np.ndarray,
    wa1: np.ndarray,
    wr0: np.ndarray,
    wr1: np.ndarray,
    shape: tuple[int, int],
) -> np.ndarray:
    polar_grid = np.zeros(shape, dtype=np.float32)
    sample_values = values.astype(np.float32)
    np.add.at(polar_grid, (a0, r0), sample_values * wa0 * wr0)
    np.add.at(polar_grid, (a1, r0), sample_values * wa1 * wr0)
    np.add.at(polar_grid, (a0, r1), sample_values * wa0 * wr1)
    np.add.at(polar_grid, (a1, r1), sample_values * wa1 * wr1)
    return polar_grid


def _sample_from_polar(
    polar_grid: np.ndarray,
    a0: np.ndarray,
    a1: np.ndarray,
    r0: np.ndarray,
    r1: np.ndarray,
    wa0: np.ndarray,
    wa1: np.ndarray,
    wr0: np.ndarray,
    wr1: np.ndarray,
) -> np.ndarray:
    sampled = (
        polar_grid[a0, r0] * wa0 * wr0
        + polar_grid[a1, r0] * wa1 * wr0
        + polar_grid[a0, r1] * wa0 * wr1
        + polar_grid[a1, r1] * wa1 * wr1
    )
    return sampled.astype(np.float32)


def _convolve_polar_response(polar_source: np.ndarray, angle_kernel: np.ndarray, range_kernel: np.ndarray) -> np.ndarray:
    polar_response = np.zeros_like(polar_source, dtype=np.float32)
    angle_center = len(angle_kernel) // 2
    for kernel_index, kernel_weight in enumerate(angle_kernel.tolist()):
        shift = kernel_index - angle_center
        if abs(kernel_weight) < 1e-9:
            continue
        polar_response += float(kernel_weight) * np.roll(polar_source, shift=shift, axis=0)
    return cv2.filter2D(
        polar_response,
        ddepth=-1,
        kernel=range_kernel[np.newaxis, :],
        borderType=cv2.BORDER_CONSTANT,
    )


def _convolve_polar_response_complex(
    polar_source: np.ndarray,
    angle_kernel: np.ndarray,
    range_kernel: np.ndarray,
) -> np.ndarray:
    response_real = np.zeros(polar_source.shape, dtype=np.float32)
    response_imag = np.zeros(polar_source.shape, dtype=np.float32)
    angle_center = len(angle_kernel) // 2
    source_real = polar_source.real.astype(np.float32)
    source_imag = polar_source.imag.astype(np.float32)
    for kernel_index, kernel_weight in enumerate(angle_kernel.tolist()):
        shift = kernel_index - angle_center
        if abs(kernel_weight) < 1e-9:
            continue
        response_real += float(kernel_weight) * np.roll(source_real, shift=shift, axis=0)
        response_imag += float(kernel_weight) * np.roll(source_imag, shift=shift, axis=0)
    filtered_real = cv2.filter2D(
        response_real,
        ddepth=-1,
        kernel=range_kernel[np.newaxis, :],
        borderType=cv2.BORDER_CONSTANT,
    )
    filtered_imag = cv2.filter2D(
        response_imag,
        ddepth=-1,
        kernel=range_kernel[np.newaxis, :],
        borderType=cv2.BORDER_CONSTANT,
    )
    return (filtered_real + 1j * filtered_imag).astype(np.complex64)


def _build_building_blind_zone_mask(
    class_map: np.ndarray,
    class_names: Sequence[str],
    geometry: Mapping[str, np.ndarray],
    scope_region: np.ndarray,
    angular_resolution_deg: float,
    range_resolution_m: float,
    meters_per_pixel: float,
) -> tuple[np.ndarray, Dict[str, object]]:
    scope_mask = scope_region.astype(bool)
    report: Dict[str, object] = {
        "applied": False,
        "occluder_classes": ["building"],
        "scope_pixel_count": int(np.count_nonzero(scope_mask)),
        "building_pixel_count": 0,
        "blind_pixel_count": 0,
        "blind_pixel_share": 0.0,
    }
    if "building" not in class_names or not np.any(scope_mask):
        return np.zeros(class_map.shape, dtype=bool), report

    building_mask_raw = (class_map == class_names.index("building")) & scope_mask
    if not np.any(building_mask_raw):
        return np.zeros(class_map.shape, dtype=bool), report

    angular_sample_deg = max(float(angular_resolution_deg) / 4.0, 0.1)
    num_angle_bins = max(32, int(np.ceil(360.0 / angular_sample_deg)))
    full_angles_deg = _polar_angles_deg(
        dx_m=geometry["dx_m"].astype(np.float32),
        dy_m=geometry["dy_m"].astype(np.float32),
    )
    full_ranges_m = geometry["slant_range_m"].astype(np.float32)
    full_angle_bins = np.floor(full_angles_deg / angular_sample_deg).astype(np.int32) % num_angle_bins
    full_ranges_m = full_ranges_m.astype(np.float32)
    origin_exclusion_px = max(10.0, 2.0 * float(range_resolution_m) / max(float(meters_per_pixel), 1e-6))
    blind_zone_mask = np.zeros(class_map.shape, dtype=bool)
    component_count, component_labels, component_stats, _ = cv2.connectedComponentsWithStats(
        building_mask_raw.astype(np.uint8),
        connectivity=8,
    )
    support_expand_bins = max(1, int(np.ceil(float(angular_resolution_deg) / max(angular_sample_deg, 1e-6))) // 2)
    range_guard_m = max(float(range_resolution_m) * 0.5, float(meters_per_pixel))
    kept_component_count = 0
    component_reports: list[Dict[str, object]] = []
    for label in range(1, component_count):
        area_px = int(component_stats[label, cv2.CC_STAT_AREA])
        if area_px <= 0:
            continue
        component_mask = component_labels == label
        ys, xs = np.nonzero(component_mask)
        component_min_distance_px = float(
            np.min(
                np.hypot(
                    geometry["dx_m"][ys, xs].astype(np.float32) / max(float(meters_per_pixel), 1e-6),
                    geometry["dy_m"][ys, xs].astype(np.float32) / max(float(meters_per_pixel), 1e-6),
                )
            )
        )
        if component_min_distance_px < origin_exclusion_px:
            component_reports.append(
                {
                    "label": int(label),
                    "area_px": area_px,
                    "shadow_pixel_count": 0,
                    "angular_bin_count": 0,
                    "skipped_near_origin": True,
                    "min_distance_px": component_min_distance_px,
                }
            )
            continue
        component_angles_deg = _polar_angles_deg(
            dx_m=geometry["dx_m"][ys, xs].astype(np.float32),
            dy_m=geometry["dy_m"][ys, xs].astype(np.float32),
        )
        component_ranges_m = geometry["slant_range_m"][ys, xs].astype(np.float32)
        component_angle_bins = np.floor(component_angles_deg / angular_sample_deg).astype(np.int32) % num_angle_bins
        component_support = np.zeros(num_angle_bins, dtype=bool)
        component_back_range_m = np.full(num_angle_bins, -1.0, dtype=np.float32)
        component_support[component_angle_bins] = True
        np.maximum.at(component_back_range_m, component_angle_bins, component_ranges_m)

        expanded_support = component_support.copy()
        expanded_back_range_m = component_back_range_m.copy()
        for shift in range(1, support_expand_bins + 1):
            rolled_support_pos = np.roll(component_support, shift)
            rolled_support_neg = np.roll(component_support, -shift)
            expanded_support |= rolled_support_pos | rolled_support_neg
            expanded_back_range_m = np.maximum(expanded_back_range_m, np.where(rolled_support_pos, np.roll(component_back_range_m, shift), -1.0))
            expanded_back_range_m = np.maximum(expanded_back_range_m, np.where(rolled_support_neg, np.roll(component_back_range_m, -shift), -1.0))

        component_shadow = (
            scope_mask
            & expanded_support[full_angle_bins]
            & (full_ranges_m > (expanded_back_range_m[full_angle_bins] + range_guard_m))
        )
        component_shadow &= ~component_mask
        if not np.any(component_shadow):
            continue
        blind_zone_mask |= component_shadow
        kept_component_count += 1
        component_reports.append(
            {
                "label": int(label),
                "area_px": area_px,
                "shadow_pixel_count": int(np.count_nonzero(component_shadow)),
                "angular_bin_count": int(np.count_nonzero(component_support)),
                "skipped_near_origin": False,
                "min_distance_px": component_min_distance_px,
            }
        )

    close_kernel_px = max(3, int(np.ceil(max(float(range_resolution_m) / max(float(meters_per_pixel), 1e-6), 1.0) / 6.0)))
    blind_zone_mask = cv2.morphologyEx(
        blind_zone_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        _kernel(close_kernel_px),
        iterations=1,
    ) > 0
    blind_zone_mask &= scope_mask
    blind_zone_mask &= ~building_mask_raw

    report = {
        "applied": True,
        "occluder_classes": ["building"],
        "scope_pixel_count": int(np.count_nonzero(scope_mask)),
        "building_pixel_count": int(np.count_nonzero(building_mask_raw)),
        "blind_pixel_count": int(np.count_nonzero(blind_zone_mask)),
        "blind_pixel_share": float(np.count_nonzero(blind_zone_mask) / max(blind_zone_mask.size, 1)),
        "angular_sample_deg": float(angular_sample_deg),
        "support_expand_bins": int(support_expand_bins),
        "range_guard_m": float(range_guard_m),
        "origin_exclusion_px": float(origin_exclusion_px),
        "building_component_count": int(max(component_count - 1, 0)),
        "shadow_component_count": int(kept_component_count),
        "component_reports": component_reports[:12],
    }
    return blind_zone_mask.astype(bool), report


def _normalize_feature_map(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    array = values.astype(np.float32)
    valid_mask = np.isfinite(array)
    if mask is not None:
        valid_mask &= mask.astype(bool)
    positive = valid_mask & (array > 0.0)
    if not np.any(positive):
        return np.zeros_like(array, dtype=np.float32)
    scale = float(np.percentile(array[positive], 99.0))
    if scale <= 1e-12:
        scale = float(array[positive].max())
    if scale <= 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return (array / scale).clip(0.0, 1.0).astype(np.float32)


def _peak_map(values: np.ndarray, kernel_size: int, mask: np.ndarray | None = None) -> np.ndarray:
    normalized = values.astype(np.float32)
    dilated = cv2.dilate(normalized, _kernel(kernel_size))
    peaks = (normalized > 0.0) & np.isclose(normalized, dilated, atol=1e-6)
    if mask is not None:
        peaks &= mask.astype(bool)
    return (normalized * peaks.astype(np.float32)).astype(np.float32)


def _build_scattering_gain_map(
    rgb_image: np.ndarray,
    class_map: np.ndarray,
    class_names: Sequence[str],
    geometry: Mapping[str, np.ndarray],
    antenna_height_m: float,
) -> tuple[np.ndarray, Dict[str, object], Dict[str, np.ndarray]]:
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gray_gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
    gray_gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    texture_gradient = cv2.GaussianBlur(np.sqrt(gray_gx * gray_gx + gray_gy * gray_gy), (0, 0), sigmaX=1.2, sigmaY=1.2)
    local_mean = cv2.GaussianBlur(gray_image, (0, 0), sigmaX=3.0, sigmaY=3.0)
    local_square_mean = cv2.GaussianBlur(gray_image * gray_image, (0, 0), sigmaX=3.0, sigmaY=3.0)
    local_variance = np.maximum(local_square_mean - local_mean * local_mean, 0.0).astype(np.float32)
    roughness_map = _normalize_feature_map(texture_gradient)
    volume_texture_map = _normalize_feature_map(local_variance)
    roughness_peak_map = _peak_map(roughness_map, kernel_size=3)
    volume_peak_map = _peak_map(volume_texture_map, kernel_size=5)

    ground_offset_m = np.maximum(geometry["ground_offset_m"].astype(np.float32), 1e-6)
    look_unit_x = (-geometry["dx_m"].astype(np.float32) / ground_offset_m).astype(np.float32)
    look_unit_y = (-geometry["dy_m"].astype(np.float32) / ground_offset_m).astype(np.float32)
    look_unit_x[geometry["ground_offset_m"] <= 1e-6] = 0.0
    look_unit_y[geometry["ground_offset_m"] <= 1e-6] = 0.0

    incidence_cos = (
        float(antenna_height_m) / np.maximum(geometry["slant_range_m"].astype(np.float32), float(antenna_height_m))
    ).clip(0.0, 1.0)
    incidence_term = np.sqrt(incidence_cos).astype(np.float32)
    grazing_term = np.sqrt(np.maximum(1.0 - incidence_cos, 0.0)).astype(np.float32)

    class_index = {name: idx for idx, name in enumerate(class_names)}
    road_like_mask = np.zeros(class_map.shape, dtype=bool)
    for road_class in ("asphalt_road", "dirt_road"):
        if road_class in class_index:
            road_like_mask |= class_map == class_index[road_class]
    road_neighborhood = cv2.dilate(road_like_mask.astype(np.uint8), _kernel(7), iterations=1).astype(bool)

    gain_map = np.ones(class_map.shape, dtype=np.float32)
    diffuse_gain_map = np.zeros(class_map.shape, dtype=np.float32)
    point_gain_map = np.zeros(class_map.shape, dtype=np.float32)
    edge_facing_map = np.zeros(class_map.shape, dtype=np.float32)
    corner_strength_map = np.zeros(class_map.shape, dtype=np.float32)
    edge_peak_map = np.zeros(class_map.shape, dtype=np.float32)
    roughness_peak_class_map = np.zeros(class_map.shape, dtype=np.float32)
    corner_peak_map = np.zeros(class_map.shape, dtype=np.float32)
    dihedral_peak_map = np.zeros(class_map.shape, dtype=np.float32)
    class_reports: Dict[str, Dict[str, object]] = {}

    for class_index, class_name in enumerate(class_names):
        mask = class_map == class_index
        if not np.any(mask):
            class_reports[class_name] = {
                "pixel_count": 0,
                "mean_gain": 0.0,
                "mean_diffuse_gain": 0.0,
                "mean_point_gain": 0.0,
                "max_gain": 0.0,
                "scattering_regime": "absent",
            }
            continue

        smoothed_mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=1.0, sigmaY=1.0)
        mask_gx = cv2.Sobel(smoothed_mask, cv2.CV_32F, 1, 0, ksize=3)
        mask_gy = cv2.Sobel(smoothed_mask, cv2.CV_32F, 0, 1, ksize=3)
        mask_edge_magnitude = np.sqrt(mask_gx * mask_gx + mask_gy * mask_gy)
        edge_strength = _normalize_feature_map(mask_edge_magnitude, mask=mask | (mask_edge_magnitude > 0.0))
        edge_norm = np.maximum(mask_edge_magnitude, 1e-6)
        normal_x = (mask_gx / edge_norm).astype(np.float32)
        normal_y = (mask_gy / edge_norm).astype(np.float32)
        edge_facing = (edge_strength * np.abs(normal_x * look_unit_x + normal_y * look_unit_y)).clip(0.0, 1.0).astype(np.float32)
        edge_facing_map[mask] = edge_facing[mask]
        edge_peak = _peak_map(edge_facing, kernel_size=5, mask=mask)
        edge_peak_map[mask] = edge_peak[mask]
        roughness_peak_class = _peak_map(roughness_map, kernel_size=3, mask=mask)
        roughness_peak_class_map[mask] = roughness_peak_class[mask]
        volume_peak_class = _peak_map(volume_texture_map, kernel_size=5, mask=mask)

        corner_strength = np.zeros(class_map.shape, dtype=np.float32)
        corner_peak = np.zeros(class_map.shape, dtype=np.float32)
        if class_name in _STRUCTURED_SURFACE_CLASSES or class_name in _POINT_SCATTER_CLASSES:
            corner_raw = cv2.cornerHarris(smoothed_mask.astype(np.float32), blockSize=2, ksize=3, k=0.04)
            corner_strength = _normalize_feature_map(np.maximum(corner_raw, 0.0), mask=mask)
            corner_strength_map[mask] = corner_strength[mask]
            corner_peak = _peak_map(corner_strength, kernel_size=3, mask=mask)
            corner_peak_map[mask] = corner_peak[mask]

        roughness_term = (roughness_map * incidence_term).astype(np.float32)
        dihedral_peak = np.zeros(class_map.shape, dtype=np.float32)
        if class_name in _SMOOTH_SURFACE_CLASSES:
            smooth_surface_scale = 0.15 if class_name == "water" else 1.0
            diffuse_values = np.clip(
                smooth_surface_scale
                * (
                    0.015 * incidence_term[mask]
                    + 0.040 * np.sqrt(np.maximum(roughness_term[mask], 0.0))
                    + 0.030 * volume_texture_map[mask]
                ),
                0.0,
                0.16,
            )
            point_values = np.clip(
                smooth_surface_scale
                * (
                    grazing_term[mask] * (1.35 * roughness_peak_class[mask] + 0.75 * edge_peak[mask])
                    + 0.25 * edge_facing[mask]
                ),
                0.0,
                1.60,
            )
            regime = "smooth_surface_microfacet"
        elif class_name in _ROUGH_SURFACE_CLASSES:
            diffuse_values = np.clip(
                0.10
                + 0.28 * np.sqrt(np.maximum(roughness_term[mask], 0.0))
                + 0.18 * volume_texture_map[mask]
                + 0.08 * edge_facing[mask],
                0.0,
                0.95,
            )
            point_values = np.clip(
                grazing_term[mask] * (0.55 * roughness_peak_class[mask] + 0.22 * edge_peak[mask])
                + 0.15 * volume_peak_class[mask],
                0.0,
                0.90,
            )
            regime = "rough_surface_diffuse"
        elif class_name in _STRUCTURED_SURFACE_CLASSES:
            dihedral_support = mask & road_neighborhood
            dihedral_raw = (
                edge_facing
                * dihedral_support.astype(np.float32)
                * np.maximum(grazing_term, 0.2).astype(np.float32)
            )
            dihedral_peak = _peak_map(dihedral_raw, kernel_size=5, mask=dihedral_support)
            dihedral_peak_map[mask] = dihedral_peak[mask]
            diffuse_values = np.clip(
                0.03
                + 0.06 * np.sqrt(np.maximum(roughness_term[mask], 0.0))
                + 0.08 * edge_facing[mask]
                + 0.05 * volume_texture_map[mask],
                0.0,
                0.36,
            )
            point_values = np.clip(
                1.25 * corner_peak[mask]
                + 1.10 * dihedral_peak[mask]
                + 0.35 * edge_peak[mask]
                + 0.10 * roughness_peak_class[mask],
                0.0,
                2.40,
            )
            regime = "structured_surface_corner_dihedral"
        elif class_name in _VOLUME_SCATTER_CLASSES:
            diffuse_values = np.clip(
                0.20
                + 0.34 * incidence_term[mask]
                + 0.28 * np.sqrt(np.maximum(volume_texture_map[mask], 0.0))
                + 0.14 * np.sqrt(np.maximum(roughness_term[mask], 0.0)),
                0.0,
                1.05,
            )
            point_values = np.clip(
                0.06 * roughness_peak_class[mask] + 0.03 * volume_peak_class[mask] + 0.02 * edge_peak[mask],
                0.0,
                0.14,
            )
            regime = "volume_scatter_textured"
        elif class_name in _POINT_SCATTER_CLASSES:
            diffuse_values = np.full(mask.sum(), 0.0, dtype=np.float32)
            point_values = np.clip(1.20 + 0.90 * corner_peak[mask] + 0.30 * edge_peak[mask], 1.0, 2.40)
            regime = "point_target"
        else:
            diffuse_values = np.clip(
                0.10 + 0.24 * np.sqrt(np.maximum(roughness_term[mask], 0.0)) + 0.14 * volume_texture_map[mask],
                0.0,
                0.75,
            )
            point_values = np.clip(0.35 * roughness_peak_class[mask] + 0.18 * edge_peak[mask], 0.0, 0.60)
            regime = "generic_surface"

        gain_values = np.clip(diffuse_values + 0.18 * point_values, 0.0, 1.45)
        gain_map[mask] = gain_values.astype(np.float32)
        diffuse_gain_map[mask] = diffuse_values.astype(np.float32)
        point_gain_map[mask] = point_values.astype(np.float32)
        class_reports[class_name] = {
            "pixel_count": int(mask.sum()),
            "mean_gain": float(gain_values.mean()),
            "max_gain": float(gain_values.max()),
            "mean_diffuse_gain": float(diffuse_values.mean()),
            "mean_point_gain": float(point_values.mean()),
            "mean_edge_facing": float(edge_facing[mask].mean()),
            "mean_edge_peak": float(edge_peak[mask].mean()),
            "mean_roughness": float(roughness_map[mask].mean()),
            "mean_roughness_peak": float(roughness_peak_class[mask].mean()),
            "mean_volume_texture": float(volume_texture_map[mask].mean()),
            "mean_incidence_term": float(incidence_term[mask].mean()),
            "scattering_regime": regime,
        }

    report: Dict[str, object] = {
        "model": "class_regime_diffuse_point_separation",
        "class_reports": class_reports,
        "global_mean_gain": float(gain_map.mean()),
        "global_max_gain": float(gain_map.max()),
        "global_mean_diffuse_gain": float(diffuse_gain_map.mean()),
        "global_mean_point_gain": float(point_gain_map.mean()),
        "global_mean_incidence_term": float(incidence_term.mean()),
        "global_mean_roughness": float(roughness_map.mean()),
    }
    debug_maps = {
        "gain_map": gain_map.astype(np.float32),
        "diffuse_gain_map": diffuse_gain_map.astype(np.float32),
        "point_gain_map": point_gain_map.astype(np.float32),
        "roughness_map": roughness_map.astype(np.float32),
        "roughness_peak_map": roughness_peak_map.astype(np.float32),
        "volume_texture_map": volume_texture_map.astype(np.float32),
        "volume_peak_map": volume_peak_map.astype(np.float32),
        "incidence_term": incidence_term.astype(np.float32),
        "grazing_term": grazing_term.astype(np.float32),
        "edge_facing_map": edge_facing_map.astype(np.float32),
        "edge_peak_map": edge_peak_map.astype(np.float32),
        "corner_strength_map": corner_strength_map.astype(np.float32),
        "corner_peak_map": corner_peak_map.astype(np.float32),
        "dihedral_peak_map": dihedral_peak_map.astype(np.float32),
    }
    return gain_map.astype(np.float32), report, debug_maps


def _keep_central_boundary_cluster(
    boundary_mask: np.ndarray,
    origin_px: tuple[float, float],
    center_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object]]:
    binary_mask = boundary_mask.astype(np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    if component_count <= 1:
        cluster_report: Dict[str, object] = {
            "component_count": 0,
            "center_radius_px": float(center_radius_px),
            "selection_mode": "empty",
            "selected_component": None,
            "kept_pixel_count": 0,
        }
        return (
            boundary_mask.astype(bool),
            np.zeros_like(boundary_mask, dtype=bool),
            boundary_mask.astype(bool),
            cluster_report,
        )

    components = []
    for label in range(1, component_count):
        area_px = int(stats[label, cv2.CC_STAT_AREA])
        if area_px <= 0:
            continue
        centroid_x = float(centroids[label][0])
        centroid_y = float(centroids[label][1])
        centroid_distance_px = float(np.hypot(centroid_x - origin_px[0], centroid_y - origin_px[1]))
        components.append(
            {
                "label": int(label),
                "area_px": area_px,
                "centroid_px": [centroid_x, centroid_y],
                "centroid_distance_px": centroid_distance_px,
            }
        )

    if not components:
        cluster_report = {
            "component_count": 0,
            "center_radius_px": float(center_radius_px),
            "selection_mode": "empty",
            "selected_component": None,
            "kept_pixel_count": 0,
        }
        return (
            np.zeros_like(boundary_mask, dtype=bool),
            np.zeros_like(boundary_mask, dtype=bool),
            np.zeros_like(boundary_mask, dtype=bool),
            cluster_report,
        )

    central_candidates = [
        component for component in components if component["centroid_distance_px"] <= float(center_radius_px)
    ]
    if central_candidates:
        selected_component = max(
            central_candidates,
            key=lambda component: (component["area_px"], -component["centroid_distance_px"]),
        )
        selection_mode = "largest_within_center_radius"
    else:
        selected_component = max(
            components,
            key=lambda component: (
                component["area_px"] / max(component["centroid_distance_px"], 1.0),
                component["area_px"],
            ),
        )
        selection_mode = "best_area_distance_score"

    selected_label = int(selected_component["label"])
    primary_mask = labels == selected_label
    sealed_primary_mask = (
        cv2.morphologyEx(primary_mask.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(5), iterations=1) > 0
    )
    free_space = (~sealed_primary_mask).astype(np.uint8)
    background_count, background_labels = cv2.connectedComponents(free_space, connectivity=8)
    if background_count > 0:
        border_labels = np.unique(
            np.concatenate(
                [
                    background_labels[0, :],
                    background_labels[-1, :],
                    background_labels[:, 0],
                    background_labels[:, -1],
                ]
            )
        )
        enclosed_region = free_space.astype(bool) & ~np.isin(background_labels, border_labels)
    else:
        enclosed_region = np.zeros_like(boundary_mask, dtype=bool)

    yy, xx = np.nonzero(primary_mask)
    if xx.size >= 3:
        hull_points = cv2.convexHull(np.stack([xx, yy], axis=1).astype(np.int32))
        support_region = np.zeros_like(boundary_mask, dtype=np.uint8)
        cv2.fillConvexPoly(support_region, hull_points, 1)
        support_region = cv2.dilate(support_region, _kernel(9), iterations=1) > 0
    else:
        support_region = sealed_primary_mask.copy()

    height, width = boundary_mask.shape
    kept_labels = {selected_label}
    retained_inner_components = []
    for component in components:
        label = int(component["label"])
        if label == selected_label:
            continue
        component_mask = labels == label
        overlap_inside_px = int(np.count_nonzero(component_mask & enclosed_region))
        centroid_x, centroid_y = component["centroid_px"]
        cx = int(np.clip(round(centroid_x), 0, width - 1))
        cy = int(np.clip(round(centroid_y), 0, height - 1))
        centroid_in_support = bool(support_region[cy, cx])
        support_overlap_px = int(np.count_nonzero(component_mask & support_region))
        support_overlap_ratio = float(support_overlap_px / max(int(component["area_px"]), 1))
        if overlap_inside_px <= 0 and not centroid_in_support and support_overlap_ratio < 0.5:
            continue
        kept_labels.add(label)
        retained_component = dict(component)
        retained_component["overlap_inside_px"] = overlap_inside_px
        retained_component["support_overlap_px"] = support_overlap_px
        retained_component["support_overlap_ratio"] = support_overlap_ratio
        retained_component["centroid_in_support"] = centroid_in_support
        retained_inner_components.append(retained_component)

    kept_mask = np.isin(labels, np.array(sorted(kept_labels), dtype=np.int32))
    top_components = sorted(components, key=lambda component: component["area_px"], reverse=True)[:5]
    cluster_report = {
        "component_count": len(components),
        "center_radius_px": float(center_radius_px),
        "selection_mode": selection_mode,
        "selected_component": selected_component,
        "enclosed_region_pixel_count": int(enclosed_region.sum()),
        "support_region_pixel_count": int(support_region.sum()),
        "kept_component_count": int(len(kept_labels)),
        "kept_inner_component_count": int(len(retained_inner_components)),
        "kept_component_labels": [int(label) for label in sorted(kept_labels)],
        "retained_inner_components": retained_inner_components,
        "kept_pixel_count": int(kept_mask.sum()),
        "top_components": top_components,
    }
    return kept_mask.astype(bool), support_region.astype(bool), primary_mask.astype(bool), cluster_report


def _estimate_primary_cluster_region(primary_boundary_mask: np.ndarray, fallback_region: np.ndarray) -> np.ndarray:
    if not np.any(primary_boundary_mask):
        return fallback_region.astype(bool)

    closed_boundary = cv2.morphologyEx(
        primary_boundary_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        _kernel(9),
        iterations=2,
    )
    contours, _ = cv2.findContours(closed_boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_region.astype(bool)

    filled_region = np.zeros_like(closed_boundary, dtype=np.uint8)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(filled_region, [largest_contour], -1, 1, thickness=cv2.FILLED)
    filled_region = cv2.morphologyEx(filled_region, cv2.MORPH_CLOSE, _kernel(9), iterations=1)
    resolved_region = filled_region > 0
    if int(np.count_nonzero(resolved_region)) < int(np.count_nonzero(primary_boundary_mask)):
        return fallback_region.astype(bool)
    return resolved_region


def _build_radar_boundaries(
    rgb_image: np.ndarray,
    class_map: np.ndarray,
    class_names: Sequence[str],
    origin_px: tuple[float, float],
    boundary_classes: Sequence[str],
    thickness_px: int,
    center_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[str, int]], Dict[str, object], np.ndarray, np.ndarray]:
    height, width = class_map.shape
    class_index = {name: idx for idx, name in enumerate(class_names)}
    all_boundaries = np.zeros((height, width), dtype=bool)
    per_class_maps: Dict[str, np.ndarray] = {}
    report: Dict[str, Dict[str, int]] = {}

    for class_name in boundary_classes:
        if class_name not in class_index:
            continue
        mask = class_map == class_index[class_name]
        if not np.any(mask):
            per_class_maps[class_name] = np.zeros((height, width), dtype=np.float32)
            report[class_name] = {"pixel_count": 0}
            continue

        boundary_ring = mask & ~cv2.erode(mask.astype(np.uint8), _kernel(3), iterations=1).astype(bool)
        full_boundary = boundary_ring

        if thickness_px > 1 and np.any(full_boundary):
            full_boundary = cv2.dilate(full_boundary.astype(np.uint8), _kernel(thickness_px), iterations=1) > 0

        all_boundaries |= full_boundary
        per_class_maps[class_name] = full_boundary.astype(np.float32)
        report[class_name] = {"pixel_count": int(full_boundary.sum())}

    filtered_boundaries, support_region, primary_boundary_mask, cluster_report = _keep_central_boundary_cluster(
        boundary_mask=all_boundaries,
        origin_px=origin_px,
        center_radius_px=center_radius_px,
    )
    all_boundaries = filtered_boundaries
    for class_name, class_boundary in per_class_maps.items():
        filtered_class_boundary = class_boundary.astype(bool) & all_boundaries
        per_class_maps[class_name] = filtered_class_boundary.astype(np.float32)
        report[class_name] = {"pixel_count": int(filtered_class_boundary.sum())}

    boundary_map = np.zeros((height, width, 3), dtype=np.uint8)
    boundary_map[all_boundaries] = 255
    overlay = rgb_image.astype(np.float32).copy()
    overlay[all_boundaries] = 0.25 * overlay[all_boundaries] + 0.75 * 255.0
    boundary_overlay = overlay.clip(0, 255).astype(np.uint8)
    return (
        boundary_map,
        boundary_overlay,
        per_class_maps,
        report,
        cluster_report,
        support_region.astype(bool),
        primary_boundary_mask.astype(bool),
    )


def _build_radar_glints(
    rgb_image: np.ndarray,
    candidate_mask: np.ndarray,
    primary_region: np.ndarray,
    blind_zone_mask: np.ndarray,
    class_map: np.ndarray,
    class_names: Sequence[str],
    geometry: Mapping[str, np.ndarray],
    base_received_power_w: np.ndarray,
    scattering_debug: Mapping[str, np.ndarray],
    meters_per_pixel: float,
    angular_resolution_deg: float,
    range_resolution_m: float,
    spill_resolution_elements: float,
    peak_gain_db: float,
    receiver_floor_dbw: float,
    noise_seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, object], Dict[str, np.ndarray]]:
    height, width = candidate_mask.shape
    zero_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    zero_float = np.zeros((height, width), dtype=np.float32)
    receiver_floor_w = float(np.asarray(_db_to_linear(receiver_floor_dbw)).reshape(-1)[0])
    default_report: Dict[str, object] = {
        "applied": False,
        "candidate_glint_pixels": int(np.count_nonzero(candidate_mask)),
        "selected_glint_count": 0,
        "angular_resolution_deg": float(angular_resolution_deg),
        "range_resolution_m": float(range_resolution_m),
        "range_resolution_px": 0.0,
        "median_cross_range_resolution_px": 0.0,
        "spill_resolution_elements": float(spill_resolution_elements),
        "max_spill_px": 0.0,
        "processing_gain_db": float(peak_gain_db),
        "background_noise_scale_w": receiver_floor_w,
        "source_cell_count": 0,
        "response_model": "resolution_cell_sinc_squared_image_space",
    }
    default_debug = {
        "seed_mask": zero_float,
        "candidate_mask": candidate_mask.astype(np.float32),
        "primary_region": primary_region.astype(np.float32),
        "blind_zone_mask": blind_zone_mask.astype(np.float32),
        "outside_attenuation": np.zeros((height, width), dtype=np.float32),
        "background_noise_w": zero_float,
        "speckle_gain": np.ones((height, width), dtype=np.float32),
        "diffuse_source_power": zero_float,
        "point_source_power": zero_float,
        "glint_power_w": zero_float,
    }
    if (
        angular_resolution_deg <= 0.0
        or range_resolution_m <= 0.0
        or meters_per_pixel <= 0.0
        or not np.any(candidate_mask)
        or not np.any(primary_region)
    ):
        return zero_rgb, rgb_image.copy(), zero_float, default_report, default_debug

    visible_region = primary_region.astype(bool) & ~blind_zone_mask.astype(bool)
    effective_candidate_mask = candidate_mask.astype(bool) & visible_region
    if not np.any(effective_candidate_mask):
        return zero_rgb, rgb_image.copy(), zero_float, default_report, default_debug

    diffuse_gain_map = np.asarray(
        scattering_debug.get("diffuse_gain_map", np.ones_like(base_received_power_w, dtype=np.float32)),
        dtype=np.float32,
    )
    point_gain_map = np.asarray(
        scattering_debug.get("point_gain_map", np.zeros_like(base_received_power_w, dtype=np.float32)),
        dtype=np.float32,
    )
    diffuse_source_power_map = np.zeros_like(base_received_power_w, dtype=np.float32)
    point_source_power_map = np.zeros_like(base_received_power_w, dtype=np.float32)
    diffuse_source_power_map[effective_candidate_mask] = (
        base_received_power_w[effective_candidate_mask].astype(np.float32)
        * np.maximum(diffuse_gain_map[effective_candidate_mask], 0.0)
    )
    point_source_power_map[effective_candidate_mask] = (
        base_received_power_w[effective_candidate_mask].astype(np.float32)
        * np.maximum(point_gain_map[effective_candidate_mask], 0.0)
    )

    ys, xs = np.nonzero(effective_candidate_mask)
    range_m = geometry["slant_range_m"][ys, xs].astype(np.float32)
    dx_m = geometry["dx_m"][ys, xs].astype(np.float32)
    dy_m = geometry["dy_m"][ys, xs].astype(np.float32)
    theta_deg = _polar_angles_deg(dx_m=dx_m, dy_m=dy_m)
    diffuse_source_power = diffuse_source_power_map[ys, xs].astype(np.float32)
    point_source_power = point_source_power_map[ys, xs].astype(np.float32)
    if float(np.max(diffuse_source_power + point_source_power)) <= 0.0:
        return zero_rgb, rgb_image.copy(), zero_float, default_report, default_debug

    range_resolution_px = max(1.0, float(range_resolution_m / meters_per_pixel))
    cross_range_resolution_m = np.maximum(
        range_m * np.deg2rad(float(angular_resolution_deg)),
        meters_per_pixel,
    ).astype(np.float32)
    median_cross_range_resolution_px = (
        float(np.median(cross_range_resolution_m / meters_per_pixel)) if cross_range_resolution_m.size else 1.0
    )
    local_max_kernel = max(5, int(np.ceil(1.5 * max(range_resolution_px, median_cross_range_resolution_px))))
    deterministic_peak_map = _peak_map(
        point_source_power_map,
        kernel_size=local_max_kernel,
        mask=effective_candidate_mask,
    )
    seed_mask = deterministic_peak_map.astype(np.float32)

    angle_cell = np.floor(theta_deg / max(float(angular_resolution_deg), 1e-6)).astype(np.int32)
    range_cell = np.floor(range_m / max(float(range_resolution_m), 1e-6)).astype(np.int32)
    cell_ids = np.stack([angle_cell, range_cell], axis=1)
    unique_cells, inverse = np.unique(cell_ids, axis=0, return_inverse=True)
    source_cell_count = int(unique_cells.shape[0])

    weights = diffuse_source_power + point_source_power + _MIN_LINEAR_POWER
    cell_weight_sum = np.bincount(inverse, weights=weights, minlength=source_cell_count).astype(np.float32)
    cell_diffuse_total = np.bincount(inverse, weights=diffuse_source_power, minlength=source_cell_count).astype(np.float32)
    cell_x = np.bincount(inverse, weights=xs.astype(np.float32) * weights, minlength=source_cell_count).astype(np.float32)
    cell_y = np.bincount(inverse, weights=ys.astype(np.float32) * weights, minlength=source_cell_count).astype(np.float32)
    cell_range_m = np.bincount(inverse, weights=range_m * weights, minlength=source_cell_count).astype(np.float32)
    cell_dx_m = np.bincount(inverse, weights=dx_m * weights, minlength=source_cell_count).astype(np.float32)
    cell_dy_m = np.bincount(inverse, weights=dy_m * weights, minlength=source_cell_count).astype(np.float32)

    peak_ys, peak_xs = np.nonzero(deterministic_peak_map > 0.0)
    cell_point_total = np.zeros(source_cell_count, dtype=np.float32)
    if peak_ys.size > 0:
        peak_range_m = geometry["slant_range_m"][peak_ys, peak_xs].astype(np.float32)
        peak_theta_deg = _polar_angles_deg(
            dx_m=geometry["dx_m"][peak_ys, peak_xs].astype(np.float32),
            dy_m=geometry["dy_m"][peak_ys, peak_xs].astype(np.float32),
        )
        peak_angle_cell = np.floor(peak_theta_deg / max(float(angular_resolution_deg), 1e-6)).astype(np.int32)
        peak_range_cell = np.floor(peak_range_m / max(float(range_resolution_m), 1e-6)).astype(np.int32)
        peak_cell_lookup = {
            (int(angle_value), int(range_value)): cell_index
            for cell_index, (angle_value, range_value) in enumerate(unique_cells.tolist())
        }
        peak_values = deterministic_peak_map[peak_ys, peak_xs].astype(np.float32)
        for peak_value, angle_value, range_value in zip(
            peak_values.tolist(),
            peak_angle_cell.tolist(),
            peak_range_cell.tolist(),
        ):
            cell_index = peak_cell_lookup.get((int(angle_value), int(range_value)))
            if cell_index is not None:
                cell_point_total[cell_index] += float(peak_value)

    processing_gain_linear = float(np.asarray(_db_to_linear(peak_gain_db)).reshape(-1)[0])
    glint_noise_rng = np.random.default_rng(None if noise_seed is None else int(noise_seed) + 101)
    signal_power_w = np.zeros((height, width), dtype=np.float32)
    speckle_gain = np.ones((height, width), dtype=np.float32)

    for cell_index in range(source_cell_count):
        weight_sum = float(cell_weight_sum[cell_index])
        if weight_sum <= 0.0:
            continue
        diffuse_total = float(cell_diffuse_total[cell_index]) * processing_gain_linear
        point_total = float(cell_point_total[cell_index]) * processing_gain_linear
        if diffuse_total <= 0.0 and point_total <= 0.0:
            continue

        clutter_sample = float(glint_noise_rng.exponential(scale=max(diffuse_total, _MIN_LINEAR_POWER))) if diffuse_total > 0.0 else 0.0
        point_sample = point_total
        total_cell_power = clutter_sample + point_sample
        if total_cell_power <= 0.0:
            continue

        cx = float(cell_x[cell_index] / max(weight_sum, _MIN_LINEAR_POWER))
        cy = float(cell_y[cell_index] / max(weight_sum, _MIN_LINEAR_POWER))
        mean_range = float(cell_range_m[cell_index] / max(weight_sum, _MIN_LINEAR_POWER))
        mean_dx = float(cell_dx_m[cell_index] / max(weight_sum, _MIN_LINEAR_POWER))
        mean_dy = float(cell_dy_m[cell_index] / max(weight_sum, _MIN_LINEAR_POWER))
        norm = float(np.hypot(mean_dx, mean_dy))
        if norm <= 1e-6:
            radial_x_px, radial_y_px = 1.0, 0.0
        else:
            radial_x_px = mean_dx / max(norm, 1e-6)
            radial_y_px = mean_dy / max(norm, 1e-6)
        cross_x_px = -radial_y_px
        cross_y_px = radial_x_px

        cross_resolution_m = max(mean_range * np.deg2rad(float(angular_resolution_deg)), meters_per_pixel)
        range_support_px = max(2, int(np.ceil(2.0 * range_resolution_m / meters_per_pixel)))
        cross_support_px = max(2, int(np.ceil(2.0 * cross_resolution_m / meters_per_pixel)))
        x0 = max(0, int(np.floor(cx - max(range_support_px, cross_support_px))))
        x1 = min(width, int(np.ceil(cx + max(range_support_px, cross_support_px) + 1)))
        y0 = max(0, int(np.floor(cy - max(range_support_px, cross_support_px))))
        y1 = min(height, int(np.ceil(cy + max(range_support_px, cross_support_px) + 1)))
        if x0 >= x1 or y0 >= y1:
            continue

        patch_y, patch_x = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        offset_x_m = (patch_x - cx) * float(meters_per_pixel)
        offset_y_m = (patch_y - cy) * float(meters_per_pixel)
        range_offset_m = offset_x_m * radial_x_px + offset_y_m * radial_y_px
        cross_offset_m = offset_x_m * cross_x_px + offset_y_m * cross_y_px
        psf = (
            np.square(np.sinc(range_offset_m / max(float(range_resolution_m), 1e-6))).astype(np.float32)
            * np.square(np.sinc(cross_offset_m / max(cross_resolution_m, 1e-6))).astype(np.float32)
        )
        psf_sum = float(psf.sum())
        if psf_sum <= 1e-12:
            continue
        psf = (psf / psf_sum).astype(np.float32)
        signal_power_w[y0:y1, x0:x1] += total_cell_power * psf
        local_speckle = np.ones_like(psf, dtype=np.float32) * max(clutter_sample / max(diffuse_total, _MIN_LINEAR_POWER), 1.0)
        speckle_gain[y0:y1, x0:x1] = np.maximum(speckle_gain[y0:y1, x0:x1], local_speckle)

    signal_gate = visible_region.astype(np.float32)
    signal_power_w *= signal_gate
    background_noise_w = glint_noise_rng.exponential(
        scale=max(receiver_floor_w, _MIN_LINEAR_POWER),
        size=signal_power_w.shape,
    ).astype(np.float32)
    glint_power_w = (signal_power_w + background_noise_w).astype(np.float32)
    outside_attenuation = signal_gate.astype(np.float32)

    positive_glint = glint_power_w[np.isfinite(glint_power_w) & (glint_power_w > 0.0)]
    white_linear = float(np.percentile(positive_glint, 99.85)) if positive_glint.size > 0 else receiver_floor_w
    glint_map = _render_positive_grayscale(
        glint_power_w,
        floor_linear=receiver_floor_w,
        white_linear=max(white_linear, receiver_floor_w * 4.0),
    )
    glint_overlay = np.clip(rgb_image.astype(np.float32) + glint_map.astype(np.float32) * 0.85, 0, 255).astype(np.uint8)
    report: Dict[str, object] = {
        "applied": True,
        "candidate_glint_pixels": int(np.count_nonzero(effective_candidate_mask)),
        "selected_glint_count": source_cell_count,
        "dominant_scatterer_count": int(np.count_nonzero(seed_mask)),
        "angular_resolution_deg": float(angular_resolution_deg),
        "range_resolution_m": float(range_resolution_m),
        "range_resolution_px": float(range_resolution_px),
        "median_cross_range_resolution_px": float(median_cross_range_resolution_px),
        "spill_resolution_elements": float(spill_resolution_elements),
        "max_spill_px": 0.0,
        "processing_gain_db": float(peak_gain_db),
        "background_noise_scale_w": float(receiver_floor_w),
        "source_cell_count": source_cell_count,
        "local_peak_kernel_px": int(local_max_kernel),
        "deterministic_peak_count": int(np.count_nonzero(deterministic_peak_map)),
        "response_model": "resolution_cell_sinc_squared_image_space",
    }
    debug_maps = {
        "seed_mask": seed_mask.astype(np.float32),
        "candidate_mask": effective_candidate_mask.astype(np.float32),
        "primary_region": primary_region.astype(np.float32),
        "blind_zone_mask": blind_zone_mask.astype(np.float32),
        "outside_attenuation": outside_attenuation.astype(np.float32),
        "background_noise_w": background_noise_w.astype(np.float32),
        "speckle_gain": speckle_gain.astype(np.float32),
        "diffuse_source_power": diffuse_source_power_map.astype(np.float32),
        "point_source_power": point_source_power_map.astype(np.float32),
        "glint_power_w": glint_power_w.astype(np.float32),
    }
    return glint_map, glint_overlay, glint_power_w.astype(np.float32), report, debug_maps


def map_radar_equation_to_pixels(
    rgb_image: np.ndarray,
    class_names: Sequence[str],
    probabilities: np.ndarray,
    pixel_rcs_map_m2: np.ndarray,
    config_path: str | Path,
    pipeline_switches: Mapping[str, bool] | None = None,
) -> RadarEquationResult:
    config = _load_radar_config(config_path)
    switches = dict(pipeline_switches or {})
    frequency_ghz = _require_float(config, "frequency_ghz")
    transmit_power_w = _require_float(config, "transmit_power_w")
    antenna_gain_db = _require_float(config, "antenna_gain_db")
    system_loss_db = _require_float(config, "system_loss_db")
    antenna_height_m = _require_float(config, "antenna_height_m")
    reference_range_m = _require_float(config, "reference_range_m")
    meters_per_pixel = _require_float(config, "meters_per_pixel")
    receiver_sensitivity_dbw = _require_float(config, "receiver_sensitivity_dbw")
    receiver_max_level_dbw = _require_float(config, "receiver_max_level_dbw")
    gaussian_noise_mean_dbw = float(config.get("gaussian_noise_mean_dbw", 0.0))
    gaussian_noise_std_dbw = float(config.get("gaussian_noise_std_dbw", 0.75))
    raw_noise_seed = config.get("gaussian_noise_seed", 42)
    gaussian_noise_seed = None if raw_noise_seed is None else int(raw_noise_seed)
    glint_angular_resolution_deg = float(config.get("glint_angular_resolution_deg", 1.0))
    glint_range_resolution_m = float(config.get("glint_range_resolution_m", 1.5))
    glint_spill_resolution_elements = float(config.get("glint_spill_resolution_elements", 2.5))
    glint_peak_gain_db = float(config.get("glint_peak_gain_db", 9.0))
    if receiver_max_level_dbw <= receiver_sensitivity_dbw:
        raise ValueError("receiver_max_level_dbw must be greater than receiver_sensitivity_dbw.")
    use_radar_boundaries = bool(switches.get("use_radar_boundaries", True))
    use_radar_glints = bool(switches.get("use_radar_glints", True))
    raw_boundary_classes = config.get("boundary_classes", ["forest", "building", "shrub"])
    if isinstance(raw_boundary_classes, Sequence) and not isinstance(raw_boundary_classes, str):
        boundary_classes = [str(value) for value in raw_boundary_classes]
    else:
        boundary_classes = ["forest", "building", "shrub"]
    boundary_thickness_px = int(config.get("boundary_thickness_px", 2))
    boundary_cluster_center_radius_px = float(config.get("boundary_cluster_center_radius_px", 0.35 * min(rgb_image.shape[:2])))
    origin_mode = str(config.get("origin_mode", "image_center")).lower()

    origin_px = _resolve_radar_origin(
        config=config,
        shape=rgb_image.shape[:2],
        probabilities=probabilities,
        class_names=class_names,
    )
    class_map = np.argmax(probabilities, axis=0).astype(np.uint8)
    if use_radar_boundaries:
        (
            boundary_map,
            boundary_overlay,
            per_class_boundaries,
            boundary_report,
            boundary_cluster_report,
            boundary_support_region,
            primary_boundary_mask,
        ) = _build_radar_boundaries(
            rgb_image=rgb_image,
            class_map=class_map,
            class_names=class_names,
            origin_px=origin_px,
            boundary_classes=boundary_classes,
            thickness_px=boundary_thickness_px,
            center_radius_px=boundary_cluster_center_radius_px,
        )
    else:
        boundary_map = np.zeros_like(rgb_image)
        boundary_overlay = rgb_image.copy()
        per_class_boundaries = {class_name: np.zeros(class_map.shape, dtype=np.float32) for class_name in boundary_classes}
        boundary_report = {class_name: {"pixel_count": 0} for class_name in boundary_classes}
        boundary_cluster_report = {
            "component_count": 0,
            "center_radius_px": float(boundary_cluster_center_radius_px),
            "selection_mode": "disabled",
            "selected_component": None,
            "kept_pixel_count": 0,
        }
        boundary_support_region = np.zeros(class_map.shape, dtype=bool)
        primary_boundary_mask = np.zeros(class_map.shape, dtype=bool)

    primary_cluster_region = _estimate_primary_cluster_region(
        primary_boundary_mask=primary_boundary_mask,
        fallback_region=boundary_support_region,
    )
    blind_zone_scope = primary_cluster_region.astype(bool)
    if not np.any(blind_zone_scope):
        blind_zone_scope = boundary_support_region.astype(bool)

    if use_radar_boundaries and np.any(boundary_support_region):
        effective_rcs_map_m2, boundary_scatter_cleanup_report = _apply_boundary_scatter_cleanup(
            pixel_rcs_map_m2=pixel_rcs_map_m2,
            support_region=boundary_support_region,
        )
    else:
        effective_rcs_map_m2 = pixel_rcs_map_m2.astype(np.float32).copy()
        boundary_scatter_cleanup_report = {
            "applied": False,
            "background_rcs_m2": None,
            "replaced_pixel_count": 0,
            "preserved_pixel_count": int(np.count_nonzero(boundary_support_region)),
        }

    geometry = _geometry(
        shape=rgb_image.shape[:2],
        origin_px=origin_px,
        meters_per_pixel=meters_per_pixel,
        antenna_height_m=antenna_height_m,
        range_bias_m=0.0 if origin_mode == "image_center" else reference_range_m,
    )
    blind_zone_mask, blind_zone_report = _build_building_blind_zone_mask(
        class_map=class_map,
        class_names=class_names,
        geometry=geometry,
        scope_region=blind_zone_scope,
        angular_resolution_deg=glint_angular_resolution_deg,
        range_resolution_m=glint_range_resolution_m,
        meters_per_pixel=meters_per_pixel,
    )
    blind_zones_map = _render_red_mask(blind_zone_mask)
    glint_signal_region = primary_cluster_region.astype(bool) & ~blind_zone_mask.astype(bool)
    scattering_gain_map, scattering_model_report, scattering_debug = _build_scattering_gain_map(
        rgb_image=rgb_image,
        class_map=class_map,
        class_names=class_names,
        geometry=geometry,
        antenna_height_m=antenna_height_m,
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
    received_signal_w = (
        radar_constant
        * np.maximum(effective_rcs_map_m2, 0.0)
        / np.maximum(geometry["slant_range_m"], 1.0) ** 4
        * np.maximum(scattering_gain_map, 0.0)
    ).astype(np.float32)
    received_signal_w[blind_zone_mask.astype(bool)] = 0.0
    raw_received_power_w = received_signal_w.copy()
    raw_received_power_dbw = _linear_to_db(raw_received_power_w)
    receiver_noise_floor_dbw = receiver_sensitivity_dbw + gaussian_noise_mean_dbw
    received_power_w, receiver_noise_w, receiver_noise_report = _apply_receiver_noise(
        signal_power_w=raw_received_power_w,
        noise_floor_dbw=receiver_noise_floor_dbw,
        seed=gaussian_noise_seed,
    )
    received_power_dbw = _linear_to_db(received_power_w)
    if use_radar_glints and np.any(primary_cluster_region):
        glint_map, glint_overlay, glint_power_w, glint_report, glint_debug = _build_radar_glints(
            rgb_image=rgb_image,
            candidate_mask=glint_signal_region,
            primary_region=primary_cluster_region,
            blind_zone_mask=blind_zone_mask,
            class_map=class_map,
            class_names=class_names,
            geometry=geometry,
            base_received_power_w=raw_received_power_w,
            scattering_debug=scattering_debug,
            meters_per_pixel=meters_per_pixel,
            angular_resolution_deg=glint_angular_resolution_deg,
            range_resolution_m=glint_range_resolution_m,
            spill_resolution_elements=glint_spill_resolution_elements,
            peak_gain_db=glint_peak_gain_db,
            receiver_floor_dbw=receiver_sensitivity_dbw,
            noise_seed=gaussian_noise_seed,
        )
    else:
        glint_map = np.zeros_like(rgb_image)
        glint_overlay = rgb_image.copy()
        glint_power_w = np.zeros(class_map.shape, dtype=np.float32)
        glint_report = {
            "applied": False,
            "candidate_glint_pixels": int(np.count_nonzero(primary_cluster_region)),
            "selected_glint_count": 0,
            "angular_resolution_deg": float(glint_angular_resolution_deg),
            "range_resolution_m": float(glint_range_resolution_m),
            "range_resolution_px": 0.0,
            "median_cross_range_resolution_px": 0.0,
            "spill_resolution_elements": float(glint_spill_resolution_elements),
            "max_spill_px": 0.0,
            "processing_gain_db": float(glint_peak_gain_db),
            "background_noise_scale_w": float(np.asarray(_db_to_linear(receiver_sensitivity_dbw)).reshape(-1)[0]),
            "source_cell_count": 0,
            "response_model": "polar_beam_clutter_plus_sparse_point_targets",
        }
        glint_debug = {
            "seed_mask": np.zeros(class_map.shape, dtype=np.float32),
            "candidate_mask": glint_signal_region.astype(np.float32),
            "primary_region": primary_cluster_region.astype(np.float32),
            "blind_zone_mask": blind_zone_mask.astype(np.float32),
            "outside_attenuation": np.zeros(class_map.shape, dtype=np.float32),
            "background_noise_w": np.zeros(class_map.shape, dtype=np.float32),
            "speckle_gain": np.ones(class_map.shape, dtype=np.float32),
            "diffuse_source_power": np.zeros(class_map.shape, dtype=np.float32),
            "point_source_power": np.zeros(class_map.shape, dtype=np.float32),
            "glint_power_w": glint_power_w.astype(np.float32),
        }
    heatmap = _render_grayscale(
        values_dbw=received_power_dbw,
        lower_dbw=receiver_sensitivity_dbw,
        upper_dbw=receiver_max_level_dbw,
    )
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
        "geometry_range_bias_m": float(0.0 if origin_mode == "image_center" else reference_range_m),
        "visual_black_level_dbw": float(receiver_sensitivity_dbw),
        "visual_white_level_dbw": float(receiver_max_level_dbw),
    }
    report: Dict[str, object] = {
        "equation": "Pr = Pt * Gt * Gr * lambda^2 * sigma / ((4*pi)^3 * R^4 * L)",
        "summary": summary,
        "boundary_report": boundary_report,
        "boundary_cluster_report": boundary_cluster_report,
        "boundary_scatter_cleanup_report": boundary_scatter_cleanup_report,
        "blind_zone_report": blind_zone_report,
        "scattering_model_report": scattering_model_report,
        "receiver_noise_report": receiver_noise_report,
        "glint_report": glint_report,
        "config": {
            "origin_mode": origin_mode,
            "frequency_ghz": frequency_ghz,
            "transmit_power_w": transmit_power_w,
            "antenna_gain_db": antenna_gain_db,
            "system_loss_db": system_loss_db,
            "antenna_height_m": antenna_height_m,
            "reference_range_m": reference_range_m,
            "meters_per_pixel": meters_per_pixel,
            "boundary_classes": boundary_classes,
            "boundary_thickness_px": boundary_thickness_px,
            "boundary_cluster_center_radius_px": boundary_cluster_center_radius_px,
            "receiver_sensitivity_dbw": receiver_sensitivity_dbw,
            "receiver_max_level_dbw": receiver_max_level_dbw,
            "gaussian_noise_mean_dbw": gaussian_noise_mean_dbw,
            "gaussian_noise_std_dbw": gaussian_noise_std_dbw,
            "gaussian_noise_seed": gaussian_noise_seed,
            "glint_angular_resolution_deg": glint_angular_resolution_deg,
            "glint_range_resolution_m": glint_range_resolution_m,
            "glint_spill_resolution_elements": glint_spill_resolution_elements,
            "glint_peak_gain_db": glint_peak_gain_db,
        },
        "pipeline_switches": dict(switches),
    }
    debug_maps: Dict[str, np.ndarray] = {
        "radar_slant_range_m": geometry["slant_range_m"].astype(np.float32),
        "radar_ground_offset_m": geometry["ground_offset_m"].astype(np.float32),
        "radar_effective_rcs_map_m2": effective_rcs_map_m2.astype(np.float32),
        "radar_blind_zone_mask": blind_zone_mask.astype(np.float32),
        "radar_blind_zones_map_rgb": blind_zones_map,
        "radar_blind_zone_scope": blind_zone_scope.astype(np.float32),
        "radar_scattering_gain_map": scattering_gain_map.astype(np.float32),
        "radar_scattering_diffuse_gain_map": scattering_debug["diffuse_gain_map"].astype(np.float32),
        "radar_scattering_point_gain_map": scattering_debug["point_gain_map"].astype(np.float32),
        "radar_scattering_roughness_map": scattering_debug["roughness_map"].astype(np.float32),
        "radar_scattering_volume_texture_map": scattering_debug["volume_texture_map"].astype(np.float32),
        "radar_scattering_incidence_term": scattering_debug["incidence_term"].astype(np.float32),
        "radar_scattering_grazing_term": scattering_debug["grazing_term"].astype(np.float32),
        "radar_scattering_edge_facing_map": scattering_debug["edge_facing_map"].astype(np.float32),
        "radar_scattering_corner_strength_map": scattering_debug["corner_strength_map"].astype(np.float32),
        "radar_scattering_dihedral_peak_map": scattering_debug["dihedral_peak_map"].astype(np.float32),
        "radar_received_power_dbw_raw": raw_received_power_dbw.astype(np.float32),
        "radar_receiver_noise_w": receiver_noise_w.astype(np.float32),
        "radar_received_power_dbw": received_power_dbw.astype(np.float32),
        "radar_equation_map_rgb": heatmap,
        "radar_equation_overlay_rgb": overlay,
        "radar_glints_map_rgb": glint_map,
        "radar_glints_overlay_rgb": glint_overlay,
        "radar_glints_power_w": glint_power_w.astype(np.float32),
        "radar_boundary_map_rgb": boundary_map,
        "radar_boundary_overlay_rgb": boundary_overlay,
        "radar_boundary_cluster_mask": (boundary_map[:, :, 0] > 0).astype(np.float32),
        "radar_boundary_support_region": boundary_support_region.astype(np.float32),
        "radar_primary_boundary_mask": primary_boundary_mask.astype(np.float32),
        "radar_primary_cluster_region": primary_cluster_region.astype(np.float32),
    }
    debug_maps["radar_glint_seed_mask"] = glint_debug["seed_mask"].astype(np.float32)
    debug_maps["radar_glint_candidate_mask"] = glint_debug["candidate_mask"].astype(np.float32)
    debug_maps["radar_glint_outside_attenuation"] = glint_debug["outside_attenuation"].astype(np.float32)
    debug_maps["radar_glint_background_noise_w"] = glint_debug["background_noise_w"].astype(np.float32)
    debug_maps["radar_glint_speckle_gain"] = glint_debug["speckle_gain"].astype(np.float32)
    debug_maps["radar_glint_diffuse_source_power"] = glint_debug["diffuse_source_power"].astype(np.float32)
    debug_maps["radar_glint_point_source_power"] = glint_debug["point_source_power"].astype(np.float32)
    for class_name, boundary_values in per_class_boundaries.items():
        debug_maps[f"radar_boundary_{class_name}"] = boundary_values.astype(np.float32)

    return RadarEquationResult(
        received_power_w=received_power_w,
        received_power_dbw=received_power_dbw,
        heatmap=heatmap,
        overlay=overlay,
        blind_zones_map=blind_zones_map,
        glint_map=glint_map,
        glint_overlay=glint_overlay,
        boundary_map=boundary_map,
        boundary_overlay=boundary_overlay,
        debug_maps=debug_maps,
        report=report,
        config=config,
        origin_px=origin_px,
    )
