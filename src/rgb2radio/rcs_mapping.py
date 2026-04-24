from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import cv2
import numpy as np

PIXEL_SIZE_METERS = 0.375
PIXEL_AREA_M2 = PIXEL_SIZE_METERS * PIXEL_SIZE_METERS
_MIN_LINEAR_RCS = 1e-12
_LOW_PERCENTILE = 5.0
_HIGH_PERCENTILE = 95.0
_VEHICLE_DISTRIBUTION_BIAS = 0.15
_SHADOW_CLASS_NAMES = {"forest_shadow", "building_shadow"}


@dataclass(frozen=True)
class RcsMappingResult:
    linear_map_m2: np.ndarray
    dbsm_map: np.ndarray
    heatmap: np.ndarray
    overlay: np.ndarray
    brightness: np.ndarray
    normalized_brightness: np.ndarray
    debug_maps: Mapping[str, np.ndarray]
    report: Mapping[str, object]
    config: Mapping[str, object]


def _load_rcs_config(path: str | Path) -> Dict[str, object]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["config_path"] = str(config_path)
    return payload


def _validate_rcs_config(config: Mapping[str, object], class_names: Sequence[str]) -> None:
    missing = [class_name for class_name in class_names if class_name not in _SHADOW_CLASS_NAMES and class_name not in config]
    if missing:
        raise ValueError(f"RCS config is missing class definitions: {missing}")


def _value_brightness(rgb_image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    return (hsv[:, :, 2] / 255.0).astype(np.float32)


def _db_to_linear(values_db: np.ndarray | float) -> np.ndarray:
    return np.power(10.0, np.asarray(values_db, dtype=np.float32) / 10.0).astype(np.float32)


def _linear_to_db(values_linear: np.ndarray) -> np.ndarray:
    return (10.0 * np.log10(np.maximum(values_linear.astype(np.float32), _MIN_LINEAR_RCS))).astype(np.float32)


def _classwise_normalize(
    brightness: np.ndarray,
    class_map: np.ndarray,
    class_names: Sequence[str],
    low_percentile: float,
    high_percentile: float,
) -> tuple[np.ndarray, Dict[str, Dict[str, float]]]:
    normalized = np.zeros_like(brightness, dtype=np.float32)
    stats: Dict[str, Dict[str, float]] = {}
    for class_index, class_name in enumerate(class_names):
        mask = class_map == class_index
        if not np.any(mask):
            continue

        values = brightness[mask].astype(np.float32)
        low = float(np.percentile(values, low_percentile))
        high = float(np.percentile(values, high_percentile))
        if high - low < 1e-6:
            scaled = np.full(values.shape, 0.5, dtype=np.float32)
        else:
            scaled = ((values - low) / (high - low)).clip(0.0, 1.0).astype(np.float32)
        normalized[mask] = scaled
        stats[class_name] = {
            "brightness_min": float(values.min()),
            "brightness_max": float(values.max()),
            "brightness_mean": float(values.mean()),
            "brightness_p_low": low,
            "brightness_p_high": high,
        }
    return normalized, stats


def _resolve_reference_range_db(config: Mapping[str, object], class_name: str) -> tuple[float, float]:
    spec = config[class_name]
    if not isinstance(spec, Mapping):
        raise ValueError(f"RCS config entry for '{class_name}' must be an object.")
    if "minimum" not in spec or "maximum" not in spec:
        raise ValueError(f"RCS config entry for '{class_name}' must contain 'minimum' and 'maximum'.")
    return float(spec["minimum"]), float(spec["maximum"])


def _vehicle_component_map(
    mask: np.ndarray,
    normalized_brightness: np.ndarray,
    min_db: float,
    max_db: float,
    distribution_bias: float,
) -> tuple[np.ndarray, list[Dict[str, float]]]:
    component_map = np.zeros_like(normalized_brightness, dtype=np.float32)
    reports: list[Dict[str, float]] = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for component_id in range(1, num_labels):
        component_mask = labels == component_id
        if not np.any(component_mask):
            continue
        component_values = normalized_brightness[component_mask].astype(np.float32)
        component_score = float(component_values.mean()) if component_values.size else 0.5
        target_dbsm = float(min_db + component_score * (max_db - min_db))
        target_linear = float(_db_to_linear(target_dbsm))
        weights = distribution_bias + component_values
        weight_sum = float(weights.sum())
        if weight_sum <= 0.0:
            weights = np.ones_like(component_values, dtype=np.float32)
            weight_sum = float(weights.sum())
        component_linear = target_linear * weights / weight_sum
        component_map[component_mask] = component_linear.astype(np.float32)
        reports.append(
            {
                "component_id": int(component_id),
                "pixel_count": int(stats[component_id, cv2.CC_STAT_AREA]),
                "component_score": component_score,
                "target_rcs_dbsm": target_dbsm,
                "target_rcs_m2": target_linear,
            }
        )
    return component_map, reports


def _render_heatmap(values_dbsm: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values_dbsm)
    if not np.any(finite):
        return np.zeros(values_dbsm.shape + (3,), dtype=np.uint8)
    samples = values_dbsm[finite]
    lower = float(np.percentile(samples, 2.0))
    upper = float(np.percentile(samples, 98.0))
    if upper - lower < 1e-6:
        upper = lower + 1e-6
    scaled = ((values_dbsm - lower) / (upper - lower)).clip(0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _shadow_indices(class_names: Sequence[str]) -> list[int]:
    return [index for index, class_name in enumerate(class_names) if class_name in _SHADOW_CLASS_NAMES]


def _nearest_majority_shadow_fill(
    shadow_mask: np.ndarray,
    class_map: np.ndarray,
    direct_linear_map: np.ndarray,
    class_names: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = class_map.shape
    shadow_indices = _shadow_indices(class_names)
    non_shadow_mask = ~np.isin(class_map, shadow_indices)
    donor_class_map = np.full(class_map.shape, -1, dtype=np.int16)
    donor_radius_map = np.zeros(class_map.shape, dtype=np.float32)
    filled_linear_map = np.zeros_like(direct_linear_map, dtype=np.float32)

    global_counts = np.bincount(class_map[non_shadow_mask].ravel(), minlength=len(class_names)) if np.any(non_shadow_mask) else np.zeros(len(class_names), dtype=np.int32)
    for shadow_index in shadow_indices:
        global_counts[shadow_index] = 0
    global_donor_class = int(np.argmax(global_counts)) if np.any(global_counts) else -1
    global_donor_value = float(direct_linear_map[class_map == global_donor_class].mean()) if global_donor_class >= 0 and np.any(class_map == global_donor_class) else float(np.mean(direct_linear_map[non_shadow_mask])) if np.any(non_shadow_mask) else float(_MIN_LINEAR_RCS)

    ys, xs = np.where(shadow_mask)
    max_radius = max(height, width)
    for y, x in zip(ys.tolist(), xs.tolist()):
        assigned = False
        for radius in range(1, max_radius + 1):
            y0 = max(0, y - radius)
            y1 = min(height, y + radius + 1)
            x0 = max(0, x - radius)
            x1 = min(width, x + radius + 1)

            local_non_shadow = non_shadow_mask[y0:y1, x0:x1]
            if not np.any(local_non_shadow):
                continue

            boundary = np.zeros_like(local_non_shadow, dtype=bool)
            boundary[0, :] = True
            boundary[-1, :] = True
            boundary[:, 0] = True
            boundary[:, -1] = True
            valid = boundary & local_non_shadow
            if not np.any(valid):
                continue

            local_labels = class_map[y0:y1, x0:x1]
            candidate_labels = local_labels[valid]
            counts = np.bincount(candidate_labels.ravel(), minlength=len(class_names))
            for shadow_index in shadow_indices:
                counts[shadow_index] = 0
            donor_class = int(np.argmax(counts))
            if counts[donor_class] <= 0:
                continue

            donor_values = direct_linear_map[y0:y1, x0:x1][valid & (local_labels == donor_class)]
            if donor_values.size == 0:
                continue

            filled_linear_map[y, x] = float(donor_values.mean())
            donor_class_map[y, x] = donor_class
            donor_radius_map[y, x] = float(radius)
            assigned = True
            break

        if not assigned:
            filled_linear_map[y, x] = global_donor_value
            donor_class_map[y, x] = np.int16(global_donor_class)
            donor_radius_map[y, x] = float(max_radius)

    return filled_linear_map, donor_class_map, donor_radius_map


def map_rcs_to_pixels(
    rgb_image: np.ndarray,
    class_map: np.ndarray,
    class_names: Sequence[str],
    config_path: str | Path,
) -> RcsMappingResult:
    config = _load_rcs_config(config_path)
    _validate_rcs_config(config, class_names)
    pixel_area_m2 = PIXEL_AREA_M2
    low_percentile = _LOW_PERCENTILE
    high_percentile = _HIGH_PERCENTILE

    brightness = _value_brightness(rgb_image)
    normalized_brightness, brightness_stats = _classwise_normalize(
        brightness=brightness,
        class_map=class_map,
        class_names=class_names,
        low_percentile=low_percentile,
        high_percentile=high_percentile,
    )

    linear_map_m2 = np.zeros_like(brightness, dtype=np.float32)
    sigma0_db_map = np.full_like(brightness, np.nan, dtype=np.float32)
    class_reports: Dict[str, Dict[str, object]] = {}
    surface_area_offset_db = float(10.0 * np.log10(pixel_area_m2))

    for class_index, class_name in enumerate(class_names):
        mask = class_map == class_index
        if not np.any(mask):
            class_reports[class_name] = {
                "pixel_count": 0,
                "pixel_share": 0.0,
            }
            continue

        if class_name in _SHADOW_CLASS_NAMES:
            continue

        min_db, max_db = _resolve_reference_range_db(config, class_name)
        brightness_values = normalized_brightness[mask].astype(np.float32)

        if class_name == "vehicle":
            component_map, component_reports = _vehicle_component_map(
                mask=mask,
                normalized_brightness=normalized_brightness,
                min_db=min_db,
                max_db=max_db,
                distribution_bias=_VEHICLE_DISTRIBUTION_BIAS,
            )
            linear_map_m2[mask] = component_map[mask]
            sigma0_db_map[mask] = np.nan
            class_reports[class_name] = {
                "pixel_count": int(mask.sum()),
                "pixel_share": float(mask.mean()),
                "reference_range_db": [min_db, max_db],
                "brightness_stats": brightness_stats.get(class_name, {}),
                "sum_rcs_m2": float(component_map[mask].sum()),
                "mean_pixel_rcs_m2": float(component_map[mask].mean()),
                "mean_pixel_rcs_dbsm": float(_linear_to_db(component_map[mask]).mean()),
                "component_reports": component_reports,
            }
            continue

        sigma0_db = (min_db + brightness_values * (max_db - min_db)).astype(np.float32)
        sigma0_db_map[mask] = sigma0_db
        per_pixel_dbsm = sigma0_db + surface_area_offset_db
        linear_map_m2[mask] = _db_to_linear(per_pixel_dbsm)
        class_reports[class_name] = {
            "pixel_count": int(mask.sum()),
            "pixel_share": float(mask.mean()),
            "reference_range_db": [min_db, max_db],
            "effective_pixel_rcs_dbsm_range": [
                float(min_db + surface_area_offset_db),
                float(max_db + surface_area_offset_db),
            ],
            "brightness_stats": brightness_stats.get(class_name, {}),
            "sum_rcs_m2": float(linear_map_m2[mask].sum()),
            "mean_pixel_rcs_m2": float(linear_map_m2[mask].mean()),
            "mean_pixel_rcs_dbsm": float(_linear_to_db(linear_map_m2[mask]).mean()),
        }

    shadow_indices = _shadow_indices(class_names)
    non_shadow_mask = ~np.isin(class_map, shadow_indices)
    if np.any(non_shadow_mask):
        shadow_mask = ~non_shadow_mask
        filled_shadow_linear, shadow_donor_classes, shadow_radius_map = _nearest_majority_shadow_fill(
            shadow_mask=shadow_mask,
            class_map=class_map,
            direct_linear_map=linear_map_m2,
            class_names=class_names,
        )
        linear_map_m2[shadow_mask] = filled_shadow_linear[shadow_mask]
        sigma0_db_map[shadow_mask] = _linear_to_db(linear_map_m2[shadow_mask]) - surface_area_offset_db

        for shadow_class_name in _SHADOW_CLASS_NAMES:
            shadow_index = class_names.index(shadow_class_name)
            shadow_class_mask = class_map == shadow_index
            donor_histogram: Dict[str, int] = {}
            if np.any(shadow_class_mask):
                donor_values, donor_counts = np.unique(shadow_donor_classes[shadow_class_mask], return_counts=True)
                for donor_index, donor_count in zip(donor_values.tolist(), donor_counts.tolist()):
                    if donor_index < 0:
                        continue
                    donor_histogram[class_names[int(donor_index)]] = int(donor_count)
                class_reports[shadow_class_name] = {
                    "pixel_count": int(shadow_class_mask.sum()),
                    "pixel_share": float(shadow_class_mask.mean()),
                    "brightness_stats": brightness_stats.get(shadow_class_name, {}),
                    "sum_rcs_m2": float(linear_map_m2[shadow_class_mask].sum()),
                    "mean_pixel_rcs_m2": float(linear_map_m2[shadow_class_mask].mean()),
                    "mean_pixel_rcs_dbsm": float(_linear_to_db(linear_map_m2[shadow_class_mask]).mean()),
                    "donor_class_histogram": donor_histogram,
                    "mean_search_radius_px": float(shadow_radius_map[shadow_class_mask].mean()),
                }
            else:
                class_reports[shadow_class_name] = {
                    "pixel_count": 0,
                    "pixel_share": 0.0,
                }
    else:
        shadow_donor_classes = np.full(class_map.shape, -1, dtype=np.int16)
        shadow_radius_map = np.zeros(class_map.shape, dtype=np.float32)

    dbsm_map = _linear_to_db(linear_map_m2)
    heatmap = _render_heatmap(dbsm_map)
    overlay = (
        rgb_image.astype(np.float32) * 0.56 + heatmap.astype(np.float32) * 0.44
    ).clip(0, 255).astype(np.uint8)

    valid_samples = dbsm_map[np.isfinite(dbsm_map)]
    summary = {
        "min_pixel_rcs_dbsm": float(valid_samples.min()) if valid_samples.size else None,
        "max_pixel_rcs_dbsm": float(valid_samples.max()) if valid_samples.size else None,
        "mean_pixel_rcs_dbsm": float(valid_samples.mean()) if valid_samples.size else None,
        "total_rcs_m2": float(linear_map_m2.sum()),
        "pixel_area_m2": pixel_area_m2,
        "config_path": str(config["config_path"]),
    }
    report: Dict[str, object] = {
        "summary": summary,
        "brightness_normalization": {
            "metric": "hsv_value",
            "low_percentile": low_percentile,
            "high_percentile": high_percentile,
        },
        "class_reports": class_reports,
        "notes": [
            "Distributed classes use brightness-normalized ranges from the config.",
            "Shadow pixels inherit EPR from the nearest ring of non-shadow classes using the most frequent class on that ring.",
            "Vehicle keeps object-level EPR and distributes it across connected components."
        ],
    }

    debug_maps: Dict[str, np.ndarray] = {
        "brightness_value": brightness.astype(np.float32),
        "brightness_normalized_by_class": normalized_brightness.astype(np.float32),
        "rcs_dbsm_map": dbsm_map.astype(np.float32),
        "rcs_heatmap_rgb": heatmap,
        "rcs_overlay_rgb": overlay,
        "shadow_donor_class_map": shadow_donor_classes.astype(np.float32),
        "shadow_search_radius_px": shadow_radius_map.astype(np.float32),
    }
    finite_sigma = np.isfinite(sigma0_db_map)
    if np.any(finite_sigma):
        sigma_debug = sigma0_db_map.copy()
        sigma_debug[~finite_sigma] = np.nanmin(sigma0_db_map[finite_sigma])
        debug_maps["sigma0_db_map"] = sigma_debug.astype(np.float32)

    return RcsMappingResult(
        linear_map_m2=linear_map_m2.astype(np.float32),
        dbsm_map=dbsm_map.astype(np.float32),
        heatmap=heatmap,
        overlay=overlay,
        brightness=brightness.astype(np.float32),
        normalized_brightness=normalized_brightness.astype(np.float32),
        debug_maps=debug_maps,
        report=report,
        config=config,
    )
