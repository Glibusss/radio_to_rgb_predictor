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


def _render_grayscale(values_dbw: np.ndarray, lower_dbw: float, upper_dbw: float) -> np.ndarray:
    finite = np.isfinite(values_dbw)
    if not np.any(finite):
        return np.zeros(values_dbw.shape + (3,), dtype=np.uint8)
    if upper_dbw - lower_dbw < 1e-6:
        upper_dbw = lower_dbw + 1e-6
    scaled = ((values_dbw - lower_dbw) / (upper_dbw - lower_dbw)).clip(0.0, 1.0)
    gray = (scaled * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def _keep_central_boundary_cluster(
    boundary_mask: np.ndarray,
    origin_px: tuple[float, float],
    center_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
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
        return boundary_mask.astype(bool), np.zeros_like(boundary_mask, dtype=bool), cluster_report

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
        return np.zeros_like(boundary_mask, dtype=bool), np.zeros_like(boundary_mask, dtype=bool), cluster_report

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
    return kept_mask.astype(bool), support_region.astype(bool), cluster_report


def _build_radar_boundaries(
    rgb_image: np.ndarray,
    class_map: np.ndarray,
    class_names: Sequence[str],
    origin_px: tuple[float, float],
    boundary_classes: Sequence[str],
    thickness_px: int,
    center_radius_px: float,
) -> tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], Dict[str, Dict[str, int]], Dict[str, object], np.ndarray]:
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

    filtered_boundaries, support_region, cluster_report = _keep_central_boundary_cluster(
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
    return boundary_map, boundary_overlay, per_class_maps, report, cluster_report, support_region.astype(bool)


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
    if receiver_max_level_dbw <= receiver_sensitivity_dbw:
        raise ValueError("receiver_max_level_dbw must be greater than receiver_sensitivity_dbw.")
    use_radar_boundaries = bool(switches.get("use_radar_boundaries", True))
    raw_boundary_classes = config.get("boundary_classes", ["forest", "building", "shrub"])
    if isinstance(raw_boundary_classes, Sequence) and not isinstance(raw_boundary_classes, str):
        boundary_classes = [str(value) for value in raw_boundary_classes]
    else:
        boundary_classes = ["forest", "building", "shrub"]
    boundary_thickness_px = int(config.get("boundary_thickness_px", 2))
    boundary_cluster_center_radius_px = float(config.get("boundary_cluster_center_radius_px", 0.35 * min(rgb_image.shape[:2])))

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
        * np.maximum(effective_rcs_map_m2, 0.0)
        / np.maximum(geometry["slant_range_m"], 1.0) ** 4
    ).astype(np.float32)
    received_power_dbw = _linear_to_db(received_power_w)
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
        "visual_black_level_dbw": float(receiver_sensitivity_dbw),
        "visual_white_level_dbw": float(receiver_max_level_dbw),
    }
    report: Dict[str, object] = {
        "equation": "Pr = Pt * Gt * Gr * lambda^2 * sigma / ((4*pi)^3 * R^4 * L)",
        "summary": summary,
        "boundary_report": boundary_report,
        "boundary_cluster_report": boundary_cluster_report,
        "boundary_scatter_cleanup_report": boundary_scatter_cleanup_report,
        "config": {
            "origin_mode": str(config.get("origin_mode", "image_center")),
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
        },
        "pipeline_switches": dict(switches),
    }
    debug_maps: Dict[str, np.ndarray] = {
        "radar_slant_range_m": geometry["slant_range_m"].astype(np.float32),
        "radar_ground_offset_m": geometry["ground_offset_m"].astype(np.float32),
        "radar_effective_rcs_map_m2": effective_rcs_map_m2.astype(np.float32),
        "radar_received_power_dbw": received_power_dbw.astype(np.float32),
        "radar_equation_map_rgb": heatmap,
        "radar_equation_overlay_rgb": overlay,
        "radar_boundary_map_rgb": boundary_map,
        "radar_boundary_overlay_rgb": boundary_overlay,
        "radar_boundary_cluster_mask": (boundary_map[:, :, 0] > 0).astype(np.float32),
        "radar_boundary_support_region": boundary_support_region.astype(np.float32),
    }
    for class_name, boundary_values in per_class_boundaries.items():
        debug_maps[f"radar_boundary_{class_name}"] = boundary_values.astype(np.float32)

    return RadarEquationResult(
        received_power_w=received_power_w,
        received_power_dbw=received_power_dbw,
        heatmap=heatmap,
        overlay=overlay,
        boundary_map=boundary_map,
        boundary_overlay=boundary_overlay,
        debug_maps=debug_maps,
        report=report,
        config=config,
        origin_px=origin_px,
    )
