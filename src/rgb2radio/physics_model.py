from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .common import normalize01, normalize_to_uint8
from .radar_reference import RadarReference
from .rcs_library import RCS_PROFILES
from .terrain_resnet import TERRAIN_CLASS_NAMES, render_terrain_map


@dataclass(frozen=True)
class RadarPhysicsConfig:
    frequency_ghz: float = 9.4
    tx_power_w: float = 50.0
    antenna_height_m: float = 3.0
    range_resolution_m: float = 1.5
    azimuth_resolution_deg: float = 1.0
    display_angle_step_deg: float = 1.0
    meters_per_pixel: float = 0.375
    reference_range_m: float = 200.0
    effective_looks: float = 2.6
    foliage_loss_np_per_m: float = 0.012
    shadow_floor: float = 0.34
    display_canvas_px: int = 768
    stc_exponent: float = 1.6
    seed: int = 42

    @property
    def wavelength_m(self) -> float:
        return 299_792_458.0 / (self.frequency_ghz * 1e9)


@dataclass(frozen=True)
class SynthesisReport:
    ground_gray: np.ndarray
    display_gray: np.ndarray
    display_optical_rgb: np.ndarray
    pure_display_gray: np.ndarray
    pure_display_optical_rgb: np.ndarray
    terrain_rgb: np.ndarray
    origin_px: Tuple[float, float]
    debug_maps: Dict[str, np.ndarray]
    metrics: Dict[str, float]
    validation: Dict[str, Any]


def _estimate_radar_origin(probabilities: np.ndarray) -> Tuple[float, float]:
    concrete = probabilities[TERRAIN_CLASS_NAMES.index("concrete_building")]
    metal = probabilities[TERRAIN_CLASS_NAMES.index("metal_building")]
    wood = probabilities[TERRAIN_CLASS_NAMES.index("wood_building")]
    asphalt = probabilities[TERRAIN_CLASS_NAMES.index("asphalt_road")]
    dirt = probabilities[TERRAIN_CLASS_NAMES.index("dirt_road")]
    weights = 0.95 * metal + 0.80 * concrete + 0.55 * wood + 0.25 * asphalt + 0.18 * dirt
    weights = cv2.GaussianBlur(weights.astype(np.float32), (0, 0), sigmaX=9.0, sigmaY=9.0)
    total = float(weights.sum())
    height, width = weights.shape
    if total < 1e-8:
        return width / 2.0, height / 2.0
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    return float((xx * weights).sum() / total), float((yy * weights).sum() / total)


def _geometry(shape: Tuple[int, int], origin_px: Tuple[float, float], config: RadarPhysicsConfig) -> Dict[str, np.ndarray]:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    dx_m = (xx - origin_px[0]) * config.meters_per_pixel
    dy_m = (yy - origin_px[1]) * config.meters_per_pixel
    ground_range_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)
    safe_ground_range_m = np.maximum(ground_range_m, config.meters_per_pixel)
    slant_range_m = np.sqrt(safe_ground_range_m * safe_ground_range_m + config.antenna_height_m ** 2)

    look_x = dx_m / safe_ground_range_m
    look_y = dy_m / safe_ground_range_m
    grazing = np.arctan2(config.antenna_height_m, safe_ground_range_m)
    azimuth_cell_m = np.maximum(
        safe_ground_range_m * np.deg2rad(config.azimuth_resolution_deg),
        config.meters_per_pixel,
    )
    return {
        "ground_range_m": ground_range_m.astype(np.float32),
        "slant_range_m": slant_range_m.astype(np.float32),
        "look_x": look_x.astype(np.float32),
        "look_y": look_y.astype(np.float32),
        "grazing_gain": (0.38 + 0.62 * normalize01(np.sin(grazing))).astype(np.float32),
        "azimuth_cell_m": azimuth_cell_m.astype(np.float32),
    }


def _semantic_structure(probabilities: np.ndarray) -> Dict[str, np.ndarray]:
    class_labels = np.argmax(probabilities, axis=0).astype(np.uint8)
    semantic_gray = (
        class_labels.astype(np.float32) / max(len(TERRAIN_CLASS_NAMES) - 1, 1) * 255.0
    ).astype(np.uint8)
    grad_x = cv2.Sobel(semantic_gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(semantic_gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.maximum(np.sqrt(grad_x * grad_x + grad_y * grad_y), 1e-6)
    edges = normalize01(grad_mag)

    building_mask = normalize01(
        probabilities[TERRAIN_CLASS_NAMES.index("concrete_building")]
        + probabilities[TERRAIN_CLASS_NAMES.index("metal_building")]
        + probabilities[TERRAIN_CLASS_NAMES.index("wood_building")]
    )
    building_gray = (building_mask * 255.0).astype(np.float32)
    corners = cv2.cornerHarris(building_gray, blockSize=2, ksize=3, k=0.04)
    corners = normalize01(np.maximum(cv2.GaussianBlur(corners, (5, 5), 0.0), 0.0))

    road_mask = normalize01(
        probabilities[TERRAIN_CLASS_NAMES.index("dirt_road")]
        + probabilities[TERRAIN_CLASS_NAMES.index("asphalt_road")]
    )
    water_mask = normalize01(
        probabilities[TERRAIN_CLASS_NAMES.index("water")]
        + probabilities[TERRAIN_CLASS_NAMES.index("rippling_water")]
    )
    forest_mask = probabilities[TERRAIN_CLASS_NAMES.index("forest")]
    compound_mask = normalize01(
        cv2.GaussianBlur(np.maximum(building_mask, 0.70 * road_mask), (0, 0), sigmaX=10.0, sigmaY=10.0)
    )
    return {
        "edges": edges.astype(np.float32),
        "corners": corners.astype(np.float32),
        "normal_x": (grad_x / grad_mag).astype(np.float32),
        "normal_y": (grad_y / grad_mag).astype(np.float32),
        "building_mask": building_mask.astype(np.float32),
        "road_mask": road_mask.astype(np.float32),
        "water_mask": water_mask.astype(np.float32),
        "forest_mask": forest_mask.astype(np.float32),
        "compound_mask": compound_mask.astype(np.float32),
    }


def _physics_maps(
    probabilities: np.ndarray,
    structure: Dict[str, np.ndarray],
    geometry: Dict[str, np.ndarray],
    config: RadarPhysicsConfig,
) -> Dict[str, np.ndarray]:
    specific_sigma = np.zeros_like(probabilities[0], dtype=np.float32)
    base_point_rcs = np.zeros_like(probabilities[0], dtype=np.float32)
    height_map = np.zeros_like(probabilities[0], dtype=np.float32)
    attenuation_np_per_m = np.zeros_like(probabilities[0], dtype=np.float32)
    scatterer_density = np.zeros_like(probabilities[0], dtype=np.float32)
    specularity = np.zeros_like(probabilities[0], dtype=np.float32)
    roughness = np.zeros_like(probabilities[0], dtype=np.float32)
    cell_area_m2 = config.meters_per_pixel ** 2

    for class_index, class_name in enumerate(TERRAIN_CLASS_NAMES):
        probs = probabilities[class_index]
        profile = RCS_PROFILES[class_name]
        specific_sigma += probs * (10.0 ** (profile.sigma0_db / 10.0)) * cell_area_m2
        base_point_rcs += probs * profile.point_rcs_m2
        height_map += probs * profile.mean_height_m
        attenuation_np_per_m += probs * profile.attenuation_np_per_m
        scatterer_density += probs * profile.scatterer_density_m2
        specularity += probs * profile.specularity
        roughness += probs * profile.roughness

    aspect_alignment = np.abs(
        structure["normal_x"] * geometry["look_x"] + structure["normal_y"] * geometry["look_y"]
    )
    aspect_gain = (0.34 + 0.66 * np.power(aspect_alignment, 1.35)).astype(np.float32)

    concrete = probabilities[TERRAIN_CLASS_NAMES.index("concrete_building")]
    metal = probabilities[TERRAIN_CLASS_NAMES.index("metal_building")]
    wood = probabilities[TERRAIN_CLASS_NAMES.index("wood_building")]
    dirt = probabilities[TERRAIN_CLASS_NAMES.index("dirt_road")]
    asphalt = probabilities[TERRAIN_CLASS_NAMES.index("asphalt_road")]
    rippling = probabilities[TERRAIN_CLASS_NAMES.index("rippling_water")]
    forest = probabilities[TERRAIN_CLASS_NAMES.index("forest")]
    building_mask = structure["building_mask"]
    road_mask = structure["road_mask"]

    point_boost = normalize01(
        0.42 * structure["corners"]
        + 0.28 * structure["edges"]
        + 0.26 * building_mask
        + 0.18 * road_mask
        + 0.16 * rippling
    )
    double_bounce = (0.55 * metal + 0.35 * concrete + 0.18 * wood) * point_boost * (0.30 + 0.70 * aspect_gain)
    road_alignment_gain = (0.26 * asphalt + 0.18 * dirt) * (0.40 + 0.60 * aspect_gain)

    specific_sigma *= geometry["grazing_gain"] * (0.78 + 0.30 * roughness + 0.34 * road_alignment_gain)
    point_rcs = base_point_rcs * np.clip(0.22 + 1.65 * point_boost + 2.10 * double_bounce + 0.40 * specularity, 0.0, None)
    occluder_strength = np.clip(0.82 * building_mask + 0.55 * forest + 0.18 * structure["corners"], 0.0, 1.0)

    return {
        "specific_sigma": specific_sigma.astype(np.float32),
        "point_rcs": point_rcs.astype(np.float32),
        "height_map": height_map.astype(np.float32),
        "attenuation_np_per_m": attenuation_np_per_m.astype(np.float32),
        "scatterer_density": scatterer_density.astype(np.float32),
        "point_boost": point_boost.astype(np.float32),
        "aspect_gain": aspect_gain.astype(np.float32),
        "occluder_strength": occluder_strength.astype(np.float32),
        "roughness": roughness.astype(np.float32),
        "specularity": specularity.astype(np.float32),
    }


def _ray_effects(
    geometry: Dict[str, np.ndarray],
    height_map: np.ndarray,
    attenuation_np_per_m: np.ndarray,
    occluder_strength: np.ndarray,
    origin_px: Tuple[float, float],
    config: RadarPhysicsConfig,
) -> Dict[str, np.ndarray]:
    ground_range = geometry["ground_range_m"]
    height, width = ground_range.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    angle = np.arctan2(yy - origin_px[1], xx - origin_px[0])

    max_range = float(np.max(ground_range))
    effective_cross_range_m = max(config.range_resolution_m * 2.5, config.meters_per_pixel * 10.0)
    num_bins = max(360, int(np.ceil(2.0 * np.pi * max_range / max(effective_cross_range_m, 1e-3))))
    flat_bins = np.floor(((angle.ravel() + np.pi) / (2.0 * np.pi)) * num_bins).astype(np.int32)
    flat_bins = np.clip(flat_bins, 0, num_bins - 1)
    flat_range = ground_range.ravel().astype(np.float32)
    flat_height = height_map.ravel().astype(np.float32)
    flat_foliage = attenuation_np_per_m.ravel().astype(np.float32)
    flat_occluder = occluder_strength.ravel().astype(np.float32)

    order = np.lexsort((flat_range, flat_bins))
    transmittance = np.ones_like(flat_range, dtype=np.float32)
    shadow = np.ones_like(flat_range, dtype=np.float32)

    current_bin = -1
    cumulative_loss = 0.0
    last_range = 0.0
    max_obstruction_angle = -1e9

    for idx in order:
        ray_bin = int(flat_bins[idx])
        radius = float(max(flat_range[idx], config.meters_per_pixel))
        if ray_bin != current_bin:
            current_bin = ray_bin
            cumulative_loss = 0.0
            last_range = 0.0
            max_obstruction_angle = -1e9

        dr = max(radius - last_range, config.meters_per_pixel)
        cumulative_loss += float(flat_foliage[idx]) * dr
        transmittance[idx] = np.exp(-2.0 * cumulative_loss)

        obstruction_height = max(flat_height[idx] - config.antenna_height_m, 0.0) * float(flat_occluder[idx])
        obstruction_angle = np.arctan2(obstruction_height, radius)
        if obstruction_angle + np.deg2rad(0.12) < max_obstruction_angle:
            excess = max_obstruction_angle - obstruction_angle
            shadow[idx] = max(config.shadow_floor, float(np.exp(-3.6 * excess)))
        max_obstruction_angle = max(max_obstruction_angle, obstruction_angle)
        last_range = radius

    transmittance = cv2.GaussianBlur(transmittance.reshape(ground_range.shape), (0, 0), sigmaX=0.8, sigmaY=0.8)
    shadow = cv2.GaussianBlur(shadow.reshape(ground_range.shape), (0, 0), sigmaX=1.6, sigmaY=1.6)
    return {
        "transmittance": transmittance.astype(np.float32),
        "shadow": shadow.astype(np.float32),
    }


def _build_scatterer_map(
    probabilities: np.ndarray,
    physics: Dict[str, np.ndarray],
    structure: Dict[str, np.ndarray],
    config: RadarPhysicsConfig,
) -> Dict[str, np.ndarray]:
    cell_area_m2 = config.meters_per_pixel ** 2
    density_per_cell = np.clip(physics["scatterer_density"] * cell_area_m2, 0.01, 0.95)

    building_mask = structure["building_mask"]
    road_mask = structure["road_mask"]
    water_mask = structure["water_mask"]
    forest_mask = structure["forest_mask"]
    compound_mask = structure["compound_mask"]
    point_boost = physics["point_boost"]

    rng = np.random.default_rng(config.seed)
    micro_random = rng.random(probabilities.shape[1:], dtype=np.float32)
    bright_random = rng.random(probabilities.shape[1:], dtype=np.float32)
    road_linearity = normalize01(cv2.GaussianBlur(road_mask * structure["edges"], (0, 0), sigmaX=2.2, sigmaY=2.2))

    micro_prob = np.clip(
        density_per_cell
        * (0.12 + 0.18 * physics["roughness"] + 0.10 * forest_mask + 0.08 * water_mask + 0.42 * compound_mask),
        0.005,
        0.58,
    )
    bright_prob = np.clip(
        0.004
        + 0.24 * point_boost
        + 0.34 * building_mask
        + 0.14 * road_linearity
        + 0.18 * compound_mask
        + 0.10 * probabilities[TERRAIN_CLASS_NAMES.index("rippling_water")],
        0.004,
        0.84,
    )

    micro_mask = (micro_random < micro_prob).astype(np.float32)
    bright_mask = (bright_random < bright_prob).astype(np.float32)

    micro_rcs = (
        micro_mask
        * physics["specific_sigma"]
        * (0.28 + 0.54 * compound_mask + 0.18 * forest_mask)
        / np.maximum(micro_prob, 1e-3)
    )
    bright_rcs = (
        bright_mask
        * physics["point_rcs"]
        * (0.60 + 0.52 * compound_mask + 0.20 * road_linearity)
        / np.maximum(bright_prob, 1e-3)
    )
    scatterer_rcs = np.maximum(micro_rcs + bright_rcs, 0.0)

    return {
        "micro_mask": micro_mask.astype(np.float32),
        "bright_mask": bright_mask.astype(np.float32),
        "micro_rcs": micro_rcs.astype(np.float32),
        "bright_rcs": bright_rcs.astype(np.float32),
        "scatterer_rcs": scatterer_rcs.astype(np.float32),
        "compound_mask": compound_mask.astype(np.float32),
        "road_linearity": road_linearity.astype(np.float32),
    }


def _strong_scatterer_overlay(
    source_power: np.ndarray,
    origin_px: Tuple[float, float],
    output_shape: Tuple[int, int],
    center_px: Tuple[float, float],
    radius_px: float,
) -> np.ndarray:
    threshold = float(np.quantile(source_power, 0.996))
    ys, xs = np.where(source_power >= threshold)
    overlay = np.zeros(output_shape, dtype=np.float32)
    if len(xs) == 0:
        return overlay

    values = source_power[ys, xs]
    order = np.argsort(-values)
    ys = ys[order][:280]
    xs = xs[order][:280]
    values = values[order][:280]

    scale = radius_px / max(
        min(
            origin_px[0],
            origin_px[1],
            source_power.shape[1] - 1.0 - origin_px[0],
            source_power.shape[0] - 1.0 - origin_px[1],
        ),
        1.0,
    )

    for y, x, value in zip(ys, xs, values):
        dx = (float(x) - origin_px[0]) * scale
        dy = (float(y) - origin_px[1]) * scale
        dist = float(np.hypot(dx, dy))
        if dist < 2.0:
            continue
        center = (int(round(center_px[0] + dx)), int(round(center_px[1] + dy)))
        tail = 6.0 + 18.0 * float(value) + 0.028 * dist
        start_scale = max((dist - 0.12 * tail) / dist, 0.0)
        end_scale = (dist + tail) / dist
        start = (int(round(center_px[0] + dx * start_scale)), int(round(center_px[1] + dy * start_scale)))
        end = (int(round(center_px[0] + dx * end_scale)), int(round(center_px[1] + dy * end_scale)))
        brightness = 0.25 + 0.85 * float(value)
        cv2.line(overlay, start, end, color=brightness, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, center, radius=1, color=min(brightness * 1.2, 1.0), thickness=-1, lineType=cv2.LINE_AA)

    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=1.0, sigmaY=1.0)
    return normalize01(overlay)


def _make_circle_mask(
    output_shape: Tuple[int, int],
    center_px: Tuple[float, float],
    radius_px: float,
) -> np.ndarray:
    yy, xx = np.mgrid[0 : output_shape[0], 0 : output_shape[1]].astype(np.float32)
    dist = np.sqrt((xx - center_px[0]) ** 2 + (yy - center_px[1]) ** 2)
    return np.clip((radius_px - dist) / max(radius_px * 0.05, 1.0), 0.0, 1.0)


def _default_display_layout(config: RadarPhysicsConfig) -> tuple[Tuple[int, int], Tuple[float, float], float]:
    side = int(max(config.display_canvas_px, 256))
    center = (side / 2.0, side / 2.0)
    radius = side * 0.46
    return (side, side), center, radius


def _reference_display_layout(reference: RadarReference) -> tuple[Tuple[int, int], Tuple[float, float], float]:
    return reference.clean_gray.shape, reference.center_px, reference.radius_px * 0.95


def _crop_display_optical(
    rgb_image: np.ndarray,
    origin_px: Tuple[float, float],
    output_shape: Tuple[int, int],
    center_px: Tuple[float, float],
    radius_px: float,
) -> np.ndarray:
    height, width = rgb_image.shape[:2]
    crop_radius = int(
        max(
            16,
            np.floor(min(origin_px[0], origin_px[1], width - 1.0 - origin_px[0], height - 1.0 - origin_px[1])),
        )
    )
    center_x = int(round(origin_px[0]))
    center_y = int(round(origin_px[1]))
    crop = rgb_image[center_y - crop_radius : center_y + crop_radius + 1, center_x - crop_radius : center_x + crop_radius + 1]
    if crop.size == 0:
        return np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)

    diameter = max(2, int(round(radius_px * 2.0)))
    resized = cv2.resize(crop, (diameter, diameter), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((output_shape[0], output_shape[1], 3), dtype=np.uint8)
    x0 = int(round(center_px[0] - diameter / 2.0))
    y0 = int(round(center_px[1] - diameter / 2.0))
    x1 = x0 + diameter
    y1 = y0 + diameter
    canvas[y0:y1, x0:x1] = resized

    mask = _make_circle_mask(canvas.shape[:2], center_px, radius_px)
    return (canvas.astype(np.float32) * mask[:, :, None]).clip(0, 255).astype(np.uint8)


def _display_max_range_m(
    source_shape: Tuple[int, int],
    origin_px: Tuple[float, float],
    config: RadarPhysicsConfig,
) -> float:
    max_radius_px = max(
        16.0,
        min(
            origin_px[0],
            origin_px[1],
            source_shape[1] - 1.0 - origin_px[0],
            source_shape[0] - 1.0 - origin_px[1],
        ),
    )
    return float(max_radius_px * config.meters_per_pixel)


def _build_resolution_cell_map(
    geometry: Dict[str, np.ndarray],
    physics: Dict[str, np.ndarray],
    structure: Dict[str, np.ndarray],
    ray_effects: Dict[str, np.ndarray],
    origin_px: Tuple[float, float],
    config: RadarPhysicsConfig,
) -> Dict[str, Any]:
    source_shape = structure["building_mask"].shape
    max_range_m = _display_max_range_m(source_shape, origin_px, config)
    angle_step_deg = max(config.display_angle_step_deg, config.azimuth_resolution_deg)
    angle_step_rad = np.deg2rad(angle_step_deg)
    num_azimuth_bins = max(1, int(np.ceil(360.0 / angle_step_deg)))
    num_range_bins = max(1, int(np.ceil(max_range_m / config.range_resolution_m)))

    height, width = source_shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    scene_angle = np.mod(np.arctan2(yy - origin_px[1], xx - origin_px[0]), 2.0 * np.pi)
    ground_range_m = geometry["ground_range_m"]
    valid_scene = ground_range_m <= max_range_m

    scene_azimuth_index = np.clip(
        np.floor(scene_angle / angle_step_rad).astype(np.int32),
        0,
        num_azimuth_bins - 1,
    )
    scene_range_index = np.clip(
        np.floor(ground_range_m / config.range_resolution_m).astype(np.int32),
        0,
        num_range_bins - 1,
    )
    flat_index = (scene_azimuth_index[valid_scene] * num_range_bins + scene_range_index[valid_scene]).astype(np.int32)
    total_bins = num_azimuth_bins * num_range_bins

    road_linearity = normalize01(cv2.GaussianBlur(structure["road_mask"] * structure["edges"], (0, 0), sigmaX=2.0, sigmaY=2.0))
    compound_strength = np.clip(0.72 * structure["compound_mask"] + 0.28 * road_linearity, 0.0, 1.0)
    deterministic_rcs = (
        0.78 * physics["specific_sigma"] * (0.58 + 0.42 * physics["roughness"])
        + 0.22 * physics["point_rcs"] * (0.18 + 0.82 * physics["point_boost"])
    )
    deterministic_rcs *= physics["aspect_gain"] * (0.74 + 0.26 * compound_strength)

    occupancy = np.bincount(flat_index, minlength=total_bins).astype(np.float32)

    def mean_per_cell(values: np.ndarray) -> np.ndarray:
        flat_values = values[valid_scene].astype(np.float32)
        summed = np.bincount(flat_index, weights=flat_values, minlength=total_bins).astype(np.float32)
        mean = summed / np.maximum(occupancy, 1.0)
        return mean.reshape(num_azimuth_bins, num_range_bins)

    cell_rcs = mean_per_cell(deterministic_rcs)
    cell_transmittance = mean_per_cell(ray_effects["transmittance"])
    cell_shadow = mean_per_cell(ray_effects["shadow"])
    cell_compound = mean_per_cell(compound_strength)
    cell_forest = mean_per_cell(structure["forest_mask"])
    cell_occupancy = occupancy.reshape(num_azimuth_bins, num_range_bins)

    radar_constant = config.tx_power_w * (config.wavelength_m ** 2) / ((4.0 * np.pi) ** 3)
    center_ground_range_m = (np.arange(num_range_bins, dtype=np.float32) + 0.5) * config.range_resolution_m
    center_slant_range_m = np.sqrt(np.maximum(center_ground_range_m, config.meters_per_pixel) ** 2 + config.antenna_height_m ** 2)
    stc_gain = np.power(
        np.maximum(center_slant_range_m / max(config.reference_range_m, 1.0), 0.15),
        config.stc_exponent,
    ).astype(np.float32)

    cell_power = (
        radar_constant
        * cell_rcs
        / np.maximum(center_slant_range_m[None, :], 1.0) ** 4
        * cell_transmittance
        * cell_shadow
        * stc_gain[None, :]
    )
    cell_power *= 0.82 + 0.18 * normalize01(cell_compound)
    cell_power += 0.015 * normalize01(cell_rcs) * np.clip(cell_forest, 0.0, 1.0)
    cell_power = np.maximum(cell_power.astype(np.float32), 0.0)

    return {
        "cell_power": cell_power,
        "cell_rcs": cell_rcs.astype(np.float32),
        "cell_transmittance": cell_transmittance.astype(np.float32),
        "cell_shadow": cell_shadow.astype(np.float32),
        "cell_compound": cell_compound.astype(np.float32),
        "cell_occupancy": cell_occupancy.astype(np.float32),
        "max_range_m": float(max_range_m),
        "angle_step_rad": float(angle_step_rad),
        "num_azimuth_bins": int(num_azimuth_bins),
        "num_range_bins": int(num_range_bins),
    }


def _render_point_display(
    cell_map: Dict[str, Any],
    output_shape: Tuple[int, int],
    center_px: Tuple[float, float],
    radius_px: float,
    config: RadarPhysicsConfig,
    params: Dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    cell_power = cell_map["cell_power"].astype(np.float32)
    valid_cells = cell_map["cell_occupancy"] > 0
    point_strength = np.zeros_like(cell_power, dtype=np.float32)
    if np.any(valid_cells):
        log_power = np.log1p(float(params["gain"]) * np.maximum(cell_power[valid_cells], 0.0))
        lower = float(np.quantile(log_power, float(params["lower_quantile"])))
        upper = float(np.quantile(log_power, float(params["upper_quantile"])))
        if upper - lower < 1e-8:
            mapped = normalize01(log_power)
        else:
            mapped = np.clip((log_power - lower) / (upper - lower), 0.0, 1.0)
        point_strength[valid_cells] = np.power(mapped.astype(np.float32), float(params["gamma"]))
        threshold = float(np.quantile(point_strength[valid_cells], float(params["threshold_quantile"])))
        keep_mask = valid_cells & (point_strength >= threshold)
    else:
        keep_mask = np.zeros_like(cell_power, dtype=bool)

    display = np.zeros(output_shape, dtype=np.float32)
    coords = np.argwhere(keep_mask)
    for azimuth_index, range_index in coords:
        strength = float(point_strength[azimuth_index, range_index])
        angle = (float(azimuth_index) + 0.5) * float(cell_map["angle_step_rad"])
        range_m = (float(range_index) + 0.5) * config.range_resolution_m
        radial_px = (range_m / max(float(cell_map["max_range_m"]), 1e-6)) * radius_px
        x = int(round(center_px[0] + radial_px * np.cos(angle)))
        y = int(round(center_px[1] + radial_px * np.sin(angle)))
        point_radius = int(round(params["point_radius"] + (1 if strength > 0.82 else 0)))
        cv2.circle(display, (x, y), radius=max(point_radius, 1), color=strength, thickness=-1, lineType=cv2.LINE_AA)

    blur_sigma = float(params["blur_sigma"])
    if blur_sigma > 0:
        display = cv2.GaussianBlur(display, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)

    circle_mask = _make_circle_mask(output_shape, center_px, radius_px)
    display = np.clip(display, 0.0, 1.0) * circle_mask
    display_u8 = (np.power(display, float(params["display_gamma"])) * 255.0).clip(0, 255).astype(np.uint8)
    return display_u8, keep_mask.astype(np.float32)


def _optimize_point_display(
    cell_map: Dict[str, Any],
    output_shape: Tuple[int, int],
    center_px: Tuple[float, float],
    radius_px: float,
    config: RadarPhysicsConfig,
    reference: RadarReference | None,
) -> tuple[np.ndarray, Dict[str, float], list[Dict[str, float]], np.ndarray]:
    candidates: list[Dict[str, float]] = []
    for threshold_quantile in (0.42, 0.50, 0.58, 0.66):
        for gamma in (0.72, 0.82, 0.92):
            for gain in (10.0, 16.0, 24.0):
                for blur_sigma in (0.20, 0.45, 0.70):
                    candidates.append(
                        {
                            "threshold_quantile": threshold_quantile,
                            "gamma": gamma,
                            "gain": gain,
                            "blur_sigma": blur_sigma,
                            "point_radius": 1.0,
                            "display_gamma": 0.94,
                            "lower_quantile": 0.46,
                            "upper_quantile": 0.997,
                        }
                    )

    history: list[Dict[str, float]] = []
    best_image: np.ndarray | None = None
    best_keep_mask = np.zeros_like(cell_map["cell_power"], dtype=np.float32)
    best_params = candidates[0]
    best_score = -1e9

    for params in candidates:
        display_gray, keep_mask = _render_point_display(
            cell_map=cell_map,
            output_shape=output_shape,
            center_px=center_px,
            radius_px=radius_px,
            config=config,
            params=params,
        )
        if reference is None:
            best_image = display_gray
            best_keep_mask = keep_mask
            best_params = params
            break

        metrics = _quality_metrics(display_gray, reference)
        score = float(metrics["masked_ncc"] - 0.18 * metrics["masked_mae"])
        history.append(
            {
                "score": score,
                "masked_ncc": float(metrics["masked_ncc"]),
                "masked_mae": float(metrics["masked_mae"]),
                **{key: float(value) for key, value in params.items()},
            }
        )
        if score > best_score:
            best_score = score
            best_image = display_gray
            best_keep_mask = keep_mask
            best_params = params.copy()

    if best_image is None:
        best_image, best_keep_mask = _render_point_display(
            cell_map=cell_map,
            output_shape=output_shape,
            center_px=center_px,
            radius_px=radius_px,
            config=config,
            params=best_params,
        )
    history = sorted(history, key=lambda item: item["score"], reverse=True)[:12]
    return best_image, best_params, history, best_keep_mask


def _render_display(
    source_power: np.ndarray,
    geometry: Dict[str, np.ndarray],
    origin_px: Tuple[float, float],
    output_shape: Tuple[int, int],
    center_px: Tuple[float, float],
    radius_px: float,
    config: RadarPhysicsConfig,
) -> np.ndarray:
    max_range_m = _display_max_range_m(source_power.shape, origin_px, config)
    angle_step_deg = max(config.display_angle_step_deg, config.azimuth_resolution_deg)
    angle_step_rad = np.deg2rad(angle_step_deg)
    num_azimuth_bins = max(1, int(np.ceil(360.0 / angle_step_deg)))
    num_range_bins = max(1, int(np.ceil(max_range_m / config.range_resolution_m)))

    height, width = source_power.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    scene_angle = np.mod(np.arctan2(yy - origin_px[1], xx - origin_px[0]), 2.0 * np.pi)
    scene_range_m = geometry["slant_range_m"]
    valid_scene = geometry["ground_range_m"] <= max_range_m

    scene_azimuth_index = np.clip(
        np.floor(scene_angle / angle_step_rad).astype(np.int32),
        0,
        num_azimuth_bins - 1,
    )
    scene_range_index = np.clip(
        np.floor(scene_range_m / config.range_resolution_m).astype(np.int32),
        0,
        num_range_bins - 1,
    )
    flat_index = (scene_azimuth_index[valid_scene] * num_range_bins + scene_range_index[valid_scene]).astype(np.int32)
    flat_power = np.maximum(source_power[valid_scene].astype(np.float32), 0.0)

    total_bins = num_azimuth_bins * num_range_bins
    integrated_power = np.bincount(flat_index, weights=flat_power, minlength=total_bins).astype(np.float32)
    occupancy = np.bincount(flat_index, minlength=total_bins).astype(np.float32)
    peak_power = np.zeros((total_bins,), dtype=np.float32)
    np.maximum.at(peak_power, flat_index, flat_power)

    integrated_power = integrated_power.reshape(num_azimuth_bins, num_range_bins)
    occupancy = occupancy.reshape(num_azimuth_bins, num_range_bins)
    peak_power = peak_power.reshape(num_azimuth_bins, num_range_bins)
    average_power = integrated_power / np.maximum(occupancy, 1.0)
    diffuse_power = integrated_power / np.sqrt(np.maximum(occupancy, 1.0))
    cell_floor = cv2.GaussianBlur(average_power, (0, 0), sigmaX=1.4, sigmaY=0.9)
    cell_power = 0.34 * peak_power + 0.56 * diffuse_power + 0.10 * cell_floor
    cell_power *= 0.52 + 0.48 * normalize01(average_power + 0.35 * diffuse_power)
    cell_power = cv2.GaussianBlur(cell_power, (0, 0), sigmaX=1.0, sigmaY=0.7)
    cell_power = np.maximum(cell_power, 0.0).astype(np.float32)
    valid_cells = occupancy > 0
    if np.any(valid_cells):
        log_power = np.log1p(16.0 * cell_power[valid_cells])
        lower = float(np.quantile(log_power, 0.50))
        upper = float(np.quantile(log_power, 0.997))
        if upper - lower < 1e-8:
            mapped_cells = normalize01(log_power)
        else:
            mapped_cells = np.clip((log_power - lower) / (upper - lower), 0.0, 1.0)
        cell_power = np.zeros_like(cell_power, dtype=np.float32)
        cell_power[valid_cells] = np.power(mapped_cells.astype(np.float32), 0.78)
    else:
        cell_power = np.zeros_like(cell_power, dtype=np.float32)

    display_yy, display_xx = np.mgrid[0 : output_shape[0], 0 : output_shape[1]].astype(np.float32)
    dx = display_xx - center_px[0]
    dy = display_yy - center_px[1]
    dist_px = np.sqrt(dx * dx + dy * dy)
    circle_mask = _make_circle_mask(output_shape, center_px, radius_px)
    display_range_m = np.clip((dist_px / max(radius_px, 1e-6)) * max_range_m, 0.0, max_range_m - 1e-6)
    display_angle = np.mod(np.arctan2(dy, dx), 2.0 * np.pi)
    display_azimuth_index = np.clip(
        np.floor(display_angle / angle_step_rad).astype(np.int32),
        0,
        num_azimuth_bins - 1,
    )
    display_range_index = np.clip(
        np.floor(display_range_m / config.range_resolution_m).astype(np.int32),
        0,
        num_range_bins - 1,
    )
    display = cell_power[display_azimuth_index, display_range_index] * circle_mask
    display = cv2.GaussianBlur(display.astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=0.8)
    low_band = normalize01(cv2.GaussianBlur(display, (0, 0), sigmaX=10.0, sigmaY=10.0))
    mid_band = normalize01(cv2.GaussianBlur(display, (0, 0), sigmaX=4.5, sigmaY=4.5))
    display = normalize01(0.78 * display + 0.14 * mid_band + 0.06 * low_band + 0.02 * circle_mask)

    ring = np.zeros_like(display, dtype=np.float32)
    cv2.circle(
        ring,
        (int(round(center_px[0])), int(round(center_px[1]))),
        int(round(radius_px)),
        color=0.10,
        thickness=max(1, int(round(radius_px * 0.01))),
        lineType=cv2.LINE_AA,
    )
    display = normalize01((display + ring) * circle_mask)
    display_u8 = normalize_to_uint8(np.power(display, 0.92))
    return (display_u8.astype(np.float32) * circle_mask).clip(0, 255).astype(np.uint8)


def _quality_metrics(display_gray: np.ndarray, reference: RadarReference) -> Dict[str, float]:
    mask = reference.clean_gray > max(12, int(np.quantile(reference.clean_gray, 0.10)))
    if int(mask.sum()) == 0:
        return {"masked_mae": 0.0, "masked_ncc": 0.0}
    synth = display_gray[mask].astype(np.float32) / 255.0
    ref = reference.clean_gray[mask].astype(np.float32) / 255.0
    mae = float(np.mean(np.abs(synth - ref)))
    synth_centered = synth - float(np.mean(synth))
    ref_centered = ref - float(np.mean(ref))
    denominator = float(np.linalg.norm(synth_centered) * np.linalg.norm(ref_centered))
    ncc = float(np.dot(synth_centered, ref_centered) / denominator) if denominator > 1e-8 else 0.0
    return {"masked_mae": mae, "masked_ncc": ncc}


def _tone_map_radar(values: np.ndarray) -> np.ndarray:
    values = np.maximum(values.astype(np.float32), 0.0)
    upper = float(np.quantile(values, 0.9995))
    lower = float(np.quantile(values, 0.85))
    if upper - lower < 1e-8:
        return normalize_to_uint8(values)
    mapped = np.clip((values - lower) / (upper - lower), 0.0, 1.0)
    mapped = np.power(mapped, 0.55)
    base = (mapped * 255.0).clip(0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(12, 12))
    return clahe.apply(base)


def synthesize_from_rgb(
    rgb_image: np.ndarray,
    terrain_probabilities: np.ndarray,
    reference: RadarReference | None = None,
    config: RadarPhysicsConfig | None = None,
) -> SynthesisReport:
    if config is None:
        config = RadarPhysicsConfig()

    origin_px = _estimate_radar_origin(terrain_probabilities)
    geometry = _geometry(rgb_image.shape[:2], origin_px, config)
    structure = _semantic_structure(terrain_probabilities)
    physics = _physics_maps(terrain_probabilities, structure, geometry, config)
    ray_effects = _ray_effects(
        geometry=geometry,
        height_map=physics["height_map"],
        attenuation_np_per_m=physics["attenuation_np_per_m"],
        occluder_strength=physics["occluder_strength"],
        origin_px=origin_px,
        config=config,
    )
    scatterers = _build_scatterer_map(terrain_probabilities, physics, structure, config)

    specific_sigma = physics["specific_sigma"]
    point_rcs = physics["point_rcs"]
    scatterer_rcs = scatterers["scatterer_rcs"]
    slant_range = np.maximum(geometry["slant_range_m"], 1.0)
    radar_constant = config.tx_power_w * (config.wavelength_m ** 2) / ((4.0 * np.pi) ** 3)

    raw_power = (radar_constant * scatterer_rcs / np.power(slant_range, 4.0)).astype(np.float32)
    raw_power *= physics["aspect_gain"] * ray_effects["transmittance"] * ray_effects["shadow"]
    stc_gain = np.power(np.maximum(slant_range / max(config.reference_range_m, 1.0), 0.15), config.stc_exponent)
    raw_power *= stc_gain.astype(np.float32)

    range_sigma_px = max(config.range_resolution_m / config.meters_per_pixel / 2.355, 0.8)
    effective_azimuth_resolution_m = np.maximum(geometry["azimuth_cell_m"], config.meters_per_pixel)
    blur_sigma_px = max(float(np.median(effective_azimuth_resolution_m)) / config.meters_per_pixel / 2.355, 0.7)
    smoothed_power = cv2.GaussianBlur(raw_power, (0, 0), sigmaX=range_sigma_px, sigmaY=blur_sigma_px)

    rng = np.random.default_rng(config.seed)
    speckle = rng.gamma(shape=config.effective_looks, scale=1.0 / config.effective_looks, size=smoothed_power.shape).astype(np.float32)
    clutter = 0.06 * cv2.GaussianBlur(normalize01(smoothed_power), (0, 0), sigmaX=1.2, sigmaY=1.2)
    distributed_floor = 0.10 * normalize01(specific_sigma) * geometry["grazing_gain"] * ray_effects["transmittance"]
    compound_glow = 0.18 * cv2.GaussianBlur(
        normalize01(raw_power) * structure["compound_mask"],
        (0, 0),
        sigmaX=3.8,
        sigmaY=3.8,
    )
    road_glow = 0.06 * cv2.GaussianBlur(
        normalize01(raw_power) * scatterers["road_linearity"],
        (0, 0),
        sigmaX=1.6,
        sigmaY=1.6,
    )
    radar_power = np.maximum(smoothed_power * speckle + clutter + distributed_floor + compound_glow + road_glow, 0.0)
    radar_log = np.log1p(8.0 * radar_power)
    radar_log = np.power(radar_log, 0.82)
    ground_gray = _tone_map_radar(radar_log)
    cell_map = _build_resolution_cell_map(
        geometry=geometry,
        physics=physics,
        structure=structure,
        ray_effects=ray_effects,
        origin_px=origin_px,
        config=config,
    )

    pure_output_shape, pure_center_px, pure_radius_px = _default_display_layout(config)
    pure_display_optical_rgb = _crop_display_optical(
        rgb_image,
        origin_px,
        pure_output_shape,
        pure_center_px,
        pure_radius_px,
    )

    if reference is not None:
        projected_shape, projected_center_px, projected_radius_px = _reference_display_layout(reference)
        display_gray, best_params, validation_history, best_keep_mask = _optimize_point_display(
            cell_map=cell_map,
            output_shape=projected_shape,
            center_px=projected_center_px,
            radius_px=projected_radius_px,
            config=config,
            reference=reference,
        )
        pure_display_gray, pure_keep_mask = _render_point_display(
            cell_map=cell_map,
            output_shape=pure_output_shape,
            center_px=pure_center_px,
            radius_px=pure_radius_px,
            config=config,
            params=best_params,
        )
        display_optical_rgb = _crop_display_optical(
            rgb_image,
            origin_px,
            projected_shape,
            projected_center_px,
            projected_radius_px,
        )
        metrics = _quality_metrics(display_gray, reference)
        validation = {
            "best_render_params": {key: float(value) for key, value in best_params.items()},
            "top_candidates": validation_history,
        }
    else:
        default_params = {
            "threshold_quantile": 0.50,
            "gamma": 0.82,
            "gain": 16.0,
            "blur_sigma": 0.45,
            "point_radius": 1.0,
            "display_gamma": 0.94,
            "lower_quantile": 0.46,
            "upper_quantile": 0.997,
        }
        pure_display_gray, pure_keep_mask = _render_point_display(
            cell_map=cell_map,
            output_shape=pure_output_shape,
            center_px=pure_center_px,
            radius_px=pure_radius_px,
            config=config,
            params=default_params,
        )
        display_gray = pure_display_gray
        display_optical_rgb = pure_display_optical_rgb
        metrics = {}
        validation = {
            "best_render_params": {key: float(value) for key, value in default_params.items()},
            "top_candidates": [],
        }
        best_keep_mask = pure_keep_mask

    terrain_rgb = render_terrain_map(terrain_probabilities)

    debug_maps = {
        "specific_sigma": normalize_to_uint8(specific_sigma),
        "point_rcs": normalize_to_uint8(point_rcs),
        "scatterer_rcs": normalize_to_uint8(scatterer_rcs),
        "bright_points": normalize_to_uint8(scatterers["bright_mask"]),
        "micro_points": normalize_to_uint8(scatterers["micro_mask"]),
        "transmittance": normalize_to_uint8(ray_effects["transmittance"]),
        "radar_shadow": normalize_to_uint8(1.0 - ray_effects["shadow"]),
        "aspect_gain": normalize_to_uint8(physics["aspect_gain"]),
        "cell_power": normalize_to_uint8(cell_map["cell_power"]),
        "cell_rcs": normalize_to_uint8(cell_map["cell_rcs"]),
        "cell_selected_points": normalize_to_uint8(best_keep_mask),
        "terrain_map": cv2.cvtColor(terrain_rgb, cv2.COLOR_RGB2BGR),
        "pure_display_gray": pure_display_gray,
        "projected_display_gray": display_gray,
    }
    if reference is not None:
        debug_maps["reference_clean"] = reference.clean_gray

    return SynthesisReport(
        ground_gray=ground_gray,
        display_gray=display_gray,
        display_optical_rgb=display_optical_rgb,
        pure_display_gray=pure_display_gray,
        pure_display_optical_rgb=pure_display_optical_rgb,
        terrain_rgb=terrain_rgb,
        origin_px=origin_px,
        debug_maps=debug_maps,
        metrics=metrics,
        validation=validation,
    )
