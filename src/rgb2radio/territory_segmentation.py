from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import cv2
import numpy as np

from .common import gray_from_rgb, normalize01
from .terrain_resnet import (
    TERRAIN_CLASS_NAMES,
    build_heuristic_score_maps,
    combine_terrain_probabilities,
    load_checkpoint_metadata,
    normalize_probabilities,
    predict_terrain_probabilities,
    render_terrain_map,
    train_terrain_resnet,
)

TERRITORY_CLASS_NAMES: Sequence[str] = (
    "forest",
    "building",
    "asphalt_road",
    "dirt_road",
    "water",
    "forest_shadow",
    "building_shadow",
    "shrub",
    "vehicle",
)

TERRITORY_CLASS_COLORS = np.array(
    [
        [40, 115, 58],
        [198, 198, 206],
        [68, 71, 78],
        [165, 117, 71],
        [58, 109, 188],
        [25, 63, 40],
        [87, 89, 102],
        [118, 166, 73],
        [224, 70, 62],
    ],
    dtype=np.uint8,
)

_BASE_CLASS_INDEX = {name: index for index, name in enumerate(TERRAIN_CLASS_NAMES)}


@dataclass(frozen=True)
class TerritorySegmentationResult:
    class_names: Sequence[str]
    class_map: np.ndarray
    probabilities: np.ndarray
    color_map: np.ndarray
    overlay: np.ndarray
    debug_maps: Mapping[str, np.ndarray]
    report: Mapping[str, object]
    training_summary: Mapping[str, object]


def _base_probabilities(base_probs: np.ndarray, name: str) -> np.ndarray:
    return base_probs[_BASE_CLASS_INDEX[name]]


def _kernel(size: int) -> np.ndarray:
    size = max(3, int(size))
    if size % 2 == 0:
        size += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    work = mask.astype(np.uint8)
    if open_size > 0:
        work = cv2.morphologyEx(work, cv2.MORPH_OPEN, _kernel(open_size))
    if close_size > 0:
        work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, _kernel(close_size))
    return work > 0


def _proximity_map(mask: np.ndarray, sigma: float, dilate_size: int = 0) -> np.ndarray:
    work = mask.astype(np.uint8)
    if dilate_size > 0:
        work = cv2.dilate(work, _kernel(dilate_size), iterations=1)
    blurred = cv2.GaussianBlur(work.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    return normalize01(blurred)


def _compute_scene_features(rgb_image: np.ndarray, score_maps: Mapping[str, np.ndarray]) -> Dict[str, np.ndarray]:
    rgb_f32 = rgb_image.astype(np.float32) / 255.0
    red, green, blue = cv2.split(rgb_f32)
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV).astype(np.float32)
    saturation = hsv[:, :, 1] / 255.0
    value = hsv[:, :, 2] / 255.0
    gray_u8 = gray_from_rgb(rgb_image)
    gray = gray_u8.astype(np.float32) / 255.0

    exg = normalize01(2.0 * green - red - blue)
    small_bright = cv2.morphologyEx(gray_u8, cv2.MORPH_TOPHAT, _kernel(9)).astype(np.float32) / 255.0
    small_dark = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, _kernel(9)).astype(np.float32) / 255.0
    local_small = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    local_large = cv2.GaussianBlur(gray, (0, 0), sigmaX=7.0, sigmaY=7.0)
    local_contrast = normalize01(np.abs(local_small - local_large))
    manmade = normalize01(
        0.38 * score_maps["hard_edges"]
        + 0.22 * score_maps["corners"]
        + 0.18 * score_maps["metallic_tone"]
        + 0.12 * score_maps["neutral"]
        + 0.10 * (1.0 - exg)
    )
    shadow_candidate = normalize01(
        score_maps["shadow"]
        * (0.55 + 0.45 * score_maps["neutral"])
        * (1.0 - 0.55 * score_maps["water_group"])
    )
    return {
        "gray": gray.astype(np.float32),
        "value": value.astype(np.float32),
        "saturation": saturation.astype(np.float32),
        "exg": exg.astype(np.float32),
        "small_bright": normalize01(small_bright).astype(np.float32),
        "small_dark": normalize01(small_dark).astype(np.float32),
        "local_contrast": local_contrast.astype(np.float32),
        "manmade": manmade.astype(np.float32),
        "shadow_candidate": shadow_candidate.astype(np.float32),
    }


def _extract_vehicle_mask(
    vehicle_response: np.ndarray,
    near_manmade: np.ndarray,
    road_mask: np.ndarray,
    building_mask: np.ndarray,
) -> np.ndarray:
    candidate = _clean_mask(vehicle_response > 0.26, open_size=0, close_size=3)
    if not np.any(candidate):
        return np.zeros_like(candidate, dtype=bool)

    neighborhood = cv2.dilate((road_mask | building_mask).astype(np.uint8), _kernel(11), iterations=1) > 0
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), connectivity=8)
    kept = np.zeros_like(candidate, dtype=bool)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        width = max(1, int(stats[label, cv2.CC_STAT_WIDTH]))
        height = max(1, int(stats[label, cv2.CC_STAT_HEIGHT]))
        if area < 4 or area > 140:
            continue
        if max(width, height) > 28:
            continue
        aspect_ratio = float(max(width, height) / max(1, min(width, height)))
        if aspect_ratio > 5.0:
            continue
        component = labels == label
        if float(np.mean(vehicle_response[component])) < 0.30:
            continue
        if not np.any(near_manmade[component] > 0.20):
            continue
        if not np.any(neighborhood[component]):
            continue
        kept |= component
    return kept


def _cleanup_probabilities(probabilities: np.ndarray) -> np.ndarray:
    min_area = {
        "forest": 140,
        "building": 20,
        "asphalt_road": 80,
        "dirt_road": 60,
        "water": 120,
        "forest_shadow": 24,
        "building_shadow": 24,
        "shrub": 36,
        "vehicle": 4,
    }
    cleaned = probabilities.astype(np.float32).copy()
    labels = np.argmax(cleaned, axis=0).astype(np.int32)
    for class_index, class_name in enumerate(TERRITORY_CLASS_NAMES):
        mask = (labels == class_index).astype(np.uint8)
        if int(mask.sum()) == 0:
            continue
        num_labels, component_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for component_id in range(1, num_labels):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area >= min_area[class_name]:
                continue
            component_mask = component_labels == component_id
            cleaned[class_index, component_mask] *= 0.02
    return normalize_probabilities(np.maximum(cleaned, 1e-7))


def _class_distribution(labels: np.ndarray) -> Dict[str, Dict[str, float | int]]:
    total = max(1, int(labels.size))
    report: Dict[str, Dict[str, float | int]] = {}
    for class_index, class_name in enumerate(TERRITORY_CLASS_NAMES):
        count = int(np.count_nonzero(labels == class_index))
        report[class_name] = {
            "pixel_count": count,
            "pixel_share": float(count / total),
        }
    return report


def render_territory_color_map(probabilities: np.ndarray) -> np.ndarray:
    labels = np.argmax(probabilities, axis=0)
    confidence = np.max(probabilities, axis=0)[:, :, None]
    color_map = TERRITORY_CLASS_COLORS[labels].astype(np.float32)
    shaded = color_map * (0.58 + 0.42 * confidence)
    return shaded.clip(0, 255).astype(np.uint8)


def render_territory_overlay(rgb_image: np.ndarray, probabilities: np.ndarray, alpha: float = 0.46) -> np.ndarray:
    color_map = render_territory_color_map(probabilities)
    blended = rgb_image.astype(np.float32) * (1.0 - alpha) + color_map.astype(np.float32) * alpha
    return blended.clip(0, 255).astype(np.uint8)


def render_territory_map(probabilities: np.ndarray) -> np.ndarray:
    return render_territory_color_map(probabilities)


def segment_territories(
    rgb_image: np.ndarray,
    model_path: str | Path,
    retrain: bool = False,
    train_epochs: int = 8,
    patch_size: int = 64,
    stride: int = 24,
    seed: int = 42,
) -> TerritorySegmentationResult:
    model_path = Path(model_path)
    training_summary: Dict[str, object] = {
        "model_path": str(model_path),
        "trained": False,
        "reused_checkpoint": False,
        "fallback_mode": "heuristic_only",
    }

    if retrain or not model_path.exists():
        try:
            training = train_terrain_resnet(
                rgb_image=rgb_image,
                output_path=model_path,
                patch_size=patch_size,
                epochs=train_epochs,
                seed=seed,
            )
            training_summary = {
                "model_path": str(training.model_path),
                "trained": True,
                "reused_checkpoint": False,
                "fallback_mode": "terrain_resnet",
                "patch_size": training.patch_size,
                "train_samples": training.train_samples,
                "val_samples": training.val_samples,
                "best_val_accuracy": training.best_val_accuracy,
                "class_histogram": training.class_histogram,
            }
        except RuntimeError as exc:
            training_summary["error"] = str(exc)

    score_maps = build_heuristic_score_maps(rgb_image)
    model_probabilities = None
    checkpoint_metadata: Mapping[str, object] | None = None
    if model_path.exists():
        model_probabilities = predict_terrain_probabilities(rgb_image, model_path, stride=stride)
        checkpoint_metadata = load_checkpoint_metadata(model_path)
        training_summary["fallback_mode"] = "terrain_resnet"
        if not bool(training_summary.get("trained")):
            training_summary["reused_checkpoint"] = True

    base_probabilities = combine_terrain_probabilities(
        score_maps=score_maps,
        model_probabilities=model_probabilities,
        model_class_histogram=checkpoint_metadata.get("class_histogram") if checkpoint_metadata else None,
    )

    base_forest = _base_probabilities(base_probabilities, "forest")
    base_building = np.maximum.reduce(
        [
            _base_probabilities(base_probabilities, "concrete_building"),
            _base_probabilities(base_probabilities, "metal_building"),
            _base_probabilities(base_probabilities, "wood_building"),
        ]
    )
    base_asphalt = _base_probabilities(base_probabilities, "asphalt_road")
    base_dirt = _base_probabilities(base_probabilities, "dirt_road")
    base_water = _base_probabilities(base_probabilities, "water") + 0.85 * _base_probabilities(base_probabilities, "rippling_water")
    base_water = normalize01(base_water)

    scene = _compute_scene_features(rgb_image, score_maps)

    forest_core = _clean_mask((base_forest > 0.42) | (score_maps["forest_group"] > 0.34), open_size=5, close_size=7)
    building_core = _clean_mask((base_building > 0.30) | (score_maps["building_group"] > 0.22), open_size=3, close_size=5)
    road_core = _clean_mask((np.maximum(base_asphalt, base_dirt) > 0.24) | (score_maps["road_group"] > 0.18), open_size=3, close_size=7)
    water_core = _clean_mask((base_water > 0.34) | (score_maps["water_group"] > 0.26), open_size=5, close_size=9)

    forest_proximity = _proximity_map(forest_core, sigma=6.5, dilate_size=9)
    building_proximity = _proximity_map(building_core, sigma=7.0, dilate_size=11)
    road_proximity = _proximity_map(road_core, sigma=6.0, dilate_size=9)
    near_manmade = np.maximum(building_proximity, road_proximity)

    shrub = normalize01(
        score_maps["vegetation"]
        * (0.58 + 0.42 * score_maps["bright"])
        * (0.65 + 0.35 * score_maps["smooth"])
        * (1.0 - 0.82 * base_forest)
        * (1.0 - 0.55 * base_water)
        * (1.0 - 0.45 * near_manmade)
        * (1.0 - 0.35 * score_maps["canopy_shadow"])
    )

    shadow_candidate = scene["shadow_candidate"]
    forest_shadow = normalize01(
        shadow_candidate
        * (0.30 + 0.70 * forest_proximity)
        * (1.0 - 0.38 * building_proximity)
        * (1.0 - 0.22 * water_core.astype(np.float32))
        * (1.0 - 0.28 * base_forest)
    )
    building_shadow = normalize01(
        shadow_candidate
        * (0.32 + 0.68 * building_proximity)
        * (0.25 + 0.75 * near_manmade)
        * (0.35 + 0.65 * score_maps["hard_edges"])
        * (1.0 - 0.25 * base_building)
        * (1.0 - 0.20 * base_water)
    )

    vehicle_response = normalize01(
        np.maximum(scene["small_bright"], 0.9 * scene["small_dark"])
        * (0.28 + 0.72 * scene["local_contrast"])
        * (0.38 + 0.62 * score_maps["corners"])
        * (0.42 + 0.58 * score_maps["hard_edges"])
        * (0.30 + 0.70 * near_manmade)
        * (1.0 - 0.70 * base_forest)
        * (1.0 - 0.55 * base_water)
    )
    vehicle_mask = _extract_vehicle_mask(
        vehicle_response=vehicle_response,
        near_manmade=near_manmade,
        road_mask=road_core,
        building_mask=building_core,
    )
    vehicle = cv2.GaussianBlur(vehicle_mask.astype(np.float32), (0, 0), sigmaX=0.8, sigmaY=0.8)
    vehicle = np.maximum(vehicle, 0.65 * vehicle_response)

    forest = normalize01(
        (0.82 * base_forest + 0.18 * score_maps["forest_group"])
        * (1.0 - 0.42 * building_proximity)
        * (1.0 - 0.36 * road_proximity)
        * (1.0 - 0.55 * np.maximum(forest_shadow, building_shadow))
    )
    building = normalize01(
        (0.84 * base_building + 0.16 * score_maps["building_group"])
        * (1.0 - 0.48 * base_water)
        * (1.0 - 0.28 * np.maximum(forest_shadow, building_shadow))
        * (1.0 - 0.35 * vehicle)
    )
    asphalt_road = normalize01(
        (0.80 * base_asphalt + 0.20 * score_maps["road_group"])
        * (1.0 - 0.38 * base_water)
        * (1.0 - 0.25 * forest)
        * (1.0 - 0.28 * building_shadow)
    )
    dirt_road = normalize01(
        (0.82 * base_dirt + 0.18 * score_maps["road_group"])
        * (1.0 - 0.34 * base_water)
        * (1.0 - 0.32 * forest)
        * (1.0 - 0.18 * building_shadow)
    )
    water = normalize01(
        (0.86 * base_water + 0.14 * score_maps["water_group"])
        * (1.0 - 0.42 * np.maximum(forest_shadow, building_shadow))
        * (1.0 - 0.20 * vehicle)
    )

    scores = np.stack(
        [
            1.18 * forest,
            1.20 * building,
            1.12 * asphalt_road,
            1.05 * dirt_road,
            1.10 * water,
            0.98 * forest_shadow,
            0.98 * building_shadow,
            1.02 * shrub,
            1.70 * vehicle,
        ],
        axis=0,
    ).astype(np.float32)
    probabilities = normalize_probabilities(np.maximum(scores, 1e-6))

    if np.any(forest_core):
        probabilities[0] *= 0.55 + 0.45 * forest_proximity
    if np.any(building_core):
        probabilities[1] *= 0.55 + 0.45 * building_proximity
    if np.any(road_core):
        probabilities[2] *= 0.62 + 0.38 * road_proximity
        probabilities[3] *= 0.62 + 0.38 * road_proximity
    if np.any(water_core):
        water_support = _proximity_map(water_core, sigma=5.0, dilate_size=7)
        probabilities[4] *= 0.58 + 0.42 * water_support
    probabilities[5] *= 0.18 + 0.82 * (forest_shadow > 0.18).astype(np.float32)
    probabilities[6] *= 0.18 + 0.82 * (building_shadow > 0.18).astype(np.float32)
    probabilities[7] *= 0.22 + 0.78 * (shrub > 0.16).astype(np.float32)
    probabilities[8] *= 0.12 + 0.88 * (vehicle > 0.20).astype(np.float32)
    probabilities = normalize_probabilities(np.maximum(probabilities, 1e-7))

    smoothed = np.stack(
        [
            cv2.GaussianBlur(probabilities[index], (0, 0), sigmaX=1.0, sigmaY=1.0)
            for index in range(probabilities.shape[0])
        ],
        axis=0,
    ).astype(np.float32)
    probabilities = normalize_probabilities(0.76 * probabilities + 0.24 * smoothed)
    probabilities = _cleanup_probabilities(probabilities)
    probabilities[8] = np.maximum(probabilities[8], vehicle.astype(np.float32))
    probabilities = normalize_probabilities(probabilities)

    class_map = np.argmax(probabilities, axis=0).astype(np.uint8)
    class_map[vehicle_mask] = np.uint8(TERRITORY_CLASS_NAMES.index("vehicle"))
    if np.any(building_shadow > 0.45):
        building_shadow_mask = (building_shadow > 0.45) & ~vehicle_mask & ~building_core & ~water_core
        class_map[building_shadow_mask] = np.uint8(TERRITORY_CLASS_NAMES.index("building_shadow"))
    if np.any(forest_shadow > 0.46):
        forest_shadow_mask = (forest_shadow > 0.46) & ~vehicle_mask & ~water_core & ~building_core
        class_map[forest_shadow_mask] = np.uint8(TERRITORY_CLASS_NAMES.index("forest_shadow"))

    one_hot = np.zeros_like(probabilities)
    for class_index in range(one_hot.shape[0]):
        one_hot[class_index] = (class_map == class_index).astype(np.float32)
    probabilities = normalize_probabilities(0.58 * probabilities + 0.42 * one_hot)

    color_map = render_territory_map(probabilities)
    overlay = render_territory_overlay(rgb_image, probabilities)
    base_map_rgb = render_terrain_map(base_probabilities)

    debug_maps: Dict[str, np.ndarray] = {
        "base_terrain_map_rgb": base_map_rgb,
        "forest_core": forest_core.astype(np.float32),
        "building_core": building_core.astype(np.float32),
        "road_core": road_core.astype(np.float32),
        "water_core": water_core.astype(np.float32),
        "forest_proximity": forest_proximity.astype(np.float32),
        "building_proximity": building_proximity.astype(np.float32),
        "road_proximity": road_proximity.astype(np.float32),
        "shadow_candidate": shadow_candidate.astype(np.float32),
        "forest_shadow_candidate": forest_shadow.astype(np.float32),
        "building_shadow_candidate": building_shadow.astype(np.float32),
        "shrub_candidate": shrub.astype(np.float32),
        "vehicle_response": vehicle_response.astype(np.float32),
        "vehicle_mask": vehicle_mask.astype(np.float32),
        "final_class_map": class_map.astype(np.uint8),
        "final_overlay_rgb": overlay,
        "territories_rgb": color_map,
    }

    for class_index, class_name in enumerate(TERRITORY_CLASS_NAMES):
        debug_maps[f"prob_{class_name}"] = probabilities[class_index].astype(np.float32)

    report: Dict[str, object] = {
        "class_distribution": _class_distribution(class_map),
        "training_summary": dict(training_summary),
        "model_path": str(model_path),
    }

    return TerritorySegmentationResult(
        class_names=TERRITORY_CLASS_NAMES,
        class_map=class_map,
        probabilities=probabilities,
        color_map=color_map,
        overlay=overlay,
        debug_maps=debug_maps,
        report=report,
        training_summary=training_summary,
    )
