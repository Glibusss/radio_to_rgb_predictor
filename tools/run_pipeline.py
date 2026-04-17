from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgb2radio.common import ensure_dir, read_rgb, resolve_existing_path, save_gray, save_json, save_rgb
from rgb2radio.physics_model import RadarPhysicsConfig, synthesize_from_rgb
from rgb2radio.radar_reference import extract_radar_reference
from rgb2radio.rcs_library import RCS_PROFILES
from rgb2radio.terrain_resnet import (
    build_heuristic_score_maps,
    combine_terrain_probabilities,
    export_training_debug_images,
    load_checkpoint_metadata,
    predict_terrain_probabilities,
    train_terrain_resnet,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="End-to-end RGB -> grayscale radar baseline.")
    parser.add_argument("--optical", type=Path, default=None)
    parser.add_argument("--radar", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/run"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/models/terrain_resnet.pt"))
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    optical_path = args.optical or resolve_existing_path(
        [
            PROJECT_ROOT / "real_data.png",
            PROJECT_ROOT / "data/raw/optical/real_data.png",
        ]
    )
    radar_path = args.radar or resolve_existing_path(
        [
            PROJECT_ROOT / "radar_display.png",
            PROJECT_ROOT / "data/raw/radar/radar_display.png",
        ]
    )

    output_dir = ensure_dir(args.output_dir)
    debug_dir = ensure_dir(output_dir / "debug")
    rgb_image = read_rgb(optical_path)
    radar_rgb = read_rgb(radar_path)
    reference = extract_radar_reference(radar_rgb)

    if args.retrain or not args.model_path.exists():
        training = train_terrain_resnet(
            rgb_image=rgb_image,
            output_path=args.model_path,
            patch_size=args.patch_size,
            epochs=args.train_epochs,
            seed=args.seed,
        )
        training_summary = {
            "model_path": str(training.model_path),
            "best_val_accuracy": training.best_val_accuracy,
            "train_samples": training.train_samples,
            "val_samples": training.val_samples,
            "class_histogram": training.class_histogram,
        }
    else:
        training_summary = {"model_path": str(args.model_path), "reused_checkpoint": True}

    export_training_debug_images(args.model_path, debug_dir / "terrain_training")
    model_probs = predict_terrain_probabilities(rgb_image, args.model_path, stride=args.stride)
    checkpoint_metadata = load_checkpoint_metadata(args.model_path)
    score_maps = build_heuristic_score_maps(rgb_image)
    terrain_probs = combine_terrain_probabilities(
        score_maps=score_maps,
        model_probabilities=model_probs,
        model_class_histogram=checkpoint_metadata.get("class_histogram"),
    )

    report = synthesize_from_rgb(
        rgb_image=rgb_image,
        terrain_probabilities=terrain_probs,
        reference=reference,
        config=RadarPhysicsConfig(seed=args.seed),
    )

    save_gray(output_dir / "synth_ground_gray.png", report.ground_gray)
    save_gray(output_dir / "pure_physics_display_gray.png", report.pure_display_gray)
    save_gray(output_dir / "projected_display_gray.png", report.display_gray)
    save_gray(output_dir / "synth_display_gray.png", report.display_gray)
    save_gray(output_dir / "reference_clean_gray.png", reference.clean_gray)
    save_rgb(output_dir / "pure_display_optical_rgb.png", report.pure_display_optical_rgb)
    save_rgb(output_dir / "projected_display_optical_rgb.png", report.display_optical_rgb)
    save_rgb(output_dir / "display_optical_rgb.png", report.display_optical_rgb)
    save_rgb(output_dir / "terrain_map_rgb.png", report.terrain_rgb)

    for name, image in report.debug_maps.items():
        target = debug_dir / f"{name}.png"
        if image.ndim == 2:
            save_gray(target, image)
        else:
            cv2.imwrite(str(target), image)

    save_json(
        output_dir / "report.json",
        {
            "optical_path": str(optical_path),
            "radar_path": str(radar_path),
            "model_path": str(args.model_path),
            "metrics": report.metrics,
            "origin_px": [float(report.origin_px[0]), float(report.origin_px[1])],
            "config": {
                "frequency_ghz": 9.4,
                "tx_power_w": 50.0,
                "antenna_height_m": 3.0,
                "range_resolution_m": 1.5,
                "azimuth_resolution_deg": 1.0,
                "display_angle_step_deg": 1.0,
                "meters_per_pixel": 0.375,
            },
            "training": training_summary,
            "validation": report.validation,
            "note": "pure_physics_display_gray.png is generated from polar range-azimuth cell aggregation without histogram matching or radar-style transfer; projected_display_gray.png is only a geometry-aligned projection for comparison against the reference.",
            "azimuth_note": "Display formation uses 1 deg azimuth cells and 1.5 m range cells, so plan-view building contours are intentionally unresolved.",
        },
    )
    save_json(
        output_dir / "rcs_map.json",
        {
            class_name: {
                "sigma0_db": profile.sigma0_db,
                "point_rcs_m2": profile.point_rcs_m2,
                "scatterer_density_m2": profile.scatterer_density_m2,
                "mean_height_m": profile.mean_height_m,
                "attenuation_np_per_m": profile.attenuation_np_per_m,
                "specularity": profile.specularity,
                "roughness": profile.roughness,
            }
            for class_name, profile in RCS_PROFILES.items()
        },
    )

    print(f"output_dir={output_dir}")
    print(f"origin_px=({report.origin_px[0]:.1f}, {report.origin_px[1]:.1f})")
    print(f"masked_mae={report.metrics['masked_mae']:.4f}")
    print(f"masked_ncc={report.metrics['masked_ncc']:.4f}")


if __name__ == "__main__":
    main()
