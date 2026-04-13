"""CLI-скрипт для запуска базового синтеза радарного изображения."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_synthesis.physics import (
    RadarConfig,
    read_rgb_image,
    save_debug_images,
    save_gray_image,
    save_rgb_image,
    synthesize_radar_image,
)
from radar_synthesis.radar_display import extract_clean_radar_reference


def build_parser() -> argparse.ArgumentParser:
    """Создает парсер аргументов командной строки для синтеза примера."""

    parser = argparse.ArgumentParser(description="Physics-aware optical-to-radar baseline.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/optical/real_data.png"),
        help="Path to the source optical image.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/synthesis/real_data_synth_display.png"),
        help="Path to the synthesized display-style radar image.",
    )
    parser.add_argument(
        "--ground-output",
        type=Path,
        default=Path("data/processed/synthesis/real_data_synth_ground.png"),
        help="Path to the synthesized ground-projected radar image.",
    )
    parser.add_argument(
        "--optical-display-output",
        type=Path,
        default=Path("data/processed/synthesis/real_data_optical_display.png"),
        help="Path to the centered circular optical crop used for display-mode comparison.",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("data/processed/debug/real_data"),
        help="Directory for intermediate debug maps.",
    )
    parser.add_argument(
        "--reference-radar",
        type=Path,
        default=Path("data/raw/radar/radar_display.png"),
        help="Optional radar screen capture used to calibrate display geometry and intensity style.",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=1.5,
        help="Ground sampling distance for the optical image.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the speckle component.",
    )
    return parser


def main() -> None:
    """Синтезирует радарные изображения и сохраняет отладочные артефакты."""

    args = build_parser().parse_args()
    config = RadarConfig(meters_per_pixel=args.meters_per_pixel, seed=args.seed)

    rgb_image = read_rgb_image(args.input)
    reference_clean = None
    reference_center = None
    reference_radius = None
    if args.reference_radar.exists():
        reference_rgb = read_rgb_image(args.reference_radar)
        reference_result = extract_clean_radar_reference(reference_rgb)
        reference_clean = reference_result.clean_image
        reference_center = reference_result.center_px
        reference_radius = reference_result.radius_px

    result = synthesize_radar_image(
        rgb_image,
        config=config,
        reference_clean_image=reference_clean,
        reference_raw_image=reference_result.raw_response if args.reference_radar.exists() else None,
        reference_style_background=reference_result.style_background if args.reference_radar.exists() else None,
        reference_style_artifacts=reference_result.style_artifacts if args.reference_radar.exists() else None,
        reference_center_px=reference_center,
        reference_radius_px=reference_radius,
    )

    save_gray_image(args.output, result.display_radar_image)
    save_gray_image(args.ground_output, result.radar_image)
    save_rgb_image(args.optical_display_output, result.display_optical_image)
    save_debug_images(args.debug_dir, result.debug_maps)

    print(f"saved={args.output}")
    print(f"ground_saved={args.ground_output}")
    print(f"optical_display_saved={args.optical_display_output}")
    print(f"origin_px=({result.radar_origin_px[0]:.1f}, {result.radar_origin_px[1]:.1f})")
    print(f"debug_dir={args.debug_dir}")


if __name__ == "__main__":
    main()
