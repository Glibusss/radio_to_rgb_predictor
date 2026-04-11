from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_synthesis.chunks import export_paired_chunks, export_unpaired_radar_bank
from radar_synthesis.physics import RadarConfig, read_rgb_image, synthesize_radar_image
from radar_synthesis.radar_display import extract_clean_radar_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare chunked datasets for optical-to-radar training.")
    parser.add_argument("--optical", type=Path, default=Path("data/raw/optical/real_data.png"))
    parser.add_argument("--radar-display", type=Path, default=Path("data/raw/radar/radar_display.png"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/chunks"))
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--meters-per-pixel", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--train-augmentations",
        nargs="+",
        default=["orig", "rot90", "rot180", "flip_lr"],
        help="Augmentations for train chunks: orig rot90 rot180 flip_lr",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    optical_image = read_rgb_image(args.optical)
    synth_result = synthesize_radar_image(
        optical_image,
        config=RadarConfig(meters_per_pixel=args.meters_per_pixel, seed=args.seed),
    )

    paired_stats = export_paired_chunks(
        optical_image=optical_image,
        radar_image=synth_result.radar_image,
        output_dir=args.output_dir / "paired",
        patch_size=args.patch_size,
        stride=args.stride,
        val_ratio=args.val_ratio,
        seed=args.seed,
        train_augmentations=args.train_augmentations,
    )

    radar_display = read_rgb_image(args.radar_display)
    radar_reference = extract_clean_radar_reference(radar_display).clean_image
    radar_bank_stats = export_unpaired_radar_bank(
        radar_image=radar_reference,
        output_dir=args.output_dir / "unpaired_radar",
        patch_size=args.patch_size,
        stride=args.stride,
    )

    print(f"paired={paired_stats}")
    print(f"unpaired_radar={radar_bank_stats}")
    print(f"output_dir={args.output_dir}")


if __name__ == "__main__":
    main()
