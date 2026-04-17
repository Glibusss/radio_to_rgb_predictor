from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgb2radio.chunking import export_paired_chunks, export_unpaired_radar_chunks
from rgb2radio.common import ensure_dir, read_rgb, resolve_existing_path, save_json
from rgb2radio.physics_model import RadarPhysicsConfig, synthesize_from_rgb
from rgb2radio.radar_reference import extract_radar_reference
from rgb2radio.terrain_resnet import (
    build_heuristic_score_maps,
    combine_terrain_probabilities,
    load_checkpoint_metadata,
    predict_terrain_probabilities,
    train_terrain_resnet,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare chunked data for future cGAN/diffusion experiments.")
    parser.add_argument("--optical", type=Path, default=None)
    parser.add_argument("--radar", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chunks"))
    parser.add_argument("--model-path", type=Path, default=Path("outputs/models/terrain_resnet.pt"))
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
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

    rgb_image = read_rgb(optical_path)
    radar_rgb = read_rgb(radar_path)
    reference = extract_radar_reference(radar_rgb)
    if not args.model_path.exists():
        train_terrain_resnet(rgb_image=rgb_image, output_path=args.model_path, seed=args.seed)

    model_probs = predict_terrain_probabilities(rgb_image, args.model_path)
    checkpoint_metadata = load_checkpoint_metadata(args.model_path)
    score_maps = build_heuristic_score_maps(rgb_image)
    terrain_probs = combine_terrain_probabilities(
        score_maps=score_maps,
        model_probabilities=model_probs,
        model_class_histogram=checkpoint_metadata.get("class_histogram"),
    )
    synthesis = synthesize_from_rgb(
        rgb_image=rgb_image,
        terrain_probabilities=terrain_probs,
        reference=reference,
        config=RadarPhysicsConfig(seed=args.seed),
    )

    output_dir = ensure_dir(args.output_dir)
    stats = {
        "paired_ground": export_paired_chunks(
            optical_image=rgb_image,
            radar_image=synthesis.ground_gray,
            output_dir=output_dir / "paired_ground",
            patch_size=args.patch_size,
            stride=args.stride,
        ),
        "paired_display_synth": export_paired_chunks(
            optical_image=synthesis.display_optical_rgb,
            radar_image=synthesis.display_gray,
            output_dir=output_dir / "paired_display_synth",
            patch_size=args.patch_size,
            stride=args.stride,
            min_content_score=10.0,
        ),
        "paired_display_real": export_paired_chunks(
            optical_image=synthesis.display_optical_rgb,
            radar_image=reference.clean_gray,
            output_dir=output_dir / "paired_display_real",
            patch_size=args.patch_size,
            stride=args.stride,
            min_content_score=10.0,
        ),
        "unpaired_radar": export_unpaired_radar_chunks(
            radar_image=reference.clean_gray,
            output_dir=output_dir / "unpaired_radar",
            patch_size=args.patch_size,
            stride=args.stride,
        ),
    }
    save_json(output_dir / "chunk_stats.json", stats)

    print(f"output_dir={output_dir}")
    print(f"stats={stats}")


if __name__ == "__main__":
    main()
