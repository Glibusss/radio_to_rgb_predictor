from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgb2radio.common import read_rgb, resolve_existing_path, save_json
from rgb2radio.terrain_resnet import export_training_debug_images, train_terrain_resnet


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the terrain ResNet used by the territory segmentation pipeline.")
    parser.add_argument(
        "--optical",
        type=Path,
        default=None,
        help="Path to optical RGB image. Defaults to real_data.png.",
    )
    parser.add_argument("--output-model", type=Path, default=Path("outputs/models/terrain_resnet.pt"))
    parser.add_argument("--debug-dir", type=Path, default=Path("output_debug/training"))
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    optical_path = args.optical or resolve_existing_path(
        [
            PROJECT_ROOT / "real_data.png",
        ]
    )

    rgb_image = read_rgb(optical_path)
    result = train_terrain_resnet(
        rgb_image=rgb_image,
        output_path=args.output_model,
        patch_size=args.patch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    export_training_debug_images(result.model_path, args.debug_dir)
    save_json(
        args.debug_dir / "training_summary.json",
        {
            "model_path": str(result.model_path),
            "patch_size": result.patch_size,
            "train_samples": result.train_samples,
            "val_samples": result.val_samples,
            "best_val_accuracy": result.best_val_accuracy,
            "class_histogram": result.class_histogram,
            "history": result.history,
        },
    )

    print(f"model={result.model_path}")
    print(f"best_val_accuracy={result.best_val_accuracy:.4f}")
    print(f"train_samples={result.train_samples}")
    print(f"val_samples={result.val_samples}")
    print(f"debug_dir={args.debug_dir}")


if __name__ == "__main__":
    main()
