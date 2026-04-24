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

from rgb2radio.common import ensure_dir, normalize_to_uint8, read_rgb, resolve_existing_path, save_json, save_rgb
from rgb2radio.terrain_resnet import export_training_debug_images
from rgb2radio.territory_segmentation import TERRITORY_CLASS_NAMES, segment_territories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Classify territories on real_data.png into 9 target classes.")
    parser.add_argument("--input", type=Path, default=None, help="Path to optical RGB image. Defaults to real_data.png.")
    parser.add_argument("--output-dir", type=Path, default=Path("output"), help="Directory for the final territories.png.")
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=Path("output_debug"),
        help="Directory for intermediate debug artifacts when --debug is enabled.",
    )
    parser.add_argument("--model-path", type=Path, default=Path("outputs/models/terrain_resnet.pt"))
    parser.add_argument("--retrain", action="store_true", help="Force terrain ResNet retraining before inference.")
    parser.add_argument("--train-epochs", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true", help="Save all intermediate images into output_debug.")
    return parser


def _save_debug_map(path: Path, image: np.ndarray) -> None:
    if image.ndim == 3 and image.shape[2] == 3:
        save_rgb(path, image.astype(np.uint8))
        return

    if image.dtype == np.bool_:
        cv2.imwrite(str(path), image.astype(np.uint8) * 255)
        return

    if np.issubdtype(image.dtype, np.integer):
        max_value = int(np.max(image)) if image.size else 0
        if 0 <= max_value <= 255:
            cv2.imwrite(str(path), image.astype(np.uint8))
            return

    cv2.imwrite(str(path), normalize_to_uint8(image.astype(np.float32)))


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input or resolve_existing_path(
        [
            PROJECT_ROOT / "real_data.png",
        ]
    )

    image_rgb = read_rgb(input_path)
    result = segment_territories(
        rgb_image=image_rgb,
        model_path=args.model_path,
        retrain=args.retrain,
        train_epochs=args.train_epochs,
        patch_size=args.patch_size,
        stride=args.stride,
        seed=args.seed,
    )

    output_dir = ensure_dir(args.output_dir)
    final_path = output_dir / "territories.png"
    overlay_path = output_dir / "territories_overlay.png"
    save_rgb(final_path, result.color_map)
    save_rgb(overlay_path, result.overlay)

    if args.debug:
        debug_dir = ensure_dir(args.debug_dir)
        save_rgb(debug_dir / "input.png", image_rgb)
        save_rgb(debug_dir / "territories.png", result.color_map)
        save_rgb(debug_dir / "territories_overlay.png", result.overlay)
        for name, image in result.debug_maps.items():
            _save_debug_map(debug_dir / f"{name}.png", image)
        if args.model_path.exists():
            export_training_debug_images(args.model_path, debug_dir / "training")
        save_json(
            debug_dir / "report.json",
            {
                "input_path": str(input_path),
                "output_path": str(final_path),
                "overlay_output_path": str(overlay_path),
                "class_names": list(TERRITORY_CLASS_NAMES),
                "report": result.report,
            },
        )

    distribution = result.report["class_distribution"]
    summary = ", ".join(
        f"{class_name}={100.0 * distribution[class_name]['pixel_share']:.1f}%"
        for class_name in TERRITORY_CLASS_NAMES
    )
    print(f"territories={final_path}")
    print(summary)
    print(f"mode={result.training_summary.get('fallback_mode', 'heuristic_only')}")
    if args.debug:
        print(f"debug_dir={args.debug_dir}")


if __name__ == "__main__":
    main()
