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

from rgb2radio.common import ensure_dir, resolve_existing_path, save_json, save_rgb
from rgb2radio.radar_territory_classifier import classify_radar_observation, load_radar_observation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the radar-only baseline territory classifier on a radar image or scalar radar map."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Radar input (.npy or image). Defaults to radar_data.PNG when available.",
    )
    parser.add_argument("--model-path", type=Path, default=Path("outputs/models/radar_territory_resnet.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/radar_classifier"))
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def main() -> None:
    args = build_parser().parse_args()
    input_path = args.input or resolve_existing_path(
        [
            PROJECT_ROOT / "radar_data.PNG",
            PROJECT_ROOT / "output" / "radar_equation_power_dbw.npy",
            PROJECT_ROOT / "output" / "radar_equation_map.png",
        ]
    )

    observation = load_radar_observation(input_path)
    result = classify_radar_observation(
        observation,
        checkpoint_path=args.model_path,
        stride=args.stride,
        batch_size=args.batch_size,
    )

    output_dir = ensure_dir(args.output_dir)
    save_rgb(output_dir / "radar_input.png", observation.visualization_rgb)
    save_rgb(output_dir / "territories.png", result.color_map)
    save_rgb(output_dir / "territories_overlay.png", result.overlay)
    _save_mask(output_dir / "active_mask.png", result.active_mask)
    np.save(output_dir / "probabilities.npy", result.probabilities.astype(np.float32))
    np.save(output_dir / "class_map.npy", result.class_map.astype(np.uint8))
    save_json(output_dir / "report.json", dict(result.report))

    distribution = result.report["class_distribution"]
    summary = ", ".join(
        f"{class_name}={100.0 * distribution[class_name]['pixel_share']:.1f}%"
        for class_name in result.class_names
    )
    print(f"territories={output_dir / 'territories.png'}")
    print(f"overlay={output_dir / 'territories_overlay.png'}")
    print(f"summary={summary}")


if __name__ == "__main__":
    main()
