from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from radar_synthesis.radar_display import extract_clean_radar_reference, read_rgb_image, save_gray_image


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a UI-cleaned radar reference image.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/raw/radar/radar_display.png"),
        help="Path to the radar screen capture.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/reference/radar_display_clean.png"),
        help="Path to the cleaned radar image.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rgb_image = read_rgb_image(args.input)
    result = extract_clean_radar_reference(rgb_image)
    save_gray_image(args.output, result.clean_image)

    print(f"saved={args.output}")
    print(f"center_px=({result.center_px[0]:.1f}, {result.center_px[1]:.1f})")
    print(f"radius_px={result.radius_px:.1f}")


if __name__ == "__main__":
    main()
