from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgb2radio.common import read_rgb, resolve_existing_path, save_gray
from rgb2radio.radar_reference import extract_radar_reference


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract a cleaned grayscale radar reference from the screen capture.")
    parser.add_argument("--radar", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("outputs/reference/reference_clean_gray.png"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    radar_path = args.radar or resolve_existing_path(
        [
            PROJECT_ROOT / "radar_display.png",
            PROJECT_ROOT / "data/raw/radar/radar_display.png",
        ]
    )
    reference = extract_radar_reference(read_rgb(radar_path))
    save_gray(args.output, reference.clean_gray)
    print(f"output={args.output}")
    print(f"center_px=({reference.center_px[0]:.1f}, {reference.center_px[1]:.1f})")
    print(f"radius_px={reference.radius_px:.1f}")


if __name__ == "__main__":
    main()
