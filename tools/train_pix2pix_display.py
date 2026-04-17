from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from rgb2radio.pix2pix_display import Pix2PixConfig, train_pix2pix_display


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a lightweight pix2pix baseline on display-aligned chunks.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pix2pix"))
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train_pix2pix_display(
        output_dir=args.output_dir,
        config=Pix2PixConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        ),
    )
    print(f"model={result.model_path}")
    print(f"output={result.output_path}")
    print(f"report={result.report_path}")
    print(f"best_epoch={result.best_epoch}")
    print(f"best_masked_ncc={result.best_masked_ncc:.4f}")
    print(f"best_masked_mae={result.best_masked_mae:.4f}")


if __name__ == "__main__":
    main()
