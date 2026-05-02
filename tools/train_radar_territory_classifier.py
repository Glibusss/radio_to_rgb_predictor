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

from rgb2radio.common import ensure_dir, read_rgb, resolve_existing_path, save_json, save_rgb
from rgb2radio.pipeline_switches import load_pipeline_switches
from rgb2radio.radar_territory_classifier import (
    RADAR_FEATURE_NAMES,
    align_label_map_to_radar,
    build_radar_feature_cube,
    classify_radar_observation,
    load_radar_observation,
    train_radar_territory_classifier,
)
from rgb2radio.territory_segmentation import TERRITORY_CLASS_COLORS, segment_territories


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a baseline territory classifier that predicts classes directly from a radar image."
    )
    parser.add_argument(
        "--optical",
        type=Path,
        default=None,
        help="Aligned optical RGB image used to generate weak labels. Defaults to real_data.png.",
    )
    parser.add_argument(
        "--radar-input",
        type=Path,
        default=None,
        help="Aligned radar input (.npy or image). Defaults to radar_data.PNG when available.",
    )
    parser.add_argument("--output-model", type=Path, default=Path("outputs/models/radar_territory_resnet.pt"))
    parser.add_argument("--terrain-model-path", type=Path, default=Path("outputs/models/terrain_resnet.pt"))
    parser.add_argument("--pipeline-config", type=Path, default=Path("config/pipeline_switches.json"))
    parser.add_argument("--debug-dir", type=Path, default=Path("output_debug/radar_training"))
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--alignment-mode",
        choices=("auto", "stretch", "contain", "cover"),
        default="auto",
        help="How to transfer optical weak labels into the radar frame when shapes differ.",
    )
    return parser


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def _build_training_sample_mask(active_mask: np.ndarray, support_mask: np.ndarray) -> np.ndarray:
    dilated = cv2.dilate(
        support_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)),
        iterations=1,
    ) > 0
    return dilated & active_mask.astype(bool)


def _render_label_map(labels: np.ndarray) -> np.ndarray:
    image = np.zeros(labels.shape + (3,), dtype=np.uint8)
    valid = (labels >= 0) & (labels < len(TERRITORY_CLASS_COLORS))
    image[valid] = TERRITORY_CLASS_COLORS[labels[valid]]
    return image


def main() -> None:
    args = build_parser().parse_args()
    optical_path = args.optical or resolve_existing_path(
        [
            PROJECT_ROOT / "real_data.png",
            PROJECT_ROOT / "real_data2.PNG",
        ]
    )
    radar_input_path = args.radar_input or resolve_existing_path(
        [
            PROJECT_ROOT / "radar_data.PNG",
            PROJECT_ROOT / "output" / "radar_equation_power_dbw.npy",
            PROJECT_ROOT / "output" / "radar_equation_map.png",
        ]
    )

    pipeline_switches = load_pipeline_switches(args.pipeline_config)
    optical_rgb = read_rgb(optical_path)
    weak_labels = segment_territories(
        rgb_image=optical_rgb,
        model_path=args.terrain_model_path,
        pipeline_switches=pipeline_switches,
    )

    radar_observation = load_radar_observation(radar_input_path)
    aligned_labels, alignment_report = align_label_map_to_radar(
        weak_labels.class_map.astype(np.int32),
        target_shape=radar_observation.intensity.shape,
        mode=args.alignment_mode,
    )
    aligned_sample_mask = aligned_labels >= 0
    training_sample_mask = _build_training_sample_mask(
        active_mask=radar_observation.active_mask,
        support_mask=radar_observation.support_mask,
    )
    training_sample_mask &= aligned_sample_mask

    feature_cube = build_radar_feature_cube(radar_observation)
    training = train_radar_territory_classifier(
        feature_cube=feature_cube,
        labels=np.maximum(aligned_labels, 0).astype(np.int32),
        active_mask=radar_observation.active_mask,
        output_path=args.output_model,
        patch_size=args.patch_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        sample_mask=training_sample_mask,
    )
    inference = classify_radar_observation(radar_observation, checkpoint_path=args.output_model)

    debug_dir = ensure_dir(args.debug_dir)
    save_rgb(debug_dir / "radar_input.png", radar_observation.visualization_rgb)
    save_rgb(debug_dir / "weak_labels.png", weak_labels.color_map)
    save_rgb(debug_dir / "weak_labels_aligned.png", _render_label_map(aligned_labels))
    save_rgb(debug_dir / "train_prediction.png", inference.color_map)
    save_rgb(debug_dir / "train_prediction_overlay.png", inference.overlay)
    _save_mask(debug_dir / "active_mask.png", radar_observation.active_mask)
    _save_mask(debug_dir / "support_mask.png", radar_observation.support_mask)
    _save_mask(debug_dir / "alignment_mask.png", aligned_sample_mask)
    _save_mask(debug_dir / "training_sample_mask.png", training_sample_mask)
    np.save(debug_dir / "train_probabilities.npy", inference.probabilities.astype(np.float32))
    save_json(
        debug_dir / "training_summary.json",
        {
            "optical_path": str(optical_path),
            "radar_input_path": str(radar_input_path),
            "alignment_report": alignment_report,
            "output_model": str(training.model_path),
            "patch_size": training.patch_size,
            "train_samples": training.train_samples,
            "val_samples": training.val_samples,
            "best_val_accuracy": training.best_val_accuracy,
            "class_histogram": training.class_histogram,
            "history": training.history,
            "feature_names": list(RADAR_FEATURE_NAMES),
            "inference_report": dict(inference.report),
        },
    )

    print(f"model={training.model_path}")
    print(f"best_val_accuracy={training.best_val_accuracy:.4f}")
    print(f"train_samples={training.train_samples}")
    print(f"val_samples={training.val_samples}")
    print(f"debug_dir={debug_dir}")


if __name__ == "__main__":
    main()
