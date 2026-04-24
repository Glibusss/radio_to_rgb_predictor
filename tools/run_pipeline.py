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
from rgb2radio.pipeline_switches import load_pipeline_switches
from rgb2radio.radar_equation import map_radar_equation_to_pixels
from rgb2radio.rcs_mapping import map_rcs_to_pixels
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
    parser.add_argument("--pipeline-config", type=Path, default=Path("config/pipeline_switches.json"))
    parser.add_argument("--rcs-config", type=Path, default=Path("config/rcs_reference.json"))
    parser.add_argument("--radar-config", type=Path, default=Path("config/radar_config.json"))
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


def _remove_if_exists(path: Path) -> None:
    if path.exists() and path.is_file():
        path.unlink()


def main() -> None:
    args = build_parser().parse_args()
    pipeline_switches = load_pipeline_switches(args.pipeline_config)
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
        pipeline_switches=pipeline_switches,
    )
    rcs_enabled = bool(pipeline_switches.get("use_rcs_mapping", True))
    radar_enabled = bool(pipeline_switches.get("use_radar_equation", True))
    rcs_result = None
    radar_result = None
    if rcs_enabled:
        rcs_result = map_rcs_to_pixels(
            rgb_image=image_rgb,
            class_map=result.class_map,
            class_names=result.class_names,
            config_path=args.rcs_config,
            pipeline_switches=pipeline_switches,
        )
    if rcs_result is not None and radar_enabled:
        radar_result = map_radar_equation_to_pixels(
            rgb_image=image_rgb,
            class_names=result.class_names,
            probabilities=result.probabilities,
            pixel_rcs_map_m2=rcs_result.linear_map_m2,
            config_path=args.radar_config,
            pipeline_switches=pipeline_switches,
        )

    output_dir = ensure_dir(args.output_dir)
    final_path = output_dir / "territories.png"
    overlay_path = output_dir / "territories_overlay.png"
    rcs_map_path = output_dir / "rcs_map.png"
    rcs_overlay_path = output_dir / "rcs_overlay.png"
    rcs_map_m2_path = output_dir / "rcs_map_m2.npy"
    rcs_map_dbsm_path = output_dir / "rcs_map_dbsm.npy"
    rcs_report_path = output_dir / "rcs_report.json"
    radar_map_path = output_dir / "radar_equation_map.png"
    radar_overlay_path = output_dir / "radar_equation_overlay.png"
    radar_boundaries_path = output_dir / "radar_boundaries.png"
    radar_boundaries_overlay_path = output_dir / "radar_boundaries_overlay.png"
    radar_power_w_path = output_dir / "radar_equation_power_w.npy"
    radar_power_dbw_path = output_dir / "radar_equation_power_dbw.npy"
    radar_report_path = output_dir / "radar_equation_report.json"
    save_rgb(final_path, result.color_map)
    save_rgb(overlay_path, result.overlay)
    if rcs_result is not None:
        save_rgb(rcs_map_path, rcs_result.heatmap)
        save_rgb(rcs_overlay_path, rcs_result.overlay)
        np.save(rcs_map_m2_path, rcs_result.linear_map_m2.astype(np.float32))
        np.save(rcs_map_dbsm_path, rcs_result.dbsm_map.astype(np.float32))
        save_json(rcs_report_path, dict(rcs_result.report))
    else:
        _remove_if_exists(rcs_map_path)
        _remove_if_exists(rcs_overlay_path)
        _remove_if_exists(rcs_map_m2_path)
        _remove_if_exists(rcs_map_dbsm_path)
        _remove_if_exists(rcs_report_path)
    if radar_result is not None:
        save_rgb(radar_map_path, radar_result.heatmap)
        save_rgb(radar_overlay_path, radar_result.overlay)
        save_rgb(radar_boundaries_path, radar_result.boundary_map)
        save_rgb(radar_boundaries_overlay_path, radar_result.boundary_overlay)
        np.save(radar_power_w_path, radar_result.received_power_w.astype(np.float32))
        np.save(radar_power_dbw_path, radar_result.received_power_dbw.astype(np.float32))
        save_json(radar_report_path, dict(radar_result.report))
    else:
        _remove_if_exists(radar_map_path)
        _remove_if_exists(radar_overlay_path)
        _remove_if_exists(radar_boundaries_path)
        _remove_if_exists(radar_boundaries_overlay_path)
        _remove_if_exists(radar_power_w_path)
        _remove_if_exists(radar_power_dbw_path)
        _remove_if_exists(radar_report_path)

    if args.debug:
        debug_dir = ensure_dir(args.debug_dir)
        save_rgb(debug_dir / "input.png", image_rgb)
        save_rgb(debug_dir / "territories.png", result.color_map)
        save_rgb(debug_dir / "territories_overlay.png", result.overlay)
        for name, image in result.debug_maps.items():
            _save_debug_map(debug_dir / f"{name}.png", image)
        if rcs_result is not None:
            save_rgb(debug_dir / "rcs_map.png", rcs_result.heatmap)
            save_rgb(debug_dir / "rcs_overlay.png", rcs_result.overlay)
            np.save(debug_dir / "rcs_map_m2.npy", rcs_result.linear_map_m2.astype(np.float32))
            np.save(debug_dir / "rcs_map_dbsm.npy", rcs_result.dbsm_map.astype(np.float32))
            for name, image in rcs_result.debug_maps.items():
                _save_debug_map(debug_dir / f"{name}.png", image)
        else:
            _remove_if_exists(debug_dir / "rcs_map.png")
            _remove_if_exists(debug_dir / "rcs_overlay.png")
            _remove_if_exists(debug_dir / "rcs_map_m2.npy")
            _remove_if_exists(debug_dir / "rcs_map_dbsm.npy")
        if radar_result is not None:
            save_rgb(debug_dir / "radar_equation_map.png", radar_result.heatmap)
            save_rgb(debug_dir / "radar_equation_overlay.png", radar_result.overlay)
            save_rgb(debug_dir / "radar_boundaries.png", radar_result.boundary_map)
            save_rgb(debug_dir / "radar_boundaries_overlay.png", radar_result.boundary_overlay)
            np.save(debug_dir / "radar_equation_power_w.npy", radar_result.received_power_w.astype(np.float32))
            np.save(debug_dir / "radar_equation_power_dbw.npy", radar_result.received_power_dbw.astype(np.float32))
            for name, image in radar_result.debug_maps.items():
                _save_debug_map(debug_dir / f"{name}.png", image)
        else:
            _remove_if_exists(debug_dir / "radar_equation_map.png")
            _remove_if_exists(debug_dir / "radar_equation_overlay.png")
            _remove_if_exists(debug_dir / "radar_boundaries.png")
            _remove_if_exists(debug_dir / "radar_boundaries_overlay.png")
            _remove_if_exists(debug_dir / "radar_equation_power_w.npy")
            _remove_if_exists(debug_dir / "radar_equation_power_dbw.npy")
        if args.model_path.exists():
            export_training_debug_images(args.model_path, debug_dir / "training")
        report_payload = {
            "input_path": str(input_path),
            "output_path": str(final_path),
            "overlay_output_path": str(overlay_path),
            "class_names": list(TERRITORY_CLASS_NAMES),
            "pipeline_config_path": str(args.pipeline_config),
            "pipeline_switches": dict(pipeline_switches),
            "report": result.report,
        }
        if rcs_result is not None:
            report_payload["rcs_map_path"] = str(rcs_map_path)
            report_payload["rcs_overlay_path"] = str(rcs_overlay_path)
            report_payload["rcs_report_path"] = str(rcs_report_path)
            report_payload["rcs_report"] = rcs_result.report
        if radar_result is not None:
            report_payload["radar_equation_map_path"] = str(radar_map_path)
            report_payload["radar_equation_overlay_path"] = str(radar_overlay_path)
            report_payload["radar_boundaries_path"] = str(radar_boundaries_path)
            report_payload["radar_boundaries_overlay_path"] = str(radar_boundaries_overlay_path)
            report_payload["radar_equation_report_path"] = str(radar_report_path)
            report_payload["radar_equation_report"] = radar_result.report
        save_json(
            debug_dir / "report.json",
            report_payload,
        )

    distribution = result.report["class_distribution"]
    summary = ", ".join(
        f"{class_name}={100.0 * distribution[class_name]['pixel_share']:.1f}%"
        for class_name in TERRITORY_CLASS_NAMES
    )
    print(f"territories={final_path}")
    if rcs_result is not None:
        print(f"rcs_map={rcs_map_path}")
        print(
            "rcs_range_dbsm="
            f"{rcs_result.report['summary']['min_pixel_rcs_dbsm']:.1f}.."
            f"{rcs_result.report['summary']['max_pixel_rcs_dbsm']:.1f}"
        )
    else:
        print("rcs_map=disabled")
    if radar_result is not None:
        print(f"radar_equation_map={radar_map_path}")
        print(
            "radar_equation_range_dbw="
            f"{radar_result.report['summary']['min_received_power_dbw']:.1f}.."
            f"{radar_result.report['summary']['max_received_power_dbw']:.1f}"
        )
    else:
        print("radar_equation_map=disabled")
    print(summary)
    print(f"mode={result.training_summary.get('fallback_mode', 'heuristic_only')}")
    if args.debug:
        print(f"debug_dir={args.debug_dir}")


if __name__ == "__main__":
    main()
