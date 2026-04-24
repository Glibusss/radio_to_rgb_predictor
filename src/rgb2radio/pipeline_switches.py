from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping


DEFAULT_PIPELINE_SWITCHES: Dict[str, bool] = {
    "train_if_missing_checkpoint": True,
    "use_model_inference": True,
    "use_shadow_refinement": True,
    "use_vehicle_refinement": True,
    "use_probability_smoothing": True,
    "use_component_cleanup": True,
    "use_rcs_mapping": True,
    "use_radar_equation": True,
    "use_radar_boundaries": True,
    "use_brightness_normalization": False,
    "use_shadow_rcs_nearest_majority": True,
    "use_vehicle_rcs_distribution": False,
}


def load_pipeline_switches(path: str | Path) -> Dict[str, bool]:
    config_path = Path(path)
    if not config_path.exists():
        return dict(DEFAULT_PIPELINE_SWITCHES)

    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("Pipeline switches config must be a JSON object.")

    unknown = sorted(set(payload) - set(DEFAULT_PIPELINE_SWITCHES))
    if unknown:
        raise ValueError(f"Unknown pipeline switches: {unknown}")

    resolved = dict(DEFAULT_PIPELINE_SWITCHES)
    for key, value in payload.items():
        if not isinstance(value, bool):
            raise ValueError(f"Pipeline switch '{key}' must be true or false.")
        resolved[key] = value
    return resolved
