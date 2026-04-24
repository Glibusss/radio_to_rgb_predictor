"""Territory segmentation utilities for this workspace."""

from .pipeline_switches import DEFAULT_PIPELINE_SWITCHES, load_pipeline_switches
from .radar_equation import RadarEquationResult, map_radar_equation_to_pixels
from .rcs_mapping import RcsMappingResult, map_rcs_to_pixels
from .terrain_resnet import (
    TERRAIN_CLASS_NAMES,
    TerrainTrainingResult,
    load_trained_model,
    predict_terrain_probabilities,
    render_terrain_map,
    train_terrain_resnet,
)
from .territory_segmentation import (
    TERRITORY_CLASS_NAMES,
    TerritorySegmentationResult,
    render_territory_map,
    segment_territories,
)

__all__ = [
    "TERRAIN_CLASS_NAMES",
    "TERRITORY_CLASS_NAMES",
    "DEFAULT_PIPELINE_SWITCHES",
    "RadarEquationResult",
    "RcsMappingResult",
    "TerrainTrainingResult",
    "TerritorySegmentationResult",
    "load_pipeline_switches",
    "load_trained_model",
    "map_radar_equation_to_pixels",
    "map_rcs_to_pixels",
    "predict_terrain_probabilities",
    "render_terrain_map",
    "render_territory_map",
    "segment_territories",
    "train_terrain_resnet",
]
