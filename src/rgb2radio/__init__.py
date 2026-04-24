"""Territory segmentation utilities for this workspace."""

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
    "RcsMappingResult",
    "TerrainTrainingResult",
    "TerritorySegmentationResult",
    "load_trained_model",
    "map_rcs_to_pixels",
    "predict_terrain_probabilities",
    "render_terrain_map",
    "render_territory_map",
    "segment_territories",
    "train_terrain_resnet",
]
