"""Fresh optical-to-radar baseline written from scratch for this workspace."""

from .physics_model import RadarPhysicsConfig, SynthesisReport, synthesize_from_rgb
from .radar_reference import RadarReference, extract_radar_reference
from .terrain_resnet import (
    TERRAIN_CLASS_NAMES,
    TerrainTrainingResult,
    load_trained_model,
    predict_terrain_probabilities,
    render_terrain_map,
    train_terrain_resnet,
)

__all__ = [
    "RadarPhysicsConfig",
    "RadarReference",
    "SynthesisReport",
    "TERRAIN_CLASS_NAMES",
    "TerrainTrainingResult",
    "extract_radar_reference",
    "load_trained_model",
    "predict_terrain_probabilities",
    "render_terrain_map",
    "synthesize_from_rgb",
    "train_terrain_resnet",
]
