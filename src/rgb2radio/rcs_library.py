from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class MaterialRCSProfile:
    sigma0_db: float
    point_rcs_m2: float
    scatterer_density_m2: float
    mean_height_m: float
    attenuation_np_per_m: float
    specularity: float
    roughness: float


RCS_PROFILES: Dict[str, MaterialRCSProfile] = {
    "forest": MaterialRCSProfile(
        sigma0_db=-10.0,
        point_rcs_m2=0.10,
        scatterer_density_m2=2.8,
        mean_height_m=7.0,
        attenuation_np_per_m=0.015,
        specularity=0.20,
        roughness=0.55,
    ),
    "water": MaterialRCSProfile(
        sigma0_db=-26.0,
        point_rcs_m2=0.005,
        scatterer_density_m2=0.05,
        mean_height_m=0.05,
        attenuation_np_per_m=0.0,
        specularity=0.95,
        roughness=0.05,
    ),
    "rippling_water": MaterialRCSProfile(
        sigma0_db=-9.0,
        point_rcs_m2=0.03,
        scatterer_density_m2=0.65,
        mean_height_m=0.08,
        attenuation_np_per_m=0.0,
        specularity=0.55,
        roughness=0.90,
    ),
    "concrete_building": MaterialRCSProfile(
        sigma0_db=-2.0,
        point_rcs_m2=3.5,
        scatterer_density_m2=0.75,
        mean_height_m=5.5,
        attenuation_np_per_m=0.0,
        specularity=0.65,
        roughness=0.55,
    ),
    "metal_building": MaterialRCSProfile(
        sigma0_db=4.0,
        point_rcs_m2=12.0,
        scatterer_density_m2=0.90,
        mean_height_m=6.0,
        attenuation_np_per_m=0.0,
        specularity=0.95,
        roughness=0.75,
    ),
    "dirt_road": MaterialRCSProfile(
        sigma0_db=-12.0,
        point_rcs_m2=0.20,
        scatterer_density_m2=0.18,
        mean_height_m=0.10,
        attenuation_np_per_m=0.0,
        specularity=0.20,
        roughness=0.70,
    ),
    "asphalt_road": MaterialRCSProfile(
        sigma0_db=-17.0,
        point_rcs_m2=0.08,
        scatterer_density_m2=0.12,
        mean_height_m=0.08,
        attenuation_np_per_m=0.0,
        specularity=0.35,
        roughness=0.25,
    ),
    "wood_building": MaterialRCSProfile(
        sigma0_db=-6.0,
        point_rcs_m2=1.4,
        scatterer_density_m2=0.55,
        mean_height_m=4.5,
        attenuation_np_per_m=0.002,
        specularity=0.35,
        roughness=0.50,
    ),
}
