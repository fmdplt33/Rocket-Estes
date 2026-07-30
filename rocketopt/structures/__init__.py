"""Structures: material database and structural margin analysis."""

from __future__ import annotations

from rocketopt.structures.analysis import (
    FlutterResult,
    MarginResult,
    StructuralReport,
    analyse_structure,
    fin_bending_stress,
    fin_divergence_velocity,
    fin_flutter_velocity,
    fin_joint_shear_stress,
    landing_deceleration,
    tube_axial_stress,
    tube_buckling_stress,
)
from rocketopt.structures.materials import (
    ADHESIVES,
    MATERIALS,
    Adhesive,
    Material,
    MaterialCategory,
    SurfaceFinish,
    get_adhesive,
    get_material,
    materials_in_category,
    nose_cone_materials,
    sheet_materials,
    tube_materials,
)

__all__ = [
    "ADHESIVES",
    "MATERIALS",
    "Adhesive",
    "FlutterResult",
    "MarginResult",
    "Material",
    "MaterialCategory",
    "StructuralReport",
    "SurfaceFinish",
    "analyse_structure",
    "fin_bending_stress",
    "fin_divergence_velocity",
    "fin_flutter_velocity",
    "fin_joint_shear_stress",
    "get_adhesive",
    "get_material",
    "landing_deceleration",
    "materials_in_category",
    "nose_cone_materials",
    "sheet_materials",
    "tube_axial_stress",
    "tube_buckling_stress",
    "tube_materials",
]
