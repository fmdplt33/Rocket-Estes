"""CAD export: solid models, 2D templates and parametric Fusion 360 scripts."""

from __future__ import annotations

from rocketopt.cad.exporters import (
    export_all,
    export_fin_dxf,
    export_fin_template_svg,
    export_profile_svg,
    export_solids,
    export_stl_parts,
)
from rocketopt.cad.fusion360 import build_fusion_script, write_fusion_script
from rocketopt.cad.mesh import TriangleMesh, write_stl
from rocketopt.cad.parts import (
    DEFAULT_TOLERANCES,
    PrintablePart,
    PrintTolerances,
    printable_parts,
)
from rocketopt.cad.validation import (
    AssemblyValidation,
    AssemblyValidationError,
    FitCheck,
    validate_printable_assembly,
)

__all__ = [
    "DEFAULT_TOLERANCES",
    "AssemblyValidation",
    "AssemblyValidationError",
    "FitCheck",
    "PrintTolerances",
    "PrintablePart",
    "TriangleMesh",
    "build_fusion_script",
    "export_all",
    "export_fin_dxf",
    "export_fin_template_svg",
    "export_profile_svg",
    "export_solids",
    "export_stl_parts",
    "printable_parts",
    "validate_printable_assembly",
    "write_fusion_script",
    "write_stl",
]
