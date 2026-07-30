"""Aerodynamics: atmosphere, Barrowman stability and component drag build-up."""

from __future__ import annotations

from rocketopt.aerodynamics.atmosphere import (
    Atmosphere,
    AtmosphereState,
    dynamic_viscosity,
    saturation_vapour_pressure,
)
from rocketopt.aerodynamics.barrowman import (
    BarrowmanResult,
    ComponentContribution,
    DampingDerivatives,
    barrowman_analysis,
    crossflow_drag_coefficient,
    damping_derivatives,
    prandtl_glauert,
    static_margin,
    stability_verdict,
)
from rocketopt.aerodynamics.drag import (
    DragBreakdown,
    base_drag_coefficient,
    drag_buildup,
    skin_friction_coefficient,
    wave_drag_coefficient,
)

__all__ = [
    "Atmosphere",
    "AtmosphereState",
    "BarrowmanResult",
    "ComponentContribution",
    "DampingDerivatives",
    "DragBreakdown",
    "barrowman_analysis",
    "base_drag_coefficient",
    "crossflow_drag_coefficient",
    "damping_derivatives",
    "drag_buildup",
    "dynamic_viscosity",
    "prandtl_glauert",
    "saturation_vapour_pressure",
    "skin_friction_coefficient",
    "stability_verdict",
    "static_margin",
    "wave_drag_coefficient",
]
