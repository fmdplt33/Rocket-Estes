"""Engineering CFD approximations: potential flow and boundary layers.

Not a Navier-Stokes solver. The inviscid field is a genuine numerical solution
of Laplace's equation for the body's displacement effect, and the boundary
layer is a genuine integration of the momentum-integral equations through that
field. Together they predict surface pressure, boundary-layer growth,
transition and separation without needing a mesh or a Reynolds-averaged
turbulence model.
"""

from __future__ import annotations

from rocketopt.cfd.boundary_layer import (
    BoundaryLayerSolution,
    FlowRegime,
    solve_boundary_layer,
)
from rocketopt.cfd.potential_flow import (
    BodyProfile,
    PotentialFlowSolution,
    body_profile,
    solve_potential_flow,
)

__all__ = [
    "BodyProfile",
    "BoundaryLayerSolution",
    "FlowRegime",
    "PotentialFlowSolution",
    "body_profile",
    "solve_boundary_layer",
    "solve_potential_flow",
]
