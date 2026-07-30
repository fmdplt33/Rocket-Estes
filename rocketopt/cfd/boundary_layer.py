"""Boundary-layer development and separation prediction.

The potential-flow solution in :mod:`rocketopt.cfd.potential_flow` gives the
velocity just outside the boundary layer. This module integrates the boundary
layer through that velocity distribution to find where it thickens, where it
transitions and where it separates.

Method
------
*Laminar region* - Thwaites' momentum-integral method [1]. The momentum
thickness follows from

``theta^2 = (0.45 nu / Ue^6) integral_0^s Ue^5 ds``

and the pressure-gradient parameter is ``lambda = (theta^2 / nu) dUe/ds``.
Laminar separation occurs at ``lambda = -0.09``.

*Transition* - by the momentum-thickness Reynolds number criterion
``Re_theta > 1.174 (1 + 22400 / Re_s) Re_s^0.46``, due to Cebeci and Smith
[2]. On a real model rocket the surface trips the layer far earlier than this
predicts, so the reported transition point is an upper bound and the drag
build-up assumes fully turbulent flow regardless.

*Turbulent region* - momentum integral with the Ludwieg-Tillmann skin-friction
law [3] and a shape factor advanced by Head's entrainment method [4].
Turbulent separation is taken at ``H = 2.4``, the value conventionally used
for an attached-to-separated transition [5].

Why it matters
--------------
Separation location sets base drag and wake width. On a rocket the flow
normally separates at the boat-tail or the base, and a boat-tail steeper than
about 12 degrees separates early enough to lose the drag benefit it was added
for. This module is what makes that visible rather than assumed.

References
----------
[1] Thwaites, B. (1949). "Approximate calculation of the laminar boundary
    layer." *Aeronautical Quarterly*, 1(3), 245-280.
[2] Cebeci, T., & Smith, A. M. O. (1974). *Analysis of Turbulent Boundary
    Layers*. Academic Press.
[3] Ludwieg, H., & Tillmann, W. (1950). "Investigations of the wall shearing
    stress in turbulent boundary layers." NACA TM 1285.
[4] Head, M. R. (1958). "Entrainment in the turbulent boundary layer." ARC
    R&M 3152.
[5] White, F. M. (2006). *Viscous Fluid Flow*, 3rd ed., Ch. 6-7.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from rocketopt.cfd.potential_flow import PotentialFlowSolution

__all__ = [
    "FlowRegime",
    "BoundaryLayerSolution",
    "solve_boundary_layer",
]

_THWAITES_SEPARATION = -0.09
"""Value of Thwaites' lambda at laminar separation [1]."""

_TURBULENT_SEPARATION_H = 2.4
"""Shape factor at turbulent separation [5]."""


class FlowRegime(str, Enum):
    """State of the boundary layer at a station."""

    LAMINAR = "laminar"
    TURBULENT = "turbulent"
    SEPARATED = "separated"

    @property
    def label(self) -> str:
        """Human-readable name."""
        return self.name.title()


@dataclass(frozen=True, slots=True)
class BoundaryLayerSolution:
    """Boundary-layer state along the body surface.

    Attributes
    ----------
    arc_length:
        Distance along the surface from the tip [m].
    momentum_thickness:
        Momentum thickness at each station [m].
    displacement_thickness:
        Displacement thickness at each station [m].
    shape_factor:
        Shape factor ``H = delta* / theta`` at each station [-].
    skin_friction:
        Local skin-friction coefficient at each station [-].
    regime:
        Flow regime at each station.
    transition_arc_length:
        Distance along the surface at which transition was predicted [m], or
        ``None`` if the layer stayed laminar.
    separation_arc_length:
        Distance at which separation was predicted [m], or ``None`` if the
        flow stayed attached to the tail.
    """

    arc_length: NDArray[np.float64]
    momentum_thickness: NDArray[np.float64]
    displacement_thickness: NDArray[np.float64]
    shape_factor: NDArray[np.float64]
    skin_friction: NDArray[np.float64]
    regime: tuple[FlowRegime, ...]
    transition_arc_length: float | None
    separation_arc_length: float | None

    @property
    def separates(self) -> bool:
        """Whether the boundary layer separates before the tail."""
        return self.separation_arc_length is not None

    @property
    def boundary_layer_thickness(self) -> NDArray[np.float64]:
        """Approximate total thickness ``delta`` at each station [m].

        For a turbulent layer ``delta`` is roughly ``7.7 theta``; for a laminar
        Blasius layer roughly ``7.5 theta``. A single factor of 7.7 is used,
        which is within the uncertainty of the method either way [5].
        """
        return 7.7 * self.momentum_thickness

    def summary(self) -> str:
        """Return a short readable description of the solution."""
        lines = []
        if self.transition_arc_length is not None:
            lines.append(
                f"Natural transition predicted at "
                f"{self.transition_arc_length * 1e3:.0f} mm along the surface "
                f"(a real finish will trip it far sooner)."
            )
        else:
            lines.append("The layer remains laminar to the tail by this criterion.")

        if self.separation_arc_length is not None:
            lines.append(
                f"Separation predicted at "
                f"{self.separation_arc_length * 1e3:.0f} mm along the surface."
            )
        else:
            lines.append("The flow stays attached to the base.")

        lines.append(
            f"Peak boundary-layer thickness "
            f"{float(np.max(self.boundary_layer_thickness)) * 1e3:.2f} mm."
        )
        return " ".join(lines)


def solve_boundary_layer(
    solution: PotentialFlowSolution,
    *,
    kinematic_viscosity: float = 1.46e-5,
    force_turbulent: bool = True,
) -> BoundaryLayerSolution:
    """Integrate the boundary layer through a potential-flow solution.

    Parameters
    ----------
    solution:
        The inviscid solution supplying the edge velocity distribution.
    kinematic_viscosity:
        Kinematic viscosity of the air [m^2/s]. The default is the
        sea-level standard value.
    force_turbulent:
        When ``True`` the layer is treated as turbulent from very near the
        tip, which is what actually happens on a model rocket: the spiral
        seam of a wound tube and ordinary paint texture trip it within a few
        centimetres. Set ``False`` to see where natural transition would
        occur on an idealised smooth body.

    Returns
    -------
    BoundaryLayerSolution
        Thicknesses, shape factor, skin friction and regime at every station.
    """
    profile = solution.profile
    s = profile.surface_arc_length()
    edge_velocity = np.maximum(solution.surface_velocity, 1e-6)
    nu = kinematic_viscosity

    n = len(s)
    theta = np.zeros(n)
    shape = np.full(n, 2.59)  # Blasius value at the tip.
    cf = np.zeros(n)
    regime: list[FlowRegime] = []

    transition_s: float | None = None
    separation_s: float | None = None

    # -- Laminar march by Thwaites -----------------------------------------
    # theta^2 = 0.45 nu / Ue^6 * integral Ue^5 ds
    integrand = edge_velocity**5
    integral = np.concatenate(
        [[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(s))]
    )
    theta_laminar_sq = 0.45 * nu * integral / np.maximum(edge_velocity**6, 1e-12)
    theta_laminar = np.sqrt(np.maximum(theta_laminar_sq, 0.0))

    due_ds = np.gradient(edge_velocity, s)
    lam = theta_laminar_sq * due_ds / nu

    turbulent = False
    separated = False

    for i in range(n):
        if separated:
            regime.append(FlowRegime.SEPARATED)
            theta[i] = theta[i - 1]
            shape[i] = shape[i - 1]
            cf[i] = 0.0
            continue

        if not turbulent:
            # Laminar separation check.
            if lam[i] < _THWAITES_SEPARATION and s[i] > 1e-4:
                # A laminar layer that separates against an adverse gradient
                # normally reattaches turbulent on a body like this, so treat
                # it as forced transition rather than terminal separation.
                turbulent = True
                transition_s = transition_s or float(s[i])
            else:
                # Natural transition by momentum-thickness Reynolds number.
                re_s = edge_velocity[i] * max(s[i], 1e-9) / nu
                re_theta = edge_velocity[i] * theta_laminar[i] / nu
                critical = 1.174 * (1.0 + 22400.0 / max(re_s, 1.0)) * re_s**0.46
                trip_early = force_turbulent and s[i] > 0.02 * s[-1]
                if re_theta > critical or trip_early:
                    turbulent = True
                    transition_s = transition_s or float(s[i])

        if not turbulent:
            theta[i] = theta_laminar[i]
            # Thwaites' correlation for the shape factor.
            shape[i] = _thwaites_shape_factor(lam[i])
            re_theta = edge_velocity[i] * theta[i] / nu
            # Laminar skin friction from Thwaites' shear correlation.
            cf[i] = 2.0 * _thwaites_shear(lam[i]) / max(re_theta, 1e-6)
            regime.append(FlowRegime.LAMINAR)
            continue

        # -- Turbulent march ------------------------------------------------
        if i == 0:
            theta[i] = max(theta_laminar[i], 1e-7)
            shape[i] = 1.4
        else:
            ds = s[i] - s[i - 1]
            re_theta = max(edge_velocity[i - 1] * theta[i - 1] / nu, 1.0)

            # Ludwieg-Tillmann skin friction.
            cf_prev = 0.246 * (10.0 ** (-0.678 * shape[i - 1])) * re_theta**-0.268

            # von Karman momentum integral:
            # dtheta/ds = cf/2 - (H + 2) theta / Ue * dUe/ds
            dtheta = (
                cf_prev / 2.0
                - (shape[i - 1] + 2.0)
                * theta[i - 1]
                / max(edge_velocity[i - 1], 1e-6)
                * due_ds[i - 1]
            ) * ds
            theta[i] = max(theta[i - 1] + dtheta, 1e-9)

            # Head's entrainment method advances the shape factor.
            shape[i] = _advance_shape_factor(
                shape[i - 1],
                theta[i - 1],
                edge_velocity[i - 1],
                due_ds[i - 1],
                cf_prev,
                ds,
            )

        re_theta = max(edge_velocity[i] * theta[i] / nu, 1.0)
        cf[i] = 0.246 * (10.0 ** (-0.678 * shape[i])) * re_theta**-0.268

        if shape[i] >= _TURBULENT_SEPARATION_H and s[i] > 0.1 * s[-1]:
            separated = True
            separation_s = float(s[i])
            regime.append(FlowRegime.SEPARATED)
        else:
            regime.append(FlowRegime.TURBULENT)

    displacement = shape * theta

    return BoundaryLayerSolution(
        arc_length=s,
        momentum_thickness=theta,
        displacement_thickness=displacement,
        shape_factor=shape,
        skin_friction=cf,
        regime=tuple(regime),
        transition_arc_length=transition_s,
        separation_arc_length=separation_s,
    )


def _thwaites_shape_factor(lam: float) -> float:
    """Shape factor from Thwaites' correlation [1].

    Parameters
    ----------
    lam:
        Thwaites' pressure-gradient parameter.

    Returns
    -------
    float
        Shape factor ``H``.
    """
    lam = float(np.clip(lam, -0.1, 0.25))
    if lam >= 0.0:
        return 2.61 - 3.75 * lam + 5.24 * lam * lam
    return 2.088 + 0.0731 / (lam + 0.14)


def _thwaites_shear(lam: float) -> float:
    """Shear correlation ``l(lambda)`` from Thwaites' method [1]."""
    lam = float(np.clip(lam, -0.1, 0.25))
    if lam >= 0.0:
        return 0.22 + 1.57 * lam - 1.8 * lam * lam
    return 0.22 + 1.402 * lam + (0.018 * lam) / (lam + 0.107)


def _advance_shape_factor(
    shape: float,
    theta: float,
    edge_velocity: float,
    due_ds: float,
    cf: float,
    ds: float,
) -> float:
    """Advance the turbulent shape factor by Head's entrainment method [4].

    Head introduces the entrainment shape parameter ``H1``, correlated with
    ``H``, and closes the problem with an entrainment rate. Integrating the
    entrainment equation gives the new ``H1``, from which ``H`` is recovered.

    Parameters
    ----------
    shape:
        Shape factor at the previous station.
    theta:
        Momentum thickness at the previous station [m].
    edge_velocity:
        Edge velocity at the previous station [m/s].
    due_ds:
        Edge velocity gradient at the previous station [1/s].
    cf:
        Skin-friction coefficient at the previous station [-].
    ds:
        Step along the surface [m].

    Returns
    -------
    float
        Shape factor at the new station, clamped to a physical range.
    """
    h1 = _entrainment_parameter(shape)
    entrainment = 0.0306 * (h1 - 3.0) ** -0.6169 if h1 > 3.0 else 0.03

    # d(theta H1)/ds = Ue^-1 * ... ; expanded for dH1/ds:
    dh1 = (
        entrainment
        - h1 * (cf / 2.0)
        + h1 * (shape + 1.0) * theta / max(edge_velocity, 1e-6) * due_ds
    ) / max(theta, 1e-9)

    h1_new = max(h1 + dh1 * ds, 3.01)
    return float(np.clip(_shape_from_entrainment(h1_new), 1.05, 3.0))


def _entrainment_parameter(shape: float) -> float:
    """Head's ``H1`` from the shape factor ``H`` [4]."""
    if shape <= 1.6:
        return 3.3 + 0.8234 * (shape - 1.1) ** -1.287
    return 3.3 + 1.5501 * (shape - 0.6778) ** -3.064


def _shape_from_entrainment(h1: float) -> float:
    """Invert :func:`_entrainment_parameter` to recover ``H`` from ``H1``."""
    if h1 <= 3.3:
        return 3.0
    # The two branches meet at H = 1.6, where H1 is about 3.6367.
    if h1 < 3.6367:
        return 0.6778 + (1.5501 / (h1 - 3.3)) ** (1.0 / 3.064)
    return 1.1 + (0.8234 / (h1 - 3.3)) ** (1.0 / 1.287)
