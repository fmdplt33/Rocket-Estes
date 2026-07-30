"""Axisymmetric potential-flow solution by an axial source distribution.

Method
------
For a slender body of revolution at small incidence, the disturbance the body
makes to a uniform stream is represented by a line of sources along its own
axis. The source strength per unit length is set by the rate at which the body
displaces fluid, which for a body of cross-sectional area ``S(x)`` is

``lambda(x) = U_inf * dS/dx``

This is the classical slender-body result of von Karman [1] and Munk [2]. The
perturbation potential at a field point is then

``phi(x, r) = -(1 / 4 pi) integral lambda(xi) / sqrt((x - xi)^2 + r^2) dxi``

and the velocity components follow by differentiation. Pressure comes from the
incompressible Bernoulli relation

``Cp = 1 - (|V| / U_inf)^2``

with a Prandtl-Glauert correction applied for compressibility.

What this is and is not
-----------------------
This is a genuine inviscid solution of Laplace's equation for the body's own
displacement effect, computed numerically. It predicts the favourable
pressure gradient over the nose, the suction peak at the shoulder and the
recovery along the body, which is what drives boundary-layer behaviour and
hence where the flow separates.

It is *not* a Navier-Stokes solution. It carries no viscosity, so it cannot
by itself predict separation, wake width or total drag - those come from the
boundary-layer solution in :mod:`rocketopt.cfd.boundary_layer` and the drag
build-up in :mod:`rocketopt.aerodynamics.drag`. Being a slender-body method it
also loses accuracy where the body is not slender: at a blunt nose tip and at
an abrupt shoulder the local Cp is indicative rather than exact.

References
----------
[1] von Karman, T. (1927). "Calculation of pressure distribution on airship
    hulls." NACA TM 574.
[2] Munk, M. M. (1924). "The aerodynamic forces on airship hulls." NACA
    Report 184.
[3] Ashley, H., & Landahl, M. (1965). *Aerodynamics of Wings and Bodies*,
    Ch. 6 (slender bodies of revolution).
[4] Anderson, J. D. (2016). *Fundamentals of Aerodynamics*, 6th ed., Ch. 6.
[5] Katz, J., & Plotkin, A. (2001). *Low-Speed Aerodynamics*, 2nd ed., Ch. 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rocketopt.geometry.components import BodyTube, Transition
from rocketopt.geometry.rocket import Rocket

__all__ = [
    "BodyProfile",
    "PotentialFlowSolution",
    "body_profile",
    "solve_potential_flow",
]

_EPSILON: Final[float] = 1.0e-9
"""Softening length used to keep the source kernel finite on the axis [m]."""


@dataclass(frozen=True, slots=True)
class BodyProfile:
    """The outer surface of a rocket as a radius distribution.

    Attributes
    ----------
    x:
        Axial stations from nose tip to tail [m], strictly increasing.
    radius:
        Body radius at each station [m].
    """

    x: NDArray[np.float64]
    radius: NDArray[np.float64]

    @property
    def area(self) -> NDArray[np.float64]:
        """Cross-sectional area at each station [m^2]."""
        return math.pi * self.radius**2

    @property
    def length(self) -> float:
        """Overall length [m]."""
        return float(self.x[-1] - self.x[0])

    @property
    def max_radius(self) -> float:
        """Maximum body radius [m]."""
        return float(np.max(self.radius))

    def surface_arc_length(self) -> NDArray[np.float64]:
        """Cumulative distance along the surface from the tip [m].

        Boundary-layer growth is driven by distance along the wetted surface,
        not by axial station, and the two differ substantially over a nose
        cone.
        """
        dx = np.diff(self.x)
        dr = np.diff(self.radius)
        ds = np.sqrt(dx * dx + dr * dr)
        return np.concatenate([[0.0], np.cumsum(ds)])


def body_profile(rocket: Rocket, n_points: int = 240) -> BodyProfile:
    """Sample a rocket's outer surface.

    Fins and the launch lug are excluded: this is the axisymmetric body the
    potential-flow model represents.

    Parameters
    ----------
    rocket:
        The design.
    n_points:
        Number of stations to sample.

    Returns
    -------
    BodyProfile
        The sampled surface.
    """
    xs: list[float] = []
    rs: list[float] = []

    # Nose, sampled from its analytic profile.
    nose_n = max(int(n_points * 0.45), 20)
    nose_x, nose_r = rocket.nose.profile_points(nose_n)
    xs.extend(float(v) for v in nose_x)
    rs.extend(float(v) for v in nose_r)

    # Body sections.
    remaining = max(n_points - nose_n, 10)
    total_section_length = sum(p.section.length for p in rocket.sections)

    for placed in rocket.sections:
        section = placed.section
        share = section.length / total_section_length if total_section_length else 1.0
        count = max(int(remaining * share), 4)
        local = np.linspace(0.0, section.length, count)[1:]  # skip duplicate joint

        for offset in local:
            xs.append(placed.position + float(offset))
            if isinstance(section, Transition):
                frac = float(offset) / section.length
                rs.append(
                    section.fore_radius
                    + frac * (section.aft_radius - section.fore_radius)
                )
            else:
                assert isinstance(section, BodyTube)
                rs.append(section.outer_radius)

    x = np.asarray(xs, dtype=np.float64)
    r = np.asarray(rs, dtype=np.float64)

    # Enforce strict monotonicity in x, which the solver requires.
    keep = np.concatenate([[True], np.diff(x) > 1e-9])
    return BodyProfile(x=x[keep], radius=r[keep])


@dataclass(frozen=True, slots=True)
class PotentialFlowSolution:
    """Inviscid flow field and surface pressure for one condition.

    Attributes
    ----------
    profile:
        The body surface that was solved.
    freestream_velocity:
        Free-stream speed [m/s].
    mach:
        Free-stream Mach number [-].
    surface_cp:
        Pressure coefficient at each profile station [-].
    surface_velocity:
        Flow speed just outside the boundary layer at each station [m/s].
    source_strength:
        Line source strength per unit length at each station [m^2/s].
    """

    profile: BodyProfile
    freestream_velocity: float
    mach: float
    surface_cp: NDArray[np.float64]
    surface_velocity: NDArray[np.float64]
    source_strength: NDArray[np.float64]

    @property
    def valid_from_index(self) -> int:
        """First station at which the slender-body result is trustworthy.

        Slender-body theory assumes the body radius is small compared with its
        length but *large* compared with the source spacing. Within a couple of
        element widths of a pointed tip the second condition fails, and the
        computed pressure there is an artefact of the discretisation rather
        than a physical result. Peak-finding skips that region.
        """
        radius = self.profile.radius
        spacing = float(np.mean(np.gradient(self.profile.x)))
        for index, r in enumerate(radius):
            if r > spacing:
                return index
        return 0

    @property
    def stagnation_index(self) -> int:
        """Index of the highest-pressure station, the stagnation region."""
        start = self.valid_from_index
        return start + int(np.argmax(self.surface_cp[start:]))

    @property
    def suction_peak_index(self) -> int:
        """Index of the lowest-pressure station, the suction peak.

        Measured over the region where slender-body theory is valid; see
        :attr:`valid_from_index`.
        """
        start = self.valid_from_index
        return start + int(np.argmin(self.surface_cp[start:]))

    @property
    def suction_peak_cp(self) -> float:
        """Minimum pressure coefficient on the body [-]."""
        return float(self.surface_cp[self.suction_peak_index])

    def adverse_gradient_start(self) -> float:
        """Axial station where the pressure gradient turns adverse [m].

        Downstream of the suction peak the pressure rises again, which is what
        eventually separates the boundary layer. Returns the station of the
        suction peak.
        """
        return float(self.profile.x[self.suction_peak_index])

    def velocity_field(
        self,
        x_grid: NDArray[np.float64],
        r_grid: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Evaluate the velocity field on a grid of points.

        Parameters
        ----------
        x_grid, r_grid:
            Arrays of the same shape giving field-point coordinates [m].

        Returns
        -------
        tuple of numpy.ndarray
            Axial and radial velocity components [m/s], same shape as input.
        """
        return _induced_velocity(
            self.profile.x,
            self.source_strength,
            np.asarray(x_grid, dtype=np.float64),
            np.asarray(r_grid, dtype=np.float64),
            self.freestream_velocity,
        )

    def pressure_field(
        self,
        x_grid: NDArray[np.float64],
        r_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Pressure coefficient on a grid of field points [-].

        Parameters
        ----------
        x_grid, r_grid:
            Field-point coordinates [m].

        Returns
        -------
        numpy.ndarray
            Pressure coefficient at each point.
        """
        u, v = self.velocity_field(x_grid, r_grid)
        speed_squared = u * u + v * v
        cp = 1.0 - speed_squared / (self.freestream_velocity**2)
        return _apply_compressibility(cp, self.mach)


def _apply_compressibility(
    cp: NDArray[np.float64] | float, mach: float
) -> NDArray[np.float64]:
    """Apply the Prandtl-Glauert correction to an incompressible Cp.

    ``Cp = Cp_incompressible / sqrt(1 - M^2)`` [4], Sec. 11.4. Frozen above
    Mach 0.8, where the correction diverges and the linearisation behind it
    has failed anyway.
    """
    m = min(abs(mach), 0.8)
    beta = math.sqrt(max(1.0 - m * m, 1e-6))
    return np.asarray(cp, dtype=np.float64) / beta


def _induced_velocity(
    source_x: NDArray[np.float64],
    strength: NDArray[np.float64],
    field_x: NDArray[np.float64],
    field_r: NDArray[np.float64],
    freestream: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Velocity induced by an axial source line plus the free stream.

    The potential of a point source of strength ``sigma`` at ``xi`` on the
    axis is ``-sigma / (4 pi d)`` with ``d`` the distance to the field point.
    Differentiating and integrating along the line gives the components below
    [3], Ch. 6.

    Parameters
    ----------
    source_x:
        Axial positions of the source elements [m].
    strength:
        Source strength per unit length at each element [m^2/s].
    field_x, field_r:
        Field-point coordinates [m], any matching shape.
    freestream:
        Free-stream speed [m/s].

    Returns
    -------
    tuple of numpy.ndarray
        Axial and radial velocity [m/s].
    """
    shape = field_x.shape
    fx = field_x.ravel()[:, None]  # (points, 1)
    fr = field_r.ravel()[:, None]

    sx = source_x[None, :]  # (1, sources)
    dx = fx - sx
    r2 = fr * fr + _EPSILON * _EPSILON
    distance = np.sqrt(dx * dx + r2)
    distance_cubed = distance**3

    # Element width for the integration, by the trapezoidal rule.
    widths = np.gradient(source_x)
    weight = (strength * widths)[None, :] / (4.0 * math.pi)

    u = freestream + np.sum(weight * dx / distance_cubed, axis=1)
    v = np.sum(weight * fr / distance_cubed, axis=1)

    return u.reshape(shape), v.reshape(shape)


def solve_potential_flow(
    rocket: Rocket,
    *,
    velocity: float,
    mach: float = 0.0,
    n_points: int = 240,
) -> PotentialFlowSolution:
    """Solve the inviscid flow about a rocket body.

    Parameters
    ----------
    rocket:
        The design.
    velocity:
        Free-stream speed [m/s].
    mach:
        Free-stream Mach number, for the compressibility correction [-].
    n_points:
        Number of surface stations.

    Returns
    -------
    PotentialFlowSolution
        Surface pressure and the source distribution needed to evaluate the
        field anywhere.

    Raises
    ------
    ValueError
        If ``velocity`` is not positive.
    """
    if velocity <= 0.0:
        raise ValueError("free-stream velocity must be positive")

    profile = body_profile(rocket, n_points)

    # Source strength follows the rate of change of cross-sectional area.
    area = profile.area
    strength = velocity * np.gradient(area, profile.x)

    # Evaluate the velocity just outside the surface. A field point on the
    # surface would sit on the singularity of its own source element, so it is
    # offset outward. Near a pointed tip the local radius tends to zero, so a
    # radius-proportional offset would place the point *inside* the singular
    # region and return a large spurious suction. The offset is therefore
    # floored at a fraction of the local element spacing, which is the length
    # scale over which the discretised source line is meaningful at all.
    spacing = np.gradient(profile.x)
    offset = np.maximum(profile.radius * 1.02, 0.75 * spacing)
    u, v = _induced_velocity(profile.x, strength, profile.x, offset, velocity)

    speed = np.sqrt(u * u + v * v)
    cp_incompressible = 1.0 - (speed / velocity) ** 2
    cp = _apply_compressibility(cp_incompressible, mach)

    # Physical bound: Cp cannot exceed 1 at a stagnation point in incompressible
    # flow. Slender-body theory can overshoot at a blunt tip, where it is known
    # to be invalid; clip rather than report an impossible value.
    cp = np.minimum(cp, 1.0)

    return PotentialFlowSolution(
        profile=profile,
        freestream_velocity=velocity,
        mach=mach,
        surface_cp=cp,
        surface_velocity=speed,
        source_strength=strength,
    )
