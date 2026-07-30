"""Parametric nose cone profiles.

Nine profile families are supported. Each is defined by its radius as a
function of axial station ``x`` measured aft from the tip, with ``x = 0`` at
the tip and ``x = L`` at the base where the radius equals ``R``.

Profile definitions
-------------------
Let ``f = x / L`` denote fractional station.

Conical
    ``y = R f``
Tangent ogive
    Circular arc of radius ``rho = (R^2 + L^2) / (2 R)`` meeting the body
    tangentially at the base.
Secant ogive
    Circular arc of radius ``rho > rho_tangent``; the arc meets the base at an
    angle, giving a fuller shoulder. Controlled by
    :attr:`NoseCone.shape_parameter` as a multiple of ``rho_tangent``.
Elliptical
    Quarter ellipse, ``y = R sqrt(1 - (1 - f)^2)``.
Parabolic series
    ``y = R (2f - K f^2) / (2 - K)`` with ``K`` in ``[0, 1]``. ``K = 0`` is a
    cone, ``K = 1`` the full parabola.
Power series
    ``y = R f^n`` with ``n`` in ``(0, 1]``. ``n = 1`` is a cone, ``n = 0.5``
    the classic 1/2-power ("parabolic") nose.
Haack series
    ``theta = arccos(1 - 2f)``,
    ``y = (R / sqrt(pi)) sqrt(theta - sin(2 theta)/2 + C sin^3(theta))``.
    ``C = 0`` gives the Von Karman / LD-Haack shape, which minimises
    pressure drag for a given length and base diameter. ``C = 1/3`` gives
    LV-Haack, which minimises drag for a given length and *volume*.

Note that LD-Haack and Von Karman are the same curve; both names are provided
because both appear in the literature and in other rocketry software.

Centre of pressure
------------------
Slender-body theory gives the nose contribution as ``CN_alpha = 2`` referenced
to the base area, with the centre of pressure at

``X_cp = L - V / A_base``

where ``V`` is the enclosed volume. This single expression reproduces every
per-shape constant Barrowman tabulates - ``2L/3`` for a cone, ``0.466 L`` for
a tangent ogive, ``L/3`` for an ellipse - so no lookup table is needed.

References
----------
[1] Crowell, G. A. (1996). *The Descriptive Geometry of Nose Cones*.
[2] Barrowman, J. S. (1967). *The Practical Calculation of the Aerodynamic
    Characteristics of Slender Finned Vehicles*. NASA TN, M.Sc. thesis,
    Catholic University of America.
[3] Haack, W. (1941). *Geschossformen kleinsten Wellenwiderstandes*.
    Bericht 139 der Lilienthal-Gesellschaft.
[4] von Karman, T. (1935). *The Problem of Resistance in Compressible Fluids*.
    Volta Congress, Rome.
[5] Niskanen, S. (2013). *OpenRocket Technical Documentation*, Sec. 3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rocketopt.structures.materials import Material, get_material

__all__ = [
    "MAX_SHOULDER_LENGTH",
    "MIN_SHOULDER_LENGTH",
    "NoseConeShape",
    "NoseCone",
    "SHAPE_PARAMETER_RANGES",
    "default_shape_parameter",
    "default_shoulder_length",
]

MIN_SHOULDER_LENGTH: Final[float] = 0.020
"""Shortest shoulder a nose cone is given by default [m].

Below about 20 mm the ejection charge works the joint in peel rather than in
shear and the shoulder starts to split away, whatever it is made of.
"""

MAX_SHOULDER_LENGTH: Final[float] = 0.040
"""Longest shoulder a nose cone is given by default [m].

Beyond 40 mm the shoulder is only taking up recovery bay; the joint is already
far stronger than the loads it sees.
"""


def default_shoulder_length(base_diameter: float, tube_length: float) -> float:
    """Return a sensible shoulder length for a cone on a given tube.

    One body diameter is the usual rule of thumb, clamped into
    ``[MIN_SHOULDER_LENGTH, MAX_SHOULDER_LENGTH]`` and to a quarter of the tube
    length so that a short airframe keeps a usable recovery bay.

    Parameters
    ----------
    base_diameter:
        Nose cone base diameter, which is the body tube's outside diameter [m].
    tube_length:
        Length of the body tube the shoulder plugs into [m].

    Returns
    -------
    float
        Shoulder length [m].
    """
    clamped = min(max(base_diameter, MIN_SHOULDER_LENGTH), MAX_SHOULDER_LENGTH)
    return min(clamped, 0.25 * tube_length)

_PROFILE_SAMPLES: Final[int] = 400
"""Stations used for numerical volume and wetted-area integration.

At 400 stations the trapezoidal volume of a cone is within 1e-6 relative
error of the analytic value, which is far below the uncertainty in any
aerodynamic coefficient it feeds.
"""


class NoseConeShape(str, Enum):
    """Nose cone profile family."""

    CONICAL = "conical"
    TANGENT_OGIVE = "tangent_ogive"
    SECANT_OGIVE = "secant_ogive"
    VON_KARMAN = "von_karman"
    HAACK_LD = "haack_ld"
    HAACK_LV = "haack_lv"
    ELLIPTICAL = "elliptical"
    PARABOLIC = "parabolic"
    POWER = "power"

    @property
    def label(self) -> str:
        """Human-readable name for reports and the UI."""
        return {
            NoseConeShape.CONICAL: "Conical",
            NoseConeShape.TANGENT_OGIVE: "Tangent ogive",
            NoseConeShape.SECANT_OGIVE: "Secant ogive",
            NoseConeShape.VON_KARMAN: "Von Karman (LD-Haack)",
            NoseConeShape.HAACK_LD: "LD-Haack (Von Karman)",
            NoseConeShape.HAACK_LV: "LV-Haack",
            NoseConeShape.ELLIPTICAL: "Elliptical",
            NoseConeShape.PARABOLIC: "Parabolic series",
            NoseConeShape.POWER: "Power series",
        }[self]

    @property
    def uses_shape_parameter(self) -> bool:
        """Whether this family has a free shape parameter."""
        return self in {
            NoseConeShape.SECANT_OGIVE,
            NoseConeShape.PARABOLIC,
            NoseConeShape.POWER,
        }

    @property
    def has_blunt_tip(self) -> bool:
        """Whether the profile leaves the tip with a finite radius of curvature.

        A power-series nose with ``n < 1`` and the Haack family both start with
        a vertical tangent, giving a very fine, fragile tip that in practice
        must be blunted. Reported so the manufacturability score can penalise
        it.
        """
        return self in {
            NoseConeShape.VON_KARMAN,
            NoseConeShape.HAACK_LD,
            NoseConeShape.HAACK_LV,
            NoseConeShape.POWER,
            NoseConeShape.ELLIPTICAL,
        }


SHAPE_PARAMETER_RANGES: Final[dict[NoseConeShape, tuple[float, float]]] = {
    # Secant ogive: multiple of the tangent-ogive radius. Above about 3 the
    # profile becomes strongly bulged and separation-prone (Crowell [1]).
    NoseConeShape.SECANT_OGIVE: (1.0, 3.0),
    # Parabolic series K, per Crowell [1].
    NoseConeShape.PARABOLIC: (0.0, 1.0),
    # Power series n. n -> 0 degenerates to a flat disc, so the lower bound is
    # held at 0.3; n = 1 is a cone.
    NoseConeShape.POWER: (0.3, 1.0),
}
"""Valid ``(low, high)`` bounds for :attr:`NoseCone.shape_parameter`."""


def default_shape_parameter(shape: NoseConeShape) -> float:
    """Return a sensible default shape parameter for a profile family.

    Parameters
    ----------
    shape:
        Profile family.

    Returns
    -------
    float
        ``1.0`` for families with no free parameter. For the parametric
        families, the value most commonly used in practice: a secant ogive at
        1.5 times the tangent radius, the full parabola (``K = 1``) and the
        1/2-power nose (``n = 0.5``).
    """
    return {
        NoseConeShape.SECANT_OGIVE: 1.5,
        NoseConeShape.PARABOLIC: 1.0,
        NoseConeShape.POWER: 0.5,
    }.get(shape, 1.0)


# Note: this dataclass deliberately does not use slots. Several of its
# properties integrate the profile over hundreds of stations, and the flight
# simulator evaluates them inside the integration loop. They are cached with
# functools.cached_property, which needs an instance __dict__. The class stays
# frozen, so caching derived values can never desynchronise from the geometry.
@dataclass(frozen=True)
class NoseCone:
    """A parametric nose cone.

    Attributes
    ----------
    shape:
        Profile family.
    length:
        Axial length from tip to base [m].
    base_radius:
        Radius at the base, matching the body tube outer radius [m].
    wall_thickness:
        Shell thickness [m] when :attr:`solid` is ``False``. Ignored for a
        solid (turned balsa or basswood) cone.
    material:
        Material the cone is made from.
    shape_parameter:
        Free parameter for the secant ogive, parabolic and power families;
        ignored otherwise. See :data:`SHAPE_PARAMETER_RANGES`.
    solid:
        ``True`` for a turned solid cone, ``False`` for a moulded or printed
        shell.
    shoulder_length:
        Length of the shoulder that plugs into the body tube [m]. Contributes
        mass but no external wetted area.
    shoulder_radius:
        Outside radius of that shoulder [m], which is the body tube's *inside*
        radius when the cone is to locate in the tube. Left at ``None`` it
        falls back to ``base_radius - wall_thickness``, the shoulder a moulded
        cone of this wall thickness naturally has.
        :func:`~rocketopt.geometry.rocket.build_rocket` sets it from the body
        tube it assembles the cone onto, so the mass model and the exported
        spigot are the same part.
    """

    shape: NoseConeShape
    length: float
    base_radius: float
    material: Material
    wall_thickness: float = 1.5e-3
    shape_parameter: float = 1.0
    solid: bool = False
    shoulder_length: float = 0.0
    shoulder_radius: float | None = None

    def __post_init__(self) -> None:
        """Validate the geometry."""
        if self.length <= 0.0:
            raise ValueError("nose cone length must be positive")
        if self.base_radius <= 0.0:
            raise ValueError("nose cone base radius must be positive")
        if self.shoulder_length < 0.0:
            raise ValueError("shoulder length must not be negative")
        if self.shoulder_radius is not None and not (
            0.0 < self.shoulder_radius <= self.base_radius
        ):
            raise ValueError(
                f"shoulder radius {self.shoulder_radius} m must be positive and "
                f"no larger than the base radius {self.base_radius} m"
            )
        if not self.solid:
            if self.wall_thickness <= 0.0:
                raise ValueError("hollow nose cone needs a positive wall thickness")
            if self.wall_thickness >= self.base_radius:
                raise ValueError(
                    f"wall thickness {self.wall_thickness} m exceeds the base "
                    f"radius {self.base_radius} m"
                )
        if self.shape.uses_shape_parameter:
            low, high = SHAPE_PARAMETER_RANGES[self.shape]
            if not low <= self.shape_parameter <= high:
                raise ValueError(
                    f"{self.shape.label} shape_parameter must lie in "
                    f"[{low}, {high}], got {self.shape_parameter}"
                )

    # -- Profile -------------------------------------------------------------

    def radius_at(self, x: NDArray[np.float64] | float) -> NDArray[np.float64]:
        """Profile radius at axial station(s) ``x`` measured aft from the tip.

        Parameters
        ----------
        x:
            Station(s) [m] in ``[0, length]``. Values outside are clamped.

        Returns
        -------
        numpy.ndarray
            Radius at each station [m].
        """
        xs = np.clip(np.atleast_1d(np.asarray(x, dtype=np.float64)), 0.0, self.length)
        f = xs / self.length
        R = self.base_radius
        L = self.length

        match self.shape:
            case NoseConeShape.CONICAL:
                y = R * f

            case NoseConeShape.TANGENT_OGIVE:
                rho = (R * R + L * L) / (2.0 * R)
                # y = sqrt(rho^2 - (L - x)^2) + R - rho   [1], Eq. 8
                inner = np.maximum(rho * rho - (L - xs) ** 2, 0.0)
                y = np.sqrt(inner) + R - rho

            case NoseConeShape.SECANT_OGIVE:
                # Circular arc of radius rho through the tip (0, 0) and the
                # base point (L, R). Rather than Crowell's rotated form [1],
                # construct the circle centre directly: it lies on the
                # perpendicular bisector of the chord joining those two points.
                # At rho = rho_tangent this reduces exactly to the tangent
                # ogive, which the unit tests assert.
                rho_tan = (R * R + L * L) / (2.0 * R)
                rho = rho_tan * self.shape_parameter
                chord = math.hypot(L, R)
                # Distance from the chord midpoint to the circle centre.
                half_chord_offset = math.sqrt(max(rho * rho - chord * chord / 4.0, 0.0))
                # Unit normal to the chord, directed below the profile.
                xc = L / 2.0 + half_chord_offset * (R / chord)
                yc = R / 2.0 - half_chord_offset * (L / chord)
                inner = np.maximum(rho * rho - (xs - xc) ** 2, 0.0)
                y = yc + np.sqrt(inner)

            case NoseConeShape.ELLIPTICAL:
                y = R * np.sqrt(np.maximum(1.0 - (1.0 - f) ** 2, 0.0))

            case NoseConeShape.PARABOLIC:
                K = self.shape_parameter
                y = R * (2.0 * f - K * f * f) / (2.0 - K)

            case NoseConeShape.POWER:
                y = R * np.power(f, self.shape_parameter)

            case NoseConeShape.VON_KARMAN | NoseConeShape.HAACK_LD:
                y = self._haack(f, C=0.0)

            case NoseConeShape.HAACK_LV:
                y = self._haack(f, C=1.0 / 3.0)

            case _:  # pragma: no cover - Enum is exhaustive
                raise ValueError(f"unhandled nose cone shape {self.shape}")

        # Every family is pointed at the tip, so the radius there is exactly
        # zero. The ogive forms reach it by subtracting two nearly equal large
        # numbers - ``sqrt(rho^2 - L^2) + R - rho`` - and leave a residue of a
        # few times 1e-14 m behind. That is meaningless as a radius but not as a
        # topology: revolved, a residual radius makes a ring of slivers at the
        # apex instead of a single point, so it is snapped away here where the
        # profile is defined rather than in each consumer.
        return np.where(xs <= 0.0, 0.0, np.maximum(y, 0.0))

    def _haack(self, f: NDArray[np.float64], *, C: float) -> NDArray[np.float64]:
        """Evaluate the Haack series profile.

        Parameters
        ----------
        f:
            Fractional station ``x / L``.
        C:
            Series constant; 0 for Von Karman / LD-Haack, 1/3 for LV-Haack.

        Returns
        -------
        numpy.ndarray
            Radius at each station [m]. Haack [3]; von Karman [4].
        """
        theta = np.arccos(np.clip(1.0 - 2.0 * f, -1.0, 1.0))
        inner = theta - np.sin(2.0 * theta) / 2.0 + C * np.sin(theta) ** 3
        return (self.base_radius / math.sqrt(math.pi)) * np.sqrt(
            np.maximum(inner, 0.0)
        )

    @cached_property
    def _cached_stations(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Profile sample arrays, computed once per instance."""
        x = np.linspace(0.0, self.length, _PROFILE_SAMPLES)
        return x, self.radius_at(x)

    def _stations(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(x, y)`` arrays sampling the profile for integration."""
        return self._cached_stations

    # -- Derived geometry ----------------------------------------------------

    @property
    def base_diameter(self) -> float:
        """Diameter at the base [m]."""
        return 2.0 * self.base_radius

    @property
    def shoulder_outer_radius(self) -> float:
        """Resolved outside radius of the shoulder [m].

        :attr:`shoulder_radius` when it was given, otherwise
        ``base_radius - wall_thickness``.
        """
        if self.shoulder_radius is not None:
            return self.shoulder_radius
        return max(self.base_radius - self.wall_thickness, 0.0)

    @property
    def shoulder_inner_radius(self) -> float:
        """Bore radius of the shoulder [m], zero when it is solid."""
        if self.solid:
            return 0.0
        return max(self.shoulder_outer_radius - self.wall_thickness, 0.0)

    @property
    def fineness_ratio(self) -> float:
        """Length divided by base diameter [-].

        The dominant parameter for nose pressure drag; below about 2 the
        pressure drag rises steeply, above about 5 skin friction dominates
        ([5], Sec. 3.4).
        """
        return self.length / self.base_diameter

    @cached_property
    def enclosed_volume(self) -> float:
        """Volume enclosed by the profile [m^3].

        Computed as the solid of revolution ``V = integral pi y^2 dx`` by the
        trapezoidal rule over :data:`_PROFILE_SAMPLES` stations.
        """
        x, y = self._stations()
        return float(np.trapezoid(math.pi * y * y, x))

    @cached_property
    def wetted_area(self) -> float:
        """External surface area exposed to the flow [m^2].

        Surface of revolution ``S = integral 2 pi y sqrt(1 + (dy/dx)^2) dx``.
        The base annulus is excluded because it mates against the body tube.
        """
        x, y = self._stations()
        dydx = np.gradient(y, x)
        return float(np.trapezoid(2.0 * math.pi * y * np.sqrt(1.0 + dydx**2), x))

    @cached_property
    def planform_area(self) -> float:
        """Side-view projected area [m^2], used by body normal-force models."""
        x, y = self._stations()
        return float(np.trapezoid(2.0 * y, x))

    @cached_property
    def material_volume(self) -> float:
        """Volume of material in the cone including its shoulder [m^3].

        A solid cone uses the full enclosed volume. A hollow cone uses a shell
        of :attr:`wall_thickness`, computed as the difference between the outer
        profile and an inner profile offset inward by the wall thickness. The
        shoulder is a tube of the same wall thickness, or solid stock on a
        turned cone.
        """
        if self.solid:
            shell = self.enclosed_volume
        else:
            x, y = self._stations()
            inner = np.maximum(y - self.wall_thickness, 0.0)
            shell = float(np.trapezoid(math.pi * (y * y - inner * inner), x))

        if self.shoulder_length > 0.0:
            shell += self._shoulder_volume

        return shell

    @property
    def _shoulder_volume(self) -> float:
        """Volume of material in the shoulder alone [m^3]."""
        r_out = self.shoulder_outer_radius
        r_in = self.shoulder_inner_radius
        return math.pi * (r_out**2 - r_in**2) * self.shoulder_length

    @property
    def mass(self) -> float:
        """Mass of the nose cone [kg]."""
        return self.material_volume * self.material.density

    @cached_property
    def centre_of_mass(self) -> float:
        """Centre of mass measured aft from the tip [m].

        For a solid cone this is the centroid of the solid of revolution. For a
        shell it is the centroid of the wall material, which sits noticeably
        further aft because the wall area grows with radius.
        """
        x, y = self._stations()
        if self.solid:
            dv = math.pi * y * y
        else:
            inner = np.maximum(y - self.wall_thickness, 0.0)
            dv = math.pi * (y * y - inner * inner)

        volume = float(np.trapezoid(dv, x))
        if volume <= 0.0:
            return self.length / 2.0
        cg_cone = float(np.trapezoid(dv * x, x)) / volume

        if self.shoulder_length <= 0.0:
            return cg_cone

        # Combine the cone with its shoulder, whose centroid sits aft of the base.
        v_shoulder = self._shoulder_volume
        cg_shoulder = self.length + self.shoulder_length / 2.0
        total = volume + v_shoulder
        return (cg_cone * volume + cg_shoulder * v_shoulder) / total

    # -- Aerodynamics --------------------------------------------------------

    @property
    def cn_alpha(self) -> float:
        """Normal-force coefficient slope per radian, referenced to base area.

        Slender-body theory gives exactly 2 for any closed nose shape at small
        angle of attack, independent of profile ([2], Eq. 3-5).
        """
        return 2.0

    @cached_property
    def centre_of_pressure(self) -> float:
        """Centre of pressure aft of the tip [m].

        From slender-body theory, ``X_cp = L - V / A_base`` ([2], Sec. 3.2).
        For a cone this evaluates to ``2L/3`` and for a tangent ogive to about
        ``0.466 L``, matching Barrowman's tabulated constants.
        """
        base_area = math.pi * self.base_radius**2
        return self.length - self.enclosed_volume / base_area

    @property
    def volume_coefficient(self) -> float:
        """Enclosed volume as a fraction of its circumscribing cylinder [-].

        A direct measure of usable internal payload volume; 1/3 for a cone,
        2/3 for an ellipse.
        """
        return self.enclosed_volume / (math.pi * self.base_radius**2 * self.length)

    @property
    def manufacturability(self) -> float:
        """Ease of fabrication, 0 (hard) to 1 (easy) [-].

        A weighted score combining the material's workability, the profile's
        geometric complexity and the fragility of its tip. Used as an
        objective term by the optimiser; it is a design heuristic, not a
        measured quantity.
        """
        shape_ease = {
            NoseConeShape.CONICAL: 1.00,  # a straight taper, trivial to turn
            NoseConeShape.POWER: 0.80,
            NoseConeShape.PARABOLIC: 0.78,
            NoseConeShape.ELLIPTICAL: 0.75,
            NoseConeShape.TANGENT_OGIVE: 0.70,
            NoseConeShape.SECANT_OGIVE: 0.60,
            NoseConeShape.VON_KARMAN: 0.55,
            NoseConeShape.HAACK_LD: 0.55,
            NoseConeShape.HAACK_LV: 0.55,
        }[self.shape]

        # Very slender cones are hard to turn without whipping in the lathe.
        slenderness_penalty = 1.0 if self.fineness_ratio <= 5.0 else 0.85
        tip_penalty = 0.9 if self.shape.has_blunt_tip else 1.0

        return float(
            np.clip(
                0.55 * shape_ease
                + 0.45 * self.material.machinability * slenderness_penalty * tip_penalty,
                0.0,
                1.0,
            )
        )

    def profile_points(self, n: int = 200) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(x, y)`` points describing the outer profile.

        Suitable for CAD revolution, SVG export and the 3D viewport.

        Parameters
        ----------
        n:
            Number of points.

        Returns
        -------
        tuple of numpy.ndarray
            Axial stations [m] and radii [m], from tip to base.
        """
        if n < 2:
            raise ValueError("n must be at least 2")
        x = np.linspace(0.0, self.length, n)
        return x, self.radius_at(x)

    @classmethod
    def from_names(
        cls,
        shape: str,
        length: float,
        base_radius: float,
        material: str,
        **kwargs: float | bool,
    ) -> NoseCone:
        """Construct from string names, for CLI and config-file use.

        Parameters
        ----------
        shape:
            Profile name matching a :class:`NoseConeShape` value.
        length:
            Length [m].
        base_radius:
            Base radius [m].
        material:
            Material name resolved by
            :func:`rocketopt.structures.materials.get_material`.
        **kwargs:
            Forwarded to the constructor.

        Returns
        -------
        NoseCone
            The constructed nose cone.
        """
        key = shape.strip().lower().replace(" ", "_").replace("-", "_")
        try:
            profile = NoseConeShape(key)
        except ValueError:
            raise ValueError(
                f"Unknown nose cone shape {shape!r}. Available: "
                f"{[s.value for s in NoseConeShape]}"
            ) from None
        return cls(
            shape=profile,
            length=length,
            base_radius=base_radius,
            material=get_material(material),
            **kwargs,  # type: ignore[arg-type]
        )
