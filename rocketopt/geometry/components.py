"""Airframe components: body tubes, transitions, fin sets and small parts.

Every component exposes a common set of properties so the assembly in
:mod:`rocketopt.geometry.rocket` can treat them uniformly:

``mass``
    Component mass [kg].
``centre_of_mass``
    Centroid measured aft of the component's own forward face [m].
``wetted_area``
    External area exposed to the flow [m^2].
``cn_alpha`` / ``centre_of_pressure``
    Barrowman normal-force slope and its application point, where the
    component generates normal force.

Fin cross-sections are defined in :mod:`rocketopt.geometry.fin_sections`, which
this module re-exports :class:`FinAirfoil` from and defers to for the section
area used by :attr:`FinSet.volume_single`.

Barrowman fin equations
-----------------------
For ``N`` fins of exposed semi-span ``s`` mounted on a body of radius ``r``,
with root chord ``c_r``, tip chord ``c_t`` and leading-edge sweep distance
``x_r``:

Normal-force slope, referenced to the body cross-sectional area::

    CN_alpha_fins = 4 N (s / d)^2
                    / (1 + sqrt(1 + (2 l / (c_r + c_t))^2))

where ``d = 2r`` and ``l`` is the mid-chord line length,
``l = sqrt(s^2 + (x_r + c_t/2 - c_r/2)^2)``.

Body-to-fin interference is accounted for by the factor::

    K_fb = 1 + r / (s + r)

Fin centre of pressure aft of the root leading edge::

    X_f = (x_r / 3)(c_r + 2 c_t)/(c_r + c_t)
          + (1/6)[(c_r + c_t) - c_r c_t / (c_r + c_t)]

References
----------
[1] Barrowman, J. S. (1967). *The Practical Calculation of the Aerodynamic
    Characteristics of Slender Finned Vehicles*. M.Sc. thesis, Catholic
    University of America.
[2] Barrowman, J. S., & Barrowman, J. A. (1966). *A Method for Calculating the
    Centre of Pressure of Model Rockets*. NARAM-8 technical report.
[3] Niskanen, S. (2013). *OpenRocket Technical Documentation*, Sec. 3.2.
[4] Hoerner, S. F. (1965). *Fluid-Dynamic Drag*, Chs. 3 and 6.
[5] Knacke, T. W. (1992). *Parachute Recovery Systems Design Manual*,
    NWC TP 6575.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Final

import numpy as np

from rocketopt.geometry.fin_sections import FinAirfoil, section_outline, swept_volume
from rocketopt.structures.materials import Material
from rocketopt.utils.constants import (
    CD_PARACHUTE_FLAT_SHEET,
    CD_PARACHUTE_HEMISPHERICAL,
    CD_STREAMER,
)

__all__ = [
    "MIN_MOTOR_TUBE_WALL",
    "MOTOR_FIT_CLEARANCE",
    "FinAirfoil",
    "RecoveryType",
    "BodyTube",
    "Transition",
    "FinSet",
    "LaunchLug",
    "MotorMount",
    "RecoveryDevice",
]


class RecoveryType(str, Enum):
    """Recovery device family."""

    PARACHUTE_FLAT = "parachute_flat"
    PARACHUTE_DOME = "parachute_dome"
    STREAMER = "streamer"
    TUMBLE = "tumble"

    @property
    def label(self) -> str:
        """Human-readable name."""
        return self.name.replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class BodyTube:
    """A cylindrical airframe section.

    Attributes
    ----------
    length:
        Axial length [m].
    outer_radius:
        Outside radius [m].
    wall_thickness:
        Wall thickness [m].
    material:
        Tube material.
    """

    length: float
    outer_radius: float
    wall_thickness: float
    material: Material

    def __post_init__(self) -> None:
        """Validate the tube geometry."""
        if self.length <= 0.0:
            raise ValueError("body tube length must be positive")
        if self.outer_radius <= 0.0:
            raise ValueError("body tube outer radius must be positive")
        if not 0.0 < self.wall_thickness < self.outer_radius:
            raise ValueError(
                f"wall thickness {self.wall_thickness} m must be positive and "
                f"less than the outer radius {self.outer_radius} m"
            )

    @property
    def inner_radius(self) -> float:
        """Inside radius [m]."""
        return self.outer_radius - self.wall_thickness

    @property
    def diameter(self) -> float:
        """Outside diameter [m]."""
        return 2.0 * self.outer_radius

    @property
    def reference_area(self) -> float:
        """Cross-sectional area based on the outside diameter [m^2].

        This is the reference area for every aerodynamic coefficient in
        RocketOpt, following Barrowman's convention [1].
        """
        return math.pi * self.outer_radius**2

    @property
    def internal_volume(self) -> float:
        """Usable internal volume [m^3]."""
        return math.pi * self.inner_radius**2 * self.length

    @property
    def material_volume(self) -> float:
        """Volume of tube wall material [m^3]."""
        return math.pi * (self.outer_radius**2 - self.inner_radius**2) * self.length

    @property
    def mass(self) -> float:
        """Tube mass [kg]."""
        return self.material_volume * self.material.density

    @property
    def centre_of_mass(self) -> float:
        """Centroid aft of the forward face [m]. A uniform tube balances at mid-length."""
        return self.length / 2.0

    @property
    def wetted_area(self) -> float:
        """External cylindrical surface area [m^2]."""
        return 2.0 * math.pi * self.outer_radius * self.length

    @property
    def planform_area(self) -> float:
        """Side-view projected area [m^2]."""
        return self.diameter * self.length

    @property
    def cn_alpha(self) -> float:
        """Normal-force slope [1/rad].

        A constant-diameter cylinder produces no normal force in Barrowman's
        potential-flow formulation [1]; body lift enters only through the
        viscous cross-flow correction applied in
        :mod:`rocketopt.aerodynamics.barrowman`.
        """
        return 0.0

    @property
    def centre_of_pressure(self) -> float:
        """Centre of pressure aft of the forward face [m].

        Undefined for a component with zero normal force; the centroid is
        returned so that weighted sums remain well behaved.
        """
        return self.length / 2.0


@dataclass(frozen=True, slots=True)
class Transition:
    """A conical shoulder or boat-tail joining two different diameters.

    A reduction in diameter aft (``aft_radius < fore_radius``) is a boat-tail,
    which recovers base drag. An increase is a shoulder.

    Attributes
    ----------
    length:
        Axial length [m].
    fore_radius:
        Radius at the forward face [m].
    aft_radius:
        Radius at the aft face [m].
    wall_thickness:
        Wall thickness [m].
    material:
        Material.
    """

    length: float
    fore_radius: float
    aft_radius: float
    wall_thickness: float
    material: Material

    def __post_init__(self) -> None:
        """Validate the transition geometry."""
        if self.length <= 0.0:
            raise ValueError("transition length must be positive")
        if self.fore_radius <= 0.0 or self.aft_radius <= 0.0:
            raise ValueError("transition radii must be positive")
        if math.isclose(self.fore_radius, self.aft_radius):
            raise ValueError(
                "a transition must change diameter; use a BodyTube for a "
                "constant-diameter section"
            )
        if self.wall_thickness <= 0.0:
            raise ValueError("transition wall thickness must be positive")

    @property
    def is_boat_tail(self) -> bool:
        """Whether the diameter reduces going aft."""
        return self.aft_radius < self.fore_radius

    @property
    def half_angle(self) -> float:
        """Half-angle of the cone [rad].

        Boat-tails steeper than about 12 degrees separate and stop recovering
        base drag ([4], Ch. 3).
        """
        return math.atan2(abs(self.aft_radius - self.fore_radius), self.length)

    @property
    def enclosed_volume(self) -> float:
        """Volume of the truncated cone [m^3]."""
        r1, r2 = self.fore_radius, self.aft_radius
        return math.pi * self.length * (r1 * r1 + r1 * r2 + r2 * r2) / 3.0

    @property
    def material_volume(self) -> float:
        """Volume of wall material [m^3], as a conical shell."""
        t = self.wall_thickness
        r1i = max(self.fore_radius - t, 0.0)
        r2i = max(self.aft_radius - t, 0.0)
        inner = math.pi * self.length * (r1i * r1i + r1i * r2i + r2i * r2i) / 3.0
        return self.enclosed_volume - inner

    @property
    def mass(self) -> float:
        """Transition mass [kg]."""
        return self.material_volume * self.material.density

    @property
    def centre_of_mass(self) -> float:
        """Centroid of the conical shell aft of its forward face [m]."""
        r1, r2 = self.fore_radius, self.aft_radius
        # Centroid of a truncated cone measured from the r1 face.
        numerator = r1 * r1 + 2.0 * r1 * r2 + 3.0 * r2 * r2
        denominator = 4.0 * (r1 * r1 + r1 * r2 + r2 * r2)
        return self.length * numerator / denominator

    @property
    def wetted_area(self) -> float:
        """Lateral surface area of the frustum [m^2]."""
        slant = math.hypot(self.length, self.aft_radius - self.fore_radius)
        return math.pi * (self.fore_radius + self.aft_radius) * slant

    @property
    def planform_area(self) -> float:
        """Side-view projected area [m^2]."""
        return (self.fore_radius + self.aft_radius) * self.length

    def cn_alpha(self, reference_radius: float) -> float:
        """Normal-force slope [1/rad] referenced to ``reference_radius``.

        Barrowman [1], Eq. 3-8: a change in cross-sectional area generates
        normal force proportional to that change::

            CN_alpha = 2 [(r_aft / r_ref)^2 - (r_fore / r_ref)^2]

        A boat-tail therefore contributes a *negative* slope, which moves the
        overall centre of pressure forward - the reason an aggressive boat-tail
        can destabilise an otherwise sound design.

        Parameters
        ----------
        reference_radius:
            Radius defining the reference area [m].

        Returns
        -------
        float
            Normal-force slope [1/rad].
        """
        ref = reference_radius
        return 2.0 * ((self.aft_radius / ref) ** 2 - (self.fore_radius / ref) ** 2)

    @property
    def centre_of_pressure(self) -> float:
        """Centre of pressure aft of the forward face [m].

        Barrowman [1], Eq. 3-9, for a conical transition::

            X = (L / 3) [1 + (1 - r_f/r_a) / (1 - (r_f/r_a)^2)]
        """
        ratio = self.fore_radius / self.aft_radius
        if math.isclose(ratio, 1.0):
            return self.length / 2.0
        return (self.length / 3.0) * (1.0 + (1.0 - ratio) / (1.0 - ratio * ratio))


@dataclass(frozen=True, slots=True)
class FinSet:
    """A set of identical trapezoidal fins equally spaced around the body.

    Attributes
    ----------
    count:
        Number of fins, 3 to 5.
    root_chord:
        Chord at the body wall [m].
    tip_chord:
        Chord at the tip [m]. Zero gives a delta fin.
    span:
        Exposed semi-span, body wall to tip [m].
    sweep_length:
        Axial distance from the root leading edge to the tip leading edge [m].
    thickness:
        Fin material thickness [m].
    material:
        Fin material.
    body_radius:
        Radius of the body the fins are mounted on [m].
    airfoil:
        Cross-section profile.
    cant_angle:
        Fin cant about the radial axis [rad]. A small cant induces roll, which
        averages out thrust misalignment at the cost of a little drag.
    fillet_radius:
        Radius of the root fillet [m]. Reduces interference drag and is the
        single most effective way to strengthen the fin-body joint.
    position:
        Axial station of the root leading edge, measured aft of the nose tip
        [m]. Set by the assembly, not by the optimiser directly.
    """

    count: int
    root_chord: float
    tip_chord: float
    span: float
    sweep_length: float
    thickness: float
    material: Material
    body_radius: float
    airfoil: FinAirfoil = FinAirfoil.ROUNDED
    cant_angle: float = 0.0
    fillet_radius: float = 0.0
    position: float = 0.0

    def __post_init__(self) -> None:
        """Validate the fin geometry."""
        if not 3 <= self.count <= 5:
            raise ValueError(f"fin count must be 3, 4 or 5; got {self.count}")
        if self.root_chord <= 0.0:
            raise ValueError("root chord must be positive")
        if self.tip_chord < 0.0:
            raise ValueError("tip chord must not be negative")
        if self.span <= 0.0:
            raise ValueError("fin span must be positive")
        if self.thickness <= 0.0:
            raise ValueError("fin thickness must be positive")
        if self.body_radius <= 0.0:
            raise ValueError("body radius must be positive")
        if self.fillet_radius < 0.0:
            raise ValueError("fillet radius must not be negative")
        if abs(self.cant_angle) > math.radians(15.0):
            raise ValueError("cant angle above 15 degrees is not a sane design")

    # -- Planform ------------------------------------------------------------

    @property
    def taper_ratio(self) -> float:
        """Tip chord divided by root chord [-]."""
        return self.tip_chord / self.root_chord

    @property
    def mean_chord(self) -> float:
        """Mean geometric chord [m]."""
        return 0.5 * (self.root_chord + self.tip_chord)

    @property
    def area_single(self) -> float:
        """Exposed planform area of one fin [m^2]."""
        return self.mean_chord * self.span

    @property
    def area_total(self) -> float:
        """Combined exposed planform area of all fins [m^2]."""
        return self.area_single * self.count

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio of one exposed fin panel [-].

        Defined as ``span^2 / area``, the convention used by the flutter
        correlation in NACA TN 4197 and by OpenRocket [3].
        """
        return self.span**2 / self.area_single

    @property
    def mid_chord_length(self) -> float:
        """Length of the mid-chord line from root to tip [m].

        The ``l`` appearing in the Barrowman fin normal-force equation [1].
        """
        mid_chord_sweep = self.sweep_length + 0.5 * (self.tip_chord - self.root_chord)
        return math.hypot(self.span, mid_chord_sweep)

    @property
    def leading_edge_sweep_angle(self) -> float:
        """Leading-edge sweep angle from the radial direction [rad]."""
        return math.atan2(self.sweep_length, self.span)

    @property
    def thickness_ratio(self) -> float:
        """Thickness divided by mean chord [-]. Drives flutter and wave drag."""
        return self.thickness / self.mean_chord

    @property
    def total_span(self) -> float:
        """Tip-to-tip span across the body [m]."""
        return 2.0 * (self.span + self.body_radius)

    # -- Mass ----------------------------------------------------------------

    @property
    def fillet_volume_single(self) -> float:
        """Volume of adhesive in one fin's pair of root fillets [m^3].

        A fillet of radius ``R`` has a cross-sectional area of
        ``(1 - pi/4) R^2`` - the square corner minus the quarter circle - and
        runs the length of the root chord on both sides of the fin.
        """
        if self.fillet_radius <= 0.0:
            return 0.0
        area = (1.0 - math.pi / 4.0) * self.fillet_radius**2
        return 2.0 * area * self.root_chord

    @property
    def volume_single(self) -> float:
        """Material volume of one fin [m^3], excluding fillets.

        Obtained by integrating the true section area over the span with
        :func:`~rocketopt.geometry.fin_sections.swept_volume`, so the reported
        fin mass is the mass of the section that
        :func:`~rocketopt.cad.parts.fin_mesh` exports - a square section fills
        its bounding box, a rounded one loses the two edge corners, the NACA
        aerofoil encloses about 68% of it and the double wedge exactly half.
        """
        return swept_volume(
            self.airfoil,
            root_chord=self.root_chord,
            tip_chord=self.tip_chord,
            span=self.span,
            thickness=self.thickness,
        )

    def section_outline(
        self, span_fraction: float = 0.0, *, samples: int = 61
    ) -> list[tuple[float, float]]:
        """Return the fin's cross-section outline at a spanwise station.

        Parameters
        ----------
        span_fraction:
            Fraction of the exposed span, 0 at the root and 1 at the tip.
        samples:
            Number of chordwise stations in the outline.

        Returns
        -------
        list of tuple
            ``(x, y)`` vertices [m] of the closed section, ``x`` aft of the
            local leading edge and ``y`` either side of the chord line.

        Raises
        ------
        ValueError
            If ``span_fraction`` is outside ``[0, 1]``.
        """
        if not 0.0 <= span_fraction <= 1.0:
            raise ValueError("span_fraction must lie in [0, 1]")
        chord = self.root_chord + (self.tip_chord - self.root_chord) * span_fraction
        return section_outline(
            self.airfoil,
            chord=chord,
            thickness=self.thickness,
            samples=samples,
        )

    @property
    def mass(self) -> float:
        """Combined mass of all fins [kg], excluding fillet adhesive."""
        return self.volume_single * self.material.density * self.count

    @property
    def centre_of_mass(self) -> float:
        """Centroid aft of the root leading edge [m].

        Area centroid of the trapezoidal planform, projected onto the axis.
        """
        c_r, c_t, sweep = self.root_chord, self.tip_chord, self.sweep_length
        denom = c_r + c_t
        if denom <= 0.0:
            return c_r / 2.0
        # Spanwise centroid position as a fraction of span.
        eta = (c_r + 2.0 * c_t) / (3.0 * denom)
        # Chordwise centroid of the local chord at that station.
        local_chord = c_r + (c_t - c_r) * eta
        return sweep * eta + local_chord / 2.0

    @property
    def wetted_area(self) -> float:
        """Total wetted area of all fins [m^2], both faces plus edges."""
        faces = 2.0 * self.area_total
        edge_length = self.span * 2.0 + self.tip_chord
        edges = edge_length * self.thickness * self.count
        return faces + edges

    # -- Aerodynamics --------------------------------------------------------

    @property
    def interference_factor(self) -> float:
        """Body-to-fin interference factor ``K_fb`` [-].

        Barrowman [1], Eq. 3-20: ``K_fb = 1 + r / (s + r)``. Flow accelerating
        around the body increases the effective fin loading.
        """
        return 1.0 + self.body_radius / (self.span + self.body_radius)

    def cn_alpha(self, reference_radius: float) -> float:
        """Normal-force slope of the fin set [1/rad].

        Barrowman [1], Eq. 3-18, with the interference factor applied::

            CN_alpha = K_fb * 4 N (s/d)^2
                       / (1 + sqrt(1 + (2 l / (c_r + c_t))^2))

        The formula is a subsonic, potential-flow result. Above about Mach 0.8
        the Prandtl-Glauert correction applied in
        :mod:`rocketopt.aerodynamics.barrowman` extends it.

        Parameters
        ----------
        reference_radius:
            Radius defining the reference area [m].

        Returns
        -------
        float
            Normal-force slope referenced to the body cross-section [1/rad].
        """
        d = 2.0 * reference_radius
        chord_sum = self.root_chord + self.tip_chord
        if chord_sum <= 0.0:
            return 0.0
        ratio = 2.0 * self.mid_chord_length / chord_sum
        base = 4.0 * self.count * (self.span / d) ** 2 / (1.0 + math.sqrt(1.0 + ratio**2))
        return self.interference_factor * base

    @property
    def centre_of_pressure(self) -> float:
        """Centre of pressure aft of the root leading edge [m].

        Barrowman [1], Eq. 3-19::

            X_f = (x_r/3)(c_r + 2 c_t)/(c_r + c_t)
                  + (1/6)[(c_r + c_t) - c_r c_t/(c_r + c_t)]
        """
        c_r, c_t, x_r = self.root_chord, self.tip_chord, self.sweep_length
        denom = c_r + c_t
        if denom <= 0.0:
            return c_r / 2.0
        term1 = (x_r / 3.0) * (c_r + 2.0 * c_t) / denom
        term2 = (1.0 / 6.0) * (denom - c_r * c_t / denom)
        return term1 + term2

    @property
    def absolute_centre_of_pressure(self) -> float:
        """Fin centre of pressure aft of the nose tip [m]."""
        return self.position + self.centre_of_pressure

    def profile_drag_factor(self) -> float:
        """Section drag multiplier relative to a square-edged plate [-].

        Hoerner [4], Ch. 6, Table 6-1. A rounded leading edge with a sharp
        trailing edge roughly halves fin profile drag compared with a
        square-cut plate, which is why it is worth the extra sanding.
        """
        return {
            FinAirfoil.SQUARE: 1.00,
            FinAirfoil.ROUNDED: 0.70,
            FinAirfoil.AIRFOIL: 0.50,
            FinAirfoil.DOUBLE_WEDGE: 0.55,
        }[self.airfoil]

    @property
    def manufacturability(self) -> float:
        """Ease of fabrication, 0 to 1 [-].

        Penalises thin stock, many fins, aggressive taper, cant and the
        harder-to-shape sections. A design heuristic used by the optimiser,
        not a measured quantity.
        """
        airfoil_ease = {
            FinAirfoil.SQUARE: 1.00,
            FinAirfoil.ROUNDED: 0.85,
            FinAirfoil.AIRFOIL: 0.55,
            FinAirfoil.DOUBLE_WEDGE: 0.60,
        }[self.airfoil]

        count_ease = {3: 1.00, 4: 0.95, 5: 0.80}[self.count]
        # Stock thinner than 1.5 mm is fragile and hard to sand square.
        thin_penalty = min(1.0, self.thickness / 1.5e-3)
        cant_penalty = 1.0 if abs(self.cant_angle) < 1e-6 else 0.80
        # A very sharp taper leaves a fragile tip.
        taper_penalty = 0.85 if self.taper_ratio < 0.15 else 1.0

        return float(
            np.clip(
                0.30 * airfoil_ease
                + 0.20 * count_ease
                + 0.20 * self.material.machinability
                + 0.30 * thin_penalty * cant_penalty * taper_penalty,
                0.0,
                1.0,
            )
        )


@dataclass(frozen=True, slots=True)
class LaunchLug:
    """A tube that guides the rocket along the launch rod.

    Attributes
    ----------
    length:
        Lug length [m].
    outer_radius:
        Outside radius [m].
    inner_radius:
        Bore radius, which must clear the launch rod [m].
    material:
        Lug material.
    position:
        Axial station of the forward end, aft of the nose tip [m].
    """

    length: float
    outer_radius: float
    inner_radius: float
    material: Material
    position: float = 0.0

    def __post_init__(self) -> None:
        """Validate the lug geometry."""
        if self.length <= 0.0:
            raise ValueError("launch lug length must be positive")
        if not 0.0 < self.inner_radius < self.outer_radius:
            raise ValueError("lug inner radius must be positive and below the outer")

    @property
    def mass(self) -> float:
        """Lug mass [kg]."""
        volume = (
            math.pi * (self.outer_radius**2 - self.inner_radius**2) * self.length
        )
        return volume * self.material.density

    @property
    def centre_of_mass(self) -> float:
        """Centroid aft of the lug's forward end [m]."""
        return self.length / 2.0

    @property
    def wetted_area(self) -> float:
        """External wetted area [m^2]."""
        return 2.0 * math.pi * self.outer_radius * self.length

    @property
    def frontal_area(self) -> float:
        """Projected frontal area [m^2], which sets its parasitic drag."""
        return math.pi * self.outer_radius**2


MOTOR_FIT_CLEARANCE: Final[float] = 0.8e-3
"""Diametral clearance between the motor case and the mount bore [m].

0.8 mm on diameter - 0.4 mm radial - is the usual fit for an Estes case in a
paper motor tube: loose enough to load a motor with cold fingers, tight enough
that the case cannot rattle sideways under thrust.
"""

MIN_MOTOR_TUBE_WALL: Final[float] = 0.4e-3
"""Thinnest motor tube wall a mount is allowed to be sized to [m]."""


@dataclass(frozen=True, slots=True)
class MotorMount:
    """The motor tube, thrust ring and centring rings.

    Attributes
    ----------
    inner_diameter:
        Bore that accepts the motor case [m].
    length:
        Motor tube length [m].
    wall_thickness:
        Motor tube wall thickness [m].
    material:
        Motor tube material.
    centring_ring_count:
        Number of centring rings. Zero when the motor tube *is* the airframe.
    body_inner_radius:
        Inside radius of the airframe the rings bridge to [m].
    position:
        Axial station of the motor tube's forward end, aft of the nose tip [m].
    """

    inner_diameter: float
    length: float
    wall_thickness: float
    material: Material
    body_inner_radius: float
    centring_ring_count: int = 2
    position: float = 0.0

    def __post_init__(self) -> None:
        """Validate the motor mount geometry."""
        if self.inner_diameter <= 0.0 or self.length <= 0.0:
            raise ValueError("motor mount dimensions must be positive")
        if self.wall_thickness <= 0.0:
            raise ValueError("motor tube wall thickness must be positive")
        if self.centring_ring_count < 0:
            raise ValueError("centring ring count must not be negative")

    @classmethod
    def for_airframe(
        cls,
        *,
        motor_diameter: float,
        motor_length: float,
        body_inner_radius: float,
        material: Material,
        length: float | None = None,
        position: float = 0.0,
        bore_clearance: float = MOTOR_FIT_CLEARANCE,
    ) -> MotorMount:
        """Size a self-centring motor mount for an airframe and a motor.

        The mount is a plain tube that fills the airframe bore: its outside
        diameter *is* the body tube's inside diameter and its bore is the motor
        case diameter plus a loading clearance. Because it contacts the bore
        along its whole length it locates the motor concentrically on its own,
        so no centring rings are fitted, and it is a single part that can be
        printed and pushed straight into the tube.

        Parameters
        ----------
        motor_diameter:
            Motor case outside diameter [m].
        motor_length:
            Motor case length [m], used as the default mount length.
        body_inner_radius:
            Inside radius of the body tube the mount sits in [m].
        material:
            Mount material.
        length:
            Mount length [m]. Defaults to the motor casing length, which
            supports the case over its full length and lets the aft face carry
            the thrust ring.
        position:
            Axial station of the mount's forward end, aft of the nose tip [m].
        bore_clearance:
            Diametral clearance between the case and the bore [m].

        Returns
        -------
        MotorMount
            A mount whose outside diameter equals the airframe bore.

        Raises
        ------
        ValueError
            If the motor plus the thinnest sensible wall will not fit inside
            the body tube, which means the design needs minimum-diameter
            construction rather than a mount.
        """
        bore = motor_diameter + bore_clearance
        wall = body_inner_radius - bore / 2.0
        if wall < MIN_MOTOR_TUBE_WALL:
            raise ValueError(
                f"a {motor_diameter * 1e3:.1f} mm motor needs a bore of "
                f"{bore * 1e3:.1f} mm, which leaves only {wall * 1e3:.2f} mm of "
                f"wall inside a {body_inner_radius * 2e3:.1f} mm airframe bore; "
                f"use a larger body tube or minimum-diameter construction"
            )
        return cls(
            inner_diameter=bore,
            length=motor_length if length is None else length,
            wall_thickness=wall,
            material=material,
            body_inner_radius=body_inner_radius,
            centring_ring_count=0,
            position=position,
        )

    @property
    def outer_radius(self) -> float:
        """Motor tube outside radius [m]."""
        return self.inner_diameter / 2.0 + self.wall_thickness

    @property
    def outer_diameter(self) -> float:
        """Motor tube outside diameter [m]."""
        return 2.0 * self.outer_radius

    @property
    def fits_airframe(self) -> bool:
        """Whether the mount will go inside the airframe bore it is sized to."""
        return self.outer_radius <= self.body_inner_radius + 1e-9

    @property
    def is_minimum_diameter(self) -> bool:
        """Whether the motor tube doubles as the airframe.

        True when the airframe bore is within a millimetre of the motor tube
        outside diameter, in which case no centring rings are needed.
        """
        return abs(self.body_inner_radius - self.outer_radius) < 1.0e-3

    @property
    def tube_mass(self) -> float:
        """Mass of the motor tube [kg]."""
        r_o = self.outer_radius
        r_i = self.inner_diameter / 2.0
        return math.pi * (r_o**2 - r_i**2) * self.length * self.material.density

    @property
    def ring_mass(self) -> float:
        """Combined mass of the centring rings [kg].

        Rings are modelled as annuli of the same thickness as the motor tube
        wall, bridging the motor tube to the airframe bore.
        """
        if self.centring_ring_count == 0 or self.is_minimum_diameter:
            return 0.0
        annulus = math.pi * (self.body_inner_radius**2 - self.outer_radius**2)
        if annulus <= 0.0:
            return 0.0
        volume = annulus * self.wall_thickness * self.centring_ring_count
        return volume * self.material.density

    @property
    def mass(self) -> float:
        """Total motor mount mass [kg]."""
        return self.tube_mass + self.ring_mass

    @property
    def centre_of_mass(self) -> float:
        """Centroid aft of the motor tube's forward end [m]."""
        return self.length / 2.0


@dataclass(frozen=True, slots=True)
class RecoveryDevice:
    """A parachute, streamer or tumble recovery configuration.

    Attributes
    ----------
    kind:
        Device family.
    diameter:
        Constructed (flat) canopy diameter [m]. For a streamer this is
        ignored; use :attr:`streamer_length` and :attr:`streamer_width`.
    areal_density:
        Mass per unit area of the canopy material [kg/m^2]. Ripstop nylon runs
        about 0.045 kg/m^2; 1.5 mil mylar about 0.035 kg/m^2.
    shroud_line_count:
        Number of shroud lines.
    shroud_line_length:
        Length of each shroud line [m].
    shroud_line_density:
        Mass per unit length of the shroud line [kg/m].
    deployment_altitude_fraction:
        Fraction of apogee altitude at which a second-stage device deploys.
        ``1.0`` deploys at apogee.
    streamer_length:
        Streamer length [m], used when ``kind`` is ``STREAMER``.
    streamer_width:
        Streamer width [m], used when ``kind`` is ``STREAMER``.
    """

    kind: RecoveryType
    diameter: float = 0.0
    areal_density: float = 0.045
    shroud_line_count: int = 6
    shroud_line_length: float = 0.0
    shroud_line_density: float = 1.5e-4
    deployment_altitude_fraction: float = 1.0
    streamer_length: float = 0.0
    streamer_width: float = 0.0

    def __post_init__(self) -> None:
        """Validate the recovery device."""
        if self.kind in {RecoveryType.PARACHUTE_FLAT, RecoveryType.PARACHUTE_DOME}:
            if self.diameter <= 0.0:
                raise ValueError("parachute diameter must be positive")
        if self.kind is RecoveryType.STREAMER:
            if self.streamer_length <= 0.0 or self.streamer_width <= 0.0:
                raise ValueError("streamer length and width must be positive")
        if self.areal_density <= 0.0:
            raise ValueError("areal density must be positive")

    @property
    def drag_coefficient(self) -> float:
        """Drag coefficient referenced to :attr:`reference_area` [-].

        Values from Knacke [5], Table 5-1, for a flat circular sheet and a
        hemispherical canopy; the streamer value is the empirical figure used
        in model rocketry.
        """
        return {
            RecoveryType.PARACHUTE_FLAT: CD_PARACHUTE_FLAT_SHEET,
            RecoveryType.PARACHUTE_DOME: CD_PARACHUTE_HEMISPHERICAL,
            RecoveryType.STREAMER: CD_STREAMER,
            # A tumbling rocket presents roughly its own side area; the
            # coefficient is applied to the airframe reference area by the
            # caller.
            RecoveryType.TUMBLE: 1.2,
        }[self.kind]

    @property
    def reference_area(self) -> float:
        """Area the drag coefficient is referenced to [m^2].

        For a canopy this is the constructed flat area; for a streamer, the
        planform area. Returns zero for tumble recovery, where the caller
        substitutes the airframe area.
        """
        if self.kind in {RecoveryType.PARACHUTE_FLAT, RecoveryType.PARACHUTE_DOME}:
            return math.pi * (self.diameter / 2.0) ** 2
        if self.kind is RecoveryType.STREAMER:
            return self.streamer_length * self.streamer_width
        return 0.0

    @property
    def drag_area(self) -> float:
        """Product of drag coefficient and reference area, ``Cd*S`` [m^2].

        This is the quantity that actually sets descent rate, so it is worth
        having directly.
        """
        return self.drag_coefficient * self.reference_area

    @property
    def mass(self) -> float:
        """Mass of canopy plus shroud lines [kg]."""
        canopy = self.reference_area * self.areal_density
        lines = (
            self.shroud_line_count * self.shroud_line_length * self.shroud_line_density
        )
        return canopy + lines

    def descent_rate(self, mass: float, air_density: float) -> float:
        """Steady-state descent rate under this device [m/s].

        Equating weight and drag, ``m g = 0.5 rho V^2 Cd S``, gives
        ``V = sqrt(2 m g / (rho Cd S))``. Knacke [5], Eq. 5-1.

        Parameters
        ----------
        mass:
            Descending mass [kg].
        air_density:
            Local air density [kg/m^3].

        Returns
        -------
        float
            Terminal descent velocity [m/s].

        Raises
        ------
        ValueError
            If the device has no drag area, which happens for tumble recovery
            where the airframe area must be supplied by the caller instead.
        """
        from rocketopt.utils.constants import G0

        if self.drag_area <= 0.0:
            raise ValueError(
                f"{self.kind.label} has no intrinsic drag area; supply the "
                f"airframe reference area explicitly"
            )
        return math.sqrt(2.0 * mass * G0 / (air_density * self.drag_area))
