"""Printable solid geometry for every component of a design.

This is where a :class:`~rocketopt.geometry.rocket.Rocket` becomes a set of
parts a slicer will accept. Each part is built as a single revolve or loft of a
closed profile - see :mod:`rocketopt.cad.mesh` - so no boolean operations are
involved and each part is a closed manifold by construction.

Interfaces
----------
Three fits decide whether the printed parts go together, and all three are
derived from the design rather than assumed:

Nose cone to body tube
    The cone carries an integrated locating spigot whose nominal outside
    diameter is the body tube's *inside* diameter, so it plugs straight into the
    tube. Its length is the design's shoulder length, or a default of one body
    diameter clamped into 20-40 mm when the design does not specify one: long
    enough that the joint carries the ejection load in shear rather than in
    peel, short enough not to eat the parachute bay.
Motor mount to body tube
    The mount's outside diameter is the body tube's inside diameter, so it
    locates itself concentrically along its whole length and needs no centring
    rings.
Motor to motor mount
    The mount's bore is the motor case diameter plus a loading clearance, and
    its length is the motor casing length, so the case is supported end to end
    and seats against a thrust ring at the aft face.

Print clearances
----------------
The *nominal* diameters above are exact matches, which would print as an
interference fit: a printer that holds +/-0.1 mm cannot make two nominally equal
diameters slide together. Each mating diameter therefore carries a small
clearance from :class:`PrintTolerances`, taken off the part that goes *inside*
so the airframe keeps its aerodynamic dimensions. The defaults suit a
well-tuned FDM machine printing PLA or PETG at 0.2 mm layers; a machine with
more elephant's foot or a press-fit preference can pass its own.

Orientation
-----------
Every part is built in millimetres with its build axis along ``+Z`` and the face
that should sit on the print bed at ``z = 0``:

* nose cone - spigot end down, tip up, which needs no support anywhere;
* body tube and motor mount - standing on one end face;
* fin - standing on its root, so the layers run across the span and the
  aerofoil is formed in the XY plane rather than stepped across the layers.

References
----------
[1] Stine, G. H., & Stine, B. (2004). *Handbook of Model Rocketry*, 7th ed.,
    Ch. 5 - shoulder length and motor mount practice.
[2] Niskanen, S. (2013). *OpenRocket Technical Documentation*, Sec. 3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from rocketopt.cad.mesh import (
    TriangleMesh,
    dedupe_polyline,
    loft,
    revolve_profile,
)
from rocketopt.geometry.components import BodyTube
from rocketopt.geometry.fin_sections import section_outline
from rocketopt.geometry.nose_cones import MAX_SHOULDER_LENGTH, MIN_SHOULDER_LENGTH
from rocketopt.geometry.rocket import Rocket
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "DEFAULT_TOLERANCES",
    "PrintTolerances",
    "PrintablePart",
    "body_tube_mesh",
    "body_tube_of",
    "body_tube_profile",
    "fin_loft_sections",
    "fin_mesh",
    "main_body_tube",
    "motor_mount_mesh",
    "motor_mount_profile",
    "motor_mount_radii",
    "nose_cone_mesh",
    "nose_cone_profile",
    "printable_parts",
    "spigot_length",
]

_MM: Final[float] = 1e3
"""Metres to millimetres. Every mesh in this module is built in millimetres."""


@dataclass(frozen=True, slots=True)
class PrintTolerances:
    """Clearances and tessellation settings for printable output.

    Attributes
    ----------
    spigot_clearance:
        Diametral clearance between the nose cone spigot and the body tube
        bore [m]. 0.3 mm on diameter gives a firm push fit that still comes
        apart by hand at the field.
    mount_clearance:
        Diametral clearance between the motor mount and the body tube bore [m].
    motor_clearance:
        Diametral clearance between the motor case and the mount bore [m]. A
        little looser than the airframe fits, because a spent case has to come
        out while still warm.
    revolve_segments:
        Facets around the circumference of a revolved part. 128 holds the
        faceting error on a 25 mm tube below 4 micrometres, two orders of
        magnitude finer than a printer can resolve, without making the file
        needlessly large.
    nose_profile_samples:
        Stations along the nose profile.
    section_samples:
        Chordwise stations in a fin section outline.
    span_sections:
        Spanwise sections lofted through a fin, at least 2.
    min_spigot_length:
        Shortest default spigot [m].
    max_spigot_length:
        Longest default spigot [m].
    """

    spigot_clearance: float = 0.30e-3
    mount_clearance: float = 0.30e-3
    motor_clearance: float = 0.40e-3
    revolve_segments: int = 128
    nose_profile_samples: int = 120
    section_samples: int = 61
    span_sections: int = 5
    min_spigot_length: float = MIN_SHOULDER_LENGTH
    max_spigot_length: float = MAX_SHOULDER_LENGTH

    def __post_init__(self) -> None:
        """Validate the tolerance set."""
        for name in ("spigot_clearance", "mount_clearance", "motor_clearance"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0e-3:
                raise ValueError(
                    f"{name} of {value * 1e3:.2f} mm is outside the sensible "
                    f"range 0 to 1 mm on diameter"
                )
        if self.revolve_segments < 12:
            raise ValueError("revolve_segments must be at least 12")
        if self.nose_profile_samples < 8:
            raise ValueError("nose_profile_samples must be at least 8")
        if self.section_samples < 5:
            raise ValueError("section_samples must be at least 5")
        if self.span_sections < 2:
            raise ValueError("span_sections must be at least 2")
        if not 0.0 < self.min_spigot_length <= self.max_spigot_length:
            raise ValueError("spigot length bounds must be positive and ordered")


DEFAULT_TOLERANCES: Final[PrintTolerances] = PrintTolerances()
"""Tolerances used when a caller does not supply their own."""


@dataclass(frozen=True, slots=True)
class PrintablePart:
    """One solid part of a design, ready to slice.

    Attributes
    ----------
    key:
        Stable identifier, also the file stem.
    name:
        Human-readable part name.
    mesh:
        The solid, in millimetres.
    description:
        One-line summary of the driving dimensions, for logs and reports.
    quantity:
        How many of this part the design needs.
    """

    key: str
    name: str
    mesh: TriangleMesh
    description: str
    quantity: int = 1

    @property
    def filename(self) -> str:
        """Suggested STL file name."""
        return f"{self.key}.stl"


def body_tube_of(rocket: Rocket) -> BodyTube:
    """Return the design's main body tube.

    Parameters
    ----------
    rocket:
        The design.

    Returns
    -------
    BodyTube
        The longest cylindrical section, which is the airframe the nose cone
        and motor mount have to fit.

    Raises
    ------
    ValueError
        If the design has no cylindrical section. :class:`Rocket` validation
        makes this unreachable for an assembled design.
    """
    tubes = [p.section for p in rocket.sections if isinstance(p.section, BodyTube)]
    if not tubes:  # pragma: no cover - Rocket.__post_init__ guarantees one
        raise ValueError("design has no cylindrical body tube")
    return max(tubes, key=lambda tube: tube.length)


# Kept as an alias because "main body tube" reads better at call sites that are
# choosing between several tubes.
main_body_tube = body_tube_of


def spigot_length(
    rocket: Rocket, tolerances: PrintTolerances = DEFAULT_TOLERANCES
) -> float:
    """Return the insertion depth of the nose cone spigot [m].

    The design's own shoulder length is used whenever it has one, because that
    is what the mass model has already accounted for. Otherwise the default is
    one body diameter - the usual rule of thumb [1] - clamped into
    ``[min_spigot_length, max_spigot_length]`` and, for a very short airframe,
    to a quarter of the body tube length so the spigot cannot swallow the
    recovery bay.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerance set supplying the default bounds.

    Returns
    -------
    float
        Spigot insertion depth [m].
    """
    if rocket.nose.shoulder_length > 0.0:
        return rocket.nose.shoulder_length

    tube = body_tube_of(rocket)
    default = min(
        max(rocket.reference_diameter, tolerances.min_spigot_length),
        tolerances.max_spigot_length,
    )
    return min(default, 0.25 * tube.length)


def _spigot_outer_radius(
    rocket: Rocket, tolerances: PrintTolerances
) -> tuple[float, float]:
    """Return the ``(nominal, printed)`` spigot outside radius [m].

    The nominal radius is the body tube bore. The printed radius is that less
    half the diametral print clearance.
    """
    nominal = rocket.nose.shoulder_outer_radius
    return nominal, nominal - tolerances.spigot_clearance / 2.0


def nose_cone_profile(
    rocket: Rocket, tolerances: PrintTolerances = DEFAULT_TOLERANCES
) -> list[tuple[float, float]]:
    """Return the closed ``(z, r)`` profile of the nose cone, in millimetres.

    The loop runs down the outer surface from the tip, across the base annulus
    to the spigot, along the spigot and back up the inside of the shell. A
    hollow cone therefore comes out as an open-ended shell of the design's wall
    thickness with the spigot bore continuous into the parachute bay; a solid
    cone comes out as solid stock with a solid spigot.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerances.

    Returns
    -------
    list of tuple
        Closed profile in millimetres, with ``z`` measured up from the aft end
        of the spigot - the face that sits on the print bed - and ``r`` the
        radius.
    """
    nose = rocket.nose
    spigot = spigot_length(rocket, tolerances)
    _, spigot_radius = _spigot_outer_radius(rocket, tolerances)

    length_mm = nose.length * _MM
    spigot_mm = spigot * _MM
    spigot_r_mm = spigot_radius * _MM
    wall_mm = nose.wall_thickness * _MM
    total_mm = length_mm + spigot_mm

    def station(axial_mm: float) -> float:
        """Convert a station measured aft of the tip into a build height."""
        return total_mm - axial_mm

    xs, rs = nose.profile_points(tolerances.nose_profile_samples)
    outer = [
        (station(float(x) * _MM), float(r) * _MM) for x, r in zip(xs, rs, strict=True)
    ]

    # Base annulus, then out along the spigot.
    profile = [
        *outer,
        (station(length_mm), spigot_r_mm),
        (station(total_mm), spigot_r_mm),
    ]

    if nose.solid:
        # Solid stock: close straight across the aft face to the axis.
        profile.append((station(total_mm), 0.0))
        return dedupe_polyline(profile, closed=True)

    bore_mm = max(spigot_r_mm - wall_mm, 0.0)
    profile.append((station(total_mm), bore_mm))
    profile.append((station(length_mm), bore_mm))

    # Up the inside of the shell. The wall is offset radially, matching the
    # shell volume NoseCone.material_volume integrates, and the cone goes solid
    # where the profile radius falls below the wall thickness.
    inner_x = np.linspace(nose.length, 0.0, tolerances.nose_profile_samples)
    inner_r = np.asarray(nose.radius_at(inner_x)) - nose.wall_thickness
    for x, r in zip(inner_x, inner_r, strict=True):
        if r <= 0.0:
            break
        profile.append((station(float(x) * _MM), float(r) * _MM))
    else:  # pragma: no cover - a profile that never reaches the axis
        x = float(inner_x[-1])
        profile.append((station(x * _MM), 0.0))
        return dedupe_polyline(profile, closed=True)

    # Interpolate the exact station where the wall closes on the axis, so the
    # solid tip is the right length rather than a sample interval too long.
    solid_x = _solid_tip_station(rocket)
    profile.append((station(solid_x * _MM), 0.0))
    return dedupe_polyline(profile, closed=True)


def _solid_tip_station(rocket: Rocket) -> float:
    """Return the station aft of the tip where the shell wall closes [m].

    Forward of this station the profile is narrower than the wall, so the cone
    is solid there. Every family grows monotonically from zero at the tip and is
    far wider than any wall by the base, so there is exactly one crossing of
    ``radius = wall_thickness`` and bisection on the analytic profile finds it -
    more precisely than refining the sampled polyline would.
    """
    nose = rocket.nose
    wall = nose.wall_thickness
    if float(nose.radius_at(nose.length)[0]) <= wall:  # pragma: no cover
        return nose.length

    low, high = 0.0, nose.length
    for _ in range(80):
        mid = 0.5 * (low + high)
        if float(nose.radius_at(mid)[0]) < wall:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def nose_cone_mesh(
    rocket: Rocket, tolerances: PrintTolerances = DEFAULT_TOLERANCES
) -> TriangleMesh:
    """Build the nose cone solid, spigot included.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerances.

    Returns
    -------
    TriangleMesh
        A closed solid in millimetres, tip up, spigot end on the bed.
    """
    return revolve_profile(
        nose_cone_profile(rocket, tolerances), segments=tolerances.revolve_segments
    )


def _tube_profile(
    length: float, inner_radius: float, outer_radius: float
) -> list[tuple[float, float]]:
    """Return the closed ``(z, r)`` profile of an annular tube, in millimetres."""
    length_mm = length * _MM
    inner_mm = inner_radius * _MM
    outer_mm = outer_radius * _MM
    return [
        (0.0, inner_mm),
        (0.0, outer_mm),
        (length_mm, outer_mm),
        (length_mm, inner_mm),
    ]


def body_tube_profile(rocket: Rocket) -> list[tuple[float, float]]:
    """Return the closed ``(z, r)`` profile of the body tube, in millimetres.

    No clearance is applied: the tube is the datum every other part is cut to,
    and shrinking it would shift the airframe's aerodynamic diameter.

    Parameters
    ----------
    rocket:
        The design.

    Returns
    -------
    list of tuple
        Closed profile in millimetres, standing on its forward face at
        ``z = 0``.
    """
    tube = body_tube_of(rocket)
    return _tube_profile(tube.length, tube.inner_radius, tube.outer_radius)


def body_tube_mesh(
    rocket: Rocket, tolerances: PrintTolerances = DEFAULT_TOLERANCES
) -> TriangleMesh:
    """Build the body tube solid at its nominal dimensions.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerances, used only for the facet count.

    Returns
    -------
    TriangleMesh
        A closed annular tube in millimetres, standing on its forward face.
    """
    return revolve_profile(
        body_tube_profile(rocket), segments=tolerances.revolve_segments
    )


def motor_mount_radii(
    rocket: Rocket, tolerances: PrintTolerances = DEFAULT_TOLERANCES
) -> tuple[float, float]:
    """Return the ``(bore, outside)`` radius of the printable motor mount [m].

    The outside radius is the body tube bore less half the print clearance, and
    the bore is the motor case radius plus half the motor clearance, so the part
    drops into the airframe and accepts the motor without reaming. Where the
    design's mount is deliberately narrower than the airframe - a hand-specified
    mount carried on centring rings - its own outside diameter is respected
    instead.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerances.

    Returns
    -------
    tuple of float
        Bore and outside radius [m].

    Raises
    ------
    ValueError
        If the bore is not smaller than the outside diameter once clearances are
        applied, which means the design cannot be built as a separate mount at
        all.
    """
    mount = rocket.motor_mount
    tube = body_tube_of(rocket)

    outer = (
        min(mount.outer_radius, tube.inner_radius) - tolerances.mount_clearance / 2.0
    )
    inner = max(
        mount.inner_diameter / 2.0,
        rocket.motor.motor.diameter / 2.0 + tolerances.motor_clearance / 2.0,
    )
    if inner >= outer:
        raise ValueError(
            f"motor mount for {rocket.motor.designation} would have a "
            f"{inner * 2e3:.1f} mm bore inside a {outer * 2e3:.1f} mm outside "
            f"diameter; the motor does not fit the airframe"
        )
    return inner, outer


def motor_mount_profile(
    rocket: Rocket, tolerances: PrintTolerances = DEFAULT_TOLERANCES
) -> list[tuple[float, float]]:
    """Return the closed ``(z, r)`` profile of the motor mount, in millimetres.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerances.

    Returns
    -------
    list of tuple
        Closed profile in millimetres, standing on its forward face at
        ``z = 0``.
    """
    inner, outer = motor_mount_radii(rocket, tolerances)
    return _tube_profile(rocket.motor_mount.length, inner, outer)


def motor_mount_mesh(
    rocket: Rocket, tolerances: PrintTolerances = DEFAULT_TOLERANCES
) -> TriangleMesh:
    """Build the motor mount solid.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerances.

    Returns
    -------
    TriangleMesh
        A closed annular tube in millimetres, standing on its forward face.
    """
    return revolve_profile(
        motor_mount_profile(rocket, tolerances), segments=tolerances.revolve_segments
    )


def _fin_span_stations(rocket: Rocket, tolerances: PrintTolerances) -> list[float]:
    """Return the spanwise fractions to loft a fin through.

    Uniform stations, plus the exact station at which the local chord falls to
    the fin thickness. That is the only kink in an otherwise ruled surface: aft
    of it the section thickness is clamped to the chord so the section stays
    valid into a vanishing tip.
    """
    fins = rocket.fins
    stations = list(np.linspace(0.0, 1.0, tolerances.span_sections))

    taper = fins.root_chord - fins.tip_chord
    if taper > 0.0 and fins.tip_chord < fins.thickness < fins.root_chord:
        clamp = (fins.root_chord - fins.thickness) / taper
        if 0.0 < clamp < 1.0:
            stations.append(clamp)

    return sorted(set(stations))


def fin_loft_sections(
    rocket: Rocket,
    tolerances: PrintTolerances = DEFAULT_TOLERANCES,
    *,
    min_chord: float = 0.0,
) -> list[tuple[float, list[tuple[float, float]]]]:
    """Return the section outlines a fin is lofted through, in millimetres.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerances, supplying the section and span sampling.
    min_chord:
        Shortest chord any section may have [m]. Zero lets a delta fin close to
        a true point, which the mesh loft handles. A BREP kernel cannot loft to
        a point, so the STEP exporter passes a small positive value and gets a
        sliver of a tip instead.

    Returns
    -------
    list of tuple
        ``(span_height_mm, outline)`` pairs ordered root to tip, where the
        outline vertices are ``(x, y)`` in millimetres with ``x`` measured aft
        of the *root* leading edge, so the leading-edge sweep is already
        applied.
    """
    fins = rocket.fins
    sections: list[tuple[float, list[tuple[float, float]]]] = []

    for eta in _fin_span_stations(rocket, tolerances):
        chord = fins.root_chord + (fins.tip_chord - fins.root_chord) * eta
        if chord < min_chord:
            chord = min_chord
        outline = section_outline(
            fins.airfoil,
            chord=chord,
            thickness=fins.thickness,
            samples=tolerances.section_samples,
        )
        lead_mm = fins.sweep_length * eta * _MM
        sections.append(
            (
                fins.span * eta * _MM,
                [(lead_mm + u * _MM, v * _MM) for u, v in outline],
            )
        )

    return sections


def fin_mesh(
    rocket: Rocket, tolerances: PrintTolerances = DEFAULT_TOLERANCES
) -> TriangleMesh:
    """Build one fin as a solid with its true aerodynamic section.

    The fin is lofted through its cross-section outlines from root to tip, so
    the exported solid carries the specified leading edge, trailing edge and
    thickness distribution instead of the flat plate an extruded planform would
    give. The planform - root chord, tip chord, span and leading-edge sweep - is
    reproduced exactly, and matches the SVG and DXF templates of the same
    design.

    The fin is not canted: a canted fin set is the same part fitted at an angle,
    so one part serves every fin in the set.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerances, supplying the section and span sampling.

    Returns
    -------
    TriangleMesh
        A closed solid in millimetres, root on the bed, span along ``+Z``,
        chord along ``+X`` and thickness across ``Y``.
    """
    return loft(
        [
            [(x, y, height) for x, y in outline]
            for height, outline in fin_loft_sections(rocket, tolerances)
        ]
    )


def printable_parts(
    rocket: Rocket, tolerances: PrintTolerances = DEFAULT_TOLERANCES
) -> tuple[PrintablePart, ...]:
    """Build every solid part of a design.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerances.

    Returns
    -------
    tuple of PrintablePart
        Nose cone, body tube, motor mount and fin, in assembly order.
    """
    tube = body_tube_of(rocket)
    mount = rocket.motor_mount
    motor = rocket.motor.motor
    fins = rocket.fins
    nominal_spigot, printed_spigot = _spigot_outer_radius(rocket, tolerances)
    spigot = spigot_length(rocket, tolerances)
    nose = rocket.nose
    wall_note = (
        "solid" if nose.solid else f"{nose.wall_thickness * _MM:.1f} mm wall"
    )

    parts = (
        PrintablePart(
            key="nose_cone",
            name="Nose cone",
            mesh=nose_cone_mesh(rocket, tolerances),
            description=(
                f"{nose.shape.label}, {nose.length * _MM:.1f} mm long, {wall_note}; "
                f"integrated spigot {printed_spigot * 2 * _MM:.2f} mm diameter x "
                f"{spigot * _MM:.1f} mm, for a {nominal_spigot * 2 * _MM:.2f} mm bore"
            ),
        ),
        PrintablePart(
            key="body_tube",
            name="Body tube",
            mesh=body_tube_mesh(rocket, tolerances),
            description=(
                f"{tube.diameter * _MM:.2f} mm outside diameter x "
                f"{tube.wall_thickness * _MM:.2f} mm wall x "
                f"{tube.length * _MM:.1f} mm; bore {tube.inner_radius * 2 * _MM:.2f} mm"
            ),
        ),
        PrintablePart(
            key="motor_mount",
            name="Motor mount",
            mesh=motor_mount_mesh(rocket, tolerances),
            description=(
                f"tube for {rocket.motor.designation}: bore "
                f"{mount.inner_diameter * _MM:.2f} mm for a "
                f"{motor.diameter * _MM:.1f} mm case, "
                f"{mount.outer_diameter * _MM:.2f} mm outside diameter x "
                f"{mount.length * _MM:.1f} mm"
            ),
        ),
        PrintablePart(
            key="fin",
            name="Fin",
            mesh=fin_mesh(rocket, tolerances),
            description=(
                f"{fins.airfoil.label.lower()} section, root "
                f"{fins.root_chord * _MM:.1f} mm, tip {fins.tip_chord * _MM:.1f} mm, "
                f"span {fins.span * _MM:.1f} mm, sweep "
                f"{fins.sweep_length * _MM:.1f} mm, "
                f"{fins.thickness * _MM:.2f} mm thick"
            ),
            quantity=fins.count,
        ),
    )

    _log.debug(
        "Built %d printable parts for %s, %d triangles in total",
        len(parts),
        rocket.name,
        sum(part.mesh.triangle_count for part in parts),
    )
    return parts
