"""Pre-export checks on the printable geometry of a design.

An STL that a slicer rejects, or worse accepts and prints wrong, is a waste of
several hours. Everything a set of parts has to satisfy before it is written is
therefore checked here, and measured off the meshes themselves rather than
recomputed from the design, so the checks cover the geometry that will actually
be sliced:

* the nose cone spigot enters the body tube bore, with a clearance that is
  present but small, and the cone's base is flush with the tube outside it;
* the motor mount enters the same bore, and sits inside the tube axially;
* the selected motor enters the mount's bore, and the mount is long enough to
  retain it;
* the fin reproduces its planform and its section, to the thickness;
* every part is a closed, manifold, outward-facing solid.

:func:`~rocketopt.cad.exporters.export_stl_parts` runs
:func:`validate_printable_assembly` and refuses to write anything if a check
fails, so a written STL set has passed all of this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from rocketopt.cad.parts import (
    DEFAULT_TOLERANCES,
    PrintablePart,
    PrintTolerances,
    body_tube_of,
    printable_parts,
    spigot_length,
)
from rocketopt.geometry.fin_sections import section_area
from rocketopt.geometry.nose_cones import MIN_SHOULDER_LENGTH
from rocketopt.geometry.rocket import Rocket
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "AssemblyValidation",
    "AssemblyValidationError",
    "FitCheck",
    "validate_printable_assembly",
]

_MM: Final[float] = 1e3
"""Metres to millimetres."""

_MAX_CLEARANCE: Final[float] = 1.0e-3
"""Largest diametral clearance a mating fit may have [m].

Beyond a millimetre on diameter the joint rattles: a nose cone can cock over far
enough to break the airflow at the shoulder, and a motor can shift under thrust.
"""

_MIN_CLEARANCE: Final[float] = 0.05e-3
"""Smallest diametral clearance a mating fit may have [m].

Two nominally equal printed diameters will not go together; 0.05 mm is the least
that survives the dimensional scatter of a well-tuned machine.
"""

_VOLUME_TOLERANCE: Final[float] = 0.02
"""Relative agreement required between a mesh volume and its analytic value."""

_NOSE_VOLUME_TOLERANCE: Final[float] = 0.05
"""Relative agreement required between the nose cone mesh and the mass model.

Looser than the other parts, because the exported spigot is a print clearance
smaller than the nominal one the mass model integrates, and on a stubby cone the
spigot is a large fraction of the material. The check is there to catch a cone
exported solid when it should be a shell, or without its spigot - both of which
are several times this band.
"""


@dataclass(frozen=True, slots=True)
class FitCheck:
    """The outcome of one check.

    Attributes
    ----------
    name:
        Short description of what was checked.
    passed:
        Whether it holds.
    detail:
        The measurement, phrased so it is useful whether the check passed or
        failed.
    """

    name: str
    passed: bool
    detail: str

    def __str__(self) -> str:
        """Return a one-line report of this check."""
        return f"[{'pass' if self.passed else 'FAIL'}] {self.name}: {self.detail}"


@dataclass(frozen=True, slots=True)
class AssemblyValidation:
    """Every check run over a design's printable parts.

    Attributes
    ----------
    checks:
        The individual outcomes, in the order they were run.
    """

    checks: tuple[FitCheck, ...]

    @property
    def passed(self) -> bool:
        """Whether every check holds."""
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[FitCheck, ...]:
        """The checks that do not hold."""
        return tuple(check for check in self.checks if not check.passed)

    def report(self) -> str:
        """Return a multi-line report of every check."""
        header = (
            f"{len(self.checks)} checks, "
            f"{len(self.checks) - len(self.failures)} passed, "
            f"{len(self.failures)} failed"
        )
        return "\n".join([header, *(str(check) for check in self.checks)])

    def raise_for_failures(self) -> None:
        """Raise if any check failed.

        Raises
        ------
        AssemblyValidationError
            Listing every failure, so one export attempt reports every problem
            rather than the first.
        """
        if self.passed:
            return
        detail = "\n".join(
            f"  - {check.name}: {check.detail}" for check in self.failures
        )
        raise AssemblyValidationError(
            f"{len(self.failures)} of {len(self.checks)} pre-export checks failed:\n"
            f"{detail}"
        )


class AssemblyValidationError(ValueError):
    """Raised when a design's parts would not assemble or would not print."""


def _fit_check(
    name: str,
    *,
    inner: float,
    outer: float,
    what: str,
    minimum: float = _MIN_CLEARANCE,
    maximum: float = _MAX_CLEARANCE,
) -> FitCheck:
    """Check that one diameter enters another with a sensible clearance.

    Parameters
    ----------
    name:
        Check name.
    inner:
        Diameter of the part that goes inside [m].
    outer:
        Diameter of the bore it goes into [m].
    what:
        Phrase naming the two parts, for the detail line.
    minimum, maximum:
        Bounds on the diametral clearance [m].

    Returns
    -------
    FitCheck
        The outcome.
    """
    clearance = outer - inner
    passed = minimum <= clearance <= maximum
    return FitCheck(
        name=name,
        passed=passed,
        detail=(
            f"{what}: {inner * _MM:.2f} mm into {outer * _MM:.2f} mm, clearance "
            f"{clearance * _MM:+.2f} mm (allowed {minimum * _MM:.2f} to "
            f"{maximum * _MM:.2f} mm)"
        ),
    )


def _mesh_checks(part: PrintablePart) -> list[FitCheck]:
    """Check that a part's mesh is a printable solid."""
    problems = part.mesh.validate()
    return [
        FitCheck(
            name=f"{part.name} mesh is a watertight, manifold solid",
            passed=not problems,
            detail=(
                f"{part.mesh.triangle_count} triangles, "
                f"{part.mesh.volume / 1e3:.2f} cm^3 enclosed"
                if not problems
                else "; ".join(problems)
            ),
        )
    ]


def _minimum_engagement(body_diameter: float) -> float:
    """Return the shortest spigot that will carry the ejection load [m].

    Half a body diameter of engagement is the classic minimum: less and the
    ejection charge racks the joint rather than shearing it. On an airframe wide
    enough that half a diameter exceeds the 20 mm floor a default spigot is
    given, that floor governs instead - a 50 mm airframe with a 20 mm shoulder is
    short but sound, and refusing to export it would be an opinion rather than a
    check.

    Parameters
    ----------
    body_diameter:
        Body tube outside diameter [m].

    Returns
    -------
    float
        Minimum acceptable insertion depth [m].
    """
    return min(0.5 * body_diameter, MIN_SHOULDER_LENGTH)


def _nose_checks(
    rocket: Rocket, part: PrintablePart, tolerances: PrintTolerances
) -> list[FitCheck]:
    """Check the nose cone against the body tube it plugs into."""
    tube = body_tube_of(rocket)
    spigot = spigot_length(rocket, tolerances)
    spigot_mm = spigot * _MM

    # The spigot is a plain cylinder, so its only vertices are the rings at each
    # end. Measure the aft face, which is the ring that has to enter the tube,
    # in a slab thin enough to exclude the base annulus at the spigot's forward
    # end. On a solid cone the inner radius comes out as zero, correctly.
    bore_mm, spigot_outer_mm = part.mesh.radial_extent(0.0, 0.25 * spigot_mm)
    _, base_mm = part.mesh.radial_extent()

    checks = [
        _fit_check(
            "Nose cone spigot enters the body tube",
            inner=2.0 * spigot_outer_mm / _MM,
            outer=2.0 * tube.inner_radius,
            what="spigot into tube bore",
            maximum=max(tolerances.spigot_clearance * 2.0, _MIN_CLEARANCE * 2.0),
        ),
        FitCheck(
            name="Nose cone spigot is sized to the body tube bore",
            passed=math.isclose(
                rocket.nose.shoulder_outer_radius, tube.inner_radius, abs_tol=1e-9
            ),
            detail=(
                f"nominal spigot {rocket.nose.shoulder_outer_radius * 2e3:.2f} mm "
                f"against a {tube.inner_radius * 2e3:.2f} mm bore"
            ),
        ),
        FitCheck(
            name="Nose cone base is flush with the body tube",
            passed=math.isclose(
                2.0 * base_mm / _MM, tube.diameter, rel_tol=1e-3, abs_tol=1e-5
            ),
            detail=(
                f"cone base {2.0 * base_mm:.2f} mm against a "
                f"{tube.diameter * _MM:.2f} mm tube"
            ),
        ),
        FitCheck(
            name="Nose cone spigot is deep enough to carry the ejection load",
            passed=spigot >= _minimum_engagement(tube.diameter) - 1e-9,
            detail=(
                f"{spigot_mm:.1f} mm of engagement, at least "
                f"{_minimum_engagement(tube.diameter) * _MM:.1f} mm needed for a "
                f"{tube.diameter * _MM:.1f} mm airframe"
            ),
        ),
    ]

    if not rocket.nose.solid:
        checks.append(
            FitCheck(
                name="Nose cone spigot is bored through to the recovery bay",
                passed=bore_mm > 0.0,
                detail=f"spigot bore {2.0 * bore_mm:.2f} mm diameter",
            )
        )

    expected = rocket.nose.material_volume * _MM**3
    checks.append(
        FitCheck(
            name="Nose cone solid matches the mass model",
            passed=math.isclose(
                part.mesh.volume, expected, rel_tol=_NOSE_VOLUME_TOLERANCE
            ),
            detail=(
                f"{part.mesh.volume / 1e3:.2f} cm^3 exported against "
                f"{expected / 1e3:.2f} cm^3 in the mass model"
            ),
        )
    )
    return checks


def _mount_checks(
    rocket: Rocket, part: PrintablePart, tolerances: PrintTolerances
) -> list[FitCheck]:
    """Check the motor mount against the body tube and the motor."""
    tube = body_tube_of(rocket)
    mount = rocket.motor_mount
    motor = rocket.motor.motor
    bore_mm, outer_mm = part.mesh.radial_extent()

    body_aft = max(placed.aft_position for placed in rocket.body_tubes)

    # A mount turned to the airframe bore locates itself, so its clearance has
    # to be small. One deliberately narrower than the bore is carried on
    # centring rings, which are fitted separately, so all that can be asked of
    # the tube itself is that it goes in at all.
    if mount.is_minimum_diameter:
        fit = _fit_check(
            "Motor mount enters the body tube",
            inner=2.0 * outer_mm / _MM,
            outer=2.0 * tube.inner_radius,
            what="mount outside diameter into tube bore",
            maximum=max(tolerances.mount_clearance * 2.0, _MIN_CLEARANCE * 2.0),
        )
    else:
        gap = 2.0 * tube.inner_radius - 2.0 * outer_mm / _MM
        fit = FitCheck(
            name="Motor mount enters the body tube",
            passed=gap >= _MIN_CLEARANCE,
            detail=(
                f"{2.0 * outer_mm:.2f} mm mount into a "
                f"{tube.inner_radius * 2e3:.2f} mm bore, {gap * _MM:+.2f} mm of "
                f"annulus for {mount.centring_ring_count} centring ring(s) to "
                f"bridge"
            ),
        )

    return [
        fit,
        _fit_check(
            "Motor enters the motor mount",
            inner=motor.diameter,
            outer=2.0 * bore_mm / _MM,
            what=f"{rocket.motor.designation} case into mount bore",
            maximum=max(tolerances.motor_clearance * 2.0, _MIN_CLEARANCE * 2.0),
        ),
        FitCheck(
            name="Motor mount is no wider than the body tube bore",
            passed=mount.outer_radius <= tube.inner_radius + 1e-9,
            detail=(
                f"mount {mount.outer_diameter * _MM:.2f} mm nominal against a "
                f"{tube.inner_radius * 2e3:.2f} mm bore, "
                + (
                    "self-centring"
                    if mount.is_minimum_diameter
                    else f"centred by {mount.centring_ring_count} ring(s)"
                )
            ),
        ),
        FitCheck(
            name="Motor mount is long enough to retain the motor",
            passed=mount.length >= motor.length - 1e-9,
            detail=(
                f"{mount.length * _MM:.1f} mm of mount for a "
                f"{motor.length * _MM:.1f} mm case"
            ),
        ),
        FitCheck(
            name="Motor mount fits within the airframe length",
            passed=mount.position + mount.length <= body_aft + 1e-9,
            detail=(
                f"mount ends at {(mount.position + mount.length) * _MM:.1f} mm, "
                f"airframe at {body_aft * _MM:.1f} mm aft of the tip"
            ),
        ),
    ]


def _fin_checks(rocket: Rocket, part: PrintablePart) -> list[FitCheck]:
    """Check that the fin solid reproduces the specified fin."""
    fins = rocket.fins
    size = part.mesh.size
    root_area = section_area(
        fins.airfoil, thickness=fins.thickness, chord=fins.root_chord
    )
    expected_chord = max(fins.root_chord, fins.sweep_length + fins.tip_chord) * _MM

    # The lofted section area integrated over the span, which is what the fin
    # would enclose if the loft reproduced the specification exactly.
    expected_volume = fins.volume_single * _MM**3

    return [
        FitCheck(
            name="Fin reproduces its planform",
            passed=(
                math.isclose(size[0], expected_chord, rel_tol=1e-6, abs_tol=1e-6)
                and math.isclose(size[2], fins.span * _MM, rel_tol=1e-6, abs_tol=1e-6)
            ),
            detail=(
                f"{size[0]:.2f} x {size[2]:.2f} mm exported against "
                f"{expected_chord:.2f} x {fins.span * _MM:.2f} mm specified"
            ),
        ),
        FitCheck(
            name="Fin reaches its specified thickness",
            passed=math.isclose(
                size[1], fins.thickness * _MM, rel_tol=1e-4, abs_tol=1e-6
            ),
            detail=(
                f"{size[1]:.3f} mm exported against "
                f"{fins.thickness * _MM:.3f} mm specified"
            ),
        ),
        FitCheck(
            name=f"Fin carries its {fins.airfoil.label.lower()} section",
            passed=math.isclose(
                part.mesh.volume, expected_volume, rel_tol=_VOLUME_TOLERANCE
            ),
            detail=(
                f"{part.mesh.volume / 1e3:.3f} cm^3 exported against "
                f"{expected_volume / 1e3:.3f} cm^3 for a "
                f"{fins.airfoil.label.lower()} section "
                f"({root_area * 1e6:.2f} mm^2 at the root)"
            ),
        ),
    ]


def validate_printable_assembly(
    rocket: Rocket,
    tolerances: PrintTolerances = DEFAULT_TOLERANCES,
    parts: tuple[PrintablePart, ...] | None = None,
) -> AssemblyValidation:
    """Check that a design's parts will assemble and will print.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Print tolerances the parts were, or will be, built with.
    parts:
        Already-built parts to check. Built from ``rocket`` when ``None``.

    Returns
    -------
    AssemblyValidation
        Every check and its outcome. Call
        :meth:`AssemblyValidation.raise_for_failures` to turn failures into an
        exception.
    """
    built = printable_parts(rocket, tolerances) if parts is None else parts
    by_key = {part.key: part for part in built}

    checks: list[FitCheck] = []
    for part in built:
        checks.extend(_mesh_checks(part))
    if "nose_cone" in by_key:
        checks.extend(_nose_checks(rocket, by_key["nose_cone"], tolerances))
    if "motor_mount" in by_key:
        checks.extend(_mount_checks(rocket, by_key["motor_mount"], tolerances))
    if "fin" in by_key:
        checks.extend(_fin_checks(rocket, by_key["fin"]))

    validation = AssemblyValidation(tuple(checks))
    _log.debug(
        "Validated %s: %d checks, %d failed",
        rocket.name,
        len(validation.checks),
        len(validation.failures),
    )
    return validation
