"""The whole rocket as one set of meshes, positioned as it is built.

:mod:`rocketopt.cad.parts` builds each component on its own, oriented for a
print bed. This module takes the same geometry and puts it where it actually
sits on the rocket, which is what a viewer, a render or an assembly check needs.

Coordinates
-----------
Millimetres, with ``+Z`` along the body axis measured aft from the nose tip -
the same station convention as the rest of the codebase, just promoted to three
dimensions. ``X`` and ``Y`` are radial, and the first fin is on ``+X``.

Tessellation
------------
:data:`DISPLAY_TOLERANCES` coarsens the sampling to roughly a tenth of the
triangles a printable part carries. On screen the difference is invisible - the
faceting is still finer than a pixel at any sensible zoom - and it is the
difference between a viewer that turns smoothly and one that does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rocketopt.cad.mesh import TriangleMesh, revolve_profile
from rocketopt.cad.parts import (
    DEFAULT_TOLERANCES,
    PrintTolerances,
    fin_mesh,
    motor_mount_radii,
    nose_cone_profile,
    spigot_length,
)
from rocketopt.geometry.components import BodyTube, Transition
from rocketopt.geometry.rocket import Rocket
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "DISPLAY_TOLERANCES",
    "AssemblyPart",
    "assembled_parts",
    "assembly_bounds",
]

_MM: Final[float] = 1e3
"""Metres to millimetres."""

DISPLAY_TOLERANCES: Final[PrintTolerances] = PrintTolerances(
    revolve_segments=40,
    nose_profile_samples=32,
    section_samples=25,
    span_sections=3,
)
"""Sampling for an on-screen assembly, about a fifteenth of a printable part.

Chosen so a whole rocket comes to a few thousand triangles, which a software
rasteriser can turn smoothly. At any sensible zoom the extra faceting is finer
than a pixel."""


@dataclass(frozen=True, slots=True)
class AssemblyPart:
    """One component, placed where it sits on the rocket.

    Attributes
    ----------
    key:
        Stable identifier. Repeated parts share it, so the three fins of a
        three-fin set are all ``"fin"``.
    name:
        Human-readable name, unique within the assembly.
    mesh:
        The solid in assembly coordinates, millimetres.
    colour:
        Suggested ``(r, g, b)`` for a viewer, 0-255.
    opaque:
        Whether the part is on the outside of the airframe. A viewer can make
        the opaque parts translucent to show what is inside.
    axisymmetric:
        Whether the part is a body of revolution about the axis. A viewer's
        section cut applies only to these: slicing a fin along its span reveals
        nothing and leaves a row of teeth where the cut crosses it.
    """

    key: str
    name: str
    mesh: TriangleMesh
    colour: tuple[int, int, int]
    opaque: bool = True
    axisymmetric: bool = True


# Muted, distinguishable, and readable against the dark theme.
_NOSE_COLOUR: Final[tuple[int, int, int]] = (214, 218, 224)
_TUBE_COLOUR: Final[tuple[int, int, int]] = (186, 158, 116)
_FIN_COLOUR: Final[tuple[int, int, int]] = (79, 156, 249)
_MOUNT_COLOUR: Final[tuple[int, int, int]] = (150, 156, 166)
_MOTOR_COLOUR: Final[tuple[int, int, int]] = (217, 164, 65)
_LUG_COLOUR: Final[tuple[int, int, int]] = (120, 126, 136)


def _offset_profile(
    profile: list[tuple[float, float]], station_mm: float
) -> list[tuple[float, float]]:
    """Shift a ``(z, r)`` profile aft to an axial station."""
    return [(z + station_mm, r) for z, r in profile]


def _tube_profile(
    station_mm: float, length_mm: float, inner_mm: float, outer_mm: float
) -> list[tuple[float, float]]:
    """Return the closed ``(z, r)`` profile of an annular tube at a station."""
    aft = station_mm + length_mm
    return [
        (station_mm, inner_mm),
        (station_mm, outer_mm),
        (aft, outer_mm),
        (aft, inner_mm),
    ]


def _fin_rotation(angle: float) -> NDArray[np.float64]:
    """Return the rotation placing a fin's local axes onto the airframe.

    A fin is built with its chord along ``+X``, its thickness across ``Y`` and
    its span along ``+Z``. On the airframe the chord runs aft along the body
    axis and the span runs radially outward at ``angle``, so the local axes map
    to::

        chord     -> +Z, the body axis
        span      -> the radial direction at ``angle``
        thickness -> the tangential direction

    The tangential image is chosen as ``span x chord`` rather than the other
    way round, which keeps the frame right-handed and the fin the right way out.

    Parameters
    ----------
    angle:
        Roll angle of the fin about the body axis [rad], zero on ``+X``.

    Returns
    -------
    numpy.ndarray
        A ``(3, 3)`` rotation whose columns are the images of the local axes.
    """
    cos, sin = math.cos(angle), math.sin(angle)
    chord = np.array([0.0, 0.0, 1.0])
    radial = np.array([cos, sin, 0.0])
    tangential = np.cross(radial, chord)
    return np.column_stack([chord, tangential, radial])


def assembled_parts(
    rocket: Rocket, tolerances: PrintTolerances = DISPLAY_TOLERANCES
) -> tuple[AssemblyPart, ...]:
    """Build every component of a design, placed as assembled.

    Parameters
    ----------
    rocket:
        The design.
    tolerances:
        Tessellation and clearances. Defaults to :data:`DISPLAY_TOLERANCES`;
        pass :data:`rocketopt.cad.parts.DEFAULT_TOLERANCES` for the same
        sampling the printable parts use.

    Returns
    -------
    tuple of AssemblyPart
        Nose cone, body sections, motor mount, the loaded motor, every fin and
        the launch lug, in millimetres about the body axis.
    """
    segments = tolerances.revolve_segments
    parts: list[AssemblyPart] = []

    # -- Nose cone. Its profile is built tip-up for printing, so it is folded
    # back to run aft from the tip like every other station in the assembly.
    nose_total_mm = (rocket.nose.length + spigot_length(rocket, tolerances)) * _MM
    nose_profile = [
        (nose_total_mm - z, r) for z, r in nose_cone_profile(rocket, tolerances)
    ]
    parts.append(
        AssemblyPart(
            key="nose_cone",
            name="Nose cone",
            mesh=revolve_profile(nose_profile, segments=segments),
            colour=_NOSE_COLOUR,
        )
    )

    # -- Body sections, each at its own station.
    for index, placed in enumerate(rocket.sections, start=1):
        section = placed.section
        station_mm = placed.position * _MM
        length_mm = section.length * _MM
        if isinstance(section, Transition):
            wall_mm = section.wall_thickness * _MM
            fore_mm = section.fore_radius * _MM
            aft_mm = section.aft_radius * _MM
            profile = [
                (station_mm, max(fore_mm - wall_mm, 0.0)),
                (station_mm, fore_mm),
                (station_mm + length_mm, aft_mm),
                (station_mm + length_mm, max(aft_mm - wall_mm, 0.0)),
            ]
            name = "Boat-tail" if section.is_boat_tail else "Shoulder"
        else:
            assert isinstance(section, BodyTube)
            profile = _tube_profile(
                station_mm,
                length_mm,
                section.inner_radius * _MM,
                section.outer_radius * _MM,
            )
            name = "Body tube"
        parts.append(
            AssemblyPart(
                key="body_tube",
                name=name if len(rocket.sections) == 1 else f"{name} {index}",
                mesh=revolve_profile(profile, segments=segments),
                colour=_TUBE_COLOUR,
            )
        )

    # -- Motor mount, inside the airframe.
    bore, outer = motor_mount_radii(rocket, tolerances)
    mount = rocket.motor_mount
    parts.append(
        AssemblyPart(
            key="motor_mount",
            name="Motor mount",
            mesh=revolve_profile(
                _tube_profile(
                    mount.position * _MM, mount.length * _MM, bore * _MM, outer * _MM
                ),
                segments=segments,
            ),
            colour=_MOUNT_COLOUR,
            opaque=False,
        )
    )

    # -- The motor itself, seated against the thrust ring.
    motor = rocket.motor.motor
    parts.append(
        AssemblyPart(
            key="motor",
            name=f"Motor {rocket.motor.designation}",
            mesh=revolve_profile(
                _tube_profile(
                    rocket.motor_position * _MM,
                    motor.length * _MM,
                    0.0,
                    motor.diameter / 2.0 * _MM,
                ),
                segments=segments,
            ),
            colour=_MOTOR_COLOUR,
            opaque=False,
        )
    )

    # -- Fins, equally spaced and canted if the design asks for it.
    fins = rocket.fins
    single = fin_mesh(rocket, tolerances)
    if abs(fins.cant_angle) > 1e-9:
        # A cant is a rotation of the fin about its own radial axis, applied
        # before it is placed on the airframe.
        cos, sin = math.cos(fins.cant_angle), math.sin(fins.cant_angle)
        cant = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
        single = single.transformed(cant)

    for index in range(fins.count):
        angle = 2.0 * math.pi * index / fins.count
        placement = np.array(
            [
                fins.body_radius * _MM * math.cos(angle),
                fins.body_radius * _MM * math.sin(angle),
                fins.position * _MM,
            ]
        )
        parts.append(
            AssemblyPart(
                key="fin",
                name=f"Fin {index + 1}",
                mesh=single.transformed(_fin_rotation(angle), placement),
                colour=_FIN_COLOUR,
                axisymmetric=False,
            )
        )

    # -- Launch lug, offset from the axis and lying along it.
    lug = rocket.launch_lug
    if lug is not None:
        # Sit the lug just outboard of the tube, between two fins.
        angle = math.pi / fins.count
        radius_mm = (rocket.reference_radius + lug.outer_radius) * _MM
        barrel = revolve_profile(
            _tube_profile(
                0.0, lug.length * _MM, lug.inner_radius * _MM, lug.outer_radius * _MM
            ),
            segments=max(16, segments // 2),
        )
        parts.append(
            AssemblyPart(
                key="launch_lug",
                name="Launch lug",
                mesh=barrel.transformed(
                    np.eye(3),
                    (
                        radius_mm * math.cos(angle),
                        radius_mm * math.sin(angle),
                        lug.position * _MM,
                    ),
                ),
                colour=_LUG_COLOUR,
                axisymmetric=False,
            )
        )

    _log.debug(
        "Assembled %s: %d parts, %d triangles",
        rocket.name,
        len(parts),
        sum(part.mesh.triangle_count for part in parts),
    )
    return tuple(parts)


def assembly_bounds(
    parts: tuple[AssemblyPart, ...],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the ``(minimum, maximum)`` corner of a whole assembly.

    Parameters
    ----------
    parts:
        The assembled parts.

    Returns
    -------
    tuple of numpy.ndarray
        Bounding box corners in millimetres.

    Raises
    ------
    ValueError
        If there are no parts to bound.
    """
    if not parts:
        raise ValueError("an empty assembly has no bounds")
    lows = np.array([part.mesh.bounds[0] for part in parts])
    highs = np.array([part.mesh.bounds[1] for part in parts])
    return lows.min(axis=0), highs.max(axis=0)


# Re-exported so callers can ask for print-quality tessellation without also
# importing rocketopt.cad.parts.
PRINT_TOLERANCES: Final[PrintTolerances] = DEFAULT_TOLERANCES
"""The printable-part sampling, for a render that should match the STL exactly."""
