"""CAD export: solid models, 2D profiles and manufacturing templates.

Two tiers of output, so that useful CAD is always produced:

*Always available* - printable STL solids of every part, SVG fin templates, an
SVG side elevation and a DXF fin outline, none of which need a third-party
dependency. The STLs come from :mod:`rocketopt.cad.parts` by way of the mesh
kernel in :mod:`rocketopt.cad.mesh`, and are checked against
:mod:`rocketopt.cad.validation` before anything is written, so an exported set
assembles and slices. The fin template is the one 2D output a builder genuinely
needs: printed at 1:1 it is glued to the fin stock and cut around.

*With CadQuery installed* - STEP solids of every component, built from the same
profiles as the STLs so the two agree.

The Fusion 360 script from :mod:`rocketopt.cad.fusion360` is always written
too, and is the better route into parametric CAD than a STEP import.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from rocketopt.cad.mesh import dedupe_polyline, write_stl
from rocketopt.cad.parts import (
    DEFAULT_TOLERANCES,
    PrintTolerances,
    body_tube_profile,
    fin_loft_sections,
    motor_mount_profile,
    nose_cone_profile,
    printable_parts,
)
from rocketopt.cad.validation import validate_printable_assembly
from rocketopt.geometry.components import Transition
from rocketopt.geometry.rocket import Rocket
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "export_fin_template_svg",
    "export_profile_svg",
    "export_fin_dxf",
    "export_solids",
    "export_stl_parts",
    "export_all",
]

_MIN_STEP_TIP_CHORD: Final[float] = 0.05e-3
"""Shortest chord a lofted STEP section may have [m].

A BREP loft cannot close on a point, so a delta fin's sharp tip becomes a
0.05 mm sliver in STEP. The STL keeps the true point, and 0.05 mm is below what
any cutter or printer resolves.
"""

_SVG_SCALE = 3.7795275591
"""Pixels per millimetre at 96 dpi, so an SVG prints at 1:1."""


def export_fin_template_svg(rocket: Rocket, path: str | Path) -> Path:
    """Write a 1:1 fin cutting template as SVG.

    Printed at 100% scale this is glued to the fin stock and cut around. A
    100 mm calibration bar is included so a mis-scaled print is obvious before
    any material is cut.

    Parameters
    ----------
    rocket:
        The design.
    path:
        Destination ``.svg`` path.

    Returns
    -------
    pathlib.Path
        The resolved path written.
    """
    fins = rocket.fins
    root = fins.root_chord * 1e3
    tip = fins.tip_chord * 1e3
    span = fins.span * 1e3
    sweep = fins.sweep_length * 1e3

    margin = 15.0
    width = max(root, sweep + tip) + 2 * margin
    height = span + 2 * margin + 30.0

    points = [
        (margin, margin + span),
        (margin + sweep, margin),
        (margin + sweep + tip, margin),
        (margin + root, margin + span),
    ]
    polygon = " ".join(f"{x:.3f},{y:.3f}" for x, y in points)

    bar_y = height - 12.0
    return _write_svg(
        path,
        width,
        height,
        f"""  <polygon points="{polygon}"
           fill="none" stroke="#000000" stroke-width="0.3"/>

  <!-- Root chord dimension -->
  <line x1="{margin:.3f}" y1="{margin + span + 6:.3f}"
        x2="{margin + root:.3f}" y2="{margin + span + 6:.3f}"
        stroke="#888888" stroke-width="0.2"/>
  <text x="{margin + root / 2:.3f}" y="{margin + span + 11:.3f}"
        font-family="sans-serif" font-size="3.5" text-anchor="middle"
        fill="#444444">root {root:.1f} mm</text>

  <!-- Span dimension -->
  <line x1="{margin - 6:.3f}" y1="{margin:.3f}"
        x2="{margin - 6:.3f}" y2="{margin + span:.3f}"
        stroke="#888888" stroke-width="0.2"/>
  <text x="{margin - 8:.3f}" y="{margin + span / 2:.3f}"
        font-family="sans-serif" font-size="3.5" text-anchor="middle"
        fill="#444444" transform="rotate(-90 {margin - 8:.3f} {margin + span / 2:.3f})"
        >span {span:.1f} mm</text>

  <!-- Print calibration bar -->
  <line x1="{margin:.3f}" y1="{bar_y:.3f}" x2="{margin + 100:.3f}" y2="{bar_y:.3f}"
        stroke="#000000" stroke-width="0.4"/>
  <line x1="{margin:.3f}" y1="{bar_y - 2:.3f}" x2="{margin:.3f}" y2="{bar_y + 2:.3f}"
        stroke="#000000" stroke-width="0.4"/>
  <line x1="{margin + 100:.3f}" y1="{bar_y - 2:.3f}"
        x2="{margin + 100:.3f}" y2="{bar_y + 2:.3f}"
        stroke="#000000" stroke-width="0.4"/>
  <text x="{margin + 50:.3f}" y="{bar_y - 3:.3f}"
        font-family="sans-serif" font-size="3.5" text-anchor="middle"
        fill="#000000">100 mm - measure this before cutting</text>

  <text x="{margin:.3f}" y="{margin - 5:.3f}"
        font-family="sans-serif" font-size="4" fill="#000000"
        >{rocket.name} - fin template, cut {fins.count} from
        {fins.thickness:.2f} mm {fins.material.display_name}</text>
""".replace(f"{fins.thickness:.2f}", f"{fins.thickness * 1e3:.2f}"),
        title=f"{rocket.name} fin template",
    )


def export_profile_svg(rocket: Rocket, path: str | Path) -> Path:
    """Write a scale side elevation of the whole rocket as SVG.

    Parameters
    ----------
    rocket:
        The design.
    path:
        Destination ``.svg`` path.

    Returns
    -------
    pathlib.Path
        The resolved path written.
    """
    fins = rocket.fins
    length_mm = rocket.length * 1e3
    half_mm = max(rocket.reference_radius, fins.span + fins.body_radius) * 1e3

    margin = 12.0
    width = length_mm + 2 * margin
    height = 2 * half_mm + 2 * margin + 20.0
    axis = margin + half_mm

    # Outer silhouette, tip to tail along the top then back along the bottom.
    xs, ys = rocket.nose.profile_points(80)
    upper = [(float(x) * 1e3, float(y) * 1e3) for x, y in zip(xs, ys, strict=True)]
    for placed in rocket.sections:
        section = placed.section
        if isinstance(section, Transition):
            upper.append((placed.position * 1e3, section.fore_radius * 1e3))
            upper.append((placed.aft_position * 1e3, section.aft_radius * 1e3))
        else:
            upper.append((placed.position * 1e3, section.outer_radius * 1e3))
            upper.append((placed.aft_position * 1e3, section.outer_radius * 1e3))

    body_points = " ".join(f"{margin + x:.3f},{axis - y:.3f}" for x, y in upper)
    body_points += " " + " ".join(
        f"{margin + x:.3f},{axis + y:.3f}" for x, y in reversed(upper)
    )

    fin_root = fins.position * 1e3
    fin_shapes = []
    for sign in (-1, 1):
        pts = [
            (fin_root, sign * fins.body_radius * 1e3),
            (fin_root + fins.sweep_length * 1e3, sign * (fins.body_radius + fins.span) * 1e3),
            (
                fin_root + (fins.sweep_length + fins.tip_chord) * 1e3,
                sign * (fins.body_radius + fins.span) * 1e3,
            ),
            (fin_root + fins.root_chord * 1e3, sign * fins.body_radius * 1e3),
        ]
        polygon = " ".join(f"{margin + x:.3f},{axis - y:.3f}" for x, y in pts)
        fin_shapes.append(
            f'  <polygon points="{polygon}" fill="#d0d8e4" '
            f'stroke="#000000" stroke-width="0.3"/>'
        )

    cg = rocket.cg_at(0.0) * 1e3
    from rocketopt.aerodynamics.barrowman import barrowman_analysis

    cp = barrowman_analysis(rocket, mach=0.3).centre_of_pressure * 1e3

    return _write_svg(
        path,
        width,
        height,
        f"""  <polygon points="{body_points}" fill="#eef1f5"
           stroke="#000000" stroke-width="0.4"/>
{chr(10).join(fin_shapes)}
  <line x1="{margin:.3f}" y1="{axis:.3f}" x2="{margin + length_mm:.3f}" y2="{axis:.3f}"
        stroke="#999999" stroke-width="0.2" stroke-dasharray="4,2"/>

  <circle cx="{margin + cg:.3f}" cy="{axis:.3f}" r="2.5"
          fill="#000000" stroke="#000000" stroke-width="0.3"/>
  <text x="{margin + cg:.3f}" y="{axis + 8:.3f}" font-family="sans-serif"
        font-size="3.5" text-anchor="middle" fill="#000000">CG</text>

  <circle cx="{margin + cp:.3f}" cy="{axis:.3f}" r="2.5"
          fill="none" stroke="#0044cc" stroke-width="0.5"/>
  <text x="{margin + cp:.3f}" y="{axis - 6:.3f}" font-family="sans-serif"
        font-size="3.5" text-anchor="middle" fill="#0044cc">CP</text>

  <text x="{margin:.3f}" y="{height - 6:.3f}" font-family="sans-serif"
        font-size="4" fill="#000000"
        >{rocket.name} - {length_mm:.0f} mm long, {rocket.reference_diameter * 1e3:.1f} mm
        diameter, static margin {(cp - cg) / (rocket.reference_diameter * 1e3):.2f} cal</text>
""",
        title=f"{rocket.name} side elevation",
    )


def export_fin_dxf(rocket: Rocket, path: str | Path) -> Path:
    """Write the fin outline as a minimal DXF for laser or CNC cutting.

    A hand-written R12 DXF is emitted rather than pulling in a DXF library.
    R12 is the most widely accepted dialect and a closed ``LWPOLYLINE`` is all
    a cutter needs.

    Parameters
    ----------
    rocket:
        The design.
    path:
        Destination ``.dxf`` path.

    Returns
    -------
    pathlib.Path
        The resolved path written.
    """
    fins = rocket.fins
    points = [
        (0.0, 0.0),
        (fins.sweep_length * 1e3, fins.span * 1e3),
        ((fins.sweep_length + fins.tip_chord) * 1e3, fins.span * 1e3),
        (fins.root_chord * 1e3, 0.0),
    ]

    lines = [
        "0", "SECTION", "2", "HEADER",
        "9", "$INSUNITS", "70", "4",  # 4 = millimetres
        "0", "ENDSEC",
        "0", "SECTION", "2", "ENTITIES",
    ]
    # A closed polyline through the four corners.
    lines += [
        "0", "LWPOLYLINE",
        "8", "FIN",
        "90", str(len(points)),
        "70", "1",  # closed
    ]
    for x, y in points:
        lines += ["10", f"{x:.4f}", "20", f"{y:.4f}"]
    lines += ["0", "ENDSEC", "0", "EOF"]

    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log.info("Wrote fin DXF to %s", file_path)
    return file_path.resolve()


def _write_svg(
    path: str | Path, width: float, height: float, body: str, *, title: str
) -> Path:
    """Write an SVG document sized in millimetres."""
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg"
     width="{width:.3f}mm" height="{height:.3f}mm"
     viewBox="0 0 {width:.3f} {height:.3f}">
  <title>{title}</title>
  <rect width="100%" height="100%" fill="#ffffff"/>
{body}</svg>
"""
    file_path.write_text(content, encoding="utf-8")
    _log.info("Wrote SVG to %s", file_path)
    return file_path.resolve()


def _dedupe(
    points: list[tuple[float, float]], tolerance: float = 1e-6
) -> list[tuple[float, float]]:
    """Remove consecutive coincident points from an open polyline.

    A thin wrapper over :func:`rocketopt.cad.mesh.dedupe_polyline`, kept because
    the CadQuery paths here work in millimetres and want a looser tolerance than
    the mesh kernel's default.

    Parameters
    ----------
    points:
        Polyline vertices in millimetres.
    tolerance:
        Distance below which two consecutive points are treated as one, in
        millimetres.

    Returns
    -------
    list of tuple
        The polyline with consecutive duplicates removed.
    """
    return dedupe_polyline(points, tolerance=tolerance)


def export_stl_parts(
    rocket: Rocket,
    directory: str | Path,
    *,
    tolerances: PrintTolerances = DEFAULT_TOLERANCES,
    validate: bool = True,
) -> list[Path]:
    """Export a printable STL of every part of a design.

    The nose cone comes out with its locating spigot, the motor mount bored to
    the motor and turned to the airframe, and the fin with its true aerodynamic
    section - see :mod:`rocketopt.cad.parts` for the interfaces and clearances.
    Each part is a closed manifold solid in millimetres, oriented with the face
    that should sit on the print bed at ``z = 0``.

    Parameters
    ----------
    rocket:
        The design.
    directory:
        Destination directory, created if needed.
    tolerances:
        Print clearances and tessellation settings.
    validate:
        Run :func:`rocketopt.cad.validation.validate_printable_assembly` and
        refuse to write anything unless every check passes. Turn this off only
        to inspect geometry that is known to be broken.

    Returns
    -------
    list of pathlib.Path
        The STL files written, in assembly order.

    Raises
    ------
    rocketopt.cad.validation.AssemblyValidationError
        If ``validate`` is set and any part would not assemble or would not
        slice. Nothing is written in that case.
    """
    out_dir = Path(directory).expanduser()
    parts = printable_parts(rocket, tolerances)

    if validate:
        validation = validate_printable_assembly(rocket, tolerances, parts)
        if not validation.passed:
            _log.error("Pre-export checks failed:\n%s", validation.report())
        validation.raise_for_failures()
        _log.info(
            "%s passed all %d pre-export checks",
            rocket.name,
            len(validation.checks),
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    written = [
        write_stl(
            part.mesh,
            out_dir / part.filename,
            name=f"{rocket.name} - {part.name}",
        )
        for part in parts
    ]
    _log.info("Exported %d printable STL part(s)", len(written))
    return written


def export_solids(
    rocket: Rocket,
    directory: str | Path,
    *,
    tolerances: PrintTolerances = DEFAULT_TOLERANCES,
) -> list[Path]:
    """Export STEP solids using CadQuery.

    The same profiles the STL exporter revolves and lofts are used here, so the
    STEP bodies carry the nose cone spigot, the airframe-diameter motor mount and
    the true fin section too. STEP is exact BREP geometry rather than a mesh,
    which is what a CAM package or a downstream parametric model wants; STLs come
    from :func:`export_stl_parts` and need no optional dependency.

    Parameters
    ----------
    rocket:
        The design.
    directory:
        Destination directory.
    tolerances:
        Print clearances and profile sampling, matching the STL parts.

    Returns
    -------
    list of pathlib.Path
        Files written.

    Raises
    ------
    ImportError
        If CadQuery is not installed. Callers should catch this and fall back
        to the dependency-free exports.
    """
    import cadquery as cq

    out_dir = Path(directory).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def revolved(profile: list[tuple[float, float]], name: str) -> None:
        """Revolve a ``(z, r)`` profile about Z and write it as STEP."""
        # On the XZ workplane the local x axis is the global X and the local y
        # axis is the global Z, so a (radius, height) point becomes (r, 0, z).
        # The axis of revolution is given in those local coordinates, hence
        # (0, 1, 0) for the global Z axis - the same orientation the mesh
        # exporter builds in.
        solid = (
            cq.Workplane("XZ")
            .polyline([(r, z) for z, r in profile])
            .close()
            .revolve(360, (0, 0, 0), (0, 1, 0))
        )
        path = out_dir / f"{name}.step"
        cq.exporters.export(solid, str(path))
        written.append(path.resolve())

    revolved(nose_cone_profile(rocket, tolerances), "nose_cone")
    revolved(body_tube_profile(rocket), "body_tube")
    revolved(motor_mount_profile(rocket, tolerances), "motor_mount")

    # -- Fin, lofted through its true section ------------------------------
    sections = fin_loft_sections(
        rocket, tolerances, min_chord=_MIN_STEP_TIP_CHORD
    )
    fin = cq.Workplane("XY")
    previous = 0.0
    for height, outline in sections:
        # Workplane offsets are relative to the current plane, so step by the
        # gap to the previous section rather than by the absolute height.
        if height > previous:
            fin = fin.workplane(offset=height - previous)
            previous = height
        fin = fin.polyline(_dedupe(outline)).close()
    fin_solid = fin.loft(ruled=True)
    fin_path = out_dir / "fin.step"
    cq.exporters.export(fin_solid, str(fin_path))
    written.append(fin_path.resolve())

    _log.info("Exported %d CadQuery STEP solid(s)", len(written))
    return written


def export_all(
    rocket: Rocket,
    directory: str | Path,
    *,
    tolerances: PrintTolerances = DEFAULT_TOLERANCES,
) -> list[Path]:
    """Export every available CAD representation of a design.

    Always writes printable STL solids, SVG templates, a DXF fin outline and the
    Fusion 360 script. Adds STEP solids when CadQuery is installed.

    Parameters
    ----------
    rocket:
        The design.
    directory:
        Destination directory, created if needed.
    tolerances:
        Print clearances and tessellation settings for the STL parts.

    Returns
    -------
    list of pathlib.Path
        Every file written.

    Raises
    ------
    rocketopt.cad.validation.AssemblyValidationError
        If the design's parts would not assemble or would not slice. The 2D
        templates are written first and are unaffected, but no STL is.
    """
    from rocketopt.cad.fusion360 import write_fusion_script

    out_dir = Path(directory).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    written = [
        export_fin_template_svg(rocket, out_dir / "fin_template.svg"),
        export_profile_svg(rocket, out_dir / "side_elevation.svg"),
        export_fin_dxf(rocket, out_dir / "fin.dxf"),
        write_fusion_script(rocket, out_dir / "fusion360_build.py"),
    ]

    written.extend(export_stl_parts(rocket, out_dir, tolerances=tolerances))

    try:
        written.extend(export_solids(rocket, out_dir, tolerances=tolerances))
    except ImportError:
        _log.info(
            "CadQuery is not installed; STEP export skipped. The printable STLs "
            "were still written. Install with: pip install 'rocketopt[cad]'"
        )

    return written
