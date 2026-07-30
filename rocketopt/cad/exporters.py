"""CAD export: solid models, 2D profiles and manufacturing templates.

Two tiers of output, so that useful CAD is always produced:

*Always available* - SVG fin templates, an SVG side elevation and a DXF fin
outline are written with no third-party dependency. The fin template is the
one output a builder genuinely needs: printed at 1:1 it is glued to the fin
stock and cut around.

*With CadQuery installed* - STEP and STL solids of every component.

The Fusion 360 script from :mod:`rocketopt.cad.fusion360` is always written
too, and is the better route into parametric CAD than a STEP import.
"""

from __future__ import annotations

import math
from pathlib import Path

from rocketopt.geometry.components import BodyTube, Transition
from rocketopt.geometry.rocket import Rocket
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "export_fin_template_svg",
    "export_profile_svg",
    "export_fin_dxf",
    "export_solids",
    "export_all",
]

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
    """Remove consecutive coincident points from a polyline.

    A zero-length edge makes the OCC kernel raise ``StdFail_NotDone`` rather
    than skipping it, so any polyline handed to CadQuery must be free of
    repeated vertices. Tolerance is in the polyline's own units, millimetres
    here.

    Parameters
    ----------
    points:
        Polyline vertices.
    tolerance:
        Distance below which two consecutive points are treated as one.

    Returns
    -------
    list of tuple
        The polyline with consecutive duplicates removed.
    """
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if cleaned and math.hypot(
            point[0] - cleaned[-1][0], point[1] - cleaned[-1][1]
        ) < tolerance:
            continue
        cleaned.append(point)
    return cleaned


def export_solids(rocket: Rocket, directory: str | Path) -> list[Path]:
    """Export STEP and STL solids using CadQuery.

    Parameters
    ----------
    rocket:
        The design.
    directory:
        Destination directory.

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

    # -- Nose cone: revolve the analytic profile ---------------------------
    xs, ys = rocket.nose.profile_points(120)
    profile = [(float(x) * 1e3, float(y) * 1e3) for x, y in zip(xs, ys, strict=True)]
    # Close the profile back along the axis. The profile already begins at the
    # tip (0, 0), so no extra start point is prepended - doing so creates a
    # duplicated vertex, and CadQuery raises StdFail_NotDone rather than
    # silently ignoring the resulting zero-length edge.
    outline = _dedupe([*profile, (profile[-1][0], 0.0)])

    nose_solid = (
        cq.Workplane("XZ")
        .polyline(outline)
        .close()
        .revolve(360, (0, 0, 0), (1, 0, 0))
    )
    nose_path = out_dir / "nose_cone.step"
    cq.exporters.export(nose_solid, str(nose_path))
    written.append(nose_path.resolve())

    # -- Body tube ---------------------------------------------------------
    body = next(
        (p.section for p in rocket.sections if isinstance(p.section, BodyTube)), None
    )
    if body is not None:
        tube = (
            cq.Workplane("XY")
            .circle(body.outer_radius * 1e3)
            .circle(body.inner_radius * 1e3)
            .extrude(body.length * 1e3)
        )
        tube_path = out_dir / "body_tube.step"
        cq.exporters.export(tube, str(tube_path))
        written.append(tube_path.resolve())

    # -- Fin ----------------------------------------------------------------
    fins = rocket.fins
    # A delta fin has zero tip chord, which collapses two corners onto one.
    fin_outline = _dedupe(
        [
            (0.0, 0.0),
            (fins.sweep_length * 1e3, fins.span * 1e3),
            ((fins.sweep_length + fins.tip_chord) * 1e3, fins.span * 1e3),
            (fins.root_chord * 1e3, 0.0),
        ]
    )
    fin_solid = (
        cq.Workplane("XY")
        .polyline(fin_outline)
        .close()
        .extrude(fins.thickness * 1e3)
    )
    for suffix in ("step", "stl"):
        fin_path = out_dir / f"fin.{suffix}"
        cq.exporters.export(fin_solid, str(fin_path))
        written.append(fin_path.resolve())

    _log.info("Exported %d CadQuery solid file(s)", len(written))
    return written


def export_all(rocket: Rocket, directory: str | Path) -> list[Path]:
    """Export every available CAD representation of a design.

    Always writes SVG templates, a DXF fin outline and the Fusion 360 script.
    Adds STEP and STL solids when CadQuery is installed.

    Parameters
    ----------
    rocket:
        The design.
    directory:
        Destination directory, created if needed.

    Returns
    -------
    list of pathlib.Path
        Every file written.
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

    try:
        written.extend(export_solids(rocket, out_dir))
    except ImportError:
        _log.info(
            "CadQuery is not installed; STEP and STL export skipped. "
            "Install with: pip install 'rocketopt[cad]'"
        )

    return written
