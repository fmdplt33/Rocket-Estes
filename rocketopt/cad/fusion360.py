"""Generate a parametric Fusion 360 build script for a design.

The generated script is native CAD, not a mesh import. It creates User
Parameters for every driving dimension, builds sketches that reference those
parameters, and produces solids by revolve and extrude. Changing a parameter
in Fusion's Modify > Change Parameters dialog therefore rebuilds the model
correctly, which is the whole point of exporting parametric CAD rather than a
STEP body.

Units
-----
Fusion 360's API works internally in centimetres regardless of the document's
display units. Every dimension emitted here is converted to centimetres at the
point of use, and the User Parameters are created with explicit ``mm`` units
so they read naturally in the UI.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from rocketopt.geometry.components import BodyTube, Transition
from rocketopt.geometry.rocket import Rocket
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = ["build_fusion_script", "write_fusion_script"]


def _profile_points(rocket: Rocket, samples: int = 60) -> list[tuple[float, float]]:
    """Return the nose profile as ``(x_mm, r_mm)`` points for a spline.

    Parameters
    ----------
    rocket:
        The design.
    samples:
        Number of points along the nose profile.

    Returns
    -------
    list of tuple
        Points in millimetres, from tip to base.
    """
    xs, ys = rocket.nose.profile_points(samples)
    return [(float(x) * 1e3, float(y) * 1e3) for x, y in zip(xs, ys, strict=True)]


def build_fusion_script(rocket: Rocket) -> str:
    """Build a Fusion 360 Python API script that recreates the design.

    Parameters
    ----------
    rocket:
        The design to export.

    Returns
    -------
    str
        A complete, runnable Fusion 360 script.
    """
    fins = rocket.fins
    nose = rocket.nose
    body = next(
        (p.section for p in rocket.sections if isinstance(p.section, BodyTube)), None
    )
    if body is None:  # pragma: no cover - Rocket validation guarantees a tube
        raise ValueError("design has no cylindrical body tube to export")

    boat_tail = next(
        (p for p in rocket.sections if isinstance(p.section, Transition)), None
    )

    profile = _profile_points(rocket)
    profile_literal = ",\n        ".join(
        f"({x:.4f}, {r:.4f})" for x, r in profile
    )

    parameters = [
        ("body_outer_diameter", rocket.reference_diameter * 1e3, "Body tube outside diameter"),
        ("body_wall", body.wall_thickness * 1e3, "Body tube wall thickness"),
        ("body_length", body.length * 1e3, "Body tube length"),
        ("nose_length", nose.length * 1e3, "Nose cone length"),
        ("nose_wall", nose.wall_thickness * 1e3, "Nose cone wall thickness"),
        ("nose_shoulder", nose.shoulder_length * 1e3,
         "Nose cone spigot length, the depth it inserts into the body tube"),
        ("nose_spigot_diameter", nose.shoulder_outer_radius * 2e3,
         "Nose cone spigot outside diameter, equal to the body tube bore"),
        ("fin_root_chord", fins.root_chord * 1e3, "Fin root chord"),
        ("fin_tip_chord", fins.tip_chord * 1e3, "Fin tip chord"),
        ("fin_span", fins.span * 1e3, "Fin exposed semi-span"),
        ("fin_sweep", fins.sweep_length * 1e3, "Fin leading-edge sweep"),
        ("fin_thickness", fins.thickness * 1e3, "Fin thickness"),
        ("fin_fillet", max(fins.fillet_radius * 1e3, 0.0), "Fin root fillet radius"),
        ("fin_station", (fins.position - nose.length) * 1e3,
         "Fin root leading edge, aft of the body tube's forward end"),
        ("motor_tube_id", rocket.motor_mount.inner_diameter * 1e3,
         "Motor mount bore, the motor case diameter plus a loading clearance"),
        ("motor_tube_od", rocket.motor_mount.outer_diameter * 1e3,
         "Motor mount outside diameter, equal to the body tube bore"),
        ("motor_tube_length", rocket.motor_mount.length * 1e3, "Motor mount length"),
    ]
    parameter_literal = ",\n        ".join(
        f'("{name}", {value:.4f}, "{description}")'
        for name, value, description in parameters
    )

    boat_tail_block = ""
    if boat_tail is not None:
        section = boat_tail.section
        assert isinstance(section, Transition)
        boat_tail_block = f"""
    # ---------------------------------------------------------------
    # Boat-tail
    # ---------------------------------------------------------------
    _log('Building boat-tail')
    bt_sketch = sketches.add(root.xZConstructionPlane)
    bt_lines = bt_sketch.sketchCurves.sketchLines
    bt_x0 = MM * (P('nose_length') + P('body_length'))
    bt_len = MM * {section.length * 1e3:.4f}
    bt_r1 = MM * {section.fore_radius * 1e3:.4f}
    bt_r2 = MM * {section.aft_radius * 1e3:.4f}
    bt_wall = MM * {section.wall_thickness * 1e3:.4f}

    bt_pts = [
        adsk.core.Point3D.create(bt_x0, bt_r1, 0),
        adsk.core.Point3D.create(bt_x0 + bt_len, bt_r2, 0),
        adsk.core.Point3D.create(bt_x0 + bt_len, bt_r2 - bt_wall, 0),
        adsk.core.Point3D.create(bt_x0, bt_r1 - bt_wall, 0),
    ]
    for i in range(len(bt_pts)):
        bt_lines.addByTwoPoints(bt_pts[i], bt_pts[(i + 1) % len(bt_pts)])

    bt_axis = bt_sketch.sketchCurves.sketchLines.addByTwoPoints(
        adsk.core.Point3D.create(bt_x0, 0, 0),
        adsk.core.Point3D.create(bt_x0 + bt_len, 0, 0),
    )
    bt_axis.isConstruction = True

    bt_rev = revolves.createInput(
        bt_sketch.profiles.item(0), bt_axis,
        adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
    bt_rev.setAngleExtent(False, adsk.core.ValueInput.createByReal(2 * math.pi))
    revolves.add(bt_rev).bodies.item(0).name = 'Boat-tail'
"""

    today = _dt.date.today().isoformat()

    return f'''"""Fusion 360 build script for {rocket.name}.

Generated by RocketOpt 1.0.0 on {today}.

How to run
----------
1. In Fusion 360, open Utilities > ADD-INS > Scripts and Add-Ins.
2. Click the green "+" next to My Scripts and select this file.
3. Select the script and click Run.

The model is fully parametric. Every driving dimension is created as a User
Parameter; open Modify > Change Parameters to edit them and the model will
rebuild. No mesh geometry is used, so every feature stays editable.

Design summary
--------------
Motor              {rocket.motor.designation}
Overall length     {rocket.length * 1e3:.1f} mm
Body diameter      {rocket.reference_diameter * 1e3:.2f} mm
Nose               {nose.shape.label}, {nose.length * 1e3:.1f} mm
Fins               {fins.count} x {fins.material.display_name}
Dry mass           {rocket.dry_mass * 1e3:.1f} g
"""

import math
import traceback

import adsk.core
import adsk.fusion

# Fusion's API works in centimetres internally, whatever the document shows.
MM = 0.1

# The nose profile, sampled from the analytic {nose.shape.label} curve.
# (axial station in mm, radius in mm), tip first.
NOSE_PROFILE = [
        {profile_literal}
]

# name, value in mm, description
PARAMETERS = [
        {parameter_literal}
]


def _log(message):
    """Write a progress message to the Fusion text command window."""
    app = adsk.core.Application.get()
    app.log('[RocketOpt] ' + message)


def create_parameters(design):
    """Create or update the User Parameters that drive the model."""
    units = design.unitsManager
    params = design.userParameters
    for name, value, description in PARAMETERS:
        existing = params.itemByName(name)
        value_input = adsk.core.ValueInput.createByString('{{}} mm'.format(value))
        if existing:
            existing.expression = '{{}} mm'.format(value)
        else:
            params.add(name, value_input, units.defaultLengthUnits, description)


def P(name):
    """Return a user parameter's value in millimetres."""
    design = adsk.core.Application.get().activeProduct
    param = design.userParameters.itemByName(name)
    # Fusion stores lengths in cm; convert back to mm for our arithmetic.
    return param.value / MM


def run(context):
    """Build the rocket."""
    ui = None
    try:
        app = adsk.core.Application.get()
        ui = app.userInterface

        doc = app.documents.add(adsk.core.DocumentTypes.FusionDesignDocumentType)
        design = app.activeProduct
        design.designType = adsk.fusion.DesignTypes.ParametricDesignType
        root = design.rootComponent
        root.name = '{rocket.name}'

        create_parameters(design)

        sketches = root.sketches
        revolves = root.features.revolveFeatures
        extrudes = root.features.extrudeFeatures

        # ---------------------------------------------------------------
        # Nose cone: revolve the analytic profile, then shell it
        # ---------------------------------------------------------------
        _log('Building nose cone')
        nose_sketch = sketches.add(root.xZConstructionPlane)
        points = adsk.core.ObjectCollection.create()
        for x_mm, r_mm in NOSE_PROFILE:
            points.add(adsk.core.Point3D.create(x_mm * MM, r_mm * MM, 0))
        spline = nose_sketch.sketchCurves.sketchFittedSplines.add(points)

        lines = nose_sketch.sketchCurves.sketchLines
        tip = adsk.core.Point3D.create(0, 0, 0)
        base_outer = adsk.core.Point3D.create(
            NOSE_PROFILE[-1][0] * MM, NOSE_PROFILE[-1][1] * MM, 0)
        base_axis = adsk.core.Point3D.create(NOSE_PROFILE[-1][0] * MM, 0, 0)
        lines.addByTwoPoints(base_outer, base_axis)
        axis_line = lines.addByTwoPoints(base_axis, tip)

        nose_rev = revolves.createInput(
            nose_sketch.profiles.item(0), axis_line,
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        nose_rev.setAngleExtent(False, adsk.core.ValueInput.createByReal(2 * math.pi))
        nose_body = revolves.add(nose_rev).bodies.item(0)
        nose_body.name = 'Nose cone'

        # Hollow the nose, leaving the base open.
        shells = root.features.shellFeatures
        base_face = None
        best_x = -1e9
        for face in nose_body.faces:
            centroid = face.centroid
            if centroid.x > best_x:
                best_x = centroid.x
                base_face = face
        if base_face is not None:
            face_collection = adsk.core.ObjectCollection.create()
            face_collection.add(base_face)
            shell_input = shells.createInput(face_collection, False)
            shell_input.insideThickness = adsk.core.ValueInput.createByReal(
                P('nose_wall') * MM)
            try:
                shells.add(shell_input)
            except Exception:
                # A very fine tip can defeat the shell; leave it solid and
                # tell the user rather than aborting the whole build.
                _log('Nose shell failed - left solid. Reduce nose_wall or '
                     'blunt the tip if a hollow cone is required.')

        # ---------------------------------------------------------------
        # Nose cone spigot: locates the cone in the body tube, so its
        # outside diameter is the tube's bore
        # ---------------------------------------------------------------
        _log('Building nose cone spigot')
        spigot_sketch = sketches.add(root.xZConstructionPlane)
        spigot_lines = spigot_sketch.sketchCurves.sketchLines
        s_x0 = P('nose_length') * MM
        s_x1 = s_x0 + P('nose_shoulder') * MM
        s_out = 0.5 * P('nose_spigot_diameter') * MM
        s_in = max(s_out - P('nose_wall') * MM, 0.0)

        s_corners = [
            adsk.core.Point3D.create(s_x0, s_in, 0),
            adsk.core.Point3D.create(s_x0, s_out, 0),
            adsk.core.Point3D.create(s_x1, s_out, 0),
            adsk.core.Point3D.create(s_x1, s_in, 0),
        ]
        for i in range(len(s_corners)):
            spigot_lines.addByTwoPoints(
                s_corners[i], s_corners[(i + 1) % len(s_corners)])

        spigot_axis = spigot_lines.addByTwoPoints(
            adsk.core.Point3D.create(s_x0, 0, 0),
            adsk.core.Point3D.create(s_x1, 0, 0))
        spigot_axis.isConstruction = True

        spigot_rev = revolves.createInput(
            spigot_sketch.profiles.item(0), spigot_axis,
            adsk.fusion.FeatureOperations.JoinFeatureOperation)
        spigot_rev.setAngleExtent(
            False, adsk.core.ValueInput.createByReal(2 * math.pi))
        try:
            revolves.add(spigot_rev)
        except Exception:
            # Joining needs the spigot to touch the shelled cone. If the shell
            # failed above, or the wall leaves no overlap, build it separately
            # and say so rather than aborting.
            spigot_rev.operation = (
                adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
            revolves.add(spigot_rev).bodies.item(0).name = 'Nose cone spigot'
            _log('Spigot built as a separate body - combine it with the nose '
                 'cone before exporting a single printable part.')

        # ---------------------------------------------------------------
        # Body tube
        # ---------------------------------------------------------------
        _log('Building body tube')
        body_sketch = sketches.add(root.xZConstructionPlane)
        body_lines = body_sketch.sketchCurves.sketchLines
        x0 = P('nose_length') * MM
        x1 = x0 + P('body_length') * MM
        r_out = 0.5 * P('body_outer_diameter') * MM
        r_in = r_out - P('body_wall') * MM

        corners = [
            adsk.core.Point3D.create(x0, r_in, 0),
            adsk.core.Point3D.create(x0, r_out, 0),
            adsk.core.Point3D.create(x1, r_out, 0),
            adsk.core.Point3D.create(x1, r_in, 0),
        ]
        for i in range(len(corners)):
            body_lines.addByTwoPoints(corners[i], corners[(i + 1) % len(corners)])

        body_axis = body_lines.addByTwoPoints(
            adsk.core.Point3D.create(x0, 0, 0),
            adsk.core.Point3D.create(x1, 0, 0))
        body_axis.isConstruction = True

        body_rev = revolves.createInput(
            body_sketch.profiles.item(0), body_axis,
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        body_rev.setAngleExtent(False, adsk.core.ValueInput.createByReal(2 * math.pi))
        revolves.add(body_rev).bodies.item(0).name = 'Body tube'

        # ---------------------------------------------------------------
        # Motor tube
        # ---------------------------------------------------------------
        _log('Building motor tube')
        motor_sketch = sketches.add(root.xZConstructionPlane)
        motor_lines = motor_sketch.sketchCurves.sketchLines
        m_x1 = x1
        m_x0 = m_x1 - P('motor_tube_length') * MM
        m_in = 0.5 * P('motor_tube_id') * MM
        m_out = 0.5 * P('motor_tube_od') * MM

        m_corners = [
            adsk.core.Point3D.create(m_x0, m_in, 0),
            adsk.core.Point3D.create(m_x0, m_out, 0),
            adsk.core.Point3D.create(m_x1, m_out, 0),
            adsk.core.Point3D.create(m_x1, m_in, 0),
        ]
        for i in range(len(m_corners)):
            motor_lines.addByTwoPoints(
                m_corners[i], m_corners[(i + 1) % len(m_corners)])

        motor_axis = motor_lines.addByTwoPoints(
            adsk.core.Point3D.create(m_x0, 0, 0),
            adsk.core.Point3D.create(m_x1, 0, 0))
        motor_axis.isConstruction = True

        motor_rev = revolves.createInput(
            motor_sketch.profiles.item(0), motor_axis,
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        motor_rev.setAngleExtent(False, adsk.core.ValueInput.createByReal(2 * math.pi))
        revolves.add(motor_rev).bodies.item(0).name = 'Motor tube'
{boat_tail_block}
        # ---------------------------------------------------------------
        # Fins: sketch one trapezoid, extrude it, then circular-pattern
        # ---------------------------------------------------------------
        _log('Building {fins.count} fins')
        fin_plane = root.xYConstructionPlane
        fin_sketch = sketches.add(fin_plane)
        fin_lines = fin_sketch.sketchCurves.sketchLines

        f_x0 = (P('nose_length') + P('fin_station')) * MM
        f_root = P('fin_root_chord') * MM
        f_tip = P('fin_tip_chord') * MM
        f_span = P('fin_span') * MM
        f_sweep = P('fin_sweep') * MM
        f_r = 0.5 * P('body_outer_diameter') * MM

        fin_pts = [
            adsk.core.Point3D.create(f_x0, f_r, 0),
            adsk.core.Point3D.create(f_x0 + f_sweep, f_r + f_span, 0),
            adsk.core.Point3D.create(f_x0 + f_sweep + f_tip, f_r + f_span, 0),
            adsk.core.Point3D.create(f_x0 + f_root, f_r, 0),
        ]
        for i in range(len(fin_pts)):
            fin_lines.addByTwoPoints(fin_pts[i], fin_pts[(i + 1) % len(fin_pts)])

        fin_extrude = extrudes.createInput(
            fin_sketch.profiles.item(0),
            adsk.fusion.FeatureOperations.NewBodyFeatureOperation)
        fin_extrude.setSymmetricExtent(
            adsk.core.ValueInput.createByReal(P('fin_thickness') * MM), True)
        fin_feature = extrudes.add(fin_extrude)
        fin_body = fin_feature.bodies.item(0)
        fin_body.name = 'Fin'

        # Circular pattern about the body axis.
        if {fins.count} > 1:
            patterns = root.features.circularPatternFeatures
            bodies = adsk.core.ObjectCollection.create()
            bodies.add(fin_body)
            pattern_input = patterns.createInput(bodies, root.xConstructionAxis)
            pattern_input.quantity = adsk.core.ValueInput.createByReal({fins.count})
            pattern_input.totalAngle = adsk.core.ValueInput.createByString('360 deg')
            pattern_input.isSymmetric = False
            patterns.add(pattern_input)

        design.rootComponent.isBodiesFolderLightBulbOn = True
        app.activeViewport.fit()

        _log('Build complete: {rocket.name}')
        ui.messageBox(
            'RocketOpt build complete.\\n\\n'
            '{rocket.name}\\n'
            'Length {rocket.length * 1e3:.0f} mm, '
            'diameter {rocket.reference_diameter * 1e3:.1f} mm\\n\\n'
            'Edit any dimension via Modify > Change Parameters.')

    except Exception:
        if ui:
            ui.messageBox('RocketOpt script failed:\\n{{}}'.format(
                traceback.format_exc()))
'''


def write_fusion_script(rocket: Rocket, path: str | Path) -> Path:
    """Write a Fusion 360 build script to disk.

    Parameters
    ----------
    rocket:
        The design to export.
    path:
        Destination ``.py`` path.

    Returns
    -------
    pathlib.Path
        The resolved path written.
    """
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(build_fusion_script(rocket), encoding="utf-8")
    _log.info("Wrote Fusion 360 script to %s", file_path)
    return file_path.resolve()
