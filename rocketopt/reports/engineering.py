"""Engineering report generation in Markdown.

The report is written to be handed to someone who was not present when the
design was produced. It therefore states the flight condition every number was
computed at, cites the model behind each section, and reproduces any modelling
caveats - such as a synthesised thrust curve - rather than hiding them.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

from rocketopt.aerodynamics.atmosphere import Atmosphere
from rocketopt.aerodynamics.barrowman import (
    barrowman_analysis,
    damping_derivatives,
    stability_verdict,
)
from rocketopt.aerodynamics.drag import drag_buildup
from rocketopt.flight.simulation import FlightResult
from rocketopt.geometry.components import BodyTube
from rocketopt.geometry.rocket import Rocket
from rocketopt.reports.bom import bill_of_materials
from rocketopt.structures.analysis import analyse_structure
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = ["build_markdown_report", "write_markdown_report"]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a Markdown table."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def build_markdown_report(rocket: Rocket, flight: FlightResult) -> str:
    """Build a complete engineering report as Markdown text.

    Parameters
    ----------
    rocket:
        The design.
    flight:
        A completed flight simulation of that design.

    Returns
    -------
    str
        The report.
    """
    today = _dt.date.today().isoformat()
    motor = rocket.motor.motor

    analysis = barrowman_analysis(rocket, mach=0.3)
    damping = damping_derivatives(rocket, t=0.0, mach=0.3)
    structure = analyse_structure(
        rocket,
        max_velocity=flight.max_velocity,
        max_dynamic_pressure=flight.max_dynamic_pressure,
        max_acceleration=flight.max_acceleration,
        landing_velocity=flight.landing_velocity,
    )

    atmosphere = Atmosphere.standard()
    state = atmosphere.state_at(0.0)
    reference_speed = max(flight.max_velocity * 0.6, 20.0)
    drag = drag_buildup(rocket, state, reference_speed)

    cg_loaded = rocket.cg_at(0.0)
    cg_burnout = rocket.cg_at(motor.burn_time)
    calibre = rocket.reference_diameter
    margin_loaded = (analysis.centre_of_pressure - cg_loaded) / calibre
    margin_burnout = (analysis.centre_of_pressure - cg_burnout) / calibre

    parts: list[str] = []

    # -- Header -------------------------------------------------------------
    parts.append(
        f"""# Engineering Report: {rocket.name}

**Motor:** {rocket.motor.designation} &nbsp;&nbsp;
**Generated:** {today} &nbsp;&nbsp;
**Tool:** RocketOpt 1.0.0

All quantities are SI unless stated. Aerodynamic coefficients are referenced
to the body cross-sectional area of {rocket.reference_area * 1e4:.2f} cm² and
the body diameter of {calibre * 1e3:.1f} mm, following Barrowman's convention.
"""
    )

    # -- Summary ------------------------------------------------------------
    parts.append(
        f"""## 1. Summary

| Quantity | Value |
|---|---|
| Apogee | **{flight.apogee:.1f} m** |
| Maximum velocity | {flight.max_velocity:.1f} m/s (Mach {flight.max_mach:.2f}) |
| Maximum acceleration | {flight.max_acceleration:.0f} m/s² ({flight.max_acceleration / 9.80665:.1f} g) |
| Rail exit velocity | {flight.rail_exit_velocity:.1f} m/s |
| Static margin (loaded / burnout) | {margin_loaded:.2f} / {margin_burnout:.2f} calibres |
| Stability verdict | {stability_verdict(min(margin_loaded, margin_burnout))} |
| Lift-off mass | {rocket.loaded_mass * 1e3:.1f} g |
| Thrust-to-weight ratio | {rocket.thrust_to_weight:.1f} |
| Coasting drag coefficient | {drag.total:.3f} |
| Descent rate | {flight.landing_velocity:.1f} m/s |
| Flight duration | {flight.flight_duration:.0f} s |
"""
    )

    # -- Geometry -----------------------------------------------------------
    body = next(
        (p.section for p in rocket.sections if isinstance(p.section, BodyTube)), None
    )
    geometry_rows = [
        ["Overall length", f"{rocket.length * 1e3:.1f}", "mm"],
        ["Body diameter", f"{calibre * 1e3:.2f}", "mm"],
        ["Fineness ratio", f"{rocket.fineness_ratio:.1f}", "-"],
        ["Nose profile", rocket.nose.shape.label, "-"],
        ["Nose length", f"{rocket.nose.length * 1e3:.1f}", "mm"],
        ["Nose fineness", f"{rocket.nose.fineness_ratio:.2f}", "-"],
        ["Nose material", rocket.nose.material.display_name, "-"],
    ]
    if body is not None:
        geometry_rows += [
            ["Body length", f"{body.length * 1e3:.1f}", "mm"],
            ["Wall thickness", f"{body.wall_thickness * 1e3:.2f}", "mm"],
            ["Body material", body.material.display_name, "-"],
        ]
    fins = rocket.fins
    geometry_rows += [
        ["Fin count", str(fins.count), "-"],
        ["Fin root chord", f"{fins.root_chord * 1e3:.1f}", "mm"],
        ["Fin tip chord", f"{fins.tip_chord * 1e3:.1f}", "mm"],
        ["Fin span (exposed)", f"{fins.span * 1e3:.1f}", "mm"],
        ["Fin sweep", f"{fins.sweep_length * 1e3:.1f}", "mm"],
        ["Fin thickness", f"{fins.thickness * 1e3:.2f}", "mm"],
        ["Fin aspect ratio", f"{fins.aspect_ratio:.2f}", "-"],
        ["Fin taper ratio", f"{fins.taper_ratio:.2f}", "-"],
        ["Fin section", fins.airfoil.label, "-"],
        ["Fin material", fins.material.display_name, "-"],
        ["Root fillet", f"{fins.fillet_radius * 1e3:.1f}", "mm"],
        ["Surface finish", rocket.surface_finish.label, "-"],
        ["Wetted area", f"{rocket.wetted_area * 1e4:.1f}", "cm²"],
    ]
    parts.append(
        "## 2. Geometry\n\n"
        + _table(["Parameter", "Value", "Units"], geometry_rows)
    )

    # -- Mass ---------------------------------------------------------------
    loaded = rocket.loaded_mass_properties
    burnout = rocket.burnout_mass_properties
    dry = rocket.dry_mass_properties
    parts.append(
        "## 3. Mass properties\n\n"
        + _table(
            ["Condition", "Mass (g)", "CG (mm aft of tip)", "Ixx (g·cm²)", "Iyy (g·cm²)"],
            [
                ["Dry (no motor)", f"{dry.mass * 1e3:.1f}", f"{dry.cg * 1e3:.1f}",
                 f"{dry.ixx * 1e7:.1f}", f"{dry.iyy * 1e7:.1f}"],
                ["Lift-off", f"{loaded.mass * 1e3:.1f}", f"{loaded.cg * 1e3:.1f}",
                 f"{loaded.ixx * 1e7:.1f}", f"{loaded.iyy * 1e7:.1f}"],
                ["Burnout", f"{burnout.mass * 1e3:.1f}", f"{burnout.cg * 1e3:.1f}",
                 f"{burnout.ixx * 1e7:.1f}", f"{burnout.iyy * 1e7:.1f}"],
            ],
        )
        + f"\n\nThe centre of gravity moves "
        f"{abs(burnout.cg - loaded.cg) * 1e3:.1f} mm "
        f"{'forward' if burnout.cg < loaded.cg else 'aft'} through the burn as "
        f"propellant is consumed, which is why stability is checked at both "
        f"conditions.\n"
    )

    # -- Propulsion ---------------------------------------------------------
    parts.append(
        f"""## 4. Propulsion

| Parameter | Value | Units |
|---|---|---|
| Designation | {rocket.motor.designation} | - |
| Impulse class | {motor.impulse_class} | - |
| Total impulse | {motor.total_impulse:.2f} | N·s |
| Average thrust | {motor.average_thrust:.2f} | N |
| Peak thrust | {motor.peak_thrust:.2f} | N |
| Burn time | {motor.burn_time:.2f} | s |
| Propellant mass | {motor.propellant_mass * 1e3:.2f} | g |
| Loaded mass | {motor.total_mass * 1e3:.2f} | g |
| Ejection delay | {rocket.motor.delay:.0f} | s |
| Thrust curve source | {motor.thrust_curve.source.value} | - |

Data source: {motor.source}
"""
    )
    provenance = motor.provenance_warning()
    if provenance:
        parts.append(f"> **Note on thrust data.** {provenance}\n")

    # -- Aerodynamics -------------------------------------------------------
    contribution_rows = [
        [c.name, f"{c.cn_alpha:.3f}", f"{c.centre_of_pressure * 1e3:.1f}"]
        for c in analysis.contributions
    ]
    contribution_rows.append(
        ["**Total**", f"**{analysis.cn_alpha:.3f}**",
         f"**{analysis.centre_of_pressure * 1e3:.1f}**"]
    )

    drag_rows = [
        [name, f"{value:.4f}", f"{100.0 * value / drag.total:.1f}%"]
        for name, value in sorted(
            drag.as_dict().items(), key=lambda kv: kv[1], reverse=True
        )
        if value > 1e-6
    ]
    drag_rows.append(["**Total**", f"**{drag.total:.4f}**", "100.0%"])

    parts.append(
        f"""## 5. Aerodynamics

Computed by the Barrowman method (Barrowman, 1967) with a Prandtl-Glauert
compressibility correction and an Allen-Perkins viscous cross-flow term.

### 5.1 Normal force and centre of pressure (Mach 0.3)

{_table(["Component", "CNα (1/rad)", "X_cp (mm)"], contribution_rows)}

### 5.2 Drag breakdown at {reference_speed:.0f} m/s (Mach {drag.mach:.2f}, Re {drag.reynolds:.2e})

{_table(["Mechanism", "C_D", "Share"], drag_rows)}

The dominant contributor is **{drag.dominant_mechanism().lower()}**.

### 5.3 Rotational damping (about the lift-off centre of gravity)

| Derivative | Value | Meaning |
|---|---|---|
| C_mq | {damping.cmq:.1f} | Pitch damping |
| C_nr | {damping.cnr:.1f} | Yaw damping |
| C_lp | {damping.clp:.3f} | Roll damping |
| C_lδ | {damping.cl_delta:.3f} | Roll moment per radian of fin cant |
"""
    )
    if drag.validity_warning:
        parts.append(f"> **Validity.** {drag.validity_warning}\n")

    # -- Stability ----------------------------------------------------------
    parts.append(
        f"""## 6. Stability

| Condition | CG (mm) | CP (mm) | Static margin (cal) | Verdict |
|---|---|---|---|---|
| Lift-off | {cg_loaded * 1e3:.1f} | {analysis.centre_of_pressure * 1e3:.1f} | {margin_loaded:.2f} | {stability_verdict(margin_loaded)} |
| Burnout | {cg_burnout * 1e3:.1f} | {analysis.centre_of_pressure * 1e3:.1f} | {margin_burnout:.2f} | {stability_verdict(margin_burnout)} |

A static margin between 1.0 and 2.5 calibres is the accepted band for a
conventional model rocket. Below 1.0 the vehicle is prone to divergence in
gusts; above 2.5 it weathercocks strongly into wind and loses altitude.
"""
    )

    # -- Structures ---------------------------------------------------------
    structure_rows = [
        [
            check.name,
            f"{check.applied:.3e}",
            f"{check.allowable:.3e}",
            check.units,
            "∞" if check.margin_of_safety == float("inf") else f"{check.margin_of_safety:+.2f}",
            "PASS" if check.passes else "**FAIL**",
        ]
        for check in structure.checks
    ]
    parts.append(
        f"""## 7. Structures

Safety factor {structure.safety_factor:.2f} applied to all material allowables.
Margin of safety is defined as allowable / applied − 1.

{_table(["Check", "Applied", "Allowable", "Units", "MS", "Verdict"], structure_rows)}

### 7.1 Fin aeroelasticity

| Quantity | Value |
|---|---|
| Flutter velocity | {structure.flutter.flutter_velocity:.0f} m/s |
| Divergence velocity | {structure.flutter.divergence_velocity:.0f} m/s |
| Peak flight velocity | {structure.flutter.max_flight_velocity:.0f} m/s |
| Verdict | {"SAFE" if structure.flutter.is_safe else "**UNSAFE**"} |

Flutter velocity is computed from NACA TN 4197. It scales with fin thickness
to the power 1.5 and with the square root of the material shear modulus, so
thickness and material stiffness are the two effective remedies. A 50% margin
over peak flight speed is required because the correlation itself carries
roughly ±20% scatter and the failure mode is sudden and total.

Limiting check: **{structure.limiting_check.name}**.
"""
    )

    # -- Flight -------------------------------------------------------------
    parts.append(
        f"""## 8. Flight simulation

Six-degree-of-freedom integration with a quaternion attitude representation,
fixed-step RK4, modelling changing mass, thrust, centre of gravity and drag,
plus the launch rail constraint, wind shear, recovery deployment and descent.

| Event | Time (s) | Altitude (m) | Velocity (m/s) |
|---|---|---|---|
| Rail exit | {flight.rail_exit_time:.3f} | - | {flight.rail_exit_velocity:.1f} |
| Burnout | {motor.burn_time:.2f} | {flight.burnout_altitude:.1f} | {flight.burnout_velocity:.1f} |
| Apogee | {flight.apogee_time:.2f} | {flight.apogee:.1f} | 0.0 |
| Ejection | {rocket.motor.ejection_time:.2f} | {flight.ejection_altitude:.1f} | - |
| Landing | {flight.landing_time:.1f} | 0.0 | {flight.landing_velocity:.1f} |

Ejection fires {abs(flight.apogee_error_vs_ejection):.1f} m
{"below" if flight.apogee_error_vs_ejection > 0 else "above"} apogee.
Maximum dynamic pressure was {flight.max_dynamic_pressure:.0f} Pa.
Landing point {flight.landing_distance:.0f} m from the pad.
"""
    )

    # -- Warnings -----------------------------------------------------------
    if flight.warnings:
        warning_lines = "\n".join(f"- {w}" for w in flight.warnings)
        parts.append(f"## 9. Warnings\n\n{warning_lines}\n")
    else:
        parts.append(
            "## 9. Warnings\n\nNone. The design satisfies every checked limit.\n"
        )

    # -- Bill of materials --------------------------------------------------
    bom = bill_of_materials(rocket)
    bom_rows = [
        [item.name, item.specification, f"{item.quantity:g}", item.units,
         f"{item.mass * 1e3:.2f}"]
        for item in bom
    ]
    total_mass = sum(item.mass for item in bom)
    bom_rows.append(["**Total**", "", "", "", f"**{total_mass * 1e3:.2f}**"])
    parts.append(
        "## 10. Bill of materials\n\n"
        + _table(["Item", "Specification", "Qty", "Units", "Mass (g)"], bom_rows)
    )

    # -- References ---------------------------------------------------------
    parts.append(
        """## 11. References

1. Barrowman, J. S. (1967). *The Practical Calculation of the Aerodynamic
   Characteristics of Slender Finned Vehicles*. M.Sc. thesis, Catholic
   University of America.
2. Barrowman, J. S., & Barrowman, J. A. (1966). *A Method for Calculating the
   Centre of Pressure of Model Rockets*. NARAM-8.
3. Hoerner, S. F. (1965). *Fluid-Dynamic Drag*. Published by the author.
4. Martin, D. J. (1958). *Summary of Flutter Experiences as a Guide to the
   Preliminary Design of Lifting Surfaces on Missiles*. NACA TN 4197.
5. NASA (1968). *Buckling of Thin-Walled Circular Cylinders*. NASA SP-8007.
6. Allen, H. J., & Perkins, E. W. (1951). *A Study of Effects of Viscosity on
   Flow Over Slender Inclined Bodies of Revolution*. NACA Report 1048.
7. U.S. Standard Atmosphere, 1976. NOAA-S/T 76-1562.
8. Crowell, G. A. (1996). *The Descriptive Geometry of Nose Cones*.
9. Niskanen, S. (2013). *OpenRocket Technical Documentation*.
10. Knacke, T. W. (1992). *Parachute Recovery Systems Design Manual*,
    NWC TP 6575.
"""
    )

    return "\n\n".join(parts)


def write_markdown_report(
    rocket: Rocket, flight: FlightResult, path: str | Path
) -> Path:
    """Write an engineering report to a Markdown file.

    Parameters
    ----------
    rocket:
        The design.
    flight:
        A completed flight simulation.
    path:
        Destination path.

    Returns
    -------
    pathlib.Path
        The resolved path written.
    """
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(build_markdown_report(rocket, flight), encoding="utf-8")
    _log.info("Wrote engineering report to %s", file_path)
    return file_path.resolve()
