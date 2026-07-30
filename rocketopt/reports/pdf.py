"""PDF report generation.

Produces a typeset engineering report with embedded plots: the trajectory, the
drag breakdown and a scale side elevation with the centre of gravity and
centre of pressure marked.

ReportLab is an optional dependency. :func:`write_pdf_report` raises a clear
``ImportError`` naming the extra to install rather than failing obscurely, and
callers that cannot guarantee it is present should fall back to the Markdown
report in :mod:`rocketopt.reports.engineering`.
"""

from __future__ import annotations

import datetime as _dt
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from rocketopt.aerodynamics.atmosphere import Atmosphere  # noqa: E402
from rocketopt.aerodynamics.barrowman import (  # noqa: E402
    barrowman_analysis,
    stability_verdict,
)
from rocketopt.aerodynamics.drag import drag_buildup  # noqa: E402
from rocketopt.flight.simulation import FlightResult  # noqa: E402
from rocketopt.geometry.components import BodyTube, Transition  # noqa: E402
from rocketopt.geometry.rocket import Rocket  # noqa: E402
from rocketopt.reports.bom import bill_of_materials  # noqa: E402
from rocketopt.structures.analysis import analyse_structure  # noqa: E402
from rocketopt.utils.logging import get_logger  # noqa: E402

_log = get_logger(__name__)

__all__ = ["write_pdf_report", "PDF_AVAILABLE"]


def _reportlab_available() -> bool:
    """Return whether ReportLab can be imported."""
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return False
    return True


PDF_AVAILABLE: bool = _reportlab_available()
"""Whether PDF output is possible in this environment."""


def _figure_to_png(figure: "plt.Figure") -> io.BytesIO:
    """Render a Matplotlib figure to an in-memory PNG buffer."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(figure)
    buffer.seek(0)
    return buffer


def _trajectory_figure(flight: FlightResult) -> io.BytesIO:
    """Plot altitude and speed against time."""
    data = flight.as_arrays()
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))

    axes[0].plot(data["time"], data["altitude"], color="#1f6fb4", linewidth=1.6)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Altitude (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate(
        f"apogee {flight.apogee:.0f} m",
        xy=(flight.apogee_time, flight.apogee),
        xytext=(8, -12),
        textcoords="offset points",
        fontsize=8,
    )

    axes[1].plot(data["time"], data["speed"], color="#2e8b57", linewidth=1.6)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0.0, min(flight.rocket.motor.ejection_time * 1.4,
                              float(data["time"][-1])))

    return _figure_to_png(figure)


def _drag_figure(rocket: Rocket, flight: FlightResult) -> io.BytesIO:
    """Plot the drag breakdown as a horizontal bar chart."""
    state = Atmosphere.standard().state_at(0.0)
    breakdown = drag_buildup(rocket, state, max(flight.max_velocity * 0.6, 20.0))

    items = [(k, v) for k, v in breakdown.as_dict().items() if v > 1e-6]
    items.sort(key=lambda kv: kv[1], reverse=True)

    figure, ax = plt.subplots(figsize=(6.2, 2.8))
    ax.barh([k for k, _ in items], [v for _, v in items], color="#1f6fb4")
    ax.invert_yaxis()
    ax.set_xlabel("$C_D$ contribution")
    ax.set_title(f"Total $C_D$ = {breakdown.total:.3f}", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    for index, (_, value) in enumerate(items):
        ax.text(value, index, f" {value:.4f}", va="center", fontsize=7)

    return _figure_to_png(figure)


def _elevation_figure(rocket: Rocket) -> io.BytesIO:
    """Draw a scale side elevation with CG and CP marked."""
    figure, ax = plt.subplots(figsize=(9.0, 2.2))

    xs, ys = rocket.nose.profile_points(120)
    upper_x = [float(v) * 1e3 for v in xs]
    upper_y = [float(v) * 1e3 for v in ys]

    for placed in rocket.sections:
        section = placed.section
        if isinstance(section, Transition):
            upper_x += [placed.position * 1e3, placed.aft_position * 1e3]
            upper_y += [section.fore_radius * 1e3, section.aft_radius * 1e3]
        else:
            assert isinstance(section, BodyTube)
            upper_x += [placed.position * 1e3, placed.aft_position * 1e3]
            upper_y += [section.outer_radius * 1e3, section.outer_radius * 1e3]

    ax.fill_between(upper_x, upper_y, [-v for v in upper_y],
                    color="#d8dee8", edgecolor="#333333", linewidth=0.8)

    fins = rocket.fins
    root = fins.position * 1e3
    body_r = fins.body_radius * 1e3
    tip_r = body_r + fins.span * 1e3
    fin_x = [
        root,
        root + fins.sweep_length * 1e3,
        root + (fins.sweep_length + fins.tip_chord) * 1e3,
        root + fins.root_chord * 1e3,
    ]
    for sign in (1, -1):
        ax.fill(
            fin_x,
            [sign * body_r, sign * tip_r, sign * tip_r, sign * body_r],
            color="#9fb8d8",
            edgecolor="#333333",
            linewidth=0.8,
        )

    cg = rocket.cg_at(0.0) * 1e3
    cp = barrowman_analysis(rocket, mach=0.3).centre_of_pressure * 1e3
    ax.plot([cg], [0], marker="o", color="black", markersize=7)
    ax.annotate("CG", (cg, 0), xytext=(0, -16), textcoords="offset points",
                ha="center", fontsize=8)
    ax.plot([cp], [0], marker="o", markerfacecolor="none",
            markeredgecolor="#1f6fb4", markersize=8, markeredgewidth=1.6)
    ax.annotate("CP", (cp, 0), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=8, color="#1f6fb4")

    ax.set_xlabel("Station aft of nose tip (mm)")
    ax.set_aspect("equal")
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)

    return _figure_to_png(figure)


def write_pdf_report(
    rocket: Rocket, flight: FlightResult, path: str | Path
) -> Path:
    """Write a typeset PDF engineering report.

    Parameters
    ----------
    rocket:
        The design.
    flight:
        A completed flight simulation of that design.
    path:
        Destination ``.pdf`` path.

    Returns
    -------
    pathlib.Path
        The resolved path written.

    Raises
    ------
    ImportError
        If ReportLab is not installed, naming the extra that provides it.
    """
    if not PDF_AVAILABLE:
        raise ImportError(
            "PDF output needs ReportLab, which is not installed. "
            "Install it with:  pip install 'rocketopt[reports]'\n"
            "The Markdown report from rocketopt.reports.engineering needs no "
            "extra dependencies."
        )

    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        spaceBefore=10,
        spaceAfter=5,
        textColor=colors.HexColor("#12304f"),
    )
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=9, leading=12)
    caption = ParagraphStyle(
        "Caption", parent=body, fontSize=7.5, textColor=colors.grey
    )

    def make_table(rows: list[list[str]], widths: list[float]) -> Table:
        """Build a consistently styled table."""
        table = Table(rows, colWidths=widths, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8edf3")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b6bfcc")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        return table

    analysis = barrowman_analysis(rocket, mach=0.3)
    structure = analyse_structure(
        rocket,
        max_velocity=flight.max_velocity,
        max_dynamic_pressure=flight.max_dynamic_pressure,
        max_acceleration=flight.max_acceleration,
        landing_velocity=flight.landing_velocity,
    )
    cg = rocket.cg_at(0.0)
    margin = (analysis.centre_of_pressure - cg) / rocket.reference_diameter

    story: list[object] = []

    story.append(Paragraph(f"Engineering Report: {rocket.name}", styles["Title"]))
    story.append(
        Paragraph(
            f"Motor {rocket.motor.designation} &nbsp;&middot;&nbsp; "
            f"generated {_dt.date.today().isoformat()} &nbsp;&middot;&nbsp; "
            f"RocketOpt 1.0.0",
            caption,
        )
    )
    story.append(Spacer(1, 6 * mm))

    # -- Elevation ---------------------------------------------------------
    story.append(Image(_elevation_figure(rocket), width=170 * mm, height=41 * mm))
    story.append(
        Paragraph(
            "Scale side elevation. The centre of pressure must lie aft of the "
            "centre of gravity for static stability.",
            caption,
        )
    )
    story.append(Spacer(1, 4 * mm))

    # -- Summary -----------------------------------------------------------
    story.append(Paragraph("1. Performance summary", heading))
    story.append(
        make_table(
            [
                ["Quantity", "Value"],
                ["Apogee", f"{flight.apogee:.1f} m"],
                ["Maximum velocity",
                 f"{flight.max_velocity:.1f} m/s (Mach {flight.max_mach:.2f})"],
                ["Maximum acceleration",
                 f"{flight.max_acceleration / 9.80665:.1f} g"],
                ["Rail exit velocity", f"{flight.rail_exit_velocity:.1f} m/s"],
                ["Thrust-to-weight ratio", f"{rocket.thrust_to_weight:.1f}"],
                ["Static margin", f"{margin:.2f} cal ({stability_verdict(margin)})"],
                ["Lift-off mass", f"{rocket.loaded_mass * 1e3:.1f} g"],
                ["Dry mass", f"{rocket.dry_mass * 1e3:.1f} g"],
                ["Descent rate", f"{flight.landing_velocity:.1f} m/s"],
                ["Flight duration", f"{flight.flight_duration:.0f} s"],
            ],
            [70 * mm, 60 * mm],
        )
    )

    # -- Geometry ----------------------------------------------------------
    story.append(Paragraph("2. Geometry", heading))
    fins = rocket.fins
    story.append(
        make_table(
            [
                ["Parameter", "Value"],
                ["Overall length", f"{rocket.length * 1e3:.1f} mm"],
                ["Body diameter", f"{rocket.reference_diameter * 1e3:.2f} mm"],
                ["Fineness ratio", f"{rocket.fineness_ratio:.1f}"],
                ["Nose", f"{rocket.nose.shape.label}, "
                         f"{rocket.nose.length * 1e3:.1f} mm"],
                ["Nose material", rocket.nose.material.display_name],
                ["Fin count", str(fins.count)],
                ["Fin planform",
                 f"root {fins.root_chord * 1e3:.1f} / tip {fins.tip_chord * 1e3:.1f} "
                 f"/ span {fins.span * 1e3:.1f} mm"],
                ["Fin thickness", f"{fins.thickness * 1e3:.2f} mm"],
                ["Fin material", fins.material.display_name],
                ["Fin section", fins.airfoil.label],
                ["Surface finish", rocket.surface_finish.label],
            ],
            [70 * mm, 60 * mm],
        )
    )

    story.append(PageBreak())

    # -- Trajectory --------------------------------------------------------
    story.append(Paragraph("3. Flight simulation", heading))
    story.append(Image(_trajectory_figure(flight), width=170 * mm, height=57 * mm))
    story.append(
        Paragraph(
            "Six-degree-of-freedom integration including changing mass, thrust "
            "and centre of gravity, the launch rail constraint, wind shear and "
            "recovery deployment.",
            caption,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        make_table(
            [
                ["Event", "Time (s)", "Altitude (m)", "Velocity (m/s)"],
                ["Rail exit", f"{flight.rail_exit_time:.3f}", "-",
                 f"{flight.rail_exit_velocity:.1f}"],
                ["Burnout", f"{rocket.motor.motor.burn_time:.2f}",
                 f"{flight.burnout_altitude:.1f}", f"{flight.burnout_velocity:.1f}"],
                ["Apogee", f"{flight.apogee_time:.2f}", f"{flight.apogee:.1f}", "0.0"],
                ["Ejection", f"{rocket.motor.ejection_time:.2f}",
                 f"{flight.ejection_altitude:.1f}", "-"],
                ["Landing", f"{flight.landing_time:.1f}", "0.0",
                 f"{flight.landing_velocity:.1f}"],
            ],
            [40 * mm, 30 * mm, 35 * mm, 35 * mm],
        )
    )

    # -- Aerodynamics ------------------------------------------------------
    story.append(Paragraph("4. Aerodynamics", heading))
    story.append(Image(_drag_figure(rocket, flight), width=130 * mm, height=59 * mm))
    story.append(
        Paragraph(
            "Drag decomposed by physical mechanism, computed by the Barrowman "
            "method with Hoerner component correlations.",
            caption,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(
        make_table(
            [["Component", "CN alpha (1/rad)", "X_cp (mm)"]]
            + [
                [c.name, f"{c.cn_alpha:.3f}", f"{c.centre_of_pressure * 1e3:.1f}"]
                for c in analysis.contributions
            ]
            + [["Total", f"{analysis.cn_alpha:.3f}",
                f"{analysis.centre_of_pressure * 1e3:.1f}"]],
            [65 * mm, 40 * mm, 35 * mm],
        )
    )

    story.append(PageBreak())

    # -- Structures --------------------------------------------------------
    story.append(Paragraph("5. Structural margins", heading))
    story.append(
        Paragraph(
            f"Safety factor {structure.safety_factor:.2f} applied to all material "
            f"allowables. Margin of safety is allowable / applied &minus; 1.",
            body,
        )
    )
    check_rows = [["Check", "Applied", "Allowable", "MS", "Verdict"]]
    for check in structure.checks:
        margin_text = (
            "inf" if check.margin_of_safety == float("inf")
            else f"{check.margin_of_safety:+.2f}"
        )
        check_rows.append(
            [
                check.name,
                f"{check.applied:.3e}",
                f"{check.allowable:.3e}",
                margin_text,
                "PASS" if check.passes else "FAIL",
            ]
        )
    story.append(make_table(check_rows, [48 * mm, 30 * mm, 30 * mm, 22 * mm, 20 * mm]))

    story.append(Spacer(1, 3 * mm))
    story.append(
        Paragraph(
            f"Fin flutter velocity {structure.flutter.flutter_velocity:.0f} m/s "
            f"against a peak flight speed of "
            f"{structure.flutter.max_flight_velocity:.0f} m/s "
            f"({'SAFE' if structure.flutter.is_safe else 'UNSAFE'}). Flutter speed "
            f"scales with fin thickness to the power 1.5 and with the square root "
            f"of shear modulus, so thickness and material stiffness are the two "
            f"effective remedies. A 50% margin is required because the NACA "
            f"TN 4197 correlation carries roughly &plusmn;20% scatter and the "
            f"failure is sudden and total.",
            body,
        )
    )

    # -- Bill of materials -------------------------------------------------
    story.append(Paragraph("6. Bill of materials", heading))
    items = bill_of_materials(rocket)
    bom_rows = [["Item", "Specification", "Qty", "Mass (g)"]]
    for item in items:
        bom_rows.append(
            [
                item.name,
                Paragraph(item.specification, ParagraphStyle("s", fontSize=7,
                                                             leading=8.5)),
                f"{item.quantity:g} {item.units}",
                f"{item.mass * 1e3:.2f}",
            ]
        )
    bom_rows.append(
        ["Total", "", "", f"{sum(i.mass for i in items) * 1e3:.2f}"]
    )
    story.append(make_table(bom_rows, [30 * mm, 85 * mm, 22 * mm, 23 * mm]))

    # -- Warnings ----------------------------------------------------------
    if flight.warnings:
        story.append(Paragraph("7. Warnings", heading))
        for warning in flight.warnings:
            story.append(Paragraph(f"&bull; {warning}", body))

    document = SimpleDocTemplate(
        str(file_path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"RocketOpt engineering report: {rocket.name}",
        author="RocketOpt 1.0.0",
    )
    document.build(story)

    _log.info("Wrote PDF report to %s", file_path)
    return file_path.resolve()
