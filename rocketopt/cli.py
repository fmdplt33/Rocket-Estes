"""Command-line interface for RocketOpt.

Run ``rocketopt --help`` for the full command list. The common workflows are::

    rocketopt motors                        # list the motor database
    rocketopt design C6-5                   # optimise a rocket for a motor
    rocketopt design C6-5 --for altitude --export out/
    rocketopt design C6-5 --bundle alpha.zip   # everything in one file
    rocketopt simulate design.yaml          # fly a saved design
    rocketopt ui                            # open the desktop interface
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from rocketopt.utils.logging import configure_logging

app = typer.Typer(
    name="rocketopt",
    help="Design, analyse and optimise high-performance model rockets.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

__all__ = ["app", "main"]


def _verdict_style(passes: bool) -> str:
    """Return a Rich style for a pass/fail verdict."""
    return "green" if passes else "bold red"


@app.command("motors")
def motors_command(
    impulse_class: Annotated[
        str | None,
        typer.Option("--class", "-c", help="Filter to one impulse class, e.g. C."),
    ] = None,
) -> None:
    """List the Estes motor database."""
    configure_logging(logging.WARNING)
    from rocketopt.propulsion.database import list_motors, motors_by_class

    motors = motors_by_class(impulse_class) if impulse_class else list_motors()
    if not motors:
        console.print(f"[red]No motors found for class {impulse_class!r}.[/red]")
        raise typer.Exit(code=1)

    table = Table(title="Estes motors", header_style="bold")
    for column, justify in (
        ("Motor", "left"),
        ("Class", "center"),
        ("Impulse (N·s)", "right"),
        ("Avg thrust (N)", "right"),
        ("Peak (N)", "right"),
        ("Burn (s)", "right"),
        ("Mass (g)", "right"),
        ("Dia (mm)", "right"),
        ("Delays (s)", "left"),
        ("Curve", "left"),
    ):
        table.add_column(column, justify=justify)  # type: ignore[arg-type]

    for motor in motors:
        table.add_row(
            motor.designation,
            motor.impulse_class,
            f"{motor.total_impulse:.2f}",
            f"{motor.average_thrust:.2f}",
            f"{motor.peak_thrust:.1f}",
            f"{motor.burn_time:.2f}",
            f"{motor.total_mass * 1e3:.1f}",
            f"{motor.diameter * 1e3:.0f}",
            ", ".join(f"{d:g}" for d in motor.delays),
            motor.thrust_curve.source.value,
        )

    console.print(table)
    console.print(
        "[dim]Curve 'synthesised' means the thrust history is modelled from "
        "certified impulse, peak thrust and burn time. Drop a ThrustCurve.org "
        ".eng file into the motors asset folder to use measured data.[/dim]"
    )


@app.command("design")
def design_command(
    motor: Annotated[str, typer.Argument(help="Motor designation, e.g. C6-5.")],
    objective: Annotated[
        str,
        typer.Option(
            "--for",
            "-f",
            help="What to optimise for: altitude, velocity, drag, duration, "
            "light, buildable.",
        ),
    ] = "altitude",
    population: Annotated[
        int, typer.Option("--population", "-p", help="Population size.")
    ] = 40,
    generations: Annotated[
        int, typer.Option("--generations", "-g", help="Number of generations.")
    ] = 20,
    max_length: Annotated[
        float, typer.Option("--max-length", help="Maximum overall length in mm.")
    ] = 1200.0,
    max_diameter: Annotated[
        float, typer.Option("--max-diameter", help="Maximum body diameter in mm.")
    ] = 60.0,
    wind: Annotated[
        float, typer.Option("--wind", help="Wind speed at 10 m in m/s.")
    ] = 0.0,
    elevation: Annotated[
        float, typer.Option("--elevation", help="Site elevation in m.")
    ] = 0.0,
    export: Annotated[
        Path | None,
        typer.Option("--export", "-e", help="Directory to write reports and CAD to."),
    ] = None,
    bundle: Annotated[
        Path | None,
        typer.Option(
            "--bundle",
            "-b",
            help="Write everything into a single .zip instead of loose files.",
        ),
    ] = None,
    seed: Annotated[
        int, typer.Option("--seed", help="Random seed for a reproducible run.")
    ] = 20260729,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Log every generation.")
    ] = False,
) -> None:
    """Design a near-optimal rocket for a motor."""
    configure_logging(logging.INFO if verbose else logging.WARNING)

    from rocketopt.flight.environment import LaunchConditions
    from rocketopt.optimisation.driver import optimise_for_motor
    from rocketopt.optimisation.problem import DesignConstraints, Objective

    objective_map = {
        "altitude": (Objective.APOGEE,),
        "velocity": (Objective.MAX_VELOCITY,),
        "drag": (Objective.LOW_DRAG,),
        "duration": (Objective.FLIGHT_DURATION,),
        "light": (Objective.LIGHT,),
        "buildable": (Objective.APOGEE, Objective.MANUFACTURABILITY),
    }
    key = objective.strip().lower()
    if key not in objective_map:
        console.print(
            f"[red]Unknown objective {objective!r}.[/red] "
            f"Choose from: {', '.join(sorted(objective_map))}"
        )
        raise typer.Exit(code=1)

    constraints = DesignConstraints(
        max_length=max_length * 1e-3,
        max_diameter=max_diameter * 1e-3,
    )
    conditions = LaunchConditions.from_inputs(
        elevation_m=elevation, wind_speed_ms=wind
    )

    console.print(
        Panel(
            f"Optimising for [bold]{key}[/bold] on a [bold]{motor}[/bold]\n"
            f"{population} designs x {generations} generations "
            f"= {population * (generations + 1)} evaluations",
            title="RocketOpt",
            border_style="blue",
        )
    )

    with console.status("Searching the design space...", spinner="dots"):
        try:
            outcome = optimise_for_motor(
                motor,
                objectives=objective_map[key],
                constraints=constraints,
                conditions=conditions,
                population=population,
                generations=generations,
                seed=seed,
                verbose=verbose,
            )
        except (KeyError, RuntimeError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    _print_outcome(outcome)

    if export is not None:
        _export_everything(outcome, export)

    if bundle is not None:
        _export_bundle(outcome, bundle)


def _export_bundle(outcome: object, path: Path) -> None:
    """Write every output for an outcome into a single archive."""
    from rocketopt.bundle import write_bundle

    rocket = outcome.best  # type: ignore[attr-defined]
    flight = outcome.flight  # type: ignore[attr-defined]

    with console.status("Bundling...", spinner="dots"):
        contents = write_bundle(rocket, path, flight=flight)

    console.print(
        f"\n[green]Bundled {contents.file_count} files "
        f"({contents.path.stat().st_size / 1024:.0f} kB) into:[/green]\n"
        f"  {contents.path}"
    )
    console.print(
        "[dim]  print/    STL solids, ready to slice\n"
        "  cad/      STEP solids, templates and the Fusion 360 script\n"
        "  reports/  engineering report and build guide[/dim]"
    )


def _print_outcome(outcome: object) -> None:
    """Print an optimisation outcome as Rich tables."""
    rocket = outcome.best  # type: ignore[attr-defined]
    flight = outcome.flight  # type: ignore[attr-defined]
    structure = outcome.structure  # type: ignore[attr-defined]

    geometry = Table(title="Optimised design", header_style="bold", show_header=False)
    geometry.add_column("Parameter")
    geometry.add_column("Value", justify="right")
    for name, value in (
        ("Motor", rocket.motor.designation),
        ("Overall length", f"{rocket.length * 1e3:.0f} mm"),
        ("Body diameter", f"{rocket.reference_diameter * 1e3:.1f} mm"),
        ("Nose", f"{rocket.nose.shape.label}, {rocket.nose.length * 1e3:.0f} mm"),
        ("Nose material", rocket.nose.material.display_name),
        (
            "Fins",
            f"{rocket.fins.count} x {rocket.fins.material.display_name}, "
            f"{rocket.fins.thickness * 1e3:.2f} mm",
        ),
        ("Fin section", rocket.fins.airfoil.label),
        (
            "Fin planform",
            f"root {rocket.fins.root_chord * 1e3:.0f} / "
            f"tip {rocket.fins.tip_chord * 1e3:.0f} / "
            f"span {rocket.fins.span * 1e3:.0f} mm",
        ),
        ("Surface finish", rocket.surface_finish.label),
        ("Dry mass", f"{rocket.dry_mass * 1e3:.1f} g"),
        ("Lift-off mass", f"{rocket.loaded_mass * 1e3:.1f} g"),
    ):
        geometry.add_row(name, str(value))
    console.print(geometry)

    performance = Table(title="Predicted flight", header_style="bold", show_header=False)
    performance.add_column("Quantity")
    performance.add_column("Value", justify="right")
    performance.add_row("Apogee", f"[bold green]{flight.apogee:.1f} m[/bold green]")
    performance.add_row(
        "Maximum velocity", f"{flight.max_velocity:.1f} m/s (Mach {flight.max_mach:.2f})"
    )
    performance.add_row(
        "Maximum acceleration",
        f"{flight.max_acceleration:.0f} m/s² ({flight.max_acceleration / 9.80665:.1f} g)",
    )
    performance.add_row("Rail exit velocity", f"{flight.rail_exit_velocity:.1f} m/s")
    performance.add_row("Thrust / weight", f"{rocket.thrust_to_weight:.1f}")
    performance.add_row("Static margin", f"{flight.min_static_margin:.2f} calibres")
    performance.add_row("Descent rate", f"{flight.landing_velocity:.1f} m/s")
    performance.add_row("Flight duration", f"{flight.flight_duration:.0f} s")
    console.print(performance)

    checks = Table(title="Structural margins", header_style="bold")
    checks.add_column("Check")
    checks.add_column("Margin", justify="right")
    checks.add_column("Verdict", justify="center")
    for check in structure.checks:
        margin = check.margin_of_safety
        margin_text = "inf" if margin == float("inf") else f"{margin:+.2f}"
        checks.add_row(
            check.name,
            margin_text,
            f"[{_verdict_style(check.passes)}]"
            f"{'PASS' if check.passes else 'FAIL'}[/]",
        )
    flutter_ok = structure.flutter.is_safe
    checks.add_row(
        "Fin flutter",
        f"{structure.flutter.flutter_velocity:.0f} m/s",
        f"[{_verdict_style(flutter_ok)}]{'SAFE' if flutter_ok else 'UNSAFE'}[/]",
    )
    console.print(checks)

    if len(outcome.objectives) >= 2:  # type: ignore[attr-defined]
        rows = outcome.pareto_table()  # type: ignore[attr-defined]
        if rows:
            pareto = Table(
                title=f"Pareto front ({len(rows)} non-dominated designs)",
                header_style="bold",
            )
            for column in rows[0]:
                pareto.add_column(column, justify="right")
            for row in sorted(
                rows, key=lambda r: list(r.values())[0], reverse=True
            )[:10]:
                pareto.add_row(*[f"{v:.3f}" for v in row.values()])
            console.print(pareto)

    if flight.warnings:
        console.print("\n[yellow]Warnings[/yellow]")
        for warning in flight.warnings:
            console.print(f"  [yellow]•[/yellow] {warning}")


def _export_everything(outcome: object, directory: Path) -> None:
    """Write reports and CAD for an outcome."""
    from rocketopt.reports.bom import write_manufacturing_guide
    from rocketopt.reports.engineering import write_markdown_report

    directory.mkdir(parents=True, exist_ok=True)
    rocket = outcome.best  # type: ignore[attr-defined]
    flight = outcome.flight  # type: ignore[attr-defined]

    written: list[Path] = [
        write_markdown_report(rocket, flight, directory / "engineering_report.md"),
        write_manufacturing_guide(rocket, directory / "manufacturing_guide.md"),
    ]

    from rocketopt.reports.pdf import PDF_AVAILABLE, write_pdf_report

    if PDF_AVAILABLE:
        written.append(
            write_pdf_report(rocket, flight, directory / "engineering_report.pdf")
        )
    else:
        console.print(
            "[yellow]ReportLab is not installed, so the PDF was skipped; the "
            "Markdown report was still written. Install with: "
            "pip install 'rocketopt[reports]'[/yellow]"
        )

    # export_all writes printable STLs, SVG, DXF and the Fusion script
    # unconditionally, and adds STEP when CadQuery is available.
    from rocketopt.cad.exporters import export_all

    written.extend(export_all(rocket, directory))

    if not any(p.suffix == ".step" for p in written):
        console.print(
            "[yellow]CadQuery is not installed, so STEP was skipped. The "
            "printable STLs were still written. Install with: "
            "pip install 'rocketopt[cad]'[/yellow]"
        )

    console.print("\n[green]Written:[/green]")
    for path in written:
        console.print(f"  {path}")


@app.command("simulate")
def simulate_command(
    motor: Annotated[str, typer.Argument(help="Motor designation, e.g. C6-5.")],
    wind: Annotated[float, typer.Option("--wind", help="Wind speed in m/s.")] = 0.0,
    elevation: Annotated[
        float, typer.Option("--elevation", help="Site elevation in m.")
    ] = 0.0,
    rail_angle: Annotated[
        float, typer.Option("--rail-angle", help="Rail tilt from vertical in degrees.")
    ] = 0.0,
) -> None:
    """Fly the reference design on a motor, without optimising."""
    configure_logging(logging.WARNING)

    from rocketopt.flight.environment import LaunchConditions
    from rocketopt.flight.simulation import SimulationConfig, simulate
    from rocketopt.ui.panels import DesignValues

    # The interface defaults describe a validated, stable reference rocket.
    from rocketopt.ui.main_window import build_rocket_from_values

    values = DesignValues(motor=motor)
    try:
        rocket = build_rocket_from_values(values)
    except (KeyError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    conditions = LaunchConditions.from_inputs(
        elevation_m=elevation, wind_speed_ms=wind, rail_angle_deg=rail_angle
    )
    result = simulate(rocket, conditions, SimulationConfig(six_dof=True))

    console.print(Panel(rocket.summary(), title="Design", border_style="blue"))
    console.print(Panel(result.summary(), title="Flight", border_style="green"))


@app.command("ui")
def ui_command() -> None:
    """Open the desktop interface."""
    from rocketopt.ui import run

    raise typer.Exit(code=run())


@app.command("validate")
def validate_command() -> None:
    """Check the physics models against published reference cases."""
    configure_logging(logging.WARNING)

    from rocketopt.aerodynamics.atmosphere import Atmosphere
    from rocketopt.geometry.nose_cones import NoseCone, NoseConeShape
    from rocketopt.structures.materials import get_material

    table = Table(title="Validation against published references", header_style="bold")
    table.add_column("Case")
    table.add_column("Computed", justify="right")
    table.add_column("Reference", justify="right")
    table.add_column("Error", justify="right")
    table.add_column("Verdict", justify="center")

    rows: list[tuple[str, float, float, float]] = []

    # USSA-1976 sea-level and 11 km values.
    std = Atmosphere.standard()
    for altitude, reference, label in (
        (0.0, 1.2250, "Air density at sea level (kg/m³)"),
        (11000.0, 0.36392, "Air density at 11 km (kg/m³)"),
        (0.0, 340.29, "Speed of sound at sea level (m/s)"),
    ):
        state = std.state_at(altitude)
        value = state.density if "density" in label else state.speed_of_sound
        rows.append((label, value, reference, 1e-3))

    # Barrowman nose CP constants.
    material = get_material("balsa")
    for shape, reference, label in (
        (NoseConeShape.CONICAL, 2.0 / 3.0, "Conical nose X_cp / L"),
        (NoseConeShape.TANGENT_OGIVE, 0.466, "Tangent ogive X_cp / L"),
        (NoseConeShape.ELLIPTICAL, 1.0 / 3.0, "Elliptical nose X_cp / L"),
    ):
        cone = NoseCone(
            shape=shape,
            length=0.15,
            base_radius=0.0122,
            material=material,
            solid=True,
        )
        rows.append((label, cone.centre_of_pressure / 0.15, reference, 5e-3))

    all_pass = True
    for label, computed, reference, tolerance in rows:
        error = abs(computed - reference) / abs(reference)
        passes = error <= tolerance
        all_pass = all_pass and passes
        table.add_row(
            label,
            f"{computed:.5g}",
            f"{reference:.5g}",
            f"{error * 100:.3f}%",
            f"[{_verdict_style(passes)}]{'PASS' if passes else 'FAIL'}[/]",
        )

    console.print(table)
    if not all_pass:
        raise typer.Exit(code=1)
    console.print("[green]All reference cases pass.[/green]")


def main() -> None:
    """Entry point for the ``rocketopt`` console script."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
