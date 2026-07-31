"""The RocketOpt main application window.

Interaction model
-----------------
Edit on the left, see the consequence immediately in the centre and right.
Every parameter change re-runs a fast analysis after a short debounce, so the
design responds as it is typed rather than behind a "calculate" button.

Optimisation is the one long operation, so it runs on a worker thread with a
progress bar and a cancel button. Its result is loaded straight back into the
design controls, which means the optimiser's answer becomes the starting point
for further manual work rather than a dead end.
"""

from __future__ import annotations

import contextlib
import math
import traceback
from dataclasses import replace

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from rocketopt.aerodynamics.barrowman import barrowman_analysis
from rocketopt.aerodynamics.drag import drag_buildup
from rocketopt.flight.environment import LaunchConditions
from rocketopt.flight.simulation import FlightResult, SimulationConfig, simulate
from rocketopt.geometry.components import (
    BodyTube,
    FinAirfoil,
    FinSet,
    LaunchLug,
    RecoveryDevice,
    RecoveryType,
)
from rocketopt.geometry.nose_cones import NoseCone, NoseConeShape
from rocketopt.geometry.rocket import Rocket, build_rocket
from rocketopt.optimisation.driver import OptimisationOutcome, optimise
from rocketopt.optimisation.ga import GAConfig, GenerationRecord
from rocketopt.optimisation.problem import (
    DesignConstraints,
    Objective,
    default_design_space,
)
from rocketopt.propulsion.database import get_motor_configuration
from rocketopt.structures.analysis import analyse_structure
from rocketopt.structures.materials import SurfaceFinish, get_material
from rocketopt.ui.animation import AnimationPage
from rocketopt.ui.panels import DesignPanel, DesignValues, ResultsPanel
from rocketopt.ui.plots import ConvergencePlot, DragPlot, ParetoPlot, TrajectoryPlot
from rocketopt.ui.rocket_view import RocketView
from rocketopt.ui.viewer3d import Viewer3DPage
from rocketopt.ui.windtunnel import WindTunnelPage
from rocketopt.utils.constants import (
    MIN_LIFTOFF_THRUST_TO_WEIGHT,
    MIN_RAIL_EXIT_VELOCITY,
)
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = ["MainWindow", "build_rocket_from_values", "conditions_from_values"]

_DEBOUNCE_MS = 220
"""Delay after the last edit before re-analysing, in milliseconds."""


def conditions_from_values(values: DesignValues) -> LaunchConditions:
    """Build launch conditions from the panel snapshot."""
    return LaunchConditions.from_inputs(
        elevation_m=values.elevation_m,
        temperature_c=values.temperature_c,
        humidity_percent=values.humidity_pct,
        wind_speed_ms=values.wind_ms,
        wind_direction_deg=values.wind_dir_deg,
        rail_length_m=values.rail_length_m,
        rail_angle_deg=values.rail_angle_deg,
    )


def build_rocket_from_values(values: DesignValues) -> Rocket:
    """Build a :class:`Rocket` from the design panel snapshot.

    Parameters
    ----------
    values:
        Panel snapshot.

    Returns
    -------
    Rocket
        The assembled design.

    Raises
    ------
    ValueError
        If the combination of values is not a buildable rocket. The message is
        shown to the user, so it must explain what to change.
    """
    radius = values.body_diameter_mm * 1e-3 / 2.0
    body_material = get_material(values.body_material)
    wall = min(values.wall_thickness_mm * 1e-3, radius * 0.4)

    body = BodyTube(
        length=values.body_length_mm * 1e-3,
        outer_radius=radius,
        wall_thickness=wall,
        material=body_material,
    )

    nose_material = get_material(values.nose_material)
    nose = NoseCone(
        shape=NoseConeShape(values.nose_shape),
        length=values.nose_fineness * 2.0 * radius,
        base_radius=radius,
        material=nose_material,
        wall_thickness=min(1.5e-3, radius * 0.3),
        solid=nose_material.category.value == "wood",
        shoulder_length=min(0.025, body.length * 0.15),
    )

    fin_material = get_material(values.fin_material)
    fins = FinSet(
        count=values.fin_count,
        root_chord=values.fin_root_mm * 1e-3,
        tip_chord=values.fin_tip_mm * 1e-3,
        span=values.fin_span_mm * 1e-3,
        sweep_length=values.fin_sweep_mm * 1e-3,
        thickness=values.fin_thickness_mm * 1e-3,
        material=fin_material,
        body_radius=radius,
        airfoil=FinAirfoil(values.fin_airfoil),
        fillet_radius=values.fillet_mm * 1e-3,
    )

    motor = get_motor_configuration(values.motor)

    # Size the canopy for a 4 m/s descent from a rough mass estimate.
    estimate = nose.mass + body.mass + fins.mass + 0.012
    area = 2.0 * estimate * 9.80665 / (1.225 * 0.75 * 16.0)
    diameter = min(max(2.0 * math.sqrt(area / math.pi), 0.15), 0.60)

    return build_rocket(
        nose=nose,
        body_tube=body,
        fins=fins,
        motor=motor,
        recovery=RecoveryDevice(
            kind=RecoveryType.PARACHUTE_FLAT,
            diameter=diameter,
            shroud_line_count=6,
            shroud_line_length=diameter,
        ),
        launch_lug=LaunchLug(
            length=0.035,
            outer_radius=0.0021,
            inner_radius=0.0016,
            material=body_material,
        ),
        nose_ballast_mass=values.ballast_g * 1e-3,
        surface_finish=SurfaceFinish[values.surface_finish],
        name="Interactive design",
    )


class _OptimiseWorker(QObject):
    """Runs an optimisation off the GUI thread.

    Cancellation is cooperative: :meth:`cancel` sets a flag that the optimiser
    polls at each generation boundary and after each evaluation, so a cancelled
    run still returns the best front found so far rather than nothing.
    """

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, float, int)

    def __init__(
        self,
        motor: str,
        objectives: tuple[Objective, ...],
        conditions: LaunchConditions,
        constraints: DesignConstraints,
        population: int,
        generations: int,
    ) -> None:
        """Store the run parameters."""
        super().__init__()
        self._motor = motor
        self._objectives = objectives
        self._conditions = conditions
        self._constraints = constraints
        self._population = population
        self._generations = generations
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation.

        Safe to call from the GUI thread: a plain bool assignment is atomic
        under the interpreter lock, and the optimiser only ever reads it.
        """
        self._cancelled = True

    def _should_cancel(self) -> bool:
        """Report whether cancellation has been requested."""
        return self._cancelled

    def _on_generation(self, record: GenerationRecord) -> None:
        """Forward a generation record to the GUI thread as a signal."""
        best = (
            record.best_objectives[0]
            if record.best_objectives and record.feasible_count
            else float("nan")
        )
        self.progress.emit(
            record.generation, self._generations, float(best), record.feasible_count
        )

    def run(self) -> None:
        """Execute the optimisation and emit the outcome."""
        try:
            motor_config = get_motor_configuration(self._motor)
            space = default_design_space(motor_config)

            outcome = optimise(
                space,
                objectives=self._objectives,
                constraints=self._constraints,
                conditions=self._conditions,
                ga_config=GAConfig(
                    population_size=self._population,
                    generations=self._generations,
                    seed=20260729,
                    verbose=False,
                ),
                progress_fn=self._on_generation,
                should_cancel=self._should_cancel,
            )
            self.finished.emit(outcome)
        except Exception as exc:  # noqa: BLE001 - must not kill the GUI thread
            _log.debug("optimisation failed: %s", traceback.format_exc())
            self.failed.emit(str(exc))


class MainWindow(QMainWindow):
    """The application's main window."""

    def __init__(self) -> None:
        """Build the window and run an initial analysis."""
        super().__init__()
        self.setWindowTitle("RocketOpt - model rocket design and optimisation")
        self.resize(1500, 940)

        self._rocket: Rocket | None = None
        self._flight: FlightResult | None = None
        self._outcome: OptimisationOutcome | None = None
        self._thread: QThread | None = None
        self._worker: _OptimiseWorker | None = None
        self._animation_stale: bool = True
        self._stale_tabs: set[QWidget] = set()
        self._drag_plot_data: tuple[object, list[float], list[float]] | None = None

        self._build_toolbar()
        self._build_docks()
        self._build_centre()
        self._build_menu()

        self.statusBar().showMessage("Ready")

        # Debounce timer so dragging a spin box does not queue dozens of runs.
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(_DEBOUNCE_MS)
        self._debounce.timeout.connect(self._analyse)

        self.design_panel.changed.connect(self._debounce.start)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self._analyse()

    # -- Construction -------------------------------------------------------

    def _build_toolbar(self) -> None:
        """Build the top toolbar with the optimisation controls."""
        bar = QToolBar("Main")
        bar.setMovable(False)
        self.addToolBar(bar)

        bar.addWidget(QLabel("  Optimise for  "))

        self.objective_box = QComboBox()
        self.objective_box.addItem("Maximum altitude", (Objective.APOGEE,))
        self.objective_box.addItem("Maximum speed", (Objective.MAX_VELOCITY,))
        self.objective_box.addItem("Minimum drag", (Objective.LOW_DRAG,))
        self.objective_box.addItem("Longest flight", (Objective.FLIGHT_DURATION,))
        self.objective_box.addItem("Lightest", (Objective.LIGHT,))
        self.objective_box.addItem(
            "Altitude vs ease of building",
            (Objective.APOGEE, Objective.MANUFACTURABILITY),
        )
        self.objective_box.addItem(
            "Altitude vs mass", (Objective.APOGEE, Objective.LIGHT)
        )
        self.objective_box.setToolTip(
            "Choosing two objectives returns a trade-off front rather than a "
            "single answer; see the Pareto tab."
        )
        bar.addWidget(self.objective_box)

        self.effort_box = QComboBox()
        self.effort_box.addItem("Quick  (24 x 10)", (24, 10))
        self.effort_box.addItem("Standard  (40 x 20)", (40, 20))
        self.effort_box.addItem("Thorough  (64 x 40)", (64, 40))
        self.effort_box.setCurrentIndex(0)
        self.effort_box.setToolTip(
            "Population size by generations. More of both searches more "
            "thoroughly and takes proportionally longer."
        )
        bar.addWidget(self.effort_box)

        self.optimise_button = QPushButton("Optimise")
        self.optimise_button.setProperty("primary", True)
        self.optimise_button.clicked.connect(self._start_optimise)
        bar.addWidget(self.optimise_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_optimise)
        bar.addWidget(self.cancel_button)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        bar.addWidget(self.progress)

    def _build_docks(self) -> None:
        """Build the left design dock and right results dock."""
        self.design_panel = DesignPanel()
        design_dock = QDockWidget("Design")
        design_dock.setWidget(self.design_panel)
        design_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        design_dock.setMinimumWidth(330)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, design_dock)

        self.results_panel = ResultsPanel()
        results_dock = QDockWidget("Results")
        results_dock.setWidget(self.results_panel)
        results_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        results_dock.setMinimumWidth(300)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, results_dock)

    def _build_centre(self) -> None:
        """Build the central tab stack."""
        self.tabs = QTabWidget()

        rocket_tab = QWidget()
        rocket_layout = QVBoxLayout(rocket_tab)
        rocket_layout.setContentsMargins(8, 8, 8, 8)
        self.rocket_view = RocketView()
        self.rocket_view.finsEdited.connect(self._on_fins_dragged)
        rocket_layout.addWidget(self.rocket_view)

        plan_controls = QHBoxLayout()
        plan_controls.setSpacing(8)
        self.plan_fit_button = QPushButton("Fit")
        self.plan_fit_button.setToolTip(
            "Frame the whole rocket again. Double-clicking the view does the "
            "same."
        )
        self.plan_fit_button.clicked.connect(self.rocket_view.fit)
        plan_controls.addWidget(self.plan_fit_button)
        plan_controls.addStretch(1)
        rocket_layout.addLayout(plan_controls)

        hint = QLabel(
            "Drag the circled points on the fin to reshape it - the numbers on "
            "the left follow, and vice versa. Scroll to zoom about the pointer "
            "and drag anywhere else to pan. The centre of pressure (CP) must "
            "sit aft of the centre of gravity (CG); the gap between them, in "
            "body diameters, is the static margin. Aim for 1.0 to 2.0."
        )
        hint.setWordWrap(True)
        hint.setProperty("muted", True)
        rocket_layout.addWidget(hint)
        self.tabs.addTab(rocket_tab, "Rocket")

        self.viewer_page = Viewer3DPage()
        self.tabs.addTab(self.viewer_page, "3D")

        self.wind_tunnel = WindTunnelPage()
        self.tabs.addTab(self.wind_tunnel, "Wind tunnel")

        self.animation_page = AnimationPage()
        self.tabs.addTab(self.animation_page, "Animation")

        self.trajectory_plot = TrajectoryPlot()
        self.tabs.addTab(self.trajectory_plot, "Trajectory")

        self.drag_plot = DragPlot()
        self.tabs.addTab(self.drag_plot, "Aerodynamics")

        self.convergence_plot = ConvergencePlot()
        self.tabs.addTab(self.convergence_plot, "Convergence")

        self.pareto_plot = ParetoPlot()
        self.tabs.addTab(self.pareto_plot, "Trade-off")

        self.setCentralWidget(self.tabs)

    def _build_menu(self) -> None:
        """Build the menu bar."""
        file_menu = self.menuBar().addMenu("&File")

        export_bundle = QAction("Export &everything to one file...", self)
        export_bundle.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_bundle.setStatusTip(
            "Write every STL, STEP, template and report into a single zip"
        )
        export_bundle.triggered.connect(self._export_bundle)
        file_menu.addAction(export_bundle)

        file_menu.addSeparator()

        export_report = QAction("Export &report...", self)
        export_report.setShortcut(QKeySequence("Ctrl+R"))
        export_report.triggered.connect(self._export_report)
        file_menu.addAction(export_report)

        export_cad = QAction("Export CAD to a &folder...", self)
        export_cad.setShortcut(QKeySequence("Ctrl+E"))
        export_cad.triggered.connect(self._export_cad)
        file_menu.addAction(export_cad)

        file_menu.addSeparator()
        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

    # -- Analysis -----------------------------------------------------------

    def _analyse(self) -> None:
        """Rebuild and re-analyse the design from the current controls."""
        values = self.design_panel.values()

        try:
            rocket = build_rocket_from_values(values)
        except (ValueError, KeyError) as exc:
            self._rocket = None
            self.rocket_view.set_rocket(None, 0.0, 0.0)
            self.viewer_page.set_rocket(None)
            self.wind_tunnel.set_rocket(None)
            self.results_panel.set_error(str(exc))
            self.statusBar().showMessage(f"Invalid design: {exc}")
            return

        conditions = conditions_from_values(values)

        try:
            # Preview fidelity while editing: 20 ms steps, apogee cut-off.
            # Within a few centimetres of the 1 ms result and fast enough that
            # the numbers track the controls without perceptible lag.
            flight = simulate(rocket, conditions, SimulationConfig.preview())
        except ValueError as exc:
            self._rocket = rocket
            analysis = barrowman_analysis(rocket, mach=0.3)
            self.rocket_view.set_rocket(
                rocket, rocket.cg_at(0.0), analysis.centre_of_pressure
            )
            self.results_panel.set_error(str(exc))
            self.statusBar().showMessage(f"Cannot fly: {exc}")
            return

        self._rocket = rocket
        self._flight = flight

        analysis = barrowman_analysis(rocket, mach=0.3)
        cg = rocket.cg_at(0.0)
        self.rocket_view.set_rocket(rocket, cg, analysis.centre_of_pressure)

        structure = analyse_structure(
            rocket,
            max_velocity=flight.max_velocity,
            max_dynamic_pressure=flight.max_dynamic_pressure,
            max_acceleration=flight.max_acceleration,
            landing_velocity=flight.landing_velocity,
        )

        state = conditions.atmosphere.state_at(conditions.ground_altitude)
        reference_speed = max(flight.max_velocity * 0.6, 20.0)
        breakdown = drag_buildup(rocket, state, reference_speed)

        speeds = [10.0 + 10.0 * i for i in range(25)]
        totals = [drag_buildup(rocket, state, v).total for v in speeds]

        margin = (analysis.centre_of_pressure - cg) / rocket.reference_diameter
        self.results_panel.update_results(
            apogee=flight.apogee,
            max_velocity=flight.max_velocity,
            static_margin=margin,
            loaded_mass=rocket.loaded_mass,
            rail_exit=flight.rail_exit_velocity,
            thrust_to_weight=rocket.thrust_to_weight,
            drag_coefficient=breakdown.total,
            landing_velocity=flight.landing_velocity,
            structure_rows=[
                (c.name, c.margin_of_safety, c.passes) for c in structure.checks
            ],
            flutter_text=(
                f"{structure.flutter.flutter_velocity:.0f} m/s"
                if structure.flutter.flutter_velocity < 1e5
                else "n/a"
            ),
            flutter_ok=structure.flutter.is_safe,
            warnings=flight.warnings,
            min_rail_exit=MIN_RAIL_EXIT_VELOCITY,
            min_twr=MIN_LIFTOFF_THRUST_TO_WEIGHT,
        )

        # Matplotlib redraws cost far more than the physics does - together they
        # were three quarters of the interactive latency. Only the visible tab
        # is redrawn; the others are marked stale and refreshed when opened.
        self._drag_plot_data = (breakdown, speeds, totals)
        self._stale_tabs = {
            self.trajectory_plot,
            self.drag_plot,
            self.viewer_page,
            self.wind_tunnel,
        }
        self._animation_stale = True
        self._refresh_visible_tab()

        self.statusBar().showMessage(
            f"Apogee {flight.apogee:.0f} m | "
            f"{rocket.loaded_mass * 1e3:.0f} g | "
            f"static margin {margin:.2f} cal | "
            f"Cd {breakdown.total:.3f}"
        )

    def _on_tab_changed(self, index: int) -> None:
        """Refresh whichever tab has just become visible, if it is stale."""
        del index
        self._refresh_visible_tab()

    def _refresh_visible_tab(self) -> None:
        """Redraw only the currently visible tab, if its data is stale.

        Deferring the redraw of hidden tabs is what keeps editing responsive:
        a Matplotlib figure redraw costs roughly as much as the entire flight
        simulation, and there is no point paying it for a plot nobody is
        looking at.
        """
        current = self.tabs.currentWidget()

        if current is self.animation_page:
            if self._animation_stale:
                self._refresh_animation()
            return

        if current in self._stale_tabs:
            if current is self.trajectory_plot:
                self.trajectory_plot.show_flight(self._flight)
            elif current is self.drag_plot and self._drag_plot_data is not None:
                breakdown, speeds, totals = self._drag_plot_data
                self.drag_plot.show_drag(breakdown, speeds, totals)
            elif current is self.viewer_page:
                self.viewer_page.set_rocket(self._rocket)
            elif current is self.wind_tunnel:
                # Solving the field costs a good fraction of a second, so it
                # only happens for a tab somebody is actually looking at.
                self.wind_tunnel.set_rocket(self._rocket)
            self._stale_tabs.discard(current)

    def _refresh_animation(self) -> None:
        """Run a full-flight simulation and load it into the animation page."""
        if self._rocket is None:
            self.animation_page.set_flight(None)
            return

        values = self.design_panel.values()
        try:
            flight = simulate(
                self._rocket,
                conditions_from_values(values),
                SimulationConfig.animation(),
            )
        except ValueError:
            self.animation_page.set_flight(None)
            return

        self.animation_page.set_flight(flight)
        self._animation_stale = False

    def _on_fins_dragged(
        self, root_mm: float, tip_mm: float, span_mm: float, sweep_mm: float
    ) -> None:
        """Apply a fin drag from the rocket view to the numeric controls.

        The spin boxes are updated with signals blocked and the re-analysis is
        triggered once at the end. Letting each ``setValue`` emit would fire
        four analyses per mouse-move event and make dragging unusable.
        """
        panel = self.design_panel
        widgets = (panel.fin_root, panel.fin_tip, panel.fin_span, panel.fin_sweep)
        for widget in widgets:
            widget.blockSignals(True)
        try:
            panel.fin_root.setValue(root_mm)
            panel.fin_tip.setValue(tip_mm)
            panel.fin_span.setValue(span_mm)
            panel.fin_sweep.setValue(sweep_mm)
        finally:
            for widget in widgets:
                widget.blockSignals(False)

        # Re-analyse immediately rather than via the debounce, so the drawing
        # tracks the pointer. The preview simulation is fast enough for this.
        self._analyse()

    # -- Optimisation --------------------------------------------------------

    def _start_optimise(self) -> None:
        """Launch an optimisation on a worker thread."""
        if self._thread is not None:
            return

        values = self.design_panel.values()
        objectives = self.objective_box.currentData()
        population, generations = self.effort_box.currentData()

        self.optimise_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress.setVisible(True)
        # Real generation count, not an indeterminate spinner: the user can see
        # how far through the run is and how the best design is improving.
        self.progress.setRange(0, generations)
        self.progress.setValue(0)
        self.progress.setFormat("starting...")
        self.statusBar().showMessage(
            f"Optimising for {self.objective_box.currentText().lower()}: "
            f"{population} designs x {generations} generations"
        )

        self._thread = QThread(self)
        self._worker = _OptimiseWorker(
            motor=values.motor,
            objectives=objectives,
            conditions=conditions_from_values(values),
            constraints=DesignConstraints(),
            population=population,
            generations=generations,
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._optimise_finished)
        self._worker.failed.connect(self._optimise_failed)
        self._worker.progress.connect(self._on_optimise_progress)
        self._thread.start()

    def _on_optimise_progress(
        self, generation: int, total: int, best: float, feasible: int
    ) -> None:
        """Update the progress bar from a generation record."""
        self.progress.setMaximum(total)
        self.progress.setValue(generation)

        unit = self.objective_box.currentData()[0].units
        if best == best:  # not NaN
            self.progress.setFormat(f"gen {generation}/{total} - {best:.0f} {unit}")
            self.statusBar().showMessage(
                f"Generation {generation} of {total}: best "
                f"{best:.1f} {unit}, {feasible} feasible designs"
            )
        else:
            self.progress.setFormat(f"gen {generation}/{total} - searching")
            self.statusBar().showMessage(
                f"Generation {generation} of {total}: no feasible design yet"
            )

    def _cancel_optimise(self) -> None:
        """Ask the worker to stop and keep whatever it has found."""
        if self._worker is not None:
            self._worker.cancel()
            self.cancel_button.setEnabled(False)
            self.progress.setFormat("stopping...")
            self.statusBar().showMessage(
                "Stopping - the best design found so far will be kept"
            )

    def _teardown_thread(self) -> None:
        """Stop and dispose of the worker thread."""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
            self._thread = None
        self._worker = None
        self.optimise_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        self.progress.setVisible(False)

    def _optimise_finished(self, outcome: OptimisationOutcome) -> None:
        """Load an optimisation result into the design controls."""
        self._outcome = outcome
        self._teardown_thread()

        values = self._values_from_rocket(outcome.best)
        # Preserve the launch conditions the user set; only geometry changes.
        current = self.design_panel.values()
        values = replace(
            values,
            elevation_m=current.elevation_m,
            temperature_c=current.temperature_c,
            humidity_pct=current.humidity_pct,
            wind_ms=current.wind_ms,
            wind_dir_deg=current.wind_dir_deg,
            rail_length_m=current.rail_length_m,
            rail_angle_deg=current.rail_angle_deg,
        )
        self.design_panel.set_values(values)

        label = outcome.objectives[0].label
        self.convergence_plot.show_history(outcome.history, label)

        rows = outcome.pareto_table()
        if len(outcome.objectives) >= 2:
            self.pareto_plot.show_front(
                rows, outcome.objectives[0].label, outcome.objectives[1].label
            )
            self.tabs.setCurrentWidget(self.pareto_plot)
        else:
            self.pareto_plot.show_front([], "", "")
            self.tabs.setCurrentWidget(self.convergence_plot)

        verb = "stopped early" if outcome.history.cancelled else "complete"
        self.statusBar().showMessage(
            f"Optimisation {verb}: apogee {outcome.flight.apogee:.0f} m "
            f"after {outcome.history.evaluations} evaluations"
        )

    def _optimise_failed(self, message: str) -> None:
        """Report an optimisation failure."""
        self._teardown_thread()
        self.statusBar().showMessage("Optimisation failed")
        QMessageBox.warning(
            self,
            "Optimisation failed",
            f"{message}\n\nTry relaxing a constraint, or choosing a motor with "
            f"more impulse.",
        )

    @staticmethod
    def _values_from_rocket(rocket: Rocket) -> DesignValues:
        """Convert an optimised rocket back into panel values."""
        body = rocket.sections[0].section
        assert isinstance(body, BodyTube)
        fins = rocket.fins
        return DesignValues(
            motor=rocket.motor.designation,
            nose_shape=rocket.nose.shape.value,
            nose_fineness=rocket.nose.fineness_ratio,
            nose_material=rocket.nose.material.name,
            body_diameter_mm=body.outer_radius * 2e3,
            body_length_mm=body.length * 1e3,
            wall_thickness_mm=body.wall_thickness * 1e3,
            body_material=body.material.name,
            fin_count=fins.count,
            fin_root_mm=fins.root_chord * 1e3,
            fin_tip_mm=fins.tip_chord * 1e3,
            fin_span_mm=fins.span * 1e3,
            fin_sweep_mm=fins.sweep_length * 1e3,
            fin_thickness_mm=fins.thickness * 1e3,
            fin_material=fins.material.name,
            fin_airfoil=fins.airfoil.value,
            fillet_mm=fins.fillet_radius * 1e3,
            surface_finish=rocket.surface_finish.name,
            ballast_g=rocket.nose_ballast_mass * 1e3,
        )

    # -- Export --------------------------------------------------------------

    def _export_report(self) -> None:
        """Write a Markdown engineering report for the current design."""
        if self._rocket is None or self._flight is None:
            QMessageBox.information(
                self, "Nothing to export", "Adjust the design to produce a result first."
            )
            return

        from rocketopt.reports.pdf import PDF_AVAILABLE

        filters = "Markdown (*.md)"
        default = "rocketopt_report.md"
        if PDF_AVAILABLE:
            filters = "PDF (*.pdf);;Markdown (*.md)"
            default = "rocketopt_report.pdf"

        path, _ = QFileDialog.getSaveFileName(
            self, "Export report", default, filters
        )
        if not path:
            return

        # Re-fly at full 6-DOF fidelity: the on-screen numbers come from the
        # coarse preview, and an exported report should not.
        try:
            flight = simulate(
                self._rocket,
                conditions_from_values(self.design_panel.values()),
                SimulationConfig(six_dof=True),
            )
        except ValueError:
            flight = self._flight

        try:
            if path.lower().endswith(".pdf"):
                from rocketopt.reports.pdf import write_pdf_report

                write_pdf_report(self._rocket, flight, path)
            else:
                from rocketopt.reports.engineering import write_markdown_report

                write_markdown_report(self._rocket, flight, path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.warning(self, "Export failed", str(exc))
            return

        self.statusBar().showMessage(f"Report written to {path}")

    def _export_bundle(self) -> None:
        """Write every output for the current design into a single archive."""
        if self._rocket is None:
            QMessageBox.information(
                self,
                "Nothing to export",
                "Adjust the design to produce a result first.",
            )
            return

        default = f"{self._rocket.name.replace(' ', '_')}.zip"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export everything", default, "Zip archive (*.zip)"
        )
        if not path:
            return

        # Re-fly at full fidelity: the on-screen numbers come from the coarse
        # preview, and an exported report should not.
        flight: FlightResult | None = self._flight
        with contextlib.suppress(ValueError):
            flight = simulate(
                self._rocket,
                conditions_from_values(self.design_panel.values()),
                SimulationConfig(six_dof=True),
            )

        self.statusBar().showMessage("Exporting...")
        try:
            from rocketopt.bundle import write_bundle

            contents = write_bundle(self._rocket, path, flight=flight)
        except Exception as exc:  # every failure here is reported to the user
            QMessageBox.warning(self, "Export failed", str(exc))
            self.statusBar().showMessage("Export failed")
            return

        size_kb = contents.path.stat().st_size / 1024
        QMessageBox.information(
            self,
            "Export complete",
            f"{contents.file_count} files written to\n{contents.path}\n"
            f"({size_kb:.0f} kB)\n\n"
            "print/ holds the STLs, ready to slice.\n"
            "cad/ holds STEP solids, templates and the Fusion 360 script.\n"
            "reports/ holds the engineering report and build guide.",
        )
        self.statusBar().showMessage(
            f"{contents.file_count} files bundled into {contents.path}"
        )

    def _export_cad(self) -> None:
        """Write CAD output for the current design."""
        if self._rocket is None:
            QMessageBox.information(
                self, "Nothing to export", "Adjust the design to produce a result first."
            )
            return

        directory = QFileDialog.getExistingDirectory(self, "Choose an export folder")
        if not directory:
            return

        try:
            from rocketopt.cad.exporters import export_all

            written = export_all(self._rocket, directory)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.warning(self, "Export failed", str(exc))
            return

        QMessageBox.information(
            self,
            "Export complete",
            "Written:\n" + "\n".join(str(p) for p in written),
        )
        self.statusBar().showMessage(f"CAD exported to {directory}")

    def closeEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        """Stop any running worker before closing."""
        self._teardown_thread()
        super().closeEvent(event)  # type: ignore[arg-type]
