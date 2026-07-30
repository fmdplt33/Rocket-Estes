"""Embedded Matplotlib canvases for trajectory, aerodynamic and Pareto plots."""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("QtAgg")

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from rocketopt.aerodynamics.drag import DragBreakdown  # noqa: E402
from rocketopt.flight.simulation import FlightResult  # noqa: E402
from rocketopt.optimisation.ga import OptimisationHistory  # noqa: E402
from rocketopt.ui.theme import (  # noqa: E402
    ACCENT,
    BORDER,
    PLOT_COLOURS,
    SURFACE,
    TEXT,
    TEXT_MUTED,
)

__all__ = ["PlotCanvas", "TrajectoryPlot", "DragPlot", "ConvergencePlot", "ParetoPlot"]


class PlotCanvas(FigureCanvasQTAgg):
    """A Matplotlib canvas styled to match the dark application theme.

    Parameters
    ----------
    parent:
        Optional parent widget.
    nrows, ncols:
        Subplot grid shape.
    """

    def __init__(self, parent: object = None, nrows: int = 1, ncols: int = 1) -> None:
        """Create a themed figure and axes."""
        self.figure = Figure(figsize=(6, 4), dpi=100, facecolor=SURFACE)
        super().__init__(self.figure)
        if parent is not None:
            self.setParent(parent)  # type: ignore[arg-type]

        self.axes = self.figure.subplots(nrows, ncols, squeeze=False)
        for ax in self.axes.flat:
            self._style_axes(ax)
        self.figure.tight_layout()

    @staticmethod
    def _style_axes(ax: object) -> None:
        """Apply the dark theme to one axes object."""
        ax.set_facecolor(SURFACE)  # type: ignore[attr-defined]
        for spine in ax.spines.values():  # type: ignore[attr-defined]
            spine.set_color(BORDER)
        ax.tick_params(colors=TEXT_MUTED, labelsize=8)  # type: ignore[attr-defined]
        ax.xaxis.label.set_color(TEXT)  # type: ignore[attr-defined]
        ax.yaxis.label.set_color(TEXT)  # type: ignore[attr-defined]
        ax.title.set_color(TEXT)  # type: ignore[attr-defined]
        ax.grid(True, color=BORDER, alpha=0.45, linewidth=0.6)  # type: ignore[attr-defined]

    def clear(self) -> None:
        """Clear every axes and re-apply the theme."""
        for ax in self.axes.flat:
            ax.clear()
            self._style_axes(ax)

    def finish(self) -> None:
        """Lay out and repaint."""
        try:
            self.figure.tight_layout()
        except (ValueError, RuntimeError):
            # tight_layout can fail on degenerate axes; the plot is still valid.
            pass
        self.draw_idle()


class TrajectoryPlot(PlotCanvas):
    """Altitude, speed, acceleration and dynamic pressure against time."""

    def __init__(self, parent: object = None) -> None:
        """Create a two-by-two trajectory panel."""
        super().__init__(parent, nrows=2, ncols=2)

    def show_flight(self, result: FlightResult | None) -> None:
        """Plot a flight result.

        Parameters
        ----------
        result:
            The flight to display, or ``None`` to show an empty panel.
        """
        self.clear()
        if result is None or not result.states:
            self.axes[0][0].text(
                0.5,
                0.5,
                "Run an analysis to see the trajectory",
                ha="center",
                va="center",
                color=TEXT_MUTED,
                transform=self.axes[0][0].transAxes,
            )
            self.finish()
            return

        data = result.as_arrays()

        panels = (
            (0, 0, "altitude", "Altitude (m)", PLOT_COLOURS[0]),
            (0, 1, "speed", "Speed (m/s)", PLOT_COLOURS[1]),
            (1, 0, "acceleration", "Acceleration (m/s$^2$)", PLOT_COLOURS[2]),
            (1, 1, "dynamic_pressure", "Dynamic pressure (Pa)", PLOT_COLOURS[3]),
        )
        for row, col, key, label, colour in panels:
            ax = self.axes[row][col]
            ax.plot(data["time"], data[key], color=colour, linewidth=1.6)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel(label)

        # Mark the key flight events on the altitude trace.
        ax = self.axes[0][0]
        for time_value, label, colour in (
            (result.rail_exit_time, "rail exit", TEXT_MUTED),
            (result.rocket.motor.motor.burn_time, "burnout", PLOT_COLOURS[2]),
            (result.apogee_time, "apogee", PLOT_COLOURS[1]),
            (result.rocket.motor.ejection_time, "ejection", PLOT_COLOURS[4]),
        ):
            if time_value and time_value > 0:
                ax.axvline(time_value, color=colour, linestyle="--", linewidth=0.9, alpha=0.8)
        ax.annotate(
            f"{result.apogee:.0f} m",
            xy=(result.apogee_time, result.apogee),
            xytext=(6, -12),
            textcoords="offset points",
            color=TEXT,
            fontsize=9,
            fontweight="bold",
        )
        self.finish()


class DragPlot(PlotCanvas):
    """Drag coefficient breakdown and its variation with speed."""

    def __init__(self, parent: object = None) -> None:
        """Create a two-panel drag view."""
        super().__init__(parent, nrows=1, ncols=2)

    def show_drag(
        self,
        breakdown: DragBreakdown | None,
        speeds: Sequence[float] = (),
        totals: Sequence[float] = (),
    ) -> None:
        """Plot a drag breakdown and a drag-versus-speed sweep.

        Parameters
        ----------
        breakdown:
            Component breakdown at a representative condition.
        speeds:
            Speeds for the sweep [m/s].
        totals:
            Total drag coefficient at each speed [-].
        """
        self.clear()
        if breakdown is None:
            self.axes[0][0].text(
                0.5,
                0.5,
                "Run an analysis to see the drag breakdown",
                ha="center",
                va="center",
                color=TEXT_MUTED,
                transform=self.axes[0][0].transAxes,
            )
            self.finish()
            return

        items = [(k, v) for k, v in breakdown.as_dict().items() if v > 1e-6]
        items.sort(key=lambda kv: kv[1], reverse=True)

        ax = self.axes[0][0]
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        colours = [PLOT_COLOURS[i % len(PLOT_COLOURS)] for i in range(len(items))]
        bars = ax.barh(labels, values, color=colours)
        ax.invert_yaxis()
        ax.set_xlabel("$C_D$ contribution")
        ax.set_title(f"Total $C_D$ = {breakdown.total:.3f}", fontsize=10)
        for bar, value in zip(bars, values, strict=True):
            ax.text(
                value,
                bar.get_y() + bar.get_height() / 2,
                f" {value:.3f}",
                va="center",
                color=TEXT_MUTED,
                fontsize=8,
            )

        ax2 = self.axes[0][1]
        if speeds and totals:
            ax2.plot(speeds, totals, color=ACCENT, linewidth=1.8)
            ax2.set_xlabel("Speed (m/s)")
            ax2.set_ylabel("$C_D$")
            ax2.set_title("Drag versus speed", fontsize=10)
        self.finish()


class ConvergencePlot(PlotCanvas):
    """Optimiser convergence: best objective and feasible count by generation."""

    def show_history(
        self, history: OptimisationHistory | None, objective_label: str = "Objective"
    ) -> None:
        """Plot convergence history.

        Parameters
        ----------
        history:
            The optimisation history, or ``None``.
        objective_label:
            Axis label for the primary objective.
        """
        self.clear()
        ax = self.axes[0][0]
        if history is None or not history.records:
            ax.text(
                0.5,
                0.5,
                "Run an optimisation to see convergence",
                ha="center",
                va="center",
                color=TEXT_MUTED,
                transform=ax.transAxes,
            )
            self.finish()
            return

        generations = [r.generation for r in history.records]
        best = [
            r.best_objectives[0] if r.best_objectives and r.feasible_count else float("nan")
            for r in history.records
        ]
        mean = [
            r.mean_objectives[0] if r.mean_objectives and r.feasible_count else float("nan")
            for r in history.records
        ]

        ax.plot(generations, best, color=PLOT_COLOURS[1], linewidth=2.0, label="Best")
        ax.plot(
            generations,
            mean,
            color=PLOT_COLOURS[0],
            linewidth=1.2,
            linestyle="--",
            label="Population mean",
        )
        ax.set_xlabel("Generation")
        ax.set_ylabel(objective_label)
        legend = ax.legend(facecolor=SURFACE, edgecolor=BORDER, fontsize=8)
        for text in legend.get_texts():
            text.set_color(TEXT)

        feasible_ax = ax.twinx()
        self._style_axes(feasible_ax)
        feasible_ax.grid(False)
        feasible_ax.fill_between(
            generations,
            [r.feasible_count for r in history.records],
            color=PLOT_COLOURS[6],
            alpha=0.18,
        )
        feasible_ax.set_ylabel("Feasible designs", color=TEXT_MUTED)
        self.finish()


class ParetoPlot(PlotCanvas):
    """Scatter of the Pareto front for a two-objective run."""

    def show_front(
        self,
        rows: Sequence[dict[str, float]],
        x_label: str,
        y_label: str,
    ) -> None:
        """Plot a Pareto front.

        Parameters
        ----------
        rows:
            Rows from
            :meth:`~rocketopt.optimisation.driver.OptimisationOutcome.pareto_table`.
        x_label, y_label:
            Keys to plot against one another.
        """
        self.clear()
        ax = self.axes[0][0]
        if not rows or x_label not in rows[0] or y_label not in rows[0]:
            ax.text(
                0.5,
                0.5,
                "Optimise with two objectives to see a trade-off front",
                ha="center",
                va="center",
                color=TEXT_MUTED,
                transform=ax.transAxes,
            )
            self.finish()
            return

        xs = [r[x_label] for r in rows]
        ys = [r[y_label] for r in rows]
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]

        ax.plot(xs, ys, color=BORDER, linewidth=1.0, zorder=1)
        ax.scatter(xs, ys, s=46, color=ACCENT, zorder=2, edgecolors=SURFACE)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title("Pareto front - every point is an optimal trade", fontsize=10)
        self.finish()
