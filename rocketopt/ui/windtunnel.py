"""A virtual wind tunnel: the flow field around the airframe, drawn.

The numbers behind this tab already exist - :mod:`rocketopt.cfd` solves the
axisymmetric potential flow and integrates the boundary layer through it - but a
column of pressure coefficients tells you very little about a shape. Seeing
where the flow accelerates, where the pressure recovers and where the boundary
layer gives up tells you which part of the airframe to change.

What is shown
-------------
Field
    Pressure coefficient over a slice through the body axis, with the body
    silhouette on top and streamlines traced through the velocity field.
    Blue is suction, red is compression, and the colour scale is symmetric
    about zero so the sign is readable at a glance.
Surface pressure
    ``Cp`` along the surface, with the stagnation point and the suction peak
    marked. The rise aft of the suction peak is the adverse gradient the
    boundary layer has to survive.
Boundary layer
    Displacement thickness and total thickness along the body, with transition
    and any separation marked. Separation ahead of the tail means a wake far
    larger than the body, which is where a boat-tail earns its keep.

The model is inviscid, axisymmetric and slender-body: it is a good guide to
where a shape is working the flow hard, and it says nothing about the fins.
"""

from __future__ import annotations

from typing import Final

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from rocketopt.aerodynamics.atmosphere import Atmosphere
from rocketopt.cfd.boundary_layer import BoundaryLayerSolution, solve_boundary_layer
from rocketopt.cfd.potential_flow import (
    BodyProfile,
    PotentialFlowSolution,
    solve_potential_flow,
)
from rocketopt.geometry.components import Transition
from rocketopt.geometry.rocket import Rocket
from rocketopt.ui.plots import PlotCanvas
from rocketopt.ui.theme import (
    ACCENT,
    BORDER,
    FAIL,
    PASS,
    SURFACE,
    TEXT,
    TEXT_MUTED,
    WARN,
)

__all__ = ["WindTunnelCanvas", "WindTunnelPage"]

_GRID_X: Final[int] = 180
"""Field grid points along the body."""

_GRID_R: Final[int] = 72
"""Field grid points out from the axis.

The field costs one source-sum per grid point, so the grid is kept to what the
contours actually resolve; the flow is smooth on this scale."""

_FIELD_HEIGHT: Final[float] = 2.1
"""Height of the field slice, in body diameters either side of the axis.

Far enough out that the flow has returned to free stream at the edge, close
enough that the picture is mostly rocket."""

_STREAMLINE_COLOUR: Final[str] = "#4a5058"
"""Streamline colour: dark, because the field they cross is pale near Cp = 0."""

_STREAMLINE_COUNT: Final[int] = 11
"""Number of streamlines seeded upstream."""

_TEST_SPEEDS: Final[tuple[tuple[str, float], ...]] = (
    ("20 m/s  -  off the rod", 20.0),
    ("50 m/s", 50.0),
    ("100 m/s  -  typical burnout", 100.0),
    ("150 m/s", 150.0),
    ("200 m/s  -  fast C-motor design", 200.0),
    ("250 m/s  -  approaching Mach 0.75", 250.0),
)
"""Test speeds offered, chosen to bracket what a model rocket actually sees."""


class WindTunnelCanvas(PlotCanvas):
    """Three linked views of the flow about a design."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Create the field axes above a pair of section plots."""
        super().__init__(parent, nrows=1, ncols=1)
        self.figure.clear()
        grid = self.figure.add_gridspec(
            2, 2, height_ratios=(1.45, 1.0), hspace=0.42, wspace=0.22
        )
        self.field_axes = self.figure.add_subplot(grid[0, :])
        self.pressure_axes = self.figure.add_subplot(grid[1, 0])
        self.layer_axes = self.figure.add_subplot(grid[1, 1])
        self.axes = np.array(
            [[self.field_axes, self.pressure_axes], [self.layer_axes, self.layer_axes]]
        )
        for ax in (self.field_axes, self.pressure_axes, self.layer_axes):
            self._style_axes(ax)

    def finish(self) -> None:
        """Lay the figure out and repaint.

        The base class calls ``tight_layout``, which cannot handle the mixed
        grid here - one wide axes above two narrow ones, with an equal aspect
        ratio on the wide one - so the margins are set explicitly instead.
        """
        self.figure.subplots_adjust(
            left=0.07, right=0.98, top=0.94, bottom=0.09, hspace=0.42, wspace=0.22
        )
        self.draw_idle()  # type: ignore[no-untyped-call]

    def show_flow(
        self,
        rocket: Rocket | None,
        *,
        velocity: float,
        streamlines: bool = True,
        force_turbulent: bool = True,
    ) -> str:
        """Solve and draw the flow about a design.

        Parameters
        ----------
        rocket:
            The design, or ``None`` to clear the view.
        velocity:
            Free-stream test speed [m/s].
        streamlines:
            Trace streamlines through the field.
        force_turbulent:
            Trip the boundary layer near the tip, which is what a real finish
            does. Turn it off to see natural transition on an ideal surface.

        Returns
        -------
        str
            A one-line summary of the result, for a status label.
        """
        self.clear()
        for ax in (self.field_axes, self.pressure_axes, self.layer_axes):
            ax.clear()
            self._style_axes(ax)

        if rocket is None:
            self.field_axes.text(
                0.5,
                0.5,
                "Adjust the design on the left to see the flow here",
                ha="center",
                va="center",
                color=TEXT_MUTED,
                transform=self.field_axes.transAxes,
            )
            self.finish()
            return ""

        state = Atmosphere.standard().state_at(0.0)
        mach = velocity / state.speed_of_sound
        solution = solve_potential_flow(rocket, velocity=velocity, mach=mach)
        layer = solve_boundary_layer(
            solution,
            kinematic_viscosity=state.kinematic_viscosity,
            force_turbulent=force_turbulent,
        )

        self._draw_field(rocket, solution, streamlines=streamlines)
        self._draw_surface_pressure(solution)
        self._draw_boundary_layer(solution, layer)
        self.finish()

        separation = (
            f"separates at {layer.separation_arc_length * 1e3:.0f} mm"
            if layer.separation_arc_length is not None
            else "attached to the tail"
        )
        return (
            f"{velocity:.0f} m/s  (Mach {mach:.2f})   -   "
            f"suction peak Cp {solution.suction_peak_cp:+.3f} at "
            f"{solution.profile.x[solution.suction_peak_index] * 1e3:.0f} mm   -   "
            f"boundary layer {separation}"
        )

    # -- Field --------------------------------------------------------------

    def _draw_field(
        self,
        rocket: Rocket,
        solution: PotentialFlowSolution,
        *,
        streamlines: bool,
    ) -> None:
        """Draw the pressure field, the body and the streamlines."""
        profile = solution.profile
        length = float(profile.x[-1])
        half_height = _FIELD_HEIGHT * rocket.reference_diameter

        x = np.linspace(-0.12 * length, 1.16 * length, _GRID_X)
        r = np.linspace(0.0, half_height, _GRID_R)
        grid_x, grid_r = np.meshgrid(x, r)

        cp = solution.pressure_field(grid_x, grid_r)
        inside = self._inside_body(profile, grid_x, grid_r)
        cp = np.ma.masked_array(cp, mask=inside)

        # A symmetric scale keeps the sign of the pressure readable; the tip
        # singularity of a slender-body solution would otherwise set the range.
        limit = max(float(np.percentile(np.abs(cp.compressed()), 99.0)), 0.05)

        # The flow is axisymmetric, so the half solved above is mirrored rather
        # than solved twice, and contoured in one pass rather than two.
        mirrored_r = np.concatenate([-grid_r[:0:-1], grid_r])
        mirrored_x = np.concatenate([grid_x[:0:-1], grid_x])
        mirrored_cp = np.ma.concatenate([cp[:0:-1], cp])  # type: ignore[no-untyped-call]

        self.field_axes.contourf(
            mirrored_x * 1e3,
            mirrored_r * 1e3,
            mirrored_cp,
            levels=np.linspace(-limit, limit, 25),
            cmap="RdBu_r",
            extend="both",
        )

        if streamlines:
            self._draw_streamlines(solution, x, half_height, profile)

        self._draw_body_outline(rocket, profile)

        self.field_axes.set_xlabel("station [mm]")
        self.field_axes.set_ylabel("radius [mm]")
        self.field_axes.set_title(
            "Pressure coefficient through the axis  -  blue is suction, red is "
            "compression",
            fontsize=9,
        )
        self.field_axes.set_xlim(x[0] * 1e3, x[-1] * 1e3)
        self.field_axes.set_ylim(-half_height * 1e3, half_height * 1e3)
        self.field_axes.set_aspect("equal", adjustable="box")
        self.field_axes.grid(False)

    @staticmethod
    def _inside_body(
        profile: BodyProfile,
        grid_x: NDArray[np.float64],
        grid_r: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return a mask of grid points inside the body."""
        surface = np.interp(
            grid_x, profile.x, profile.radius, left=0.0, right=profile.radius[-1]
        )
        within = (grid_x >= profile.x[0]) & (grid_x <= profile.x[-1])
        inside: NDArray[np.bool_] = (grid_r < surface) & within
        return inside

    def _draw_streamlines(
        self,
        solution: PotentialFlowSolution,
        x: NDArray[np.float64],
        half_height: float,
        profile: BodyProfile,
    ) -> None:
        """Trace streamlines from upstream by integrating the velocity field.

        Matplotlib's own streamplot needs a regular grid and draws happily
        through the body. Marching the seeds forward instead keeps every line
        outside the surface, which is what makes the picture readable.
        """
        px = np.full(_STREAMLINE_COUNT, float(x[0]))
        pr = np.linspace(0.02, 1.0, _STREAMLINE_COUNT) * half_height
        step = float(x[-1] - x[0]) / (_GRID_X * 1.2)
        steps = int(_GRID_X * 1.6)

        # All the seeds are marched together, so the field is evaluated once per
        # step rather than once per step per line. That is the difference
        # between a redraw you notice and one you do not.
        track_x = np.empty((steps + 1, _STREAMLINE_COUNT))
        track_r = np.empty_like(track_x)
        track_x[0], track_r[0] = px, pr

        for index in range(steps):
            u, v = solution.velocity_field(px, np.maximum(pr, 1e-5))
            speed = np.hypot(u, v)
            speed[speed < 1e-9] = 1e-9
            px = px + step * u / speed
            pr = pr + step * v / speed
            # Never let a line cross into the body.
            surface = np.interp(px, profile.x, profile.radius, left=0.0, right=0.0)
            pr = np.maximum(pr, surface)
            track_x[index + 1], track_r[index + 1] = px, pr

        beyond = track_x > x[-1]
        track_x = np.where(beyond, np.nan, track_x)
        track_r = np.where(beyond, np.nan, track_r)

        for sign in (1.0, -1.0):
            self.field_axes.plot(
                track_x * 1e3,
                sign * track_r * 1e3,
                color=_STREAMLINE_COLOUR,
                linewidth=0.7,
                alpha=0.85,
                zorder=4,
            )

    def _draw_body_outline(self, rocket: Rocket, profile: BodyProfile) -> None:
        """Fill the body silhouette over the field."""
        xs = list(profile.x)
        rs = list(profile.radius)
        # Close the silhouette around the tail.
        aft = rocket.sections[-1].section
        base = (
            aft.aft_radius if isinstance(aft, Transition) else aft.outer_radius
        )
        xs.append(float(profile.x[-1]))
        rs.append(base)

        upper = np.array(rs)
        self.field_axes.fill_between(
            np.array(xs) * 1e3,
            -upper * 1e3,
            upper * 1e3,
            facecolor=SURFACE,
            edgecolor=TEXT_MUTED,
            linewidth=1.1,
            zorder=5,
        )

    # -- Sections -----------------------------------------------------------

    def _draw_surface_pressure(self, solution: PotentialFlowSolution) -> None:
        """Plot the surface pressure coefficient with its features marked."""
        ax = self.pressure_axes
        start = solution.valid_from_index
        x = solution.profile.x[start:] * 1e3
        cp = solution.surface_cp[start:]

        ax.plot(x, cp, color=ACCENT, linewidth=1.6)
        ax.axhline(0.0, color=BORDER, linewidth=0.8)

        peak = solution.suction_peak_index
        ax.plot(
            solution.profile.x[peak] * 1e3,
            solution.surface_cp[peak],
            "v",
            color=WARN,
            markersize=7,
            label=f"suction peak {solution.suction_peak_cp:+.2f}",
        )
        stagnation = solution.stagnation_index
        ax.plot(
            solution.profile.x[stagnation] * 1e3,
            solution.surface_cp[stagnation],
            "^",
            color=PASS,
            markersize=7,
            label="stagnation",
        )

        # Cp is conventionally drawn with suction upwards.
        ax.invert_yaxis()
        ax.set_xlabel("station [mm]")
        ax.set_ylabel("$C_p$")
        ax.set_title("Surface pressure", fontsize=9)
        legend = ax.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER)
        for text in legend.get_texts():
            text.set_color(TEXT)

    def _draw_boundary_layer(
        self, solution: PotentialFlowSolution, layer: BoundaryLayerSolution
    ) -> None:
        """Plot boundary-layer growth with transition and separation marked."""
        ax = self.layer_axes
        arc = layer.arc_length * 1e3

        ax.plot(
            arc,
            layer.boundary_layer_thickness * 1e6,
            color=ACCENT,
            linewidth=1.6,
            label="total thickness",
        )
        ax.plot(
            arc,
            layer.displacement_thickness * 1e6,
            color=TEXT_MUTED,
            linewidth=1.2,
            linestyle="--",
            label="displacement",
        )

        if layer.transition_arc_length is not None:
            ax.axvline(
                layer.transition_arc_length * 1e3,
                color=WARN,
                linewidth=1.1,
                linestyle=":",
                label="transition",
            )
        if layer.separation_arc_length is not None:
            ax.axvline(
                layer.separation_arc_length * 1e3,
                color=FAIL,
                linewidth=1.4,
                label="separation",
            )

        del solution
        ax.set_xlabel("surface distance [mm]")
        ax.set_ylabel("thickness [µm]")
        ax.set_title("Boundary layer", fontsize=9)
        legend = ax.legend(fontsize=7, facecolor=SURFACE, edgecolor=BORDER)
        for text in legend.get_texts():
            text.set_color(TEXT)


class WindTunnelPage(QWidget):
    """The wind tunnel view with its test controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(parent)
        self._rocket: Rocket | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.canvas = WindTunnelCanvas()
        layout.addWidget(self.canvas, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("Test speed"))
        self.speed_box = QComboBox()
        for label, speed in _TEST_SPEEDS:
            self.speed_box.addItem(label, speed)
        self.speed_box.setCurrentIndex(2)
        self.speed_box.currentIndexChanged.connect(self._refresh)
        controls.addWidget(self.speed_box)

        self.streamline_box = QCheckBox("Streamlines")
        self.streamline_box.setChecked(True)
        self.streamline_box.toggled.connect(self._refresh)
        controls.addWidget(self.streamline_box)

        self.turbulent_box = QCheckBox("Tripped boundary layer")
        self.turbulent_box.setChecked(True)
        self.turbulent_box.setToolTip(
            "A wound tube's seam and ordinary paint trip the boundary layer "
            "within a few centimetres of the tip, so this is the realistic "
            "case. Uncheck to see where a perfectly smooth body would "
            "transition on its own."
        )
        self.turbulent_box.toggled.connect(self._refresh)
        controls.addWidget(self.turbulent_box)

        controls.addStretch(1)

        self.readout = QLabel("")
        self.readout.setProperty("muted", True)
        self.readout.setAlignment(Qt.AlignmentFlag.AlignRight)
        controls.addWidget(self.readout)

        layout.addLayout(controls)

    def set_rocket(self, rocket: Rocket | None) -> None:
        """Show the flow about a design.

        Parameters
        ----------
        rocket:
            The design, or ``None`` to clear the view.
        """
        self._rocket = rocket
        self._refresh()

    def _refresh(self) -> None:
        """Re-solve and redraw at the current settings."""
        summary = self.canvas.show_flow(
            self._rocket,
            velocity=float(self.speed_box.currentData()),
            streamlines=self.streamline_box.isChecked(),
            force_turbulent=self.turbulent_box.isChecked(),
        )
        self.readout.setText(summary)
