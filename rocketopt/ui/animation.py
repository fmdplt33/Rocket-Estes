"""Animated flight replay: the rocket flying along its altitude-time curve.

The rocket travels along the altitude-against-time trace, pitching to follow
the curve's tangent, so it climbs steeply under power, arcs over at apogee and
descends under the canopy. A speed-against-time trace sits beneath on the same
time axis, so the two align vertically and can be read together.

Plotting altitude against *time* rather than against downrange distance is a
deliberate choice: in still air a downrange plot is a vertical line and shows
nothing, whereas the altitude-time curve is the shape a flyer recognises and
carries the whole flight profile. Horizontal drift is reported numerically in
the telemetry panel instead.

Events are derived from the simulated trajectory rather than assumed: top speed
and maximum dynamic pressure are located by scanning the samples, so they land
wherever the physics actually put them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPolygonF,
)
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from rocketopt.flight.simulation import FlightPhase, FlightResult
from rocketopt.ui.theme import (
    ACCENT,
    BORDER,
    FAIL,
    PASS,
    SURFACE,
    SURFACE_RAISED,
    TEXT,
    TEXT_MUTED,
    WARN,
)

__all__ = ["FlightEvent", "AnimationCanvas", "AnimationPage"]

_FRAME_MS = 25
"""Timer interval, giving 40 frames per second."""


@dataclass(frozen=True, slots=True)
class FlightEvent:
    """A marked point on the altitude-time curve.

    Attributes
    ----------
    label:
        Short name shown on the marker.
    detail:
        Value text shown under the label.
    time:
        Time of the event [s].
    altitude:
        Altitude at the event [m].
    colour:
        Marker colour.
    """

    label: str
    detail: str
    time: float
    altitude: float
    colour: str


def extract_events(result: FlightResult) -> list[FlightEvent]:
    """Derive the notable events from a flight result.

    Maximum speed and maximum dynamic pressure are found by scanning the
    trajectory samples rather than assumed to coincide with burnout, because on
    a long-burning motor they do not.

    Parameters
    ----------
    result:
        A completed flight simulation.

    Returns
    -------
    list of FlightEvent
        Events in time order.
    """
    if not result.states:
        return []

    states = result.states
    events: list[FlightEvent] = []

    def at_time(t: float):
        """Return the sample nearest a given time."""
        return min(states, key=lambda s: abs(s.time - t))

    if result.rail_exit_time > 0.0:
        sample = at_time(result.rail_exit_time)
        events.append(
            FlightEvent(
                "Rail exit",
                f"{result.rail_exit_velocity:.0f} m/s",
                result.rail_exit_time,
                sample.altitude,
                TEXT_MUTED,
            )
        )

    peak_accel = max(states, key=lambda s: float((s.acceleration**2).sum()))
    magnitude = math.sqrt(float((peak_accel.acceleration**2).sum()))
    events.append(
        FlightEvent(
            "Max acceleration",
            f"{magnitude / 9.80665:.0f} g",
            peak_accel.time,
            peak_accel.altitude,
            FAIL,
        )
    )

    peak_q = max(states, key=lambda s: s.dynamic_pressure)
    events.append(
        FlightEvent(
            "Max Q",
            f"{peak_q.dynamic_pressure / 1000.0:.1f} kPa",
            peak_q.time,
            peak_q.altitude,
            WARN,
        )
    )

    peak_v = max(states, key=lambda s: s.speed)
    events.append(
        FlightEvent(
            "Top speed",
            f"{peak_v.speed:.0f} m/s  Mach {peak_v.mach:.2f}",
            peak_v.time,
            peak_v.altitude,
            ACCENT,
        )
    )

    burn_time = result.rocket.motor.motor.burn_time
    sample = at_time(burn_time)
    events.append(
        FlightEvent(
            "Booster cutoff",
            f"{result.burnout_velocity:.0f} m/s at {result.burnout_altitude:.0f} m",
            burn_time,
            result.burnout_altitude or sample.altitude,
            WARN,
        )
    )

    events.append(
        FlightEvent(
            "Apogee",
            f"{result.apogee:.0f} m",
            result.apogee_time,
            result.apogee,
            PASS,
        )
    )

    ejection_time = result.rocket.motor.ejection_time
    if ejection_time <= states[-1].time:
        sample = at_time(ejection_time)
        events.append(
            FlightEvent(
                "Ejection",
                f"{result.ejection_altitude:.0f} m",
                ejection_time,
                result.ejection_altitude or sample.altitude,
                FAIL,
            )
        )

    if result.landing_time > 0.0:
        events.append(
            FlightEvent(
                "Landing",
                f"{result.landing_velocity:.1f} m/s, "
                f"{result.landing_distance:.0f} m out",
                result.landing_time,
                0.0,
                TEXT_MUTED,
            )
        )

    events.sort(key=lambda e: e.time)
    return events


class AnimationCanvas(QWidget):
    """Paints the altitude-time curve, the rocket on it, and the speed trace."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise an empty canvas."""
        super().__init__(parent)
        self._result: FlightResult | None = None
        self._events: list[FlightEvent] = []
        self._time: float = 0.0
        self._ascent_only: bool = False
        self.setMinimumHeight(380)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_flight(self, result: FlightResult | None) -> None:
        """Load a flight to replay."""
        self._result = result
        self._events = extract_events(result) if result else []
        self._time = 0.0
        self.update()

    def set_time(self, t: float) -> None:
        """Move the playhead to time ``t`` [s]."""
        self._time = t
        self.update()

    def set_ascent_only(self, ascent_only: bool) -> None:
        """Restrict the time axis to the powered and coasting phases."""
        self._ascent_only = ascent_only
        self.update()

    @property
    def duration(self) -> float:
        """Total flight duration [s]."""
        if self._result is None or not self._result.states:
            return 0.0
        return self._result.states[-1].time

    @property
    def window(self) -> float:
        """Time span currently shown on the axis [s]."""
        duration = self.duration
        if not self._ascent_only or self._result is None:
            return duration
        ejection = self._result.rocket.motor.ejection_time
        return min(duration, max(ejection * 1.25, 1.0))

    def state_at(self, t: float):
        """Return the trajectory sample nearest time ``t``."""
        if self._result is None or not self._result.states:
            return None
        return min(self._result.states, key=lambda s: abs(s.time - t))

    # -- Painting -----------------------------------------------------------

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        """Render the current animation frame."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(SURFACE))

        result = self._result
        if result is None or not result.states:
            painter.setPen(QPen(QColor(TEXT_MUTED)))
            font = QFont()
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Adjust the design, then press Play to watch the flight",
            )
            painter.end()
            return

        states = result.states
        left = 76
        right_margin = 190
        top = 34
        bottom_margin = 44

        width = self.width() - left - right_margin
        if width < 120:
            painter.end()
            return

        # Split the plotting area: altitude above, speed below, sharing the
        # same time axis so the two curves line up vertically.
        available = self.height() - top - bottom_margin
        speed_height = min(max(int(available * 0.30), 80), 150)
        gap = 46
        altitude_height = available - speed_height - gap
        if altitude_height < 90:
            painter.end()
            return

        window = max(self.window, 1e-6)
        # Ten percent of headroom above apogee. Without it the apogee marker
        # and its label sit on the panel's top edge, underneath the title.
        max_altitude = max(max(s.altitude for s in states), 1.0) * 1.10

        def tx(t: float) -> float:
            """Time to screen x."""
            return left + (min(t, window) / window) * width

        def ay(altitude: float) -> float:
            """Altitude to screen y in the upper panel."""
            return top + altitude_height - (altitude / max_altitude) * altitude_height

        self._draw_axes(
            painter, left, top, width, altitude_height, max_altitude, "m", 5
        )

        title = QFont()
        title.setPointSize(9)
        painter.setFont(title)
        painter.setPen(QPen(QColor(TEXT_MUTED)))
        painter.drawText(
            QRectF(left, top - 20, 480, 16),
            Qt.AlignmentFlag.AlignLeft,
            "Altitude against time - the rocket flies along this curve",
        )
        self._draw_event_lines(painter, tx, top, altitude_height, window)
        self._draw_altitude_curve(painter, states, tx, ay, window)
        self._draw_events(painter, tx, ay, window)

        current = self.state_at(self._time)
        if current is not None:
            self._draw_rocket_on_curve(painter, states, current, tx, ay, window)

        self._draw_time_axis(painter, tx, top + altitude_height, window)

        speed_top = top + altitude_height + gap
        self._draw_speed_panel(
            painter, states, left, speed_top, width, speed_height, tx, window
        )

        if current is not None:
            self._draw_readouts(painter, current)

        painter.end()

    def _draw_axes(
        self,
        painter: QPainter,
        left: int,
        top: float,
        width: int,
        height: int,
        maximum: float,
        unit: str,
        divisions: int,
    ) -> None:
        """Draw horizontal gridlines with value labels."""
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        raw_step = maximum / divisions
        exponent = math.floor(math.log10(raw_step)) if raw_step > 0 else 0
        base = 10.0**exponent
        step = min([1, 2, 5, 10], key=lambda m: abs(m * base - raw_step)) * base

        value = 0.0
        while value <= maximum * 1.001:
            y = top + height - (value / maximum) * height
            painter.setPen(QPen(QColor(BORDER), 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(left, y), QPointF(left + width, y))
            painter.setPen(QPen(QColor(TEXT_MUTED)))
            painter.drawText(
                QRectF(left - 70, y - 8, 62, 16),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{value:.0f} {unit}",
            )
            value += step

        painter.setPen(QPen(QColor(BORDER), 1.2))
        painter.drawLine(QPointF(left, top), QPointF(left, top + height))
        painter.drawLine(
            QPointF(left, top + height), QPointF(left + width, top + height)
        )

    def _draw_event_lines(
        self, painter: QPainter, tx, top: float, height: int, window: float
    ) -> None:
        """Draw vertical event lines through the altitude panel."""
        for event in self._events:
            if event.time > window:
                continue
            colour = QColor(event.colour)
            colour.setAlpha(140 if event.time <= self._time else 50)
            painter.setPen(QPen(colour, 1, Qt.PenStyle.DashLine))
            x = tx(event.time)
            painter.drawLine(QPointF(x, top), QPointF(x, top + height))

    def _draw_altitude_curve(
        self, painter: QPainter, states, tx, ay, window: float
    ) -> None:
        """Draw the whole altitude curve faintly, the flown part brightly."""
        full = QPainterPath()
        flown = QPainterPath()
        started = False

        for index, s in enumerate(states):
            if s.time > window:
                break
            point = QPointF(tx(s.time), ay(s.altitude))
            if index == 0:
                full.moveTo(point)
            else:
                full.lineTo(point)
            if s.time <= self._time:
                if not started:
                    flown.moveTo(point)
                    started = True
                else:
                    flown.lineTo(point)

        painter.setPen(QPen(QColor(BORDER), 1.6, Qt.PenStyle.DashLine))
        painter.drawPath(full)

        if started:
            # Fill under the flown portion so progress reads at a glance.
            fill = QPainterPath(flown)
            fill.lineTo(QPointF(tx(min(self._time, window)), ay(0.0)))
            fill.lineTo(QPointF(tx(0.0), ay(0.0)))
            fill.closeSubpath()
            shade = QColor(ACCENT)
            shade.setAlpha(38)
            painter.setBrush(QBrush(shade))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(fill)

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(ACCENT), 2.6))
            painter.drawPath(flown)

    def _draw_events(self, painter: QPainter, tx, ay, window: float) -> None:
        """Draw event markers on the curve with collision-avoided labels."""
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)

        occupied: dict[bool, list[float]] = {True: [], False: []}
        label_height = 28.0

        for index, event in enumerate(self._events):
            if event.time > window:
                continue

            reached = event.time <= self._time
            colour = QColor(event.colour)
            if not reached:
                colour.setAlpha(75)

            x = tx(event.time)
            y = ay(event.altitude)

            painter.setPen(QPen(colour, 1.6))
            painter.setBrush(QBrush(colour) if reached else Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QRectF(x - 4.5, y - 4.5, 9.0, 9.0))

            # Choose the side by available space, not by alternating blindly:
            # an event early in the flight sits hard against the left axis and
            # a left-hand label would be clipped off the edge.
            label_width = 140.0
            room_left = x - label_width - 20.0 >= 0.0
            room_right = x + label_width + 20.0 <= self.width() - 186.0
            if room_left and room_right:
                to_right = index % 2 == 0
            else:
                to_right = room_right

            label_x = x + 14 if to_right else x - (label_width + 12.0)
            alignment = (
                Qt.AlignmentFlag.AlignLeft if to_right else Qt.AlignmentFlag.AlignRight
            )

            label_y = y - 8
            for _ in range(len(self._events)):
                clash = next(
                    (
                        used
                        for used in occupied[to_right]
                        if abs(used - label_y) < label_height
                    ),
                    None,
                )
                if clash is None:
                    break
                label_y = clash + label_height
            occupied[to_right].append(label_y)

            painter.setPen(QPen(colour, 1.0))
            elbow = x + (11 if to_right else -11)
            painter.drawLine(QPointF(x + (5 if to_right else -5), y), QPointF(elbow, y))
            if abs(label_y + 8 - y) > 1.0:
                painter.drawLine(QPointF(elbow, y), QPointF(elbow, label_y + 8))

            text_colour = QColor(TEXT if reached else TEXT_MUTED)
            if not reached:
                text_colour.setAlpha(120)

            bold = QFont()
            bold.setPointSize(8)
            bold.setBold(reached)
            painter.setFont(bold)
            painter.setPen(QPen(text_colour))
            painter.drawText(
                QRectF(label_x, label_y - 6, 140, 13),
                alignment | Qt.AlignmentFlag.AlignVCenter,
                event.label,
            )
            painter.setFont(font)
            painter.setPen(QPen(QColor(TEXT_MUTED)))
            painter.drawText(
                QRectF(label_x, label_y + 6, 140, 13),
                alignment | Qt.AlignmentFlag.AlignVCenter,
                event.detail,
            )

    def _curve_tangent(self, states, current, tx, ay, window: float) -> float:
        """Screen-space heading of the altitude curve at the playhead [rad].

        Taken from the curve as drawn, not from the physical velocity vector:
        the rocket is travelling along this trace, so following its visual
        slope is what makes the motion read correctly. It pitches over at
        apogee and points downward on the way back.

        Parameters
        ----------
        states:
            Trajectory samples.
        current:
            The sample at the playhead.
        tx, ay:
            Screen mapping functions.
        window:
            Time span shown.

        Returns
        -------
        float
            Heading in screen coordinates, where zero points right.
        """
        span = max(window * 0.012, 1e-3)
        before = self.state_at(max(current.time - span, 0.0))
        after = self.state_at(min(current.time + span, states[-1].time))
        if before is None or after is None:
            return 0.0

        dx = tx(after.time) - tx(before.time)
        dy = ay(after.altitude) - ay(before.altitude)
        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            return 0.0
        return math.atan2(dy, dx)

    def _draw_rocket_on_curve(
        self, painter: QPainter, states, current, tx, ay, window: float
    ) -> None:
        """Draw the rocket at the playhead, pitched along the curve."""
        x = tx(min(current.time, window))
        y = ay(current.altitude)
        angle = self._curve_tangent(states, current, tx, ay, window)

        painter.save()
        painter.translate(x, y)
        painter.rotate(math.degrees(angle))

        length = 26.0
        half_width = 5.0

        if current.thrust > 0.0 and current.phase in {
            FlightPhase.ON_RAIL,
            FlightPhase.BOOST,
        }:
            plume = QPolygonF(
                [
                    QPointF(-length * 0.5, -half_width * 0.75),
                    QPointF(-length * 0.5 - 30.0, 0.0),
                    QPointF(-length * 0.5, half_width * 0.75),
                ]
            )
            flame = QColor(WARN)
            flame.setAlpha(210)
            painter.setBrush(QBrush(flame))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(plume)

        fins = QPolygonF(
            [
                QPointF(-length * 0.5, -half_width),
                QPointF(-length * 0.5 - 6.0, -half_width - 5.0),
                QPointF(-length * 0.5 - 6.0, half_width + 5.0),
                QPointF(-length * 0.5, half_width),
            ]
        )
        painter.setBrush(QBrush(QColor(ACCENT)))
        painter.setPen(QPen(QColor(ACCENT).darker(140), 1.0))
        painter.drawPolygon(fins)

        body = QPolygonF(
            [
                QPointF(length * 0.5, 0.0),
                QPointF(length * 0.12, -half_width),
                QPointF(-length * 0.5, -half_width),
                QPointF(-length * 0.5, half_width),
                QPointF(length * 0.12, half_width),
            ]
        )
        painter.setBrush(QBrush(QColor(TEXT)))
        painter.setPen(QPen(QColor(SURFACE_RAISED), 1.0))
        painter.drawPolygon(body)

        painter.restore()

        # Parachute, drawn upright rather than rotated with the airframe.
        if current.phase is FlightPhase.DESCENT:
            canopy = QRectF(x - 17, y - 32, 34, 20)
            painter.setPen(QPen(QColor(PASS), 2.0))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawArc(canopy, 0, 180 * 16)
            painter.drawLine(QPointF(x - 16, y - 22), QPointF(x, y - 4))
            painter.drawLine(QPointF(x + 16, y - 22), QPointF(x, y - 4))

    def _draw_time_axis(
        self, painter: QPainter, tx, baseline: float, window: float
    ) -> None:
        """Label the shared time axis."""
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(TEXT_MUTED)))

        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            t = window * fraction
            painter.drawText(
                QRectF(tx(t) - 30, baseline + 4, 60, 14),
                Qt.AlignmentFlag.AlignCenter,
                f"{t:.1f} s",
            )

        # The panel title lives above the plot, not below it: putting it here
        # placed it directly on top of the speed panel's own title.

    def _draw_speed_panel(
        self,
        painter: QPainter,
        states,
        left: int,
        top: float,
        width: int,
        height: int,
        tx,
        window: float,
    ) -> None:
        """Draw the speed trace on the same time axis as the altitude curve."""
        # Headroom above the peak, for the same reason as the altitude panel.
        max_speed = max(max(s.speed for s in states), 1.0) * 1.12

        def sy(speed: float) -> float:
            """Speed to screen y."""
            return top + height - (speed / max_speed) * height

        self._draw_axes(painter, left, top, width, height, max_speed, "m/s", 3)

        for event in self._events:
            if event.time > window:
                continue
            colour = QColor(event.colour)
            colour.setAlpha(120 if event.time <= self._time else 45)
            painter.setPen(QPen(colour, 1, Qt.PenStyle.DashLine))
            x = tx(event.time)
            painter.drawLine(QPointF(x, top), QPointF(x, top + height))

        full = QPainterPath()
        flown = QPainterPath()
        started = False
        for index, s in enumerate(states):
            if s.time > window:
                break
            point = QPointF(tx(s.time), sy(s.speed))
            if index == 0:
                full.moveTo(point)
            else:
                full.lineTo(point)
            if s.time <= self._time:
                if not started:
                    flown.moveTo(point)
                    started = True
                else:
                    flown.lineTo(point)

        painter.setPen(QPen(QColor(BORDER), 1.4, Qt.PenStyle.DashLine))
        painter.drawPath(full)
        if started:
            painter.setPen(QPen(QColor(PASS), 2.4))
            painter.drawPath(flown)

        current = self.state_at(self._time)
        if current is not None and current.time <= window:
            x = tx(current.time)
            y = sy(current.speed)
            painter.setBrush(QBrush(QColor(PASS)))
            painter.setPen(QPen(QColor(TEXT), 1.5))
            painter.drawEllipse(QRectF(x - 5, y - 5, 10, 10))

            bold = QFont()
            bold.setPointSize(9)
            bold.setBold(True)
            painter.setFont(bold)
            painter.setPen(QPen(QColor(TEXT)))
            label_x = min(x + 10, left + width - 84)
            label_y = y + 6 if (y - top) < 26 else y - 20
            painter.drawText(
                QRectF(label_x, label_y, 84, 16),
                Qt.AlignmentFlag.AlignLeft,
                f"{current.speed:.0f} m/s",
            )

        title = QFont()
        title.setPointSize(9)
        painter.setFont(title)
        painter.setPen(QPen(QColor(TEXT_MUTED)))
        painter.drawText(
            QRectF(left, top - 18, 400, 16),
            Qt.AlignmentFlag.AlignLeft,
            "Speed against time",
        )

    def _draw_readouts(self, painter: QPainter, state) -> None:
        """Draw the live telemetry panel on the right."""
        panel_x = self.width() - 178
        panel_y = 34

        painter.setPen(QPen(QColor(BORDER)))
        painter.setBrush(QBrush(QColor(SURFACE_RAISED)))
        painter.drawRoundedRect(QRectF(panel_x, panel_y, 164, 216), 6, 6)

        accel = math.sqrt(float((state.acceleration**2).sum()))
        rows = [
            ("Time", f"{state.time:.2f} s"),
            ("Phase", state.phase.label),
            ("Altitude", f"{state.altitude:.0f} m"),
            ("Speed", f"{state.speed:.0f} m/s"),
            ("Vertical", f"{state.vertical_velocity:+.0f} m/s"),
            ("Mach", f"{state.mach:.2f}"),
            ("Acceleration", f"{accel / 9.80665:.1f} g"),
            ("Thrust", f"{state.thrust:.1f} N"),
            ("Mass", f"{state.mass * 1e3:.0f} g"),
            ("Downrange", f"{state.downrange:.0f} m"),
        ]

        label_font = QFont()
        label_font.setPointSize(8)
        value_font = QFont()
        value_font.setPointSize(9)
        value_font.setBold(True)

        y = panel_y + 14
        for label, value in rows:
            painter.setFont(label_font)
            painter.setPen(QPen(QColor(TEXT_MUTED)))
            painter.drawText(
                QRectF(panel_x + 10, y, 78, 16), Qt.AlignmentFlag.AlignLeft, label
            )
            painter.setFont(value_font)
            painter.setPen(QPen(QColor(TEXT)))
            painter.drawText(
                QRectF(panel_x + 82, y, 74, 16), Qt.AlignmentFlag.AlignRight, value
            )
            y += 20


class AnimationPage(QWidget):
    """The animation tab: canvas plus playback controls."""

    scrubbed = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.canvas = AnimationCanvas()
        layout.addWidget(self.canvas, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        self.play_button = QPushButton("Play")
        self.play_button.setProperty("primary", True)
        self.play_button.setFixedWidth(84)
        self.play_button.clicked.connect(self._toggle)
        controls.addWidget(self.play_button)

        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self.restart)
        controls.addWidget(self.restart_button)

        self.speed_box = QComboBox()
        for label, factor in (
            ("0.1x slow", 0.1),
            ("0.25x", 0.25),
            ("0.5x", 0.5),
            ("1x real time", 1.0),
            ("2x", 2.0),
            ("5x", 5.0),
        ):
            self.speed_box.addItem(label, factor)
        self.speed_box.setCurrentIndex(2)
        self.speed_box.setToolTip(
            "Playback rate. Boost lasts under two seconds, so 0.25x or slower "
            "is usually needed to see the powered phase."
        )
        controls.addWidget(self.speed_box)

        self.span_box = QComboBox()
        self.span_box.addItem("Whole flight", False)
        self.span_box.addItem("Ascent only", True)
        self.span_box.setToolTip(
            "Descent under the canopy takes most of the flight time, so the "
            "ascent view spreads the powered and coasting phases across the "
            "full width."
        )
        self.span_box.currentIndexChanged.connect(self._on_span_changed)
        controls.addWidget(self.span_box)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setToolTip("Drag to scrub through the flight.")
        self.slider.sliderMoved.connect(self._on_scrub)
        self.slider.sliderPressed.connect(self.pause)
        controls.addWidget(self.slider, stretch=1)

        self.time_label = QLabel("0.00 s")
        self.time_label.setFixedWidth(104)
        self.time_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        controls.addWidget(self.time_label)

        layout.addLayout(controls)

        hint = QLabel(
            "The rocket flies along its altitude-time curve, pitching to follow "
            "the slope. Events are read from the simulated trajectory, so top "
            "speed and max Q appear wherever the physics puts them."
        )
        hint.setWordWrap(True)
        hint.setProperty("muted", True)
        layout.addWidget(hint)

        self._timer = QTimer(self)
        self._timer.setInterval(_FRAME_MS)
        self._timer.timeout.connect(self._tick)
        self._time = 0.0

    def set_flight(self, result: FlightResult | None) -> None:
        """Load a flight, resetting playback."""
        self.pause()
        self._time = 0.0
        self.canvas.set_flight(result)
        self.slider.setValue(0)
        self._update_time_label()

    def _on_span_changed(self, index: int) -> None:
        """Switch between the whole flight and the ascent only."""
        del index
        self.canvas.set_ascent_only(bool(self.span_box.currentData()))
        # Keep the playhead inside the new window.
        self._time = min(self._time, self.canvas.window)
        self.canvas.set_time(self._time)
        self._update_time_label()

    def _toggle(self) -> None:
        """Play or pause."""
        if self._timer.isActive():
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        """Start playback, restarting if it has already finished."""
        if self.canvas.window <= 0.0:
            return
        if self._time >= self.canvas.window:
            self._time = 0.0
        self._timer.start()
        self.play_button.setText("Pause")

    def pause(self) -> None:
        """Pause playback."""
        self._timer.stop()
        self.play_button.setText("Play")

    def restart(self) -> None:
        """Rewind to ignition and play."""
        self._time = 0.0
        self.canvas.set_time(0.0)
        self.slider.setValue(0)
        self._update_time_label()
        self.play()

    def _tick(self) -> None:
        """Advance the playhead one frame."""
        window = self.canvas.window
        if window <= 0.0:
            self.pause()
            return

        factor = float(self.speed_box.currentData())
        self._time += (_FRAME_MS / 1000.0) * factor

        if self._time >= window:
            self._time = window
            self.pause()

        self.canvas.set_time(self._time)
        self.slider.setValue(int(1000.0 * self._time / window))
        self._update_time_label()

    def _on_scrub(self, value: int) -> None:
        """Handle the user dragging the scrub slider."""
        window = self.canvas.window
        if window <= 0.0:
            return
        self._time = window * value / 1000.0
        self.canvas.set_time(self._time)
        self._update_time_label()

    def _update_time_label(self) -> None:
        """Refresh the time readout."""
        self.time_label.setText(f"{self._time:.2f} / {self.canvas.window:.1f} s")
