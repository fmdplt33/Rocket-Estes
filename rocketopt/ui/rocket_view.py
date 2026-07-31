"""Interactive scale side-elevation view of a rocket.

Drawn with :class:`QPainter` rather than a 3D toolkit. For judging whether a
design is sensible, a to-scale side elevation showing the fin planform and the
relative positions of CG and CP conveys more, faster, than a shaded 3D model -
and it adds no dependency beyond Qt itself.

Direct fin editing
------------------
The fin planform carries three drag handles which between them control all four
fin dimensions:

===============  =========================================================
Handle           Effect
===============  =========================================================
Root leading     Dragging forward lengthens the root chord. The root
edge             trailing edge stays flush with the aft end of the body, so
                 this is the dimension a builder actually sets.
Tip leading      Horizontal drag sets leading-edge sweep; vertical drag
edge             sets exposed span.
Tip trailing     Horizontal drag sets the tip chord. Dragging it onto the
edge             tip leading edge gives a delta fin.
===============  =========================================================

Editing is two-way: the handles move when the numeric fields change, and the
numeric fields update as the handles are dragged. Neither is the master copy -
they are two views of the same values.

Navigation
----------
The view fits the whole rocket by default. The wheel zooms about the pointer, so
whatever is under it stays under it, and dragging anywhere but on a handle pans.
Zoomed in far enough to see a fin fillet or the wall of a tube, the handles keep
working, so a fin can be shaped at whatever scale suits. Double-clicking, or the
Fit button beside the view, returns to the whole rocket.

The static margin is the most important number on the screen, so the gap
between the CG and CP markers is annotated directly with its value in calibres
and colour-coded against the stability band.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QToolTip, QWidget

from rocketopt.geometry.components import Transition
from rocketopt.geometry.rocket import Rocket
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
from rocketopt.utils.constants import (
    MAX_STATIC_MARGIN_CALIBRES,
    MIN_STATIC_MARGIN_CALIBRES,
)

__all__ = ["RocketView", "FinHandle"]

_MARGIN_PX = 56
"""Padding around the drawing, leaving room for markers and labels."""

_HANDLE_RADIUS = 6.0
"""Drawn radius of a drag handle, in pixels."""

_GRAB_RADIUS = 13.0
"""Pointer distance within which a handle is grabbed, in pixels."""

_MIN_ZOOM = 0.4
_MAX_ZOOM = 60.0
"""Zoom limits relative to the fitted scale. The upper end resolves a 0.1 mm
wall on a 25 mm tube; the lower end keeps the rocket from vanishing."""

_ZOOM_PER_NOTCH = 1.2
"""Zoom factor for one wheel detent."""

# Fin dimension limits, matching the spin-box ranges in the design panel so
# that dragging can never produce a value the numeric fields would reject.
_MIN_CHORD_MM = 10.0
_MAX_CHORD_MM = 200.0
_MIN_SPAN_MM = 5.0
_MAX_SPAN_MM = 150.0
_MIN_TIP_MM = 0.0
_MAX_SWEEP_MM = 200.0


class FinHandle(str, Enum):
    """Which fin control point is being dragged."""

    ROOT_LEADING = "root_leading"
    TIP_LEADING = "tip_leading"
    TIP_TRAILING = "tip_trailing"

    @property
    def hint(self) -> str:
        """Tooltip explaining what this handle does."""
        return {
            FinHandle.ROOT_LEADING: "Drag to set root chord",
            FinHandle.TIP_LEADING: "Drag to set sweep and span",
            FinHandle.TIP_TRAILING: "Drag to set tip chord",
        }[self]


@dataclass(slots=True)
class _Transform:
    """The model-to-screen mapping from the most recent paint."""

    scale: float = 1.0
    origin_x: float = 0.0
    axis_y: float = 0.0

    def to_screen(self, x: float, r: float) -> QPointF:
        """Map a model station and radius to a screen point."""
        return QPointF(self.origin_x + x * self.scale, self.axis_y - r * self.scale)

    def to_model_x(self, screen_x: float) -> float:
        """Map a screen x back to an axial station [m]."""
        return (screen_x - self.origin_x) / self.scale

    def to_model_r(self, screen_y: float) -> float:
        """Map a screen y back to a radius [m]."""
        return (self.axis_y - screen_y) / self.scale


class RocketView(QWidget):
    """A to-scale, directly editable side elevation of the current design.

    Signals
    -------
    finsEdited(float, float, float, float)
        Emitted while a handle is dragged, carrying root chord, tip chord,
        span and sweep, all in millimetres.
    """

    finsEdited = Signal(float, float, float, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise an empty view."""
        super().__init__(parent)
        self._rocket: Rocket | None = None
        self._cg: float = 0.0
        self._cp: float = 0.0
        self._transform = _Transform()
        self._dragging: FinHandle | None = None
        self._hovered: FinHandle | None = None
        self._editable: bool = True
        self._zoom: float = 1.0
        self._pan = QPointF(0.0, 0.0)
        self._panning_from: QPointF | None = None

        self.setMinimumHeight(260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def set_rocket(self, rocket: Rocket | None, cg: float, cp: float) -> None:
        """Update the displayed design.

        Parameters
        ----------
        rocket:
            The design, or ``None`` to clear the view.
        cg:
            Centre of gravity aft of the nose tip [m].
        cp:
            Centre of pressure aft of the nose tip [m].
        """
        self._rocket = rocket
        self._cg = cg
        self._cp = cp
        self.update()

    def set_editable(self, editable: bool) -> None:
        """Enable or disable direct fin editing."""
        self._editable = editable
        if not editable:
            self._dragging = None
            self._hovered = None
        self.update()

    # -- View navigation -----------------------------------------------------

    def fit(self) -> None:
        """Return to the fitted view of the whole rocket."""
        self._zoom = 1.0
        self._pan = QPointF(0.0, 0.0)
        self.update()

    @property
    def zoom(self) -> float:
        """Current magnification relative to the fitted scale [-]."""
        return self._zoom

    def _base_transform(self) -> _Transform | None:
        """Return the fitted, unzoomed and unpanned model-to-screen mapping.

        Shared by painting and by the wheel handler, which has to know where a
        point would land before and after a zoom to keep it under the pointer.
        """
        rocket = self._rocket
        if rocket is None:
            return None

        usable_width = self.width() - 2 * _MARGIN_PX
        usable_height = self.height() - 2 * _MARGIN_PX
        if usable_width <= 20 or usable_height <= 20:
            return None

        half_height = max(
            rocket.reference_radius,
            rocket.fins.span + rocket.fins.body_radius,
        )
        scale = min(
            usable_width / rocket.length,
            usable_height / (2.0 * half_height),
        )
        return _Transform(
            scale=scale, origin_x=float(_MARGIN_PX), axis_y=self.height() / 2.0
        )

    def _current_transform(self) -> _Transform | None:
        """Return the mapping in force, zoom and pan included."""
        base = self._base_transform()
        if base is None:
            return None
        return _Transform(
            scale=base.scale * self._zoom,
            origin_x=base.origin_x + self._pan.x(),
            axis_y=base.axis_y + self._pan.y(),
        )

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        """Zoom about the pointer."""
        base = self._base_transform()
        if base is None:
            return
        notches = event.angleDelta().y() / 120.0
        if not notches:
            return

        zoom = min(max(self._zoom * _ZOOM_PER_NOTCH**notches, _MIN_ZOOM), _MAX_ZOOM)
        if zoom == self._zoom:
            return

        # Hold the model point under the pointer still: solve the pan that maps
        # it back to the same screen position at the new scale.
        current = self._current_transform()
        assert current is not None
        pointer = event.position()
        model_x = current.to_model_x(pointer.x())
        model_r = current.to_model_r(pointer.y())

        self._zoom = zoom
        scale = base.scale * zoom
        self._pan = QPointF(
            pointer.x() - base.origin_x - model_x * scale,
            pointer.y() - base.axis_y + model_r * scale,
        )
        self.update()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        """Return to the fitted view."""
        del event
        self.fit()

    # -- Handle geometry ----------------------------------------------------

    def _handle_positions(self) -> dict[FinHandle, QPointF]:
        """Return the screen position of each fin drag handle."""
        rocket = self._rocket
        if rocket is None:
            return {}

        fins = rocket.fins
        t = self._transform
        body_r = fins.body_radius
        tip_r = body_r + fins.span

        return {
            FinHandle.ROOT_LEADING: t.to_screen(fins.position, body_r),
            FinHandle.TIP_LEADING: t.to_screen(
                fins.position + fins.sweep_length, tip_r
            ),
            FinHandle.TIP_TRAILING: t.to_screen(
                fins.position + fins.sweep_length + fins.tip_chord, tip_r
            ),
        }

    def _handle_at(self, point: QPointF) -> FinHandle | None:
        """Return the handle near ``point``, or ``None``."""
        best: FinHandle | None = None
        best_distance = _GRAB_RADIUS
        for handle, position in self._handle_positions().items():
            distance = math.hypot(
                point.x() - position.x(), point.y() - position.y()
            )
            if distance <= best_distance:
                best = handle
                best_distance = distance
        return best

    # -- Mouse interaction ---------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        """Grab a handle, or begin a pan.

        The left button edits a fin when it lands on a handle and pans
        otherwise, so the same button does the obvious thing in both places.
        """
        if self._rocket is None:
            return

        handle = (
            self._handle_at(event.position())
            if self._editable and event.button() is Qt.MouseButton.LeftButton
            else None
        )
        if handle is not None:
            self._dragging = handle
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            self.update()
            return

        if event.button() in (
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.RightButton,
        ):
            self._panning_from = event.position()
            self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        """Pan, update the fin being dragged, or highlight a hovered handle."""
        if self._rocket is None:
            return

        if self._panning_from is not None:
            self._pan += event.position() - self._panning_from
            self._panning_from = event.position()
            self.update()
            return

        if not self._editable:
            return

        if self._dragging is None:
            hovered = self._handle_at(event.position())
            if hovered is not self._hovered:
                self._hovered = hovered
                self.setCursor(
                    Qt.CursorShape.OpenHandCursor
                    if hovered
                    else Qt.CursorShape.ArrowCursor
                )
                if hovered is not None:
                    QToolTip.showText(
                        event.globalPosition().toPoint(), hovered.hint, self
                    )
                self.update()
            return

        self._apply_drag(event.position())

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        """End a drag or a pan."""
        del event
        if self._panning_from is not None:
            self._panning_from = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        if self._dragging is not None:
            self._dragging = None
            self.setCursor(
                Qt.CursorShape.OpenHandCursor
                if self._hovered
                else Qt.CursorShape.ArrowCursor
            )
            self.update()

    def leaveEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        """Clear the hover highlight when the pointer leaves."""
        del event
        if self._hovered is not None:
            self._hovered = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()

    def _apply_drag(self, point: QPointF) -> None:
        """Translate a drag position into new fin dimensions and emit them.

        Values are clamped to the same limits the numeric spin boxes enforce,
        so dragging can never create a fin the design panel would refuse.
        """
        rocket = self._rocket
        if rocket is None:
            return

        fins = rocket.fins
        t = self._transform

        root_mm = fins.root_chord * 1e3
        tip_mm = fins.tip_chord * 1e3
        span_mm = fins.span * 1e3
        sweep_mm = fins.sweep_length * 1e3

        model_x_mm = t.to_model_x(point.x()) * 1e3
        model_r_mm = abs(t.to_model_r(point.y())) * 1e3
        body_r_mm = fins.body_radius * 1e3

        # The root trailing edge is pinned to the aft end of the body, so the
        # root chord is measured forward from there.
        root_trailing_mm = (fins.position + fins.root_chord) * 1e3

        if self._dragging is FinHandle.ROOT_LEADING:
            root_mm = root_trailing_mm - model_x_mm

        elif self._dragging is FinHandle.TIP_LEADING:
            span_mm = model_r_mm - body_r_mm
            root_leading_mm = root_trailing_mm - root_mm
            sweep_mm = model_x_mm - root_leading_mm

        elif self._dragging is FinHandle.TIP_TRAILING:
            root_leading_mm = root_trailing_mm - root_mm
            tip_mm = model_x_mm - root_leading_mm - sweep_mm

        root_mm = min(max(root_mm, _MIN_CHORD_MM), _MAX_CHORD_MM)
        span_mm = min(max(span_mm, _MIN_SPAN_MM), _MAX_SPAN_MM)
        sweep_mm = min(max(sweep_mm, 0.0), _MAX_SWEEP_MM)
        # A tip chord cannot exceed the root chord in this planform family.
        tip_mm = min(max(tip_mm, _MIN_TIP_MM), root_mm)

        self.finsEdited.emit(root_mm, tip_mm, span_mm, sweep_mm)

    # -- Painting ----------------------------------------------------------

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        """Render the rocket elevation."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(SURFACE))

        rocket = self._rocket
        if rocket is None:
            self._draw_placeholder(painter)
            painter.end()
            return

        transform = self._current_transform()
        if transform is None:
            painter.end()
            return

        self._transform = transform
        t = transform
        scale = t.scale

        def sx(x: float) -> float:
            """Model axial station to screen x."""
            return t.origin_x + x * t.scale

        def sy(r: float) -> float:
            """Model radius to screen y."""
            return t.axis_y - r * t.scale

        self._draw_centreline(painter, sx(0.0), sx(rocket.length), t.axis_y)
        self._draw_fins(painter, rocket, sx, sy)
        self._draw_body(painter, rocket, sx, sy)
        self._draw_motor(painter, rocket, sx, sy)
        self._draw_markers(painter, rocket, sx, t.axis_y)
        if self._editable:
            self._draw_handles(painter)
        self._draw_scale_bar(painter, rocket, scale, t.origin_x)

        painter.end()

    def _draw_placeholder(self, painter: QPainter) -> None:
        """Draw the empty-state message."""
        painter.setPen(QPen(QColor(TEXT_MUTED)))
        font = QFont()
        font.setPointSize(11)
        painter.setFont(font)
        painter.drawText(
            self.rect(),
            Qt.AlignmentFlag.AlignCenter,
            "Adjust the design on the left to see it here",
        )

    def _draw_centreline(
        self, painter: QPainter, x0: float, x1: float, axis_y: float
    ) -> None:
        """Draw the body axis as a dashed centreline."""
        painter.setPen(QPen(QColor(BORDER), 1, Qt.PenStyle.DashLine))
        painter.drawLine(QPointF(x0 - 20, axis_y), QPointF(x1 + 20, axis_y))

    def _draw_body(
        self,
        painter: QPainter,
        rocket: Rocket,
        sx: Callable[[float], float],
        sy: Callable[[float], float],
    ) -> None:
        """Draw the nose and body outline as a filled silhouette."""
        path = QPainterPath()

        xs, ys = rocket.nose.profile_points(120)
        path.moveTo(QPointF(sx(float(xs[0])), sy(float(ys[0]))))
        for x, y in zip(xs, ys, strict=True):
            path.lineTo(QPointF(sx(float(x)), sy(float(y))))

        for placed in rocket.sections:
            section = placed.section
            if isinstance(section, Transition):
                path.lineTo(QPointF(sx(placed.position), sy(section.fore_radius)))
                path.lineTo(QPointF(sx(placed.aft_position), sy(section.aft_radius)))
            else:
                path.lineTo(QPointF(sx(placed.position), sy(section.outer_radius)))
                path.lineTo(QPointF(sx(placed.aft_position), sy(section.outer_radius)))

        for placed in reversed(rocket.sections):
            section = placed.section
            if isinstance(section, Transition):
                path.lineTo(QPointF(sx(placed.aft_position), sy(-section.aft_radius)))
                path.lineTo(QPointF(sx(placed.position), sy(-section.fore_radius)))
            else:
                path.lineTo(QPointF(sx(placed.aft_position), sy(-section.outer_radius)))
                path.lineTo(QPointF(sx(placed.position), sy(-section.outer_radius)))

        for x, y in zip(reversed(xs), reversed(ys), strict=True):
            path.lineTo(QPointF(sx(float(x)), sy(-float(y))))
        path.closeSubpath()

        painter.setBrush(QBrush(QColor(62, 68, 78)))
        painter.setPen(QPen(QColor(TEXT_MUTED), 1.4))
        painter.drawPath(path)

    def _draw_fins(
        self,
        painter: QPainter,
        rocket: Rocket,
        sx: Callable[[float], float],
        sy: Callable[[float], float],
    ) -> None:
        """Draw the fin planform above and below the body."""
        fins = rocket.fins
        root_x = fins.position
        body_r = fins.body_radius
        tip_r = body_r + fins.span

        active = self._dragging is not None or self._hovered is not None
        fill = QColor(ACCENT).darker(150 if active else 160)
        painter.setBrush(QBrush(fill))
        painter.setPen(QPen(QColor(ACCENT), 1.8 if active else 1.4))

        for sign in (1.0, -1.0):
            path = QPainterPath()
            path.moveTo(QPointF(sx(root_x), sy(sign * body_r)))
            path.lineTo(QPointF(sx(root_x + fins.sweep_length), sy(sign * tip_r)))
            path.lineTo(
                QPointF(
                    sx(root_x + fins.sweep_length + fins.tip_chord), sy(sign * tip_r)
                )
            )
            path.lineTo(QPointF(sx(root_x + fins.root_chord), sy(sign * body_r)))
            path.closeSubpath()
            painter.drawPath(path)

    def _draw_motor(
        self,
        painter: QPainter,
        rocket: Rocket,
        sx: Callable[[float], float],
        sy: Callable[[float], float],
    ) -> None:
        """Outline the installed motor inside the airframe."""
        motor = rocket.motor.motor
        x0 = rocket.motor_position
        radius = motor.diameter / 2.0

        rect = QRectF(
            QPointF(sx(x0), sy(radius)),
            QPointF(sx(x0 + motor.length), sy(-radius)),
        )
        painter.setBrush(QBrush(QColor(WARN).darker(200)))
        painter.setPen(QPen(QColor(WARN), 1.2, Qt.PenStyle.DashLine))
        painter.drawRect(rect)

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(WARN)))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, rocket.motor.designation)

    def _draw_handles(self, painter: QPainter) -> None:
        """Draw the fin drag handles, highlighting hover and drag states."""
        for handle, position in self._handle_positions().items():
            is_active = handle is self._dragging
            is_hovered = handle is self._hovered

            radius = _HANDLE_RADIUS + (2.0 if is_active or is_hovered else 0.0)
            rect = QRectF(
                position.x() - radius,
                position.y() - radius,
                2 * radius,
                2 * radius,
            )

            if is_active:
                painter.setBrush(QBrush(QColor(TEXT)))
                painter.setPen(QPen(QColor(TEXT), 2))
            elif is_hovered:
                painter.setBrush(QBrush(QColor(ACCENT)))
                painter.setPen(QPen(QColor(TEXT), 1.6))
            else:
                painter.setBrush(QBrush(QColor(SURFACE)))
                painter.setPen(QPen(QColor(ACCENT), 1.8))
            painter.drawEllipse(rect)

    def _draw_markers(
        self,
        painter: QPainter,
        rocket: Rocket,
        sx: Callable[[float], float],
        axis_y: float,
    ) -> None:
        """Draw the CG and CP markers and annotate the static margin."""
        calibre = rocket.reference_diameter
        margin = (self._cp - self._cg) / calibre if calibre > 0 else 0.0

        if margin < MIN_STATIC_MARGIN_CALIBRES:
            margin_colour = FAIL if margin < 0.5 else WARN
        elif margin > MAX_STATIC_MARGIN_CALIBRES:
            margin_colour = WARN
        else:
            margin_colour = PASS

        cg_x = sx(self._cg)
        cp_x = sx(self._cp)
        marker_y = axis_y

        painter.setPen(QPen(QColor(margin_colour), 2))
        painter.drawLine(QPointF(cg_x, marker_y), QPointF(cp_x, marker_y))

        label_y = max(marker_y - self.height() * 0.32, 18.0)
        painter.setPen(QPen(QColor(margin_colour), 1, Qt.PenStyle.DotLine))
        for x in (cg_x, cp_x):
            painter.drawLine(QPointF(x, label_y + 16), QPointF(x, marker_y - 10))
        painter.setPen(QPen(QColor(margin_colour), 2))
        painter.drawLine(QPointF(cg_x, label_y + 16), QPointF(cp_x, label_y + 16))

        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        centre = 0.5 * (cg_x + cp_x)
        painter.drawText(
            QRectF(centre - 80.0, label_y - 4, 160.0, 20.0),
            Qt.AlignmentFlag.AlignCenter,
            f"static margin {margin:+.2f} cal",
        )

        self._draw_cg_marker(painter, cg_x, marker_y)
        self._draw_cp_marker(painter, cp_x, marker_y)

        font.setBold(False)
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(TEXT_MUTED)))
        painter.drawText(QPointF(cg_x - 8, marker_y + 30), "CG")
        painter.drawText(QPointF(cp_x - 8, marker_y + 30), "CP")

    def _draw_cg_marker(self, painter: QPainter, x: float, y: float) -> None:
        """Draw the standard quartered-circle centre-of-gravity symbol."""
        radius = 8.0
        rect = QRectF(x - radius, y - radius, 2 * radius, 2 * radius)
        painter.setPen(QPen(QColor(TEXT), 1.5))
        painter.setBrush(QBrush(QColor(TEXT)))
        painter.drawPie(rect, 0, 90 * 16)
        painter.drawPie(rect, 180 * 16, 90 * 16)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)

    def _draw_cp_marker(self, painter: QPainter, x: float, y: float) -> None:
        """Draw the centre-of-pressure marker as an open circle with a cross."""
        radius = 8.0
        rect = QRectF(x - radius, y - radius, 2 * radius, 2 * radius)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(ACCENT), 2))
        painter.drawEllipse(rect)
        painter.drawLine(QPointF(x - radius, y), QPointF(x + radius, y))
        painter.drawLine(QPointF(x, y - radius), QPointF(x, y + radius))

    def _draw_scale_bar(
        self, painter: QPainter, rocket: Rocket, scale: float, origin_x: float
    ) -> None:
        """Draw a labelled scale bar so dimensions can be read off directly.

        The bar is sized from what is on screen rather than from the rocket, so
        it stays a useful length at any zoom: a quarter of the visible width,
        rounded to one significant figure.
        """
        del rocket
        target = (self.width() / scale) / 4.0
        exponent = math.floor(math.log10(target)) if target > 0 else -2
        step = 10.0**exponent
        bar_m = round(target / step) * step
        if bar_m <= 0:
            return

        y = self.height() - 24.0
        x0 = float(_MARGIN_PX)
        x1 = x0 + bar_m * scale
        del origin_x

        painter.setPen(QPen(QColor(TEXT_MUTED), 1.4))
        painter.drawLine(QPointF(x0, y), QPointF(x1, y))
        painter.drawLine(QPointF(x0, y - 4), QPointF(x0, y + 4))
        painter.drawLine(QPointF(x1, y - 4), QPointF(x1, y + 4))

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(
            QRectF(x0, y + 4, x1 - x0, 16),
            Qt.AlignmentFlag.AlignCenter,
            f"{bar_m * 1e3:.3g} mm",
        )

        painter.setPen(QPen(QColor(TEXT_MUTED)))
        note = "Drag the circles on the fin to reshape it"
        if abs(self._zoom - 1.0) > 1e-3:
            note = f"{self._zoom:.1f}x  -  double-click to fit"
        painter.drawText(
            QRectF(x1 + 14, y - 4, 360, 16),
            Qt.AlignmentFlag.AlignLeft,
            note,
        )
