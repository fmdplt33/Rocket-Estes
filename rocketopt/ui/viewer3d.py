"""An interactive shaded 3D view of the assembled rocket.

Rendered by a small software rasteriser rather than OpenGL or VTK. The meshes
already exist - :mod:`rocketopt.cad.assembly` places the same solids that get
exported - and a rocket is a few thousand triangles, which Qt fills fast enough
to orbit smoothly. That keeps the 3D view working in the base install, with no
GPU, no driver and no extra dependency, which matters more here than the last
few frames per second would.

How it draws
------------
Every triangle in the assembly is transformed into camera space at once with
NumPy, back-facing triangles are dropped, the rest are sorted far-to-near and
filled in that order - the painter's algorithm. Triangles are shaded by the
angle between their normal and a light fixed to the camera, so the shape reads
correctly from any direction without needing a depth buffer.

The painter's algorithm cannot resolve triangles that pass through one another,
which on a rocket means the odd flicker where a fin root enters the body tube.
For judging a design that is a fair trade for the simplicity.

Controls
--------
============================  ===========================================
Drag with the left button     Orbit
Drag with the right button    Pan
Wheel                         Zoom
============================  ===========================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QMouseEvent,
    QPainter,
    QPen,
    QPolygonF,
    QWheelEvent,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from rocketopt.cad.assembly import AssemblyPart, assembled_parts
from rocketopt.geometry.rocket import Rocket
from rocketopt.ui.theme import BORDER, SURFACE, TEXT_MUTED

__all__ = ["Viewer3D", "Viewer3DPage"]

_AMBIENT: Final[float] = 0.34
"""Fraction of a surface's colour that shows where no light reaches it."""

_ZOOM_PER_NOTCH: Final[float] = 1.15
"""Zoom factor for one wheel detent."""

_MIN_ELEVATION: Final[float] = -85.0
_MAX_ELEVATION: Final[float] = 85.0
"""Elevation limits [deg], short of the poles where the up vector degenerates."""

_FOCAL: Final[float] = 1.6
"""Focal length as a multiple of the viewport half-height; about a 64 degree
field of view, wide enough to see a whole rocket without obvious distortion."""

_FIT_FRACTION: Final[float] = 0.88
"""Fraction of the viewport a fitted assembly fills, leaving a little air."""

_XRAY_ALPHA: Final[int] = 96
"""Opacity of the airframe skin in x-ray, out of 255."""

_MIN_FACET_AREA: Final[float] = 0.2
"""Projected area below which a triangle is skipped [px^2].

Edge-on facets of a cylinder collapse to slivers thinner than a pixel. Skipping
them is invisible and takes a useful slice off the triangle count."""

STANDARD_VIEWS: Final[dict[str, tuple[float, float]]] = {
    "Three-quarter": (52.0, 22.0),
    "Side": (90.0, 0.0),
    "Top": (90.0, 84.0),
    "Front": (0.0, 0.0),
    "Aft": (180.0, 0.0),
}
"""Named ``(azimuth, elevation)`` camera positions, in degrees."""


@dataclass(slots=True)
class _Camera:
    """An orbit camera looking at a point in the assembly.

    Attributes
    ----------
    azimuth, elevation:
        Orbit angles [deg]. Azimuth turns about the world ``Y`` axis, so the
        body axis stays horizontal on screen.
    distance:
        Eye distance from the target [mm].
    target:
        Point the camera looks at [mm].
    """

    azimuth: float = 52.0
    elevation: float = 22.0
    distance: float = 600.0
    target: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    def basis(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return the camera's ``(right, up, forward)`` unit axes.

        Forward points from the eye towards the target, so a point's depth is
        its projection onto it.
        """
        azimuth = math.radians(self.azimuth)
        elevation = math.radians(self.elevation)
        # Direction from the target out to the eye. World up is +Y and the body
        # axis is +Z, so this puts the rocket across the screen rather than up
        # and down it.
        to_eye = np.array(
            [
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
                math.cos(elevation) * math.cos(azimuth),
            ]
        )
        forward = -to_eye
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        return right, up, forward

    @property
    def eye(self) -> NDArray[np.float64]:
        """Position of the eye [mm]."""
        _, _, forward = self.basis()
        return self.target - forward * self.distance


class Viewer3D(QWidget):
    """A shaded, orbitable view of the assembled rocket."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialise an empty viewer."""
        super().__init__(parent)
        self._parts: tuple[AssemblyPart, ...] = ()
        self._camera = _Camera()
        self._home_distance = 600.0
        self._last_pos: QPointF | None = None
        self._orbiting = False
        self._panning = False
        self._cutaway = False
        self._xray = False
        self._rocket_length = 0.0
        # A design can arrive before the widget has been laid out, and the fit
        # depends on the viewport aspect ratio, so it is deferred to the paint.
        self._needs_fit = False

        # Flattened triangle soup, rebuilt only when the design changes.
        self._vertices: NDArray[np.float64] = np.zeros((0, 3))
        self._faces: NDArray[np.int64] = np.zeros((0, 3), dtype=np.int64)
        self._colours: NDArray[np.float64] = np.zeros((0, 3))
        self._skin: NDArray[np.bool_] = np.zeros(0, dtype=bool)
        self._sectionable: NDArray[np.bool_] = np.zeros(0, dtype=bool)
        self._centroids: NDArray[np.float64] = np.zeros((0, 3))

        self.setMinimumHeight(320)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    # -- Contents -----------------------------------------------------------

    def set_rocket(self, rocket: Rocket | None) -> None:
        """Rebuild the scene for a design.

        Parameters
        ----------
        rocket:
            The design, or ``None`` to clear the view.
        """
        if rocket is None:
            self._parts = ()
            self._vertices = np.zeros((0, 3))
            self._faces = np.zeros((0, 3), dtype=np.int64)
            self._rocket_length = 0.0
            self.update()
            return

        self._parts = assembled_parts(rocket)
        self._rocket_length = rocket.length * 1e3
        self._flatten()
        self._needs_fit = True
        self.reset_view()
        self.update()

    def _flatten(self) -> None:
        """Concatenate every part into one array of triangles.

        Transforming and sorting one array is what keeps the frame time in
        milliseconds; doing it part by part in Python is an order of magnitude
        slower.
        """
        vertices: list[NDArray[np.float64]] = []
        faces: list[NDArray[np.int64]] = []
        colours: list[NDArray[np.float64]] = []
        skin: list[NDArray[np.bool_]] = []
        sectionable: list[NDArray[np.bool_]] = []

        offset = 0
        for part in self._parts:
            mesh = part.mesh
            vertices.append(mesh.vertices)
            faces.append(mesh.faces + offset)
            colours.append(
                np.tile(
                    np.asarray(part.colour, dtype=np.float64),
                    (mesh.triangle_count, 1),
                )
            )
            skin.append(np.full(mesh.triangle_count, part.opaque, dtype=bool))
            sectionable.append(
                np.full(mesh.triangle_count, part.axisymmetric, dtype=bool)
            )
            offset += mesh.vertex_count

        self._vertices = np.concatenate(vertices) if vertices else np.zeros((0, 3))
        self._faces = (
            np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64)
        )
        self._colours = np.concatenate(colours) if colours else np.zeros((0, 3))
        self._skin = np.concatenate(skin) if skin else np.zeros(0, dtype=bool)
        self._sectionable = (
            np.concatenate(sectionable) if sectionable else np.zeros(0, dtype=bool)
        )
        # Cached because the cutaway test needs them on every frame and they
        # never change while the design does not.
        self._centroids = (
            self._vertices[self._faces].mean(axis=1)
            if len(self._faces)
            else np.zeros((0, 3))
        )

    @property
    def triangle_count(self) -> int:
        """Number of triangles in the scene."""
        return len(self._faces)

    # -- Camera -------------------------------------------------------------

    def reset_view(self) -> None:
        """Frame the whole assembly."""
        if not self._parts:
            return

        low = self._vertices.min(axis=0)
        high = self._vertices.max(axis=0)
        self._camera.target = 0.5 * (low + high)

        # Measure the vertices themselves across the screen axes. A bounding
        # sphere would frame the rocket for its diagonal, and even a bounding
        # box has corners the rocket never reaches - a fin's span crossed with
        # the nose's station - so either would leave the viewport half empty.
        right, up, forward = self._camera.basis()
        relative = self._vertices - self._camera.target
        lateral = np.abs(relative @ right)
        vertical = np.abs(relative @ up)
        # How far each corner reaches towards the camera. Under perspective a
        # near corner projects larger than a far one at the same offset, so the
        # standoff has to be solved per corner rather than from the extremes.
        towards = -(relative @ forward)

        # Half-angles of the frustum, as tangents.
        tan_vertical = 1.0 / _FOCAL
        tan_horizontal = tan_vertical * max(self.width(), 1) / max(self.height(), 1)

        # A corner at offset ``o`` and depth ``d - t`` fits when
        # ``o / (d - t) <= tan * fill``, so each corner sets a lower bound on d.
        bounds = (
            np.maximum(
                lateral / (tan_horizontal * _FIT_FRACTION),
                vertical / (tan_vertical * _FIT_FRACTION),
            )
            + towards
        )

        self._home_distance = max(float(bounds.max()), 1.0)
        self._camera.distance = self._home_distance
        self._needs_fit = False

    def fit(self) -> None:
        """Reframe the assembly and repaint."""
        self.reset_view()
        self.update()

    def set_view(self, name: str) -> None:
        """Snap the camera to a named standard view.

        Parameters
        ----------
        name:
            A key of :data:`STANDARD_VIEWS`.
        """
        azimuth, elevation = STANDARD_VIEWS[name]
        self._camera.azimuth = azimuth
        self._camera.elevation = elevation
        # The fit depends on which way the assembly is presented, so a standard
        # view reframes as well as turning.
        self.reset_view()
        self.update()

    def set_cutaway(self, cutaway: bool) -> None:
        """Show or hide the near half of the airframe."""
        self._cutaway = cutaway
        self.update()

    def set_xray(self, xray: bool) -> None:
        """Make the outer skin translucent so the internals show through."""
        self._xray = xray
        self.update()

    # -- Interaction --------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        """Begin an orbit or a pan."""
        self._last_pos = event.position()
        if event.button() is Qt.MouseButton.LeftButton:
            self._orbiting = True
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._panning = True
            self.setCursor(Qt.CursorShape.SizeAllCursor)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        """Orbit or pan with the pointer."""
        if self._last_pos is None or not (self._orbiting or self._panning):
            return

        delta = event.position() - self._last_pos
        self._last_pos = event.position()

        if self._orbiting:
            self._camera.azimuth = (self._camera.azimuth - delta.x() * 0.4) % 360.0
            self._camera.elevation = min(
                max(self._camera.elevation + delta.y() * 0.4, _MIN_ELEVATION),
                _MAX_ELEVATION,
            )
        else:
            right, up, _ = self._camera.basis()
            # One pixel of drag moves the target by one pixel at the target's
            # depth, so the model tracks the pointer.
            scale = self._camera.distance / (_FOCAL * max(self.height(), 1) / 2.0)
            self._camera.target = (
                self._camera.target - right * delta.x() * scale + up * delta.y() * scale
            )
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        """End an orbit or pan."""
        del event
        self._orbiting = False
        self._panning = False
        self._last_pos = None
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def resizeEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        """Refit if the viewport changes before the view has been touched."""
        super().resizeEvent(event)  # type: ignore[arg-type]
        if self._parts and self._camera.distance == self._home_distance:
            self._needs_fit = True

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt override
        """Reframe the assembly."""
        del event
        self.fit()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        """Zoom towards or away from the target."""
        notches = event.angleDelta().y() / 120.0
        if not notches:
            return
        self._camera.distance = min(
            max(
                self._camera.distance / (_ZOOM_PER_NOTCH**notches),
                self._home_distance * 0.05,
            ),
            self._home_distance * 12.0,
        )
        self.update()

    # -- Rendering ----------------------------------------------------------

    def _project(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
        """Project every vertex to screen space.

        Returns
        -------
        tuple
            Screen coordinates ``(n, 2)``, depth along the view axis ``(n,)``,
            and a mask of the vertices in front of the eye.
        """
        right, up, forward = self._camera.basis()
        relative = self._vertices - self._camera.eye
        depth = relative @ forward
        # Keep the divide finite for vertices at or behind the eye; they are
        # masked out by the caller anyway.
        safe = np.where(depth > 1e-6, depth, 1e-6)

        focal = _FOCAL * self.height() / 2.0
        screen = np.empty((len(self._vertices), 2))
        screen[:, 0] = self.width() / 2.0 + focal * (relative @ right) / safe
        screen[:, 1] = self.height() / 2.0 - focal * (relative @ up) / safe
        return screen, depth, depth > 1e-6

    def _visible_faces(self) -> NDArray[np.bool_]:
        """Return a mask of the triangles the current toggles let through.

        The cutaway plane contains the body axis and turns with the camera, so
        the cut always opens towards the viewer however the model is orbited -
        a fixed world plane would show the section edge-on from half the orbit.
        """
        keep = np.ones(len(self._faces), dtype=bool)
        if not self._cutaway:
            return keep

        _, _, forward = self._camera.basis()
        # The camera direction with the body axis taken out of it: the radial
        # direction the viewer is looking from.
        towards_viewer = -forward.copy()
        towards_viewer[2] = 0.0
        norm = float(np.linalg.norm(towards_viewer))
        if norm < 1e-9:  # looking straight down the axis; nothing to cut
            return keep
        towards_viewer /= norm

        # The cut follows facet edges rather than slicing them, which is enough
        # to show wall thicknesses and what sits inside. Fins and the lug are
        # left whole; sectioning them would only leave teeth across the cut.
        near = self._centroids @ towards_viewer > 0.0
        keep &= ~(near & self._sectionable)
        return keep

    def paintEvent(self, event: object) -> None:  # noqa: N802 - Qt override
        """Render the assembly."""
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(SURFACE))

        if not len(self._faces) or self.width() < 10 or self.height() < 10:
            self._draw_placeholder(painter)
            painter.end()
            return

        if self._needs_fit:
            self.reset_view()

        screen, depth, in_front = self._project()

        keep = self._visible_faces()
        faces = self._faces[keep]
        colours = self._colours[keep]
        skin = self._skin[keep]
        if not len(faces):
            painter.end()
            return

        # Only triangles wholly in front of the eye are drawn. Clipping the
        # ones that straddle it would add a lot of machinery to fix a case that
        # only arises when the camera is inside the rocket.
        visible = in_front[faces].all(axis=1)
        faces, colours, skin = faces[visible], colours[visible], skin[visible]
        if not len(faces):
            painter.end()
            return

        corners = screen[faces]
        # Signed area of the projected triangle. Negative means the triangle is
        # wound the other way on screen, so it is facing away: cull it.
        edge1 = corners[:, 1] - corners[:, 0]
        edge2 = corners[:, 2] - corners[:, 0]
        area = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
        facing = area < -_MIN_FACET_AREA
        faces, colours, corners = faces[facing], colours[facing], corners[facing]
        skin = skin[facing]
        if not len(faces):
            painter.end()
            return

        shade = self._shading(faces)
        order = np.argsort(-depth[faces].mean(axis=1))

        painter.setPen(Qt.PenStyle.NoPen)
        shaded = np.clip(colours * shade[:, None], 0.0, 255.0)
        # In x-ray the skin is drawn translucent. Because the triangles are
        # already ordered back to front, plain alpha blending composites them
        # correctly with no further work.
        alpha = np.where(skin & self._xray, _XRAY_ALPHA, 255)
        for index in order:
            triangle = corners[index]
            painter.setBrush(
                QBrush(
                    QColor(
                        int(shaded[index, 0]),
                        int(shaded[index, 1]),
                        int(shaded[index, 2]),
                        int(alpha[index]),
                    )
                )
            )
            painter.drawPolygon(
                QPolygonF([QPointF(x, y) for x, y in triangle])
            )

        self._draw_overlay(painter, len(faces))
        painter.end()

    def _shading(self, faces: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return a brightness multiplier for each triangle.

        A single light sits just above and to the left of the eye, so the
        lighting turns with the camera and no face is ever left unlit.
        """
        corners = self._vertices[faces]
        normals = np.cross(
            corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        )
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0.0] = 1.0
        normals /= lengths[:, None]

        right, up, forward = self._camera.basis()
        light = -forward + 0.45 * up - 0.35 * right
        light /= np.linalg.norm(light)

        diffuse = np.clip(normals @ light, 0.0, 1.0)
        return _AMBIENT + (1.0 - _AMBIENT) * diffuse

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

    def _draw_overlay(self, painter: QPainter, drawn: int) -> None:
        """Draw the scale note and triangle count."""
        # The triangle loop leaves a solid brush set, and drawRect fills with
        # the current brush - which would flood the viewport with the colour of
        # whichever triangle happened to be drawn last.
        painter.setBrush(Qt.BrushStyle.NoBrush)

        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QPen(QColor(TEXT_MUTED)))
        painter.drawText(
            10,
            self.height() - 10,
            f"{self._rocket_length:.0f} mm overall  -  {drawn} of "
            f"{self.triangle_count} triangles  -  drag to orbit, "
            f"right-drag to pan, wheel to zoom",
        )

        painter.setPen(QPen(QColor(BORDER)))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))


class Viewer3DPage(QWidget):
    """The 3D viewer with its view controls."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the page."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self.viewer = Viewer3D()
        layout.addWidget(self.viewer, stretch=1)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("View"))
        self.view_box = QComboBox()
        for name in STANDARD_VIEWS:
            self.view_box.addItem(name)
        self.view_box.currentTextChanged.connect(self.viewer.set_view)
        controls.addWidget(self.view_box)

        self.fit_button = QPushButton("Fit")
        self.fit_button.setToolTip("Reframe the whole rocket. Double-click the view "
                                   "to do the same.")
        self.fit_button.clicked.connect(self.viewer.fit)
        controls.addWidget(self.fit_button)

        self.xray_box = QCheckBox("X-ray")
        self.xray_box.setToolTip(
            "Make the airframe translucent to see the motor mount and the "
            "motor inside it, in place."
        )
        self.xray_box.toggled.connect(self.viewer.set_xray)
        controls.addWidget(self.xray_box)

        self.cutaway_box = QCheckBox("Cutaway")
        self.cutaway_box.setToolTip(
            "Cut the half nearest the camera away, along the body axis, to "
            "show wall thicknesses and the parts inside."
        )
        self.cutaway_box.toggled.connect(self.viewer.set_cutaway)
        controls.addWidget(self.cutaway_box)

        controls.addStretch(1)
        layout.addLayout(controls)

        hint = QLabel(
            "The same solids that get exported, assembled: the nose cone with "
            "its spigot in the tube, the motor mount bridging the bore to the "
            "motor, and the fins with their true section."
        )
        hint.setWordWrap(True)
        hint.setProperty("muted", True)
        layout.addWidget(hint)

    def set_rocket(self, rocket: Rocket | None) -> None:
        """Show a design.

        Parameters
        ----------
        rocket:
            The design, or ``None`` to clear the view.
        """
        self.viewer.set_rocket(rocket)
