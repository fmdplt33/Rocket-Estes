"""Tests for the interactive views: 3D viewer, plan navigation, wind tunnel.

These need PySide6 and are skipped without it. They run offscreen, so no
display is required; what they check is the arithmetic behind the views - where
the camera ends up, that a zoom holds the point under the pointer, that the
flow solver is actually driven - rather than what any of it looks like.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest


def _has_pyside() -> bool:
    """Return whether PySide6 is importable."""
    try:
        import PySide6  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = [
    pytest.mark.requires_ui,
    pytest.mark.skipif(not _has_pyside(), reason="PySide6 is not installed"),
]


@pytest.fixture(scope="module")
def qt_app():
    """A single offscreen Qt application for the module."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# 3D viewer
# ---------------------------------------------------------------------------


def test_camera_basis_is_orthonormal_and_right_handed(qt_app) -> None:
    """The camera frame is a proper rotation at every orbit angle."""
    from rocketopt.ui.viewer3d import _Camera

    for azimuth in (0.0, 47.0, 180.0, 305.0):
        for elevation in (-84.0, -20.0, 0.0, 35.0, 84.0):
            camera = _Camera(azimuth=azimuth, elevation=elevation)
            right, up, forward = camera.basis()

            for axis in (right, up, forward):
                assert float(np.linalg.norm(axis)) == pytest.approx(1.0, abs=1e-12)
            assert float(right @ up) == pytest.approx(0.0, abs=1e-12)
            assert float(right @ forward) == pytest.approx(0.0, abs=1e-12)
            assert np.allclose(np.cross(right, up), -forward, atol=1e-12)


def test_the_eye_sits_its_distance_from_the_target(qt_app) -> None:
    """The camera is exactly ``distance`` away, looking at the target."""
    from rocketopt.ui.viewer3d import _Camera

    camera = _Camera(azimuth=33.0, elevation=17.0, distance=420.0)
    camera.target = np.array([10.0, -5.0, 140.0])
    _, _, forward = camera.basis()

    offset = camera.target - camera.eye
    assert float(np.linalg.norm(offset)) == pytest.approx(420.0, rel=1e-12)
    assert np.allclose(offset / 420.0, forward, atol=1e-12)


def test_the_viewer_frames_the_whole_rocket(qt_app, reference_rocket) -> None:
    """After a fit, every vertex projects inside the viewport."""
    from rocketopt.ui.viewer3d import STANDARD_VIEWS, Viewer3D

    viewer = Viewer3D()
    viewer.resize(900, 500)
    viewer.set_rocket(reference_rocket)

    assert viewer.triangle_count > 1000

    for name in STANDARD_VIEWS:
        viewer.set_view(name)
        screen, _, in_front = viewer._project()

        assert in_front.all(), f"{name}: geometry behind the eye"
        assert screen[:, 0].min() >= 0.0, name
        assert screen[:, 0].max() <= viewer.width(), name
        assert screen[:, 1].min() >= 0.0, name
        assert screen[:, 1].max() <= viewer.height(), name

    # And it really does fill the frame rather than sitting in the middle of it.
    viewer.set_view("Side")
    screen, _, _ = viewer._project()
    width = screen[:, 0].max() - screen[:, 0].min()
    assert width > 0.7 * viewer.width()


def test_zooming_in_and_out_returns_to_the_same_place(qt_app, reference_rocket) -> None:
    """Zoom is bounded and reversible, and Fit restores the framing."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    from rocketopt.ui.viewer3d import Viewer3D

    viewer = Viewer3D()
    viewer.resize(900, 500)
    viewer.set_rocket(reference_rocket)
    fitted = viewer._camera.distance

    def wheel(notches: int) -> None:
        viewer.wheelEvent(
            QWheelEvent(
                QPointF(450, 250),
                QPointF(450, 250),
                QPoint(0, 0),
                QPoint(0, 120 * notches),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
        )

    for _ in range(4):
        wheel(1)
    assert viewer._camera.distance < fitted
    for _ in range(4):
        wheel(-1)
    assert viewer._camera.distance == pytest.approx(fitted, rel=1e-9)

    # Far past the stops the distance clamps rather than running away.
    for _ in range(60):
        wheel(1)
    assert viewer._camera.distance >= fitted * 0.04
    viewer.fit()
    assert viewer._camera.distance == pytest.approx(fitted, rel=1e-9)


def test_the_cutaway_and_xray_change_what_is_drawn(qt_app, reference_rocket) -> None:
    """Sectioning drops the near half; the fins are left whole."""
    from rocketopt.ui.viewer3d import Viewer3D

    viewer = Viewer3D()
    viewer.resize(900, 500)
    viewer.set_rocket(reference_rocket)

    whole = int(viewer._visible_faces().sum())
    viewer.set_cutaway(True)
    sectioned = viewer._visible_faces()

    assert int(sectioned.sum()) < whole
    # Every fin triangle survives: only the bodies of revolution are cut.
    assert bool(sectioned[~viewer._sectionable].all())

    # X-ray hides nothing; it only changes how the skin is painted.
    viewer.set_cutaway(False)
    viewer.set_xray(True)
    assert int(viewer._visible_faces().sum()) == whole


def test_the_viewer_survives_an_empty_design(qt_app) -> None:
    """Clearing the design leaves a drawable, empty viewer."""
    from PySide6.QtGui import QPixmap

    from rocketopt.ui.viewer3d import Viewer3DPage

    page = Viewer3DPage()
    page.resize(600, 400)
    page.set_rocket(None)

    assert page.viewer.triangle_count == 0
    page.viewer.render(QPixmap(page.viewer.size()))  # must not raise


# ---------------------------------------------------------------------------
# Plan view navigation
# ---------------------------------------------------------------------------


def test_zoom_holds_the_point_under_the_pointer(qt_app, reference_rocket) -> None:
    """Whatever is under the pointer stays under it through a zoom."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    from rocketopt.ui.rocket_view import RocketView

    view = RocketView()
    view.resize(1000, 420)
    view.set_rocket(reference_rocket, 0.15, 0.18)

    pointer = QPointF(640.0, 190.0)
    before = view._current_transform()
    assert before is not None
    anchor = (before.to_model_x(pointer.x()), before.to_model_r(pointer.y()))

    for _ in range(6):
        view.wheelEvent(
            QWheelEvent(
                pointer,
                pointer,
                QPoint(0, 0),
                QPoint(0, 120),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
        )

    assert view.zoom > 2.0
    after = view._current_transform()
    assert after is not None
    landed = after.to_screen(*anchor)
    assert landed.x() == pytest.approx(pointer.x(), abs=1e-6)
    assert landed.y() == pytest.approx(pointer.y(), abs=1e-6)


def test_zoom_is_bounded_and_fit_resets_it(qt_app, reference_rocket) -> None:
    """The zoom stops at its limits, and Fit puts the whole rocket back."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent

    from rocketopt.ui.rocket_view import _MAX_ZOOM, _MIN_ZOOM, RocketView

    view = RocketView()
    view.resize(1000, 420)
    view.set_rocket(reference_rocket, 0.15, 0.18)

    def wheel(notches: int) -> None:
        view.wheelEvent(
            QWheelEvent(
                QPointF(500, 210),
                QPointF(500, 210),
                QPoint(0, 0),
                QPoint(0, 120 * notches),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False,
            )
        )

    for _ in range(80):
        wheel(1)
    assert view.zoom == pytest.approx(_MAX_ZOOM)
    for _ in range(160):
        wheel(-1)
    assert view.zoom == pytest.approx(_MIN_ZOOM)

    view.fit()
    assert view.zoom == pytest.approx(1.0)


def test_panning_moves_the_view_not_the_design(qt_app, reference_rocket) -> None:
    """Dragging off a handle pans, and leaves the fins alone."""
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    from rocketopt.ui.rocket_view import RocketView

    view = RocketView()
    view.resize(1000, 420)
    view.set_rocket(reference_rocket, 0.15, 0.18)

    edits: list[tuple[float, ...]] = []
    view.finsEdited.connect(lambda *values: edits.append(values))

    before = view._current_transform()
    assert before is not None

    def mouse(kind, x, y, button):
        return QMouseEvent(
            kind,
            QPointF(x, y),
            button,
            button,
            Qt.KeyboardModifier.NoModifier,
        )

    # A point well clear of the fin handles, which sit at the tail.
    view.mousePressEvent(
        mouse(QMouseEvent.Type.MouseButtonPress, 120, 60, Qt.MouseButton.LeftButton)
    )
    view.mouseMoveEvent(
        mouse(QMouseEvent.Type.MouseMove, 190, 100, Qt.MouseButton.NoButton)
    )
    view.mouseReleaseEvent(
        mouse(QMouseEvent.Type.MouseButtonRelease, 190, 100, Qt.MouseButton.LeftButton)
    )

    after = view._current_transform()
    assert after is not None
    assert after.origin_x == pytest.approx(before.origin_x + 70.0)
    assert after.axis_y == pytest.approx(before.axis_y + 40.0)
    assert after.scale == pytest.approx(before.scale)
    assert not edits, "panning must not edit the fins"


def test_a_fin_handle_still_drags_when_zoomed_in(qt_app, reference_rocket) -> None:
    """Zoom does not break direct fin editing."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent, QWheelEvent

    from rocketopt.ui.rocket_view import FinHandle, RocketView

    view = RocketView()
    view.resize(1000, 420)
    view.set_rocket(reference_rocket, 0.15, 0.18)

    view.wheelEvent(
        QWheelEvent(
            QPointF(800, 210),
            QPointF(800, 210),
            QPoint(0, 0),
            QPoint(0, 240),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
    )
    view._transform = view._current_transform()

    handles = view._handle_positions()
    grab = handles[FinHandle.TIP_LEADING]

    edits: list[tuple[float, ...]] = []
    view.finsEdited.connect(lambda *values: edits.append(values))

    view.mousePressEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            grab,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert view._dragging is FinHandle.TIP_LEADING

    view.mouseMoveEvent(
        QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(grab.x() + 6.0, grab.y() - 12.0),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )

    assert edits, "dragging a handle while zoomed should still edit the fin"
    root, tip, span, sweep = edits[-1]
    assert span > reference_rocket.fins.span * 1e3
    assert root == pytest.approx(reference_rocket.fins.root_chord * 1e3, abs=1e-6)
    del tip, sweep


# ---------------------------------------------------------------------------
# Wind tunnel
# ---------------------------------------------------------------------------


def test_the_wind_tunnel_solves_and_reports(qt_app, reference_rocket) -> None:
    """The tunnel drives the solvers and reports what they found."""
    from rocketopt.ui.windtunnel import WindTunnelPage

    page = WindTunnelPage()
    page.resize(1000, 700)
    page.set_rocket(reference_rocket)

    summary = page.readout.text()
    assert "Mach" in summary
    assert "suction peak" in summary
    assert "boundary layer" in summary

    # Three axes, all populated.
    assert page.canvas.field_axes.collections
    assert page.canvas.pressure_axes.lines
    assert page.canvas.layer_axes.lines


def test_the_wind_tunnel_follows_its_test_speed(qt_app, reference_rocket) -> None:
    """Changing the test speed re-solves at the new condition."""
    from rocketopt.ui.windtunnel import WindTunnelPage

    page = WindTunnelPage()
    page.resize(1000, 700)
    page.set_rocket(reference_rocket)

    page.speed_box.setCurrentIndex(0)
    slow = page.readout.text()
    page.speed_box.setCurrentIndex(page.speed_box.count() - 1)
    fast = page.readout.text()

    assert slow != fast
    assert "20 m/s" in slow
    assert "250 m/s" in fast
    # Compressibility shows up as a higher Mach number, not just a higher speed.
    slow_mach = float(slow.split("Mach ")[1].split(")")[0])
    fast_mach = float(fast.split("Mach ")[1].split(")")[0])
    assert fast_mach > slow_mach * 10.0


def test_the_wind_tunnel_survives_an_empty_design(qt_app) -> None:
    """With no design the tunnel draws a prompt rather than failing."""
    from rocketopt.ui.windtunnel import WindTunnelPage

    page = WindTunnelPage()
    page.resize(800, 600)
    page.set_rocket(None)

    assert page.readout.text() == ""
    assert page.canvas.field_axes.texts


def test_the_pressure_field_is_symmetric_about_the_axis(
    qt_app, reference_rocket
) -> None:
    """The flow model is axisymmetric, so the drawn field must be too."""
    from rocketopt.cfd.potential_flow import solve_potential_flow

    solution = solve_potential_flow(reference_rocket, velocity=100.0, mach=0.29)
    x = np.linspace(0.02, 0.25, 24)
    r = np.full_like(x, 0.03)

    above = solution.pressure_field(x, r)
    below = solution.pressure_field(x, -r)

    assert np.allclose(above, below, atol=1e-12)


# ---------------------------------------------------------------------------
# The window as a whole
# ---------------------------------------------------------------------------


def test_the_window_opens_with_every_tab(qt_app) -> None:
    """The main window builds, analyses a design and offers all its views."""
    from rocketopt.ui.main_window import MainWindow

    window = MainWindow()
    window.resize(1400, 900)

    tabs = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert tabs[:3] == ["Rocket", "3D", "Wind tunnel"]
    assert {"Animation", "Trajectory", "Aerodynamics"} <= set(tabs)

    assert window._rocket is not None
    assert math.isfinite(window._rocket.length)

    # Opening a tab is what populates it, so each one has to survive being shown.
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)
        qt_app.processEvents()

    assert window.viewer_page.viewer.triangle_count > 0
    assert "Mach" in window.wind_tunnel.readout.text()
    window.close()
