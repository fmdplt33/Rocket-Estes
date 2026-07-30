"""Design input panel and results panel.

The design panel is deliberately organised in the order a builder thinks about
a rocket - motor first, then nose, body, fins, finish - rather than in the
order the solver happens to want its variables. Every control carries a
tooltip explaining what the parameter does and which way improves performance,
so the interface teaches as it is used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QLabel,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from rocketopt.geometry.components import FinAirfoil
from rocketopt.geometry.nose_cones import NoseConeShape
from rocketopt.propulsion.database import list_motors
from rocketopt.structures.materials import (
    SurfaceFinish,
    nose_cone_materials,
    sheet_materials,
    tube_materials,
)
from rocketopt.ui.theme import FAIL, PASS, TEXT_MUTED, WARN

__all__ = ["DesignValues", "DesignPanel", "ResultsPanel"]


@dataclass(slots=True)
class DesignValues:
    """Plain snapshot of everything the design panel controls.

    The defaults describe an Estes Alpha III-class rocket that is stable,
    flutter-safe and passes every structural check, so the application opens
    on a sound design rather than one the user must first repair.

    A note on the default nose material: a *solid balsa* nose of this size
    weighs so little that the centre of gravity sits too far aft, giving a
    static margin below 1.0 calibre. A moulded plastic shell puts mass at the
    front where stability needs it, which is why production kits in this class
    use one.
    """

    motor: str = "C6-5"
    nose_shape: str = "tangent_ogive"
    nose_fineness: float = 3.0
    nose_material: str = "pla"
    body_diameter_mm: float = 24.8
    body_length_mm: float = 216.0
    wall_thickness_mm: float = 0.45
    body_material: str = "kraft_paper"
    fin_count: int = 3
    fin_root_mm: float = 55.0
    fin_tip_mm: float = 30.0
    fin_span_mm: float = 42.0
    fin_sweep_mm: float = 25.0
    fin_thickness_mm: float = 2.38
    fin_material: str = "balsa"
    fin_airfoil: str = "rounded"
    fillet_mm: float = 3.0
    surface_finish: str = "REGULAR_PAINT"
    ballast_g: float = 0.0
    # Launch conditions
    elevation_m: float = 0.0
    temperature_c: float = 15.0
    humidity_pct: float = 50.0
    wind_ms: float = 0.0
    wind_dir_deg: float = 270.0
    rail_length_m: float = 0.91
    rail_angle_deg: float = 0.0


def _spin(
    minimum: float,
    maximum: float,
    value: float,
    step: float,
    decimals: int,
    suffix: str,
    tooltip: str,
) -> QDoubleSpinBox:
    """Build a configured double spin box."""
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setValue(value)
    box.setSingleStep(step)
    box.setDecimals(decimals)
    box.setSuffix(suffix)
    box.setToolTip(tooltip)
    box.setKeyboardTracking(False)
    return box


class DesignPanel(QScrollArea):
    """Scrollable panel of every editable design parameter.

    Emits :attr:`changed` whenever any control is edited, so the host window
    can re-run the analysis.
    """

    changed = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the panel."""
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        layout.addWidget(self._build_motor_group())
        layout.addWidget(self._build_nose_group())
        layout.addWidget(self._build_body_group())
        layout.addWidget(self._build_fin_group())
        layout.addWidget(self._build_finish_group())
        layout.addWidget(self._build_conditions_group())
        layout.addStretch(1)

        self.setWidget(container)
        self._connect_all()

    # -- Group builders ----------------------------------------------------

    def _build_motor_group(self) -> QGroupBox:
        """Motor selection."""
        group = QGroupBox("1. Motor")
        form = QFormLayout(group)

        self.motor = QComboBox()
        for motor in list_motors():
            for delay in motor.delays:
                label = (
                    f"{motor.designation}-{int(delay)}"
                    if float(delay).is_integer()
                    else f"{motor.designation}-{delay:g}"
                )
                self.motor.addItem(
                    f"{label}   ({motor.total_impulse:.1f} N·s, "
                    f"{motor.impulse_class})",
                    label,
                )
        self._select_data(self.motor, "C6-5")
        self.motor.setToolTip(
            "Estes motor and ejection delay. The delay should fire the "
            "parachute at apogee; the results panel reports how far off it is."
        )
        form.addRow("Motor", self.motor)
        return group

    def _build_nose_group(self) -> QGroupBox:
        """Nose cone parameters."""
        group = QGroupBox("2. Nose cone")
        form = QFormLayout(group)

        self.nose_shape = QComboBox()
        for shape in NoseConeShape:
            self.nose_shape.addItem(shape.label, shape.value)
        self._select_data(self.nose_shape, "tangent_ogive")
        self.nose_shape.setToolTip(
            "Profile family. Von Karman gives the least pressure drag for a "
            "given length and diameter; conical is the easiest to make."
        )
        form.addRow("Shape", self.nose_shape)

        self.nose_fineness = _spin(
            1.0, 8.0, 3.0, 0.1, 2, "",
            "Nose length divided by body diameter. Below about 2 pressure "
            "drag climbs steeply; above about 5 the extra wetted area costs "
            "more than the pressure drag it saves."
        )
        form.addRow("Fineness ratio", self.nose_fineness)

        self.nose_material = QComboBox()
        for material in nose_cone_materials():
            self.nose_material.addItem(material.display_name, material.name)
        self._select_data(self.nose_material, "pla")
        self.nose_material.setToolTip(
            "Nose material. A moulded or printed shell is heavier than turned "
            "balsa and moves the centre of gravity forward, which usually "
            "helps stability on a short model."
        )
        form.addRow("Material", self.nose_material)
        return group

    def _build_body_group(self) -> QGroupBox:
        """Body tube parameters."""
        group = QGroupBox("3. Body")
        form = QFormLayout(group)

        self.body_diameter = _spin(
            10.0, 80.0, 24.8, 0.1, 2, " mm",
            "Outside diameter. This sets the aerodynamic reference area, so "
            "the smallest tube that clears the motor is almost always fastest."
        )
        form.addRow("Diameter", self.body_diameter)

        self.body_length = _spin(
            50.0, 900.0, 216.0, 5.0, 1, " mm",
            "Body tube length. Longer moves the centre of gravity forward and "
            "adds stability, at the cost of mass and skin friction."
        )
        form.addRow("Length", self.body_length)

        self.wall_thickness = _spin(
            0.2, 3.0, 0.45, 0.05, 2, " mm",
            "Tube wall thickness. Snapped to the nearest stock size for the "
            "chosen material when the design is built."
        )
        form.addRow("Wall", self.wall_thickness)

        self.body_material = QComboBox()
        for material in tube_materials():
            self.body_material.addItem(material.display_name, material.name)
        self._select_data(self.body_material, "kraft_paper")
        form.addRow("Material", self.body_material)
        return group

    def _build_fin_group(self) -> QGroupBox:
        """Fin parameters."""
        group = QGroupBox("4. Fins")
        form = QFormLayout(group)

        self.fin_count = QSpinBox()
        self.fin_count.setRange(3, 5)
        self.fin_count.setValue(3)
        self.fin_count.setToolTip(
            "Three fins give the least drag; four are easier to align "
            "accurately; five add stability at a clear drag cost."
        )
        form.addRow("Count", self.fin_count)

        self.fin_root = _spin(
            10.0, 200.0, 55.0, 1.0, 1, " mm", "Chord where the fin meets the body."
        )
        form.addRow("Root chord", self.fin_root)

        self.fin_tip = _spin(
            0.0, 200.0, 30.0, 1.0, 1, " mm",
            "Chord at the fin tip. Zero gives a delta fin."
        )
        form.addRow("Tip chord", self.fin_tip)

        self.fin_span = _spin(
            5.0, 150.0, 42.0, 1.0, 1, " mm",
            "Exposed span from the body wall to the tip. The strongest lever "
            "on stability, and on fin drag."
        )
        form.addRow("Span", self.fin_span)

        self.fin_sweep = _spin(
            0.0, 200.0, 25.0, 1.0, 1, " mm",
            "How far aft the tip leading edge sits relative to the root "
            "leading edge. Sweep moves the centre of pressure aft."
        )
        form.addRow("Sweep", self.fin_sweep)

        self.fin_thickness = _spin(
            0.4, 8.0, 2.38, 0.1, 2, " mm",
            "Fin stock thickness. Flutter speed scales with thickness to the "
            "power 1.5, so thin fins on a fast rocket will shed."
        )
        form.addRow("Thickness", self.fin_thickness)

        self.fin_material = QComboBox()
        for material in sheet_materials():
            self.fin_material.addItem(material.display_name, material.name)
        self._select_data(self.fin_material, "balsa")
        self.fin_material.setToolTip(
            "Fin material. Stiffer materials raise the flutter speed as the "
            "square root of shear modulus."
        )
        form.addRow("Material", self.fin_material)

        self.fin_airfoil = QComboBox()
        for airfoil in FinAirfoil:
            self.fin_airfoil.addItem(airfoil.label, airfoil.value)
        self._select_data(self.fin_airfoil, "rounded")
        self.fin_airfoil.setToolTip(
            "Cross-section. Rounding the leading edge and tapering the "
            "trailing edge roughly halves fin pressure drag."
        )
        form.addRow("Section", self.fin_airfoil)

        self.fillet = _spin(
            0.0, 15.0, 3.0, 0.5, 2, " mm",
            "Root fillet radius. The single most effective way to strengthen "
            "the fin joint and cut interference drag."
        )
        form.addRow("Fillet", self.fillet)
        return group

    def _build_finish_group(self) -> QGroupBox:
        """Finish and ballast."""
        group = QGroupBox("5. Finish and ballast")
        form = QFormLayout(group)

        self.surface_finish = QComboBox()
        for finish in SurfaceFinish:
            self.surface_finish.addItem(
                f"{finish.label} ({float(finish) * 1e6:.1f} µm)", finish.name
            )
        self._select_data(self.surface_finish, "REGULAR_PAINT")
        self.surface_finish.setToolTip(
            "Surface roughness sets skin-friction drag, which is usually the "
            "largest single drag contributor on a model rocket."
        )
        form.addRow("Surface", self.surface_finish)

        self.ballast = _spin(
            0.0, 100.0, 0.0, 0.5, 1, " g",
            "Mass added in the nose to move the centre of gravity forward. "
            "Use only if the design is otherwise unstable; it always costs "
            "altitude."
        )
        form.addRow("Nose ballast", self.ballast)
        return group

    def _build_conditions_group(self) -> QGroupBox:
        """Launch conditions."""
        group = QGroupBox("6. Launch conditions")
        form = QFormLayout(group)

        self.elevation = _spin(
            -100.0, 4000.0, 0.0, 10.0, 0, " m", "Site elevation above sea level."
        )
        form.addRow("Elevation", self.elevation)

        self.temperature = _spin(
            -30.0, 50.0, 15.0, 1.0, 1, " °C", "Air temperature at the pad."
        )
        form.addRow("Temperature", self.temperature)

        self.humidity = _spin(
            0.0, 100.0, 50.0, 5.0, 0, " %",
            "Relative humidity. Moist air is slightly less dense than dry air."
        )
        form.addRow("Humidity", self.humidity)

        self.wind = _spin(
            0.0, 20.0, 0.0, 0.5, 1, " m/s", "Wind speed at 10 m above ground."
        )
        form.addRow("Wind speed", self.wind)

        self.wind_dir = _spin(
            0.0, 360.0, 270.0, 10.0, 0, " °",
            "Direction the wind blows from, clockwise from north."
        )
        form.addRow("Wind from", self.wind_dir)

        self.rail_length = _spin(
            0.3, 3.0, 0.91, 0.05, 2, " m",
            "Guided rod or rail length. The rocket must be fast enough "
            "leaving it for the fins to work."
        )
        form.addRow("Rail length", self.rail_length)

        self.rail_angle = _spin(
            0.0, 30.0, 0.0, 1.0, 0, " °",
            "Tilt from vertical, into wind. Capped at 30 degrees by the NAR "
            "safety code."
        )
        form.addRow("Rail angle", self.rail_angle)
        return group

    # -- Wiring ------------------------------------------------------------

    @staticmethod
    def _select_data(combo: QComboBox, value: str) -> None:
        """Select the combo entry whose user data equals ``value``."""
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _editable_widgets(self) -> list[QWidget]:
        """Every control that can change the design.

        PySide6's ``findChildren`` accepts one type at a time, unlike PyQt's
        tuple form, so the three widget classes are collected separately.
        """
        widgets: list[QWidget] = []
        widgets.extend(self.findChildren(QDoubleSpinBox))
        widgets.extend(self.findChildren(QSpinBox))
        widgets.extend(self.findChildren(QComboBox))
        return widgets

    def _connect_all(self) -> None:
        """Connect every control to the ``changed`` signal.

        ``valueChanged`` and ``currentIndexChanged`` both carry an argument
        while ``changed`` carries none, so each is wrapped in a lambda that
        discards it. Connecting them directly raises a TypeError on every
        edit.
        """
        for spin in self.findChildren(QDoubleSpinBox):
            spin.valueChanged.connect(lambda _value: self.changed.emit())
        for int_spin in self.findChildren(QSpinBox):
            int_spin.valueChanged.connect(lambda _value: self.changed.emit())
        for combo in self.findChildren(QComboBox):
            combo.currentIndexChanged.connect(lambda _index: self.changed.emit())

    def values(self) -> DesignValues:
        """Return the current control values as a snapshot."""
        return DesignValues(
            motor=self.motor.currentData(),
            nose_shape=self.nose_shape.currentData(),
            nose_fineness=self.nose_fineness.value(),
            nose_material=self.nose_material.currentData(),
            body_diameter_mm=self.body_diameter.value(),
            body_length_mm=self.body_length.value(),
            wall_thickness_mm=self.wall_thickness.value(),
            body_material=self.body_material.currentData(),
            fin_count=self.fin_count.value(),
            fin_root_mm=self.fin_root.value(),
            fin_tip_mm=self.fin_tip.value(),
            fin_span_mm=self.fin_span.value(),
            fin_sweep_mm=self.fin_sweep.value(),
            fin_thickness_mm=self.fin_thickness.value(),
            fin_material=self.fin_material.currentData(),
            fin_airfoil=self.fin_airfoil.currentData(),
            fillet_mm=self.fillet.value(),
            surface_finish=self.surface_finish.currentData(),
            ballast_g=self.ballast.value(),
            elevation_m=self.elevation.value(),
            temperature_c=self.temperature.value(),
            humidity_pct=self.humidity.value(),
            wind_ms=self.wind.value(),
            wind_dir_deg=self.wind_dir.value(),
            rail_length_m=self.rail_length.value(),
            rail_angle_deg=self.rail_angle.value(),
        )

    def set_values(self, values: DesignValues, *, block: bool = True) -> None:
        """Load a design snapshot into the controls.

        Parameters
        ----------
        values:
            The snapshot to apply.
        block:
            Suppress ``changed`` while loading, so an optimiser result does
            not trigger one re-analysis per control.
        """
        widgets = self._editable_widgets()
        if block:
            for widget in widgets:
                widget.blockSignals(True)
        try:
            self._select_data(self.motor, values.motor)
            self._select_data(self.nose_shape, values.nose_shape)
            self.nose_fineness.setValue(values.nose_fineness)
            self._select_data(self.nose_material, values.nose_material)
            self.body_diameter.setValue(values.body_diameter_mm)
            self.body_length.setValue(values.body_length_mm)
            self.wall_thickness.setValue(values.wall_thickness_mm)
            self._select_data(self.body_material, values.body_material)
            self.fin_count.setValue(values.fin_count)
            self.fin_root.setValue(values.fin_root_mm)
            self.fin_tip.setValue(values.fin_tip_mm)
            self.fin_span.setValue(values.fin_span_mm)
            self.fin_sweep.setValue(values.fin_sweep_mm)
            self.fin_thickness.setValue(values.fin_thickness_mm)
            self._select_data(self.fin_material, values.fin_material)
            self._select_data(self.fin_airfoil, values.fin_airfoil)
            self.fillet.setValue(values.fillet_mm)
            self._select_data(self.surface_finish, values.surface_finish)
            self.ballast.setValue(values.ballast_g)
        finally:
            if block:
                for widget in widgets:
                    widget.blockSignals(False)
        self.changed.emit()


class _Metric(QWidget):
    """A single labelled headline number with an optional colour."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        """Build the metric tile."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)

        self._value = QLabel("-")
        self._value.setProperty("metric", True)

        self._caption = QLabel(caption)
        self._caption.setProperty("muted", True)

        layout.addWidget(self._value)
        layout.addWidget(self._caption)

    def set_value(self, text: str, colour: str | None = None) -> None:
        """Update the displayed value and its colour."""
        self._value.setText(text)
        self._value.setStyleSheet(f"color: {colour};" if colour else "")


class ResultsPanel(QWidget):
    """Headline results, stability verdict and warnings."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Build the results panel."""
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        grid_box = QGroupBox("Performance")
        grid = QGridLayout(grid_box)
        self.metrics: dict[str, _Metric] = {}
        captions = [
            ("apogee", "Apogee"),
            ("max_velocity", "Max speed"),
            ("stability", "Static margin"),
            ("mass", "Lift-off mass"),
            ("rail", "Rail exit"),
            ("twr", "Thrust / weight"),
            ("cd", "Coast drag Cd"),
            ("landing", "Landing speed"),
        ]
        for index, (key, caption) in enumerate(captions):
            metric = _Metric(caption)
            self.metrics[key] = metric
            grid.addWidget(metric, index // 2, index % 2)
        layout.addWidget(grid_box)

        self.structure_box = QGroupBox("Structural margins")
        structure_layout = QVBoxLayout(self.structure_box)
        self.structure_label = QLabel("Run an analysis")
        self.structure_label.setWordWrap(True)
        self.structure_label.setTextFormat(Qt.TextFormat.RichText)
        structure_layout.addWidget(self.structure_label)
        layout.addWidget(self.structure_box)

        self.warning_box = QGroupBox("Warnings")
        warning_layout = QVBoxLayout(self.warning_box)
        self.warning_label = QLabel("None")
        self.warning_label.setWordWrap(True)
        self.warning_label.setTextFormat(Qt.TextFormat.RichText)
        warning_layout.addWidget(self.warning_label)
        layout.addWidget(self.warning_box)

        layout.addStretch(1)

    def set_error(self, message: str) -> None:
        """Show an error state, clearing the metrics."""
        for metric in self.metrics.values():
            metric.set_value("-")
        self.structure_label.setText(
            f"<span style='color:{FAIL}'>{message}</span>"
        )
        self.warning_label.setText("-")

    def update_results(
        self,
        *,
        apogee: float,
        max_velocity: float,
        static_margin: float,
        loaded_mass: float,
        rail_exit: float,
        thrust_to_weight: float,
        drag_coefficient: float,
        landing_velocity: float,
        structure_rows: list[tuple[str, float, bool]],
        flutter_text: str,
        flutter_ok: bool,
        warnings: list[str],
        min_rail_exit: float,
        min_twr: float,
    ) -> None:
        """Populate every field from an analysis result."""
        self.metrics["apogee"].set_value(f"{apogee:.0f} m")
        self.metrics["max_velocity"].set_value(f"{max_velocity:.0f} m/s")

        if static_margin < 1.0:
            margin_colour = FAIL if static_margin < 0.5 else WARN
        elif static_margin > 2.5:
            margin_colour = WARN
        else:
            margin_colour = PASS
        self.metrics["stability"].set_value(f"{static_margin:.2f} cal", margin_colour)

        self.metrics["mass"].set_value(f"{loaded_mass * 1e3:.0f} g")
        self.metrics["rail"].set_value(
            f"{rail_exit:.0f} m/s", PASS if rail_exit >= min_rail_exit else FAIL
        )
        self.metrics["twr"].set_value(
            f"{thrust_to_weight:.1f}", PASS if thrust_to_weight >= min_twr else FAIL
        )
        self.metrics["cd"].set_value(f"{drag_coefficient:.3f}")
        self.metrics["landing"].set_value(
            f"{landing_velocity:.1f} m/s", PASS if landing_velocity <= 6.0 else WARN
        )

        rows = []
        for name, margin, passes in structure_rows:
            colour = PASS if passes else FAIL
            margin_text = "inf" if margin == float("inf") else f"{margin:+.1f}"
            rows.append(
                f"<tr><td>{name}</td>"
                f"<td align='right' style='color:{colour}'>{margin_text}</td></tr>"
            )
        flutter_colour = PASS if flutter_ok else FAIL
        rows.append(
            f"<tr><td>Fin flutter</td>"
            f"<td align='right' style='color:{flutter_colour}'>{flutter_text}</td></tr>"
        )
        self.structure_label.setText(
            "<table width='100%' cellspacing='2'>"
            f"<tr><td style='color:{TEXT_MUTED}'>check</td>"
            f"<td align='right' style='color:{TEXT_MUTED}'>margin</td></tr>"
            + "".join(rows)
            + "</table>"
        )

        if warnings:
            items = "".join(f"<li>{w}</li>" for w in warnings)
            self.warning_label.setText(
                f"<ul style='margin-left:-20px;color:{WARN}'>{items}</ul>"
            )
        else:
            self.warning_label.setText(
                f"<span style='color:{PASS}'>No warnings - design is within "
                f"every limit.</span>"
            )
