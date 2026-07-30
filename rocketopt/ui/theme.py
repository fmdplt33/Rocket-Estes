"""Dark theme palette and stylesheet for the RocketOpt application.

The palette is built around a neutral dark grey rather than pure black, which
reduces halation against the light-on-dark text, and reserves saturated colour
for meaning: green for a passing margin, amber for a caution, red for a
violated constraint. Nothing else in the interface is allowed to use those
three hues, so a coloured element always carries information.
"""

from __future__ import annotations

from typing import Final

from PySide6.QtGui import QColor, QPalette

__all__ = [
    "BACKGROUND",
    "SURFACE",
    "SURFACE_RAISED",
    "BORDER",
    "TEXT",
    "TEXT_MUTED",
    "ACCENT",
    "PASS",
    "WARN",
    "FAIL",
    "PLOT_COLOURS",
    "build_palette",
    "STYLESHEET",
]

BACKGROUND: Final[str] = "#1b1d21"
SURFACE: Final[str] = "#24272c"
SURFACE_RAISED: Final[str] = "#2c3036"
BORDER: Final[str] = "#3a3f47"
TEXT: Final[str] = "#e6e8ea"
TEXT_MUTED: Final[str] = "#9aa1aa"
ACCENT: Final[str] = "#4f9cf9"

PASS: Final[str] = "#4caf7d"
WARN: Final[str] = "#d9a441"
FAIL: Final[str] = "#e05c5c"

PLOT_COLOURS: Final[tuple[str, ...]] = (
    "#4f9cf9",
    "#4caf7d",
    "#d9a441",
    "#c471d4",
    "#e05c5c",
    "#48b8c4",
    "#9aa1aa",
)
"""Qualitative series colours for plots, ordered for maximum separation."""


def build_palette() -> QPalette:
    """Return the application's :class:`QPalette`.

    Returns
    -------
    QPalette
        A dark palette consistent with :data:`STYLESHEET`.
    """
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(SURFACE))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#0d1117"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(SURFACE_RAISED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(TEXT_MUTED)
    )
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(TEXT_MUTED)
    )
    return palette


STYLESHEET: Final[str] = f"""
QWidget {{
    background-color: {BACKGROUND};
    color: {TEXT};
    font-size: 13px;
}}

QMainWindow::separator {{
    background: {BORDER};
    width: 1px;
    height: 1px;
}}

QDockWidget {{
    titlebar-close-icon: none;
    font-weight: 600;
}}
QDockWidget::title {{
    background: {SURFACE_RAISED};
    padding: 7px 10px;
    border-bottom: 1px solid {BORDER};
}}

QGroupBox {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 6px;
    margin-top: 16px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {TEXT_MUTED};
}}

QLabel[muted="true"] {{
    color: {TEXT_MUTED};
}}
QLabel[metric="true"] {{
    font-size: 19px;
    font-weight: 600;
}}

QPushButton {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 5px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background: #343941; }}
QPushButton:pressed {{ background: #1f2226; }}
QPushButton:disabled {{ color: {TEXT_MUTED}; background: {SURFACE}; }}
QPushButton[primary="true"] {{
    background: {ACCENT};
    color: #0d1117;
    border: none;
    font-weight: 600;
}}
QPushButton[primary="true"]:hover {{ background: #66aaff; }}
QPushButton[primary="true"]:disabled {{ background: #35506f; color: #7d8894; }}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 6px;
    min-height: 20px;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE_RAISED};
    selection-background-color: {ACCENT};
    selection-color: #0d1117;
    border: 1px solid {BORDER};
}}

QSlider::groove:horizontal {{
    background: {BORDER};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {ACCENT};
    width: 14px;
    margin: -6px 0;
    border-radius: 7px;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 6px;
    background: {SURFACE};
}}
QTabBar::tab {{
    background: transparent;
    padding: 8px 18px;
    border-bottom: 2px solid transparent;
    color: {TEXT_MUTED};
}}
QTabBar::tab:selected {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover {{ color: {TEXT}; }}

QTableWidget {{
    background: {SURFACE};
    gridline-color: {BORDER};
    border: none;
}}
QHeaderView::section {{
    background: {SURFACE_RAISED};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px;
    font-weight: 600;
}}

QProgressBar {{
    background: {SURFACE_RAISED};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    height: 18px;
}}
QProgressBar::chunk {{
    background: {ACCENT};
    border-radius: 3px;
}}

QToolBar {{
    background: {SURFACE};
    border-bottom: 1px solid {BORDER};
    spacing: 8px;
    padding: 6px;
}}

QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {BORDER};
}}

QScrollBar:vertical {{
    background: transparent;
    width: 10px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: #4b515a; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}

QToolTip {{
    background: {SURFACE_RAISED};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 5px;
}}
"""
"""Qt stylesheet applied to the whole application."""
