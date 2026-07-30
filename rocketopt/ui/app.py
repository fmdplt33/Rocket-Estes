"""Application entry point for the RocketOpt desktop interface."""

from __future__ import annotations

import sys

from rocketopt.utils.logging import configure_logging, get_logger

_log = get_logger(__name__)

__all__ = ["run"]


def run(argv: list[str] | None = None) -> int:
    """Start the RocketOpt desktop application.

    Parameters
    ----------
    argv:
        Command-line arguments. Defaults to :data:`sys.argv`.

    Returns
    -------
    int
        Process exit code.

    Raises
    ------
    SystemExit
        With a helpful message if PySide6 is not installed.
    """
    configure_logging()

    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise SystemExit(
            "The RocketOpt interface needs PySide6, which is not installed.\n"
            "Install it with:  pip install 'rocketopt[ui]'"
        ) from exc

    from rocketopt.ui.main_window import MainWindow
    from rocketopt.ui.theme import STYLESHEET, build_palette

    app = QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("RocketOpt")
    app.setOrganizationName("RocketOpt")
    app.setStyle("Fusion")
    app.setPalette(build_palette())
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()

    _log.info("RocketOpt interface started")
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
