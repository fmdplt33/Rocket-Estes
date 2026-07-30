"""The RocketOpt desktop interface.

Importing this package does *not* import PySide6; the Qt-dependent modules are
imported only when :func:`rocketopt.ui.app.run` is called. That keeps
``import rocketopt`` working on a headless install with no GUI toolkit
available.
"""

from __future__ import annotations

__all__ = ["run"]


def run(argv: list[str] | None = None) -> int:
    """Start the desktop application.

    Parameters
    ----------
    argv:
        Command-line arguments.

    Returns
    -------
    int
        Process exit code.
    """
    from rocketopt.ui.app import run as _run

    return _run(argv)
