"""Centralised logging configuration for RocketOpt.

Uses :mod:`rich` for coloured console output when available and degrades to the
standard library formatter otherwise, so the package never hard-fails on a
minimal install.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Final

_DEFAULT_FORMAT: Final[str] = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DEFAULT_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"
_ROOT_LOGGER_NAME: Final[str] = "rocketopt"

_configured: bool = False


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_file: Path | None = None,
    use_rich: bool = True,
    force: bool = False,
) -> logging.Logger:
    """Configure the ``rocketopt`` logger hierarchy.

    Parameters
    ----------
    level:
        Logging level for the console handler, e.g. ``logging.DEBUG`` or
        ``"INFO"``.
    log_file:
        Optional path to a file that receives an uncoloured ``DEBUG``-level
        copy of every record. Parent directories are created on demand.
    use_rich:
        When ``True`` and :mod:`rich` is installed, use ``RichHandler`` for the
        console stream.
    force:
        Re-configure even if this function has already run in this process.

    Returns
    -------
    logging.Logger
        The configured root ``rocketopt`` logger.
    """
    global _configured

    logger = logging.getLogger(_ROOT_LOGGER_NAME)

    if _configured and not force:
        return logger

    # Remove handlers from a previous configuration so ``force`` is idempotent.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    console_handler: logging.Handler
    if use_rich:
        try:
            from rich.logging import RichHandler

            console_handler = RichHandler(
                rich_tracebacks=True,
                show_path=False,
                omit_repeated_times=False,
            )
            console_handler.setFormatter(
                logging.Formatter("%(message)s", datefmt="[%X]")
            )
        except ImportError:  # pragma: no cover - depends on optional install
            console_handler = logging.StreamHandler(stream=sys.stderr)
            console_handler.setFormatter(
                logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
            )
    else:
        console_handler = logging.StreamHandler(stream=sys.stderr)
        console_handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        )

    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
        )
        logger.addHandler(file_handler)

    _configured = True
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the ``rocketopt`` namespace.

    Parameters
    ----------
    name:
        Usually ``__name__``. A leading ``"rocketopt."`` is stripped so that
        callers may pass either the bare module name or the dotted path.

    Returns
    -------
    logging.Logger
        A logger whose records flow to the handlers installed by
        :func:`configure_logging`.
    """
    short = name
    prefix = f"{_ROOT_LOGGER_NAME}."
    if short.startswith(prefix):
        short = short[len(prefix) :]
    elif short == _ROOT_LOGGER_NAME:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{short}")


__all__ = ["configure_logging", "get_logger"]
