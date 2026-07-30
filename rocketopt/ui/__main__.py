"""Allow the interface to be launched with ``python -m rocketopt.ui``."""

from __future__ import annotations

from rocketopt.ui.app import run

if __name__ == "__main__":
    raise SystemExit(run())
