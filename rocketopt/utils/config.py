"""Configuration loading, persistence and well-known package paths.

RocketOpt persists designs and run settings as YAML. Every persisted document
carries a ``schema_version`` so that future format changes can be migrated
rather than silently misread.

The helpers here are deliberately generic: any :class:`pydantic.BaseModel` can
be round-tripped through :func:`save_model` / :func:`load_model`.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Final, TypeVar

import yaml
from pydantic import BaseModel, Field, field_validator

from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

SCHEMA_VERSION: Final[int] = 1
"""Version stamped into every document written by :func:`save_model`."""

PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
"""Filesystem root of the installed ``rocketopt`` package."""

ASSETS_DIR: Final[Path] = PACKAGE_ROOT / "assets"
"""Directory holding bundled motor files and other static data."""

MOTOR_ASSETS_DIR: Final[Path] = ASSETS_DIR / "motors"
"""Directory scanned for user-supplied RASP ``.eng`` motor files."""

_ModelT = TypeVar("_ModelT", bound=BaseModel)


class ConfigError(RuntimeError):
    """Raised when a configuration document cannot be read or validated."""


class AppConfig(BaseModel):
    """Application-level settings that are not part of a rocket design.

    Attributes
    ----------
    output_dir:
        Directory that reports, CAD exports and plots are written to.
    log_level:
        Console logging level name.
    random_seed:
        Seed for the optimiser's random number generator. Fixing this makes
        optimisation runs reproducible, which matters for regression tests.
    max_workers:
        Upper bound on worker processes used to evaluate an optimiser
        population. ``1`` disables multiprocessing.
    """

    output_dir: Path = Field(default=Path("rocketopt_output"))
    log_level: str = Field(default="INFO")
    random_seed: int = Field(default=20260729, ge=0)
    max_workers: int = Field(default=1, ge=1, le=256)

    @field_validator("log_level")
    @classmethod
    def _validate_level(cls, value: str) -> str:
        """Reject log level names the standard library will not accept."""
        allowed = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return upper

    def ensure_output_dir(self) -> Path:
        """Create :attr:`output_dir` if needed and return its resolved path."""
        resolved = self.output_dir.expanduser().resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        return resolved


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML document into a dictionary.

    Parameters
    ----------
    path:
        Path to the YAML file.

    Returns
    -------
    dict
        Parsed mapping. An empty file yields an empty dictionary.

    Raises
    ------
    ConfigError
        If the file is missing, unreadable, malformed, or does not contain a
        mapping at the top level.
    """
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise ConfigError(f"Configuration file not found: {file_path}")

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            # safe_load refuses arbitrary Python object construction.
            data = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {file_path}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read {file_path}: {exc}") from exc

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Expected a mapping at the top level of {file_path}, "
            f"got {type(data).__name__}"
        )
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> Path:
    """Write a dictionary to a YAML file, creating parent directories.

    Parameters
    ----------
    data:
        Mapping to serialise.
    path:
        Destination path.

    Returns
    -------
    pathlib.Path
        The resolved path that was written.

    Raises
    ------
    ConfigError
        If the file cannot be written.
    """
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                data,
                handle,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
    except OSError as exc:
        raise ConfigError(f"Cannot write {file_path}: {exc}") from exc
    _log.debug("Wrote YAML document to %s", file_path)
    return file_path.resolve()


def save_model(model: BaseModel, path: str | Path, *, label: str | None = None) -> Path:
    """Serialise a Pydantic model to YAML with provenance metadata.

    The written document has the shape::

        schema_version: 1
        created: "2026-07-29"        # ISO-8601, %Y-%m-%d
        kind: "RocketDesign"
        label: "optimised-C6-5"      # optional, omitted when None
        data: {...}                  # the model's own fields

    Parameters
    ----------
    model:
        Any Pydantic model instance.
    path:
        Destination path.
    label:
        Optional human-readable name stored alongside the payload.

    Returns
    -------
    pathlib.Path
        The resolved path that was written.
    """
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created": _dt.date.today().isoformat(),
        "kind": type(model).__name__,
    }
    if label is not None:
        document["label"] = label
    # mode="json" converts Path, Enum and numpy scalars into YAML-safe types.
    document["data"] = model.model_dump(mode="json")
    return save_yaml(document, path)


def load_model(model_cls: type[_ModelT], path: str | Path) -> _ModelT:
    """Load and validate a Pydantic model previously written by :func:`save_model`.

    Documents written by an older ``schema_version`` are accepted with a
    warning; a *newer* version is refused because this build cannot know what
    the extra fields mean.

    Parameters
    ----------
    model_cls:
        The Pydantic model class to validate against.
    path:
        Path to the YAML document.

    Returns
    -------
    _ModelT
        The validated model instance.

    Raises
    ------
    ConfigError
        If the document is malformed, was written by a newer schema version, or
        fails validation against ``model_cls``.
    """
    document = load_yaml(path)

    version = document.get("schema_version", SCHEMA_VERSION)
    if not isinstance(version, int):
        raise ConfigError(f"schema_version must be an integer, got {version!r}")
    if version > SCHEMA_VERSION:
        raise ConfigError(
            f"{path} was written by schema version {version}, but this build of "
            f"RocketOpt understands at most version {SCHEMA_VERSION}. Upgrade "
            f"RocketOpt to read this file."
        )
    if version < SCHEMA_VERSION:
        _log.warning(
            "%s uses schema version %d (current is %d); loading with defaults "
            "for any newly added fields.",
            path,
            version,
            SCHEMA_VERSION,
        )

    payload = document.get("data")
    if not isinstance(payload, dict):
        raise ConfigError(f"{path} has no 'data' mapping to load a model from")

    try:
        return model_cls.model_validate(payload)
    except Exception as exc:  # pydantic.ValidationError and friends
        raise ConfigError(f"{path} is not a valid {model_cls.__name__}: {exc}") from exc


def load_app_config(path: str | Path | None = None) -> AppConfig:
    """Load :class:`AppConfig`, falling back to defaults when no file is given.

    Parameters
    ----------
    path:
        Optional path to a YAML file containing ``AppConfig`` fields at the top
        level (not wrapped in a ``data`` key).

    Returns
    -------
    AppConfig
        Validated application configuration.

    Raises
    ------
    ConfigError
        If the file exists but does not validate.
    """
    if path is None:
        return AppConfig()

    raw = load_yaml(path)
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:
        raise ConfigError(f"{path} is not a valid AppConfig: {exc}") from exc


__all__ = [
    "SCHEMA_VERSION",
    "PACKAGE_ROOT",
    "ASSETS_DIR",
    "MOTOR_ASSETS_DIR",
    "ConfigError",
    "AppConfig",
    "load_yaml",
    "save_yaml",
    "save_model",
    "load_model",
    "load_app_config",
]
