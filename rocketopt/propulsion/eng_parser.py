"""Reader and writer for RASP ``.eng`` motor files.

The RASP format is the de-facto interchange format for rocket motor
static-test data and is what ThrustCurve.org distributes. A file contains one
or more motor definitions; each definition is a header line followed by
``time thrust`` sample pairs and terminated by a sample at zero thrust.

Header line fields, in order::

    designation  diameter_mm  length_mm  delays  propellant_kg  total_kg  manufacturer

For example::

    ; Estes C6, NAR certified
    C6 18 70 0-3-5-7 0.0108 0.0258 Estes
       0.031 0.946
       0.092 4.826
       ...
       1.850 0.000

Lines beginning with ``;`` are comments. Note that the two mass fields are in
**kilograms** while the two dimension fields are in **millimetres** - an
inconsistency inherent to the format, not a bug here.

References
----------
[1] Coker, J. (2020). *RASP Engine File Format Specification*, ThrustCurve.org.
[2] Stine, G. H., & Stine, B. (2004). *Handbook of Model Rocketry*, 7th ed.,
    Appendix - motor data conventions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

import numpy as np

from rocketopt.propulsion.motor import Motor, ThrustCurve, ThrustCurveSource
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "EngParseError",
    "parse_eng_file",
    "parse_eng_text",
    "write_eng_file",
    "load_motors_from_directory",
]

_COMMENT_PREFIX: Final[str] = ";"
_DELAY_SPLIT: Final[re.Pattern[str]] = re.compile(r"[-,/]")


class EngParseError(ValueError):
    """Raised when a ``.eng`` file cannot be parsed."""


def _strip_comment(line: str) -> str:
    """Remove a trailing RASP comment and surrounding whitespace."""
    idx = line.find(_COMMENT_PREFIX)
    if idx >= 0:
        line = line[:idx]
    return line.strip()


def _parse_delays(token: str, designation: str) -> tuple[float, ...]:
    """Parse the delay field of a RASP header line.

    Delays appear as a separated list such as ``0-3-5-7``. The sentinel ``P``
    or ``None`` marks a plugged motor with no ejection charge; that is
    represented here as a single zero delay.

    Parameters
    ----------
    token:
        The raw delay field.
    designation:
        Motor designation, used only for error messages.

    Returns
    -------
    tuple of float
        Sorted, de-duplicated delays [s].

    Raises
    ------
    EngParseError
        If a delay token is not numeric.
    """
    cleaned = token.strip().upper()
    if cleaned in {"P", "NONE", "0"}:
        return (0.0,)

    delays: list[float] = []
    for part in _DELAY_SPLIT.split(cleaned):
        if not part:
            continue
        if part == "P":
            # A plugged option listed alongside timed delays.
            delays.append(0.0)
            continue
        try:
            delays.append(float(part))
        except ValueError as exc:
            raise EngParseError(
                f"{designation}: cannot parse delay {part!r} in field {token!r}"
            ) from exc

    if not delays:
        raise EngParseError(f"{designation}: delay field {token!r} lists no delays")
    return tuple(sorted(set(delays)))


def parse_eng_text(text: str, *, origin: str = "<string>") -> list[Motor]:
    """Parse the contents of a RASP ``.eng`` document.

    Parameters
    ----------
    text:
        Full file contents.
    origin:
        Name used in error messages, normally the file path.

    Returns
    -------
    list of Motor
        Every motor defined in the document, in file order.

    Raises
    ------
    EngParseError
        If the document is malformed.
    """
    motors: list[Motor] = []

    header: list[str] | None = None
    samples: list[tuple[float, float]] = []
    header_line_no = 0

    def flush(at_line: int) -> None:
        """Convert the accumulated header and samples into a Motor."""
        nonlocal header, samples
        if header is None:
            return
        if len(samples) < 2:
            raise EngParseError(
                f"{origin}:{header_line_no}: motor {header[0]!r} has "
                f"{len(samples)} thrust sample(s); at least 2 are required"
            )
        motors.append(_build_motor(header, samples, origin, header_line_no))
        header = None
        samples = []
        del at_line

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw)
        if not line:
            continue

        fields = line.split()

        # A header line starts with a non-numeric designation token.
        is_header = True
        try:
            float(fields[0])
            is_header = False
        except ValueError:
            is_header = True

        if is_header:
            flush(line_no)
            if len(fields) < 7:
                raise EngParseError(
                    f"{origin}:{line_no}: header line needs 7 fields "
                    f"(designation diameter length delays propellant total "
                    f"manufacturer), got {len(fields)}: {line!r}"
                )
            header = fields
            header_line_no = line_no
            continue

        if header is None:
            raise EngParseError(
                f"{origin}:{line_no}: thrust sample before any motor header"
            )
        if len(fields) < 2:
            raise EngParseError(
                f"{origin}:{line_no}: expected 'time thrust', got {line!r}"
            )
        try:
            samples.append((float(fields[0]), float(fields[1])))
        except ValueError as exc:
            raise EngParseError(
                f"{origin}:{line_no}: non-numeric thrust sample {line!r}"
            ) from exc

    flush(len(text.splitlines()) + 1)

    if not motors:
        raise EngParseError(f"{origin}: no motor definitions found")
    return motors


def _build_motor(
    header: list[str],
    samples: list[tuple[float, float]],
    origin: str,
    line_no: int,
) -> Motor:
    """Assemble a :class:`Motor` from a parsed header and sample list.

    Parameters
    ----------
    header:
        Whitespace-split header fields.
    samples:
        ``(time, thrust)`` pairs in file order.
    origin:
        File name, for error messages.
    line_no:
        Header line number, for error messages.

    Returns
    -------
    Motor
        The parsed motor, with a ``MEASURED`` thrust curve.

    Raises
    ------
    EngParseError
        If any field is non-numeric or physically invalid.
    """
    designation = header[0]
    try:
        diameter_mm = float(header[1])
        length_mm = float(header[2])
        propellant_kg = float(header[4])
        total_kg = float(header[5])
    except ValueError as exc:
        raise EngParseError(
            f"{origin}:{line_no}: non-numeric field in header for {designation!r}"
        ) from exc

    manufacturer = " ".join(header[6:])
    delays = _parse_delays(header[3], designation)

    times = np.array([s[0] for s in samples], dtype=np.float64)
    thrusts = np.array([s[1] for s in samples], dtype=np.float64)

    # RASP files conventionally omit the (0, 0) origin. Prepend it so the
    # ignition transient is integrated rather than extrapolated flat.
    if times[0] > 0.0:
        times = np.concatenate([[0.0], times])
        thrusts = np.concatenate([[0.0], thrusts])

    # Some files repeat the final time stamp; drop non-increasing duplicates.
    keep = np.concatenate([[True], np.diff(times) > 0.0])
    if not keep.all():
        _log.debug(
            "%s:%d: dropped %d duplicate time stamp(s) from %s",
            origin,
            line_no,
            int((~keep).sum()),
            designation,
        )
    times = times[keep]
    thrusts = thrusts[keep]

    try:
        curve = ThrustCurve(
            times=times,
            thrusts=thrusts,
            source=ThrustCurveSource.MEASURED,
        )
    except ValueError as exc:
        raise EngParseError(
            f"{origin}:{line_no}: invalid thrust curve for {designation!r}: {exc}"
        ) from exc

    try:
        return Motor(
            designation=designation,
            manufacturer=manufacturer,
            diameter=diameter_mm * 1e-3,
            length=length_mm * 1e-3,
            total_mass=total_kg,
            propellant_mass=propellant_kg,
            thrust_curve=curve,
            delays=delays,
            source=f"RASP file {origin}",
        )
    except ValueError as exc:
        raise EngParseError(
            f"{origin}:{line_no}: invalid motor definition for {designation!r}: {exc}"
        ) from exc


def parse_eng_file(path: str | Path) -> list[Motor]:
    """Parse a RASP ``.eng`` file from disk.

    Parameters
    ----------
    path:
        Path to the ``.eng`` file.

    Returns
    -------
    list of Motor
        Every motor defined in the file.

    Raises
    ------
    EngParseError
        If the file is missing, unreadable or malformed.
    """
    file_path = Path(path).expanduser()
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise EngParseError(f"Cannot read {file_path}: {exc}") from exc
    return parse_eng_text(text, origin=str(file_path))


def write_eng_file(motor: Motor, path: str | Path, *, n_points: int = 32) -> Path:
    """Write a motor to a RASP ``.eng`` file.

    Useful for exporting a synthesised curve so it can be inspected in
    OpenRocket or RockSim.

    Parameters
    ----------
    motor:
        Motor to serialise.
    path:
        Destination path.
    n_points:
        Number of thrust samples to write.

    Returns
    -------
    pathlib.Path
        The resolved path written.
    """
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    curve = motor.thrust_curve.resampled(n_points)
    delay_field = "-".join(
        str(int(d)) if float(d).is_integer() else f"{d:g}" for d in motor.delays
    )

    lines = [
        f"; {motor.manufacturer} {motor.designation}",
        f"; Total impulse {motor.total_impulse:.3f} N-s, "
        f"average thrust {motor.average_thrust:.2f} N, "
        f"peak thrust {motor.peak_thrust:.2f} N",
        f"; Thrust curve provenance: {motor.thrust_curve.source.value}",
        f"; Written by RocketOpt from source: {motor.source}",
        f"{motor.designation} {motor.diameter * 1e3:g} {motor.length * 1e3:g} "
        f"{delay_field} {motor.propellant_mass:.4f} {motor.total_mass:.4f} "
        f"{motor.manufacturer}",
    ]
    # The RASP convention omits the (0, 0) origin and ends at zero thrust.
    for t, f in zip(curve.times[1:], curve.thrusts[1:], strict=True):
        lines.append(f"   {t:.4f} {f:.4f}")
    if curve.thrusts[-1] != 0.0:
        lines.append(f"   {curve.burn_time:.4f} 0.0000")

    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _log.debug("Wrote %s to %s", motor.designation, file_path)
    return file_path.resolve()


def load_motors_from_directory(directory: str | Path) -> dict[str, Motor]:
    """Load every ``.eng`` file in a directory.

    Files that fail to parse are logged and skipped rather than aborting the
    scan, so one malformed download does not prevent the rest from loading.

    Parameters
    ----------
    directory:
        Directory to scan, non-recursively.

    Returns
    -------
    dict
        Mapping of motor designation to :class:`Motor`. Empty if the directory
        does not exist.
    """
    dir_path = Path(directory).expanduser()
    if not dir_path.is_dir():
        return {}

    motors: dict[str, Motor] = {}
    for eng_path in sorted(dir_path.glob("*.eng")):
        try:
            for motor in parse_eng_file(eng_path):
                if motor.designation in motors:
                    _log.warning(
                        "Duplicate motor %s in %s; keeping the first definition",
                        motor.designation,
                        eng_path.name,
                    )
                    continue
                motors[motor.designation] = motor
        except EngParseError as exc:
            _log.warning("Skipping %s: %s", eng_path.name, exc)

    if motors:
        _log.info("Loaded %d measured motor curve(s) from %s", len(motors), dir_path)
    return motors
