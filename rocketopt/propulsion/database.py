"""The Estes motor database.

Certified scalar data
---------------------
Each entry records the values published on the Estes motor data sheet and
carried in the NAR Standards & Testing certification listing: case diameter
and length, loaded and propellant mass, total impulse, peak thrust, burn time
and the available ejection delays. Average thrust is *derived* as
``total_impulse / burn_time`` rather than stored, so it can never disagree
with the other two.

Thrust curves
-------------
Built-in curves are synthesised by
:func:`rocketopt.propulsion.motor.synthesise_black_powder_curve`, which
reproduces the certified total impulse, peak thrust and burn time exactly.
They are *models*, not static-test measurements.

To use real measured data, download the motor's ``.eng`` file from
ThrustCurve.org into the directory returned by
:func:`user_motor_directory` (or any directory passed to
:func:`load_database`). Measured curves automatically take precedence over the
synthesised ones, and :meth:`Motor.provenance_warning` becomes empty so that
reports stop carrying the caveat.

A caveat on specific impulse
----------------------------
Specific impulse derived from the published figures as
``I_total / (m_propellant * g0)`` ranges from about 61 s (B4) to 96 s (E16)
across this table. Black-powder propellant does not really vary that much;
the spread comes from Estes not defining "propellant weight" consistently
between data sheets - for some motors the figure appears to include the delay
and ejection composition, which contribute no useful impulse.

This does not corrupt any result computed here. The published propellant mass
is used only for the vehicle mass history, which is exactly what it measures,
and thrust comes from the certified impulse and the thrust curve, never from
Isp. :attr:`~rocketopt.propulsion.motor.Motor.specific_impulse` is reported
for reference and should be read as "impulse per unit of quoted propellant
mass", not as a combustion efficiency.

References
----------
[1] Estes Industries. *Model Rocket Engine Performance Data* (technical
    specification sheets shipped with each motor pack, and the Estes
    Educator motor chart).
[2] National Association of Rocketry, Standards & Testing Committee.
    Certified motor listings.
[3] Coker, J. ThrustCurve.org motor database, which republishes the NAR and
    TMT certification data used here.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

from rocketopt.propulsion.eng_parser import load_motors_from_directory
from rocketopt.propulsion.motor import (
    Motor,
    MotorConfiguration,
    synthesise_black_powder_curve,
)
from rocketopt.utils.config import MOTOR_ASSETS_DIR
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "CertifiedMotorData",
    "CERTIFIED_MOTORS",
    "load_database",
    "get_motor",
    "get_motor_configuration",
    "list_motors",
    "motors_by_class",
    "motors_fitting_diameter",
    "user_motor_directory",
]


@dataclass(frozen=True, slots=True)
class CertifiedMotorData:
    """Published certification data for one motor.

    Attributes
    ----------
    designation:
        Motor code without delay, e.g. ``"C6"``.
    diameter_mm:
        Case outside diameter [mm].
    length_mm:
        Case length [mm].
    total_mass_g:
        Loaded mass [g].
    propellant_mass_g:
        Propellant mass [g].
    total_impulse:
        Certified total impulse [N*s].
    peak_thrust:
        Certified maximum thrust [N].
    burn_time:
        Certified burn duration [s].
    delays:
        Available ejection delays [s]; ``0.0`` denotes a booster motor.
    notes:
        Free-text remarks shown in the UI and reports.
    """

    designation: str
    diameter_mm: float
    length_mm: float
    total_mass_g: float
    propellant_mass_g: float
    total_impulse: float
    peak_thrust: float
    burn_time: float
    delays: tuple[float, ...]
    notes: str = ""

    @property
    def average_thrust(self) -> float:
        """Derived average thrust [N]."""
        return self.total_impulse / self.burn_time

    def to_motor(self) -> Motor:
        """Build a :class:`Motor` with a synthesised thrust curve.

        Returns
        -------
        Motor
            Motor whose curve matches this record's certified scalars.
        """
        curve = synthesise_black_powder_curve(
            total_impulse=self.total_impulse,
            peak_thrust=self.peak_thrust,
            burn_time=self.burn_time,
        )
        return Motor(
            designation=self.designation,
            manufacturer="Estes",
            diameter=self.diameter_mm * 1e-3,
            length=self.length_mm * 1e-3,
            total_mass=self.total_mass_g * 1e-3,
            propellant_mass=self.propellant_mass_g * 1e-3,
            thrust_curve=curve,
            delays=self.delays,
            source="Estes published data sheet / NAR certification [1][2][3]",
        )


# ---------------------------------------------------------------------------
# Certified data table
# ---------------------------------------------------------------------------
#
# Ordered by impulse class then by average thrust. Every row satisfies
# total_impulse <= the class ceiling (A 2.5, B 5, C 10, D 20, E 40, F 80 N*s)
# and has peak_thrust > total_impulse / burn_time, which the burn model
# requires.

CERTIFIED_MOTORS: Final[dict[str, CertifiedMotorData]] = {
    entry.designation: entry
    for entry in (
        CertifiedMotorData(
            designation="A8",
            diameter_mm=18.0,
            length_mm=70.0,
            total_mass_g=16.2,
            propellant_mass_g=3.12,
            total_impulse=2.50,
            peak_thrust=9.7,
            burn_time=0.73,
            delays=(3.0, 5.0),
            notes="The standard 18 mm A. Low thrust; needs a light airframe.",
        ),
        CertifiedMotorData(
            designation="A10",
            diameter_mm=13.0,
            length_mm=45.0,
            total_mass_g=8.5,
            propellant_mass_g=3.30,
            total_impulse=2.50,
            peak_thrust=13.0,
            burn_time=0.44,
            delays=(0.0, 3.0, 5.0),
            notes=(
                "13 mm mini motor. Highest thrust-to-mass in the A class and "
                "much lighter than the A8, so it suits minimum-diameter models."
            ),
        ),
        CertifiedMotorData(
            designation="B4",
            diameter_mm=18.0,
            length_mm=70.0,
            total_mass_g=19.8,
            propellant_mass_g=8.33,
            total_impulse=5.00,
            peak_thrust=12.8,
            burn_time=1.16,
            delays=(2.0, 4.0, 6.0),
            notes="Long, soft B burn. Good for heavier or draggy models.",
        ),
        CertifiedMotorData(
            designation="B6",
            diameter_mm=18.0,
            length_mm=70.0,
            total_mass_g=19.5,
            propellant_mass_g=6.24,
            total_impulse=5.00,
            peak_thrust=12.1,
            burn_time=1.02,
            delays=(0.0, 2.0, 4.0, 6.0),
            notes="The most widely flown B. Booster variant B6-0 available.",
        ),
        CertifiedMotorData(
            designation="C5",
            diameter_mm=18.0,
            length_mm=70.0,
            total_mass_g=24.0,
            propellant_mass_g=10.8,
            total_impulse=9.00,
            peak_thrust=13.0,
            burn_time=1.80,
            delays=(3.0,),
            notes="Long-burn C, single delay option. Limited availability.",
        ),
        CertifiedMotorData(
            designation="C6",
            diameter_mm=18.0,
            length_mm=70.0,
            total_mass_g=25.8,
            propellant_mass_g=10.8,
            total_impulse=8.82,
            peak_thrust=14.09,
            burn_time=1.86,
            delays=(0.0, 3.0, 5.0, 7.0),
            notes=(
                "The reference Estes motor. Widest delay selection and the "
                "best-characterised curve in the range."
            ),
        ),
        CertifiedMotorData(
            designation="C11",
            diameter_mm=24.0,
            length_mm=70.0,
            total_mass_g=25.6,
            propellant_mass_g=10.9,
            total_impulse=8.80,
            peak_thrust=21.6,
            burn_time=0.80,
            delays=(0.0, 3.0, 5.0, 7.0),
            notes=(
                "24 mm C with more than twice the average thrust of a C6. "
                "Lifts heavier models but burns out much sooner."
            ),
        ),
        CertifiedMotorData(
            designation="D12",
            diameter_mm=24.0,
            length_mm=70.0,
            total_mass_g=42.0,
            propellant_mass_g=21.1,
            total_impulse=16.84,
            peak_thrust=29.7,
            burn_time=1.40,
            delays=(0.0, 3.0, 5.0, 7.0),
            notes="The workhorse 24 mm D. Excellent thrust-to-impulse balance.",
        ),
        CertifiedMotorData(
            designation="E12",
            diameter_mm=24.0,
            length_mm=95.0,
            total_mass_g=56.9,
            propellant_mass_g=35.8,
            total_impulse=28.45,
            peak_thrust=26.5,
            burn_time=2.37,
            delays=(0.0, 4.0, 6.0, 8.0),
            notes=(
                "Long 24 mm E. Modest peak thrust for its impulse, so it wants "
                "a light model and a long launch rod."
            ),
        ),
        CertifiedMotorData(
            designation="E16",
            diameter_mm=24.0,
            length_mm=95.0,
            total_mass_g=56.0,
            propellant_mass_g=32.0,
            total_impulse=30.00,
            peak_thrust=35.0,
            burn_time=1.88,
            delays=(4.0, 6.0, 8.0),
            notes="Higher-thrust E for heavier airframes than the E12 suits.",
        ),
        CertifiedMotorData(
            designation="F15",
            diameter_mm=29.0,
            length_mm=114.0,
            total_mass_g=102.0,
            propellant_mass_g=60.0,
            total_impulse=49.60,
            peak_thrust=25.2,
            burn_time=3.18,
            delays=(0.0, 4.0, 6.0, 8.0),
            notes=(
                "The largest black-powder motor Estes sells. Long, sustained "
                "burn; the highest average-to-peak thrust ratio in the range."
            ),
        ),
    )
}
"""Published certification data for every Estes motor RocketOpt knows about."""


def user_motor_directory() -> Path:
    """Return the directory scanned for user-supplied ``.eng`` files.

    Returns
    -------
    pathlib.Path
        ``rocketopt/assets/motors`` inside the installed package. Drop
        ThrustCurve.org downloads here to override the synthesised curves.
    """
    return MOTOR_ASSETS_DIR


@lru_cache(maxsize=8)
def load_database(eng_directory: Path | None = None) -> dict[str, Motor]:
    """Build the motor database, preferring measured curves where available.

    Parameters
    ----------
    eng_directory:
        Directory scanned for RASP ``.eng`` files. Defaults to
        :func:`user_motor_directory`. Pass an explicit path to load a private
        motor collection.

    Returns
    -------
    dict
        Mapping of designation to :class:`Motor`. Includes every certified
        Estes motor plus any additional motors found in ``.eng`` files.

    Notes
    -----
    The result is cached. Call ``load_database.cache_clear()`` after adding a
    new ``.eng`` file within a running process.
    """
    directory = eng_directory if eng_directory is not None else user_motor_directory()

    # Start from the synthesised set so every certified motor is always present.
    database: dict[str, Motor] = {
        designation: data.to_motor()
        for designation, data in CERTIFIED_MOTORS.items()
    }

    measured = load_motors_from_directory(directory)
    for designation, motor in measured.items():
        if designation in database:
            _log.info(
                "Using measured thrust curve for %s in place of the "
                "synthesised one",
                designation,
            )
        database[designation] = motor

    return database


def _split_designation(name: str) -> tuple[str, float | None]:
    """Split ``"C6-5"`` into ``("C6", 5.0)``, or ``"C6"`` into ``("C6", None)``.

    Parameters
    ----------
    name:
        Motor designation, with or without a delay suffix.

    Returns
    -------
    tuple
        The bare designation and the delay in seconds, or ``None`` when no
        delay was given.
    """
    cleaned = name.strip().upper().replace(" ", "")
    if "-" not in cleaned:
        return cleaned, None

    base, _, delay_token = cleaned.rpartition("-")
    if not base:
        return cleaned, None
    try:
        return base, float(delay_token)
    except ValueError:
        # A hyphen that is part of the name rather than a delay separator.
        return cleaned, None


def get_motor(name: str, *, eng_directory: Path | None = None) -> Motor:
    """Look up a motor by designation, with or without a delay suffix.

    Parameters
    ----------
    name:
        Designation such as ``"C6"`` or ``"C6-5"``. Case-insensitive.
    eng_directory:
        Optional override for the ``.eng`` search directory.

    Returns
    -------
    Motor
        The matching motor. Any delay suffix is ignored; use
        :func:`get_motor_configuration` to keep it.

    Raises
    ------
    KeyError
        If no motor matches, listing the available designations.
    """
    database = load_database(eng_directory)
    base, _ = _split_designation(name)

    if base in database:
        return database[base]

    raise KeyError(
        f"Unknown motor {name!r}. Available motors: {sorted(database)}"
    )


def get_motor_configuration(
    name: str, *, eng_directory: Path | None = None
) -> MotorConfiguration:
    """Look up a motor and delay together, e.g. ``"C6-5"``.

    Parameters
    ----------
    name:
        Designation including delay, such as ``"D12-5"``. When no delay is
        given the motor's longest available delay is selected, which is the
        usual choice for a high-performance single-stage flight.
    eng_directory:
        Optional override for the ``.eng`` search directory.

    Returns
    -------
    MotorConfiguration
        Motor paired with the requested (or defaulted) delay.

    Raises
    ------
    KeyError
        If the motor is unknown.
    ValueError
        If the motor exists but is not sold with the requested delay.
    """
    motor = get_motor(name, eng_directory=eng_directory)
    _, delay = _split_designation(name)

    if delay is None:
        # Longest delay: the right default when coasting to apogee.
        return motor.with_delay(max(motor.delays))
    return motor.with_delay(delay)


def list_motors(*, eng_directory: Path | None = None) -> list[Motor]:
    """Return every motor in the database, ordered by total impulse.

    Parameters
    ----------
    eng_directory:
        Optional override for the ``.eng`` search directory.

    Returns
    -------
    list of Motor
        Sorted ascending by total impulse, then by average thrust.
    """
    database = load_database(eng_directory)
    return sorted(
        database.values(),
        key=lambda m: (m.total_impulse, m.average_thrust),
    )


def motors_by_class(
    letter: str, *, eng_directory: Path | None = None
) -> list[Motor]:
    """Return every motor in one NAR impulse class.

    Parameters
    ----------
    letter:
        Class letter, e.g. ``"C"``. Case-insensitive.
    eng_directory:
        Optional override for the ``.eng`` search directory.

    Returns
    -------
    list of Motor
        Motors in that class, ordered by average thrust.
    """
    target = letter.strip().upper()
    return sorted(
        (m for m in list_motors(eng_directory=eng_directory) if m.impulse_class == target),
        key=lambda m: m.average_thrust,
    )


def motors_fitting_diameter(
    body_diameter: float,
    *,
    min_wall_clearance: float = 0.0,
    eng_directory: Path | None = None,
) -> list[Motor]:
    """Return motors that physically fit inside a body tube.

    Parameters
    ----------
    body_diameter:
        Body tube *inside* diameter [m].
    min_wall_clearance:
        Additional radial clearance required between motor case and tube wall
        [m], for a motor mount tube or centring rings.
    eng_directory:
        Optional override for the ``.eng`` search directory.

    Returns
    -------
    list of Motor
        Motors whose case diameter plus twice the clearance does not exceed
        ``body_diameter``, ordered by total impulse descending so the most
        capable fitting motor comes first.
    """
    limit = body_diameter - 2.0 * min_wall_clearance
    fitting = [
        m for m in list_motors(eng_directory=eng_directory) if m.diameter <= limit
    ]
    return sorted(fitting, key=lambda m: m.total_impulse, reverse=True)
