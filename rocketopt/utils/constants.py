"""Physical and engineering constants used throughout RocketOpt.

All values are SI unless the symbol name says otherwise. Each constant carries
the source it was taken from so that results remain traceable.

References
----------
[1] U.S. Standard Atmosphere, 1976. NOAA-S/T 76-1562, NASA-TM-X-74335.
[2] CODATA 2018 recommended values of the fundamental physical constants.
[3] WGS-84 / EGM-96 normal gravity, NIMA TR8350.2, 3rd ed.
[4] Sutton, G. P., & Biblarz, O. (2016). *Rocket Propulsion Elements*, 9th ed.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Gravitation
# ---------------------------------------------------------------------------

G0: Final[float] = 9.80665
"""Standard acceleration of gravity [m/s^2]. Definition, CGPM 1901; see [1]."""

EARTH_RADIUS_MEAN: Final[float] = 6_371_008.8
"""Mean volumetric Earth radius [m], WGS-84 derived; see [3]."""

# ---------------------------------------------------------------------------
# Air / thermodynamics (U.S. Standard Atmosphere 1976, sea level) [1]
# ---------------------------------------------------------------------------

R_UNIVERSAL: Final[float] = 8.31446261815324
"""Universal molar gas constant [J/(mol*K)]. CODATA 2018 exact value [2]."""

M_AIR_DRY: Final[float] = 0.0289644
"""Molar mass of dry air [kg/mol], USSA-1976 sea-level composition [1]."""

M_WATER: Final[float] = 0.01801528
"""Molar mass of water vapour [kg/mol]."""

R_AIR: Final[float] = R_UNIVERSAL / M_AIR_DRY
"""Specific gas constant of dry air [J/(kg*K)] = 287.0528... [1]."""

R_VAPOUR: Final[float] = R_UNIVERSAL / M_WATER
"""Specific gas constant of water vapour [J/(kg*K)] = 461.52... ."""

GAMMA_AIR: Final[float] = 1.4
"""Ratio of specific heats for air, diatomic ideal-gas value [1]."""

T0_SEA_LEVEL: Final[float] = 288.15
"""USSA-1976 sea-level temperature [K] [1]."""

P0_SEA_LEVEL: Final[float] = 101_325.0
"""USSA-1976 sea-level pressure [Pa] [1]."""

RHO0_SEA_LEVEL: Final[float] = 1.225
"""USSA-1976 sea-level density [kg/m^3] [1]."""

A0_SEA_LEVEL: Final[float] = 340.294
"""USSA-1976 sea-level speed of sound [m/s] [1]."""

LAPSE_TROPOSPHERE: Final[float] = -0.0065
"""Troposphere temperature lapse rate [K/m], 0-11 km geopotential [1]."""

H_TROPOPAUSE: Final[float] = 11_000.0
"""Geopotential altitude of the tropopause [m] [1]."""

# Sutherland's law coefficients for air viscosity.
# White, F. M. (2006). *Viscous Fluid Flow*, 3rd ed., Eq. 1-36.
SUTHERLAND_MU_REF: Final[float] = 1.716e-5
"""Reference dynamic viscosity [Pa*s] at ``SUTHERLAND_T_REF``."""

SUTHERLAND_T_REF: Final[float] = 273.15
"""Reference temperature for Sutherland's law [K]."""

SUTHERLAND_S: Final[float] = 110.4
"""Sutherland constant for air [K]."""

# ---------------------------------------------------------------------------
# Aerodynamic modelling constants
# ---------------------------------------------------------------------------

RE_CRITICAL: Final[float] = 5.0e5
"""Critical Reynolds number for laminar-to-turbulent transition on a flat
plate. Schlichting, H. (1979). *Boundary-Layer Theory*, 7th ed., Ch. XVII."""

MACH_INCOMPRESSIBLE_LIMIT: Final[float] = 0.3
"""Below this Mach number compressibility corrections are negligible
(<2% density error). Anderson, J. D. (2016). *Fundamentals of
Aerodynamics*, 6th ed., Sec. 8.5."""

MACH_TRANSONIC_ONSET: Final[float] = 0.8
"""Mach number at which transonic wave-drag rise begins for slender bodies.
Hoerner, S. F. (1965). *Fluid-Dynamic Drag*, Ch. 16."""

MACH_SUPERSONIC: Final[float] = 1.2
"""Above this Mach number, fully supersonic wave-drag models are used."""

# ---------------------------------------------------------------------------
# Recovery / descent
# ---------------------------------------------------------------------------

CD_PARACHUTE_FLAT_SHEET: Final[float] = 0.75
"""Drag coefficient of a flat circular sheet parachute referenced to the
canopy's flat (constructed) area. Knacke, T. W. (1992). *Parachute Recovery
Systems Design Manual*, NWC TP 6575, Table 5-1."""

CD_PARACHUTE_HEMISPHERICAL: Final[float] = 1.50
"""Drag coefficient of a hemispherical (dome) canopy, flat-area reference.
Knacke (1992), Table 5-1."""

CD_STREAMER: Final[float] = 0.30
"""Effective drag coefficient of a trailing streamer referenced to streamer
planform area. Empirical; Van Milligan, T. (2000), *Apogee Peak of Flight* 34."""

# ---------------------------------------------------------------------------
# Safety / certification
# ---------------------------------------------------------------------------

MIN_LIFTOFF_THRUST_TO_WEIGHT: Final[float] = 5.0
"""Minimum thrust-to-weight ratio recommended for a safe rail exit on a model
rocket. NAR Model Rocket Safety Code; Nakka, R. (2001) and standard model
rocketry practice both quote 5:1 as the design minimum."""

MIN_RAIL_EXIT_VELOCITY: Final[float] = 15.0
"""Minimum recommended velocity [m/s] at departure from the launch rail so
that the fins have sufficient authority. OpenRocket Technical Documentation
(Niskanen, 2013), Sec. 3.3."""

MIN_STATIC_MARGIN_CALIBRES: Final[float] = 1.0
"""Minimum static margin in body calibres for a conventionally stable model
rocket. Barrowman, J. S. (1970). *The Practical Calculation of the
Aerodynamic Characteristics of Slender Finned Vehicles*, NARAM-8 report."""

MAX_STATIC_MARGIN_CALIBRES: Final[float] = 2.5
"""Above roughly 2.5 calibres a rocket becomes over-stable and weathercocks
excessively into the wind. Barrowman (1970); OpenRocket Technical
Documentation (Niskanen, 2013), Sec. 3.3."""

__all__ = [
    "G0",
    "EARTH_RADIUS_MEAN",
    "R_UNIVERSAL",
    "M_AIR_DRY",
    "M_WATER",
    "R_AIR",
    "R_VAPOUR",
    "GAMMA_AIR",
    "T0_SEA_LEVEL",
    "P0_SEA_LEVEL",
    "RHO0_SEA_LEVEL",
    "A0_SEA_LEVEL",
    "LAPSE_TROPOSPHERE",
    "H_TROPOPAUSE",
    "SUTHERLAND_MU_REF",
    "SUTHERLAND_T_REF",
    "SUTHERLAND_S",
    "RE_CRITICAL",
    "MACH_INCOMPRESSIBLE_LIMIT",
    "MACH_TRANSONIC_ONSET",
    "MACH_SUPERSONIC",
    "CD_PARACHUTE_FLAT_SHEET",
    "CD_PARACHUTE_HEMISPHERICAL",
    "CD_STREAMER",
    "MIN_LIFTOFF_THRUST_TO_WEIGHT",
    "MIN_RAIL_EXIT_VELOCITY",
    "MIN_STATIC_MARGIN_CALIBRES",
    "MAX_STATIC_MARGIN_CALIBRES",
]
