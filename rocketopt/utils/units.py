"""Unit conversion helpers.

RocketOpt computes exclusively in SI (metre, kilogram, second, kelvin, newton,
pascal). These helpers exist for the input and reporting boundaries only, where
model rocketry conventionally uses millimetres, grams, feet and degrees.

Conversion factors are exact by definition unless noted:

* 1 inch = 25.4 mm exactly (international yard and pound agreement, 1959)
* 1 foot = 0.3048 m exactly
* 1 lbf  = 4.4482216152605 N exactly (NIST SP 811, Appendix B.8)
* 1 oz   = 28.349523125 g exactly
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Exact conversion factors
# ---------------------------------------------------------------------------

MM_PER_INCH: Final[float] = 25.4
M_PER_FOOT: Final[float] = 0.3048
N_PER_LBF: Final[float] = 4.4482216152605
G_PER_OZ: Final[float] = 28.349523125
KG_PER_LB: Final[float] = 0.45359237
PA_PER_PSI: Final[float] = N_PER_LBF / (MM_PER_INCH * 1e-3) ** 2
"""Pascals per pound-force per square inch = 6894.757... Pa/psi."""

PA_PER_HPA: Final[float] = 100.0
MS_PER_KNOT: Final[float] = 1852.0 / 3600.0
"""Metres per second per knot; 1 nautical mile = 1852 m exactly (BIPM)."""
MS_PER_MPH: Final[float] = M_PER_FOOT * 5280.0 / 3600.0
MS_PER_KMH: Final[float] = 1000.0 / 3600.0

KELVIN_OFFSET: Final[float] = 273.15
"""0 degC in kelvin, exact by definition of the Celsius scale."""


# ---------------------------------------------------------------------------
# Length
# ---------------------------------------------------------------------------


def mm_to_m(value: float) -> float:
    """Convert millimetres to metres."""
    return value * 1e-3


def m_to_mm(value: float) -> float:
    """Convert metres to millimetres."""
    return value * 1e3


def inch_to_m(value: float) -> float:
    """Convert inches to metres."""
    return value * MM_PER_INCH * 1e-3


def m_to_inch(value: float) -> float:
    """Convert metres to inches."""
    return value / (MM_PER_INCH * 1e-3)


def foot_to_m(value: float) -> float:
    """Convert feet to metres."""
    return value * M_PER_FOOT


def m_to_foot(value: float) -> float:
    """Convert metres to feet."""
    return value / M_PER_FOOT


# ---------------------------------------------------------------------------
# Mass
# ---------------------------------------------------------------------------


def g_to_kg(value: float) -> float:
    """Convert grams to kilograms."""
    return value * 1e-3


def kg_to_g(value: float) -> float:
    """Convert kilograms to grams."""
    return value * 1e3


def oz_to_kg(value: float) -> float:
    """Convert avoirdupois ounces to kilograms."""
    return value * G_PER_OZ * 1e-3


def kg_to_oz(value: float) -> float:
    """Convert kilograms to avoirdupois ounces."""
    return value / (G_PER_OZ * 1e-3)


def lb_to_kg(value: float) -> float:
    """Convert avoirdupois pounds to kilograms."""
    return value * KG_PER_LB


def kg_to_lb(value: float) -> float:
    """Convert kilograms to avoirdupois pounds."""
    return value / KG_PER_LB


# ---------------------------------------------------------------------------
# Force / pressure
# ---------------------------------------------------------------------------


def lbf_to_n(value: float) -> float:
    """Convert pounds-force to newtons."""
    return value * N_PER_LBF


def n_to_lbf(value: float) -> float:
    """Convert newtons to pounds-force."""
    return value / N_PER_LBF


def psi_to_pa(value: float) -> float:
    """Convert pounds-force per square inch to pascals."""
    return value * PA_PER_PSI


def pa_to_psi(value: float) -> float:
    """Convert pascals to pounds-force per square inch."""
    return value / PA_PER_PSI


def hpa_to_pa(value: float) -> float:
    """Convert hectopascals (millibars) to pascals."""
    return value * PA_PER_HPA


def pa_to_hpa(value: float) -> float:
    """Convert pascals to hectopascals (millibars)."""
    return value / PA_PER_HPA


def mpa_to_pa(value: float) -> float:
    """Convert megapascals to pascals."""
    return value * 1e6


def pa_to_mpa(value: float) -> float:
    """Convert pascals to megapascals."""
    return value * 1e-6


# ---------------------------------------------------------------------------
# Temperature
# ---------------------------------------------------------------------------


def celsius_to_kelvin(value: float) -> float:
    """Convert degrees Celsius to kelvin."""
    return value + KELVIN_OFFSET


def kelvin_to_celsius(value: float) -> float:
    """Convert kelvin to degrees Celsius."""
    return value - KELVIN_OFFSET


def fahrenheit_to_kelvin(value: float) -> float:
    """Convert degrees Fahrenheit to kelvin."""
    return (value - 32.0) * 5.0 / 9.0 + KELVIN_OFFSET


def kelvin_to_fahrenheit(value: float) -> float:
    """Convert kelvin to degrees Fahrenheit."""
    return (value - KELVIN_OFFSET) * 9.0 / 5.0 + 32.0


# ---------------------------------------------------------------------------
# Velocity
# ---------------------------------------------------------------------------


def knot_to_ms(value: float) -> float:
    """Convert knots to metres per second."""
    return value * MS_PER_KNOT


def ms_to_knot(value: float) -> float:
    """Convert metres per second to knots."""
    return value / MS_PER_KNOT


def mph_to_ms(value: float) -> float:
    """Convert miles per hour to metres per second."""
    return value * MS_PER_MPH


def ms_to_mph(value: float) -> float:
    """Convert metres per second to miles per hour."""
    return value / MS_PER_MPH


def kmh_to_ms(value: float) -> float:
    """Convert kilometres per hour to metres per second."""
    return value * MS_PER_KMH


def ms_to_kmh(value: float) -> float:
    """Convert metres per second to kilometres per hour."""
    return value / MS_PER_KMH


# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------


def gcm3_to_kgm3(value: float) -> float:
    """Convert grams per cubic centimetre to kilograms per cubic metre."""
    return value * 1000.0


def kgm3_to_gcm3(value: float) -> float:
    """Convert kilograms per cubic metre to grams per cubic centimetre."""
    return value / 1000.0


def gsm_to_kgm2(value: float) -> float:
    """Convert grams per square metre (areal density) to kg/m^2."""
    return value * 1e-3


def kgm2_to_gsm(value: float) -> float:
    """Convert kilograms per square metre to grams per square metre."""
    return value * 1e3


__all__ = [
    "MM_PER_INCH",
    "M_PER_FOOT",
    "N_PER_LBF",
    "G_PER_OZ",
    "KG_PER_LB",
    "PA_PER_PSI",
    "PA_PER_HPA",
    "MS_PER_KNOT",
    "MS_PER_MPH",
    "MS_PER_KMH",
    "KELVIN_OFFSET",
    "mm_to_m",
    "m_to_mm",
    "inch_to_m",
    "m_to_inch",
    "foot_to_m",
    "m_to_foot",
    "g_to_kg",
    "kg_to_g",
    "oz_to_kg",
    "kg_to_oz",
    "lb_to_kg",
    "kg_to_lb",
    "lbf_to_n",
    "n_to_lbf",
    "psi_to_pa",
    "pa_to_psi",
    "hpa_to_pa",
    "pa_to_hpa",
    "mpa_to_pa",
    "pa_to_mpa",
    "celsius_to_kelvin",
    "kelvin_to_celsius",
    "fahrenheit_to_kelvin",
    "kelvin_to_fahrenheit",
    "knot_to_ms",
    "ms_to_knot",
    "mph_to_ms",
    "ms_to_mph",
    "kmh_to_ms",
    "ms_to_kmh",
    "gcm3_to_kgm3",
    "kgm3_to_gcm3",
    "gsm_to_kgm2",
    "kgm2_to_gsm",
]
