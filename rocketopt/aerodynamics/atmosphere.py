"""Atmospheric model with humidity and non-standard surface conditions.

Implements the U.S. Standard Atmosphere 1976 troposphere and lower
stratosphere, extended in two ways that matter for a real launch day:

*Non-standard surface conditions*
    The temperature and pressure profiles are re-based on the measured
    conditions at the launch site rather than on 15 degC and 1013.25 hPa.

*Humidity*
    Moist air is *less* dense than dry air at the same pressure and
    temperature, because water vapour has a lower molar mass than air
    (18.02 vs 28.96 g/mol). This is often assumed backwards. The effect is
    small - about 1% density reduction at 30 degC and 100% relative humidity -
    but it is cheap to include and moves apogee by a metre or two.

Altitude convention
-------------------
Every ``altitude`` argument in this module is **geopotential** altitude above
mean sea level, which is the variable the hydrostatic equation integrates and
the one standard-atmosphere tables are defined against. Geometric altitude
differs by ``Z - H = Z^2 / (r0 + Z)`` - about 10 m at 8 km and under 0.2 m at
1 km. For a model rocket the distinction is far below every other uncertainty
in the model, but it is stated here because silently mixing the two produces a
0.15% pressure error that is otherwise very hard to account for.

Density of moist air uses the partial-pressure form::

    rho = (p_d / (R_d T)) + (p_v / (R_v T))

where ``p_v`` is the vapour partial pressure and ``p_d = p - p_v`` the dry-air
partial pressure. Saturation vapour pressure comes from the Buck equation,
which is accurate to better than 0.05% over -40 to +50 degC.

References
----------
[1] U.S. Standard Atmosphere, 1976. NOAA-S/T 76-1562, NASA-TM-X-74335.
[2] Buck, A. L. (1981). "New Equations for Computing Vapor Pressure and
    Enhancement Factor." *Journal of Applied Meteorology*, 20(12), 1527-1532.
[3] Picard, A., et al. (2008). "Revised formula for the density of moist air
    (CIPM-2007)." *Metrologia*, 45(2), 149-155.
[4] White, F. M. (2006). *Viscous Fluid Flow*, 3rd ed., Eq. 1-36 (Sutherland).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rocketopt.utils.constants import (
    GAMMA_AIR,
    H_TROPOPAUSE,
    LAPSE_TROPOSPHERE,
    P0_SEA_LEVEL,
    R_AIR,
    R_VAPOUR,
    SUTHERLAND_MU_REF,
    SUTHERLAND_S,
    SUTHERLAND_T_REF,
    T0_SEA_LEVEL,
)
from rocketopt.utils.constants import G0 as _G0

__all__ = [
    "AtmosphereState",
    "Atmosphere",
    "saturation_vapour_pressure",
    "dynamic_viscosity",
]


def saturation_vapour_pressure(temperature: float) -> float:
    """Saturation vapour pressure of water over liquid water [Pa].

    Buck equation [2], Eq. 4a, valid from -40 to +50 degC::

        e_s = 611.21 exp[(18.678 - T_c/234.5) (T_c / (257.14 + T_c))]

    Parameters
    ----------
    temperature:
        Air temperature [K].

    Returns
    -------
    float
        Saturation vapour pressure [Pa].
    """
    t_c = temperature - 273.15
    return 611.21 * math.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def dynamic_viscosity(temperature: float) -> float:
    """Dynamic viscosity of air by Sutherland's law [Pa*s].

    White [4], Eq. 1-36::

        mu = mu_ref (T / T_ref)^1.5 (T_ref + S) / (T + S)

    Parameters
    ----------
    temperature:
        Air temperature [K].

    Returns
    -------
    float
        Dynamic viscosity [Pa*s].
    """
    t_ratio = temperature / SUTHERLAND_T_REF
    return (
        SUTHERLAND_MU_REF
        * t_ratio**1.5
        * (SUTHERLAND_T_REF + SUTHERLAND_S)
        / (temperature + SUTHERLAND_S)
    )


@dataclass(frozen=True, slots=True)
class AtmosphereState:
    """Atmospheric properties at one altitude.

    Attributes
    ----------
    altitude:
        Geometric altitude above mean sea level [m].
    temperature:
        Static temperature [K].
    pressure:
        Static pressure [Pa].
    density:
        Mass density [kg/m^3].
    speed_of_sound:
        Speed of sound [m/s].
    dynamic_viscosity:
        Dynamic viscosity [Pa*s].
    """

    altitude: float
    temperature: float
    pressure: float
    density: float
    speed_of_sound: float
    dynamic_viscosity: float

    @property
    def kinematic_viscosity(self) -> float:
        """Kinematic viscosity ``mu / rho`` [m^2/s]."""
        return self.dynamic_viscosity / self.density

    def mach(self, velocity: float) -> float:
        """Mach number for a given speed.

        Parameters
        ----------
        velocity:
            Speed [m/s].

        Returns
        -------
        float
            Mach number [-].
        """
        return abs(velocity) / self.speed_of_sound

    def reynolds(self, velocity: float, length: float) -> float:
        """Reynolds number based on ``length``.

        ``Re = rho V L / mu``.

        Parameters
        ----------
        velocity:
            Speed [m/s].
        length:
            Reference length [m].

        Returns
        -------
        float
            Reynolds number [-].
        """
        return self.density * abs(velocity) * length / self.dynamic_viscosity

    def dynamic_pressure(self, velocity: float) -> float:
        """Dynamic pressure ``q = 0.5 rho V^2`` [Pa].

        Parameters
        ----------
        velocity:
            Speed [m/s].

        Returns
        -------
        float
            Dynamic pressure [Pa].
        """
        return 0.5 * self.density * velocity * velocity


@dataclass(frozen=True, slots=True)
class Atmosphere:
    """A layered atmosphere re-based on measured launch-site conditions.

    Attributes
    ----------
    site_elevation:
        Launch site elevation above mean sea level [m].
    site_temperature:
        Measured air temperature at the site [K].
    site_pressure:
        Measured *station* pressure at the site [Pa]. This is the absolute
        pressure at the launch site, not the sea-level-corrected QNH that
        aviation weather reports quote.
    relative_humidity:
        Relative humidity at the site, 0 to 1.
    lapse_rate:
        Temperature lapse rate through the troposphere [K/m]. Defaults to the
        USSA-1976 standard value; a measured sounding can be supplied instead.
    """

    site_elevation: float = 0.0
    site_temperature: float = T0_SEA_LEVEL
    site_pressure: float = P0_SEA_LEVEL
    relative_humidity: float = 0.0
    lapse_rate: float = LAPSE_TROPOSPHERE

    def __post_init__(self) -> None:
        """Validate the atmospheric definition."""
        if self.site_temperature <= 0.0:
            raise ValueError("site temperature must be above absolute zero")
        if self.site_pressure <= 0.0:
            raise ValueError("site pressure must be positive")
        if not 0.0 <= self.relative_humidity <= 1.0:
            raise ValueError(
                f"relative_humidity must lie in [0, 1]; got {self.relative_humidity}. "
                f"Pass a fraction, not a percentage."
            )
        if self.lapse_rate > 0.0:
            raise ValueError(
                "a positive lapse rate means temperature rising with altitude; "
                "supply a negative value for a normal troposphere"
            )

    @classmethod
    def standard(cls) -> Atmosphere:
        """Return the dry U.S. Standard Atmosphere at sea level."""
        return cls()

    @classmethod
    def from_conditions(
        cls,
        *,
        elevation_m: float = 0.0,
        temperature_c: float = 15.0,
        pressure_pa: float | None = None,
        humidity_percent: float = 0.0,
    ) -> Atmosphere:
        """Build an atmosphere from conventionally-quoted launch-day values.

        Parameters
        ----------
        elevation_m:
            Site elevation above mean sea level [m].
        temperature_c:
            Air temperature [degC].
        pressure_pa:
            Station pressure [Pa]. When ``None``, the standard pressure for
            the given elevation is used.
        humidity_percent:
            Relative humidity [%], 0 to 100.

        Returns
        -------
        Atmosphere
            The configured atmosphere.
        """
        if pressure_pa is None:
            pressure_pa = cls._standard_pressure_at(elevation_m)
        return cls(
            site_elevation=elevation_m,
            site_temperature=temperature_c + 273.15,
            site_pressure=pressure_pa,
            relative_humidity=humidity_percent / 100.0,
        )

    @staticmethod
    def _standard_pressure_at(altitude: float) -> float:
        """Standard atmospheric pressure at a geopotential altitude [Pa].

        USSA-1976 [1] troposphere barometric formula.
        """
        if altitude <= H_TROPOPAUSE:
            t = T0_SEA_LEVEL + LAPSE_TROPOSPHERE * altitude
            return P0_SEA_LEVEL * (t / T0_SEA_LEVEL) ** (
                -_G0 / (LAPSE_TROPOSPHERE * R_AIR)
            )
        # Isothermal lower stratosphere above 11 km.
        t_trop = T0_SEA_LEVEL + LAPSE_TROPOSPHERE * H_TROPOPAUSE
        p_trop = P0_SEA_LEVEL * (t_trop / T0_SEA_LEVEL) ** (
            -_G0 / (LAPSE_TROPOSPHERE * R_AIR)
        )
        return p_trop * math.exp(
            -_G0 * (altitude - H_TROPOPAUSE) / (R_AIR * t_trop)
        )

    def temperature_at(self, altitude: float) -> float:
        """Static temperature at an altitude above mean sea level [K].

        Parameters
        ----------
        altitude:
            Geometric altitude AMSL [m].

        Returns
        -------
        float
            Temperature [K], floored at 180 K so that extreme extrapolation
            cannot produce a non-physical value.
        """
        height_above_site = altitude - self.site_elevation
        if altitude <= H_TROPOPAUSE:
            return max(self.site_temperature + self.lapse_rate * height_above_site, 180.0)
        # Isothermal above the tropopause.
        trop_height = H_TROPOPAUSE - self.site_elevation
        return max(self.site_temperature + self.lapse_rate * trop_height, 180.0)

    def pressure_at(self, altitude: float) -> float:
        """Static pressure at an altitude above mean sea level [Pa].

        Integrates the hydrostatic equation through the lapse layer, re-based
        on the measured site pressure.

        Parameters
        ----------
        altitude:
            Geometric altitude AMSL [m].

        Returns
        -------
        float
            Pressure [Pa].
        """
        t_site = self.site_temperature
        t_here = self.temperature_at(altitude)

        if altitude <= H_TROPOPAUSE:
            # p = p0 (T/T0)^(-g / (L R))
            exponent = -_G0 / (self.lapse_rate * R_AIR)
            return self.site_pressure * (t_here / t_site) ** exponent

        # Pressure at the tropopause, then isothermal above it.
        t_trop = self.temperature_at(H_TROPOPAUSE)
        exponent = -_G0 / (self.lapse_rate * R_AIR)
        p_trop = self.site_pressure * (t_trop / t_site) ** exponent
        return p_trop * math.exp(-_G0 * (altitude - H_TROPOPAUSE) / (R_AIR * t_trop))

    def density_at(self, altitude: float) -> float:
        """Air density at an altitude above mean sea level [kg/m^3].

        Accounts for humidity via partial pressures [3]. Relative humidity is
        held constant with altitude, which is a coarse assumption but has a
        negligible effect over a model rocket's altitude band.

        Parameters
        ----------
        altitude:
            Geometric altitude AMSL [m].

        Returns
        -------
        float
            Density [kg/m^3].
        """
        t = self.temperature_at(altitude)
        p = self.pressure_at(altitude)

        if self.relative_humidity <= 0.0:
            return p / (R_AIR * t)

        p_vapour = self.relative_humidity * saturation_vapour_pressure(t)
        # Vapour pressure can never exceed the total pressure.
        p_vapour = min(p_vapour, p)
        p_dry = p - p_vapour
        return p_dry / (R_AIR * t) + p_vapour / (R_VAPOUR * t)

    def speed_of_sound_at(self, altitude: float) -> float:
        """Speed of sound at an altitude above mean sea level [m/s].

        ``a = sqrt(gamma R T)``. The humidity correction to ``gamma`` and to
        the effective gas constant is below 0.3% even at saturation, so dry-air
        values are used.

        Parameters
        ----------
        altitude:
            Geometric altitude AMSL [m].

        Returns
        -------
        float
            Speed of sound [m/s].
        """
        return math.sqrt(GAMMA_AIR * R_AIR * self.temperature_at(altitude))

    def state_at(self, altitude: float) -> AtmosphereState:
        """Return every atmospheric property at an altitude.

        Parameters
        ----------
        altitude:
            Geometric altitude AMSL [m].

        Returns
        -------
        AtmosphereState
            Bundled properties, which is cheaper than calling each accessor
            separately inside a simulation loop.
        """
        t = self.temperature_at(altitude)
        p = self.pressure_at(altitude)
        return AtmosphereState(
            altitude=altitude,
            temperature=t,
            pressure=p,
            density=self.density_at(altitude),
            speed_of_sound=math.sqrt(GAMMA_AIR * R_AIR * t),
            dynamic_viscosity=dynamic_viscosity(t),
        )

    def gravity_at(self, altitude: float) -> float:
        """Acceleration due to gravity at an altitude [m/s^2].

        Inverse-square variation with geocentric radius. Over a 1 km flight
        this changes gravity by only 0.03%, but it costs nothing to include
        and matters for the high-altitude edge of the F15's capability.

        Parameters
        ----------
        altitude:
            Geometric altitude AMSL [m].

        Returns
        -------
        float
            Local gravitational acceleration [m/s^2].
        """
        from rocketopt.utils.constants import EARTH_RADIUS_MEAN

        ratio = EARTH_RADIUS_MEAN / (EARTH_RADIUS_MEAN + altitude)
        return _G0 * ratio * ratio
