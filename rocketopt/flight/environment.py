"""Launch conditions: rail geometry, wind field and the launch site.

Coordinate frame
----------------
An east-north-up (ENU) inertial frame anchored at the launch pad:

* ``x`` points east
* ``y`` points north
* ``z`` points up

Wind direction follows the meteorological convention: it is the direction the
wind blows *from*, measured clockwise from north. A "270 degree wind" is a
westerly, blowing towards the east, so its velocity vector points along +x.

Wind profile
------------
Wind speed increases with height above ground through the atmospheric surface
layer. The power law is used::

    V(z) = V_ref (z / z_ref)^alpha

with ``alpha`` near 1/7 over open terrain. This is the standard engineering
profile for the first hundred metres and is adequate for a model rocket, which
spends most of its flight well inside the surface layer.

References
----------
[1] Hsu, S. A., Meindl, E. A., & Gilhousen, D. B. (1994). "Determining the
    Power-Law Wind-Profile Exponent under Near-Neutral Stability Conditions at
    Sea." *Journal of Applied Meteorology*, 33(6), 757-765.
[2] Counihan, J. (1975). "Adiabatic atmospheric boundary layers." *Atmospheric
    Environment*, 9(10), 871-905.
[3] National Association of Rocketry. *Model Rocket Safety Code*, 2023 rev.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rocketopt.aerodynamics.atmosphere import Atmosphere

__all__ = ["WindModel", "LaunchRail", "LaunchConditions"]

_REFERENCE_HEIGHT: float = 10.0
"""Standard anemometer height [m]; met reports quote wind at this height."""


@dataclass(frozen=True, slots=True)
class WindModel:
    """A sheared wind field with optional gusting.

    Attributes
    ----------
    speed:
        Wind speed at the 10 m reference height [m/s].
    direction:
        Direction the wind blows *from*, clockwise from north [rad].
    shear_exponent:
        Power-law exponent. 1/7 (0.143) over open grassland, up to 0.4 over
        built-up terrain [1][2].
    turbulence_intensity:
        Standard deviation of gusts as a fraction of mean speed. Zero gives a
        steady wind, which is the right choice for repeatable optimisation.
    reference_height:
        Height the quoted speed applies at [m].
    """

    speed: float = 0.0
    direction: float = 0.0
    shear_exponent: float = 1.0 / 7.0
    turbulence_intensity: float = 0.0
    reference_height: float = _REFERENCE_HEIGHT

    def __post_init__(self) -> None:
        """Validate the wind model."""
        if self.speed < 0.0:
            raise ValueError("wind speed must not be negative")
        if not 0.0 <= self.shear_exponent <= 1.0:
            raise ValueError("shear exponent must lie in [0, 1]")
        if self.turbulence_intensity < 0.0:
            raise ValueError("turbulence intensity must not be negative")
        if self.reference_height <= 0.0:
            raise ValueError("reference height must be positive")

    @classmethod
    def from_degrees(
        cls,
        speed: float,
        direction_deg: float,
        **kwargs: float,
    ) -> WindModel:
        """Build a wind model with the direction given in degrees.

        Parameters
        ----------
        speed:
            Wind speed at reference height [m/s].
        direction_deg:
            Direction the wind blows from, clockwise from north [deg].
        **kwargs:
            Forwarded to the constructor.

        Returns
        -------
        WindModel
            The configured wind model.
        """
        return cls(speed=speed, direction=math.radians(direction_deg), **kwargs)

    def speed_at(self, height: float) -> float:
        """Mean wind speed at a height above ground level [m/s].

        Parameters
        ----------
        height:
            Height above ground [m]. Values below 1 m are clamped, since the
            power law is singular at the surface.

        Returns
        -------
        float
            Mean wind speed [m/s].
        """
        if self.speed <= 0.0:
            return 0.0
        h = max(height, 1.0)
        return self.speed * (h / self.reference_height) ** self.shear_exponent

    def velocity_at(
        self,
        height: float,
        rng: np.random.Generator | None = None,
    ) -> NDArray[np.float64]:
        """Wind velocity vector at a height, in the ENU frame [m/s].

        A wind blowing *from* bearing ``theta`` has a velocity vector pointing
        towards ``theta + 180 deg``, so the east and north components are
        ``-V sin(theta)`` and ``-V cos(theta)``.

        Parameters
        ----------
        height:
            Height above ground [m].
        rng:
            Random generator used to add a gust component when
            :attr:`turbulence_intensity` is non-zero. When ``None``, only the
            mean wind is returned, which keeps simulations deterministic.

        Returns
        -------
        numpy.ndarray
            Wind velocity ``[east, north, up]`` [m/s].
        """
        magnitude = self.speed_at(height)
        if rng is not None and self.turbulence_intensity > 0.0:
            magnitude += rng.normal(0.0, self.turbulence_intensity * magnitude)

        east = -magnitude * math.sin(self.direction)
        north = -magnitude * math.cos(self.direction)
        return np.array([east, north, 0.0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class LaunchRail:
    """The launch rod or rail that guides the rocket until it has flying speed.

    Attributes
    ----------
    length:
        Usable guided length [m]. The standard Estes rod is 0.91 m (3 ft).
    angle_from_vertical:
        Tilt away from vertical [rad]. The NAR safety code limits this to
        30 degrees [3].
    azimuth:
        Compass bearing the rail is tilted towards, clockwise from north
        [rad]. Convention is to tilt into the wind so the rocket weathercocks
        back over the pad.
    """

    length: float = 0.91
    angle_from_vertical: float = 0.0
    azimuth: float = 0.0

    def __post_init__(self) -> None:
        """Validate the rail geometry."""
        if self.length <= 0.0:
            raise ValueError("launch rail length must be positive")
        if not 0.0 <= self.angle_from_vertical <= math.radians(30.0):
            raise ValueError(
                "launch angle must lie between 0 and 30 degrees from vertical, "
                "per the NAR Model Rocket Safety Code"
            )

    @classmethod
    def from_degrees(
        cls,
        length: float = 0.91,
        angle_deg: float = 0.0,
        azimuth_deg: float = 0.0,
    ) -> LaunchRail:
        """Build a rail with angles given in degrees.

        Parameters
        ----------
        length:
            Guided length [m].
        angle_deg:
            Tilt from vertical [deg].
        azimuth_deg:
            Bearing tilted towards, clockwise from north [deg].

        Returns
        -------
        LaunchRail
            The configured rail.
        """
        return cls(
            length=length,
            angle_from_vertical=math.radians(angle_deg),
            azimuth=math.radians(azimuth_deg),
        )

    @property
    def direction(self) -> NDArray[np.float64]:
        """Unit vector along the rail in the ENU frame, pointing skyward."""
        tilt = self.angle_from_vertical
        return np.array(
            [
                math.sin(tilt) * math.sin(self.azimuth),
                math.sin(tilt) * math.cos(self.azimuth),
                math.cos(tilt),
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class LaunchConditions:
    """Everything about the launch site and the day.

    Attributes
    ----------
    atmosphere:
        Atmospheric model, already re-based on site conditions.
    wind:
        Wind field.
    rail:
        Launch rail geometry.
    """

    atmosphere: Atmosphere
    wind: WindModel
    rail: LaunchRail

    @classmethod
    def standard(cls) -> LaunchConditions:
        """Return still air at sea level on a vertical 0.91 m rod."""
        return cls(
            atmosphere=Atmosphere.standard(),
            wind=WindModel(),
            rail=LaunchRail(),
        )

    @classmethod
    def from_inputs(
        cls,
        *,
        elevation_m: float = 0.0,
        temperature_c: float = 15.0,
        pressure_pa: float | None = None,
        humidity_percent: float = 0.0,
        wind_speed_ms: float = 0.0,
        wind_direction_deg: float = 0.0,
        rail_length_m: float = 0.91,
        rail_angle_deg: float = 0.0,
        rail_azimuth_deg: float | None = None,
        turbulence_intensity: float = 0.0,
    ) -> LaunchConditions:
        """Build launch conditions from the values a user actually types in.

        Parameters
        ----------
        elevation_m:
            Site elevation above mean sea level [m].
        temperature_c:
            Air temperature [degC].
        pressure_pa:
            Station pressure [Pa]; standard for the elevation when ``None``.
        humidity_percent:
            Relative humidity [%].
        wind_speed_ms:
            Wind speed at 10 m [m/s].
        wind_direction_deg:
            Direction the wind blows from, clockwise from north [deg].
        rail_length_m:
            Guided rail length [m].
        rail_angle_deg:
            Rail tilt from vertical [deg].
        rail_azimuth_deg:
            Bearing the rail is tilted towards [deg]. When ``None``, the rail
            is tilted *into* the wind, which is standard practice: the rocket
            then weathercocks back towards the pad rather than downwind.
        turbulence_intensity:
            Gust standard deviation as a fraction of mean wind speed.

        Returns
        -------
        LaunchConditions
            The configured conditions.
        """
        if rail_azimuth_deg is None:
            # Tilt into the wind: the wind comes *from* wind_direction_deg, so
            # tilting towards that bearing points the rocket upwind.
            rail_azimuth_deg = wind_direction_deg

        return cls(
            atmosphere=Atmosphere.from_conditions(
                elevation_m=elevation_m,
                temperature_c=temperature_c,
                pressure_pa=pressure_pa,
                humidity_percent=humidity_percent,
            ),
            wind=WindModel.from_degrees(
                wind_speed_ms,
                wind_direction_deg,
                turbulence_intensity=turbulence_intensity,
            ),
            rail=LaunchRail.from_degrees(
                length=rail_length_m,
                angle_deg=rail_angle_deg,
                azimuth_deg=rail_azimuth_deg,
            ),
        )

    @property
    def ground_altitude(self) -> float:
        """Site elevation above mean sea level [m]."""
        return self.atmosphere.site_elevation
