"""Validation against published reference data.

These are the tests that matter most: they check RocketOpt against numbers
published by someone else, rather than against itself. A change that breaks one
of these has changed the physics, not just the code.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocketopt.aerodynamics.atmosphere import Atmosphere
from rocketopt.geometry.nose_cones import NoseCone, NoseConeShape
from rocketopt.propulsion.database import CERTIFIED_MOTORS, list_motors
from rocketopt.structures.analysis import fin_flutter_velocity
from rocketopt.structures.materials import get_material

pytestmark = pytest.mark.validation


# ---------------------------------------------------------------------------
# U.S. Standard Atmosphere 1976, Table I
# ---------------------------------------------------------------------------

# USSA-1976 tabulated against *geopotential* altitude, which is the variable
# the barometric formula integrates and the convention this model uses.
#
# Note on a subtlety that first showed up as a failing test here: standard
# atmosphere tables are published against both geopotential (H) and geometric
# (Z) altitude, and the two differ by about 10 m at 8 km. Mixing rows - a
# geopotential temperature against a geometric pressure - produces an apparent
# 0.15% error in an otherwise exact model. Every row below is geopotential.
#
# geopotential altitude [m], temperature [K], pressure [Pa], density [kg/m^3], a [m/s]
USSA_1976 = [
    (0.0, 288.15, 101325.0, 1.22500, 340.29),
    (1000.0, 281.65, 89874.6, 1.11164, 336.43),
    (2000.0, 275.15, 79495.2, 1.00649, 332.53),
    (5000.0, 255.65, 54019.9, 0.73612, 320.55),
    (8000.0, 236.15, 35599.8, 0.52516, 308.06),
    (11000.0, 216.65, 22632.1, 0.36392, 295.07),
]


@pytest.mark.parametrize(("altitude", "temperature", "pressure", "density", "sound"), USSA_1976)
def test_standard_atmosphere_matches_ussa1976(
    standard_atmosphere: Atmosphere,
    altitude: float,
    temperature: float,
    pressure: float,
    density: float,
    sound: float,
) -> None:
    """Atmospheric properties match the USSA-1976 tabulated values."""
    state = standard_atmosphere.state_at(altitude)
    assert state.temperature == pytest.approx(temperature, rel=1e-4)
    assert state.pressure == pytest.approx(pressure, rel=1e-3)
    assert state.density == pytest.approx(density, rel=1e-3)
    assert state.speed_of_sound == pytest.approx(sound, rel=1e-3)


def test_moist_air_is_less_dense_than_dry_air() -> None:
    """Humidity reduces density, because water vapour is lighter than air.

    This is frequently assumed backwards. Water has a molar mass of
    18.02 g/mol against dry air's 28.96, so replacing air molecules with
    vapour at fixed pressure and temperature *lowers* the density.
    """
    dry = Atmosphere.from_conditions(temperature_c=30.0, humidity_percent=0.0)
    saturated = Atmosphere.from_conditions(temperature_c=30.0, humidity_percent=100.0)

    assert saturated.density_at(0.0) < dry.density_at(0.0)
    # The effect is real but small: about 1-2% at 30 degC.
    reduction = 1.0 - saturated.density_at(0.0) / dry.density_at(0.0)
    assert 0.005 < reduction < 0.030


# ---------------------------------------------------------------------------
# Barrowman nose-cone centre of pressure
# ---------------------------------------------------------------------------

# Barrowman (1967) and Crowell (1996) tabulate X_cp as a fraction of length.
NOSE_CP_REFERENCE = [
    (NoseConeShape.CONICAL, 2.0 / 3.0, 1.0 / 3.0),
    (NoseConeShape.ELLIPTICAL, 1.0 / 3.0, 2.0 / 3.0),
    (NoseConeShape.TANGENT_OGIVE, 0.466, 0.534),
]


@pytest.mark.parametrize(("shape", "cp_fraction", "volume_fraction"), NOSE_CP_REFERENCE)
def test_nose_cone_cp_matches_barrowman(
    shape: NoseConeShape, cp_fraction: float, volume_fraction: float
) -> None:
    """Nose CP and volume coefficient match the published constants.

    All nine profiles are handled by the single slender-body expression
    ``X_cp = L - V / A_base``; these three have tabulated values to check it
    against.
    """
    cone = NoseCone(
        shape=shape,
        length=0.15,
        base_radius=0.0122,
        material=get_material("balsa"),
        solid=True,
    )
    assert cone.centre_of_pressure / cone.length == pytest.approx(cp_fraction, abs=2e-3)
    assert cone.volume_coefficient == pytest.approx(volume_fraction, abs=3e-3)


def test_von_karman_volume_coefficient() -> None:
    """The Von Karman (LD-Haack) nose encloses exactly half its cylinder.

    A known analytic property of the C = 0 Haack series.
    """
    cone = NoseCone(
        shape=NoseConeShape.VON_KARMAN,
        length=0.15,
        base_radius=0.0122,
        material=get_material("balsa"),
        solid=True,
    )
    assert cone.volume_coefficient == pytest.approx(0.5, abs=2e-3)


def test_ld_haack_is_von_karman() -> None:
    """LD-Haack and Von Karman name the same curve, so must be identical."""
    kwargs = {
        "length": 0.15,
        "base_radius": 0.0122,
        "material": get_material("balsa"),
        "solid": True,
    }
    vk = NoseCone(shape=NoseConeShape.VON_KARMAN, **kwargs)
    ld = NoseCone(shape=NoseConeShape.HAACK_LD, **kwargs)

    x = np.linspace(0.0, 0.15, 200)
    assert np.allclose(vk.radius_at(x), ld.radius_at(x), atol=1e-15)


@pytest.mark.parametrize(
    ("shape", "parameter"),
    [(NoseConeShape.POWER, 1.0), (NoseConeShape.PARABOLIC, 0.0)],
)
def test_degenerate_profiles_collapse_to_a_cone(
    shape: NoseConeShape, parameter: float
) -> None:
    """Power n=1 and parabolic K=0 are both exactly a cone."""
    kwargs = {
        "length": 0.15,
        "base_radius": 0.0122,
        "material": get_material("balsa"),
        "solid": True,
    }
    cone = NoseCone(shape=NoseConeShape.CONICAL, **kwargs)
    degenerate = NoseCone(shape=shape, shape_parameter=parameter, **kwargs)

    x = np.linspace(0.0, 0.15, 200)
    assert np.allclose(cone.radius_at(x), degenerate.radius_at(x), atol=1e-12)


def test_secant_ogive_reduces_to_tangent_ogive() -> None:
    """A secant ogive at the tangent radius *is* the tangent ogive.

    This is the regression test for a real bug: the original implementation
    used a rotated-arc form that produced nonsense (X_cp/L of -51428). The
    direct circle construction that replaced it must reduce exactly.
    """
    kwargs = {
        "length": 0.15,
        "base_radius": 0.0122,
        "material": get_material("balsa"),
        "solid": True,
    }
    tangent = NoseCone(shape=NoseConeShape.TANGENT_OGIVE, **kwargs)
    secant = NoseCone(
        shape=NoseConeShape.SECANT_OGIVE, shape_parameter=1.0, **kwargs
    )

    x = np.linspace(0.0, 0.15, 500)
    assert np.allclose(tangent.radius_at(x), secant.radius_at(x), atol=1e-12)


@pytest.mark.parametrize("ratio", [1.0, 1.5, 2.0, 3.0])
def test_secant_ogive_endpoints_and_monotonicity(ratio: float) -> None:
    """A secant ogive starts at zero radius, ends at the base, and never dips."""
    cone = NoseCone(
        shape=NoseConeShape.SECANT_OGIVE,
        length=0.15,
        base_radius=0.0122,
        material=get_material("balsa"),
        solid=True,
        shape_parameter=ratio,
    )
    x = np.linspace(0.0, 0.15, 500)
    y = cone.radius_at(x)

    assert y[0] == pytest.approx(0.0, abs=1e-12)
    assert y[-1] == pytest.approx(0.0122, abs=1e-12)
    assert np.all(np.diff(y) >= -1e-12)


# ---------------------------------------------------------------------------
# Motor database
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("designation", sorted(CERTIFIED_MOTORS))
def test_synthesised_curve_reproduces_certified_scalars(designation: str) -> None:
    """Each synthesised thrust curve matches its certified data.

    The burn model solves a shape parameter so that total impulse, peak thrust
    and burn time are all reproduced simultaneously. If that solve regresses,
    integrated performance silently changes.
    """
    certified = CERTIFIED_MOTORS[designation]
    motor = certified.to_motor()

    assert motor.total_impulse == pytest.approx(certified.total_impulse, rel=1e-3)
    assert motor.peak_thrust == pytest.approx(certified.peak_thrust, rel=1e-3)
    assert motor.burn_time == pytest.approx(certified.burn_time, rel=1e-3)
    assert motor.average_thrust == pytest.approx(certified.average_thrust, rel=1e-3)


@pytest.mark.parametrize("designation", sorted(CERTIFIED_MOTORS))
def test_impulse_accumulation_is_monotonic_and_exact(designation: str) -> None:
    """Delivered impulse rises monotonically and ends at the total.

    Regression test for the cumulative-impulse optimisation, which replaced an
    O(n) re-integration on every call with a precomputed lookup.
    """
    motor = CERTIFIED_MOTORS[designation].to_motor()
    curve = motor.thrust_curve

    previous = -1.0
    for i in range(41):
        t = motor.burn_time * i / 40.0
        value = curve.impulse_to(t)
        assert value >= previous - 1e-12
        previous = value

    assert curve.impulse_to(0.0) == pytest.approx(0.0, abs=1e-12)
    assert curve.impulse_to(motor.burn_time) == pytest.approx(
        motor.total_impulse, rel=1e-9
    )
    assert curve.impulse_to(motor.burn_time * 10) == pytest.approx(
        motor.total_impulse, rel=1e-9
    )


@pytest.mark.parametrize("designation", sorted(CERTIFIED_MOTORS))
def test_motor_mass_conserves(designation: str) -> None:
    """Motor mass falls from loaded to burnout and never outside that range."""
    motor = CERTIFIED_MOTORS[designation].to_motor()

    assert motor.mass_at(-1.0) == pytest.approx(motor.total_mass)
    assert motor.mass_at(0.0) == pytest.approx(motor.total_mass)
    assert motor.mass_at(motor.burn_time) == pytest.approx(motor.burnout_mass)
    assert motor.mass_at(1e6) == pytest.approx(motor.burnout_mass)

    masses = [motor.mass_at(motor.burn_time * i / 20.0) for i in range(21)]
    assert all(b <= a + 1e-12 for a, b in zip(masses, masses[1:], strict=False))


def test_impulse_class_boundaries() -> None:
    """Impulse class letters follow the NAR doubling scheme."""
    from rocketopt.propulsion.motor import impulse_class

    assert impulse_class(2.5) == "A"
    assert impulse_class(2.51) == "B"
    assert impulse_class(5.0) == "B"
    assert impulse_class(5.01) == "C"
    assert impulse_class(10.0) == "C"
    assert impulse_class(20.0) == "D"
    assert impulse_class(40.0) == "E"
    assert impulse_class(80.0) == "F"


def test_every_motor_class_letter_is_consistent() -> None:
    """Each motor's class letter agrees with its own total impulse."""
    from rocketopt.propulsion.motor import impulse_class

    for motor in list_motors():
        assert motor.impulse_class == impulse_class(motor.total_impulse)


# ---------------------------------------------------------------------------
# Fin flutter, NACA TN 4197
# ---------------------------------------------------------------------------


def test_flutter_scales_with_thickness_to_the_three_halves() -> None:
    """Flutter velocity scales as (t/c)^1.5, so doubling thickness gives 2^1.5.

    The dominant term in the NACA TN 4197 correlation. Getting this exponent
    wrong would make thin fins look survivable when they are not.
    """
    common = {
        "shear_modulus": 0.21e9,
        "aspect_ratio": 1.0,
        "taper_ratio": 0.55,
        "ambient_pressure": 101325.0,
        "speed_of_sound": 340.29,
    }
    thin = fin_flutter_velocity(thickness_ratio=0.02, **common)
    thick = fin_flutter_velocity(thickness_ratio=0.04, **common)

    assert thick / thin == pytest.approx(2.0**1.5, rel=1e-9)


def test_flutter_scales_with_root_shear_modulus() -> None:
    """Flutter velocity scales as sqrt(G): 100x stiffer gives 10x the speed."""
    common = {
        "aspect_ratio": 1.0,
        "taper_ratio": 0.55,
        "thickness_ratio": 0.05,
        "ambient_pressure": 101325.0,
        "speed_of_sound": 340.29,
    }
    soft = fin_flutter_velocity(shear_modulus=0.21e9, **common)
    stiff = fin_flutter_velocity(shear_modulus=21.0e9, **common)

    assert stiff / soft == pytest.approx(10.0, rel=1e-9)


def test_flutter_is_most_critical_at_low_altitude() -> None:
    """Flutter speed rises with altitude because ambient pressure falls.

    Sea level is therefore the conservative place to check it, which is what
    ``analyse_structure`` does by default.
    """
    atmosphere = Atmosphere.standard()
    common = {
        "shear_modulus": 0.21e9,
        "aspect_ratio": 1.0,
        "taper_ratio": 0.55,
        "thickness_ratio": 0.05,
    }
    speeds = []
    for altitude in (0.0, 1000.0, 3000.0):
        state = atmosphere.state_at(altitude)
        speeds.append(
            fin_flutter_velocity(
                ambient_pressure=state.pressure,
                speed_of_sound=state.speed_of_sound,
                **common,
            )
        )

    assert speeds[0] < speeds[1] < speeds[2]


def test_cone_volume_matches_analytic() -> None:
    """The numerically integrated cone volume matches pi r^2 L / 3."""
    radius, length = 0.0122, 0.15
    cone = NoseCone(
        shape=NoseConeShape.CONICAL,
        length=length,
        base_radius=radius,
        material=get_material("balsa"),
        solid=True,
    )
    analytic = math.pi * radius * radius * length / 3.0
    assert cone.enclosed_volume == pytest.approx(analytic, rel=1e-4)
