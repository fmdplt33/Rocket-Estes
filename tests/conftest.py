"""Shared fixtures for the RocketOpt test suite."""

from __future__ import annotations

import pytest

from rocketopt.aerodynamics.atmosphere import Atmosphere
from rocketopt.flight.environment import LaunchConditions
from rocketopt.flight.simulation import SimulationConfig, simulate
from rocketopt.geometry.components import (
    BodyTube,
    FinAirfoil,
    FinSet,
    LaunchLug,
    RecoveryDevice,
    RecoveryType,
)
from rocketopt.geometry.nose_cones import NoseCone, NoseConeShape
from rocketopt.geometry.rocket import Rocket, build_rocket
from rocketopt.propulsion.database import get_motor_configuration
from rocketopt.structures.materials import SurfaceFinish, get_material


@pytest.fixture(scope="session")
def reference_rocket() -> Rocket:
    """An Estes Alpha III-class reference design on a C6-5.

    The motor mount is left to ``build_rocket``, which turns a one-piece mount
    to the body tube bore. That is 8 g of solid annulus against the 1.4 g of the
    kit's paper tube and two centring rings, and 8 g at the tail of a 26 g
    airframe costs about a quarter of a calibre of static margin, so the design
    carries the 2 g of nose ballast needed to keep the margin an Alpha III flies
    with. A design that wants the lighter ring-carried mount passes one to
    ``build_rocket`` explicitly.

    Session-scoped because it is immutable and its derived geometry is cached;
    rebuilding it per test would dominate the suite's run time.
    """
    kraft = get_material("kraft_paper")
    balsa = get_material("balsa")
    pla = get_material("pla")

    return build_rocket(
        nose=NoseCone(
            shape=NoseConeShape.TANGENT_OGIVE,
            length=0.0744,
            base_radius=0.0124,
            material=pla,
            wall_thickness=1.2e-3,
            shoulder_length=0.019,
        ),
        body_tube=BodyTube(
            length=0.216, outer_radius=0.0124, wall_thickness=0.45e-3, material=kraft
        ),
        fins=FinSet(
            count=3,
            root_chord=0.055,
            tip_chord=0.030,
            span=0.042,
            sweep_length=0.025,
            thickness=2.38e-3,
            material=balsa,
            body_radius=0.0124,
            airfoil=FinAirfoil.ROUNDED,
            fillet_radius=3.0e-3,
        ),
        motor=get_motor_configuration("C6-5"),
        recovery=RecoveryDevice(
            kind=RecoveryType.PARACHUTE_FLAT,
            diameter=0.305,
            shroud_line_count=6,
            shroud_line_length=0.30,
        ),
        launch_lug=LaunchLug(
            length=0.035, outer_radius=0.0021, inner_radius=0.0016, material=kraft
        ),
        nose_ballast_mass=2.0e-3,
        surface_finish=SurfaceFinish.REGULAR_PAINT,
        name="Reference Alpha III class",
    )


@pytest.fixture(scope="session")
def standard_conditions() -> LaunchConditions:
    """Still air at sea level on a vertical 0.91 m rod."""
    return LaunchConditions.standard()


@pytest.fixture(scope="session")
def reference_flight(reference_rocket, standard_conditions):
    """A completed 3-DOF flight of the reference rocket."""
    return simulate(
        reference_rocket, standard_conditions, SimulationConfig(six_dof=False)
    )


@pytest.fixture(scope="session")
def standard_atmosphere() -> Atmosphere:
    """The dry U.S. Standard Atmosphere at sea level."""
    return Atmosphere.standard()
