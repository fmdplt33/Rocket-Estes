"""Geometry: nose cones, airframe components, assemblies and mass properties."""

from __future__ import annotations

from rocketopt.geometry.components import (
    BodyTube,
    FinAirfoil,
    FinSet,
    LaunchLug,
    MotorMount,
    RecoveryDevice,
    RecoveryType,
    Transition,
)
from rocketopt.geometry.mass_properties import MassProperties, combine
from rocketopt.geometry.nose_cones import (
    SHAPE_PARAMETER_RANGES,
    NoseCone,
    NoseConeShape,
    default_shape_parameter,
)
from rocketopt.geometry.rocket import PlacedSection, Rocket, build_rocket

__all__ = [
    "SHAPE_PARAMETER_RANGES",
    "BodyTube",
    "FinAirfoil",
    "FinSet",
    "LaunchLug",
    "MassProperties",
    "MotorMount",
    "NoseCone",
    "NoseConeShape",
    "PlacedSection",
    "RecoveryDevice",
    "RecoveryType",
    "Rocket",
    "Transition",
    "build_rocket",
    "combine",
    "default_shape_parameter",
]
