"""Flight: launch conditions, rigid-body dynamics and trajectory simulation."""

from __future__ import annotations

from rocketopt.flight.dynamics import (
    euler_angles,
    quaternion_from_axis_angle,
    quaternion_from_vectors,
    quaternion_multiply,
    quaternion_normalise,
    quaternion_to_matrix,
    rotate_body_to_inertial,
    rotate_inertial_to_body,
)
from rocketopt.flight.environment import LaunchConditions, LaunchRail, WindModel
from rocketopt.flight.simulation import (
    FlightPhase,
    FlightResult,
    FlightState,
    SimulationConfig,
    simulate,
)

__all__ = [
    "FlightPhase",
    "FlightResult",
    "FlightState",
    "LaunchConditions",
    "LaunchRail",
    "SimulationConfig",
    "WindModel",
    "euler_angles",
    "quaternion_from_axis_angle",
    "quaternion_from_vectors",
    "quaternion_multiply",
    "quaternion_normalise",
    "quaternion_to_matrix",
    "rotate_body_to_inertial",
    "rotate_inertial_to_body",
    "simulate",
]
