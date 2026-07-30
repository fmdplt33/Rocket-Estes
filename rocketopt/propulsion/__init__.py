"""Propulsion: motor models, thrust curves and the Estes motor database."""

from __future__ import annotations

from rocketopt.propulsion.database import (
    CERTIFIED_MOTORS,
    CertifiedMotorData,
    get_motor,
    get_motor_configuration,
    list_motors,
    load_database,
    motors_by_class,
    motors_fitting_diameter,
    user_motor_directory,
)
from rocketopt.propulsion.eng_parser import (
    EngParseError,
    load_motors_from_directory,
    parse_eng_file,
    parse_eng_text,
    write_eng_file,
)
from rocketopt.propulsion.motor import (
    Motor,
    MotorConfiguration,
    ThrustCurve,
    ThrustCurveSource,
    impulse_class,
    synthesise_black_powder_curve,
)

__all__ = [
    "CERTIFIED_MOTORS",
    "CertifiedMotorData",
    "EngParseError",
    "Motor",
    "MotorConfiguration",
    "ThrustCurve",
    "ThrustCurveSource",
    "get_motor",
    "get_motor_configuration",
    "impulse_class",
    "list_motors",
    "load_database",
    "load_motors_from_directory",
    "motors_by_class",
    "motors_fitting_diameter",
    "parse_eng_file",
    "parse_eng_text",
    "synthesise_black_powder_curve",
    "user_motor_directory",
    "write_eng_file",
]
