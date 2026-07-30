"""RocketOpt - automated design and optimisation of Estes-class model rockets.

RocketOpt sizes, analyses, optimises and exports high-performance model
rockets. Every physical quantity is computed in SI units from published
aerospace relations; each equation carries a citation in its docstring.

Layers
------
``rocketopt.propulsion``
    Estes motor database, thrust-curve interpolation and RASP ``.eng`` import.
``rocketopt.geometry``
    Parametric nose cones, body tubes, transitions, fin sets and the rocket
    assembly, with mass and inertia properties.
``rocketopt.aerodynamics``
    Atmosphere model, Barrowman normal-force and centre-of-pressure analysis,
    component drag build-up and rotational damping derivatives.
``rocketopt.flight``
    Six-degree-of-freedom trajectory simulation including launch rail, wind,
    recovery deployment and descent.
``rocketopt.structures``
    Fin flutter, tube buckling and stress, fin bending, landing and joint
    loads.
``rocketopt.optimisation``
    Multi-objective design optimisation via NSGA-II with optional gradient
    refinement.
``rocketopt.cfd``
    Panel-method and boundary-layer approximations producing pressure fields,
    streamlines and separation estimates.
``rocketopt.cad``
    Solid-model generation and STEP/STL/DXF/SVG export, plus a parametric
    Fusion 360 script generator.
``rocketopt.reports``
    Engineering, stability, aerodynamic and manufacturing documentation.

Examples
--------
>>> from rocketopt import quick_design
>>> result = quick_design("C6-5")           # doctest: +SKIP
>>> round(result.apogee_m)                  # doctest: +SKIP
312
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"
__author__ = "RocketOpt Contributors"
__license__ = "MIT"

if TYPE_CHECKING:  # pragma: no cover - import-time typing only
    from rocketopt.optimisation.driver import OptimisationOutcome


def quick_design(motor: str, **kwargs: Any) -> OptimisationOutcome:
    """Design and simulate a near-optimal rocket for ``motor`` in one call.

    This is a thin convenience wrapper over
    :func:`rocketopt.optimisation.driver.optimise_for_motor` intended for
    interactive use. Import the underlying modules directly for full control.

    Parameters
    ----------
    motor:
        Estes motor designation, e.g. ``"C6-5"`` or ``"D12-5"``.
    **kwargs:
        Forwarded to :func:`rocketopt.optimisation.driver.optimise_for_motor`.

    Returns
    -------
    OptimisationOutcome
        The optimised design, its flight simulation and the Pareto front.
    """
    from rocketopt.optimisation.driver import optimise_for_motor

    return optimise_for_motor(motor, **kwargs)


__all__ = ["__version__", "__author__", "__license__", "quick_design"]
