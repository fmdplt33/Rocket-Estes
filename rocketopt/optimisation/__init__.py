"""Optimisation: design space, NSGA-II engine and high-level drivers."""

from __future__ import annotations

from rocketopt.optimisation.driver import (
    OptimisationOutcome,
    optimise,
    optimise_for_motor,
)
from rocketopt.optimisation.ga import (
    GAConfig,
    GenerationRecord,
    Individual,
    OptimisationHistory,
    crowding_distance,
    fast_non_dominated_sort,
    nsga2,
)
from rocketopt.optimisation.problem import (
    DesignConstraints,
    DesignSpace,
    DesignVariable,
    Evaluation,
    Objective,
    VariableKind,
    decode,
    default_design_space,
    evaluate,
)

__all__ = [
    "DesignConstraints",
    "DesignSpace",
    "DesignVariable",
    "Evaluation",
    "GAConfig",
    "GenerationRecord",
    "Individual",
    "Objective",
    "OptimisationHistory",
    "OptimisationOutcome",
    "VariableKind",
    "crowding_distance",
    "decode",
    "default_design_space",
    "evaluate",
    "fast_non_dominated_sort",
    "nsga2",
    "optimise",
    "optimise_for_motor",
]
