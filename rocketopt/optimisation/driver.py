"""High-level optimisation entry points.

This is the layer most users touch: give it a motor designation and it returns
a near-optimal rocket, its flight, its structural assessment and the Pareto
front that produced it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Sequence

from rocketopt.flight.environment import LaunchConditions
from rocketopt.flight.simulation import FlightResult, SimulationConfig, simulate
from rocketopt.geometry.rocket import Rocket
from rocketopt.optimisation.ga import (
    GAConfig,
    GenerationRecord,
    Individual,
    OptimisationHistory,
    nsga2,
)
from rocketopt.optimisation.problem import (
    DesignConstraints,
    DesignSpace,
    Evaluation,
    Objective,
    decode,
    default_design_space,
    evaluate,
)
from rocketopt.propulsion.database import get_motor_configuration
from rocketopt.structures.analysis import StructuralReport, analyse_structure
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = ["OptimisationOutcome", "optimise", "optimise_for_motor"]


@dataclass(slots=True)
class OptimisationOutcome:
    """Everything an optimisation run produces.

    Attributes
    ----------
    best:
        The design selected as the single best compromise.
    flight:
        A full 6-DOF simulation of :attr:`best`.
    structure:
        Structural assessment of :attr:`best` at its flight loads.
    history:
        Convergence record and the raw Pareto front.
    space:
        The design space that was searched.
    objectives:
        Objectives that were optimised, in order.
    pareto_designs:
        Decoded designs on the Pareto front, paired with their evaluations.
    """

    best: Rocket
    flight: FlightResult
    structure: StructuralReport
    history: OptimisationHistory
    space: DesignSpace
    objectives: tuple[Objective, ...]
    pareto_designs: list[tuple[Rocket, Evaluation]] = field(default_factory=list)

    @property
    def apogee(self) -> float:
        """Apogee of the selected design [m]."""
        return self.flight.apogee

    def summary(self) -> str:
        """Return a readable multi-line summary of the outcome."""
        lines = [
            self.best.summary(),
            "",
            self.flight.summary(),
            "",
            self.structure.summary(),
            "",
            f"Optimisation: {self.history.evaluations} evaluations, "
            f"{len(self.history.pareto_front)} designs on the Pareto front",
        ]
        converged = self.history.converged_generation
        if converged is not None:
            lines.append(f"  Converged by generation {converged}")
        else:
            lines.append(
                "  Still improving when the run ended; consider more generations"
            )
        return "\n".join(lines)

    def pareto_table(self) -> list[dict[str, float]]:
        """Return the Pareto front as rows for a table or plot.

        Returns
        -------
        list of dict
            One row per non-dominated design, with each objective under its
            own label plus a few headline geometry figures.
        """
        rows: list[dict[str, float]] = []
        for rocket, ev in self.pareto_designs:
            row: dict[str, float] = {}
            for objective, value in zip(self.objectives, ev.objectives, strict=True):
                row[objective.label] = value
            row["Length (mm)"] = rocket.length * 1e3
            row["Diameter (mm)"] = rocket.reference_diameter * 1e3
            row["Dry mass (g)"] = rocket.dry_mass * 1e3
            row["Static margin (cal)"] = (
                ev.flight.min_static_margin if ev.flight else 0.0
            )
            rows.append(row)
        return rows


def _select_best(
    front: Sequence[Individual],
    objectives: Sequence[Objective],
) -> Individual:
    """Choose one design from the Pareto front as the headline answer.

    The front is by definition a set of incomparable trade-offs, so picking a
    single winner requires a stated rule. The rule used here is to maximise
    the *first* objective, breaking ties by crowding distance. The first
    objective is whichever the caller listed first, so the caller retains
    control without needing to supply weights.

    Parameters
    ----------
    front:
        Non-dominated individuals.
    objectives:
        The objectives, in order.

    Returns
    -------
    Individual
        The selected design.

    Raises
    ------
    ValueError
        If the front is empty.
    """
    if not front:
        raise ValueError("cannot select from an empty Pareto front")
    del objectives  # Rule uses objective 0 only; kept for signature clarity.
    return max(front, key=lambda ind: (ind.objectives[0], ind.crowding))


def optimise(
    space: DesignSpace,
    *,
    objectives: Sequence[Objective] = (Objective.APOGEE,),
    constraints: DesignConstraints | None = None,
    conditions: LaunchConditions | None = None,
    ga_config: GAConfig | None = None,
    seed_designs: Sequence[Sequence[float]] | None = None,
    final_sim_config: SimulationConfig | None = None,
    progress_fn: Callable[[GenerationRecord], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> OptimisationOutcome:
    """Optimise a design space and return the best design with its analysis.

    The search itself uses the fast point-mass flight model; the winning
    design is then re-flown in full 6-DOF so the reported numbers come from
    the higher-fidelity model.

    Parameters
    ----------
    space:
        Design space to search.
    objectives:
        Objectives to maximise, in priority order. The first is used to pick
        the single headline design from the Pareto front.
    constraints:
        Feasibility limits. Defaults to :class:`DesignConstraints`.
    conditions:
        Launch conditions. Defaults to still air at sea level.
    ga_config:
        Genetic algorithm settings.
    seed_designs:
        Optional starting design vectors.
    final_sim_config:
        Simulation settings for the final re-flight. Defaults to 6-DOF.

    Returns
    -------
    OptimisationOutcome
        The selected design, its flight and structural analysis, and the
        Pareto front.

    Raises
    ------
    RuntimeError
        If no feasible design was found. The message lists which constraints
        were hardest to satisfy, which is far more useful than a bare failure.
    """
    constraints = constraints or DesignConstraints()
    conditions = conditions or LaunchConditions.standard()
    ga_config = ga_config or GAConfig()
    objectives = tuple(objectives)

    if not objectives:
        raise ValueError("at least one objective is required")

    def evaluate_fn(genes: list[float]) -> Evaluation:
        return evaluate(
            space,
            genes,
            objectives=objectives,
            constraints=constraints,
            conditions=conditions,
        )

    _log.info(
        "Optimising %s for %s over %d variables, %d objectives",
        space.motor.designation,
        ", ".join(o.label for o in objectives),
        space.dimension,
        len(objectives),
    )

    # Always seed a conventional baseline unless the caller supplied its own
    # starting designs. Without it a small population can contain no feasible
    # design at all and the run fails outright; with it the optimiser is
    # guaranteed to return something at least as good as a sound conventional
    # rocket.
    if seed_designs is None:
        seed_designs = [space.baseline_vector()]

    history = nsga2(
        evaluate_fn,
        space.dimension,
        ga_config,
        seed_designs=seed_designs,
        progress_fn=progress_fn,
        should_cancel=should_cancel,
    )

    feasible = [ind for ind in history.pareto_front if ind.feasible]
    if not feasible:
        raise RuntimeError(_infeasible_message(history, constraints))

    best_individual = _select_best(feasible, objectives)
    best_rocket = decode(space, best_individual.genes)

    # Re-fly the winner at full fidelity.
    flight = simulate(
        best_rocket,
        conditions,
        final_sim_config or SimulationConfig(six_dof=True),
    )
    structure = analyse_structure(
        best_rocket,
        max_velocity=flight.max_velocity,
        max_dynamic_pressure=flight.max_dynamic_pressure,
        max_acceleration=flight.max_acceleration,
        landing_velocity=flight.landing_velocity,
        safety_factor=constraints.safety_factor,
    )

    pareto_designs: list[tuple[Rocket, Evaluation]] = []
    for individual in feasible:
        if individual.evaluation is None:
            continue
        try:
            pareto_designs.append(
                (decode(space, individual.genes), individual.evaluation)
            )
        except ValueError:  # pragma: no cover - front members always decode
            continue

    return OptimisationOutcome(
        best=best_rocket,
        flight=flight,
        structure=structure,
        history=history,
        space=space,
        objectives=objectives,
        pareto_designs=pareto_designs,
    )


def _infeasible_message(
    history: OptimisationHistory,
    constraints: DesignConstraints,
) -> str:
    """Build an explanatory message when no feasible design was found."""
    del constraints
    best_violation = min((r.min_violation for r in history.records), default=float("inf"))
    return (
        f"No feasible design was found in {history.evaluations} evaluations. "
        f"The closest candidate still violated its constraints by {best_violation:.3g} "
        f"(normalised). The usual causes are a maximum length or diameter too "
        f"small for the motor, a static margin band too narrow to hit, or a "
        f"flutter margin unreachable with the permitted fin materials. Relax "
        f"the binding constraint or allow a stiffer fin material."
    )


def optimise_for_motor(
    motor: str,
    *,
    objectives: Sequence[Objective] | Sequence[str] = (Objective.APOGEE,),
    constraints: DesignConstraints | None = None,
    conditions: LaunchConditions | None = None,
    population: int = 64,
    generations: int = 40,
    seed: int | None = 20260729,
    verbose: bool = True,
) -> OptimisationOutcome:
    """Design a near-optimal rocket for a named motor.

    This is the one-line entry point behind :func:`rocketopt.quick_design`.

    Parameters
    ----------
    motor:
        Motor designation such as ``"C6-5"``. When no delay is given, the
        motor's longest delay is used.
    objectives:
        Objectives to maximise, as :class:`Objective` members or their string
        values.
    constraints:
        Feasibility limits.
    conditions:
        Launch conditions.
    population:
        Genetic algorithm population size.
    generations:
        Number of generations.
    seed:
        Random seed for a reproducible run. Pass ``None`` for a fresh search.
    verbose:
        Log progress each generation.

    Returns
    -------
    OptimisationOutcome
        The optimised design and its analysis.
    """
    motor_config = get_motor_configuration(motor)
    space = default_design_space(motor_config)

    resolved = tuple(
        o if isinstance(o, Objective) else Objective(str(o)) for o in objectives
    )

    return optimise(
        space,
        objectives=resolved,
        constraints=constraints,
        conditions=conditions,
        ga_config=GAConfig(
            population_size=population,
            generations=generations,
            seed=seed,
            verbose=verbose,
        ),
    )
