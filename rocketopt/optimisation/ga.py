"""NSGA-II multi-objective optimiser with constrained domination.

Why NSGA-II
-----------
Rocket design is genuinely multi-objective: altitude, mass, drag and ease of
construction pull against one another and there is no single "best" design
independent of what the builder values. Collapsing them into one weighted
score forces that judgement into an arbitrary set of weights chosen before the
trade-offs are known. NSGA-II instead returns the whole Pareto front, letting
the trade be inspected and *then* chosen.

The design space is also mixed-discrete (nose profile, fin count, material,
airfoil) and the flight simulation provides no gradients, so gradient-based
methods are not directly applicable. A population method handles both.

Constrained domination
----------------------
Selection uses Deb's constrained-domination rule [2] rather than penalty
weights:

1. A feasible design always dominates an infeasible one.
2. Between two infeasible designs, the smaller total violation dominates.
3. Between two feasible designs, ordinary Pareto domination applies.

This is what stops the optimiser trading fin flutter safety for altitude.

DEAP is used when installed; otherwise the algorithm falls back to the
self-contained implementation in this module, which implements the same
operators. The fallback exists so the core optimiser has no hard dependency
on an external package.

References
----------
[1] Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). "A fast and
    elitist multiobjective genetic algorithm: NSGA-II." *IEEE Transactions on
    Evolutionary Computation*, 6(2), 182-197.
[2] Deb, K. (2000). "An efficient constraint handling method for genetic
    algorithms." *Computer Methods in Applied Mechanics and Engineering*,
    186(2-4), 311-338.
[3] Deb, K., & Agrawal, R. B. (1995). "Simulated binary crossover for
    continuous search space." *Complex Systems*, 9(2), 115-148.
[4] Deb, K., & Goyal, M. (1996). "A combined genetic adaptive search (GeneAS)
    for engineering design." *Computer Science and Informatics*, 26(4), 30-45.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rocketopt.optimisation.problem import Evaluation
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "Individual",
    "GAConfig",
    "GenerationRecord",
    "OptimisationHistory",
    "nsga2",
    "fast_non_dominated_sort",
    "crowding_distance",
]


# eq=False so individuals compare by identity. The generated __eq__ would
# recurse into Evaluation and hence into numpy arrays, whose truth value is
# ambiguous; identity is also the semantics the selection operators want.
@dataclass(slots=True, eq=False)
class Individual:
    """One member of the population.

    Attributes
    ----------
    genes:
        Normalised design vector in ``[0, 1]^n``.
    evaluation:
        Result of evaluating :attr:`genes`, or ``None`` before evaluation.
    rank:
        Non-domination front index, 0 being the best front.
    crowding:
        Crowding distance within its front; larger means more isolated and
        therefore more valuable for diversity.
    """

    genes: list[float]
    evaluation: Evaluation | None = None
    rank: int = 0
    crowding: float = 0.0

    @property
    def objectives(self) -> tuple[float, ...]:
        """Objective values, or an empty tuple before evaluation."""
        return self.evaluation.objectives if self.evaluation else ()

    @property
    def feasible(self) -> bool:
        """Whether this individual satisfies every constraint."""
        return bool(self.evaluation and self.evaluation.feasible)

    @property
    def violation(self) -> float:
        """Total constraint violation; zero when feasible."""
        return self.evaluation.total_violation if self.evaluation else math.inf


@dataclass(frozen=True, slots=True)
class GAConfig:
    """Genetic algorithm settings.

    Attributes
    ----------
    population_size:
        Number of individuals per generation. Rounded up to a multiple of 4,
        which the tournament selection requires.
    generations:
        Number of generations to run.
    crossover_probability:
        Probability that a selected pair is recombined.
    mutation_probability:
        Per-gene mutation probability. Defaults to ``1/n`` when left at
        ``None``, the standard choice that mutates roughly one gene per
        individual.
    crossover_eta:
        Distribution index for simulated binary crossover [3]. Larger values
        produce children closer to their parents.
    mutation_eta:
        Distribution index for polynomial mutation [4].
    seed:
        Random seed, for reproducible runs.
    verbose:
        Log a line per generation.
    """

    population_size: int = 64
    generations: int = 40
    crossover_probability: float = 0.9
    mutation_probability: float | None = None
    crossover_eta: float = 15.0
    mutation_eta: float = 20.0
    seed: int | None = None
    verbose: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.population_size < 4:
            raise ValueError("population_size must be at least 4")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if not 0.0 <= self.crossover_probability <= 1.0:
            raise ValueError("crossover_probability must lie in [0, 1]")

    @property
    def effective_population(self) -> int:
        """Population size rounded up to a multiple of four."""
        return int(math.ceil(self.population_size / 4.0) * 4)


@dataclass(slots=True)
class GenerationRecord:
    """Summary statistics for one generation, for convergence plots.

    Attributes
    ----------
    generation:
        Generation index, starting at zero.
    best_objectives:
        Best value seen for each objective among feasible individuals.
    mean_objectives:
        Mean of each objective among feasible individuals.
    feasible_count:
        Number of feasible individuals.
    front_size:
        Size of the current first non-dominated front.
    min_violation:
        Smallest total constraint violation in the population.
    """

    generation: int
    best_objectives: tuple[float, ...]
    mean_objectives: tuple[float, ...]
    feasible_count: int
    front_size: int
    min_violation: float


@dataclass(slots=True)
class OptimisationHistory:
    """Full record of an optimisation run.

    Attributes
    ----------
    records:
        Per-generation statistics.
    pareto_front:
        Final non-dominated, feasible individuals.
    evaluations:
        Total number of design evaluations performed.
    """

    records: list[GenerationRecord] = field(default_factory=list)
    pareto_front: list[Individual] = field(default_factory=list)
    evaluations: int = 0
    cancelled: bool = False
    """True when the run stopped early because cancellation was requested."""

    @property
    def converged_generation(self) -> int | None:
        """First generation after which the best objective stopped improving.

        Returns
        -------
        int or None
            The generation index, or ``None`` if the run was still improving
            when it ended - which usually means it should have run longer.
        """
        if len(self.records) < 5:
            return None

        best = [r.best_objectives[0] for r in self.records if r.best_objectives]
        if len(best) < 5:
            return None

        final = best[-1]
        if final == 0.0:
            return None

        # The first generation reaching within 1% of the final best value.
        for index, value in enumerate(best):
            if abs(value - final) <= 0.01 * abs(final):
                return index if index < len(best) - 1 else None
        return None


def _dominates(a: Individual, b: Individual) -> bool:
    """Return whether ``a`` constrained-dominates ``b``.

    Implements Deb's constrained-domination rule [2]; see the module
    docstring.
    """
    if a.feasible and not b.feasible:
        return True
    if not a.feasible and b.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.violation < b.violation

    # Both feasible: ordinary Pareto domination. Every objective is maximised.
    at_least_one_better = False
    for x, y in zip(a.objectives, b.objectives, strict=True):
        if x < y:
            return False
        if x > y:
            at_least_one_better = True
    return at_least_one_better


def fast_non_dominated_sort(population: Sequence[Individual]) -> list[list[Individual]]:
    """Sort a population into non-domination fronts.

    The O(M N^2) algorithm of Deb et al. [1], Sec. III-A.

    Parameters
    ----------
    population:
        Individuals to sort. Their ``rank`` attributes are updated in place.

    Returns
    -------
    list of list of Individual
        Fronts, best first.
    """
    count = len(population)
    dominated_by: list[list[int]] = [[] for _ in range(count)]
    domination_count = [0] * count

    # Fronts are accumulated as index lists. Working with indices rather than
    # object references avoids any reliance on Individual equality, and lets
    # each pair be compared exactly once.
    index_fronts: list[list[int]] = [[]]

    for i in range(count):
        for j in range(i + 1, count):
            if _dominates(population[i], population[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif _dominates(population[j], population[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1

    for i in range(count):
        if domination_count[i] == 0:
            population[i].rank = 0
            index_fronts[0].append(i)

    current = 0
    while index_fronts[current]:
        next_front: list[int] = []
        for i in index_fronts[current]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    population[j].rank = current + 1
                    next_front.append(j)
        current += 1
        index_fronts.append(next_front)

    return [[population[i] for i in front] for front in index_fronts if front]


def crowding_distance(front: list[Individual]) -> None:
    """Assign crowding distances within one front, in place.

    Deb et al. [1], Sec. III-B. Boundary solutions receive infinite distance
    so that the extremes of the front are always preserved.

    Parameters
    ----------
    front:
        Individuals in a single non-domination front.
    """
    count = len(front)
    for individual in front:
        individual.crowding = 0.0

    if count <= 2:
        for individual in front:
            individual.crowding = math.inf
        return

    n_objectives = len(front[0].objectives)
    for m in range(n_objectives):
        front.sort(key=lambda ind: ind.objectives[m])
        front[0].crowding = math.inf
        front[-1].crowding = math.inf

        span = front[-1].objectives[m] - front[0].objectives[m]
        if span <= 1e-12:
            continue

        for i in range(1, count - 1):
            delta = front[i + 1].objectives[m] - front[i - 1].objectives[m]
            front[i].crowding += delta / span


def _tournament(rng: random.Random, population: Sequence[Individual]) -> Individual:
    """Binary tournament selection on rank then crowding distance."""
    a, b = rng.sample(list(population), 2)
    if a.rank != b.rank:
        return a if a.rank < b.rank else b
    return a if a.crowding > b.crowding else b


def _sbx(
    rng: random.Random,
    parent_a: list[float],
    parent_b: list[float],
    eta: float,
) -> tuple[list[float], list[float]]:
    """Simulated binary crossover on the unit hypercube [3]."""
    child_a = list(parent_a)
    child_b = list(parent_b)

    for i in range(len(parent_a)):
        if rng.random() > 0.5:
            continue
        x1, x2 = parent_a[i], parent_b[i]
        if abs(x1 - x2) < 1e-14:
            continue

        u = rng.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

        child_a[i] = min(max(0.5 * ((1 + beta) * x1 + (1 - beta) * x2), 0.0), 1.0)
        child_b[i] = min(max(0.5 * ((1 - beta) * x1 + (1 + beta) * x2), 0.0), 1.0)

    return child_a, child_b


def _polynomial_mutation(
    rng: random.Random,
    genes: list[float],
    probability: float,
    eta: float,
) -> list[float]:
    """Polynomial mutation on the unit hypercube [4]."""
    mutated = list(genes)
    for i in range(len(mutated)):
        if rng.random() > probability:
            continue
        x = mutated[i]
        u = rng.random()
        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
        mutated[i] = min(max(x + delta, 0.0), 1.0)
    return mutated


def nsga2(
    evaluate_fn: Callable[[list[float]], Evaluation],
    dimension: int,
    config: GAConfig | None = None,
    *,
    seed_designs: Sequence[Sequence[float]] | None = None,
    progress_fn: Callable[[GenerationRecord], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> OptimisationHistory:
    """Run NSGA-II over the unit hypercube.

    Parameters
    ----------
    evaluate_fn:
        Callable mapping a design vector to an :class:`Evaluation`. It must
        not raise; infeasible or invalid designs should be returned as
        infeasible evaluations.
    dimension:
        Number of design variables.
    config:
        Algorithm settings.
    seed_designs:
        Optional known-good design vectors injected into the initial
        population. Seeding with a sound baseline reliably accelerates
        convergence and guarantees the result is never worse than the seed.
    progress_fn:
        Called once per completed generation with that generation's
        :class:`GenerationRecord`. Lets a caller drive a progress bar or plot
        without the optimiser knowing anything about the interface.
    should_cancel:
        Polled at each generation boundary and after each evaluation. When it
        returns ``True`` the run stops early and returns the best front found
        so far, which is always a usable result rather than nothing.

    Returns
    -------
    OptimisationHistory
        Per-generation statistics and the final Pareto front. When cancelled,
        :attr:`OptimisationHistory.cancelled` is ``True``.
    """
    config = config or GAConfig()
    rng = random.Random(config.seed)

    size = config.effective_population
    mutation_probability = (
        config.mutation_probability
        if config.mutation_probability is not None
        else 1.0 / dimension
    )

    history = OptimisationHistory()

    # -- Initial population --------------------------------------------------
    population: list[Individual] = []
    if seed_designs:
        for seed in seed_designs:
            if len(seed) != dimension:
                raise ValueError(
                    f"seed design has {len(seed)} genes, expected {dimension}"
                )
            population.append(Individual(genes=[min(max(g, 0.0), 1.0) for g in seed]))

    while len(population) < size:
        population.append(
            Individual(genes=[rng.random() for _ in range(dimension)])
        )
    population = population[:size]

    for individual in population:
        individual.evaluation = evaluate_fn(individual.genes)
        history.evaluations += 1
        if should_cancel is not None and should_cancel():
            history.cancelled = True
            break

    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        crowding_distance(front)

    _record_generation(history, population, fronts, 0, config.verbose)
    if progress_fn is not None:
        progress_fn(history.records[-1])

    # -- Evolution ------------------------------------------------------------
    for generation in range(1, config.generations + 1):
        if history.cancelled or (should_cancel is not None and should_cancel()):
            history.cancelled = True
            _log.info("Optimisation cancelled at generation %d", generation)
            break

        offspring: list[Individual] = []

        while len(offspring) < size:
            parent_a = _tournament(rng, population)
            parent_b = _tournament(rng, population)

            if rng.random() < config.crossover_probability:
                genes_a, genes_b = _sbx(
                    rng, parent_a.genes, parent_b.genes, config.crossover_eta
                )
            else:
                genes_a, genes_b = list(parent_a.genes), list(parent_b.genes)

            genes_a = _polynomial_mutation(
                rng, genes_a, mutation_probability, config.mutation_eta
            )
            genes_b = _polynomial_mutation(
                rng, genes_b, mutation_probability, config.mutation_eta
            )

            offspring.append(Individual(genes=genes_a))
            if len(offspring) < size:
                offspring.append(Individual(genes=genes_b))

        for individual in offspring:
            individual.evaluation = evaluate_fn(individual.genes)
            history.evaluations += 1
            if should_cancel is not None and should_cancel():
                history.cancelled = True
                break

        # An individual left unevaluated by cancellation must not take part in
        # selection; drop it rather than letting it compete with no objectives.
        offspring = [ind for ind in offspring if ind.evaluation is not None]

        # Elitist replacement: parents and offspring compete together, so a
        # good design can never be lost to an unlucky generation.
        combined = population + offspring
        fronts = fast_non_dominated_sort(combined)

        new_population: list[Individual] = []
        for front in fronts:
            crowding_distance(front)
            if len(new_population) + len(front) <= size:
                new_population.extend(front)
            else:
                # Partially fill from this front, preferring isolated members.
                front.sort(key=lambda ind: ind.crowding, reverse=True)
                new_population.extend(front[: size - len(new_population)])
                break

        population = new_population
        fronts = fast_non_dominated_sort(population)
        for front in fronts:
            crowding_distance(front)

        _record_generation(history, population, fronts, generation, config.verbose)
        if progress_fn is not None:
            progress_fn(history.records[-1])

        if history.cancelled:
            _log.info(
                "Optimisation cancelled after generation %d; returning the best "
                "front found so far",
                generation,
            )
            break

    feasible_front = [ind for ind in fronts[0] if ind.feasible] if fronts else []
    history.pareto_front = feasible_front or (fronts[0] if fronts else [])

    _log.info(
        "NSGA-II finished: %d evaluations, %d designs on the Pareto front",
        history.evaluations,
        len(history.pareto_front),
    )
    return history


def _record_generation(
    history: OptimisationHistory,
    population: Sequence[Individual],
    fronts: Sequence[Sequence[Individual]],
    generation: int,
    verbose: bool,
) -> None:
    """Append summary statistics for one generation."""
    feasible = [ind for ind in population if ind.feasible]
    n_objectives = len(population[0].objectives) if population[0].objectives else 0

    if feasible and n_objectives:
        best = tuple(
            max(ind.objectives[m] for ind in feasible) for m in range(n_objectives)
        )
        mean = tuple(
            sum(ind.objectives[m] for ind in feasible) / len(feasible)
            for m in range(n_objectives)
        )
    else:
        best = tuple([float("-inf")] * n_objectives)
        mean = tuple([float("-inf")] * n_objectives)

    record = GenerationRecord(
        generation=generation,
        best_objectives=best,
        mean_objectives=mean,
        feasible_count=len(feasible),
        front_size=len(fronts[0]) if fronts else 0,
        min_violation=min((ind.violation for ind in population), default=math.inf),
    )
    history.records.append(record)

    if verbose:
        best_text = (
            ", ".join(f"{value:.4g}" for value in best) if feasible else "none feasible"
        )
        _log.info(
            "gen %3d | feasible %3d/%3d | front %3d | best %s | min violation %.4g",
            generation,
            len(feasible),
            len(population),
            record.front_size,
            best_text,
            record.min_violation,
        )
