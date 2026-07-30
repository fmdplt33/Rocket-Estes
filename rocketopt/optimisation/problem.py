"""Design space, decoding and evaluation for the optimiser.

The optimiser searches a fixed-length real vector in ``[0, 1]^n``. Each design
variable owns a slice of that vector and knows how to map its own normalised
coordinate onto a physical value - a length in metres, a discrete stock
thickness, or a categorical choice such as a nose profile or fin material.

Normalising everything to the unit hypercube keeps the genetic operators
simple and unbiased: a mutation of a given magnitude means the same thing for
every variable regardless of its physical units, so no variable dominates the
search purely because it happens to be measured in millimetres.

Constraint handling
-------------------
Constraints are *not* folded into the objectives as weighted penalties.
Weighted penalties silently trade safety against performance, which is exactly
wrong for a constraint like fin flutter, where any violation is catastrophic.
Instead each candidate carries a total constraint violation, and the NSGA-II
selection in :mod:`rocketopt.optimisation.ga` treats any feasible design as
dominating any infeasible one. A design that flutters can therefore never win
on altitude.

References
----------
[1] Deb, K., et al. (2002). "A fast and elitist multiobjective genetic
    algorithm: NSGA-II." *IEEE Transactions on Evolutionary Computation*,
    6(2), 182-197.
[2] Deb, K. (2000). "An efficient constraint handling method for genetic
    algorithms." *Computer Methods in Applied Mechanics and Engineering*,
    186(2-4), 311-338.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from rocketopt.aerodynamics.barrowman import barrowman_analysis
from rocketopt.flight.environment import LaunchConditions
from rocketopt.flight.simulation import FlightResult, SimulationConfig, simulate
from rocketopt.geometry.components import (
    BodyTube,
    FinAirfoil,
    FinSet,
    LaunchLug,
    MotorMount,
    RecoveryDevice,
    RecoveryType,
)
from rocketopt.geometry.nose_cones import (
    SHAPE_PARAMETER_RANGES,
    NoseCone,
    NoseConeShape,
)
from rocketopt.geometry.rocket import Rocket, build_rocket
from rocketopt.propulsion.motor import MotorConfiguration
from rocketopt.structures.analysis import analyse_structure
from rocketopt.structures.materials import (
    SurfaceFinish,
    get_material,
    nose_cone_materials,
    sheet_materials,
    tube_materials,
)
from rocketopt.utils.constants import (
    MAX_STATIC_MARGIN_CALIBRES,
    MIN_LIFTOFF_THRUST_TO_WEIGHT,
    MIN_RAIL_EXIT_VELOCITY,
    MIN_STATIC_MARGIN_CALIBRES,
)
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "Objective",
    "VariableKind",
    "DesignVariable",
    "DesignSpace",
    "DesignConstraints",
    "Evaluation",
    "default_design_space",
    "decode",
    "evaluate",
]


class Objective(str, Enum):
    """A quantity the optimiser can be asked to improve.

    Every objective is defined so that **larger is better**; the genetic
    algorithm maximises without exception, which removes a whole class of
    sign-convention bugs.
    """

    APOGEE = "apogee"
    """Peak altitude [m]."""

    MAX_VELOCITY = "max_velocity"
    """Peak speed [m/s]."""

    LOW_DRAG = "low_drag"
    """Negative of the coasting drag coefficient [-]."""

    FLIGHT_DURATION = "flight_duration"
    """Total time from ignition to landing [s]."""

    PAYLOAD = "payload"
    """Payload mass carried [kg]."""

    MANUFACTURABILITY = "manufacturability"
    """Ease of construction, 0 to 1 [-]."""

    LIGHT = "light"
    """Negative of dry mass [kg]."""

    @property
    def label(self) -> str:
        """Human-readable name."""
        return {
            Objective.APOGEE: "Maximum altitude",
            Objective.MAX_VELOCITY: "Maximum velocity",
            Objective.LOW_DRAG: "Minimum drag",
            Objective.FLIGHT_DURATION: "Maximum flight duration",
            Objective.PAYLOAD: "Maximum payload",
            Objective.MANUFACTURABILITY: "Ease of construction",
            Objective.LIGHT: "Minimum mass",
        }[self]

    @property
    def units(self) -> str:
        """Units the raw value is reported in."""
        return {
            Objective.APOGEE: "m",
            Objective.MAX_VELOCITY: "m/s",
            Objective.LOW_DRAG: "-",
            Objective.FLIGHT_DURATION: "s",
            Objective.PAYLOAD: "kg",
            Objective.MANUFACTURABILITY: "-",
            Objective.LIGHT: "kg",
        }[self]


class VariableKind(str, Enum):
    """How a design variable maps from its normalised coordinate."""

    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    STOCK = "stock"
    """Continuous in the search, snapped to a purchasable size on decode."""


@dataclass(frozen=True, slots=True)
class DesignVariable:
    """One searchable design parameter.

    Attributes
    ----------
    name:
        Identifier used in reports and result dictionaries.
    kind:
        Mapping rule; see :class:`VariableKind`.
    low:
        Lower physical bound, for continuous and integer variables.
    high:
        Upper physical bound.
    choices:
        Allowed values for a categorical variable.
    units:
        Unit label for reporting.
    description:
        What the variable controls.
    """

    name: str
    kind: VariableKind
    low: float = 0.0
    high: float = 1.0
    choices: tuple[object, ...] = field(default=())
    units: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        """Validate the variable definition."""
        if self.kind is VariableKind.CATEGORICAL:
            if not self.choices:
                raise ValueError(f"{self.name}: a categorical variable needs choices")
        elif self.high <= self.low:
            raise ValueError(
                f"{self.name}: high ({self.high}) must exceed low ({self.low})"
            )

    def decode(self, u: float) -> object:
        """Map a normalised coordinate onto a physical value.

        Parameters
        ----------
        u:
            Normalised coordinate, clamped to ``[0, 1]``.

        Returns
        -------
        object
            A float for continuous and stock variables, an int for integer
            variables, or the selected element for a categorical variable.
        """
        u = min(max(u, 0.0), 1.0)

        if self.kind is VariableKind.CATEGORICAL:
            # Map [0, 1] onto the choice indices, keeping the final choice
            # from occupying a vanishingly thin slice at u == 1.
            index = min(int(u * len(self.choices)), len(self.choices) - 1)
            return self.choices[index]

        value = self.low + u * (self.high - self.low)

        if self.kind is VariableKind.INTEGER:
            return int(round(value))

        return value

    def encode(self, value: object) -> float:
        """Map a physical value back onto its normalised coordinate.

        The inverse of :meth:`decode`, used to place a known design into the
        optimiser's search space as a seed.

        Parameters
        ----------
        value:
            A physical value, or one of :attr:`choices` for a categorical
            variable.

        Returns
        -------
        float
            Normalised coordinate in ``[0, 1]``. Categorical choices map to the
            centre of their slice, so that a small mutation does not
            immediately flip the choice.

        Raises
        ------
        ValueError
            If a categorical value is not one of the allowed choices.
        """
        if self.kind is VariableKind.CATEGORICAL:
            try:
                index = self.choices.index(value)
            except ValueError:
                raise ValueError(
                    f"{self.name}: {value!r} is not one of {self.choices}"
                ) from None
            return (index + 0.5) / len(self.choices)

        span = self.high - self.low
        if span <= 0.0:  # pragma: no cover - guarded by __post_init__
            return 0.0
        return min(max((float(value) - self.low) / span, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class DesignConstraints:
    """Limits a design must respect to be considered feasible.

    Attributes
    ----------
    max_length:
        Maximum overall length [m].
    max_diameter:
        Maximum body diameter [m].
    max_dry_mass:
        Maximum structural mass [kg].
    min_static_margin:
        Minimum static margin [calibres].
    max_static_margin:
        Maximum static margin [calibres].
    min_thrust_to_weight:
        Minimum lift-off thrust-to-weight ratio [-].
    min_rail_exit_velocity:
        Minimum speed leaving the rail [m/s].
    flutter_margin:
        Required ratio of flutter speed to peak flight speed [-].
    safety_factor:
        Structural safety factor applied to material allowables.
    max_landing_velocity:
        Maximum acceptable descent speed at touchdown [m/s].
    required_payload:
        Payload mass that must be carried [kg].
    """

    max_length: float = 1.2
    max_diameter: float = 0.060
    max_dry_mass: float = 0.500
    min_static_margin: float = MIN_STATIC_MARGIN_CALIBRES
    max_static_margin: float = MAX_STATIC_MARGIN_CALIBRES
    min_thrust_to_weight: float = MIN_LIFTOFF_THRUST_TO_WEIGHT
    min_rail_exit_velocity: float = MIN_RAIL_EXIT_VELOCITY
    flutter_margin: float = 1.5
    safety_factor: float = 1.5
    max_landing_velocity: float = 6.0
    required_payload: float = 0.0


def default_design_space(motor: MotorConfiguration) -> DesignSpace:
    """Build a sensible design space for a given motor.

    Bounds are scaled to the motor so that the search is neither absurdly
    wide nor artificially narrow. A body tube cannot be smaller than the
    motor it must contain, and there is no point searching a two-metre
    airframe for an A8.

    Parameters
    ----------
    motor:
        The motor the rocket is designed around.

    Returns
    -------
    DesignSpace
        The variable set to search.
    """
    motor_obj = motor.motor
    # The body must clear the motor plus a motor tube wall and a little slop.
    min_body_radius = motor_obj.diameter / 2.0 + 1.2e-3
    max_body_radius = max(min_body_radius * 2.2, min_body_radius + 0.010)

    # Length scales with total impulse: bigger motors want longer airframes.
    impulse_scale = (motor_obj.total_impulse / 8.82) ** 0.4
    min_body_length = max(motor_obj.length * 1.4, 0.10)
    max_body_length = min(0.90, 0.30 + 0.45 * impulse_scale)

    variables = (
        DesignVariable(
            name="nose_shape",
            kind=VariableKind.CATEGORICAL,
            choices=tuple(NoseConeShape),
            description="Nose cone profile family",
        ),
        DesignVariable(
            name="nose_shape_parameter",
            kind=VariableKind.CONTINUOUS,
            low=0.0,
            high=1.0,
            description="Normalised free parameter for parametric nose families",
        ),
        DesignVariable(
            name="nose_fineness",
            kind=VariableKind.CONTINUOUS,
            low=1.5,
            high=6.0,
            units="-",
            description="Nose length divided by body diameter",
        ),
        DesignVariable(
            name="nose_material",
            kind=VariableKind.CATEGORICAL,
            choices=tuple(m.name for m in nose_cone_materials()),
            description="Nose cone material",
        ),
        DesignVariable(
            name="body_radius",
            kind=VariableKind.CONTINUOUS,
            low=min_body_radius,
            high=max_body_radius,
            units="m",
            description="Body tube outer radius",
        ),
        DesignVariable(
            name="body_length",
            kind=VariableKind.CONTINUOUS,
            low=min_body_length,
            high=max_body_length,
            units="m",
            description="Body tube length",
        ),
        DesignVariable(
            name="body_material",
            kind=VariableKind.CATEGORICAL,
            choices=tuple(m.name for m in tube_materials()),
            description="Body tube material",
        ),
        DesignVariable(
            name="wall_thickness",
            kind=VariableKind.STOCK,
            low=0.30e-3,
            high=1.60e-3,
            units="m",
            description="Body tube wall thickness",
        ),
        DesignVariable(
            name="fin_count",
            kind=VariableKind.INTEGER,
            low=3,
            high=5,
            units="-",
            description="Number of fins",
        ),
        DesignVariable(
            name="fin_root_chord",
            kind=VariableKind.CONTINUOUS,
            low=0.020,
            high=0.120,
            units="m",
            description="Fin root chord",
        ),
        DesignVariable(
            name="fin_taper_ratio",
            kind=VariableKind.CONTINUOUS,
            low=0.0,
            high=1.0,
            units="-",
            description="Tip chord as a fraction of root chord",
        ),
        DesignVariable(
            name="fin_span",
            kind=VariableKind.CONTINUOUS,
            low=0.015,
            high=0.090,
            units="m",
            description="Exposed fin semi-span",
        ),
        DesignVariable(
            name="fin_sweep_fraction",
            kind=VariableKind.CONTINUOUS,
            low=0.0,
            high=1.2,
            units="-",
            description="Leading-edge sweep as a fraction of span",
        ),
        DesignVariable(
            name="fin_thickness",
            kind=VariableKind.STOCK,
            low=0.79e-3,
            high=6.35e-3,
            units="m",
            description="Fin stock thickness",
        ),
        DesignVariable(
            name="fin_material",
            kind=VariableKind.CATEGORICAL,
            choices=tuple(m.name for m in sheet_materials()),
            description="Fin material",
        ),
        DesignVariable(
            name="fin_airfoil",
            kind=VariableKind.CATEGORICAL,
            choices=tuple(FinAirfoil),
            description="Fin cross-section",
        ),
        DesignVariable(
            name="fillet_ratio",
            kind=VariableKind.CONTINUOUS,
            low=0.0,
            high=2.0,
            units="-",
            description="Fillet radius as a multiple of fin thickness",
        ),
        DesignVariable(
            name="surface_finish",
            kind=VariableKind.CATEGORICAL,
            choices=tuple(SurfaceFinish),
            description="External surface finish",
        ),
        DesignVariable(
            name="nose_ballast",
            kind=VariableKind.CONTINUOUS,
            low=0.0,
            high=0.030,
            units="kg",
            description="Ballast mass added in the nose",
        ),
    )

    return DesignSpace(variables=variables, motor=motor)


@dataclass(frozen=True, slots=True)
class DesignSpace:
    """The set of variables the optimiser searches.

    Attributes
    ----------
    variables:
        Ordered design variables. The order fixes the meaning of each gene.
    motor:
        Motor the design is built around.
    """

    variables: tuple[DesignVariable, ...]
    motor: MotorConfiguration

    @property
    def dimension(self) -> int:
        """Number of genes in a design vector."""
        return len(self.variables)

    def index_of(self, name: str) -> int:
        """Return the gene index of a named variable.

        Parameters
        ----------
        name:
            Variable name.

        Returns
        -------
        int
            Index into a design vector.

        Raises
        ------
        KeyError
            If no variable has that name.
        """
        for index, variable in enumerate(self.variables):
            if variable.name == name:
                return index
        raise KeyError(f"no design variable named {name!r}")

    def baseline_vector(self) -> list[float]:
        """Return a normalised vector describing a known-sound conventional design.

        This exists so the optimiser never starts from nothing. In a
        nineteen-dimensional space with a stability band, a flutter margin and
        a full set of structural checks, a small random population can easily
        contain no feasible design at all - the user then gets a failure
        instead of an answer. Seeding one conventional design guarantees a
        feasible starting point, so the search can only improve on it.

        The baseline is a three-fin design of the kind that has flown reliably
        for decades, not a tuned optimum.

        One caveat that is a property of the motors rather than of this
        function: on an E or F the baseline will not meet the 15 m/s rail-exit
        constraint from a standard 0.91 m rod. An F15 is a long, soft burn and
        leaves that rod at about 12.5 m/s; roughly 1.3 m of rail is needed.
        Pass a longer ``rail_length_m`` in the launch conditions for those
        motors, which is what a flyer does in practice.

        Everything is scaled to the motor rather than fixed. A fixed-size fin
        set is feasible on an 18 mm C but not on a 24 mm D: the larger motor is
        heavier and sits at the tail, pulling the centre of gravity aft, so
        proportionally smaller fins leave too little static margin. Fin span
        and chord therefore scale with body diameter, and fin thickness with it
        too, since a faster vehicle needs a stiffer fin to stay clear of
        divergence.

        Returns
        -------
        list of float
            A design vector in ``[0, 1]^n``, ordered to match
            :attr:`variables`.
        """
        motor = self.motor.motor

        body_radius = motor.diameter / 2.0 + 2.6e-3
        body_diameter = 2.0 * body_radius

        # Proportions taken from the Alpha III, which these reproduce at 18 mm.
        fin_span = 1.80 * body_diameter
        fin_root = 2.40 * body_diameter
        # A thicker fin on a bigger, faster rocket. Flutter and divergence
        # speeds both scale with thickness to the power 1.5, so a modest
        # increase buys a large aeroelastic margin. The seed is deliberately
        # conservative here; the optimiser thins the fin again wherever the
        # flutter constraint allows it.
        if motor.total_impulse <= 10.0:
            fin_thickness = max(2.38e-3, body_diameter / 10.0)
        else:
            fin_thickness = max(3.0e-3, body_diameter / 7.0)

        body_length = max(motor.length * 2.6, 0.18, fin_root / 0.8)

        # Fin material by impulse class. Balsa is the traditional choice and is
        # fine on an A to C, but a D-class rocket passes 200 m/s and balsa fins
        # of the size needed for stability would flutter well inside the flight
        # envelope. Flutter speed goes as the square root of shear modulus, so
        # basswood buys 1.7x over balsa and plywood a little more again. This
        # is the substitution a designer makes for the same reason.
        if motor.total_impulse <= 10.0:
            fin_material = "balsa"
        elif motor.total_impulse <= 20.0:
            fin_material = "basswood"
        else:
            fin_material = "birch_plywood"

        targets: dict[str, object] = {
            "nose_shape": NoseConeShape.TANGENT_OGIVE,
            "nose_shape_parameter": 0.5,
            "nose_fineness": 3.0,
            "nose_material": "pla",
            # A little above the minimum bore so the motor tube certainly fits.
            "body_radius": body_radius,
            "body_length": body_length,
            "body_material": "kraft_paper",
            "wall_thickness": 0.45e-3,
            "fin_count": 3,
            "fin_root_chord": fin_root,
            "fin_taper_ratio": 0.55,
            "fin_span": fin_span,
            "fin_sweep_fraction": 0.6,
            "fin_thickness": fin_thickness,
            "fin_material": fin_material,
            "fin_airfoil": FinAirfoil.ROUNDED,
            "fillet_ratio": 1.2,
            "surface_finish": SurfaceFinish.REGULAR_PAINT,
            "nose_ballast": 0.0,
        }

        vector: list[float] = []
        for variable in self.variables:
            if variable.name in targets:
                vector.append(variable.encode(targets[variable.name]))
            else:  # pragma: no cover - every variable is covered above
                vector.append(0.5)

        return self._repair_stability(vector)

    def _repair_stability(self, vector: list[float]) -> list[float]:
        """Grow the fins and body until the baseline is comfortably stable.

        Scaled proportions alone cannot cover the whole motor range: a D12 is
        42 g on a thirty-gram airframe, so over half the lift-off mass sits at
        the tail and the centre of gravity is dragged aft further than any
        fixed fin proportion compensates for.

        Rather than hand-tune a proportion per motor, the baseline repairs
        itself. Span is increased first because it is the strongest lever on
        the centre of pressure, then body length, which moves the fins aft of
        the centre of gravity. Only the Barrowman analysis is needed, which
        costs microseconds - no flight simulation is run here.

        Parameters
        ----------
        vector:
            The proportionally-scaled starting vector.

        Returns
        -------
        list of float
            A vector meeting the static-margin constraint with a margin to
            spare, or the best reached if the variable bounds prevent it.
        """
        span_index = self.index_of("fin_span")
        length_index = self.index_of("body_length")
        burn_time = self.motor.motor.burn_time

        # Aim above the bare minimum so the seed is not sitting on the
        # constraint boundary, where a single mutation makes it infeasible.
        target = MIN_STATIC_MARGIN_CALIBRES + 0.25

        best = list(vector)
        for _ in range(30):
            try:
                rocket = decode(self, vector)
            except (ValueError, KeyError, AssertionError):
                return best

            analysis = barrowman_analysis(rocket, mach=0.3)
            calibre = rocket.reference_diameter
            worst = min(
                (analysis.centre_of_pressure - rocket.cg_at(0.0)) / calibre,
                (analysis.centre_of_pressure - rocket.cg_at(burn_time)) / calibre,
            )
            if worst >= target:
                return vector

            best = list(vector)
            if vector[span_index] < 0.985:
                vector[span_index] = min(vector[span_index] + 0.05, 1.0)
            elif vector[length_index] < 0.985:
                vector[length_index] = min(vector[length_index] + 0.05, 1.0)
            else:
                break

        return best

    def decode_all(self, vector: Sequence[float]) -> dict[str, object]:
        """Decode a full design vector into a named parameter dictionary.

        Parameters
        ----------
        vector:
            Normalised design vector of length :attr:`dimension`.

        Returns
        -------
        dict
            Physical parameter values keyed by variable name.

        Raises
        ------
        ValueError
            If the vector length does not match the space dimension.
        """
        if len(vector) != self.dimension:
            raise ValueError(
                f"design vector has {len(vector)} genes but the space has "
                f"{self.dimension} variables"
            )
        return {
            variable.name: variable.decode(float(u))
            for variable, u in zip(self.variables, vector, strict=True)
        }


def decode(space: DesignSpace, vector: Sequence[float]) -> Rocket:
    """Build a :class:`Rocket` from a normalised design vector.

    Several parameters are expressed as *ratios* rather than absolute values -
    nose fineness rather than nose length, fin taper ratio rather than tip
    chord, fillet radius as a multiple of fin thickness. This keeps the search
    space well conditioned: a given normalised coordinate produces a sensibly
    proportioned rocket at any body diameter, so the optimiser does not waste
    its population on geometrically absurd candidates.

    Parameters
    ----------
    space:
        The design space.
    vector:
        Normalised design vector.

    Returns
    -------
    Rocket
        The decoded design.

    Raises
    ------
    ValueError
        If the decoded parameters cannot form a valid rocket. Callers should
        treat this as an infeasible candidate rather than an error.
    """
    p = space.decode_all(vector)
    motor = space.motor
    motor_obj = motor.motor

    body_radius = float(p["body_radius"])
    body_length = float(p["body_length"])
    wall_material = get_material(str(p["body_material"]))
    wall_thickness = wall_material.nearest_stock_thickness(float(p["wall_thickness"]))
    # A wall cannot be thicker than the tube it forms.
    wall_thickness = min(wall_thickness, body_radius * 0.4)

    body_tube = BodyTube(
        length=body_length,
        outer_radius=body_radius,
        wall_thickness=wall_thickness,
        material=wall_material,
    )

    # -- Nose ------------------------------------------------------------
    nose_shape = p["nose_shape"]
    assert isinstance(nose_shape, NoseConeShape)
    nose_length = float(p["nose_fineness"]) * 2.0 * body_radius

    shape_parameter = 1.0
    if nose_shape.uses_shape_parameter:
        low, high = SHAPE_PARAMETER_RANGES[nose_shape]
        shape_parameter = low + float(p["nose_shape_parameter"]) * (high - low)

    nose_material = get_material(str(p["nose_material"]))
    # Wood is turned solid; thermoplastics are printed as a shell.
    solid = nose_material.category.value == "wood"

    nose = NoseCone(
        shape=nose_shape,
        length=nose_length,
        base_radius=body_radius,
        material=nose_material,
        wall_thickness=min(1.5e-3, body_radius * 0.3),
        shape_parameter=shape_parameter,
        solid=solid,
        shoulder_length=min(0.025, body_length * 0.15),
    )

    # -- Fins --------------------------------------------------------------
    fin_material = get_material(str(p["fin_material"]))
    fin_thickness = fin_material.nearest_stock_thickness(float(p["fin_thickness"]))
    root_chord = float(p["fin_root_chord"])
    # The fin root must fit on the body tube.
    root_chord = min(root_chord, body_length * 0.85)
    span = float(p["fin_span"])
    airfoil = p["fin_airfoil"]
    assert isinstance(airfoil, FinAirfoil)

    fins = FinSet(
        count=int(p["fin_count"]),
        root_chord=root_chord,
        tip_chord=root_chord * float(p["fin_taper_ratio"]),
        span=span,
        sweep_length=span * float(p["fin_sweep_fraction"]),
        thickness=fin_thickness,
        material=fin_material,
        body_radius=body_radius,
        airfoil=airfoil,
        fillet_radius=float(p["fillet_ratio"]) * fin_thickness,
    )

    # -- Motor mount --------------------------------------------------------
    mount_length = motor_obj.length
    if mount_length > body_length:
        raise ValueError(
            f"motor {motor.designation} is longer than the {body_length * 1e3:.0f} mm "
            f"body tube"
        )
    motor_mount = MotorMount(
        inner_diameter=motor_obj.diameter + 0.8e-3,
        length=mount_length,
        wall_thickness=wall_thickness,
        material=wall_material,
        body_inner_radius=body_tube.inner_radius,
        centring_ring_count=2,
        position=nose_length + body_length - mount_length,
    )

    # -- Recovery -----------------------------------------------------------
    # Size the canopy for a sensible descent rate given the expected mass.
    # A rough dry-mass estimate is enough; the simulation will report the
    # actual descent rate and the constraint check will police it.
    estimated_mass = (
        nose.mass + body_tube.mass + fins.mass + motor_mount.mass + 0.010
    )
    target_descent = 4.0
    required_area = (
        2.0 * estimated_mass * 9.80665 / (1.225 * 0.75 * target_descent**2)
    )
    chute_diameter = min(max(2.0 * math.sqrt(required_area / math.pi), 0.15), 0.60)

    recovery = RecoveryDevice(
        kind=RecoveryType.PARACHUTE_FLAT,
        diameter=chute_diameter,
        shroud_line_count=6,
        shroud_line_length=chute_diameter,
    )

    lug = LaunchLug(
        length=0.035,
        outer_radius=0.0021,
        inner_radius=0.0016,
        material=wall_material,
    )

    surface_finish = p["surface_finish"]
    assert isinstance(surface_finish, SurfaceFinish)

    return build_rocket(
        nose=nose,
        body_tube=body_tube,
        fins=fins,
        motor=motor,
        recovery=recovery,
        motor_mount=motor_mount,
        launch_lug=lug,
        nose_ballast_mass=float(p["nose_ballast"]),
        surface_finish=surface_finish,
        name=f"Optimised {motor.designation}",
    )


@dataclass(slots=True)
class Evaluation:
    """The result of evaluating one candidate design.

    Attributes
    ----------
    rocket:
        The decoded design, or ``None`` if decoding failed.
    flight:
        The simulated flight, or ``None`` if the design could not fly.
    objectives:
        Objective values in the order requested, all to be maximised.
    violations:
        Named constraint violations; each value is the amount by which the
        constraint is breached, in its own units, and is zero when satisfied.
    feasible:
        Whether every constraint is satisfied.
    error:
        Message explaining why evaluation failed, if it did.
    """

    rocket: Rocket | None = None
    flight: FlightResult | None = None
    objectives: tuple[float, ...] = field(default=())
    violations: dict[str, float] = field(default_factory=dict)
    feasible: bool = False
    error: str = ""

    @property
    def total_violation(self) -> float:
        """Sum of all normalised constraint violations [-].

        Used by the constrained-domination rule [2]: among two infeasible
        designs, the one with the smaller total violation is preferred.
        """
        return sum(self.violations.values())


def _objective_value(
    objective: Objective,
    rocket: Rocket,
    flight: FlightResult,
) -> float:
    """Compute one objective, always oriented so that larger is better."""
    match objective:
        case Objective.APOGEE:
            return flight.apogee
        case Objective.MAX_VELOCITY:
            return flight.max_velocity
        case Objective.LOW_DRAG:
            coasting = [
                s.drag_coefficient
                for s in flight.states
                if s.phase.value == "coast" and s.drag_coefficient > 0.0
            ]
            mean_cd = sum(coasting) / len(coasting) if coasting else 1.0
            return -mean_cd
        case Objective.FLIGHT_DURATION:
            return flight.flight_duration
        case Objective.PAYLOAD:
            return rocket.payload_mass
        case Objective.MANUFACTURABILITY:
            return rocket.manufacturability
        case Objective.LIGHT:
            return -rocket.dry_mass
        case _:  # pragma: no cover - Enum is exhaustive
            raise ValueError(f"unhandled objective {objective}")


def evaluate(
    space: DesignSpace,
    vector: Sequence[float],
    *,
    objectives: Sequence[Objective],
    constraints: DesignConstraints,
    conditions: LaunchConditions,
    sim_config: SimulationConfig | None = None,
) -> Evaluation:
    """Decode, simulate and score one candidate design.

    Any failure - an invalid geometry, a rocket that cannot lift off - is
    caught and returned as an infeasible :class:`Evaluation` rather than
    raised. An optimiser must be able to propose nonsense and be told it is
    nonsense without the run aborting.

    Parameters
    ----------
    space:
        The design space.
    vector:
        Normalised design vector.
    objectives:
        Objectives to score, in order.
    constraints:
        Feasibility limits.
    conditions:
        Launch conditions to fly in.
    sim_config:
        Simulation settings. Defaults to fast point-mass mode, which is what
        the inner loop of an optimisation should use.

    Returns
    -------
    Evaluation
        Objectives, constraint violations and the underlying design.
    """
    sim_config = sim_config or SimulationConfig.fast()

    try:
        rocket = decode(space, vector)
    except (ValueError, KeyError, AssertionError) as exc:
        return Evaluation(error=f"invalid geometry: {exc}", violations={"geometry": 1.0})

    try:
        flight = simulate(rocket, conditions, sim_config)
    except ValueError as exc:
        return Evaluation(
            rocket=rocket,
            error=f"unflyable: {exc}",
            violations={"liftoff": 1.0},
        )

    violations: dict[str, float] = {}

    def breach(name: str, amount: float, scale: float) -> None:
        """Record a violation, normalised by a characteristic scale."""
        if amount > 0.0:
            violations[name] = amount / scale

    # -- Geometry limits ---------------------------------------------------
    breach("length", rocket.length - constraints.max_length, constraints.max_length)
    breach(
        "diameter",
        rocket.reference_diameter - constraints.max_diameter,
        constraints.max_diameter,
    )
    breach("dry_mass", rocket.dry_mass - constraints.max_dry_mass, constraints.max_dry_mass)
    breach(
        "payload",
        constraints.required_payload - rocket.payload_mass,
        max(constraints.required_payload, 1e-3),
    )

    # -- Stability ----------------------------------------------------------
    analysis = barrowman_analysis(rocket, mach=0.3)
    margin_loaded = (
        analysis.centre_of_pressure - rocket.cg_at(0.0)
    ) / rocket.reference_diameter
    margin_burnout = (
        analysis.centre_of_pressure - rocket.cg_at(rocket.motor.motor.burn_time)
    ) / rocket.reference_diameter
    worst_low = min(margin_loaded, margin_burnout)
    worst_high = max(margin_loaded, margin_burnout)

    breach("static_margin_low", constraints.min_static_margin - worst_low, 1.0)
    breach("static_margin_high", worst_high - constraints.max_static_margin, 1.0)

    # -- Launch safety -------------------------------------------------------
    breach(
        "thrust_to_weight",
        constraints.min_thrust_to_weight - rocket.thrust_to_weight,
        constraints.min_thrust_to_weight,
    )
    breach(
        "rail_exit",
        constraints.min_rail_exit_velocity - flight.rail_exit_velocity,
        constraints.min_rail_exit_velocity,
    )
    breach(
        "landing_velocity",
        flight.landing_velocity - constraints.max_landing_velocity,
        constraints.max_landing_velocity,
    )

    # -- Structure -----------------------------------------------------------
    try:
        structure = analyse_structure(
            rocket,
            max_velocity=flight.max_velocity,
            max_dynamic_pressure=flight.max_dynamic_pressure,
            max_acceleration=flight.max_acceleration,
            landing_velocity=flight.landing_velocity,
            safety_factor=constraints.safety_factor,
        )
    except ValueError as exc:
        return Evaluation(
            rocket=rocket,
            flight=flight,
            error=f"structural analysis failed: {exc}",
            violations={"structure": 1.0},
        )

    required_flutter = constraints.flutter_margin * flight.max_velocity
    breach(
        "flutter",
        required_flutter - structure.flutter.flutter_velocity,
        max(required_flutter, 1.0),
    )
    breach(
        "divergence",
        required_flutter - structure.flutter.divergence_velocity,
        max(required_flutter, 1.0),
    )
    for check in structure.checks:
        if not check.passes:
            # Utilisation above 1 is the fractional overload.
            breach(check.name.lower().replace(" ", "_"), check.utilisation - 1.0, 1.0)

    values = tuple(_objective_value(o, rocket, flight) for o in objectives)

    return Evaluation(
        rocket=rocket,
        flight=flight,
        objectives=values,
        violations=violations,
        feasible=not violations,
    )
