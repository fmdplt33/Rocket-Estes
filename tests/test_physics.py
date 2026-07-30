"""Internal consistency and physical-sanity tests.

Where :mod:`test_validation` checks against published numbers, these check that
the model is self-consistent and behaves the way physics requires: mass is
conserved, drag opposes motion, damping opposes rotation, and the fast
approximations agree with the accurate ones.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from rocketopt.aerodynamics.barrowman import (
    barrowman_analysis,
    damping_derivatives,
    prandtl_glauert,
)
from rocketopt.aerodynamics.drag import drag_buildup, skin_friction_coefficient
from rocketopt.cfd import solve_boundary_layer, solve_potential_flow
from rocketopt.flight.dynamics import (
    quaternion_from_vectors,
    quaternion_multiply,
    quaternion_normalise,
    quaternion_to_matrix,
    rotate_body_to_inertial,
    rotate_inertial_to_body,
)
from rocketopt.flight.simulation import SimulationConfig, simulate
from rocketopt.geometry.mass_properties import MassProperties, combine


# ---------------------------------------------------------------------------
# Mass properties
# ---------------------------------------------------------------------------


def test_mass_is_conserved(reference_rocket) -> None:
    """Loaded mass equals dry mass plus the motor's loaded mass."""
    expected = reference_rocket.dry_mass + reference_rocket.motor.motor.total_mass
    assert reference_rocket.loaded_mass == pytest.approx(expected, rel=1e-12)


def test_centre_of_gravity_moves_forward_through_the_burn(reference_rocket) -> None:
    """Burning aft propellant moves the centre of gravity forward."""
    loaded = reference_rocket.cg_at(0.0)
    burnout = reference_rocket.cg_at(reference_rocket.motor.motor.burn_time)
    assert burnout < loaded


def test_combine_reduces_to_a_single_part() -> None:
    """Combining one component returns it unchanged."""
    part = MassProperties(mass=0.5, cg=0.3, ixx=1e-5, iyy=2e-4)
    assert combine([part]) == part


def test_parallel_axis_theorem() -> None:
    """Two point masses give the textbook transverse inertia about their CG."""
    left = MassProperties(mass=1.0, cg=0.0, ixx=0.0, iyy=0.0)
    right = MassProperties(mass=1.0, cg=2.0, ixx=0.0, iyy=0.0)
    combined = combine([left, right])

    assert combined.mass == pytest.approx(2.0)
    assert combined.cg == pytest.approx(1.0)
    # Two 1 kg masses 1 m either side of the CG: I = 2 * m d^2 = 2.
    assert combined.iyy == pytest.approx(2.0)


def test_combine_rejects_zero_mass() -> None:
    """A centre of gravity is undefined without mass."""
    with pytest.raises(ValueError, match="zero total mass"):
        combine([MassProperties(mass=0.0, cg=0.0, ixx=0.0, iyy=0.0)])


# ---------------------------------------------------------------------------
# Quaternions
# ---------------------------------------------------------------------------


def test_quaternion_round_trip_is_identity() -> None:
    """Rotating to the body frame and back returns the original vector."""
    rng = np.random.default_rng(20260729)
    for _ in range(20):
        q = quaternion_normalise(rng.normal(size=4))
        v = rng.normal(size=3)
        assert np.allclose(
            rotate_inertial_to_body(q, rotate_body_to_inertial(q, v)), v, atol=1e-12
        )


def test_quaternion_matrix_is_orthonormal() -> None:
    """A unit quaternion maps to a proper rotation matrix."""
    rng = np.random.default_rng(7)
    for _ in range(20):
        q = quaternion_normalise(rng.normal(size=4))
        m = quaternion_to_matrix(q)
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(m)) == pytest.approx(1.0, abs=1e-12)


def test_quaternion_from_vectors_aligns_them() -> None:
    """The shortest-arc quaternion actually rotates source onto target."""
    rng = np.random.default_rng(11)
    for _ in range(20):
        a = rng.normal(size=3)
        b = rng.normal(size=3)
        q = quaternion_from_vectors(a, b)
        rotated = rotate_body_to_inertial(q, a / np.linalg.norm(a))
        assert np.allclose(rotated, b / np.linalg.norm(b), atol=1e-9)


def test_quaternion_multiply_identity() -> None:
    """Multiplying by the identity quaternion changes nothing."""
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    q = quaternion_normalise(np.array([0.3, -0.2, 0.9, 0.1]))
    assert np.allclose(quaternion_multiply(q, identity), q, atol=1e-15)


def test_antiparallel_vectors_do_not_produce_nan() -> None:
    """The 180-degree case is handled rather than dividing by zero."""
    q = quaternion_from_vectors(np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]))
    assert np.all(np.isfinite(q))
    rotated = rotate_body_to_inertial(q, np.array([1.0, 0, 0]))
    assert np.allclose(rotated, [-1.0, 0.0, 0.0], atol=1e-9)


# ---------------------------------------------------------------------------
# Aerodynamics
# ---------------------------------------------------------------------------


def test_centre_of_pressure_is_aft_of_the_nose(reference_rocket) -> None:
    """A finned rocket's CP sits well aft, driven by the fins."""
    result = barrowman_analysis(reference_rocket, mach=0.3)
    assert result.centre_of_pressure > reference_rocket.nose.length
    assert result.centre_of_pressure < reference_rocket.length
    assert result.cn_alpha > 0.0


def test_reference_design_is_stable(reference_rocket) -> None:
    """The reference rocket sits inside the accepted stability band."""
    from rocketopt.aerodynamics.barrowman import static_margin

    loaded = static_margin(reference_rocket, t=0.0)
    burnout = static_margin(
        reference_rocket, t=reference_rocket.motor.motor.burn_time
    )
    assert 1.0 <= loaded <= 2.5
    assert 1.0 <= burnout <= 2.5
    # Stability improves through the burn as the CG moves forward.
    assert burnout > loaded


def test_prandtl_glauert_increases_with_mach() -> None:
    """The compressibility factor grows with Mach and is frozen transonically."""
    assert prandtl_glauert(0.0) == pytest.approx(1.0)
    assert prandtl_glauert(0.5) > prandtl_glauert(0.3)
    # Frozen above the transonic onset rather than diverging at Mach 1.
    assert prandtl_glauert(0.95) == pytest.approx(prandtl_glauert(0.8))
    assert math.isfinite(prandtl_glauert(1.0))


def test_damping_derivatives_oppose_rotation(reference_rocket) -> None:
    """Damping derivatives must be negative, or the rocket is a divergent
    oscillator."""
    damping = damping_derivatives(reference_rocket, t=0.0, mach=0.3)
    assert damping.cmq < 0.0
    assert damping.cnr < 0.0
    assert damping.clp < 0.0
    # Fin cant produces a positive roll moment per positive cant angle.
    assert damping.cl_delta > 0.0


def test_skin_friction_is_continuous_through_transition() -> None:
    """No step change at the laminar-turbulent transition Reynolds number.

    A discontinuity here would put a false cliff in the optimiser's objective.
    """
    kwargs = {"roughness": 20e-6, "characteristic_length": 0.3, "mach": 0.0}
    below = skin_friction_coefficient(4.99e5, **kwargs)
    at = skin_friction_coefficient(5.0e5, **kwargs)
    above = skin_friction_coefficient(5.01e5, **kwargs)

    assert below == pytest.approx(at, rel=1e-3)
    assert at == pytest.approx(above, rel=1e-3)


def test_rougher_finish_increases_drag(reference_rocket, standard_atmosphere) -> None:
    """A rougher surface must never reduce drag."""
    from dataclasses import replace

    from rocketopt.structures.materials import SurfaceFinish

    state = standard_atmosphere.state_at(0.0)
    previous = 0.0
    for finish in (
        SurfaceFinish.MIRROR,
        SurfaceFinish.POLISHED,
        SurfaceFinish.SMOOTH_PAINT,
        SurfaceFinish.REGULAR_PAINT,
        SurfaceFinish.UNFINISHED,
        SurfaceFinish.ROUGH_UNFINISHED,
    ):
        rocket = replace(reference_rocket, surface_finish=finish)
        cd = drag_buildup(rocket, state, 100.0).total
        assert cd >= previous - 1e-9
        previous = cd


def test_base_drag_falls_while_the_motor_burns(
    reference_rocket, standard_atmosphere
) -> None:
    """Exhaust raises base pressure, so base drag drops during boost."""
    state = standard_atmosphere.state_at(0.0)
    coasting = drag_buildup(reference_rocket, state, 100.0, motor_burning=False)
    burning = drag_buildup(reference_rocket, state, 100.0, motor_burning=True)
    assert burning.base < coasting.base


def test_drag_components_sum_to_the_total(
    reference_rocket, standard_atmosphere
) -> None:
    """The breakdown must add up, or the report misleads."""
    state = standard_atmosphere.state_at(0.0)
    breakdown = drag_buildup(reference_rocket, state, 100.0)
    assert sum(breakdown.as_dict().values()) == pytest.approx(
        breakdown.total, rel=1e-12
    )


def test_crossflow_only_acts_at_angle_of_attack(reference_rocket) -> None:
    """Viscous cross-flow vanishes at zero incidence, as sin^2 requires."""
    straight = barrowman_analysis(reference_rocket, mach=0.3, angle_of_attack=0.0)
    inclined = barrowman_analysis(
        reference_rocket, mach=0.3, angle_of_attack=math.radians(8.0)
    )
    assert straight.crossflow_cn == pytest.approx(0.0, abs=1e-12)
    assert inclined.crossflow_cn > 0.0


# ---------------------------------------------------------------------------
# Flight simulation
# ---------------------------------------------------------------------------


def test_flight_reaches_apogee_and_lands(reference_flight) -> None:
    """A complete flight has a positive apogee and returns to the ground."""
    assert reference_flight.apogee > 0.0
    assert reference_flight.apogee_time > 0.0
    assert reference_flight.landing_time > reference_flight.apogee_time
    assert reference_flight.landing_velocity > 0.0


def test_apogee_is_the_maximum_sampled_altitude(reference_flight) -> None:
    """The reported apogee agrees with the trajectory it came from."""
    highest = max(state.altitude for state in reference_flight.states)
    assert reference_flight.apogee == pytest.approx(highest, rel=2e-3)


def test_rail_exit_precedes_burnout(reference_flight) -> None:
    """The rocket leaves the rail long before the motor stops."""
    burn_time = reference_flight.rocket.motor.motor.burn_time
    assert 0.0 < reference_flight.rail_exit_time < burn_time


def test_fast_and_full_simulations_agree(reference_rocket, standard_conditions) -> None:
    """The optimiser's fast model matches the full 6-DOF model on apogee.

    If this drifts, the optimiser is ranking designs by something other than
    what the final report will show.
    """
    accurate = simulate(
        reference_rocket, standard_conditions, SimulationConfig(six_dof=True)
    )
    fast = simulate(reference_rocket, standard_conditions, SimulationConfig.fast())
    preview = simulate(
        reference_rocket, standard_conditions, SimulationConfig.preview()
    )

    assert fast.apogee == pytest.approx(accurate.apogee, rel=2e-3)
    assert preview.apogee == pytest.approx(accurate.apogee, rel=5e-3)
    assert fast.max_velocity == pytest.approx(accurate.max_velocity, rel=2e-3)


def test_wind_pushes_the_landing_downrange(reference_rocket) -> None:
    """Wind must move the landing point away from the pad."""
    from rocketopt.flight.environment import LaunchConditions

    calm = simulate(
        reference_rocket,
        LaunchConditions.from_inputs(wind_speed_ms=0.0),
        SimulationConfig(six_dof=False),
    )
    windy = simulate(
        reference_rocket,
        LaunchConditions.from_inputs(wind_speed_ms=6.0, wind_direction_deg=270.0),
        SimulationConfig(six_dof=False),
    )
    assert windy.landing_distance > calm.landing_distance


def test_unflyable_design_is_rejected_clearly(reference_rocket) -> None:
    """A rocket that cannot lift off raises rather than silently misbehaving."""
    from dataclasses import replace

    from rocketopt.flight.environment import LaunchConditions

    # Two kilograms of ballast on a C6 gives a thrust-to-weight well below 1.
    overloaded = replace(reference_rocket, nose_ballast_mass=2.0)
    with pytest.raises(ValueError, match="thrust-to-weight"):
        simulate(overloaded, LaunchConditions.standard(), SimulationConfig.fast())


# ---------------------------------------------------------------------------
# CFD approximations
# ---------------------------------------------------------------------------


def test_potential_flow_respects_the_stagnation_limit(reference_rocket) -> None:
    """Incompressible Cp cannot exceed 1 anywhere on the body."""
    solution = solve_potential_flow(reference_rocket, velocity=100.0)
    assert float(np.max(solution.surface_cp)) <= 1.0 + 1e-9


def test_potential_flow_relaxes_to_the_free_stream(reference_rocket) -> None:
    """Far from the body the disturbance vanishes."""
    solution = solve_potential_flow(reference_rocket, velocity=100.0)
    x = np.array([[reference_rocket.length * 0.5]])
    r = np.array([[reference_rocket.reference_radius * 300.0]])
    u, v = solution.velocity_field(x, r)

    assert float(u[0, 0]) == pytest.approx(100.0, rel=1e-3)
    assert abs(float(v[0, 0])) < 0.5


def test_flow_accelerates_over_the_nose_then_recovers(reference_rocket) -> None:
    """The suction peak sits near the shoulder, not at the tip.

    Regression test for a real bug: sampling the surface too close to the
    axial source line at a pointed tip produced a spurious Cp of -1.7 there
    and reported the tip as the suction peak.
    """
    solution = solve_potential_flow(reference_rocket, velocity=100.0)
    peak_x = float(solution.profile.x[solution.suction_peak_index])
    nose_length = reference_rocket.nose.length

    assert 0.3 * nose_length < peak_x < 1.5 * nose_length
    assert solution.suction_peak_cp < 0.0
    # Along the parallel body the flow returns to the free-stream speed.
    assert float(solution.surface_velocity[-1]) == pytest.approx(100.0, rel=0.05)


def test_design_variable_encode_inverts_decode() -> None:
    """Encoding a decoded value returns the original coordinate."""
    from rocketopt.geometry.components import FinAirfoil
    from rocketopt.optimisation.problem import DesignVariable, VariableKind

    continuous = DesignVariable(
        name="span", kind=VariableKind.CONTINUOUS, low=0.015, high=0.09
    )
    for u in (0.0, 0.25, 0.5, 0.9, 1.0):
        assert continuous.encode(continuous.decode(u)) == pytest.approx(u, abs=1e-12)

    categorical = DesignVariable(
        name="airfoil", kind=VariableKind.CATEGORICAL, choices=tuple(FinAirfoil)
    )
    for choice in FinAirfoil:
        assert categorical.decode(categorical.encode(choice)) is choice


def test_baseline_design_is_feasible() -> None:
    """The seeded baseline satisfies every constraint on every motor.

    The optimiser seeds this design so that a small population can never
    return "no feasible design found". If the baseline itself stops being
    feasible, that guarantee silently disappears.
    """
    from rocketopt.flight.environment import LaunchConditions
    from rocketopt.optimisation.problem import (
        DesignConstraints,
        Objective,
        default_design_space,
        evaluate,
    )
    from rocketopt.propulsion.database import get_motor_configuration

    constraints = DesignConstraints()

    # Rail length is the flyer's choice, not the design's, and it has to suit
    # the motor. An F15 is a long, soft burn - 25 N peak on a 160 g vehicle -
    # which leaves a standard 0.91 m rod at about 12.5 m/s, below the 15 m/s
    # the fins need. Reaching 15 m/s takes roughly 1.3 m of rail. That is a
    # real property of the motor, so the larger motors are tested on the
    # longer rail a flyer would actually use.
    rails = {
        "A8-3": 0.91,
        "B6-4": 0.91,
        "C6-5": 0.91,
        "D12-5": 0.91,
        "E12-6": 1.5,
        "F15-6": 1.5,
    }

    for designation, rail_length in rails.items():
        conditions = LaunchConditions.from_inputs(rail_length_m=rail_length)
        space = default_design_space(get_motor_configuration(designation))
        result = evaluate(
            space,
            space.baseline_vector(),
            objectives=(Objective.APOGEE,),
            constraints=constraints,
            conditions=conditions,
        )
        assert result.feasible, (
            f"baseline for {designation} violates {sorted(result.violations)}"
        )
        assert result.objectives[0] > 0.0


def test_optimiser_returns_a_design_even_with_a_tiny_budget() -> None:
    """A very small run still yields a usable answer, thanks to the seed.

    Regression test for a real failure: eight designs over three generations
    found nothing feasible at all in the nineteen-dimensional space, and the
    command-line tool exited with an error instead of a rocket.
    """
    from rocketopt.optimisation.driver import optimise_for_motor

    outcome = optimise_for_motor(
        "C6-5", population=8, generations=2, seed=3, verbose=False
    )
    assert outcome.flight.apogee > 0.0
    assert outcome.structure.passes


def test_boundary_layer_grows_along_the_body(reference_rocket) -> None:
    """Momentum thickness increases from the tip towards the tail."""
    solution = solve_potential_flow(reference_rocket, velocity=100.0)
    layer = solve_boundary_layer(solution)

    assert layer.momentum_thickness[-1] > layer.momentum_thickness[1]
    assert np.all(layer.shape_factor >= 1.0)
    assert np.all(layer.momentum_thickness >= 0.0)
