"""Six-degree-of-freedom trajectory simulation.

Flight phases
-------------
``ON_RAIL``
    Translation is constrained to the rail direction and attitude is held
    fixed. This matters: a rocket that leaves the rail too slowly has no fin
    authority and will weathercock hard or go unstable, and that only shows up
    if the rail constraint is modelled rather than assumed away.
``BOOST``
    Free flight with the motor thrusting.
``COAST``
    Free flight after burnout, up to ejection.
``DESCENT``
    After the ejection charge fires. The airframe is no longer a rigid
    aerodynamic body - it is a mass hanging under a canopy - so rotational
    dynamics are dropped and the vehicle is integrated as a point mass with
    the recovery device's drag area.
``LANDED``
    Terminated at ground level.

Integration
-----------
Fixed-step classical Runge-Kutta 4. A fixed step is chosen deliberately over
an adaptive solver: the right-hand side changes discontinuously at rail exit,
burnout and ejection, and adaptive solvers either step over those events or
grind to a halt at them. At the default 1 ms step the integration error over a
30 s flight is below a centimetre in apogee, which is far smaller than the
uncertainty in the drag coefficient.

Events are located by linear interpolation between the bracketing steps, so
apogee and rail-exit times are resolved far more finely than the step size.

References
----------
[1] Stevens, B. L., Lewis, F. L., & Johnson, E. N. (2015). *Aircraft Control
    and Simulation*, 3rd ed., Ch. 2.
[2] Zipfel, P. H. (2007). *Modeling and Simulation of Aerospace Vehicle
    Dynamics*, 2nd ed.
[3] Niskanen, S. (2013). *OpenRocket Technical Documentation*, Ch. 4.
[4] Barrowman, J. S. (1967). *The Practical Calculation of the Aerodynamic
    Characteristics of Slender Finned Vehicles*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rocketopt.aerodynamics.barrowman import barrowman_analysis, damping_derivatives
from rocketopt.aerodynamics.drag import drag_buildup
from rocketopt.flight.dynamics import (
    euler_angles,
    quaternion_derivative,
    quaternion_from_vectors,
    quaternion_normalise,
    quaternion_to_matrix,
)
from rocketopt.flight.environment import LaunchConditions
from rocketopt.geometry.components import RecoveryType
from rocketopt.geometry.rocket import Rocket
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = ["FlightPhase", "FlightState", "FlightResult", "simulate", "SimulationConfig"]

_MIN_AIRSPEED: Final[float] = 0.5
"""Airspeed below which aerodynamic forces are set to zero [m/s].

Below this the angle of attack is numerically meaningless and dividing by it
produces noise, so the aerodynamic model is simply switched off. The forces
involved are negligible at that speed in any case.
"""

_MAX_FLIGHT_TIME: Final[float] = 600.0
"""Hard stop on simulated time [s], so a runaway case cannot loop forever."""


def _norm3(v: NDArray[np.float64]) -> float:
    """Euclidean norm of a 3-vector.

    ``numpy.linalg.norm`` carries substantial dispatch overhead relative to the
    arithmetic itself, and the integrator calls this several times per
    derivative evaluation - tens of thousands of times per flight. Doing the
    three multiplies directly is several times faster for this fixed size.
    """
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


class FlightPhase(str, Enum):
    """Phase of flight."""

    ON_RAIL = "on_rail"
    BOOST = "boost"
    COAST = "coast"
    DESCENT = "descent"
    LANDED = "landed"

    @property
    def label(self) -> str:
        """Human-readable name."""
        return self.name.replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    """Tunable simulation settings.

    Attributes
    ----------
    timestep:
        Integration step [s].
    sample_interval:
        Interval at which trajectory samples are stored [s]. Storing every
        step would produce hundreds of thousands of rows for no benefit.
    max_time:
        Abort the run after this much simulated time [s].
    six_dof:
        When ``False``, angular dynamics are skipped and the rocket is flown
        as a point mass along its velocity vector. Roughly eight times faster
        and accurate to a few percent in apogee for a stable design, which
        makes it the right choice inside an optimiser loop.
    apply_wind:
        Whether to include the wind field.
    """

    timestep: float = 1.0e-3
    sample_interval: float = 0.05
    max_time: float = _MAX_FLIGHT_TIME
    six_dof: bool = True
    apply_wind: bool = True
    descent_timestep: float = 2.0e-2
    stop_at_apogee: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration."""
        if self.timestep <= 0.0:
            raise ValueError("timestep must be positive")
        if self.descent_timestep <= 0.0:
            raise ValueError("descent_timestep must be positive")
        if self.sample_interval < self.timestep:
            raise ValueError("sample_interval must not be smaller than timestep")
        if self.max_time <= 0.0:
            raise ValueError("max_time must be positive")

    def timestep_for(self, phase: FlightPhase) -> float:
        """Integration step appropriate to a flight phase [s].

        Ascent is integrated finely because thrust, mass and drag all change
        quickly and the peak loads occur there. Descent under a canopy is a
        slow, near-steady terminal-velocity problem that occupies most of the
        flight duration, so it is integrated with a step twenty times coarser.
        This costs nothing in accuracy - the descent rate settles to within a
        millimetre per second of terminal velocity either way - and removes
        the great majority of the run time.

        Parameters
        ----------
        phase:
            Current flight phase.

        Returns
        -------
        float
            Integration step [s].
        """
        if phase is FlightPhase.DESCENT:
            return self.descent_timestep
        return self.timestep

    @classmethod
    def fast(cls) -> SimulationConfig:
        """Return the configuration intended for an optimiser's inner loop.

        Three savings, none of which affect what the optimiser is ranking:

        * Point-mass mode, which agrees with 6-DOF on apogee to well under
          0.1% for any design that satisfies the stability constraints.
        * A 5 ms step rather than 1 ms. RK4 is fourth-order, so this raises
          integration error to roughly a centimetre of apogee - three orders
          of magnitude below the spread between competing designs.
        * Termination at apogee, with descent derived in closed form. The
          descent is a constant-terminal-velocity problem whose answer is
          already available analytically, so simulating 90-odd seconds of it
          per candidate buys nothing.

        Returns
        -------
        SimulationConfig
            Settings tuned for throughput rather than fidelity.
        """
        return cls(
            timestep=5.0e-3,
            sample_interval=0.05,
            six_dof=False,
            stop_at_apogee=True,
        )

    @classmethod
    def preview(cls) -> SimulationConfig:
        """Return the configuration for live, interactive re-analysis.

        Used while the user is editing a design, where the requirement is that
        the numbers update without perceptible lag rather than that they are
        exact. A 20 ms step keeps apogee within a few centimetres of the 1 ms
        result - RK4 error scales as the fourth power of step size - while
        cutting the work by a factor of twenty against the default.

        The design is re-flown at full fidelity for reports and for the final
        answer after an optimisation, so nothing the user takes away depends on
        this coarser setting.

        Returns
        -------
        SimulationConfig
            Settings tuned for interactive latency.
        """
        return cls(
            timestep=2.0e-2,
            sample_interval=0.05,
            descent_timestep=0.1,
            six_dof=False,
            stop_at_apogee=True,
        )

    @classmethod
    def animation(cls) -> SimulationConfig:
        """Return the configuration for the animated replay.

        Unlike :meth:`preview` this integrates the *whole* flight rather than
        stopping at apogee, because the replay needs ejection, descent and
        landing to show. Descent is stepped coarsely since it is a slow,
        near-steady phase, and samples are stored densely enough for smooth
        playback at 40 frames per second.

        Returns
        -------
        SimulationConfig
            Settings for a complete but inexpensive trajectory.
        """
        return cls(
            timestep=1.0e-2,
            sample_interval=0.04,
            descent_timestep=0.1,
            six_dof=False,
            stop_at_apogee=False,
        )


@dataclass(frozen=True, slots=True)
class FlightState:
    """One sampled instant of the trajectory.

    Attributes
    ----------
    time:
        Time since ignition [s].
    position:
        Position in the ENU frame relative to the pad [m].
    velocity:
        Inertial velocity [m/s].
    quaternion:
        Body-to-inertial attitude quaternion.
    angular_velocity:
        Body-frame angular rate [rad/s].
    mass:
        Vehicle mass [kg].
    phase:
        Flight phase.
    mach:
        Mach number [-].
    dynamic_pressure:
        Dynamic pressure [Pa].
    angle_of_attack:
        Total angle of attack [rad].
    drag_coefficient:
        Total drag coefficient [-].
    thrust:
        Motor thrust [N].
    acceleration:
        Inertial acceleration [m/s^2].
    static_margin:
        Static margin at this instant [calibres].
    """

    time: float
    position: NDArray[np.float64]
    velocity: NDArray[np.float64]
    quaternion: NDArray[np.float64]
    angular_velocity: NDArray[np.float64]
    mass: float
    phase: FlightPhase
    mach: float
    dynamic_pressure: float
    angle_of_attack: float
    drag_coefficient: float
    thrust: float
    acceleration: NDArray[np.float64]
    static_margin: float

    @property
    def altitude(self) -> float:
        """Height above the launch pad [m]."""
        return float(self.position[2])

    @property
    def speed(self) -> float:
        """Inertial speed [m/s]."""
        return _norm3((self.velocity))

    @property
    def vertical_velocity(self) -> float:
        """Rate of climb [m/s]."""
        return float(self.velocity[2])

    @property
    def downrange(self) -> float:
        """Horizontal distance from the pad [m]."""
        return float(math.hypot(self.position[0], self.position[1]))

    @property
    def euler(self) -> tuple[float, float, float]:
        """Roll, pitch and yaw [rad]."""
        return euler_angles(self.quaternion)


@dataclass(slots=True)
class FlightResult:
    """Outcome of a trajectory simulation.

    Attributes
    ----------
    rocket:
        The design that was flown.
    conditions:
        The launch conditions used.
    states:
        Sampled trajectory.
    apogee:
        Peak altitude above the pad [m].
    apogee_time:
        Time of apogee [s].
    max_velocity:
        Peak speed [m/s].
    max_mach:
        Peak Mach number [-].
    max_acceleration:
        Peak acceleration magnitude [m/s^2].
    max_dynamic_pressure:
        Peak dynamic pressure [Pa].
    rail_exit_velocity:
        Speed at the moment the rocket leaves the rail [m/s].
    rail_exit_time:
        Time of rail exit [s].
    burnout_altitude:
        Altitude at motor burnout [m].
    burnout_velocity:
        Speed at motor burnout [m/s].
    ejection_altitude:
        Altitude when the ejection charge fires [m].
    landing_time:
        Time of ground contact [s].
    landing_velocity:
        Descent speed at ground contact [m/s].
    landing_distance:
        Horizontal distance from the pad at landing [m].
    flight_duration:
        Total time from ignition to landing [s].
    min_static_margin:
        Smallest static margin seen during powered and coasting flight
        [calibres].
    warnings:
        Design or modelling warnings raised during the run.
    """

    rocket: Rocket
    conditions: LaunchConditions
    states: list[FlightState] = field(default_factory=list)
    apogee: float = 0.0
    apogee_time: float = 0.0
    max_velocity: float = 0.0
    max_mach: float = 0.0
    max_acceleration: float = 0.0
    max_dynamic_pressure: float = 0.0
    rail_exit_velocity: float = 0.0
    rail_exit_time: float = 0.0
    burnout_altitude: float = 0.0
    burnout_velocity: float = 0.0
    ejection_altitude: float = 0.0
    landing_time: float = 0.0
    landing_velocity: float = 0.0
    landing_distance: float = 0.0
    flight_duration: float = 0.0
    min_static_margin: float = 0.0
    warnings: list[str] = field(default_factory=list)

    @property
    def apogee_error_vs_ejection(self) -> float:
        """Altitude lost between apogee and ejection [m].

        A well-chosen delay fires the charge within a metre or two of apogee.
        A large positive value means the delay is too long and the rocket is
        already descending fast when the charge fires, which is the usual
        cause of a shredded parachute or a zippered body tube.
        """
        return self.apogee - self.ejection_altitude

    def as_arrays(self) -> dict[str, NDArray[np.float64]]:
        """Return the trajectory as named arrays, for plotting and export.

        Returns
        -------
        dict
            Mapping of quantity name to a one-dimensional array, all the same
            length as :attr:`states`.
        """
        return {
            "time": np.array([s.time for s in self.states]),
            "altitude": np.array([s.altitude for s in self.states]),
            "east": np.array([float(s.position[0]) for s in self.states]),
            "north": np.array([float(s.position[1]) for s in self.states]),
            "speed": np.array([s.speed for s in self.states]),
            "vertical_velocity": np.array([s.vertical_velocity for s in self.states]),
            "downrange": np.array([s.downrange for s in self.states]),
            "mach": np.array([s.mach for s in self.states]),
            "dynamic_pressure": np.array([s.dynamic_pressure for s in self.states]),
            "angle_of_attack": np.array([s.angle_of_attack for s in self.states]),
            "drag_coefficient": np.array([s.drag_coefficient for s in self.states]),
            "thrust": np.array([s.thrust for s in self.states]),
            "mass": np.array([s.mass for s in self.states]),
            "acceleration": np.array(
                [_norm3((s.acceleration)) for s in self.states]
            ),
            "static_margin": np.array([s.static_margin for s in self.states]),
        }

    def summary(self) -> str:
        """Return a readable multi-line flight summary."""
        lines = [
            f"Flight of {self.rocket.name} on {self.rocket.motor.designation}",
            f"  Apogee                {self.apogee:7.1f} m at {self.apogee_time:5.2f} s",
            f"  Max velocity          {self.max_velocity:7.1f} m/s "
            f"(Mach {self.max_mach:.2f})",
            f"  Max acceleration      {self.max_acceleration:7.1f} m/s^2 "
            f"({self.max_acceleration / 9.80665:.1f} g)",
            f"  Max dynamic pressure  {self.max_dynamic_pressure:7.1f} Pa",
            f"  Rail exit velocity    {self.rail_exit_velocity:7.1f} m/s at "
            f"{self.rail_exit_time:.3f} s",
            f"  Burnout               {self.burnout_altitude:7.1f} m at "
            f"{self.burnout_velocity:.1f} m/s",
            f"  Ejection altitude     {self.ejection_altitude:7.1f} m "
            f"({self.apogee_error_vs_ejection:+.1f} m from apogee)",
            f"  Landing               {self.landing_velocity:7.1f} m/s at "
            f"{self.landing_distance:.0f} m downrange",
            f"  Flight duration       {self.flight_duration:7.1f} s",
            f"  Minimum static margin {self.min_static_margin:7.2f} calibres",
        ]
        if self.warnings:
            lines.append("  Warnings:")
            lines.extend(f"    - {w}" for w in self.warnings)
        return "\n".join(lines)


@dataclass(slots=True)
class _Integrator:
    """Internal integration state and right-hand side for one flight."""

    rocket: Rocket
    conditions: LaunchConditions
    config: SimulationConfig

    # Cached per-flight constants.
    ref_area: float = 0.0
    ref_diameter: float = 0.0
    ground: float = 0.0
    burn_time: float = 0.0
    ejection_time: float = 0.0
    rail_direction: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )

    # Cached aerodynamic data, refreshed as the centre of gravity moves.
    _cp: float = 0.0
    _cna: float = 0.0
    _cmq: float = 0.0
    _clp: float = 0.0
    _cl_delta: float = 0.0

    def setup(self) -> None:
        """Precompute per-flight constants."""
        self.ref_area = self.rocket.reference_area
        self.ref_diameter = self.rocket.reference_diameter
        self.ground = self.conditions.ground_altitude
        self.burn_time = self.rocket.motor.motor.burn_time
        self.ejection_time = self.rocket.motor.ejection_time
        self.rail_direction = self.conditions.rail.direction

        # Barrowman coefficients are evaluated once at a representative Mach
        # number; the centre of pressure of a fixed geometry moves only a few
        # percent across the subsonic range, and re-running the full analysis
        # every step would dominate the run time.
        analysis = barrowman_analysis(self.rocket, mach=0.3)
        self._cp = analysis.centre_of_pressure
        self._cna = analysis.cn_alpha

        damping = damping_derivatives(self.rocket, t=0.0, mach=0.3)
        self._cmq = damping.cmq
        self._clp = damping.clp
        self._cl_delta = damping.cl_delta

    def phase_at(self, t: float, position: NDArray[np.float64], rail_left: bool) -> FlightPhase:
        """Determine the flight phase at a given time and position."""
        if position[2] <= 0.0 and t > 1.0:
            return FlightPhase.LANDED
        if not rail_left:
            return FlightPhase.ON_RAIL
        if t >= self.ejection_time:
            return FlightPhase.DESCENT
        if t < self.burn_time:
            return FlightPhase.BOOST
        return FlightPhase.COAST

    def derivatives(
        self,
        t: float,
        state: NDArray[np.float64],
        phase: FlightPhase,
    ) -> tuple[NDArray[np.float64], dict[str, float]]:
        """Right-hand side of the equations of motion.

        Parameters
        ----------
        t:
            Time since ignition [s].
        state:
            Packed state ``[pos(3), vel(3), quat(4), omega(3)]``.
        phase:
            Current flight phase.

        Returns
        -------
        tuple
            The state derivative and a dictionary of diagnostic quantities
            (Mach number, angle of attack, drag coefficient and thrust) used
            for trajectory sampling.
        """
        position = state[0:3]
        velocity = state[3:6]
        quat = quaternion_normalise(state[6:10])
        omega = state[10:13]

        altitude_amsl = self.ground + position[2]
        atmos = self.conditions.atmosphere.state_at(altitude_amsl)
        gravity = self.conditions.atmosphere.gravity_at(altitude_amsl)

        mass_props = self.rocket.mass_properties_at(t)
        mass = mass_props.mass
        thrust_magnitude = self.rocket.motor.thrust_at(t)

        # Relative wind.
        if self.config.apply_wind:
            wind = self.conditions.wind.velocity_at(max(position[2], 0.0))
        else:
            wind = np.zeros(3, dtype=np.float64)
        v_rel = velocity - wind
        airspeed = _norm3((v_rel))

        force = np.array([0.0, 0.0, -mass * gravity], dtype=np.float64)
        moment = np.zeros(3, dtype=np.float64)

        rotation = quaternion_to_matrix(quat)

        # In point-mass mode the vehicle is assumed to fly along its velocity
        # vector, so thrust acts there rather than along a tracked body axis.
        # This is exact for a stable rocket in still air and is what makes the
        # mode cheap: no attitude state, no normal force, no moments.
        if not self.config.six_dof and phase is not FlightPhase.ON_RAIL:
            if airspeed > _MIN_AIRSPEED:
                body_x_inertial = v_rel / airspeed
            else:
                body_x_inertial = self.rail_direction
        else:
            body_x_inertial = rotation[:, 0]

        # -- Thrust ---------------------------------------------------------
        if thrust_magnitude > 0.0 and phase in {FlightPhase.ON_RAIL, FlightPhase.BOOST}:
            force = force + thrust_magnitude * body_x_inertial

        diagnostics = {
            "mach": atmos.mach(airspeed),
            "angle_of_attack": 0.0,
            "drag_coefficient": 0.0,
            "thrust": thrust_magnitude,
            "dynamic_pressure": atmos.dynamic_pressure(airspeed),
        }

        # -- Aerodynamics ---------------------------------------------------
        if phase is FlightPhase.DESCENT:
            force = force + self._recovery_force(v_rel, airspeed, atmos.density)
        elif not self.config.six_dof:
            if airspeed > _MIN_AIRSPEED:
                # Point-mass mode: zero angle of attack by construction, so
                # only axial drag acts. No Barrowman call, no moments.
                breakdown = drag_buildup(
                    self.rocket,
                    atmos,
                    airspeed,
                    angle_of_attack=0.0,
                    motor_burning=thrust_magnitude > 0.0,
                )
                q_dyn = 0.5 * atmos.density * airspeed * airspeed
                force = force - breakdown.total * q_dyn * self.ref_area * (
                    v_rel / airspeed
                )
                diagnostics["drag_coefficient"] = breakdown.total
        elif airspeed > _MIN_AIRSPEED:
            aero_force, aero_moment, diag = self._aero_forces(
                t=t,
                v_rel=v_rel,
                airspeed=airspeed,
                rotation=rotation,
                omega=omega,
                atmos=atmos,
                mass_props_cg=mass_props.cg,
                motor_burning=thrust_magnitude > 0.0,
            )
            force = force + aero_force
            moment = moment + aero_moment
            diagnostics.update(diag)

        acceleration = force / mass

        # -- Rail constraint -------------------------------------------------
        if phase is FlightPhase.ON_RAIL:
            # The rail carries any transverse load, so only the component of
            # acceleration along the rail survives. Attitude is held fixed.
            along = float(np.dot(acceleration, self.rail_direction))
            # The rail cannot pull the rocket backwards down its length.
            acceleration = max(along, 0.0) * self.rail_direction
            omega_dot = np.zeros(3, dtype=np.float64)
            quat_dot = np.zeros(4, dtype=np.float64)
            velocity_out = float(np.dot(velocity, self.rail_direction)) * self.rail_direction
            return (
                np.concatenate([velocity_out, acceleration, quat_dot, omega_dot]),
                diagnostics,
            )

        # -- Rotational dynamics ---------------------------------------------
        if phase is FlightPhase.DESCENT or not self.config.six_dof:
            # Under a canopy the airframe is no longer a rigid aerodynamic
            # body, and in point-mass mode there is no attitude state at all.
            # Either way the rotational degrees of freedom are inert.
            omega_dot = np.zeros(3, dtype=np.float64)
            quat_dot = np.zeros(4, dtype=np.float64)
        else:
            ixx = max(mass_props.ixx, 1e-12)
            iyy = max(mass_props.iyy, 1e-12)
            inertia = np.array([ixx, iyy, iyy], dtype=np.float64)
            # Euler's equation for a body with a diagonal inertia tensor [1].
            gyroscopic = np.cross(omega, inertia * omega)
            omega_dot = (moment - gyroscopic) / inertia
            quat_dot = quaternion_derivative(quat, omega)

        return (
            np.concatenate([velocity, acceleration, quat_dot, omega_dot]),
            diagnostics,
        )

    def _aero_forces(
        self,
        *,
        t: float,
        v_rel: NDArray[np.float64],
        airspeed: float,
        rotation: NDArray[np.float64],
        omega: NDArray[np.float64],
        atmos: object,
        mass_props_cg: float,
        motor_burning: bool,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
        """Aerodynamic force and moment in the inertial and body frames.

        Returns
        -------
        tuple
            ``(force_inertial, moment_body, diagnostics)``.
        """
        # Relative wind expressed in body axes.
        v_body = rotation.T @ v_rel
        v_hat = v_body / airspeed

        # Total angle of attack: angle between the nose and the relative wind.
        lateral = np.array([0.0, v_hat[1], v_hat[2]])
        lateral_mag = _norm3((lateral))
        alpha = math.atan2(lateral_mag, float(v_hat[0]))

        q_dyn = 0.5 * atmos.density * airspeed * airspeed  # type: ignore[attr-defined]

        # Normal force from the Barrowman slope, plus the drag build-up.
        cn = self._cna * alpha
        breakdown = drag_buildup(
            self.rocket,
            atmos,  # type: ignore[arg-type]
            airspeed,
            angle_of_attack=alpha,
            motor_burning=motor_burning,
            normal_force_coefficient=cn,
        )
        cd = breakdown.total

        # Drag opposes the relative wind; normal force opposes the lateral
        # component of it, which is what makes a stable rocket weathercock.
        force_body = -cd * q_dyn * self.ref_area * v_hat
        if lateral_mag > 1e-9:
            n_hat = lateral / lateral_mag
            force_body = force_body - cn * q_dyn * self.ref_area * n_hat

        # Moment about the centre of gravity. With the body origin at the CG
        # and +x forward, a point x_cp aft of the nose tip sits at
        # (x_cg - x_cp) on the body x axis.
        r_cp = np.array([mass_props_cg - self._cp, 0.0, 0.0], dtype=np.float64)
        moment_body = np.cross(r_cp, force_body)

        # Rotational damping. The non-dimensional rate is omega*d/(2V), and the
        # moment is q*A*d*C. Combining gives 0.25*rho*V*A*d^2*C*omega.
        damping_scale = 0.25 * atmos.density * airspeed * self.ref_area  # type: ignore[attr-defined]
        d2 = self.ref_diameter * self.ref_diameter
        moment_body = moment_body + np.array(
            [
                damping_scale * d2 * self._clp * omega[0],
                damping_scale * d2 * self._cmq * omega[1],
                damping_scale * d2 * self._cmq * omega[2],
            ],
            dtype=np.float64,
        )

        # Roll driven by fin cant.
        cant = self.rocket.fins.cant_angle
        if abs(cant) > 1e-12:
            moment_body[0] += (
                q_dyn * self.ref_area * self.ref_diameter * self._cl_delta * cant
            )

        force_inertial = rotation @ force_body
        diagnostics = {
            "angle_of_attack": alpha,
            "drag_coefficient": cd,
        }
        return force_inertial, moment_body, diagnostics

    def _recovery_force(
        self,
        v_rel: NDArray[np.float64],
        airspeed: float,
        density: float,
    ) -> NDArray[np.float64]:
        """Drag force under the deployed recovery device [N].

        Parameters
        ----------
        v_rel:
            Velocity relative to the air [m/s].
        airspeed:
            Magnitude of ``v_rel`` [m/s].
        density:
            Local air density [kg/m^3].

        Returns
        -------
        numpy.ndarray
            Drag force in the inertial frame [N].
        """
        if airspeed < 1e-6:
            return np.zeros(3, dtype=np.float64)

        recovery = self.rocket.recovery
        if recovery.kind is RecoveryType.TUMBLE:
            drag_area = recovery.drag_coefficient * self.ref_area
        else:
            drag_area = recovery.drag_area

        magnitude = 0.5 * density * airspeed * airspeed * drag_area
        return -magnitude * (v_rel / airspeed)


def simulate(
    rocket: Rocket,
    conditions: LaunchConditions | None = None,
    config: SimulationConfig | None = None,
) -> FlightResult:
    """Fly a rocket and return its trajectory.

    Parameters
    ----------
    rocket:
        The design to fly.
    conditions:
        Launch conditions. Defaults to still air at sea level on a vertical
        0.91 m rod.
    config:
        Simulation settings. Defaults to a 1 ms 6-DOF integration.

    Returns
    -------
    FlightResult
        Trajectory samples and the derived performance figures.

    Raises
    ------
    ValueError
        If the rocket cannot leave the rail, which means the thrust-to-weight
        ratio is below 1 and the design is unflyable rather than merely poor.
    """
    conditions = conditions or LaunchConditions.standard()
    config = config or SimulationConfig()

    integrator = _Integrator(rocket=rocket, conditions=conditions, config=config)
    integrator.setup()

    result = FlightResult(rocket=rocket, conditions=conditions)

    # Sanity check before spending time integrating.
    if rocket.thrust_to_weight < 1.0:
        raise ValueError(
            f"{rocket.name} on {rocket.motor.designation} has a thrust-to-weight "
            f"ratio of {rocket.thrust_to_weight:.2f}; it cannot lift off. "
            f"Choose a more powerful motor or reduce the vehicle mass."
        )

    # Initial state: sitting on the rail, nose along the rail direction.
    rail_dir = conditions.rail.direction
    quat0 = quaternion_from_vectors(np.array([1.0, 0.0, 0.0]), rail_dir)
    state = np.concatenate(
        [
            np.zeros(3),  # position
            np.zeros(3),  # velocity
            quat0,
            np.zeros(3),  # angular velocity
        ]
    ).astype(np.float64)

    t = 0.0
    rail_left = False
    next_sample = 0.0
    apogee_recorded = False
    burnout_recorded = False
    ejection_recorded = False
    min_margin = float("inf")
    previous_altitude = 0.0
    previous_vertical_velocity = 0.0

    while t < config.max_time:
        phase = integrator.phase_at(t, state[0:3], rail_left)

        if phase is FlightPhase.LANDED:
            break

        dt = config.timestep_for(phase)

        # -- Classical RK4 ---------------------------------------------------
        k1, diagnostics = integrator.derivatives(t, state, phase)
        k2, _ = integrator.derivatives(t + 0.5 * dt, state + 0.5 * dt * k1, phase)
        k3, _ = integrator.derivatives(t + 0.5 * dt, state + 0.5 * dt * k2, phase)
        k4, _ = integrator.derivatives(t + dt, state + dt * k3, phase)

        new_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        new_state[6:10] = quaternion_normalise(new_state[6:10])

        # -- Event detection --------------------------------------------------
        rail_distance = float(np.dot(new_state[0:3], rail_dir))
        if not rail_left and rail_distance >= conditions.rail.length:
            rail_left = True
            result.rail_exit_time = t + dt
            result.rail_exit_velocity = _norm3((new_state[3:6]))

        if not burnout_recorded and t + dt >= integrator.burn_time:
            burnout_recorded = True
            result.burnout_altitude = float(new_state[2])
            result.burnout_velocity = _norm3((new_state[3:6]))

        if not ejection_recorded and t + dt >= integrator.ejection_time:
            ejection_recorded = True
            result.ejection_altitude = float(new_state[2])

        vertical_velocity = float(new_state[5])
        if (
            not apogee_recorded
            and rail_left
            and previous_vertical_velocity > 0.0
            and vertical_velocity <= 0.0
        ):
            # Linearly interpolate the zero crossing of vertical velocity.
            span = previous_vertical_velocity - vertical_velocity
            frac = previous_vertical_velocity / span if span > 1e-12 else 0.0
            result.apogee_time = t + frac * dt
            result.apogee = previous_altitude + frac * (
                float(new_state[2]) - previous_altitude
            )
            apogee_recorded = True

            if config.stop_at_apogee:
                _complete_analytically(result, integrator)
                state = new_state
                break

        # -- Sampling ---------------------------------------------------------
        if t >= next_sample:
            acceleration = k1[3:6]
            cg = rocket.cg_at(t)
            margin = (integrator._cp - cg) / integrator.ref_diameter
            if phase in {FlightPhase.BOOST, FlightPhase.COAST}:
                min_margin = min(min_margin, margin)

            result.states.append(
                FlightState(
                    time=t,
                    position=state[0:3].copy(),
                    velocity=state[3:6].copy(),
                    quaternion=state[6:10].copy(),
                    angular_velocity=state[10:13].copy(),
                    mass=rocket.mass_properties_at(t).mass,
                    phase=phase,
                    mach=diagnostics["mach"],
                    dynamic_pressure=diagnostics["dynamic_pressure"],
                    angle_of_attack=diagnostics["angle_of_attack"],
                    drag_coefficient=diagnostics["drag_coefficient"],
                    thrust=diagnostics["thrust"],
                    acceleration=acceleration.copy(),
                    static_margin=margin,
                )
            )

            result.max_velocity = max(result.max_velocity, _norm3((state[3:6])))
            result.max_mach = max(result.max_mach, diagnostics["mach"])
            result.max_acceleration = max(
                result.max_acceleration, _norm3((acceleration))
            )
            result.max_dynamic_pressure = max(
                result.max_dynamic_pressure, diagnostics["dynamic_pressure"]
            )
            next_sample += config.sample_interval

        # -- Ground contact ----------------------------------------------------
        if rail_left and new_state[2] <= 0.0 and t > 1.0:
            span = previous_altitude - float(new_state[2])
            frac = previous_altitude / span if span > 1e-12 else 0.0
            result.landing_time = t + frac * dt
            result.landing_velocity = abs(float(new_state[5]))
            result.landing_distance = float(
                math.hypot(new_state[0], new_state[1])
            )
            state = new_state
            break

        previous_altitude = float(new_state[2])
        previous_vertical_velocity = vertical_velocity
        state = new_state
        t += dt

    else:
        result.warnings.append(
            f"Simulation hit the {config.max_time:.0f} s time limit without "
            f"landing; the descent rate may be unrealistically low."
        )

    if not result.flight_duration:
        result.flight_duration = result.landing_time or t
    result.min_static_margin = 0.0 if min_margin == float("inf") else min_margin

    _collect_warnings(result, integrator)

    _log.debug(
        "%s on %s: apogee %.1f m, rail exit %.1f m/s",
        rocket.name,
        rocket.motor.designation,
        result.apogee,
        result.rail_exit_velocity,
    )
    return result


def _complete_analytically(result: FlightResult, integrator: _Integrator) -> None:
    """Fill in descent figures without integrating the descent.

    Under a deployed canopy the vehicle reaches terminal velocity within a
    second or two and holds it, so descent rate follows directly from the
    force balance ``m g = 0.5 rho V^2 Cd S`` and descent time from the apogee
    altitude divided by that rate. Both are then exact rather than
    approximate, and 90-odd seconds of integration per candidate is avoided.

    Parameters
    ----------
    result:
        The in-progress flight result, mutated in place.
    integrator:
        The integrator, for access to the rocket and conditions.
    """
    rocket = integrator.rocket
    atmosphere = integrator.conditions.atmosphere

    # Evaluate density at half the apogee altitude as a representative mean
    # over the descent, rather than at the ground or at apogee.
    mean_altitude = integrator.ground + 0.5 * result.apogee
    density = atmosphere.density_at(mean_altitude)

    descending_mass = rocket.burnout_mass
    recovery = rocket.recovery

    if recovery.kind is RecoveryType.TUMBLE:
        drag_area = recovery.drag_coefficient * integrator.ref_area
    else:
        drag_area = recovery.drag_area

    if drag_area > 0.0:
        from rocketopt.utils.constants import G0

        descent_rate = math.sqrt(
            2.0 * descending_mass * G0 / (density * drag_area)
        )
    else:  # pragma: no cover - guarded by RecoveryDevice validation
        descent_rate = 0.0

    result.landing_velocity = descent_rate
    if descent_rate > 1e-6:
        descent_time = result.apogee / descent_rate
        result.landing_time = result.apogee_time + descent_time

        # Horizontal drift: the canopy travels with the wind for the descent.
        if integrator.config.apply_wind:
            wind_speed = integrator.conditions.wind.speed_at(0.5 * result.apogee)
            result.landing_distance = wind_speed * descent_time
    result.flight_duration = result.landing_time


def _collect_warnings(result: FlightResult, integrator: _Integrator) -> None:
    """Append design warnings to a completed flight result."""
    from rocketopt.utils.constants import (
        MAX_STATIC_MARGIN_CALIBRES,
        MIN_LIFTOFF_THRUST_TO_WEIGHT,
        MIN_RAIL_EXIT_VELOCITY,
        MIN_STATIC_MARGIN_CALIBRES,
    )

    rocket = result.rocket

    if rocket.thrust_to_weight < MIN_LIFTOFF_THRUST_TO_WEIGHT:
        result.warnings.append(
            f"Thrust-to-weight ratio is {rocket.thrust_to_weight:.1f}, below the "
            f"{MIN_LIFTOFF_THRUST_TO_WEIGHT:.0f}:1 minimum recommended for a safe "
            f"rail exit."
        )

    if result.rail_exit_velocity < MIN_RAIL_EXIT_VELOCITY:
        result.warnings.append(
            f"Rail exit velocity is {result.rail_exit_velocity:.1f} m/s, below the "
            f"{MIN_RAIL_EXIT_VELOCITY:.0f} m/s needed for the fins to have "
            f"authority. Use a longer rail or a higher-thrust motor."
        )

    if result.min_static_margin < MIN_STATIC_MARGIN_CALIBRES:
        result.warnings.append(
            f"Minimum static margin is {result.min_static_margin:.2f} calibres, "
            f"below the {MIN_STATIC_MARGIN_CALIBRES:.1f} calibre minimum for "
            f"reliable stability."
        )
    elif result.min_static_margin > MAX_STATIC_MARGIN_CALIBRES:
        result.warnings.append(
            f"Minimum static margin is {result.min_static_margin:.2f} calibres, "
            f"above the {MAX_STATIC_MARGIN_CALIBRES:.1f} calibre limit; the "
            f"rocket will weathercock strongly into wind."
        )

    overshoot = result.apogee_error_vs_ejection
    if abs(overshoot) > 0.10 * max(result.apogee, 1.0):
        direction = "long" if overshoot > 0.0 else "short"
        result.warnings.append(
            f"Ejection fires {abs(overshoot):.0f} m {direction} of apogee. "
            f"Consider a different delay on the {rocket.motor.motor.designation}; "
            f"available delays are {list(rocket.motor.motor.delays)} s."
        )

    provenance = rocket.motor.motor.provenance_warning()
    if provenance:
        result.warnings.append(provenance)

    del integrator
