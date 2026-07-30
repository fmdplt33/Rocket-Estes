"""The rocket assembly: component stack, geometry and mass properties.

A :class:`Rocket` is an ordered stack of components. The nose tip is the
origin of the body axis and ``x`` increases aft. Component axial positions are
derived from the stack rather than stored, so a design can never contain a
gap or an overlap.

Aerodynamic coefficients are deliberately *not* computed here. Geometry and
mass live in this module; :mod:`rocketopt.aerodynamics` consumes a
:class:`Rocket` and returns coefficients. That split keeps the geometry layer
free of any Mach- or altitude-dependent state.

References
----------
[1] Barrowman, J. S. (1967). *The Practical Calculation of the Aerodynamic
    Characteristics of Slender Finned Vehicles*.
[2] Niskanen, S. (2013). *OpenRocket Technical Documentation*.
[3] National Association of Rocketry. *Model Rocket Safety Code*, 2023 rev.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING

from rocketopt.geometry.components import (
    BodyTube,
    FinSet,
    LaunchLug,
    MotorMount,
    RecoveryDevice,
    Transition,
)
from rocketopt.geometry.mass_properties import (
    MassProperties,
    combine,
    fin_set_properties,
    launch_lug_properties,
    motor_mount_properties,
    nose_cone_properties,
    point_mass,
    transition_properties,
    tube_properties,
)
from rocketopt.geometry.nose_cones import NoseCone
from rocketopt.structures.materials import SurfaceFinish
from rocketopt.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from rocketopt.propulsion.motor import MotorConfiguration

_log = get_logger(__name__)

__all__ = ["Rocket", "PlacedSection", "build_rocket"]

Section = BodyTube | Transition


@dataclass(frozen=True, slots=True)
class PlacedSection:
    """A body section together with its resolved axial station.

    Attributes
    ----------
    section:
        The component.
    position:
        Axial station of its forward face, aft of the nose tip [m].
    """

    section: Section
    position: float

    @property
    def aft_position(self) -> float:
        """Axial station of the section's aft face [m]."""
        return self.position + self.section.length


# Note: no slots. The structural mass properties and wetted areas are
# time-invariant but expensive to compute (they integrate the nose profile and
# fin planform over hundreds of stations), and the flight simulator asks for
# them on every derivative evaluation - roughly 240,000 times per flight.
# They are cached with functools.cached_property, which requires an instance
# __dict__. The class remains frozen, so a cached value can never go stale.
@dataclass(frozen=True)
class Rocket:
    """A complete model rocket design.

    Attributes
    ----------
    nose:
        Nose cone, always at ``x = 0``.
    sections:
        Body sections in order from front to back. Must contain at least one
        :class:`~rocketopt.geometry.components.BodyTube`.
    fins:
        The fin set.
    motor_mount:
        Motor tube and centring rings.
    motor:
        The motor and ejection delay flown.
    recovery:
        Recovery device.
    launch_lug:
        Launch lug, or ``None`` for a rail-guided or tower-launched model.
    payload_mass:
        Mass of any payload carried [kg].
    payload_position:
        Axial station of the payload's centre of gravity [m]. Defaults to just
        aft of the nose cone when left at ``None``.
    nose_ballast_mass:
        Ballast added in the nose to move the centre of gravity forward [kg].
    surface_finish:
        External finish, which sets skin-friction roughness.
    name:
        Human-readable design name.
    """

    nose: NoseCone
    sections: tuple[PlacedSection, ...]
    fins: FinSet
    motor_mount: MotorMount
    motor: MotorConfiguration
    recovery: RecoveryDevice
    launch_lug: LaunchLug | None = None
    payload_mass: float = 0.0
    payload_position: float | None = None
    nose_ballast_mass: float = 0.0
    surface_finish: SurfaceFinish = SurfaceFinish.SMOOTH_PAINT
    name: str = "Unnamed"

    def __post_init__(self) -> None:
        """Validate the assembly."""
        if not self.sections:
            raise ValueError("a rocket needs at least one body section")
        if not any(isinstance(p.section, BodyTube) for p in self.sections):
            raise ValueError("a rocket needs at least one cylindrical body tube")
        if self.payload_mass < 0.0 or self.nose_ballast_mass < 0.0:
            raise ValueError("payload and ballast masses must not be negative")

        # The stack must be contiguous: each section starts where the last ended.
        expected = self.nose.length
        for placed in self.sections:
            if not math.isclose(placed.position, expected, abs_tol=1e-9):
                raise ValueError(
                    f"section stack is not contiguous: expected a section at "
                    f"x={expected:.6f} m but found one at x={placed.position:.6f} m"
                )
            expected = placed.aft_position

        # The motor must physically fit the mount, and the mount the airframe.
        if self.motor.motor.diameter > self.motor_mount.inner_diameter + 1e-6:
            raise ValueError(
                f"motor {self.motor.designation} is "
                f"{self.motor.motor.diameter * 1e3:.1f} mm diameter but the "
                f"motor mount bore is only "
                f"{self.motor_mount.inner_diameter * 1e3:.1f} mm"
            )
        if self.motor.motor.length > self.motor_mount.length + 1e-6:
            raise ValueError(
                f"motor {self.motor.designation} is "
                f"{self.motor.motor.length * 1e3:.0f} mm long but the motor "
                f"tube is only {self.motor_mount.length * 1e3:.0f} mm"
            )

    # -- Overall geometry ----------------------------------------------------

    @property
    def length(self) -> float:
        """Overall length from nose tip to the aft-most face [m]."""
        return self.sections[-1].aft_position

    @cached_property
    def reference_radius(self) -> float:
        """Radius defining the aerodynamic reference area [m].

        Barrowman's convention is the maximum body radius [1]; for a
        conventional model that is the main body tube.
        """
        radii = []
        for placed in self.sections:
            if isinstance(placed.section, BodyTube):
                radii.append(placed.section.outer_radius)
            else:
                radii.append(max(placed.section.fore_radius, placed.section.aft_radius))
        return max(radii)

    @property
    def reference_diameter(self) -> float:
        """Reference diameter, one body calibre [m]."""
        return 2.0 * self.reference_radius

    @property
    def reference_area(self) -> float:
        """Aerodynamic reference area [m^2]."""
        return math.pi * self.reference_radius**2

    @property
    def fineness_ratio(self) -> float:
        """Overall length divided by reference diameter [-]."""
        return self.length / self.reference_diameter

    @property
    def body_tubes(self) -> tuple[PlacedSection, ...]:
        """Every cylindrical section in the stack."""
        return tuple(p for p in self.sections if isinstance(p.section, BodyTube))

    @property
    def transitions(self) -> tuple[PlacedSection, ...]:
        """Every conical transition in the stack."""
        return tuple(p for p in self.sections if isinstance(p.section, Transition))

    @cached_property
    def wetted_area(self) -> float:
        """Total external area exposed to the flow [m^2]."""
        area = self.nose.wetted_area
        area += sum(p.section.wetted_area for p in self.sections)
        area += self.fins.wetted_area
        if self.launch_lug is not None:
            area += self.launch_lug.wetted_area
        return area

    @cached_property
    def body_planform_area(self) -> float:
        """Side-view projected area of the nose and body [m^2].

        Used by the viscous cross-flow term in the Barrowman analysis.
        """
        return self.nose.planform_area + sum(
            p.section.planform_area for p in self.sections
        )

    @property
    def base_area(self) -> float:
        """Area of the aft-facing base [m^2].

        The annulus around the motor is what actually sees base drag; the
        motor nozzle itself is not a base while the motor is burning.
        """
        aft = self.sections[-1].section
        radius = aft.aft_radius if isinstance(aft, Transition) else aft.outer_radius
        return math.pi * radius**2

    # -- Mass properties -----------------------------------------------------

    @property
    def _payload_station(self) -> float:
        """Resolved axial station of the payload centre of gravity [m]."""
        if self.payload_position is not None:
            return self.payload_position
        # Default: immediately aft of the nose shoulder, the usual bay location.
        return self.nose.length + self.nose.shoulder_length + 0.02

    @cached_property
    def _structural_parts(self) -> tuple[MassProperties, ...]:
        """Mass properties of every part except the motor.

        Cached: nothing here depends on time, and rebuilding it inside the
        simulation loop dominated the run time before this was memoised.
        """
        parts = [nose_cone_properties(self.nose)]

        for placed in self.sections:
            if isinstance(placed.section, BodyTube):
                parts.append(tube_properties(placed.section, placed.position))
            else:
                parts.append(transition_properties(placed.section, placed.position))

        parts.append(fin_set_properties(self.fins))
        parts.append(motor_mount_properties(self.motor_mount))

        if self.launch_lug is not None:
            parts.append(launch_lug_properties(self.launch_lug))

        if self.recovery.mass > 0.0:
            # The recovery device packs into the tube just aft of the nose.
            parts.append(
                point_mass(self.recovery.mass, self.nose.length + 0.03)
            )
        if self.payload_mass > 0.0:
            parts.append(point_mass(self.payload_mass, self._payload_station))
        if self.nose_ballast_mass > 0.0:
            # Ballast is packed into the very tip, where it is most effective.
            parts.append(point_mass(self.nose_ballast_mass, self.nose.length * 0.15))

        return tuple(parts)

    @cached_property
    def dry_mass_properties(self) -> MassProperties:
        """Mass properties with no motor installed."""
        return combine(list(self._structural_parts))

    @property
    def dry_mass(self) -> float:
        """Structural mass with no motor [kg]."""
        return self.dry_mass_properties.mass

    @property
    def motor_position(self) -> float:
        """Axial station of the motor's forward face [m].

        The motor is seated against the thrust ring at the aft end of the
        motor tube, so its forward face sits one motor length forward of the
        motor tube's aft end.
        """
        mount_aft = self.motor_mount.position + self.motor_mount.length
        return mount_aft - self.motor.motor.length

    def mass_properties_at(self, t: float) -> MassProperties:
        """Mass properties at time ``t`` after ignition.

        Parameters
        ----------
        t:
            Time since ignition [s]. Negative values are treated as zero.

        Returns
        -------
        MassProperties
            Assembly mass, centre of gravity and inertias, including the
            motor's remaining propellant.
        """
        motor = self.motor.motor
        motor_mass = motor.mass_at(max(t, 0.0))
        motor_cg = self.motor_position + motor.centre_of_mass_offset(max(t, 0.0))

        # The motor is a slender cylinder; include its own inertia so that the
        # roll and pitch moments are not underestimated at lift-off, when the
        # motor is a large fraction of total mass.
        r = motor.diameter / 2.0
        motor_props = MassProperties(
            mass=motor_mass,
            cg=motor_cg,
            ixx=motor_mass * r * r / 2.0,
            iyy=motor_mass * (3.0 * r * r + motor.length**2) / 12.0,
        )

        return combine([*self._structural_parts, motor_props])

    @property
    def loaded_mass_properties(self) -> MassProperties:
        """Mass properties at ignition, with a full motor."""
        return self.mass_properties_at(0.0)

    @property
    def burnout_mass_properties(self) -> MassProperties:
        """Mass properties at motor burnout."""
        return self.mass_properties_at(self.motor.motor.burn_time)

    @property
    def loaded_mass(self) -> float:
        """Lift-off mass including a full motor [kg]."""
        return self.loaded_mass_properties.mass

    @property
    def burnout_mass(self) -> float:
        """Mass at burnout [kg]."""
        return self.burnout_mass_properties.mass

    def cg_at(self, t: float) -> float:
        """Centre of gravity aft of the nose tip at time ``t`` [m]."""
        return self.mass_properties_at(t).cg

    # -- Performance indicators ---------------------------------------------

    @property
    def thrust_to_weight(self) -> float:
        """Average thrust divided by lift-off weight [-].

        The NAR Model Rocket Safety Code and common practice call for at least
        5:1 for a safe rod exit [3].
        """
        from rocketopt.utils.constants import G0

        return self.motor.motor.average_thrust / (self.loaded_mass * G0)

    @property
    def peak_thrust_to_weight(self) -> float:
        """Peak thrust divided by lift-off weight [-]."""
        from rocketopt.utils.constants import G0

        return self.motor.motor.peak_thrust / (self.loaded_mass * G0)

    @property
    def propellant_fraction(self) -> float:
        """Propellant mass divided by lift-off mass [-]."""
        return self.motor.motor.propellant_mass / self.loaded_mass

    @property
    def manufacturability(self) -> float:
        """Overall ease of construction, 0 to 1 [-].

        A weighted blend of the nose and fin scores plus a penalty for
        unusually many body sections. A design heuristic used as an
        optimisation objective, not a measured quantity.
        """
        section_penalty = 1.0 if len(self.sections) <= 2 else 0.85
        return float(
            min(
                1.0,
                (0.40 * self.nose.manufacturability
                 + 0.45 * self.fins.manufacturability
                 + 0.15)
                * section_penalty,
            )
        )

    def with_motor(self, motor: MotorConfiguration) -> Rocket:
        """Return a copy of this design flying a different motor.

        Parameters
        ----------
        motor:
            The replacement motor configuration.

        Returns
        -------
        Rocket
            A new design; this one is unchanged.
        """
        return replace(self, motor=motor)

    def with_ballast(self, ballast_mass: float) -> Rocket:
        """Return a copy with a different nose ballast mass.

        Parameters
        ----------
        ballast_mass:
            Ballast mass [kg].

        Returns
        -------
        Rocket
            A new design; this one is unchanged.
        """
        return replace(self, nose_ballast_mass=ballast_mass)

    def summary(self) -> str:
        """Return a short multi-line description for logs and the CLI."""
        loaded = self.loaded_mass_properties
        return (
            f"{self.name}\n"
            f"  Motor              {self.motor.designation}\n"
            f"  Length             {self.length * 1e3:.0f} mm\n"
            f"  Diameter           {self.reference_diameter * 1e3:.1f} mm\n"
            f"  Fineness ratio     {self.fineness_ratio:.1f}\n"
            f"  Nose               {self.nose.shape.label}, "
            f"{self.nose.length * 1e3:.0f} mm\n"
            f"  Fins               {self.fins.count} x {self.fins.material.display_name}, "
            f"{self.fins.thickness * 1e3:.2f} mm\n"
            f"  Dry mass           {self.dry_mass * 1e3:.1f} g\n"
            f"  Lift-off mass      {loaded.mass * 1e3:.1f} g\n"
            f"  CG at lift-off     {loaded.cg * 1e3:.0f} mm aft of tip\n"
            f"  Thrust/weight      {self.thrust_to_weight:.1f}"
        )


def build_rocket(
    *,
    nose: NoseCone,
    body_tube: BodyTube,
    fins: FinSet,
    motor: MotorConfiguration,
    recovery: RecoveryDevice,
    motor_mount: MotorMount | None = None,
    boat_tail: Transition | None = None,
    launch_lug: LaunchLug | None = None,
    fin_root_position: float | None = None,
    payload_mass: float = 0.0,
    nose_ballast_mass: float = 0.0,
    surface_finish: SurfaceFinish = SurfaceFinish.SMOOTH_PAINT,
    name: str = "Unnamed",
) -> Rocket:
    """Assemble a conventional single-stage rocket, resolving all positions.

    This is the normal way to build a :class:`Rocket`. It stacks the nose,
    body tube and optional boat-tail, seats the motor mount at the aft end and
    places the fin root so the fins end flush with the aft end of the body
    unless told otherwise.

    Parameters
    ----------
    nose:
        Nose cone.
    body_tube:
        Main body tube.
    fins:
        Fin set. Its ``position`` is overwritten by this function.
    motor:
        Motor and delay.
    recovery:
        Recovery device.
    motor_mount:
        Motor mount. When ``None``, one is sized automatically to the motor
        with a snug fit and two centring rings.
    boat_tail:
        Optional aft transition.
    launch_lug:
        Optional launch lug. When supplied, its ``position`` is overwritten to
        sit near the centre of gravity.
    fin_root_position:
        Axial station of the fin root leading edge [m]. Defaults to placing
        the fin root trailing edge flush with the aft end of the body tube.
    payload_mass:
        Payload mass [kg].
    nose_ballast_mass:
        Nose ballast mass [kg].
    surface_finish:
        External finish.
    name:
        Design name.

    Returns
    -------
    Rocket
        The assembled design.

    Raises
    ------
    ValueError
        If the geometry is inconsistent, for example a fin root that would
        extend beyond the aft end of the airframe.
    """
    placed: list[PlacedSection] = [PlacedSection(body_tube, nose.length)]
    if boat_tail is not None:
        placed.append(PlacedSection(boat_tail, placed[-1].aft_position))

    body_aft = placed[0].aft_position

    if motor_mount is None:
        motor_obj = motor.motor
        # A snug motor tube: 0.4 mm radial clearance is the usual fit for a
        # paper motor tube on an Estes case.
        mount_length = motor_obj.length
        motor_mount = MotorMount(
            inner_diameter=motor_obj.diameter + 0.8e-3,
            length=mount_length,
            wall_thickness=body_tube.wall_thickness,
            material=body_tube.material,
            body_inner_radius=body_tube.inner_radius,
            centring_ring_count=2,
            position=body_aft - mount_length,
        )
    if motor_mount.position + motor_mount.length > body_aft + 1e-9:
        raise ValueError(
            "the motor mount extends beyond the aft end of the body tube; "
            "lengthen the body tube or shorten the motor mount"
        )

    if fin_root_position is None:
        fin_root_position = body_aft - fins.root_chord
    if fin_root_position < nose.length - 1e-9:
        raise ValueError(
            f"fin root at x={fin_root_position:.4f} m would sit forward of the "
            f"nose/body joint at x={nose.length:.4f} m; shorten the root chord "
            f"or lengthen the body tube"
        )
    if fin_root_position + fins.root_chord > body_aft + 1e-9:
        raise ValueError(
            "fins extend beyond the aft end of the body tube; shorten the root "
            "chord or move the fins forward"
        )

    positioned_fins = replace(fins, position=fin_root_position)

    positioned_lug = launch_lug
    if launch_lug is not None:
        # Place the lug near mid-body, which is close enough to the centre of
        # gravity to avoid inducing a pitching moment on the rod.
        positioned_lug = replace(
            launch_lug, position=nose.length + 0.45 * body_tube.length
        )

    rocket = Rocket(
        nose=nose,
        sections=tuple(placed),
        fins=positioned_fins,
        motor_mount=motor_mount,
        motor=motor,
        recovery=recovery,
        launch_lug=positioned_lug,
        payload_mass=payload_mass,
        nose_ballast_mass=nose_ballast_mass,
        surface_finish=surface_finish,
        name=name,
    )

    _log.debug(
        "Built %s: %.0f mm long, %.1f g loaded, T/W %.1f",
        name,
        rocket.length * 1e3,
        rocket.loaded_mass * 1e3,
        rocket.thrust_to_weight,
    )
    return rocket
