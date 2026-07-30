"""Structural analysis: flutter, buckling, stress, landing and joint loads.

Each check returns both the computed quantity and a margin of safety, defined
throughout as::

    MS = allowable / applied - 1

so that a positive margin means the structure survives and zero means it is
exactly at its limit. This is the convention used across aerospace structural
practice and avoids the ambiguity of a bare "safety factor".

References
----------
[1] Martin, D. J. (1958). *Summary of Flutter Experiences as a Guide to the
    Preliminary Design of Lifting Surfaces on Missiles*. NACA TN 4197.
[2] NASA (1968). *Buckling of Thin-Walled Circular Cylinders*. NASA SP-8007.
[3] Timoshenko, S. P., & Gere, J. M. (1961). *Theory of Elastic Stability*,
    2nd ed., Ch. 11.
[4] Hoerner, S. F. (1965). *Fluid-Dynamic Drag*.
[5] Peery, D. J., & Azar, J. J. (1982). *Aircraft Structures*, 2nd ed.
[6] Niskanen, S. (2013). *OpenRocket Technical Documentation*.
[7] Petrie, E. M. (2007). *Handbook of Adhesives and Sealants*, 2nd ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rocketopt.structures.materials import Adhesive, get_adhesive
from rocketopt.utils.constants import G0

# The geometry layer imports rocketopt.structures.materials, which means
# importing rocketopt.geometry here at module scope would close an import
# cycle: geometry.components -> structures (package __init__) -> analysis ->
# geometry.components. materials is a leaf module with no geometry
# dependency, whereas this module sits above geometry, so the geometry
# imports are deferred to type-checking time and to the point of use.
if TYPE_CHECKING:  # pragma: no cover
    from rocketopt.geometry.components import BodyTube
    from rocketopt.geometry.rocket import Rocket

__all__ = [
    "MarginResult",
    "FlutterResult",
    "StructuralReport",
    "fin_flutter_velocity",
    "fin_divergence_velocity",
    "fin_bending_stress",
    "tube_buckling_stress",
    "tube_axial_stress",
    "landing_deceleration",
    "fin_joint_shear_stress",
    "analyse_structure",
]


@dataclass(frozen=True, slots=True)
class MarginResult:
    """A single structural check.

    Attributes
    ----------
    name:
        Check name.
    applied:
        Applied load or stress, in the units named by :attr:`units`.
    allowable:
        Allowable value in the same units.
    units:
        Unit label for reporting.
    note:
        Optional commentary.
    """

    name: str
    applied: float
    allowable: float
    units: str
    note: str = ""

    @property
    def margin_of_safety(self) -> float:
        """``allowable / applied - 1`` [-].

        Returns positive infinity when nothing is applied, which is the
        correct reading: an unloaded member cannot fail.
        """
        if abs(self.applied) < 1e-12:
            return math.inf
        return self.allowable / abs(self.applied) - 1.0

    @property
    def passes(self) -> bool:
        """Whether the margin of safety is non-negative."""
        return self.margin_of_safety >= 0.0

    @property
    def utilisation(self) -> float:
        """Applied divided by allowable [-]. Above 1 means failure."""
        if abs(self.allowable) < 1e-12:
            return math.inf
        return abs(self.applied) / self.allowable

    def __str__(self) -> str:
        """Return a one-line summary."""
        verdict = "PASS" if self.passes else "FAIL"
        margin = self.margin_of_safety
        margin_text = "inf" if math.isinf(margin) else f"{margin:+.2f}"
        return (
            f"{self.name:<28} {self.applied:11.4g} / {self.allowable:11.4g} "
            f"{self.units:<8} MS {margin_text:>7}  {verdict}"
        )


@dataclass(frozen=True, slots=True)
class FlutterResult:
    """Fin aeroelastic stability.

    Attributes
    ----------
    flutter_velocity:
        Critical flutter speed [m/s].
    divergence_velocity:
        Critical static divergence speed [m/s].
    max_flight_velocity:
        Peak speed the vehicle actually reaches [m/s].
    altitude:
        Altitude the check was performed at [m].
    """

    flutter_velocity: float
    divergence_velocity: float
    max_flight_velocity: float
    altitude: float

    @property
    def flutter_margin(self) -> float:
        """Flutter speed divided by flight speed, minus one [-]."""
        if self.max_flight_velocity < 1e-9:
            return math.inf
        return self.flutter_velocity / self.max_flight_velocity - 1.0

    @property
    def is_safe(self) -> bool:
        """Whether flutter and divergence speeds both exceed flight speed.

        A 50% margin is required rather than a bare pass. Flutter is a
        catastrophic, self-amplifying failure with no warning: the fin
        departs in a fraction of a second. The flutter correlation itself
        carries roughly +/-20% scatter [1], so flying close to the predicted
        boundary is not defensible.
        """
        required = 1.5 * self.max_flight_velocity
        return self.flutter_velocity >= required and self.divergence_velocity >= required


def fin_flutter_velocity(
    *,
    shear_modulus: float,
    aspect_ratio: float,
    taper_ratio: float,
    thickness_ratio: float,
    ambient_pressure: float,
    speed_of_sound: float,
) -> float:
    """Critical fin flutter velocity [m/s].

    The standard preliminary-design correlation from NACA TN 4197 [1]::

        V_f = a * sqrt( G / ( 1.337 A^3 P (lambda + 1)
                              / ( 2 (A + 2) (t/c)^3 ) ) )

    where ``A`` is the fin aspect ratio, ``lambda`` the taper ratio, ``t/c``
    the thickness-to-chord ratio, ``P`` the ambient static pressure, ``G`` the
    fin material's shear modulus and ``a`` the local speed of sound.

    The dominant term is ``(t/c)^3``: halving fin thickness cuts the flutter
    speed by a factor of nearly three. This is why thin fins on a fast model
    shed, and why the optimiser must carry this as a hard constraint rather
    than a soft penalty.

    Parameters
    ----------
    shear_modulus:
        Fin material shear modulus G [Pa].
    aspect_ratio:
        Exposed panel aspect ratio ``span^2 / area`` [-].
    taper_ratio:
        Tip chord over root chord [-].
    thickness_ratio:
        Thickness over mean chord [-].
    ambient_pressure:
        Static pressure at the analysis altitude [Pa].
    speed_of_sound:
        Local speed of sound [m/s].

    Returns
    -------
    float
        Flutter velocity [m/s]. Returns infinity for a degenerate fin with
        zero thickness ratio or zero aspect ratio, where the correlation does
        not apply.

    Raises
    ------
    ValueError
        If any input is non-positive where the correlation requires positive
        values.
    """
    if shear_modulus <= 0.0:
        raise ValueError("shear modulus must be positive")
    if ambient_pressure <= 0.0:
        raise ValueError("ambient pressure must be positive")
    if speed_of_sound <= 0.0:
        raise ValueError("speed of sound must be positive")
    if thickness_ratio <= 0.0 or aspect_ratio <= 0.0:
        return math.inf

    numerator = 1.337 * aspect_ratio**3 * ambient_pressure * (taper_ratio + 1.0)
    denominator = 2.0 * (aspect_ratio + 2.0) * thickness_ratio**3
    denominator_term = numerator / denominator

    if denominator_term <= 0.0:
        return math.inf
    return speed_of_sound * math.sqrt(shear_modulus / denominator_term)


def fin_divergence_velocity(
    *,
    youngs_modulus: float,
    thickness: float,
    root_chord: float,
    span: float,
    ambient_pressure: float,
    speed_of_sound: float,
) -> float:
    """Critical static divergence velocity of a fin [m/s].

    Divergence is the static counterpart of flutter: at sufficient dynamic
    pressure the aerodynamic moment produced by a small twist exceeds the
    fin's torsional stiffness and the twist runs away. For an unswept
    cantilever plate the divergence dynamic pressure follows from equating
    the aerodynamic moment slope to the torsional stiffness [1][3]::

        q_div = G J / (e a S b)

    Modelling the fin as a thin rectangular plate with torsional constant
    ``J = c t^3 / 3``, taking the aerodynamic centre at quarter chord and the
    elastic axis at mid chord (so the eccentricity ``e`` is ``c/4``), and
    using a lift-curve slope of ``2 pi`` gives the expression used here.

    Parameters
    ----------
    youngs_modulus:
        Fin material Young's modulus [Pa]. Converted internally to a shear
        modulus assuming a Poisson ratio of 0.3.
    thickness:
        Fin thickness [m].
    root_chord:
        Root chord [m].
    span:
        Exposed semi-span [m].
    ambient_pressure:
        Static pressure [Pa].
    speed_of_sound:
        Local speed of sound [m/s].

    Returns
    -------
    float
        Divergence velocity [m/s], or infinity for a degenerate fin.
    """
    if thickness <= 0.0 or root_chord <= 0.0 or span <= 0.0:
        return math.inf
    if ambient_pressure <= 0.0 or speed_of_sound <= 0.0:
        raise ValueError("pressure and speed of sound must be positive")

    shear_modulus = youngs_modulus / (2.0 * (1.0 + 0.3))
    torsion_constant = root_chord * thickness**3 / 3.0
    eccentricity = root_chord / 4.0
    lift_slope = 2.0 * math.pi
    area = root_chord * span

    q_divergence = (shear_modulus * torsion_constant) / (
        eccentricity * lift_slope * area * span
    )
    if q_divergence <= 0.0:
        return math.inf

    # q = 0.5 rho V^2, and rho = gamma P / a^2 for an ideal gas.
    density = 1.4 * ambient_pressure / (speed_of_sound**2)
    return math.sqrt(2.0 * q_divergence / density)


def fin_bending_stress(
    *,
    normal_force_per_fin: float,
    span: float,
    root_chord: float,
    thickness: float,
    taper_ratio: float,
) -> float:
    """Peak bending stress at the fin root [Pa].

    The fin is treated as a cantilever plate carrying a spanwise-distributed
    aerodynamic load. The load centroid of a trapezoidal planform sits at::

        y_bar = (span / 3) (1 + 2 lambda) / (1 + lambda)

    giving a root bending moment ``M = N y_bar``. The root section is a
    rectangle of width ``c_root`` and depth ``t``, so its section modulus is
    ``Z = c_root t^2 / 6`` and the peak fibre stress is ``M / Z`` [5], Ch. 5.

    Parameters
    ----------
    normal_force_per_fin:
        Aerodynamic normal force carried by one fin [N].
    span:
        Exposed semi-span [m].
    root_chord:
        Root chord [m].
    thickness:
        Fin thickness [m].
    taper_ratio:
        Tip chord over root chord [-].

    Returns
    -------
    float
        Peak bending stress [Pa].
    """
    if thickness <= 0.0 or root_chord <= 0.0:
        raise ValueError("fin thickness and root chord must be positive")

    centroid = (span / 3.0) * (1.0 + 2.0 * taper_ratio) / (1.0 + taper_ratio)
    moment = abs(normal_force_per_fin) * centroid
    section_modulus = root_chord * thickness**2 / 6.0
    return moment / section_modulus


def tube_buckling_stress(tube: BodyTube) -> float:
    """Allowable axial compressive stress before shell buckling [Pa].

    A thin-walled cylinder in axial compression fails by shell buckling long
    before it reaches its material compressive strength. Classical theory [3]
    gives::

        sigma_cr = E t / (r sqrt(3 (1 - nu^2)))

    Real shells buckle at a fraction of this because of unavoidable geometric
    imperfections. NASA SP-8007 [2] gives the empirical knockdown factor::

        gamma = 1 - 0.901 (1 - exp(-phi)),  phi = (1/16) sqrt(r / t)

    which for a typical model rocket tube (``r/t`` around 35) lands near 0.5.
    Ignoring this knockdown would overestimate the allowable by roughly a
    factor of two.

    Parameters
    ----------
    tube:
        The body tube.

    Returns
    -------
    float
        Allowable compressive stress [Pa], the lesser of the knocked-down
        buckling stress and the material's compressive strength.
    """
    material = tube.material
    radius = tube.outer_radius - tube.wall_thickness / 2.0  # mid-surface radius
    thickness = tube.wall_thickness
    nu = material.poisson_ratio

    classical = material.youngs_modulus * thickness / (
        radius * math.sqrt(3.0 * (1.0 - nu * nu))
    )

    phi = (1.0 / 16.0) * math.sqrt(radius / thickness)
    knockdown = 1.0 - 0.901 * (1.0 - math.exp(-phi))

    return min(classical * knockdown, material.compressive_strength)


def tube_axial_stress(tube: BodyTube, axial_force: float) -> float:
    """Axial stress in a tube wall [Pa].

    Parameters
    ----------
    tube:
        The body tube.
    axial_force:
        Axial force carried [N]. Sign is ignored.

    Returns
    -------
    float
        Axial stress [Pa].
    """
    area = math.pi * (tube.outer_radius**2 - tube.inner_radius**2)
    if area <= 0.0:
        raise ValueError("tube wall area must be positive")
    return abs(axial_force) / area


def landing_deceleration(
    landing_velocity: float,
    *,
    crush_distance: float = 0.02,
) -> float:
    """Deceleration on ground impact [m/s^2].

    Energy is absorbed over a short crushing distance as the nose cone,
    body tube and ground deform. Constant deceleration over that distance
    gives ``a = v^2 / (2 s)``.

    The default 20 mm represents impact on firm grass, which is what a model
    rocket normally lands on. Concrete would be an order of magnitude harsher,
    and this figure should be reduced accordingly for a hard-surface site.

    Parameters
    ----------
    landing_velocity:
        Descent speed at impact [m/s].
    crush_distance:
        Effective stopping distance [m].

    Returns
    -------
    float
        Deceleration magnitude [m/s^2].
    """
    if crush_distance <= 0.0:
        raise ValueError("crush distance must be positive")
    return landing_velocity * landing_velocity / (2.0 * crush_distance)


def fin_joint_shear_stress(
    *,
    normal_force_per_fin: float,
    root_chord: float,
    fillet_radius: float,
    fin_thickness: float,
) -> float:
    """Shear stress in the fin-to-body adhesive joint [Pa].

    The bond carries the fin's aerodynamic load into the airframe. Without a
    fillet the effective bond width is just the fin thickness, which is a very
    small area. A fillet of radius ``R`` adds ``2R`` of bonded width - one
    fillet each side - and is the single most effective structural
    improvement available on a model rocket.

    Parameters
    ----------
    normal_force_per_fin:
        Aerodynamic normal force on one fin [N].
    root_chord:
        Root chord, which sets the bond length [m].
    fillet_radius:
        Fillet radius [m].
    fin_thickness:
        Fin thickness [m].

    Returns
    -------
    float
        Average shear stress in the joint [Pa].
    """
    bond_width = fin_thickness + 2.0 * fillet_radius
    bond_area = bond_width * root_chord
    if bond_area <= 0.0:
        raise ValueError("bond area must be positive")
    return abs(normal_force_per_fin) / bond_area


@dataclass(frozen=True, slots=True)
class StructuralReport:
    """Complete structural assessment of a design at its worst flight condition.

    Attributes
    ----------
    flutter:
        Fin aeroelastic result.
    checks:
        Every margin-of-safety check performed.
    safety_factor:
        Design safety factor the allowables were reduced by.
    limiting_check:
        The check with the smallest margin.
    """

    flutter: FlutterResult
    checks: tuple[MarginResult, ...]
    safety_factor: float

    @property
    def limiting_check(self) -> MarginResult:
        """The check with the smallest margin of safety."""
        return min(self.checks, key=lambda c: c.margin_of_safety)

    @property
    def minimum_margin(self) -> float:
        """Smallest margin of safety across all checks [-]."""
        return self.limiting_check.margin_of_safety

    @property
    def passes(self) -> bool:
        """Whether every check passes and the fins are flutter-safe."""
        return all(c.passes for c in self.checks) and self.flutter.is_safe

    def summary(self) -> str:
        """Return a readable multi-line structural summary."""
        lines = [
            f"Structural assessment (safety factor {self.safety_factor:.2f})",
            f"  {'check':<28} {'applied':>11} / {'allowable':>11} {'units':<8} "
            f"{'MS':>10}  verdict",
        ]
        lines.extend(f"  {check}" for check in self.checks)
        lines.append(
            f"  Fin flutter velocity        {self.flutter.flutter_velocity:11.1f} m/s "
            f"vs {self.flutter.max_flight_velocity:.1f} m/s flight "
            f"({'SAFE' if self.flutter.is_safe else 'UNSAFE'})"
        )
        lines.append(
            f"  Fin divergence velocity     {self.flutter.divergence_velocity:11.1f} m/s"
        )
        lines.append(f"  Limiting check: {self.limiting_check.name}")
        return "\n".join(lines)


def analyse_structure(
    rocket: Rocket,
    *,
    max_velocity: float,
    max_dynamic_pressure: float,
    max_acceleration: float,
    landing_velocity: float,
    analysis_altitude: float = 0.0,
    max_angle_of_attack: float = math.radians(5.0),
    safety_factor: float = 1.5,
    adhesive: Adhesive | None = None,
) -> StructuralReport:
    """Run every structural check against a flight's peak loads.

    Parameters
    ----------
    rocket:
        The design.
    max_velocity:
        Peak airspeed from the trajectory [m/s].
    max_dynamic_pressure:
        Peak dynamic pressure [Pa].
    max_acceleration:
        Peak acceleration magnitude [m/s^2].
    landing_velocity:
        Descent speed at ground impact [m/s].
    analysis_altitude:
        Altitude to evaluate atmospheric properties at [m]. Flutter is most
        critical at *low* altitude where pressure is highest, so the default
        of sea level is the conservative choice.
    max_angle_of_attack:
        Worst-case angle of attack for fin loading [rad]. Five degrees is a
        reasonable upper bound for a stable rocket in moderate wind.
    safety_factor:
        Factor by which material allowables are reduced.
    adhesive:
        Adhesive used at the fin joints. Defaults to five-minute epoxy.

    Returns
    -------
    StructuralReport
        All checks plus the flutter assessment.

    Raises
    ------
    ValueError
        If ``safety_factor`` is less than 1.
    """
    if safety_factor < 1.0:
        raise ValueError("safety factor must be at least 1.0")

    from rocketopt.aerodynamics.atmosphere import Atmosphere
    from rocketopt.geometry.components import BodyTube

    atmosphere = Atmosphere.standard()
    state = atmosphere.state_at(analysis_altitude)

    fins = rocket.fins
    fin_material = fins.material
    adhesive = adhesive or get_adhesive("epoxy_5min")

    # -- Flutter and divergence ------------------------------------------
    flutter_velocity = fin_flutter_velocity(
        shear_modulus=fin_material.shear_modulus,
        aspect_ratio=fins.aspect_ratio,
        taper_ratio=fins.taper_ratio,
        thickness_ratio=fins.thickness_ratio,
        ambient_pressure=state.pressure,
        speed_of_sound=state.speed_of_sound,
    )
    divergence_velocity = fin_divergence_velocity(
        youngs_modulus=fin_material.youngs_modulus,
        thickness=fins.thickness,
        root_chord=fins.root_chord,
        span=fins.span,
        ambient_pressure=state.pressure,
        speed_of_sound=state.speed_of_sound,
    )
    flutter = FlutterResult(
        flutter_velocity=flutter_velocity,
        divergence_velocity=divergence_velocity,
        max_flight_velocity=max_velocity,
        altitude=analysis_altitude,
    )

    checks: list[MarginResult] = []

    # -- Fin bending -------------------------------------------------------
    # Normal force shared across the fin set at the worst-case incidence.
    fin_cna = fins.cn_alpha(rocket.reference_radius)
    total_normal_force = (
        fin_cna * max_angle_of_attack * max_dynamic_pressure * rocket.reference_area
    )
    per_fin = total_normal_force / fins.count

    bending_stress = fin_bending_stress(
        normal_force_per_fin=per_fin,
        span=fins.span,
        root_chord=fins.root_chord,
        thickness=fins.thickness,
        taper_ratio=fins.taper_ratio,
    )
    checks.append(
        MarginResult(
            name="Fin root bending",
            applied=bending_stress,
            allowable=fin_material.tensile_strength / safety_factor,
            units="Pa",
            note=f"at {math.degrees(max_angle_of_attack):.0f} deg angle of attack",
        )
    )

    # -- Fin joint shear ---------------------------------------------------
    joint_stress = fin_joint_shear_stress(
        normal_force_per_fin=per_fin,
        root_chord=fins.root_chord,
        fillet_radius=fins.fillet_radius,
        fin_thickness=fins.thickness,
    )
    # The joint is only as good as the weaker of glue and substrate bond.
    joint_allowable = (
        adhesive.lap_shear_strength * fin_material.bondability / safety_factor
    )
    checks.append(
        MarginResult(
            name="Fin joint shear",
            applied=joint_stress,
            allowable=joint_allowable,
            units="Pa",
            note=f"{adhesive.display_name}, "
            f"{fins.fillet_radius * 1e3:.1f} mm fillet",
        )
    )

    # -- Body tube axial compression under thrust -------------------------
    main_tube = next(
        (p.section for p in rocket.sections if isinstance(p.section, BodyTube)),
        None,
    )
    if main_tube is not None:
        # The tube carries the inertial reaction of everything ahead of it.
        forward_mass = rocket.loaded_mass - rocket.motor.motor.total_mass
        axial_force = forward_mass * max_acceleration
        axial_stress = tube_axial_stress(main_tube, axial_force)
        checks.append(
            MarginResult(
                name="Body tube compression",
                applied=axial_stress,
                allowable=tube_buckling_stress(main_tube) / safety_factor,
                units="Pa",
                note="shell buckling with NASA SP-8007 knockdown",
            )
        )

    # -- Landing impact ----------------------------------------------------
    deceleration = landing_deceleration(landing_velocity)
    # Treat the nose cone as the impacting member; it must survive its own
    # inertial load plus that of the mass behind it.
    impact_force = rocket.dry_mass * deceleration
    if main_tube is not None:
        impact_stress = tube_axial_stress(main_tube, impact_force)
        checks.append(
            MarginResult(
                name="Landing impact",
                applied=impact_stress,
                allowable=tube_buckling_stress(main_tube) / safety_factor,
                units="Pa",
                note=f"{landing_velocity:.1f} m/s onto firm grass "
                f"({deceleration / G0:.0f} g)",
            )
        )

    return StructuralReport(
        flutter=flutter,
        checks=tuple(checks),
        safety_factor=safety_factor,
    )
