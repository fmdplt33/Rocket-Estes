"""Component drag build-up.

Total zero-lift drag is assembled from physically separate mechanisms rather
than a single fitted coefficient, so that the optimiser can see *why* one
design is draggier than another:

======================  ====================================================
Mechanism               Model
======================  ====================================================
Skin friction           Flat-plate correlation with laminar/turbulent
                        transition and a surface-roughness floor, scaled by
                        component form factors.
Nose pressure drag      Zero for a smooth, slender profile; a
                        ``sin^2(theta)`` term for conical and blunt shapes.
Base drag               Empirical base-pressure coefficient on the aft
                        annulus, reduced while the motor is exhausting.
Fin pressure drag       Leading-edge stagnation plus trailing-edge base drag,
                        scaled by the section profile.
Fin interference        Junction drag at the fin root, reduced by a fillet.
Launch lug              Bluff-body drag on the projected frontal area.
Wave drag               Zero below the drag-divergence Mach number, then a
                        transonic rise scaled by nose fineness ratio.
Induced drag            ``CN^2`` term active only at non-zero angle of attack.
======================  ====================================================

Validity
--------
Every correlation here is a subsonic engineering model. Estes-class flights
top out near Mach 0.5 to 0.7, comfortably inside that envelope. The wave-drag
term exists so that the model degrades sensibly rather than silently
under-predicting if pushed beyond it, and
:attr:`DragBreakdown.validity_warning` says so explicitly whenever the
analysed Mach number leaves the validated range.

References
----------
[1] Hoerner, S. F. (1965). *Fluid-Dynamic Drag*. Published by the author.
    Ch. 2 (skin friction), Ch. 3 (bodies, base drag), Ch. 6 (wings and fins),
    Ch. 8 (interference), Ch. 16 (compressibility).
[2] Niskanen, S. (2013). *OpenRocket Technical Documentation*, Sec. 3.4.
[3] Schlichting, H. (1979). *Boundary-Layer Theory*, 7th ed., Ch. XXI.
[4] Barrowman, J. S. (1967). *The Practical Calculation of the Aerodynamic
    Characteristics of Slender Finned Vehicles*.
[5] Anderson, J. D. (2016). *Fundamentals of Aerodynamics*, 6th ed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rocketopt.aerodynamics.atmosphere import AtmosphereState
from rocketopt.geometry.components import FinAirfoil, Transition
from rocketopt.geometry.rocket import Rocket
from rocketopt.utils.constants import MACH_TRANSONIC_ONSET, RE_CRITICAL

__all__ = [
    "DragBreakdown",
    "skin_friction_coefficient",
    "base_drag_coefficient",
    "wave_drag_coefficient",
    "drag_buildup",
]

_MACH_VALIDATED_LIMIT: float = 0.8
"""Upper Mach number for which these correlations are considered validated."""

_RE_FULLY_TURBULENT: float = 5.0e6
"""Reynolds number above which the boundary layer is treated as fully
turbulent. Between :data:`~rocketopt.utils.constants.RE_CRITICAL` and this
value the transition point moves progressively forward along the body, so the
friction coefficient is blended rather than switched [3], Ch. XVII."""


def skin_friction_coefficient(
    reynolds: float,
    *,
    roughness: float,
    characteristic_length: float,
    mach: float = 0.0,
    assume_turbulent: bool = True,
) -> float:
    """Flat-plate skin-friction coefficient with a roughness floor.

    Three regimes are combined:

    *Laminar* (``Re < 5e5``)
        Blasius solution, ``Cf = 1.328 / sqrt(Re)`` [3].

    *Turbulent, hydraulically smooth*
        ``Cf = 1 / (1.50 ln(Re) - 5.6)^2``, the correlation OpenRocket uses
        [2], Eq. 3.81, which fits the Karman-Schoenherr law closely over the
        Reynolds range of interest.

    *Fully rough*
        ``Cf = 0.032 (Rs / L)^0.2`` [1], Ch. 2. Once the roughness elements
        protrude through the viscous sublayer, friction stops falling with
        Reynolds number. The larger of the smooth and rough values is taken.

    A compressibility correction ``(1 - 0.1 M^2)`` is then applied [2],
    Eq. 3.83.

    Parameters
    ----------
    reynolds:
        Reynolds number based on ``characteristic_length`` [-].
    roughness:
        RMS surface roughness height [m].
    characteristic_length:
        Length the Reynolds number is based on [m].
    mach:
        Mach number, for the compressibility correction [-].
    assume_turbulent:
        When ``True`` (the default) the boundary layer is treated as turbulent
        from the nose tip. This is the correct assumption for a model rocket:
        the spiral seam of a wound tube, the launch lug, the fin root joints
        and ordinary paint texture all trip the boundary layer within the
        first few centimetres, so the long laminar run that a smooth flat
        plate would enjoy never materialises. OpenRocket makes the same
        assumption for the same reason [2], Sec. 3.4.1. Set ``False`` only for
        a competition model with a genuinely polished, seamless surface.

    Returns
    -------
    float
        Skin-friction coefficient referenced to wetted area [-].
    """
    if reynolds < 1.0e4:
        # Below this the vehicle is barely moving; clamp to the laminar value
        # at Re = 1e4 so the coefficient cannot blow up as V approaches zero.
        reynolds = 1.0e4

    cf_laminar = 1.328 / math.sqrt(reynolds)
    cf_turbulent = 1.0 / (1.50 * math.log(reynolds) - 5.6) ** 2

    if roughness > 0.0 and characteristic_length > 0.0:
        cf_rough = 0.032 * (roughness / characteristic_length) ** 0.2
        cf_turbulent = max(cf_turbulent, cf_rough)

    if assume_turbulent:
        cf = cf_turbulent
    elif reynolds <= RE_CRITICAL:
        cf = cf_laminar
    elif reynolds >= _RE_FULLY_TURBULENT:
        cf = cf_turbulent
    else:
        # Transition is progressive, not a step: the transition point marches
        # forward along the body over roughly a decade of Reynolds number.
        # Blending linearly in log(Re) keeps the coefficient continuous, which
        # matters because a discontinuity here would put a false cliff in the
        # optimiser's objective function.
        frac = math.log(reynolds / RE_CRITICAL) / math.log(
            _RE_FULLY_TURBULENT / RE_CRITICAL
        )
        cf = cf_laminar + frac * (cf_turbulent - cf_laminar)

    return cf * (1.0 - 0.1 * mach * mach)


def base_drag_coefficient(mach: float, *, motor_burning: bool) -> float:
    """Base drag coefficient referenced to the base area.

    Hoerner [1], Ch. 3, gives a base-pressure coefficient near 0.12 to 0.14
    for a blunt-based body of revolution at low subsonic speed, rising with
    Mach number as ``0.12 + 0.13 M^2`` [2], Eq. 3.87.

    While the motor is burning, exhaust fills the base region and raises the
    base pressure. The reduction to 30% of the coasting value follows the
    treatment in [2], Sec. 3.4.3.

    Parameters
    ----------
    mach:
        Mach number [-].
    motor_burning:
        Whether the motor is producing thrust.

    Returns
    -------
    float
        Base drag coefficient referenced to base area [-].
    """
    cd = 0.12 + 0.13 * mach * mach
    return cd * 0.3 if motor_burning else cd


def wave_drag_coefficient(mach: float, fineness_ratio: float) -> float:
    """Transonic and supersonic wave drag referenced to body frontal area.

    Below the drag-divergence Mach number there is no wave drag. Above it, the
    coefficient rises steeply to a peak just above Mach 1 and then falls.
    The peak magnitude scales inversely with the square of the nose fineness
    ratio, which is the leading-order result of slender-body theory [5],
    Ch. 12, and matches the trend Hoerner tabulates [1], Ch. 16.

    This term is an engineering approximation. An Estes-class model does not
    reach these speeds; it exists so the model fails safe rather than
    silently under-predicting drag.

    Parameters
    ----------
    mach:
        Mach number [-].
    fineness_ratio:
        Nose cone length divided by base diameter [-].

    Returns
    -------
    float
        Wave drag coefficient referenced to body frontal area [-].
    """
    if mach <= MACH_TRANSONIC_ONSET:
        return 0.0

    # Peak wave drag scales as 1/fineness^2 for a slender nose.
    peak = 0.9 / max(fineness_ratio, 0.5) ** 2

    if mach < 1.1:
        # Smooth rise from the divergence Mach number to the peak.
        frac = (mach - MACH_TRANSONIC_ONSET) / (1.1 - MACH_TRANSONIC_ONSET)
        return peak * frac * frac

    # Gradual decay above the peak, as the shock system sweeps back.
    return peak / (1.0 + 0.7 * (mach - 1.1))


@dataclass(frozen=True, slots=True)
class DragBreakdown:
    """Zero-lift drag decomposed by mechanism.

    All coefficients are referenced to the vehicle reference area so they sum
    directly to :attr:`total`.

    Attributes
    ----------
    skin_friction:
        Viscous friction over every wetted surface.
    nose_pressure:
        Pressure (form) drag of the nose profile.
    base:
        Base drag on the aft-facing annulus.
    fin_pressure:
        Fin leading- and trailing-edge pressure drag.
    fin_interference:
        Extra drag at the fin-body junction.
    launch_lug:
        Bluff-body drag of the launch lug.
    wave:
        Compressibility wave drag.
    induced:
        Drag induced by generating normal force at angle of attack.
    mach:
        Mach number analysed.
    reynolds:
        Body-length Reynolds number analysed.
    validity_warning:
        Non-empty when the flight condition sits outside the range these
        correlations are validated for.
    """

    skin_friction: float
    nose_pressure: float
    base: float
    fin_pressure: float
    fin_interference: float
    launch_lug: float
    wave: float
    induced: float
    mach: float
    reynolds: float
    validity_warning: str = ""

    @property
    def parasitic(self) -> float:
        """Total drag excluding the induced (angle-of-attack) contribution."""
        return (
            self.skin_friction
            + self.nose_pressure
            + self.base
            + self.fin_pressure
            + self.fin_interference
            + self.launch_lug
            + self.wave
        )

    @property
    def total(self) -> float:
        """Total drag coefficient referenced to the body cross-section [-]."""
        return self.parasitic + self.induced

    def as_dict(self) -> dict[str, float]:
        """Return the breakdown as a mapping, for reports and plots."""
        return {
            "Skin friction": self.skin_friction,
            "Nose pressure": self.nose_pressure,
            "Base": self.base,
            "Fin pressure": self.fin_pressure,
            "Fin interference": self.fin_interference,
            "Launch lug": self.launch_lug,
            "Wave": self.wave,
            "Induced": self.induced,
        }

    def dominant_mechanism(self) -> str:
        """Name the largest single contributor, for report commentary."""
        return max(self.as_dict().items(), key=lambda kv: kv[1])[0]


def _nose_pressure_drag(rocket: Rocket, mach: float) -> float:
    """Nose profile pressure drag referenced to the body area.

    A smooth, slender profile recovers essentially all its stagnation pressure
    at subsonic speed, so pressure drag is negligible. A conical nose does not:
    the flow must turn through the cone half-angle at the tip, which costs
    roughly ``0.8 sin^2(theta)`` [1], Ch. 3.

    Parameters
    ----------
    rocket:
        The design.
    mach:
        Mach number [-].

    Returns
    -------
    float
        Pressure drag coefficient [-].
    """
    from rocketopt.geometry.nose_cones import NoseConeShape

    nose = rocket.nose
    if nose.shape is NoseConeShape.CONICAL:
        half_angle = math.atan2(nose.base_radius, nose.length)
        cd = 0.8 * math.sin(half_angle) ** 2
    else:
        # Curved profiles: pressure drag is small but not exactly zero, and
        # grows as the profile gets stubbier.
        cd = 0.02 / max(nose.fineness_ratio, 0.5) ** 2

    # Any shoulder or boat-tail transition adds its own form drag.
    for placed in rocket.sections:
        section = placed.section
        if isinstance(section, Transition):
            angle = section.half_angle
            if section.is_boat_tail:
                # A shallow boat-tail *reduces* net drag by shrinking the base;
                # that credit is taken in the base term, so only the additional
                # form drag of a steep taper is charged here.
                if angle > math.radians(12.0):
                    cd += 0.6 * math.sin(angle - math.radians(12.0)) ** 2
            else:
                area_ratio = (section.aft_radius / rocket.reference_radius) ** 2
                cd += 0.8 * math.sin(angle) ** 2 * area_ratio

    return cd * (1.0 + 0.2 * mach * mach)


def _fin_pressure_drag(rocket: Rocket, mach: float) -> float:
    """Fin edge pressure drag, dominated by trailing-edge base drag.

    This term covers only what the skin-friction term does not. The viscous
    drag of the fin surfaces, including the thickness effect, is already
    charged through the ``1 + 2(t/c)`` form factor applied to the fin wetted
    area, so charging full stagnation pressure on the fin frontal area here
    would double-count it heavily - on a typical model the fin frontal area
    is over half the body cross-section.

    What remains is the base drag of a blunt trailing edge, referenced to the
    fins' combined edge frontal area. Hoerner [1], Ch. 6, gives a blunt
    trailing edge a base-pressure coefficient close to that of any other
    rearward-facing step, near 0.14; tapering the trailing edge to a point
    largely removes it, which is the main reason a shaped fin is worth the
    sanding.

    Parameters
    ----------
    rocket:
        The design.
    mach:
        Mach number [-].

    Returns
    -------
    float
        Fin pressure drag coefficient referenced to body area [-].
    """
    fins = rocket.fins
    frontal = fins.span * fins.thickness * fins.count
    if frontal <= 0.0:
        return 0.0

    # Base-pressure coefficient of the trailing edge, by section shape.
    edge_cd = {
        FinAirfoil.SQUARE: 0.14,  # full blunt-base drag on the cut edge
        FinAirfoil.ROUNDED: 0.10,  # rounded LE helps, TE is still blunt
        FinAirfoil.AIRFOIL: 0.02,  # tapers to a sharp TE, base drag nearly gone
        FinAirfoil.DOUBLE_WEDGE: 0.03,
    }[fins.airfoil]

    cd = edge_cd * frontal / rocket.reference_area
    return cd * (1.0 + 0.15 * mach * mach)


def _fin_interference_drag(rocket: Rocket, fin_drag: float) -> float:
    """Junction drag where the fin roots meet the body.

    The corner between fin and body forces two boundary layers to merge, which
    thickens the flow and separates it in the corner. Hoerner [1], Ch. 8,
    gives junction interference as a substantial fraction of the appendage's
    own drag, falling sharply once a fillet is added.

    The fillet credit here is an engineering approximation: interference falls
    from 15% of fin drag with a sharp corner to about 4% once the fillet
    radius reaches roughly one fin thickness, which is the point of
    diminishing returns commonly quoted in model rocketry practice.

    Parameters
    ----------
    rocket:
        The design.
    fin_drag:
        Combined fin skin-friction and pressure drag coefficient [-].

    Returns
    -------
    float
        Interference drag coefficient referenced to body area [-].
    """
    fins = rocket.fins
    if fins.thickness <= 0.0:
        return 0.0

    fillet_ratio = min(fins.fillet_radius / fins.thickness, 1.0)
    factor = 0.15 - 0.11 * fillet_ratio
    return factor * fin_drag


def drag_buildup(
    rocket: Rocket,
    state: AtmosphereState,
    velocity: float,
    *,
    angle_of_attack: float = 0.0,
    motor_burning: bool = False,
    normal_force_coefficient: float = 0.0,
    assume_turbulent: bool = True,
) -> DragBreakdown:
    """Compute the full drag breakdown at one flight condition.

    Parameters
    ----------
    rocket:
        The design.
    state:
        Local atmospheric properties.
    velocity:
        Airspeed [m/s].
    angle_of_attack:
        Angle of attack [rad], used for the induced-drag term.
    motor_burning:
        Whether the motor is producing thrust, which raises base pressure.
    normal_force_coefficient:
        Normal-force coefficient at this condition, from
        :func:`~rocketopt.aerodynamics.barrowman.barrowman_analysis`. Used for
        induced drag; pass zero to omit it.

    Returns
    -------
    DragBreakdown
        Drag decomposed by mechanism, all referenced to the body area.
    """
    speed = abs(velocity)
    mach = state.mach(speed)
    length = rocket.length
    reynolds = state.reynolds(speed, length)
    a_ref = rocket.reference_area
    roughness = float(rocket.surface_finish)

    # -- Skin friction, with per-component form factors -------------------
    cf = skin_friction_coefficient(
        reynolds,
        roughness=roughness,
        characteristic_length=length,
        mach=mach,
        assume_turbulent=assume_turbulent,
    )

    # Body form factor: a slender body's boundary layer sees a mild pressure
    # gradient, raising friction above the flat-plate value [2], Eq. 3.85.
    body_wetted = rocket.nose.wetted_area + sum(
        p.section.wetted_area for p in rocket.sections
    )
    body_form = 1.0 + 1.0 / (2.0 * max(rocket.fineness_ratio, 1.0))

    # Fin form factor: thickness accelerates the flow over the section.
    fin_form = 1.0 + 2.0 * rocket.fins.thickness_ratio

    friction_area = body_form * body_wetted + fin_form * rocket.fins.wetted_area
    if rocket.launch_lug is not None:
        friction_area += rocket.launch_lug.wetted_area

    cd_friction = cf * friction_area / a_ref

    # -- Pressure and base ------------------------------------------------
    cd_nose = _nose_pressure_drag(rocket, mach)

    cd_base_ref_base = base_drag_coefficient(mach, motor_burning=motor_burning)
    # Only the annulus outside the motor nozzle is a true base while burning.
    base_area = rocket.base_area
    if motor_burning:
        base_area = max(base_area - rocket.motor.motor.cross_sectional_area, 0.0)
    cd_base = cd_base_ref_base * base_area / a_ref

    # -- Fins --------------------------------------------------------------
    cd_fin_pressure = _fin_pressure_drag(rocket, mach)
    fin_friction = cf * fin_form * rocket.fins.wetted_area / a_ref
    cd_interference = _fin_interference_drag(rocket, fin_friction + cd_fin_pressure)

    # -- Launch lug --------------------------------------------------------
    cd_lug = 0.0
    if rocket.launch_lug is not None:
        # A short cylinder in the boundary layer; the 0.6 factor accounts for
        # it sitting inside the retarded flow near the body [1], Ch. 8.
        cd_lug = 0.6 * rocket.launch_lug.frontal_area / a_ref

    # -- Wave --------------------------------------------------------------
    cd_wave = wave_drag_coefficient(mach, rocket.nose.fineness_ratio)

    # -- Induced -----------------------------------------------------------
    # At angle of attack the normal force has a streamwise component.
    cd_induced = 0.0
    if abs(angle_of_attack) > 1e-9 and abs(normal_force_coefficient) > 1e-12:
        cd_induced = abs(normal_force_coefficient * math.sin(angle_of_attack))

    warning = ""
    if mach > _MACH_VALIDATED_LIMIT:
        warning = (
            f"Mach {mach:.2f} is above the {_MACH_VALIDATED_LIMIT} limit these "
            f"subsonic drag correlations are validated for; the wave-drag term "
            f"is an approximation and total drag should be treated as "
            f"indicative only."
        )

    return DragBreakdown(
        skin_friction=cd_friction,
        nose_pressure=cd_nose,
        base=cd_base,
        fin_pressure=cd_fin_pressure,
        fin_interference=cd_interference,
        launch_lug=cd_lug,
        wave=cd_wave,
        induced=cd_induced,
        mach=mach,
        reynolds=reynolds,
        validity_warning=warning,
    )
