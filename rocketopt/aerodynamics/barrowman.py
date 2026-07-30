"""Barrowman stability analysis: normal force, centre of pressure and damping.

What Barrowman's method does and does not cover
-----------------------------------------------
The classical method [1][2] is a slender-body potential-flow result valid at
small angle of attack and subsonic speed. It gives the normal-force slope of
each component and the point at which that force acts. Three extensions are
applied here, each of which matters for a real model rocket:

*Compressibility*
    The Prandtl-Glauert factor ``1 / sqrt(1 - M^2)`` scales the fin
    normal-force slope with Mach number [3]. Above Mach 0.8 the correction is
    frozen at its Mach 0.8 value, because Prandtl-Glauert diverges at Mach 1
    and no model in this class flies transonic in a controlled way. That
    freeze is a modelling choice and is flagged in
    :attr:`BarrowmanResult.transonic_warning`.

*Viscous cross-flow*
    A cylinder at angle of attack sheds vortices and generates normal force
    that potential flow misses entirely. Barrowman gives the body zero lift;
    in reality a long body contributes appreciably above a few degrees. The
    Allen-Perkins cross-flow term [4] is added, which grows as ``sin^2(alpha)``
    and therefore vanishes at the small angles where the classical method is
    already correct.

*Rotational damping*
    Pitch, yaw and roll damping derivatives are computed by strip integration
    so that the 6-DOF simulation has physically-based rates rather than
    arbitrary numbers.

Sign and coordinate conventions
-------------------------------
``x`` is measured aft from the nose tip. All coefficients are referenced to
the body cross-sectional area ``A_ref`` and diameter ``d``. A positive
normal-force slope pushes the nose away from the free stream; the centre of
pressure must lie *aft* of the centre of gravity for static stability.

References
----------
[1] Barrowman, J. S. (1967). *The Practical Calculation of the Aerodynamic
    Characteristics of Slender Finned Vehicles*. M.Sc. thesis, Catholic
    University of America.
[2] Barrowman, J. S., & Barrowman, J. A. (1966). *A Method for Calculating the
    Centre of Pressure of Model Rockets*. NARAM-8.
[3] Anderson, J. D. (2016). *Fundamentals of Aerodynamics*, 6th ed., Sec. 11.4
    (Prandtl-Glauert rule).
[4] Allen, H. J., & Perkins, E. W. (1951). *A Study of Effects of Viscosity on
    Flow Over Slender Inclined Bodies of Revolution*. NACA Report 1048.
[5] Hoerner, S. F. (1965). *Fluid-Dynamic Drag*, Ch. 3 (cross-flow drag of
    cylinders).
[6] Niskanen, S. (2013). *OpenRocket Technical Documentation*, Sec. 3.2, 3.5.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rocketopt.geometry.components import BodyTube, Transition
from rocketopt.geometry.rocket import Rocket
from rocketopt.utils.constants import (
    MACH_TRANSONIC_ONSET,
    MAX_STATIC_MARGIN_CALIBRES,
    MIN_STATIC_MARGIN_CALIBRES,
)

__all__ = [
    "ComponentContribution",
    "BarrowmanResult",
    "barrowman_analysis",
    "prandtl_glauert",
    "crossflow_drag_coefficient",
    "static_margin",
]

_CROSSFLOW_ETA: float = 0.65
"""Cross-flow drag proportionality factor for a finite cylinder.

Allen and Perkins [4] express body cross-flow lift as
``eta * c_dn * (A_plan / A_ref) * sin^2(alpha)``, where ``eta`` accounts for
the finite length of the body relative to an infinite cylinder. Hoerner [5],
Ch. 3, gives eta near 0.65 for the length-to-diameter ratios typical of model
rockets (8 to 20).
"""


def prandtl_glauert(mach: float) -> float:
    """Prandtl-Glauert compressibility factor [-].

    Returns ``1 / sqrt(1 - M^2)`` for subsonic flow [3]. Above
    :data:`~rocketopt.utils.constants.MACH_TRANSONIC_ONSET` the factor is held
    at its value there, because the correction diverges at Mach 1 and would
    otherwise produce unbounded normal-force slopes.

    Parameters
    ----------
    mach:
        Mach number [-].

    Returns
    -------
    float
        Compressibility scale factor, always at least 1.
    """
    m = min(abs(mach), MACH_TRANSONIC_ONSET)
    return 1.0 / math.sqrt(max(1.0 - m * m, 1e-6))


def crossflow_drag_coefficient(mach: float) -> float:
    """Cross-flow drag coefficient of a circular cylinder [-].

    Hoerner [5], Ch. 3. At the Reynolds numbers a model rocket body sees in
    cross-flow (10^4 to 10^5) the sub-critical value is close to 1.2, rising
    slowly as the cross-flow Mach number increases.

    Parameters
    ----------
    mach:
        Free-stream Mach number [-].

    Returns
    -------
    float
        Cross-flow drag coefficient [-].
    """
    if mach <= 0.5:
        return 1.2
    # Mild compressibility rise towards the transonic drag peak.
    return 1.2 + 0.6 * min(mach - 0.5, 0.5)


@dataclass(frozen=True, slots=True)
class ComponentContribution:
    """One component's contribution to the vehicle normal force.

    Attributes
    ----------
    name:
        Component label.
    cn_alpha:
        Normal-force slope referenced to the vehicle reference area [1/rad].
    centre_of_pressure:
        Application point aft of the nose tip [m].
    """

    name: str
    cn_alpha: float
    centre_of_pressure: float

    @property
    def moment_arm_product(self) -> float:
        """Product ``CN_alpha * X_cp``, the term summed to locate the vehicle CP."""
        return self.cn_alpha * self.centre_of_pressure


@dataclass(frozen=True, slots=True)
class BarrowmanResult:
    """Outcome of a Barrowman analysis at one flight condition.

    Attributes
    ----------
    cn_alpha:
        Vehicle normal-force slope [1/rad], referenced to the body area.
    centre_of_pressure:
        Vehicle centre of pressure aft of the nose tip [m].
    normal_force_coefficient:
        Normal-force coefficient at the analysed angle of attack [-],
        including the viscous cross-flow contribution.
    mach:
        Mach number analysed [-].
    angle_of_attack:
        Angle of attack analysed [rad].
    contributions:
        Per-component breakdown.
    crossflow_cn:
        Normal force from viscous cross-flow alone [-].
    transonic_warning:
        Non-empty when the Prandtl-Glauert correction was frozen because the
        Mach number exceeded the transonic onset.
    """

    cn_alpha: float
    centre_of_pressure: float
    normal_force_coefficient: float
    mach: float
    angle_of_attack: float
    contributions: tuple[ComponentContribution, ...] = field(default=())
    crossflow_cn: float = 0.0
    transonic_warning: str = ""

    def stability_calibres(self, cg: float, reference_diameter: float) -> float:
        """Static margin in body calibres for a given centre of gravity.

        Parameters
        ----------
        cg:
            Centre of gravity aft of the nose tip [m].
        reference_diameter:
            One body calibre [m].

        Returns
        -------
        float
            ``(X_cp - X_cg) / d``. Positive means statically stable.
        """
        return (self.centre_of_pressure - cg) / reference_diameter


def _body_crossflow(rocket: Rocket, alpha: float, mach: float) -> tuple[float, float]:
    """Viscous cross-flow normal force and its application point.

    Allen and Perkins [4]::

        CN_crossflow = eta * c_dn * (A_planform / A_ref) * sin^2(alpha)

    acting at the centroid of the projected side area.

    Parameters
    ----------
    rocket:
        The design.
    alpha:
        Angle of attack [rad].
    mach:
        Mach number [-].

    Returns
    -------
    tuple
        ``(CN, X_cp)`` - the cross-flow normal-force coefficient [-] and its
        application point aft of the nose tip [m].
    """
    if abs(alpha) < 1e-9:
        return 0.0, 0.0

    planform = rocket.body_planform_area
    if planform <= 0.0:
        return 0.0, 0.0

    c_dn = crossflow_drag_coefficient(mach)
    cn = (
        _CROSSFLOW_ETA
        * c_dn
        * (planform / rocket.reference_area)
        * math.sin(alpha) ** 2
        * math.copysign(1.0, alpha)
    )

    # Centroid of the side-view area: area-weighted mean station.
    moment = 0.0
    area = 0.0

    nose_area = rocket.nose.planform_area
    # The nose planform centroid, computed from its own profile.
    x_nose, y_nose = rocket.nose.profile_points(200)
    if nose_area > 0.0:
        nose_centroid = float(np.trapezoid(2.0 * y_nose * x_nose, x_nose)) / nose_area
        moment += nose_area * nose_centroid
        area += nose_area

    for placed in rocket.sections:
        a = placed.section.planform_area
        if isinstance(placed.section, Transition):
            r1, r2 = placed.section.fore_radius, placed.section.aft_radius
            # Centroid of a trapezoid of depths 2*r1 and 2*r2.
            centroid_local = (
                placed.section.length * (r1 + 2.0 * r2) / (3.0 * (r1 + r2))
            )
        else:
            centroid_local = placed.section.length / 2.0
        moment += a * (placed.position + centroid_local)
        area += a

    x_cp = moment / area if area > 0.0 else rocket.length / 2.0
    return cn, x_cp


def barrowman_analysis(
    rocket: Rocket,
    *,
    mach: float = 0.3,
    angle_of_attack: float = 0.0,
) -> BarrowmanResult:
    """Run a Barrowman stability analysis at one flight condition.

    Parameters
    ----------
    rocket:
        The design to analyse.
    mach:
        Free-stream Mach number [-].
    angle_of_attack:
        Angle of attack [rad]. The normal-force slope is independent of this;
        it affects only the viscous cross-flow term and the returned
        :attr:`BarrowmanResult.normal_force_coefficient`.

    Returns
    -------
    BarrowmanResult
        Vehicle normal-force slope, centre of pressure and a per-component
        breakdown.
    """
    ref_radius = rocket.reference_radius
    beta = prandtl_glauert(mach)

    contributions: list[ComponentContribution] = []

    # Nose: CN_alpha = 2 for any closed shape, at X = L - V/A_base.
    contributions.append(
        ComponentContribution(
            name=f"Nose ({rocket.nose.shape.label})",
            cn_alpha=rocket.nose.cn_alpha,
            centre_of_pressure=rocket.nose.centre_of_pressure,
        )
    )

    # Transitions: any change of cross-sectional area generates normal force.
    for index, placed in enumerate(rocket.sections, start=1):
        section = placed.section
        if isinstance(section, BodyTube):
            # No potential-flow contribution from a constant-diameter tube.
            continue
        contributions.append(
            ComponentContribution(
                name=(
                    f"{'Boat-tail' if section.is_boat_tail else 'Shoulder'} {index}"
                ),
                cn_alpha=section.cn_alpha(ref_radius),
                centre_of_pressure=placed.position + section.centre_of_pressure,
            )
        )

    # Fins, with the Prandtl-Glauert correction applied.
    fin_cna = rocket.fins.cn_alpha(ref_radius) * beta
    contributions.append(
        ComponentContribution(
            name=f"{rocket.fins.count} fins",
            cn_alpha=fin_cna,
            centre_of_pressure=rocket.fins.absolute_centre_of_pressure,
        )
    )

    # A launch lug is small but sits far from the CG; include it for
    # completeness. Its normal-force slope is that of a short cylinder.
    if rocket.launch_lug is not None:
        lug = rocket.launch_lug
        lug_cna = 2.0 * lug.frontal_area / rocket.reference_area
        contributions.append(
            ComponentContribution(
                name="Launch lug",
                cn_alpha=lug_cna,
                centre_of_pressure=lug.position + lug.centre_of_mass,
            )
        )

    total_cna = sum(c.cn_alpha for c in contributions)
    if abs(total_cna) < 1e-12:
        raise ValueError(
            "total normal-force slope is zero, so a centre of pressure is "
            "undefined; check that the design has fins"
        )
    x_cp = sum(c.moment_arm_product for c in contributions) / total_cna

    # Viscous cross-flow, which shifts the CP aft at larger angles of attack.
    cn_crossflow, x_crossflow = _body_crossflow(rocket, angle_of_attack, mach)
    cn_potential = total_cna * angle_of_attack
    cn_total = cn_potential + cn_crossflow

    if abs(cn_total) > 1e-12 and abs(cn_crossflow) > 1e-12:
        x_cp = (cn_potential * x_cp + cn_crossflow * x_crossflow) / cn_total

    warning = ""
    if abs(mach) > MACH_TRANSONIC_ONSET:
        warning = (
            f"Mach {mach:.2f} exceeds the transonic onset of "
            f"{MACH_TRANSONIC_ONSET}; the Prandtl-Glauert correction is frozen "
            f"at its Mach {MACH_TRANSONIC_ONSET} value and the normal-force "
            f"slope is therefore approximate."
        )

    return BarrowmanResult(
        cn_alpha=total_cna,
        centre_of_pressure=x_cp,
        normal_force_coefficient=cn_total,
        mach=mach,
        angle_of_attack=angle_of_attack,
        contributions=tuple(contributions),
        crossflow_cn=cn_crossflow,
        transonic_warning=warning,
    )


def static_margin(
    rocket: Rocket,
    *,
    t: float = 0.0,
    mach: float = 0.3,
) -> float:
    """Static margin in body calibres at a given time after ignition.

    Parameters
    ----------
    rocket:
        The design.
    t:
        Time since ignition [s]. The centre of gravity moves forward as
        propellant burns, so the margin generally increases through the burn.
    mach:
        Mach number for the aerodynamic analysis [-].

    Returns
    -------
    float
        ``(X_cp - X_cg) / d`` [calibres].
    """
    result = barrowman_analysis(rocket, mach=mach)
    cg = rocket.cg_at(t)
    return (result.centre_of_pressure - cg) / rocket.reference_diameter


@dataclass(frozen=True, slots=True)
class DampingDerivatives:
    """Rotational damping derivatives about the centre of gravity.

    All are referenced to the body area and diameter, and all are negative for
    a stable vehicle - they oppose the rotation that produces them.

    Attributes
    ----------
    cmq:
        Pitch damping derivative ``dCm / d(q d / 2V)`` [1/rad].
    cnr:
        Yaw damping derivative. Equal to :attr:`cmq` for an axisymmetric body.
    clp:
        Roll damping derivative ``dCl / d(p d / 2V)`` [1/rad].
    cl_delta:
        Roll moment produced per radian of fin cant [1/rad].
    """

    cmq: float
    cnr: float
    clp: float
    cl_delta: float


def damping_derivatives(
    rocket: Rocket,
    *,
    t: float = 0.0,
    mach: float = 0.3,
) -> DampingDerivatives:
    """Compute pitch, yaw and roll damping derivatives.

    Pitch damping is obtained by summing each lifting component's contribution
    weighted by the square of its moment arm [6], Sec. 3.5::

        Cmq = -2 sum_i CN_alpha_i ((x_i - x_cg) / d)^2

    Roll damping is obtained by strip integration across the fin span. A strip
    at radius ``y`` sees a local incidence ``p y / V``, so::

        Clp = -2 N a_0 / (d^2 A_ref) * integral y^2 c(y) dy

    where ``a_0`` is the fin section lift slope implied by the Barrowman fin
    normal-force slope, keeping the two models consistent with one another.

    Parameters
    ----------
    rocket:
        The design.
    t:
        Time since ignition [s], which fixes the centre of gravity.
    mach:
        Mach number [-].

    Returns
    -------
    DampingDerivatives
        The four derivatives.
    """
    result = barrowman_analysis(rocket, mach=mach)
    cg = rocket.cg_at(t)
    d = rocket.reference_diameter
    a_ref = rocket.reference_area

    # Pitch and yaw damping from every lifting surface.
    cmq = -2.0 * sum(
        c.cn_alpha * ((c.centre_of_pressure - cg) / d) ** 2 for c in result.contributions
    )

    # Roll: strip-integrate over the exposed fin span.
    fins = rocket.fins
    n_strips = 100
    s = np.linspace(0.0, fins.span, n_strips)
    frac = s / fins.span
    chord = fins.root_chord + frac * (fins.tip_chord - fins.root_chord)
    y = fins.body_radius + s

    # Section lift slope implied by the Barrowman fin CN_alpha, per fin,
    # referenced to fin planform area rather than body area.
    fin_cna_ref = fins.cn_alpha(rocket.reference_radius) * prandtl_glauert(mach)
    a0 = fin_cna_ref * a_ref / (fins.count * fins.area_single)

    integral_y2c = float(np.trapezoid(y * y * chord, s))
    integral_yc = float(np.trapezoid(y * chord, s))

    clp = -2.0 * fins.count * a0 * integral_y2c / (d * d * a_ref)
    cl_delta = fins.count * a0 * integral_yc / (a_ref * d)

    return DampingDerivatives(cmq=cmq, cnr=cmq, clp=clp, cl_delta=cl_delta)


def stability_verdict(margin_calibres: float) -> str:
    """Return a plain-language verdict on a static margin.

    Parameters
    ----------
    margin_calibres:
        Static margin [calibres].

    Returns
    -------
    str
        One of ``"unstable"``, ``"marginal"``, ``"stable"`` or
        ``"over-stable"``, using the conventional 1.0 to 2.5 calibre band.
    """
    if margin_calibres < 0.0:
        return "unstable"
    if margin_calibres < MIN_STATIC_MARGIN_CALIBRES:
        return "marginal"
    if margin_calibres <= MAX_STATIC_MARGIN_CALIBRES:
        return "stable"
    return "over-stable"
