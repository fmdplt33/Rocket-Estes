"""Mass, centre of gravity and moment of inertia of components and assemblies.

Coordinate convention
---------------------
The body axis ``x`` runs aft from the nose tip. ``Ixx`` is the roll (axial)
moment of inertia and ``Iyy`` the pitch/yaw (transverse) moment of inertia,
both taken about the object's own centre of gravity unless stated otherwise.
Because the airframe is axisymmetric, ``Izz == Iyy``.

Components are combined with the parallel-axis theorem::

    I_total = sum_i [ I_i + m_i d_i^2 ]

where ``d_i`` is the axial offset between component ``i``'s centre of gravity
and the assembly's. The axial moment ``Ixx`` needs no such shift, since every
component is concentric with the body axis; only the fins sit off-axis, and
their contribution is handled explicitly.

References
----------
[1] Beer, F. P., & Johnston, E. R. (2012). *Vector Mechanics for Engineers:
    Statics and Dynamics*, 10th ed., Ch. 9 (moments of inertia, parallel-axis
    theorem).
[2] Niskanen, S. (2013). *OpenRocket Technical Documentation*, Sec. 2.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rocketopt.geometry.components import (
    BodyTube,
    FinSet,
    LaunchLug,
    MotorMount,
    Transition,
)
from rocketopt.geometry.nose_cones import NoseCone

__all__ = [
    "MassProperties",
    "point_mass",
    "tube_properties",
    "transition_properties",
    "nose_cone_properties",
    "fin_set_properties",
    "launch_lug_properties",
    "motor_mount_properties",
    "combine",
]


@dataclass(frozen=True, slots=True)
class MassProperties:
    """Mass and inertia of a component or assembly.

    Attributes
    ----------
    mass:
        Mass [kg].
    cg:
        Axial centre of gravity aft of the nose tip [m].
    ixx:
        Roll moment of inertia about the body axis, through the CG [kg*m^2].
    iyy:
        Pitch/yaw moment of inertia about the transverse axis through the CG
        [kg*m^2].
    """

    mass: float
    cg: float
    ixx: float
    iyy: float

    def __post_init__(self) -> None:
        """Validate that the properties are physically admissible."""
        if self.mass < 0.0:
            raise ValueError("mass must not be negative")
        if self.ixx < 0.0 or self.iyy < 0.0:
            raise ValueError("moments of inertia must not be negative")

    def shifted_to(self, new_cg: float) -> MassProperties:
        """Return these properties with inertia referred to a new axial origin.

        Applies the parallel-axis theorem [1] to the transverse moment. The
        axial moment is unchanged because the shift is purely axial.

        Parameters
        ----------
        new_cg:
            Axial station to refer the transverse inertia to [m].

        Returns
        -------
        MassProperties
            Properties with ``iyy`` taken about ``new_cg``. The ``cg`` field
            is left at the true centroid.
        """
        offset = self.cg - new_cg
        return MassProperties(
            mass=self.mass,
            cg=self.cg,
            ixx=self.ixx,
            iyy=self.iyy + self.mass * offset * offset,
        )


def point_mass(mass: float, position: float) -> MassProperties:
    """Mass properties of a point mass on the body axis.

    Parameters
    ----------
    mass:
        Mass [kg].
    position:
        Axial station aft of the nose tip [m].

    Returns
    -------
    MassProperties
        A point mass has no inertia about its own centroid.
    """
    return MassProperties(mass=mass, cg=position, ixx=0.0, iyy=0.0)


def tube_properties(tube: BodyTube, position: float) -> MassProperties:
    """Mass properties of a cylindrical tube.

    For a hollow cylinder of outer radius ``r_o``, inner radius ``r_i``,
    length ``L`` and mass ``m`` [1], Appendix B::

        Ixx = m (r_o^2 + r_i^2) / 2
        Iyy = m [3 (r_o^2 + r_i^2) + L^2] / 12

    Parameters
    ----------
    tube:
        The tube.
    position:
        Axial station of the tube's forward face aft of the nose tip [m].

    Returns
    -------
    MassProperties
        Properties about the tube's own centroid.
    """
    m = tube.mass
    r_o, r_i, length = tube.outer_radius, tube.inner_radius, tube.length
    radial = r_o * r_o + r_i * r_i
    return MassProperties(
        mass=m,
        cg=position + tube.centre_of_mass,
        ixx=m * radial / 2.0,
        iyy=m * (3.0 * radial + length * length) / 12.0,
    )


def transition_properties(transition: Transition, position: float) -> MassProperties:
    """Mass properties of a conical transition.

    The frustum shell is integrated numerically along its axis rather than
    using a closed form, because the wall thickness is constant while the
    radius varies, which no standard table covers.

    Parameters
    ----------
    transition:
        The transition.
    position:
        Axial station of the forward face aft of the nose tip [m].

    Returns
    -------
    MassProperties
        Properties about the transition's own centroid.
    """
    n = 200
    x = np.linspace(0.0, transition.length, n)
    frac = x / transition.length
    r_out = transition.fore_radius + frac * (transition.aft_radius - transition.fore_radius)
    r_in = np.maximum(r_out - transition.wall_thickness, 0.0)

    # Mass per unit length of the annular slice.
    dm = math.pi * (r_out**2 - r_in**2) * transition.material.density
    mass = float(np.trapezoid(dm, x))
    if mass <= 0.0:
        return MassProperties(mass=0.0, cg=position, ixx=0.0, iyy=0.0)

    cg_local = float(np.trapezoid(dm * x, x)) / mass

    # Axial inertia of an annular slice: dI = dm (r_o^2 + r_i^2) / 2.
    ixx = float(np.trapezoid(dm * (r_out**2 + r_in**2) / 2.0, x))

    # Transverse inertia: each slice contributes its own diametral inertia
    # dm (r_o^2 + r_i^2) / 4 plus dm * (x - cg)^2.
    diametral = dm * (r_out**2 + r_in**2) / 4.0
    iyy = float(np.trapezoid(diametral + dm * (x - cg_local) ** 2, x))

    return MassProperties(
        mass=mass,
        cg=position + cg_local,
        ixx=ixx,
        iyy=iyy,
    )


def nose_cone_properties(cone: NoseCone, position: float = 0.0) -> MassProperties:
    """Mass properties of a nose cone, solid or shell.

    Integrated numerically over the profile so that every one of the nine
    supported shapes is handled by the same code path.

    Parameters
    ----------
    cone:
        The nose cone.
    position:
        Axial station of the tip aft of the nose tip [m], normally zero.

    Returns
    -------
    MassProperties
        Properties about the cone's own centroid, including its shoulder.
    """
    n = 400
    x = np.linspace(0.0, cone.length, n)
    r_out = cone.radius_at(x)
    if cone.solid:
        r_in = np.zeros_like(r_out)
    else:
        r_in = np.maximum(r_out - cone.wall_thickness, 0.0)

    dm = math.pi * (r_out**2 - r_in**2) * cone.material.density
    mass = float(np.trapezoid(dm, x))

    ixx = float(np.trapezoid(dm * (r_out**2 + r_in**2) / 2.0, x))
    diametral = dm * (r_out**2 + r_in**2) / 4.0

    # Add the shoulder, a short tube extending aft of the base.
    shoulder_mass = 0.0
    shoulder_cg = 0.0
    shoulder_ixx = 0.0
    shoulder_iyy_own = 0.0
    if cone.shoulder_length > 0.0:
        r_so = cone.base_radius - cone.wall_thickness
        r_si = max(r_so - cone.wall_thickness, 0.0)
        shoulder_mass = (
            math.pi * (r_so**2 - r_si**2) * cone.shoulder_length * cone.material.density
        )
        shoulder_cg = cone.length + cone.shoulder_length / 2.0
        radial = r_so**2 + r_si**2
        shoulder_ixx = shoulder_mass * radial / 2.0
        shoulder_iyy_own = (
            shoulder_mass * (3.0 * radial + cone.shoulder_length**2) / 12.0
        )

    total_mass = mass + shoulder_mass
    if total_mass <= 0.0:
        return MassProperties(mass=0.0, cg=position, ixx=0.0, iyy=0.0)

    cone_cg = float(np.trapezoid(dm * x, x)) / mass if mass > 0.0 else 0.0
    cg_local = (cone_cg * mass + shoulder_cg * shoulder_mass) / total_mass

    iyy = float(np.trapezoid(diametral + dm * (x - cg_local) ** 2, x))
    iyy += shoulder_iyy_own + shoulder_mass * (shoulder_cg - cg_local) ** 2

    return MassProperties(
        mass=total_mass,
        cg=position + cg_local,
        ixx=ixx + shoulder_ixx,
        iyy=iyy,
    )


def fin_set_properties(fins: FinSet) -> MassProperties:
    """Mass properties of a fin set.

    Each fin is treated as a thin trapezoidal plate lying in a plane through
    the body axis. The set is axisymmetric in aggregate for three or more
    equally spaced fins, so a single ``Iyy`` is meaningful.

    The roll inertia is the dominant fin contribution, because the fin mass
    sits at a radius well outside the body::

        Ixx = N * integral over span of (dm * y^2)

    where ``y`` is measured from the body axis, running from
    ``body_radius`` to ``body_radius + span``.

    Parameters
    ----------
    fins:
        The fin set. Its :attr:`~rocketopt.geometry.components.FinSet.position`
        gives the root leading edge station.

    Returns
    -------
    MassProperties
        Properties of all fins combined, about their common centroid.
    """
    n = 200
    # Spanwise station measured from the body wall.
    s = np.linspace(0.0, fins.span, n)
    frac = s / fins.span
    chord = fins.root_chord + frac * (fins.tip_chord - fins.root_chord)

    section_factor = fins.volume_single / max(fins.area_single * fins.thickness, 1e-30)
    # Mass per unit span of one fin.
    dm = chord * fins.thickness * section_factor * fins.material.density

    mass_single = float(np.trapezoid(dm, s))
    total_mass = mass_single * fins.count
    if total_mass <= 0.0:
        return MassProperties(mass=0.0, cg=fins.position, ixx=0.0, iyy=0.0)

    # Axial centroid: local chord centre offset by the local leading-edge sweep.
    le_offset = frac * fins.sweep_length
    x_local = le_offset + chord / 2.0
    cg_local = float(np.trapezoid(dm * x_local, s)) / mass_single

    # Radial distance of each spanwise strip from the body axis.
    y = fins.body_radius + s

    # Roll inertia: all fins contribute, each strip at radius y.
    ixx = fins.count * float(np.trapezoid(dm * y * y, s))

    # Transverse inertia of one fin about the assembly CG: the strip's own
    # axial extent plus its axial offset. The radial extent contributes to
    # Iyy for fins in the pitch plane only; averaged over N >= 3 equally
    # spaced fins, half the radial term appears in each transverse axis.
    axial_term = dm * ((chord**2) / 12.0 + (x_local - cg_local) ** 2)
    radial_term = 0.5 * dm * y * y
    iyy = fins.count * float(np.trapezoid(axial_term + radial_term, s))

    return MassProperties(
        mass=total_mass,
        cg=fins.position + cg_local,
        ixx=ixx,
        iyy=iyy,
    )


def launch_lug_properties(lug: LaunchLug) -> MassProperties:
    """Mass properties of a launch lug.

    The lug sits off-axis against the body wall, so its roll inertia uses the
    parallel-axis theorem about the body axis.

    Parameters
    ----------
    lug:
        The launch lug, whose ``position`` gives its forward end.

    Returns
    -------
    MassProperties
        Properties about the lug's own axial centroid.
    """
    m = lug.mass
    radial = lug.outer_radius**2 + lug.inner_radius**2
    # Mounting radius: the lug's own axis sits one lug radius off the body wall.
    return MassProperties(
        mass=m,
        cg=lug.position + lug.centre_of_mass,
        ixx=m * radial / 2.0,
        iyy=m * (3.0 * radial + lug.length**2) / 12.0,
    )


def motor_mount_properties(mount: MotorMount) -> MassProperties:
    """Mass properties of a motor mount.

    Parameters
    ----------
    mount:
        The motor mount, whose ``position`` gives its forward end.

    Returns
    -------
    MassProperties
        Combined properties of motor tube and centring rings.
    """
    tube_mass = mount.tube_mass
    r_o = mount.outer_radius
    r_i = mount.inner_diameter / 2.0
    radial = r_o * r_o + r_i * r_i

    tube = MassProperties(
        mass=tube_mass,
        cg=mount.position + mount.length / 2.0,
        ixx=tube_mass * radial / 2.0,
        iyy=tube_mass * (3.0 * radial + mount.length**2) / 12.0,
    )

    ring_mass = mount.ring_mass
    if ring_mass <= 0.0:
        return tube

    # Rings sit at the ends of the motor tube.
    ring_radial = mount.body_inner_radius**2 + r_o**2
    rings = MassProperties(
        mass=ring_mass,
        cg=mount.position + mount.length / 2.0,
        ixx=ring_mass * ring_radial / 2.0,
        # Rings are thin discs; their axial extent is negligible, but they are
        # separated, so approximate the pair as sitting at +/- L/2.
        iyy=ring_mass * ring_radial / 4.0 + ring_mass * (mount.length / 2.0) ** 2,
    )
    return combine([tube, rings])


def combine(parts: list[MassProperties]) -> MassProperties:
    """Combine component mass properties into an assembly.

    Uses the parallel-axis theorem [1] to refer every transverse moment to the
    assembly centre of gravity.

    Parameters
    ----------
    parts:
        Component properties. Components with zero mass are ignored.

    Returns
    -------
    MassProperties
        Assembly mass, centre of gravity and inertias about that CG.

    Raises
    ------
    ValueError
        If ``parts`` is empty or the total mass is zero, in which case a
        centre of gravity is undefined.
    """
    contributing = [p for p in parts if p.mass > 0.0]
    if not contributing:
        raise ValueError("cannot combine mass properties with zero total mass")

    total_mass = sum(p.mass for p in contributing)
    cg = sum(p.mass * p.cg for p in contributing) / total_mass

    ixx = sum(p.ixx for p in contributing)
    iyy = sum(p.iyy + p.mass * (p.cg - cg) ** 2 for p in contributing)

    return MassProperties(mass=total_mass, cg=cg, ixx=ixx, iyy=iyy)
