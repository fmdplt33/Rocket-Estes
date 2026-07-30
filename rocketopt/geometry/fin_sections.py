"""Fin cross-section geometry: thickness distributions and section outlines.

A fin is specified by its planform (root chord, tip chord, span, sweep) and by
a *section* - the shape of its cross-section normal to the span. The section is
what separates a sanded aerofoil from a square-cut plate, and it is the part
most often lost when a design is exported to CAD as a simple extrusion.

This module is the single definition of that geometry. Everything that needs a
fin section takes it from here:

* :class:`~rocketopt.geometry.components.FinSet` computes its material volume
  by integrating :func:`section_area` over the span, so the reported fin mass
  is the mass of the section that will actually be built.
* :mod:`rocketopt.cad.parts` lofts :func:`section_outline` from root to tip to
  build the exported solid, so the STL carries the true leading edge, trailing
  edge and thickness distribution.

Section definitions
-------------------
``xi`` denotes fractional chord, 0 at the leading edge and 1 at the trailing
edge. ``t`` is the maximum section thickness and ``c`` the local chord. Each
family is defined by its half-thickness ``y(xi)``, and every family reaches
exactly ``t / 2`` at its thickest point.

Square
    ``y = t / 2`` throughout: a flat plate with square-cut edges. Blunt at both
    the leading and the trailing edge.
Rounded
    A flat plate whose leading and trailing edges are turned to semicircles of
    radius ``t / 2``. The straight middle is untouched, so
    ``y = t / 2`` for ``t / 2 <= xi c <= c - t / 2``.
Aerofoil
    The NACA four-digit symmetric thickness distribution [1]::

        y = 5 t (0.2969 sqrt(xi) - 0.1260 xi - 0.3516 xi^2
                 + 0.2843 xi^3 - 0.1036 xi^4)

    normalised so that its maximum is exactly ``t / 2``. The ``-0.1036``
    quartic coefficient is the closed-trailing-edge variant, which gives a
    genuinely sharp trailing edge instead of the 0.21% gap left by the original
    ``-0.1015``. A rounded leading edge of finite radius tapering to a sharp
    trailing edge - the section a builder produces by sanding.
Double wedge
    ``y = t xi`` forward of mid-chord and ``y = t (1 - xi)`` aft: a symmetric
    diamond with sharp leading and trailing edges, lowest wave drag of the four
    at supersonic speed.

Degenerate chords
-----------------
Where the local chord falls below the thickness - the last fraction of a
millimetre of a delta fin, for instance - a square or rounded section would
self-intersect. The section thickness is therefore clamped to the local chord,
which turns the rounded section smoothly into a circle of the chord's diameter
and keeps the lofted solid valid all the way to a sharp tip.

References
----------
[1] Jacobs, E. N., Ward, K. E., & Pinkerton, R. M. (1933). *The Characteristics
    of 78 Related Airfoil Sections from Tests in the Variable-Density Wind
    Tunnel*. NACA Report 460.
[2] Hoerner, S. F. (1965). *Fluid-Dynamic Drag*, Ch. 6.
[3] Abbott, I. H., & von Doenhoff, A. E. (1959). *Theory of Wing Sections*,
    Sec. 6.4.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Final

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "FinAirfoil",
    "half_thickness",
    "peak_station",
    "section_area",
    "section_area_ratio",
    "section_outline",
    "swept_volume",
]


class FinAirfoil(str, Enum):
    """Fin cross-section profile.

    The drag multipliers quoted in
    :meth:`~rocketopt.geometry.components.FinSet.profile_drag_factor` are
    relative to a square-edged flat plate of the same thickness, following
    Hoerner [2], Ch. 6.
    """

    SQUARE = "square"
    """Flat plate, sharp square edges. Simplest to make, highest drag."""

    ROUNDED = "rounded"
    """Flat plate with leading and trailing edges rounded to a semicircle."""

    AIRFOIL = "airfoil"
    """Rounded leading edge tapering to a sharp trailing edge."""

    DOUBLE_WEDGE = "double_wedge"
    """Symmetric diamond section; lowest wave drag at supersonic speed."""

    @property
    def label(self) -> str:
        """Human-readable name."""
        return self.name.replace("_", " ").title()

    @property
    def has_blunt_leading_edge(self) -> bool:
        """Whether the section has finite thickness at the leading edge."""
        return self is FinAirfoil.SQUARE

    @property
    def has_blunt_trailing_edge(self) -> bool:
        """Whether the section has finite thickness at the trailing edge."""
        return self is FinAirfoil.SQUARE


_NACA_COEFFICIENTS: Final[tuple[float, float, float, float, float]] = (
    0.2969,
    -0.1260,
    -0.3516,
    0.2843,
    -0.1036,
)
"""NACA four-digit thickness coefficients, closed-trailing-edge variant [1]."""


def _naca_shape(xi: NDArray[np.float64]) -> NDArray[np.float64]:
    """Evaluate the NACA thickness distribution, normalised to a peak of 1.

    Parameters
    ----------
    xi:
        Fractional chord in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        Half-thickness as a fraction of the maximum half-thickness, so that the
        peak value is exactly 1.
    """
    a0, a1, a2, a3, a4 = _NACA_COEFFICIENTS
    raw = (
        a0 * np.sqrt(xi)
        + a1 * xi
        + a2 * xi**2
        + a3 * xi**3
        + a4 * xi**4
    )
    return np.maximum(raw, 0.0) / _NACA_PEAK


def _naca_peak() -> tuple[float, float]:
    """Return the ``(station, value)`` of the NACA polynomial's maximum.

    The polynomial has a single interior maximum, near 30% chord. Its derivative
    is solved for rather than sampled, so the normalisation is exact to machine
    precision and the station can be added to the sampled outline - otherwise a
    section's thickest point falls between two samples and the exported solid is
    a shade thinner than specified.
    """
    from scipy.optimize import brentq

    a0, a1, a2, a3, a4 = _NACA_COEFFICIENTS

    def raw(xi: float) -> float:
        return a0 * math.sqrt(xi) + a1 * xi + a2 * xi**2 + a3 * xi**3 + a4 * xi**4

    def slope(xi: float) -> float:
        return (
            0.5 * a0 / math.sqrt(xi)
            + a1
            + 2.0 * a2 * xi
            + 3.0 * a3 * xi**2
            + 4.0 * a4 * xi**3
        )

    station = float(brentq(slope, 0.05, 0.90, xtol=1e-15))
    return station, raw(station)


_NACA_PEAK_STATION, _NACA_PEAK = _naca_peak()
"""Fractional chord of the NACA section's thickest point, and the peak value.

The peak is close to 0.1 by construction of the four-digit family; it is
resolved exactly so that the generated section reaches the specified thickness
rather than 99.6% of it.
"""


def peak_station(airfoil: FinAirfoil) -> float:
    """Return the fractional chord at which a section is thickest.

    Parameters
    ----------
    airfoil:
        Section family.

    Returns
    -------
    float
        Fractional chord of the thickest point. The square section is of uniform
        thickness and the rounded section has a flat middle, so mid-chord is
        returned for both - it is thickest there as much as anywhere.
    """
    if airfoil is FinAirfoil.AIRFOIL:
        return _NACA_PEAK_STATION
    return 0.5

_NACA_AREA_RATIO: Final[float] = (
    2.0 * _NACA_COEFFICIENTS[0] / 3.0
    + _NACA_COEFFICIENTS[1] / 2.0
    + _NACA_COEFFICIENTS[2] / 3.0
    + _NACA_COEFFICIENTS[3] / 4.0
    + _NACA_COEFFICIENTS[4] / 5.0
) / _NACA_PEAK
"""Area of the NACA section divided by ``t * c``, about 0.687.

The section area is ``2 * (t / 2) * c * mean(shape)``, so the ratio is just the
mean of the normalised shape. The closed-trailing-edge coefficients sum to zero
at ``xi = 1``, so the polynomial is non-negative throughout ``[0, 1]`` and the
integral of the normalised shape is exact term by term.
"""


def _effective_thickness(thickness: float, chord: float) -> float:
    """Return the section thickness clamped to fit inside the local chord.

    Parameters
    ----------
    thickness:
        Nominal maximum thickness [m].
    chord:
        Local chord [m].

    Returns
    -------
    float
        ``min(thickness, chord)``, never negative. A section thicker than its
        chord cannot be drawn without self-intersecting; clamping keeps the
        geometry valid in the vanishing chord at the tip of a delta fin.
    """
    return max(min(thickness, chord), 0.0)


def half_thickness(
    airfoil: FinAirfoil,
    xi: NDArray[np.float64] | float,
    *,
    thickness: float,
    chord: float,
) -> NDArray[np.float64]:
    """Half-thickness of a fin section at fractional chord ``xi``.

    Parameters
    ----------
    airfoil:
        Section family.
    xi:
        Fractional chord, 0 at the leading edge and 1 at the trailing edge.
        Values outside ``[0, 1]`` are clamped.
    thickness:
        Maximum section thickness [m].
    chord:
        Local chord [m].

    Returns
    -------
    numpy.ndarray
        Half-thickness at each station [m]. The maximum over ``xi`` equals half
        the effective thickness exactly.

    Raises
    ------
    ValueError
        If the thickness is not positive.
    """
    if thickness <= 0.0:
        raise ValueError("section thickness must be positive")

    stations = np.clip(np.atleast_1d(np.asarray(xi, dtype=np.float64)), 0.0, 1.0)
    t = _effective_thickness(thickness, chord)
    if t <= 0.0 or chord <= 0.0:
        return np.zeros_like(stations)

    match airfoil:
        case FinAirfoil.SQUARE:
            return np.full_like(stations, t / 2.0)

        case FinAirfoil.ROUNDED:
            r = t / 2.0
            x = stations * chord
            y = np.full_like(stations, r)
            # Semicircular leading edge, centred one radius aft of the nose.
            nose = x < r
            y[nose] = np.sqrt(np.maximum(r * r - (r - x[nose]) ** 2, 0.0))
            # Semicircular trailing edge, mirrored about the mid-chord.
            tail = x > chord - r
            d = chord - x[tail]
            y[tail] = np.sqrt(np.maximum(r * r - (r - d) ** 2, 0.0))
            return y

        case FinAirfoil.AIRFOIL:
            return (t / 2.0) * _naca_shape(stations)

        case FinAirfoil.DOUBLE_WEDGE:
            return t * np.minimum(stations, 1.0 - stations)

        case _:  # pragma: no cover - the Enum is exhaustive
            raise ValueError(f"unhandled fin section {airfoil}")


def section_area_ratio(
    airfoil: FinAirfoil, *, thickness: float, chord: float
) -> float:
    """Section area divided by ``thickness * chord`` [-].

    Every family has a closed-form ratio, so no numerical integration is
    needed. The rounded section is the only one whose ratio depends on the
    thickness-to-chord ratio, because the material removed by rounding the two
    edges is a fixed area rather than a fixed fraction.

    Parameters
    ----------
    airfoil:
        Section family.
    thickness:
        Maximum section thickness [m].
    chord:
        Local chord [m].

    Returns
    -------
    float
        Area ratio in ``(0, 1]``. Zero for a vanishing chord.

    Raises
    ------
    ValueError
        If the thickness is not positive.
    """
    if thickness <= 0.0:
        raise ValueError("section thickness must be positive")
    if chord <= 0.0:
        return 0.0

    t = _effective_thickness(thickness, chord)
    match airfoil:
        case FinAirfoil.SQUARE:
            return 1.0
        case FinAirfoil.ROUNDED:
            # A rectangle t by c, less the corner area left outside each
            # semicircular edge: 2 r^2 - pi r^2 / 2 per edge, with r = t / 2.
            return 1.0 - (t / chord) * (1.0 - math.pi / 4.0)
        case FinAirfoil.AIRFOIL:
            return _NACA_AREA_RATIO
        case FinAirfoil.DOUBLE_WEDGE:
            return 0.5
        case _:  # pragma: no cover - the Enum is exhaustive
            raise ValueError(f"unhandled fin section {airfoil}")


def section_area(airfoil: FinAirfoil, *, thickness: float, chord: float) -> float:
    """Cross-sectional area of a fin section [m^2].

    Parameters
    ----------
    airfoil:
        Section family.
    thickness:
        Maximum section thickness [m].
    chord:
        Local chord [m].

    Returns
    -------
    float
        Enclosed area of the section [m^2].
    """
    if chord <= 0.0:
        return 0.0
    t = _effective_thickness(thickness, chord)
    ratio = section_area_ratio(airfoil, thickness=thickness, chord=chord)
    return ratio * t * chord


def swept_volume(
    airfoil: FinAirfoil,
    *,
    root_chord: float,
    tip_chord: float,
    span: float,
    thickness: float,
) -> float:
    """Material volume of one linearly tapered fin [m^3].

    The section area is integrated along the span, which is the volume the
    exported solid encloses. For the square, aerofoil and double-wedge families
    the area is proportional to the local chord and the integral is exactly the
    planform area times the constant area-per-unit-chord; the rounded family
    carries an additional term, and the thickness clamp near a vanishing tip
    chord makes all four slightly non-linear. The integral is therefore
    evaluated numerically, over enough stations that the result is exact to
    well below a milligram of material.

    Parameters
    ----------
    airfoil:
        Section family.
    root_chord:
        Chord at the root [m].
    tip_chord:
        Chord at the tip [m].
    span:
        Exposed semi-span [m].
    thickness:
        Maximum section thickness [m].

    Returns
    -------
    float
        Volume of one fin [m^3], excluding any root fillet.
    """
    if span <= 0.0 or root_chord <= 0.0:
        return 0.0

    eta = np.linspace(0.0, 1.0, _SPAN_INTEGRATION_STATIONS)
    chords = root_chord + (tip_chord - root_chord) * eta
    areas = np.array(
        [section_area(airfoil, thickness=thickness, chord=float(c)) for c in chords]
    )
    return float(np.trapezoid(areas, eta * span))


_SPAN_INTEGRATION_STATIONS: Final[int] = 201
"""Spanwise stations used by :func:`swept_volume`.

The integrand is linear in the local chord for three of the four families, and
its only kink is where the thickness clamp engages very close to a vanishing
tip, so 201 stations put the trapezoidal error far below a milligram.
"""


def section_outline(
    airfoil: FinAirfoil,
    *,
    chord: float,
    thickness: float,
    samples: int = 61,
) -> list[tuple[float, float]]:
    """Return the closed outline of a fin section.

    The outline runs counter-clockwise in the ``(chord, thickness)`` plane:
    forward along the upper surface from the leading edge to the trailing edge,
    then back along the lower surface. Blunt edges contribute two distinct
    points and so close the loop through a straight edge; sharp edges
    contribute a single shared point.

    Stations are cosine-spaced, which concentrates points where the curvature
    is highest - at the leading edge - and so resolves the nose radius with far
    fewer points than uniform spacing would need.

    Parameters
    ----------
    airfoil:
        Section family.
    chord:
        Local chord [m]. A chord of zero returns the single point ``(0, 0)``,
        the degenerate section at the tip of a delta fin.
    thickness:
        Maximum section thickness [m].
    samples:
        Number of chordwise stations, at least 5.

    Returns
    -------
    list of tuple
        ``(x, y)`` vertices [m] with ``x`` measured aft of the leading edge and
        ``y`` either side of the chord line. The first vertex is not repeated
        at the end; the loop closes implicitly.

    Raises
    ------
    ValueError
        If ``samples`` is below 5.
    """
    if samples < 5:
        raise ValueError("a section outline needs at least 5 stations")
    if chord <= 0.0:
        return [(0.0, 0.0)]

    # Cosine spacing: dense at the leading edge, where the radius is smallest.
    beta = np.linspace(0.0, math.pi, samples)
    xi = 0.5 * (1.0 - np.cos(beta))

    # Add the thickest station unless the grid already lands on it, so the
    # outline reaches the full specified thickness instead of falling between two
    # samples. Inserting a station the grid already has would put two coincident
    # vertices in the outline, and a lofted surface through a zero-length edge
    # has a hole in it.
    peak = peak_station(airfoil)
    if float(np.abs(xi - peak).min()) > _STATION_TOLERANCE:
        xi = np.sort(np.append(xi, peak))

    y = half_thickness(airfoil, xi, thickness=thickness, chord=chord)
    x = xi * chord

    upper = [(float(xv), float(yv)) for xv, yv in zip(x, y, strict=True)]
    lower = [(float(xv), -float(yv)) for xv, yv in zip(x, y, strict=True)]

    # Share the leading- and trailing-edge points when the section closes to a
    # sharp edge there; a duplicated vertex would create a zero-length edge.
    start = 1 if abs(upper[0][1]) < _SHARP_EDGE_TOLERANCE else 0
    end = len(lower) - 1 if abs(upper[-1][1]) < _SHARP_EDGE_TOLERANCE else len(lower)

    return upper + list(reversed(lower[start:end]))


_SHARP_EDGE_TOLERANCE: Final[float] = 1e-12
"""Half-thickness below which an edge counts as sharp, in metres."""

_STATION_TOLERANCE: Final[float] = 1e-9
"""Fractional-chord distance below which two stations count as the same."""
