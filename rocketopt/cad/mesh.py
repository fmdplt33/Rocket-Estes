"""Triangle meshes, STL output and the watertightness checks that back them.

A slicer will accept almost any file it is handed and then fail, quietly or
loudly, on geometry that is not a closed solid. This module therefore treats
validity as a property that is *proved* before a file is written rather than
hoped for afterwards:

* Surfaces are built by revolving or lofting closed profiles, so no boolean
  operations are involved and no seams have to be stitched. The seam ring of a
  revolve shares its vertices by index rather than duplicating them.
* :meth:`TriangleMesh.validate` checks the three properties a printable solid
  must have - every edge used by exactly two triangles (watertight), every
  directed edge used exactly once (consistently oriented), and a positive
  enclosed volume (normals outward) - and
  :func:`rocketopt.cad.exporters.export_stl_parts` refuses to write a mesh that
  fails any of them.

Everything here is plain NumPy, so STL export does not need CadQuery. The
CadQuery path in :mod:`rocketopt.cad.exporters` builds the same profiles as
BREP solids for STEP output.

Units
-----
Meshes carry whatever units their profiles were built in.
:mod:`rocketopt.cad.parts` builds in millimetres, which is what STL consumers
assume.

References
----------
[1] 3D Systems (1989). *StereoLithography Interface Specification*.
[2] Mirtich, B. (1996). *Fast and Accurate Computation of Polyhedral Mass
    Properties*. Journal of Graphics Tools, 1(2), 31-50.
"""

from __future__ import annotations

import itertools
import math
import struct
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from numpy.typing import NDArray

from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "TriangleMesh",
    "dedupe_polyline",
    "loft",
    "revolve_profile",
    "write_stl",
]

_DEGENERATE_AREA: Final[float] = 1e-14
"""Triangle area below which a facet is discarded as degenerate.

In square millimetres, so this drops facets narrower than about 10 nanometres -
the collapsed quads at a cone apex or a sharp fin tip - while keeping every
facet a slicer could resolve.
"""

_AXIS_TOLERANCE: Final[float] = 1e-9
"""Radius below which a revolved profile vertex is taken to be on the axis.

A picometre in millimetre units. Analytic profiles do not always land exactly on
zero - a tangent ogive tip is the difference of two nearly equal large numbers
and lands a few times 1e-14 away - and a residual radius would sweep a ring of
sliver facets instead of collapsing to a single apex vertex.
"""


@dataclass(frozen=True)
class TriangleMesh:
    """An indexed triangle mesh.

    Attributes
    ----------
    vertices:
        ``(n, 3)`` array of vertex coordinates.
    faces:
        ``(m, 3)`` array of vertex indices, each triangle wound
        counter-clockwise seen from outside the solid.
    """

    vertices: NDArray[np.float64]
    faces: NDArray[np.int64]

    def __post_init__(self) -> None:
        """Validate the array shapes and index range."""
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must be an (n, 3) array")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            raise ValueError("faces must be an (m, 3) array")
        if len(self.faces) and (
            self.faces.min() < 0 or self.faces.max() >= len(self.vertices)
        ):
            raise ValueError("a face references a vertex that does not exist")

    # -- Basic measures ------------------------------------------------------

    @property
    def vertex_count(self) -> int:
        """Number of vertices."""
        return len(self.vertices)

    @property
    def triangle_count(self) -> int:
        """Number of triangles."""
        return len(self.faces)

    @property
    def bounds(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Axis-aligned bounding box as ``(minimum, maximum)`` corners."""
        if not len(self.vertices):  # pragma: no cover - guarded by callers
            zero = np.zeros(3)
            return zero, zero
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def size(self) -> NDArray[np.float64]:
        """Bounding box extent along each axis."""
        low, high = self.bounds
        return high - low

    def radial_extent(
        self, z_low: float = -math.inf, z_high: float = math.inf
    ) -> tuple[float, float]:
        """Smallest and largest radius about the Z axis within a slab.

        Measuring a printed part's fit means measuring the geometry that will
        actually be sliced, so the mating diameters are read back off the mesh
        rather than recomputed from the design.

        Parameters
        ----------
        z_low, z_high:
            Bounds of the slab to measure, inclusive.

        Returns
        -------
        tuple of float
            ``(minimum, maximum)`` radius of the vertices inside the slab.

        Raises
        ------
        ValueError
            If no vertex lies inside the slab.
        """
        inside = self.vertices[
            (self.vertices[:, 2] >= z_low) & (self.vertices[:, 2] <= z_high)
        ]
        if not len(inside):
            raise ValueError(
                f"no vertex lies between z={z_low:.6g} and z={z_high:.6g}"
            )
        radii = np.hypot(inside[:, 0], inside[:, 1])
        return float(radii.min()), float(radii.max())

    def _triangles(self) -> tuple[NDArray[np.float64], ...]:
        """Return the three corner arrays of every triangle."""
        return (
            self.vertices[self.faces[:, 0]],
            self.vertices[self.faces[:, 1]],
            self.vertices[self.faces[:, 2]],
        )

    @property
    def face_normals(self) -> NDArray[np.float64]:
        """Unit normal of every triangle, following its winding."""
        a, b, c = self._triangles()
        normals = np.asarray(np.cross(b - a, c - a), dtype=np.float64)
        lengths = np.linalg.norm(normals, axis=1)
        lengths[lengths == 0.0] = 1.0
        unit: NDArray[np.float64] = normals / lengths[:, None]
        return unit

    @property
    def surface_area(self) -> float:
        """Total area of every triangle."""
        a, b, c = self._triangles()
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())

    @property
    def volume(self) -> float:
        """Signed enclosed volume.

        The divergence theorem applied to each triangle, summing the signed
        volumes of the tetrahedra it forms with the origin [2]. Positive when
        the winding puts the normals outward, which is the sign convention STL
        consumers expect.
        """
        a, b, c = self._triangles()
        return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)

    # -- Validity ------------------------------------------------------------

    def _edge_counts(self) -> tuple[Counter[tuple[int, int]], Counter[tuple[int, int]]]:
        """Return counts of directed and undirected edges."""
        directed: Counter[tuple[int, int]] = Counter()
        undirected: Counter[tuple[int, int]] = Counter()
        for i, j, k in self.faces.tolist():
            for a, b in ((i, j), (j, k), (k, i)):
                directed[(a, b)] += 1
                undirected[(a, b) if a < b else (b, a)] += 1
        return directed, undirected

    def validate(self) -> tuple[str, ...]:
        """Return every reason this mesh is not a printable solid.

        Returns
        -------
        tuple of str
            One message per problem found, empty when the mesh is a closed,
            consistently oriented, outward-facing manifold. Suitable for
            putting straight into an error message.
        """
        problems: list[str] = []

        if self.triangle_count < 4:
            problems.append(
                f"a closed solid needs at least 4 triangles, this has "
                f"{self.triangle_count}"
            )

        a, b, c = self._triangles()
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        degenerate = int((areas <= _DEGENERATE_AREA).sum())
        if degenerate:
            problems.append(f"{degenerate} triangle(s) have no area")

        directed, undirected = self._edge_counts()

        boundary = [edge for edge, count in undirected.items() if count == 1]
        if boundary:
            problems.append(
                f"{len(boundary)} edge(s) belong to only one triangle, so the "
                f"surface is not watertight"
            )
        excess = [edge for edge, count in undirected.items() if count > 2]
        if excess:
            problems.append(
                f"{len(excess)} edge(s) are shared by more than two triangles, "
                f"so the surface is not manifold"
            )
        reversed_edges = [edge for edge, count in directed.items() if count > 1]
        if reversed_edges:
            problems.append(
                f"{len(reversed_edges)} edge(s) are traversed the same way twice, "
                f"so neighbouring triangles disagree on which side is outside"
            )

        if self.volume <= 0.0:
            problems.append(
                f"enclosed volume is {self.volume:.6g}, so the triangle normals "
                f"point inward"
            )

        return tuple(problems)

    @property
    def is_valid_solid(self) -> bool:
        """Whether the mesh is a closed, outward-facing manifold."""
        return not self.validate()

    # -- Transforms ----------------------------------------------------------

    def flipped(self) -> TriangleMesh:
        """Return a copy with every triangle wound the other way."""
        return TriangleMesh(self.vertices, self.faces[:, ::-1].copy())

    def translated(self, offset: Sequence[float]) -> TriangleMesh:
        """Return a copy moved by ``offset``.

        Parameters
        ----------
        offset:
            Three-component translation.

        Returns
        -------
        TriangleMesh
            The translated mesh.
        """
        shift = np.asarray(offset, dtype=np.float64)
        return TriangleMesh(self.vertices + shift, self.faces)

    def transformed(
        self,
        rotation: NDArray[np.float64],
        translation: Sequence[float] | NDArray[np.float64] = (0.0, 0.0, 0.0),
    ) -> TriangleMesh:
        """Return a copy rotated and then translated.

        Parameters
        ----------
        rotation:
            ``(3, 3)`` matrix applied to every vertex. Its columns are the
            images of the local axes.
        translation:
            Three-component offset applied after the rotation.

        Returns
        -------
        TriangleMesh
            The transformed mesh. A rotation that reflects - one with a negative
            determinant - turns the solid inside out, so the winding is reversed
            to compensate and the normals keep pointing outward.

        Raises
        ------
        ValueError
            If the matrix is not ``(3, 3)`` or is singular.
        """
        matrix = np.asarray(rotation, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("rotation must be a (3, 3) matrix")

        determinant = float(np.linalg.det(matrix))
        if abs(determinant) < 1e-12:
            raise ValueError("rotation matrix is singular")

        moved = self.vertices @ matrix.T + np.asarray(translation, dtype=np.float64)
        faces = self.faces[:, ::-1].copy() if determinant < 0.0 else self.faces
        return TriangleMesh(moved, faces)

    def scaled(self, factor: float) -> TriangleMesh:
        """Return a copy scaled about the origin.

        Parameters
        ----------
        factor:
            Uniform scale factor, which must be positive so the winding is
            preserved.

        Returns
        -------
        TriangleMesh
            The scaled mesh.
        """
        if factor <= 0.0:
            raise ValueError("scale factor must be positive")
        return TriangleMesh(self.vertices * factor, self.faces)

    def oriented_outward(self) -> TriangleMesh:
        """Return a copy whose triangle normals certainly point outward."""
        return self.flipped() if self.volume < 0.0 else self


def _clean(
    vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]]
) -> TriangleMesh:
    """Assemble a mesh, dropping degenerate facets and unused vertices.

    Parameters
    ----------
    vertices:
        Vertex coordinates.
    faces:
        Triangles as vertex index triples.

    Returns
    -------
    TriangleMesh
        A mesh with outward normals, no zero-area triangles and no vertices
        that no triangle references.
    """
    vertex_array = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    face_array = np.asarray(faces, dtype=np.int64).reshape(-1, 3)

    if len(face_array):
        a = vertex_array[face_array[:, 0]]
        b = vertex_array[face_array[:, 1]]
        c = vertex_array[face_array[:, 2]]
        areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
        face_array = face_array[areas > _DEGENERATE_AREA]

    # Re-index so that collapsed rings leave no orphan vertices behind.
    used = np.unique(face_array)
    remap = np.full(len(vertex_array), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    return TriangleMesh(vertex_array[used], remap[face_array]).oriented_outward()


def dedupe_polyline(
    points: Sequence[tuple[float, float]],
    *,
    tolerance: float = 1e-9,
    closed: bool = False,
) -> list[tuple[float, float]]:
    """Remove consecutive coincident vertices from a polyline.

    A zero-length edge is not harmless. Swept or revolved it produces a
    zero-area facet, which has to be discarded, and discarding it without also
    merging the two coincident rings leaves the neighbouring strips unmatched -
    a hole. The OCC kernel is less forgiving still and raises
    ``StdFail_NotDone`` rather than skipping the edge.

    Parameters
    ----------
    points:
        Polyline vertices.
    tolerance:
        Distance below which two consecutive vertices are treated as one, in the
        polyline's own units.
    closed:
        Also collapse a duplicated vertex across the wrap from the last vertex
        back to the first, which matters for a closed profile.

    Returns
    -------
    list of tuple
        The polyline with consecutive duplicates removed.
    """
    cleaned: list[tuple[float, float]] = []
    for point in points:
        if (
            cleaned
            and math.hypot(point[0] - cleaned[-1][0], point[1] - cleaned[-1][1])
            < tolerance
        ):
            continue
        cleaned.append(tuple(point))  # type: ignore[arg-type]

    if closed:
        while (
            len(cleaned) > 1
            and math.hypot(
                cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]
            )
            < tolerance
        ):
            cleaned.pop()

    return cleaned


def revolve_profile(
    profile: Sequence[tuple[float, float]], *, segments: int = 180
) -> TriangleMesh:
    """Revolve a closed profile about the Z axis into a solid of revolution.

    Parameters
    ----------
    profile:
        Closed loop of ``(z, r)`` vertices bounding the material in the
        half-plane, with ``r >= 0``. The loop closes implicitly from the last
        vertex back to the first; a repeated first vertex, and any other
        consecutive duplicate, is removed by :func:`dedupe_polyline` first.
        Vertices within :data:`_AXIS_TOLERANCE` of ``r = 0`` collapse onto the
        axis, which is how a cone apex or a solid end face is expressed; a whole
        edge lying on the axis generates no surface, as it encloses nothing.
    segments:
        Number of facets around the circumference, at least 12.

    Returns
    -------
    TriangleMesh
        A closed solid with outward normals, whose seam at ``theta = 0`` shares
        vertices by index rather than duplicating them.

    Raises
    ------
    ValueError
        If the profile has fewer than three vertices, contains a negative
        radius, or ``segments`` is below 12.
    """
    if segments < 12:
        raise ValueError("a revolved solid needs at least 12 segments")
    if any(r < 0.0 for _, r in profile):
        raise ValueError("a revolved profile cannot have a negative radius")

    profile = dedupe_polyline(profile, closed=True)
    if len(profile) < 3:
        raise ValueError("a revolved profile needs at least 3 distinct vertices")

    angles = np.linspace(0.0, 2.0 * math.pi, segments, endpoint=False)
    cos, sin = np.cos(angles), np.sin(angles)

    vertices: list[tuple[float, float, float]] = []
    # Index of the first vertex of each profile station's ring, or the single
    # index of a station collapsed onto the axis.
    rings: list[list[int]] = []
    for z, r in profile:
        if r <= _AXIS_TOLERANCE:
            rings.append([len(vertices)])
            vertices.append((0.0, 0.0, float(z)))
            continue
        start = len(vertices)
        for k in range(segments):
            vertices.append((float(r * cos[k]), float(r * sin[k]), float(z)))
        rings.append([start + k for k in range(segments)])

    faces: list[tuple[int, int, int]] = []
    count = len(profile)
    for i in range(count):
        lower, upper = rings[i], rings[(i + 1) % count]
        if len(lower) == 1 and len(upper) == 1:
            # Both ends on the axis: the edge sweeps out nothing.
            continue
        for k in range(segments):
            nxt = (k + 1) % segments
            if len(lower) == 1:
                faces.append((lower[0], upper[k], upper[nxt]))
            elif len(upper) == 1:
                faces.append((lower[k], upper[0], lower[nxt]))
            else:
                faces.append((lower[k], upper[k], upper[nxt]))
                faces.append((lower[k], upper[nxt], lower[nxt]))

    mesh = _clean(vertices, faces)
    _log.debug(
        "Revolved %d-vertex profile into %d triangles",
        len(profile),
        mesh.triangle_count,
    )
    return mesh


def loft(
    sections: Sequence[Sequence[tuple[float, float, float]]],
) -> TriangleMesh:
    """Loft a stack of closed sections into a solid, capping both ends.

    Consecutive sections are joined by a ruled surface, which is exactly right
    for a linearly tapered fin: every point of the section traces a straight
    line from root to tip.

    Parameters
    ----------
    sections:
        Two or more sections, ordered along the loft. Each is a closed loop of
        ``(x, y, z)`` vertices with the first vertex not repeated, and all
        loops must have the same number of vertices so that corresponding
        points can be joined. A section of a single vertex is a degenerate
        section - the sharp tip of a delta fin - and may only be the first or
        last section.

    Returns
    -------
    TriangleMesh
        A closed solid with outward normals.

    Raises
    ------
    ValueError
        If fewer than two sections are given, if the loops have different
        lengths, or if a degenerate section appears in the middle of the stack.
    """
    if len(sections) < 2:
        raise ValueError("a loft needs at least 2 sections")

    widths = {len(section) for section in sections if len(section) > 1}
    if len(widths) != 1:
        raise ValueError(
            f"every non-degenerate loft section must have the same number of "
            f"vertices; got {sorted(widths)}"
        )
    width = widths.pop()
    if width < 3:
        raise ValueError("a loft section needs at least 3 vertices")
    for index, section in enumerate(sections[1:-1], start=1):
        if len(section) == 1:
            raise ValueError(
                f"section {index} collapses to a point in the middle of the loft"
            )

    vertices: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for section in sections:
        start = len(vertices)
        vertices.extend((float(x), float(y), float(z)) for x, y, z in section)
        rings.append([start + i for i in range(len(section))])

    faces: list[tuple[int, int, int]] = []
    for lower, upper in itertools.pairwise(rings):
        for i in range(width):
            nxt = (i + 1) % width
            if len(upper) == 1:
                faces.append((lower[i], lower[nxt], upper[0]))
            elif len(lower) == 1:
                faces.append((lower[0], upper[nxt], upper[i]))
            else:
                faces.append((lower[i], lower[nxt], upper[nxt]))
                faces.append((lower[i], upper[nxt], upper[i]))

    # Cap each open end with a fan from its centroid. The sections are convex
    # in their own plane, so a centroid fan is always valid.
    for ring, outward in ((rings[0], False), (rings[-1], True)):
        if len(ring) == 1:
            continue
        centre = tuple(
            float(v) for v in np.mean([vertices[i] for i in ring], axis=0)
        )
        hub = len(vertices)
        vertices.append(centre)  # type: ignore[arg-type]
        for i in range(width):
            nxt = (i + 1) % width
            if outward:
                faces.append((ring[i], ring[nxt], hub))
            else:
                faces.append((ring[nxt], ring[i], hub))

    mesh = _clean(vertices, faces)
    _log.debug(
        "Lofted %d sections into %d triangles", len(sections), mesh.triangle_count
    )
    return mesh


def write_stl(
    mesh: TriangleMesh,
    path: str | Path,
    *,
    name: str = "part",
    binary: bool = True,
) -> Path:
    """Write a mesh as an STL file.

    Parameters
    ----------
    mesh:
        The mesh to write. Its winding is used as-is, so pass a mesh that has
        been validated.
    path:
        Destination path.
    name:
        Solid name recorded in the file. Binary STL has only an 80-byte
        comment header, so long names are truncated.
    binary:
        Write binary STL, the format every slicer prefers and roughly a fifth
        the size of ASCII. ASCII is useful for inspecting a file by eye.

    Returns
    -------
    pathlib.Path
        The resolved path written.
    """
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)

    normals = mesh.face_normals
    corners = mesh.vertices[mesh.faces]

    if binary:
        header = f"RocketOpt {name}".encode("ascii", errors="replace")[:80]
        chunks = [header.ljust(80, b"\0"), struct.pack("<I", mesh.triangle_count)]
        for normal, triangle in zip(normals, corners, strict=True):
            chunks.append(
                struct.pack(
                    "<12fH",
                    *normal,
                    *triangle[0],
                    *triangle[1],
                    *triangle[2],
                    0,
                )
            )
        file_path.write_bytes(b"".join(chunks))
    else:
        lines = [f"solid {name}"]
        for normal, triangle in zip(normals, corners, strict=True):
            lines.append(
                f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}"
            )
            lines.append("    outer loop")
            lines.extend(
                f"      vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}" for v in triangle
            )
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append(f"endsolid {name}")
        file_path.write_text("\n".join(lines) + "\n", encoding="ascii")

    _log.info(
        "Wrote %s STL (%d triangles) to %s",
        "binary" if binary else "ASCII",
        mesh.triangle_count,
        file_path,
    )
    return file_path.resolve()
