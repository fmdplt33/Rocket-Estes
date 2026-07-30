"""Tests for printable CAD geometry: meshes, fin sections, fits and STL output.

These check the geometry itself, not that a function ran. Volumes are compared
against closed-form values, the exported STL is parsed back off disk and
re-measured, and the interfaces between parts are measured from the meshes that
will be sliced.
"""

from __future__ import annotations

import math
import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rocketopt.cad.mesh import TriangleMesh, loft, revolve_profile, write_stl
from rocketopt.cad.parts import (
    DEFAULT_TOLERANCES,
    PrintTolerances,
    body_tube_mesh,
    body_tube_of,
    fin_mesh,
    motor_mount_mesh,
    motor_mount_radii,
    nose_cone_mesh,
    printable_parts,
    spigot_length,
)
from rocketopt.cad.validation import (
    AssemblyValidationError,
    validate_printable_assembly,
)
from rocketopt.geometry.components import (
    MIN_MOTOR_TUBE_WALL,
    BodyTube,
    FinAirfoil,
    MotorMount,
)
from rocketopt.geometry.fin_sections import (
    half_thickness,
    peak_station,
    section_area,
    section_area_ratio,
    section_outline,
    swept_volume,
)
from rocketopt.geometry.nose_cones import (
    MAX_SHOULDER_LENGTH,
    MIN_SHOULDER_LENGTH,
)


def _polygon_area(points: list[tuple[float, float]]) -> float:
    """Return the enclosed area of a closed polygon by the shoelace formula."""
    array = np.asarray(points, dtype=np.float64)
    x, y = array[:, 0], array[:, 1]
    return 0.5 * abs(
        float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))
    )


def _read_binary_stl(path: Path) -> TriangleMesh:
    """Parse a binary STL back into a mesh, to check what was written.

    Vertices are welded by exact coordinate match, which is enough to recover the
    original topology because the exporter writes coordinates straight from a
    shared vertex array.
    """
    raw = path.read_bytes()
    count = struct.unpack("<I", raw[80:84])[0]
    vertices: list[tuple[float, float, float]] = []
    index: dict[tuple[float, float, float], int] = {}
    faces: list[tuple[int, int, int]] = []

    for i in range(count):
        offset = 84 + 50 * i
        values = struct.unpack("<12fH", raw[offset : offset + 50])
        corners = []
        for j in range(3):
            point = tuple(float(v) for v in values[3 + 3 * j : 6 + 3 * j])
            if point not in index:
                index[point] = len(vertices)
                vertices.append(point)  # type: ignore[arg-type]
            corners.append(index[point])
        faces.append((corners[0], corners[1], corners[2]))

    return TriangleMesh(
        np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    )


# ---------------------------------------------------------------------------
# Mesh kernel
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        # Solid cylinder, radius 5, height 10.
        ([(0.0, 0.0), (0.0, 5.0), (10.0, 5.0), (10.0, 0.0)], math.pi * 25.0 * 10.0),
        # Annular tube, bore 3, outside 5, height 10.
        ([(0.0, 3.0), (0.0, 5.0), (10.0, 5.0), (10.0, 3.0)], math.pi * 16.0 * 10.0),
        # Cone, apex on the axis.
        ([(0.0, 0.0), (10.0, 5.0), (10.0, 0.0)], math.pi * 25.0 * 10.0 / 3.0),
    ],
)
def test_revolved_solids_have_the_analytic_volume(
    profile: list[tuple[float, float]], expected: float
) -> None:
    """A revolved profile encloses the volume its profile implies."""
    mesh = revolve_profile(profile, segments=720)

    assert mesh.validate() == ()
    # A faceted polygon inscribes the circle, so the mesh is a shade smaller.
    assert mesh.volume == pytest.approx(expected, rel=1e-4)
    assert mesh.volume < expected


def test_revolve_shares_its_seam_vertices() -> None:
    """The seam at theta = 0 is shared, not duplicated.

    A duplicated seam ring is the classic way to produce a mesh that looks
    closed, passes an eyeball check and then leaks in a slicer.
    """
    mesh = revolve_profile(
        [(0.0, 3.0), (0.0, 5.0), (10.0, 5.0), (10.0, 3.0)], segments=32
    )

    assert mesh.vertex_count == 4 * 32
    assert mesh.validate() == ()


def test_revolve_tolerates_a_repeated_profile_vertex() -> None:
    """A coincident profile vertex must not open a hole in the surface.

    Two consecutive profile points at the same place sweep out a zero-area strip.
    Dropping those facets without also merging the two rings would leave the
    neighbouring strips unmatched, so the profile is deduplicated first.
    """
    mesh = revolve_profile(
        [(0.0, 3.0), (0.0, 5.0), (0.0, 5.0), (10.0, 5.0), (10.0, 3.0)], segments=360
    )

    assert mesh.validate() == ()
    assert mesh.volume == pytest.approx(math.pi * 16.0 * 10.0, rel=1e-4)


def test_revolve_collapses_a_near_zero_radius_onto_the_axis() -> None:
    """A tip radius of a few times 1e-14 is an apex, not a ring.

    Regression test for a real failure. A tangent ogive reaches its tip radius
    by subtracting two nearly equal large numbers and can land 3e-14 from zero.
    Revolved, that swept a ring of sliver facets; the slivers were then dropped
    as degenerate and left the apex open, so the optimiser's own designs failed
    the watertightness check on export.
    """
    mesh = revolve_profile(
        [(0.0, 3e-14), (10.0, 5.0), (10.0, 0.0)], segments=64
    )

    assert mesh.validate() == ()
    apex = mesh.vertices[mesh.vertices[:, 2] < 1e-9]
    assert len(apex) == 1, "the tip should collapse to a single vertex"


def test_every_nose_profile_starts_exactly_on_the_axis() -> None:
    """Every family is pointed, so its tip radius is exactly zero."""
    from rocketopt.geometry.nose_cones import (
        NoseCone,
        NoseConeShape,
        default_shape_parameter,
    )
    from rocketopt.structures.materials import get_material

    for shape in NoseConeShape:
        cone = NoseCone(
            shape=shape,
            length=0.0723,
            base_radius=0.0112,
            material=get_material("pla"),
            shape_parameter=default_shape_parameter(shape),
        )
        assert float(cone.radius_at(0.0)[0]) == 0.0, shape.label
        assert float(cone.profile_points(50)[1][0]) == 0.0, shape.label


def test_revolve_rejects_impossible_profiles() -> None:
    """Bad profiles are refused rather than silently producing rubbish."""
    with pytest.raises(ValueError, match="at least 3 distinct vertices"):
        revolve_profile([(0.0, 1.0), (1.0, 1.0)])
    with pytest.raises(ValueError, match="at least 3 distinct vertices"):
        # Three vertices, but two of them are the same point.
        revolve_profile([(0.0, 1.0), (1.0, 1.0), (1.0, 1.0)])
    with pytest.raises(ValueError, match="negative radius"):
        revolve_profile([(0.0, 1.0), (1.0, -1.0), (1.0, 0.0)])
    with pytest.raises(ValueError, match="at least 12 segments"):
        revolve_profile([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)], segments=6)


def test_lofted_solids_have_the_analytic_volume() -> None:
    """A loft between sections encloses the volume the sections imply."""
    base = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 4.0, 0.0), (0.0, 4.0, 0.0)]
    top = [(x, y, 7.0) for x, y, _ in base]

    box = loft([base, top])
    assert box.validate() == ()
    assert box.volume == pytest.approx(10.0 * 4.0 * 7.0)

    # A section that collapses to a point gives a pyramid, a third of the box.
    pyramid = loft([base, [(5.0, 2.0, 7.0)]])
    assert pyramid.validate() == ()
    assert pyramid.volume == pytest.approx(10.0 * 4.0 * 7.0 / 3.0)


def test_loft_rejects_mismatched_or_degenerate_sections() -> None:
    """Sections that cannot be joined are refused."""
    square = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    triangle = [(0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.5, 1.0, 1.0)]

    with pytest.raises(ValueError, match="at least 2 sections"):
        loft([square])
    with pytest.raises(ValueError, match="same number of vertices"):
        loft([square, triangle])
    with pytest.raises(ValueError, match="collapses to a point in the middle"):
        loft([square, [(0.5, 0.5, 1.0)], [(x, y, 2.0) for x, y, _ in square]])


def test_validate_catches_holes_and_inverted_normals() -> None:
    """The mesh checks detect the two ways a solid is not printable."""
    mesh = revolve_profile(
        [(0.0, 0.0), (0.0, 5.0), (10.0, 5.0), (10.0, 0.0)], segments=32
    )
    assert mesh.is_valid_solid

    holed = TriangleMesh(mesh.vertices, mesh.faces[:-1])
    problems = holed.validate()
    assert any("not watertight" in problem for problem in problems)

    inverted = TriangleMesh(mesh.vertices, mesh.faces[:, ::-1].copy())
    assert any("point inward" in problem for problem in inverted.validate())
    assert any("disagree" not in problem for problem in inverted.validate())


def test_written_stl_reparses_to_the_same_solid(tmp_path: Path) -> None:
    """A written binary STL is well formed and describes the same solid."""
    mesh = revolve_profile(
        [(0.0, 3.0), (0.0, 5.0), (10.0, 5.0), (10.0, 3.0)], segments=64
    )
    path = write_stl(mesh, tmp_path / "tube.stl", name="tube")

    raw = path.read_bytes()
    assert len(raw) == 84 + 50 * mesh.triangle_count
    assert struct.unpack("<I", raw[80:84])[0] == mesh.triangle_count
    assert raw[:9] == b"RocketOpt"

    reparsed = _read_binary_stl(path)
    assert reparsed.triangle_count == mesh.triangle_count
    assert reparsed.validate() == (), "the file on disk is not a closed solid"
    assert reparsed.volume == pytest.approx(mesh.volume, rel=1e-6)


def test_ascii_stl_is_written_when_asked(tmp_path: Path) -> None:
    """ASCII STL carries one facet block per triangle."""
    mesh = revolve_profile([(0.0, 0.0), (5.0, 5.0), (5.0, 0.0)], segments=16)
    path = write_stl(mesh, tmp_path / "cone.stl", name="cone", binary=False)
    text = path.read_text(encoding="ascii")

    assert text.startswith("solid cone")
    assert text.rstrip().endswith("endsolid cone")
    assert text.count("facet normal") == mesh.triangle_count
    assert text.count("outer loop") == mesh.triangle_count


# ---------------------------------------------------------------------------
# Fin sections
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("airfoil", list(FinAirfoil))
def test_every_section_reaches_its_specified_thickness(airfoil: FinAirfoil) -> None:
    """Each section reaches exactly its specified thickness, and never exceeds it."""
    thickness, chord = 2.4e-3, 60.0e-3
    stations = np.linspace(0.0, 1.0, 5001)
    y = half_thickness(airfoil, stations, thickness=thickness, chord=chord)

    assert float(y.min()) >= 0.0
    assert float(y.max()) <= thickness / 2.0 + 1e-18
    # The thickest station is known analytically, so the peak is exact there
    # rather than merely close on a sampled grid.
    peak = half_thickness(
        airfoil, peak_station(airfoil), thickness=thickness, chord=chord
    )
    assert float(peak[0]) == pytest.approx(thickness / 2.0, rel=1e-12)

    # The exported outline includes that station, so the solid is full thickness.
    outline = section_outline(airfoil, chord=chord, thickness=thickness)
    assert max(abs(y) for _, y in outline) == pytest.approx(
        thickness / 2.0, rel=1e-12
    )


@pytest.mark.parametrize("airfoil", list(FinAirfoil))
def test_section_area_ratio_matches_the_outline_it_describes(
    airfoil: FinAirfoil,
) -> None:
    """The closed-form area ratio agrees with the outline that is exported."""
    thickness, chord = 2.4e-3, 60.0e-3
    outline = section_outline(
        airfoil, chord=chord, thickness=thickness, samples=2001
    )

    measured = _polygon_area(outline) / (thickness * chord)
    assert measured == pytest.approx(
        section_area_ratio(airfoil, thickness=thickness, chord=chord), rel=1e-4
    )


def test_section_area_ratios_rank_the_families_correctly() -> None:
    """A square plate encloses most material and a diamond exactly half."""
    thickness, chord = 2.4e-3, 60.0e-3

    def ratio(airfoil: FinAirfoil) -> float:
        return section_area_ratio(airfoil, thickness=thickness, chord=chord)

    assert ratio(FinAirfoil.SQUARE) == 1.0
    assert ratio(FinAirfoil.DOUBLE_WEDGE) == pytest.approx(0.5)
    # The NACA section is the published 0.685 of its bounding box.
    assert ratio(FinAirfoil.AIRFOIL) == pytest.approx(0.685, abs=0.01)
    assert (
        ratio(FinAirfoil.DOUBLE_WEDGE)
        < ratio(FinAirfoil.AIRFOIL)
        < ratio(FinAirfoil.ROUNDED)
        < ratio(FinAirfoil.SQUARE)
    )


def test_edges_are_sharp_or_blunt_as_the_family_requires() -> None:
    """Leading and trailing edges close the way each section is defined to."""
    thickness, chord = 2.0e-3, 50.0e-3

    def edges(airfoil: FinAirfoil) -> tuple[float, float]:
        y = half_thickness(
            airfoil, np.array([0.0, 1.0]), thickness=thickness, chord=chord
        )
        return float(y[0]), float(y[1])

    assert edges(FinAirfoil.SQUARE) == (thickness / 2.0, thickness / 2.0)
    assert edges(FinAirfoil.ROUNDED) == pytest.approx((0.0, 0.0))
    assert edges(FinAirfoil.DOUBLE_WEDGE) == pytest.approx((0.0, 0.0))
    # A rounded leading edge tapering to a sharp trailing edge.
    leading, trailing = edges(FinAirfoil.AIRFOIL)
    assert leading == pytest.approx(0.0, abs=1e-12)
    assert trailing == pytest.approx(0.0, abs=1e-12)


def test_rounded_edges_are_semicircles_of_the_half_thickness() -> None:
    """The rounded section's edges follow a circle of radius t/2."""
    thickness, chord = 3.0e-3, 40.0e-3
    radius = thickness / 2.0
    # A quarter of the way round the leading edge circle.
    x = radius * (1.0 - math.cos(math.radians(45.0)))
    y = half_thickness(
        FinAirfoil.ROUNDED, x / chord, thickness=thickness, chord=chord
    )

    assert float(y[0]) == pytest.approx(radius * math.sin(math.radians(45.0)))


def test_a_vanishing_chord_gives_a_degenerate_section() -> None:
    """At the sharp tip of a delta fin the section collapses to a point."""
    assert section_outline(FinAirfoil.ROUNDED, chord=0.0, thickness=2e-3) == [
        (0.0, 0.0)
    ]
    assert section_area(FinAirfoil.ROUNDED, thickness=2e-3, chord=0.0) == 0.0
    # Below the thickness the section is clamped, so it never self-intersects.
    thin = section_outline(FinAirfoil.SQUARE, chord=1e-3, thickness=2e-3)
    assert _polygon_area(thin) == pytest.approx(1e-3 * 1e-3, rel=1e-6)


def test_swept_volume_matches_the_closed_form() -> None:
    """Integrating the section over the span reproduces the analytic volume."""
    root, tip, span, thickness = 60e-3, 30e-3, 45e-3, 2.4e-3
    planform = 0.5 * (root + tip) * span

    for airfoil, ratio in (
        (FinAirfoil.SQUARE, 1.0),
        (FinAirfoil.DOUBLE_WEDGE, 0.5),
    ):
        volume = swept_volume(
            airfoil, root_chord=root, tip_chord=tip, span=span, thickness=thickness
        )
        assert volume == pytest.approx(ratio * thickness * planform, rel=1e-6)

    # The rounded section loses a fixed area at each edge, independent of chord,
    # so its volume is the plate less that area times the span.
    rounded = swept_volume(
        FinAirfoil.ROUNDED,
        root_chord=root,
        tip_chord=tip,
        span=span,
        thickness=thickness,
    )
    expected = thickness * planform - thickness**2 * (1.0 - math.pi / 4.0) * span
    assert rounded == pytest.approx(expected, rel=1e-6)


def test_fin_mass_follows_the_section(reference_rocket) -> None:
    """Changing the section changes the fin mass in the right direction."""
    fins = reference_rocket.fins

    def mass(airfoil: FinAirfoil) -> float:
        return replace(fins, airfoil=airfoil).mass

    assert (
        mass(FinAirfoil.DOUBLE_WEDGE)
        < mass(FinAirfoil.AIRFOIL)
        < mass(FinAirfoil.ROUNDED)
        < mass(FinAirfoil.SQUARE)
    )


# ---------------------------------------------------------------------------
# Nose cone spigot
# ---------------------------------------------------------------------------


def test_nose_spigot_is_sized_to_the_body_tube_bore(reference_rocket) -> None:
    """The spigot's nominal diameter is the tube bore, exactly."""
    tube = body_tube_of(reference_rocket)

    assert reference_rocket.nose.shoulder_outer_radius == pytest.approx(
        tube.inner_radius, abs=1e-12
    )


def test_nose_spigot_mesh_enters_the_body_tube_mesh(reference_rocket) -> None:
    """Measured off both meshes, the spigot slides into the tube with clearance."""
    nose = nose_cone_mesh(reference_rocket)
    tube = body_tube_mesh(reference_rocket)
    spigot_mm = spigot_length(reference_rocket) * 1e3

    _, spigot_radius = nose.radial_extent(0.0, 0.25 * spigot_mm)
    tube_bore, _ = tube.radial_extent()

    clearance = 2.0 * (tube_bore - spigot_radius)
    assert clearance == pytest.approx(
        DEFAULT_TOLERANCES.spigot_clearance * 1e3, abs=0.02
    )
    assert 0.0 < clearance < 0.6


def test_nose_mesh_carries_the_spigot_and_the_full_profile(reference_rocket) -> None:
    """The exported cone is the profile plus the spigot, and nothing else."""
    mesh = nose_cone_mesh(reference_rocket)
    nose = reference_rocket.nose
    tube = body_tube_of(reference_rocket)

    height = nose.length + spigot_length(reference_rocket)
    assert mesh.size[2] == pytest.approx(height * 1e3, rel=1e-9)
    # The widest point is the base, which must match the tube it butts against.
    assert mesh.size[0] == pytest.approx(tube.diameter * 1e3, rel=1e-3)
    assert mesh.validate() == ()
    # A shell, so it encloses far less than the envelope of the profile.
    assert mesh.volume < 0.5 * nose.enclosed_volume * 1e9
    assert mesh.volume == pytest.approx(nose.material_volume * 1e9, rel=0.02)


def test_a_solid_nose_exports_solid(reference_rocket) -> None:
    """A turned cone is exported as solid stock with a solid spigot."""
    balsa = replace(reference_rocket.nose, solid=True)
    turned = replace(reference_rocket, nose=balsa)
    mesh = nose_cone_mesh(turned)

    assert mesh.validate() == ()
    assert mesh.volume == pytest.approx(balsa.material_volume * 1e9, rel=0.02)
    # No bore: the axis is reached at the aft face.
    bore, _ = mesh.radial_extent(0.0, 0.1)
    assert bore == pytest.approx(0.0)


def test_a_design_without_a_shoulder_is_given_a_default_spigot() -> None:
    """build_rocket resolves a missing shoulder into a usable spigot."""
    from rocketopt.geometry.components import (
        FinSet,
        RecoveryDevice,
        RecoveryType,
    )
    from rocketopt.geometry.nose_cones import NoseCone, NoseConeShape
    from rocketopt.geometry.rocket import build_rocket
    from rocketopt.propulsion.database import get_motor_configuration
    from rocketopt.structures.materials import get_material

    kraft = get_material("kraft_paper")
    tube = BodyTube(
        length=0.30, outer_radius=0.0124, wall_thickness=0.45e-3, material=kraft
    )
    rocket = build_rocket(
        nose=NoseCone(
            shape=NoseConeShape.CONICAL,
            length=0.06,
            base_radius=0.0124,
            material=get_material("pla"),
            shoulder_length=0.0,
        ),
        body_tube=tube,
        fins=FinSet(
            count=3,
            root_chord=0.05,
            tip_chord=0.025,
            span=0.04,
            sweep_length=0.02,
            thickness=2.4e-3,
            material=get_material("balsa"),
            body_radius=0.0124,
        ),
        motor=get_motor_configuration("C6-5"),
        recovery=RecoveryDevice(kind=RecoveryType.PARACHUTE_FLAT, diameter=0.30),
        name="No shoulder",
    )

    spigot = spigot_length(rocket)
    assert MIN_SHOULDER_LENGTH <= spigot <= MAX_SHOULDER_LENGTH
    assert rocket.nose.shoulder_length == pytest.approx(spigot)
    assert rocket.nose.shoulder_outer_radius == pytest.approx(tube.inner_radius)
    assert nose_cone_mesh(rocket).validate() == ()


def test_a_spigot_wider_than_the_tube_bore_is_refused(reference_rocket) -> None:
    """A cone that cannot enter the tube it sits on is a build error."""
    oversized = replace(
        reference_rocket.nose, shoulder_radius=reference_rocket.nose.base_radius
    )
    with pytest.raises(ValueError, match="shoulder is"):
        replace(reference_rocket, nose=oversized)


# ---------------------------------------------------------------------------
# Motor mount
# ---------------------------------------------------------------------------


def test_generated_mount_spans_bore_to_motor(reference_rocket) -> None:
    """The generated mount's outside diameter is the bore and its bore the motor."""
    mount = reference_rocket.motor_mount
    tube = body_tube_of(reference_rocket)
    motor = reference_rocket.motor.motor

    assert mount.outer_radius == pytest.approx(tube.inner_radius, abs=1e-12)
    assert mount.inner_diameter > motor.diameter
    assert mount.inner_diameter == pytest.approx(motor.diameter + 0.8e-3)
    assert mount.length == pytest.approx(motor.length)
    # It contacts the bore along its whole length, so rings would be redundant.
    assert mount.centring_ring_count == 0
    assert mount.ring_mass == 0.0
    assert mount.is_minimum_diameter


def test_mount_mesh_fits_the_tube_and_accepts_the_motor(reference_rocket) -> None:
    """Measured off the mesh, the mount goes in the tube and the motor in it."""
    mesh = motor_mount_mesh(reference_rocket)
    tube = body_tube_of(reference_rocket)
    motor = reference_rocket.motor.motor

    bore, outer = mesh.radial_extent()
    assert mesh.validate() == ()
    assert mesh.size[2] == pytest.approx(motor.length * 1e3, rel=1e-9)
    assert 2.0 * outer < tube.inner_radius * 2e3
    assert 2.0 * bore > motor.diameter * 1e3
    assert 2.0 * (tube.inner_radius * 1e3 - outer) == pytest.approx(
        DEFAULT_TOLERANCES.mount_clearance * 1e3, abs=0.02
    )
    assert 2.0 * bore - motor.diameter * 1e3 >= (
        DEFAULT_TOLERANCES.motor_clearance * 1e3
    )


def test_mount_is_refused_when_the_motor_will_not_fit(reference_rocket) -> None:
    """A bore too small for the motor plus a wall is an error, not a thin mount."""
    motor = reference_rocket.motor.motor
    with pytest.raises(ValueError, match="minimum-diameter construction"):
        MotorMount.for_airframe(
            motor_diameter=motor.diameter,
            motor_length=motor.length,
            body_inner_radius=motor.diameter / 2.0 + MIN_MOTOR_TUBE_WALL / 2.0,
            material=reference_rocket.motor_mount.material,
        )


def test_a_ring_carried_mount_is_still_exported(reference_rocket) -> None:
    """A hand-specified narrow mount is respected, not widened to the bore."""
    narrow = MotorMount(
        inner_diameter=reference_rocket.motor.motor.diameter + 0.8e-3,
        length=reference_rocket.motor_mount.length,
        wall_thickness=0.45e-3,
        material=reference_rocket.motor_mount.material,
        body_inner_radius=reference_rocket.motor_mount.body_inner_radius,
        centring_ring_count=2,
        position=reference_rocket.motor_mount.position,
    )
    ringed = replace(reference_rocket, motor_mount=narrow)

    assert not narrow.is_minimum_diameter
    _, outer = motor_mount_mesh(ringed).radial_extent()
    assert 2.0 * outer == pytest.approx(narrow.outer_diameter * 1e3, abs=0.31)
    # And it still validates: the rings bridge the annulus that is left.
    validation = validate_printable_assembly(ringed)
    assert validation.passed, validation.report()


def test_the_mount_bore_is_refused_when_it_exceeds_the_outside(
    reference_rocket,
) -> None:
    """A mount with no wall left once clearances are applied is reported."""
    bore = body_tube_of(reference_rocket).inner_radius
    # A wall thin enough that the print clearance eats all of it: the mount is
    # nominally the right size, but nothing can be made from it.
    hairline = replace(
        reference_rocket.motor_mount,
        inner_diameter=2.0 * (bore - DEFAULT_TOLERANCES.mount_clearance / 2.0),
        wall_thickness=DEFAULT_TOLERANCES.mount_clearance / 2.0,
    )
    with pytest.raises(ValueError, match="does not fit the airframe"):
        motor_mount_radii(replace(reference_rocket, motor_mount=hairline))


# ---------------------------------------------------------------------------
# Fin solids
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("airfoil", list(FinAirfoil))
def test_exported_fin_carries_its_section(reference_rocket, airfoil) -> None:
    """The lofted fin encloses the volume of the section it was specified with."""
    fins = replace(reference_rocket.fins, airfoil=airfoil)
    rocket = replace(reference_rocket, fins=fins)
    mesh = fin_mesh(rocket)

    assert mesh.validate() == ()
    assert mesh.volume == pytest.approx(fins.volume_single * 1e9, rel=5e-3)
    # Planform and thickness are reproduced exactly, not approximated.
    assert mesh.size[0] == pytest.approx(
        max(fins.root_chord, fins.sweep_length + fins.tip_chord) * 1e3
    )
    assert mesh.size[1] == pytest.approx(fins.thickness * 1e3, rel=1e-9)
    assert mesh.size[2] == pytest.approx(fins.span * 1e3, rel=1e-9)


def test_an_aerofoil_fin_is_not_a_flat_plate(reference_rocket) -> None:
    """The whole point: the section changes the solid, not just a label."""
    plate = fin_mesh(
        replace(
            reference_rocket,
            fins=replace(reference_rocket.fins, airfoil=FinAirfoil.SQUARE),
        )
    )
    aerofoil = fin_mesh(
        replace(
            reference_rocket,
            fins=replace(reference_rocket.fins, airfoil=FinAirfoil.AIRFOIL),
        )
    )

    # Same bounding box, materially less material inside it.
    assert np.allclose(plate.size, aerofoil.size)
    assert aerofoil.volume < 0.75 * plate.volume


def test_fin_thickness_holds_across_the_span(reference_rocket) -> None:
    """The section keeps its full thickness from root to tip, as sheet stock does."""
    fins = replace(reference_rocket.fins, airfoil=FinAirfoil.AIRFOIL)
    mesh = fin_mesh(replace(reference_rocket, fins=fins))
    span_mm = fins.span * 1e3

    for fraction in (0.0, 0.5, 0.99):
        z = fraction * span_mm
        near = mesh.vertices[np.abs(mesh.vertices[:, 2] - z) < 1e-6]
        if not len(near):
            continue
        assert float(np.abs(near[:, 1]).max() * 2.0) == pytest.approx(
            fins.thickness * 1e3, rel=1e-6
        )


def test_a_delta_fin_lofts_to_a_point(reference_rocket) -> None:
    """A zero tip chord closes on a single vertex and stays a valid solid."""
    delta = replace(
        reference_rocket, fins=replace(reference_rocket.fins, tip_chord=0.0)
    )
    mesh = fin_mesh(delta)

    assert mesh.validate() == ()
    tip = mesh.vertices[mesh.vertices[:, 2] > delta.fins.span * 1e3 - 1e-9]
    assert len(tip) == 1, "the tip should be a single point"


# ---------------------------------------------------------------------------
# Assembly validation and export
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("airfoil", list(FinAirfoil))
def test_the_reference_assembly_validates(reference_rocket, airfoil) -> None:
    """Every part of the reference design assembles and would slice."""
    rocket = replace(
        reference_rocket, fins=replace(reference_rocket.fins, airfoil=airfoil)
    )
    validation = validate_printable_assembly(rocket)

    assert validation.passed, validation.report()
    assert validation.failures == ()
    assert "0 failed" in validation.report()


def test_validation_catches_an_interference_fit(reference_rocket) -> None:
    """With no clearance the parts would not go together, and that is reported."""
    tight = PrintTolerances(
        spigot_clearance=0.0, mount_clearance=0.0, motor_clearance=0.0
    )
    validation = validate_printable_assembly(reference_rocket, tight)

    assert not validation.passed
    failed = {check.name for check in validation.failures}
    assert "Nose cone spigot enters the body tube" in failed
    assert "Motor mount enters the body tube" in failed
    assert "Motor enters the motor mount" in failed
    assert "[FAIL]" in validation.report()


def test_export_writes_nothing_when_validation_fails(
    reference_rocket, tmp_path: Path
) -> None:
    """A design that would not assemble produces no misleading STL."""
    from rocketopt.cad.exporters import export_stl_parts

    tight = PrintTolerances(spigot_clearance=0.0)
    with pytest.raises(AssemblyValidationError, match="pre-export checks failed"):
        export_stl_parts(reference_rocket, tmp_path, tolerances=tight)

    assert list(tmp_path.glob("*.stl")) == []


def test_every_exported_stl_is_a_closed_solid(reference_rocket, tmp_path: Path) -> None:
    """Parsed back off disk, each part is watertight, manifold and outward-facing."""
    from rocketopt.cad.exporters import export_stl_parts

    written = export_stl_parts(reference_rocket, tmp_path)
    assert {path.name for path in written} == {
        "nose_cone.stl",
        "body_tube.stl",
        "motor_mount.stl",
        "fin.stl",
    }

    expected = {part.key: part.mesh for part in printable_parts(reference_rocket)}
    for path in written:
        mesh = _read_binary_stl(path)
        assert mesh.validate() == (), f"{path.name}: {mesh.validate()}"
        assert mesh.volume == pytest.approx(expected[path.stem].volume, rel=1e-6)


def test_parts_are_oriented_for_printing(reference_rocket) -> None:
    """Each part stands on the bed at z = 0 with its build axis up."""
    for part in printable_parts(reference_rocket):
        low, high = part.mesh.bounds
        assert low[2] == pytest.approx(0.0, abs=1e-9), f"{part.key} floats off the bed"
        assert high[2] > 0.0
        assert part.description, f"{part.key} has no description"

    quantities = {part.key: part.quantity for part in printable_parts(reference_rocket)}
    assert quantities["fin"] == reference_rocket.fins.count
    assert quantities["nose_cone"] == 1


def test_print_tolerances_are_validated() -> None:
    """A tolerance set that makes no sense is refused at construction."""
    with pytest.raises(ValueError, match="outside the sensible range"):
        PrintTolerances(spigot_clearance=5e-3)
    with pytest.raises(ValueError, match="revolve_segments"):
        PrintTolerances(revolve_segments=4)
    with pytest.raises(ValueError, match="span_sections"):
        PrintTolerances(span_sections=1)
