"""Tests for the assembled 3D model and the single-file export bundle."""

from __future__ import annotations

import math
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rocketopt.bundle import write_bundle
from rocketopt.cad.assembly import (
    DISPLAY_TOLERANCES,
    _fin_rotation,
    assembled_parts,
    assembly_bounds,
)
from rocketopt.cad.mesh import TriangleMesh, revolve_profile
from rocketopt.cad.parts import DEFAULT_TOLERANCES, body_tube_of

# ---------------------------------------------------------------------------
# Mesh transforms
# ---------------------------------------------------------------------------


def test_a_rotation_moves_a_mesh_without_changing_it() -> None:
    """A rigid transform preserves volume, and the winding stays outward."""
    tube = revolve_profile(
        [(0.0, 3.0), (0.0, 5.0), (10.0, 5.0), (10.0, 3.0)], segments=48
    )
    angle = math.radians(37.0)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    moved = tube.transformed(rotation, (100.0, -20.0, 5.0))

    assert moved.validate() == ()
    assert moved.volume == pytest.approx(tube.volume, rel=1e-9)
    assert moved.bounds[0][2] == pytest.approx(5.0)


def test_a_reflection_keeps_the_normals_outward() -> None:
    """A mirrored solid is re-wound rather than left inside out."""
    tube = revolve_profile(
        [(0.0, 3.0), (0.0, 5.0), (10.0, 5.0), (10.0, 3.0)], segments=48
    )
    mirror = np.diag([1.0, 1.0, -1.0])

    mirrored = tube.transformed(mirror)

    assert mirrored.volume > 0.0
    assert mirrored.validate() == ()


def test_transform_rejects_a_bad_matrix() -> None:
    """A matrix that is not a usable transform is refused."""
    mesh = revolve_profile([(0.0, 0.0), (1.0, 1.0), (1.0, 0.0)], segments=16)
    with pytest.raises(ValueError, match=r"\(3, 3\) matrix"):
        mesh.transformed(np.eye(2))
    with pytest.raises(ValueError, match="singular"):
        mesh.transformed(np.zeros((3, 3)))


def test_the_fin_placement_is_a_proper_rotation() -> None:
    """Fins are placed by a right-handed rotation, so they are not mirrored."""
    for degrees in (0.0, 90.0, 137.0, 250.0):
        rotation = _fin_rotation(math.radians(degrees))
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_every_assembled_part_is_a_valid_solid(reference_rocket) -> None:
    """The assembly is made of the same closed solids as the export."""
    parts = assembled_parts(reference_rocket)

    assert {part.key for part in parts} >= {
        "nose_cone",
        "body_tube",
        "motor_mount",
        "motor",
        "fin",
    }
    for part in parts:
        assert part.mesh.validate() == (), f"{part.name}: {part.mesh.validate()}"
        assert part.mesh.volume > 0.0


def test_parts_sit_at_their_design_stations(reference_rocket) -> None:
    """Each part is placed at the station the design puts it at."""
    parts = {part.name: part for part in assembled_parts(reference_rocket)}
    nose = reference_rocket.nose
    tube = body_tube_of(reference_rocket)
    mount = reference_rocket.motor_mount

    # The nose tip is the origin, and the cone runs aft from it.
    nose_low, nose_high = parts["Nose cone"].mesh.bounds
    assert nose_low[2] == pytest.approx(0.0, abs=1e-6)
    assert nose_high[2] == pytest.approx(
        (nose.length + nose.shoulder_length) * 1e3, rel=1e-6
    )

    tube_low, tube_high = parts["Body tube"].mesh.bounds
    assert tube_low[2] == pytest.approx(nose.length * 1e3, rel=1e-9)
    assert tube_high[2] == pytest.approx((nose.length + tube.length) * 1e3, rel=1e-9)

    # The spigot overlaps the tube, which is what holds the cone on.
    assert nose_high[2] > tube_low[2]

    mount_low, mount_high = parts["Motor mount"].mesh.bounds
    assert mount_low[2] == pytest.approx(mount.position * 1e3, rel=1e-9)
    assert mount_high[2] == pytest.approx(
        (mount.position + mount.length) * 1e3, rel=1e-9
    )


def test_fins_are_equally_spaced_around_the_body(reference_rocket) -> None:
    """Every fin is on the body at its own angle, all at the same radius."""
    fins = [
        part for part in assembled_parts(reference_rocket) if part.key == "fin"
    ]
    assert len(fins) == reference_rocket.fins.count

    tip_radius = (
        reference_rocket.fins.body_radius + reference_rocket.fins.span
    ) * 1e3
    # The tip's corners sit half a thickness either side of the fin's plane, so
    # their distance from the axis is a hair more than the span reaches.
    corner_radius = math.hypot(tip_radius, reference_rocket.fins.thickness * 5e2)

    angles = []
    for part in fins:
        radii = np.hypot(part.mesh.vertices[:, 0], part.mesh.vertices[:, 1])
        assert float(radii.max()) == pytest.approx(corner_radius, rel=1e-6)
        # Average over the whole tip so the fin's own thickness, which puts its
        # surfaces a degree or so either side of the plane, cancels out.
        tip = part.mesh.vertices[radii > 0.999 * float(radii.max())]
        centre = tip.mean(axis=0)
        angles.append(math.degrees(math.atan2(centre[1], centre[0])) % 360.0)

    def gap(first: float, second: float) -> float:
        """Smallest angle between two bearings [deg], wrap included."""
        return abs((first - second + 180.0) % 360.0 - 180.0)

    spacing = 360.0 / reference_rocket.fins.count
    for index in range(reference_rocket.fins.count):
        assert min(gap(angle, index * spacing) for angle in angles) < 0.5


def test_the_assembly_spans_the_whole_rocket(reference_rocket) -> None:
    """The assembly reaches from the nose tip to the tail and out to the fins."""
    low, high = assembly_bounds(assembled_parts(reference_rocket))

    assert low[2] == pytest.approx(0.0, abs=1e-6)
    assert high[2] == pytest.approx(reference_rocket.length * 1e3, rel=1e-6)
    span = (reference_rocket.fins.body_radius + reference_rocket.fins.span) * 1e3
    assert max(abs(low[0]), abs(high[0])) == pytest.approx(span, rel=1e-3)


def test_display_tessellation_is_lighter_than_the_printable_one(
    reference_rocket,
) -> None:
    """The viewer's mesh is coarser than the exported one, and still valid."""
    display = assembled_parts(reference_rocket, DISPLAY_TOLERANCES)
    printable = assembled_parts(reference_rocket, DEFAULT_TOLERANCES)

    display_count = sum(part.mesh.triangle_count for part in display)
    printable_count = sum(part.mesh.triangle_count for part in printable)

    assert display_count < printable_count / 5
    assert all(part.mesh.validate() == () for part in display)

    # Coarser sampling must not move the geometry, only facet it more roughly.
    # A faceted cylinder inscribes the true one, so the coarse mesh is always
    # the smaller of the two - by a few percent on the launch lug, which is the
    # smallest part and carries the fewest segments.
    for coarse, fine in zip(display, printable, strict=True):
        assert coarse.mesh.volume == pytest.approx(fine.mesh.volume, rel=0.04)
        assert coarse.mesh.volume <= fine.mesh.volume * 1.0001


def test_a_canted_fin_set_still_assembles(reference_rocket) -> None:
    """Fin cant rotates each fin without breaking the solid."""
    canted = replace(
        reference_rocket,
        fins=replace(reference_rocket.fins, cant_angle=math.radians(3.0)),
    )
    fins = [part for part in assembled_parts(canted) if part.key == "fin"]

    assert len(fins) == canted.fins.count
    for part in fins:
        assert part.mesh.validate() == ()


def test_fins_are_excluded_from_the_section_cut(reference_rocket) -> None:
    """Only the bodies of revolution are marked for sectioning."""
    parts = {part.key: part for part in assembled_parts(reference_rocket)}

    assert parts["nose_cone"].axisymmetric
    assert parts["body_tube"].axisymmetric
    assert parts["motor_mount"].axisymmetric
    assert not parts["fin"].axisymmetric


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def test_bundle_holds_every_output_in_one_file(
    reference_rocket, reference_flight, tmp_path: Path
) -> None:
    """One archive carries the printable solids, the CAD and the reports."""
    contents = write_bundle(
        reference_rocket, tmp_path / "design.zip", flight=reference_flight
    )

    assert contents.path.exists()
    assert contents.path.suffix == ".zip"
    assert zipfile.is_zipfile(contents.path)

    names = set(contents.names)
    assert "README.txt" in names
    assert {
        "print/nose_cone.stl",
        "print/body_tube.stl",
        "print/motor_mount.stl",
        "print/fin.stl",
    } <= names
    assert {"cad/fin_template.svg", "cad/fin.dxf", "cad/fusion360_build.py"} <= names
    assert {
        "reports/engineering_report.md",
        "reports/manufacturing_guide.md",
    } <= names

    with zipfile.ZipFile(contents.path) as archive:
        assert archive.testzip() is None
        # The STLs must survive the round trip as real geometry.
        for name in names:
            if name.endswith(".stl"):
                assert len(archive.read(name)) > 84
        readme = archive.read("README.txt").decode("utf-8")

    assert reference_rocket.name in readme
    assert reference_rocket.motor.designation in readme
    # The manifest states the fits, which is the point of shipping one file.
    bore = body_tube_of(reference_rocket).inner_radius * 2e3
    assert f"{bore:.2f} mm" in readme
    assert "Print clearances" in readme

    assert contents.uncompressed_bytes > contents.path.stat().st_size


def test_bundle_names_itself_from_the_design(
    reference_rocket, reference_flight, tmp_path: Path
) -> None:
    """Given a directory, the archive takes the design's name."""
    contents = write_bundle(reference_rocket, tmp_path, flight=reference_flight)

    assert contents.path.parent == tmp_path.resolve()
    assert contents.path.name == "Reference_Alpha_III_class.zip"


def test_bundle_can_skip_the_reports(
    reference_rocket, tmp_path: Path
) -> None:
    """Without reports the bundle needs no flight, so it is much quicker."""
    contents = write_bundle(
        reference_rocket, tmp_path / "cad_only.zip", include_reports=False
    )

    assert not any(name.startswith("reports/") for name in contents.names)
    assert any(name.startswith("print/") for name in contents.names)


def test_bundle_refuses_a_design_that_would_not_assemble(
    reference_rocket, tmp_path: Path
) -> None:
    """A failed pre-export check stops the bundle being written at all."""
    from rocketopt.cad.parts import PrintTolerances
    from rocketopt.cad.validation import AssemblyValidationError

    target = tmp_path / "broken.zip"
    with pytest.raises(AssemblyValidationError):
        write_bundle(
            reference_rocket,
            target,
            include_reports=False,
            tolerances=PrintTolerances(spigot_clearance=0.0),
        )

    assert not target.exists()


def test_mesh_translation_and_scaling(reference_rocket) -> None:
    """The simple transforms behave, and scaling is refused if it would flip."""
    part = assembled_parts(reference_rocket)[0].mesh

    moved = part.translated((1.0, 2.0, 3.0))
    assert moved.bounds[0][0] == pytest.approx(part.bounds[0][0] + 1.0)
    assert moved.volume == pytest.approx(part.volume, rel=1e-9)

    bigger = part.scaled(2.0)
    assert bigger.volume == pytest.approx(part.volume * 8.0, rel=1e-9)

    with pytest.raises(ValueError, match="must be positive"):
        part.scaled(-1.0)


def test_a_mesh_reports_its_own_size(reference_rocket) -> None:
    """Bounds, size and surface area agree with the geometry."""
    mesh = assembled_parts(reference_rocket)[1].mesh
    assert isinstance(mesh, TriangleMesh)
    low, high = mesh.bounds

    assert np.allclose(mesh.size, high - low)
    assert mesh.surface_area > 0.0
    assert mesh.vertex_count > 0
