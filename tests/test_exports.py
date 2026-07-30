"""Tests for report and CAD export.

These check that exported files are *valid*, not merely that a function ran:
STEP files must carry an ISO-10303 header and re-import as real geometry, the
PDF must have a PDF header and multiple pages, and the Fusion 360 script must
parse as Python.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pytest

from rocketopt.cad.exporters import (
    export_all,
    export_fin_dxf,
    export_fin_template_svg,
    export_profile_svg,
)
from rocketopt.cad.fusion360 import build_fusion_script, write_fusion_script
from rocketopt.reports.bom import bill_of_materials, build_manufacturing_guide
from rocketopt.reports.engineering import build_markdown_report

def _has_cadquery() -> bool:
    """Return whether CadQuery is importable."""
    try:
        import cadquery  # noqa: F401
    except ImportError:
        return False
    return True


def _has_reportlab() -> bool:
    """Return whether ReportLab is importable."""
    try:
        import reportlab  # noqa: F401
    except ImportError:
        return False
    return True


# ---------------------------------------------------------------------------
# Dependency-free exports
# ---------------------------------------------------------------------------


def test_fin_template_svg_is_well_formed(reference_rocket, tmp_path: Path) -> None:
    """The fin template is valid XML and carries a calibration bar."""
    from xml.etree import ElementTree

    path = export_fin_template_svg(reference_rocket, tmp_path / "fin.svg")
    text = path.read_text(encoding="utf-8")

    ElementTree.fromstring(text)  # raises if malformed
    assert "100 mm" in text, "print calibration bar is missing"
    # Dimensions must be in millimetres so it prints at 1:1.
    assert 'width="' in text and "mm" in text


def test_profile_svg_is_well_formed(reference_rocket, tmp_path: Path) -> None:
    """The side elevation is valid XML and marks CG and CP."""
    from xml.etree import ElementTree

    path = export_profile_svg(reference_rocket, tmp_path / "profile.svg")
    text = path.read_text(encoding="utf-8")

    ElementTree.fromstring(text)
    assert ">CG<" in text
    assert ">CP<" in text


def test_fin_dxf_has_a_closed_polyline(reference_rocket, tmp_path: Path) -> None:
    """The DXF contains a closed LWPOLYLINE in millimetres."""
    path = export_fin_dxf(reference_rocket, tmp_path / "fin.dxf")
    text = path.read_text(encoding="utf-8")

    assert "LWPOLYLINE" in text
    assert text.strip().endswith("EOF")
    # $INSUNITS 4 is millimetres.
    assert "$INSUNITS" in text
    # Closed flag: group code 70 with value 1.
    lines = text.splitlines()
    seventy = [i for i, line in enumerate(lines) if line.strip() == "70"]
    assert any(lines[i + 1].strip() == "1" for i in seventy)


def test_fusion_script_parses_as_python(reference_rocket) -> None:
    """The generated Fusion 360 script must be syntactically valid."""
    source = build_fusion_script(reference_rocket)
    ast.parse(source)

    assert "adsk.fusion" in source
    assert "def run(context)" in source
    # It must be parametric: user parameters, not baked-in numbers only.
    assert "userParameters" in source
    assert "fin_root_chord" in source


def test_fusion_script_parameters_match_the_design(
    reference_rocket, tmp_path: Path
) -> None:
    """Parameter values in the script match the rocket they came from."""
    path = write_fusion_script(reference_rocket, tmp_path / "build.py")
    source = path.read_text(encoding="utf-8")

    tree = ast.parse(source)
    parameters: dict[str, float] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "PARAMETERS" for t in node.targets
        ):
            for element in node.value.elts:  # type: ignore[attr-defined]
                name = ast.literal_eval(element.elts[0])
                value = ast.literal_eval(element.elts[1])
                parameters[name] = value

    assert parameters["fin_root_chord"] == pytest.approx(
        reference_rocket.fins.root_chord * 1e3, abs=1e-3
    )
    assert parameters["fin_span"] == pytest.approx(
        reference_rocket.fins.span * 1e3, abs=1e-3
    )
    assert parameters["body_outer_diameter"] == pytest.approx(
        reference_rocket.reference_diameter * 1e3, abs=1e-3
    )


def test_markdown_report_has_every_section(reference_rocket, reference_flight) -> None:
    """The Markdown report contains all eleven numbered sections."""
    text = build_markdown_report(reference_rocket, reference_flight)
    for section in (
        "## 1. Summary",
        "## 2. Geometry",
        "## 3. Mass properties",
        "## 4. Propulsion",
        "## 5. Aerodynamics",
        "## 6. Stability",
        "## 7. Structures",
        "## 8. Flight simulation",
        "## 9. Warnings",
        "## 10. Bill of materials",
        "## 11. References",
    ):
        assert section in text, f"missing {section}"


def test_bill_of_materials_mass_is_consistent(reference_rocket) -> None:
    """BOM masses account for most of the dry mass.

    They will not match exactly: the BOM lists purchasable items and fillet
    adhesive, while the dry mass also includes ballast and payload. The check
    is that nothing large has been left off the list.
    """
    items = bill_of_materials(reference_rocket)
    total = sum(item.mass for item in items)

    assert total > 0.5 * reference_rocket.dry_mass
    assert total < 1.6 * reference_rocket.dry_mass
    assert all(item.specification for item in items), "an item has no specification"


def test_manufacturing_guide_mentions_the_key_dimensions(reference_rocket) -> None:
    """The build guide states the fin geometry a builder must cut to."""
    guide = build_manufacturing_guide(reference_rocket)
    fins = reference_rocket.fins

    assert f"{fins.root_chord * 1e3:.1f} mm" in guide
    assert f"{fins.span * 1e3:.1f} mm" in guide
    assert str(fins.count) in guide
    assert "Pre-flight checks" in guide


def test_export_all_writes_the_dependency_free_set(
    reference_rocket, tmp_path: Path
) -> None:
    """SVG, DXF, the Fusion script and every printable STL are always produced."""
    written = export_all(reference_rocket, tmp_path)
    names = {p.name for p in written}

    assert {"fin_template.svg", "side_elevation.svg", "fin.dxf",
            "fusion360_build.py"} <= names
    # STL export needs no optional dependency, so it is never skipped.
    assert {"nose_cone.stl", "body_tube.stl", "motor_mount.stl", "fin.stl"} <= names
    assert all(p.stat().st_size > 100 for p in written)


# ---------------------------------------------------------------------------
# Optional extras
# ---------------------------------------------------------------------------


@pytest.mark.requires_cad
@pytest.mark.skipif(not _has_cadquery(), reason="CadQuery is not installed")
def test_step_export_produces_real_geometry(reference_rocket, tmp_path: Path) -> None:
    """Exported STEP files are valid and agree with the printable meshes.

    Cross-validation between two independent kernels: the STEP bodies come from
    OpenCascade and the STLs from the mesh kernel in ``rocketopt.cad.mesh``, but
    both are built from the same profiles, so their volumes must agree to within
    the mesh's faceting error.

    Also a regression test for a real failure: the nose profile already begins at
    the tip, so prepending an explicit (0, 0) produced a duplicated vertex and
    the OCC kernel raised StdFail_NotDone instead of ignoring the resulting
    zero-length edge.
    """
    import cadquery as cq

    from rocketopt.cad.exporters import export_solids
    from rocketopt.cad.parts import printable_parts

    written = export_solids(reference_rocket, tmp_path)
    names = {p.name for p in written}
    assert {
        "nose_cone.step",
        "body_tube.step",
        "motor_mount.step",
        "fin.step",
    } <= names

    for path in written:
        assert path.stat().st_size > 200
        assert path.suffix == ".step", "STLs come from export_stl_parts"
        head = path.read_text(encoding="utf-8", errors="replace")[:300]
        assert "ISO-10303" in head

    meshes = {part.key: part.mesh for part in printable_parts(reference_rocket)}
    for path in written:
        solid = cq.importers.importStep(str(path))
        assert solid.val().isValid(), f"{path.name} is not a valid solid"
        # Faceting makes the mesh very slightly smaller than the exact body.
        assert solid.val().Volume() == pytest.approx(
            meshes[path.stem].volume, rel=2e-3
        ), f"{path.name} disagrees with the mesh it was built alongside"


@pytest.mark.requires_cad
@pytest.mark.skipif(not _has_cadquery(), reason="CadQuery is not installed")
def test_delta_fin_exports_without_degenerate_edges(
    reference_rocket, tmp_path: Path
) -> None:
    """A zero tip chord collapses two corners; export must still succeed."""
    from dataclasses import replace

    from rocketopt.cad.exporters import export_solids

    delta = replace(reference_rocket, fins=replace(reference_rocket.fins, tip_chord=0.0))
    written = export_solids(delta, tmp_path)
    assert any(p.name == "fin.step" for p in written)


@pytest.mark.skipif(not _has_reportlab(), reason="ReportLab is not installed")
def test_pdf_report_is_a_valid_multipage_pdf(
    reference_rocket, reference_flight, tmp_path: Path
) -> None:
    """The PDF has a valid header, several pages and embedded plots."""
    from rocketopt.reports.pdf import write_pdf_report

    path = write_pdf_report(reference_rocket, reference_flight, tmp_path / "r.pdf")
    raw = path.read_bytes()

    assert raw.startswith(b"%PDF-")
    assert raw.rstrip().endswith(b"%%EOF")
    # Embedded plot images make the file substantially larger than text alone.
    assert len(raw) > 40_000

    pages = raw.count(b"/Type /Page") - raw.count(b"/Type /Pages")
    assert pages >= 3


def test_pdf_reports_a_clear_error_when_unavailable(monkeypatch) -> None:
    """Without ReportLab the failure names the extra to install."""
    import rocketopt.reports.pdf as pdf_module

    monkeypatch.setattr(pdf_module, "PDF_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"rocketopt\[reports\]"):
        pdf_module.write_pdf_report(None, None, "unused.pdf")  # type: ignore[arg-type]


def test_dedupe_removes_only_consecutive_duplicates() -> None:
    """The polyline cleaner keeps distinct points and drops repeats."""
    from rocketopt.cad.exporters import _dedupe

    points = [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    cleaned = _dedupe(points)

    assert cleaned == [(0.0, 0.0), (1.0, 0.0), (0.0, 0.0)]
    # Points a hair apart but above tolerance are kept.
    assert len(_dedupe([(0.0, 0.0), (1e-3, 0.0)])) == 2
    assert math.isclose(cleaned[1][0], 1.0)
