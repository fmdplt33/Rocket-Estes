"""Bill of materials and build documentation.

The bill of materials lists what a builder must actually buy or cut, with
stock sizes rather than the solver's continuous values, so it can be taken
straight to a workbench.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rocketopt.geometry.components import BodyTube, Transition
from rocketopt.geometry.rocket import Rocket
from rocketopt.utils.logging import get_logger

_log = get_logger(__name__)

__all__ = [
    "BomItem",
    "bill_of_materials",
    "build_manufacturing_guide",
    "write_manufacturing_guide",
]


@dataclass(frozen=True, slots=True)
class BomItem:
    """One line of the bill of materials.

    Attributes
    ----------
    name:
        Part name.
    specification:
        Material and dimensions as a builder would order them.
    quantity:
        How many are needed.
    units:
        Unit of quantity, e.g. ``"off"`` or ``"mm"``.
    mass:
        Total mass of this line [kg].
    estimated_cost:
        Indicative cost of the material used.
    """

    name: str
    specification: str
    quantity: float
    units: str
    mass: float
    estimated_cost: float = 0.0


def bill_of_materials(rocket: Rocket) -> list[BomItem]:
    """Build the bill of materials for a design.

    Parameters
    ----------
    rocket:
        The design.

    Returns
    -------
    list of BomItem
        One line per purchasable or cuttable item.
    """
    items: list[BomItem] = []

    nose = rocket.nose
    items.append(
        BomItem(
            name="Nose cone",
            specification=(
                f"{nose.shape.label}, {nose.length * 1e3:.0f} mm long, "
                f"{nose.base_diameter * 1e3:.1f} mm base, "
                f"{nose.material.display_name}"
                + ("" if nose.solid else f", {nose.wall_thickness * 1e3:.1f} mm wall")
            ),
            quantity=1,
            units="off",
            mass=nose.mass,
            estimated_cost=nose.material_volume * nose.material.density
            * nose.material.cost_per_kg,
        )
    )

    for index, placed in enumerate(rocket.sections, start=1):
        section = placed.section
        if isinstance(section, BodyTube):
            items.append(
                BomItem(
                    name=f"Body tube {index}",
                    specification=(
                        f"{section.material.display_name}, "
                        f"{section.diameter * 1e3:.1f} mm OD x "
                        f"{section.wall_thickness * 1e3:.2f} mm wall, "
                        f"{section.length * 1e3:.0f} mm long"
                    ),
                    quantity=1,
                    units="off",
                    mass=section.mass,
                    estimated_cost=section.mass * section.material.cost_per_kg,
                )
            )
        elif isinstance(section, Transition):
            items.append(
                BomItem(
                    name=f"{'Boat-tail' if section.is_boat_tail else 'Shoulder'} {index}",
                    specification=(
                        f"{section.material.display_name}, "
                        f"{section.fore_radius * 2e3:.1f} to "
                        f"{section.aft_radius * 2e3:.1f} mm, "
                        f"{section.length * 1e3:.0f} mm long"
                    ),
                    quantity=1,
                    units="off",
                    mass=section.mass,
                    estimated_cost=section.mass * section.material.cost_per_kg,
                )
            )

    fins = rocket.fins
    # Sheet area needed, with 20% allowance for the cutting layout.
    sheet_area = fins.area_single * fins.count * 1.2
    items.append(
        BomItem(
            name="Fins",
            specification=(
                f"{fins.material.display_name}, {fins.thickness * 1e3:.2f} mm sheet; "
                f"root {fins.root_chord * 1e3:.0f} mm, tip {fins.tip_chord * 1e3:.0f} mm, "
                f"span {fins.span * 1e3:.0f} mm, sweep {fins.sweep_length * 1e3:.0f} mm, "
                f"{fins.airfoil.label.lower()} section; "
                f"at least {sheet_area * 1e4:.0f} cm² of sheet"
            ),
            quantity=fins.count,
            units="off",
            mass=fins.mass,
            estimated_cost=fins.mass * 1.2 * fins.material.cost_per_kg,
        )
    )

    mount = rocket.motor_mount
    items.append(
        BomItem(
            name="Motor mount",
            specification=(
                f"{mount.material.display_name}, "
                f"{mount.inner_diameter * 1e3:.1f} mm ID x "
                f"{mount.outer_diameter * 1e3:.1f} mm OD "
                f"({mount.wall_thickness * 1e3:.2f} mm wall), "
                f"{mount.length * 1e3:.0f} mm long"
                + (
                    "; a self-centring fit in the airframe bore, print or turn "
                    "it as one piece"
                    if mount.is_minimum_diameter
                    else ""
                )
            ),
            quantity=1,
            units="off",
            mass=mount.tube_mass,
            estimated_cost=mount.tube_mass * mount.material.cost_per_kg,
        )
    )
    if mount.ring_mass > 0.0:
        items.append(
            BomItem(
                name="Centring rings",
                specification=(
                    f"{mount.material.display_name}, "
                    f"{mount.outer_radius * 2e3:.1f} mm ID x "
                    f"{mount.body_inner_radius * 2e3:.1f} mm OD"
                ),
                quantity=mount.centring_ring_count,
                units="off",
                mass=mount.ring_mass,
                estimated_cost=mount.ring_mass * mount.material.cost_per_kg,
            )
        )

    if rocket.launch_lug is not None:
        lug = rocket.launch_lug
        items.append(
            BomItem(
                name="Launch lug",
                specification=(
                    f"{lug.inner_radius * 2e3:.1f} mm bore, "
                    f"{lug.length * 1e3:.0f} mm long"
                ),
                quantity=1,
                units="off",
                mass=lug.mass,
                estimated_cost=lug.mass * lug.material.cost_per_kg,
            )
        )

    recovery = rocket.recovery
    if recovery.mass > 0.0:
        if recovery.diameter > 0.0:
            spec = (
                f"{recovery.diameter * 1e3:.0f} mm flat canopy, "
                f"{recovery.shroud_line_count} shroud lines at "
                f"{recovery.shroud_line_length * 1e3:.0f} mm"
            )
        else:
            spec = (
                f"{recovery.streamer_width * 1e3:.0f} x "
                f"{recovery.streamer_length * 1e3:.0f} mm streamer"
            )
        items.append(
            BomItem(
                name="Recovery device",
                specification=spec,
                quantity=1,
                units="off",
                mass=recovery.mass,
                estimated_cost=0.0,
            )
        )

    # Consumables that carry real mass but are not "parts".
    fillet_volume = fins.fillet_volume_single * fins.count
    if fillet_volume > 0.0:
        # Epoxy density from the adhesive table.
        fillet_mass = fillet_volume * 1150.0
        items.append(
            BomItem(
                name="Fillet adhesive",
                specification=(
                    f"Epoxy, {fins.fillet_radius * 1e3:.1f} mm radius fillets, "
                    f"{2 * fins.count} fillets total"
                ),
                quantity=fillet_volume * 1e6,
                units="ml",
                mass=fillet_mass,
                estimated_cost=0.0,
            )
        )

    if rocket.nose_ballast_mass > 0.0:
        items.append(
            BomItem(
                name="Nose ballast",
                specification="Lead shot or modelling clay, packed into the tip",
                quantity=rocket.nose_ballast_mass * 1e3,
                units="g",
                mass=rocket.nose_ballast_mass,
            )
        )

    return items


def build_manufacturing_guide(rocket: Rocket) -> str:
    """Build a step-by-step construction guide as Markdown.

    Parameters
    ----------
    rocket:
        The design.

    Returns
    -------
    str
        The guide.
    """
    fins = rocket.fins
    mount = rocket.motor_mount
    body = next(
        (p.section for p in rocket.sections if isinstance(p.section, BodyTube)), None
    )
    body_length = body.length if body else 0.0

    fin_spacing = 360.0 / fins.count
    fin_station = fins.position - rocket.nose.length

    items = bill_of_materials(rocket)
    bom_lines = "\n".join(
        f"| {i.name} | {i.specification} | {i.quantity:g} {i.units} | "
        f"{i.mass * 1e3:.2f} g |"
        for i in items
    )

    return f"""# Manufacturing and Assembly Guide: {rocket.name}

## Bill of materials

| Item | Specification | Quantity | Mass |
|---|---|---|---|
{bom_lines}

## Cutting list

**Body tube.** Cut to {body_length * 1e3:.0f} mm. Cut square: a tube cut off
square by more than about a degree will set the fins crooked no matter how
carefully they are aligned.

**Fins.** Mark {fins.count} identical fins on
{fins.thickness * 1e3:.2f} mm {fins.material.display_name.lower()}:

- Root chord {fins.root_chord * 1e3:.1f} mm
- Tip chord {fins.tip_chord * 1e3:.1f} mm
- Span {fins.span * 1e3:.1f} mm
- Leading edge swept {fins.sweep_length * 1e3:.1f} mm aft from root to tip

Stack all {fins.count} blanks and sand them together to a single template.
Fins that differ from one another induce roll and cost altitude.

**Fin section.** Shape to a {fins.airfoil.label.lower()} section.
{_airfoil_instruction(rocket)}

**Motor mount.** {mount.inner_diameter * 1e3:.1f} mm bore x
{mount.outer_diameter * 1e3:.1f} mm outside diameter, cut to
{mount.length * 1e3:.0f} mm.

## Assembly

1. **Motor mount.** {_mount_instruction(rocket)}

2. **Install the mount.** Slide it into the body tube so its aft face ends
   flush with the aft end of the tube, and glue it. Check it has gone in
   square: a mount glued in crooked points the thrust off the axis.

3. **Mark the fin lines.** Wrap a strip of paper around the tube, mark the
   circumference, divide it into {fins.count} equal parts
   ({fin_spacing:.0f}° apart) and transfer the marks. Extend them into
   straight lines using a door frame or an aluminium angle as a straight edge.

4. **Attach the fins.** The fin root leading edge sits
   {fin_station * 1e3:.0f} mm aft of the body tube's forward end. Glue one fin
   at a time and let each set before starting the next. Check each fin is
   square to the body from two directions.

5. **Fillets.** Apply a {fins.fillet_radius * 1e3:.1f} mm radius fillet to both
   sides of every fin root.
   {_fillet_note(rocket)}

6. **Launch lug.** Glue the lug along one of the fin lines so it sits in a fin's
   wake rather than in clean flow.

7. **Recovery.** Attach the shock cord to the body tube, fold the parachute and
   pack it with flame-resistant wadding between it and the motor.

8. **Finish.** {_finish_instruction(rocket)}

## Pre-flight checks

- Confirm the balance point is {rocket.cg_at(0.0) * 1e3:.0f} mm aft of the nose
  tip **with a motor installed**. Check by balancing the rocket on a straight
  edge.
- Confirm the nose cone is a friction fit: tight enough not to fall out, loose
  enough to be pulled off by hand.
- Confirm all fins are firmly attached. A fin that can be twisted by hand will
  come off in flight.
"""


def _mount_instruction(rocket: Rocket) -> str:
    """Return fitting instructions for the motor mount.

    A mount turned to the airframe bore locates itself and needs no rings; a
    narrower one has to be centred by them.
    """
    mount = rocket.motor_mount
    thrust_ring = (
        "Add a thrust ring or engine hook at the forward end so the motor "
        "cannot be driven into the airframe under thrust."
    )
    if mount.is_minimum_diameter:
        return (
            f"The mount is {mount.outer_diameter * 1e3:.1f} mm outside diameter "
            f"against a {mount.body_inner_radius * 2e3:.1f} mm airframe bore, so "
            f"it centres itself along its whole length and needs no centring "
            f"rings. Ease the outside diameter with fine paper until it slides "
            f"in with firm hand pressure, no more. {thrust_ring}"
        )
    return (
        f"Fit the centring rings to the motor tube, one flush with the aft end "
        f"and one {mount.length * 0.7e3:.0f} mm forward. {thrust_ring}"
    )


def _airfoil_instruction(rocket: Rocket) -> str:
    """Return shaping instructions for the chosen fin section."""
    from rocketopt.geometry.components import FinAirfoil

    return {
        FinAirfoil.SQUARE: (
            "Leave the edges square. This is the simplest option and the "
            "draggiest; rounding the leading edge alone recovers most of the "
            "difference for a few minutes of sanding."
        ),
        FinAirfoil.ROUNDED: (
            "Round the leading and trailing edges to a semicircle. Sand along "
            "the span, not across it, so the edge stays straight."
        ),
        FinAirfoil.AIRFOIL: (
            "Round the leading edge and taper the aft two-thirds of the chord "
            "to a sharp trailing edge. Mark the taper start line on both faces "
            "first so the section stays symmetric."
        ),
        FinAirfoil.DOUBLE_WEDGE: (
            "Taper from a sharp leading edge to full thickness at mid-chord, "
            "then back to a sharp trailing edge. Mark the mid-chord line on "
            "both faces before sanding."
        ),
    }[rocket.fins.airfoil]


def _fillet_note(rocket: Rocket) -> str:
    """Return a note explaining the fillet's structural role."""
    fins = rocket.fins
    if fins.fillet_radius <= 0.0:
        return (
            "No fillet is specified. Consider adding one anyway: it is the "
            "single most effective way to strengthen the fin joint, and it "
            "reduces interference drag."
        )
    bond_gain = (fins.thickness + 2 * fins.fillet_radius) / fins.thickness
    return (
        f"The fillet widens the bonded area by a factor of {bond_gain:.1f} "
        f"compared with the bare fin root, and reduces junction interference "
        f"drag. Do not skip it."
    )


def _finish_instruction(rocket: Rocket) -> str:
    """Return finishing instructions matched to the specified surface finish."""
    finish = rocket.surface_finish
    roughness_um = float(finish) * 1e6
    base = (
        f"The design assumes a {finish.label.lower()} finish "
        f"({roughness_um:.1f} µm roughness), which the drag model depends on."
    )
    if roughness_um <= 2.0:
        return (
            base
            + " Fill the tube's spiral groove with sanding sealer, sand to "
            "1200 grit, then polish. This is a competition-grade finish and "
            "takes several hours."
        )
    if roughness_um <= 20.0:
        return (
            base
            + " Fill the spiral groove, spray-paint, then wet-sand lightly to "
            "remove orange peel."
        )
    if roughness_um <= 60.0:
        return base + " Fill the spiral groove and apply a normal coat of paint."
    return (
        base
        + " Little or no finishing is assumed. Sealing and painting the tube "
        "would reduce drag noticeably; re-run the analysis with a smoother "
        "finish to see how much."
    )


def write_manufacturing_guide(rocket: Rocket, path: str | Path) -> Path:
    """Write the manufacturing guide to a Markdown file.

    Parameters
    ----------
    rocket:
        The design.
    path:
        Destination path.

    Returns
    -------
    pathlib.Path
        The resolved path written.
    """
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(build_manufacturing_guide(rocket), encoding="utf-8")
    _log.info("Wrote manufacturing guide to %s", file_path)
    return file_path.resolve()
