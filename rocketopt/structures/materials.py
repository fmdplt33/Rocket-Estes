"""Material, adhesive and surface-finish property database.

Every entry records mechanical properties needed by the structural and
aerodynamic layers, plus manufacturing metadata (cost, workability, stock
thicknesses) used by the optimiser's manufacturability objective.

Property sources
----------------
[W]  Forest Products Laboratory (2010). *Wood Handbook: Wood as an Engineering
     Material*. General Technical Report FPL-GTR-190, USDA Forest Service.
     Tables 5-3 (mechanical properties) and 4-3 (specific gravity), clear
     straight-grained wood at 12% moisture content.
[G]  Military Specification MIL-I-24768/27 (GEE-F, glass-cloth epoxy) and
     manufacturer data sheets for NEMA grade G-10/FR-4 laminate.
[C]  Daniel, I. M., & Ishai, O. (2006). *Engineering Mechanics of Composite
     Materials*, 2nd ed., Table A.2 - carbon/epoxy quasi-isotropic laminate.
[P]  Manufacturer data sheets for FDM thermoplastics, values quoted for
     solid (>=98% infill) specimens loaded in the XY print plane.
[A]  ASM International (1990). *Metals Handbook*, Vol. 2, 10th ed. -
     aluminium alloy 6061-T6.
[K]  Estes Industries body-tube specifications; spiral-wound kraft paperboard
     properties from Mark's Standard Handbook, 11th ed., Sec. 6.

Note on wood
------------
Wood is strongly orthotropic. The values below are the along-grain (L)
direction, which is how fin stock and nose-cone blanks are cut. The transverse
moduli are roughly one twentieth of the along-grain value; :attr:`Material.
transverse_modulus_ratio` records this so the fin flutter model can account
for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Final

__all__ = [
    "MaterialCategory",
    "Material",
    "Adhesive",
    "SurfaceFinish",
    "MATERIALS",
    "ADHESIVES",
    "get_material",
    "get_adhesive",
    "materials_in_category",
    "sheet_materials",
    "tube_materials",
    "nose_cone_materials",
]


class MaterialCategory(str, Enum):
    """Broad manufacturing family a material belongs to."""

    PAPER = "paper"
    WOOD = "wood"
    COMPOSITE = "composite"
    THERMOPLASTIC = "thermoplastic"
    METAL = "metal"


class SurfaceFinish(float, Enum):
    """Representative RMS surface roughness height in metres.

    Skin-friction drag depends on the ratio of roughness height to body length
    through the "admissible roughness" criterion, so the finish a builder
    applies is a genuine design variable.

    Values follow the finish categories tabulated in the OpenRocket Technical
    Documentation (Niskanen, S., 2013, *OpenRocket Technical Documentation*,
    Table 3.1), which are in turn drawn from Hoerner, S. F. (1965),
    *Fluid-Dynamic Drag*, Chapter 5. They are typical values; a specific
    build may differ by a factor of two either way.
    """

    ROUGH_UNFINISHED = 500e-6
    """Bare spiral-wound tube, visible spiral groove."""

    UNFINISHED = 150e-6
    """Sanded but unsealed cardboard or wood."""

    REGULAR_PAINT = 60e-6
    """Brushed or rattle-can paint over filled grain."""

    SMOOTH_PAINT = 20e-6
    """Sprayed and lightly wet-sanded paint."""

    POLISHED = 2e-6
    """Wet-sanded to 1200 grit and polished."""

    MIRROR = 0.5e-6
    """Competition-grade polished and waxed finish."""

    @property
    def label(self) -> str:
        """Human-readable name for reports and the UI."""
        return self.name.replace("_", " ").title()


@dataclass(frozen=True, slots=True)
class Material:
    """Mechanical and manufacturing properties of a structural material.

    All properties are SI. Strengths are ultimate values unless the material
    is ductile, in which case ``tensile_strength`` is the 0.2% offset yield
    strength (this is the conservative choice for a structural check).

    Attributes
    ----------
    name:
        Lower-case unique key used for lookup and serialisation.
    display_name:
        Human-readable name for reports.
    category:
        Manufacturing family; see :class:`MaterialCategory`.
    density:
        Mass density [kg/m^3].
    youngs_modulus:
        Along-grain / in-plane Young's modulus E [Pa].
    shear_modulus:
        In-plane shear modulus G [Pa]. Governs fin flutter.
    poisson_ratio:
        Poisson's ratio nu [-], dimensionless.
    tensile_strength:
        Ultimate (or yield, for ductile metals) tensile strength [Pa].
    compressive_strength:
        Ultimate compressive strength [Pa].
    shear_strength:
        Ultimate in-plane shear strength [Pa].
    transverse_modulus_ratio:
        E_transverse / E_longitudinal [-]. 1.0 for isotropic materials.
    cost_per_kg:
        Indicative material cost [currency units/kg] for the manufacturability
        objective. Relative magnitudes are what matter, not absolute values.
    machinability:
        Ease of cutting and shaping with hobby tools, 0 (hard) to 1 (trivial).
    bondability:
        Quality of a typical adhesive joint to this material, 0 to 1. Used to
        derate joint allowables in :mod:`rocketopt.structures.loads`.
    stock_thicknesses:
        Commercially available sheet thicknesses [m]. Empty when the material
        is not supplied as sheet.
    is_isotropic:
        Whether the material may be treated as isotropic in stress analysis.
    source:
        Citation key into this module's docstring reference list.
    """

    name: str
    display_name: str
    category: MaterialCategory
    density: float
    youngs_modulus: float
    shear_modulus: float
    poisson_ratio: float
    tensile_strength: float
    compressive_strength: float
    shear_strength: float
    transverse_modulus_ratio: float
    cost_per_kg: float
    machinability: float
    bondability: float
    source: str
    stock_thicknesses: tuple[float, ...] = field(default=())
    is_isotropic: bool = True

    def __post_init__(self) -> None:
        """Validate that the property set is physically admissible."""
        if self.density <= 0.0:
            raise ValueError(f"{self.name}: density must be positive")
        if self.youngs_modulus <= 0.0 or self.shear_modulus <= 0.0:
            raise ValueError(f"{self.name}: elastic moduli must be positive")
        # Thermodynamic admissibility for an isotropic solid requires
        # -1 < nu < 0.5; auxetic materials are outside this database's scope.
        if not 0.0 <= self.poisson_ratio < 0.5:
            raise ValueError(
                f"{self.name}: poisson_ratio {self.poisson_ratio} outside [0, 0.5)"
            )
        if not 0.0 <= self.machinability <= 1.0:
            raise ValueError(f"{self.name}: machinability must lie in [0, 1]")
        if not 0.0 <= self.bondability <= 1.0:
            raise ValueError(f"{self.name}: bondability must lie in [0, 1]")
        if any(t <= 0.0 for t in self.stock_thicknesses):
            raise ValueError(f"{self.name}: stock thicknesses must be positive")

    @property
    def specific_stiffness(self) -> float:
        """Young's modulus per unit density E/rho [m^2/s^2].

        The figure of merit for a stiffness-critical, mass-limited part such
        as a fin. Ashby, M. F. (2011). *Materials Selection in Mechanical
        Design*, 4th ed., Ch. 5.
        """
        return self.youngs_modulus / self.density

    @property
    def specific_strength(self) -> float:
        """Tensile strength per unit density sigma/rho [m^2/s^2].

        Figure of merit for a strength-critical, mass-limited part.
        Ashby (2011), Ch. 5.
        """
        return self.tensile_strength / self.density

    @property
    def transverse_modulus(self) -> float:
        """Young's modulus across the grain / fibre direction [Pa]."""
        return self.youngs_modulus * self.transverse_modulus_ratio

    def nearest_stock_thickness(self, target: float) -> float:
        """Snap a continuous thickness to the nearest purchasable sheet.

        The optimiser searches thickness as a continuous variable; this maps
        the result back onto stock a builder can actually buy.

        Parameters
        ----------
        target:
            Desired thickness [m].

        Returns
        -------
        float
            Nearest available stock thickness [m], or ``target`` unchanged if
            this material has no discrete stock sizes.
        """
        if not self.stock_thicknesses:
            return target
        return min(self.stock_thicknesses, key=lambda t: abs(t - target))


@dataclass(frozen=True, slots=True)
class Adhesive:
    """Properties of a bonding agent used at component joints.

    Attributes
    ----------
    name:
        Lower-case unique key.
    display_name:
        Human-readable name.
    lap_shear_strength:
        Ultimate lap-shear strength on a well-prepared joint [Pa]. Values are
        for the adhesive itself; the effective joint allowable is derated by
        the substrate's :attr:`Material.bondability`.
    peel_strength:
        Resistance to peel loading [N/m]. Low for rigid adhesives.
    density:
        Cured density [kg/m^3], used to account for fillet mass.
    cure_time_s:
        Time to handling strength [s].
    gap_filling:
        Whether the adhesive tolerates a loose-fitting joint.
    source:
        Citation for the quoted strength.
    """

    name: str
    display_name: str
    lap_shear_strength: float
    peel_strength: float
    density: float
    cure_time_s: float
    gap_filling: bool
    source: str

    def __post_init__(self) -> None:
        """Validate the adhesive property set."""
        if self.lap_shear_strength <= 0.0:
            raise ValueError(f"{self.name}: lap shear strength must be positive")
        if self.density <= 0.0:
            raise ValueError(f"{self.name}: density must be positive")


# ---------------------------------------------------------------------------
# Material database
# ---------------------------------------------------------------------------
#
# Thicknesses are quoted in metres. Imperial stock sizes are converted exactly
# (1 in = 25.4 mm): 1/32 in = 0.79375 mm, 1/16 in = 1.5875 mm,
# 3/32 in = 2.38125 mm, 1/8 in = 3.175 mm, 3/16 in = 4.7625 mm,
# 1/4 in = 6.35 mm.

_MATERIAL_LIST: Final[tuple[Material, ...]] = (
    Material(
        name="kraft_paper",
        display_name="Spiral-wound kraft paper tube",
        category=MaterialCategory.PAPER,
        density=690.0,
        youngs_modulus=2.4e9,
        shear_modulus=0.90e9,
        poisson_ratio=0.33,
        tensile_strength=27.0e6,
        compressive_strength=18.0e6,
        shear_strength=8.0e6,
        transverse_modulus_ratio=0.45,  # cross-machine direction of the ply
        cost_per_kg=18.0,
        machinability=0.95,
        bondability=0.95,
        source="[K]",
        stock_thicknesses=(0.35e-3, 0.45e-3, 0.55e-3, 0.70e-3, 1.00e-3),
        is_isotropic=False,
    ),
    Material(
        name="balsa",
        display_name="Balsa (light grade)",
        category=MaterialCategory.WOOD,
        density=160.0,
        youngs_modulus=3.4e9,
        shear_modulus=0.21e9,
        poisson_ratio=0.29,
        tensile_strength=21.6e6,
        compressive_strength=14.9e6,
        shear_strength=2.1e6,
        transverse_modulus_ratio=0.045,
        cost_per_kg=95.0,
        machinability=1.00,
        bondability=0.90,
        source="[W]",
        stock_thicknesses=(0.79375e-3, 1.5875e-3, 2.38125e-3, 3.175e-3, 4.7625e-3, 6.35e-3),
        is_isotropic=False,
    ),
    Material(
        name="basswood",
        display_name="Basswood",
        category=MaterialCategory.WOOD,
        density=420.0,
        youngs_modulus=10.1e9,
        shear_modulus=0.63e9,
        poisson_ratio=0.30,
        tensile_strength=60.0e6,
        compressive_strength=32.6e6,
        shear_strength=6.5e6,
        transverse_modulus_ratio=0.055,
        cost_per_kg=42.0,
        machinability=0.90,
        bondability=0.90,
        source="[W]",
        stock_thicknesses=(1.5875e-3, 2.38125e-3, 3.175e-3, 4.7625e-3, 6.35e-3),
        is_isotropic=False,
    ),
    Material(
        name="birch_plywood",
        display_name="Aircraft birch plywood",
        category=MaterialCategory.WOOD,
        density=680.0,
        youngs_modulus=8.5e9,
        shear_modulus=0.70e9,
        poisson_ratio=0.30,
        tensile_strength=40.0e6,
        compressive_strength=35.0e6,
        shear_strength=9.0e6,
        # Cross-plying largely equalises the in-plane moduli.
        transverse_modulus_ratio=0.75,
        cost_per_kg=35.0,
        machinability=0.75,
        bondability=0.88,
        source="[W]",
        stock_thicknesses=(0.79375e-3, 1.5875e-3, 2.38125e-3, 3.175e-3, 4.7625e-3),
        is_isotropic=False,
    ),
    Material(
        name="g10_fr4",
        display_name="G-10 / FR-4 glass-epoxy laminate",
        category=MaterialCategory.COMPOSITE,
        density=1850.0,
        youngs_modulus=18.6e9,
        shear_modulus=5.5e9,
        poisson_ratio=0.14,
        tensile_strength=262.0e6,
        compressive_strength=420.0e6,
        shear_strength=130.0e6,
        transverse_modulus_ratio=0.92,  # woven cloth, near-balanced
        cost_per_kg=60.0,
        machinability=0.35,  # abrasive, blunts tools, requires dust control
        bondability=0.70,  # needs abrading and solvent wipe
        source="[G]",
        stock_thicknesses=(0.79375e-3, 1.5875e-3, 2.38125e-3, 3.175e-3, 4.7625e-3),
    ),
    Material(
        name="carbon_fibre",
        display_name="Carbon/epoxy quasi-isotropic laminate",
        category=MaterialCategory.COMPOSITE,
        density=1550.0,
        youngs_modulus=55.0e9,
        shear_modulus=21.0e9,
        poisson_ratio=0.31,
        tensile_strength=550.0e6,
        compressive_strength=400.0e6,
        shear_strength=90.0e6,
        transverse_modulus_ratio=1.00,  # quasi-isotropic lay-up by definition
        cost_per_kg=220.0,
        machinability=0.30,
        bondability=0.65,
        source="[C]",
        stock_thicknesses=(0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3, 3.0e-3),
    ),
    Material(
        name="phenolic",
        display_name="Kraft-phenolic tube",
        category=MaterialCategory.COMPOSITE,
        density=1350.0,
        youngs_modulus=6.9e9,
        shear_modulus=2.6e9,
        poisson_ratio=0.33,
        tensile_strength=62.0e6,
        compressive_strength=130.0e6,
        shear_strength=20.0e6,
        transverse_modulus_ratio=0.60,
        cost_per_kg=45.0,
        machinability=0.60,  # brittle, chips when drilled
        bondability=0.85,
        source="[G]",
        stock_thicknesses=(0.8e-3, 1.2e-3, 1.6e-3, 2.4e-3),
    ),
    Material(
        name="pla",
        display_name="PLA (FDM printed, solid)",
        category=MaterialCategory.THERMOPLASTIC,
        density=1240.0,
        youngs_modulus=3.3e9,
        shear_modulus=1.20e9,
        poisson_ratio=0.36,
        tensile_strength=46.0e6,
        compressive_strength=60.0e6,
        shear_strength=25.0e6,
        # Interlayer (Z) strength is the weak axis of any FDM part.
        transverse_modulus_ratio=0.55,
        cost_per_kg=25.0,
        machinability=0.80,
        bondability=0.55,  # low surface energy; CA bonds poorly without prep
        source="[P]",
        is_isotropic=False,
    ),
    Material(
        name="petg",
        display_name="PETG (FDM printed, solid)",
        category=MaterialCategory.THERMOPLASTIC,
        density=1270.0,
        youngs_modulus=2.1e9,
        shear_modulus=0.77e9,
        poisson_ratio=0.38,
        tensile_strength=47.0e6,
        compressive_strength=55.0e6,
        shear_strength=22.0e6,
        transverse_modulus_ratio=0.70,  # better layer fusion than PLA
        cost_per_kg=28.0,
        machinability=0.70,
        bondability=0.50,
        source="[P]",
        is_isotropic=False,
    ),
    Material(
        name="abs",
        display_name="ABS (FDM printed, solid)",
        category=MaterialCategory.THERMOPLASTIC,
        density=1040.0,
        youngs_modulus=2.2e9,
        shear_modulus=0.80e9,
        poisson_ratio=0.35,
        tensile_strength=39.0e6,
        compressive_strength=48.0e6,
        shear_strength=20.0e6,
        transverse_modulus_ratio=0.60,
        cost_per_kg=26.0,
        machinability=0.85,
        bondability=0.75,  # solvent-weldable with acetone
        source="[P]",
        is_isotropic=False,
    ),
    Material(
        name="nylon_pa12",
        display_name="Nylon PA12 (SLS sintered)",
        category=MaterialCategory.THERMOPLASTIC,
        density=1010.0,
        youngs_modulus=1.7e9,
        shear_modulus=0.60e9,
        poisson_ratio=0.40 - 1e-9,  # quoted 0.40; kept strictly below 0.5
        tensile_strength=48.0e6,
        compressive_strength=52.0e6,
        shear_strength=24.0e6,
        transverse_modulus_ratio=0.95,  # sintering gives near-isotropy
        cost_per_kg=85.0,
        machinability=0.75,
        bondability=0.45,
        source="[P]",
    ),
    Material(
        name="aluminium_6061",
        display_name="Aluminium 6061-T6",
        category=MaterialCategory.METAL,
        density=2700.0,
        youngs_modulus=68.9e9,
        shear_modulus=26.0e9,
        poisson_ratio=0.33,
        tensile_strength=276.0e6,  # 0.2% offset yield, the conservative choice
        compressive_strength=276.0e6,
        shear_strength=207.0e6,
        transverse_modulus_ratio=1.00,
        cost_per_kg=12.0,
        machinability=0.55,
        bondability=0.60,  # requires etch or abrasion for a reliable bond
        source="[A]",
        stock_thicknesses=(0.5e-3, 0.8e-3, 1.0e-3, 1.6e-3, 2.0e-3, 3.0e-3),
    ),
)

MATERIALS: Final[dict[str, Material]] = {m.name: m for m in _MATERIAL_LIST}
"""Registry of every known material, keyed by :attr:`Material.name`."""


# ---------------------------------------------------------------------------
# Adhesive database
# ---------------------------------------------------------------------------
#
# Lap-shear strengths are typical published values for well-prepared joints.
# Where a joint fails in the substrate rather than the glue line (as PVA on
# wood normally does) the quoted figure is the substrate-limited value.

_ADHESIVE_LIST: Final[tuple[Adhesive, ...]] = (
    Adhesive(
        name="pva",
        display_name="PVA wood glue",
        lap_shear_strength=10.0e6,
        peel_strength=300.0,
        density=1100.0,
        cure_time_s=1800.0,
        gap_filling=False,
        source="Forest Products Laboratory FPL-GTR-190, Ch. 10",
    ),
    Adhesive(
        name="ca_thin",
        display_name="Cyanoacrylate, thin",
        lap_shear_strength=15.0e6,
        peel_strength=120.0,  # brittle glue line, poor peel resistance
        density=1050.0,
        cure_time_s=20.0,
        gap_filling=False,
        source="Petrie, E. M. (2007). Handbook of Adhesives and Sealants, 2nd ed.",
    ),
    Adhesive(
        name="ca_medium",
        display_name="Cyanoacrylate, medium (gap filling)",
        lap_shear_strength=13.0e6,
        peel_strength=150.0,
        density=1060.0,
        cure_time_s=60.0,
        gap_filling=True,
        source="Petrie (2007), Ch. 9",
    ),
    Adhesive(
        name="epoxy_5min",
        display_name="Five-minute epoxy",
        lap_shear_strength=17.0e6,
        peel_strength=700.0,
        density=1150.0,
        cure_time_s=300.0,
        gap_filling=True,
        source="Petrie (2007), Ch. 8",
    ),
    Adhesive(
        name="epoxy_structural",
        display_name="Structural epoxy (30 min)",
        lap_shear_strength=25.0e6,
        peel_strength=1500.0,
        density=1180.0,
        cure_time_s=1800.0,
        gap_filling=True,
        source="Petrie (2007), Ch. 8; ASTM D1002 lap-shear on abraded substrate",
    ),
)

ADHESIVES: Final[dict[str, Adhesive]] = {a.name: a for a in _ADHESIVE_LIST}
"""Registry of every known adhesive, keyed by :attr:`Adhesive.name`."""


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def get_material(name: str) -> Material:
    """Look up a material by name.

    Parameters
    ----------
    name:
        Material key, case-insensitive; spaces and hyphens are normalised to
        underscores so ``"Birch Plywood"`` resolves to ``"birch_plywood"``.

    Returns
    -------
    Material
        The matching material.

    Raises
    ------
    KeyError
        If no material matches, with the available names listed.
    """
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return MATERIALS[key]
    except KeyError:
        raise KeyError(
            f"Unknown material {name!r}. Available: {sorted(MATERIALS)}"
        ) from None


def get_adhesive(name: str) -> Adhesive:
    """Look up an adhesive by name.

    Parameters
    ----------
    name:
        Adhesive key, case-insensitive; spaces and hyphens are normalised to
        underscores.

    Returns
    -------
    Adhesive
        The matching adhesive.

    Raises
    ------
    KeyError
        If no adhesive matches, with the available names listed.
    """
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    try:
        return ADHESIVES[key]
    except KeyError:
        raise KeyError(
            f"Unknown adhesive {name!r}. Available: {sorted(ADHESIVES)}"
        ) from None


def materials_in_category(category: MaterialCategory) -> tuple[Material, ...]:
    """Return every material belonging to ``category``, ordered by name."""
    return tuple(
        sorted(
            (m for m in MATERIALS.values() if m.category is category),
            key=lambda m: m.name,
        )
    )


def sheet_materials() -> tuple[Material, ...]:
    """Return materials available as flat sheet, i.e. valid fin stock.

    Returns
    -------
    tuple of Material
        Materials with at least one entry in :attr:`Material.stock_thicknesses`,
        plus the printable thermoplastics, which can be produced at an
        arbitrary thickness. Ordered by name.
    """
    candidates = [
        m
        for m in MATERIALS.values()
        if m.stock_thicknesses or m.category is MaterialCategory.THERMOPLASTIC
    ]
    return tuple(sorted(candidates, key=lambda m: m.name))


def tube_materials() -> tuple[Material, ...]:
    """Return materials that body tubes are manufactured from.

    Metals are excluded: an aluminium airframe is impractical for an
    Estes-class model and would violate the NAR Model Rocket Safety Code,
    which requires lightweight non-metal body parts.
    """
    allowed = {"kraft_paper", "phenolic", "g10_fr4", "carbon_fibre"}
    return tuple(sorted((MATERIALS[n] for n in allowed), key=lambda m: m.name))


def nose_cone_materials() -> tuple[Material, ...]:
    """Return materials suitable for a nose cone.

    Balsa and basswood are turned; thermoplastics are printed; plywood is
    excluded because it cannot be turned to a smooth ogive.
    """
    allowed = {"balsa", "basswood", "pla", "petg", "abs", "nylon_pa12"}
    return tuple(sorted((MATERIALS[n] for n in allowed), key=lambda m: m.name))
