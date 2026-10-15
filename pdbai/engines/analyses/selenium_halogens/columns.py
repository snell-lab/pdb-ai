"""The selenium_halogen_sites table, and the atom column that links atoms to it.

Each row is one selenium or halogen atom of one model. The site key
columns name the atom, and the bonded_ ones the nearest atom it is bonded to,
which are missing for an ion. Both sets link to the atom level through the
standard site link, so they hold what the atom table holds for the same
atom.

The row sets the atom beside two references within its own structure:
the atoms it is bonded to, and the other atoms of its component. An atom
lost to radiation is modelled, when it is modelled at all, at a lower
occupancy or a higher B-factor than the atoms around it.

Most rows of a large run are chloride ions, since chloride is in most
crystallization buffers. They are rows for the reason bromide and iodide
are: the table holds every atom of its elements and leaves the choice of
which to read to the analysis, through parent_type.

The table has no cohort, technique, or temperature column, and holds no
verdict.
"""

from __future__ import annotations

from ...core import (
    PROFILE_RADIATION_SENSITIVITY,
    PROFILE_RESEARCH_READY,
    ColumnGranularity,
    ColumnSpec,
    TableColumnSpec,
    TableSpec,
    site_atom_link,
    site_key_columns,
)
from ..site_geometry import SPECIAL_POSITION_CUTOFF_A
from .constants import (
    BOND_TOLERANCE_A,
    PARENT_TYPES,
    SITE_ELEMENTS,
    SITE_STATUSES,
)

BONDED_PREFIX = "bonded_"
SELENIUM_HALOGEN_SITES_TABLE_NAME = "selenium_halogen_sites"
SELENIUM_HALOGEN_LINK_COLUMN = "selenium_halogen_site_id"

_TOLERANCE = f"{BOND_TOLERANCE_A:g} Å"

_MEASURE_COLUMNS: tuple[TableColumnSpec, ...] = (
    TableColumnSpec(
        "element",
        categories=SITE_ELEMENTS,
        description=(
            "Element symbol of the atom, in upper case: SE, F, CL, BR, or I. "
            "Fluorine and chlorine are read as the lighter halogens the "
            "others are compared with."
        ),
    ),
    TableColumnSpec(
        "parent_type",
        categories=PARENT_TYPES,
        description=(
            "The kind of component the atom belongs to. ion is a "
            "component whose only non-hydrogen atom is the atom, "
            "selenomethionine is MSE, polymer_residue is any other component "
            "with a place in a polymer sequence, such as a brominated "
            "nucleotide, and ligand is any other component. Most ions of a "
            "large run are the chloride of a buffer."
        ),
    ),
    TableColumnSpec(
        "component_atom_count",
        dtype="Int64",
        description=(
            "Non-hydrogen atoms the atom table holds with coordinates for the "
            "atom's component, the atom included."
        ),
    ),
    TableColumnSpec(
        "n_bonded_atoms",
        dtype="Int64",
        description=(
            "Atoms of the same model bonded to the atom: non-hydrogen, "
            "non-metal atoms with an occupancy that is not zero, within the "
            f"sum of the two covalent radii plus {_TOLERANCE}. Zero for an ion, "
            "and for a halogen bound only to a metal."
        ),
    ),
    TableColumnSpec(
        "bonded_element",
        description="Element of the nearest bonded atom. Missing when there is none.",
    ),
    TableColumnSpec(
        "bond_distance_min_A",
        dtype="float64",
        description=(
            "Distance in Å to the nearest bonded atom. Bond lengths are "
            "restrained in refinement, so read it with the other measures. "
            "Missing without a bonded atom."
        ),
    ),
    TableColumnSpec(
        "bond_distance_max_A",
        dtype="float64",
        description=(
            "Distance in Å to the farthest bonded atom, which differs from "
            "the nearest where the atom has two bonds, as the selenium of "
            "selenomethionine does."
        ),
    ),
    TableColumnSpec("occupancy", dtype="float64", description="Occupancy of the atom."),
    TableColumnSpec(
        "b_factor_A2",
        dtype="float64",
        description=(
            "Isotropic B-factor of the atom in Å². A neutron B-factor is "
            "refined against nuclear density and is not directly comparable "
            "with an X-ray one."
        ),
    ),
    TableColumnSpec(
        "bonded_b_factor_mean_A2",
        dtype="float64",
        description="Mean isotropic B-factor in Å² of the bonded atoms.",
    ),
    TableColumnSpec(
        "b_factor_ratio_to_bonded",
        dtype="float64",
        description=(
            "b_factor_A2 divided by bonded_b_factor_mean_A2. Missing when "
            "either is missing or not positive. For selenomethionine it "
            "compares with b_factor_ratio_to_reference of a met_thioether row "
            "of the residue_sites table, which sets the sulfur of methionine "
            "against the same two carbons."
        ),
    ),
    TableColumnSpec(
        "component_occupancy_mean",
        dtype="float64",
        description=(
            "Mean occupancy of the other atoms of the component, those that "
            "are not selenium or a halogen. An atom deposited below it "
            "was modelled as partly lost. Missing for an ion."
        ),
    ),
    TableColumnSpec(
        "component_b_factor_mean_A2",
        dtype="float64",
        description=(
            "Mean isotropic B-factor in Å² of the same atoms, those with an "
            "occupancy that is not zero."
        ),
    ),
    TableColumnSpec(
        "b_factor_ratio_to_component",
        dtype="float64",
        description=(
            "b_factor_A2 divided by component_b_factor_mean_A2. Missing when "
            "either is missing or not positive."
        ),
    ),
    TableColumnSpec(
        "on_special_position",
        dtype="boolean",
        description=(
            "True when a symmetry image of the atom lies within "
            f"{SPECIAL_POSITION_CUTOFF_A:g} Å of it, so the atom sits on a "
            "symmetry element and its occupancy is expected to be a fraction, "
            "as a halide ion's often is. Missing when the file gives no usable "
            "unit cell and space group."
        ),
    ),
    TableColumnSpec(
        "site_status",
        categories=SITE_STATUSES,
        description=(
            "ok, or the first gap that applies. no_bonded_atom when no atom "
            "lies within bonding distance, which is every ion and any atom "
            "whose bond was refined longer than the limit, and "
            "b_factor_unavailable when b_factor_ratio_to_bonded is missing."
        ),
    ),
)

SELENIUM_HALOGEN_SITES_TABLE = TableSpec(
    name=SELENIUM_HALOGEN_SITES_TABLE_NAME,
    description=(
        "One row per selenium, fluorine, chlorine, bromine, or iodine atom of "
        "each model that has "
        "coordinates, in the conformer the atom table keeps, whatever its "
        "occupancy. The row sets the atom beside the atoms it is bonded to "
        "and the other atoms of its component. Heavy metals are rows of the "
        "metal_sites table instead. It makes no judgment of the site."
    ),
    key=("site_id",),
    columns=(
        *site_key_columns(),
        *site_key_columns(BONDED_PREFIX),
        *_MEASURE_COLUMNS,
    ),
    links=(
        site_atom_link(role="site"),
        site_atom_link(BONDED_PREFIX, role="bonded"),
    ),
)

SELENIUM_HALOGEN_LINK_COLUMN_SPEC = ColumnSpec(
    name=SELENIUM_HALOGEN_LINK_COLUMN,
    group="Selenium and Halogen Fields",
    granularity=ColumnGranularity.ATOM,
    dtype="text",
    description=(
        "The atom's own site_id when it is a selenium or halogen "
        "atom with a row in the selenium_halogen_sites table, and missing on every "
        "other atom."
    ),
    profiles=frozenset({PROFILE_RESEARCH_READY, PROFILE_RADIATION_SENSITIVITY}),
)

__all__ = [
    "BONDED_PREFIX",
    "SELENIUM_HALOGEN_LINK_COLUMN",
    "SELENIUM_HALOGEN_LINK_COLUMN_SPEC",
    "SELENIUM_HALOGEN_SITES_TABLE",
    "SELENIUM_HALOGEN_SITES_TABLE_NAME",
]
