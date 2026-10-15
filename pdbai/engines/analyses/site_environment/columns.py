"""The site_environment table, and the atom column of the engine.

Each row is one group of the residue_sites table, with the same site_id, so
the two tables join one to one. A row describes what surrounds the group in
its own model: what it is hydrogen bonded to, the nearest disulfide, metal,
aromatic ring, cation, and ligand, how densely it is packed, how much of it
the solvent reaches and how deep it lies, and, where the entry has a unit
cell, how near the neighbouring molecules of the crystal come.

Every column but the crystal contacts and the waters can be taken from a
bare model, such as a predicted one, and gives the same value for a model
with or without its waters and ligands, apart from the ligand distance. A
missing value says that what the column looks for is not in the model. A
missing distance to a metal means the model has no metal atom, not that one
is far away. The crystal contacts are missing for a model without a unit
cell, and crystal_contacts_status says so. The water columns are missing for
a model without water.

The atom column atom_depth_A is apart from the table. It holds the depth the
table takes for its groups, taken for every protein atom.

The table has no technique, resolution, or cohort column. Those come from
the structures table at analysis time. It holds no verdict either.
"""

from __future__ import annotations

from ...core import (
    PROFILE_RADIATION_SENSITIVITY,
    ColumnGranularity,
    ColumnSpec,
    TableColumnSpec,
    TableSpec,
    site_atom_link,
    site_key_columns,
)
from ..residue_sites.constants import SITE_CLASS_NAMES
from .constants import (
    AROMATIC_RESIDUE_NAMES,
    CONTACT_COUNT_RADIUS_A,
    CONTACTS_STATUSES,
    ENVIRONMENT_STATUSES,
    EXPOSED_ATOM_AREA_A2,
    HBOND_DISTANCE_A,
    PACKING_INNER_RADIUS_A,
    PACKING_OUTER_RADIUS_A,
    SALT_BRIDGE_DISTANCE_A,
    SPHERE_POINT_COUNT,
    SULFUR_HBOND_DISTANCE_A,
    SYMMETRY_SEARCH_RADIUS_A,
)

SITE_ENVIRONMENT_TABLE_NAME = "site_environment"
DEPTH_COLUMN = "atom_depth_A"

_HBOND = (
    f"within {HBOND_DISTANCE_A:.1f} Å of a nitrogen or oxygen of the group, or "
    f"{SULFUR_HBOND_DISTANCE_A:.1f} Å of its sulfur"
)
_DEPTH = (
    "the distance in Å from the atom to the nearest protein atom of its model "
    f"with a solvent-accessible area of at least {EXPOSED_ATOM_AREA_A2:g} Å², "
    "zero for an atom at the surface"
)

_MEASURE_COLUMNS: tuple[TableColumnSpec, ...] = (
    TableColumnSpec(
        "site_class",
        categories=SITE_CLASS_NAMES,
        description=(
            "The kind of group, as the residue_sites table gives it on the row "
            "with the same site_id."
        ),
    ),
    TableColumnSpec(
        "n_group_atoms",
        dtype="Int64",
        description=(
            "Site atoms of the group with coordinates and an occupancy that is "
            "not zero, which every measure of the row is taken over. Missing "
            "when there are none."
        ),
    ),
    TableColumnSpec(
        "hbond_protein_count",
        dtype="Int64",
        description=(
            "Nitrogen and oxygen atoms of other amino acids of the model "
            f"{_HBOND}. A count by distance alone, without hydrogen positions "
            "or angles, so it holds every possible hydrogen bond and some that "
            "are not. Missing for a group without nitrogen, oxygen, or sulfur, "
            "such as the delta carbons of Leu."
        ),
    ),
    TableColumnSpec(
        "hbond_water_count",
        dtype="Int64",
        description=(
            f"Water oxygens of the model {_HBOND}, by distance alone. Waters of "
            "symmetry mates are not seen. Missing for a group without nitrogen, "
            "oxygen, or sulfur, and for a model without water, such as a "
            "predicted one."
        ),
    ),
    TableColumnSpec(
        "salt_bridge_partner_count",
        dtype="Int64",
        description=(
            "Residues of Lys, Arg, and His with a nitrogen of their charged "
            f"group (NZ; NE, NH1, NH2; ND1, NE2) within "
            f"{SALT_BRIDGE_DISTANCE_A:.1f} Å of an oxygen of the group. Read "
            "for the carboxylates of Asp, Glu, and the C-terminus only, and "
            "missing on every other class. Whether a His is charged depends on "
            "its surroundings, which the count does not judge."
        ),
    ),
    TableColumnSpec(
        "nearest_cation_distance_A",
        dtype="float64",
        description=(
            "Distance in Å from the nearest group atom to the nearest nitrogen "
            "of the charged group of a Lys, Arg, or His of another residue of "
            "the model. Missing when the model has none."
        ),
    ),
    TableColumnSpec(
        "in_disulfide",
        dtype="boolean",
        description=(
            "For a cysteine sulfur, True when its SG lies within the disulfide "
            "engine's search distance of the SG of another cysteine of the "
            "model, which takes in stretched bonds. Missing on every other "
            "class."
        ),
    ),
    TableColumnSpec(
        "nearest_disulfide_sulfur_distance_A",
        dtype="float64",
        description=(
            "Distance in Å from the nearest group atom to the nearest SG of a "
            "disulfide, found as in_disulfide, leaving out the group's own "
            "residue and, for a cysteine, the residue it is bonded to. Missing "
            "when the model has no other disulfide."
        ),
    ),
    TableColumnSpec(
        "nearest_metal_distance_A",
        dtype="float64",
        description=(
            "Distance in Å from the nearest group atom to the nearest metal "
            "atom of the model, as the residue_sites table takes it. Missing "
            "when the model has no metal atom."
        ),
    ),
    TableColumnSpec(
        "nearest_aromatic_ring_distance_A",
        dtype="float64",
        description=(
            "Distance in Å from the nearest group atom to the nearest centroid "
            "of the aromatic ring of a Phe, Tyr, Trp, or His of another residue "
            "of the model, each ring of Trp counted alone. A ring is read when "
            "all its atoms are modelled. Missing when the model has none. "
            "Electrons and holes may move between such rings and sensitive "
            "groups, which is why it is recorded."
        ),
    ),
    TableColumnSpec(
        "nearest_aromatic_res_name",
        categories=AROMATIC_RESIDUE_NAMES,
        description="The residue of that nearest ring, PHE, TYR, TRP, or HIS.",
    ),
    TableColumnSpec(
        "nearest_ligand_distance_A",
        dtype="float64",
        description=(
            "Distance in Å from the nearest group atom to the nearest heavy "
            "atom of the model that is not of an amino acid of a polymer, a "
            "water, or a metal, such as a ligand, a cofactor, an ion, or a "
            "nucleotide. Missing when the model has none, which is so of "
            "every predicted model of a protein alone."
        ),
    ),
    TableColumnSpec(
        "heavy_atom_count_6A",
        dtype="Int64",
        description=(
            "Protein heavy atoms of the model within "
            f"{PACKING_INNER_RADIUS_A:.0f} Å of the centroid of the group's "
            "atoms, leaving out the group's own atoms. Protein atoms are those "
            "of standard amino acids and of the modified residues of a "
            "polymer, so waters and ligands are not counted, and the atoms of "
            "symmetry mates are not either."
        ),
    ),
    TableColumnSpec(
        "heavy_atom_count_10A",
        dtype="Int64",
        description=(
            f"The same count within {PACKING_OUTER_RADIUS_A:.0f} Å, a measure "
            "of burial."
        ),
    ),
    TableColumnSpec(
        "site_sasa_A2",
        dtype="float64",
        description=(
            "Solvent-accessible area in Å² of the group's atoms among the "
            "protein atoms of their model, by the method of Shrake and Rupley "
            f"with the SASA engine's radii, probe, and {SPHERE_POINT_COUNT} test "
            "points per atom. Waters and ligands do not cover it."
        ),
    ),
    TableColumnSpec(
        "site_relative_sasa",
        dtype="float64",
        description=(
            "site_sasa_A2 over the area of the same atoms with their own "
            "residue alone, so 1 for a group the rest of the model does not "
            "touch and 0 for one it covers. Missing when the area of the "
            "residue alone is zero."
        ),
    ),
    TableColumnSpec(
        "site_depth_A",
        dtype="float64",
        description=(
            f"The mean over the group's atoms of {_DEPTH}. A buried group has "
            "a depth of several Å."
        ),
    ),
    TableColumnSpec(
        "water_count_6A",
        dtype="Int64",
        description=(
            "Water oxygens of the model within "
            f"{CONTACT_COUNT_RADIUS_A:.0f} Å of a group atom. Missing for a "
            "model without water. How many waters a depositor models depends "
            "on the resolution and on the depositor, so compare it within one "
            "structure or between structures of like resolution."
        ),
    ),
    TableColumnSpec(
        "nearest_symmetry_mate_distance_A",
        dtype="float64",
        description=(
            "Distance in Å from the nearest group atom to the nearest heavy "
            "atom of a symmetry mate, an image of the model's non-water atoms "
            "under the space group and the translations of the lattice. Images "
            "that build the biological assembly are mates too. Missing when "
            f"no mate comes within {SYMMETRY_SEARCH_RADIUS_A:.0f} Å, and for a "
            "model without a unit cell, which crystal_contacts_status tells "
            "apart."
        ),
    ),
    TableColumnSpec(
        "symmetry_mate_atom_count_6A",
        dtype="Int64",
        description=(
            "Heavy atoms of symmetry mates within "
            f"{CONTACT_COUNT_RADIUS_A:.0f} Å of a group atom. Missing for a "
            "model without a unit cell."
        ),
    ),
    TableColumnSpec(
        "crystal_contacts_status",
        categories=CONTACTS_STATUSES,
        description=(
            "ok when a symmetry mate lies within "
            f"{SYMMETRY_SEARCH_RADIUS_A:.0f} Å of the group, none_in_range "
            "when none does, and no_unit_cell when the entry has no usable cell "
            "and space group, as for a predicted model or an NMR ensemble. "
            "Missing on a row without group atoms in an entry with a cell."
        ),
    ),
    TableColumnSpec(
        "environment_status",
        categories=ENVIRONMENT_STATUSES,
        description=(
            "ok, or no_group_atoms when the model holds none of the group's "
            "site atoms with an occupancy that is not zero, so that every "
            "measure of the row is missing."
        ),
    ),
)

# The columns the row takes from its residue_sites row.
KEY_COLUMNS: tuple[str, ...] = (
    *(column.name for column in site_key_columns()),
    "site_class",
)

SITE_ENVIRONMENT_TABLE = TableSpec(
    name=SITE_ENVIRONMENT_TABLE_NAME,
    description=(
        "One row per group of the residue_sites table, with the same site_id: "
        "the surroundings of the group in its own model, its hydrogen bonds "
        "and salt bridges, the nearest disulfide, metal, aromatic ring, cation, "
        "and ligand, the protein atoms packed about it, its solvent-accessible "
        "area and depth, and, for an entry with a unit cell, its distance from "
        "the neighbouring molecules of the crystal. The row makes no judgment "
        "of the site."
    ),
    key=("site_id",),
    columns=(*site_key_columns(), *_MEASURE_COLUMNS),
    links=(site_atom_link(),),
)

DEPTH_COLUMN_SPEC = ColumnSpec(
    name=DEPTH_COLUMN,
    group="Site Environment Fields",
    granularity=ColumnGranularity.ATOM,
    dtype="numeric",
    description=(
        f"For every protein atom, {_DEPTH}. Protein atoms are the heavy atoms "
        "of standard amino acids and of the modified residues of a polymer "
        "with an occupancy that is not zero, and only they cover one another, "
        "so a predicted model gives the same depth as a deposited one of the "
        "same coordinates. Missing on every other atom."
    ),
    # Only the profile for models of damage switches the engine on, since it
    # computes the solvent-accessible area of every protein atom.
    profiles=frozenset({PROFILE_RADIATION_SENSITIVITY}),
)

__all__ = [
    "DEPTH_COLUMN",
    "DEPTH_COLUMN_SPEC",
    "KEY_COLUMNS",
    "SITE_ENVIRONMENT_TABLE",
    "SITE_ENVIRONMENT_TABLE_NAME",
]
