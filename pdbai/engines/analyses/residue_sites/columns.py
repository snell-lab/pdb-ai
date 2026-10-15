"""The residue_sites table, and the two atom columns of the engine.

Each row is one group of atoms in one residue of one model: the atoms that
specific radiation damage removes, or the atoms of a control residue of the
same shape. The site key columns name the row by its key atom, the atom the
group is bonded to, which stays when the group is lost. A residue cut back
to that atom therefore still has a row, with no site atoms modelled.

The atom column residue_site_id holds the row's site_id on every site atom
the model holds and is missing everywhere else, so the density, the
B-factor, and every other atom column of a site are one join away.

The atom column b_factor_ratio_to_parent is apart from the table. It holds
the ratio the table takes for its groups, taken for every atom of a standard
amino acid against the atom it hangs from.

The table has no cohort, technique, or temperature column. Those come from
the structures table at analysis time. It holds no verdict either. A ratio
or a count of missing atoms states what the model holds and leaves what that
means to the reader.
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
from .constants import (
    METAL_CONTACT_DISTANCE_A,
    SENSITIVE_CLASS_NAMES,
    SITE_CLASS_NAMES,
    SITE_CLASSES,
    SITE_STATUSES,
)

RESIDUE_SITES_TABLE_NAME = "residue_sites"
RESIDUE_SITE_LINK_COLUMN = "residue_site_id"
B_FACTOR_RATIO_COLUMN = "b_factor_ratio_to_parent"

_METAL_CONTACT = f"{METAL_CONTACT_DISTANCE_A:.1f} Å"


def _class_list() -> str:
    """Return every class with its atoms, as the site_class description gives it.

    A class deposited under more than one code lists the atoms of each code.
    """
    codes: dict[str, list[str]] = {}
    for site in SITE_CLASSES:
        residue = site.residue or "any standard amino acid with an OXT"
        codes.setdefault(site.name, []).append(
            f"{residue}: {', '.join(site.site_atoms)}, attached to "
            f"{' and '.join(site.reference_atoms)}"
        )
    return "; ".join(f"{name} ({'; '.join(parts)})" for name, parts in codes.items())


_MEASURE_COLUMNS: tuple[TableColumnSpec, ...] = (
    TableColumnSpec(
        "site_class",
        categories=SITE_CLASS_NAMES,
        description=(
            "The kind of group, with its site atoms and the reference atoms "
            f"they are attached to: {_class_list()}. The row's site key "
            "columns name the first reference atom."
        ),
    ),
    TableColumnSpec(
        "is_control",
        dtype="boolean",
        description=(
            "True for a group of the same shape as a sensitive one, which the "
            "same damage does not remove. False for a sensitive group."
        ),
    ),
    TableColumnSpec(
        "control_for",
        categories=SENSITIVE_CLASS_NAMES,
        description=(
            "The sensitive class a control group stands beside. Missing on a "
            "sensitive group. Asn and Gln have the shape of Asp and Glu. The "
            "others are approximate, since no standard residue has the shape "
            "of Tyr, Met, Cys, or His without the sensitive atoms and with the "
            "same elements. Phe, Leu, and Ile have the shape of Tyr and Met "
            "with a carbon in the atom's place, Thr has an oxygen on a carbon "
            "as Tyr has, Ser is Cys with an oxygen in place of the sulfur, and "
            "the ring carbons of Phe are an aromatic ring on CG as the "
            "imidazole of His is, with carbon in place of its two nitrogens. "
            "The indole of Trp, the oxidation products, and the C-terminus "
            "have no control."
        ),
    ),
    TableColumnSpec(
        "n_site_atoms_expected",
        dtype="Int64",
        description="Site atoms the class defines.",
    ),
    TableColumnSpec(
        "n_site_atoms_modelled",
        dtype="Int64",
        description=(
            "Site atoms the atom table holds with coordinates, whatever their "
            "occupancy. Fewer than expected means the model was cut back, "
            "which depositors do for disorder as well as for damage."
        ),
    ),
    TableColumnSpec(
        "n_site_atoms_zero_occupancy",
        dtype="Int64",
        description="Modelled site atoms deposited with an occupancy of zero or below.",
    ),
    TableColumnSpec(
        "site_occupancy_min",
        dtype="float64",
        description="Lowest occupancy among the modelled site atoms.",
    ),
    TableColumnSpec(
        "site_occupancy_mean",
        dtype="float64",
        description="Mean occupancy of the modelled site atoms.",
    ),
    TableColumnSpec(
        "site_b_factor_mean_A2",
        dtype="float64",
        description=(
            "Mean isotropic B-factor in Å² of the modelled site atoms whose "
            "occupancy is not zero. A neutron B-factor is refined against "
            "nuclear density and is not directly comparable with an X-ray one."
        ),
    ),
    TableColumnSpec(
        "site_b_factor_max_A2",
        dtype="float64",
        description="Highest isotropic B-factor in Å² among the same site atoms.",
    ),
    TableColumnSpec(
        "reference_b_factor_mean_A2",
        dtype="float64",
        description=(
            "Mean isotropic B-factor in Å² of the modelled reference atoms "
            "whose occupancy is not zero."
        ),
    ),
    TableColumnSpec(
        "b_factor_ratio_to_reference",
        dtype="float64",
        description=(
            "site_b_factor_mean_A2 divided by reference_b_factor_mean_A2, a "
            "comparison within one residue. Missing when either is missing or "
            "not positive."
        ),
    ),
    TableColumnSpec(
        "attachment_distance_min_A",
        dtype="float64",
        description=(
            "Shortest of the bonds that attach the group, in Å, each between a "
            "reference atom and a site atom. For a carboxylate it is the one "
            "bond that decarboxylation breaks. Bond lengths are restrained in "
            "refinement, so read it with the other measures."
        ),
    ),
    TableColumnSpec(
        "attachment_distance_max_A",
        dtype="float64",
        description="Longest of the same bonds in Å.",
    ),
    TableColumnSpec(
        "has_alternate_conformer",
        dtype="boolean",
        description=(
            "True when a modelled site atom carries an alternate-conformation "
            "identifier. The row describes the conformer the atom table keeps."
        ),
    ),
    TableColumnSpec(
        "nearest_metal_distance_A",
        dtype="float64",
        description=(
            "Distance in Å from the nearest site atom to the nearest metal "
            "atom of the same model, among the atoms of the file only. "
            "Missing when the model has no metal atom or no site atom."
        ),
    ),
    TableColumnSpec(
        "metal_bound",
        dtype="boolean",
        description=(
            f"True when nearest_metal_distance_A is at most {_METAL_CONTACT}. "
            "A group bound to a metal may be protected, so such rows can be "
            "set apart. False otherwise, and never missing. A metal of a "
            "symmetry mate is not seen."
        ),
    ),
    TableColumnSpec(
        "site_status",
        categories=SITE_STATUSES,
        description=(
            "ok, or the first gap that applies. site_atoms_missing when the "
            "model holds fewer site atoms than the class defines, "
            "zero_occupancy when a modelled site atom has an occupancy of "
            "zero, and b_factor_unavailable when the ratio is missing."
        ),
    ),
)

RESIDUE_SITES_TABLE = TableSpec(
    name=RESIDUE_SITES_TABLE_NAME,
    description=(
        "One row per radiation-sensitive group of each polymer residue and "
        "model, and per control group of the same shape, in the conformer the "
        "atom table keeps. The groups are the carboxylates of Asp, Glu, and "
        "the C-terminus, the sulfurs of Met and Cys, the hydroxyl of Tyr, the "
        "indole of Trp, and the imidazole of His, with the amides of Asn and "
        "Gln, the ring carbons of Phe, the delta carbons of Leu and Ile, and "
        "the hydroxyls of Thr and Ser as controls. A cysteine is a row "
        "whether or not it is in a disulfide, which the disulfides table "
        "says. A residue deposited oxidized at one of these groups, a "
        "cysteine as CSO, CSD, or OCS, a methionine as its sulfoxide or "
        "sulfone, a tryptophan hydroxylated, as a hydroperoxide, or as "
        "kynurenine, a histidine with an oxygen on its ring, or a tyrosine "
        "or phenylalanine with a second oxygen or a first, is a row for the "
        "oxygens added. A residue is a row when the model holds its key atom. "
        "The row makes no judgment of the site."
    ),
    key=("site_id",),
    columns=(*site_key_columns(), *_MEASURE_COLUMNS),
    links=(site_atom_link(),),
)

RESIDUE_SITE_LINK_COLUMN_SPEC = ColumnSpec(
    name=RESIDUE_SITE_LINK_COLUMN,
    group="Residue Site Fields",
    granularity=ColumnGranularity.ATOM,
    dtype="text",
    description=(
        "The site_id of the atom's row in the residue_sites table when the "
        "atom is a site atom of one, and missing on every other atom."
    ),
    profiles=frozenset({PROFILE_RESEARCH_READY, PROFILE_RADIATION_SENSITIVITY}),
)

B_FACTOR_RATIO_COLUMN_SPEC = ColumnSpec(
    name=B_FACTOR_RATIO_COLUMN,
    group="Residue Site Fields",
    granularity=ColumnGranularity.ATOM,
    dtype="numeric",
    description=(
        "The atom's isotropic B-factor divided by the mean B-factor of its "
        "parents, the atoms it is bonded to one bond nearer the CA, for "
        "every atom of the 20 standard amino acids but the CA. It is the "
        "ratio the residue_sites table takes for a few named groups, taken "
        "for every atom. A B-factor rises along any side chain, so compare "
        "one atom type across structures or doses, not one atom with "
        "another. Missing on the CA, on atoms of any other component, and "
        "where the atom or its parent has no coordinates, an occupancy of "
        "zero, or a B-factor that is not positive. A neutron B-factor is not "
        "directly comparable with an X-ray one."
    ),
    profiles=frozenset({PROFILE_RESEARCH_READY, PROFILE_RADIATION_SENSITIVITY}),
)

__all__ = [
    "B_FACTOR_RATIO_COLUMN",
    "B_FACTOR_RATIO_COLUMN_SPEC",
    "RESIDUE_SITES_TABLE",
    "RESIDUE_SITES_TABLE_NAME",
    "RESIDUE_SITE_LINK_COLUMN",
    "RESIDUE_SITE_LINK_COLUMN_SPEC",
]
