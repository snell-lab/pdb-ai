"""The cofactor_sites table, and the atom column that links atoms to it.

Each row is one cofactor of one model: a flavin, a tetrapyrrole, a
fluorescent-protein chromophore, a retinal, or a p-coumaric acid. The site
key columns name the row by the key atom of its class. The measures describe
the cofactor's reactive group, the ring or conjugated chain where reduction
or isomerization shows, beside the rest of the component.

A geometry column belongs to one class and is missing on the rows of every
other, so the table has the same columns for every cofactor.

The atom column cofactor_site_id holds the row's site_id on every atom of
the reactive group and is missing everywhere else.

The table has no cohort, technique, or temperature column, and holds no
verdict. An angle or a distance states the geometry of the model and leaves
what it means to the reader.
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
from .constants import (
    COFACTOR_CLASS_NAMES,
    COFACTOR_STATUSES,
    LINK_SEARCH_RADIUS_A,
)

COFACTOR_SITES_TABLE_NAME = "cofactor_sites"
COFACTOR_LINK_COLUMN = "cofactor_site_id"

_TORSION_NOTE = (
    "In degrees, from −180 to 180, where 0 is cis and 180 is trans. Missing on "
    "rows of every other class, and when one of the four atoms is not modelled."
)

_MEASURE_COLUMNS: tuple[TableColumnSpec, ...] = (
    TableColumnSpec(
        "cofactor_class",
        categories=COFACTOR_CLASS_NAMES,
        description=(
            "The kind of cofactor, recognized by the atom names of its "
            "reactive group. flavin is a component with the isoalloxazine "
            "ring, named by its N5. tetrapyrrole is one with the four pyrrole "
            "nitrogens and the four bridging carbons of a heme, a "
            "chlorophyll, or a pheophytin, named by its NA. fp_chromophore is "
            "a polymer residue with the imidazolinone ring and methine bridge "
            "of a fluorescent-protein chromophore, named by its CB2. retinal "
            "is RET, named by its C15. coumaric_acid is HC4, named by its C3."
        ),
    ),
    TableColumnSpec(
        "component_atom_count",
        dtype="Int64",
        description=(
            "Non-hydrogen atoms the atom table holds with coordinates for the "
            "component."
        ),
    ),
    TableColumnSpec(
        "n_group_atoms_expected",
        dtype="Int64",
        description=(
            "Atoms of the reactive group the class defines: 14 for the "
            "isoalloxazine ring, 24 for the tetrapyrrole macrocycle, 7 for "
            "the imidazolinone ring and bridge of a chromophore, 11 for the "
            "retinal chain from C5 to C15, and 9 for p-coumaric acid without "
            "its oxygens."
        ),
    ),
    TableColumnSpec(
        "n_group_atoms_modelled",
        dtype="Int64",
        description="Group atoms the atom table holds with coordinates.",
    ),
    TableColumnSpec(
        "group_occupancy_min",
        dtype="float64",
        description="Lowest occupancy among the modelled group atoms.",
    ),
    TableColumnSpec(
        "group_occupancy_mean",
        dtype="float64",
        description="Mean occupancy of the modelled group atoms.",
    ),
    TableColumnSpec(
        "group_b_factor_mean_A2",
        dtype="float64",
        description=(
            "Mean isotropic B-factor in Å² of the modelled group atoms whose "
            "occupancy is not zero. A neutron B-factor is not directly "
            "comparable with an X-ray one."
        ),
    ),
    TableColumnSpec(
        "rest_b_factor_mean_A2",
        dtype="float64",
        description=(
            "Mean isotropic B-factor in Å² of the component's other "
            "non-hydrogen atoms whose occupancy is not zero. Missing when the "
            "group is the whole component."
        ),
    ),
    TableColumnSpec(
        "b_factor_ratio_to_rest",
        dtype="float64",
        description=(
            "group_b_factor_mean_A2 divided by rest_b_factor_mean_A2. Missing "
            "when either is missing or not positive."
        ),
    ),
    TableColumnSpec(
        "group_planarity_rmsd_A",
        dtype="float64",
        description=(
            "Root-mean-square distance in Å of the modelled group atoms from "
            "the least-squares plane through them. Planarity is restrained in "
            "refinement, more tightly at lower resolution. Missing with fewer "
            "than four group atoms."
        ),
    ),
    TableColumnSpec(
        "flavin_butterfly_angle_deg",
        dtype="float64",
        description=(
            "For a flavin, the angle in degrees, from 0 to 90, between the "
            "least-squares planes of the two outer rings of the isoalloxazine "
            "ring, the pyrimidine ring (N1, C2, N3, C4, C4A, C10) and the "
            "benzene ring (C5A, C6, C7, C8, C9, C9A). A flat ring gives 0, and "
            "a reduced flavin bends about its N5–N10 axis. Missing on rows of "
            "every other class, and when a ring atom is not modelled."
        ),
    ),
    TableColumnSpec(
        "chromophore_tau_deg",
        dtype="float64",
        description=(
            "For a fluorescent-protein chromophore, the torsion "
            f"N2–CA2–CB2–CG2 about the bridge's double bond. {_TORSION_NOTE}"
        ),
    ),
    TableColumnSpec(
        "chromophore_phi_deg",
        dtype="float64",
        description=(
            "For a fluorescent-protein chromophore, the torsion "
            "CA2–CB2–CG2–CD1 about the bridge's single bond, with CD2 where "
            f"the component has no CD1. {_TORSION_NOTE}"
        ),
    ),
    TableColumnSpec(
        "retinal_c11_c12_torsion_deg",
        dtype="float64",
        description=(
            "For a retinal, the torsion C10–C11–C12–C13 about the C11=C12 "
            f"bond, which light isomerizes in visual rhodopsins. {_TORSION_NOTE}"
        ),
    ),
    TableColumnSpec(
        "retinal_c13_c14_torsion_deg",
        dtype="float64",
        description=(
            "For a retinal, the torsion C12–C13–C14–C15 about the C13=C14 "
            f"bond, which light isomerizes in microbial rhodopsins. {_TORSION_NOTE}"
        ),
    ),
    TableColumnSpec(
        "coumaric_c2_c3_torsion_deg",
        dtype="float64",
        description=(
            "For a p-coumaric acid, the torsion C1–C2–C3–C1' about the C2=C3 "
            f"bond, which light isomerizes. {_TORSION_NOTE}"
        ),
    ),
    TableColumnSpec(
        "protein_link_distance_A",
        dtype="float64",
        description=(
            "Distance in Å from the cofactor's linking atom to the nearest "
            "protein atom of the kind it bonds to, the C15 of a retinal to a "
            "lysine NZ, and the C1 of a p-coumaric acid to a cysteine SG, "
            f"among the atoms of the file within {LINK_SEARCH_RADIUS_A:g} Å. "
            "It is given as measured and asserts no bond. Missing on rows of "
            "every other class, and when no such atom lies within the radius."
        ),
    ),
    TableColumnSpec(
        "central_metal_element",
        description=(
            "For a tetrapyrrole, the element of the metal atom of the "
            "component that lies nearest the centre of its four pyrrole "
            "nitrogens, in upper case. Missing for a macrocycle without a "
            "metal, such as a pheophytin, and on rows of every other class."
        ),
    ),
    TableColumnSpec(
        "central_metal_site_id",
        description=(
            "site_id of that metal atom, which is its site_id in the "
            "metal_sites table when the metal-site engine ran. Missing when "
            "there is none."
        ),
    ),
    TableColumnSpec(
        "metal_out_of_plane_A",
        dtype="float64",
        description=(
            "For a tetrapyrrole, the distance in Å of the central metal from "
            "the least-squares plane of the four pyrrole nitrogens, without a "
            "sign. Missing without a central metal or one of the nitrogens."
        ),
    ),
    TableColumnSpec(
        "metal_nitrogen_distance_mean_A",
        dtype="float64",
        description=(
            "For a tetrapyrrole, the mean distance in Å from the central metal "
            "to the modelled pyrrole nitrogens. Missing without a central "
            "metal."
        ),
    ),
    TableColumnSpec(
        "cofactor_status",
        categories=COFACTOR_STATUSES,
        description=(
            "ok, or the first gap that applies. group_atoms_missing when the "
            "model holds fewer group atoms than the class defines, and "
            "b_factor_unavailable when group_b_factor_mean_A2 is missing."
        ),
    ),
)

COFACTOR_SITES_TABLE = TableSpec(
    name=COFACTOR_SITES_TABLE_NAME,
    description=(
        "One row per cofactor of each model whose reactive group is "
        "recognized, a flavin, a tetrapyrrole such as a heme or a "
        "chlorophyll, a fluorescent-protein chromophore, a retinal, or a "
        "p-coumaric acid, in the conformer the atom table keeps. The row "
        "makes no judgment of the site."
    ),
    key=("site_id",),
    columns=(*site_key_columns(), *_MEASURE_COLUMNS),
    links=(site_atom_link(),),
)

COFACTOR_LINK_COLUMN_SPEC = ColumnSpec(
    name=COFACTOR_LINK_COLUMN,
    group="Cofactor Fields",
    granularity=ColumnGranularity.ATOM,
    dtype="text",
    description=(
        "The site_id of the atom's row in the cofactor_sites table when the "
        "atom belongs to the reactive group of a cofactor there, and missing "
        "on every other atom."
    ),
    # The general profile leaves cofactors out. Their geometry is restrained in
    # refinement and has shown no difference between cohorts, so the engine is
    # switched on by the radiation profile, or by its own option.
    profiles=frozenset({PROFILE_RADIATION_SENSITIVITY}),
)

__all__ = [
    "COFACTOR_LINK_COLUMN",
    "COFACTOR_LINK_COLUMN_SPEC",
    "COFACTOR_SITES_TABLE",
    "COFACTOR_SITES_TABLE_NAME",
]
