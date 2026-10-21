"""The columns the parser writes, declared in display order.

Every column that structure/mmcif_parser.py names itself is declared here.
That covers the renamed atom-site items, the derived values, and the fields
copied from the RCSB entry record. A declaration gives the column's display
group, granularity, type, profiles, and star-schema roles. The catalog's
display order starts with these columns, in this order. The catalog refuses
an engine column that takes one of their names.

A few raw mmCIF items are declared as well. They are the atom serial number,
the raw model number, the B factor, and the identity items the star schema
builds its keys from. The other raw items the parser keeps are not
declared, and neither are its _pdbai_ helper columns. dataset_shapes places
each of those by its name.

A description that quotes a parser threshold or list of names builds its text
from the parser's own constant, so the text cannot drift from the code. The
citation columns come from the RCSB entry record, so each of their
descriptions ends with the same sentence on when the value is missing.
"""

from __future__ import annotations

from ..engines.core import (
    MINIMAL_PROFILE,
    ColumnDtype,
    ColumnGranularity,
    ColumnSpec,
)
from ..structure.mmcif_parser import (
    BACKBONE_ATOM_NAMES,
    HIGH_TITLE_DOSE_MGY,
    MEAN_RESIDUE_MASS_DA,
)
from ..structure.residue_identity import RESIDUE_NUMBERING_COLUMN
from .contracts import ColumnRole, DatasetColumnSpec

# The backbone atom names as a description lists them, in the parser's order.
_BACKBONE_ATOM_LIST = (
    ", ".join(BACKBONE_ATOM_NAMES[:-1]) + f", or {BACKBONE_ATOM_NAMES[-1]}"
)

_ATOM_IDENTITY_ROLES: frozenset[ColumnRole] = frozenset({"atom_identity"})
_OPTIONAL_RESIDUE_KEY_ROLES: frozenset[ColumnRole] = frozenset({"optional_residue_key"})
_OPTIONAL_RESIDUE_KEY_AND_CHAIN_ROLES: frozenset[ColumnRole] = frozenset(
    {"optional_residue_key", "chain_identity"}
)

# The sentence that ends the description of a column read from the RCSB entry
# record, which an uploaded file does not have.
_FROM_RCSB = (
    " From the RCSB entry record, so it is missing when the record reports none "
    "and always for an uploaded structure."
)


def _parser_column(
    name: str,
    *,
    granularity: ColumnGranularity,
    dtype: ColumnDtype,
    profiles: frozenset[str] = frozenset(),
    roles: frozenset[ColumnRole] = frozenset(),
    description: str,
) -> DatasetColumnSpec:
    return DatasetColumnSpec(
        ColumnSpec(
            name,
            group="PDB Fields",
            granularity=granularity,
            dtype=dtype,
            description=description,
            profiles=profiles,
        ),
        engine=None,
        roles=roles,
    )


def _citation_column(
    name: str, *, dtype: ColumnDtype, description: str
) -> DatasetColumnSpec:
    return DatasetColumnSpec(
        ColumnSpec(
            name,
            group="Citation Fields",
            granularity=ColumnGranularity.CITATION,
            dtype=dtype,
            description=description + _FROM_RCSB,
        ),
        engine=None,
    )


# One entry per column. Their order is the display order. It is also the
# order in which the star schema lists its key columns, which
# columns_with_role reads.
BASE_COLUMN_SPECS: tuple[DatasetColumnSpec, ...] = (
    _parser_column(
        "pdb_id",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        roles=frozenset(
            {"chain_author_identity", "residue_author_identity", "structure_key"}
        ),
        description=(
            "Four-character PDB entry accession in upper case for a downloaded "
            "entry. For an uploaded file it is the entry identifier taken from "
            "the file name, in upper case, with a suffix such as _2 when two "
            "files share a name."
        ),
    ),
    _parser_column(
        "chain_id",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        roles=frozenset({"chain_author_identity", "residue_author_identity"}),
        description=(
            "Chain identifier. It is the author-assigned _atom_site.auth_asym_id, "
            "or _atom_site.label_asym_id for a structure whose residue_numbering "
            "is label."
        ),
    ),
    _parser_column(
        "res_num",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        roles=frozenset({"residue_author_identity"}),
        description=(
            "Residue sequence identifier. It is the author-assigned "
            "_atom_site.auth_seq_id, or _atom_site.label_seq_id for a structure "
            "whose residue_numbering is label, where it is empty for a residue "
            "outside a polymer."
        ),
    ),
    _parser_column(
        "res_name",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        roles=frozenset({"residue_author_identity"}),
        description=(
            "Residue or component name. It is the author-assigned "
            "_atom_site.auth_comp_id, or _atom_site.label_comp_id on an atom "
            "whose author name is missing."
        ),
    ),
    _parser_column(
        "atom_name",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        roles=_ATOM_IDENTITY_ROLES,
        description="Atom name from _atom_site.label_atom_id.",
    ),
    _parser_column(
        "element",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        roles=_ATOM_IDENTITY_ROLES,
        description="Chemical element symbol from _atom_site.type_symbol.",
    ),
    _parser_column(
        "is_hetatm",
        granularity=ColumnGranularity.ATOM,
        dtype="boolean",
        profiles=MINIMAL_PROFILE,
        roles=_ATOM_IDENTITY_ROLES,
        description="True when _atom_site.group_PDB classifies the site as HETATM.",
    ),
    _parser_column(
        "occupancy",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        roles=_ATOM_IDENTITY_ROLES,
        description="Refined atom-site occupancy from _atom_site.occupancy.",
    ),
    _parser_column(
        "occupancy_all_conformers",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        description=(
            "The atom's occupancy summed over every alternate conformer the file "
            "models it in, including conformers the parser drops; equal to "
            "occupancy for an atom without alternates. Below one, the atom's "
            "occupancy was refined below full rather than shared between "
            "conformers. Missing when the atom has no alternates and no readable "
            "occupancy."
        ),
    ),
    _parser_column(
        "conformer_count",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        description=(
            "Number of alternate conformers (distinct _atom_site.label_alt_id "
            "values) the file models the atom in, including those the parser "
            "drops; 1 for an atom without alternates."
        ),
    ),
    _parser_column(
        "model_num",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        roles=_OPTIONAL_RESIDUE_KEY_AND_CHAIN_ROLES,
        description=(
            "Coordinate-model number containing the atom site, from "
            "_atom_site.pdbx_PDB_model_num."
        ),
    ),
    _parser_column(
        "_atom_site.id",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        description="Atom-site row identifier assigned in the source mmCIF file.",
    ),
    _parser_column(
        "_atom_site.label_asym_id",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        roles=_OPTIONAL_RESIDUE_KEY_AND_CHAIN_ROLES,
        description="PDB-assigned asymmetric-unit chain identifier used by mmCIF relationships.",
    ),
    _parser_column(
        "_atom_site.label_entity_id",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        roles=_OPTIONAL_RESIDUE_KEY_AND_CHAIN_ROLES,
        description="Identifier of the mmCIF entity to which the atom site belongs.",
    ),
    _parser_column(
        "_atom_site.label_seq_id",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        roles=_OPTIONAL_RESIDUE_KEY_ROLES,
        description="PDB-assigned polymer sequence position within the atom site's entity.",
    ),
    _parser_column(
        "_atom_site.pdbx_PDB_ins_code",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        roles=_OPTIONAL_RESIDUE_KEY_ROLES,
        description="Author insertion code that disambiguates residues sharing a sequence number.",
    ),
    _parser_column(
        "_atom_site.pdbx_PDB_model_num",
        granularity=ColumnGranularity.ATOM,
        dtype="text",
        description=(
            "Raw mmCIF coordinate-model number, which model_num repeats. The "
            "star schema builds its keys from model_num, so this column plays "
            "no part in them."
        ),
    ),
    _parser_column(
        "is_backbone",
        granularity=ColumnGranularity.ATOM,
        dtype="boolean",
        profiles=MINIMAL_PROFILE,
        roles=_ATOM_IDENTITY_ROLES,
        description=f"True for non-HETATM atoms named {_BACKBONE_ATOM_LIST}.",
    ),
    _parser_column(
        "is_sidechain",
        granularity=ColumnGranularity.ATOM,
        dtype="boolean",
        roles=_ATOM_IDENTITY_ROLES,
        description="True for non-HETATM atoms not classified as protein backbone atoms.",
    ),
    _parser_column(
        "x",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        description="Atom Cartesian x coordinate in Å from _atom_site.Cartn_x.",
    ),
    _parser_column(
        "y",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        description="Atom Cartesian y coordinate in Å from _atom_site.Cartn_y.",
    ),
    _parser_column(
        "z",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        description="Atom Cartesian z coordinate in Å from _atom_site.Cartn_z.",
    ),
    _parser_column(
        "_atom_site.B_iso_or_equiv",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        description="Isotropic or equivalent isotropic atomic displacement parameter in Å².",
    ),
    _parser_column(
        "residue_mean_bfactor",
        granularity=ColumnGranularity.RESIDUE,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        description=(
            "Mean isotropic B factor across atom rows sharing one canonical "
            "model-aware residue identity. Atoms without a reported B factor are "
            "excluded from the mean. The value is missing when no atom in the "
            "residue reports one, and for a row without a canonical residue "
            "identity."
        ),
    ),
    _parser_column(
        "u11",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        description=(
            "U[1][1] component of the atom's anisotropic displacement tensor "
            "in Å². Missing when the atom has no anisotropic displacement record."
        ),
    ),
    _parser_column(
        "u22",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        description=(
            "U[2][2] component of the atom's anisotropic displacement tensor "
            "in Å². Missing when the atom has no anisotropic displacement record."
        ),
    ),
    _parser_column(
        "u33",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        description=(
            "U[3][3] component of the atom's anisotropic displacement tensor "
            "in Å². Missing when the atom has no anisotropic displacement record."
        ),
    ),
    _parser_column(
        "u12",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        description=(
            "U[1][2] component of the atom's anisotropic displacement tensor "
            "in Å². Missing when the atom has no anisotropic displacement record."
        ),
    ),
    _parser_column(
        "u13",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        description=(
            "U[1][3] component of the atom's anisotropic displacement tensor "
            "in Å². Missing when the atom has no anisotropic displacement record."
        ),
    ),
    _parser_column(
        "u23",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        description=(
            "U[2][3] component of the atom's anisotropic displacement tensor "
            "in Å². Missing when the atom has no anisotropic displacement record."
        ),
    ),
    _parser_column(
        "u_equivalent",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        description=(
            "Equivalent isotropic displacement U, calculated as "
            "(U11 + U22 + U33) / 3 in Å². Missing when the atom has no "
            "anisotropic displacement record."
        ),
    ),
    _parser_column(
        "b_anisotropy",
        granularity=ColumnGranularity.ATOM,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        description=(
            "Ratio of the largest to the smallest eigenvalue of the atom's "
            "anisotropic U tensor, where 1 indicates isotropic displacement. "
            "Missing when any of the six components is missing or the tensor "
            "is not positive definite."
        ),
    ),
    _parser_column(
        RESIDUE_NUMBERING_COLUMN,
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        description=(
            "Which identifiers chain_id and res_num hold for the structure. It is "
            "author when the file gives an author chain and an author residue "
            "number on at least one atom, and both columns then hold the author "
            "items on every atom. It is label when the file gives no author chain "
            "or no author residue number, and both columns then hold the label "
            "items. The two numberings are never mixed within a structure."
        ),
    ),
    _parser_column(
        "resolution",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        description=(
            "Resolution of the refined model in Å: _refine.ls_d_res_high, else "
            "_refine_hist.d_res_high, else _reflns.d_resolution_high, else the "
            "smallest _reflns_shell.d_res_high, else the RCSB entry's value. "
            "Missing when none reports one. A joint X-ray/neutron entry has a "
            "refinement record for each method, and this is the first one's, "
            "of whichever method the file lists first. resolution_xray and "
            "resolution_neutron hold each method's own."
        ),
    ),
    _parser_column(
        "resolution_xray",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Resolution in Å of the refinement against the X-ray data, the "
            "_refine.ls_d_res_high of the record that _refine.pdbx_refine_id "
            "names X-RAY DIFFRACTION. In a joint X-ray/neutron entry it is the "
            "X-ray data's resolution whichever record comes first. Missing "
            "when the file has no such record, as for a neutron-only entry."
        ),
    ),
    _parser_column(
        "resolution_neutron",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Resolution in Å of the refinement against the neutron data, the "
            "_refine.ls_d_res_high of the record that _refine.pdbx_refine_id "
            "names NEUTRON DIFFRACTION. In a joint X-ray/neutron entry it is "
            "the neutron data's resolution whichever record comes first. "
            "Missing when the file has no such record, as for an X-ray-only "
            "entry."
        ),
    ),
    *(
        _parser_column(
            f"{prefix}_{suffix}",
            granularity=ColumnGranularity.STRUCTURE,
            dtype="numeric",
            description=(
                f"{label} of the refinement against the {method} data, the "
                f"{item} of the record that _refine.pdbx_refine_id names "
                f"{named}. In a joint X-ray/neutron entry it is the {method} "
                "data's value whichever record comes first, where r_work and "
                "r_free hold the first record's. Missing when the file has no "
                "such record or the record states none."
            ),
        )
        for suffix, method, named in (
            ("xray", "X-ray", "X-RAY DIFFRACTION"),
            ("neutron", "neutron", "NEUTRON DIFFRACTION"),
        )
        for prefix, label, item in (
            ("r_work", "R-work", "_refine.ls_R_factor_R_work"),
            ("r_free", "R-free", "_refine.ls_R_factor_R_free"),
        )
    ),
    _parser_column(
        "r_work",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        description=(
            "Refinement R factor calculated for reflections in the working set."
            + _FROM_RCSB
        ),
    ),
    _parser_column(
        "r_free",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        profiles=MINIMAL_PROFILE,
        description=(
            "Cross-validation R factor calculated for the held-out reflection set."
            + _FROM_RCSB
        ),
    ),
    _parser_column(
        "refinement_program",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=(
            "Name of the program the model was refined with, the _software.name "
            "of the file's first _software record classified as refinement, as "
            "the file writes it. An entry that names several refinement "
            "programs has its first here and all of them in "
            "refinement_programs. Missing when the file names none."
        ),
    ),
    _parser_column(
        "refinement_program_version",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=(
            "Version of refinement_program, the _software.version of the same "
            "record, as the depositors wrote it and in no fixed format. Missing "
            "when that record states none."
        ),
    ),
    _parser_column(
        "refinement_programs",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=(
            "Every refinement program the entry names (_software.name of the "
            "records classified as refinement), each listed once, "
            "semicolon-separated in file order. It equals refinement_program "
            "for an entry that names one. Names are as the file writes them, so "
            "SHELXL and SHELXL-97, or BUSTER and BUSTER-TNT, are listed apart. "
            "The file does not say which of several programs wrote the "
            "deposited B factors. Missing when the file names none."
        ),
    ),
    _parser_column(
        "tls_group_count",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Number of TLS groups the refinement used, the number of "
            "_pdbx_refine_tls records in the file. 0 for a file that reports a "
            "refinement, a _refine category, and no TLS group. Missing for a "
            "file that reports no refinement, which says nothing of TLS."
        ),
    ),
    _parser_column(
        "tls_residual_adp_flag",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=(
            "The archive's note on the B factors of an entry refined with TLS, "
            "the first _refine.pdbx_TLS_residual_ADP_flag the file states, as "
            "written. LIKELY RESIDUAL marks an entry whose "
            "_atom_site.B_iso_or_equiv probably leaves out the TLS "
            "contribution, so its B factors do not compare with those of other "
            "entries. Missing when the file states none, as for most entries "
            "with TLS groups, so missing does not say the B factors are whole."
        ),
    ),
    _parser_column(
        "anisotropic_atom_fraction",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Fraction (0-1) of the file's atom sites that have an anisotropic "
            "displacement record (_atom_site_anisotrop), over every alternate "
            "conformer and model, with atoms of element H, D, or T left out. 0 for "
            "a file without such records, which is how an isotropic refinement "
            "is deposited. A refinement with TLS groups can write a record for "
            "every atom of a group, so read it beside tls_group_count. Missing "
            "only for a file without a non-hydrogen atom."
        ),
    ),
    _parser_column(
        "space_group",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=("Hermann–Mauguin symbol of the crystal space group." + _FROM_RCSB),
    ),
    _parser_column(
        "reflns_completeness",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Percentage of theoretically possible reflections observed." + _FROM_RCSB
        ),
    ),
    _parser_column(
        "cc_half",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Correlation coefficient between randomly divided half datasets (CC1/2)."
            + _FROM_RCSB
        ),
    ),
    _parser_column(
        "i_over_sigma",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Reported mean of each reflection's intensity divided by its "
            "uncertainty, from _reflns.pdbx_netI_over_sigmaI." + _FROM_RCSB
        ),
    ),
    _parser_column(
        "wilson_b",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Wilson-plot estimate of the overall isotropic B factor in Å²." + _FROM_RCSB
        ),
    ),
    _parser_column(
        "multiplicity",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Mean number of observations contributing to each unique reflection."
            + _FROM_RCSB
        ),
    ),
    _parser_column(
        "unit_cell_volume",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Unit-cell volume in Å³ calculated from the file's _cell lengths and "
            "angles. Missing when the file reports no complete cell, when the cell "
            "cannot close, and for the wwPDB placeholder cell of 1 Å edges and "
            "right angles written for entries without a crystal."
        ),
    ),
    _parser_column(
        "solvent_content",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Crystal solvent-volume fraction (0-1). Taken from the deposited "
            "_exptl_crystal.density_percent_sol, else converted from the deposited "
            "Matthews coefficient, else estimated from unit-cell volume, space-group "
            f"symmetry count, and {MEAN_RESIDUE_MASS_DA:g} Da per polymer residue of "
            "the first model, HETATM-flagged modified residues included. Each source "
            "is used only when it gives a fraction inside (0, 1), otherwise the next "
            "is tried. Missing when none does."
        ),
    ),
    _parser_column(
        "radiation_wavelength",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=("Primary reported diffraction wavelength in Å." + _FROM_RCSB),
    ),
    _parser_column(
        "title_dose_MGy",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Absorbed X-ray dose as the entry title states it, in megagray (MGy), "
            'read from the first number followed by "MGy", "kGy", or "Gy" in the '
            "title, where dose-series entries state it, a value in kGy divided by "
            "1,000 and one in Gy by 1,000,000. A number may be written as "
            '"7.88x10^3". A dose rate is not read. The archive records no dose, so '
            "this is the depositor's statement and not a measured value, and a "
            "title copied from another entry or at odds with its paper gives a "
            "wrong one. Missing when the title states none, as for most entries, "
            "and always for an uploaded structure."
        ),
    ),
    _parser_column(
        "title",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=(
            "Title of the entry, _struct.title. It is where depositors state a "
            "dose, a time delay, or a resting or intermediate state, none of "
            "which the archive records in a field of its own." + _FROM_RCSB
        ),
    ),
    _parser_column(
        "keywords",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=(
            "Free-text keywords of the entry, _struct_keywords.text, as the "
            "depositors wrote them." + _FROM_RCSB
        ),
    ),
    _parser_column(
        "mentions_time_resolved",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="boolean",
        description=(
            "True when the title or the keywords name a time-resolved or "
            "pump-probe experiment, in any spelling of those two words. This "
            "is a text match, not a classification. A resting-state structure "
            "from a time-resolved series is true as well, and the title says "
            "which state the model shows. Missing when the record has neither "
            "a title nor keywords, and always for an uploaded structure."
        ),
    ),
    _parser_column(
        "is_serial_crystallography",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="boolean",
        description=(
            "True when the first diffraction record says that several crystals "
            "contributed to each data set, _diffrn.pdbx_serial_crystal_experiment. "
            "False when it says one crystal did. Missing when the record does not "
            "say, which many serial entries deposited before the field existed "
            "leave out, and always for an uploaded structure."
        ),
    ),
    _parser_column(
        "sample_delivery_method",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=(
            "How a serial-crystallography sample reached the beam, such as "
            "injection or fixed target, from the first "
            "_pdbx_serial_crystallography_sample_delivery record. Missing when "
            "none is reported, as for a single-crystal experiment." + _FROM_RCSB
        ),
    ),
    _parser_column(
        "xfel_pulse_duration_fs",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Average X-ray pulse duration in femtoseconds, from the first "
            "_pdbx_serial_crystallography_measurement record. Missing when none "
            "is reported, which is the case for most free-electron laser entries."
            + _FROM_RCSB
        ),
    ),
    _parser_column(
        "is_high_title_dose",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="boolean",
        description=(
            "True when the dose stated in the entry title exceeds "
            f"{HIGH_TITLE_DOSE_MGY:g} MGy. Missing when the title states none, "
            "which is the case for most entries, so absent is not the same as "
            "below threshold."
        ),
    ),
    _parser_column(
        "temperature_K",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Reported crystal temperature during data collection in kelvin."
            + _FROM_RCSB
        ),
    ),
    _parser_column(
        "experimental_methods",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        description=(
            "Every experimental method the entry reports (_exptl.method), "
            "each listed once, semicolon-separated in deposition order; a joint "
            "X-ray and neutron refinement lists both. From the RCSB entry record, "
            "or from the file's own _exptl category for uploaded structures. "
            "Missing when neither reports one."
        ),
    ),
    _parser_column(
        "is_multi_method",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="boolean",
        profiles=MINIMAL_PROFILE,
        description=(
            "True when the entry reports more than one experimental method, so its "
            "model was refined against several kinds of data; false for a single "
            "method; missing when no method was reported."
        ),
    ),
    _parser_column(
        "first_radiation_source",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        description=(
            "The first radiation source the entry reports, as the depositors "
            "wrote it in the first _diffrn_source.source row, without the "
            "normalization radiation_sources applies." + _FROM_RCSB
        ),
    ),
    _parser_column(
        "radiation_sources",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        profiles=MINIMAL_PROFILE,
        description=(
            "Every radiation source the entry reports (_diffrn_source.source), each "
            "listed once, semicolon-separated in deposition order and normalized to "
            "the archive's canonical spellings (SYNCHROTRON, FREE ELECTRON LASER, "
            "ROTATING ANODE, SEALED TUBE, LIQUID ANODE; a joint X-ray/neutron entry "
            "also lists its NUCLEAR REACTOR or SPALLATION SOURCE). This is the "
            "column the xray_source intake filter matches against. Missing when no "
            "source is reported."
        ),
    ),
    _parser_column(
        "synchrotron_site",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=(
            "Reported synchrotron facility at which data were collected." + _FROM_RCSB
        ),
    ),
    _parser_column(
        "detector_type",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=("Reported diffraction-detector type or model." + _FROM_RCSB),
    ),
    _parser_column(
        "collection_year",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Calendar year of diffraction data collection, from the first "
            "_diffrn_detector.pdbx_collection_date reported to RCSB. Missing when "
            "the depositors did not record a collection date, and always for an "
            "uploaded structure."
        ),
    ),
    _parser_column(
        "deposition_year",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=("Calendar year of the RCSB deposition date." + _FROM_RCSB),
    ),
    _parser_column(
        "beamline",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=(
            "Synchrotron beamline from _diffrn_source.pdbx_synchrotron_beamline, "
            "falling back to the source's make and model in _diffrn_source.type "
            "when no beamline is reported." + _FROM_RCSB
        ),
    ),
    _parser_column(
        "crystal_pH",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=("Reported pH of the crystallization condition." + _FROM_RCSB),
    ),
    _parser_column(
        "crystallization_method",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="text",
        description=("Reported method used to grow the crystal." + _FROM_RCSB),
    ),
    _parser_column(
        "num_crystals",
        granularity=ColumnGranularity.STRUCTURE,
        dtype="numeric",
        description=(
            "Reported number of crystals used for the experiment." + _FROM_RCSB
        ),
    ),
    _citation_column(
        "citation_title",
        dtype="text",
        description="Title of the primary publication associated with the PDB entry.",
    ),
    _citation_column(
        "citation_journal_abbrev",
        dtype="text",
        description="Abbreviated journal name for the primary publication.",
    ),
    _citation_column(
        "citation_year",
        dtype="numeric",
        description="Publication year of the primary citation.",
    ),
    _citation_column(
        "citation_doi",
        dtype="text",
        description="Digital Object Identifier of the primary citation.",
    ),
    _citation_column(
        "citation_pubmed_id",
        dtype="numeric",
        description="PubMed identifier of the primary citation.",
    ),
    _citation_column(
        "citation_authors",
        dtype="text",
        description="Semicolon-separated authors of the primary citation.",
    ),
    _citation_column(
        "deposit_authors",
        dtype="text",
        description="Semicolon-separated authors credited with the PDB deposition.",
    ),
)

__all__ = ["BASE_COLUMN_SPECS"]
