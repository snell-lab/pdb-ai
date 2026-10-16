"""The disulfides table, and the atom column that links atoms to it.

Each row names two cysteine residues of one model that may be joined by a
disulfide. A row is a candidate pair, since a deposited record or a short
distance does not prove a bond, and metal_bridged marks the rows where both
sulfurs are bound to one metal atom. The first residue fills the unprefixed
site key columns and the second the partner_ ones, each named by its
representative SG, the conformer the atom table keeps. Both sets link to the
atom level through the standard site link, so they hold what the atom table
holds for the same atom. res_name is therefore the label residue name where
the file gives no author one, and chain_id and res_num are the label chain
and number for a structure the parser numbered by its label identifiers. The
measure columns then summarize every alternate conformer of the two
residues, which the atom table no longer holds.

The atom column disulfide_site_id holds a representative SG's own site_id on
that SG's atom row and is missing everywhere else. An SG paired with two
partners is in two rows, and its own identifier still names it without
ambiguity. Joining the atoms to the table therefore matches it against either
site_id or partner_site_id. No curated profile keeps _atom_site.label_alt_id,
so under a profile only this one-column join is available.
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
from ..struct_conn import DETECTION_METHODS
from .constants import (
    DISULFIDE_SEARCH_CUTOFF_A,
    METAL_CONTACT_DISTANCE_A,
    OCCUPANCY_SUM_DECIMALS,
    PAIR_STATUSES,
)

# The two distances as the descriptions state them.
_SEARCH_LIMIT = f"{DISULFIDE_SEARCH_CUTOFF_A:.1f} Å"
_METAL_CONTACT = f"{METAL_CONTACT_DISTANCE_A:.1f} Å"

# The prefix of the second residue's site key columns.
PARTNER_PREFIX = "partner_"

DISULFIDE_LINK_COLUMN = "disulfide_site_id"

DISULFIDES_TABLE_NAME = "disulfides"

_MEASURE_COLUMNS: tuple[TableColumnSpec, ...] = (
    TableColumnSpec(
        "detection_method",
        categories=DETECTION_METHODS,
        description=(
            "How the pair was found in this model. It is struct_conn when a "
            "deposited _struct_conn disulf record within one molecule joins it, "
            "distance when two usable SG conformers that can coexist lie at "
            f"most {_SEARCH_LIMIT} apart, the limit included and with no lower "
            "limit, and struct_conn_and_distance when both found it. A "
            "struct_conn row is not evidence of a cleaved bond. Its SG atoms "
            "lie beyond the limit, as in some NMR models, or have no usable "
            "conformers to measure, as when a residue is deposited at zero "
            "occupancy."
        ),
    ),
    TableColumnSpec(
        "metal_bridged",
        dtype="boolean",
        description=(
            "True when a usable SG conformer of each residue lies within "
            f"{_METAL_CONTACT} of the same metal atom of the same model, the "
            "limit included, where the two SG conformers can coexist with each "
            "other and each can coexist with the metal atom. False otherwise, "
            "and never missing. A metal atom is any atom whose "
            "_atom_site.type_symbol names a metallic element, with coordinates "
            "and an occupancy that is not zero or below, so the flag is False "
            "on every row of a file that names no elements. Only the atoms of "
            "the file are read, so a metal of a symmetry mate is not seen. The "
            "flag never adds, removes, or changes a row. A flagged row is at a "
            "metal site and should be excluded or kept apart before any "
            "disulfide measure is read."
        ),
    ),
    TableColumnSpec(
        "ss_distance_min_A",
        dtype="float64",
        description=(
            "Shortest SG–SG distance in Å over every pair of usable SG "
            "conformers, one from each residue, that can coexist. A usable "
            "conformer has coordinates and an occupancy that is not zero or "
            "below, and two conformers can coexist when their alternate "
            "location labels are equal or either has none. The value is not "
            "rounded. It is missing when no such pair exists, and pair_status "
            "says why."
        ),
    ),
    TableColumnSpec(
        "ss_distance_max_A",
        dtype="float64",
        description=(
            "Longest SG–SG distance in Å over the same pairs of conformers as "
            "ss_distance_min_A. It equals the minimum when one pair was "
            "measured."
        ),
    ),
    TableColumnSpec(
        "sg_conformer_occupancy_min",
        dtype="float64",
        description=(
            "Lowest occupancy of any single SG conformer of either residue, "
            "over every SG row the structure file holds for them, rows at zero "
            "occupancy included. It is below 1 for any split SG, so on its own "
            "it does not measure damage. It is missing when any of those "
            "occupancies is missing."
        ),
    ),
    TableColumnSpec(
        "sg_total_occupancy_min",
        dtype="float64",
        description=(
            "The lower of the two residues' SG occupancies, each summed over "
            "the residue's conformers and rounded to "
            f"{OCCUPANCY_SUM_DECIMALS} decimals. It is not a measure of SG loss "
            "on its own, since a residue or chain modelled at partial or zero "
            "occupancy as a whole lowers it, and so, rarely, do occupancies "
            "deposited to two decimals that sum to 0.98 or 0.99. "
            "sg_occupancy_deficit_max measures the SG against its own CB "
            "instead. The value is not clamped, and it is missing when any SG "
            "occupancy is missing."
        ),
    ),
    TableColumnSpec(
        "sg_occupancy_deficit_max",
        dtype="float64",
        description=(
            "The larger of the two residues' SG occupancy deficits, each the "
            "sum of the residue's CB occupancies less the sum of its SG "
            f"occupancies, rounded to {OCCUPANCY_SUM_DECIMALS} decimals. It is "
            "0 when the SG is modelled as fully as the CB, whatever the "
            "rounding of deposited occupancies or the occupancy of the whole "
            "residue, and above 0 when part of the SG is not modelled, as when "
            "SG 0.75 sits on CB 1.00 (0.25). It is the Figure 3 read-out of SG "
            "occupancy below 1. The value is not clamped, and it is missing "
            "when a residue has no CB row or any SG or CB occupancy of the two "
            "is missing."
        ),
    ),
    TableColumnSpec(
        "sg_has_alternate_conformers",
        dtype="boolean",
        description=(
            "True when the SG rows of either residue carry two or more distinct "
            "non-empty alternate location labels, and False otherwise. A lone "
            "labelled SG, such as A at 0.6 with no B, is False. Never missing."
        ),
    ),
    TableColumnSpec(
        "sg_cb_b_factor_ratio_min",
        dtype="float64",
        description=(
            "The smaller of the two residues' ratios of the B-factor of the "
            "representative SG to that of the representative CB, so it does "
            "not depend on which residue is first. A residue's ratio is missing "
            "when it has no CB in a conformer that can coexist with its SG, or "
            "when either B-factor is missing, not finite, or zero or below. It "
            "is also missing when either atom's occupancy is zero or below, "
            "since such an atom was placed without being seen in the data and "
            "its B-factor is not a measurement. Non-diffraction models may carry values that are not B-factors, "
            "which are reported as measured, so filter by technique at "
            "analysis time. This value is missing when either ratio is. The "
            "representative SG and CB belong to the conformer the atom table "
            "keeps, which may be a broken conformer of higher occupancy, "
            "except on a row whose pair_status is sg_not_in_atom_table, where "
            "they come from the conformer of the fallback SG. The atom table "
            "keeps the conformer of highest mean occupancy over all the "
            "residue's atoms, so in a neutron model the occupancies of H and "
            "D atoms can decide it. An intact residue's ratio is typically "
            "above 1, grouped or constant-B refinement gives exactly 1 for "
            "reasons unrelated to damage, and neutron and X-ray ratios are not "
            "directly comparable."
        ),
    ),
    TableColumnSpec(
        "sg_cb_b_factor_ratio_max",
        dtype="float64",
        description=(
            "The larger of the two residues' SG/CB B-factor ratios, missing "
            "under the same rules as sg_cb_b_factor_ratio_min."
        ),
    ),
    TableColumnSpec(
        "pair_status",
        categories=PAIR_STATUSES,
        description=(
            "ok, or why a value of the row is missing. It is no_usable_sg when "
            "a residue has no usable SG conformer, no_compatible_conformers "
            "when no SG conformers of the two can coexist, occupancy_missing "
            "when an SG or CB occupancy is missing, cb_missing when a residue "
            "has no representative CB, b_factor_unavailable when a "
            "representative B-factor is missing, not finite, or zero or below, "
            "or belongs to an atom at zero occupancy, "
            "and sg_not_in_atom_table when a residue's representative SG is a "
            "conformer the atom table dropped, so no atom links to it. The "
            "first that applies, in that order, is given. A usable conformer "
            "has coordinates and an occupancy that is not zero or below, so "
            "no_usable_sg does not mean that the sulfur is absent from the "
            "model. A whole residue deposited at zero occupancy, with its SG "
            "coordinates present, has this status."
        ),
    ),
)

DISULFIDES_TABLE = TableSpec(
    name=DISULFIDES_TABLE_NAME,
    # The description states both distances, so the manifest records them.
    description=(
        "One row per candidate disulfide, a pair of cysteine residues in one "
        "model that a deposited _struct_conn disulf record joins within one "
        f"molecule, or whose SG atoms lie at most {_SEARCH_LIMIT} apart in "
        "usable conformers that can coexist. A row does not assert a bond. The "
        "distance search has no lower limit, so an implausibly short distance "
        "is reported as measured. metal_bridged "
        f"marks the rows whose two sulfurs lie within {_METAL_CONTACT} of one "
        "metal atom. Alternate conformers are summarized in the row. site_id "
        "and partner_site_id name each residue's representative SG, the "
        "conformer the atom table keeps, and the key columns hold what the "
        "atom table holds for that atom. A "
        "pair whose SG atoms lie beyond the limit in every conformer is in no "
        "row unless a record joins it, so the rows leave out the most "
        "separated conformers of damaged bonds. A cysteine with no SG atom in "
        "the file is in no row either. The run report's "
        "disulfide_cysteine_without_sg_count counts such cysteines, which the "
        "table cannot see. It is not a count of damaged bonds. When overlapping "
        "chain copies model the disorder of a whole molecule, a cysteine of "
        "one copy can pair by distance with its own copy or with another "
        "cysteine's copy. A distance-only row between chains whose residues "
        "each have a copy at the same residue number in another chain is "
        "likely such an artefact."
    ),
    key=("site_id", f"{PARTNER_PREFIX}site_id"),
    columns=(
        *site_key_columns(),
        *site_key_columns(PARTNER_PREFIX),
        *_MEASURE_COLUMNS,
    ),
    links=(site_atom_link(), site_atom_link(PARTNER_PREFIX, role="partner")),
)

DISULFIDE_LINK_COLUMN_SPEC = ColumnSpec(
    name=DISULFIDE_LINK_COLUMN,
    group="Disulfide Fields",
    granularity=ColumnGranularity.ATOM,
    dtype="text",
    description=(
        "The atom's own site_id when it is the representative SG of a cysteine "
        "in the disulfides table, and missing on every other atom. An atom "
        "joins its rows where this equals site_id or partner_site_id there."
    ),
    profiles=frozenset({PROFILE_RESEARCH_READY, PROFILE_RADIATION_SENSITIVITY}),
)

__all__ = [
    "DISULFIDES_TABLE",
    "DISULFIDES_TABLE_NAME",
    "DISULFIDE_LINK_COLUMN",
    "DISULFIDE_LINK_COLUMN_SPEC",
    "PARTNER_PREFIX",
]
