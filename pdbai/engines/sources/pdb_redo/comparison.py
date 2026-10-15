"""PDB-Redo's atom values set beside the RCSB atoms they pair with.

compare_with_rcsb_atoms pairs the atoms of the two models through
structure.alignment, which matches them by author chain, residue, and atom
name. Each RCSB atom then gets PDB-Redo's coordinates. It also gets the difference, PDB-Redo minus
RCSB, in each coordinate and in B factor, occupancy, and equivalent U. Each
atom is flagged as revised or not. Every row repeats the alignment's match
rate and ambiguous-atom counts.

An atom is revised when any of its differences exceeds 0.001. An atom with
no PDB-Redo partner has empty columns, and its revised flag is missing
rather than False. So an atom that could not be compared is never reported
as unchanged. A match rate below 95% is logged as a warning, since the
PDB-Redo columns of that structure are then incomplete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ....structure import alignment

logger = logging.getLogger(__name__)

# The match rate below which the comparison warns.
_LOW_MATCH_RATE_THRESHOLD = 0.95

# One tolerance applies to every compared value, each in its own unit. That
# is Å for a coordinate, Å² for a B factor or U, and no unit for occupancy.
_REVISION_TOLERANCE = 1e-3

# Each RCSB atom column compared, and the column its difference goes to.
_DELTA_OUTPUT_NAMES = {
    "x": "pdb_redo_x_delta",
    "y": "pdb_redo_y_delta",
    "z": "pdb_redo_z_delta",
    "_atom_site.B_iso_or_equiv": "pdb_redo_bfactor_delta",
    "occupancy": "pdb_redo_occupancy_delta",
    "u_equivalent": "pdb_redo_u_equivalent_delta",
}

# The compared columns whose PDB-Redo value is also written as it is.
_RAW_VALUE_OUTPUT_NAMES = {
    "x": "pdb_redo_x",
    "y": "pdb_redo_y",
    "z": "pdb_redo_z",
}

# Structure-level results of the alignment, repeated on every atom row.
_ALIGNMENT_OUTPUT_NAMES = (
    "pdb_redo_alignment_match_rate",
    "pdb_redo_alignment_rcsb_ambiguous_atom_count",
    "pdb_redo_alignment_pdb_redo_ambiguous_atom_count",
)

# Whether any of an atom's differences exceeds the revision tolerance.
_REVISED_OUTPUT_NAME = "pdb_redo_revised"

# Every column compare_with_rcsb_atoms returns, in the order the engine declares them.
OUTPUT_COLUMNS = (
    *_RAW_VALUE_OUTPUT_NAMES.values(),
    *_DELTA_OUTPUT_NAMES.values(),
    _REVISED_OUTPUT_NAME,
    *_ALIGNMENT_OUTPUT_NAMES,
)


@dataclass(frozen=True)
class PdbRedoComparisonResult:
    """The comparison columns of one entry, and the alignment behind them.

    The columns share the index of the RCSB atom rows. The alignment is None
    when either model has no atoms, since no alignment was then tried.
    """

    columns: pd.DataFrame
    alignment: alignment.AlignmentDiagnostics | None


def compare_with_rcsb_atoms(
    rcsb_dataframe: pd.DataFrame,
    pdb_redo_dataframe: pd.DataFrame,
    *,
    pdb_id: str,
) -> PdbRedoComparisonResult:
    """Return PDB-Redo's values and differences for every RCSB atom.

    Every output column is present, and every one is empty when either
    frame has no rows. The alignment columns repeat one value on every row.
    The value and difference columns are empty for an atom without a
    partner, and so is its revised flag. A compared column missing from
    either frame leaves its difference empty, and PDB-Redo coordinates are
    written whenever PDB-Redo has them. A value that is not a number counts
    as missing. ``pdb_id`` only labels log messages.
    """
    n = len(rcsb_dataframe)
    columns = pd.DataFrame(
        {name: np.full(n, np.nan) for name in OUTPUT_COLUMNS},
        index=rcsb_dataframe.index,
    )
    columns[_REVISED_OUTPUT_NAME] = pd.Series(
        pd.NA, index=rcsb_dataframe.index, dtype="boolean"
    )
    if n == 0 or pdb_redo_dataframe.empty:
        return PdbRedoComparisonResult(columns=columns, alignment=None)

    alignment_result = alignment.align(
        rcsb_dataframe,
        pdb_redo_dataframe,
        pdb_id=pdb_id,
        caller="pdb_redo_engine",
        carry_columns=tuple(_DELTA_OUTPUT_NAMES),
    )
    merged = alignment_result.columns
    diagnostics = alignment_result.diagnostics
    # A rejected alignment has already logged its own warning, and an entry
    # without protein heavy atoms has no rate to judge.
    if (
        diagnostics.rejected_reason is None
        and diagnostics.eligible_atom_count > 0
        and diagnostics.match_rate < _LOW_MATCH_RATE_THRESHOLD
    ):
        logger.warning(
            "RCSB/PDB-Redo atom alignment for %s matched only %.1f%% of protein "
            "heavy atoms (%d/%d). The two sources model different protein atoms "
            "(alt-locs, truncated side chains, renamed residues). Delta and RABDAM "
            "columns derived from PDB-Redo are incomplete for this structure.",
            pdb_id,
            diagnostics.match_rate * 100.0,
            diagnostics.matched_eligible_atom_count,
            diagnostics.eligible_atom_count,
        )
    columns["pdb_redo_alignment_match_rate"] = diagnostics.match_rate
    columns["pdb_redo_alignment_rcsb_ambiguous_atom_count"] = (
        diagnostics.rcsb_ambiguous_atom_count
    )
    columns["pdb_redo_alignment_pdb_redo_ambiguous_atom_count"] = (
        diagnostics.pdb_redo_ambiguous_atom_count
    )

    any_revised = np.zeros(n, dtype=bool)
    compared = np.zeros(n, dtype=bool)
    for source_column, output_column in _DELTA_OUTPUT_NAMES.items():
        # The alignment returns each carried column with the suffix _pdb_redo,
        # and leaves out a column PDB-Redo's frame lacks.
        pdb_redo_column = f"{source_column}_pdb_redo"
        if pdb_redo_column not in merged.columns:
            continue
        pdb_redo_values = np.asarray(
            pd.to_numeric(merged[pdb_redo_column], errors="coerce"), dtype=float
        )
        if source_column in _RAW_VALUE_OUTPUT_NAMES:
            columns[_RAW_VALUE_OUTPUT_NAMES[source_column]] = pdb_redo_values
        if source_column not in rcsb_dataframe.columns:
            continue
        rcsb_values = np.asarray(
            pd.to_numeric(rcsb_dataframe[source_column], errors="coerce"), dtype=float
        )
        delta = pdb_redo_values - rcsb_values
        columns[output_column] = delta
        finite_delta = np.isfinite(delta)
        compared |= finite_delta
        any_revised |= finite_delta & (np.abs(delta) > _REVISION_TOLERANCE)
    revised = pd.Series(pd.NA, index=rcsb_dataframe.index, dtype="boolean")
    revised.loc[compared] = any_revised[compared]
    columns[_REVISED_OUTPUT_NAME] = revised

    return PdbRedoComparisonResult(columns=columns, alignment=diagnostics)


__all__ = ["OUTPUT_COLUMNS", "PdbRedoComparisonResult", "compare_with_rcsb_atoms"]
