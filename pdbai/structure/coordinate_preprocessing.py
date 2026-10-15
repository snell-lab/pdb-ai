"""Choosing one alternate conformer for each residue of a coordinate table.

A crystallographic model can hold part of a residue in several alternate
conformations. Each atom of a conformer is labelled by _atom_site.label_alt_id
and weighted by its occupancy. Most calculations need one set of coordinates,
so the parser keeps one conformer of each residue. This module decides which.

The choice is made for the whole residue, not atom by atom. Choosing atom by
atom can assemble a residue whose atoms never coexisted, such as the backbone
of conformer A with the side chain of conformer B. Residues are told apart by
the model-aware key of residue_identity.py, so each model makes its own
choice. An atom without a residue identity counts as a residue of its own,
since grouping such atoms would make unrelated atoms compete. Every alternate
atom must have a finite occupancy. Reading a missing one as zero or one would
decide the choice by a guess. An occupancy is read by the grammar of
cif_numbers.py, as the parser reads it, so one refined with a standard
uncertainty, such as 0.50(2), counts as 0.50 here too.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..refused_values import listed_values
from .cif_numbers import coerce_cif_numbers
from .residue_identity import build_residue_identities, clean_string_series

_ALT_ID_COLUMN = "_atom_site.label_alt_id"
_OCCUPANCY_COLUMN = "_atom_site.occupancy"

# The decimal places a mean occupancy is rounded to before conformers are
# compared. Occupancies are recorded to a few decimals. Two means that differ
# only beyond this are the same mean summed in a different order. They must
# tie, so that table order decides and rounding error does not.
_MEAN_OCCUPANCY_DECIMALS = 9


def _alternate_occupancies(
    dataframe: pd.DataFrame, positions: npt.NDArray[np.intp]
) -> npt.NDArray[np.float64]:
    """Return the occupancies of the rows at positions, as finite floats.

    A standard uncertainty in parentheses is dropped. Raises ValueError
    naming the occupancy column when it is absent or when any of those rows
    holds a value that is not a finite number.
    """
    if _OCCUPANCY_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Column {_OCCUPANCY_COLUMN!r} is missing, but atoms have alternate "
            f"location identifiers in {_ALT_ID_COLUMN!r}, and the conformer kept "
            f"is chosen by occupancy."
        )
    raw = dataframe[_OCCUPANCY_COLUMN].iloc[positions]
    occupancy = coerce_cif_numbers(raw).to_numpy(dtype=np.float64, na_value=np.nan)
    refused = ~np.isfinite(occupancy)
    if refused.any():
        raise ValueError(
            f"Column {_OCCUPANCY_COLUMN!r} must hold a finite number for every "
            f"atom with an alternate location identifier, since the conformer "
            f"kept is chosen by occupancy, but holds {listed_values(raw[refused])}."
        )
    return occupancy


def highest_occupancy_conformer_mask(
    dataframe: pd.DataFrame,
) -> npt.NDArray[np.bool_]:
    """Return a mask of the rows to keep so each residue has one conformer.

    The mask is positional, with one entry per row in row order. A row
    without an alternate identifier is shared by every conformer, so it is
    always True, whatever its occupancy. An identifier may be letters,
    digits, or several characters. In each residue, the rows of the
    identifier with the highest mean occupancy are True. Means are compared
    to nine decimal places, and the identifier that appears first wins a tie.
    Raises ValueError naming the occupancy column when a row with an
    alternate identifier has no finite occupancy.
    """
    alt_ids = clean_string_series(dataframe, _ALT_ID_COLUMN).to_numpy()
    alt_positions = np.flatnonzero(alt_ids != "")
    selected = np.ones(len(dataframe), dtype=np.bool_)
    if alt_positions.size == 0:
        return selected

    occupancy = _alternate_occupancies(dataframe, alt_positions)
    residue_keys = build_residue_identities(dataframe).residue_keys.iloc[alt_positions]
    residue_codes = residue_keys.factorize()[0]
    # Factorizing gives every unidentified row the code -1. A negative code
    # derived from its position makes each such row a residue of its own.
    # Such a code cannot collide with the codes of identified residues, which
    # are not negative.
    residue_codes = np.where(residue_codes >= 0, residue_codes, -1 - alt_positions)

    alternates = pd.DataFrame(
        {
            "residue": residue_codes,
            "alt_id": alt_ids[alt_positions],
            "occupancy": occupancy,
            "position": alt_positions,
        }
    )
    conformers = (
        alternates.groupby(["residue", "alt_id"], sort=False)
        .agg(
            mean_occupancy=("occupancy", "mean"),
            first_position=("position", "min"),
        )
        .reset_index()
    )
    conformers["mean_occupancy"] = conformers["mean_occupancy"].round(
        _MEAN_OCCUPANCY_DECIMALS
    )
    kept = conformers.sort_values(
        ["residue", "mean_occupancy", "first_position"],
        ascending=[True, False, True],
    ).drop_duplicates("residue")
    kept_alt_id = alternates["residue"].map(kept.set_index("residue")["alt_id"])
    selected[alt_positions] = (kept_alt_id == alternates["alt_id"]).to_numpy()
    return selected


__all__ = ["highest_occupancy_conformer_mask"]
