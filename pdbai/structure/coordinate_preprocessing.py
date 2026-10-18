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

The choice also records what the dropped conformers held, which the kept rows
would otherwise lose. For each atom modelled in alternate conformers, it sums
the atom's occupancy over all of them and counts them. An atom is one atom
name within one residue. A total below one then marks an occupancy refined
below one, such as an atom modelled as partly lost, rather than one shared
between conformers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..refused_values import listed_values
from .cif_numbers import coerce_cif_numbers
from .residue_identity import build_residue_identities, clean_string_series

_ALT_ID_COLUMN = "_atom_site.label_alt_id"
_ATOM_NAME_COLUMN = "_atom_site.label_atom_id"
_OCCUPANCY_COLUMN = "_atom_site.occupancy"

# The decimal places a mean occupancy is rounded to before conformers are
# compared. Occupancies are recorded to a few decimals. Two means that differ
# only beyond this are the same mean summed in a different order. They must
# tie, so that table order decides and rounding error does not. Summed
# occupancies are rounded the same way, so 0.56 and 0.44 sum to exactly one.
_MEAN_OCCUPANCY_DECIMALS = 9


@dataclass(frozen=True)
class ConformerChoice:
    """The rows to keep, and each row's atom summed over all its conformers.

    Every array is positional, with one entry per row in row order. keep is
    True for the rows of the conformer kept. occupancy_total is the occupancy
    of the row's atom summed over every alternate conformer the file models
    it in, and NaN for a row without an alternate identifier, whose own
    occupancy is its total. conformer_count is the number of alternate
    identifiers the row's atom is modelled under, and 1 for a row without
    one.
    """

    keep: npt.NDArray[np.bool_]
    occupancy_total: npt.NDArray[np.float64]
    conformer_count: npt.NDArray[np.int64]


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


def choose_conformers(dataframe: pd.DataFrame) -> ConformerChoice:
    """Return the conformer kept for each residue, with each atom's totals.

    A row without an alternate identifier is shared by every conformer, so it
    is always kept, whatever its occupancy. An identifier may be letters,
    digits, or several characters. In each residue, the rows of the
    identifier with the highest mean occupancy are kept. Means are compared
    to nine decimal places, and the identifier that appears first wins a tie.

    The totals group the alternate rows of one residue by atom name. An
    alternate row without an atom name counts as an atom of its own, so
    unnamed atoms are never summed together. Raises ValueError naming the
    occupancy column when a row with an alternate identifier has no finite
    occupancy.
    """
    alt_ids = clean_string_series(dataframe, _ALT_ID_COLUMN).to_numpy()
    alt_positions = np.flatnonzero(alt_ids != "")
    selected = np.ones(len(dataframe), dtype=np.bool_)
    occupancy_total = np.full(len(dataframe), np.nan, dtype=np.float64)
    conformer_count = np.ones(len(dataframe), dtype=np.int64)
    if alt_positions.size == 0:
        return ConformerChoice(selected, occupancy_total, conformer_count)

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

    atom_names = clean_string_series(dataframe, _ATOM_NAME_COLUMN).to_numpy()[
        alt_positions
    ]
    atom_codes = (
        alternates.assign(atom=atom_names)
        .groupby(["residue", "atom"], sort=False)
        .ngroup()
        .to_numpy()
    )
    # As with residues, an unnamed atom takes a negative code of its own.
    atom_codes = np.where(atom_names != "", atom_codes, -1 - alt_positions)
    by_atom = alternates.groupby(atom_codes, sort=False)
    occupancy_total[alt_positions] = (
        by_atom["occupancy"].transform("sum").round(_MEAN_OCCUPANCY_DECIMALS).to_numpy()
    )
    conformer_count[alt_positions] = by_atom["alt_id"].transform("nunique").to_numpy()
    return ConformerChoice(selected, occupancy_total, conformer_count)


def highest_occupancy_conformer_mask(
    dataframe: pd.DataFrame,
) -> npt.NDArray[np.bool_]:
    """Return a mask of the rows to keep so each residue has one conformer.

    The mask is choose_conformers(dataframe).keep, positional, with one entry
    per row in row order. Raises ValueError naming the occupancy column when
    a row with an alternate identifier has no finite occupancy.
    """
    return choose_conformers(dataframe).keep


__all__ = ["ConformerChoice", "choose_conformers", "highest_occupancy_conformer_mask"]
