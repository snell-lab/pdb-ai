"""Scoring one structure file and carrying its scores onto the RCSB atoms.

The engine can score two files per entry. One is the file the atom table
was parsed from, which is RCSB's file or an upload. The other is PDB-Redo's
re-refinement. Scores of the first share the atom table's serial numbers,
so they are joined on _atom_site.id. The two refinements number their
atoms differently, so a join on serials would pair unrelated atoms and
give no sign of error. Scores of PDB-Redo's file are therefore bridged by
atom identity through structure.alignment. The caller states which case
applies. The merge never falls back to serial numbers.

A serial number that rabdam-custom gave to two atoms pairs with neither. When
no score reaches any atom, because the bridge cannot be built, is rejected,
or pairs nothing, or because no serial number is in the atom table, the
per-atom columns stay empty and a reason is returned.

Bnet is one number per structure and needs no pairing. It therefore
reaches the atom table even when no per-atom score does. A file that
rabdam-custom cannot score gives a warning and empty columns, not a failed
structure. Only a missing installation raises, because it would fail every
structure alike.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ....structure import alignment
from .integration import (
    RabdamCustomError,
    RabdamCustomNotInstalledError,
    RabdamCustomResult,
    run_rabdam_custom_for_structure,
)


@dataclass(frozen=True)
class PlacedRabdamScores:
    """One source's RABDAM columns on the atom table, and how they got there.

    alignment holds the identity bridge's diagnostics when a bridge ran.
    unaligned_reason says why no per-atom score reached the atom table,
    although the source was scored. It is None when at least one did, and
    when there were no scores to place. While it is set, the per-atom
    columns are empty.

    unplaced_atom_count counts the scored atoms that reached no row while
    others did. rabdam-custom keeps the highest-occupancy conformer of each
    atom, as the Bnet reference was computed, while the atom table keeps one
    conformer per residue. Where the two choose differently, the atom the
    scores belong to has no row. A bridged atom can also find no partner.
    """

    columns: pd.DataFrame
    alignment: alignment.AlignmentDiagnostics | None = None
    unaligned_reason: str | None = None
    unplaced_atom_count: int = 0


def score_structure_file(
    pdb_id: str,
    cif_path: Path,
    structure_dataframe: pd.DataFrame,
    warnings: list[str],
) -> RabdamCustomResult | None:
    """Score a structure file, or return None after appending a warning.

    None means rabdam-custom could not score the file, and the warning
    appended to ``warnings`` says why. rabdam-custom's own warnings are
    appended as well. The percentile is matched on the resolution in
    ``structure_dataframe``, the atom table, when it has one. Every source
    of an entry is therefore ranked at the same resolution. The X-ray
    refinement's own resolution, resolution_xray, comes first. The
    resolution column may hold a joint entry's neutron refinement instead,
    and BDamage scores the X-ray model. Raise RabdamCustomNotInstalledError
    when the package cannot be imported.
    """
    resolution_override: float | None = None
    for column in ("resolution_xray", "resolution"):
        if column not in structure_dataframe.columns:
            continue
        resolutions = pd.to_numeric(
            structure_dataframe[column], errors="coerce"
        ).dropna()
        if not resolutions.empty:
            resolution_override = float(resolutions.iloc[0])
            break

    try:
        rabdam_result = run_rabdam_custom_for_structure(
            pdb_id, cif_path, resolution_override=resolution_override
        )
    except RabdamCustomNotInstalledError:
        raise
    except RabdamCustomError as exception:
        # Some of rabdam-custom's messages end in a period and some do not.
        warnings.append(
            f"rabdam-custom failed for {pdb_id}: {str(exception).rstrip('.')}. "
            "BDamage columns will be empty for this structure."
        )
        return None

    warnings.extend(rabdam_result.warnings)
    return rabdam_result


def place_scores_on_atom_table(
    atom_dataframe: pd.DataFrame,
    rabdam_result: RabdamCustomResult | None,
    pdb_redo_dataframe: pd.DataFrame | None = None,
    *,
    pdb_id: str = "",
    suffix: str = "",
    requires_alignment: bool,
) -> PlacedRabdamScores:
    """Return one source's bdamage, packing_density, and bnet columns.

    The column names end in ``suffix``, and the rows match
    ``atom_dataframe``. Every value is missing when ``rabdam_result`` is
    None. Bnet fills every row whenever it was produced.

    Set ``requires_alignment`` when the scores came from PDB-Redo's file.
    Its atoms are then paired with the RCSB atoms by identity, through
    ``pdb_redo_dataframe``, the parsed PDB-Redo atom table. Otherwise the
    scores must come from the file ``atom_dataframe`` was parsed from, and
    they are joined on _atom_site.id. ``pdb_id`` only labels the bridge's
    log messages.
    """
    bdamage_column = f"bdamage{suffix}"
    packing_density_column = f"packing_density{suffix}"
    bnet_column = f"bnet{suffix}"
    n = len(atom_dataframe)
    columns = pd.DataFrame(
        {
            bdamage_column: np.full(n, np.nan),
            packing_density_column: np.full(n, np.nan),
            bnet_column: np.full(n, np.nan),
        },
        index=atom_dataframe.index,
    )

    if rabdam_result is None:
        return PlacedRabdamScores(columns=columns)

    if rabdam_result.bnet is not None:
        columns[bnet_column] = rabdam_result.bnet

    if rabdam_result.per_atom.empty:
        return PlacedRabdamScores(
            columns=columns,
            unaligned_reason="it gave no scored atom a serial number",
        )

    # mmCIF gives each atom its own _atom_site.id. A serial shared by two
    # scored atoms names neither, and the lookup below needs unique keys.
    per_atom = rabdam_result.per_atom
    rabdam_atoms = per_atom[~per_atom["_atom_site.id"].duplicated(keep=False)]

    if not requires_alignment:
        if "_atom_site.id" not in atom_dataframe.columns:
            return PlacedRabdamScores(
                columns=columns,
                unaligned_reason="the atom table carries no _atom_site.id column",
            )
        ids = atom_dataframe["_atom_site.id"].astype(str)
        lookup = rabdam_atoms.set_index("_atom_site.id")
        columns[bdamage_column] = ids.map(lookup["bdamage"]).to_numpy(dtype=float)
        columns[packing_density_column] = ids.map(lookup["packing_density"]).to_numpy(
            dtype=float
        )
        return _placed(
            columns,
            bdamage_column,
            len(rabdam_atoms),
            "none of its atom serial numbers is in the atom table",
        )

    if pdb_redo_dataframe is None or pdb_redo_dataframe.empty:
        return PlacedRabdamScores(
            columns=columns,
            unaligned_reason=(
                "no parsed PDB-Redo atom table was available to pair its atoms "
                "by identity"
            ),
        )

    if "_atom_site.id" not in pdb_redo_dataframe.columns:
        return PlacedRabdamScores(
            columns=columns,
            unaligned_reason="the PDB-Redo atom table carries no _atom_site.id column",
        )

    # The bridge is PDB-Redo's atom table with the scores joined on PDB-Redo
    # serials. The aligner then carries the scores across by identity.
    bridge = pdb_redo_dataframe[
        [
            "_atom_site.id",
            *alignment.identity_columns(pdb_redo_dataframe),
            *alignment.coordinate_columns(pdb_redo_dataframe),
        ]
    ].astype({"_atom_site.id": str})
    bridge = bridge.merge(
        rabdam_atoms[["_atom_site.id", "bdamage", "packing_density"]],
        on="_atom_site.id",
        how="inner",
    )
    alignment_result = alignment.align(
        atom_dataframe,
        bridge,
        pdb_id=pdb_id,
        caller="rabdam_engine",
        carry_columns=("bdamage", "packing_density"),
    )
    carried = alignment_result.columns
    columns[bdamage_column] = carried["bdamage_pdb_redo"].to_numpy(dtype=float)
    columns[packing_density_column] = carried["packing_density_pdb_redo"].to_numpy(
        dtype=float
    )

    diagnostics = alignment_result.diagnostics
    if diagnostics.rejected_reason is not None:
        return PlacedRabdamScores(
            columns=columns,
            alignment=diagnostics,
            unaligned_reason=diagnostics.rejected_reason,
        )
    placed = _placed(
        columns,
        bdamage_column,
        len(rabdam_atoms),
        "none of its scored atoms paired with an RCSB atom by identity",
    )
    return dataclasses.replace(placed, alignment=diagnostics)


def _placed(
    columns: pd.DataFrame,
    bdamage_column: str,
    scored_atom_count: int,
    reason_if_none: str,
) -> PlacedRabdamScores:
    # Each scored atom reaches at most one row, by its unique serial or the
    # bridge's one-to-one pairing, so the rows with a score count the atoms
    # placed.
    placed_atom_count = int(columns[bdamage_column].notna().sum())
    if not placed_atom_count:
        return PlacedRabdamScores(columns=columns, unaligned_reason=reason_if_none)
    return PlacedRabdamScores(
        columns=columns,
        unplaced_atom_count=max(0, scored_atom_count - placed_atom_count),
    )


# The scores compared between the PDB-Redo and RCSB models when both are
# scored. A matches flag promises that the two values differ by no more than
# AGREEMENT_TOLERANCE, and its description in engine.py prints the number. So
# the tolerance is one absolute difference for every score, as the flag says.
# The flag asks whether the two refinements gave the same value, not whether
# the values are close in a scientific sense. PDB-AI chose the number, and
# RABDAM defines none. BDamage and Bnet are of order one, so the number sits
# far above floating-point noise in them. For bnet_percentile, a percentage
# from 0 to 100, it is a thousandth of a percentage point. A relative
# tolerance would loosen that test at high percentiles and make it almost
# impossible to pass near zero. For packing density, which counts atoms, it
# means equality.
COMPARED_SCORES = ("bdamage", "packing_density", "bnet", "bnet_percentile")
AGREEMENT_TOLERANCE = 1e-3


def add_agreement_flags(atom_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add a matches flag, such as bdamage_matches, for each shared score.

    The DataFrame is changed in place and returned. A flag is missing where
    either value is, so an atom the bridge could not pair does not read as a
    disagreement.
    """
    for score in COMPARED_SCORES:
        pdb_redo_column, rcsb_column = f"{score}_pdb_redo", f"{score}_rcsb"
        if (
            pdb_redo_column not in atom_dataframe.columns
            or rcsb_column not in atom_dataframe.columns
        ):
            continue
        pdb_redo_values = pd.to_numeric(
            atom_dataframe[pdb_redo_column], errors="coerce"
        )
        rcsb_values = pd.to_numeric(atom_dataframe[rcsb_column], errors="coerce")
        both_present = pdb_redo_values.notna() & rcsb_values.notna()
        agrees = (pdb_redo_values - rcsb_values).abs() <= AGREEMENT_TOLERANCE
        atom_dataframe[f"{score}_matches"] = agrees.astype("boolean").where(
            both_present
        )
    return atom_dataframe


__all__ = [
    "AGREEMENT_TOLERANCE",
    "COMPARED_SCORES",
    "PlacedRabdamScores",
    "add_agreement_flags",
    "place_scores_on_atom_table",
    "score_structure_file",
]
