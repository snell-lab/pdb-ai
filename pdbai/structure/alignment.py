"""Pairing each RCSB atom with the same atom in its PDB-Redo re-refinement.

RCSB and PDB-Redo write separate mmCIF files for one entry. Their atom serial
numbers drift apart once the two refinements treat waters, hydrogens,
alternate conformers, or ligands differently. So every merge of PDB-Redo
values onto the RCSB atoms pairs atoms by identity instead. An atom's identity
is its author residue key from residue_identity.py plus its atom name. Label
fields are left out because PDB-Redo renumbers them. The alternate identifier
is left out because re-refinement can change which conformer it keeps.

Pairing is one to one or not at all. An atom without an identity is left
unmatched. So is an atom that shares its identity with another atom of its
file, and the aligner counts and logs those atoms.

Identity alone can still pair different atoms, so the aligner also measures
how far apart paired atoms lie. When the median distance exceeds 1 Å, the
whole alignment is rejected and every carried column is left empty. The
aligner only measures the match rate. The PDB-Redo comparison decides when a
rate is low, because the RABDAM bridge aligns a chosen subset on purpose.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .elements import HYDROGEN_ELEMENTS
from .residue_identity import (
    ResidueIdentities,
    ResidueKey,
    build_author_residue_keys,
    build_residue_identities,
    clean_string_series,
)

logger = logging.getLogger(__name__)

# Real re-refinements move matched atoms a median of 0.05 to 0.3 Å. Mis-paired
# structures sit tens of ångströms apart. The limit leaves a wide margin on
# both sides.
_MISPAIRING_SHIFT_LIMIT_ANGSTROM = 1.0

# The parser's named columns the aligner reads. Every table it is given is
# one parse_mmcif_atom_rows wrote, or one narrowed to identity_columns and
# coordinate_columns, so it holds them and the mmCIF items they were read
# from are never needed here.
_ATOM_NAME_COLUMN = "atom_name"
_ELEMENT_COLUMN = "element"
_COORDINATE_COLUMNS = ("x", "y", "z")

# Every column that build_residue_identities or the atom-name lookup reads. A
# DataFrame narrowed to these columns keeps the identity of each atom.
_IDENTITY_COLUMNS = (
    "pdb_id",
    "model_num",
    "_atom_site.pdbx_PDB_model_num",
    "residue_numbering",
    "chain_id",
    "_atom_site.auth_asym_id",
    "_atom_site.label_asym_id",
    "res_num",
    "_atom_site.auth_seq_id",
    "_atom_site.label_seq_id",
    "_atom_site.pdbx_PDB_ins_code",
    "res_name",
    "_atom_site.auth_comp_id",
    "_atom_site.label_comp_id",
    "is_hetatm",
    "_atom_site.group_PDB",
    "_pdbai_polymer_type",
    "_pdbai_entity_type",
    "_pdbai_chem_comp_type",
    _ATOM_NAME_COLUMN,
)

# An atom's identity across files, which is its author residue key followed
# by its upper-cased atom name.
_AtomIdentity = tuple[str, ...]


@dataclass(frozen=True)
class AlignmentDiagnostics:
    """How well the RCSB and PDB-Redo atoms of one structure paired."""

    # The share of eligible RCSB atoms that found a PDB-Redo partner. It is
    # matched_eligible_atom_count over eligible_atom_count. It is 0.0 when no
    # atom is eligible or the alignment was rejected.
    match_rate: float
    # RCSB atoms of any kind that found a PDB-Redo partner.
    matched_atom_count: int
    # The atoms on each side left unmatched because another atom in the same
    # file shares their identity.
    rcsb_ambiguous_atom_count: int
    pdb_redo_ambiguous_atom_count: int
    # How many RCSB atoms are eligible, and how many of those found a
    # partner. protein_heavy_atom_mask selects the eligible atoms, and the
    # match rate is measured over them.
    eligible_atom_count: int
    matched_eligible_atom_count: int
    # Why the distance check rejected the alignment, naming the median
    # distance between paired atoms, or None when it did not reject it.
    rejected_reason: str | None
    # Which RCSB chain IDs the pairing used. "author" is the normal case.
    # "rcsb_label" means PDB-Redo wrote the RCSB label chain IDs as its author
    # chains. Some entries with multi-letter chains come back that way, with
    # AAA, BBB, and CCC renamed to A, B, and C.
    chain_id_source: Literal["author", "rcsb_label"]


@dataclass(frozen=True)
class AlignmentResult:
    """PDB-Redo columns carried onto the RCSB atoms, and how well atoms paired.

    ``columns`` has one row per RCSB atom, under the RCSB DataFrame's index
    and in its order. It has one column per carried PDB-Redo column, named
    with the suffix _pdb_redo. An unmatched atom's values are missing.
    """

    columns: pd.DataFrame
    diagnostics: AlignmentDiagnostics


def identity_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return the columns of the DataFrame that atom identities are read from.

    A caller that narrows a DataFrame before alignment keeps these, so every
    atom keeps the identity it had in the full DataFrame.
    """
    return [column for column in _IDENTITY_COLUMNS if column in dataframe.columns]


def coordinate_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return the Cartesian coordinate columns present in the DataFrame.

    A caller that narrows a DataFrame before alignment keeps these, so the
    distance check can still measure how far paired atoms moved.
    """
    return [column for column in _COORDINATE_COLUMNS if column in dataframe.columns]


def protein_heavy_atom_mask(
    dataframe: pd.DataFrame, identities: ResidueIdentities | None = None
) -> npt.NDArray[np.bool_]:
    """Return which atoms both refinements are expected to model.

    These are the protein heavy atoms. PDB-Redo strips deposited hydrogens
    and remodels waters. Counting those atoms as misses would make a match
    rate report modelling choices instead of alignment quality. Hydrogen and
    its isotopes are recognized by element symbol, and an atom without a symbol
    counts as heavy. When the DataFrame has no protein heavy atom, as in a
    nucleic-acid entry, every heavy atom counts. When it has no heavy atom
    either, every atom counts. Pass ``identities`` when the DataFrame's
    residue identities are already built.
    """
    element = clean_string_series(dataframe, _ELEMENT_COLUMN).str.upper()
    heavy = ~element.isin(HYDROGEN_ELEMENTS).to_numpy(dtype=bool)
    if identities is None:
        identities = build_residue_identities(dataframe)
    protein_heavy = identities.protein_polymer.to_numpy(dtype=bool) & heavy
    if protein_heavy.any():
        return protein_heavy
    if heavy.any():
        return heavy
    return np.ones(len(dataframe), dtype=bool)


def _atom_identities(
    identities: ResidueIdentities, atom_names: pd.Series
) -> tuple[list[_AtomIdentity | None], int]:
    """Return each atom's identity, or None, and how many atoms share one.

    An atom has no identity when its author residue key or its atom name is
    missing. It also loses its identity when another atom of the same file
    shares it.
    """
    residue_keys: list[ResidueKey | None] = build_author_residue_keys(
        identities
    ).tolist()
    keys: list[_AtomIdentity | None] = [
        (*residue_key, atom_name) if residue_key is not None and atom_name else None
        for residue_key, atom_name in zip(residue_keys, atom_names, strict=True)
    ]
    counts = Counter(key for key in keys if key is not None)
    shared = sum(count for count in counts.values() if count > 1)
    unique = [key if key is not None and counts[key] == 1 else None for key in keys]
    return unique, shared


def _partner_positions(
    rcsb_keys: list[_AtomIdentity | None], pdb_redo_keys: list[_AtomIdentity | None]
) -> npt.NDArray[np.intp]:
    """Return each RCSB atom's PDB-Redo row position, or -1 for none."""
    pdb_redo_positions = {
        key: position for position, key in enumerate(pdb_redo_keys) if key is not None
    }
    return np.fromiter(
        (
            pdb_redo_positions.get(key, -1) if key is not None else -1
            for key in rcsb_keys
        ),
        dtype=np.intp,
        count=len(rcsb_keys),
    )


def _with_label_chains_as_author(
    identities: ResidueIdentities,
) -> ResidueIdentities | None:
    """Return ``identities`` with label chain IDs in place of author chains.

    An atom without a label chain ID keeps its author chain. Returns None when
    the substitution would change no atom's chain.
    """
    label_chain = identities.label_chain
    chain = label_chain.where(label_chain.ne(""), identities.auth_chain)
    if chain.eq(identities.auth_chain).all():
        return None
    return replace(identities, auth_chain=chain)


def _coordinate_matrix(dataframe: pd.DataFrame) -> npt.NDArray[np.float64] | None:
    """Return one row of x, y, z per atom, or None when an axis has no column.

    A coordinate that is not a number reads as NaN.
    """
    if not set(_COORDINATE_COLUMNS) <= set(dataframe.columns):
        return None
    return np.column_stack(
        [
            np.asarray(
                pd.to_numeric(dataframe[axis], errors="coerce"), dtype=np.float64
            )
            for axis in _COORDINATE_COLUMNS
        ]
    )


def _median_shift(
    rcsb_dataframe: pd.DataFrame,
    pdb_redo_dataframe: pd.DataFrame,
    partners: npt.NDArray[np.intp],
) -> float | None:
    """Return the median distance in Å between paired atoms, or None.

    partners holds each RCSB atom's PDB-Redo row position, or -1. A pair
    with a coordinate that is missing or not a number is left out. None is
    returned when either side lacks a coordinate axis or no pair is left.
    """
    rcsb_coordinates = _coordinate_matrix(rcsb_dataframe)
    pdb_redo_coordinates = _coordinate_matrix(pdb_redo_dataframe)
    if rcsb_coordinates is None or pdb_redo_coordinates is None:
        return None
    matched = partners >= 0
    shifts = np.linalg.norm(
        rcsb_coordinates[matched] - pdb_redo_coordinates[partners[matched]], axis=1
    )
    finite_shifts = shifts[np.isfinite(shifts)]
    return float(np.median(finite_shifts)) if finite_shifts.size else None


def align(
    rcsb_dataframe: pd.DataFrame,
    pdb_redo_dataframe: pd.DataFrame,
    *,
    pdb_id: str,
    caller: str,
    carry_columns: Sequence[str],
) -> AlignmentResult:
    """Return the PDB-Redo values of carry_columns paired onto the RCSB atoms.

    A carried column that pdb_redo_dataframe lacks is left out of the result,
    since callers name every column they could use. Sometimes no eligible
    RCSB atom pairs on author chain IDs. The RCSB label chain IDs are then
    tried in their place, because PDB-Redo writes them as author chains for
    some entries with multi-letter chains. That pairing must still pass the
    distance check, and the diagnostics record which chain IDs were used.
    pdb_id and ``caller``, the name of the calling engine, only label the
    log messages.
    """
    carried = [
        column
        for column in dict.fromkeys(carry_columns)
        if column in pdb_redo_dataframe.columns
    ]
    label = f"{pdb_id} ({caller})"

    pdb_redo_keys, pdb_redo_ambiguous = _atom_identities(
        build_residue_identities(pdb_redo_dataframe),
        clean_string_series(pdb_redo_dataframe, _ATOM_NAME_COLUMN).str.upper(),
    )
    rcsb_identities = build_residue_identities(rcsb_dataframe)
    rcsb_atom_names = clean_string_series(rcsb_dataframe, _ATOM_NAME_COLUMN).str.upper()
    rcsb_keys, rcsb_ambiguous = _atom_identities(rcsb_identities, rcsb_atom_names)
    partners = _partner_positions(rcsb_keys, pdb_redo_keys)

    eligible = protein_heavy_atom_mask(rcsb_dataframe, rcsb_identities)
    eligible_count = int(eligible.sum())
    chain_id_source: Literal["author", "rcsb_label"] = "author"
    relabelled = (
        _with_label_chains_as_author(rcsb_identities)
        if eligible_count and not (eligible & (partners >= 0)).any()
        else None
    )
    if relabelled is not None:
        retry_keys, retry_ambiguous = _atom_identities(relabelled, rcsb_atom_names)
        retry_partners = _partner_positions(retry_keys, pdb_redo_keys)
        if (eligible & (retry_partners >= 0)).any():
            partners, rcsb_ambiguous = retry_partners, retry_ambiguous
            chain_id_source = "rcsb_label"
            logger.warning(
                "RCSB/PDB-Redo atom alignment for %s paired no protein heavy atom "
                "on author chain IDs; the PDB-Redo file carries the RCSB label "
                "chain IDs instead, so the alignment used those.",
                label,
            )

    if rcsb_ambiguous or pdb_redo_ambiguous:
        logger.warning(
            "RCSB/PDB-Redo atom alignment for %s found duplicate atom identities "
            "(%d RCSB, %d PDB-Redo) that cannot be told apart by chain, residue, and "
            "atom name. Those atoms are left unmatched rather than paired arbitrarily.",
            label,
            rcsb_ambiguous,
            pdb_redo_ambiguous,
        )

    rejected_reason: str | None = None
    median_shift = _median_shift(rcsb_dataframe, pdb_redo_dataframe, partners)
    if median_shift is not None and median_shift > _MISPAIRING_SHIFT_LIMIT_ANGSTROM:
        rejected_reason = (
            f"matched atoms are a median {median_shift:.1f} Å apart "
            f"(limit {_MISPAIRING_SHIFT_LIMIT_ANGSTROM:g} Å), so the "
            "identity match paired different atoms"
        )
        logger.warning(
            "RCSB/PDB-Redo atom alignment for %s was rejected: %s. Every column "
            "carried from PDB-Redo is left empty for this structure.",
            label,
            rejected_reason,
        )
        partners = np.full(len(rcsb_dataframe), -1, dtype=np.intp)

    matched = partners >= 0
    matched_eligible_count = int((matched & eligible).sum())
    # Every unmatched atom has position -1, which reindexing leaves missing.
    columns = (
        pdb_redo_dataframe[carried]
        .reset_index(drop=True)
        .reindex(partners)
        .set_axis(rcsb_dataframe.index)
        .add_suffix("_pdb_redo")
    )
    diagnostics = AlignmentDiagnostics(
        match_rate=matched_eligible_count / eligible_count if eligible_count else 0.0,
        matched_atom_count=int(matched.sum()),
        rcsb_ambiguous_atom_count=rcsb_ambiguous,
        pdb_redo_ambiguous_atom_count=pdb_redo_ambiguous,
        eligible_atom_count=eligible_count,
        matched_eligible_atom_count=matched_eligible_count,
        rejected_reason=rejected_reason,
        chain_id_source=chain_id_source,
    )
    return AlignmentResult(columns=columns, diagnostics=diagnostics)


__all__ = [
    "AlignmentDiagnostics",
    "AlignmentResult",
    "align",
    "coordinate_columns",
    "identity_columns",
    "protein_heavy_atom_mask",
]
