"""Component categories, protein radii, chain ends, and relative areas.

This module decides what kind of thing each atom and residue is, for the
SASA calculation in compute.py. Each function reads the atom table, or the
residue identities built from it, and none of them computes an area.

An atom's component category decides, in compute.py, whether it occludes the
bound context. Whether an atom belongs to a polymer comes from
residue_identity.py, which puts entity metadata before the ATOM or HETATM
record. Any named non-polymer that no list covers counts as a ligand, so a
ligand missing from the lists still occludes. Chains, residues, and their
names are read only from those residue identities, never from chain_id or
res_num, so a structure numbered by its label items is classified as any
other.

Chain ends are found conservatively. A residue at either end of a chain is
terminal, and so is each residue beside a gap or a broken peptide bond. Such
a residue lacks a neighbour that the Tien maxima assume, so its relative area
could read too high. A gap inside a chain therefore counts as two chain ends.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from contextlib import suppress

import numpy as np
import pandas as pd

from ....structure.residue_identity import (
    ResidueIdentities,
    ResidueKey,
    first_nonempty_string_series,
)
from .constants import (
    DSSP_BACKBONE_RADII,
    DSSP_RADIUS_SIDECHAIN,
    EXPERIMENTAL_ADDITIVE_IDS,
    KNOWN_COFACTOR_IDS,
    MAX_ASA_TIEN2013,
    METAL_ELEMENTS,
    RASA_BURIED_BELOW,
    RASA_EXPOSED_FROM,
    UNKNOWN_COMPONENT_IDS,
    WATER_COMPONENT_IDS,
)


def normalized_elements(dataframe: pd.DataFrame) -> pd.Series:
    """Return each atom's upper-case element symbol, or "" if it has none."""
    return first_nonempty_string_series(
        dataframe,
        ("element", "_atom_site.type_symbol"),
    ).str.upper()


def normalized_atom_names(dataframe: pd.DataFrame) -> pd.Series:
    """Return each atom's name in upper case, or "" if it has none.

    The atom_name column is read first, then the label and author atom names.
    """
    return first_nonempty_string_series(
        dataframe,
        ("atom_name", "_atom_site.label_atom_id", "_atom_site.auth_atom_id"),
    ).str.upper()


def component_categories(
    identities: ResidueIdentities,
    elements: pd.Series,
) -> pd.Series:
    """Return each atom's component category, under the index of ``elements``.

    A polymer atom is protein_polymer or nonprotein_polymer by its entity. A
    non-polymer atom is water, experimental_additive, or ligand_or_cofactor
    when its component ID is on a list in constants.py. A residue that no list
    covers is a metal if it is a single atom of a metal element. Otherwise it
    is ligand_or_cofactor if it has a component name other than UNK, UNL, or
    UNX. All that remains is unknown_nonpolymer, including any unlisted
    non-polymer atom that has no residue identity.
    """
    categories = pd.Series("unknown_nonpolymer", index=elements.index, dtype=object)
    categories.loc[identities.polymer & ~identities.protein_polymer] = (
        "nonprotein_polymer"
    )
    categories.loc[identities.protein_polymer] = "protein_polymer"

    nonpolymer = ~identities.polymer
    residue_names = identities.res_name
    categories.loc[nonpolymer & residue_names.isin(WATER_COMPONENT_IDS)] = "water"
    categories.loc[nonpolymer & residue_names.isin(EXPERIMENTAL_ADDITIVE_IDS)] = (
        "experimental_additive"
    )
    categories.loc[nonpolymer & residue_names.isin(KNOWN_COFACTOR_IDS)] = (
        "ligand_or_cofactor"
    )

    # The remaining non-polymer residues are judged whole, so that every atom
    # of one residue gets the same category.
    grouped_positions: dict[ResidueKey, list[int]] = {}
    nonpolymer_rows = nonpolymer.to_numpy(dtype=bool)
    for position, key in enumerate(identities.residue_keys.tolist()):
        if key is not None and nonpolymer_rows[position]:
            grouped_positions.setdefault(key, []).append(position)
    for positions in grouped_positions.values():
        rows = np.asarray(positions, dtype=int)
        if (categories.iloc[rows] != "unknown_nonpolymer").any():
            continue
        if len(rows) == 1 and elements.iloc[rows[0]] in METAL_ELEMENTS:
            categories.iloc[rows] = "metal"
            continue
        residue_name = residue_names.iloc[rows[0]]
        if residue_name and residue_name not in UNKNOWN_COMPONENT_IDS:
            categories.iloc[rows] = "ligand_or_cofactor"
    return categories


def protein_radii(atom_names: pd.Series) -> np.ndarray:
    """Return the DSSP radius in Å of each protein atom, by its upper-case name.

    N, CA, C, and O take their backbone radii, and any other name takes the
    side-chain radius. The radii suit protein atoms only.
    """
    return np.asarray(
        [DSSP_BACKBONE_RADII.get(name, DSSP_RADIUS_SIDECHAIN) for name in atom_names],
        dtype=float,
    )


def rasa_and_burial(
    residue_name: str,
    residue_sasa: float,
) -> tuple[float, bool, bool] | None:
    """Return a residue's relative area and whether it is buried or exposed.

    The relative area is ``residue_sasa``, in Å², divided by the Tien maximum
    for ``residue_name``, an upper-case component ID, and it is not clipped
    at 1. Whether the residue is buried or exposed follows the cut-offs in
    constants.py, and a residue between the two is neither. Returns None for a
    residue with no Tien maximum.
    """
    maximum = MAX_ASA_TIEN2013.get(residue_name)
    if maximum is None:
        return None
    rasa = residue_sasa / maximum
    return rasa, rasa < RASA_BURIED_BELOW, rasa >= RASA_EXPOSED_FROM


def terminal_residues(
    identities: ResidueIdentities,
    residue_positions: Sequence[np.ndarray],
    valid_atom_mask: np.ndarray,
    atom_names: pd.Series,
    coordinates: np.ndarray,
) -> set[int]:
    """Return the indices of the residues at a chain end or beside a break.

    ``residue_positions`` holds the atom positions of each residue, and a
    residue's index is its place in that sequence. Residues fall into chains by
    structure, model, and chain ID. Within a chain, residues are ordered by
    label_seq_id where it is a number, then by auth_seq_id where only that is
    a number, and then by table position. A break is a step in label_seq_id
    other than one, or a peptide bond that cannot be confirmed. Confirming the
    bond needs the C of one residue and the N of the next among the atoms
    ``valid_atom_mask`` marks, lying 0.8 to 2.0 Å apart.
    """
    first_position = [int(positions[0]) for positions in residue_positions]
    chains: dict[tuple[str, str, str], list[int]] = {}
    for residue, position in enumerate(first_position):
        chain_key = (
            identities.pdb_id.iloc[position],
            identities.model.iloc[position],
            identities.chain.iloc[position],
        )
        chains.setdefault(chain_key, []).append(residue)

    def _sort_value(residue: int) -> tuple[int, float, int]:
        position = first_position[residue]
        label_value = identities.label_seq.iloc[position]
        auth_value = identities.auth_seq.iloc[position]
        for rank, value in ((0, label_value), (1, auth_value)):
            with suppress(ValueError):
                return rank, float(value), position
        return 2, float(position), position

    names = atom_names.to_numpy(dtype=object)

    def _first_valid_atom(residue: int, name: str) -> int | None:
        for position in residue_positions[residue]:
            if names[position] == name and valid_atom_mask[position]:
                return int(position)
        return None

    terminal: set[int] = set()
    for chain_residues in chains.values():
        chain_residues.sort(key=_sort_value)
        terminal.add(chain_residues[0])
        terminal.add(chain_residues[-1])
        for left, right in itertools.pairwise(chain_residues):
            left_label = identities.label_seq.iloc[first_position[left]]
            right_label = identities.label_seq.iloc[first_position[right]]
            sequence_break = False
            # A missing or nonnumeric label_seq_id leaves the bond test to
            # decide.
            with suppress(ValueError):
                sequence_break = float(right_label) - float(left_label) != 1.0

            left_c = _first_valid_atom(left, "C")
            right_n = _first_valid_atom(right, "N")
            # A peptide C-N bond is 1.33 Å long. The window allows for a
            # strained or poorly modelled bond, and it rejects the gap that a
            # missing residue leaves.
            bonded = (
                left_c is not None
                and right_n is not None
                and 0.8 <= math.dist(coordinates[left_c], coordinates[right_n]) <= 2.0
            )
            if sequence_break or not bonded:
                terminal.add(left)
                terminal.add(right)
    return terminal


__all__ = [
    "component_categories",
    "normalized_atom_names",
    "normalized_elements",
    "protein_radii",
    "rasa_and_burial",
    "terminal_residues",
]
