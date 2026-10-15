"""Each atom's B-factor against the atom it hangs from.

b_factor_ratio_to_parent gives, for every atom of the 20 standard amino
acids, its B-factor over the mean B-factor of its parents, the atoms it is
bonded to that lie one bond nearer the CA. It is the measure the
residue_sites table takes for a few named groups, taken for every atom, so
that an analysis can ask which atoms move with dose without naming any in
advance.

A B-factor rises along a side chain in any structure, so a ratio above one
means little by itself. What a ratio is compared with is the same atom type
in another structure, or at another dose.

PARENT_ATOMS holds the bonds. It was derived from coordinates, as the bonded
atoms of each atom that a walk outward from the CA reaches one step earlier,
and it is fixed chemistry: the N, the C, and the CB hang from the CA, the O
and the OXT from the C, and each side chain as listed. An atom that closes a
ring has two parents, as the CZ of Phe has CE1 and CE2. The CD of Pro hangs
from the N, which is the shorter way round its ring. The CA has no parent
and no ratio.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..site_geometry import SiteAtoms

_BACKBONE: dict[str, tuple[str, ...]] = {
    "N": ("CA",),
    "C": ("CA",),
    "O": ("C",),
    "OXT": ("C",),
    "CB": ("CA",),
}

# The side-chain atoms of each residue beyond the CB, each with its parents.
_SIDE_CHAINS: dict[str, dict[str, tuple[str, ...]]] = {
    "ALA": {},
    "ARG": {
        "CG": ("CB",),
        "CD": ("CG",),
        "NE": ("CD",),
        "CZ": ("NE",),
        "NH1": ("CZ",),
        "NH2": ("CZ",),
    },
    "ASN": {"CG": ("CB",), "OD1": ("CG",), "ND2": ("CG",)},
    "ASP": {"CG": ("CB",), "OD1": ("CG",), "OD2": ("CG",)},
    "CYS": {"SG": ("CB",)},
    "GLN": {"CG": ("CB",), "CD": ("CG",), "OE1": ("CD",), "NE2": ("CD",)},
    "GLU": {"CG": ("CB",), "CD": ("CG",), "OE1": ("CD",), "OE2": ("CD",)},
    "GLY": {},
    "HIS": {
        "CG": ("CB",),
        "ND1": ("CG",),
        "CD2": ("CG",),
        "CE1": ("ND1",),
        "NE2": ("CD2",),
    },
    "ILE": {"CG1": ("CB",), "CG2": ("CB",), "CD1": ("CG1",)},
    "LEU": {"CG": ("CB",), "CD1": ("CG",), "CD2": ("CG",)},
    "LYS": {"CG": ("CB",), "CD": ("CG",), "CE": ("CD",), "NZ": ("CE",)},
    "MET": {"CG": ("CB",), "SD": ("CG",), "CE": ("SD",)},
    "PHE": {
        "CG": ("CB",),
        "CD1": ("CG",),
        "CD2": ("CG",),
        "CE1": ("CD1",),
        "CE2": ("CD2",),
        "CZ": ("CE1", "CE2"),
    },
    "PRO": {"CG": ("CB",), "CD": ("N",)},
    "SER": {"OG": ("CB",)},
    "THR": {"OG1": ("CB",), "CG2": ("CB",)},
    "TRP": {
        "CG": ("CB",),
        "CD1": ("CG",),
        "CD2": ("CG",),
        "NE1": ("CD1",),
        "CE2": ("CD2",),
        "CE3": ("CD2",),
        "CZ2": ("CE2",),
        "CZ3": ("CE3",),
        "CH2": ("CZ2", "CZ3"),
    },
    "TYR": {
        "CG": ("CB",),
        "CD1": ("CG",),
        "CD2": ("CG",),
        "CE1": ("CD1",),
        "CE2": ("CD2",),
        "CZ": ("CE1", "CE2"),
        "OH": ("CZ",),
    },
    "VAL": {"CG1": ("CB",), "CG2": ("CB",)},
}

PARENT_ATOMS: dict[str, dict[str, tuple[str, ...]]] = {
    residue: {
        **{
            atom: parents
            for atom, parents in _BACKBONE.items()
            if residue != "GLY" or atom != "CB"
        },
        **side_chain,
    }
    for residue, side_chain in _SIDE_CHAINS.items()
}

_BONDS = pd.DataFrame(
    [
        (residue, atom, parent)
        for residue, atoms in PARENT_ATOMS.items()
        for atom, parents in atoms.items()
        for parent in parents
    ],
    columns=["res_name", "atom_name", "parent_name"],
)


def b_factor_ratio_to_parent(table: SiteAtoms) -> npt.NDArray[np.float64]:
    """Return each atom's B-factor over the mean B-factor of its parents.

    The result has a value per row of the atom table, in row order. A row
    has a ratio when it is an atom of a standard amino acid deposited as an
    ATOM record, with a parent listed in PARENT_ATOMS that the residue
    holds, and when it and that parent have coordinates, an occupancy that
    is not zero, and a positive B-factor. Every other row is NaN. Where a
    residue holds one of an atom's two parents, the ratio is taken against
    that one.
    """
    count = len(table.b_factor)
    ratio = np.full(count, np.nan)
    measured = (
        table.usable
        & ~table.is_hetatm
        & np.isfinite(table.b_factor)
        & (table.b_factor > 0)
    )
    atoms = pd.DataFrame(
        {
            "row": np.arange(count),
            "residue": table.residue_index,
            "res_name": table.res_name,
            "atom_name": table.atom_name,
            "b_factor": table.b_factor,
        }
    )[measured].drop_duplicates(["residue", "atom_name"])
    pairs = atoms.merge(_BONDS, on=["res_name", "atom_name"])
    parents = atoms[["residue", "atom_name", "b_factor"]].rename(
        columns={"atom_name": "parent_name", "b_factor": "parent_b_factor"}
    )
    pairs = pairs.merge(parents, on=["residue", "parent_name"])
    if pairs.empty:
        return ratio
    means = pairs.groupby("row").agg(
        b_factor=("b_factor", "first"), parent=("parent_b_factor", "mean")
    )
    ratio[means.index.to_numpy()] = (means["b_factor"] / means["parent"]).to_numpy()
    return ratio


__all__ = ["PARENT_ATOMS", "b_factor_ratio_to_parent"]
