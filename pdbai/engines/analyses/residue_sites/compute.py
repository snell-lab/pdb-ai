"""Finding the sensitive groups of each residue and measuring them.

compute_structure_residue_sites takes the atom table of one structure and
returns the residue_sites table, the two atom columns, and the counts the run
report records. Nothing here judges a site.

The atom table gives the atoms, one conformer of each residue as the parser
chose it, so a group is measured in the conformer every other column
describes. A residue is read when its atoms are ATOM records, which leaves
out a free amino acid bound as a ligand. An oxidized cysteine is the
exception. It is a modified residue, deposited as HETATM records with a
place in the polymer sequence, and is read as such. Hydrogen and deuterium atoms are
never read, since atoms are chosen by name, so a neutron model is read as an
X-ray model is.

Each class of SITE_CLASSES applies to the residues of its name, and the
C-terminal carboxylate to any standard amino acid that models an OXT. A
residue with the class's key atom is a row, whatever else of the group the
model holds. One without it is counted and has no row, since nothing would
name it.

Occupancies are summarized over every site atom with coordinates. B-factors
are summarized over those whose occupancy is not zero, since the B-factor of
an atom deposited at zero occupancy describes nothing. Metal atoms are those
of the file only, so a metal of a symmetry mate is not seen.

Rows follow the residues in file order and, within a residue, the classes in
the order SITE_CLASSES lists them, so a rerun gives the same table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ...core import site_key_dataframe
from ..metal_sites.constants import STANDARD_AMINO_ACIDS
from ..site_geometry import SiteAtoms, finite_mean, positive_ratio, read_site_atoms
from .columns import RESIDUE_SITES_TABLE
from .constants import (
    METAL_CONTACT_DISTANCE_A,
    RESIDUE_SITES_STATUS_NO_SITES,
    RESIDUE_SITES_STATUS_OK,
    SITE_CLASSES,
    SITE_STATUS_ATOMS_MISSING,
    SITE_STATUS_B_FACTOR_UNAVAILABLE,
    SITE_STATUS_OK,
    SITE_STATUS_ZERO_OCCUPANCY,
    ResidueSiteClass,
)
from .parents import b_factor_ratio_to_parent

_CLASSES_BY_RESIDUE: dict[str, tuple[ResidueSiteClass, ...]] = {
    name: tuple(site for site in SITE_CLASSES if site.residue == name)
    for name in {site.residue for site in SITE_CLASSES if site.residue is not None}
}
_ANY_RESIDUE_CLASSES = tuple(site for site in SITE_CLASSES if site.residue is None)


@dataclass(frozen=True)
class ResidueSiteComputation:
    """The residue_sites table of one structure, and what the run report records."""

    table: pd.DataFrame
    # The row's site_id on each modelled site atom, on the atom table's index.
    link: pd.Series
    # Each atom's B-factor over that of its parents, on the atom table's index.
    b_factor_ratio: pd.Series
    status: str
    # Residues of a class's name that lack its key atom, which no row holds.
    without_key_atom_count: int
    warnings: tuple[str, ...]


def _metal_trees(table: SiteAtoms) -> dict[object, cKDTree]:
    """Return a search tree over the usable metal atoms of each model."""
    metals = np.flatnonzero(table.usable & table.is_metal)
    trees: dict[object, cKDTree] = {}
    for model in dict.fromkeys(table.model[metals].tolist()):
        trees[model] = cKDTree(table.xyz[metals[table.model[metals] == model]])
    return trees


def _measure(
    site: ResidueSiteClass,
    named: dict[str, int],
    table: SiteAtoms,
    metals: cKDTree | None,
) -> tuple[dict[str, object], list[int]]:
    """Return one row's measures and the atom rows of its modelled site atoms."""
    site_rows = [named[name] for name in site.site_atoms if name in named]
    occupancy = table.occupancy[site_rows]
    measured = [row for row in site_rows if table.usable[row]]
    site_b = [float(value) for value in table.b_factor[measured]]
    reference_b = finite_mean(
        float(table.b_factor[named[name]])
        for name in site.reference_atoms
        if name in named and table.usable[named[name]]
    )
    site_b_mean = finite_mean(site_b)
    ratio = positive_ratio(site_b_mean, reference_b)
    attachments = [
        math.dist(table.xyz[named[first]].tolist(), table.xyz[named[second]].tolist())
        for first, second in site.attachments
        if first in named and second in named
    ]
    nearest_metal = math.nan
    if metals is not None and measured:
        nearest_metal = float(np.min(metals.query(table.xyz[measured])[0]))
    zero_count = int((np.isfinite(occupancy) & (occupancy <= 0.0)).sum())
    if len(site_rows) < len(site.site_atoms):
        status = SITE_STATUS_ATOMS_MISSING
    elif zero_count:
        status = SITE_STATUS_ZERO_OCCUPANCY
    elif not math.isfinite(ratio):
        status = SITE_STATUS_B_FACTOR_UNAVAILABLE
    else:
        status = SITE_STATUS_OK
    finite_b = [value for value in site_b if math.isfinite(value)]
    return (
        {
            "site_class": site.name,
            "is_control": site.control_for is not None,
            "control_for": site.control_for if site.control_for else pd.NA,
            "n_site_atoms_expected": len(site.site_atoms),
            "n_site_atoms_modelled": len(site_rows),
            "n_site_atoms_zero_occupancy": zero_count,
            "site_occupancy_min": (
                float(np.nanmin(occupancy))
                if np.isfinite(occupancy).any()
                else math.nan
            ),
            "site_occupancy_mean": finite_mean(float(value) for value in occupancy),
            "site_b_factor_mean_A2": site_b_mean,
            "site_b_factor_max_A2": max(finite_b) if finite_b else math.nan,
            "reference_b_factor_mean_A2": reference_b,
            "b_factor_ratio_to_reference": ratio,
            "attachment_distance_min_A": min(attachments) if attachments else math.nan,
            "attachment_distance_max_A": max(attachments) if attachments else math.nan,
            "has_alternate_conformer": any(
                table.altloc[row] != "" for row in site_rows
            ),
            "nearest_metal_distance_A": nearest_metal,
            "metal_bound": (
                math.isfinite(nearest_metal)
                and nearest_metal <= METAL_CONTACT_DISTANCE_A
            ),
            "site_status": status,
        },
        site_rows,
    )


def compute_structure_residue_sites(atoms: pd.DataFrame) -> ResidueSiteComputation:
    """Return the residue_sites table, link column, and report of one structure.

    ``atoms`` is the structure's atom table. A structure without a residue
    of any class gives an empty table and the status no_sites.
    """
    table = read_site_atoms(atoms)
    link = pd.Series(pd.NA, index=atoms.index, dtype=object)
    b_factor_ratio = pd.Series(b_factor_ratio_to_parent(table), index=atoms.index)
    metal_trees = _metal_trees(table)

    key_rows: list[int] = []
    measures: list[dict[str, object]] = []
    site_atom_rows: list[list[int]] = []
    without_key = 0
    for residue, rows in enumerate(table.residues):
        first = int(rows[0])
        name = str(table.res_name[first])
        classes = _CLASSES_BY_RESIDUE.get(name, ())
        if table.is_hetatm[first]:
            # A modified residue is HETATM records with a place in a polymer
            # sequence. A free amino acid bound as a ligand has no such place.
            if table.label_seq[first] == "":
                continue
            classes = tuple(site for site in classes if site.modified)
        elif name in STANDARD_AMINO_ACIDS:
            classes = (*classes, *_ANY_RESIDUE_CLASSES)
        if not classes:
            continue
        named = table.named(residue)
        for site in classes:
            if any(required not in named for required in site.requires):
                continue
            key_row = named.get(site.key_atom)
            if key_row is None:
                without_key += 1
                continue
            row, site_rows = _measure(
                site, named, table, metal_trees.get(table.model[first])
            )
            key_rows.append(key_row)
            measures.append(row)
            site_atom_rows.append(site_rows)

    warnings: list[str] = []
    if without_key:
        warnings.append(
            f"{without_key} residue(s) of a site class lack the class's key "
            "atom, so no row holds them."
        )
    if not key_rows:
        return ResidueSiteComputation(
            table=RESIDUE_SITES_TABLE.empty_dataframe(),
            link=link,
            b_factor_ratio=b_factor_ratio,
            status=RESIDUE_SITES_STATUS_NO_SITES,
            without_key_atom_count=without_key,
            warnings=tuple(warnings),
        )
    keys = site_key_dataframe(atoms.iloc[key_rows]).reset_index(drop=True)
    result = pd.concat([keys, pd.DataFrame(measures)], axis=1)
    # Two residues the file gives one identifier would share a site_id.
    repeated = result.duplicated(subset="site_id")
    if repeated.any():
        warnings.append(
            f"{int(repeated.sum())} group(s) share a site_id with an earlier "
            "group of the file, so no row holds them."
        )
    site_ids = result["site_id"].tolist()
    for position, site_rows in enumerate(site_atom_rows):
        if not repeated.iloc[position]:
            link.iloc[site_rows] = site_ids[position]
    return ResidueSiteComputation(
        table=result[~repeated].reset_index(drop=True),
        link=link,
        b_factor_ratio=b_factor_ratio,
        status=RESIDUE_SITES_STATUS_OK,
        without_key_atom_count=without_key,
        warnings=tuple(warnings),
    )


__all__ = ["ResidueSiteComputation", "compute_structure_residue_sites"]
