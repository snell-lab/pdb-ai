"""Finding each selenium and halogen atom and what it is bonded to.

compute_structure_selenium_halogens takes the atom table of one structure and
returns the selenium_halogen_sites table, the atom link column, and the
status the run report records. Nothing here judges a site.

The atom table gives the atoms, one conformer of each residue as the parser
chose it, so a site is measured in the conformer every other column
describes. A selenium or halogen atom is a row when it has coordinates,
whatever its occupancy, since an atom deposited at zero occupancy is one a depositor
modelled as lost.

Model by model, the atoms it is bonded to are those within the sum
of the two covalent radii plus BOND_TOLERANCE_A. A bonded atom has
coordinates and an occupancy that is not zero, and is not hydrogen,
deuterium, or a metal. Bonds are found among the atoms of the file only,
and across components as well as within one.

The other atoms of the atom's component, those that are not themselves
selenium or a halogen, give the second reference. Their occupancies
are averaged over every such atom with coordinates and their B-factors over
those whose occupancy is not zero.

Rows follow the atoms in file order, so a rerun gives the same table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gemmi
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ...core import site_key_dataframe
from ..site_geometry import (
    SiteAtoms,
    finite_mean,
    on_special_position,
    positive_ratio,
    read_site_atoms,
    read_unit_cell,
)
from .columns import BONDED_PREFIX, SELENIUM_HALOGEN_SITES_TABLE
from .constants import (
    BOND_TOLERANCE_A,
    PARENT_ION,
    PARENT_LIGAND,
    PARENT_POLYMER_RESIDUE,
    PARENT_SELENOMETHIONINE,
    SELENIUM_HALOGENS_STATUS_NONE,
    SELENIUM_HALOGENS_STATUS_OK,
    SELENOMETHIONINE_RESIDUE_NAME,
    SITE_ELEMENTS,
    SITE_STATUS_B_FACTOR_UNAVAILABLE,
    SITE_STATUS_NO_BONDED_ATOM,
    SITE_STATUS_OK,
)


@dataclass(frozen=True)
class SeleniumHalogenComputation:
    """The selenium_halogen_sites table of one structure, and what the run report records."""

    table: pd.DataFrame
    # The atoms' own site_id on their rows of the atom table, on its index.
    link: pd.Series
    status: str


def covalent_radius(element: str) -> float:
    """Return gemmi's covalent radius of an element in Å, or NaN when unknown."""
    radius = float(gemmi.Element(element).covalent_r)
    return radius if radius > 0 else math.nan


def _bonded(
    row: int,
    table: SiteAtoms,
    tree: cKDTree,
    tree_rows: np.ndarray,
    radii: dict[str, float],
    reach: float,
) -> list[tuple[float, int]]:
    """Return the atoms bonded to one such atom, as distance and row, nearest first."""
    own = radii[str(table.element[row])]
    found: list[tuple[float, int]] = []
    for index in tree.query_ball_point(table.xyz[row], reach):
        other = int(tree_rows[index])
        if other == row:
            continue
        radius = radii.get(str(table.element[other]), math.nan)
        distance = math.dist(table.xyz[row].tolist(), table.xyz[other].tolist())
        if 0.0 < distance <= own + radius + BOND_TOLERANCE_A:
            found.append((distance, other))
    return sorted(found)


def _parent_type(row: int, table: SiteAtoms, component_size: int) -> str:
    if component_size == 1:
        return PARENT_ION
    if table.res_name[row] == SELENOMETHIONINE_RESIDUE_NAME:
        return PARENT_SELENOMETHIONINE
    return PARENT_POLYMER_RESIDUE if table.label_seq[row] != "" else PARENT_LIGAND


def compute_structure_selenium_halogens(
    atoms: pd.DataFrame,
) -> SeleniumHalogenComputation:
    """Return the selenium_halogen_sites table, link column, and status of one structure.

    ``atoms`` is the structure's atom table. A structure without a selenium
    or halogen atom gives an empty table and the status
    no_sites.
    """
    table = read_site_atoms(atoms)
    link = pd.Series(pd.NA, index=atoms.index, dtype=object)
    is_site = np.isin(table.element, list(SITE_ELEMENTS))
    site_rows = np.flatnonzero(table.placed & is_site)
    if site_rows.size == 0:
        return SeleniumHalogenComputation(
            table=SELENIUM_HALOGEN_SITES_TABLE.empty_dataframe(),
            link=link,
            status=SELENIUM_HALOGENS_STATUS_NONE,
        )
    cell = read_unit_cell(atoms)
    partners = table.usable & ~table.is_metal
    radii = {
        element: covalent_radius(element)
        for element in dict.fromkeys(table.element[partners | is_site].tolist())
        if element
    }
    reach = (
        2 * max(radii[element] for element in SITE_ELEMENTS if element in radii)
        + BOND_TOLERANCE_A
    )
    trees: dict[object, tuple[cKDTree, np.ndarray]] = {}
    for model in dict.fromkeys(table.model[site_rows].tolist()):
        rows = np.flatnonzero(partners & (table.model == model))
        if rows.size:
            trees[model] = (cKDTree(table.xyz[rows]), rows)

    measures: list[dict[str, object]] = []
    nearest_rows: list[int | None] = []
    for row in site_rows.tolist():
        search = trees.get(table.model[row])
        bonded = (
            _bonded(row, table, search[0], search[1], radii, reach) if search else []
        )
        component = table.residues[int(table.residue_index[row])]
        component = component[table.placed[component]]
        others = component[~is_site[component]]
        distances = [distance for distance, _row in bonded]
        b_factor = float(table.b_factor[row])
        bonded_b = finite_mean(float(table.b_factor[other]) for _d, other in bonded)
        component_b = finite_mean(
            float(table.b_factor[other]) for other in others if table.usable[other]
        )
        ratio = positive_ratio(b_factor, bonded_b)
        if not bonded:
            status = SITE_STATUS_NO_BONDED_ATOM
        elif not math.isfinite(ratio):
            status = SITE_STATUS_B_FACTOR_UNAVAILABLE
        else:
            status = SITE_STATUS_OK
        special = on_special_position(cell, table.xyz[row].tolist())
        nearest_rows.append(bonded[0][1] if bonded else None)
        measures.append(
            {
                "element": str(table.element[row]),
                "parent_type": _parent_type(row, table, len(component)),
                "component_atom_count": len(component),
                "n_bonded_atoms": len(bonded),
                "bonded_element": (
                    str(table.element[bonded[0][1]]) if bonded else pd.NA
                ),
                "bond_distance_min_A": min(distances) if distances else math.nan,
                "bond_distance_max_A": max(distances) if distances else math.nan,
                "occupancy": float(table.occupancy[row]),
                "b_factor_A2": b_factor,
                "bonded_b_factor_mean_A2": bonded_b,
                "b_factor_ratio_to_bonded": ratio,
                "component_occupancy_mean": finite_mean(
                    float(value) for value in table.occupancy[others]
                ),
                "component_b_factor_mean_A2": component_b,
                "b_factor_ratio_to_component": positive_ratio(b_factor, component_b),
                "on_special_position": pd.NA if special is None else special,
                "site_status": status,
            }
        )

    keys = site_key_dataframe(atoms.iloc[site_rows]).reset_index(drop=True)
    with_bond = [index for index, row in enumerate(nearest_rows) if row is not None]
    bonded_keys = (
        site_key_dataframe(
            atoms.iloc[[row for row in nearest_rows if row is not None]],
            BONDED_PREFIX,
        )
        .set_axis(with_bond, axis=0)
        .reindex(range(len(site_rows)))
    )
    link.iloc[site_rows] = keys["site_id"].to_numpy(dtype=object)
    return SeleniumHalogenComputation(
        table=pd.concat([keys, bonded_keys, pd.DataFrame(measures)], axis=1),
        link=link,
        status=SELENIUM_HALOGENS_STATUS_OK,
    )


__all__ = [
    "SeleniumHalogenComputation",
    "compute_structure_selenium_halogens",
    "covalent_radius",
]
