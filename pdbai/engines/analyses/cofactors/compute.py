"""Recognizing each cofactor and measuring its reactive group.

compute_structure_cofactors takes the atom table of one structure and
returns the cofactor_sites table, the atom link column, and the status the
run report records. Nothing here judges a site.

The atom table gives the atoms, one conformer of each residue as the parser
chose it, so a cofactor is measured in the conformer every other column
describes. Hydrogen and deuterium atoms are never read, so a neutron model
is read as an X-ray model is.

A component is of a class when it holds every atom name of the class's
signature, is one of the components the class is limited to where it names
any, and has a place in a polymer sequence where the class asks for one. The
classes are tried in the order COFACTOR_CLASSES lists them, and the first
that fits is taken. The 20 standard amino acids are never tried. Every
signature holds its class's key atom, so a component of a class is a row.

Occupancies are summarized over every group atom with coordinates. B-factors
are summarized over those whose occupancy is not zero. The protein atom a
cofactor is linked to is looked for among the atoms of the file only.

Rows follow the components in file order, so a rerun gives the same table.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ...core import site_key_dataframe
from ..metal_sites.constants import STANDARD_AMINO_ACIDS
from ..site_geometry import (
    SiteAtoms,
    finite_mean,
    fit_plane,
    plane_angle_deg,
    positive_ratio,
    read_site_atoms,
    torsion_deg,
)
from .columns import COFACTOR_SITES_TABLE
from .constants import (
    COFACTOR_CLASSES,
    COFACTOR_STATUS_ATOMS_MISSING,
    COFACTOR_STATUS_B_FACTOR_UNAVAILABLE,
    COFACTOR_STATUS_OK,
    COFACTORS_STATUS_NONE,
    COFACTORS_STATUS_OK,
    LINK_SEARCH_RADIUS_A,
    TORSION_COLUMNS,
    AtomNames,
    CofactorClass,
)

# The fewest atoms a signature holds, below which no component is tried.
_SMALLEST_SIGNATURE = min(len(cofactor.signature) for cofactor in COFACTOR_CLASSES)


@dataclass(frozen=True)
class CofactorComputation:
    """The cofactor_sites table of one structure, and what the run report records."""

    table: pd.DataFrame
    # The row's site_id on each modelled group atom, on the atom table's index.
    link: pd.Series
    status: str


def classify_component(
    res_name: str, atom_names: Iterable[str], *, in_polymer: bool
) -> CofactorClass | None:
    """Return the class a component is of, by its name and atom names, or None."""
    names = set(atom_names)
    for cofactor in COFACTOR_CLASSES:
        if cofactor.components is not None and res_name not in cofactor.components:
            continue
        if cofactor.polymer_only and not in_polymer:
            continue
        if cofactor.signature <= names:
            return cofactor
    return None


def _row_of(names: AtomNames, named: dict[str, int]) -> int | None:
    """Return the row of an atom by the first of its names the residue holds."""
    return next((named[name] for name in names if name in named), None)


def _rows_of(atoms: Iterable[AtomNames], named: dict[str, int]) -> list[int] | None:
    """Return the rows of every atom listed, or None when one is not modelled."""
    rows = [_row_of(names, named) for names in atoms]
    return (
        None if any(row is None for row in rows) else [r for r in rows if r is not None]
    )


def _link_distance(
    cofactor: CofactorClass, named: dict[str, int], table: SiteAtoms, model: object
) -> float:
    """Return the distance from the linking atom to its nearest protein partner."""
    if cofactor.link is None:
        return math.nan
    atom, residue, partner = cofactor.link
    if atom not in named:
        return math.nan
    partners = np.flatnonzero(
        table.usable
        & (table.model == model)
        & (table.res_name == residue)
        & (table.atom_name == partner)
    )
    if partners.size == 0:
        return math.nan
    offsets = table.xyz[partners] - table.xyz[named[atom]]
    nearest = float(np.sqrt(np.square(offsets).sum(axis=1)).min())
    return nearest if nearest <= LINK_SEARCH_RADIUS_A else math.nan


def _metal_measures(
    cofactor: CofactorClass,
    named: dict[str, int],
    component: np.ndarray,
    table: SiteAtoms,
) -> tuple[int | None, float, float]:
    """Return the central metal's row, its out-of-plane distance, and mean M–N."""
    if cofactor.metal_plane is None:
        return None, math.nan, math.nan
    nitrogens = [named[name] for name in cofactor.metal_plane if name in named]
    metals = component[table.is_metal[component] & table.placed[component]]
    if metals.size == 0 or not nitrogens:
        return None, math.nan, math.nan
    positions = table.xyz[nitrogens]
    separations = np.sqrt(
        np.square(table.xyz[metals] - positions.mean(axis=0)).sum(axis=1)
    )
    metal = int(metals[int(np.argmin(separations))])
    plane = (
        fit_plane(positions) if len(nitrogens) == len(cofactor.metal_plane) else None
    )
    return (
        metal,
        plane.distance(table.xyz[metal]) if plane is not None else math.nan,
        float(np.sqrt(np.square(positions - table.xyz[metal]).sum(axis=1)).mean()),
    )


def _measure(
    cofactor: CofactorClass,
    named: dict[str, int],
    component: np.ndarray,
    table: SiteAtoms,
) -> tuple[dict[str, object], list[int], int | None]:
    """Return one row's measures, its group atom rows, and its central metal row."""
    group = [
        row
        for row in (_row_of(names, named) for names in cofactor.group)
        if row is not None
    ]
    occupancy = table.occupancy[group]
    in_group = set(group)
    group_b = finite_mean(
        float(table.b_factor[row]) for row in group if table.usable[row]
    )
    rest_b = finite_mean(
        float(table.b_factor[row])
        for row in component.tolist()
        if row not in in_group and table.usable[row]
    )
    group_plane = fit_plane(table.xyz[group]) if len(group) >= 4 else None
    butterfly = math.nan
    if cofactor.fold_planes is not None:
        rings = [_rows_of(ring, named) for ring in cofactor.fold_planes]
        if rings[0] is not None and rings[1] is not None:
            first = fit_plane(table.xyz[rings[0]])
            second = fit_plane(table.xyz[rings[1]])
            if first is not None and second is not None:
                butterfly = plane_angle_deg(first, second)
    torsions: dict[str, float] = dict.fromkeys(TORSION_COLUMNS, math.nan)
    for column, atoms in cofactor.torsions:
        rows = _rows_of(atoms, named)
        if rows is not None:
            torsions[column] = torsion_deg(table.xyz[rows])
    metal, out_of_plane, metal_nitrogen = _metal_measures(
        cofactor, named, component, table
    )
    if len(group) < len(cofactor.group):
        status = COFACTOR_STATUS_ATOMS_MISSING
    elif not math.isfinite(group_b):
        status = COFACTOR_STATUS_B_FACTOR_UNAVAILABLE
    else:
        status = COFACTOR_STATUS_OK
    first_row = int(component[0])
    return (
        {
            "cofactor_class": cofactor.name,
            "component_atom_count": len(component),
            "n_group_atoms_expected": len(cofactor.group),
            "n_group_atoms_modelled": len(group),
            "group_occupancy_min": (
                float(np.nanmin(occupancy))
                if np.isfinite(occupancy).any()
                else math.nan
            ),
            "group_occupancy_mean": finite_mean(float(value) for value in occupancy),
            "group_b_factor_mean_A2": group_b,
            "rest_b_factor_mean_A2": rest_b,
            "b_factor_ratio_to_rest": positive_ratio(group_b, rest_b),
            "group_planarity_rmsd_A": (
                group_plane.rmsd if group_plane is not None else math.nan
            ),
            "flavin_butterfly_angle_deg": butterfly,
            **torsions,
            "protein_link_distance_A": _link_distance(
                cofactor, named, table, table.model[first_row]
            ),
            "central_metal_element": (
                str(table.element[metal]) if metal is not None else pd.NA
            ),
            "metal_out_of_plane_A": out_of_plane,
            "metal_nitrogen_distance_mean_A": metal_nitrogen,
            "cofactor_status": status,
        },
        group,
        metal,
    )


def compute_structure_cofactors(atoms: pd.DataFrame) -> CofactorComputation:
    """Return the cofactor_sites table, link column, and report of one structure.

    ``atoms`` is the structure's atom table. A structure without a
    recognized cofactor gives an empty table and the status no_cofactors.
    """
    table = read_site_atoms(atoms)
    link = pd.Series(pd.NA, index=atoms.index, dtype=object)

    key_rows: list[int] = []
    measures: list[dict[str, object]] = []
    group_rows: list[list[int]] = []
    metal_rows: list[int | None] = []
    for residue, rows in enumerate(table.residues):
        first = int(rows[0])
        if (
            len(rows) < _SMALLEST_SIGNATURE
            or table.res_name[first] in STANDARD_AMINO_ACIDS
        ):
            continue
        named = table.named(residue)
        cofactor = classify_component(
            str(table.res_name[first]), named, in_polymer=table.label_seq[first] != ""
        )
        if cofactor is None:
            continue
        row, group, metal = _measure(cofactor, named, rows[table.placed[rows]], table)
        key_rows.append(named[cofactor.key_atom])
        measures.append(row)
        group_rows.append(group)
        metal_rows.append(metal)

    if not key_rows:
        return CofactorComputation(
            table=COFACTOR_SITES_TABLE.empty_dataframe(),
            link=link,
            status=COFACTORS_STATUS_NONE,
        )
    keys = site_key_dataframe(atoms.iloc[key_rows]).reset_index(drop=True)
    result = pd.concat([keys, pd.DataFrame(measures)], axis=1)
    with_metal = [index for index, row in enumerate(metal_rows) if row is not None]
    metal_ids = site_key_dataframe(
        atoms.iloc[[row for row in metal_rows if row is not None]]
    )["site_id"].set_axis(with_metal)
    result["central_metal_site_id"] = metal_ids.reindex(range(len(key_rows)))
    for position, group in enumerate(group_rows):
        link.iloc[group] = result["site_id"].iloc[position]
    return CofactorComputation(table=result, link=link, status=COFACTORS_STATUS_OK)


__all__ = ["CofactorComputation", "classify_component", "compute_structure_cofactors"]
