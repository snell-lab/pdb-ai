"""What the residue-site, heavy-atom, and cofactor engines share.

read_site_atoms turns the atom table of one structure into arrays, a value
per row in row order, and groups the rows by residue. The atom table keeps
one conformer of each residue, so an atom name is found at most once in a
residue, and each of the three engines measures a site in the conformer
every other column describes.

The rest is geometry on those arrays: a mean that passes over missing
values, a torsion angle, a least-squares plane, and the test for an atom on
a symmetry element. None of it judges a site.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import gemmi
import numpy as np
import numpy.typing as npt
import pandas as pd

from ...structure.cif_numbers import coerce_cif_numbers
from ...structure.residue_identity import clean_string_series
from .metal_sites.constants import HYDROGEN_ELEMENTS, METAL_ELEMENTS

# Two images of one atom closer than this in Å are one atom on a special
# position, the cut-off the metal-site engine uses for the same purpose.
SPECIAL_POSITION_CUTOFF_A = 0.8

_CELL_ITEMS = (
    "_cell.length_a",
    "_cell.length_b",
    "_cell.length_c",
    "_cell.angle_alpha",
    "_cell.angle_beta",
    "_cell.angle_gamma",
)
_SPACE_GROUP_ITEMS = ("_symmetry.space_group_name_H-M", "_space_group.name_H-M_alt")


@dataclass(frozen=True)
class SiteAtoms:
    """The atom table's columns as arrays, and its rows grouped by residue."""

    element: npt.NDArray[np.object_]
    res_name: npt.NDArray[np.object_]
    atom_name: npt.NDArray[np.object_]
    model: npt.NDArray[np.object_]
    altloc: npt.NDArray[np.object_]
    # The label sequence number, empty for a component outside a polymer.
    label_seq: npt.NDArray[np.object_]
    is_hetatm: npt.NDArray[np.bool_]
    xyz: npt.NDArray[np.float64]
    occupancy: npt.NDArray[np.float64]
    b_factor: npt.NDArray[np.float64]
    # Rows with coordinates that are not hydrogen, deuterium, or tritium.
    placed: npt.NDArray[np.bool_]
    # The placed rows whose occupancy is not zero or below. A missing
    # occupancy does not make a row unusable, since unknown is not zero.
    usable: npt.NDArray[np.bool_]
    is_metal: npt.NDArray[np.bool_]
    # The residue each row belongs to, as an index into residues.
    residue_index: npt.NDArray[np.intp]
    # The rows of each residue, in row order. A residue is one component of
    # one model, told apart by chain, residue number, insertion code, and
    # name.
    residues: tuple[npt.NDArray[np.intp], ...]

    def named(self, residue: int) -> dict[str, int]:
        """Return the placed rows of one residue by atom name, the first of each."""
        found: dict[str, int] = {}
        for row in self.residues[residue].tolist():
            if self.placed[row]:
                found.setdefault(str(self.atom_name[row]), row)
        return found


def _text(
    atoms: pd.DataFrame, column: str, *, upper: bool = False
) -> npt.NDArray[np.object_]:
    values = clean_string_series(atoms, column)
    if upper:
        values = values.str.upper()
    return values.to_numpy(dtype=object)


def _numbers(atoms: pd.DataFrame, column: str) -> npt.NDArray[np.float64]:
    if column not in atoms.columns:
        return np.full(len(atoms), np.nan)
    return pd.to_numeric(atoms[column], errors="coerce").to_numpy(
        dtype=np.float64, na_value=np.nan
    )


def read_site_atoms(atoms: pd.DataFrame) -> SiteAtoms:
    """Return the atom table's columns as arrays, with its rows grouped by residue."""
    element = _text(atoms, "element", upper=True)
    res_name = _text(atoms, "res_name", upper=True)
    model = _text(atoms, "model_num")
    xyz = np.column_stack([_numbers(atoms, name) for name in ("x", "y", "z")])
    occupancy = _numbers(atoms, "occupancy")
    placed = np.isfinite(xyz).all(axis=1) & ~np.isin(element, list(HYDROGEN_ELEMENTS))
    keys = pd.DataFrame(
        {
            "model": model,
            "chain": _text(atoms, "chain_id"),
            "number": _text(atoms, "res_num"),
            "insertion": _text(atoms, "_atom_site.pdbx_PDB_ins_code"),
            "name": res_name,
        }
    )
    residue_index = keys.groupby(list(keys.columns), sort=False).ngroup().to_numpy()
    order = np.argsort(residue_index, kind="stable")
    counts = np.bincount(residue_index) if len(residue_index) else np.array([], int)
    residues = tuple(np.split(order, np.cumsum(counts)[:-1])) if len(counts) else ()
    is_hetatm = (
        atoms["is_hetatm"].fillna(False).to_numpy(dtype=bool)
        if "is_hetatm" in atoms.columns
        else np.zeros(len(atoms), dtype=bool)
    )
    return SiteAtoms(
        element=element,
        res_name=res_name,
        atom_name=_text(atoms, "atom_name"),
        model=model,
        altloc=_text(atoms, "_atom_site.label_alt_id"),
        label_seq=_text(atoms, "_atom_site.label_seq_id"),
        is_hetatm=is_hetatm,
        xyz=xyz,
        occupancy=occupancy,
        b_factor=_numbers(atoms, "_atom_site.B_iso_or_equiv"),
        placed=placed,
        usable=placed & ~(np.isfinite(occupancy) & (occupancy <= 0.0)),
        is_metal=np.isin(element, list(METAL_ELEMENTS)),
        residue_index=residue_index.astype(np.intp),
        residues=residues,
    )


def finite_mean(values: Iterable[float]) -> float:
    """Return the mean of the values that are numbers, or NaN without any."""
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def positive_ratio(numerator: float, denominator: float) -> float:
    """Return numerator over denominator, or NaN unless both are positive numbers."""
    if (
        math.isfinite(numerator)
        and math.isfinite(denominator)
        and numerator > 0
        and denominator > 0
    ):
        return numerator / denominator
    return math.nan


def torsion_deg(points: npt.NDArray[np.float64]) -> float:
    """Return the torsion angle in degrees about the middle two of four points.

    The sign follows the IUPAC convention, so 0 is cis and 180 is trans, and
    the value lies in (−180, 180]. NaN stands for points that do not define
    the angle, as when three of them lie on one line.
    """
    first = points[1] - points[0]
    axis = points[2] - points[1]
    last = points[3] - points[2]
    normal_1 = np.cross(first, axis)
    normal_2 = np.cross(axis, last)
    length = math.sqrt(float(np.dot(axis, axis)))
    if (
        length == 0
        or float(np.dot(normal_1, normal_1)) == 0
        or float(np.dot(normal_2, normal_2)) == 0
    ):
        return math.nan
    angle = math.degrees(
        math.atan2(
            float(np.dot(np.cross(normal_1, normal_2), axis)) / length,
            float(np.dot(normal_1, normal_2)),
        )
    )
    return 180.0 if angle == -180.0 else angle


@dataclass(frozen=True)
class Plane:
    """The least-squares plane through a set of points."""

    centroid: npt.NDArray[np.float64]
    normal: npt.NDArray[np.float64]
    # The root-mean-square distance in Å of the fitted points from the plane.
    rmsd: float

    def distance(self, point: npt.NDArray[np.float64]) -> float:
        """Return the unsigned distance in Å of a point from the plane."""
        return abs(float(np.dot(point - self.centroid, self.normal)))


def fit_plane(points: npt.NDArray[np.float64]) -> Plane | None:
    """Return the least-squares plane through the points, or None for fewer than three."""
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    # The plane's normal is the direction the points spread least along.
    spreads, directions = np.linalg.eigh(np.cov((points - centroid).T, bias=True))
    return Plane(
        centroid=centroid,
        normal=directions[:, 0],
        rmsd=math.sqrt(max(float(spreads[0]), 0.0)),
    )


def plane_angle_deg(first: Plane, second: Plane) -> float:
    """Return the angle in degrees between two planes, from 0 to 90."""
    cosine = min(1.0, abs(float(np.dot(first.normal, second.normal))))
    return math.degrees(math.acos(cosine))


def read_unit_cell(atoms: pd.DataFrame) -> gemmi.UnitCell | None:
    """Return the entry's unit cell with its symmetry images, or None.

    The cell and the space group are read from the first row of the atom
    table, which repeats the file's _cell and _symmetry items on every row.
    None stands for a table without those items, a cell that is a
    placeholder, and a space group gemmi does not know.
    """
    if atoms.empty or any(item not in atoms.columns for item in _CELL_ITEMS):
        return None
    numbers = coerce_cif_numbers(
        pd.Series([atoms[item].iloc[0] for item in _CELL_ITEMS], dtype=object)
    ).to_numpy(dtype=np.float64, na_value=np.nan)
    if not (np.isfinite(numbers) & (numbers > 0)).all():
        return None
    cell = gemmi.UnitCell(*(float(value) for value in numbers))
    if not cell.is_crystal():
        return None
    name = next(
        (
            text
            for item in _SPACE_GROUP_ITEMS
            if item in atoms.columns
            and (text := str(clean_string_series(atoms.iloc[:1], item).iloc[0]))
        ),
        None,
    )
    if name is None:
        return None
    # gemmi returns None for an unknown name, although its stubs say not.
    space_group = cast(
        "gemmi.SpaceGroup | None",
        gemmi.find_spacegroup_by_name(name, cell.alpha, cell.gamma),
    )
    if space_group is None:
        return None
    structure = gemmi.Structure()
    structure.cell = cell
    structure.spacegroup_hm = space_group.xhm()
    structure.setup_cell_images()
    return structure.cell


def on_special_position(
    cell: gemmi.UnitCell | None, position: Sequence[float]
) -> bool | None:
    """Return whether a symmetry image of a position lies on top of it, or None.

    True when an image lies within SPECIAL_POSITION_CUTOFF_A, so the atom
    sits on a symmetry element and its occupancy is expected to be a
    fraction. None without a usable cell and space group.
    """
    if cell is None:
        return None
    return (
        cell.is_special_position(
            gemmi.Position(*(float(value) for value in position)),
            SPECIAL_POSITION_CUTOFF_A,
        )
        > 0
    )


__all__ = [
    "SPECIAL_POSITION_CUTOFF_A",
    "Plane",
    "SiteAtoms",
    "finite_mean",
    "fit_plane",
    "on_special_position",
    "plane_angle_deg",
    "positive_ratio",
    "read_site_atoms",
    "read_unit_cell",
    "torsion_deg",
]
