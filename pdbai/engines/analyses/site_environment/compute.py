"""Measuring the surroundings of each radiation-sensitive group.

compute_structure_site_environment takes the atom table of one structure and
returns the site_environment table, the atom depth column, and the status
the run report records. Nothing here judges a site.

The sites are those of the residue_sites table, found by that engine's own
function, so a row here has the site_id of its row there. A group's atoms
are its site atoms with coordinates and an occupancy that is not zero, in
the conformer the atom table keeps. A row without any is kept, with every
measure missing.

Each model is read on its own, and only its own atoms are seen, except for
the crystal contacts. The protein atoms of a model are the heavy atoms of
its standard amino acids and of the modified residues of its polymers, with
an occupancy that is not zero. They alone give the packing counts, the
solvent-accessible area, and the depth, so those three are the same for a
model with or without its ligands and waters, and can be taken from a
predicted model. Waters are the oxygens of WATER_RESIDUE_NAMES. Atoms of
the group's own residue are never its partners.

The crystal contacts come from the images of the model under the space
group of the entry and the translations of its lattice, built here from the
cell gemmi reads. The image of the model under the identity with no
translation is the model itself and is left out. The images include those
that make up the biological assembly, so a contact may be an interface of
the assembly as well as one of the crystal.

Rows follow the residue_sites table, so a rerun gives the same table.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import gemmi
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial import cKDTree

from ..disulfides.constants import DISULFIDE_SEARCH_CUTOFF_A
from ..metal_sites.constants import STANDARD_AMINO_ACIDS
from ..residue_sites.compute import compute_structure_residue_sites
from ..sasa.classification import protein_radii
from ..sasa.sphere import compute_sasa
from ..site_geometry import SiteAtoms, read_site_atoms, read_unit_cell
from .columns import KEY_COLUMNS, SITE_ENVIRONMENT_TABLE
from .constants import (
    AROMATIC_RINGS,
    CARBOXYLATE_CLASSES,
    CATIONIC_NITROGENS,
    CONTACT_COUNT_RADIUS_A,
    CONTACTS_STATUS_NO_UNIT_CELL,
    CONTACTS_STATUS_NONE_IN_RANGE,
    CONTACTS_STATUS_OK,
    CYS_SULFUR_CLASS,
    ENVIRONMENT_STATUS_NO_GROUP_ATOMS,
    ENVIRONMENT_STATUS_OK,
    EXPOSED_ATOM_AREA_A2,
    HBOND_DISTANCE_A,
    PACKING_INNER_RADIUS_A,
    PACKING_OUTER_RADIUS_A,
    SALT_BRIDGE_DISTANCE_A,
    SITE_ENVIRONMENT_STATUS_NO_SITES,
    SITE_ENVIRONMENT_STATUS_OK,
    SPHERE_POINT_COUNT,
    SULFUR_HBOND_DISTANCE_A,
    SYMMETRY_SEARCH_RADIUS_A,
    WATER_RESIDUE_NAMES,
)

_POLAR_ELEMENTS = ("N", "O", "S")
_PARTNER_ELEMENTS = ("N", "O")


@dataclass(frozen=True)
class SiteEnvironmentComputation:
    """The site_environment table of one structure, and what the run report records."""

    table: pd.DataFrame
    # Each protein atom's depth in Å, on the atom table's index.
    depth: pd.Series
    status: str


def _empty_rows() -> npt.NDArray[np.intp]:
    return np.empty(0, dtype=np.intp)


def _nearest(
    points: npt.NDArray[np.float64], targets: npt.NDArray[np.float64]
) -> tuple[float, int]:
    """Return the least distance from any point to any target, and that target."""
    if len(points) == 0 or len(targets) == 0:
        return math.nan, -1
    distances = np.linalg.norm(points[:, None, :] - targets[None, :, :], axis=2)
    position = int(np.argmin(distances))
    return float(distances.flat[position]), position % len(targets)


def symmetry_mates(
    cell: gemmi.UnitCell, xyz: npt.NDArray[np.float64], reach: float
) -> npt.NDArray[np.float64]:
    """Return the positions of symmetry-mate atoms within ``reach`` Å of the model's box.

    ``xyz`` holds the model's atoms. Every image of them under the space
    group's operations and the lattice's translations is tried, save the
    model itself, and the atoms of each that fall within the model's box,
    widened by ``reach`` on every side, are returned.
    """
    if len(xyz) == 0:
        return np.empty((0, 3))
    orth = np.array(cell.orth.mat.tolist())
    frac = np.array(cell.frac.mat.tolist())
    low, high = xyz.min(axis=0) - reach, xyz.max(axis=0) + reach
    corners = np.array(list(itertools.product(*zip(low, high, strict=True))))
    corners_frac = corners @ frac.T
    box_low, box_high = corners_frac.min(axis=0), corners_frac.max(axis=0)
    fractional = xyz @ frac.T
    operations = [(np.eye(3), np.zeros(3))] + [
        (np.array(image.mat.tolist()), np.array(image.vec.tolist()))
        for image in cell.images
    ]
    found: list[npt.NDArray[np.float64]] = []
    for index, (rotation, translation) in enumerate(operations):
        image = fractional @ rotation.T + translation
        image_low, image_high = image.min(axis=0), image.max(axis=0)
        # Only a whole-cell shift that brings the image's box over the
        # model's can bring one of its atoms near the model.
        ranges = [
            range(
                math.ceil(box_low[axis] - image_high[axis]),
                math.floor(box_high[axis] - image_low[axis]) + 1,
            )
            for axis in range(3)
        ]
        for shift in itertools.product(*ranges):
            if index == 0 and shift == (0, 0, 0):
                continue
            positions = (image + np.array(shift, dtype=float)) @ orth.T
            inside = ((positions >= low) & (positions <= high)).all(axis=1)
            if inside.any():
                found.append(positions[inside])
    return np.concatenate(found) if found else np.empty((0, 3))


class _Model:
    """The atoms of one model, sorted into what a group's measures look for."""

    def __init__(
        self, table: SiteAtoms, model: object, cell: gemmi.UnitCell | None
    ) -> None:
        self.table = table
        in_model = (table.model == model) & table.usable
        is_water = np.isin(table.res_name, list(WATER_RESIDUE_NAMES))
        in_polymer = (
            ~table.is_hetatm & np.isin(table.res_name, list(STANDARD_AMINO_ACIDS))
        ) | (table.is_hetatm & (table.label_seq != ""))
        protein = in_model & in_polymer & ~is_water & ~table.is_metal
        self.protein = np.flatnonzero(protein)
        self.water = np.flatnonzero(in_model & is_water & (table.element == "O"))
        self.metal = np.flatnonzero(in_model & table.is_metal)
        self.ligand = np.flatnonzero(
            in_model & ~in_polymer & ~is_water & ~table.is_metal
        )
        self.partner = self.protein[
            np.isin(table.element[self.protein], _PARTNER_ELEMENTS)
        ]
        self.cation = np.array(
            [
                row
                for row in self.protein.tolist()
                if table.atom_name[row]
                in CATIONIC_NITROGENS.get(str(table.res_name[row]), ())
            ],
            dtype=np.intp,
        )
        self.protein_tree = (
            cKDTree(table.xyz[self.protein]) if len(self.protein) else None
        )
        self.partner_tree = (
            cKDTree(table.xyz[self.partner]) if len(self.partner) else None
        )
        self.water_tree = cKDTree(table.xyz[self.water]) if len(self.water) else None
        self.area = np.full(len(table.xyz), math.nan)
        self.depth = np.full(len(table.xyz), math.nan)
        if len(self.protein):
            self.area[self.protein] = compute_sasa(
                table.xyz[self.protein],
                protein_radii(pd.Series(table.atom_name[self.protein]).str.upper()),
                SPHERE_POINT_COUNT,
            )
            exposed = self.protein[self.area[self.protein] >= EXPOSED_ATOM_AREA_A2]
            if len(exposed):
                self.depth[self.protein] = cKDTree(table.xyz[exposed]).query(
                    table.xyz[self.protein]
                )[0]
        self._find_disulfides()
        self._find_rings()
        self.mate_tree: cKDTree | None = None
        if cell is not None:
            mates = symmetry_mates(
                cell,
                table.xyz[np.concatenate([self.protein, self.metal, self.ligand])],
                SYMMETRY_SEARCH_RADIUS_A,
            )
            self.mate_tree = cKDTree(mates if len(mates) else np.empty((0, 3)))

    def _find_disulfides(self) -> None:
        """Find the cysteine sulfurs bonded to the sulfur of another residue."""
        table = self.table
        sulfurs = self.protein[
            (table.res_name[self.protein] == "CYS")
            & (table.atom_name[self.protein] == "SG")
        ]
        self.partners_of: dict[int, set[int]] = {}
        bonded: set[int] = set()
        if len(sulfurs) > 1:
            pairs = cKDTree(table.xyz[sulfurs]).query_pairs(DISULFIDE_SEARCH_CUTOFF_A)
            for first, second in pairs:
                one, other = int(sulfurs[first]), int(sulfurs[second])
                residues = (
                    int(table.residue_index[one]),
                    int(table.residue_index[other]),
                )
                if residues[0] == residues[1]:
                    continue
                bonded.update((one, other))
                self.partners_of.setdefault(residues[0], set()).add(residues[1])
                self.partners_of.setdefault(residues[1], set()).add(residues[0])
        self.disulfide_sulfur = np.array(sorted(bonded), dtype=np.intp)

    def _find_rings(self) -> None:
        """Find the centroid of every aromatic ring the model holds whole."""
        table = self.table
        centroids: list[npt.NDArray[np.float64]] = []
        owners: list[int] = []
        for residue in dict.fromkeys(
            int(row) for row in table.residue_index[self.protein]
        ):
            rows: list[int] = table.residues[residue].tolist()
            name = str(table.res_name[rows[0]])
            if name not in AROMATIC_RINGS:
                continue
            named = {
                str(table.atom_name[row]): row for row in rows if table.usable[row]
            }
            for ring in AROMATIC_RINGS[name]:
                if all(atom in named for atom in ring):
                    centroids.append(
                        table.xyz[[named[atom] for atom in ring]].mean(axis=0)
                    )
                    owners.append(residue)
        self.ring_centroid = np.array(centroids).reshape(-1, 3)
        self.ring_residue = np.array(owners, dtype=np.intp)

    def isolated_area(self, group: npt.NDArray[np.intp], residue: int) -> float:
        """Return the group's solvent-accessible area with its own residue alone."""
        rows = np.intersect1d(self.table.residues[residue], self.protein)
        targets = np.searchsorted(rows, group)
        areas = compute_sasa(
            self.table.xyz[rows],
            protein_radii(pd.Series(self.table.atom_name[rows]).str.upper()),
            SPHERE_POINT_COUNT,
            targets,
        )
        return float(areas.sum())


def _distinct_near(
    tree: cKDTree | None,
    rows: npt.NDArray[np.intp],
    points: npt.NDArray[np.float64],
    radii: list[float],
) -> set[int]:
    """Return the rows of the tree within each point's radius of that point."""
    if tree is None or len(points) == 0:
        return set()
    found: set[int] = set()
    for point, radius in zip(points, radii, strict=True):
        found.update(int(rows[index]) for index in tree.query_ball_point(point, radius))
    return found


def _measure(
    site_class: str, group: npt.NDArray[np.intp], model: _Model
) -> dict[str, object]:
    """Return the environment measures of one group."""
    table = model.table
    xyz = table.xyz[group]
    own = int(table.residue_index[group[0]])
    centroid = xyz.mean(axis=0)
    group_rows = set(group.tolist())
    residue_of = table.residue_index

    polar = group[np.isin(table.element[group], _POLAR_ELEMENTS)]
    radii = [
        SULFUR_HBOND_DISTANCE_A if table.element[row] == "S" else HBOND_DISTANCE_A
        for row in polar.tolist()
    ]
    protein_partners = {
        row
        for row in _distinct_near(
            model.partner_tree, model.partner, table.xyz[polar], radii
        )
        if residue_of[row] != own
    }
    water_partners = _distinct_near(
        model.water_tree, model.water, table.xyz[polar], radii
    )
    has_water = len(model.water) > 0

    salt_bridges: object = pd.NA
    if site_class in CARBOXYLATE_CLASSES:
        oxygens = group[table.element[group] == "O"]
        near = [
            row
            for row in model.cation.tolist()
            if residue_of[row] != own
            and len(oxygens)
            and float(
                np.min(np.linalg.norm(table.xyz[oxygens] - table.xyz[row], axis=1))
            )
            <= SALT_BRIDGE_DISTANCE_A
        ]
        salt_bridges = len({int(residue_of[row]) for row in near})
    cations = model.cation[residue_of[model.cation] != own]
    excluded = {own, *model.partners_of.get(own, set())}
    sulfurs = np.array(
        [
            row
            for row in model.disulfide_sulfur.tolist()
            if residue_of[row] not in excluded
        ],
        dtype=np.intp,
    )
    rings = model.ring_residue != own
    ring_distance, ring = _nearest(xyz, model.ring_centroid[rings])

    packing: dict[str, int] = {}
    for name, radius in (
        ("heavy_atom_count_6A", PACKING_INNER_RADIUS_A),
        ("heavy_atom_count_10A", PACKING_OUTER_RADIUS_A),
    ):
        found = (
            model.protein[model.protein_tree.query_ball_point(centroid, radius)]
            if model.protein_tree is not None
            else _empty_rows()
        )
        packing[name] = len(set(found.tolist()) - group_rows)

    site_area = float(np.nansum(model.area[group]))
    isolated = model.isolated_area(group, own)
    waters_near = _distinct_near(
        model.water_tree, model.water, xyz, [CONTACT_COUNT_RADIUS_A] * len(group)
    )

    contacts_status = CONTACTS_STATUS_NO_UNIT_CELL
    mate_distance = math.nan
    mate_count: object = pd.NA
    if model.mate_tree is not None:
        mate_count = 0
        if model.mate_tree.n:
            distances = model.mate_tree.query(
                xyz, distance_upper_bound=SYMMETRY_SEARCH_RADIUS_A
            )[0]
            mate_count = len(
                _distinct_near(
                    model.mate_tree,
                    np.arange(model.mate_tree.n),
                    xyz,
                    [CONTACT_COUNT_RADIUS_A] * len(group),
                )
            )
            if np.isfinite(distances).any():
                mate_distance = float(np.min(distances))
        contacts_status = (
            CONTACTS_STATUS_OK
            if math.isfinite(mate_distance)
            else CONTACTS_STATUS_NONE_IN_RANGE
        )

    return {
        "n_group_atoms": len(group),
        "hbond_protein_count": len(protein_partners) if len(polar) else pd.NA,
        "hbond_water_count": (
            len(water_partners) if len(polar) and has_water else pd.NA
        ),
        "salt_bridge_partner_count": salt_bridges,
        "nearest_cation_distance_A": _nearest(xyz, table.xyz[cations])[0],
        "in_disulfide": (
            own in model.partners_of if site_class == CYS_SULFUR_CLASS else pd.NA
        ),
        "nearest_disulfide_sulfur_distance_A": _nearest(xyz, table.xyz[sulfurs])[0],
        "nearest_metal_distance_A": _nearest(xyz, table.xyz[model.metal])[0],
        "nearest_aromatic_ring_distance_A": ring_distance,
        "nearest_aromatic_res_name": (
            str(table.res_name[table.residues[model.ring_residue[rings][ring]][0]])
            if ring >= 0
            else pd.NA
        ),
        "nearest_ligand_distance_A": _nearest(xyz, table.xyz[model.ligand])[0],
        **packing,
        "site_sasa_A2": site_area,
        "site_relative_sasa": site_area / isolated if isolated > 0 else math.nan,
        "site_depth_A": float(np.mean(model.depth[group])),
        "water_count_6A": len(waters_near) if has_water else pd.NA,
        "nearest_symmetry_mate_distance_A": mate_distance,
        "symmetry_mate_atom_count_6A": mate_count,
        "crystal_contacts_status": contacts_status,
        "environment_status": ENVIRONMENT_STATUS_OK,
    }


def compute_structure_site_environment(
    atoms: pd.DataFrame,
) -> SiteEnvironmentComputation:
    """Return the site_environment table, depth column, and report of one structure.

    ``atoms`` is the structure's atom table. A structure without a group of
    any residue-site class gives an empty table and the status no_sites.
    """
    residue_sites = compute_structure_residue_sites(atoms)
    sites = residue_sites.table
    table = read_site_atoms(atoms)
    cell = read_unit_cell(atoms)
    # Every model is read, with or without a site, so that every protein
    # atom has a depth.
    models = {
        model: _Model(table, model, cell)
        for model in dict.fromkeys(table.model[table.usable].tolist())
    }
    depth = np.full(len(atoms), math.nan)
    for model in models.values():
        depth[model.protein] = model.depth[model.protein]

    # The residue-site engine links every modelled site atom to its row.
    group_of: dict[str, list[int]] = {}
    for row, site_id in enumerate(residue_sites.link.tolist()):
        if isinstance(site_id, str) and table.usable[row]:
            group_of.setdefault(site_id, []).append(row)

    measures: list[dict[str, object]] = []
    for site_id, site_class in zip(sites["site_id"], sites["site_class"], strict=True):
        group = np.array(group_of.get(str(site_id), []), dtype=np.intp)
        if not len(group):
            measures.append(
                {
                    "environment_status": ENVIRONMENT_STATUS_NO_GROUP_ATOMS,
                    "crystal_contacts_status": (
                        CONTACTS_STATUS_NO_UNIT_CELL if cell is None else pd.NA
                    ),
                }
            )
            continue
        model = models[table.model[group[0]]]
        measures.append(_measure(str(site_class), group, model))

    if sites.empty:
        return SiteEnvironmentComputation(
            table=SITE_ENVIRONMENT_TABLE.empty_dataframe(),
            depth=pd.Series(depth, index=atoms.index),
            status=SITE_ENVIRONMENT_STATUS_NO_SITES,
        )
    result = pd.concat(
        [
            sites[list(KEY_COLUMNS)].reset_index(drop=True),
            pd.DataFrame(
                measures,
                columns=SITE_ENVIRONMENT_TABLE.column_names[len(KEY_COLUMNS) :],
            ),
        ],
        axis=1,
    )
    return SiteEnvironmentComputation(
        table=result,
        depth=pd.Series(depth, index=atoms.index),
        status=SITE_ENVIRONMENT_STATUS_OK,
    )


__all__ = [
    "SiteEnvironmentComputation",
    "compute_structure_site_environment",
    "symmetry_mates",
]
