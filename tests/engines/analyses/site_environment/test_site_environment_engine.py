from __future__ import annotations

import dataclasses
import itertools
import math
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd
import pytest
from site_cif import SiteAtom, parse_site_atoms, residue, write_site_cif

from pdbai.engines.analyses.residue_sites.compute import (
    compute_structure_residue_sites,
)
from pdbai.engines.analyses.site_environment import SITE_ENVIRONMENT_ENGINE
from pdbai.engines.analyses.site_environment.columns import (
    DEPTH_COLUMN,
    SITE_ENVIRONMENT_TABLE,
)
from pdbai.engines.analyses.site_environment.compute import (
    SiteEnvironmentComputation,
    compute_structure_site_environment,
    symmetry_mates,
)
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.compatibility import require_compatible_options
from pdbai.engines.core import EngineContext, EngineOptions
from pdbai.engines.core.tables import validate_tables
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files
from pdbai.profiles import get_profile

_ON = {"include_site_environment": True}
_ASP_ID = "1SIT:1:A:1::CB:"

# An aspartate along x, with its carboxylate oxygens 1 Å either side of it.
_ASP = residue(
    "ASP",
    1,
    {
        "CA": (-1.5, 0, 0),
        "CB": (0, 0, 0),
        "CG": (1.5, 0, 0),
        "OD1": (2.1, 1, 0),
        "OD2": (2.1, -1, 0),
    },
)


def _water(number: int, xyz: tuple[float, float, float]) -> SiteAtom:
    return SiteAtom("HOH", number, "O", xyz, element="O", hetatm=True)


def _compute(
    tmp_path: Path, atoms: list[SiteAtom], *, space_group: str | None = "P 1"
) -> SiteEnvironmentComputation:
    return compute_structure_site_environment(
        parse_site_atoms(write_site_cif(tmp_path, atoms, space_group=space_group))
    )


def _row(computation: SiteEnvironmentComputation, site_id: str) -> dict[str, Any]:
    """Return one row as plain values by column name."""
    rows = computation.table[computation.table["site_id"] == site_id]
    assert len(rows) == 1
    return {str(name): value for name, value in rows.iloc[0].to_dict().items()}


def test_every_residue_site_is_a_row_with_its_site_id(tmp_path: Path) -> None:
    atoms = [
        *_ASP,
        *residue("CYS", 2, {"CB": (0, 9, 0), "SG": (0, 10.8, 0)}),
        *residue("LEU", 3, {"CG": (9, 0, 0), "CD1": (10.5, 0, 0)}),
    ]
    table = parse_site_atoms(write_site_cif(tmp_path, atoms))

    environment = compute_structure_site_environment(table).table
    sites = compute_structure_residue_sites(table).table

    assert environment["site_id"].tolist() == sites["site_id"].tolist()
    assert environment["site_class"].tolist() == [
        "asp_carboxylate",
        "cys_sulfur",
        "leu_delta_carbons",
    ]
    assert environment["n_group_atoms"].tolist() == [3, 1, 1]


def test_hydrogen_bonds_and_salt_bridges_are_counted_by_distance(
    tmp_path: Path,
) -> None:
    atoms = [
        *_ASP,
        # NZ is 3.0 Å from OD1, a salt bridge and a hydrogen bond.
        *residue("LYS", 2, {"CE": (2.1, 5.5, 0), "NZ": (2.1, 4.0, 0)}),
        # OG is 3.4 Å from OD2, and its CB too far to count for anything.
        *residue("SER", 3, {"CB": (2.1, -5.9, 0), "OG": (2.1, -4.4, 0)}),
        # One water 2.8 Å from OD1 and one 4.5 Å from it, near but not bonded.
        _water(101, (2.1, 1, 2.8)),
        _water(102, (2.1, 1, -4.5)),
    ]

    row = _row(_compute(tmp_path, atoms), _ASP_ID)

    assert row["hbond_protein_count"] == 2
    assert row["hbond_water_count"] == 1
    assert row["water_count_6A"] == 2
    assert row["salt_bridge_partner_count"] == 1
    assert row["nearest_cation_distance_A"] == pytest.approx(3.0)
    assert row["environment_status"] == "ok"


def test_a_group_does_not_count_its_own_residue(tmp_path: Path) -> None:
    # The backbone nitrogen of the aspartate itself is 2.5 Å from OD1.
    asp = residue("ASP", 1, {"N": (2.1, 3.5, 0), **{a.name: a.xyz for a in _ASP}})

    row = _row(_compute(tmp_path, asp), _ASP_ID)

    assert row["hbond_protein_count"] == 0
    assert row["salt_bridge_partner_count"] == 0


def test_what_a_class_or_model_cannot_have_is_missing(tmp_path: Path) -> None:
    atoms = [
        *residue("LEU", 1, {"CG": (0, 0, 0), "CD1": (1.5, 0, 0), "CD2": (0, 1.5, 0)}),
        *residue("THR", 2, {"CB": (0, 9, 0), "OG1": (1.4, 9, 0)}),
    ]

    computation = _compute(tmp_path, atoms)
    leucine = _row(computation, "1SIT:1:A:1::CG:")
    threonine = _row(computation, "1SIT:1:A:2::CB:")

    # A carbon group has no hydrogen bonds, and only a carboxylate has salt
    # bridges, or a cysteine a disulfide.
    assert pd.isna(leucine["hbond_protein_count"])
    assert threonine["hbond_protein_count"] == 0
    assert pd.isna(threonine["salt_bridge_partner_count"])
    assert pd.isna(threonine["in_disulfide"])
    # Without a water in the model, the water columns are unknown, not zero.
    assert pd.isna(threonine["hbond_water_count"])
    assert pd.isna(threonine["water_count_6A"])
    for name in (
        "nearest_cation_distance_A",
        "nearest_disulfide_sulfur_distance_A",
        "nearest_metal_distance_A",
        "nearest_aromatic_ring_distance_A",
        "nearest_ligand_distance_A",
    ):
        assert math.isnan(threonine[name])
    assert pd.isna(threonine["nearest_aromatic_res_name"])


def _hexagon(
    centre: tuple[float, float, float], names: tuple[str, ...]
) -> dict[str, tuple[float, float, float]]:
    """Return ring atoms 1.39 Å from a centre, in the plane normal to z."""
    x, y, z = centre
    return {
        name: (
            x + 1.39 * math.cos(math.pi * index / 3),
            y + 1.39 * math.sin(math.pi * index / 3),
            z,
        )
        for index, name in enumerate(names)
    }


def test_the_nearest_disulfide_metal_ring_and_ligand_are_measured(
    tmp_path: Path,
) -> None:
    ring = ("CG", "CD1", "CE1", "CZ", "CE2", "CD2")

    def cysteine(number: int, x: float) -> list[SiteAtom]:
        return residue("CYS", number, {"CB": (x, -1.8, 20), "SG": (x, 0, 20)})

    atoms = [
        *_ASP,
        *residue("PHE", 2, _hexagon((1.5, 0, 5), ring)),
        SiteAtom("ZN", 201, "ZN", (2.1, 1, -6), element="ZN", hetatm=True),
        SiteAtom("GOL", 301, "C1", (-5, 0, 0), element="C", hetatm=True),
        # Residues 3 and 4 are bonded, 2.0 Å apart, and 5 and 6 as well.
        *cysteine(3, 1.5),
        *cysteine(4, 3.5),
        *cysteine(5, 1.5 + 30),
        *cysteine(6, 3.5 + 30),
        # A free cysteine 10 Å from residue 5.
        *residue("CYS", 7, {"CB": (31.5, -1.8, 30), "SG": (31.5, 0, 30)}),
    ]

    computation = _compute(tmp_path, atoms)
    asp = _row(computation, _ASP_ID)
    bonded = _row(computation, "1SIT:1:A:3::CB:")
    free = _row(computation, "1SIT:1:A:7::CB:")

    assert asp["nearest_aromatic_ring_distance_A"] == pytest.approx(5.0)
    assert asp["nearest_aromatic_res_name"] == "PHE"
    assert asp["nearest_metal_distance_A"] == pytest.approx(6.0)
    assert asp["nearest_ligand_distance_A"] == pytest.approx(6.5)
    assert asp["nearest_disulfide_sulfur_distance_A"] == pytest.approx(20.0)
    # A cysteine's own bond is not its nearest disulfide.
    assert bonded["in_disulfide"] is True
    assert bonded["nearest_disulfide_sulfur_distance_A"] == pytest.approx(30.0)
    assert free["in_disulfide"] is False
    assert free["nearest_disulfide_sulfur_distance_A"] == pytest.approx(10.0)


def _cube(spacing: float) -> list[SiteAtom]:
    """Return a 5 by 5 by 5 block of carbon atoms with a Ser hydroxyl at its centre."""
    atoms: list[SiteAtom] = []
    for number, steps in enumerate(itertools.product(range(-2, 3), repeat=3), 10):
        if steps in ((0, 0, 0), (1, 0, 0)):
            continue
        x, y, z = (spacing * step for step in steps)
        atoms.append(SiteAtom("ALA", number, "CB", (x, y, z), seq=number))
    return [*atoms, *residue("SER", 1, {"OG": (0, 0, 0), "CB": (spacing, 0, 0)})]


def test_a_group_in_the_open_is_exposed_and_one_inside_is_buried(
    tmp_path: Path,
) -> None:
    open_row = _row(_compute(tmp_path, _ASP), _ASP_ID)
    buried = _compute(tmp_path, _cube(3.5))
    row = _row(buried, "1SIT:1:A:1::CB:")

    assert open_row["site_relative_sasa"] == pytest.approx(1.0)
    assert open_row["site_depth_A"] == pytest.approx(0.0)
    assert row["site_sasa_A2"] == pytest.approx(0.0)
    assert row["site_relative_sasa"] == pytest.approx(0.0)
    # The outer layer of the block is two spacings from its centre.
    assert row["site_depth_A"] == pytest.approx(7.0)
    # Face and edge neighbours lie within 6 Å, and within 10 Å every point
    # of the block whose steps square to 8 or less, but the hydroxyl itself.
    assert row["heavy_atom_count_6A"] == 18
    assert row["heavy_atom_count_10A"] == 92


def test_every_protein_atom_gets_a_depth(tmp_path: Path) -> None:
    atoms = [*_cube(3.5), _water(500, (30, 30, 30))]
    table = parse_site_atoms(write_site_cif(tmp_path, atoms))

    depth = compute_structure_site_environment(table).depth

    assert depth.index.equals(table.index)
    by_position = dict(
        zip(map(tuple, table[["x", "y", "z"]].to_numpy()), depth, strict=True)
    )
    assert by_position[(0.0, 0.0, 0.0)] == pytest.approx(7.0)
    assert by_position[(7.0, 7.0, 7.0)] == pytest.approx(0.0)
    assert math.isnan(by_position[(30.0, 30.0, 30.0)])


def test_crystal_contacts_come_from_the_images_of_the_model(tmp_path: Path) -> None:
    # In a cubic P 1 cell of 40 Å, a glycine at x = 39 has an image at x = −1,
    # 2.69 Å from CG.
    neighbour = residue("GLY", 2, {"CA": (39.0, 1, 0)})

    touching = _row(_compute(tmp_path, [*_ASP, *neighbour]), _ASP_ID)
    alone = _row(_compute(tmp_path, _ASP), _ASP_ID)
    no_cell = _row(_compute(tmp_path, [*_ASP, *neighbour], space_group=None), _ASP_ID)

    assert touching["nearest_symmetry_mate_distance_A"] == pytest.approx(
        math.dist((1.5, 0, 0), (-1.0, 1, 0))
    )
    assert touching["symmetry_mate_atom_count_6A"] == 1
    assert touching["crystal_contacts_status"] == "ok"
    # The lattice puts the aspartate's own images 40 Å away.
    assert math.isnan(alone["nearest_symmetry_mate_distance_A"])
    assert alone["symmetry_mate_atom_count_6A"] == 0
    assert alone["crystal_contacts_status"] == "none_in_range"
    assert math.isnan(no_cell["nearest_symmetry_mate_distance_A"])
    assert pd.isna(no_cell["symmetry_mate_atom_count_6A"])
    assert no_cell["crystal_contacts_status"] == "no_unit_cell"
    # Everything else is the same without a cell.
    assert no_cell["heavy_atom_count_10A"] == touching["heavy_atom_count_10A"]


def test_symmetry_mates_agree_with_gemmi_in_an_orthorhombic_cell() -> None:
    structure = gemmi.Structure()
    structure.cell = gemmi.UnitCell(31, 47, 53, 90, 90, 90)
    structure.spacegroup_hm = "P 21 21 21"
    structure.setup_cell_images()
    cell = structure.cell
    points = np.random.default_rng(7).uniform(-5, 40, size=(40, 3))

    mates = symmetry_mates(cell, points, 12.0).tolist()

    for point in points[:10]:
        expected = min(
            cell.find_nearest_image(
                gemmi.Position(*point), gemmi.Position(*other), gemmi.Asu.Different
            ).dist()
            for other in points
        )
        found = min(math.dist(point.tolist(), mate) for mate in mates)
        assert found >= expected - 1e-9
        if expected <= 12.0:
            assert found == pytest.approx(expected)


def test_a_group_without_atoms_is_a_row_with_nothing_measured(
    tmp_path: Path,
) -> None:
    glu = residue("GLU", 2, {"CA": (0, 0, 0), "CB": (1.5, 0, 0), "CG": (3, 0, 0)})

    row = _row(_compute(tmp_path, glu), "1SIT:1:A:2::CG:")

    assert row["environment_status"] == "no_group_atoms"
    assert pd.isna(row["n_group_atoms"])
    assert math.isnan(row["site_depth_A"])
    assert pd.isna(row["crystal_contacts_status"])


def test_an_atom_at_zero_occupancy_is_not_part_of_the_group(tmp_path: Path) -> None:
    asp = [
        dataclasses.replace(atom, occupancy=0.0) if atom.name == "OD2" else atom
        for atom in _ASP
    ]

    row = _row(_compute(tmp_path, asp), _ASP_ID)

    assert row["n_group_atoms"] == 2


def test_the_table_fits_its_declaration(tmp_path: Path) -> None:
    computation = _compute(tmp_path, [*_ASP, _water(101, (5, 5, 5))])

    fixed = validate_tables(
        "site_environment",
        [SITE_ENVIRONMENT_TABLE],
        {"site_environment": computation.table},
    )["site_environment"]

    assert list(fixed.columns) == list(SITE_ENVIRONMENT_TABLE.column_names)
    assert len(fixed) == 1


def _context(path: Path) -> EngineContext:
    return EngineContext(
        path.stem, parse_site_atoms(path), EngineOptions(values=_ON), source="upload"
    )


def test_the_engine_reports_the_structure(tmp_path: Path) -> None:
    result = SITE_ENVIRONMENT_ENGINE.run(_context(write_site_cif(tmp_path, _ASP)))

    assert set(result.tables) == {"site_environment"}
    assert result.columns is not None
    assert list(result.columns.columns) == [DEPTH_COLUMN]
    assert result.report["site_environment_status"] == "ok"


def test_a_structure_without_sites_leaves_its_table_out(tmp_path: Path) -> None:
    glycine = residue("GLY", 1, {"N": (0, 0, 0), "CA": (1.5, 0, 0)})

    result = SITE_ENVIRONMENT_ENGINE.run(_context(write_site_cif(tmp_path, glycine)))

    assert result.tables == {}
    assert result.report["site_environment_status"] == "no_sites"
    assert result.columns is not None
    assert result.columns[DEPTH_COLUMN].tolist() == [0.0, 0.0]


def test_the_engine_is_off_by_default_and_runs_for_every_technique() -> None:
    registry = get_engine_registry()

    assert (
        registry.with_defaults(EngineOptions()).flag("include_site_environment")
        is False
    )
    assert SITE_ENVIRONMENT_ENGINE.spec.techniques is None
    require_compatible_options(
        registry.with_defaults(
            EngineOptions(technique="neutron_diffraction", values=_ON)
        )
    )


def test_only_the_radiation_profile_switches_the_engine_on() -> None:
    assert "include_site_environment" in get_profile("radiation_sensitivity").toggles
    assert "include_site_environment" not in get_profile("research_ready").toggles


def test_a_build_writes_the_table_and_the_depth_of_every_atom(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    working_directory = tmp_path / "work"

    build_atom_site_dataset_from_mmcif_files(
        [write_site_cif(tmp_path, _ASP)],
        working_directory,
        run_id="environment",
        engine_options=dataclasses.replace(
            offline_options, dataset_shape="star_schema"
        ).with_values(_ON),
    )

    run_directory = working_directory / "runs" / "environment"
    rows = pd.read_csv(
        run_directory / "dataset_site_environment.csv", keep_default_na=False
    )
    atoms = pd.read_csv(run_directory / "dataset_atoms.csv")
    assert list(rows.columns) == list(SITE_ENVIRONMENT_TABLE.column_names)
    assert rows["site_id"].tolist() == [_ASP_ID]
    assert rows["crystal_contacts_status"].tolist() == ["none_in_range"]
    assert atoms[DEPTH_COLUMN].tolist() == [0.0] * len(_ASP)
