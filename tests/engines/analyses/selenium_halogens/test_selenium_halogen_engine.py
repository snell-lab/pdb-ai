from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from site_cif import SiteAtom, parse_site_atoms, residue, write_site_cif

from pdbai.engines.analyses.selenium_halogens import SELENIUM_HALOGEN_ENGINE
from pdbai.engines.analyses.selenium_halogens.columns import (
    SELENIUM_HALOGEN_LINK_COLUMN,
    SELENIUM_HALOGEN_SITES_TABLE,
)
from pdbai.engines.analyses.selenium_halogens.compute import (
    SeleniumHalogenComputation,
    compute_structure_selenium_halogens,
    covalent_radius,
)
from pdbai.engines.analyses.selenium_halogens.constants import BOND_TOLERANCE_A
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.compatibility import require_compatible_options
from pdbai.engines.core import EngineContext, EngineOptions
from pdbai.engines.core.tables import validate_tables
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files

_ON = {"include_selenium_halogens": True}

# A selenomethionine with both Se–C bonds at 1.95 Å, and a selenium whose
# B-factor is twice the mean of the two carbons it is bonded to.
_MSE = residue(
    "MSE",
    1,
    {"CB": (-1.5, 0, 0), "CG": (0, 0, 0), "SE": (1.95, 0, 0), "CE": (1.95, 1.95, 0)},
    b_factors={"CB": 10, "CG": 10, "SE": 30, "CE": 20},
    elements={"SE": "SE"},
    hetatm=True,
)

# A ligand whose bromine is deposited at half the occupancy of its carbons.
_LIGAND = residue(
    "LIG",
    201,
    {"C1": (10, 0, 0), "C2": (11.4, 0, 0), "BR1": (8.1, 0, 0)},
    b_factors={"C1": 20, "C2": 30, "BR1": 50},
    occupancies={"BR1": 0.5},
    elements={"BR1": "BR"},
    polymer=False,
    hetatm=True,
)


def _bromide(number: int, xyz: tuple[float, float, float]) -> SiteAtom:
    return SiteAtom("BR", number, "BR", xyz, element="BR", hetatm=True)


def _compute(
    tmp_path: Path, atoms: list[SiteAtom], *, space_group: str | None = "P 1"
) -> SeleniumHalogenComputation:
    path = write_site_cif(tmp_path, atoms, space_group=space_group)
    return compute_structure_selenium_halogens(parse_site_atoms(path))


def _row(computation: SeleniumHalogenComputation, site_id: str) -> dict[str, Any]:
    """Return one row as plain values by column name."""
    rows = computation.table[computation.table["site_id"] == site_id]
    assert len(rows) == 1
    return {str(name): value for name, value in rows.iloc[0].to_dict().items()}


def test_the_selenium_of_selenomethionine_is_bonded_to_two_carbons(
    tmp_path: Path,
) -> None:
    computation = _compute(tmp_path, _MSE)
    row = _row(computation, "1SIT:1:A:1::SE:")

    assert computation.status == "ok"
    assert row["element"] == "SE"
    assert row["parent_type"] == "selenomethionine"
    assert row["component_atom_count"] == 4
    assert row["n_bonded_atoms"] == 2
    assert row["bonded_site_id"] == "1SIT:1:A:1::CG:"
    assert row["bonded_element"] == "C"
    assert row["bond_distance_min_A"] == pytest.approx(1.95)
    assert row["bond_distance_max_A"] == pytest.approx(1.95)
    assert row["bonded_b_factor_mean_A2"] == pytest.approx(15.0)
    assert row["b_factor_ratio_to_bonded"] == pytest.approx(2.0)
    assert row["component_b_factor_mean_A2"] == pytest.approx(40 / 3)
    assert row["site_status"] == "ok"


def test_a_halogen_is_set_beside_the_rest_of_its_ligand(tmp_path: Path) -> None:
    row = _row(_compute(tmp_path, _LIGAND), "1SIT:1:A:201::BR1:")

    assert row["element"] == "BR"
    assert row["parent_type"] == "ligand"
    assert row["n_bonded_atoms"] == 1
    assert row["bonded_atom_name"] == "C1"
    assert row["bond_distance_min_A"] == pytest.approx(1.9)
    assert row["occupancy"] == pytest.approx(0.5)
    assert row["component_occupancy_mean"] == pytest.approx(1.0)
    assert row["b_factor_ratio_to_bonded"] == pytest.approx(2.5)
    assert row["b_factor_ratio_to_component"] == pytest.approx(2.0)


def test_the_lighter_halogens_are_rows_and_no_halogen_is_a_reference(
    tmp_path: Path,
) -> None:
    # C–F at 1.35 Å and C–Cl at 1.74 Å on one ligand, with the chlorine and
    # the fluorine at B-factors the carbons do not have.
    ligand = residue(
        "LIG",
        201,
        {
            "C1": (0, 0, 0),
            "C2": (1.4, 0, 0),
            "F1": (-1.35, 0, 0),
            "CL1": (1.4, 1.74, 0),
        },
        b_factors={"C1": 20, "C2": 20, "F1": 30, "CL1": 60},
        occupancies={"CL1": 0.5},
        elements={"F1": "F", "CL1": "CL"},
        polymer=False,
        hetatm=True,
    )

    computation = _compute(tmp_path, ligand)
    fluorine = _row(computation, "1SIT:1:A:201::F1:")
    chlorine = _row(computation, "1SIT:1:A:201::CL1:")

    assert computation.table["element"].tolist() == ["F", "CL"]
    assert (fluorine["bonded_atom_name"], chlorine["bonded_atom_name"]) == ("C1", "C2")
    assert fluorine["bond_distance_min_A"] == pytest.approx(1.35)
    assert chlorine["bond_distance_min_A"] == pytest.approx(1.74)
    # The reference is the two carbons alone, for either halogen.
    assert fluorine["component_b_factor_mean_A2"] == pytest.approx(20.0)
    assert fluorine["b_factor_ratio_to_component"] == pytest.approx(1.5)
    assert chlorine["b_factor_ratio_to_component"] == pytest.approx(3.0)
    assert chlorine["component_occupancy_mean"] == pytest.approx(1.0)
    assert chlorine["occupancy"] == pytest.approx(0.5)


def test_a_chloride_ion_and_a_chlorine_on_a_metal_have_no_bonded_atom(
    tmp_path: Path,
) -> None:
    chloride = SiteAtom("CL", 301, "CL", (20, 20, 20), element="CL", hetatm=True)
    complex_ = residue(
        "CPT",
        401,
        {"PT1": (0, 0, 0), "CL1": (2.3, 0, 0), "N1": (0, 2.0, 0)},
        elements={"PT1": "PT", "CL1": "CL"},
        polymer=False,
        hetatm=True,
    )

    table = _compute(tmp_path, [chloride, *complex_]).table

    assert table["res_name"].tolist() == ["CL", "CPT"]
    assert table["parent_type"].tolist() == ["ion", "ligand"]
    assert table["n_bonded_atoms"].tolist() == [0, 0]
    assert table["site_status"].tolist() == ["no_bonded_atom"] * 2


def test_a_halogenated_residue_of_a_polymer_is_told_from_a_ligand(
    tmp_path: Path,
) -> None:
    nucleotide = residue(
        "BRU",
        5,
        {"C5": (0, 0, 0), "BR": (1.9, 0, 0)},
        elements={"BR": "BR"},
        hetatm=True,
    )

    row = _row(_compute(tmp_path, nucleotide), "1SIT:1:A:5::BR:")

    assert row["parent_type"] == "polymer_residue"


def test_an_ion_has_a_row_without_a_bonded_atom(tmp_path: Path) -> None:
    computation = _compute(tmp_path, [_bromide(301, (20, 20, 20))])
    row = _row(computation, "1SIT:1:A:301::BR:")

    assert row["parent_type"] == "ion"
    assert row["n_bonded_atoms"] == 0
    assert pd.isna(row["bonded_site_id"])
    assert pd.isna(row["bonded_element"])
    assert math.isnan(row["bond_distance_min_A"])
    assert math.isnan(row["component_occupancy_mean"])
    assert row["site_status"] == "no_bonded_atom"


def test_a_bond_is_found_up_to_the_radii_plus_the_tolerance(tmp_path: Path) -> None:
    limit = covalent_radius("C") + covalent_radius("I") + BOND_TOLERANCE_A

    def iodide_beside_carbon(distance: float) -> list[SiteAtom]:
        return residue(
            "LIG",
            1,
            {"C1": (0, 0, 0), "I1": (distance, 0, 0)},
            elements={"I1": "I"},
            polymer=False,
            hetatm=True,
        )

    inside = _row(
        _compute(tmp_path, iodide_beside_carbon(limit - 0.01)), "1SIT:1:A:1::I1:"
    )
    outside = _row(
        _compute(tmp_path, iodide_beside_carbon(limit + 0.01)), "1SIT:1:A:1::I1:"
    )

    assert limit == pytest.approx(2.57)
    assert inside["n_bonded_atoms"] == 1
    assert outside["n_bonded_atoms"] == 0
    assert outside["site_status"] == "no_bonded_atom"


def test_hydrogens_metals_and_atoms_at_zero_occupancy_are_never_bonded(
    tmp_path: Path,
) -> None:
    atoms = [
        *_LIGAND,
        SiteAtom("LIG", 201, "H1", (8.1, 1.0, 0), element="H", hetatm=True),
        SiteAtom("LIG", 201, "D1", (8.1, -1.0, 0), element="D", hetatm=True),
        SiteAtom("HG", 401, "HG", (8.1, 0, 2.0), element="HG", hetatm=True),
        SiteAtom("LIG", 201, "C9", (6.2, 0, 0), occupancy=0.0, hetatm=True),
    ]

    row = _row(_compute(tmp_path, atoms), "1SIT:1:A:201::BR1:")

    assert row["n_bonded_atoms"] == 1
    assert row["bonded_atom_name"] == "C1"


def test_an_atom_at_zero_occupancy_is_still_a_row(tmp_path: Path) -> None:
    lost = [
        dataclasses.replace(atom, occupancy=0.0) if atom.name == "BR1" else atom
        for atom in _LIGAND
    ]

    row = _row(_compute(tmp_path, lost), "1SIT:1:A:201::BR1:")

    assert row["occupancy"] == pytest.approx(0.0)
    assert row["n_bonded_atoms"] == 1


def test_an_atom_on_a_symmetry_element_is_marked(tmp_path: Path) -> None:
    # In P 1 2 1 the twofold axis runs along b through the origin.
    atoms = [_bromide(301, (0, 5, 0)), _bromide(302, (10, 5, 10))]

    table = _compute(tmp_path, atoms, space_group="P 1 2 1").table

    assert table["on_special_position"].tolist() == [True, False]
    assert _compute(tmp_path, atoms).table["on_special_position"].tolist() == [
        False,
        False,
    ]
    without_cell = _compute(tmp_path, atoms, space_group=None).table
    assert without_cell["on_special_position"].isna().all()


def test_each_atom_links_to_its_own_row(tmp_path: Path) -> None:
    atoms = parse_site_atoms(write_site_cif(tmp_path, [*_MSE, *_LIGAND]))
    computation = compute_structure_selenium_halogens(atoms)

    linked = computation.link.dropna()
    assert linked.tolist() == computation.table["site_id"].tolist()
    assert atoms.loc[linked.index, "element"].str.upper().tolist() == ["SE", "BR"]
    assert computation.link.index.equals(atoms.index)


def test_the_table_fits_its_declaration(tmp_path: Path) -> None:
    computation = _compute(tmp_path, [*_MSE, _bromide(301, (20, 20, 20))])

    fixed = validate_tables(
        "selenium_halogens",
        [SELENIUM_HALOGEN_SITES_TABLE],
        {"selenium_halogen_sites": computation.table},
    )["selenium_halogen_sites"]

    assert list(fixed.columns) == list(SELENIUM_HALOGEN_SITES_TABLE.column_names)
    assert fixed["bonded_site_id"].isna().tolist() == [False, True]


def _context(path: Path) -> EngineContext:
    return EngineContext(
        path.stem, parse_site_atoms(path), EngineOptions(values=_ON), source="upload"
    )


def test_the_engine_reports_the_structure(tmp_path: Path) -> None:
    result = SELENIUM_HALOGEN_ENGINE.run(_context(write_site_cif(tmp_path, _MSE)))

    assert set(result.tables) == {"selenium_halogen_sites"}
    assert result.columns is not None
    assert list(result.columns.columns) == [SELENIUM_HALOGEN_LINK_COLUMN]
    assert result.report["selenium_halogens_status"] == "ok"


def test_a_structure_without_such_atoms_leaves_its_table_out(tmp_path: Path) -> None:
    methionine = residue("MET", 1, {"CG": (0, 0, 0), "SD": (1.8, 0, 0)})

    result = SELENIUM_HALOGEN_ENGINE.run(_context(write_site_cif(tmp_path, methionine)))

    assert result.tables == {}
    assert result.report["selenium_halogens_status"] == "no_sites"


def test_the_engine_is_off_by_default_and_runs_for_every_technique() -> None:
    registry = get_engine_registry()

    assert (
        registry.with_defaults(EngineOptions()).flag("include_selenium_halogens")
        is False
    )
    assert SELENIUM_HALOGEN_ENGINE.spec.techniques is None
    require_compatible_options(
        registry.with_defaults(
            EngineOptions(technique="neutron_diffraction", values=_ON)
        )
    )


def test_a_build_writes_the_table_and_links_atoms_to_it(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    working_directory = tmp_path / "work"

    build_atom_site_dataset_from_mmcif_files(
        [write_site_cif(tmp_path, [*_MSE, _bromide(301, (20, 20, 20))])],
        working_directory,
        run_id="halogens",
        engine_options=dataclasses.replace(
            offline_options, dataset_shape="star_schema"
        ).with_values(_ON),
    )

    run_directory = working_directory / "runs" / "halogens"
    sites = pd.read_csv(
        run_directory / "dataset_selenium_halogen_sites.csv", keep_default_na=False
    )
    atoms = pd.read_csv(run_directory / "dataset_atoms.csv", keep_default_na=False)
    assert list(sites.columns) == list(SELENIUM_HALOGEN_SITES_TABLE.column_names)
    assert sites["site_id"].tolist() == ["1SIT:1:A:1::SE:", "1SIT:1:A:301::BR:"]
    assert sites["bonded_site_id"].tolist() == ["1SIT:1:A:1::CG:", ""]
    linked = atoms[atoms[SELENIUM_HALOGEN_LINK_COLUMN] != ""]
    assert sorted(linked[SELENIUM_HALOGEN_LINK_COLUMN]) == sorted(sites["site_id"])
