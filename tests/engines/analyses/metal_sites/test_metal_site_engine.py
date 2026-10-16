from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pdbai.engines.analyses.metal_sites import reference as reference_module
from pdbai.engines.analyses.metal_sites.columns import (
    METAL_BONDS_TABLE,
    METAL_LINK_COLUMN,
    METAL_PAIRS_TABLE,
    METAL_SITES_TABLE,
)
from pdbai.engines.analyses.metal_sites.compute import (
    MetalComputation,
    compute_structure_metal_sites,
    donor_rule,
    reference_key,
)
from pdbai.engines.analyses.metal_sites.constants import (
    BOND_STATUSES,
    DONOR_AMINO_ACID,
    DONOR_WATER,
    FIRST_SPHERE_TOLERANCE_A,
    METAL_ELEMENTS,
)
from pdbai.engines.analyses.metal_sites.engine import METAL_SITE_ENGINE
from pdbai.engines.analyses.struct_conn import DETECTION_METHODS
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.compatibility import require_compatible_options
from pdbai.engines.core import (
    STRUCTURE_CIF_PATH,
    EngineContext,
    EngineOptions,
)
from pdbai.engines.core.tables import validate_tables
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files
from pdbai.structure import parse_mmcif_atom_rows, read_mmcif_as_flat_dict

from .conftest import metal_cif_text

_ON = {"include_metal_sites": True}
_ZINC = "1MTL:1:A:401::ZN:"
_CALCIUM = "1MTL:1:A:402::CA:"
_CADMIUM = "1MTL:1:A:403::CD:"
_IRON = "1MTL:1:A:404::FE:"
_MANGANESE = "1MTL:1:A:405::MN:"
_ZINC_MODEL_2 = "1MTL:2:A:401::ZN:"


_NO_METAL_ATOMS = (
    "ATOM 1 N N . GLY A 1 1 ? 1 2 3 1.00 20 101 GLY A N 1\n"
    "ATOM 2 O O . GLY A 1 1 ? 2 2 3 1.00 20 101 GLY A O 1\n"
)


@pytest.fixture
def no_metal_cif(tmp_path: Path) -> Path:
    """Write a file with one glycine and no metal atom, and return its path."""
    path = tmp_path / "1nom.cif"
    path.write_text(metal_cif_text(atoms=_NO_METAL_ATOMS), encoding="utf-8")
    return path


def _compute(path: Path) -> tuple[MetalComputation, pd.DataFrame]:
    atoms = parse_mmcif_atom_rows(path, {}, structure_id=path.stem).dataframe
    return compute_structure_metal_sites(atoms, read_mmcif_as_flat_dict(path)), atoms


def _bonds_of(bonds: pd.DataFrame, site_id: str) -> pd.DataFrame:
    return bonds[bonds["site_id"] == site_id].set_index("donor_site_id")


def _site(computation: MetalComputation, site_id: str) -> dict[str, Any]:
    """Return one site row as plain values by column name."""
    rows = computation.sites[computation.sites["site_id"] == site_id]
    assert len(rows) == 1
    return {str(name): value for name, value in rows.iloc[0].to_dict().items()}


def _bond(computation: MetalComputation, site_id: str, donor: str) -> dict[str, Any]:
    """Return one bond row as plain values by column name."""
    bonds = computation.bonds
    rows = bonds[(bonds["site_id"] == site_id) & (bonds["donor_site_id"] == donor)]
    assert len(rows) == 1
    return {str(name): value for name, value in rows.iloc[0].to_dict().items()}


def test_every_metal_atom_of_every_model_is_a_site(metal_cif: Path) -> None:
    computation, atoms = _compute(metal_cif)

    assert computation.sites["site_id"].tolist() == [
        _ZINC,
        _CALCIUM,
        _CADMIUM,
        _IRON,
        _MANGANESE,
        _ZINC_MODEL_2,
    ]
    assert computation.sites["metal_element"].tolist() == [
        "ZN",
        "CA",
        "CD",
        "FE",
        "MN",
        "ZN",
    ]
    linked = computation.link.dropna()
    assert linked.tolist() == computation.sites["site_id"].tolist()
    assert computation.link.index.equals(atoms.index)
    assert atoms.loc[linked.index, "element"].str.upper().tolist() == [
        "ZN",
        "CA",
        "CD",
        "FE",
        "MN",
        "ZN",
    ]


def test_the_zinc_site_holds_its_inferred_and_declared_bonds(metal_cif: Path) -> None:
    computation, _atoms = _compute(metal_cif)
    bonds = _bonds_of(computation.bonds, _ZINC)

    assert bonds.index.tolist() == [
        "1MTL:1:A:102::NE2:",
        "1MTL:1:A:103::ND1:",
        "1MTL:1:A:104::SG:",
        "1MTL:1:A:105::ND2:",
        "1MTL:1:A:501::O:",
    ]
    assert bonds["distance_A"].tolist() == pytest.approx([2.03, 2.23, 2.31, 3.10, 2.50])
    assert bonds["detection_method"].tolist() == [
        "struct_conn_and_distance",
        "distance",
        "distance",
        "struct_conn",
        "distance",
    ]
    assert bonds["reference_key"].tolist() == [
        "HIS:N:ZN",
        "HIS:N:ZN",
        "CYS:S:ZN",
        "*:N:ZN",
        "HOH:O:ZN",
    ]
    assert bonds["first_sphere_cutoff_A"].tolist() == pytest.approx(
        [2.03 + FIRST_SPHERE_TOLERANCE_A] * 2 + [3.06, 2.78, 2.84]
    )
    deviations: list[float] = bonds["distance_deviation_A"].tolist()
    assert deviations[:3] + deviations[4:] == pytest.approx(
        [0.0, 0.20, 0.0, 0.41], abs=1e-9
    )
    assert math.isnan(deviations[3])
    assert bonds["bond_status"].tolist() == [
        "ok",
        "ok",
        "ok",
        "no_reference",
        "ok",
    ]
    assert bonds["donor_rule"].tolist()[3] == "outside_typical_donor_list"
    assert bonds["donor_class"].tolist()[4] == DONOR_WATER


def test_a_bond_z_divides_the_deviation_by_the_combined_uncertainty(
    metal_cif: Path,
) -> None:
    computation, _atoms = _compute(metal_cif)
    # 19 non-hydrogen atoms in model 1, one of them at zero occupancy.
    expected_dpi = 1.28 * 18.0**0.5 * 27000.0 ** (1 / 3) * 20000.0 ** (-5 / 6) * 0.2

    assert computation.dpi == pytest.approx(expected_dpi)
    assert computation.sites["dpi_A"].tolist() == pytest.approx([expected_dpi] * 6)
    bond = _bond(computation, _ZINC, "1MTL:1:A:103::ND1:")
    assert bond["reference_distance_A"] == 2.03
    assert bond["reference_sd_A"] == 0.05
    assert bond["distance_z"] == pytest.approx(
        0.20 / math.sqrt(expected_dpi**2 + 0.05**2)
    )


def test_hydrogens_empty_atoms_and_backbone_nitrogens_are_not_donors(
    metal_cif: Path,
) -> None:
    computation, _atoms = _compute(metal_cif)
    donors = set(computation.bonds["donor_site_id"])

    assert "1MTL:1:A:102::HE2:" not in donors
    assert "1MTL:1:A:106::O:" not in donors
    assert "1MTL:1:A:107::N:" not in donors
    # The second water lies beyond the first sphere.
    assert "1MTL:1:A:502::O:" not in donors


def test_a_donor_of_a_symmetry_mate_is_found_and_named(metal_cif: Path) -> None:
    computation, _atoms = _compute(metal_cif)
    bond = _bond(computation, _CALCIUM, "1MTL:1:A:110::OD1:")

    assert bond["distance_A"] == pytest.approx(2.36)
    assert bond["donor_symmetry_operator"] == "1_655"
    assert bool(bond["is_symmetry_mate"]) is True
    assert bond["detection_method"] == "struct_conn_and_distance"
    site = _site(computation, _CALCIUM)
    assert site["n_symmetry_bonds"] == 1
    assert site["coordination_number"] == 1


def test_a_metal_without_reference_distances_gets_no_inferred_bond(
    metal_cif: Path,
) -> None:
    computation, _atoms = _compute(metal_cif)
    site = _site(computation, _CADMIUM)

    assert _bonds_of(computation.bonds, _CADMIUM).empty
    assert site["metal_site_status"] == "no_donors_found"
    assert site["coordination_number"] == 0
    assert math.isnan(site["donor_distance_mean_A"])


def test_atoms_of_the_metals_own_component_are_never_donors(metal_cif: Path) -> None:
    computation, _atoms = _compute(metal_cif)
    site = _site(computation, _IRON)

    assert _bonds_of(computation.bonds, _IRON).index.tolist() == ["1MTL:1:A:130::NE2:"]
    assert site["parent_type"] == "heme"
    assert site["component_atom_count"] == 3
    assert site["distance_deviation_mean_A"] == pytest.approx(0.0, abs=1e-9)


def test_only_the_first_residue_of_a_sequence_gives_a_terminal_nitrogen(
    metal_cif: Path,
) -> None:
    computation, _atoms = _compute(metal_cif)
    bond = _bond(computation, _MANGANESE, "1MTL:1:A:101::N:")

    assert bond["donor_rule"] == "n_terminal_nitrogen"
    assert bond["reference_kind"] == "element_fallback"
    assert bond["reference_key"] == "*:N:MN"
    assert math.isnan(bond["distance_deviation_A"])
    assert bond["bond_status"] == "no_reference"


def test_each_model_is_measured_on_its_own_coordinates(metal_cif: Path) -> None:
    computation, _atoms = _compute(metal_cif)
    bonds = _bonds_of(computation.bonds, _ZINC_MODEL_2)

    assert bonds.index.tolist() == ["1MTL:2:A:102::NE2:"]
    assert bonds["distance_A"].tolist() == pytest.approx([2.50])
    site = _site(computation, _ZINC_MODEL_2)
    assert math.isnan(site["nearest_metal_distance_A"])


def test_site_rows_summarize_their_bonds(metal_cif: Path) -> None:
    computation, _atoms = _compute(metal_cif)
    site = _site(computation, _ZINC)

    assert site["parent_type"] == "ion"
    assert site["coordination_number"] == 5
    assert site["n_declared_bonds"] == 2
    assert (site["n_donor_N"], site["n_donor_O"], site["n_donor_S"]) == (3, 1, 1)
    assert site["n_donor_water"] == 1
    assert site["n_reference_bonds"] == 4
    assert site["donor_distance_min_A"] == pytest.approx(2.03)
    assert site["donor_distance_max_A"] == pytest.approx(3.10)
    assert site["distance_deviation_mean_A"] == pytest.approx((0.20 + 0.41) / 4)
    assert site["b_factor_A2"] == 15.0
    assert site["donor_b_factor_mean_A2"] == pytest.approx((20 + 30 + 10 + 20 + 30) / 5)
    assert site["b_factor_ratio_to_donors"] == pytest.approx(15 / 22)
    # The cadmium is the nearest other metal of model 1.
    assert site["nearest_metal_site_id"] == _CADMIUM
    assert site["nearest_metal_distance_A"] == pytest.approx(math.sqrt(150))
    assert site["n_metals_within_6A"] == 0
    assert bool(site["on_special_position"]) is False


def test_unresolved_records_are_counted_and_degrade_the_status(
    metal_cif: Path,
) -> None:
    computation, _atoms = _compute(metal_cif)

    # metalc4 names a water the file lacks, and model 2 has no Asn 105.
    assert computation.unresolved_record_count == 2
    assert computation.status == "degraded"
    assert any("2 deposited" in warning for warning in computation.warnings)


def test_without_a_cell_only_the_file_is_searched(tmp_path: Path) -> None:
    path = tmp_path / "1mtl.cif"
    path.write_text(metal_cif_text(cell=False), encoding="utf-8")

    computation, _atoms = _compute(path)

    assert _bonds_of(computation.bonds, _CALCIUM).empty
    assert math.isnan(computation.dpi)
    assert computation.sites["on_special_position"].isna().all()
    assert any("Symmetry mates were not searched" in w for w in computation.warnings)
    zinc_bonds = _bonds_of(computation.bonds, _ZINC)
    assert len(zinc_bonds) == 5
    assert (zinc_bonds.drop("1MTL:1:A:105::ND2:")["bond_status"] == "no_dpi").all()


def test_a_structure_without_metals_gives_empty_tables(no_metal_cif: Path) -> None:
    computation, atoms = _compute(no_metal_cif)

    assert computation.status == "no_metals"
    assert computation.sites.empty
    assert computation.bonds.empty
    assert list(computation.sites.columns) == list(METAL_SITES_TABLE.column_names)
    assert computation.link.isna().all()
    assert len(computation.link) == len(atoms)


def test_both_tables_fit_their_declarations(metal_cif: Path) -> None:
    computation, _atoms = _compute(metal_cif)

    tables = validate_tables(
        "metal_sites",
        (METAL_SITES_TABLE, METAL_BONDS_TABLE),
        {"metal_sites": computation.sites, "metal_ligand_bonds": computation.bonds},
    )

    assert len(tables["metal_sites"]) == 6
    assert len(tables["metal_ligand_bonds"]) == 9
    assert set(tables["metal_ligand_bonds"]["detection_method"]) <= set(
        DETECTION_METHODS
    )
    assert set(tables["metal_ligand_bonds"]["bond_status"]) <= set(BOND_STATUSES)


@pytest.mark.parametrize(
    ("res_name", "atom_name", "element", "label_seq", "donor_class", "expected"),
    [
        ("HOH", "O", "O", "", DONOR_WATER, "water_oxygen"),
        ("HIS", "NE2", "N", "5", DONOR_AMINO_ACID, "typical_sidechain_donor"),
        ("LEU", "O", "O", "5", DONOR_AMINO_ACID, "backbone_carbonyl_oxygen"),
        ("LEU", "N", "N", "1", DONOR_AMINO_ACID, "n_terminal_nitrogen"),
        ("LEU", "N", "N", "5", DONOR_AMINO_ACID, "outside_typical_donor_list"),
        ("LEU", "OXT", "O", "9", DONOR_AMINO_ACID, "c_terminal_oxygen"),
        ("ARG", "NH1", "N", "5", DONOR_AMINO_ACID, "outside_typical_donor_list"),
        ("MSE", "SE", "SE", "5", "other", "outside_typical_donor_list"),
    ],
)
def test_donor_rules(
    res_name: str,
    atom_name: str,
    element: str,
    label_seq: str,
    donor_class: str,
    expected: str,
) -> None:
    assert donor_rule(res_name, atom_name, element, label_seq, donor_class) == expected


def test_reference_keys_follow_the_tables_conventions() -> None:
    assert reference_key("HOH", "O", "O", DONOR_WATER, "ZN") == ("HOH", "O", "ZN")
    assert reference_key("GLY", "O", "O", DONOR_AMINO_ACID, "CA") == ("CA", "O", "CA")
    assert reference_key("ASP", "OD2", "O", DONOR_AMINO_ACID, "MG") == (
        "ASP",
        "O",
        "MG",
    )
    assert reference_key("CYS", "SG", "S", DONOR_AMINO_ACID, "FE") == ("CYS", "S", "FE")
    assert reference_key("GLY", "N", "N", DONOR_AMINO_ACID, "CU")[0] == "NTERM"
    assert reference_key("GLY", "OXT", "O", DONOR_AMINO_ACID, "CU")[0] == "CTERM"


def test_the_bundled_tables_match_their_manifest() -> None:
    receipts = reference_module.reference_receipts()

    assert set(receipts) == {
        "metal_sites_metal_donor_distances",
        "metal_sites_metallocofactors",
    }
    for receipt in receipts.values():
        assert isinstance(receipt, dict)
        assert receipt["hash_matches_manifest"] is True
    manifest = json.loads(
        reference_module.REFERENCE_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    distances = reference_module.literature_distances()
    assert (
        len(distances) == manifest["metal_donor_distances"]["reference_distance_count"]
    )
    assert distances[("HIS", "N", "ZN")] == (2.03, 0.05)
    assert {metal for _residue, _atom, metal in distances} <= METAL_ELEMENTS
    classes = reference_module.cofactor_classes()
    assert len(classes) == manifest["metallocofactors"]["entry_count"]
    assert classes["HEM"] == "heme"
    assert classes["SF4"] == "cluster"
    assert reference_module.element_targets()[("K", "O")] == 2.82


# A manganese cluster of two atoms 2.7 Å apart, a second component that
# overlaps it with a copy of its first atom 0.2 Å away, and a zinc ion far
# from all three.
_CLUSTER_ATOMS = (
    "HETATM 1 MN MN1 . OEX B 2 . ? 0.0  0 0 0.60 15 601 OEX A MN1 1\n"
    "HETATM 2 MN MN2 . OEX B 2 . ? 2.7  0 0 0.60 15 601 OEX A MN2 1\n"
    "HETATM 3 MN MN1 . OEX C 2 . ? 0.2  0 0 0.40 15 602 OEX A MN1 1\n"
    "HETATM 4 ZN ZN  . ZN  D 3 . ? 20.0 20 20 1.00 15 701 ZN  A ZN  1\n"
)


def test_metal_atoms_near_each_other_are_measured_pair_by_pair(
    tmp_path: Path,
) -> None:
    path = tmp_path / "1clu.cif"
    path.write_text(metal_cif_text(atoms=_CLUSTER_ATOMS), encoding="utf-8")
    computation, _atoms = _compute(path)

    fixed = validate_tables(
        "metal_sites", [METAL_PAIRS_TABLE], {"metal_metal_distances": computation.pairs}
    )["metal_metal_distances"]

    assert fixed[["site_id", "partner_site_id"]].values.tolist() == [
        ["1CLU:1:A:601::MN1:", "1CLU:1:A:601::MN2:"],
        ["1CLU:1:A:601::MN1:", "1CLU:1:A:602::MN1:"],
        ["1CLU:1:A:601::MN2:", "1CLU:1:A:602::MN1:"],
    ]
    assert fixed["distance_A"].tolist() == pytest.approx([2.7, 0.2, 2.5])
    # Only the first pair lies within one component. The other two join
    # the cluster to its overlapping copy.
    assert fixed["same_component"].tolist() == [True, False, False]
    assert set(fixed["site_id"]) | set(fixed["partner_site_id"]) <= set(
        computation.sites["site_id"]
    )
    # The nearest metal of the first atom is its own copy, not its neighbour.
    assert _site(computation, "1CLU:1:A:601::MN1:")[
        "nearest_metal_distance_A"
    ] == pytest.approx(0.2)


def test_metal_atoms_far_apart_give_no_pair(metal_cif: Path) -> None:
    computation, _atoms = _compute(metal_cif)

    assert computation.pairs.empty
    assert list(computation.pairs.columns) == list(METAL_PAIRS_TABLE.column_names)


def _context(path: Path, *, with_file: bool = True) -> EngineContext:
    atoms = parse_mmcif_atom_rows(path, {}, structure_id=path.stem).dataframe
    context = EngineContext(
        path.stem, atoms, EngineOptions(values=_ON), source="upload"
    )
    if with_file:
        context.artifacts.put(STRUCTURE_CIF_PATH, path)
    return context


def test_the_engine_needs_the_structure_file(metal_cif: Path) -> None:
    with pytest.raises(RuntimeError, match="No structure file was available"):
        METAL_SITE_ENGINE.run(_context(metal_cif, with_file=False))


def test_the_engine_reports_the_fixture(metal_cif: Path) -> None:
    result = METAL_SITE_ENGINE.run(_context(metal_cif))

    assert set(result.tables) == {"metal_sites", "metal_ligand_bonds"}
    assert result.columns is not None
    assert list(result.columns.columns) == [METAL_LINK_COLUMN]
    assert result.report["metal_sites_status"] == "degraded"
    assert result.report["metal_unresolved_record_count"] == 2
    assert result.report["metal_dpi_A"] == pytest.approx(0.008488, abs=1e-6)
    assert result.warnings and result.warnings[0].startswith("Metal sites: 1mtl: ")


def test_a_structure_without_metals_leaves_its_tables_out(no_metal_cif: Path) -> None:
    result = METAL_SITE_ENGINE.run(_context(no_metal_cif))

    assert result.tables == {}
    assert result.report["metal_sites_status"] == "no_metals"
    assert result.report["metal_dpi_A"] is None
    assert result.warnings == ()


def test_the_engine_is_off_by_default_and_runs_for_every_technique() -> None:
    registry = get_engine_registry()
    options = registry.with_defaults(EngineOptions())

    assert options.flag("include_metal_sites") is False
    assert METAL_SITE_ENGINE.spec.techniques is None
    require_compatible_options(
        registry.with_defaults(
            EngineOptions(technique="neutron_diffraction", values=_ON)
        )
    )


def test_a_build_writes_both_tables_and_links_atoms_to_them(
    metal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    working_directory = tmp_path / "work"

    result = build_atom_site_dataset_from_mmcif_files(
        [metal_cif],
        working_directory,
        run_id="metals",
        engine_options=dataclasses.replace(
            offline_options, dataset_shape="star_schema"
        ).with_values(_ON),
    )

    run_directory = working_directory / "runs" / "metals"
    sites = pd.read_csv(
        run_directory / "dataset_metal_sites.csv", keep_default_na=False
    )
    bonds = pd.read_csv(
        run_directory / "dataset_metal_ligand_bonds.csv", keep_default_na=False
    )
    atoms = pd.read_csv(run_directory / "dataset_atoms.csv", keep_default_na=False)
    assert list(result.pdb_ids) == ["1MTL"]
    assert list(sites.columns) == list(METAL_SITES_TABLE.column_names)
    assert list(bonds.columns) == list(METAL_BONDS_TABLE.column_names)
    assert len(sites) == 6
    assert len(bonds) == 9
    linked = atoms[atoms[METAL_LINK_COLUMN] != ""]
    assert sorted(linked[METAL_LINK_COLUMN]) == sorted(sites["site_id"])
    assert set(bonds["site_id"]) <= set(sites["site_id"])
