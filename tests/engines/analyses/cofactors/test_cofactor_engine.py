from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
import pandas as pd
import pytest
from site_cif import SiteAtom, parse_site_atoms, residue, write_site_cif

from pdbai.engines.analyses.cofactors import COFACTOR_ENGINE
from pdbai.engines.analyses.cofactors.columns import (
    COFACTOR_LINK_COLUMN,
    COFACTOR_SITES_TABLE,
)
from pdbai.engines.analyses.cofactors.compute import (
    CofactorComputation,
    classify_component,
    compute_structure_cofactors,
)
from pdbai.engines.analyses.cofactors.constants import (
    COFACTOR_CLASSES,
    TORSION_COLUMNS,
)
from pdbai.engines.analyses.site_geometry import fit_plane, torsion_deg
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.compatibility import require_compatible_options
from pdbai.engines.core import EngineContext, EngineOptions
from pdbai.engines.core.tables import validate_tables
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files
from pdbai.profiles import get_profile

_ON = {"include_cofactors": True}

Xyz = tuple[float, float, float]

# The atom names of the isoalloxazine ring as FMN gives them, the pyrimidine
# ring and then the benzene ring, with N5 and N10 between the two.
_PYRIMIDINE = ("N1", "C2", "N3", "C4", "C4A", "C10")
_BENZENE = ("C5A", "C6", "C7", "C8", "C9", "C9A")


def _ring(names: tuple[str, ...], centre_x: float, tilt_deg: float) -> dict[str, Xyz]:
    """Return a hexagon about (centre_x, 0, 0), tilted about the x axis."""
    tilt = math.radians(tilt_deg)
    ring: dict[str, Xyz] = {}
    for index, name in enumerate(names):
        angle = math.radians(60 * index)
        x, y = centre_x + 1.4 * math.cos(angle), 1.4 * math.sin(angle)
        ring[name] = (x, y * math.cos(tilt), y * math.sin(tilt))
    return ring


def _flavin(tilt_deg: float = 0.0, comp: str = "FMN") -> list[SiteAtom]:
    """Return a flavin whose benzene ring is tilted from its pyrimidine ring."""
    names = {"C4A": "C4X", "C5A": "C5X"} if comp == "FAD" else {}
    atoms = {
        **_ring(_PYRIMIDINE, -2.5, 0.0),
        **_ring(_BENZENE, 2.5, tilt_deg),
        "N5": (0.0, 1.2, 0.0),
        "N10": (0.0, -1.2, 0.0),
        "C7M": (5.0, 1.5, 0.0),
        "C8M": (5.0, -1.5, 0.0),
        "C1'": (0.0, -2.7, 0.0),
    }
    return residue(
        comp,
        401,
        {names.get(name, name): xyz for name, xyz in atoms.items()},
        b_factors={"C1'": 40.0},
        polymer=False,
        hetatm=True,
    )


def _heme(metal: str | None = "FE", height: float = 0.4) -> list[SiteAtom]:
    """Return a macrocycle with its nitrogens 2 Å from the axis the metal sits on."""
    atoms: dict[str, Xyz] = {
        "NA": (2, 0, 0),
        "NB": (0, 2, 0),
        "NC": (-2, 0, 0),
        "ND": (0, -2, 0),
        "CHA": (3, 3, 0),
        "CHB": (-3, 3, 0),
        "CHC": (-3, -3, 0),
        "CHD": (3, -3, 0),
    }
    elements: dict[str, str] = {}
    if metal is not None:
        atoms[metal] = (0, 0, height)
        elements[metal] = metal
    return residue(
        "HEM" if metal == "FE" else "CLA",
        501,
        atoms,
        elements=elements,
        polymer=False,
        hetatm=True,
    )


def _retinal(c13_c14_deg: float = 180.0) -> list[SiteAtom]:
    """Return the chain of a retinal from C10, twisted about its C13=C14 bond."""
    twist = math.radians(c13_c14_deg)
    atoms: dict[str, Xyz] = {
        "C10": (1.0, 2.0, 0.0),
        "C11": (1.0, 1.0, 0.0),
        "C12": (0.0, 1.0, 0.0),
        "C13": (0.0, 0.0, 0.0),
        "C14": (0.0, 0.0, 1.4),
        "C15": (-math.sin(twist), math.cos(twist), 1.4),
    }
    return residue("RET", 301, atoms, polymer=False, hetatm=True)


def _compute(tmp_path: Path, atoms: list[SiteAtom]) -> CofactorComputation:
    return compute_structure_cofactors(
        parse_site_atoms(write_site_cif(tmp_path, atoms))
    )


def _only_row(computation: CofactorComputation) -> dict[str, Any]:
    """Return the table's one row as plain values by column name."""
    assert len(computation.table) == 1
    return {
        str(name): value for name, value in computation.table.iloc[0].to_dict().items()
    }


def test_every_class_can_be_named_and_measured() -> None:
    for cofactor in COFACTOR_CLASSES:
        group = {name for names in cofactor.group for name in names}
        assert cofactor.key_atom in cofactor.signature
        assert cofactor.key_atom in group
    assert len(set(TORSION_COLUMNS)) == len(TORSION_COLUMNS)
    assert set(TORSION_COLUMNS) <= set(COFACTOR_SITES_TABLE.column_names)


@pytest.mark.parametrize(
    ("res_name", "names", "in_polymer", "expected"),
    [
        ("FAD", {"N5", "N10", "C9A", "C10", "C7M", "C8M", "C4X"}, False, "flavin"),
        ("XYZ", {"N5", "N10", "C9A", "C10", "C7M", "C8M"}, False, "flavin"),
        (
            "HEM",
            {"NA", "NB", "NC", "ND", "CHA", "CHB", "CHC", "CHD"},
            False,
            "tetrapyrrole",
        ),
        ("CRO", {"C1", "N2", "N3", "C2", "CA2", "CB2", "CG2"}, True, "fp_chromophore"),
        ("CRO", {"C1", "N2", "N3", "C2", "CA2", "CB2", "CG2"}, False, None),
        ("RET", {"C11", "C12", "C13", "C14", "C15"}, False, "retinal"),
        ("BCR", {"C11", "C12", "C13", "C14", "C15"}, False, None),
        ("HC4", {"C1", "C2", "C3", "C1'"}, False, "coumaric_acid"),
        ("FAD", {"N5", "N10", "C9A", "C10"}, False, None),
    ],
)
def test_a_cofactor_is_recognized_by_its_atom_names(
    res_name: str, names: set[str], in_polymer: bool, expected: str | None
) -> None:
    found = classify_component(res_name, names, in_polymer=in_polymer)

    assert (found.name if found else None) == expected


@pytest.mark.parametrize("angle", [0.0, 90.0, -90.0, 180.0, 37.5, -142.0])
def test_a_torsion_agrees_with_gemmi(angle: float) -> None:
    turn = math.radians(angle)
    points = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [math.cos(turn), math.sin(turn), 1.0],
        ]
    )

    expected = math.degrees(
        gemmi.calculate_dihedral(*(gemmi.Position(*point) for point in points))
    )

    assert torsion_deg(points) == pytest.approx(angle)
    assert torsion_deg(points) == pytest.approx(
        180.0 if expected == -180.0 else expected
    )


def test_a_torsion_through_a_straight_line_is_missing() -> None:
    points = np.array([[0.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 1, 0]])

    assert math.isnan(torsion_deg(points))


def test_a_plane_gives_distances_from_itself() -> None:
    points = np.array([[2.0, 0, 0.1], [0, 2.0, -0.1], [-2.0, 0, 0.1], [0, -2.0, -0.1]])

    plane = fit_plane(points)

    assert plane is not None
    assert plane.rmsd == pytest.approx(0.1)
    assert plane.distance(np.array([0.0, 0.0, 0.7])) == pytest.approx(0.7)
    assert fit_plane(points[:2]) is None


def test_a_flat_flavin_has_no_butterfly_bend(tmp_path: Path) -> None:
    row = _only_row(_compute(tmp_path, _flavin()))

    assert row["site_id"] == "1SIT:1:A:401::N5:"
    assert row["cofactor_class"] == "flavin"
    assert (row["n_group_atoms_expected"], row["n_group_atoms_modelled"]) == (14, 14)
    assert row["component_atom_count"] == 17
    assert row["flavin_butterfly_angle_deg"] == pytest.approx(0.0, abs=1e-6)
    assert row["group_planarity_rmsd_A"] == pytest.approx(0.0, abs=1e-6)
    assert row["group_b_factor_mean_A2"] == pytest.approx(20.0)
    assert row["rest_b_factor_mean_A2"] == pytest.approx(80 / 3)
    assert row["b_factor_ratio_to_rest"] == pytest.approx(0.75)
    assert math.isnan(row["chromophore_tau_deg"])
    assert math.isnan(row["metal_out_of_plane_A"])
    assert row["cofactor_status"] == "ok"


@pytest.mark.parametrize("comp", ["FMN", "FAD"])
def test_a_bent_flavin_gives_the_angle_between_its_outer_rings(
    tmp_path: Path, comp: str
) -> None:
    row = _only_row(_compute(tmp_path, _flavin(20.0, comp)))

    assert row["n_group_atoms_modelled"] == 14
    # The file holds four decimals of each coordinate.
    assert row["flavin_butterfly_angle_deg"] == pytest.approx(20.0, abs=0.01)
    assert row["group_planarity_rmsd_A"] > 0.1


def test_the_adenine_carbons_of_fad_are_not_taken_for_ring_atoms(
    tmp_path: Path,
) -> None:
    adenine = [
        SiteAtom("FAD", 401, name, (30.0, 30.0 + offset, 30.0), hetatm=True)
        for offset, name in enumerate(("C4A", "C5A"))
    ]

    row = _only_row(_compute(tmp_path, [*_flavin(20.0, "FAD"), *adenine]))

    assert row["flavin_butterfly_angle_deg"] == pytest.approx(20.0, abs=0.01)


def test_a_flavin_without_a_ring_atom_has_no_angle(tmp_path: Path) -> None:
    atoms = [atom for atom in _flavin(20.0) if atom.name != "C7"]

    row = _only_row(_compute(tmp_path, atoms))

    assert row["n_group_atoms_modelled"] == 13
    assert math.isnan(row["flavin_butterfly_angle_deg"])
    assert row["cofactor_status"] == "group_atoms_missing"


def test_a_heme_metal_is_measured_from_the_plane_of_its_nitrogens(
    tmp_path: Path,
) -> None:
    row = _only_row(_compute(tmp_path, _heme()))

    assert row["site_id"] == "1SIT:1:A:501::NA:"
    assert row["cofactor_class"] == "tetrapyrrole"
    assert row["central_metal_element"] == "FE"
    assert row["central_metal_site_id"] == "1SIT:1:A:501::FE:"
    assert row["metal_out_of_plane_A"] == pytest.approx(0.4)
    assert row["metal_nitrogen_distance_mean_A"] == pytest.approx(math.sqrt(4.16))
    assert math.isnan(row["flavin_butterfly_angle_deg"])
    assert row["cofactor_status"] == "group_atoms_missing"


def test_a_macrocycle_without_a_metal_has_no_metal_measures(tmp_path: Path) -> None:
    row = _only_row(_compute(tmp_path, _heme(metal=None)))

    assert pd.isna(row["central_metal_element"])
    assert pd.isna(row["central_metal_site_id"])
    assert math.isnan(row["metal_out_of_plane_A"])
    assert math.isnan(row["metal_nitrogen_distance_mean_A"])


def test_a_chlorophyll_names_its_magnesium(tmp_path: Path) -> None:
    row = _only_row(_compute(tmp_path, _heme("MG", 0.0)))

    assert row["central_metal_element"] == "MG"
    assert row["metal_out_of_plane_A"] == pytest.approx(0.0, abs=1e-9)
    assert row["metal_nitrogen_distance_mean_A"] == pytest.approx(2.0)


@pytest.mark.parametrize("angle", [180.0, 0.0, -150.0])
def test_a_retinal_gives_the_torsion_about_its_isomerizable_bonds(
    tmp_path: Path, angle: float
) -> None:
    lysine = residue("LYS", 216, {"NZ": (0.0, 0.0, 2.7)})

    row = _only_row(_compute(tmp_path, [*_retinal(angle), *lysine]))

    assert row["site_id"] == "1SIT:1:A:301::C15:"
    assert row["cofactor_class"] == "retinal"
    assert row["retinal_c13_c14_torsion_deg"] == pytest.approx(angle, abs=0.01)
    assert row["retinal_c11_c12_torsion_deg"] == pytest.approx(180.0)
    assert row["protein_link_distance_A"] == pytest.approx(
        math.sqrt(1 + 1.3**2), abs=1e-3
    )


def test_a_link_is_missing_beyond_the_search_radius(tmp_path: Path) -> None:
    lysine = residue("LYS", 216, {"NZ": (0.0, 0.0, 9.0)})

    row = _only_row(_compute(tmp_path, [*_retinal(), *lysine]))

    assert math.isnan(row["protein_link_distance_A"])


def test_a_chromophore_gives_the_twist_of_its_bridge(tmp_path: Path) -> None:
    chromophore = residue(
        "CRO",
        66,
        {
            "N2": (1.0, 0.0, 0.0),
            "CA2": (0.0, 0.0, 0.0),
            "CB2": (0.0, 0.0, 1.4),
            "CG2": (1.0, 0.0, 1.4),
            "CD2": (1.0, 1.0, 1.4),
            "C1": (2.0, 0.0, -1.0),
            "N3": (1.0, 0.0, -2.0),
            "C2": (0.0, 0.0, -1.4),
        },
        hetatm=True,
    )

    row = _only_row(_compute(tmp_path, chromophore))

    assert row["site_id"] == "1SIT:1:A:66::CB2:"
    assert row["cofactor_class"] == "fp_chromophore"
    assert row["chromophore_tau_deg"] == pytest.approx(0.0, abs=1e-6)
    # CD2 stands in where the component has no CD1.
    assert row["chromophore_phi_deg"] == pytest.approx(90.0)
    assert math.isnan(row["retinal_c13_c14_torsion_deg"])


def test_group_atoms_link_to_their_row_and_hydrogens_do_not(tmp_path: Path) -> None:
    hydrogen = SiteAtom("FMN", 401, "H6", (3.2, 2.2, 0.0), element="H", hetatm=True)
    atoms = parse_site_atoms(write_site_cif(tmp_path, [*_flavin(), hydrogen]))
    computation = compute_structure_cofactors(atoms)

    linked = computation.link.dropna()
    assert len(linked) == 14
    assert set(linked) == {"1SIT:1:A:401::N5:"}
    assert "H6" not in set(atoms.loc[linked.index, "atom_name"])
    assert _only_row(computation)["component_atom_count"] == 17


def test_standard_residues_and_small_components_are_never_cofactors(
    tmp_path: Path,
) -> None:
    atoms = [
        *residue("ALA", 1, {"N": (0, 0, 0), "CA": (1.5, 0, 0), "C": (3, 0, 0)}),
        SiteAtom("HOH", 601, "O", (9, 9, 9), hetatm=True),
    ]
    computation = _compute(tmp_path, atoms)

    assert computation.table.empty
    assert computation.status == "no_cofactors"


def test_the_table_fits_its_declaration(tmp_path: Path) -> None:
    computation = _compute(tmp_path, [*_flavin(), *_heme(), *_retinal()])

    fixed = validate_tables(
        "cofactors", [COFACTOR_SITES_TABLE], {"cofactor_sites": computation.table}
    )["cofactor_sites"]

    assert list(fixed.columns) == list(COFACTOR_SITES_TABLE.column_names)
    assert fixed["cofactor_class"].tolist() == ["flavin", "tetrapyrrole", "retinal"]
    assert fixed["central_metal_site_id"].isna().tolist() == [True, False, True]


def _context(path: Path) -> EngineContext:
    return EngineContext(
        path.stem, parse_site_atoms(path), EngineOptions(values=_ON), source="upload"
    )


def test_the_engine_reports_the_structure(tmp_path: Path) -> None:
    result = COFACTOR_ENGINE.run(_context(write_site_cif(tmp_path, _flavin())))

    assert set(result.tables) == {"cofactor_sites"}
    assert result.columns is not None
    assert list(result.columns.columns) == [COFACTOR_LINK_COLUMN]
    assert result.report["cofactors_status"] == "ok"


def test_a_structure_without_cofactors_leaves_its_table_out(tmp_path: Path) -> None:
    alanine = residue("ALA", 1, {"N": (0, 0, 0), "CA": (1.5, 0, 0)})

    result = COFACTOR_ENGINE.run(_context(write_site_cif(tmp_path, alanine)))

    assert result.tables == {}
    assert result.report["cofactors_status"] == "no_cofactors"


def test_the_engine_is_off_by_default_and_runs_for_every_technique() -> None:
    registry = get_engine_registry()

    assert registry.with_defaults(EngineOptions()).flag("include_cofactors") is False
    assert COFACTOR_ENGINE.spec.techniques is None
    require_compatible_options(
        registry.with_defaults(
            EngineOptions(technique="neutron_diffraction", values=_ON)
        )
    )


def test_only_the_radiation_profile_switches_the_engine_on() -> None:
    assert "include_cofactors" in get_profile("radiation_sensitivity").toggles
    assert "include_cofactors" not in get_profile("research_ready").toggles


def test_a_build_writes_the_table_and_links_atoms_to_it(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    working_directory = tmp_path / "work"

    build_atom_site_dataset_from_mmcif_files(
        [write_site_cif(tmp_path, [*_flavin(20.0), *_heme()])],
        working_directory,
        run_id="cofactors",
        engine_options=dataclasses.replace(
            offline_options, dataset_shape="star_schema"
        ).with_values(_ON),
    )

    run_directory = working_directory / "runs" / "cofactors"
    sites = pd.read_csv(
        run_directory / "dataset_cofactor_sites.csv", keep_default_na=False
    )
    atoms = pd.read_csv(run_directory / "dataset_atoms.csv", keep_default_na=False)
    assert list(sites.columns) == list(COFACTOR_SITES_TABLE.column_names)
    assert sites["site_id"].tolist() == ["1SIT:1:A:401::N5:", "1SIT:1:A:501::NA:"]
    assert float(sites["flavin_butterfly_angle_deg"].iloc[0]) == pytest.approx(
        20.0, abs=0.01
    )
    linked = atoms[atoms[COFACTOR_LINK_COLUMN] != ""]
    assert linked[COFACTOR_LINK_COLUMN].value_counts().to_dict() == {
        "1SIT:1:A:401::N5:": 14,
        "1SIT:1:A:501::NA:": 8,
    }
