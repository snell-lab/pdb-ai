from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from site_cif import SiteAtom, parse_site_atoms, residue, write_site_cif

from pdbai.engines.analyses.residue_sites import RESIDUE_SITE_ENGINE
from pdbai.engines.analyses.residue_sites.columns import (
    B_FACTOR_RATIO_COLUMN,
    RESIDUE_SITE_LINK_COLUMN,
    RESIDUE_SITES_TABLE,
)
from pdbai.engines.analyses.residue_sites.compute import (
    ResidueSiteComputation,
    compute_structure_residue_sites,
)
from pdbai.engines.analyses.residue_sites.constants import (
    SENSITIVE_CLASS_NAMES,
    SITE_CLASSES,
)
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.compatibility import require_compatible_options
from pdbai.engines.core import EngineContext, EngineOptions
from pdbai.engines.core.tables import validate_tables
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files

_ON = {"include_residue_sites": True}

# An aspartate whose group atoms rise in B-factor away from CB. CB–CG is
# 1.5 Å along x, and the mean B of CG, OD1, and OD2 is three times that of CB.
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
    b_factors={"CB": 10, "CG": 20, "OD1": 30, "OD2": 40},
)


def _compute(tmp_path: Path, atoms: list[SiteAtom]) -> ResidueSiteComputation:
    return compute_structure_residue_sites(
        parse_site_atoms(write_site_cif(tmp_path, atoms))
    )


def _row(computation: ResidueSiteComputation, site_id: str) -> dict[str, Any]:
    """Return one row as plain values by column name."""
    rows = computation.table[computation.table["site_id"] == site_id]
    assert len(rows) == 1
    return {str(name): value for name, value in rows.iloc[0].to_dict().items()}


def test_every_control_names_a_sensitive_class() -> None:
    assert {site.control_for for site in SITE_CLASSES if site.control_for} <= set(
        SENSITIVE_CLASS_NAMES
    )
    for site in SITE_CLASSES:
        assert site.key_atom == site.reference_atoms[0]
        assert {first for first, _second in site.attachments} <= set(
            site.reference_atoms
        )
        assert {second for _first, second in site.attachments} <= set(site.site_atoms)


def test_a_carboxylate_is_set_beside_the_atom_it_is_attached_to(
    tmp_path: Path,
) -> None:
    computation = _compute(tmp_path, _ASP)
    row = _row(computation, "1SIT:1:A:1::CB:")

    assert computation.status == "ok"
    assert row["site_class"] == "asp_carboxylate"
    assert row["is_control"] is False
    assert pd.isna(row["control_for"])
    assert (row["n_site_atoms_expected"], row["n_site_atoms_modelled"]) == (3, 3)
    assert row["site_b_factor_mean_A2"] == pytest.approx(30.0)
    assert row["site_b_factor_max_A2"] == pytest.approx(40.0)
    assert row["reference_b_factor_mean_A2"] == pytest.approx(10.0)
    assert row["b_factor_ratio_to_reference"] == pytest.approx(3.0)
    assert row["attachment_distance_min_A"] == pytest.approx(1.5)
    assert row["site_occupancy_min"] == pytest.approx(1.0)
    assert row["has_alternate_conformer"] is False
    assert row["metal_bound"] is False
    assert math.isnan(row["nearest_metal_distance_A"])
    assert row["site_status"] == "ok"


def test_site_atoms_link_to_their_row(tmp_path: Path) -> None:
    atoms = parse_site_atoms(write_site_cif(tmp_path, _ASP))
    computation = compute_structure_residue_sites(atoms)

    linked = computation.link.dropna()
    assert atoms.loc[linked.index, "atom_name"].tolist() == ["CG", "OD1", "OD2"]
    assert set(linked) == {"1SIT:1:A:1::CB:"}
    assert computation.link.index.equals(atoms.index)


def test_a_side_chain_cut_back_to_its_key_atom_is_still_a_row(tmp_path: Path) -> None:
    glu = residue("GLU", 2, {"CA": (0, 0, 0), "CB": (1.5, 0, 0), "CG": (3, 0, 0)})

    row = _row(_compute(tmp_path, glu), "1SIT:1:A:2::CG:")

    assert row["site_class"] == "glu_carboxylate"
    assert row["n_site_atoms_modelled"] == 0
    assert row["site_status"] == "site_atoms_missing"
    assert math.isnan(row["b_factor_ratio_to_reference"])
    assert math.isnan(row["site_occupancy_min"])


def test_a_residue_without_its_key_atom_is_counted_and_has_no_row(
    tmp_path: Path,
) -> None:
    computation = _compute(
        tmp_path, residue("ASP", 3, {"N": (0, 0, 0), "CA": (1, 0, 0)})
    )

    assert computation.table.empty
    assert computation.status == "no_sites"
    assert computation.without_key_atom_count == 1
    assert len(computation.warnings) == 1


def test_controls_name_the_class_they_stand_beside(tmp_path: Path) -> None:
    atoms = [
        *residue("ASN", 1, {"CB": (0, 0, 0), "CG": (1.5, 0, 0), "OD1": (2, 1, 0)}),
        *residue("GLN", 2, {"CG": (0, 5, 0), "CD": (1.5, 5, 0)}),
        *residue(
            "PHE", 3, {"CE1": (0, 10, 0), "CE2": (2.4, 10, 0), "CZ": (1.2, 10.7, 0)}
        ),
        *residue(
            "LEU", 4, {"CG": (0, 15, 0), "CD1": (1.5, 15, 0), "CD2": (0, 16.5, 0)}
        ),
        *residue("ILE", 5, {"CG1": (0, 20, 0), "CD1": (1.5, 20, 0)}),
        *residue("THR", 6, {"CB": (0, 25, 0), "OG1": (1.4, 25, 0)}),
        *residue("SER", 7, {"CB": (0, 30, 0), "OG": (1.4, 30, 0)}),
    ]

    table = _compute(tmp_path, atoms).table

    assert table["site_class"].tolist() == [
        "asn_amide",
        "gln_amide",
        "phe_para_carbon",
        "leu_delta_carbons",
        "ile_delta_carbon",
        "thr_hydroxyl",
        "ser_hydroxyl",
    ]
    assert table["control_for"].tolist() == [
        "asp_carboxylate",
        "glu_carboxylate",
        "tyr_hydroxyl",
        "met_thioether",
        "met_thioether",
        "tyr_hydroxyl",
        "cys_sulfur",
    ]
    assert table["is_control"].tolist() == [True] * 7
    assert table["atom_name"].tolist() == ["CB", "CG", "CE1", "CG", "CG1", "CB", "CB"]
    assert table["n_site_atoms_modelled"].tolist() == [2, 1, 1, 2, 1, 1, 1]


def test_a_cysteine_sulfur_is_a_row_bonded_or_not(tmp_path: Path) -> None:
    def cysteine(number: int, x: float) -> list[SiteAtom]:
        return residue(
            "CYS",
            number,
            {"CB": (x, 0, 0), "SG": (x, 1.8, 0)},
            b_factors={"CB": 10, "SG": 25},
        )

    # The sulfurs of residues 1 and 2 are 2.0 Å apart, as in a disulfide.
    # Residue 3 is free, and its sulfur lies 2.3 Å from a zinc atom.
    zinc = SiteAtom("ZN", 101, "ZN", (20, 1.8, 2.3), element="ZN", hetatm=True)
    atoms = [*cysteine(1, 0.0), *cysteine(2, 2.0), *cysteine(3, 20.0), zinc]

    computation = _compute(tmp_path, atoms)

    assert computation.table["site_class"].tolist() == ["cys_sulfur"] * 3
    assert computation.table["is_control"].tolist() == [False] * 3
    assert computation.table["metal_bound"].tolist() == [False, False, True]
    row = _row(computation, "1SIT:1:A:1::CB:")
    assert row["b_factor_ratio_to_reference"] == pytest.approx(2.5)
    assert row["attachment_distance_min_A"] == pytest.approx(1.8)


def test_the_methionine_sulfur_is_measured_against_both_carbons(
    tmp_path: Path,
) -> None:
    met = residue(
        "MET",
        1,
        {"CG": (0, 0, 0), "SD": (1.8, 0, 0), "CE": (1.8, 1.9, 0)},
        b_factors={"CG": 10, "SD": 30, "CE": 20},
    )

    row = _row(_compute(tmp_path, met), "1SIT:1:A:1::CG:")

    assert row["site_class"] == "met_thioether"
    assert row["reference_b_factor_mean_A2"] == pytest.approx(15.0)
    assert row["b_factor_ratio_to_reference"] == pytest.approx(2.0)
    assert row["attachment_distance_min_A"] == pytest.approx(1.8)
    assert row["attachment_distance_max_A"] == pytest.approx(1.9)


def test_the_rings_of_trp_and_his_are_groups_beside_the_ring_of_phe(
    tmp_path: Path,
) -> None:
    indole = ("CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2")
    trp = residue(
        "TRP",
        1,
        {
            "CB": (-1.5, 0, 0),
            "CG": (0, 0, 0),
            **{name: (1.4, float(index), 0) for index, name in enumerate(indole)},
        },
        b_factors={"CG": 10, **dict.fromkeys(indole, 15.0)},
    )
    his = residue(
        "HIS",
        2,
        {
            "CG": (0, 20, 0),
            "ND1": (1.4, 20, 0),
            "CD2": (0.4, 21.3, 0),
            "CE1": (2.1, 21, 0),
            "NE2": (1.6, 22, 0),
        },
    )
    phe = residue(
        "PHE",
        3,
        {
            "CG": (0, 30, 0),
            "CD1": (1.4, 30, 0),
            "CD2": (0, 31.4, 0),
            "CE1": (2.1, 31.2, 0),
            "CE2": (0.7, 32.6, 0),
            "CZ": (2.1, 32.6, 0),
        },
    )

    computation = _compute(tmp_path, [*trp, *his, *phe])
    table = computation.table

    assert table[["atom_name", "site_class"]].values.tolist() == [
        ["CG", "trp_indole"],
        ["CG", "his_imidazole"],
        ["CE1", "phe_para_carbon"],
        ["CG", "phe_ring_carbons"],
    ]
    assert table["n_site_atoms_modelled"].tolist() == [8, 4, 1, 4]
    assert table["control_for"].tolist()[2:] == ["tyr_hydroxyl", "his_imidazole"]
    assert table["is_control"].tolist() == [False, False, True, True]
    row = _row(computation, "1SIT:1:A:1::CG:")
    assert row["b_factor_ratio_to_reference"] == pytest.approx(1.5)
    assert row["attachment_distance_min_A"] == pytest.approx(1.4)
    # Each ring atom of Phe links to one row, CZ to the para carbon's.
    assert computation.link.dropna().value_counts().to_dict() == {
        "1SIT:1:A:1::CG:": 8,
        "1SIT:1:A:2::CG:": 4,
        "1SIT:1:A:3::CE1:": 1,
        "1SIT:1:A:3::CG:": 4,
    }


def test_residues_deposited_oxidized_are_rows_for_the_oxygens_they_gained(
    tmp_path: Path,
) -> None:
    def oxidized(comp: str, number: int, bonds: dict[str, str]) -> list[SiteAtom]:
        y = float(number)
        atoms = {"CA": (-3.0, y, 0.0), "CB": (-1.5, y, 0.0)}
        for index, (anchor, oxygen) in enumerate(bonds.items()):
            atoms[anchor] = (float(index), y, 0.0)
            atoms[oxygen] = (float(index), y, 1.4)
        return residue(comp, number, atoms, hetatm=True)

    atoms = [
        *oxidized("SME", 10, {"S": "OE"}),
        *oxidized("MHO", 20, {"SD": "OD1"}),
        *oxidized("TRO", 30, {"CD1": "OD1"}),
        *oxidized("0AF", 40, {"CZ2": "O1"}),
        *oxidized("TOQ", 50, {"CH2": "O6", "CZ2": "O7"}),
        # A hydroperoxide whose second oxygen the model left out.
        *oxidized("TOX", 60, {"NE1": "O1"}),
        *oxidized("KYN", 70, {"C1": "O2"}),
        *oxidized("OHI", 80, {"CE1": "O12"}),
        *oxidized("DAH", 90, {"CE2": "OE2"}),
        *oxidized("MTY", 100, {"CE2": "OH"}),
        *oxidized("OMT", 110, {"SD": "OD1"}),
    ]

    table = _compute(tmp_path, atoms).table

    assert table[["atom_name", "site_class"]].values.tolist() == [
        ["S", "met_sulfoxide"],
        ["SD", "met_sulfoxide"],
        ["CD1", "trp_hydroxylated"],
        ["CZ2", "trp_hydroxylated"],
        ["CH2", "trp_hydroxylated"],
        ["NE1", "trp_hydroperoxide"],
        ["C1", "trp_kynurenine"],
        ["CE1", "his_oxo"],
        ["CE2", "tyr_dopa"],
        ["CE2", "phe_hydroxylated"],
        ["SD", "met_sulfone"],
    ]
    assert table["n_site_atoms_expected"].tolist() == [1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 2]
    assert table["n_site_atoms_modelled"].tolist() == [1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1]
    assert table["is_control"].tolist() == [False] * 11


def test_a_site_atom_at_zero_occupancy_gives_no_b_factor(tmp_path: Path) -> None:
    tyr = residue(
        "TYR", 1, {"CZ": (0, 0, 0), "OH": (1.4, 0, 0)}, occupancies={"OH": 0.0}
    )

    row = _row(_compute(tmp_path, tyr), "1SIT:1:A:1::CZ:")

    assert row["site_class"] == "tyr_hydroxyl"
    assert row["n_site_atoms_modelled"] == 1
    assert row["n_site_atoms_zero_occupancy"] == 1
    assert row["site_occupancy_min"] == pytest.approx(0.0)
    assert math.isnan(row["site_b_factor_mean_A2"])
    assert row["site_status"] == "zero_occupancy"


def test_a_c_terminus_is_a_row_where_the_file_models_its_second_oxygen(
    tmp_path: Path,
) -> None:
    backbone = {"CA": (0.0, 0.0, 0.0), "C": (1.5, 0.0, 0.0), "O": (2.1, 1.0, 0.0)}
    atoms = [
        *residue("ALA", 1, backbone),
        *residue("ASP", 2, {**backbone, "OXT": (2.1, -1.0, 0.0), "CB": (0, 0, 9)}),
    ]

    table = _compute(tmp_path, atoms).table

    assert table[["res_num", "atom_name", "site_class"]].values.tolist() == [
        ["2", "CB", "asp_carboxylate"],
        ["2", "CA", "c_terminal_carboxylate"],
    ]
    assert table["n_site_atoms_modelled"].tolist() == [0, 3]


def test_a_group_bound_to_a_metal_is_flagged(tmp_path: Path) -> None:
    zinc = SiteAtom("ZN", 101, "ZN", (4.1, 1, 0), element="ZN", hetatm=True)

    row = _row(_compute(tmp_path, [*_ASP, zinc]), "1SIT:1:A:1::CB:")

    assert row["nearest_metal_distance_A"] == pytest.approx(2.0)
    assert row["metal_bound"] is True


def test_a_free_amino_acid_bound_as_a_ligand_is_not_read(tmp_path: Path) -> None:
    ligand = [dataclasses.replace(atom, hetatm=True, seq=None) for atom in _ASP]

    assert _compute(tmp_path, ligand).table.empty


def test_each_model_and_alternate_conformer_is_read_as_the_atom_table_keeps_it(
    tmp_path: Path,
) -> None:
    second = [dataclasses.replace(atom, model=2) for atom in _ASP]
    disordered = [
        dataclasses.replace(atom, altloc="A", occupancy=0.6)
        if atom.name == "OD1"
        else atom
        for atom in _ASP
    ]

    computation = _compute(tmp_path, [*disordered, *second])

    assert computation.table["site_id"].tolist() == [
        "1SIT:1:A:1::CB:",
        "1SIT:2:A:1::CB:",
    ]
    first = _row(computation, "1SIT:1:A:1::CB:")
    assert first["has_alternate_conformer"] is True
    assert first["site_occupancy_min"] == pytest.approx(0.6)
    assert _row(computation, "1SIT:2:A:1::CB:")["has_alternate_conformer"] is False


def test_an_oxidized_cysteine_is_read_as_a_modified_residue(tmp_path: Path) -> None:
    def oxidized(comp: str, number: int, oxygens: tuple[str, ...]) -> list[SiteAtom]:
        y = float(number)
        atoms = {"CB": (0.0, y, 0.0), "SG": (1.8, y, 0.0)}
        atoms |= {name: (2.5, y, 1.0 + index) for index, name in enumerate(oxygens)}
        return residue(comp, number, atoms, b_factors={"SG": 10}, hetatm=True)

    atoms = [
        *oxidized("CSO", 10, ("OD",)),
        *oxidized("CSD", 20, ("OD1", "OD2")),
        *oxidized("OCS", 30, ("OD1", "OD2", "OD3")),
        # The second sulfenic acid has lost its oxygen in the model.
        *oxidized("CSO", 40, ()),
    ]

    computation = _compute(tmp_path, atoms)

    assert computation.table["site_class"].tolist() == [
        "cys_sulfenic_acid",
        "cys_sulfinic_acid",
        "cys_sulfonic_acid",
        "cys_sulfenic_acid",
    ]
    assert computation.table["atom_name"].tolist() == ["SG"] * 4
    assert computation.table["n_site_atoms_modelled"].tolist() == [1, 2, 3, 0]
    assert computation.table["is_control"].tolist() == [False] * 4
    first = _row(computation, "1SIT:1:A:10::SG:")
    assert first["b_factor_ratio_to_reference"] == pytest.approx(2.0)
    assert _row(computation, "1SIT:1:A:40::SG:")["site_status"] == "site_atoms_missing"


def test_an_oxidized_cysteine_outside_a_polymer_is_not_read(tmp_path: Path) -> None:
    ligand = residue(
        "CSO",
        10,
        {"CB": (0, 0, 0), "SG": (1.8, 0, 0), "OD": (2.5, 0, 1)},
        polymer=False,
        hetatm=True,
    )

    assert _compute(tmp_path, ligand).table.empty


def test_every_atom_is_set_beside_the_atom_it_hangs_from(tmp_path: Path) -> None:
    atoms = parse_site_atoms(write_site_cif(tmp_path, _ASP))
    computation = compute_structure_residue_sites(atoms)

    ratio = dict(zip(atoms["atom_name"], computation.b_factor_ratio, strict=True))
    # The CA has no parent. CB hangs from CA, CG from CB, each oxygen from CG.
    assert math.isnan(ratio["CA"])
    assert ratio["CB"] == pytest.approx(10 / 20)
    assert ratio["CG"] == pytest.approx(20 / 10)
    assert ratio["OD1"] == pytest.approx(30 / 20)
    assert ratio["OD2"] == pytest.approx(40 / 20)
    assert computation.b_factor_ratio.index.equals(atoms.index)


def test_an_atom_that_closes_a_ring_is_set_beside_both_its_parents(
    tmp_path: Path,
) -> None:
    ring = {"CE1": (0.0, 0.0, 0.0), "CE2": (2.4, 0.0, 0.0), "CZ": (1.2, 0.7, 0.0)}
    phe = residue("PHE", 1, ring, b_factors={"CE1": 10, "CE2": 30, "CZ": 40})
    one_parent = residue(
        "PHE",
        2,
        {"CE1": (0, 9, 0), "CZ": (1.2, 9.7, 0)},
        b_factors={"CE1": 10, "CZ": 40},
    )
    atoms = parse_site_atoms(write_site_cif(tmp_path, [*phe, *one_parent]))

    ratio = compute_structure_residue_sites(atoms).b_factor_ratio

    by_atom = dict(
        zip(zip(atoms["res_num"], atoms["atom_name"], strict=True), ratio, strict=True)
    )
    assert by_atom[("1", "CZ")] == pytest.approx(40 / 20)
    # Where the residue holds one of the two, the ratio is taken against it.
    assert by_atom[("2", "CZ")] == pytest.approx(40 / 10)
    # CE1 hangs from CD1, which neither residue holds.
    assert math.isnan(by_atom[("1", "CE1")])


def test_no_ratio_is_given_without_a_usable_atom_and_parent(tmp_path: Path) -> None:
    atoms = [
        # The parent of OH is at zero occupancy.
        *residue(
            "TYR", 1, {"CZ": (0, 0, 0), "OH": (1.4, 0, 0)}, occupancies={"CZ": 0.0}
        ),
        # A B-factor of zero is a placeholder.
        *residue("SER", 2, {"CB": (0, 5, 0), "OG": (1.4, 5, 0)}, b_factors={"CB": 0.0}),
        # A free amino acid bound as a ligand is not read.
        *residue(
            "SER", 3, {"CB": (0, 9, 0), "OG": (1.4, 9, 0)}, polymer=False, hetatm=True
        ),
        # Neither is a component that is not a standard amino acid.
        *residue("CSO", 4, {"CB": (0, 13, 0), "SG": (1.8, 13, 0)}, hetatm=True),
    ]
    table = parse_site_atoms(write_site_cif(tmp_path, atoms))

    assert compute_structure_residue_sites(table).b_factor_ratio.isna().all()


def test_the_table_fits_its_declaration(tmp_path: Path) -> None:
    computation = _compute(tmp_path, _ASP)

    fixed = validate_tables(
        "residue_sites", [RESIDUE_SITES_TABLE], {"residue_sites": computation.table}
    )["residue_sites"]

    assert list(fixed.columns) == list(RESIDUE_SITES_TABLE.column_names)
    assert len(fixed) == 1


def _context(path: Path) -> EngineContext:
    return EngineContext(
        path.stem, parse_site_atoms(path), EngineOptions(values=_ON), source="upload"
    )


def test_the_engine_reports_the_structure(tmp_path: Path) -> None:
    atoms = [*_ASP, *residue("ASP", 3, {"N": (9, 9, 9), "CA": (10, 9, 9)})]

    result = RESIDUE_SITE_ENGINE.run(_context(write_site_cif(tmp_path, atoms)))

    assert set(result.tables) == {"residue_sites"}
    assert result.columns is not None
    assert list(result.columns.columns) == [
        RESIDUE_SITE_LINK_COLUMN,
        B_FACTOR_RATIO_COLUMN,
    ]
    assert result.report["residue_sites_status"] == "ok"
    assert result.report["residue_site_without_key_atom_count"] == 1
    assert result.warnings and result.warnings[0].startswith("Residue sites: 1sit: ")


def test_a_structure_without_sites_leaves_its_table_out(tmp_path: Path) -> None:
    glycine = residue("GLY", 1, {"N": (0, 0, 0), "CA": (1.5, 0, 0)})

    result = RESIDUE_SITE_ENGINE.run(_context(write_site_cif(tmp_path, glycine)))

    assert result.tables == {}
    assert result.report["residue_sites_status"] == "no_sites"
    assert result.warnings == ()


def test_the_engine_is_off_by_default_and_runs_for_every_technique() -> None:
    registry = get_engine_registry()

    assert (
        registry.with_defaults(EngineOptions()).flag("include_residue_sites") is False
    )
    assert RESIDUE_SITE_ENGINE.spec.techniques is None
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
        [write_site_cif(tmp_path, _ASP)],
        working_directory,
        run_id="residues",
        engine_options=dataclasses.replace(
            offline_options, dataset_shape="star_schema"
        ).with_values(_ON),
    )

    run_directory = working_directory / "runs" / "residues"
    sites = pd.read_csv(
        run_directory / "dataset_residue_sites.csv", keep_default_na=False
    )
    atoms = pd.read_csv(run_directory / "dataset_atoms.csv", keep_default_na=False)
    assert list(sites.columns) == list(RESIDUE_SITES_TABLE.column_names)
    assert sites["site_id"].tolist() == ["1SIT:1:A:1::CB:"]
    linked = atoms[atoms[RESIDUE_SITE_LINK_COLUMN] != ""]
    assert linked["atom_name"].tolist() == ["CG", "OD1", "OD2"]
    assert set(linked[RESIDUE_SITE_LINK_COLUMN]) == set(sites["site_id"])
    ratios = atoms.set_index("atom_name")[B_FACTOR_RATIO_COLUMN]
    assert float(ratios["CG"]) == pytest.approx(2.0)
    assert ratios["CA"] == ""
