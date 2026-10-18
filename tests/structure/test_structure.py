from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest

from pdbai.structure.mmcif_parser import (
    AtomSiteParseResult,
    parse_mmcif_atom_rows,
)

_ATOM_SITE_HEADER = """loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.auth_seq_id
_atom_site.auth_asym_id
_atom_site.pdbx_PDB_model_num
"""


def _write_cif(tmp_path: Path, atom_rows: str, *, preamble: str = "") -> Path:
    path = tmp_path / "1abc.cif"
    path.write_text(
        f"data_1abc\n{preamble}{_ATOM_SITE_HEADER}{atom_rows}",
        encoding="utf-8",
    )
    return path


def _parse(path: Path, metadata: Mapping[str, object]) -> AtomSiteParseResult:
    return parse_mmcif_atom_rows(path, metadata, structure_id="1abc")


def test_mmcif_parser_builds_atom_rows_and_selects_one_conformer(
    minimal_cif: Path,
) -> None:
    result = _parse(
        minimal_cif,
        {
            "radiation_dose_MGy": 6.0,
            "citation_doi": "10.1000/example",
        },
    )

    dataframe = result.dataframe
    assert len(dataframe) == 4
    assert dataframe["_atom_site.id"].tolist() == ["1", "3", "4", "5"]
    assert dataframe["pdb_id"].unique().tolist() == ["1ABC"]
    assert dataframe["resolution"].unique().tolist() == [1.5]
    assert dataframe["unit_cell_volume"].iloc[0] == 6000.0
    assert dataframe["citation_doi"].unique().tolist() == ["10.1000/example"]
    assert dataframe["is_high_radiation_dose"].all()
    protein = dataframe.loc[~dataframe["is_hetatm"]]
    assert protein["residue_mean_bfactor"].unique().tolist() == [30.0]
    assert protein["is_backbone"].all()
    assert bool(dataframe.loc[dataframe["atom_name"] == "ZN", "is_hetatm"].iloc[0])


def test_kept_rows_carry_their_atoms_occupancy_summed_over_conformers(
    minimal_cif: Path,
) -> None:
    # The CA is modelled as A at 0.40 and B at 0.80. B is kept, and its row
    # still records both conformers. Atoms without alternates are their own
    # total.
    dataframe = _parse(minimal_cif, {}).dataframe
    ca = dataframe.loc[dataframe["atom_name"] == "CA"].iloc[0]
    others = dataframe.loc[dataframe["atom_name"] != "CA"]

    assert ca["occupancy"] == 0.8
    assert ca["occupancy_all_conformers"] == pytest.approx(1.2)
    assert ca["conformer_count"] == 2
    assert others["occupancy_all_conformers"].tolist() == others["occupancy"].tolist()
    assert others["conformer_count"].tolist() == [1] * len(others)


def test_parser_uses_rcsb_resolution_when_cif_has_no_resolution(
    minimal_cif: Path,
) -> None:
    text = minimal_cif.read_text(encoding="utf-8")
    minimal_cif.write_text(
        text.replace("_refine.ls_d_res_high 1.50", "_refine.ls_d_res_high ?"),
        encoding="utf-8",
    )

    result = _parse(minimal_cif, {"resolution_rcsb": "2.25"})

    assert result.dataframe["resolution"].unique().tolist() == [2.25]


def test_residue_mean_bfactor_separates_coordinate_models(tmp_path: Path) -> None:

    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
        "ATOM 2 C CA GLY A 1 7 ? 2 0 0 1.00 90 7 X 2\n",
    )

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["residue_mean_bfactor"].tolist() == [10.0, 90.0]


def test_residue_mean_bfactor_excludes_missing_values(tmp_path: Path) -> None:

    cif = _write_cif(
        tmp_path,
        "ATOM 1 C N  GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
        "ATOM 2 C CA GLY A 1 7 ? 2 0 0 1.00  ? 7 X 1\n"
        "ATOM 3 C C  GLY A 1 7 ? 3 0 0 1.00 40 7 X 1\n",
    )

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["residue_mean_bfactor"].unique().tolist() == [25.0]


def test_residue_mean_bfactor_is_missing_when_no_atom_has_one(
    tmp_path: Path,
) -> None:
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C N  GLY A 1 7 ? 1 0 0 1.00 ? 7 X 1\n"
        "ATOM 2 C CA GLY A 1 7 ? 2 0 0 1.00 ? 7 X 1\n",
    )

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["residue_mean_bfactor"].isna().all()


def test_unreported_resolution_and_cell_are_missing_not_zero(
    tmp_path: Path,
) -> None:

    cif = _write_cif(tmp_path, "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n")

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["resolution"].isna().all()
    assert dataframe["unit_cell_volume"].isna().all()
    assert dataframe["solvent_content"].isna().all()


@pytest.mark.parametrize("placeholder", ["?", "."])
def test_cell_placeholders_parse_instead_of_raising(
    tmp_path: Path, placeholder: str
) -> None:

    preamble = (
        f"_cell.length_a {placeholder}\n"
        "_cell.length_b 20\n"
        "_cell.length_c 30\n"
        "_cell.angle_alpha 90\n"
        "_cell.angle_beta 90\n"
        "_cell.angle_gamma 90\n"
    )
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n",
        preamble=preamble,
    )

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["unit_cell_volume"].isna().all()


_ORTHORHOMBIC_CELL = (
    "_cell.length_a 40\n"
    "_cell.length_b 50\n"
    "_cell.length_c 60\n"
    "_cell.angle_alpha 90\n"
    "_cell.angle_beta 90\n"
    "_cell.angle_gamma 90\n"
    "_cell.Z_PDB 8\n"
    "_symmetry.space_group_name_H-M 'P 21 21 21'\n"
)
_TWO_RESIDUES = (
    "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
    "ATOM 2 C CA GLY A 1 8 ? 2 0 0 1.00 10 8 X 1\n"
    "HETATM 3 O O HOH B 2 . ? 3 0 0 1.00 10 1 W 1\n"
)


def test_solvent_content_uses_deposited_percent_first(tmp_path: Path) -> None:

    cif = _write_cif(
        tmp_path,
        _TWO_RESIDUES,
        preamble=(
            f"{_ORTHORHOMBIC_CELL}"
            "_exptl_crystal.density_Matthews 2.68\n"
            "_exptl_crystal.density_percent_sol 54.17\n"
        ),
    )

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["solvent_content"].unique().tolist() == pytest.approx([0.5417])


def test_solvent_content_converts_deposited_matthews_coefficient(
    tmp_path: Path,
) -> None:

    cif = _write_cif(
        tmp_path,
        _TWO_RESIDUES,
        preamble=(
            f"{_ORTHORHOMBIC_CELL}"
            "_exptl_crystal.density_Matthews 2.68\n"
            "_exptl_crystal.density_percent_sol ?\n"
        ),
    )

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["solvent_content"].unique().tolist() == pytest.approx(
        [1.0 - 1.23 / 2.68]
    )


def test_solvent_content_estimate_uses_symmetry_count_not_z_pdb(
    tmp_path: Path,
) -> None:
    # P 21 21 21 has four symmetry operations. Z_PDB, here 8, counts chains
    # per cell and must not enter the estimate. The asymmetric unit holds two
    # polymer residues of 110 Da each. The water is not a polymer residue.

    cif = _write_cif(tmp_path, _TWO_RESIDUES, preamble=_ORTHORHOMBIC_CELL)

    dataframe = _parse(cif, {}).dataframe

    matthews = (40 * 50 * 60) / (4 * 2 * 110.0)
    assert dataframe["solvent_content"].unique().tolist() == pytest.approx(
        [1.0 - 1.23 / matthews]
    )


def test_solvent_content_is_missing_without_symmetry_or_deposited_value(
    tmp_path: Path,
) -> None:

    preamble = _ORTHORHOMBIC_CELL.replace(
        "_symmetry.space_group_name_H-M 'P 21 21 21'\n", ""
    )
    cif = _write_cif(tmp_path, _TWO_RESIDUES, preamble=preamble)

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["unit_cell_volume"].notna().all()
    assert dataframe["solvent_content"].isna().all()


def test_solvent_content_is_missing_rather_than_clipped(tmp_path: Path) -> None:
    # A cell too small for its contents gives a Matthews coefficient below
    # 1.23 Å³/Da, and so a negative solvent fraction. Clipped to 0.0, it would
    # read as a real value.

    preamble = _ORTHORHOMBIC_CELL.replace("_cell.length_a 40", "_cell.length_a 0.2")
    cif = _write_cif(tmp_path, _TWO_RESIDUES, preamble=preamble)

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["solvent_content"].isna().all()


@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        ({}, None),
        ({"radiation_dose_MGy": 1.0}, False),
        ({"radiation_dose_MGy": 9.0}, True),
    ],
)
def test_radiation_flags_distinguish_unreported_from_below_threshold(
    tmp_path: Path, metadata: dict[str, object], expected: bool | None
) -> None:

    cif = _write_cif(tmp_path, "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n")

    dataframe = _parse(cif, metadata).dataframe

    flag = dataframe["is_high_radiation_dose"]
    assert isinstance(flag.dtype, pd.BooleanDtype)
    if expected is None:
        assert flag.isna().all()
    else:
        assert flag.tolist() == [expected]


def test_multi_row_non_atom_categories_are_not_broadcast(tmp_path: Path) -> None:

    preamble = """loop_
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.mon_id
1 GLY
2 ALA
3 SER

"""
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
        "ATOM 2 C CB GLY A 1 7 ? 2 0 0 1.00 20 7 X 1\n",
        preamble=preamble,
    )

    dataframe = _parse(cif, {}).dataframe

    assert "_pdbx_poly_seq_scheme.mon_id" not in dataframe.columns
    assert "_pdbx_poly_seq_scheme.seq_id" not in dataframe.columns
    # Nor may the rows be joined into one semicolon-separated text value.
    assert not any(
        isinstance(value, str) and ";" in value
        for value in dataframe.iloc[0].tolist()
        if isinstance(value, str)
    )


def test_single_value_non_atom_categories_still_broadcast(tmp_path: Path) -> None:

    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
        "ATOM 2 C CB GLY A 1 7 ? 2 0 0 1.00 20 7 X 1\n",
        preamble="_exptl.method 'X-RAY DIFFRACTION'\n",
    )

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["_exptl.method"].tolist() == [
        "X-RAY DIFFRACTION",
        "X-RAY DIFFRACTION",
    ]


def test_cell_measurement_items_broadcast_like_other_entry_items(
    tmp_path: Path,
) -> None:
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
        "ATOM 2 C CB GLY A 1 7 ? 2 0 0 1.00 20 7 X 1\n",
        preamble="_cell_measurement.temp 100\n_cell_measurement.reflns_used 5120\n",
    )

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["_cell_measurement.temp"].tolist() == ["100", "100"]
    assert dataframe["_cell_measurement.reflns_used"].tolist() == ["5120", "5120"]


def test_parser_reports_every_file_method_even_when_not_broadcast(
    tmp_path: Path,
) -> None:
    preamble = """loop_
_exptl.entry_id
_exptl.method
1abc 'X-RAY DIFFRACTION'
1abc 'NEUTRON DIFFRACTION'

"""
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n",
        preamble=preamble,
    )

    result = _parse(cif, {})

    # The two-row loop is not broadcast onto atoms, but the parse result still
    # carries both methods so intake can classify a joint refinement.
    assert "_exptl.method" not in result.dataframe.columns
    assert result.experimental_methods == ("X-RAY DIFFRACTION", "NEUTRON DIFFRACTION")
    assert result.is_integrative is False
    assert result.dataframe["experimental_methods"].tolist() == [
        "X-RAY DIFFRACTION; NEUTRON DIFFRACTION"
    ]
    assert result.dataframe["is_multi_method"].tolist() == [True]


def test_rcsb_methods_take_precedence_over_the_file_for_the_columns(
    tmp_path: Path,
) -> None:
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n",
        preamble="_exptl.method 'X-RAY DIFFRACTION'\n",
    )

    from_rcsb = _parse(cif, {"experimental_methods": ["ELECTRON MICROSCOPY"]}).dataframe
    from_file = _parse(cif, {}).dataframe
    unknown = _parse(
        _write_cif(tmp_path, "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"), {}
    ).dataframe

    assert from_rcsb["experimental_methods"].tolist() == ["ELECTRON MICROSCOPY"]
    assert from_file["experimental_methods"].tolist() == ["X-RAY DIFFRACTION"]
    assert from_file["is_multi_method"].tolist() == [False]
    assert bool(unknown["experimental_methods"].isna().all())
    assert bool(unknown["is_multi_method"].isna().all())


def test_radiation_sources_are_normalized_and_prefer_rcsb(tmp_path: Path) -> None:
    preamble = """loop_
_diffrn_source.diffrn_id
_diffrn_source.source
1 'Rotating Copper Anode'
2 'NUCLEAR REACTOR'

"""
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n",
        preamble=preamble,
    )

    from_file = _parse(cif, {})
    from_rcsb = _parse(cif, {"radiation_sources": ["free-electron laser"]}).dataframe
    none = _parse(
        _write_cif(tmp_path, "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"), {}
    ).dataframe

    assert from_file.radiation_sources == ("Rotating Copper Anode", "NUCLEAR REACTOR")
    assert from_file.dataframe["radiation_sources"].tolist() == [
        "ROTATING ANODE; NUCLEAR REACTOR"
    ]
    assert from_rcsb["radiation_sources"].tolist() == ["FREE ELECTRON LASER"]
    assert bool(none["radiation_sources"].isna().all())


def test_integrative_models_are_recognized_by_their_ihm_categories(
    tmp_path: Path,
) -> None:
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n",
        preamble="_ihm_model_list.model_id 1\n",
    )

    result = _parse(cif, {})

    assert result.is_integrative is True
    assert result.experimental_methods == ()


def test_matching_row_count_does_not_align_a_category_onto_atoms(
    tmp_path: Path,
) -> None:

    preamble = """loop_
_entity.id
_entity.type
1 polymer
2 non-polymer

"""
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
        "ATOM 2 C CB GLY A 1 7 ? 2 0 0 1.00 20 7 X 1\n",
        preamble=preamble,
    )

    dataframe = _parse(cif, {}).dataframe

    assert "_entity.type" not in dataframe.columns

    assert dataframe["_pdbai_entity_type"].tolist() == ["polymer", "polymer"]


def test_b_anisotropy_is_computed_for_valid_tensors(tmp_path: Path) -> None:

    preamble = """loop_
_atom_site_anisotrop.id
_atom_site_anisotrop.U[1][1]
_atom_site_anisotrop.U[2][2]
_atom_site_anisotrop.U[3][3]
_atom_site_anisotrop.U[1][2]
_atom_site_anisotrop.U[1][3]
_atom_site_anisotrop.U[2][3]
1 0.4 0.2 0.1 0.0 0.0 0.0

"""
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
        "ATOM 2 C CB GLY A 1 7 ? 2 0 0 1.00 20 7 X 1\n",
        preamble=preamble,
    )

    dataframe = _parse(cif, {}).dataframe

    first = dataframe.loc[dataframe["_atom_site.id"] == "1"].iloc[0]
    assert first["u_equivalent"] == pytest.approx((0.4 + 0.2 + 0.1) / 3.0)

    assert first["b_anisotropy"] == pytest.approx(4.0)
    assert pd.isna(
        dataframe.loc[dataframe["_atom_site.id"] == "2", "b_anisotropy"].iloc[0]
    )
    # A one-row category is otherwise repeated on every atom. The anisotropic
    # loop describes single atoms, so its one row is joined to atom 1 and its
    # items do not become columns.
    assert not dataframe.columns.str.startswith("_atom_site_anisotrop.").any()


def test_repeated_anisotrop_id_does_not_duplicate_atom_rows(
    tmp_path: Path,
) -> None:

    preamble = """loop_
_atom_site_anisotrop.id
_atom_site_anisotrop.U[1][1]
_atom_site_anisotrop.U[2][2]
_atom_site_anisotrop.U[3][3]
_atom_site_anisotrop.U[1][2]
_atom_site_anisotrop.U[1][3]
_atom_site_anisotrop.U[2][3]
1 0.1 0.1 0.1 0.0 0.0 0.0
1 0.9 0.9 0.9 0.0 0.0 0.0
2 0.2 0.2 0.2 0.0 0.0 0.0

"""
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
        "ATOM 2 C CB GLY A 1 7 ? 2 0 0 1.00 20 7 X 1\n",
        preamble=preamble,
    )

    dataframe = _parse(cif, {}).dataframe

    assert len(dataframe) == 2
    assert dataframe["_atom_site.id"].tolist() == ["1", "2"]

    assert pd.isna(dataframe.loc[dataframe["_atom_site.id"] == "1", "u11"].iloc[0])
    assert dataframe.loc[dataframe["_atom_site.id"] == "2", "u11"].iloc[0] == 0.2


_ONE_ATOM = "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"


def test_an_unreadable_alternate_occupancy_fails_the_entry_with_runtime_error(
    minimal_cif: Path,
) -> None:
    text = minimal_cif.read_text(encoding="utf-8")
    minimal_cif.write_text(
        text.replace("CA A GLY A 1 1 ? 1 0 0 0.40", "CA A GLY A 1 1 ? 1 0 0 ?"),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"1ABC.*_atom_site\.occupancy"):
        _parse(minimal_cif, {})


def test_a_b_factor_that_is_not_a_number_fails_the_entry(tmp_path: Path) -> None:
    cif = _write_cif(tmp_path, "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 high 7 X 1\n")

    with pytest.raises(
        RuntimeError, match=r"Error parsing 1ABC: .*B_iso_or_equiv.*'high'"
    ):
        _parse(cif, {})


def test_numbers_are_read_without_their_standard_uncertainty(tmp_path: Path) -> None:
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C N  GLY A 1 7 ? 1 0 0 1.00 10(2) 7 X 1\n"
        "ATOM 2 C CA GLY A 1 7 ? 2 0 0 1.00 20.0(15) 7 X 1\n",
        preamble="_refine.ls_d_res_high 1.80(5)\n",
    )

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["residue_mean_bfactor"].unique().tolist() == [15.0]
    assert dataframe["resolution"].unique().tolist() == [1.8]


def test_refinement_resolution_is_preferred_to_the_data_collection_limit(
    tmp_path: Path,
) -> None:
    cif = _write_cif(
        tmp_path,
        _ONE_ATOM,
        preamble="_reflns.d_resolution_high 1.95\n_refine.ls_d_res_high 2.10\n",
    )

    dataframe = _parse(cif, {"resolution_rcsb": 3.0}).dataframe

    assert dataframe["resolution"].tolist() == [2.1]


def test_resolution_shells_give_their_smallest_limit(tmp_path: Path) -> None:
    preamble = """loop_
_reflns_shell.pdbx_ordinal
_reflns_shell.d_res_high
1 2.40
2 2.05
3 2.20

"""
    cif = _write_cif(tmp_path, _ONE_ATOM, preamble=preamble)

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["resolution"].tolist() == [2.05]


def test_a_record_type_other_than_atom_or_hetatm_fails_the_entry(
    tmp_path: Path,
) -> None:
    cif = _write_cif(tmp_path, "ANISOU 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n")

    with pytest.raises(RuntimeError, match=r"group_PDB.*'ANISOU'"):
        _parse(cif, {})


def test_a_bad_record_type_on_a_dropped_conformer_fails_the_entry(
    minimal_cif: Path,
) -> None:
    # Atom 2 is conformer A, which the parser drops in favour of B.
    text = minimal_cif.read_text(encoding="utf-8")
    minimal_cif.write_text(
        text.replace("ATOM 2 C CA A", "ANISOU 2 C CA A"), encoding="utf-8"
    )

    with pytest.raises(
        RuntimeError,
        match=r"group_PDB must be ATOM or HETATM on every row, but holds 'ANISOU'",
    ):
        _parse(minimal_cif, {})


def test_an_atom_site_loop_without_coordinates_fails_the_entry(
    tmp_path: Path,
) -> None:
    path = tmp_path / "1abc.cif"
    path.write_text(
        "data_1abc\nloop_\n_atom_site.group_PDB\n_atom_site.id\n"
        "_atom_site.label_atom_id\nATOM 1 CA\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=r"Cartn_x, _atom_site\.Cartn_y"):
        _parse(path, {})


def test_an_entity_id_on_two_rows_fails_the_entry(tmp_path: Path) -> None:
    preamble = """loop_
_entity.id
_entity.type
1 polymer
1 water

"""
    cif = _write_cif(tmp_path, _ONE_ATOM, preamble=preamble)

    with pytest.raises(RuntimeError, match=r"_entity\.id names '1' on two rows"):
        _parse(cif, {})


def test_a_file_without_atoms_gives_an_empty_dataframe_and_its_methods(
    tmp_path: Path,
) -> None:
    path = tmp_path / "1abc.cif"
    path.write_text("data_1abc\n_exptl.method 'X-RAY DIFFRACTION'\n", encoding="utf-8")

    result = _parse(path, {})

    assert result.dataframe.empty
    assert result.experimental_methods == ("X-RAY DIFFRACTION",)


def test_entry_id_is_the_structure_id_upper_cased_whatever_the_file_says(
    tmp_path: Path,
) -> None:
    path = tmp_path / "1crn.cif"
    path.write_text(f"data_1crn\n{_ATOM_SITE_HEADER}{_ONE_ATOM}", encoding="utf-8")

    dataframe = parse_mmcif_atom_rows(path, {}, structure_id="9xyz").dataframe

    assert dataframe["pdb_id"].tolist() == ["9XYZ"]


def test_quoted_values_and_rcsb_text_lose_surrounding_whitespace(
    tmp_path: Path,
) -> None:
    cif = _write_cif(
        tmp_path, _ONE_ATOM, preamble="_struct.title '  A padded title '\n"
    )

    dataframe = _parse(cif, {"citation_title": " A title \n"}).dataframe

    assert dataframe["_struct.title"].tolist() == ["A padded title"]
    assert dataframe["citation_title"].tolist() == ["A title"]


def test_rcsb_metadata_that_is_not_a_number_fails_the_entry(tmp_path: Path) -> None:
    cif = _write_cif(tmp_path, _ONE_ATOM)

    with pytest.raises(RuntimeError, match=r"'radiation_dose_MGy' is 'high'"):
        _parse(cif, {"radiation_dose_MGy": "high"})


def test_repeated_rcsb_methods_and_sources_are_listed_once(tmp_path: Path) -> None:
    cif = _write_cif(tmp_path, _ONE_ATOM)

    dataframe = _parse(
        cif,
        {
            "experimental_methods": ["X-RAY DIFFRACTION", "X-RAY DIFFRACTION"],
            "radiation_sources": ["SYNCHROTRON", "Synchrotron", "SYNCHROTRON"],
        },
    ).dataframe

    assert dataframe["experimental_methods"].tolist() == ["X-RAY DIFFRACTION"]
    assert dataframe["is_multi_method"].tolist() == [False]
    assert dataframe["radiation_sources"].tolist() == ["SYNCHROTRON"]


def test_missing_solvent_content_is_a_missing_number(tmp_path: Path) -> None:
    cif = _write_cif(tmp_path, _ONE_ATOM)

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["solvent_content"].dtype == "float64"
    assert dataframe["solvent_content"].isna().all()


def test_placeholder_cell_of_a_non_crystal_entry_has_no_volume(
    tmp_path: Path,
) -> None:
    preamble = (
        "_cell.length_a 1.000\n_cell.length_b 1.000\n_cell.length_c 1.000\n"
        "_cell.angle_alpha 90.00\n_cell.angle_beta 90.00\n_cell.angle_gamma 90.00\n"
    )
    cif = _write_cif(tmp_path, _ONE_ATOM, preamble=preamble)

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["unit_cell_volume"].isna().all()
    assert dataframe["solvent_content"].isna().all()


def test_non_physical_deposited_matthews_coefficient_falls_back_to_the_estimate(
    tmp_path: Path,
) -> None:
    cif = _write_cif(
        tmp_path,
        _TWO_RESIDUES,
        preamble=f"{_ORTHORHOMBIC_CELL}_exptl_crystal.density_Matthews 0.9\n",
    )

    dataframe = _parse(cif, {}).dataframe

    matthews = (40 * 50 * 60) / (4 * 2 * 110.0)
    assert dataframe["solvent_content"].unique().tolist() == pytest.approx(
        [1.0 - 1.23 / matthews]
    )


def test_solvent_content_estimate_counts_hetatm_polymer_residues(
    tmp_path: Path,
) -> None:
    # Selenomethionine is written as HETATM but belongs to the polymer entity,
    # so the asymmetric unit holds three polymer residues, not two.
    preamble = f"""{_ORTHORHOMBIC_CELL}loop_
_entity.id
_entity.type
1 polymer
2 water

"""
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
        "HETATM 2 C CA MSE A 1 8 ? 2 0 0 1.00 10 8 X 1\n"
        "ATOM 3 C CA GLY A 1 9 ? 3 0 0 1.00 10 9 X 1\n"
        "HETATM 4 O O HOH B 2 . ? 4 0 0 1.00 10 1 W 1\n",
        preamble=preamble,
    )

    dataframe = _parse(cif, {}).dataframe

    matthews = (40 * 50 * 60) / (4 * 3 * 110.0)
    assert dataframe["solvent_content"].unique().tolist() == pytest.approx(
        [1.0 - 1.23 / matthews]
    )


def test_rhombohedral_cell_counts_the_operations_of_its_own_setting(
    tmp_path: Path,
) -> None:
    # R 3 on rhombohedral axes has three operations; the hexagonal setting of
    # the same name has nine, with three times the cell volume.
    preamble = (
        "_cell.length_a 50\n_cell.length_b 50\n_cell.length_c 50\n"
        "_cell.angle_alpha 80\n_cell.angle_beta 80\n_cell.angle_gamma 80\n"
        "_symmetry.space_group_name_H-M 'R 3'\n"
    )
    cif = _write_cif(tmp_path, _TWO_RESIDUES, preamble=preamble)

    dataframe = _parse(cif, {}).dataframe

    volume = float(dataframe["unit_cell_volume"].iloc[0])
    matthews = volume / (3 * 2 * 110.0)
    assert dataframe["solvent_content"].unique().tolist() == pytest.approx(
        [1.0 - 1.23 / matthews]
    )


def test_b_anisotropy_is_missing_for_a_tensor_that_is_not_positive_definite(
    tmp_path: Path,
) -> None:
    preamble = """loop_
_atom_site_anisotrop.id
_atom_site_anisotrop.U[1][1]
_atom_site_anisotrop.U[2][2]
_atom_site_anisotrop.U[3][3]
_atom_site_anisotrop.U[1][2]
_atom_site_anisotrop.U[1][3]
_atom_site_anisotrop.U[2][3]
1 0.1 0.1 0.1 0.2 0.0 0.0

"""
    cif = _write_cif(tmp_path, _ONE_ATOM, preamble=preamble)

    dataframe = _parse(cif, {}).dataframe

    assert dataframe["u_equivalent"].tolist() == pytest.approx([0.1])
    assert dataframe["b_anisotropy"].isna().all()


def test_structure_without_anisotropic_tensors_has_missing_tensor_columns(
    tmp_path: Path,
) -> None:
    cif = _write_cif(tmp_path, _ONE_ATOM)

    dataframe = _parse(cif, {}).dataframe

    columns = ["u11", "u22", "u33", "u12", "u13", "u23", "u_equivalent"]
    assert dataframe[[*columns, "b_anisotropy"]].isna().all().all()
    assert (dataframe[columns].dtypes == "float64").all()


def test_bare_placeholders_are_missing_and_quoted_ones_are_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "1abc.cif"
    path.write_text(
        "data_1abc\n"
        "_struct.title '?'\n"
        "loop_\n"
        "_atom_site.group_PDB\n"
        "_atom_site.id\n"
        "_atom_site.Cartn_x\n"
        "_atom_site.Cartn_y\n"
        "_atom_site.Cartn_z\n"
        "_atom_site.pdbx_formal_charge\n"
        "ATOM 1 0 0 0 ?\n"
        "ATOM 2 0 0 0 '?'\n"
        "ATOM 3 0 0 0 .\n"
        "ATOM 4 0 0 0 '.'\n",
        encoding="utf-8",
    )

    dataframe = _parse(path, {}).dataframe

    charges = dataframe["_atom_site.pdbx_formal_charge"]
    assert charges.isna().tolist() == [True, False, True, False]
    assert charges.dropna().tolist() == ["?", "."]
    assert dataframe["_struct.title"].tolist() == ["?"] * 4


def test_coordinates_occupancy_and_model_number_are_numbers(
    minimal_cif: Path,
) -> None:
    dataframe = _parse(minimal_cif, {}).dataframe

    for column in ("x", "y", "z", "occupancy"):
        assert dataframe[column].dtype == "float64", column
    assert dataframe["model_num"].dtype == "Int64"
    assert dataframe["x"].tolist() == [
        float(text) for text in dataframe["_atom_site.Cartn_x"]
    ]
    assert dataframe["model_num"].tolist() == [
        int(text) for text in dataframe["_atom_site.pdbx_PDB_model_num"]
    ]


def test_a_model_number_that_is_not_whole_fails_the_entry(tmp_path: Path) -> None:
    cif = _write_cif(tmp_path, _ONE_ATOM.replace(" X 1\n", " X 1.5\n"))

    with pytest.raises(RuntimeError, match=r"pdbx_PDB_model_num .* not whole"):
        _parse(cif, {})


_SCHEMES = """loop_
_pdbx_poly_seq_scheme.asym_id
_pdbx_poly_seq_scheme.seq_id
_pdbx_poly_seq_scheme.pdb_mon_id
A 6 ?
A 7 GLY
A 8 GLY
#
loop_
_pdbx_nonpoly_scheme.asym_id
_pdbx_nonpoly_scheme.pdb_seq_num
B 1
#
"""


def test_a_file_with_every_listed_residue_has_no_warning(tmp_path: Path) -> None:
    # Residue 6 was not modelled, so it is not expected to have atoms.
    result = _parse(_write_cif(tmp_path, _TWO_RESIDUES, preamble=_SCHEMES), {})

    assert result.warnings == ()


def test_a_file_cut_short_inside_its_atom_sites_is_flagged(tmp_path: Path) -> None:
    rows = _TWO_RESIDUES.splitlines(keepends=True)[0]

    result = _parse(_write_cif(tmp_path, rows, preamble=_SCHEMES), {})

    assert result.warnings == (
        "1ABC: the file lists 2 modelled residue(s) that have no atom sites, so "
        "it may be truncated.",
    )
    assert len(result.dataframe) == 1


def test_an_obsolete_entry_names_its_replacements(tmp_path: Path) -> None:
    preamble = (
        "_pdbx_database_status.status_code OBS\n"
        "_pdbx_database_PDB_obs_spr.id OBSLTE\n"
        "_pdbx_database_PDB_obs_spr.pdb_id '2HHB 3HHB 4HHB'\n"
    )

    result = _parse(_write_cif(tmp_path, _TWO_RESIDUES, preamble=preamble), {})

    assert result.warnings == (
        "1ABC is obsolete in the PDB archive; it was replaced by 2HHB, 3HHB, 4HHB.",
    )


def test_author_identifiers_name_the_atoms_of_a_file_that_gives_them(
    minimal_cif: Path,
) -> None:
    result = _parse(minimal_cif, {})

    dataframe = result.dataframe
    # The named columns hold the author items as the file writes them, and
    # the label items of the fixture differ from every one of them.
    for column, item in (
        ("chain_id", "_atom_site.auth_asym_id"),
        ("res_num", "_atom_site.auth_seq_id"),
        ("res_name", "_atom_site.auth_comp_id"),
    ):
        assert dataframe[column].tolist() == dataframe[item].tolist()
    assert dataframe["chain_id"].tolist() == ["X", "X", "X", "X"]
    assert dataframe["res_num"].tolist() == ["7", "7", "7", "8"]
    assert dataframe["res_name"].tolist() == ["GLY", "GLY", "GLY", "ZN"]
    assert dataframe["residue_numbering"].unique().tolist() == ["author"]
    assert result.warnings == ()


def test_one_atom_without_an_author_number_keeps_author_numbering(
    tmp_path: Path,
) -> None:
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C N  GLY A 1 3 ? 1 0 0 1.00 10 7 X 1\n"
        "ATOM 2 C CA GLY A 1 3 ? 2 0 0 1.00 10 ? X 1\n",
    )

    result = _parse(cif, {})

    assert result.dataframe["chain_id"].tolist() == ["X", "X"]
    assert result.dataframe["res_num"].tolist() == ["7", ""]
    assert result.dataframe["residue_numbering"].tolist() == ["author", "author"]
    assert result.warnings == ()


def test_a_file_without_author_residue_numbers_is_named_by_its_label_items(
    tmp_path: Path,
) -> None:
    # The header has no _atom_site.auth_comp_id, so the residue name is the
    # label component under either numbering.
    cif = _write_cif(
        tmp_path,
        "ATOM 1 C CA CYS A 1 3 ? 1 0 0 1.00 10 ? X 1\n"
        "ATOM 2 C CA GLY A 1 4 ? 2 0 0 1.00 10 ? X 1\n",
    )

    result = _parse(cif, {})

    dataframe = result.dataframe
    assert dataframe["chain_id"].tolist() == ["A", "A"]
    assert dataframe["res_num"].tolist() == ["3", "4"]
    assert dataframe["res_name"].tolist() == ["CYS", "GLY"]
    assert dataframe["residue_numbering"].tolist() == ["label", "label"]
    assert result.warnings == (
        "1ABC: the file gives no author chain IDs or no author residue numbers, "
        "so chain_id and res_num hold its label identifiers "
        "(_atom_site.label_asym_id and _atom_site.label_seq_id).",
    )


def test_the_numbering_is_decided_before_the_conformer_choice(tmp_path: Path) -> None:
    # Only the dropped conformer carries an author number. The file still
    # gives one, so an engine that reads every conformer decides the same.
    header = _ATOM_SITE_HEADER.replace(
        "_atom_site.label_comp_id\n",
        "_atom_site.label_alt_id\n_atom_site.label_comp_id\n",
    )
    path = tmp_path / "1abc.cif"
    path.write_text(
        f"data_1abc\n{header}"
        "ATOM 1 C CA A GLY A 1 3 ? 1 0 0 0.80 10 ? X 1\n"
        "ATOM 2 C CA B GLY A 1 3 ? 2 0 0 0.20 10 7 X 1\n",
        encoding="utf-8",
    )

    dataframe = _parse(path, {}).dataframe

    assert dataframe["_atom_site.id"].tolist() == ["1"]
    assert dataframe["residue_numbering"].tolist() == ["author"]
    assert dataframe["res_num"].tolist() == [""]


def _refine_loop(*rows: tuple[str, str]) -> str:
    """Return a _refine loop of (method, resolution) rows, quoting the method."""
    lines = "".join(f"'{method}' {resolution}\n" for method, resolution in rows)
    return f"loop_\n_refine.pdbx_refine_id\n_refine.ls_d_res_high\n{lines}\n"


@pytest.mark.parametrize(
    ("rows", "first"),
    [
        ((("X-RAY DIFFRACTION", "1.60"), ("NEUTRON DIFFRACTION", "2.15")), 1.6),
        ((("NEUTRON DIFFRACTION", "2.15"), ("X-RAY DIFFRACTION", "1.60")), 2.15),
    ],
)
def test_a_joint_entry_gives_each_method_its_own_resolution_in_either_order(
    tmp_path: Path, rows: tuple[tuple[str, str], ...], first: float
) -> None:
    cif = _write_cif(tmp_path, _ONE_ATOM, preamble=_refine_loop(*rows))

    dataframe = _parse(cif, {}).dataframe

    # resolution follows the file's order, so it names either method.
    assert dataframe["resolution"].tolist() == [first]
    assert dataframe["resolution_xray"].tolist() == [1.6]
    assert dataframe["resolution_neutron"].tolist() == [2.15]


def test_a_single_method_entry_leaves_the_other_method_s_resolution_missing(
    tmp_path: Path,
) -> None:
    # _write_cif writes one path, so each file is parsed before the next.
    xray = _write_cif(
        tmp_path, _ONE_ATOM, preamble=_refine_loop(("X-RAY DIFFRACTION", "1.80"))
    )
    xray_rows = _parse(xray, {}).dataframe
    neutron = _write_cif(
        tmp_path, _ONE_ATOM, preamble=_refine_loop(("neutron diffraction", "2.2"))
    )
    neutron_rows = _parse(neutron, {}).dataframe

    assert xray_rows["resolution_xray"].tolist() == [1.8]
    assert xray_rows["resolution_neutron"].isna().all()
    # The method is matched in the archive's canonical spelling.
    assert neutron_rows["resolution_neutron"].tolist() == [2.2]
    assert neutron_rows["resolution_xray"].isna().all()


def test_a_refinement_that_names_no_method_takes_the_entry_s_only_method(
    tmp_path: Path,
) -> None:
    unnamed = "_refine.ls_d_res_high 1.90\n"
    cif = _write_cif(tmp_path, _ONE_ATOM, preamble=unnamed)

    from_record = _parse(cif, {"experimental_methods": ["NEUTRON DIFFRACTION"]})
    two_methods = _parse(
        cif,
        {"experimental_methods": ["X-RAY DIFFRACTION", "NEUTRON DIFFRACTION"]},
    )
    no_method = _parse(cif, {})
    # _write_cif writes one path, so this file replaces the one parsed above.
    from_file = _parse(
        _write_cif(
            tmp_path,
            _ONE_ATOM,
            preamble=unnamed + "_exptl.method 'X-RAY DIFFRACTION'\n",
        ),
        {},
    )

    assert from_record.dataframe["resolution_neutron"].tolist() == [1.9]
    assert from_record.dataframe["resolution_xray"].isna().all()
    assert from_file.dataframe["resolution_xray"].tolist() == [1.9]
    # Nothing says which of two methods an unnamed refinement belongs to.
    for unassigned in (two_methods.dataframe, no_method.dataframe):
        assert unassigned["resolution"].tolist() == [1.9]
        assert unassigned["resolution_xray"].isna().all()
        assert unassigned["resolution_neutron"].isna().all()


def test_a_method_resolution_is_the_first_one_reported_for_that_method(
    tmp_path: Path,
) -> None:
    preamble = _refine_loop(
        ("X-RAY DIFFRACTION", "?"),
        ("X-RAY DIFFRACTION", "0"),
        ("X-RAY DIFFRACTION", "1.75(2)"),
        ("X-RAY DIFFRACTION", "1.20"),
        ("ELECTRON CRYSTALLOGRAPHY", "3.0"),
    )

    dataframe = _parse(_write_cif(tmp_path, _ONE_ATOM, preamble=preamble), {}).dataframe

    # A missing or zero resolution is not one, and a method without a column
    # of its own is left out.
    assert dataframe["resolution_xray"].tolist() == [1.75]
    assert dataframe["resolution_neutron"].isna().all()


def test_a_method_resolution_that_is_not_a_number_fails_the_entry(
    tmp_path: Path,
) -> None:
    preamble = _refine_loop(
        ("X-RAY DIFFRACTION", "1.6"), ("NEUTRON DIFFRACTION", "coarse")
    )
    cif = _write_cif(tmp_path, _ONE_ATOM, preamble=preamble)

    with pytest.raises(RuntimeError, match=r"_refine\.ls_d_res_high"):
        _parse(cif, {})


def test_the_record_s_serial_and_time_resolved_flags_stay_missing_when_unsaid(
    tmp_path: Path,
) -> None:
    cif = _write_cif(tmp_path, _ONE_ATOM)

    said = _parse(
        cif,
        {
            "title": "Resting state",
            "keywords": "time-resolved",
            "mentions_time_resolved": True,
            "is_serial_crystallography": False,
            "sample_delivery_method": "injection",
            "xfel_pulse_duration_fs": 30.0,
        },
    ).dataframe
    unsaid = _parse(cif, {"title": "Resting state"}).dataframe

    assert said["title"].tolist() == ["Resting state"]
    assert said["keywords"].tolist() == ["time-resolved"]
    assert said["mentions_time_resolved"].tolist() == [True]
    assert said["is_serial_crystallography"].tolist() == [False]
    assert said["sample_delivery_method"].tolist() == ["injection"]
    assert said["xfel_pulse_duration_fs"].tolist() == [30.0]
    for flag in ("mentions_time_resolved", "is_serial_crystallography"):
        assert str(unsaid[flag].dtype) == "boolean"
        assert unsaid[flag].isna().all()
    assert unsaid["keywords"].isna().all()


def test_a_joint_entry_gives_each_method_its_own_r_factors(tmp_path: Path) -> None:
    preamble = """loop_
_refine.pdbx_refine_id
_refine.ls_d_res_high
_refine.ls_R_factor_R_work
_refine.ls_R_factor_R_free
'NEUTRON DIFFRACTION' 2.15 0.210 0.250
'X-RAY DIFFRACTION' 1.60 0.160 ?

"""
    cif = _write_cif(tmp_path, _ONE_ATOM, preamble=preamble)

    dataframe = _parse(cif, {"r_work": 0.21, "r_free": 0.25}).dataframe

    # r_work and r_free hold the first record's, the neutron one here.
    assert dataframe["r_work"].tolist() == [0.21]
    assert dataframe["r_work_neutron"].tolist() == [0.21]
    assert dataframe["r_free_neutron"].tolist() == [0.25]
    assert dataframe["r_work_xray"].tolist() == [0.16]
    assert dataframe["r_free_xray"].isna().all()
    assert dataframe["resolution_xray"].tolist() == [1.6]


_REFINEMENT_PREAMBLE = """_refine.ls_d_res_high 1.10
_refine.pdbx_TLS_residual_ADP_flag 'LIKELY RESIDUAL'

loop_
_software.classification
_software.name
_software.version
'data reduction' XDS ?
refinement REFMAC 5.8.0238
refinement PHENIX 1.16

loop_
_pdbx_refine_tls.id
_pdbx_refine_tls.method
1 refined
2 refined

loop_
_atom_site_anisotrop.id
_atom_site_anisotrop.U[1][1]
_atom_site_anisotrop.U[2][2]
_atom_site_anisotrop.U[3][3]
_atom_site_anisotrop.U[1][2]
_atom_site_anisotrop.U[1][3]
_atom_site_anisotrop.U[2][3]
1 0.4 0.2 0.1 0.0 0.0 0.0
3 0.4 0.2 0.1 0.0 0.0 0.0

"""


def test_how_the_entry_was_refined_is_read_from_the_file(tmp_path: Path) -> None:
    cif = _write_cif(
        tmp_path,
        "ATOM 1 N N  GLY A 1 7 ? 1 0 0 1.00 10 7 X 1\n"
        "ATOM 2 C CA GLY A 1 7 ? 2 0 0 1.00 10 7 X 1\n"
        "ATOM 3 C C  GLY A 1 7 ? 3 0 0 1.00 10 7 X 1\n"
        "ATOM 4 O O  GLY A 1 7 ? 4 0 0 1.00 10 7 X 1\n"
        "ATOM 5 H H  GLY A 1 7 ? 5 0 0 1.00 10 7 X 1\n",
        preamble=_REFINEMENT_PREAMBLE,
    )

    # The record plays no part, so an uploaded file has these columns too.
    dataframe = _parse(cif, {}).dataframe

    assert dataframe["refinement_program"].unique().tolist() == ["REFMAC"]
    assert dataframe["refinement_program_version"].unique().tolist() == ["5.8.0238"]
    assert dataframe["refinement_programs"].unique().tolist() == ["REFMAC; PHENIX"]
    assert str(dataframe["tls_group_count"].dtype) == "Int64"
    assert dataframe["tls_group_count"].unique().tolist() == [2]
    assert dataframe["tls_residual_adp_flag"].unique().tolist() == ["LIKELY RESIDUAL"]
    # Two of the four atoms that are not hydrogen have a tensor.
    assert dataframe["anisotropic_atom_fraction"].unique().tolist() == [0.5]
    # The several rows of _software are still not spread over the atoms.
    assert "_software.name" not in dataframe.columns


def test_an_unstated_refinement_is_missing_and_no_tensor_is_a_fraction_of_zero(
    tmp_path: Path,
) -> None:
    dataframe = _parse(_write_cif(tmp_path, _ONE_ATOM), {}).dataframe

    for column in (
        "refinement_program",
        "refinement_program_version",
        "refinement_programs",
        "tls_residual_adp_flag",
    ):
        assert dataframe[column].isna().all(), column
    # The file has no _refine category, so it does not say it used no TLS.
    assert str(dataframe["tls_group_count"].dtype) == "Int64"
    assert dataframe["tls_group_count"].isna().all()
    assert dataframe["anisotropic_atom_fraction"].tolist() == [0.0]


def test_a_refinement_without_tls_groups_has_a_count_of_zero(
    minimal_cif: Path,
) -> None:
    dataframe = _parse(minimal_cif, {}).dataframe

    assert dataframe["tls_group_count"].unique().tolist() == [0]


def test_the_anisotropic_fraction_counts_the_conformers_the_parser_drops(
    minimal_cif: Path,
) -> None:
    text = minimal_cif.read_text(encoding="utf-8")
    tensors = (
        "loop_\n_atom_site_anisotrop.id\n_atom_site_anisotrop.U[1][1]\n2 0.1\n3 0.1\n\n"
    )
    minimal_cif.write_text(
        text.replace(
            "loop_\n_atom_site.group_PDB", f"{tensors}loop_\n_atom_site.group_PDB"
        ),
        encoding="utf-8",
    )

    dataframe = _parse(minimal_cif, {}).dataframe

    # Atom 2 is the conformer the parser drops. It is one of the file's five
    # atom sites, two of which have a tensor.
    assert dataframe["_atom_site.id"].tolist() == ["1", "3", "4", "5"]
    assert dataframe["anisotropic_atom_fraction"].unique().tolist() == [0.4]
