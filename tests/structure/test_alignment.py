from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from pdbai.structure import alignment
from pdbai.structure.residue_identity import (
    build_author_residue_keys,
    build_residue_identities,
)


def _dataframe(
    *,
    atom_names: list[str],
    res_nums: list[str] | None = None,
    x_values: list[float],
    label_seqs: list[str] | None = None,
    auth_chains: list[str] | None = None,
    label_chains: list[str] | None = None,
    res_names: list[str] | None = None,
) -> pd.DataFrame:
    count = len(atom_names)
    numbers = res_nums or ["10"] * count
    return pd.DataFrame(
        {
            "pdb_id": ["1abc"] * count,
            "model_num": ["1"] * count,
            "_atom_site.label_asym_id": label_chains or auth_chains or ["A"] * count,
            "_atom_site.label_seq_id": label_seqs or numbers,
            "_atom_site.auth_asym_id": auth_chains or ["A"] * count,
            "_atom_site.auth_seq_id": numbers,
            "_atom_site.pdbx_PDB_ins_code": [""] * count,
            "res_name": res_names or ["SER"] * count,
            "atom_name": atom_names,
            "x": x_values,
        }
    )


def _coordinate_dataframe(
    *,
    n: int,
    offset: float,
    chains: list[str] | None = None,
) -> pd.DataFrame:
    # Each residue has one CA atom on the x axis, 1 Å from the next and moved
    # by offset Å. Every atom is then exactly offset Å from its partner in a
    # chain built with offset 0.
    dataframe = _dataframe(
        atom_names=["CA"] * n,
        res_nums=[str(i) for i in range(1, n + 1)],
        x_values=[float(i) + offset for i in range(n)],
        auth_chains=chains,
    )
    dataframe["y"] = [0.0] * n
    dataframe["z"] = [0.0] * n
    return dataframe


def _align(
    rcsb: pd.DataFrame,
    pdb_redo: pd.DataFrame,
    *,
    pdb_id: str = "1ABC",
    carry_columns: Sequence[str] = ("x",),
) -> alignment.AlignmentResult:
    return alignment.align(
        rcsb,
        pdb_redo,
        pdb_id=pdb_id,
        caller="test_engine",
        carry_columns=carry_columns,
    )


def test_align_matches_distinct_atoms_one_to_one() -> None:
    rcsb = _dataframe(
        atom_names=["OG", "CB"], res_nums=["10", "11"], x_values=[1.0, 2.0]
    )
    pdb_redo = _dataframe(
        atom_names=["OG", "CB"], res_nums=["10", "11"], x_values=[9.0, 8.0]
    )

    result = _align(rcsb, pdb_redo)
    merged = result.columns

    assert result.diagnostics.match_rate == 1.0
    assert result.diagnostics.matched_atom_count == 2
    assert result.diagnostics.rcsb_ambiguous_atom_count == 0
    assert result.diagnostics.pdb_redo_ambiguous_atom_count == 0
    assert merged["x_pdb_redo"].tolist() == [9.0, 8.0]


def test_align_matches_atom_names_regardless_of_case() -> None:
    rcsb = _dataframe(atom_names=["og"], x_values=[1.0])
    pdb_redo = _dataframe(atom_names=["OG"], x_values=[9.0])

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.match_rate == 1.0
    assert result.columns["x_pdb_redo"].tolist() == [9.0]


def test_align_leaves_duplicate_identities_unmatched_instead_of_raising() -> None:
    rcsb = _dataframe(atom_names=["OG", "OG"], x_values=[1.0, 1.1])
    pdb_redo = _dataframe(atom_names=["OG", "OG"], x_values=[2.0, 2.1])

    result = _align(rcsb, pdb_redo)
    merged = result.columns

    assert len(merged) == len(rcsb)
    assert result.diagnostics.match_rate == 0.0
    assert result.diagnostics.rcsb_ambiguous_atom_count == 2
    assert result.diagnostics.pdb_redo_ambiguous_atom_count == 2
    assert merged["x_pdb_redo"].isna().all()


def test_align_leaves_atoms_unmatched_when_only_one_side_is_ambiguous() -> None:
    rcsb = _dataframe(atom_names=["OG"], x_values=[1.0])
    pdb_redo = _dataframe(atom_names=["OG", "OG"], x_values=[2.0, 2.1])

    result = _align(rcsb, pdb_redo)
    merged = result.columns

    assert len(merged) == 1
    assert result.diagnostics.match_rate == 0.0
    assert result.diagnostics.rcsb_ambiguous_atom_count == 0
    assert result.diagnostics.pdb_redo_ambiguous_atom_count == 2
    assert merged["x_pdb_redo"].isna().all()


def test_align_still_pairs_the_distinct_atoms_beside_a_duplicate() -> None:
    rcsb = _dataframe(
        atom_names=["OG", "OG", "CB"], res_nums=["10", "10", "11"], x_values=[1, 2, 3]
    )
    pdb_redo = _dataframe(
        atom_names=["CB", "OG"], res_nums=["11", "10"], x_values=[7.0, 8.0]
    )

    result = _align(rcsb, pdb_redo)

    assert result.columns["x_pdb_redo"].tolist()[2] == 7.0
    assert result.columns["x_pdb_redo"].iloc[:2].isna().all()
    assert result.diagnostics.matched_atom_count == 1
    assert result.diagnostics.rcsb_ambiguous_atom_count == 2


def test_align_keeps_nameless_atoms_apart() -> None:
    rcsb = _dataframe(atom_names=["", ""], x_values=[1.0, 1.1])
    pdb_redo = _dataframe(atom_names=["", ""], x_values=[2.0, 2.1])

    result = _align(rcsb, pdb_redo)
    merged = result.columns

    assert len(merged) == len(rcsb)
    assert result.diagnostics.match_rate == 0.0
    assert result.diagnostics.rcsb_ambiguous_atom_count == 0
    assert merged["x_pdb_redo"].isna().all()


def test_align_leaves_atoms_without_an_author_sequence_number_unmatched() -> None:
    rcsb = _dataframe(atom_names=["CA"], res_nums=["?"], x_values=[1.0])
    pdb_redo = _dataframe(atom_names=["CA"], res_nums=["?"], x_values=[2.0])

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.match_rate == 0.0
    assert result.columns["x_pdb_redo"].isna().all()


def test_align_keeps_models_apart() -> None:
    rcsb = _dataframe(atom_names=["CA", "CA"], x_values=[1.0, 2.0])
    rcsb["model_num"] = ["1", "2"]
    pdb_redo = _dataframe(atom_names=["CA", "CA"], x_values=[8.0, 9.0])
    pdb_redo["model_num"] = ["2", "1"]

    result = _align(rcsb, pdb_redo)

    assert result.columns["x_pdb_redo"].tolist() == [9.0, 8.0]


def test_align_warns_about_ambiguous_identities(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rcsb = _dataframe(atom_names=["OG", "OG"], x_values=[1.0, 1.1])
    pdb_redo = _dataframe(atom_names=["OG"], x_values=[2.0])

    _align(rcsb, pdb_redo)

    assert "duplicate atom identities (2 RCSB, 0 PDB-Redo)" in caplog.text
    assert "1ABC (test_engine)" in caplog.text


def test_align_stays_quiet_when_every_atom_is_distinct(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rcsb = _dataframe(
        atom_names=["OG", "CB"], res_nums=["10", "11"], x_values=[1.0, 2.0]
    )
    pdb_redo = _dataframe(
        atom_names=["OG", "CB"], res_nums=["10", "11"], x_values=[9.0, 8.0]
    )

    _align(rcsb, pdb_redo)

    assert caplog.text == ""


def test_align_preserves_rcsb_row_order_and_count() -> None:
    rcsb = _dataframe(
        atom_names=["N", "CA", "C"], res_nums=["1", "2", "3"], x_values=[1.0, 2.0, 3.0]
    )
    pdb_redo = _dataframe(
        atom_names=["C", "CA", "N"], res_nums=["3", "2", "1"], x_values=[7.0, 8.0, 9.0]
    )

    result = _align(rcsb, pdb_redo)
    merged = result.columns

    assert result.diagnostics.match_rate == 1.0
    assert merged["x_pdb_redo"].tolist() == [9.0, 8.0, 7.0]
    assert list(merged.columns) == ["x_pdb_redo"]


def test_align_returns_columns_under_the_rcsb_index() -> None:
    rcsb = _dataframe(
        atom_names=["N", "CA"], res_nums=["1", "2"], x_values=[1.0, 2.0]
    ).set_axis([40, 7])
    pdb_redo = _dataframe(
        atom_names=["CA", "N"], res_nums=["2", "1"], x_values=[8.0, 9.0]
    ).set_axis([3, 3])

    merged = _align(rcsb, pdb_redo).columns

    assert merged.index.tolist() == [40, 7]
    assert merged["x_pdb_redo"].tolist() == [9.0, 8.0]


def test_carry_columns_limits_what_crosses_the_merge() -> None:
    rcsb = _dataframe(
        atom_names=["OG", "CB"], res_nums=["10", "11"], x_values=[1.0, 2.0]
    )
    pdb_redo = _dataframe(
        atom_names=["OG", "CB"], res_nums=["10", "11"], x_values=[9.0, 8.0]
    )
    pdb_redo["unwanted"] = [1.0, 2.0]

    merged = _align(rcsb, pdb_redo, carry_columns=["x"]).columns

    assert list(merged.columns) == ["x_pdb_redo"]


def test_carry_columns_ignores_names_the_dataframe_does_not_have() -> None:
    rcsb = _dataframe(atom_names=["OG"], res_nums=["10"], x_values=[1.0])
    pdb_redo = _dataframe(atom_names=["OG"], res_nums=["10"], x_values=[9.0])

    result = _align(rcsb, pdb_redo, carry_columns=["x", "absent_column"])

    assert result.diagnostics.match_rate == 1.0
    assert list(result.columns.columns) == ["x_pdb_redo"]
    assert result.columns["x_pdb_redo"].tolist() == [9.0]


def _fully_populated_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pdb_id": ["1abc"],
            "model_num": ["1"],
            "_atom_site.pdbx_PDB_model_num": ["1"],
            "chain_id": ["A"],
            "_atom_site.auth_asym_id": ["A"],
            "_atom_site.label_asym_id": ["A"],
            "res_num": ["10"],
            "_atom_site.auth_seq_id": ["10"],
            "_atom_site.label_seq_id": ["10"],
            "_atom_site.pdbx_PDB_ins_code": [""],
            "res_name": ["MSE"],
            "_atom_site.auth_comp_id": ["MSE"],
            "_atom_site.label_comp_id": ["MSE"],
            "is_hetatm": [True],
            "_atom_site.group_PDB": ["HETATM"],
            "_pdbai_polymer_type": ["polypeptide(L)"],
            "_pdbai_entity_type": ["polymer"],
            "_pdbai_chem_comp_type": ["L-peptide linking"],
            "atom_name": ["CA"],
            "_atom_site.label_atom_id": ["CA"],
            "value": [1.0],
        }
    )


def _assert_same_identities(full: pd.DataFrame, narrowed: pd.DataFrame) -> None:
    full_identities = build_residue_identities(full)
    narrowed_identities = build_residue_identities(narrowed)
    for field in dataclasses.fields(full_identities):
        assert (
            getattr(full_identities, field.name).tolist()
            == getattr(narrowed_identities, field.name).tolist()
        ), field.name


@pytest.mark.parametrize(
    "dropped",
    [
        (),
        ("model_num", "chain_id", "res_num", "res_name", "is_hetatm", "atom_name"),
        ("_pdbai_polymer_type", "_pdbai_entity_type", "_atom_site.label_seq_id"),
        ("_pdbai_polymer_type", "_pdbai_entity_type"),
        ("is_hetatm", "_pdbai_chem_comp_type", "res_name", "_atom_site.auth_comp_id"),
    ],
)
def test_narrowing_to_identity_columns_keeps_every_residue_identity(
    dropped: tuple[str, ...],
) -> None:
    # Every case after the first drops the columns that identity is read
    # from when they exist. A fallback then decides identity, such as the
    # mmCIF spelling of a field or a weaker test for a polymer. The narrowing
    # must keep that fallback.
    full = _fully_populated_dataframe().drop(columns=list(dropped))
    narrowed = full[alignment.identity_columns(full)]

    assert "value" not in narrowed.columns
    _assert_same_identities(full, narrowed)


def test_identity_columns_keep_the_atom_name_under_either_spelling() -> None:
    full = _fully_populated_dataframe().drop(columns=["atom_name"])

    assert "_atom_site.label_atom_id" in alignment.identity_columns(full)


def test_narrowing_a_dataframe_preserves_every_author_key() -> None:
    full = _fully_populated_dataframe()
    narrowed = full[alignment.identity_columns(full)]

    assert (
        build_author_residue_keys(build_residue_identities(full)).tolist()
        == build_author_residue_keys(build_residue_identities(narrowed)).tolist()
    )
    key = build_author_residue_keys(build_residue_identities(full)).iloc[0]
    assert key is not None
    assert key[0] == "auth"


def test_a_narrowed_side_still_aligns_against_a_full_dataframe() -> None:
    full = _fully_populated_dataframe()
    full["_pdbai_polymer_type"] = [""]
    full["_pdbai_entity_type"] = ["non-polymer"]
    narrowed = full[alignment.identity_columns(full)].copy()
    narrowed["score"] = [42.0]

    result = _align(full, narrowed, carry_columns=["score"])

    assert result.diagnostics.match_rate == 1.0
    assert result.columns["score_pdb_redo"].tolist() == [42.0]


def test_align_survives_pdb_redo_label_seq_renumbering() -> None:
    # PDB-Redo numbers label_seq_id consecutively from the first observed
    # residue, so an RCSB chain whose first modelled residue is label 3 comes
    # back as label 1. Author numbering is untouched. Regression for 9GA6,
    # where every "matched" atom was paired with residue i+2.
    rcsb = _dataframe(
        atom_names=["N", "CA", "N", "CA"],
        res_nums=["3", "3", "4", "4"],
        label_seqs=["3", "3", "4", "4"],
        res_names=["THR", "THR", "GLY", "GLY"],
        x_values=[1.0, 2.0, 3.0, 4.0],
    )
    pdb_redo = _dataframe(
        atom_names=["N", "CA", "N", "CA"],
        res_nums=["3", "3", "4", "4"],
        label_seqs=["1", "1", "2", "2"],
        res_names=["THR", "THR", "GLY", "GLY"],
        x_values=[1.1, 2.1, 3.1, 4.1],
    )

    result = _align(rcsb, pdb_redo, pdb_id="9GA6")

    assert result.diagnostics.match_rate == 1.0
    assert result.columns["x_pdb_redo"].tolist() == [1.1, 2.1, 3.1, 4.1]


def test_align_survives_pdb_redo_label_asym_relabelling() -> None:
    # PDB-Redo assigns label_asym_id sequentially, so one label can name a
    # different author chain in each file. Here labels A and C name author
    # chains H and A in the RCSB file, and A and H in PDB-Redo's. Regression
    # for 9NQ3, where a 100% "match" paired every atom with the wrong chain.
    rcsb = _dataframe(
        atom_names=["CA", "CA"],
        res_nums=["10", "10"],
        auth_chains=["H", "A"],
        label_chains=["A", "C"],
        x_values=[1.0, 50.0],
    )
    pdb_redo = _dataframe(
        atom_names=["CA", "CA"],
        res_nums=["10", "10"],
        auth_chains=["A", "H"],
        label_chains=["A", "C"],
        x_values=[50.1, 1.1],
    )

    result = _align(rcsb, pdb_redo, pdb_id="9NQ3")

    assert result.diagnostics.match_rate == 1.0
    assert result.diagnostics.chain_id_source == "author"
    assert result.columns["x_pdb_redo"].tolist() == [1.1, 50.1]


def test_align_never_pairs_atoms_of_different_residues() -> None:
    # A shared author number and atom name do not make two atoms the same
    # when the residue names differ.
    rcsb = _dataframe(
        atom_names=["CA"], res_nums=["10"], res_names=["THR"], x_values=[1.0]
    )
    pdb_redo = _dataframe(
        atom_names=["CA"], res_nums=["10"], res_names=["GLY"], x_values=[2.0]
    )

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.match_rate == 0.0
    assert result.columns["x_pdb_redo"].isna().all()


def test_align_distinguishes_insertion_codes() -> None:
    rcsb = _dataframe(
        atom_names=["CA", "CA"], res_nums=["10", "10"], x_values=[1.0, 2.0]
    )
    rcsb["_atom_site.pdbx_PDB_ins_code"] = ["", "A"]
    pdb_redo = _dataframe(
        atom_names=["CA", "CA"], res_nums=["10", "10"], x_values=[2.5, 1.5]
    )
    pdb_redo["_atom_site.pdbx_PDB_ins_code"] = ["A", ""]

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.match_rate == 1.0
    assert result.columns["x_pdb_redo"].tolist() == [1.5, 2.5]


def test_mispairing_guard_rejects_atoms_that_moved_too_far(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Every atom matches by identity, yet each pair is 30 Å apart, which
    # re-refinement cannot produce. Regression for chain-relabelled entries
    # such as 9NQ3, where a 100% key match paired every atom with the wrong
    # chain.
    rcsb = _coordinate_dataframe(n=5, offset=0.0)
    pdb_redo = _coordinate_dataframe(n=5, offset=30.0)

    result = _align(rcsb, pdb_redo, pdb_id="9NQ3")

    assert result.diagnostics.rejected_reason is not None
    assert "median 30.0 Å apart" in result.diagnostics.rejected_reason
    assert result.diagnostics.match_rate == 0.0
    assert result.diagnostics.matched_atom_count == 0
    assert result.diagnostics.matched_eligible_atom_count == 0
    assert result.diagnostics.eligible_atom_count == 5
    assert result.columns["x_pdb_redo"].isna().all()
    assert "9NQ3 (test_engine) was rejected" in caplog.text
    assert "30.0 Å" in caplog.text


@pytest.mark.parametrize(("offset", "rejected"), [(1.0, False), (1.01, True)])
def test_mispairing_guard_rejects_only_a_median_above_one_angstrom(
    offset: float, rejected: bool
) -> None:
    rcsb = _coordinate_dataframe(n=3, offset=0.0)
    pdb_redo = _coordinate_dataframe(n=3, offset=offset)

    result = _align(rcsb, pdb_redo)

    assert (result.diagnostics.rejected_reason is not None) is rejected


def test_mispairing_guard_accepts_ordinary_re_refinement_shifts() -> None:
    rcsb = _coordinate_dataframe(n=5, offset=0.0)
    pdb_redo = _coordinate_dataframe(n=5, offset=0.2)

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.rejected_reason is None
    assert result.diagnostics.match_rate == 1.0
    assert result.columns["x_pdb_redo"].notna().all()


def test_mispairing_guard_tolerates_a_few_large_local_rebuilds() -> None:
    # The guard judges the median shift, so one re-traced residue out of five
    # does not condemn the structure.
    rcsb = _coordinate_dataframe(n=5, offset=0.0)
    pdb_redo = _coordinate_dataframe(n=5, offset=0.1)
    pdb_redo.loc[2, "x"] = 40.0

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.rejected_reason is None
    assert result.diagnostics.match_rate == 1.0


def test_mispairing_guard_measures_only_paired_atoms_with_coordinates() -> None:
    rcsb = _coordinate_dataframe(n=4, offset=0.0)
    pdb_redo = _coordinate_dataframe(n=4, offset=30.0)
    # A missing coordinate and an atom with no partner give no shift to
    # measure. Both are left out, so the median is 30 Å instead of NaN, which
    # would let the pairing pass.
    pdb_redo.loc[0, "y"] = np.nan
    pdb_redo = pdb_redo.drop(index=3)

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.rejected_reason is not None
    assert "median 30.0 Å apart" in result.diagnostics.rejected_reason


def test_mispairing_guard_is_skipped_without_coordinates() -> None:
    rcsb = _dataframe(
        atom_names=["OG", "CB"], res_nums=["10", "11"], x_values=[1.0, 2.0]
    )
    pdb_redo = _dataframe(
        atom_names=["OG", "CB"], res_nums=["10", "11"], x_values=[9.0, 8.0]
    )
    # Only x is present, and the guard needs all three coordinates on both
    # sides.

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.rejected_reason is None
    assert result.diagnostics.match_rate == 1.0


def test_mispairing_guard_is_skipped_when_one_side_lacks_coordinates() -> None:
    rcsb = _coordinate_dataframe(n=3, offset=0.0)
    pdb_redo = _coordinate_dataframe(n=3, offset=30.0).drop(columns=["z"])

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.rejected_reason is None


def test_mispairing_guard_reads_mmcif_coordinate_columns() -> None:
    rcsb = _coordinate_dataframe(n=3, offset=0.0)
    pdb_redo = _coordinate_dataframe(n=3, offset=30.0).rename(
        columns={
            "x": "_atom_site.Cartn_x",
            "y": "_atom_site.Cartn_y",
            "z": "_atom_site.Cartn_z",
        }
    )

    result = _align(rcsb, pdb_redo, carry_columns=["_atom_site.Cartn_x"])

    assert result.diagnostics.rejected_reason is not None
    assert "median 30.0 Å apart" in result.diagnostics.rejected_reason


def test_coordinate_columns_reports_whichever_aliases_exist() -> None:
    dataframe = pd.DataFrame({"x": [0.0], "_atom_site.Cartn_y": [0.0], "other": [1]})

    assert alignment.coordinate_columns(dataframe) == ["x", "_atom_site.Cartn_y"]


def _protein_and_solvent_dataframe(
    *, include_hydrogen: bool, include_water: bool
) -> pd.DataFrame:
    base = {"pdb_id": "1abc", "model_num": "1", "chain_id": "A", "y": 0.0, "z": 0.0}
    protein = {
        **base,
        "res_num": "10",
        "res_name": "SER",
        "_pdbai_entity_type": "polymer",
        "_pdbai_polymer_type": "polypeptide(L)",
    }
    rows = [
        {**protein, "atom_name": "N", "element": "N", "x": 0.0},
        {**protein, "atom_name": "CA", "element": "C", "x": 1.0},
    ]
    if include_hydrogen:
        rows.append({**protein, "atom_name": "H", "element": "H", "x": 0.5})
    if include_water:
        rows.append(
            {
                **base,
                "res_num": "300",
                "res_name": "HOH",
                "_pdbai_entity_type": "water",
                "_pdbai_polymer_type": "",
                "atom_name": "O",
                "element": "O",
                "x": 9.0,
            }
        )
    return pd.DataFrame(rows)


def test_match_rate_ignores_hydrogens_and_waters_pdb_redo_does_not_model() -> None:
    rcsb = _protein_and_solvent_dataframe(include_hydrogen=True, include_water=True)
    pdb_redo = _protein_and_solvent_dataframe(
        include_hydrogen=False, include_water=False
    )

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.match_rate == 1.0
    assert result.diagnostics.eligible_atom_count == 2
    assert result.diagnostics.matched_eligible_atom_count == 2
    assert result.diagnostics.matched_atom_count == 2


def test_match_rate_still_drops_when_protein_atoms_are_missing() -> None:
    rcsb = _protein_and_solvent_dataframe(include_hydrogen=False, include_water=False)
    pdb_redo = rcsb.iloc[[0]].copy()

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.match_rate == 0.5
    assert result.diagnostics.eligible_atom_count == 2
    assert result.diagnostics.matched_eligible_atom_count == 1


def test_align_only_measures_and_never_warns_about_a_low_rate(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # The RABDAM bridge aligns a deliberate subset of atoms, so a low rate is
    # coverage, not a fault. The comparison engine owns the warning.
    rcsb = _protein_and_solvent_dataframe(include_hydrogen=False, include_water=False)
    pdb_redo = rcsb.iloc[[0]].copy()

    _align(rcsb, pdb_redo)

    assert caplog.text == ""


def test_match_rate_falls_back_to_heavy_atoms_without_protein() -> None:
    dataframe = _dataframe(
        atom_names=["P", "OP1"], res_nums=["1", "2"], x_values=[0.0, 1.0]
    )
    dataframe["element"] = ["P", "O"]
    dataframe["_pdbai_entity_type"] = ["polymer", "polymer"]
    dataframe["_pdbai_polymer_type"] = ["polyribonucleotide", "polyribonucleotide"]

    result = _align(dataframe, dataframe.copy())

    assert result.diagnostics.eligible_atom_count == 2
    assert result.diagnostics.match_rate == 1.0


def test_match_rate_is_zero_for_an_empty_rcsb_dataframe() -> None:
    rcsb = _dataframe(atom_names=[], x_values=[])
    pdb_redo = _dataframe(atom_names=["CA"], x_values=[1.0])

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.eligible_atom_count == 0
    assert result.diagnostics.match_rate == 0.0
    assert result.columns.empty


def test_protein_heavy_atom_mask_reads_the_mmcif_element_column() -> None:
    dataframe = _protein_and_solvent_dataframe(
        include_hydrogen=True, include_water=True
    ).rename(columns={"element": "_atom_site.type_symbol"})

    mask = alignment.protein_heavy_atom_mask(dataframe)

    assert mask.tolist() == [True, True, False, False]


def test_protein_heavy_atom_mask_excludes_deuterium_in_any_case() -> None:
    dataframe = _protein_and_solvent_dataframe(
        include_hydrogen=True, include_water=False
    )
    dataframe["element"] = ["N", "C", "d"]

    mask = alignment.protein_heavy_atom_mask(dataframe)

    assert mask.tolist() == [True, True, False]


def test_protein_heavy_atom_mask_counts_atoms_without_an_element_as_heavy() -> None:
    dataframe = _protein_and_solvent_dataframe(
        include_hydrogen=True, include_water=True
    ).drop(columns=["element"])

    mask = alignment.protein_heavy_atom_mask(dataframe)

    assert mask.tolist() == [True, True, True, False]


def test_protein_heavy_atom_mask_counts_every_atom_when_none_is_heavy() -> None:
    dataframe = _protein_and_solvent_dataframe(
        include_hydrogen=True, include_water=False
    )
    dataframe["element"] = ["H", "H", "D"]

    mask = alignment.protein_heavy_atom_mask(dataframe)

    assert mask.tolist() == [True, True, True]


def test_protein_heavy_atom_mask_is_empty_for_an_empty_dataframe() -> None:
    mask = alignment.protein_heavy_atom_mask(pd.DataFrame())

    assert mask.dtype == bool
    assert mask.shape == (0,)


def test_align_falls_back_to_rcsb_label_chains_when_pdb_redo_renamed_them(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Regression for 9BB0 and 9BEX. RCSB author chains AAA and BBB, with
    # labels A and B, came back from PDB-Redo as author chains A and B. No
    # author key matches, so the aligner retries with the RCSB label chains
    # in the author slot.
    rcsb = _dataframe(
        atom_names=["CA", "CA"],
        res_nums=["10", "10"],
        auth_chains=["AAA", "BBB"],
        label_chains=["A", "B"],
        x_values=[1.0, 50.0],
    )
    pdb_redo = _dataframe(
        atom_names=["CA", "CA"],
        res_nums=["10", "10"],
        auth_chains=["B", "A"],
        label_chains=["B", "A"],
        x_values=[50.1, 1.1],
    )

    result = _align(rcsb, pdb_redo, pdb_id="9BB0")

    assert result.diagnostics.chain_id_source == "rcsb_label"
    assert result.diagnostics.match_rate == 1.0
    assert result.columns["x_pdb_redo"].tolist() == [1.1, 50.1]
    assert (
        "9BB0 (test_engine) paired no protein heavy atom on author chain IDs"
        in caplog.text
    )


def test_label_chain_fallback_counts_the_ambiguity_of_the_pairing_it_used() -> None:
    # Two author chains share one label chain, so the fallback's identities
    # repeat where the author identities did not.
    rcsb = _dataframe(
        atom_names=["CA", "CA", "CB"],
        res_nums=["10", "10", "10"],
        auth_chains=["AAA", "BBB", "AAA"],
        label_chains=["A", "A", "A"],
        x_values=[1.0, 2.0, 3.0],
    )
    pdb_redo = _dataframe(
        atom_names=["CB"], res_nums=["10"], auth_chains=["A"], x_values=[3.1]
    )

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.chain_id_source == "rcsb_label"
    assert result.diagnostics.rcsb_ambiguous_atom_count == 2
    assert result.columns["x_pdb_redo"].tolist()[2] == 3.1


def test_label_chain_fallback_still_faces_the_mispairing_guard() -> None:
    # The fallback pairs on label chains. When those pairs sit 30 Å apart,
    # the guard rejects the alignment as it would an author-chain pairing.
    rcsb = _coordinate_dataframe(n=5, offset=0.0, chains=["AAA"] * 5)
    rcsb["_atom_site.label_asym_id"] = ["A"] * 5
    pdb_redo = _coordinate_dataframe(n=5, offset=30.0, chains=["A"] * 5)

    result = _align(rcsb, pdb_redo, pdb_id="9BB0")

    assert result.diagnostics.chain_id_source == "rcsb_label"
    assert result.diagnostics.rejected_reason is not None
    assert result.diagnostics.match_rate == 0.0
    assert result.columns["x_pdb_redo"].isna().all()


def test_label_chain_fallback_does_nothing_when_labels_equal_author_chains() -> None:
    # The RCSB label chain equals its author chain, so putting labels in the
    # author slot changes no key. The atoms stay unmatched, and the fallback
    # is not reported as used.
    rcsb = _dataframe(
        atom_names=["CA"], auth_chains=["A"], label_chains=["A"], x_values=[1.0]
    )
    pdb_redo = _dataframe(
        atom_names=["CA"], auth_chains=["Q"], label_chains=["Q"], x_values=[1.1]
    )

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.chain_id_source == "author"
    assert result.diagnostics.match_rate == 0.0


def test_label_chain_fallback_keeps_author_pairing_when_labels_pair_nothing() -> None:
    rcsb = _dataframe(
        atom_names=["CA"], auth_chains=["AAA"], label_chains=["B"], x_values=[1.0]
    )
    pdb_redo = _dataframe(
        atom_names=["CA"], auth_chains=["Q"], label_chains=["Q"], x_values=[1.1]
    )

    result = _align(rcsb, pdb_redo)

    assert result.diagnostics.chain_id_source == "author"
    assert result.diagnostics.match_rate == 0.0
