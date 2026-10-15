from __future__ import annotations

import pandas as pd
import pytest

from pdbai.dataset_shapes import (
    base_tables,
    filtered_base_tables,
    with_key_source_columns,
)
from pdbai.engines.core import DATASET_SHAPES


def _atom_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pdb_id": ["1ABC", "1ABC", "1ABC"],
            "chain_id": ["A", "A", "B"],
            "res_num": ["1", "1", "2"],
            "res_name": ["GLY", "GLY", "ALA"],
            "atom_name": ["N", "CA", "N"],
            "element": ["N", "C", "N"],
            "is_hetatm": [False, False, False],
            "occupancy": [1.0, 1.0, 1.0],
            "model_num": [1, 1, 1],
            "_atom_site.label_asym_id": ["A", "A", "B"],
            "_atom_site.label_entity_id": ["1", "1", "2"],
            "_atom_site.label_seq_id": ["1", "1", "1"],
            "_atom_site.pdbx_PDB_ins_code": ["", "", ""],
            "_atom_site.pdbx_PDB_model_num": ["1", "1", "1"],
            "_pdbai_entity_type": ["polymer", "polymer", "polymer"],
            "x": [0.0, 1.0, 2.0],
            "residue_mean_bfactor": [12.0, 12.0, 18.0],
            "resolution": [1.5, 1.5, 1.5],
            "citation_title": ["Example", "Example", "Example"],
            "_cell.length_a": [10.0, 10.0, 10.0],
            "custom_atom_score": [0.1, 0.2, 0.3],
        }
    )


def test_flat_shape_writes_the_dataframe_as_its_one_table() -> None:
    dataframe = _atom_dataframe()

    tables = base_tables(dataframe, "flat")

    assert list(tables) == list(DATASET_SHAPES["flat"].tables)
    assert tables["flat"] is dataframe


def test_star_schema_places_columns_at_their_declared_granularity() -> None:
    tables = base_tables(_atom_dataframe(), "star_schema")

    assert list(tables) == list(DATASET_SHAPES["star_schema"].tables)
    assert len(tables["atoms"]) == 3
    assert len(tables["residues"]) == 2
    assert len(tables["chains"]) == 2
    assert len(tables["structures"]) == 1
    assert len(tables["citations"]) == 1

    assert "custom_atom_score" in tables["atoms"]
    assert "residue_mean_bfactor" in tables["residues"]
    assert "resolution" in tables["structures"]
    assert "_cell.length_a" in tables["structures"]
    assert "citation_title" in tables["citations"]
    assert "resolution" not in tables["atoms"]
    assert "citation_title" not in tables["structures"]
    assert tables["atoms"]["residue_key"].nunique() == 2


def test_star_schema_tables_lead_with_their_keys() -> None:
    tables = base_tables(_atom_dataframe(), "star_schema")

    assert list(tables["atoms"].columns[:12]) == [
        "residue_key",
        "pdb_id",
        "chain_id",
        "res_num",
        "res_name",
        "model_num",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "atom_name",
        "element",
    ]
    assert "_atom_site.pdbx_PDB_model_num" in tables["atoms"].columns[12:]
    assert "_atom_site.pdbx_PDB_model_num" not in tables["residues"]
    assert list(tables["residues"].columns[:3]) == [
        "residue_key",
        "chain_key",
        "pdb_id",
    ]
    assert list(tables["chains"].columns) == [
        "chain_key",
        "pdb_id",
        "chain_id",
        "model_num",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
    ]
    assert list(tables["structures"].columns[:1]) == ["pdb_id"]
    assert list(tables["citations"].columns) == ["pdb_id", "citation_title"]


def test_star_schema_requires_public_residue_keys() -> None:
    with pytest.raises(KeyError, match="requires the residue author identity columns"):
        base_tables(pd.DataFrame({"pdb_id": ["1ABC"]}), "star_schema")


def test_star_schema_splits_a_dataframe_without_rows_into_header_only_tables() -> None:
    tables = base_tables(_atom_dataframe().iloc[0:0], "star_schema")

    assert all(table.empty for table in tables.values())
    assert "custom_atom_score" in tables["atoms"]
    assert "resolution" in tables["structures"]


def test_invalid_residue_identity_uses_distinct_per_row_surrogates() -> None:
    dataframe = pd.DataFrame(
        {
            "pdb_id": ["1ABC", "1ABC"],
            "chain_id": ["", ""],
            "res_num": ["", ""],
            "res_name": ["", ""],
            "atom_name": ["X1", "X2"],
            "x": [1.0, 2.0],
        }
    )

    tables = base_tables(dataframe, "star_schema")

    assert tables["atoms"]["residue_key"].nunique() == 2
    assert len(tables["residues"]) == 2


def test_undeclared_columns_are_atom_level_unless_non_atom_mmcif_items(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataframe = _atom_dataframe()
    dataframe["_refine.ls_d_res_high"] = 1.5
    dataframe["_atom_site.label_alt_id"] = "."
    # A parser annotation is named like an mmCIF item but has no category, so
    # it stays with the atoms.
    dataframe["_pdbai_entity_type"] = ["polymer", "polymer", "non-polymer"]

    tables = base_tables(dataframe, "star_schema")

    assert "_refine.ls_d_res_high" in tables["structures"]
    assert "_atom_site.label_alt_id" in tables["atoms"]
    assert "_pdbai_entity_type" in tables["atoms"]
    assert "_pdbai_entity_type" not in tables["structures"]
    assert caplog.text == ""


def test_undeclared_citation_items_go_to_the_citations_table() -> None:
    dataframe = _atom_dataframe()
    # The parser repeats a one-row category's items on every atom.
    dataframe["_citation.title"] = "Example"
    dataframe["_citation_author.name"] = "Doe, J."
    dataframe["_audit_author.name"] = "Roe, R."

    tables = base_tables(dataframe, "star_schema")

    assert list(tables["citations"].columns) == [
        "pdb_id",
        "citation_title",
        "_citation.title",
        "_citation_author.name",
        "_audit_author.name",
    ]
    for column in ("_citation.title", "_citation_author.name", "_audit_author.name"):
        assert column not in tables["structures"]
    assert "_cell.length_a" in tables["structures"]


def test_star_schema_takes_a_value_the_first_row_lacks_from_a_later_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataframe = _atom_dataframe()
    dataframe["residue_mean_bfactor"] = [None, 1.0, 18.0]
    dataframe["resolution"] = [None, None, 1.5]

    tables = base_tables(dataframe, "star_schema")

    assert tables["residues"]["residue_mean_bfactor"].tolist() == [1.0, 18.0]
    assert tables["structures"]["resolution"].tolist() == [1.5]
    assert caplog.text == ""


def test_star_schema_warns_when_a_residue_column_varies_within_a_residue(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataframe = _atom_dataframe()
    dataframe.loc[1, "residue_mean_bfactor"] = 99.0

    tables = base_tables(dataframe, "star_schema")

    assert "'residue_mean_bfactor' varying within one residue" in caplog.text
    assert tables["residues"]["residue_mean_bfactor"].tolist() == [12.0, 18.0]


def test_star_schema_warns_when_a_citation_column_varies_within_a_structure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    dataframe = _atom_dataframe()
    dataframe.loc[2, "citation_title"] = "Another"

    tables = base_tables(dataframe, "star_schema")

    assert "'citation_title' varying within one structure" in caplog.text
    assert tables["citations"]["citation_title"].tolist() == ["Example"]


def test_a_star_schema_filter_keeps_the_identity_columns_its_keys_need() -> None:
    dataframe = _atom_dataframe()
    filtered = dataframe[["pdb_id", "x"]]

    restored = with_key_source_columns(filtered, dataframe, "star_schema")

    assert restored.columns[:2].tolist() == ["pdb_id", "x"]
    assert restored.columns[2:11].tolist() == [
        "chain_id",
        "res_num",
        "res_name",
        "model_num",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_pdbai_entity_type",
    ]
    assert restored["chain_id"].tolist() == dataframe["chain_id"].tolist()
    assert with_key_source_columns(filtered, dataframe, "flat") is filtered


def test_star_schema_takes_back_the_parser_helper_columns_a_filter_removed() -> None:
    dataframe = _atom_dataframe()
    dataframe["_pdbai_entity_type"] = "polymer"
    dataframe["_pdbai_polymer_type"] = "polypeptide(L)"
    filtered = dataframe[["pdb_id", "x"]]

    restored = with_key_source_columns(filtered, dataframe, "star_schema")

    assert restored["_pdbai_entity_type"].tolist() == ["polymer"] * 3
    assert restored["_pdbai_polymer_type"].tolist() == ["polypeptide(L)"] * 3


def test_a_filtered_star_schema_drops_the_helper_columns_once_keys_are_built() -> None:
    dataframe = _atom_dataframe()
    dataframe["_pdbai_entity_type"] = "polymer"
    dataframe["_pdbai_polymer_type"] = "polypeptide(L)"

    tables = filtered_base_tables(dataframe[["pdb_id", "x"]], dataframe, "star_schema")
    kept = filtered_base_tables(
        dataframe[["pdb_id", "x", "_pdbai_entity_type"]], dataframe, "star_schema"
    )

    assert not [
        column
        for table in tables.values()
        for column in table.columns
        if column.startswith("_pdbai_")
    ]
    assert (
        "chain_id" in tables["atoms"].columns or "chain_id" in tables["chains"].columns
    )
    assert any("_pdbai_entity_type" in table.columns for table in kept.values())
    assert not any("_pdbai_polymer_type" in table.columns for table in kept.values())
