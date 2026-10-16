from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pandas as pd
import pytest

from pdbai.structure.residue_identity import (
    atom_site_residue_identity,
    atom_site_residue_names,
    atom_site_residue_numbering,
    author_chain_and_number,
    build_author_residue_keys,
    build_residue_identities,
    clean_string_series,
    first_nonempty_string_series,
    first_numeric_series,
)


def _residue(**columns: object) -> pd.DataFrame:
    base: dict[str, object] = {
        "pdb_id": "1ABC",
        "chain_id": "A",
        "res_num": "10",
        "res_name": "GLY",
    }
    return pd.DataFrame({name: [value] for name, value in {**base, **columns}.items()})


def test_clean_string_series_strips_text_and_blanks_missing_values() -> None:
    dataframe = pd.DataFrame({"text": [" A ", "?", ".", None, "<NA>", "nan"]})

    result = clean_string_series(dataframe, "text")

    assert result.tolist() == ["A", "", "", "", "", ""]


def test_clean_string_series_reads_a_nullable_integer_column() -> None:
    dataframe = pd.DataFrame({"number": pd.Series([7, pd.NA], dtype="Int64")})

    assert clean_string_series(dataframe, "number").tolist() == ["7", ""]


def test_clean_string_series_gives_empty_text_for_an_absent_column() -> None:
    result = clean_string_series(pd.DataFrame({"other": [1, 2]}), "text")

    assert result.tolist() == ["", ""]


def test_first_nonempty_string_series_takes_the_first_column_with_text() -> None:
    dataframe = pd.DataFrame(
        {"first": ["A", "", "?"], "second": ["B", "C", ""]},
    )

    result = first_nonempty_string_series(dataframe, ("first", "absent", "second"), "Z")

    assert result.tolist() == ["A", "C", "Z"]


def test_first_numeric_series_falls_through_missing_values_in_order() -> None:
    dataframe = pd.DataFrame(
        {"first": ["1.5", "?", None, " "], "second": [9.0, 2.0, 3.0, math.nan]},
    )

    result = first_numeric_series(dataframe, ("absent", "first", "second"))

    assert result.tolist()[:3] == [1.5, 2.0, 3.0]
    assert math.isnan(result.iloc[3])
    assert result.dtype == "float64"


def test_first_numeric_series_uses_the_default_only_when_every_column_is_absent() -> (
    None
):
    absent = first_numeric_series(
        pd.DataFrame({"other": [1, 2]}), ("value",), default_if_absent=1.0
    )
    present = first_numeric_series(
        pd.DataFrame({"value": [None, "0.5"]}), ("value",), default_if_absent=1.0
    )

    assert absent.tolist() == [1.0, 1.0]
    assert math.isnan(present.iloc[0])
    assert present.iloc[1] == 0.5


def test_first_numeric_series_refuses_text_that_is_not_a_number() -> None:
    dataframe = pd.DataFrame({"value": ["1.0", "1.2.3", "?"]})

    with pytest.raises(
        ValueError, match=r"'value' holds value\(s\) that are not numbers: '1\.2\.3'"
    ):
        first_numeric_series(dataframe, ("value",))


def test_a_polymer_residue_is_keyed_by_label_fields_and_others_by_author() -> None:
    label_fields = {"_atom_site.label_asym_id": "B", "_atom_site.label_seq_id": "3"}
    dataframe = pd.concat(
        [
            _residue(_pdbai_entity_type="polymer", **label_fields),
            _residue(_pdbai_entity_type="non-polymer", **label_fields),
        ],
        ignore_index=True,
    )

    identities = build_residue_identities(dataframe)

    # Without a model column, every row is in model 1.
    assert identities.residue_keys.tolist() == [
        ("label", "1ABC", "1", "B", "3"),
        ("auth", "1ABC", "1", "A", "10", "", "GLY"),
    ]


def test_a_residue_without_author_chain_or_number_has_no_key() -> None:
    identities = build_residue_identities(_residue(res_num="?", is_hetatm=True))

    assert identities.residue_keys.tolist() == [None]
    assert identities.residue_join_keys.tolist() == [None]


def test_group_pdb_decides_hetatm_when_the_flag_column_is_absent() -> None:
    dataframe = pd.concat(
        [
            _residue(**{"_atom_site.group_PDB": "HETATM"}, res_name="HOH"),
            _residue(**{"_atom_site.group_PDB": "ATOM"}),
        ],
        ignore_index=True,
    )

    identities = build_residue_identities(dataframe)

    assert identities.polymer.tolist() == [False, True]


def test_an_unknown_group_pdb_is_refused_rather_than_read_as_atom() -> None:
    dataframe = _residue(**{"_atom_site.group_PDB": "ANISOU"})

    with pytest.raises(ValueError, match="_atom_site.group_PDB.*'ANISOU'"):
        build_residue_identities(dataframe)


def test_a_missing_hetatm_flag_reads_as_atom() -> None:
    # After annotation is_hetatm is a nullable Boolean column. Without the
    # flag the residue is polymer, as an ATOM record's would be.
    dataframe = _residue()
    dataframe["is_hetatm"] = pd.Series([pd.NA], dtype="boolean")

    assert build_residue_identities(dataframe).polymer.tolist() == [True]


def test_a_hetatm_flag_that_is_not_a_boolean_is_refused() -> None:
    dataframe = _residue(is_hetatm="false")

    with pytest.raises(ValueError, match="'is_hetatm'.*'false'"):
        build_residue_identities(dataframe)


def test_entity_metadata_outranks_the_hetatm_record() -> None:
    dataframe = pd.concat(
        [
            _residue(
                res_name="MSE",
                is_hetatm=True,
                _pdbai_entity_type="polymer",
                _pdbai_polymer_type="polypeptide(L)",
                **{"_atom_site.label_asym_id": "A", "_atom_site.label_seq_id": "5"},
            ),
            _residue(
                res_name="ATP",
                res_num="501",
                is_hetatm=False,
                _pdbai_entity_type="non-polymer",
                **{"_atom_site.label_asym_id": "C", "_atom_site.label_seq_id": ""},
            ),
        ],
        ignore_index=True,
    )

    identities = build_residue_identities(dataframe)

    assert identities.polymer.tolist() == [True, False]
    assert identities.protein_polymer.tolist() == [True, False]
    assert identities.residue_join_keys.tolist() == [
        "label|1ABC|1|A|5",
        "auth|1ABC|1|A|501||ATP",
    ]
    assert identities.chain_keys.tolist() == ["chain|1ABC|1|A", "chain|1ABC|1|C"]


def test_a_nucleic_acid_polymer_is_not_protein() -> None:
    identities = build_residue_identities(
        _residue(
            res_name="DA",
            is_hetatm=False,
            _pdbai_entity_type="polymer",
            _pdbai_polymer_type="polydeoxyribonucleotide",
        )
    )

    assert identities.polymer.tolist() == [True]
    assert identities.protein_polymer.tolist() == [False]


def test_component_ids_decide_protein_without_a_polymer_type() -> None:
    # SEP is phosphoserine. It is given in lower case to show that component
    # IDs are compared in upper case.
    dataframe = pd.concat(
        [
            _residue(res_name="sep", is_hetatm=True, _pdbai_entity_type="polymer"),
            _residue(res_name="DA", is_hetatm=False, res_num="11"),
        ],
        ignore_index=True,
    )

    identities = build_residue_identities(dataframe)

    assert identities.res_name.tolist() == ["SEP", "DA"]
    assert identities.protein_polymer.tolist() == [True, False]


def test_author_keys_ignore_label_fields_and_keep_the_model() -> None:
    dataframe = pd.concat(
        [
            _residue(model_num="1", **{"_atom_site.label_seq_id": "1"}),
            _residue(model_num="2", **{"_atom_site.label_seq_id": "7"}),
        ],
        ignore_index=True,
    )

    keys = build_author_residue_keys(build_residue_identities(dataframe))

    assert keys.tolist() == [
        ("auth", "1ABC", "1", "A", "10", "", "GLY"),
        ("auth", "1ABC", "2", "A", "10", "", "GLY"),
    ]


def test_residue_identity_is_model_aware_and_prefers_label_ids() -> None:
    dataframe = pd.DataFrame(
        {
            "pdb_id": ["1ABC", "1ABC", "1ABC"],
            "model_num": [1, 1, 2],
            "chain_id": ["X", "X", "X"],
            "res_num": ["10", "10", "10"],
            "res_name": ["GLY", "GLY", "GLY"],
            "_atom_site.label_asym_id": ["A", "B", "A"],
            "_atom_site.label_seq_id": ["1", "1", "1"],
            "_atom_site.label_entity_id": ["1", "1", "1"],
            "_pdbai_entity_type": ["polymer", "polymer", "polymer"],
            "is_hetatm": [False, False, False],
        }
    )

    identities = build_residue_identities(dataframe)

    assert identities.residue_join_keys.nunique() == 3
    assert identities.residue_join_keys.tolist() == [
        "label|1ABC|1|A|1",
        "label|1ABC|1|B|1",
        "label|1ABC|2|A|1",
    ]


def test_author_identity_keeps_insertion_code_and_residue_name() -> None:
    dataframe = pd.concat(
        [
            _residue(**{"_atom_site.pdbx_PDB_ins_code": "A"}),
            _residue(**{"_atom_site.pdbx_PDB_ins_code": "B"}),
            _residue(res_name="ALA", **{"_atom_site.pdbx_PDB_ins_code": "B"}),
        ],
        ignore_index=True,
    )

    identities = build_residue_identities(dataframe)

    assert identities.residue_keys.tolist() == [
        ("auth", "1ABC", "1", "A", "10", "A", "GLY"),
        ("auth", "1ABC", "1", "A", "10", "B", "GLY"),
        ("auth", "1ABC", "1", "A", "10", "B", "ALA"),
    ]


def _atom_site(
    items: Mapping[str, Sequence[str | None]] | None = None,
) -> pd.DataFrame:
    """Return _atom_site rows whose author and label identifiers differ."""
    base: dict[str, Sequence[str | None]] = {
        "_atom_site.auth_asym_id": ["X", "X", "Y"],
        "_atom_site.auth_seq_id": ["7", "8", "7"],
        "_atom_site.auth_comp_id": ["GLY", "CYS", "ZN"],
        "_atom_site.label_asym_id": ["A", "A", "B"],
        "_atom_site.label_seq_id": ["1", "2", None],
        "_atom_site.label_comp_id": ["GLY", "CYS", "ZN"],
    }
    return pd.DataFrame({**base, **(items or {})})


def test_a_file_with_author_identifiers_is_named_by_them() -> None:
    identity = atom_site_residue_identity(_atom_site())

    assert identity.numbering == "author"
    assert identity.chain_id.tolist() == ["X", "X", "Y"]
    assert identity.res_num.tolist() == ["7", "8", "7"]
    assert identity.res_name.tolist() == ["GLY", "CYS", "ZN"]


def test_a_row_without_an_author_identifier_does_not_switch_the_numbering() -> None:
    # One atom lacks its author number and another its author chain. Label
    # values on those rows alone would mix the two numberings.
    atom_site = _atom_site(
        {
            "_atom_site.auth_seq_id": ["7", None, "7"],
            "_atom_site.auth_asym_id": ["X", "X", ""],
        }
    )

    identity = atom_site_residue_identity(atom_site)

    assert identity.numbering == "author"
    assert identity.chain_id.tolist() == ["X", "X", ""]
    assert identity.res_num.tolist() == ["7", "", "7"]


@pytest.mark.parametrize(
    "without",
    [
        {"_atom_site.auth_seq_id": [None, None, None]},
        {"_atom_site.auth_seq_id": ["", None, ""]},
        {"_atom_site.auth_asym_id": [None, None, None]},
    ],
)
def test_a_file_without_an_author_identifier_is_named_by_its_label_items(
    without: dict[str, Sequence[str | None]],
) -> None:
    identity = atom_site_residue_identity(_atom_site(without))

    assert identity.numbering == "label"
    assert identity.chain_id.tolist() == ["A", "A", "B"]
    # A residue outside a polymer has no label number.
    assert identity.res_num.tolist() == ["1", "2", ""]


@pytest.mark.parametrize(
    "absent", ["_atom_site.auth_seq_id", "_atom_site.auth_asym_id"]
)
def test_an_absent_author_item_gives_label_numbering(absent: str) -> None:
    atom_site = _atom_site().drop(columns=absent)

    assert atom_site_residue_numbering(atom_site) == "label"
    assert atom_site_residue_identity(atom_site).chain_id.tolist() == ["A", "A", "B"]


def test_the_numbering_is_read_from_a_flat_dictionary_as_from_a_frame() -> None:
    author: dict[str, list[str | None]] = {
        "_atom_site.auth_asym_id": ["X", None],
        "_atom_site.auth_seq_id": [None, "7"],
    }

    assert atom_site_residue_numbering(author) == "author"
    assert (
        atom_site_residue_numbering({**author, "_atom_site.auth_seq_id": [None, None]})
        == "label"
    )


def test_a_selection_of_rows_takes_the_numbering_of_its_file() -> None:
    # Alone, the third row has no author chain and would read as label
    # numbered, so the numbering of the whole file is passed with it.
    atom_site = _atom_site({"_atom_site.auth_asym_id": ["X", "X", None]})
    numbering = atom_site_residue_numbering(atom_site)
    selection = atom_site.iloc[[2]]

    assert atom_site_residue_numbering(selection) == "label"
    identity = atom_site_residue_identity(selection, numbering=numbering)
    assert identity.numbering == "author"
    assert identity.chain_id.tolist() == [""]
    assert identity.res_num.tolist() == ["7"]


def test_a_residue_name_falls_back_to_the_label_component_row_by_row() -> None:
    atom_site = _atom_site(
        {
            "_atom_site.auth_comp_id": ["SEP", None, ""],
            "_atom_site.label_comp_id": ["SER", "CYS", "ZN"],
        }
    )

    assert atom_site_residue_names(atom_site).tolist() == ["SEP", "CYS", "ZN"]
    assert atom_site_residue_names(
        atom_site.drop(columns="_atom_site.auth_comp_id")
    ).tolist() == ["SER", "CYS", "ZN"]
    # The name does not depend on the numbering.
    assert atom_site_residue_identity(atom_site).numbering == "author"
    assert atom_site_residue_identity(atom_site).res_name.tolist() == [
        "SEP",
        "CYS",
        "ZN",
    ]


def _label_numbered_table(**columns: object) -> pd.DataFrame:
    """Return a polymer and a ligand atom as the parser names them by label."""
    base: dict[str, object] = {
        "pdb_id": ["9LAB", "9LAB"],
        "residue_numbering": ["label", "label"],
        "chain_id": ["A", "B"],
        "res_num": ["1", ""],
        "res_name": ["CYS", "ZN"],
        "_atom_site.label_asym_id": ["A", "B"],
        "_atom_site.label_seq_id": ["1", None],
        "_pdbai_entity_type": ["polymer", "non-polymer"],
        "is_hetatm": [False, True],
    }
    return pd.DataFrame({**base, **columns})


def test_label_values_in_the_named_columns_are_never_read_as_author_fields() -> None:
    table = _label_numbered_table()

    chain, number = author_chain_and_number(table)
    identities = build_residue_identities(table)

    assert chain.tolist() == ["", ""]
    assert number.tolist() == ["", ""]
    # The polymer residue keeps its label key, and no residue gets an author
    # key made of label values, which would pair it with another file's atoms.
    assert identities.residue_keys.tolist() == [("label", "9LAB", "1", "A", "1"), None]
    assert build_author_residue_keys(identities).tolist() == [None, None]


def test_author_items_still_key_a_label_numbered_table() -> None:
    # The file gave author numbers but no author chain, so its named columns
    # hold label values while its author numbers remain.
    table = _label_numbered_table(
        **{
            "_atom_site.auth_asym_id": ["X", "X"],
            "_atom_site.auth_seq_id": ["7", "8"],
        }
    )

    chain, number = author_chain_and_number(table)

    assert chain.tolist() == ["X", "X"]
    assert number.tolist() == ["7", "8"]
    assert build_residue_identities(table).residue_keys.tolist() == [
        ("label", "9LAB", "1", "A", "1"),
        ("auth", "9LAB", "1", "X", "8", "", "ZN"),
    ]


def test_a_table_without_the_numbering_column_reads_as_author_numbered() -> None:
    chain, number = author_chain_and_number(_residue())

    assert chain.tolist() == ["A"]
    assert number.tolist() == ["10"]
