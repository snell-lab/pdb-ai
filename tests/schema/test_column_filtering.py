from __future__ import annotations

import pandas as pd
import pytest

from pdbai.column_filtering import (
    filter_dataframe_columns,
    kept_columns,
    matches_glob,
    missing_columns_warning,
    parse_column_filter,
    parse_column_filter_mode,
    parse_column_list,
    resolve_requested_columns,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("include", "include"), (" EXCLUDE\n", "exclude"), (None, None), ("  ", None)],
)
def test_parse_column_filter_mode_ignores_case_and_blanks(
    value: object, expected: str | None
) -> None:
    assert parse_column_filter_mode(value) == expected


@pytest.mark.parametrize("value", ["keep", "includes", 1])
def test_parse_column_filter_mode_rejects_other_values(value: object) -> None:
    with pytest.raises(
        ValueError,
        match=rf"^column_filter_mode must be one of include, exclude; got {value!r}\.$",
    ):
        parse_column_filter_mode(value)


def test_parse_column_filter_mode_names_the_callers_key() -> None:
    with pytest.raises(ValueError, match=r"^--mode must be one of include"):
        parse_column_filter_mode("keep", "--mode")


def test_parse_column_filter_reads_a_mode_with_its_columns() -> None:
    assert parse_column_filter(" Include ", "pdb_id, x") == ("include", ["pdb_id", "x"])
    assert parse_column_filter(None, None) == (None, [])
    assert parse_column_filter("", " ") == (None, [])


@pytest.mark.parametrize(
    ("mode", "column_list"), [("include", None), ("exclude", " , "), (None, "x")]
)
def test_parse_column_filter_rejects_a_mode_or_columns_given_alone(
    mode: str | None, column_list: str | None
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^column_filter_mode and column_filter_list must be given "
        r"together; got mode .* and columns .*\.$",
    ):
        parse_column_filter(mode, column_list)


def test_parse_column_filter_names_the_callers_keys() -> None:
    with pytest.raises(
        ValueError, match=r"^mode and list must be given together; got mode None"
    ):
        parse_column_filter(None, "x", mode_key="mode", list_key="list")
    with pytest.raises(ValueError, match=r"^mode must be one of"):
        parse_column_filter("keep", "x", mode_key="mode", list_key="list")


def test_parse_column_list_splits_on_every_delimiter() -> None:
    assert parse_column_list("pdb_id, chain_id;x\r\ny\rz\n\n") == [
        "pdb_id",
        "chain_id",
        "x",
        "y",
        "z",
    ]
    assert parse_column_list(None) == []


def test_parse_column_list_splits_each_string_of_a_list() -> None:
    assert parse_column_list(["pdb_id, chain_id", "x\ny", " "]) == [
        "pdb_id",
        "chain_id",
        "x",
        "y",
    ]


def test_parse_column_list_materializes_generic_iterables() -> None:
    columns = (column for column in ["pdb_id, chain_id", "x"])

    assert parse_column_list(columns) == ["pdb_id", "chain_id", "x"]


def test_resolve_matches_exact_case_insensitive_normalized_and_glob_names() -> None:
    available = [
        "pdb_id",
        "_cell.length_a",
        "native_quality_score",
        "pdb_redo_quality_score",
        "unused",
    ]

    matched, missing = resolve_requested_columns(
        available,
        ["PDB_ID", "_cell_length_a", "*_quality_score", "does_not_exist", "*_nothing"],
    )

    assert matched == [
        "pdb_id",
        "_cell.length_a",
        "native_quality_score",
        "pdb_redo_quality_score",
    ]
    assert missing == ["does_not_exist", "*_nothing"]


def test_resolve_prefers_the_exact_spelling_and_lists_each_column_once() -> None:
    matched, missing = resolve_requested_columns(
        ["b_iso", "B_iso", "x"], ["B_iso", "x", "X", "*"]
    )

    assert matched == ["B_iso", "x", "b_iso"]
    assert missing == []


def test_a_name_folding_to_several_columns_by_case_matches_none() -> None:
    # No column is spelled b_iso, and ignoring case it could be either B_iso
    # or B_ISO.
    matched, missing = resolve_requested_columns(["B_iso", "B_ISO", "x"], ["b_iso"])

    assert matched == []
    assert missing == ["b_iso"]


def test_a_name_folding_to_several_columns_by_punctuation_matches_none() -> None:
    available = ["_cell.length_a", "cell-length-a", "x"]

    matched, missing = resolve_requested_columns(available, ["cell_length_a"])

    assert matched == []
    assert missing == ["cell_length_a"]


def test_an_exact_spelling_wins_over_an_ambiguous_fold() -> None:
    available = ["B_iso", "B_ISO", "b_iso", "_cell.length_a", "_cell_length_a"]

    matched, missing = resolve_requested_columns(available, ["b_iso", "_cell_length_a"])

    assert matched == ["b_iso", "_cell_length_a"]
    assert missing == []


def test_a_single_case_fold_wins_over_an_ambiguous_punctuation_fold() -> None:
    # Ignoring case is tried first, so it settles the name before the looser
    # fold by letters and digits is consulted.
    matched, missing = resolve_requested_columns(["B_iso", "b.iso"], ["b_iso"])

    assert matched == ["B_iso"]
    assert missing == []


def test_a_name_of_no_letters_or_digits_matches_no_column_by_its_fold() -> None:
    matched, missing = resolve_requested_columns(["_", "x"], ["-"])

    assert matched == []
    assert missing == ["-"]


def test_only_the_star_is_a_glob_wildcard() -> None:
    # Some mmCIF names carry brackets, which fnmatch would read as a
    # character class.
    assert matches_glob("_refine.aniso_B[1][1]", "_REFINE.aniso_b[1]*")
    assert not matches_glob("_refine.aniso_B11", "_refine.aniso_B[1]*")
    assert not matches_glob("b_iso", "b?iso")
    assert matches_glob("b_iso", "b*iso")


def test_include_keeps_the_matched_columns_in_request_order() -> None:
    dataframe = pd.DataFrame({"pdb_id": ["1ABC"], "a": [1], "b": [2], "c": [3]})

    filtered = filter_dataframe_columns(dataframe, "include", ["c", "A", "missing"])

    assert filtered.columns.tolist() == ["c", "a"]


def test_exclude_removes_the_matched_columns_and_keeps_source_order() -> None:
    dataframe = pd.DataFrame({"a": [1], "b": [2], "c": [3]})

    filtered = filter_dataframe_columns(dataframe, "exclude", ["b", "missing"])

    assert filtered.columns.tolist() == ["a", "c"]


def test_an_include_matching_nothing_keeps_only_the_structure_identifier() -> None:
    dataframe = pd.DataFrame({"pdb_id": ["1ABC"], "x": [1.0]})

    filtered = filter_dataframe_columns(dataframe, "include", ["*_score"])

    assert filtered.columns.tolist() == ["pdb_id"]
    assert filtered["pdb_id"].tolist() == ["1ABC"]


def test_an_exclude_matching_everything_keeps_the_structure_identifier() -> None:
    dataframe = pd.DataFrame({"pdb_id": ["1ABC"], "x": [1.0]})

    filtered = filter_dataframe_columns(dataframe, "exclude", ["*"])

    assert filtered.columns.tolist() == ["pdb_id"]


def test_an_exclude_matching_nothing_keeps_every_column() -> None:
    dataframe = pd.DataFrame({"pdb_id": ["1ABC"], "x": [1.0]})

    filtered = filter_dataframe_columns(dataframe, "exclude", ["*_score"])

    assert filtered.columns.tolist() == ["pdb_id", "x"]


def test_kept_columns_follow_the_request_for_include_and_the_source_for_exclude() -> (
    None
):
    available = ["a", "b", "c"]

    assert kept_columns(available, "include", ["c", "a"]) == ["c", "a"]
    assert kept_columns(available, "exclude", ["c", "a"]) == ["b"]


def test_kept_columns_keep_the_structure_identifier_when_nothing_else_is_kept() -> None:
    assert kept_columns(["pdb_id", "x"], "exclude", ["pdb_id", "x"]) == ["pdb_id"]
    assert kept_columns(["pdb_id", "x"], "include", []) == ["pdb_id"]


def test_kept_columns_keep_every_column_when_nothing_is_kept_and_no_identifier() -> (
    None
):
    assert kept_columns(["a", "b"], "exclude", ["a", "b"]) == ["a", "b"]


def test_missing_columns_warning_lists_each_name_once_in_order() -> None:
    assert missing_columns_warning(["z", "a", "z"], ["x"]) == (
        "The following columns were not found and were ignored: a, z"
    )


def test_missing_columns_warning_names_the_columns_an_ambiguous_name_could_mean() -> (
    None
):
    available = ["B_iso", "B_ISO", "_cell.length_a", "cell-length-a"]
    requested = ["b_iso", "cell_length_a", "z", "*_nothing"]
    _matched, missing = resolve_requested_columns(available, requested)

    assert missing_columns_warning(missing, available) == (
        "The following columns were not found and were ignored: *_nothing, z. "
        "The following names match several columns ignoring case or punctuation "
        "and were ignored: b_iso (B_iso, B_ISO); "
        "cell_length_a (_cell.length_a, cell-length-a)"
    )


def test_missing_columns_warning_lists_only_ambiguous_names_when_all_are() -> None:
    assert missing_columns_warning(["b_iso"], ["B_iso", "B_ISO"]) == (
        "The following names match several columns ignoring case or punctuation "
        "and were ignored: b_iso (B_iso, B_ISO)"
    )
