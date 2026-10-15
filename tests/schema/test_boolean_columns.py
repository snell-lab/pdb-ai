from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pdbai.engines.analyses.rabdam.calculation import add_agreement_flags
from pdbai.engines.core import (
    ColumnSpec,
    EngineContext,
    EngineOptions,
    EnginePhase,
    EngineResult,
    EngineSpec,
)
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files
from pdbai.orchestration.annotation import annotate_structure
from pdbai.schema import declared_boolean_columns, normalize_boolean_columns


class _FlagEngine:
    spec = EngineSpec(
        name="flagger",
        display_name="Flagger",
        phase=EnginePhase.ANNOTATION,
        columns=(ColumnSpec("is_flagged", group="Flag Fields", dtype="boolean"),),
    )

    def run(self, context: EngineContext) -> EngineResult:
        return EngineResult()


def _normalized(values: pd.Series) -> pd.Series:
    return normalize_boolean_columns(pd.DataFrame({"is_hetatm": values}))["is_hetatm"]


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ([True, False, None], [True, False, pd.NA]),
        ([1.0, 0.0, np.nan], [True, False, pd.NA]),
        ([1, 0, None], [True, False, pd.NA]),
        ([np.True_, np.False_, None], [True, False, pd.NA]),
        ([None, None], [pd.NA, pd.NA]),
    ],
)
def test_accepted_representations_normalize(
    values: list[object], expected: list[object]
) -> None:
    result = _normalized(pd.Series(values, dtype=object))

    assert result.dtype == "boolean"
    assert result.tolist() == expected


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "float64"])
def test_numeric_columns_of_zero_and_one_normalize(dtype: str) -> None:
    result = _normalized(pd.Series([1, 0, None], dtype=dtype))

    assert result.dtype == "boolean"
    assert result.tolist() == [True, False, pd.NA]


def test_a_numpy_boolean_column_becomes_nullable() -> None:
    result = _normalized(pd.Series([True, False], dtype=bool))

    assert result.dtype == "boolean"
    assert result.tolist() == [True, False]


def test_normalizing_keeps_the_index() -> None:
    result = _normalized(pd.Series([1.0, None, True], index=["a", "b", "c"]))

    assert result.index.tolist() == ["a", "b", "c"]
    assert result.tolist() == [True, pd.NA, True]


def test_a_column_with_no_rows_becomes_an_empty_boolean_column() -> None:
    result = _normalized(pd.Series([], dtype=object))

    assert result.dtype == "boolean"
    assert result.empty


@pytest.mark.parametrize(
    "values",
    [
        ["true", "false"],
        ["True", "False"],
        [2, 0],
        [0.5, 1.0],
        [-1, 1],
        ["yes", "no"],
        [""],
    ],
)
def test_ambiguous_representations_fail_loudly(values: list[object]) -> None:

    with pytest.raises(ValueError, match="declared boolean but holds non-boolean"):
        _normalized(pd.Series(values, dtype=object))


@pytest.mark.parametrize("dtype", ["Int64", "float64"])
def test_a_numeric_column_holding_another_number_is_refused(dtype: str) -> None:
    with pytest.raises(ValueError, match="'is_hetatm'.*2"):
        _normalized(pd.Series([1, 2, None], dtype=dtype))


def test_failure_names_the_column_and_the_offending_value() -> None:
    dataframe = pd.DataFrame(
        {"bdamage_matches": pd.Series([True, "maybe"], dtype=object)}
    )

    with pytest.raises(ValueError, match=r"'bdamage_matches'.*'maybe'"):
        normalize_boolean_columns(dataframe)


def test_failure_lists_the_first_values_in_order_and_counts_the_rest() -> None:
    # The repeated 9 counts once, so seven values are refused. The message
    # lists five in order of appearance and counts the other two.
    values = pd.Series([10, 9, 2, 9, 3, 4, 5, 6], dtype=object)

    with pytest.raises(ValueError, match=r"are 10, 9, 2, 3, 4, and 2 more\."):
        _normalized(values)


def test_normalize_boolean_columns_converts_declared_columns_only() -> None:
    # A flag with a missing value becomes a float column of 1.0, 0.0, and NaN,
    # which np.where builds here.
    dataframe = pd.DataFrame(
        {
            "is_hetatm": np.where([True, True, False], [True, False, True], np.nan),
            "u_equivalent": [1.0, 0.0, 2.0],
            "not_a_declared_column": [1.0, np.nan, 3.0],
        }
    )

    result = normalize_boolean_columns(dataframe)

    assert dataframe["is_hetatm"].dtype == np.dtype("float64")
    assert result["is_hetatm"].dtype == "boolean"
    assert result["is_hetatm"].tolist() == [True, False, pd.NA]
    assert result["u_equivalent"].dtype == np.dtype("float64")
    assert result["not_a_declared_column"].dtype == np.dtype("float64")


def test_normalize_boolean_columns_leaves_a_conforming_dataframe_untouched() -> None:
    dataframe = pd.DataFrame({"is_hetatm": pd.Series([True, pd.NA], dtype="boolean")})

    assert normalize_boolean_columns(dataframe) is dataframe


def test_normalize_boolean_columns_rejects_a_nonconforming_producer() -> None:
    dataframe = pd.DataFrame({"is_hetatm": ["true", "false"]})

    with pytest.raises(ValueError, match="'is_hetatm'"):
        normalize_boolean_columns(dataframe)


def test_a_declared_column_held_twice_is_refused() -> None:
    dataframe = pd.DataFrame([[1.0, 0.0]], columns=["is_hetatm", "is_hetatm"])

    with pytest.raises(ValueError, match="is_hetatm appear more than once"):
        normalize_boolean_columns(dataframe)


def test_an_engine_registered_later_has_its_boolean_columns_checked(
    install_engines: Callable[..., None],
) -> None:
    assert "is_flagged" not in declared_boolean_columns()

    install_engines(_FlagEngine())

    assert "is_flagged" in declared_boolean_columns()
    with pytest.raises(ValueError, match="'is_flagged'"):
        normalize_boolean_columns(pd.DataFrame({"is_flagged": ["probably"]}))


def test_rabdam_agreement_flags_are_nullable_booleans() -> None:
    atom_dataframe = pd.DataFrame(
        {
            "bdamage_pdb_redo": [1.0, 1.0, np.nan, 2.0],
            "bdamage_rcsb": [1.0, 5.0, 2.0, np.nan],
        }
    )

    result = add_agreement_flags(atom_dataframe)

    assert result["bdamage_matches"].dtype == "boolean"
    assert result["bdamage_matches"].tolist() == [True, False, pd.NA, pd.NA]


def _annotated_dataframe(
    atom_dataframe: pd.DataFrame, options: EngineOptions
) -> pd.DataFrame:
    context = EngineContext(
        pdb_id="1ABC", atom_dataframe=pd.DataFrame(), options=options
    )
    return annotate_structure(atom_dataframe, context).dataframe


def test_annotation_normalizes_a_float_encoded_flag_from_any_producer(
    offline_options: EngineOptions,
) -> None:

    atom_dataframe = pd.DataFrame(
        {
            "pdb_id": ["1ABC"] * 3,
            "chain_id": ["A"] * 3,
            "res_num": ["1", "2", "3"],
            "atom_name": ["N", "CA", "C"],
            "is_hetatm": np.where([True, True, False], [True, False, True], np.nan),
        }
    )

    dataframe = _annotated_dataframe(atom_dataframe, offline_options)

    assert dataframe["is_hetatm"].dtype == "boolean"
    assert dataframe["is_hetatm"].tolist() == [True, False, pd.NA]


def test_annotation_rejects_a_producer_emitting_an_unusable_flag(
    offline_options: EngineOptions,
) -> None:
    atom_dataframe = pd.DataFrame(
        {
            "pdb_id": ["1ABC"],
            "chain_id": ["A"],
            "res_num": ["1"],
            "atom_name": ["N"],
            "is_hetatm": ["probably"],
        }
    )

    with pytest.raises(ValueError, match="'is_hetatm'"):
        _annotated_dataframe(atom_dataframe, offline_options)


def test_every_declared_boolean_column_survives_the_pipeline_as_true_false(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:

    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        tmp_path / "work",
        output_filename="atoms.csv",
        run_id="boolean-contract",
        engine_options=offline_options,
    )

    text = Path(result.csv_path).read_text(encoding="utf-8")
    header = text.splitlines()[0].split(",")
    dataframe = pd.read_csv(result.csv_path)
    declared = declared_boolean_columns()
    present = [name for name in header if name in declared]

    assert present, "expected at least one declared boolean column in the output"
    for name in present:
        rendered = {value for value in dataframe[name].tolist() if not pd.isna(value)}
        assert rendered <= {True, False}, (name, rendered)

    # pandas reads 1.0 and 0 back as numbers equal to True and False, so the
    # set check above cannot catch them. The raw text is checked as well.
    for line in text.splitlines()[1:]:
        fields = dict(zip(header, line.split(","), strict=True))
        for name in present:
            assert not re.fullmatch(r"[01](\.0)?", fields[name]), (name, fields[name])
