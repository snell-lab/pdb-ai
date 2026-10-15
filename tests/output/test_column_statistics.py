from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from pdbai.reporting.column_statistics import (
    MAX_DISTINCT_COUNT,
    ColumnStatistics,
    read_column_statistics,
)


def test_column_statistics_are_order_independent_for_mixed_kinds(
    tmp_path: Path,
) -> None:
    columns = ["mixed_number", "mixed_boolean", "boolean_and_number", "literal_na"]
    forward_path = tmp_path / "forward.csv"
    reversed_path = tmp_path / "reversed.csv"
    pd.DataFrame(
        {
            "mixed_number": ["oops", "1"],
            "mixed_boolean": ["true", "oops"],
            "boolean_and_number": ["true", "1"],
            "literal_na": ["NA", "N/A"],
        }
    ).to_csv(forward_path, index=False)
    pd.DataFrame(
        {
            "mixed_number": ["1", "oops"],
            "mixed_boolean": ["oops", "true"],
            "boolean_and_number": ["1", "true"],
            "literal_na": ["N/A", "NA"],
        }
    ).to_csv(reversed_path, index=False)

    forward = read_column_statistics(forward_path, columns, rows_per_chunk=1)
    reverse = read_column_statistics(reversed_path, columns, rows_per_chunk=1)

    assert forward == reverse
    assert forward["mixed_number"].kind == "text"
    assert forward["mixed_number"].minimum_length == 1
    assert forward["mixed_number"].maximum_length == 4
    assert forward["mixed_boolean"].kind == "text"
    assert forward["boolean_and_number"] == ColumnStatistics(
        "text", 0.0, 2, minimum_length=1, maximum_length=4
    )
    assert forward["literal_na"].kind == "text"
    assert forward["literal_na"].percent_null == 0.0


def test_column_statistics_summarize_numbers_across_chunks(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame(
        {
            "value": [1.0, None, 2.0, 3.0, 4.0],
            "single": [None, None, 7.5, None, None],
        }
    ).to_csv(path, index=False)

    statistics = read_column_statistics(path, ["value", "single"], rows_per_chunk=2)

    value = statistics["value"]
    assert value.kind == "numeric"
    assert value.percent_null == 20.0
    assert (value.minimum, value.maximum) == (1.0, 4.0)
    assert value.mean == pytest.approx(2.5)
    assert value.standard_deviation == pytest.approx(math.sqrt(1.25))
    assert value.distinct_count == 4
    assert (value.minimum_length, value.maximum_length) == (None, None)
    assert statistics["single"].mean == 7.5
    assert statistics["single"].standard_deviation == 0.0


def test_constant_column_of_large_numbers_has_no_spread(tmp_path: Path) -> None:
    # Unit-cell volumes are large enough that a running sum of their squares
    # loses the column's spread to rounding.
    path = tmp_path / "table.csv"
    pd.DataFrame({"unit_cell_volume": [9876543.1] * 300}).to_csv(path, index=False)

    statistics = read_column_statistics(path, ["unit_cell_volume"], rows_per_chunk=7)

    assert statistics["unit_cell_volume"].standard_deviation == pytest.approx(
        0.0, abs=1e-6
    )


def test_values_that_are_not_finite_numbers_make_a_column_text(
    tmp_path: Path,
) -> None:
    path = tmp_path / "table.csv"
    path.write_text("infinite,negative_infinite,overflowing\n1,2,3\ninf,-inf,1e400\n")

    statistics = read_column_statistics(
        path, ["infinite", "negative_infinite", "overflowing"]
    )

    assert statistics["infinite"] == ColumnStatistics(
        "text", 0.0, 2, minimum_length=1, maximum_length=3
    )
    assert statistics["negative_infinite"].kind == "text"
    assert statistics["overflowing"].kind == "text"


def test_column_statistics_classify_booleans_and_empty_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame(
        {
            "flag": ["True", "false", "TRUE", None],
            "blank": [None, None, None, None],
            "text": ["a", "bcd", "true", "1"],
        }
    ).to_csv(path, index=False)

    statistics = read_column_statistics(
        path, ["flag", "blank", "text"], rows_per_chunk=3
    )

    assert statistics["flag"] == ColumnStatistics("boolean", 25.0, 3)
    assert statistics["blank"] == ColumnStatistics("empty", 100.0, 0)
    assert statistics["text"] == ColumnStatistics(
        "text", 0.0, 4, minimum_length=1, maximum_length=4
    )


def test_booleans_are_read_ignoring_surrounding_spaces(tmp_path: Path) -> None:
    path = tmp_path / "table.csv"
    path.write_text("flag\n true \nFALSE \n")

    statistics = read_column_statistics(path, ["flag"])

    assert statistics["flag"] == ColumnStatistics("boolean", 0.0, 2)


def test_column_statistics_of_a_table_with_no_rows(tmp_path: Path) -> None:
    # An engine table no structure wrote rows for is published as a header.
    path = tmp_path / "table.csv"
    path.write_text("value\n")

    statistics = read_column_statistics(path, ["value"])

    assert statistics["value"] == ColumnStatistics("empty", 0.0, 0)


@pytest.mark.parametrize("rows_per_chunk", [7, 1_000])
def test_distinct_values_are_counted_up_to_the_cap(
    tmp_path: Path, rows_per_chunk: int
) -> None:
    path = tmp_path / "table.csv"
    rows = range(2 * MAX_DISTINCT_COUNT)
    pd.DataFrame(
        {
            "at_cap": [f"v{row % MAX_DISTINCT_COUNT}" for row in rows],
            "past_cap": [f"v{row}" for row in rows],
        }
    ).to_csv(path, index=False)

    statistics = read_column_statistics(
        path, ["at_cap", "past_cap"], rows_per_chunk=rows_per_chunk
    )

    assert statistics["at_cap"].distinct_count == MAX_DISTINCT_COUNT
    assert statistics["past_cap"].distinct_count is None


def test_column_statistics_require_every_column_in_the_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "table.csv"
    pd.DataFrame({"present": ["a"]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="missing"):
        read_column_statistics(path, ["present", "missing"])
