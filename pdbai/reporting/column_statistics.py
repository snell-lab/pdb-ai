"""Per-column statistics of a CSV table, computed in bounded memory.

The run summary's data dictionary describes each assembled table as it lies
on disk, and a table can hold far more rows than fit in memory. The table is
therefore read a chunk of rows at a time. Between chunks, each column keeps
only its counts, its extremes, a running mean and spread, and a capped set of
distinct values. Memory therefore stays flat however many rows the table has.

A column's kind is read from its values rather than from its declaration, so
the statistics describe what the file holds. A column counts as numeric or
Boolean only when every one of its values is, and a single stray value makes
it text. Each chunk's mean and spread are merged exactly into the column's,
so the statistics do not depend on the chunk size beyond rounding.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Distinct values are counted up to this many. Identifier columns would
# otherwise keep every value in memory, and the run summary only needs to set
# low-cardinality columns apart from the rest.
MAX_DISTINCT_COUNT = 100

# Rows read at once. With the table's width, this sets the memory a read
# takes.
DEFAULT_ROWS_PER_CHUNK = 100_000

# What a column's values read as, ignoring the spaces around each value.
# "boolean" means each value is true or false, in any letter case. "numeric"
# means each value parses as a finite number, so "inf", "nan", and "1e400"
# make a column text. "text" covers every other column, and "empty" means the
# column holds no values at all.
type ValueKind = Literal["empty", "boolean", "numeric", "text"]


@dataclass(frozen=True)
class ColumnStatistics:
    """The statistics of one column, taken over every row of a table.

    The percentage of empty cells runs from 0 to 100, and is 0.0 for a table
    with no rows. A numeric column has a ``minimum``, ``maximum``, ``mean``,
    and population standard deviation, which is 0.0 for a lone value. A text
    column has the lengths of its shortest and longest values. A Boolean or
    empty column has neither, and a field its kind does not have is None.
    """

    kind: ValueKind
    percent_null: float
    # None when the column holds more than MAX_DISTINCT_COUNT distinct values.
    distinct_count: int | None
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    standard_deviation: float | None = None
    minimum_length: int | None = None
    maximum_length: int | None = None


@dataclass
class _ColumnAccumulator:
    kind: ValueKind = "empty"
    row_count: int = 0
    null_count: int = 0
    # None once the column holds more than MAX_DISTINCT_COUNT distinct values.
    distinct_values: set[str] | None = field(default_factory=set[str])
    minimum_length: int | None = None
    maximum_length: int | None = None
    # Taken over the chunks whose values are all numbers. They are reported
    # only if the column stays numeric to its last chunk.
    number_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float = 0.0
    squared_deviations: float = 0.0

    def add(self, values: pd.Series) -> None:
        """Take in one chunk of the column's values, read as text."""
        self.row_count += len(values)
        present = values.dropna()
        self.null_count += len(values) - len(present)
        if present.empty:
            return

        if self.distinct_values is not None:
            # One value past the cap is enough to tell the column exceeds it.
            self.distinct_values.update(present.unique()[: MAX_DISTINCT_COUNT + 1])
            if len(self.distinct_values) > MAX_DISTINCT_COUNT:
                self.distinct_values = None

        lengths = present.str.len()
        shortest, longest = int(lengths.min()), int(lengths.max())
        self.minimum_length = (
            shortest
            if self.minimum_length is None
            else min(self.minimum_length, shortest)
        )
        self.maximum_length = (
            longest
            if self.maximum_length is None
            else max(self.maximum_length, longest)
        )

        if self.kind == "text":
            # A text column stays text, so its later chunks need not be read
            # as Booleans or numbers.
            return

        if present.str.strip().str.lower().isin(("true", "false")).all():
            self._merge_kind("boolean")
            return
        # A value that is not a number coerces to NaN, which is not finite.
        numbers = pd.to_numeric(present, errors="coerce")
        if not np.isfinite(numbers).all():
            self._merge_kind("text")
            return
        self._merge_kind("numeric")
        self._add_numbers(numbers)

    def _merge_kind(self, kind: ValueKind) -> None:
        # Chunks that disagree make the column text, whichever comes first.
        if self.kind == "empty":
            self.kind = kind
        elif self.kind != kind:
            self.kind = "text"

    def _add_numbers(self, numbers: pd.Series) -> None:
        # The chunk's mean and squared deviations are merged into the running
        # ones (Chan, Golub, and LeVeque). A running sum of squares would lose
        # the spread of large values to rounding, and give a constant column
        # of unit-cell volumes a nonzero standard deviation.
        count = len(numbers)
        total = self.number_count + count
        chunk_mean = float(numbers.mean())
        delta = chunk_mean - self.mean
        self.mean += delta * count / total
        self.squared_deviations += (
            float(((numbers - chunk_mean) ** 2).sum())
            + delta**2 * self.number_count * count / total
        )
        self.number_count = total
        smallest, largest = float(numbers.min()), float(numbers.max())
        self.minimum = smallest if self.minimum is None else min(self.minimum, smallest)
        self.maximum = largest if self.maximum is None else max(self.maximum, largest)

    def statistics(self) -> ColumnStatistics:
        percent_null = self.null_count / self.row_count * 100 if self.row_count else 0.0
        distinct_count = (
            None if self.distinct_values is None else len(self.distinct_values)
        )
        if self.kind == "numeric":
            return ColumnStatistics(
                self.kind,
                percent_null,
                distinct_count,
                minimum=self.minimum,
                maximum=self.maximum,
                mean=self.mean,
                standard_deviation=math.sqrt(
                    self.squared_deviations / self.number_count
                ),
            )
        if self.kind == "text":
            return ColumnStatistics(
                self.kind,
                percent_null,
                distinct_count,
                minimum_length=self.minimum_length,
                maximum_length=self.maximum_length,
            )
        return ColumnStatistics(self.kind, percent_null, distinct_count)


def read_column_statistics(
    csv_path: Path,
    columns: Sequence[str],
    *,
    rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK,
) -> dict[str, ColumnStatistics]:
    """Return statistics for some columns of a CSV file, keyed by column name.

    Every value is read as text and only an empty cell counts as null, so a
    literal "NA" is a value like any other. Raises ValueError when a column is
    not in the file.
    """
    accumulators = {column: _ColumnAccumulator() for column in columns}
    for chunk in pd.read_csv(
        csv_path,
        # pandas' stubs refuse a Sequence[str] here, since a str is one too,
        # so the columns are passed as a list.
        usecols=list(columns),
        chunksize=rows_per_chunk,
        dtype=str,
        keep_default_na=False,
        na_values=[""],
    ):
        for column, accumulator in accumulators.items():
            accumulator.add(chunk[column])
    return {
        column: accumulator.statistics() for column, accumulator in accumulators.items()
    }


__all__ = [
    "DEFAULT_ROWS_PER_CHUNK",
    "MAX_DISTINCT_COUNT",
    "ColumnStatistics",
    "ValueKind",
    "read_column_statistics",
]
