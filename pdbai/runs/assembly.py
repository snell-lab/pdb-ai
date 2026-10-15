"""Joining each table's partial CSV files into one dataset file.

A table's partials are joined under one header, in the order given. Memory
stays bounded however large the dataset is. When every partial has exactly the
output's columns, their rows are copied as bytes, a chunk at a time. Otherwise
each row is read and written again under the output's columns, a row at a
time, with the columns a partial lacks left empty. Both ways give the same
bytes for the same rows, because partials.py writes every partial in the
format the csv module writes here.
"""

from __future__ import annotations

import csv
import io
import itertools
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import BinaryIO

from ..atomic import write_atomically
from .partials import CSV_LINE_TERMINATOR

# Receives the number of partials joined so far, the number to join, and the
# file name of the partial just joined.
type AssemblyProgressCallback = Callable[[int, int, str], None]

# How much of a partial is copied at a time, which bounds the memory a copy
# needs.
_CHUNK_BYTES = 65536
_QUOTE_OR_NEWLINE = re.compile(rb'[\n"]')


def _read_header(path: Path) -> list[str]:
    # Rows cannot be placed under columns that share a name.
    with path.open("r", encoding="utf-8", newline="") as stream:
        header = list(next(csv.reader(stream), ()))
    if not header:
        raise ValueError(f"Partial {path.name} has no header.")
    if len(set(header)) != len(header):
        raise ValueError(f"Partial {path.name} names a column more than once.")
    return header


def _header_line(columns: Sequence[str]) -> bytes:
    buffer = io.StringIO(newline="")
    csv.writer(buffer, lineterminator=CSV_LINE_TERMINATOR).writerow(columns)
    return buffer.getvalue().encode("utf-8")


def _copy_rows(path: Path, output_stream: BinaryIO) -> int:
    """Copy a partial's rows, but not its header, and return their count.

    A row ends at a line feed outside quotes. A quote inside a quoted field
    is doubled, so each quote can simply flip between inside and outside.
    Neither a quote nor a line feed occurs within a longer UTF-8 character,
    so a chunk may end anywhere.

    A file that ends partway through a row raises ValueError, since its last
    row would be joined to the next partial's first.
    """
    # The header's line end is counted like a row's, so the count starts
    # below zero.
    row_count = -1
    in_quotes = False
    last_byte = b""
    with path.open("rb") as input_stream:
        while chunk := input_stream.read(_CHUNK_BYTES):
            rows_start = 0
            for match in _QUOTE_OR_NEWLINE.finditer(chunk):
                if match.group() == b'"':
                    in_quotes = not in_quotes
                elif not in_quotes:
                    row_count += 1
                    if row_count == 0:
                        rows_start = match.end()
            if row_count >= 0:
                output_stream.write(chunk[rows_start:])
            last_byte = chunk[-1:]
    if in_quotes or last_byte != b"\n":
        raise ValueError(f"Partial {path.name} ends partway through a row.")
    return row_count


def _concatenate_partials(
    partials: Sequence[Path],
    output_path: Path,
    output_columns: Sequence[str],
    progress_callback: AssemblyProgressCallback | None,
) -> int:
    row_count = 0
    with output_path.open("wb") as output_stream:
        output_stream.write(_header_line(output_columns))
        for position, path in enumerate(partials, start=1):
            row_count += _copy_rows(path, output_stream)
            if progress_callback is not None:
                progress_callback(position, len(partials), path.name)
    return row_count


def _reconcile_partials(
    partials: Sequence[Path],
    output_path: Path,
    output_columns: Sequence[str],
    progress_callback: AssemblyProgressCallback | None,
) -> int:
    row_count = 0
    with output_path.open("w", encoding="utf-8", newline="") as output_stream:
        writer = csv.DictWriter(
            output_stream,
            fieldnames=output_columns,
            lineterminator=CSV_LINE_TERMINATOR,
        )
        writer.writeheader()
        for position, path in enumerate(partials, start=1):
            with path.open("r", encoding="utf-8", newline="") as input_stream:
                for row in csv.DictReader(input_stream):
                    # The reader collects the fields beyond the header
                    # under the key None.
                    if None in row:
                        raise ValueError(
                            f"Partial {path.name} has a row with more fields "
                            "than its header."
                        )
                    writer.writerow(row)
                    row_count += 1
            if progress_callback is not None:
                progress_callback(position, len(partials), path.name)
    return row_count


def assemble_final_csv(
    partials: Sequence[Path],
    output_path: Path,
    column_order: Sequence[str] = (),
    progress_callback: AssemblyProgressCallback | None = None,
) -> int:
    """Join one table's partials into ``output_path`` and return its row count.

    The rows are written in the order of ``partials``, under one header, and
    the count leaves the header out. The columns of ``column_order`` come
    first, in that order, whether or not a partial has them. Columns that
    only the partials have follow, in the order first seen. Without any
    partials, the file is the header alone. ``progress_callback`` is called
    once for each partial joined.

    The file replaces ``output_path`` only once it is complete. A reader
    sees the previous file or the new one, and an error leaves the previous
    file. A table with no column to write raises ValueError. So does a
    partial that has no header, repeats a column name, ends partway through
    a row, or has a row longer than its header. A partial that does not
    exist raises FileNotFoundError.
    """
    headers = [_read_header(path) for path in partials]
    output_columns = list(
        dict.fromkeys([*column_order, *itertools.chain.from_iterable(headers)])
    )
    if not output_columns:
        raise ValueError(f"No columns are known for {output_path.name}.")
    assemble = (
        _concatenate_partials
        if all(header == output_columns for header in headers)
        else _reconcile_partials
    )
    with write_atomically(output_path) as scratch_path:
        return assemble(partials, scratch_path, output_columns, progress_callback)


def assemble_final_csvs(
    partials_by_table: Mapping[str, Sequence[Path]],
    output_paths_by_table: Mapping[str, Path],
    column_order_by_table: Mapping[str, Sequence[str]],
    progress_callback: AssemblyProgressCallback | None = None,
) -> dict[str, int]:
    """Join each table's partials into its file and return the row counts.

    Each table is joined as ``assemble_final_csv`` does, and the same errors
    are raised. A table without partials is written as its header alone, from
    its column order. A table that ``column_order_by_table`` leaves out has
    only the columns of its partials.

    ``progress_callback`` is called once for each partial joined. Its position
    and total count the partials of all the tables together.
    """
    partial_count = sum(len(partials) for partials in partials_by_table.values())
    row_counts: dict[str, int] = {}
    joined_before = 0
    for table, partials in partials_by_table.items():

        def report(
            position: int, _: int, name: str, joined_before: int = joined_before
        ) -> None:
            if progress_callback is not None:
                progress_callback(joined_before + position, partial_count, name)

        row_counts[table] = assemble_final_csv(
            partials,
            output_paths_by_table[table],
            column_order_by_table.get(table, ()),
            report,
        )
        joined_before += len(partials)
    return row_counts


def read_csv_columns(csv_path: Path) -> list[str]:
    """Return the columns of a CSV file's header, or [] for an empty file."""
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        return list(next(csv.reader(stream), ()))


def read_preview_rows(csv_path: Path, limit: int) -> list[dict[str, object]]:
    """Return the first ``limit`` rows of a CSV file, each by column name."""
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        return [dict(row) for row in itertools.islice(csv.DictReader(stream), limit)]


__all__ = [
    "AssemblyProgressCallback",
    "assemble_final_csv",
    "assemble_final_csvs",
    "read_csv_columns",
    "read_preview_rows",
]
