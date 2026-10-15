"""Partial CSV files, each holding one structure's rows of one table.

A build writes the partials of each structure once its tables are complete.
The work done so far then survives a pause or a crash. A resumed run skips
each structure that is recorded done and has all its partials in place.
When the build ends, assembly joins each table's partials into one dataset
file.

Every partial is UTF-8 and ends its lines with CRLF on every operating
system, as the csv module does. The same rows therefore give the same bytes,
and the same SHA-256, wherever the run is built. Assembly relies on this
format too, since it copies a partial's rows as bytes whenever it can.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from ..atomic import write_atomically
from .storage import RunDirectory

# The line end of every CSV file a run writes, partials and datasets alike.
CSV_LINE_TERMINATOR = "\r\n"


def write_partial_csvs(
    tables: Mapping[str, pd.DataFrame], run_directory: RunDirectory, pdb_id: str
) -> dict[str, Path]:
    """Write one structure's partial CSV for each table and return the paths.

    ``tables`` holds the structure's rows for each table, by table name. The
    paths are returned under the same names. The files go in the partials
    directory of ``run_directory`` and are named after ``pdb_id``.

    Each file is UTF-8 with a header line and no index column, and replaces
    any partial an earlier build left for the structure. A table without rows
    is written as its header alone. A missing value and an empty string are
    both written as an empty field.

    A table without columns raises ValueError before anything is written,
    since its file would have no header to join others under. A ``pdb_id``
    or table name that cannot be part of a file name also raises ValueError.
    When an error stops the writing partway, the files already written are
    removed before it is raised, and the partials they replaced are gone too.
    """
    for table, dataframe in tables.items():
        if dataframe.columns.empty:
            raise ValueError(f"Table {table!r} of {pdb_id} has no columns to write.")
    paths: dict[str, Path] = {}
    try:
        for table, dataframe in tables.items():
            path = run_directory.partial_path_for(pdb_id, table)
            # Assembly copies a partial's rows as bytes and counts them by
            # their line ends. It relies on this encoding and line end, and
            # on quotes inside a field being doubled, as pandas does by
            # default.
            with write_atomically(path) as scratch:
                dataframe.to_csv(
                    scratch,
                    index=False,
                    encoding="utf-8",
                    lineterminator=CSV_LINE_TERMINATOR,
                )
            paths[table] = path
    except BaseException:
        # An earlier build may have recorded the entry done. Files left in
        # place would then be assembled with that build's other partials.
        # Removing them makes the entry count as not done, so it is built
        # again.
        for path in paths.values():
            path.unlink(missing_ok=True)
        raise
    return paths


__all__ = ["CSV_LINE_TERMINATOR", "write_partial_csvs"]
