"""Publishing a run's assembled tables into its working directory.

The command line and the Python API copy each table out of the run directory
to the top of the working directory. The web app does not, and serves a
run's files from its run directory instead. The primary table takes the
requested filename, or the first free of pdb_atom_site_dataset.csv,
pdb_atom_site_dataset_1.csv, and so on. Every other table sits beside the
primary one. Its file takes the primary name with an underscore and the
table's name added before .csv.

The primary table's file carries no table name, even in the star schema,
whose primary table is dataset_atoms.csv in the run directory. A dataset of
either shape is thus published under the name the user asked for. A
requested name is written over, so a run settled again replaces its earlier
copy. An automatic name is claimed instead, so two runs never share one.
"""

from __future__ import annotations

import shutil
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

from ..atomic import write_atomically
from ..provenance import file_receipt
from .storage import output_file_path

_DEFAULT_BASE_NAME = "pdb_atom_site_dataset"


@dataclass(frozen=True)
class PublishedTables:
    """Where a run's tables were published, and receipts for the copies."""

    primary: Path
    # Every other table's copy, by table name.
    additional: dict[str, Path]
    receipts: list[dict[str, object]]


def published_table_paths(
    working_directory: Path, output_filename: str, table_names: Sequence[str]
) -> dict[str, Path]:
    """Return the path each table is published at under a primary filename.

    ``table_names`` lists the run's tables with the primary table first, and
    the paths are returned in that order. The primary table takes
    ``output_filename``, which ends in .csv. A table whose filename is not a
    valid output filename raises ValueError, as when the table name makes it
    too long. So does a filename that names something other than a file.
    """
    primary_table, *other_tables = table_names
    stem = output_filename.removesuffix(".csv")
    return {
        primary_table: output_file_path(working_directory, output_filename),
        **{
            table: output_file_path(working_directory, f"{stem}_{table}.csv")
            for table in other_tables
        },
    }


def _claim_all_or_none(paths: Sequence[Path]) -> bool:
    # Every path is created as an empty file, or none when any of them exists.
    # Any other error that stops the claim partway is raised, after the files
    # already created are removed.
    claimed: list[Path] = []
    try:
        for path in paths:
            with path.open("x"):
                pass
            claimed.append(path)
    except BaseException as exception:
        for path in claimed:
            path.unlink(missing_ok=True)
        if isinstance(exception, FileExistsError):
            return False
        raise
    return True


def _claim_automatic_paths(
    working_directory: Path, table_names: Sequence[str]
) -> dict[str, Path]:
    """Claim the first free automatic name for all tables and return the paths.

    A name is free when none of the tables' files under it exists. Each file
    is claimed by creating it exclusively. Two runs finishing at once
    therefore cannot take the same name, and no run writes over files
    another left behind.
    """
    counter = 0
    while True:
        suffix = f"_{counter}" if counter else ""
        paths = published_table_paths(
            working_directory, f"{_DEFAULT_BASE_NAME}{suffix}.csv", table_names
        )
        if _claim_all_or_none(list(paths.values())):
            return paths
        counter += 1


def _copy_all_or_none(sources: Mapping[str, Path], targets: Mapping[str, Path]) -> None:
    # Every copy is complete in its scratch file before any target is
    # replaced, so a copy that fails leaves every target as it was.
    with ExitStack() as stack:
        for table, target in targets.items():
            scratch = stack.enter_context(write_atomically(target))
            shutil.copyfile(sources[table], scratch)


def publish_tables(
    working_directory: Path,
    output_filename: str | None,
    table_paths: Mapping[str, Path],
) -> PublishedTables:
    """Copy a run's tables into the working directory.

    ``table_paths`` holds the assembled file of every table, with the primary
    table first. The primary table is published as ``output_filename``, or
    under the first free automatic name when that is None. The result holds
    the published paths and a receipt for each copy.

    A requested filename replaces any file of that name, and the other
    tables' files replace theirs. A filename that is not a valid output
    filename raises ValueError.

    A reader never sees a table half-copied. Under an automatic name it may
    see a claimed file that is still empty, and a process killed while
    copying leaves the empty files behind. A copy that fails raises OSError.
    No table is then published, and the files claimed for an automatic name
    are removed.
    """
    table_names = list(table_paths)
    if output_filename is not None:
        targets = published_table_paths(working_directory, output_filename, table_names)
        _copy_all_or_none(table_paths, targets)
    else:
        targets = _claim_automatic_paths(working_directory, table_names)
        try:
            _copy_all_or_none(table_paths, targets)
        except BaseException:
            for target in targets.values():
                target.unlink(missing_ok=True)
            raise
    primary_table, *other_tables = table_names
    return PublishedTables(
        primary=targets[primary_table],
        additional={table: targets[table] for table in other_tables},
        receipts=[
            file_receipt(path, role=f"published_dataset_{table}")
            for table, path in targets.items()
        ],
    )


__all__ = ["PublishedTables", "publish_tables", "published_table_paths"]
