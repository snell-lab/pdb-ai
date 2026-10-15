from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from pdbai.runs import RunDirectory, write_partial_csvs


def test_partial_writers_give_each_table_a_file_named_after_the_structure(
    tmp_path: Path,
) -> None:
    run_directory = RunDirectory(tmp_path)
    flat_paths = write_partial_csvs(
        {"flat": pd.DataFrame({"note": ["a,b"]})}, run_directory, "1ABC"
    )
    star_paths = write_partial_csvs(
        {
            "atoms": pd.DataFrame({"pdb_id": ["1ABC"]}),
            "residues": pd.DataFrame({"residue_key": ["1ABC|A|1"]}),
        },
        run_directory,
        "2XYZ",
    )

    partials = tmp_path / "partials"
    assert flat_paths == {"flat": partials / "flat" / "1ABC.csv"}
    assert flat_paths["flat"].read_bytes() == b'note\r\n"a,b"\r\n'
    assert star_paths == {
        "atoms": partials / "atoms" / "2XYZ.csv",
        "residues": partials / "residues" / "2XYZ.csv",
    }


@pytest.mark.parametrize("line_separator", ["\n", "\r\n"])
def test_partials_are_the_same_bytes_on_every_operating_system(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line_separator: str
) -> None:
    monkeypatch.setattr(os, "linesep", line_separator)

    paths = write_partial_csvs(
        {"flat": pd.DataFrame({"x": [1, 2], "note": ["a\nb", None]})},
        RunDirectory(tmp_path),
        "1ABC",
    )

    assert paths["flat"].read_bytes() == b'x,note\r\n1,"a\nb"\r\n2,\r\n'


def test_a_table_without_rows_is_written_as_its_header(tmp_path: Path) -> None:
    paths = write_partial_csvs(
        {"flat": pd.DataFrame({"a": [], "b": []})}, RunDirectory(tmp_path), "1ABC"
    )

    assert paths["flat"].read_bytes() == b"a,b\r\n"


def test_a_table_without_columns_is_refused_before_anything_is_written(
    tmp_path: Path,
) -> None:
    # pandas writes such a table as blank lines with no header, which
    # assembly cannot join.
    tables = {
        "atoms": pd.DataFrame({"pdb_id": ["1ABC"]}),
        "residues": pd.DataFrame(index=range(3)),
    }

    with pytest.raises(ValueError, match="'residues' of 1ABC has no columns"):
        write_partial_csvs(tables, RunDirectory(tmp_path), "1ABC")

    assert not (tmp_path / "partials").exists()


class _UnwritableDataFrame(pd.DataFrame):
    def to_csv(self, *args: object, **kwargs: object) -> None:  # type: ignore[override]
        raise OSError(28, "No space left on device")


def test_a_failed_write_leaves_none_of_the_tables_it_wrote(tmp_path: Path) -> None:
    run_directory = RunDirectory(tmp_path)
    earlier = write_partial_csvs(
        {
            "atoms": pd.DataFrame({"pdb_id": ["old"]}),
            "residues": pd.DataFrame({"pdb_id": ["old"]}),
        },
        run_directory,
        "1ABC",
    )

    with pytest.raises(OSError, match="No space left"):
        write_partial_csvs(
            {
                "atoms": pd.DataFrame({"pdb_id": ["new"]}),
                "residues": _UnwritableDataFrame({"pdb_id": ["new"]}),
            },
            run_directory,
            "1ABC",
        )

    # A new atoms partial beside the old residues partial would assemble the
    # entry from two builds. With the atoms partial gone, the entry counts
    # as not done and is processed again.
    assert not earlier["atoms"].exists()
    assert earlier["residues"].read_bytes() == b"pdb_id\r\nold\r\n"
