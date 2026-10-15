from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pdbai.runs import RunDirectory, write_partial_csvs
from pdbai.runs.assembly import (
    assemble_final_csv,
    assemble_final_csvs,
    read_preview_rows,
)


def _partial(path: Path, content: bytes) -> Path:
    path.write_bytes(content)
    return path


def _written_partials(tmp_path: Path, *tables: pd.DataFrame) -> list[Path]:
    run_directory = RunDirectory(tmp_path / "run")
    return [
        write_partial_csvs({"flat": table}, run_directory, f"ENTRY{index}")["flat"]
        for index, table in enumerate(tables)
    ]


def test_assembled_tables_end_every_line_the_same_way(tmp_path: Path) -> None:
    partials = _written_partials(
        tmp_path,
        pd.DataFrame({"pdb_id": ["1ABC", "1ABC"], "x": [1.0, 2.0]}),
        pd.DataFrame({"pdb_id": ["2XYZ"], "x": [3.0]}),
    )
    output = tmp_path / "dataset.csv"

    assert assemble_final_csv(partials, output) == 3
    assert output.read_bytes() == (b"pdb_id,x\r\n1ABC,1.0\r\n1ABC,2.0\r\n2XYZ,3.0\r\n")


def test_copying_and_reconciling_give_the_same_bytes(tmp_path: Path) -> None:
    # Each value is one the two paths could write differently: a float that
    # needs 17 digits, a quote, a line break, and two missing values.
    rows = pd.DataFrame(
        {
            "pdb_id": ["1ABC", "1ABC", "1ABC"],
            "note": ['say "hi"', "two\nlines", None],
            "x": [0.30000000000000004, float("nan"), 2.0],
        }
    )
    copied = tmp_path / "copied.csv"
    reconciled = tmp_path / "reconciled.csv"

    assemble_final_csv(_written_partials(tmp_path / "a", rows), copied)
    # A header-only partial with fewer columns makes assembly read and
    # rewrite every row. The output keeps the same columns.
    header_only = pd.DataFrame({"pdb_id": pd.Series([], dtype="string")})
    assemble_final_csv(
        _written_partials(tmp_path / "b", rows, header_only),
        reconciled,
        column_order=["pdb_id", "note", "x"],
    )

    assert copied.read_bytes() == reconciled.read_bytes()


def test_rows_are_counted_across_chunks_and_quoted_line_breaks(
    tmp_path: Path,
) -> None:
    # Each partial is many times the 64 KiB chunk the copy reads, so chunks
    # end at many points inside quoted values.
    rows = pd.DataFrame(
        {
            "pdb_id": ["1ABC"] * 20000,
            "note": ['é "quoted"\nand\r\nbroken, 日本'] * 20000,
        }
    )
    output = tmp_path / "dataset.csv"

    assert assemble_final_csv(_written_partials(tmp_path, rows, rows), output) == 40000
    assembled = pd.read_csv(output)
    assert len(assembled) == 40000
    assert set(assembled["note"]) == {'é "quoted"\nand\r\nbroken, 日本'}


def test_a_header_with_a_quoted_line_break_is_left_out_whole(tmp_path: Path) -> None:
    partials = [
        _partial(tmp_path / "first.csv", b'"a\nb",c\r\n1,2\r\n'),
        _partial(tmp_path / "second.csv", b'"a\nb",c\r\n3,4\r\n'),
    ]
    output = tmp_path / "dataset.csv"

    assert assemble_final_csv(partials, output) == 2
    assert output.read_bytes() == b'"a\nb",c\r\n1,2\r\n3,4\r\n'


def test_requested_columns_come_first_and_columns_a_partial_lacks_are_blank(
    tmp_path: Path,
) -> None:
    partials = [
        _partial(tmp_path / "first.csv", b"pdb_id,a\r\n1ABC,1\r\n"),
        _partial(tmp_path / "second.csv", b"pdb_id,b\r\n2XYZ,2\r\n"),
    ]
    output = tmp_path / "dataset.csv"

    assert assemble_final_csv(partials, output, column_order=["pdb_id", "b"]) == 2
    assert output.read_bytes() == b"pdb_id,b,a\r\n1ABC,,1\r\n2XYZ,2,\r\n"


@pytest.mark.parametrize("cut_short", [b"a,b\r\n1,2", b'a,b\r\n1,"2\r\n'])
def test_a_partial_that_ends_partway_through_a_row_is_refused(
    tmp_path: Path, cut_short: bytes
) -> None:
    partials = [
        _partial(tmp_path / "first.csv", cut_short),
        _partial(tmp_path / "second.csv", b"a,b\r\n3,4\r\n"),
    ]
    output = tmp_path / "dataset.csv"
    output.write_bytes(b"previous")

    with pytest.raises(ValueError, match="first.csv ends partway through a row"):
        assemble_final_csv(partials, output)

    assert output.read_bytes() == b"previous"


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"a,a\r\n1,2\r\n", "names a column more than once"),
        (b"", "has no header"),
        (b"\r\n\r\n", "has no header"),
    ],
)
def test_a_partial_with_an_unusable_header_is_refused(
    tmp_path: Path, content: bytes, message: str
) -> None:
    partial = _partial(tmp_path / "first.csv", content)

    with pytest.raises(ValueError, match=message):
        assemble_final_csv([partial], tmp_path / "dataset.csv")


def test_a_row_longer_than_its_header_is_refused(tmp_path: Path) -> None:
    partial = _partial(tmp_path / "first.csv", b"a,b\r\n1,2,3\r\n")

    with pytest.raises(ValueError, match="more fields than its header"):
        assemble_final_csv([partial], tmp_path / "dataset.csv", column_order=["b"])


def test_a_missing_partial_is_an_error_not_lost_rows(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        assemble_final_csv([tmp_path / "gone.csv"], tmp_path / "dataset.csv")


def test_a_table_without_partials_is_its_header(tmp_path: Path) -> None:
    partial = _partial(tmp_path / "first.csv", b"pdb_id\r\n1ABC\r\n")
    outputs = {"flat": tmp_path / "dataset.csv", "sites": tmp_path / "sites.csv"}

    row_counts = assemble_final_csvs(
        {"flat": [partial], "sites": []}, outputs, {"sites": ["site_id", "metal"]}
    )

    assert row_counts == {"flat": 1, "sites": 0}
    assert outputs["sites"].read_bytes() == b"site_id,metal\r\n"


def test_a_table_without_partials_or_columns_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No columns are known for sites.csv"):
        assemble_final_csvs({"sites": []}, {"sites": tmp_path / "sites.csv"}, {})


def test_progress_counts_the_partials_of_every_table(tmp_path: Path) -> None:
    first = _partial(tmp_path / "first.csv", b"a\r\n1\r\n")
    second = _partial(tmp_path / "second.csv", b"a\r\n2\r\n")
    third = _partial(tmp_path / "third.csv", b"b\r\n3\r\n")
    reports: list[tuple[int, int, str]] = []

    assemble_final_csvs(
        {"atoms": [first, second], "residues": [third]},
        {"atoms": tmp_path / "atoms.csv", "residues": tmp_path / "residues.csv"},
        {},
        lambda current, total, name: reports.append((current, total, name)),
    )

    assert reports == [(1, 3, "first.csv"), (2, 3, "second.csv"), (3, 3, "third.csv")]


def test_an_error_in_the_progress_callback_stops_assembly(tmp_path: Path) -> None:
    partial = _partial(tmp_path / "first.csv", b"a\r\n1\r\n")
    output = tmp_path / "dataset.csv"

    def fail(current: int, total: int, name: str) -> None:
        raise OSError("event log is not writable")

    with pytest.raises(OSError, match="event log"):
        assemble_final_csv([partial], output, progress_callback=fail)

    assert not output.exists()


def test_a_preview_holds_no_more_rows_than_asked_for(tmp_path: Path) -> None:
    table = _partial(tmp_path / "dataset.csv", b"a,b\r\n1,2\r\n3,4\r\n")

    assert read_preview_rows(table, limit=0) == []
    assert read_preview_rows(table, limit=1) == [{"a": "1", "b": "2"}]
    assert len(read_preview_rows(table, limit=25)) == 2
