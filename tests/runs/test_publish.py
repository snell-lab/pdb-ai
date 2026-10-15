from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from pdbai.runs.publish import publish_tables, published_table_paths

STAR_TABLES = ("atoms", "residues", "chains")


def _assembled_tables(directory: Path, tables: tuple[str, ...]) -> dict[str, Path]:
    directory.mkdir(parents=True)
    paths = {table: directory / f"dataset_{table}.csv" for table in tables}
    for table, path in paths.items():
        path.write_bytes(f"{table}_id\r\n1\r\n".encode())
    return paths


def _names(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir() if path.is_file())


def test_automatic_names_count_up_from_the_unnumbered_name(tmp_path: Path) -> None:
    tables = _assembled_tables(tmp_path / "run", ("flat",))

    first = publish_tables(tmp_path, None, tables)
    second = publish_tables(tmp_path, None, tables)

    assert first.primary == tmp_path / "pdb_atom_site_dataset.csv"
    assert second.primary == tmp_path / "pdb_atom_site_dataset_1.csv"
    assert first.additional == {}
    assert second.primary.read_bytes() == b"flat_id\r\n1\r\n"


def test_the_other_tables_are_named_after_the_primary_table(tmp_path: Path) -> None:
    tables = _assembled_tables(tmp_path / "run", STAR_TABLES)

    published = publish_tables(tmp_path, "cohort.v2.csv", tables)

    assert published.primary == tmp_path / "cohort.v2.csv"
    assert published.additional == {
        "residues": tmp_path / "cohort.v2_residues.csv",
        "chains": tmp_path / "cohort.v2_chains.csv",
    }
    assert published.additional["chains"].read_bytes() == b"chains_id\r\n1\r\n"
    assert [receipt["role"] for receipt in published.receipts] == [
        "published_dataset_atoms",
        "published_dataset_residues",
        "published_dataset_chains",
    ]


def test_an_automatic_name_is_free_only_when_every_table_file_is(
    tmp_path: Path,
) -> None:
    # A run whose primary file was moved away left its other tables behind.
    left_behind = tmp_path / "pdb_atom_site_dataset_residues.csv"
    left_behind.write_bytes(b"another run's rows")
    tables = _assembled_tables(tmp_path / "run", STAR_TABLES)

    published = publish_tables(tmp_path, None, tables)

    assert published.primary == tmp_path / "pdb_atom_site_dataset_1.csv"
    assert left_behind.read_bytes() == b"another run's rows"
    assert _names(tmp_path) == [
        "pdb_atom_site_dataset_1.csv",
        "pdb_atom_site_dataset_1_chains.csv",
        "pdb_atom_site_dataset_1_residues.csv",
        "pdb_atom_site_dataset_residues.csv",
    ]


def test_a_requested_name_replaces_the_files_of_that_name(tmp_path: Path) -> None:
    (tmp_path / "mine.csv").write_bytes(b"earlier")
    tables = _assembled_tables(tmp_path / "run", ("flat",))

    published = publish_tables(tmp_path, "mine.csv", tables)

    assert published.primary.read_bytes() == b"flat_id\r\n1\r\n"


def test_a_failed_copy_under_an_automatic_name_leaves_nothing(tmp_path: Path) -> None:
    tables = _assembled_tables(tmp_path / "run", STAR_TABLES)
    tables["chains"].unlink()

    with pytest.raises(FileNotFoundError):
        publish_tables(tmp_path, None, tables)

    assert _names(tmp_path) == []
    # The name was given back, so the next run takes it.
    tables["chains"].write_bytes(b"chains_id\r\n1\r\n")
    assert publish_tables(tmp_path, None, tables).primary.name == (
        "pdb_atom_site_dataset.csv"
    )


def test_a_failed_copy_under_a_requested_name_keeps_the_earlier_files(
    tmp_path: Path,
) -> None:
    tables = _assembled_tables(tmp_path / "run", STAR_TABLES)
    publish_tables(tmp_path, "cohort.csv", tables)
    tables["atoms"].write_bytes(b"atoms_id\r\n2\r\n")
    tables["chains"].unlink()

    with pytest.raises(FileNotFoundError):
        publish_tables(tmp_path, "cohort.csv", tables)

    assert (tmp_path / "cohort.csv").read_bytes() == b"atoms_id\r\n1\r\n"
    assert _names(tmp_path) == [
        "cohort.csv",
        "cohort_chains.csv",
        "cohort_residues.csv",
    ]


def test_a_table_name_that_makes_a_filename_too_long_is_refused(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at most 217 UTF-8 bytes"):
        published_table_paths(tmp_path, "a" * 209 + ".csv", ["atoms", "residues"])


def test_a_symlink_where_a_table_would_be_published_is_refused(
    tmp_path: Path,
) -> None:
    try:
        (tmp_path / "cohort_residues.csv").symlink_to(tmp_path / "elsewhere.csv")
    except OSError as exception:
        pytest.skip(f"Symlinks are unavailable: {exception}")

    with pytest.raises(ValueError, match="symbolic link"):
        published_table_paths(tmp_path, "cohort.csv", ["atoms", "residues"])


def test_a_claim_that_fails_partway_leaves_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables = _assembled_tables(tmp_path / "run", STAR_TABLES)
    real_open: Callable[..., Any] = Path.open

    def refuse_chains(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name.endswith("_chains.csv"):
            raise PermissionError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", refuse_chains)

    with pytest.raises(PermissionError):
        publish_tables(tmp_path, None, tables)

    assert _names(tmp_path) == []
