from __future__ import annotations

from pathlib import Path

import pytest

from pdbai.runs import RunDirectory, validate_output_filename, validate_run_id
from pdbai.runs.storage import (
    deletion_name,
    output_file_path,
    runs_root,
    settings_hash,
)


def test_settings_hash_of_stored_runs_does_not_change() -> None:
    # A stored run resumes only under the hash it was created with.
    assert settings_hash({"a": 1, "b": [1, 2]}) == "e6f20bdc3757e5e7"


def test_settings_hash_refuses_a_value_json_cannot_hold() -> None:
    # A set has no JSON form. Its text would not do either, since a set of
    # strings prints in a different order in each process.
    with pytest.raises(TypeError):
        settings_hash({"x": {1, 2}})


@pytest.mark.parametrize(
    "reserved_name",
    [
        "NUL .csv",
        "CON .csv",
        "aux.csv",
        "con.csv",
        "COM1.tar.csv",
        "COM¹.csv",
        "LPT³.csv",
        "CONIN$.csv",
        "CONOUT$.csv",
    ],
)
def test_output_filenames_refuse_every_windows_device_name(reserved_name: str) -> None:
    with pytest.raises(ValueError, match="reserved filename"):
        validate_output_filename(reserved_name)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("", "must not be empty"),
        ("atoms.csv ", "whitespace"),
        ("atoms.csv.", "period"),
        ("at\x07oms.csv", "control characters"),
        ("at:oms.csv", "reserved filename characters"),
        ("a\ud800.csv", "valid Unicode"),
        ("atoms", "ends in .csv"),
        ("atoms.txt", "ends in .csv"),
        (".csv", "start with a period"),
        ("..csv", "start with a period"),
        (".atoms.csv", "start with a period"),
    ],
)
def test_output_filenames_name_the_rule_they_break(name: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_output_filename(name)


def test_an_output_filename_leaves_room_for_its_scratch_file(tmp_path: Path) -> None:
    longest = "a" * 213 + ".csv"
    assert validate_output_filename(longest) == longest
    # The scratch file for the longest name has a name 255 bytes long. That
    # is the most a common file system allows, and the file can be created.
    (tmp_path / f".{longest}.{'0' * 32}.tmp").touch()

    with pytest.raises(ValueError, match="at most 217 UTF-8 bytes"):
        validate_output_filename("a" * 214 + ".csv")


def test_an_output_filename_cannot_name_a_directory(tmp_path: Path) -> None:
    (tmp_path / "taken.csv").mkdir()

    with pytest.raises(ValueError, match="not a file"):
        output_file_path(tmp_path, "taken.csv")


@pytest.mark.parametrize("run_id", ["-x", ".hidden", "_x", "a" * 129, "é"])
def test_run_ids_refuse_a_bad_start_and_too_many_characters(run_id: str) -> None:
    with pytest.raises(ValueError, match="Run ID"):
        validate_run_id(run_id)


def test_run_ids_may_be_128_characters() -> None:
    assert validate_run_id("a" * 128) == "a" * 128


def test_a_symlinked_runs_directory_is_refused(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    try:
        (workspace / "runs").symlink_to(outside, target_is_directory=True)
    except OSError as exception:
        pytest.skip(f"Symlinks are unavailable: {exception}")

    with pytest.raises(ValueError, match="symbolic link"):
        runs_root(workspace)


def test_each_table_has_its_own_folder_of_partials(tmp_path: Path) -> None:
    run_directory = RunDirectory(tmp_path)

    # Suppose every partial sat in one folder, named after its structure and
    # table. On a file system that ignores case, the flat partial of X__SITES
    # and the sites partial of X would then be one file.
    first = run_directory.partial_path_for("X", "sites")
    second = run_directory.partial_path_for("X__SITES", "flat")

    assert first == tmp_path / "partials" / "sites" / "X.csv"
    assert str(first).casefold() != str(second).casefold()


@pytest.mark.parametrize("pdb_id", ["../outside", "a/b", "", "CON"])
def test_a_partial_cannot_be_named_outside_its_folder(
    tmp_path: Path, pdb_id: str
) -> None:
    with pytest.raises(ValueError):
        RunDirectory(tmp_path).partial_path_for(pdb_id, "flat")


def test_a_recorded_partial_outside_the_partials_directory_is_not_followed(
    tmp_path: Path,
) -> None:
    run_directory = RunDirectory(tmp_path / "run")
    run_directory.ensure()
    inside = run_directory.partials_directory / "1ABC.csv"
    inside.write_text("pdb_id\r\n", encoding="utf-8")
    outside = tmp_path / "secret.csv"
    outside.write_text("pdb_id\r\n", encoding="utf-8")

    assert run_directory.contained_partial_path("partials/1ABC.csv") == inside
    assert run_directory.contained_partial_path(str(outside)) is None
    assert run_directory.contained_partial_path("partials/../../secret.csv") is None
    assert run_directory.contained_partial_path("partials/missing.csv") is None
    assert run_directory.contained_partial_path(None) is None


def test_a_deletion_name_is_never_a_run_id() -> None:
    name = deletion_name("a" * 128)

    assert name.startswith(".deleting-" + "a" * 128 + "-")
    assert name != deletion_name("a" * 128)
    with pytest.raises(ValueError, match="Run ID"):
        validate_run_id(name)
