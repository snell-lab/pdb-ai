from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from pdbai.runs import RunState, open_run, run_table_names
from pdbai.runs import state as state_module
from pdbai.runs.state import read_run_record

SETTINGS: dict[str, object] = {"dataset_shape": "flat"}


def _loaded_run(tmp_path: Path, *pdb_ids: str) -> RunState:
    run_state = open_run(tmp_path, "run")
    run_state.load_or_initialize(
        pdb_ids or ["1ABC"], SETTINGS, schema_fingerprint="test-schema"
    )
    return run_state


def test_an_empty_run_id_is_refused_rather_than_replaced(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Run ID must not be empty"):
        open_run(tmp_path, "")


def test_opening_a_run_creates_nothing(tmp_path: Path) -> None:
    # A reader must not bring back the directory of a run just deleted.
    run_state = open_run(tmp_path, "deleted")

    assert run_state.manifest() == {}
    assert run_state.read_job_state() is None
    assert not (tmp_path / "runs").exists()


_CHANGES: dict[str, Callable[[RunState], None]] = {
    "status": lambda run: run.set_run_status("completed"),
    "entry": lambda run: run.update_entry("1ABC", state="done"),
    "receipts": lambda run: run.set_output_receipts([]),
    "engine tables": lambda run: run.set_engine_tables([]),
    "available columns": lambda run: run.add_available_columns(["pdb_id"]),
    "published filename": lambda run: run.set_published_filename("atoms.csv"),
}


@pytest.mark.parametrize("change", sorted(_CHANGES))
def test_a_run_that_is_not_loaded_cannot_be_changed(
    tmp_path: Path, change: str
) -> None:
    with pytest.raises(RuntimeError, match="is not loaded"):
        _CHANGES[change](open_run(tmp_path, "run"))


def test_resuming_keeps_the_order_of_identifiers_without_repeats(
    tmp_path: Path,
) -> None:
    first = [f"{number}ABC" for number in range(1000, 1500)]
    _loaded_run(tmp_path, *first)

    manifest = open_run(tmp_path, "run").load_or_initialize(
        ["9XYZ", *reversed(first), "9XYZ", "8XYZ"],
        SETTINGS,
        schema_fingerprint="test-schema",
    )

    assert manifest["pdb_ids"] == [*first, "9XYZ", "8XYZ"]
    assert manifest["entries"]["8XYZ"] == {"state": "pending"}


def test_resuming_a_large_run_does_not_search_the_list_for_each_identifier(
    tmp_path: Path,
) -> None:
    pdb_ids = [f"ID{number}" for number in range(100_000)]
    _loaded_run(tmp_path, *pdb_ids)
    resumed = open_run(tmp_path, "run")
    start = time.perf_counter()

    resumed.load_or_initialize(pdb_ids, SETTINGS, schema_fingerprint="test-schema")

    # A search of the list for each identifier took about 20 seconds at
    # this size.
    assert time.perf_counter() - start < 10


def test_a_record_read_while_the_log_is_folded_keeps_every_update(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = _loaded_run(tmp_path)
    writer.update_entry("1ABC", state="done")
    read_json_object = state_module.read_json_object
    folded = False

    def read_then_fold(path: Path) -> dict[str, object] | None:
        nonlocal folded
        manifest = read_json_object(path)
        if not folded:
            # The writer folds the log after the reader reads the manifest
            # and before it reads the log.
            folded = True
            writer.set_run_status("completed")
        return manifest

    monkeypatch.setattr(state_module, "read_json_object", read_then_fold)

    record = read_run_record(writer.directory)

    assert record is not None
    assert record.entries is not None
    assert record.entries["1ABC"]["state"] == "done"
    assert record.manifest["run_status"] == "completed"


def test_an_update_after_a_line_cut_short_is_kept(tmp_path: Path) -> None:
    run_state = _loaded_run(tmp_path)
    with run_state.directory.entries_path.open("ab") as stream:
        stream.write(b'{"pdb_id": "1ABC", "fields": {"sta')

    run_state.update_entry("1ABC", state="done")

    assert open_run(tmp_path, "run").manifest()["entries"]["1ABC"]["state"] == "done"


def test_the_job_state_reads_as_missing_when_it_cannot_be_read(
    tmp_path: Path,
) -> None:
    run_state = open_run(tmp_path, "run")
    run_state.write_job_state({"status": "running"})
    assert run_state.read_job_state() == {"status": "running"}

    run_state.directory.job_state_path.write_bytes(b"\xff\xfe not text")
    assert run_state.read_job_state() is None
    run_state.directory.job_state_path.write_text("[1, 2]", encoding="utf-8")
    assert run_state.read_job_state() is None
    run_state.directory.job_state_path.unlink()
    assert run_state.read_job_state() is None


def test_adopting_a_schema_drops_the_columns_recorded_under_the_old_one(
    tmp_path: Path,
) -> None:
    run_state = open_run(tmp_path, "run")
    run_state.load_or_initialize(["1ABC"], SETTINGS, schema_fingerprint="old")
    # An entry that stopped before it was done has already recorded its
    # columns.
    run_state.add_available_columns(["pdb_id", "dropped_column"])
    run_state.set_engine_tables([{"name": "old_table"}])

    resumed = open_run(tmp_path, "run")
    manifest = resumed.load_or_initialize(["1ABC"], SETTINGS, schema_fingerprint="new")

    assert manifest["schema_fingerprint"] == "new"
    assert "engine_tables" not in manifest
    assert resumed.available_columns() == []


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        ("not json", "unreadable manifest"),
        ({"settings_hash": ""}, "invalid settings hash"),
        ({"settings": None}, "incomplete manifest"),
        ({"entries": None}, "incomplete manifest"),
    ],
)
def test_a_damaged_manifest_cannot_be_resumed(
    tmp_path: Path, damage: str | dict[str, object], message: str
) -> None:
    run_state = _loaded_run(tmp_path)
    path = run_state.directory.manifest_path
    if isinstance(damage, str):
        path.write_text(damage, encoding="utf-8")
    else:
        manifest: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        path.write_text(json.dumps({**manifest, **damage}), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        open_run(tmp_path, "run").load_or_initialize(
            ["1ABC"], SETTINGS, schema_fingerprint="test-schema"
        )


def test_a_status_without_an_error_leaves_no_error(tmp_path: Path) -> None:
    run_state = _loaded_run(tmp_path)
    run_state.set_run_status("error", error="network down")

    run_state.set_run_status("error", error=None)

    assert "error" not in run_state.manifest()


def test_a_manifest_cannot_name_a_base_table_as_an_engine_table() -> None:
    # Only the first record, sites, names an engine table. The others name
    # the flat table, a star-schema table, a path, and sites again.
    manifest = {
        "settings": {"dataset_shape": "flat"},
        "engine_tables": [
            {"name": "sites"},
            {"name": "flat"},
            {"name": "structures"},
            {"name": "../escape"},
            {"name": "sites"},
        ],
    }

    assert run_table_names(manifest) == ["flat", "sites"]


def test_a_run_without_a_directory_is_not_held_and_is_not_created(
    tmp_path: Path,
) -> None:
    run_state = open_run(tmp_path, "absent")

    assert run_state.is_held() is False
    assert not (tmp_path / "runs").exists()


def test_the_probe_sees_a_holder_and_leaves_the_run_free(tmp_path: Path) -> None:
    run_state = _loaded_run(tmp_path)
    assert run_state.is_held() is False

    # Another handle stands in for another process, since an flock belongs to
    # one open file.
    with open_run(tmp_path, "run").claimed():
        assert run_state.is_held() is True

    assert run_state.is_held() is False
    with run_state.claimed():
        assert open_run(tmp_path, "run").is_held() is True
