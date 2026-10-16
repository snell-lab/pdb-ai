from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path

import pytest

from pdbai.atomic import write_text_atomically
from pdbai.runs import RunState, list_run_summaries, open_run
from pdbai.runs import listing as listing_module
from pdbai.runs import state as state_module
from pdbai.runs.storage import deletion_name

SETTINGS: dict[str, object] = {"dataset_shape": "flat"}
PDB_IDS = ["1AAA", "2BBB", "3CCC", "4DDD", "5EEE", "6FFF"]


def _loaded_run(tmp_path: Path, run_id: str = "run") -> RunState:
    run_state = open_run(tmp_path, run_id)
    run_state.load_or_initialize(PDB_IDS, SETTINGS, schema_fingerprint="test-schema")
    return run_state


def _counts(tmp_path: Path) -> dict[str, int]:
    (summary,) = list_run_summaries(tmp_path)
    counts = summary["counts"]
    return {
        state: count
        for state in ("done", "failed", "refused", "pending", "total")
        if (count := counts[state])
    }


def test_a_poll_while_the_log_is_emptied_does_not_lose_later_updates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_state = _loaded_run(tmp_path)
    run_state.update_entry("1AAA", state="done")
    run_state.update_entry("2BBB", state="done")
    assert _counts(tmp_path) == {"done": 2, "pending": 4, "total": 6}

    def poll_before_the_log_is_emptied(path: Path, text: str) -> None:
        if path == run_state.directory.entries_path:
            # The fold has replaced the manifest but not yet emptied the log.
            list_run_summaries(tmp_path)
        write_text_atomically(path, text)

    monkeypatch.setattr(
        state_module, "write_text_atomically", poll_before_the_log_is_emptied
    )
    run_state.add_available_columns(["pdb_id"])
    monkeypatch.undo()

    # Four new lines make the log longer than the one the poll read. The
    # log's size then gives no sign it was emptied, and an offset kept from
    # that poll would skip the first new lines.
    for pdb_id in PDB_IDS[2:]:
        run_state.update_entry(pdb_id, state="failed")

    assert _counts(tmp_path) == {"done": 2, "failed": 4, "total": 6}


def test_polls_follow_the_log_and_the_manifest(tmp_path: Path) -> None:
    run_state = _loaded_run(tmp_path)
    assert _counts(tmp_path) == {"pending": 6, "total": 6}

    run_state.update_entry("1AAA", state="done")
    assert _counts(tmp_path) == {"done": 1, "pending": 5, "total": 6}

    run_state.set_run_status("paused")
    run_state.update_entry("2BBB", state="refused")
    run_state.update_entry("3CCC", state="not a state")
    run_state.update_entry("9ZZZ", row_count=3)

    (summary,) = list_run_summaries(tmp_path)
    assert summary["status"] == "paused"
    assert summary["pdb_count"] == 6
    assert summary["counts"] == {
        "done": 1,
        "failed": 0,
        "refused": 1,
        "pending": 5,
        "total": 7,
    }


def test_a_summary_is_the_callers_to_change(tmp_path: Path) -> None:
    _loaded_run(tmp_path)

    (summary,) = list_run_summaries(tmp_path)
    summary["settings"]["dataset_shape"] = "changed"
    summary["counts"]["done"] = 99

    (again,) = list_run_summaries(tmp_path)
    assert again["settings"] == SETTINGS
    assert again["counts"]["done"] == 0


def test_runs_are_listed_by_when_their_status_last_changed(tmp_path: Path) -> None:
    older = _loaded_run(tmp_path, "older")
    _loaded_run(tmp_path, "newer")
    assert [run["run_id"] for run in list_run_summaries(tmp_path)] == [
        "newer",
        "older",
    ]

    older.set_run_status("paused")

    assert [run["run_id"] for run in list_run_summaries(tmp_path)] == [
        "older",
        "newer",
    ]


def test_a_timestamp_that_is_no_number_does_not_break_the_listing(
    tmp_path: Path,
) -> None:
    _loaded_run(tmp_path, "sound")
    damaged = _loaded_run(tmp_path, "damaged").directory.manifest_path
    manifest = json.loads(damaged.read_text(encoding="utf-8"))
    damaged.write_text(
        json.dumps({**manifest, "status_updated_at": "yesterday", "created_at": True}),
        encoding="utf-8",
    )

    summaries = list_run_summaries(tmp_path)

    assert [run["run_id"] for run in summaries] == ["sound", "damaged"]
    assert summaries[1]["updated_at"] is None


def test_a_deleted_run_is_forgotten(tmp_path: Path) -> None:
    kept = _loaded_run(tmp_path, "kept").directory.root
    deleted = _loaded_run(tmp_path, "deleted").directory.root
    list_run_summaries(tmp_path)
    assert {kept, deleted} <= set(listing_module._CACHE)  # pyright: ignore[reportPrivateUsage]

    shutil.rmtree(deleted)

    assert [run["run_id"] for run in list_run_summaries(tmp_path)] == ["kept"]
    assert deleted not in listing_module._CACHE  # pyright: ignore[reportPrivateUsage]
    assert kept in listing_module._CACHE  # pyright: ignore[reportPrivateUsage]


def test_a_run_has_a_dataset_once_its_primary_table_is_assembled(
    tmp_path: Path,
) -> None:
    run_state = open_run(tmp_path, "run")
    run_state.load_or_initialize(
        PDB_IDS, {"dataset_shape": "star_schema"}, schema_fingerprint="test-schema"
    )
    run_state.directory.dataset_path_for("residues").touch()
    assert list_run_summaries(tmp_path)[0]["has_dataset"] is False

    run_state.directory.dataset_path_for("atoms").touch()

    assert list_run_summaries(tmp_path)[0]["has_dataset"] is True


def test_only_directories_with_a_manifest_are_runs(tmp_path: Path) -> None:
    run_state = _loaded_run(tmp_path)
    runs = run_state.directory.root.parent
    (runs / "no-manifest").mkdir()
    (runs / "a-file").touch()
    try:
        (runs / "linked").symlink_to(run_state.directory.root)
    except OSError as exception:
        pytest.skip(f"Symlinks are unavailable: {exception}")

    assert [run["run_id"] for run in list_run_summaries(tmp_path)] == ["run"]


def test_polls_from_several_threads_agree(tmp_path: Path) -> None:
    run_state = _loaded_run(tmp_path)
    failures: list[BaseException] = []

    def poll() -> None:
        try:
            for _ in range(50):
                list_run_summaries(tmp_path)
        except BaseException as exception:
            failures.append(exception)

    threads = [threading.Thread(target=poll) for _ in range(4)]
    for thread in threads:
        thread.start()
    for number in range(200):
        # Each identifier is new to the manifest. So each update adds an
        # entry while the polling threads count the entries.
        run_state.update_entry(f"NEW{number}", state="done")
    for thread in threads:
        thread.join()

    assert failures == []
    assert _counts(tmp_path) == {"done": 200, "pending": 6, "total": 206}


def test_a_run_directory_being_deleted_is_not_listed(tmp_path: Path) -> None:
    run_state = _loaded_run(tmp_path)
    # The renamed directory's manifest still names the run. Only the
    # directory's name keeps it out of the listing.
    run_state.directory.root.rename(
        run_state.directory.root.with_name(deletion_name("run"))
    )

    assert list_run_summaries(tmp_path) == []
