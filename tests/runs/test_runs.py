from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import pytest

from pdbai.engines.core import EngineOptions
from pdbai.runs import RunDirectory, RunLockedError, list_run_summaries, open_run
from pdbai.runs.storage import output_file_path, settings_hash


def test_settings_hash_is_order_independent_but_value_sensitive() -> None:
    assert settings_hash({"a": 1, "b": True}) == settings_hash({"b": True, "a": 1})
    assert settings_hash({"a": 1}) != settings_hash({"a": 2})


@pytest.mark.parametrize(
    "unsafe_name",
    [
        ".",
        "..",
        "../outside",
        "nested/run",
        r"..\outside",
        "/tmp/outside",
        "NUL",
    ],
)
def test_run_ids_and_output_filenames_reject_unsafe_path_components(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    with pytest.raises(ValueError):
        open_run(tmp_path, unsafe_name)
    with pytest.raises(ValueError):
        output_file_path(tmp_path, unsafe_name)


@pytest.mark.parametrize("unsafe_run_id", ["two words", "fragment#id"])
def test_run_ids_are_url_safe(tmp_path: Path, unsafe_run_id: str) -> None:
    with pytest.raises(ValueError, match="ASCII letters"):
        open_run(tmp_path, unsafe_run_id)


def test_run_and_output_paths_accept_safe_names(tmp_path: Path) -> None:
    state = open_run(tmp_path, "experiment-1")

    assert state.directory.root == (tmp_path / "runs" / "experiment-1").resolve()
    assert output_file_path(tmp_path, "atoms.csv") == (tmp_path / "atoms.csv").resolve()


def test_the_flat_table_is_the_unsuffixed_dataset_file(tmp_path: Path) -> None:
    run_directory = RunDirectory(tmp_path)

    assert run_directory.dataset_path_for("flat") == tmp_path / "dataset.csv"
    assert run_directory.dataset_path_for("atoms") == tmp_path / "dataset_atoms.csv"


def test_resolved_paths_reject_symlinks_that_escape_their_roots(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    runs = tmp_path / "workspace" / "runs"
    runs.mkdir(parents=True)
    run_link = runs / "linked-run"
    output_link = tmp_path / "workspace" / "linked-output.csv"
    try:
        run_link.symlink_to(outside, target_is_directory=True)
        output_link.symlink_to(outside / "output.csv")
    except OSError as exception:
        pytest.skip(f"Symlinks are unavailable: {exception}")

    with pytest.raises(ValueError, match="symbolic link"):
        open_run(tmp_path / "workspace", "linked-run")
    with pytest.raises(ValueError, match="symbolic link"):
        output_file_path(tmp_path / "workspace", "linked-output.csv")


def test_run_manifest_resume_requires_matching_settings_and_existing_partials(
    tmp_path: Path,
) -> None:
    state = open_run(tmp_path, "test-run")
    manifest = state.load_or_initialize(
        ["1ABC"], {"dataset_shape": "flat"}, schema_fingerprint="test-schema"
    )
    assert manifest["pdb_ids"] == ["1ABC"]

    relative_path = "partials/1ABC.csv"
    state.update_entry("1ABC", state="done", partial_paths={"flat": relative_path})
    assert state.done_pdb_ids() == set()

    partial = state.directory.root / relative_path
    partial.write_text("pdb_id\n1ABC\n", encoding="utf-8")
    assert state.done_pdb_ids() == {"1ABC"}

    resumed = open_run(tmp_path, "test-run")
    resumed_manifest = resumed.load_or_initialize(
        ["1ABC", "2XYZ"], {"dataset_shape": "flat"}, schema_fingerprint="test-schema"
    )
    assert resumed_manifest["pdb_ids"] == ["1ABC", "2XYZ"]

    with pytest.raises(ValueError, match="different settings"):
        resumed.load_or_initialize(
            ["1ABC"], {"dataset_shape": "star_schema"}, schema_fingerprint="test-schema"
        )


def test_run_manifest_keeps_one_provenance_session_per_invocation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = 0

    def fake_session(
        *, reference_data: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        nonlocal captured
        captured += 1
        return {
            "captured_at": f"session-{captured}",
            "reference_data": dict(reference_data or {}),
        }

    monkeypatch.setattr(
        "pdbai.runs.state.capture_execution_session",
        fake_session,
    )
    settings = {"dataset_shape": "flat", "include_rabdam": False}
    state = open_run(tmp_path, "provenance-run")
    state.load_or_initialize(["1ABC"], settings, schema_fingerprint="test-schema")
    resumed = open_run(tmp_path, "provenance-run")
    manifest = resumed.load_or_initialize(
        ["1ABC"],
        settings,
        reference_data={"table": {"sha256": "abc"}},
        schema_fingerprint="test-schema",
    )

    provenance = manifest["provenance"]
    assert [session["captured_at"] for session in provenance["execution_sessions"]] == [
        "session-1",
        "session-2",
    ]
    # Each session records the reference data its own invocation used.
    assert [
        session["reference_data"] for session in provenance["execution_sessions"]
    ] == [{}, {"table": {"sha256": "abc"}}]


def test_run_manifest_merges_columns_observed_in_later_structures(
    tmp_path: Path,
) -> None:
    state = open_run(tmp_path, "evolving-schema")
    state.load_or_initialize(
        ["1ABC", "2XYZ"], {"dataset_shape": "flat"}, schema_fingerprint="test-schema"
    )

    state.add_canonical_columns("flat", ["pdb_id", "first_value"])
    state.add_canonical_columns("flat", ["pdb_id", "later_value", "first_value"])

    assert state.canonical_columns_by_table()["flat"] == [
        "pdb_id",
        "first_value",
        "later_value",
    ]


def test_run_summary_counts_logged_entries_and_keeps_the_recorded_status(
    tmp_path: Path,
) -> None:
    state = open_run(tmp_path, "summary")
    state.load_or_initialize(
        ["1ABC", "2XYZ"], {"dataset_shape": "flat"}, schema_fingerprint="test-schema"
    )
    state.update_entry("1ABC", state="done")
    state.update_entry("2XYZ", state="failed", error="bad input")

    summaries = list_run_summaries(tmp_path)

    assert len(summaries) == 1
    # Only settling a run completes it, so a run whose worker stopped after its
    # last entry is not taken for a finished one.
    assert summaries[0]["status"] == "running"
    assert summaries[0]["counts"] == {
        "done": 1,
        "failed": 1,
        "refused": 0,
        "pending": 0,
        "total": 2,
    }


def test_a_corrupt_manifest_reads_as_missing_rather_than_raising(
    tmp_path: Path,
) -> None:
    state = open_run(tmp_path, "corrupt")
    state.directory.ensure()
    state.directory.manifest_path.write_text('{"pdb_ids": ["1ABC"', encoding="utf-8")

    assert open_run(tmp_path, "corrupt").manifest() == {}


def test_last_progress_is_read_from_the_end_of_the_event_log(
    tmp_path: Path,
) -> None:
    state = open_run(tmp_path, "events")
    state.load_or_initialize(
        ["1ABC"], EngineOptions().as_settings_dict(), schema_fingerprint="test-schema"
    )
    # The log grows to about 190 kB, far past the 8 KiB tail the listing
    # reads.
    with state.directory.events_path.open("a", encoding="utf-8") as stream:
        for index in range(5000):
            stream.write(json.dumps({"stage": "download", "current": index}) + "\n")

    summaries = list_run_summaries(tmp_path)

    assert len(summaries) == 1
    assert summaries[0]["last_progress"] == {"stage": "download", "current": 4999}


def test_last_progress_handles_a_log_shorter_than_the_tail_window(
    tmp_path: Path,
) -> None:
    state = open_run(tmp_path, "short")
    state.load_or_initialize(
        ["1ABC"], EngineOptions().as_settings_dict(), schema_fingerprint="test-schema"
    )
    state.directory.events_path.write_text(
        json.dumps({"stage": "finalize", "current": 1}) + "\n", encoding="utf-8"
    )

    summaries = list_run_summaries(tmp_path)

    assert summaries[0]["last_progress"] == {"stage": "finalize", "current": 1}


def test_last_progress_is_absent_for_an_empty_or_unparseable_log(
    tmp_path: Path,
) -> None:
    empty = open_run(tmp_path, "empty")
    empty.load_or_initialize(
        ["1ABC"], EngineOptions().as_settings_dict(), schema_fingerprint="test-schema"
    )
    empty.directory.events_path.write_text("", encoding="utf-8")

    broken = open_run(tmp_path, "broken")
    broken.load_or_initialize(
        ["1ABC"], EngineOptions().as_settings_dict(), schema_fingerprint="test-schema"
    )
    broken.directory.events_path.write_text("{not json\n", encoding="utf-8")

    by_id = {summary["run_id"]: summary for summary in list_run_summaries(tmp_path)}

    assert by_id["empty"]["last_progress"] is None
    assert by_id["broken"]["last_progress"] is None


def test_run_manifest_records_a_selection_as_input_provenance_not_a_setting(
    tmp_path: Path,
) -> None:
    settings = EngineOptions(values={"include_rabdam": False}).as_settings_dict()
    first = {"description": "XFEL", "pdb_ids": ["11YO"], "total_count": 1}
    state = open_run(tmp_path, "selected-run")

    manifest = state.load_or_initialize(
        ["11YO"], settings, selection=first, schema_fingerprint="test-schema"
    )

    assert manifest["selection"] == first
    assert manifest["provenance"]["execution_sessions"][0]["selection"] == first
    assert "selection" not in manifest["settings"]

    # A resume with the same settings is compatible with or without a new
    # selection. The original cohort record stands, and the new selection is
    # kept with its session.
    later = {"description": "XFEL", "pdb_ids": ["11YO", "13CH"], "total_count": 2}
    resumed = open_run(tmp_path, "selected-run").load_or_initialize(
        ["11YO", "13CH"], settings, selection=later, schema_fingerprint="test-schema"
    )
    assert resumed["selection"] == first
    assert resumed["pdb_ids"] == ["11YO", "13CH"]
    assert resumed["provenance"]["execution_sessions"][1]["selection"] == later
    plain = open_run(tmp_path, "selected-run").load_or_initialize(
        ["11YO"], settings, schema_fingerprint="test-schema"
    )
    assert "selection" not in plain["provenance"]["execution_sessions"][2]

    (summary,) = list_run_summaries(tmp_path)
    assert summary["selection"] == first
    untouched = open_run(tmp_path, "typed-run").load_or_initialize(
        ["1ABC"], settings, schema_fingerprint="test-schema"
    )
    assert "selection" not in untouched
    assert (
        next(s for s in list_run_summaries(tmp_path) if s["run_id"] == "typed-run")[
            "selection"
        ]
        is None
    )


def test_resuming_a_run_marks_it_running_and_clears_an_old_error(
    tmp_path: Path,
) -> None:
    state = open_run(tmp_path, "resumed")
    state.load_or_initialize(["1ABC"], {"a": 1}, schema_fingerprint="test-schema")
    state.set_run_status("error", error="network down")

    resumed = open_run(tmp_path, "resumed")
    manifest = resumed.load_or_initialize(
        ["1ABC"], {"a": 1}, schema_fingerprint="test-schema"
    )

    assert manifest["run_status"] == "running"
    assert "error" not in manifest


def test_a_later_status_clears_an_earlier_error(tmp_path: Path) -> None:
    state = open_run(tmp_path, "run")
    state.load_or_initialize(["1ABC"], {"a": 1}, schema_fingerprint="test-schema")
    state.set_run_status("error", error="network down")
    state.set_run_status("completed")

    assert "error" not in open_run(tmp_path, "run").manifest()


def test_only_one_holder_can_process_a_run_at_a_time(tmp_path: Path) -> None:
    first = open_run(tmp_path, "shared")
    second = open_run(tmp_path, "shared")

    with first.claimed(), pytest.raises(RunLockedError), second.claimed():
        pass

    with second.claimed():
        pass


def test_a_run_written_under_another_column_schema_cannot_be_resumed(
    tmp_path: Path,
) -> None:
    state = open_run(tmp_path, "run")
    state.load_or_initialize(["1ABC"], {"a": 1}, schema_fingerprint="old")
    state.update_entry("1ABC", state="done")

    with pytest.raises(ValueError, match="different column schema"):
        open_run(tmp_path, "run").load_or_initialize(
            ["1ABC"], {"a": 1}, schema_fingerprint="new"
        )


def test_a_run_with_nothing_finished_adopts_the_current_schema(
    tmp_path: Path,
) -> None:
    open_run(tmp_path, "run").load_or_initialize(
        ["1ABC"], {"a": 1}, schema_fingerprint="old"
    )

    manifest = open_run(tmp_path, "run").load_or_initialize(
        ["1ABC"], {"a": 1}, schema_fingerprint="new"
    )

    assert manifest["schema_fingerprint"] == "new"


def test_entry_updates_append_to_a_log_instead_of_rewriting_the_manifest(
    tmp_path: Path,
) -> None:
    state = open_run(tmp_path, "run")
    state.load_or_initialize(
        ["1ABC", "2XYZ"], {"a": 1}, schema_fingerprint="test-schema"
    )
    manifest_before = state.directory.manifest_path.read_bytes()

    state.update_entry("1ABC", state="done", row_count=3)

    assert state.directory.manifest_path.read_bytes() == manifest_before
    assert state.directory.entries_path.read_text(encoding="utf-8").count("\n") == 1
    reopened = open_run(tmp_path, "run").manifest()
    assert reopened["entries"]["1ABC"]["state"] == "done"
    assert reopened["entries"]["1ABC"]["row_count"] == 3
    assert reopened["entries"]["2XYZ"] == {"state": "pending"}


def test_settling_a_run_folds_the_log_into_the_manifest(tmp_path: Path) -> None:
    state = open_run(tmp_path, "run")
    state.load_or_initialize(["1ABC"], {"a": 1}, schema_fingerprint="test-schema")
    state.update_entry("1ABC", state="done")

    state.set_run_status("completed")

    assert state.directory.entries_path.read_text(encoding="utf-8") == ""
    stored = json.loads(state.directory.manifest_path.read_text(encoding="utf-8"))
    assert stored["entries"]["1ABC"]["state"] == "done"


def test_a_torn_last_log_line_is_ignored(tmp_path: Path) -> None:
    state = open_run(tmp_path, "run")
    state.load_or_initialize(["1ABC"], {"a": 1}, schema_fingerprint="test-schema")
    state.update_entry("1ABC", state="done")
    with state.directory.entries_path.open("a", encoding="utf-8") as stream:
        stream.write('{"pdb_id": "1ABC", "fields": {"state": "fai')

    assert open_run(tmp_path, "run").manifest()["entries"]["1ABC"]["state"] == "done"


def test_run_summaries_follow_the_log_between_polls(tmp_path: Path) -> None:
    state = open_run(tmp_path, "run")
    state.load_or_initialize(
        ["1ABC", "2XYZ"], {"a": 1}, schema_fingerprint="test-schema"
    )
    assert list_run_summaries(tmp_path)[0]["counts"]["pending"] == 2

    state.update_entry("1ABC", state="done")
    state.update_entry("2XYZ", state="refused")

    counts = list_run_summaries(tmp_path)[0]["counts"]
    assert counts == {"done": 1, "failed": 0, "refused": 1, "pending": 0, "total": 2}
