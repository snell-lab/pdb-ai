from __future__ import annotations

import csv
import dataclasses
from pathlib import Path

import pandas as pd
import pytest

from pdbai.engines.core import EngineOptions
from pdbai.results import (
    DatasetBuildResult,
    NoDatasetError,
    PdbProcessResult,
    RunPausedError,
)
from pdbai.runs import (
    RunState,
    list_run_summaries,
    open_run,
    settle_run,
    write_partial_csvs,
)
from pdbai.runs import settlement as settlement_module


def _run_with_entries(
    working_directory: Path,
    options: EngineOptions,
    done: tuple[str, ...] = ("1ABC",),
    failed: tuple[str, ...] = (),
    pending: tuple[str, ...] = (),
) -> RunState:
    run_state = open_run(working_directory, "run")
    run_state.load_or_initialize(
        [*done, *failed, *pending],
        options.as_settings_dict(),
        schema_fingerprint="test-schema",
    )
    tables = [*_TABLES[options.dataset_shape]]
    for pdb_id in done:
        paths = write_partial_csvs(
            {
                table: pd.DataFrame({"pdb_id": [pdb_id, pdb_id], "x": [1.0, 2.0]})
                for table in tables
            },
            run_state.directory,
            pdb_id,
        )
        for table in tables:
            run_state.add_canonical_columns(table, ["pdb_id", "x"])
        run_state.add_available_columns(["pdb_id", "x", "y"])
        run_state.update_entry(
            pdb_id,
            state="done",
            partial_paths={
                table: path.relative_to(run_state.directory.root).as_posix()
                for table, path in paths.items()
            },
        )
    for pdb_id in failed:
        run_state.update_entry(pdb_id, state="failed", error="invalid coordinates")
    return run_state


_TABLES = {
    "flat": ("flat",),
    "star_schema": ("atoms", "residues", "chains", "structures", "citations"),
}


def _settle(
    run_state: RunState,
    working_directory: Path,
    options: EngineOptions,
    *,
    output_filename: str | None = None,
    paused: bool = False,
    publish: bool = True,
    column_filter_list: tuple[str, ...] = (),
) -> DatasetBuildResult:
    manifest = run_state.manifest()
    return settle_run(
        run_state=run_state,
        working_directory=working_directory,
        output_filename=output_filename,
        pdb_ids=list(manifest["pdb_ids"]),
        pdb_results=[
            PdbProcessResult(
                pdb_id=pdb_id,
                state=entry.get("state", "pending"),
                download_ok=True,
                parse_ok=True,
            )
            for pdb_id, entry in manifest["entries"].items()
        ],
        warnings=[],
        options=options,
        column_filter_mode="include" if column_filter_list else None,
        column_filter_list=column_filter_list,
        paused=paused,
        progress_callback=None,
        publish=publish,
        source="download",
    )


def test_columns_follow_the_run_order_not_the_order_entries_finished(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    run_state = open_run(tmp_path, "run")
    run_state.load_or_initialize(
        ["1ABC", "2XYZ"],
        offline_options.as_settings_dict(),
        schema_fingerprint="test-schema",
    )
    columns_by_id = {
        "1ABC": ["pdb_id", "first", "shared"],
        "2XYZ": ["pdb_id", "second", "shared"],
    }
    # 2XYZ finished first, so the columns were recorded in its order.
    for pdb_id in ("2XYZ", "1ABC"):
        columns = columns_by_id[pdb_id]
        paths = write_partial_csvs(
            {"flat": pd.DataFrame({column: [pdb_id] for column in columns})},
            run_state.directory,
            pdb_id,
        )
        run_state.add_canonical_columns("flat", columns)
        run_state.update_entry(
            pdb_id,
            state="done",
            partial_paths={
                "flat": paths["flat"].relative_to(run_state.directory.root).as_posix()
            },
        )

    result = _settle(run_state, tmp_path, offline_options)

    header = ["pdb_id", "first", "shared", "second"]
    assert result.csv_path.read_text(encoding="utf-8").splitlines()[0] == ",".join(
        header
    )
    assert run_state.canonical_columns_by_table()["flat"] == header


def test_a_settled_run_is_completed_and_published(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    run_state = _run_with_entries(tmp_path, offline_options, done=("1ABC", "2XYZ"))

    result = _settle(run_state, tmp_path, offline_options)

    assert result.csv_path == tmp_path / "pdb_atom_site_dataset.csv"
    assert result.atom_count == 4
    assert result.column_count == 2
    assert result.preview_rows[0] == {"pdb_id": "1ABC", "x": "1.0"}
    assert result.column_filter is None
    assert result.paused is False
    manifest = run_state.manifest()
    assert manifest["run_status"] == "completed"
    assert manifest["published_filename"] == "pdb_atom_site_dataset.csv"
    assert [receipt["role"] for receipt in manifest["provenance"]["outputs"]] == [
        "dataset_flat",
        "published_dataset_flat",
    ]


def test_a_run_settled_again_is_published_over_its_earlier_copy(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    run_state = _run_with_entries(
        tmp_path, offline_options, done=("1ABC",), pending=("2XYZ",)
    )

    paused = _settle(run_state, tmp_path, offline_options, paused=True)
    finished = _settle(run_state, tmp_path, offline_options)

    assert paused.csv_path == finished.csv_path
    assert [path.name for path in tmp_path.glob("*.csv")] == [
        "pdb_atom_site_dataset.csv"
    ]


def test_a_paused_run_is_assembled_from_its_done_entries(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    run_state = _run_with_entries(
        tmp_path, offline_options, done=("1ABC",), pending=("2XYZ",)
    )

    result = _settle(run_state, tmp_path, offline_options, paused=True)

    assert result.atom_count == 2
    assert result.paused is True
    assert "1 completed structure(s)" in result.warnings[-1]
    assert run_state.manifest()["run_status"] == "paused"


def test_a_run_paused_before_any_entry_is_done_stays_paused(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    run_state = _run_with_entries(tmp_path, offline_options, done=(), pending=("1ABC",))

    with pytest.raises(RunPausedError):
        _settle(run_state, tmp_path, offline_options, paused=True)

    assert run_state.manifest()["run_status"] == "paused"
    assert run_state.directory.report_path.exists()


def test_without_publishing_the_result_points_into_the_run_directory(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    options = dataclasses.replace(offline_options, dataset_shape="star_schema")
    run_state = _run_with_entries(tmp_path, options)

    result = _settle(run_state, tmp_path, options, publish=False)

    root = run_state.directory.root
    assert result.csv_path == root / "dataset_atoms.csv"
    assert result.additional_csv_paths["residues"] == root / "dataset_residues.csv"
    assert list(tmp_path.glob("*.csv")) == []
    manifest = run_state.manifest()
    assert "published_filename" not in manifest
    assert all(
        str(receipt["role"]).startswith("dataset_")
        for receipt in manifest["provenance"]["outputs"]
    )


def test_settlement_records_why_no_entry_produced_data(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    run_state = _run_with_entries(tmp_path, offline_options, done=(), failed=("1ABC",))

    with pytest.raises(NoDatasetError, match="1ABC: invalid coordinates"):
        _settle(run_state, tmp_path, offline_options)

    manifest = run_state.manifest()
    assert manifest["run_status"] == "error"
    assert "1ABC: invalid coordinates" in manifest["error"]
    with run_state.directory.report_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["state"] == "failed"
    assert run_state.directory.summary_path.exists()


def test_the_reason_names_only_the_first_failed_entries(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    failed = tuple(f"{number}ABC" for number in range(1000, 1025))
    run_state = _run_with_entries(tmp_path, offline_options, done=(), failed=failed)

    with pytest.raises(NoDatasetError, match="1019ABC.*; and 5 more$"):
        _settle(run_state, tmp_path, offline_options)


def test_a_run_that_cannot_be_published_is_left_in_error(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    run_state = _run_with_entries(tmp_path, offline_options)
    (tmp_path / "taken.csv").mkdir()

    with pytest.raises(ValueError, match="not a file"):
        _settle(run_state, tmp_path, offline_options, output_filename="taken.csv")

    manifest = run_state.manifest()
    assert manifest["run_status"] == "error"
    assert "not a file" in manifest["error"]
    # Every entry is done, and the run is still not taken for a completed one.
    assert list_run_summaries(tmp_path)[0]["status"] == "error"


def test_running_out_of_memory_is_recorded_with_what_to_do(
    tmp_path: Path, offline_options: EngineOptions, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_state = _run_with_entries(tmp_path, offline_options)

    def out_of_memory(*args: object, **kwargs: object) -> dict[str, int]:
        raise MemoryError

    monkeypatch.setattr(settlement_module, "assemble_final_csvs", out_of_memory)

    with pytest.raises(RuntimeError, match="ran out of memory.*resume the run"):
        _settle(run_state, tmp_path, offline_options)

    manifest = run_state.manifest()
    assert manifest["run_status"] == "error"
    assert "runs/run/partials/" in manifest["error"]


def test_a_recorded_partial_outside_the_run_is_not_assembled(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    run_state = _run_with_entries(tmp_path, offline_options, done=("1ABC", "2XYZ"))
    outside = tmp_path / "secret.csv"
    outside.write_bytes(b"pdb_id,x\r\nSECRET,0\r\n")
    run_state.update_entry("2XYZ", partial_paths={"flat": str(outside)})

    result = _settle(run_state, tmp_path, offline_options)

    assert result.atom_count == 2
    assert b"SECRET" not in result.csv_path.read_bytes()


def test_the_result_says_what_the_column_filter_asked_for_and_found(
    tmp_path: Path, offline_options: EngineOptions
) -> None:
    run_state = _run_with_entries(tmp_path, offline_options)

    result = _settle(
        run_state, tmp_path, offline_options, column_filter_list=("x", "absent")
    )

    assert result.column_filter is not None
    assert result.column_filter.to_dict() == {
        "mode": "include",
        "matched_columns": ["x"],
        "missing_columns": ["absent"],
        "requested_columns": ["x", "absent"],
        "available_columns": ["pdb_id", "x", "y"],
    }
    assert "absent" in result.warnings[-1]
