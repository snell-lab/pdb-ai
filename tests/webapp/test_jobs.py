from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from flask import Flask
from flask.testing import FlaskClient
from pytest import LogCaptureFixture, MonkeyPatch

from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import EngineOptions
from pdbai.orchestration import api as api_module
from pdbai.orchestration.build import DatasetBuild
from pdbai.results import ColumnFilterSummary, DatasetBuildResult, PdbProcessResult
from pdbai.runs import RunState, open_run
from pdbai.schema import schema_fingerprint
from pdbai.webapp import create_app, jobs
from pdbai.webapp.factory import job_service as _service
from pdbai.webapp.jobs import JobService, RunAlreadyActiveError, effective_status
from webapp.conftest import _wait_for_job

# Options that run no engine needing an install or the network.
_OPTIONS = EngineOptions(values={"include_rabdam": False})


def _app(tmp_path: Path, monkeypatch: MonkeyPatch) -> tuple[Flask, FlaskClient]:
    monkeypatch.chdir(tmp_path)
    app = create_app()
    app.testing = True
    return app, app.test_client()


def _status(client: FlaskClient, run_id: str) -> Mapping[str, Any]:
    response = client.get(f"/api/runs/{run_id}/status")
    assert response.status_code == 200
    return cast(Mapping[str, Any], response.get_json())


def _spawn(service: JobService, run_id: str, *, resume: bool = False) -> None:
    service.spawn_download_build(
        run_id,
        pdb_ids=["1ABC", "2XYZ"],
        engine_options=_OPTIONS,
        column_filter_mode=None,
        column_filter_list=None,
        resume=resume,
    )


def _result(tmp_path: Path) -> DatasetBuildResult:
    return DatasetBuildResult(
        csv_path=tmp_path / "dataset.csv",
        run_report_path=tmp_path / "run_report.csv",
        run_summary_path=tmp_path / "run_summary.md",
        atom_count=7,
        column_count=3,
        pdb_ids=["1ABC", "2XYZ"],
        preview_rows=[],
        warnings=["careful"],
        pdb_results=[
            PdbProcessResult(pdb_id="1ABC", state="done"),
            PdbProcessResult(pdb_id="2XYZ", state="refused"),
        ],
        column_filter=ColumnFilterSummary(
            mode="exclude",
            matched_columns=["note"],
            missing_columns=[],
            requested_columns=["note"],
            available_columns=["note", "x"],
        ),
    )


def _no_downloads(*_args: object, **_kwargs: object) -> None:
    """Record no entry, as a cancelled build would, so the run pauses."""
    return None


def _finished_build(tmp_path: Path) -> Callable[..., DatasetBuildResult]:
    def build(**_kwargs: object) -> DatasetBuildResult:
        return _result(tmp_path)

    return build


def _run_with_done_entry(working_directory: Path, run_id: str) -> RunState:
    """Return a download run paused with 1ABC done and 2XYZ pending.

    Its settings are those a build under _OPTIONS records.
    """
    active = get_engine_registry().with_defaults(_OPTIONS)
    settings: dict[str, object] = {
        "column_filter_mode": None,
        "column_filter_list": [],
        **active.as_settings_dict(),
    }
    run_state = open_run(working_directory, run_id)
    run_state.load_or_initialize(
        ["1ABC", "2XYZ"],
        settings,
        schema_fingerprint=schema_fingerprint(active, source="download"),
    )
    partial = run_state.directory.partials_directory / "1ABC.csv"
    partial.write_text("pdb_id,atom_name\n1ABC,CA\n", encoding="utf-8")
    run_state.update_entry(
        "1ABC", state="done", partial_paths={"flat": "partials/1ABC.csv"}
    )
    return run_state


def _job_file(tmp_path: Path, run_id: str) -> Path:
    return open_run(tmp_path / "workspace", run_id).directory.job_state_path


def test_a_second_worker_cannot_start_on_a_live_run(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)
    open_run(tmp_path / "workspace", "busy").directory.ensure()

    with service.claimed("busy"):
        started = client.post("/api/build", json={"pdbIds": ["1ABC"], "runId": "busy"})
        resumed = client.post("/api/runs/busy/resume")
        with pytest.raises(RunAlreadyActiveError), service.claimed("busy"):
            pass
        # The refused claim must not release the claim it collided with.
        assert service.is_held("busy")

    assert started.status_code == 409
    assert resumed.status_code == 409
    # Leaving the block releases its claim, so the run can be claimed again.
    with service.claimed("busy"):
        assert service.is_held("busy")
    assert not service.is_held("busy")


def test_a_live_job_blocks_a_claim_and_a_deletion_until_it_ends(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)
    release = threading.Event()

    def blocked_build(**_kwargs: object) -> DatasetBuildResult:
        release.wait(10.0)
        return _result(tmp_path)

    monkeypatch.setattr(jobs, "build_atom_site_dataset", blocked_build)
    open_run(tmp_path / "workspace", "live").directory.ensure()
    _spawn(service, "live")

    assert service.live_run_ids() == frozenset({"live"})
    assert service.is_held("live")
    with pytest.raises(RunAlreadyActiveError), service.claimed("live"):
        pass
    assert client.delete("/api/runs/live").status_code == 409
    assert _status(client, "live")["status"] == "running"

    release.set()
    _wait_for_job(service, "live")
    assert service.live_run_ids() == frozenset()
    assert client.delete("/api/runs/live").status_code == 200


def test_a_failed_build_is_recorded_logged_and_released(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)

    def process_downloads(build: DatasetBuild, **_kwargs: object) -> None:
        build.report_progress("download", "1ABC")
        raise OSError("disk full")

    monkeypatch.setattr(api_module, "process_downloads", process_downloads)
    with caplog.at_level(logging.ERROR, logger="pdbai.webapp.jobs"):
        _spawn(service, "failing")
        _wait_for_job(service, "failing")

    status = _status(client, "failing")
    assert status["status"] == "error"
    assert status["error"] == "disk full"
    assert "The build of run 'failing' failed." in caplog.text
    assert "OSError: disk full" in caplog.text
    run_state = open_run(tmp_path / "workspace", "failing")
    assert not run_state.is_held()
    manifest = run_state.manifest()
    assert (manifest["run_status"], manifest["error"]) == ("error", "disk full")
    # The build held the run, so its state is on disk and served from there.
    assert service.in_memory_status("failing") is None
    assert run_state.read_job_state() == {
        "status": "error",
        "progress": {"stage": "download", "current": 1, "total": 2, "subject": "1ABC"},
        "result": None,
        "error": "disk full",
    }


def test_an_error_without_a_message_is_named_by_its_type(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)

    def exit_build(**_kwargs: object) -> DatasetBuildResult:
        raise SystemExit

    monkeypatch.setattr(jobs, "build_atom_site_dataset", exit_build)
    _spawn(service, "exited")
    _wait_for_job(service, "exited")

    status = _status(client, "exited")
    assert (status["status"], status["error"]) == ("error", "SystemExit")
    assert not service.is_held("exited")


def test_a_run_paused_before_any_entry_is_done_has_no_result(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)
    monkeypatch.setattr(api_module, "process_downloads", _no_downloads)

    _spawn(service, "early-pause")
    _wait_for_job(service, "early-pause")

    status = _status(client, "early-pause")
    assert (status["status"], status["result"], status["error"]) == (
        "paused",
        None,
        None,
    )
    assert _job_file(tmp_path, "early-pause").is_file()


def test_a_run_paused_with_a_done_entry_keeps_its_partial_result(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)
    _run_with_done_entry(tmp_path / "workspace", "partial")
    monkeypatch.setattr(api_module, "process_downloads", _no_downloads)

    _spawn(service, "partial", resume=True)
    _wait_for_job(service, "partial")

    status = _status(client, "partial")
    assert status["status"] == "paused"
    result = cast(Mapping[str, Any], status["result"])
    assert result["csvUrl"] == "/api/runs/partial/files/dataset.csv"
    assert result["summary"]["pdbCount"] == 1
    assert result["summary"]["requestedCount"] == 2


def test_the_status_carries_a_summary_but_not_every_entry_or_event(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)
    monkeypatch.setattr(jobs, "build_atom_site_dataset", _finished_build(tmp_path))

    _spawn(service, "summary")
    _wait_for_job(service, "summary")

    status = _status(client, "summary")
    assert set(status) == {"runId", "status", "progress", "result", "error"}
    result = cast(Mapping[str, Any], status["result"])
    assert "pdbResults" not in result
    # The refused 2XYZ was requested but is not in the dataset, so pdbCount
    # is 1.
    assert result["summary"] == {
        "pdbCount": 1,
        "requestedCount": 2,
        "atomCount": 7,
        "columnCount": 3,
    }
    assert result["runSummaryUrl"] == "/api/runs/summary/files/run_summary.md"
    assert result["columnFilter"]["matchedColumns"] == ["note"]
    assert result["warnings"] == ["careful"]


def _write_run(
    working_directory: Path, run_id: str, run_status: str, job: object
) -> None:
    run_state = open_run(working_directory, run_id)
    run_state.load_or_initialize(
        ["1ABC"], _OPTIONS.as_settings_dict(), schema_fingerprint="test-schema"
    )
    run_state.set_run_status(
        cast(Any, run_status), error="disk full" if run_status == "error" else None
    )
    run_state.directory.job_state_path.write_text(json.dumps(job), encoding="utf-8")


def test_a_restarted_server_reads_each_settled_job_back(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    working_directory = tmp_path / "workspace"
    result = {"csvUrl": "/api/runs/done/files/dataset.csv"}
    _write_run(
        working_directory,
        "done",
        "completed",
        {"status": "completed", "progress": None, "result": result, "error": None},
    )
    _write_run(
        working_directory,
        "halted",
        "paused",
        {"status": "paused", "progress": None, "result": None, "error": None},
    )
    _write_run(
        working_directory,
        "broken",
        "error",
        {"status": "error", "progress": None, "result": None, "error": "old reason"},
    )
    _app_unused, client = _app(tmp_path, monkeypatch)

    done = _status(client, "done")
    assert (done["status"], done["result"], done["runId"]) == (
        "completed",
        result,
        "done",
    )
    assert _status(client, "halted")["status"] == "paused"
    broken = _status(client, "broken")
    # The manifest and job.json disagree on the error, and the manifest wins.
    assert (broken["status"], broken["error"]) == ("error", "disk full")


@pytest.mark.parametrize(
    "record",
    [
        [1, 2],
        "running",
        {"status": 3},
        {"status": "finished"},
        {"status": "completed", "progress": "half"},
        {"status": "completed", "progress": {"stage": "download", "current": True}},
        {"status": "completed", "result": ["dataset.csv"]},
        {"status": "error", "error": 7},
    ],
)
def test_a_malformed_job_file_is_refused(
    tmp_path: Path, monkeypatch: MonkeyPatch, record: object
) -> None:
    run_state = open_run(tmp_path / "workspace", "damaged")
    run_state.directory.ensure()
    run_state.directory.job_state_path.write_text(json.dumps(record), encoding="utf-8")
    _app_unused, client = _app(tmp_path, monkeypatch)

    response = client.get("/api/runs/damaged/status")

    assert response.status_code == 404
    assert cast(Mapping[str, Any], response.get_json())["error"] == "Run not found."


def test_a_run_another_process_holds_is_running_not_stale(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _app_unused, client = _app(tmp_path, monkeypatch)
    run_state = open_run(tmp_path / "workspace", "elsewhere")
    run_state.load_or_initialize(
        ["1ABC"], _OPTIONS.as_settings_dict(), schema_fingerprint="test-schema"
    )

    # Another handle stands in for another process, since an flock belongs to
    # one open file.
    with open_run(tmp_path / "workspace", "elsewhere").claimed():
        listed = cast(Mapping[str, Any], client.get("/api/runs").get_json())["runs"]
        assert (listed[0]["status"], listed[0]["live"]) == ("running", False)
        assert _status(client, "elsewhere")["status"] == "running"
        assert client.post("/api/runs/elsewhere/resume").status_code == 409

    listed = cast(Mapping[str, Any], client.get("/api/runs").get_json())["runs"]
    assert listed[0]["status"] == "stale"
    assert _status(client, "elsewhere")["status"] == "stale"


def test_a_request_on_a_held_run_never_writes_its_job_file(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, _client_unused = _app(tmp_path, monkeypatch)
    service = _service(app)
    run_state = open_run(tmp_path / "workspace", "held")
    run_state.load_or_initialize(
        ["1ABC"], _OPTIONS.as_settings_dict(), schema_fingerprint="test-schema"
    )
    settled: dict[str, object] = {
        "status": "completed",
        "progress": None,
        "result": {},
        "error": None,
    }
    run_state.write_job_state(settled)

    # The build route refuses a held run before it starts a job. So the test
    # starts the job directly, as a request that races the other process
    # would.
    with open_run(tmp_path / "workspace", "held").claimed():
        _spawn(service, "held", resume=True)
        _wait_for_job(service, "held")

    in_memory = service.in_memory_status("held")
    assert in_memory is not None
    assert in_memory["status"] == "error"
    assert "already being processed" in str(in_memory["error"])
    assert run_state.read_job_state() == settled
    assert run_state.manifest()["run_status"] == "running"


def test_a_refused_job_reports_its_error_once_then_defers_to_the_holder(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)
    run_state = open_run(tmp_path / "workspace", "held")
    run_state.load_or_initialize(
        ["1ABC"], _OPTIONS.as_settings_dict(), schema_fingerprint="test-schema"
    )

    with open_run(tmp_path / "workspace", "held").claimed():
        _spawn(service, "held", resume=True)
        _wait_for_job(service, "held")

        refused = _status(client, "held")
        assert refused["status"] == "error"
        assert "already being processed" in str(refused["error"])
        holder = _status(client, "held")
        assert (holder["status"], holder["error"]) == ("running", None)


def test_a_request_with_other_settings_keeps_a_completed_result(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, _client_unused = _app(tmp_path, monkeypatch)
    service = _service(app)
    run_state = _run_with_done_entry(tmp_path / "workspace", "finished")
    run_state.set_run_status("completed")
    settled: dict[str, object] = {
        "status": "completed",
        "progress": None,
        "result": {},
        "error": None,
    }
    run_state.write_job_state(settled)

    service.spawn_download_build(
        "finished",
        pdb_ids=["1ABC"],
        engine_options=EngineOptions(
            values={"include_rabdam": False, "include_sasa": True}
        ),
        column_filter_mode=None,
        column_filter_list=None,
        resume=True,
    )
    _wait_for_job(service, "finished")

    in_memory = service.in_memory_status("finished")
    assert in_memory is not None and "different settings" in str(in_memory["error"])
    assert open_run(tmp_path / "workspace", "finished").read_job_state() == settled


def test_a_build_refused_before_its_claim_leaves_no_run_behind(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)

    def refused_build(**_kwargs: object) -> DatasetBuildResult:
        raise RuntimeError("RABDAM is not installed.")

    monkeypatch.setattr(jobs, "build_atom_site_dataset", refused_build)
    _spawn(service, "never")
    _wait_for_job(service, "never")

    assert _status(client, "never")["error"] == "RABDAM is not installed."
    assert not (tmp_path / "workspace" / "runs" / "never").exists()


def test_progress_is_written_at_most_once_a_second(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, _client_unused = _app(tmp_path, monkeypatch)
    service = _service(app)
    written: list[object] = []

    def process_downloads(build: DatasetBuild, **_kwargs: object) -> None:
        for pdb_id in ("1ABC", "2XYZ", "1ABC"):
            build.report_progress("download", pdb_id)
            state = build.run_state.read_job_state() or {}
            written.append(cast(Mapping[str, Any], state["progress"])["subject"])
        raise OSError("stop")

    monkeypatch.setattr(api_module, "process_downloads", process_downloads)
    _spawn(service, "chatty")
    _wait_for_job(service, "chatty")

    assert written == ["1ABC", "1ABC", "1ABC"]


def test_a_job_state_that_cannot_be_written_is_logged_and_kept(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)

    def read_only(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(RunState, "write_job_state", read_only)
    monkeypatch.setattr(jobs, "build_atom_site_dataset", _finished_build(tmp_path))
    with caplog.at_level(logging.WARNING, logger="pdbai.webapp.jobs"):
        _spawn(service, "doomed")
        _wait_for_job(service, "doomed")

    assert "Could not write the job state of run 'doomed'." in caplog.text
    assert _status(client, "doomed")["status"] == "completed"


def test_pausing_reaches_only_a_live_job(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, client = _app(tmp_path, monkeypatch)
    service = _service(app)
    cancelled: list[bool] = []

    def pausable_build(**kwargs: object) -> DatasetBuildResult:
        token = cast(threading.Event, kwargs["cancel_token"])
        cancelled.append(token.wait(10.0))
        return _result(tmp_path)

    monkeypatch.setattr(jobs, "build_atom_site_dataset", pausable_build)
    open_run(tmp_path / "workspace", "deleting").directory.ensure()
    with service.claimed("deleting"):
        assert client.post("/api/runs/deleting/pause").status_code == 409
    assert not service.request_pause("absent")

    _spawn(service, "pausable")
    assert client.post("/api/runs/pausable/pause").status_code == 200
    _wait_for_job(service, "pausable")

    assert cancelled == [True]
    assert not service.request_pause("pausable")


def test_a_thread_that_cannot_start_releases_the_run_and_its_uploads(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, _client_unused = _app(tmp_path, monkeypatch)
    service = _service(app)
    upload_directory = tmp_path / "uploads"
    upload_directory.mkdir()

    def no_threads(_thread: threading.Thread) -> None:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(threading.Thread, "start", no_threads)
    with pytest.raises(RuntimeError, match="can't start new thread"):
        service.spawn_upload_build(
            "unstarted",
            mmcif_files=[upload_directory / "1abc.cif"],
            engine_options=_OPTIONS,
            column_filter_mode=None,
            column_filter_list=None,
            resume=False,
            upload_directory=upload_directory,
        )

    assert not service.is_held("unstarted")
    assert service.in_memory_status("unstarted") is None
    assert not upload_directory.exists()


def test_forgetting_drops_only_a_job_that_has_ended(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app, _client_unused = _app(tmp_path, monkeypatch)
    service = _service(app)
    release = threading.Event()

    def blocked_build(**_kwargs: object) -> DatasetBuildResult:
        release.wait(10.0)
        raise RuntimeError("stopped")

    monkeypatch.setattr(jobs, "build_atom_site_dataset", blocked_build)
    _spawn(service, "kept")
    service.forget("kept")
    assert service.in_memory_status("kept") is not None

    release.set()
    _wait_for_job(service, "kept")
    # The build never held the run, so its failure is kept only in memory.
    assert service.in_memory_status("kept") is not None
    service.forget("kept")
    assert service.in_memory_status("kept") is None


@pytest.mark.parametrize(
    ("config", "key"),
    [
        ({"MAX_PDB_IDS": 0}, "MAX_PDB_IDS"),
        ({"MAX_PDB_IDS": "1O00"}, "MAX_PDB_IDS"),
        ({"MAX_UPLOAD_FILES": True}, "MAX_UPLOAD_FILES"),
        ({"WORKING_DIRECTORY": 5}, "WORKING_DIRECTORY"),
    ],
)
def test_an_unusable_setting_stops_the_app_from_starting(
    tmp_path: Path, monkeypatch: MonkeyPatch, config: dict[str, object], key: str
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=key):
        create_app(config)


def test_entry_counts_are_unlimited_unless_a_deployment_sets_a_cap(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    default = _service(create_app())
    capped = _service(create_app({"MAX_PDB_IDS": "7", "MAX_UPLOAD_FILES": 3.0}))

    assert (default.max_pdb_ids, default.max_upload_files) == (None, None)
    assert (capped.max_pdb_ids, capped.max_upload_files) == (7, 3)
    assert default.working_directory == tmp_path / "workspace"
    assert default.dataset_directory == tmp_path / "workspace" / "datasets"


@pytest.mark.parametrize(
    ("status", "held", "shown"),
    [
        ("running", True, "running"),
        ("running", False, "stale"),
        ("completed", False, "completed"),
        ("error", True, "error"),
        ("finished", False, "unknown"),
        (None, False, "unknown"),
        (3, False, "unknown"),
    ],
)
def test_the_status_a_client_sees(status: object, held: bool, shown: str) -> None:
    assert effective_status(status, held=held) == shown


def test_a_result_names_no_local_path(tmp_path: Path) -> None:
    upload = tmp_path / "uploads" / "0f" / "1" / "junk.cif"
    result = DatasetBuildResult(
        csv_path=tmp_path / "dataset.csv",
        run_report_path=tmp_path / "run_report.csv",
        run_summary_path=tmp_path / "run_summary.md",
        atom_count=0,
        column_count=0,
        pdb_ids=["JUNK"],
        preview_rows=[],
        warnings=[
            f"JUNK: parse failed: {upload}:1:0: expected a data block",
            f"Partials kept in {tmp_path / 'runs' / 'run'}.",
            "Files from https://files.rcsb.org/download/1abc.cif are unaffected.",
        ],
    )

    scrubbed = jobs._without_local_paths(  # pyright: ignore[reportPrivateUsage]
        result, [upload], tmp_path
    )

    assert scrubbed.warnings == [
        "JUNK: parse failed: junk.cif:1:0: expected a data block",
        f"Partials kept in {Path('runs') / 'run'}.",
        "Files from https://files.rcsb.org/download/1abc.cif are unaffected.",
    ]
