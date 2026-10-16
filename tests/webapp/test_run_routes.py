from __future__ import annotations

import io
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from flask.testing import FlaskClient
from pytest import LogCaptureFixture, MonkeyPatch
from werkzeug.test import TestResponse

from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import EngineOptions
from pdbai.results import DatasetBuildResult
from pdbai.runs import RunState, open_run
from pdbai.webapp import jobs
from pdbai.webapp.factory import job_service as _service
from pdbai.webapp.jobs import JobService
from webapp.conftest import _client, _file_url, _wait_for_job

# Options that run no engine needing an install or the network.
_OPTIONS = EngineOptions(values={"include_rabdam": False})


def _run(
    tmp_path: Path, run_id: str, pdb_ids: list[str] | None = None, **settings: object
) -> RunState:
    """Return a download run whose settings a resume can turn into options."""
    run_state = open_run(tmp_path / "workspace", run_id)
    run_state.load_or_initialize(
        ["1ABC"] if pdb_ids is None else pdb_ids,
        {
            "column_filter_mode": None,
            "column_filter_list": [],
            **get_engine_registry().with_defaults(_OPTIONS).as_settings_dict(),
            **settings,
        },
        schema_fingerprint="test-schema",
    )
    return run_state


def _result(run_state: RunState) -> DatasetBuildResult:
    return DatasetBuildResult(
        csv_path=run_state.directory.dataset_path_for("flat"),
        run_report_path=run_state.directory.report_path,
        run_summary_path=run_state.directory.summary_path,
        atom_count=1,
        column_count=1,
        pdb_ids=["1ABC"],
        preview_rows=[],
        warnings=[],
    )


def _error_of(response: TestResponse) -> str:
    return str(cast(Mapping[str, object], response.get_json())["error"])


def _listed(client: FlaskClient) -> dict[str, Mapping[str, Any]]:
    response = client.get("/api/runs")
    assert response.status_code == 200
    runs = cast(Mapping[str, Any], response.get_json())["runs"]
    return {run["runId"]: run for run in runs}


def _block_builds(monkeypatch: MonkeyPatch, tmp_path: Path) -> threading.Event:
    """Make every build wait for the returned event, so its run stays live."""
    release = threading.Event()

    def blocked_build(**kwargs: object) -> DatasetBuildResult:
        release.wait(10.0)
        return _result(open_run(tmp_path / "workspace", cast(str, kwargs["run_id"])))

    monkeypatch.setattr(jobs, "build_atom_site_dataset", blocked_build)
    return release


def _spawn(service: JobService, run_id: str) -> None:
    service.spawn_download_build(
        run_id,
        pdb_ids=["1ABC"],
        engine_options=_OPTIONS,
        column_filter_mode=None,
        column_filter_list=None,
        resume=False,
        file_url=_file_url(run_id),
    )


def test_the_runs_route_lists_a_run_no_worker_holds_as_stale(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    # The server stopped after the last entry was recorded and before the run
    # was settled.
    _run(tmp_path, "stopped-run").update_entry("1ABC", state="done")
    _run(tmp_path, "settled-run").set_run_status("error", error="disk full")

    runs = _listed(client)

    assert set(runs["stopped-run"]) == {
        "runId",
        "status",
        "createdAt",
        "updatedAt",
        "settings",
        "pdbCount",
        "selection",
        "counts",
        "lastProgress",
        "hasDataset",
        "error",
        "live",
    }
    assert runs["stopped-run"]["status"] == "stale"
    assert runs["stopped-run"]["live"] is False
    assert runs["stopped-run"]["pdbCount"] == 1
    assert runs["stopped-run"]["counts"]["done"] == 1
    # Settings are data, so their keys keep the manifest's snake_case.
    assert "column_filter_mode" in runs["stopped-run"]["settings"]
    assert runs["settled-run"]["status"] == "error"
    assert runs["settled-run"]["error"] == "disk full"


def test_a_run_this_server_builds_is_listed_live_and_running(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    service = _service(cast(Any, client.application))
    release = _block_builds(monkeypatch, tmp_path)
    _run(tmp_path, "live-run")
    _spawn(service, "live-run")

    listed = _listed(client)["live-run"]

    release.set()
    _wait_for_job(service, "live-run")
    assert (listed["live"], listed["status"]) == (True, "running")


def test_a_symlinked_runs_directory_is_a_server_error_without_its_path(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    client = _client(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (tmp_path / "workspace" / "runs").symlink_to(elsewhere, target_is_directory=True)

    with caplog.at_level(logging.ERROR):
        response = client.get("/api/runs")

    assert response.status_code == 500
    assert _error_of(response) == "The runs directory cannot be read."
    assert "The runs directory was refused." in caplog.text


def test_status_of_an_unknown_run_is_not_found(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/api/runs/absent/status")

    assert response.status_code == 404
    assert _error_of(response) == "Run not found."


def test_status_of_a_run_without_a_job_file_comes_from_its_manifest(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    # A run the command line built has a manifest but no job.json.
    _run(tmp_path, "cli-run").set_run_status("paused")

    payload = cast(Mapping[str, Any], client.get("/api/runs/cli-run/status").get_json())

    assert payload["runId"] == "cli-run"
    assert payload["status"] == "paused"


def test_status_and_listing_agree_on_a_run_that_failed_before_settling(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    run_state = _run(tmp_path, "failed-run")
    # The build recorded its failure in the manifest, and the server stopped
    # before job.json was brought up to date.
    run_state.write_job_state(
        {"status": "running", "progress": None, "result": None, "error": None}
    )
    run_state.set_run_status("error", error="boom")

    status = cast(
        Mapping[str, Any], client.get("/api/runs/failed-run/status").get_json()
    )
    listed = _listed(client)["failed-run"]

    assert (status["status"], status["error"]) == ("error", "boom")
    assert (listed["status"], listed["error"]) == ("error", "boom")


def test_status_of_a_run_left_running_by_a_restart_is_stale(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    run_state = open_run(tmp_path / "workspace", "run-a")
    run_state.write_job_state({"status": "running", "progress": None})

    payload = cast(Mapping[str, Any], client.get("/api/runs/run-a/status").get_json())

    assert payload["status"] == "stale"


def test_run_file_route_serves_only_published_outputs(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    run_state = _run(tmp_path, "run-a", dataset_shape="star_schema")
    directory = run_state.directory
    directory.dataset_path_for("residues").write_text("x\n", encoding="utf-8")
    directory.report_path.write_text("pdb_id\n", encoding="utf-8")
    directory.summary_path.write_text("# Summary\n", encoding="utf-8")
    # These files exist, so each 404 below comes from the rule on what a run
    # publishes and not from a missing file.
    directory.dataset_path_for("flat").write_text("x\n", encoding="utf-8")
    directory.lock_path.write_text("", encoding="utf-8")
    assert directory.manifest_path.is_file()

    report = client.get("/api/runs/run-a/files/run_report.csv")

    assert report.status_code == 200
    assert report.headers["Content-Disposition"] == (
        "attachment; filename=run-a_run_report.csv"
    )
    assert client.get("/api/runs/run-a/files/run_summary.md").data == b"# Summary\n"
    assert client.get("/api/runs/run-a/files/dataset_residues.csv").status_code == 200
    # A star-schema run publishes no dataset.csv, so the route does not
    # serve one even when it is on disk.
    assert client.get("/api/runs/run-a/files/dataset.csv").status_code == 404
    assert client.get("/api/runs/run-a/files/manifest.json").status_code == 404
    assert client.get("/api/runs/run-a/files/run.lock").status_code == 404
    assert client.get("/api/runs/..%2F..%2Fx/files/dataset.csv").status_code in {
        400,
        404,
    }


def test_run_file_route_does_not_find_an_unknown_run_or_a_symlink(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    run_state = _run(tmp_path, "run-a")
    secret = tmp_path / "secret.csv"
    secret.write_text("secret\n", encoding="utf-8")
    run_state.directory.dataset_path_for("flat").symlink_to(secret)

    unknown = client.get("/api/runs/absent/files/dataset.csv")
    linked = client.get("/api/runs/run-a/files/dataset.csv")

    assert (unknown.status_code, _error_of(unknown)) == (404, "Run not found.")
    assert (linked.status_code, _error_of(linked)) == (404, "File not found.")


def test_download_all_includes_machine_readable_provenance(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    state = _run(tmp_path, "publication-run")
    state.directory.dataset_path_for("flat").write_text(
        "pdb_id\n1ABC\n", encoding="utf-8"
    )
    state.directory.report_path.write_text(
        "pdb_id,success\n1ABC,True\n", encoding="utf-8"
    )
    state.directory.summary_path.write_text("# Summary\n", encoding="utf-8")

    response = client.get("/api/runs/publication-run/download-all")

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        assert set(archive.namelist()) == {
            "dataset.csv",
            "manifest.json",
            "run_report.csv",
            "run_summary.md",
        }
        manifest = json.loads(archive.read("manifest.json"))
    assert manifest["provenance"]["execution_sessions"]


def test_download_all_includes_engine_tables_named_in_the_manifest(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    state = _run(tmp_path, "tables-run")
    state.set_engine_tables([{"name": "metal_sites"}, {"name": "../secret"}])
    state.directory.dataset_path_for("flat").write_text(
        "pdb_id\n1ABC\n", encoding="utf-8"
    )
    state.directory.dataset_path_for("metal_sites").write_text(
        "site_id\n", encoding="utf-8"
    )
    (state.directory.root.parent / "secret.csv").write_text("secret", encoding="utf-8")

    response = client.get("/api/runs/tables-run/download-all")

    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        assert {"dataset.csv", "dataset_metal_sites.csv"} <= set(archive.namelist())
        assert not any("secret" in name for name in archive.namelist())


def test_download_all_without_a_table_writes_nothing(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    state = _run(tmp_path, "empty-run")
    state.directory.report_path.write_text("pdb_id\n", encoding="utf-8")

    response = client.get("/api/runs/empty-run/download-all")

    assert response.status_code == 404
    assert not (tmp_path / "workspace" / "tmp").exists()


def test_a_download_archive_is_removed_once_the_response_closes(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    state = _run(tmp_path, "tmp-run")
    state.directory.dataset_path_for("flat").write_text(
        "pdb_id\n1ABC\n", encoding="utf-8"
    )
    system_temp_before = set(Path(tempfile.gettempdir()).glob("download-all-*.zip"))

    response = client.get("/api/runs/tmp-run/download-all")
    with zipfile.ZipFile(io.BytesIO(response.data)) as archive:
        assert "dataset.csv" in archive.namelist()
    response.close()

    assert list((tmp_path / "workspace" / "tmp").iterdir()) == []
    assert set(Path(tempfile.gettempdir()).glob("download-all-*.zip")) == (
        system_temp_before
    )


def test_abandoned_download_archives_are_swept(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    state = _run(tmp_path, "sweep-run")
    state.directory.dataset_path_for("flat").write_text(
        "pdb_id\n1ABC\n", encoding="utf-8"
    )
    scratch = tmp_path / "workspace" / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    abandoned = scratch / "download-all-old.zip"
    abandoned.write_bytes(b"stale")
    recent = scratch / "download-all-recent.zip"
    recent.write_bytes(b"fresh")
    old_time = time.time() - (2 * 60 * 60)
    os.utime(abandoned, (old_time, old_time))

    response = client.get("/api/runs/sweep-run/download-all")

    assert response.status_code == 200
    assert not abandoned.exists()
    assert recent.exists()


def test_each_run_links_to_the_files_in_its_own_run_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    service = _service(cast(Any, client.application))

    def fake_build(**kwargs: object) -> DatasetBuildResult:
        return _result(open_run(tmp_path / "workspace", cast(str, kwargs["run_id"])))

    monkeypatch.setattr(jobs, "build_atom_site_dataset", fake_build)
    payloads = {}
    for run_id, marker in (("run-a", "A"), ("run-b", "B")):
        directory = _run(tmp_path, run_id).directory
        directory.dataset_path_for("flat").write_text(
            f"pdb_id\n{marker}\n", encoding="utf-8"
        )
        directory.report_path.write_text(
            f"pdb_id,success\n{marker},True\n", encoding="utf-8"
        )
        _spawn(service, run_id)
        _wait_for_job(service, run_id)
        status = client.get(f"/api/runs/{run_id}/status").get_json()
        payloads[run_id] = cast(Mapping[str, Any], status)["result"]

    assert payloads["run-a"]["runReportUrl"] == "/api/runs/run-a/files/run_report.csv"
    assert (
        client.get(payloads["run-a"]["runReportUrl"]).data
        == b"pdb_id,success\nA,True\n"
    )
    assert client.get(payloads["run-b"]["csvUrl"]).data == b"pdb_id\nB\n"
    # Nothing is copied beside the workspace for a web run.
    assert list((tmp_path / "workspace" / "datasets").iterdir()) == []


def test_pause_reaches_only_a_run_this_server_builds(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    service = _service(cast(Any, client.application))
    release = _block_builds(monkeypatch, tmp_path)
    _run(tmp_path, "idle-run")
    _run(tmp_path, "live-run")
    _spawn(service, "live-run")

    bad = client.post("/api/runs/..bad/pause")
    idle = client.post("/api/runs/idle-run/pause")
    live = client.post("/api/runs/live-run/pause")

    release.set()
    _wait_for_job(service, "live-run")
    assert bad.status_code == 400
    assert (idle.status_code, _error_of(idle)) == (
        409,
        "Run is not being built by this server.",
    )
    assert live.status_code == 200
    assert live.get_json() == {"status": "pausing", "runId": "live-run"}


def test_resume_starts_a_build_of_the_recorded_run(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    service = _service(cast(Any, client.application))
    _run(tmp_path, "paused-run", ["1ABC", "2XYZ"], column_filter_mode="exclude")
    calls: list[dict[str, object]] = []

    def recording_build(**kwargs: object) -> DatasetBuildResult:
        calls.append(kwargs)
        return _result(open_run(tmp_path / "workspace", "paused-run"))

    monkeypatch.setattr(jobs, "build_atom_site_dataset", recording_build)

    response = client.post("/api/runs/paused-run/resume")
    _wait_for_job(service, "paused-run")

    assert response.status_code == 200
    assert response.get_json() == {"status": "resumed", "runId": "paused-run"}
    (call,) = calls
    assert (call["run_id"], call["resume"], call["pdb_ids"]) == (
        "paused-run",
        True,
        ["1ABC", "2XYZ"],
    )
    assert call["column_filter_mode"] == "exclude"


@pytest.mark.parametrize(
    ("settings", "pdb_ids", "message"),
    [
        ({"source": "upload"}, ["1ABC"], "cannot be resumed from the web app"),
        ({}, [], "No PDB IDs are recorded for this run."),
        ({"column_filter_mode": "sideways"}, ["1ABC"], "column_filter_mode"),
        (
            {"engine_settings": {"include_sasa": "false"}},
            ["1ABC"],
            "Run settings are unusable.",
        ),
    ],
)
def test_resume_refuses_a_run_it_cannot_rebuild(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    settings: dict[str, object],
    pdb_ids: list[str],
    message: str,
) -> None:
    client = _client(tmp_path, monkeypatch)
    service = _service(cast(Any, client.application))
    _run(tmp_path, "stored-run", pdb_ids, **settings)

    response = client.post("/api/runs/stored-run/resume")

    assert response.status_code == 409
    assert message in _error_of(response)
    assert service.in_memory_status("stored-run") is None


def test_resume_does_not_find_a_run_without_a_manifest(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    (tmp_path / "workspace" / "runs" / "bare-run").mkdir(parents=True)

    bare = client.post("/api/runs/bare-run/resume")
    unknown = client.post("/api/runs/absent/resume")

    assert (bare.status_code, _error_of(bare)) == (404, "Run not found.")
    assert (unknown.status_code, _error_of(unknown)) == (404, "Run not found.")


def test_resume_refuses_a_run_another_process_holds_without_touching_it(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    run_state = _run(tmp_path, "held-run")
    job_state: dict[str, object] = {
        "status": "paused",
        "progress": None,
        "result": None,
        "error": None,
    }
    run_state.write_job_state(job_state)

    # Another handle stands in for another process, since an flock belongs to
    # one open file.
    with open_run(tmp_path / "workspace", "held-run").claimed():
        response = client.post("/api/runs/held-run/resume")

    assert (response.status_code, _error_of(response)) == (
        409,
        "Run 'held-run' is already being built or deleted; try again later.",
    )
    assert run_state.read_job_state() == job_state


def test_delete_refuses_a_run_another_process_holds(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    run_state = open_run(tmp_path / "workspace", "run-a")
    run_state.directory.ensure()

    with open_run(tmp_path / "workspace", "run-a").claimed():
        # Another handle stands in for another process, since an flock
        # belongs to one open file.
        response = client.delete("/api/runs/run-a")
        assert response.status_code == 409
        assert run_state.directory.root.is_dir()

    deleted = client.delete("/api/runs/run-a")

    assert deleted.get_json() == {"status": "deleted", "runId": "run-a"}
    assert not run_state.directory.root.exists()
    assert list((tmp_path / "workspace" / "runs").iterdir()) == []


def test_delete_refuses_a_run_held_in_this_server(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    service = _service(cast(Any, client.application))
    run_state = _run(tmp_path, "claimed-run")

    with service.claimed("claimed-run"):
        response = client.delete("/api/runs/claimed-run")

    assert (response.status_code, _error_of(response)) == (
        409,
        "Run 'claimed-run' is already being built or deleted; try again later.",
    )
    assert run_state.directory.manifest_path.is_file()


def test_delete_does_not_find_an_unknown_run(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.delete("/api/runs/absent")

    assert (response.status_code, _error_of(response)) == (404, "Run not found.")


def test_delete_forgets_the_run_s_ended_job(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    service = _service(cast(Any, client.application))

    def failed_build(**_kwargs: object) -> DatasetBuildResult:
        raise RuntimeError("stopped")

    monkeypatch.setattr(jobs, "build_atom_site_dataset", failed_build)
    _run(tmp_path, "ended-run")
    _spawn(service, "ended-run")
    _wait_for_job(service, "ended-run")
    # The build never held the run, so its failure is kept only in memory.
    assert service.in_memory_status("ended-run") is not None

    assert client.delete("/api/runs/ended-run").status_code == 200
    assert service.in_memory_status("ended-run") is None


def test_a_run_that_cannot_be_removed_is_a_server_error_and_is_not_listed(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    client = _client(tmp_path, monkeypatch)
    _run(tmp_path, "stuck-run")

    def refuse(path: object) -> None:
        raise OSError(f"Permission denied: {path}")

    monkeypatch.setattr(shutil, "rmtree", refuse)
    with caplog.at_level(logging.ERROR):
        response = client.delete("/api/runs/stuck-run")

    assert response.status_code == 500
    assert _error_of(response) == "The run could not be deleted."
    assert "Run 'stuck-run' could not be deleted." in caplog.text
    # The directory was renamed away from its run ID before the removal
    # failed. So no route finds the run, though its manifest still names it.
    (tombstone,) = (tmp_path / "workspace" / "runs").iterdir()
    assert tombstone.name.startswith(".deleting-stuck-run-")
    assert _listed(client) == {}
    assert client.get("/api/runs/stuck-run/status").status_code == 404
