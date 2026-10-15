from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from pytest import MonkeyPatch, mark
from werkzeug.test import TestResponse

from pdbai.engines.core import EngineOptions
from pdbai.runs import open_run
from pdbai.selection import Selection, SelectionResult
from pdbai.webapp import create_app, jobs
from pdbai.webapp.factory import job_service as _service
from pdbai.webapp.options import parse_pdb_ids
from webapp.conftest import _client, _options, _wait_for_job


def _error_of(response: TestResponse) -> str:
    error = cast(dict[str, object], response.get_json())["error"]
    assert isinstance(error, str)
    return error


def _stub_build(tmp_path: Path, monkeypatch: MonkeyPatch) -> list[dict[str, object]]:
    """Replace the pipeline with a stub and return the list of its calls.

    The stub records each call's keyword arguments and returns at once, so a
    request that starts a job downloads nothing.
    """
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("pdb_id\n1ABC\n", encoding="utf-8")
    received: list[dict[str, object]] = []

    def fake_build(*_args: object, **kwargs: object) -> SimpleNamespace:
        received.append(kwargs)
        return SimpleNamespace(
            csv_path=csv_path,
            run_report_path=csv_path,
            run_summary_path=csv_path,
            additional_csv_paths={},
            pdb_ids=cast(list[str], kwargs["pdb_ids"]),
            atom_count=1,
            column_count=1,
            warnings=[],
            preview_rows=[],
            pdb_results=[],
            column_filter=None,
            paused=False,
        )

    monkeypatch.setattr(jobs, "build_atom_site_dataset", fake_build)
    return received


def test_build_route_validates_the_xray_source_filter(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    assert _options({"xraySource": "free_electron_laser"}).xray_source == (
        "free_electron_laser"
    )
    assert _options({"xraySource": ""}).xray_source == "all"
    assert _options({"xraySource": "ALL"}).xray_source == "all"

    unknown = client.post("/api/build", json={"pdbIds": ["1ABC"], "xraySource": "xfel"})
    wrong_technique = client.post(
        "/api/build",
        json={
            "pdbIds": ["1ABC"],
            "technique": "solution_nmr",
            "xraySource": "synchrotron",
        },
    )

    assert unknown.status_code == 400
    assert "xraySource must be one of" in str(
        cast(dict[str, object], unknown.get_json())["error"]
    )
    assert wrong_technique.status_code == 400
    assert "applies only to technique" in str(
        cast(dict[str, object], wrong_technique.get_json())["error"]
    )


def test_build_route_rejects_an_engine_the_technique_rules_out(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/build",
        json={
            "pdbIds": ["1ABC"],
            "technique": "electron_microscopy",
            "includeRabdam": True,
        },
    )
    unknown = client.post("/api/build", json={"pdbIds": ["1ABC"], "technique": "cryo"})

    assert response.status_code == 400
    payload = cast(dict[str, object], response.get_json())
    assert (
        "RABDAM (include_rabdam) is not available for Cryogenic electron microscopy"
        in str(payload["error"])
    )
    assert unknown.status_code == 400
    assert "technique must be one of" in str(
        cast(dict[str, object], unknown.get_json())["error"]
    )
    assert not (tmp_path / "workspace" / "runs").exists()


def test_build_route_rejects_an_unknown_column_filter_mode(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/build",
        json={"pdbIds": ["1ABC"], "columnFilterMode": "keep", "columnFilterList": "x"},
    )

    assert response.status_code == 400
    assert "columnFilterMode must be" in str(
        cast(dict[str, object], response.get_json())["error"]
    )
    assert not (tmp_path / "workspace" / "runs").exists()


@mark.parametrize(
    ("request_fields", "message"),
    [
        ({"columnFilterMode": "include"}, "must be given together"),
        ({"columnFilterList": "x"}, "must be given together"),
        ({"datasetShape": "star"}, "datasetShape must be one of"),
    ],
)
def test_build_route_rejects_an_incomplete_filter_or_unknown_shape(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    request_fields: dict[str, str],
    message: str,
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post("/api/build", json={"pdbIds": ["1ABC"], **request_fields})

    assert response.status_code == 400
    assert message in str(cast(dict[str, object], response.get_json())["error"])
    assert not (tmp_path / "workspace" / "runs").exists()


def test_build_route_rejects_an_unknown_profile(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/build", json={"pdbIds": ["1ABC"], "profileId": "ml_maximal"}
    )

    assert response.status_code == 400
    assert "profileId must be one of" in str(
        cast(dict[str, object], response.get_json())["error"]
    )
    assert not (tmp_path / "workspace" / "runs").exists()


def test_build_route_rejects_invalid_pdb_ids_before_starting_a_job(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post("/api/build", json={"pdbIds": ["../secret"]})

    assert response.status_code == 400
    payload = cast(dict[str, object], response.get_json())
    assert "digit from 1 to 9 followed by three" in str(payload["error"])
    assert not (tmp_path / "workspace" / "runs").exists()


def test_build_route_caps_the_number_of_distinct_pdb_ids(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    received = _stub_build(tmp_path, monkeypatch)
    app = create_app({"MAX_PDB_IDS": 3})
    app.testing = True
    client = app.test_client()

    # The three spellings of 1ABC normalize to one ID, so the request names
    # exactly 3 distinct IDs.
    at_cap = client.post(
        "/api/build",
        json={"pdbIds": ["1ABC", "1abc", " 1ABC", "2XYZ", "3DEF"], "runId": "cap"},
    )
    _wait_for_job(_service(app), "cap")
    over_cap = client.post(
        "/api/build", json={"pdbIds": ["1ABC", "2XYZ", "3DEF", "4HHB"]}
    )

    assert at_cap.status_code == 200
    assert len(received) == 1
    assert over_cap.status_code == 400
    assert _error_of(over_cap) == (
        "A request may name at most 3 distinct PDB IDs; got 4."
    )


def test_pdb_ids_accept_the_separators_the_browser_splits_on(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # Unstubbed, the build would download 5MCC and 4HHB from RCSB and
    # PDB-Redo in a daemon thread that outlives the test. So the pipeline is
    # stubbed, and the test waits for the job to end.
    monkeypatch.chdir(tmp_path)
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("pdb_id\n5MCC\n", encoding="utf-8")
    report_path = tmp_path / "run_report.csv"
    report_path.write_text("pdb_id,success\n5MCC,True\n", encoding="utf-8")
    requested: list[list[str]] = []

    techniques: list[str] = []

    def fake_build(*_args: object, **kwargs: object) -> SimpleNamespace:
        requested.append(cast(list[str], kwargs["pdb_ids"]))
        techniques.append(cast(EngineOptions, kwargs["engine_options"]).technique)
        return SimpleNamespace(
            csv_path=csv_path,
            run_report_path=report_path,
            run_summary_path=csv_path,
            additional_csv_paths={},
            pdb_ids=cast(list[str], kwargs["pdb_ids"]),
            atom_count=1,
            column_count=1,
            warnings=[],
            preview_rows=[],
            pdb_results=[],
            column_filter=None,
            paused=False,
        )

    monkeypatch.setattr(jobs, "build_atom_site_dataset", fake_build)
    app = create_app()
    app.testing = True
    client = app.test_client()

    response = client.post(
        "/api/build",
        json={"pdbIds": "5MCC;4HHB", "technique": "electron_microscopy"},
    )

    assert response.status_code == 200
    payload = cast(Mapping[str, Any], response.get_json())
    service = _service(app)
    _wait_for_job(service, cast(str, payload["runId"]))

    # The route split the string before handing it to the pipeline.
    assert requested == [["5MCC", "4HHB"]]
    assert techniques == ["electron_microscopy"]
    assert service.live_run_ids() == frozenset()
    assert parse_pdb_ids("5MCC;4HHB\t1ABC 2XYZ,3DEF") == [
        "5MCC",
        "4HHB",
        "1ABC",
        "2XYZ",
        "3DEF",
    ]


def test_build_route_accepts_thousands_of_ids_without_a_cap(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    received: list[list[str]] = []
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("pdb_id\n1000\n", encoding="utf-8")

    def fake_build(*_args: object, **kwargs: object) -> SimpleNamespace:
        received.append(cast(list[str], kwargs["pdb_ids"]))
        return SimpleNamespace(
            csv_path=csv_path,
            run_report_path=csv_path,
            run_summary_path=csv_path,
            additional_csv_paths={},
            pdb_ids=cast(list[str], kwargs["pdb_ids"]),
            atom_count=1,
            column_count=1,
            warnings=[],
            preview_rows=[],
            pdb_results=[],
            column_filter=None,
            paused=False,
        )

    monkeypatch.setattr(jobs, "build_atom_site_dataset", fake_build)
    app = create_app()
    app.testing = True
    client = app.test_client()
    pdb_ids = [f"{number:04d}" for number in range(1000, 3500)]

    response = client.post(
        "/api/build", json={"pdbIds": pdb_ids, "includeRabdam": False}
    )

    assert response.status_code == 200
    run_id = cast(str, cast(Mapping[str, Any], response.get_json())["runId"])
    _wait_for_job(_service(app), run_id)
    assert received == [pdb_ids]


def test_build_route_answers_with_the_run_id_and_passes_run_id_and_resume_on(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    received = _stub_build(tmp_path, monkeypatch)
    app = create_app()
    app.testing = True
    client = app.test_client()
    service = _service(app)

    started = client.post(
        "/api/build", json={"pdbIds": ["1ABC"], "runId": "fresh", "resume": "false"}
    )
    _wait_for_job(service, "fresh")
    open_run(tmp_path / "workspace", "paused").directory.ensure()
    resumed = client.post(
        "/api/build", json={"pdbIds": ["1ABC"], "runId": "paused", "resume": True}
    )
    _wait_for_job(service, "paused")

    assert started.status_code == 200
    assert started.get_json() == {"runId": "fresh"}
    assert resumed.get_json() == {"runId": "paused"}
    assert [(call["run_id"], call["resume"]) for call in received] == [
        ("fresh", False),
        ("paused", True),
    ]


@mark.parametrize("resume", [1, "yes"])
def test_build_route_refuses_a_resume_that_is_not_a_flag(
    tmp_path: Path, monkeypatch: MonkeyPatch, resume: object
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/build", json={"pdbIds": ["1ABC"], "runId": "run", "resume": resume}
    )

    assert response.status_code == 400
    assert _error_of(response) == f"resume must be true or false; got {resume!r}."
    assert not (tmp_path / "workspace" / "runs").exists()


def test_build_route_refuses_a_number_as_run_id_or_pdb_ids(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    run_id = client.post("/api/build", json={"pdbIds": ["1ABC"], "runId": 123})
    pdb_ids = client.post("/api/build", json={"pdbIds": 5})

    assert run_id.status_code == 400
    assert _error_of(run_id) == "Run ID must be a string, not 123."
    assert pdb_ids.status_code == 400
    assert _error_of(pdb_ids) == "pdbIds must be text or a list of text; got 5."
    assert not (tmp_path / "workspace" / "runs").exists()


def test_build_route_resumes_only_a_named_run_that_exists(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    unnamed = client.post("/api/build", json={"pdbIds": ["1ABC"], "resume": True})
    missing = client.post(
        "/api/build", json={"pdbIds": ["1ABC"], "runId": "gone", "resume": True}
    )

    assert unnamed.status_code == 400
    assert "resume requires a runId" in _error_of(unnamed)
    assert missing.status_code == 404
    assert _error_of(missing) == "Run not found."
    assert not (tmp_path / "workspace" / "runs").exists()


def _settled_run(tmp_path: Path, run_id: str) -> dict[str, object]:
    """Leave a run whose job state says it completed, and return that state."""
    run_state = open_run(tmp_path / "workspace", run_id)
    run_state.load_or_initialize(
        ["1ABC"], _options({}).as_settings_dict(), schema_fingerprint="test-schema"
    )
    settled: dict[str, object] = {
        "status": "completed",
        "progress": None,
        "result": {},
        "error": None,
    }
    run_state.write_job_state(settled)
    return settled


def test_build_route_refuses_an_existing_run_without_resume(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    settled = _settled_run(tmp_path, "done")

    response = client.post("/api/build", json={"pdbIds": ["1ABC"], "runId": "done"})

    assert response.status_code == 409
    assert _error_of(response) == (
        "Run 'done' already exists; resume it or choose another run ID."
    )
    assert open_run(tmp_path / "workspace", "done").read_job_state() == settled


def test_build_route_refuses_a_run_another_process_holds(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    settled = _settled_run(tmp_path, "held")
    run_state = open_run(tmp_path / "workspace", "held")

    with run_state.claimed():
        response = client.post(
            "/api/build", json={"pdbIds": ["1ABC"], "runId": "held", "resume": True}
        )

    assert response.status_code == 409
    assert "already being built" in _error_of(response)
    assert run_state.read_job_state() == settled


def _selection_record(*pdb_ids: str, technique: str) -> dict[str, object]:
    return SelectionResult(
        filters=Selection(technique),
        pdb_ids=pdb_ids,
        total_count=len(pdb_ids),
        service="rcsb_search",
        service_url="https://search.example/query",
        query={"query": {}},
        retrieved_at=1_700_000_000.0,
    ).to_dict()


def test_build_route_refuses_a_selection_describing_another_request(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    other_ids = client.post(
        "/api/build",
        json={
            "pdbIds": ["1ABC", "4HHB"],
            "selection": _selection_record("1ABC", technique="x_ray_diffraction"),
        },
    )
    other_technique = client.post(
        "/api/build",
        json={
            "pdbIds": ["1ABC"],
            "technique": "electron_microscopy",
            "selection": _selection_record("1ABC", technique="x_ray_diffraction"),
        },
    )

    assert other_ids.status_code == 400
    assert "same PDB IDs as pdbIds" in _error_of(other_ids)
    assert other_technique.status_code == 400
    assert "requested technique and X-ray source" in _error_of(other_technique)
    assert not (tmp_path / "workspace" / "runs").exists()
