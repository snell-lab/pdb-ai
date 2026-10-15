from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import requests
from fake_http import FakeResponse, FakeSession
from flask.testing import FlaskClient
from pytest import LogCaptureFixture, MonkeyPatch

from pdbai.selection import (
    Selection,
    SelectionResult,
    build_search_request,
    search_archive,
)
from pdbai.webapp import create_app, jobs, select_routes
from pdbai.webapp.factory import job_service
from webapp.conftest import _client, _wait_for_job


def _selection_result(
    *pdb_ids: str, total: int | None = None, limit: int | None = None
) -> SelectionResult:
    return SelectionResult(
        filters=Selection("x_ray_diffraction", "free_electron_laser", limit=limit),
        pdb_ids=tuple(pdb_ids),
        total_count=total if total is not None else len(pdb_ids),
        service="rcsb_search",
        service_url="https://search.example/query",
        query={"query": {}},
        retrieved_at=1_700_000_000.0,
        # In these fixtures, a larger total stands for matches the limit
        # dropped.
        truncated=total is not None and total > len(pdb_ids),
    )


def _hits(*identifiers: str) -> dict[str, object]:
    return {
        "total_count": len(identifiers),
        "result_set": [{"identifier": item, "score": 1.0} for item in identifiers],
    }


def _search_through(monkeypatch: MonkeyPatch, session: FakeSession) -> None:
    """Run the route's real search_archive against the scripted session."""

    def search(criteria: Selection, *, timeout: float) -> SelectionResult:
        return search_archive(criteria, session=session.as_session(), timeout=timeout)

    monkeypatch.setattr(select_routes, "search_archive", search)


def _recording_search(monkeypatch: MonkeyPatch) -> list[Selection]:
    asked: list[Selection] = []

    def search(criteria: Selection, *, timeout: float) -> SelectionResult:
        asked.append(criteria)
        return _selection_result("11YO", limit=criteria.limit)

    monkeypatch.setattr(select_routes, "search_archive", search)
    return asked


def _capped_client(tmp_path: Path, monkeypatch: MonkeyPatch) -> FlaskClient:
    monkeypatch.chdir(tmp_path)
    app = create_app({"MAX_PDB_IDS": 2})
    app.testing = True
    return app.test_client()


def _json(response: Any) -> Mapping[str, Any]:
    return cast(Mapping[str, Any], response.get_json())


def test_select_route_answers_with_the_record_and_its_summary(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    asked: list[Selection] = []

    def search(criteria: Selection, *, timeout: float) -> SelectionResult:
        asked.append(criteria)
        return _selection_result("11YO", "13CH", total=1041, limit=criteria.limit)

    monkeypatch.setattr(select_routes, "search_archive", search)
    client = _capped_client(tmp_path, monkeypatch)

    response = client.post(
        "/api/select",
        json={"technique": "x_ray_diffraction", "xraySource": "free_electron_laser"},
    )

    assert response.status_code == 200
    payload = _json(response)
    # MAX_PDB_IDS caps a selection as it caps a build request.
    assert asked == [Selection("x_ray_diffraction", "free_electron_laser", limit=2)]
    record = cast(Mapping[str, Any], payload["selection"])
    assert record == _selection_result("11YO", "13CH", total=1041, limit=2).to_dict()
    # The identifiers are sent once, inside the record.
    assert "pdbIds" not in payload
    assert {key: value for key, value in payload.items() if key != "selection"} == {
        "description": "X-ray diffraction (XRD), X-ray free-electron laser (XFEL)",
        "totalCount": 1041,
        "returnedCount": 2,
        "truncated": True,
        "retrievedAtIso": "2023-11-14T22:13:20Z",
        "warnings": [],
        "maxPdbIds": 2,
    }


def test_select_route_takes_the_smaller_of_the_limit_and_the_cap(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    asked = _recording_search(monkeypatch)
    client = _capped_client(tmp_path, monkeypatch)

    for limit in (5, 1, "", "1"):
        assert client.post("/api/select", json={"limit": limit}).status_code == 200

    assert [criteria.limit for criteria in asked] == [2, 1, 2, 1]


def test_select_route_reads_the_limit_strictly(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    asked = _recording_search(monkeypatch)
    client = _client(tmp_path, monkeypatch)

    for limit in (None, "", "5", 5.0):
        assert client.post("/api/select", json={"limit": limit}).status_code == 200
    refused = [
        client.post("/api/select", json={"limit": limit})
        for limit in (True, 0, -1, 5.5, "five")
    ]

    # With no deployment cap, no limit means every match.
    assert [criteria.limit for criteria in asked] == [None, None, 5, 5]
    assert [response.status_code for response in refused] == [400] * 5
    assert all("limit" in str(_json(response)["error"]) for response in refused)


def test_select_route_searches_with_the_range_filters_it_was_sent(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    session = FakeSession(FakeResponse(_hits("13CH", "11YO")))
    _search_through(monkeypatch, session)
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/select",
        json={
            "technique": "x_ray_diffraction",
            "xraySource": "synchrotron",
            "minTemperatureK": "273",
            "maxTemperatureK": "",
            "maxResolutionA": 2.5,
        },
    )

    assert response.status_code == 200
    payload = _json(response)
    criteria = Selection(
        "x_ray_diffraction",
        "synchrotron",
        bounds={"min_temperature_K": 273, "max_resolution_A": 2.5},
    )
    assert session.json_bodies == [build_search_request(criteria)]
    assert payload["description"] == (
        "X-ray diffraction (XRD), Synchrotron, temperature at least 273 K, "
        "resolution at most 2.5 Å"
    )
    record = cast(Mapping[str, Any], payload["selection"])
    assert record["filters"] == criteria.to_dict()
    assert record["query"] == build_search_request(criteria)
    # The record the page sends back with a build still passes as provenance.
    assert SelectionResult.from_dict(json.loads(json.dumps(record))).filters == criteria


def test_select_route_searches_with_the_accessions_it_was_sent(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    asked = _recording_search(monkeypatch)
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/select",
        json={
            "technique": "neutron_diffraction",
            "uniprotAccessions": "P0DTD1, p00698",
        },
    )
    refused = client.post("/api/select", json={"uniprotAccessions": "P00698 lysozyme"})

    assert response.status_code == 200
    assert asked == [
        Selection("neutron_diffraction", uniprot_accessions=("P00698", "P0DTD1"))
    ]
    assert asked[0].describe() == "Neutron diffraction, one of 2 UniProt accessions"
    assert refused.status_code == 400
    assert str(_json(refused)["error"]) == (
        "uniprotAccessions must hold UniProt accessions; got 'lysozyme'."
    )


def test_select_route_refuses_a_range_filter_it_cannot_search_with(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    asked = _recording_search(monkeypatch)
    client = _client(tmp_path, monkeypatch)

    refused = [
        client.post("/api/select", json=body)
        for body in (
            {"minTemperatureK": "1e2"},
            {"maxResolutionA": True},
            {"maxTemperatureK": 0},
            {"minTemperatureK": 300, "maxTemperatureK": 100},
        )
    ]

    assert [response.status_code for response in refused] == [400] * 4
    assert [str(_json(response)["error"]) for response in refused] == [
        "minTemperatureK must be a positive number; got '1e2'.",
        "maxResolutionA must be a positive number; got True.",
        "maxTemperatureK must be a positive number; got 0.",
        "The minimum temperature (300 K) is above the maximum (100 K), so no "
        "entry can match.",
    ]
    assert not asked


def test_select_route_rejects_bad_input(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    asked = _recording_search(monkeypatch)
    client = _client(tmp_path, monkeypatch)

    not_json = client.post("/api/select", data="x", content_type="text/plain")
    not_object = client.post("/api/select", json=["x"])
    wrong_combination = client.post(
        "/api/select", json={"technique": "solution_nmr", "xraySource": "synchrotron"}
    )
    unknown = client.post("/api/select", json={"technique": "cryo"})
    not_text = client.post("/api/select", json={"technique": 3})

    assert not_json.status_code == 415
    assert not_object.status_code == 400
    assert wrong_combination.status_code == 400
    assert "applies only to technique" in str(_json(wrong_combination)["error"])
    assert unknown.status_code == 400
    assert not_text.status_code == 400
    assert str(_json(not_text)["error"]).startswith("technique must be text")
    assert not asked


def test_select_route_refuses_a_cross_origin_request(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    asked = _recording_search(monkeypatch)
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/select", json={}, headers={"Origin": "https://attacker.example"}
    )

    assert response.status_code == 403
    assert not asked


def test_select_route_reports_an_unreachable_archive_as_502(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    session = FakeSession(requests.ConnectionError("connection refused"))
    _search_through(monkeypatch, session)
    client = _client(tmp_path, monkeypatch)

    with caplog.at_level(logging.WARNING):
        response = client.post("/api/select", json={})

    assert response.status_code == 502
    assert _json(response)["error"] == "PDB search failed: connection refused"
    # The route passes its own 20-second timeout to the search.
    assert [timeout for _url, timeout in session.calls] == [20.0]
    assert any(
        record.levelno == logging.WARNING and "connection refused" in record.message
        for record in caplog.records
    )


def test_select_route_answers_a_search_with_no_matches(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _search_through(monkeypatch, FakeSession(FakeResponse(status_code=204)))
    client = _client(tmp_path, monkeypatch)

    payload = _json(client.post("/api/select", json={"technique": "solution_nmr"}))

    assert payload["totalCount"] == 0
    assert payload["returnedCount"] == 0
    assert payload["truncated"] is False
    assert cast(Mapping[str, Any], payload["selection"])["pdb_ids"] == []


def test_select_route_passes_the_search_warnings_on(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    session = FakeSession(FakeResponse(_hits("11YO", "not an id")))
    _search_through(monkeypatch, session)
    client = _client(tmp_path, monkeypatch)

    payload = _json(client.post("/api/select", json={}))

    assert payload["warnings"] == [
        "1 search hit(s) were not valid PDB identifiers and were ignored."
    ]
    record = cast(Mapping[str, Any], payload["selection"])
    assert record["warnings"] == payload["warnings"]


def test_a_selection_from_the_select_route_is_accepted_by_the_build_route(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    received: dict[str, Any] = {}
    csv_path = tmp_path / "dataset.csv"
    csv_path.write_text("pdb_id\n11YO\n", encoding="utf-8")
    report_path = tmp_path / "run_report.csv"
    report_path.write_text("pdb_id\n11YO\n", encoding="utf-8")

    def fake_build(*_args: object, **kwargs: object) -> SimpleNamespace:
        received.update(kwargs)
        return SimpleNamespace(
            csv_path=csv_path,
            run_report_path=report_path,
            run_summary_path=report_path,
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
    _search_through(monkeypatch, FakeSession(FakeResponse(_hits("13CH", "11YO"))))
    monkeypatch.chdir(tmp_path)
    app = create_app()
    app.testing = True
    client = app.test_client()
    settings = {"technique": "x_ray_diffraction", "xraySource": "free_electron_laser"}

    selected = _json(client.post("/api/select", json=settings))
    # The browser keeps the record as parsed JSON and posts it back unchanged.
    record = json.loads(json.dumps(selected["selection"]))
    malformed = client.post(
        "/api/build", json={"pdbIds": ["11YO"], "selection": {"pdb_ids": "11YO"}}
    )
    response = client.post(
        "/api/build",
        json={
            **settings,
            "pdbIds": record["pdb_ids"],
            "includeRabdam": False,
            "selection": record,
        },
    )

    assert malformed.status_code == 400
    assert "Invalid selection record" in str(_json(malformed)["error"])
    assert response.status_code == 200
    _wait_for_job(job_service(app), cast(str, _json(response)["runId"]))
    assert received["pdb_ids"] == ["11YO", "13CH"]
    search_result = received["search_result"]
    assert isinstance(search_result, SelectionResult)
    assert search_result.to_dict() == record
