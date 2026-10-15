from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import cast

import pytest
from pytest import MonkeyPatch
from werkzeug.exceptions import BadRequest, NotFound
from werkzeug.test import TestResponse

from pdbai.runs import validate_run_id
from pdbai.webapp.route_helpers import (
    existing_run,
    remove_stale_files,
    run_id_to_build,
)
from webapp.conftest import _client


def test_build_route_rejects_an_unsafe_run_id(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/build",
        json={"pdbIds": ["1ABC"], "runId": ".."},
    )

    assert response.status_code == 400
    payload = cast(dict[str, object], response.get_json())
    assert "Run ID must be a single name" in str(payload["error"])
    assert not (tmp_path / "workspace" / "runs").exists()


def test_delete_route_cannot_escape_the_runs_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    sentinel = tmp_path / "workspace" / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    response = client.delete("/api/runs/%2E%2E")

    assert response.status_code == 400
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_build_route_requires_a_json_content_type(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/build",
        data=json.dumps({"pdbIds": ["1ABC"], "runId": "csrf"}),
        content_type="text/plain;charset=UTF-8",
    )

    assert response.status_code == 415
    payload = cast(dict[str, object], response.get_json())
    assert "application/json" in str(payload["error"])
    assert not (tmp_path / "workspace" / "runs").exists()


def _error_of(response: TestResponse) -> str:
    payload = cast(dict[str, object], response.get_json())
    error = payload["error"]
    assert isinstance(error, str)
    return error


@pytest.mark.parametrize("run_id", [0, False])
def test_build_route_refuses_a_falsy_run_id_rather_than_starting_a_new_run(
    tmp_path: Path, monkeypatch: MonkeyPatch, run_id: object
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post("/api/build", json={"pdbIds": ["1ABC"], "runId": run_id})

    assert response.status_code == 400
    assert _error_of(response).startswith("Run ID must be a string")
    assert not (tmp_path / "workspace" / "runs").exists()


@pytest.mark.parametrize("value", [None, ""])
def test_a_missing_or_empty_run_id_names_a_new_run(
    tmp_path: Path, monkeypatch: MonkeyPatch, value: object
) -> None:
    client = _client(tmp_path, monkeypatch)

    def build(raw_run_id: object) -> str:
        return run_id_to_build(
            tmp_path / "workspace", raw_run_id, resume=False, is_held=lambda _: False
        )

    with client.application.test_request_context():
        first = build(value)
        second = build(value)

    assert first != second
    assert validate_run_id(first) == first


def test_existing_run_does_not_find_a_file_at_the_run_path(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    working_directory = tmp_path / "workspace"
    (working_directory / "runs").mkdir()
    (working_directory / "runs" / "afile").write_text("not a run", encoding="utf-8")
    (working_directory / "runs" / "arun").mkdir()

    with client.application.test_request_context():
        with pytest.raises(NotFound, match="Run not found."):
            existing_run(working_directory, "afile")
        with pytest.raises(NotFound):
            existing_run(working_directory, "missing")
        with pytest.raises(BadRequest):
            existing_run(working_directory, "..")
        run_state = existing_run(working_directory, "arun")

    assert run_state.directory.root == (working_directory / "runs" / "arun").resolve()


def test_status_route_does_not_find_a_run_under_a_symlinked_runs_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    elsewhere = tmp_path / "elsewhere"
    (elsewhere / "run-a").mkdir(parents=True)
    (tmp_path / "workspace" / "runs").symlink_to(elsewhere, target_is_directory=True)

    response = client.get("/api/runs/run-a/status")

    assert response.status_code == 404
    assert str(tmp_path) not in _error_of(response)


def test_build_route_refuses_malformed_json(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/api/build", data='{"pdbIds": [', content_type="application/json"
    )

    assert response.status_code == 400
    assert _error_of(response) == "Request body must be valid JSON."
    assert not (tmp_path / "workspace" / "runs").exists()


def test_build_route_refuses_a_body_that_is_not_an_object(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.post("/api/build", json=["1ABC"])

    assert response.status_code == 400
    assert _error_of(response) == "Request body must be a JSON object."
    assert not (tmp_path / "workspace" / "runs").exists()


_AN_HOUR_AND_A_BIT_AGO = time.time() - 60 * 60 - 60


def _age(path: Path) -> None:
    os.utime(
        path,
        (_AN_HOUR_AND_A_BIT_AGO, _AN_HOUR_AND_A_BIT_AGO),
        follow_symlinks=False,
    )


def test_remove_stale_files_leaves_directories_and_removes_dangling_links(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "old_filtered_dir.csv"
    directory.mkdir()
    (directory / "inside.csv").write_text("keep", encoding="utf-8")
    _age(directory)
    dangling = tmp_path / "gone_filtered_link.csv"
    dangling.symlink_to(tmp_path / "missing.csv")
    _age(dangling)
    stale = tmp_path / "old_filtered_file.csv"
    stale.write_text("old", encoding="utf-8")
    _age(stale)

    remove_stale_files(tmp_path, "*_filtered_*.csv")

    assert (directory / "inside.csv").read_text(encoding="utf-8") == "keep"
    assert not dangling.is_symlink()
    assert not stale.exists()


def test_remove_stale_files_removes_an_old_link_but_not_its_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("keep", encoding="utf-8")
    _age(target)
    old_link = tmp_path / "old_filtered_link.csv"
    old_link.symlink_to(target)
    _age(old_link)
    young_link = tmp_path / "young_filtered_link.csv"
    young_link.symlink_to(target)

    remove_stale_files(tmp_path, "*_filtered_*.csv")

    assert not old_link.is_symlink()
    assert young_link.is_symlink()
    assert target.read_text(encoding="utf-8") == "keep"


def test_remove_stale_files_leaves_young_files(tmp_path: Path) -> None:
    young = tmp_path / "young_filtered_file.csv"
    young.write_text("new", encoding="utf-8")
    unmatched = tmp_path / "old.csv"
    unmatched.write_text("old", encoding="utf-8")
    _age(unmatched)

    remove_stale_files(tmp_path, "*_filtered_*.csv")

    assert young.read_text(encoding="utf-8") == "new"
    assert unmatched.exists()
