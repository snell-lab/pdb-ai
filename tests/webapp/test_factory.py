from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from pytest import MonkeyPatch

from pdbai.engines.core import ALL_XRAY_SOURCES, DATASET_SHAPES, DEFAULT_DATASET_SHAPE
from pdbai.profiles import PROFILES, get_profile
from pdbai.webapp import create_app
from webapp.conftest import _client


def test_index_and_profile_api_are_available(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    index = client.get("/")
    profiles = client.get("/api/profiles")

    assert index.status_code == 200
    assert b"PDB-AI" in index.data
    assert profiles.status_code == 200
    payload: object = profiles.get_json()
    assert isinstance(payload, list)
    raw_profiles = cast(list[object], payload)
    assert all(isinstance(profile, Mapping) for profile in raw_profiles)
    profile_payloads = cast(list[Mapping[str, object]], raw_profiles)
    assert [profile["id"] for profile in profile_payloads] == [
        profile.id for profile in PROFILES
    ]
    assert (tmp_path / "workspace" / "datasets").is_dir()


def test_profile_api_reports_a_profile_under_camel_case_keys(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    profile = get_profile("ml_minimal")

    payload = cast(list[dict[str, object]], client.get("/api/profiles").get_json())

    assert next(item for item in payload if item["id"] == "ml_minimal") == {
        "id": "ml_minimal",
        "name": profile.name,
        "description": profile.description,
        "filterMode": "include",
        "columns": list(profile.columns),
        "toggles": ["include_rabdam", "include_sasa", "include_validation"],
    }


def test_index_renders_a_control_for_every_declared_engine_setting(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # The engines panel is generated from the declarations, so a new engine's
    # settings appear in the form without editing the template.
    from pdbai.engines.catalog import get_engine_registry

    html = _client(tmp_path, monkeypatch).get("/").data.decode()

    for option in get_engine_registry().option_specs():
        assert f'data-option="{option.name}"' in html
        assert f'name="{option.json_key}"' in html
        if option.refines:
            assert f'data-refines="{option.refines}"' in html


def test_index_offers_every_dataset_shape_with_the_default_selected(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    html = _client(tmp_path, monkeypatch).get("/").data.decode()

    for shape in DATASET_SHAPES.values():
        assert f'<option value="{shape.name}"' in html
    assert f'<option value="{DEFAULT_DATASET_SHAPE}" selected>' in html


def test_index_renders_the_technique_selector_and_api_lists_compatibility(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    index = client.get("/")
    techniques = client.get("/api/techniques")

    page = index.get_data(as_text=True)
    assert 'id="technique"' in page
    assert 'value="x_ray_diffraction" selected' in page
    assert "Cryogenic electron microscopy (cryo-EM)" in page
    assert "Solution nuclear magnetic resonance (solution NMR)" in page
    assert "X-ray free-electron laser (XFEL)" in page
    assert techniques.status_code == 200
    payload = cast(Mapping[str, Any], techniques.get_json())
    assert payload["default"] == "x_ray_diffraction"
    em = next(
        cast(Mapping[str, Any], item)
        for item in payload["techniques"]
        if cast(Mapping[str, Any], item)["id"] == "electron_microscopy"
    )
    assert em["acronym"] == "cryo-EM"
    assert em["displayLabel"] == "Cryogenic electron microscopy (cryo-EM)"
    ids = [cast(Mapping[str, Any], item)["id"] for item in payload["techniques"]]
    assert "solution_nmr" in ids and "integrative" in ids
    rabdam = next(
        cast(Mapping[str, Any], record)
        for record in payload["compatibility"]
        if cast(Mapping[str, Any], record)["toggle"] == "include_rabdam"
    )
    assert rabdam["techniques"] == ["x_ray_diffraction"]
    assert 'id="xraySource"' in page
    sources = cast(Mapping[str, Any], payload["xraySources"])
    assert sources["technique"] == "x_ray_diffraction"
    assert sources["all"] == ALL_XRAY_SOURCES
    assert [cast(Mapping[str, Any], s)["id"] for s in sources["sources"]] == [
        "synchrotron",
        "free_electron_laser",
        "rotating_anode",
        "sealed_tube",
        "liquid_anode",
    ]


def test_download_route_serves_only_from_dataset_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    filtered = f"result_filtered_{'a' * 32}.csv"
    dataset = tmp_path / "workspace" / "datasets" / filtered
    dataset.write_bytes(b"pdb_id\n1ABC\n")
    outside = tmp_path / "secret.txt"
    outside.write_text("secret", encoding="utf-8")

    download = client.get(f"/downloads/{filtered}")
    traversal = client.get("/downloads/../secret.txt")

    assert download.status_code == 200
    assert download.data == b"pdb_id\n1ABC\n"
    assert "attachment" in download.headers["Content-Disposition"]
    assert traversal.status_code == 404


def test_state_changing_requests_from_a_foreign_origin_are_rejected(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    foreign = client.post(
        "/api/build",
        json={"pdbIds": ["1ABC"], "runId": "evil"},
        headers={"Origin": "https://attacker.example"},
    )
    same_origin = client.post(
        "/api/build",
        json={"pdbIds": ["../bad"]},
        headers={"Origin": "http://localhost"},
    )

    assert foreign.status_code == 403
    assert not (tmp_path / "workspace" / "runs" / "evil").exists()

    assert same_origin.status_code == 400


def test_reads_are_not_blocked_by_the_origin_guard(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    response = client.get("/api/profiles", headers={"Origin": "https://elsewhere.test"})

    assert response.status_code == 200


def test_oversized_requests_are_refused_with_the_standard_error_shape(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert create_app().config["MAX_CONTENT_LENGTH"] == 512 * 1024 * 1024
    client = create_app({"MAX_CONTENT_LENGTH": 64}).test_client()

    response = client.post(
        "/api/build-upload",
        data={"mmcifFiles": (io.BytesIO(b"x" * 4096), "big.cif")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 413
    payload = cast(dict[str, object], response.get_json())
    assert "error" in payload


def test_unhandled_errors_return_json_without_leaking_detail(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app()

    @app.route("/api/boom")
    def _boom() -> str:
        raise RuntimeError("secret internal path /etc/passwd")

    _ = _boom
    client = app.test_client()
    missing = client.get("/api/does-not-exist")
    broken = client.get("/api/boom")

    assert missing.status_code == 404
    assert cast(dict[str, object], missing.get_json())["error"]
    assert broken.status_code == 500
    payload = cast(dict[str, object], broken.get_json())
    assert payload["error"] == "Internal server error."
    assert "passwd" not in broken.get_data(as_text=True)


def test_requests_for_a_host_other_than_loopback_are_refused(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # A DNS-rebinding page reaches the server under its own hostname.
    client = _client(tmp_path, monkeypatch)

    assert client.get("/", headers={"Host": "attacker.example:8001"}).status_code == 400
    assert client.get("/", headers={"Host": "127.0.0.1:8001"}).status_code == 200
    assert client.get("/", headers={"Host": "localhost:8001"}).status_code == 200
    untrusted = client.get("/", headers={"Host": "attacker.example"})
    assert cast(dict[str, object], untrusted.get_json())["error"]


def test_a_host_named_in_trusted_hosts_is_answered(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    client = create_app({"TRUSTED_HOSTS": ["lab.example"]}).test_client()

    assert client.get("/", headers={"Host": "lab.example:8001"}).status_code == 200
    assert client.get("/", headers={"Host": "localhost:8001"}).status_code == 400


def test_index_renders_the_archive_selection_control(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    page = client.get("/").get_data(as_text=True)

    assert 'id="selectFromPdbButton"' in page
    assert '"select": "/api/select"' in page


def test_a_method_not_allowed_response_keeps_its_allow_header(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    response = _client(tmp_path, monkeypatch).post("/api/profiles")

    assert response.status_code == 405
    assert "GET" in response.headers["Allow"]
    assert response.content_type == "application/json"
    assert cast(dict[str, object], response.get_json())["error"]


def test_allowed_origins_extend_the_apps_own_origin(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    app = create_app({"ALLOWED_ORIGINS": ["https://ui.example/"]})
    client = app.test_client()

    def post_from(origin: str) -> int:
        return client.post(
            "/api/build", json={"pdbIds": ["../bad"]}, headers={"Origin": origin}
        ).status_code

    # A 400 means the origin passed and the bad ID was refused after it.
    assert post_from("https://ui.example") == 400
    assert post_from("http://localhost") == 400
    assert post_from("https://attacker.example") == 403
    assert post_from("null") == 403


def test_allowed_origins_given_as_one_string_are_refused_at_startup(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="ALLOWED_ORIGINS must be a list"):
        create_app({"ALLOWED_ORIGINS": "https://ui.example"})
    with pytest.raises(ValueError, match="ALLOWED_ORIGINS must hold only text"):
        create_app({"ALLOWED_ORIGINS": ["https://ui.example", 8001]})


def test_downloads_serve_only_filtered_tables(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    datasets = tmp_path / "workspace" / "datasets"
    filtered = f"upload_filtered_{'a' * 32}.csv"
    folder = f"sub_filtered_{'b' * 32}.csv"
    link = f"link_filtered_{'c' * 32}.csv"
    (datasets / filtered).write_bytes(b"pdb_id\n1ABC\n")
    (datasets / f".{filtered}.abc.tmp").write_text("partial", encoding="utf-8")
    (datasets / "result.csv").write_text("pdb_id\n", encoding="utf-8")
    (datasets / "notes.txt").write_text("notes", encoding="utf-8")
    (datasets / folder).mkdir()
    (datasets / folder / "x.csv").write_text("nested", encoding="utf-8")
    secret = tmp_path / "secret.csv"
    secret.write_text("secret", encoding="utf-8")
    (datasets / link).symlink_to(secret)

    assert client.get(f"/downloads/{filtered}").data == b"pdb_id\n1ABC\n"
    refused = (
        f".{filtered}.abc.tmp",
        "result.csv",
        "notes.txt",
        folder,
        f"{folder}/x.csv",
        link,
    )
    for name in refused:
        assert client.get(f"/downloads/{name}").status_code == 404, name


def test_a_custom_working_directory_holds_the_dataset_directory(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    create_app({"WORKING_DIRECTORY": tmp_path / "elsewhere"})

    assert (tmp_path / "elsewhere" / "datasets").is_dir()
    assert not (tmp_path / "workspace").exists()


def test_the_favicon_path_redirects_to_the_static_icon(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    response = _client(tmp_path, monkeypatch).get("/favicon.ico")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/static/favicon.svg")
