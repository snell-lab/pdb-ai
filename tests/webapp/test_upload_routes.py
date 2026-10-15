from __future__ import annotations

import gzip
import io
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from flask import Flask
from flask.testing import FlaskClient
from pytest import MonkeyPatch

from pdbai.orchestration.upload import identify_uploads
from pdbai.runs import open_run
from pdbai.webapp import create_app, jobs, upload_routes
from pdbai.webapp.factory import job_service as _service
from webapp.conftest import _client, _options, _wait_for_job


def _uploads(tmp_path: Path) -> list[Path]:
    root = tmp_path / "workspace" / "uploads"
    return list(root.iterdir()) if root.exists() else []


def _datasets(tmp_path: Path) -> list[str]:
    return sorted(path.name for path in (tmp_path / "workspace" / "datasets").iterdir())


def _post_upload(
    client: FlaskClient, files: object, **fields: str
) -> tuple[int, dict[str, Any]]:
    response = client.post(
        "/api/build-upload",
        data={"mmcifFiles": files, **fields},
        content_type="multipart/form-data",
    )
    return response.status_code, cast(dict[str, Any], response.get_json())


def _post_filter(
    client: FlaskClient, content: bytes, filename: str = "d.csv", **fields: str
) -> tuple[int, dict[str, Any]]:
    response = client.post(
        "/api/filter-upload",
        data={"file": (io.BytesIO(content), filename), **fields},
        content_type="multipart/form-data",
    )
    return response.status_code, cast(dict[str, Any], response.get_json())


def _recording_build(monkeypatch: MonkeyPatch) -> list[dict[str, object]]:
    """Replace the build with a stub and return the list of its calls.

    The stub records the files it was given and then raises, which ends the
    job before any processing.
    """
    calls: list[dict[str, object]] = []

    def fake_build(*_args: object, **kwargs: object) -> object:
        paths = cast(list[Path], kwargs["mmcif_files"])
        calls.append(
            {
                "paths": paths,
                "contents": [path.read_bytes() for path in paths],
                "uploads": identify_uploads(paths),
                "resume": kwargs["resume"],
            }
        )
        raise RuntimeError("stop after inspecting the inputs")

    monkeypatch.setattr(jobs, "build_atom_site_dataset_from_mmcif_files", fake_build)
    return calls


def _app(tmp_path: Path, monkeypatch: MonkeyPatch, **config: object) -> Flask:
    monkeypatch.chdir(tmp_path)
    app = create_app(config)
    app.testing = True
    return app


def _finish(client: FlaskClient, payload: Mapping[str, Any]) -> None:
    _wait_for_job(_service(client.application), payload["runId"])


def test_upload_route_rejects_an_engine_the_technique_rules_out_before_saving(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(
        client,
        (io.BytesIO(b"data_1ABC\n"), "structure.cif"),
        technique="solution_nmr",
        includePdbRedo="true",
    )

    assert status == 400
    assert "PDB-Redo comparison (include_pdb_redo) is not available" in str(
        payload["error"]
    )
    assert _uploads(tmp_path) == []


def test_upload_route_caps_the_number_of_files(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _app(tmp_path, monkeypatch, MAX_UPLOAD_FILES=2).test_client()

    status, payload = _post_upload(
        client,
        [
            (io.BytesIO(b"data_A\n"), "a.cif"),
            (io.BytesIO(b"data_B\n"), "b.cif"),
            (io.BytesIO(b"data_C\n"), "c.cif"),
        ],
    )

    assert status == 400
    assert "At most 2 files" in str(payload["error"])
    assert _uploads(tmp_path) == []


def test_the_file_cap_ignores_an_empty_file_field(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    calls = _recording_build(monkeypatch)
    client = _app(tmp_path, monkeypatch, MAX_UPLOAD_FILES=1).test_client()

    status, payload = _post_upload(
        client,
        [(io.BytesIO(b"data_A\n"), "a.cif"), (io.BytesIO(b""), "")],
    )

    assert status == 200
    _finish(client, payload)
    uploads = cast(list[Any], calls[0]["uploads"])
    assert [upload.pdb_id for upload in uploads] == ["A"]


def test_upload_route_rejects_a_live_run_and_removes_its_uploads(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    app = _app(tmp_path, monkeypatch)
    client = app.test_client()
    with _service(app).claimed("busy-upload"):
        status, payload = _post_upload(
            client,
            (io.BytesIO(b"data_1ABC\n"), "structure.cif"),
            runId="busy-upload",
        )

    assert status == 409
    assert payload["error"] == (
        "Run 'busy-upload' is already being built or deleted; try again later."
    )
    assert _uploads(tmp_path) == []


def _settled_run(tmp_path: Path, run_id: str) -> dict[str, object]:
    """Leave a run whose job state says it completed, and return that state."""
    run_state = open_run(tmp_path / "workspace", run_id)
    run_state.load_or_initialize(
        ["A"], _options({}).as_settings_dict(), schema_fingerprint="test-schema"
    )
    settled: dict[str, object] = {
        "status": "completed",
        "progress": None,
        "result": {},
        "error": None,
    }
    run_state.write_job_state(settled)
    return settled


def test_upload_route_resumes_only_a_run_that_exists(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(
        client, (io.BytesIO(b"data_A\n"), "a.cif"), runId="gone", resume="true"
    )

    assert status == 404
    assert payload["error"] == "Run not found."
    assert _uploads(tmp_path) == []
    assert not (tmp_path / "workspace" / "runs" / "gone").exists()


def test_upload_route_refuses_an_existing_run_without_resume(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    settled = _settled_run(tmp_path, "done")

    status, payload = _post_upload(
        client, (io.BytesIO(b"data_A\n"), "a.cif"), runId="done"
    )

    assert status == 409
    assert payload["error"] == (
        "Run 'done' already exists; resume it or choose another run ID."
    )
    assert _uploads(tmp_path) == []
    assert open_run(tmp_path / "workspace", "done").read_job_state() == settled


def test_upload_route_refuses_a_run_another_process_holds(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    settled = _settled_run(tmp_path, "held")
    run_state = open_run(tmp_path / "workspace", "held")

    with run_state.claimed():
        status, payload = _post_upload(
            client, (io.BytesIO(b"data_A\n"), "a.cif"), runId="held", resume="true"
        )

    assert status == 409
    assert payload["error"] == (
        "Run 'held' is already being built or deleted; try again later."
    )
    assert _uploads(tmp_path) == []
    assert run_state.read_job_state() == settled


def test_an_invalid_run_id_is_refused_before_saving(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(
        client, (io.BytesIO(b"data_A\n"), "a.cif"), runId="../escape"
    )

    assert status == 400
    assert "Run ID" in payload["error"]
    assert _uploads(tmp_path) == []


def test_a_misspelled_resume_flag_is_a_bad_request(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(
        client, (io.BytesIO(b"data_A\n"), "a.cif"), runId="r1", resume="flase"
    )

    assert status == 400
    assert payload["error"] == "resume must be true or false; got 'flase'."
    assert _uploads(tmp_path) == []


def test_resume_without_a_run_id_is_a_bad_request(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(
        client, (io.BytesIO(b"data_A\n"), "a.cif"), resume="true"
    )

    assert status == 400
    assert "resume requires a runId" in payload["error"]


def test_resume_reaches_the_build(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    calls = _recording_build(monkeypatch)
    client = _client(tmp_path, monkeypatch)
    _settled_run(tmp_path, "r1")

    status, payload = _post_upload(
        client, (io.BytesIO(b"data_A\n"), "a.cif"), runId="r1", resume="true"
    )

    assert status == 200
    assert payload == {"runId": "r1"}
    _finish(client, payload)
    assert calls[0]["resume"] is True


def test_compressed_uploads_are_expanded_and_removed_once_the_build_ends(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    calls = _recording_build(monkeypatch)
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(
        client,
        (io.BytesIO(gzip.compress(b"data_1ABC\n")), "1abc.cif.gz"),
        technique="electron_microscopy",
    )

    assert status == 200
    _finish(client, payload)
    assert calls[0]["contents"] == [b"data_1ABC\n"]
    assert [path.name for path in cast(list[Path], calls[0]["paths"])] == ["1abc.cif"]
    assert _uploads(tmp_path) == []


def test_files_of_one_name_are_numbered_by_the_build(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    calls = _recording_build(monkeypatch)
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(
        client,
        [
            (io.BytesIO(b"data_A\n"), "x.cif"),
            (io.BytesIO(b"data_B\n"), "x.cif"),
            (io.BytesIO(b"data_C\n"), "x_1.cif"),
        ],
    )

    assert status == 200
    _finish(client, payload)
    uploads = cast(list[Any], calls[0]["uploads"])
    assert [upload.pdb_id for upload in uploads] == ["X", "X_2", "X_1"]
    assert [upload.to_settings()["source_filename"] for upload in uploads] == [
        "x.cif",
        "x.cif",
        "x_1.cif",
    ]
    assert calls[0]["contents"] == [b"data_A\n", b"data_B\n", b"data_C\n"]


def test_a_path_or_device_name_is_saved_under_a_portable_base_name(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    calls = _recording_build(monkeypatch)
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(
        client,
        [
            (io.BytesIO(b"data_A\n"), "../../escape.cif"),
            (io.BytesIO(b"data_B\n"), "/home/me/my file.mmCIF"),
            (io.BytesIO(b"data_C\n"), "NUL.cif"),
        ],
    )

    assert status == 200
    _finish(client, payload)
    paths = cast(list[Path], calls[0]["paths"])
    assert [path.name for path in paths] == ["escape.cif", "my_file.mmCIF", "NUL_.cif"]
    uploads_root = (tmp_path / "workspace" / "uploads").resolve()
    assert all(uploads_root in path.resolve().parents for path in paths)


def test_a_name_with_no_ascii_letter_or_digit_is_refused(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(
        client,
        [(io.BytesIO(b"data_A\n"), "a.cif"), (io.BytesIO(b"data_B\n"), "中文.cif")],
    )

    assert status == 400
    assert "'中文.cif'" in payload["error"]
    assert "no ASCII letter or digit" in payload["error"]
    assert _uploads(tmp_path) == []


def test_a_name_that_is_only_a_folder_is_refused(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(client, (io.BytesIO(b"data_A\n"), "folder/"))

    assert status == 400
    assert "'folder/' is not an mmCIF file" in payload["error"]


def test_an_unsupported_extension_is_refused_before_any_file_is_saved(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(
        client,
        [
            (io.BytesIO(b"data_A\n"), "a.cif"),
            (io.BytesIO(b"notes"), "notes.txt.gz"),
        ],
    )

    assert status == 400
    assert "'notes.txt.gz' is not an mmCIF file" in payload["error"]
    assert _uploads(tmp_path) == []


def test_a_corrupt_gzip_file_is_a_bad_request_and_leaves_nothing(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    truncated = gzip.compress(b"data_1ABC\n" * 100)[:-12]

    for content in (b"not gzip at all", truncated, b"\x1f\x8b\x08\x00" + b"\0" * 20):
        status, payload = _post_upload(client, (io.BytesIO(content), "x.cif.gz"))

        assert status == 400
        assert payload["error"] == "Upload 'x.cif.gz' is not a readable gzip file."
        assert _uploads(tmp_path) == []


def test_files_expanding_past_the_request_cap_are_refused_and_removed(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _app(tmp_path, monkeypatch, MAX_CONTENT_LENGTH=100_000).test_client()
    # Each file expands to 60,000 bytes, under the cap alone but over it
    # together. Compressed, each is far smaller than the cap.
    bomb = gzip.compress(b"\0" * 60_000)

    status, payload = _post_upload(
        client,
        [(io.BytesIO(bomb), "a.cif.gz"), (io.BytesIO(bomb), "b.cif.gz")],
    )

    assert status == 413
    assert "expand to more than the 100000 bytes" in payload["error"]
    assert _uploads(tmp_path) == []


def test_a_failed_write_is_a_server_error_that_leaves_nothing(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    def full_disk(*_args: object) -> None:
        raise OSError(28, "No space left on device", "/private/path")

    monkeypatch.setattr(upload_routes, "_copy_within_budget", full_disk)
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_upload(client, (io.BytesIO(b"data_A\n"), "a.cif"))

    assert status == 500
    assert payload == {"error": "Internal server error."}
    assert _uploads(tmp_path) == []


def test_upload_folders_a_stopped_server_left_are_removed_at_startup(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    leftover = tmp_path / "workspace" / "uploads" / "0123" / "1" / "a.cif"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("data_A\n", encoding="utf-8")

    _client(tmp_path, monkeypatch)

    assert _uploads(tmp_path) == []


def test_filter_upload_streams_the_selected_columns(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    rows = "".join(f"{i},{i * 2},x{i}\n" for i in range(40))

    status, payload = _post_filter(
        client,
        ("pdb_id,b_iso,note\n" + rows).encode(),
        columnFilterMode="exclude",
        columnFilterList="note",
    )

    assert status == 200
    assert payload["summary"] == {"rowCount": 40, "columnCount": 2}
    assert len(payload["preview"]) == 25
    assert payload["preview"][1] == {"pdb_id": "1", "b_iso": "2"}
    assert payload["columnFilter"] == {
        "mode": "exclude",
        "matchedColumns": ["note"],
        "missingColumns": [],
        "requestedColumns": ["note"],
        "availableColumns": ["pdb_id", "b_iso", "note"],
    }
    downloaded = client.get(payload["csvUrl"]).data.decode()
    assert downloaded.splitlines()[:2] == ["pdb_id,b_iso", "0,0"]
    # Only the result remains, and no scratch file beside it.
    leftovers = _datasets(tmp_path)
    assert len(leftovers) == 1 and "_filtered_" in leftovers[0]


def test_filter_upload_without_a_filter_copies_the_table(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_filter(client, b"pdb_id,x\n1ABC,1\n")

    assert status == 200
    assert payload["columnFilter"] is None
    assert payload["summary"] == {"rowCount": 1, "columnCount": 2}


def test_filter_upload_refuses_an_unknown_filter_mode(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_filter(
        client,
        b"pdb_id,note\n1ABC,x\n",
        columnFilterMode="sideways",
        columnFilterList="note",
    )

    assert status == 400
    assert payload["error"].startswith(
        "columnFilterMode must be one of include, exclude"
    )
    assert "'sideways'" in payload["error"]
    assert _datasets(tmp_path) == []


def test_filter_upload_refuses_a_mode_without_a_list(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_filter(
        client, b"pdb_id,note\n1ABC,x\n", columnFilterMode="include"
    )

    assert status == 400
    assert payload["error"].startswith(
        "columnFilterMode and columnFilterList must be given together"
    )
    assert _datasets(tmp_path) == []


def test_an_exclude_filter_matching_every_column_keeps_the_identifier(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_filter(
        client,
        b"pdb_id,x\n1ABC,1\n",
        columnFilterMode="exclude",
        columnFilterList="*",
    )

    assert status == 200
    assert payload["summary"] == {"rowCount": 1, "columnCount": 1}
    assert payload["preview"] == [{"pdb_id": "1ABC"}]


def test_an_include_filter_matching_nothing_leaves_the_table_with_a_warning(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_filter(
        client,
        b"pdb_id,x\n1ABC,1\n",
        columnFilterMode="include",
        columnFilterList="missing",
    )

    assert status == 200
    assert payload["summary"] == {"rowCount": 1, "columnCount": 2}
    assert payload["warnings"] == [
        "No columns matched the requested column filter; dataset left unchanged.",
        "The following columns were not found and were ignored: missing",
    ]


def test_filter_upload_refuses_a_table_it_cannot_read(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    oversized_field = b"pdb_id,x\n1ABC,1\n2ABC," + b"y" * 200_000 + b"\n"
    cases = [
        (b"", "The CSV file is empty; its first row must name the columns."),
        (b"pdb_\xffid,x\n1ABC,1\n", "The CSV file must be UTF-8 text."),
        (b"pdb_id,x\n1ABC,\xff\n", "The CSV file must be UTF-8 text."),
        (oversized_field, "The CSV file could not be parsed: field larger than"),
        (
            b"pdb_id,x,pdb_id\n1ABC,1,2\n",
            "The CSV header names these columns more than once: ['pdb_id'].",
        ),
        (
            b"pdb_id,x\n1ABC,1\n2ABC,1,extra\n",
            "Line 3 of the CSV file has 3 fields; the header has 2.",
        ),
    ]

    for content, message in cases:
        status, payload = _post_filter(client, content)

        assert status == 400
        assert payload["error"].startswith(message)
        assert _datasets(tmp_path) == []


def test_a_long_upload_name_gets_a_short_servable_name(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)

    status, payload = _post_filter(
        client, b"pdb_id\n1ABC\n", filename="data/" + "x" * 300 + ".csv"
    )

    assert status == 200
    (name,) = _datasets(tmp_path)
    assert name.startswith("x" * 100 + "_filtered_") and len(name) == 100 + 10 + 32 + 4
    assert client.get(payload["csvUrl"]).data == b"pdb_id\r\n1ABC\r\n"


def test_filter_upload_removes_only_its_own_stale_tables(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    client = _client(tmp_path, monkeypatch)
    datasets = tmp_path / "workspace" / "datasets"
    stale = datasets / f"old_filtered_{'a' * 32}.csv"
    other = datasets / "result_filtered_by_hand.csv"
    fresh = datasets / f"new_filtered_{'b' * 32}.csv"
    for path in (stale, other, fresh):
        path.write_text("pdb_id\n", encoding="utf-8")
    two_hours_ago = time.time() - 2 * 60 * 60
    for path in (stale, other):
        os.utime(path, (two_hours_ago, two_hours_ago))

    status, _payload = _post_filter(client, b"pdb_id\n1ABC\n")

    assert status == 200
    assert not stale.exists()
    assert other.exists() and fresh.exists()
