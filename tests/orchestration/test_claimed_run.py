from __future__ import annotations

from pathlib import Path

import pytest

from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import EngineOptions
from pdbai.orchestration import api as api_module
from pdbai.orchestration.api import build_atom_site_dataset
from pdbai.orchestration.build import DatasetBuild
from pdbai.results import RunPausedError
from pdbai.runs import open_run

_OPTIONS = EngineOptions(values={"include_rabdam": False})


def _failing_downloads(
    monkeypatch: pytest.MonkeyPatch, exception: BaseException
) -> None:
    def process_downloads(build: DatasetBuild, **_kwargs: object) -> None:
        build.report_progress("download", build.pdb_ids[0])
        raise exception

    monkeypatch.setattr(api_module, "process_downloads", process_downloads)


def _no_downloads(*_args: object, **_kwargs: object) -> None:
    """Record no entry, as a cancelled build would, so the run pauses."""
    return None


def _build(tmp_path: Path, run_id: str, *, resume: bool = False) -> None:
    build_atom_site_dataset(
        ["1ABC"],
        tmp_path,
        engine_options=_OPTIONS,
        run_id=run_id,
        resume=resume,
        publish=False,
    )


def test_a_failure_before_settlement_leaves_the_run_in_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_downloads(monkeypatch, OSError("disk full"))

    with pytest.raises(OSError, match="disk full"):
        _build(tmp_path, "failed")

    manifest = open_run(tmp_path, "failed").manifest()
    assert manifest["run_status"] == "error"
    assert manifest["error"] == "disk full"
    assert not open_run(tmp_path, "failed").is_held()


def test_a_failure_without_a_message_is_recorded_by_its_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_downloads(monkeypatch, AssertionError())

    with pytest.raises(AssertionError):
        _build(tmp_path, "silent")

    assert open_run(tmp_path, "silent").manifest()["error"] == "AssertionError"


def test_an_interruption_leaves_the_run_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _failing_downloads(monkeypatch, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        _build(tmp_path, "interrupted")

    assert open_run(tmp_path, "interrupted").manifest()["run_status"] == "running"


def test_a_pause_before_any_entry_is_done_stays_paused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(api_module, "process_downloads", _no_downloads)

    with pytest.raises(RunPausedError):
        _build(tmp_path, "paused")

    assert open_run(tmp_path, "paused").manifest()["run_status"] == "paused"


def test_a_claim_refused_for_other_settings_leaves_the_run_as_it_was(
    tmp_path: Path,
) -> None:
    run_state = open_run(tmp_path, "settled")
    run_state.load_or_initialize(
        ["1ABC"],
        get_engine_registry()
        .with_defaults(
            EngineOptions(values={"include_rabdam": False, "include_sasa": True})
        )
        .as_settings_dict(),
        schema_fingerprint="test-schema",
    )
    run_state.set_run_status("completed")

    with pytest.raises(ValueError, match="different settings"):
        _build(tmp_path, "settled", resume=True)

    manifest = open_run(tmp_path, "settled").manifest()
    assert manifest["run_status"] == "completed"
    assert "error" not in manifest
