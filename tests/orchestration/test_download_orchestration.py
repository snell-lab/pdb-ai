from __future__ import annotations

import csv
import gc
import json
import threading
import time
import weakref
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pytest
import requests
from fake_http import FakeResponse, FakeSession

from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import (
    ENTRY_RECORD,
    STRUCTURE_CIF_PATH,
    STRUCTURE_METADATA,
    EngineContext,
    EngineOptions,
    EnginePhaseResult,
    EngineServices,
    ReportValue,
)
from pdbai.engines.sources.http_cache import CACHE_TTL_SECONDS
from pdbai.orchestration import download as download_module
from pdbai.orchestration import processing as processing_module
from pdbai.orchestration.annotation import AnnotatedStructure
from pdbai.orchestration.api import build_atom_site_dataset
from pdbai.orchestration.build import DatasetBuild
from pdbai.orchestration.download import DownloadAcquirer, acquire_concurrently
from pdbai.orchestration.processing import AcquiredEntry, process_entry
from pdbai.results import MetadataFetchResult, PdbProcessResult, ProcessState
from pdbai.runs import ProgressCallback, ProgressStage, open_run
from pdbai.schema import schema_fingerprint
from pdbai.structure.mmcif_parser import AtomSiteParseResult


def _build_settings(options: EngineOptions) -> dict[str, object]:
    """Return the settings a download build without a column filter records."""
    return {
        "column_filter_mode": None,
        "column_filter_list": [],
        **get_engine_registry().with_defaults(options).as_settings_dict(),
    }


def _download_fingerprint(options: EngineOptions) -> str:
    """Return the schema fingerprint a download build under options records."""
    active = get_engine_registry().with_defaults(options)
    return schema_fingerprint(active, source="download")


def _build(
    tmp_path: Path,
    *,
    pdb_ids: Sequence[str] = ("1ABC",),
    progress_callback: ProgressCallback | None = None,
    cancel_token: threading.Event | None = None,
) -> DatasetBuild:
    active = get_engine_registry().with_defaults(EngineOptions())
    run_state = open_run(tmp_path, "download-test")
    run_state.load_or_initialize(
        list(pdb_ids), active.as_settings_dict(), schema_fingerprint="test-schema"
    )
    return DatasetBuild(
        run_state=run_state,
        working_directory=tmp_path,
        pdb_ids=list(pdb_ids),
        options=active,
        progress_callback=progress_callback,
        cancel_token=cancel_token,
    )


def _resumed_build(
    tmp_path: Path,
    run_id: str,
    entries: dict[str, ProcessState],
    *,
    progress_callback: ProgressCallback | None = None,
    cancel_token: threading.Event | None = None,
) -> DatasetBuild:
    """Return a build resuming a run whose entries are in the given states.

    Each done entry has a partial on disk.
    """
    options = get_engine_registry().with_defaults(
        EngineOptions(values={"include_rabdam": False})
    )
    run_state = open_run(tmp_path, run_id)
    run_state.load_or_initialize(
        list(entries), _build_settings(options), schema_fingerprint="test-schema"
    )
    for pdb_id, state in entries.items():
        if state == "done":
            partial = run_state.directory.partials_directory / f"{pdb_id}.csv"
            partial.write_text(f"pdb_id,atom_name\n{pdb_id},CA\n", encoding="utf-8")
            run_state.update_entry(
                pdb_id,
                state="done",
                partial_paths={"flat": f"partials/{pdb_id}.csv"},
            )
        elif state == "failed":
            run_state.update_entry(pdb_id, state="failed", error="network unavailable")
    return DatasetBuild(
        run_state=run_state,
        working_directory=tmp_path,
        pdb_ids=list(entries),
        options=options,
        output_filename=f"{run_id}.csv",
        progress_callback=progress_callback,
        resume=True,
        cancel_token=cancel_token,
    )


def _acquirer(tmp_path: Path) -> DownloadAcquirer:
    return DownloadAcquirer(
        options=EngineOptions(),
        services=EngineServices(
            session=requests.Session(), timeout=1.0, cache_root=tmp_path / "mmcif"
        ),
        working_directory=tmp_path,
    )


def _acquired(
    result: PdbProcessResult,
    *,
    cif_path: Path | None = None,
    input_receipts: tuple[dict[str, object], ...] = (),
    acquisition_result: EnginePhaseResult | None = None,
) -> AcquiredEntry:
    context = EngineContext(result.pdb_id, pd.DataFrame(), EngineOptions())
    if cif_path is not None:
        context.artifacts.put(STRUCTURE_CIF_PATH, cif_path)
    return AcquiredEntry(
        result=result,
        context=context,
        input_receipts=input_receipts,
        acquisition_result=acquisition_result or EnginePhaseResult(),
    )


class _FakeAcquirer:
    """Acquire entries without the network, noting each acquisition started.

    ``before_return`` runs inside the acquisition, on its worker thread.
    """

    def __init__(self, before_return: Callable[[str], object] | None = None) -> None:
        self.before_return = before_return
        self.started: list[str] = []
        self.lock = threading.Lock()

    def acquire(self, pdb_id: str) -> AcquiredEntry:
        with self.lock:
            self.started.append(pdb_id)
        if self.before_return is not None:
            self.before_return(pdb_id)
        return _acquired(PdbProcessResult(pdb_id=pdb_id))


def _ids(count: int) -> list[str]:
    return [f"{index:04d}" for index in range(count)]


def test_acquisition_error_becomes_the_entrys_download_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def exploding_fetch(*_args: object, **_kwargs: object) -> MetadataFetchResult:
        raise RuntimeError("network exploded")

    monkeypatch.setattr(download_module, "fetch_rcsb_metadata", exploding_fetch)
    build = _build(tmp_path)

    acquired = _acquirer(tmp_path).acquire("1ABC")
    process_entry(build, acquired)

    message = "Download failed for 1ABC: network exploded"
    assert acquired.result.download_error == message
    entry = build.run_state.manifest()["entries"]["1ABC"]
    assert entry["state"] == "failed"
    assert entry["download_error"] == message
    assert entry["inputs"] == []
    assert build.warnings == [message]


def test_cancelled_build_starts_no_acquisition(tmp_path: Path) -> None:
    cancel_token = threading.Event()
    cancel_token.set()
    acquirer = _FakeAcquirer()

    acquired = list(
        acquire_concurrently(
            acquirer.acquire,
            ["1ABC"],
            build=_build(tmp_path, cancel_token=cancel_token),
            worker_count=1,
        )
    )

    assert acquired == []
    assert acquirer.started == []


def test_cancel_while_processing_waits_yields_nothing_more(tmp_path: Path) -> None:
    cancel_token = threading.Event()
    reported = threading.Event()

    def note_report(*_args: object) -> None:
        reported.set()

    def cancel_once_reported(_pdb_id: str) -> None:
        # The loop reports progress just before it waits, so the cancel
        # arrives while processing waits on this acquisition.
        reported.wait(timeout=5)
        cancel_token.set()

    acquirer = _FakeAcquirer(before_return=cancel_once_reported)
    ids = _ids(10)
    build = _build(
        tmp_path,
        pdb_ids=ids,
        progress_callback=note_report,
        cancel_token=cancel_token,
    )

    acquired = list(
        acquire_concurrently(acquirer.acquire, ids, build=build, worker_count=1)
    )

    assert acquired == []
    # Each worker keeps two entries in flight. The one worker may start the
    # queued second entry before the loop cancels it.
    assert acquirer.started in (ids[:1], ids[:2])


def test_acquisition_runs_at_most_two_entries_ahead_per_worker(
    tmp_path: Path,
) -> None:
    # A cohort of thousands must not queue thousands of finished downloads
    # ahead of processing.
    acquirer = _FakeAcquirer()
    ids = _ids(40)
    build = _build(tmp_path, pdb_ids=ids)
    taken: list[str] = []
    ahead: list[int] = []

    for acquired in acquire_concurrently(
        acquirer.acquire, ids, build=build, worker_count=2
    ):
        # The sleep makes processing slower than acquisition, so the workers
        # fill the window.
        time.sleep(0.002)
        with acquirer.lock:
            ahead.append(len(acquirer.started) - len(taken))
        taken.append(acquired.result.pdb_id)

    assert sorted(taken) == ids
    assert max(ahead) <= 4


def test_waiting_on_a_download_reports_the_download_stage(tmp_path: Path) -> None:
    reported: list[tuple[ProgressStage, int, int, str]] = []
    was_reported = threading.Event()

    def record(stage: ProgressStage, current: int, total: int, pdb_id: str) -> None:
        reported.append((stage, current, total, pdb_id))
        was_reported.set()

    acquirer = _FakeAcquirer(before_return=lambda _pdb_id: was_reported.wait(5))
    build = _build(tmp_path, progress_callback=record)

    yielded = list(
        acquire_concurrently(acquirer.acquire, ["1ABC"], build=build, worker_count=1)
    )

    assert [acquired.result.pdb_id for acquired in yielded] == ["1ABC"]
    assert reported == [("download", 1, 1, "1ABC")]


def test_queued_acquisitions_are_cancelled_when_processing_raises(
    tmp_path: Path,
) -> None:
    def first_at_once(pdb_id: str) -> None:
        # Every entry but the first stays busy long enough for the loop to
        # cancel the queued ones.
        if pdb_id != "0000":
            time.sleep(0.2)

    acquirer = _FakeAcquirer(before_return=first_at_once)
    ids = _ids(10)
    build = _build(tmp_path, pdb_ids=ids)

    with pytest.raises(RuntimeError, match="processing failed"):
        for _acquired_entry in acquire_concurrently(
            acquirer.acquire, ids, build=build, worker_count=2
        ):
            raise RuntimeError("processing failed")

    # Two workers keep four entries in flight. Both workers stay busy with
    # slow entries, so at least one queued entry is cancelled before it
    # starts. Closing the executor alone would have run all four.
    assert len(acquirer.started) < 4


def test_taken_entry_is_unreachable_once_the_next_is_taken(
    tmp_path: Path,
) -> None:
    # A long build may hold only one structure's data at a time.
    acquirer = _FakeAcquirer()
    ids = _ids(4)
    build = _build(tmp_path, pdb_ids=ids)
    taken: list[weakref.ref[AcquiredEntry]] = []

    for acquired in acquire_concurrently(
        acquirer.acquire, ids, build=build, worker_count=1
    ):
        gc.collect()
        assert [ref() for ref in taken] == [None] * len(taken)
        taken.append(weakref.ref(acquired))

    assert len(taken) == 4


def test_acquisition_publishes_the_metadata_and_entry_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def metadata(*_args: object, **_kwargs: object) -> MetadataFetchResult:
        return MetadataFetchResult(
            metadata={"experimental_methods": ["X-RAY DIFFRACTION"]},
            ok=True,
            entry_record={"rcsb_id": "1ABC"},
        )

    seen: list[EngineContext] = []

    def fake_phase(context: EngineContext, *_args: object, **_kwargs: object) -> object:
        seen.append(context)
        return EnginePhaseResult()

    monkeypatch.setattr(download_module, "fetch_rcsb_metadata", metadata)
    monkeypatch.setattr(download_module, "run_engine_phase", fake_phase)

    acquired = _acquirer(tmp_path).acquire("1ABC")

    assert seen and seen[0].artifacts.get(STRUCTURE_METADATA) == {
        "experimental_methods": ["X-RAY DIFFRACTION"]
    }
    assert seen[0].artifacts.get(ENTRY_RECORD) == {"rcsb_id": "1ABC"}
    # The context the acquisition engines filled is the one annotation reads.
    assert acquired.context is seen[0]
    # The stub engines put no structure file in the context, so the entry
    # does not count as downloaded.
    assert acquired.result.download_ok is False


def test_stale_metadata_cache_is_not_receipted_as_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_root = tmp_path / "mmcif"
    cache_root.mkdir()
    (cache_root / "1abc_metadata.json").write_text(
        json.dumps({"cached_at": time.time() - CACHE_TTL_SECONDS - 1, "entry": {}}),
        encoding="utf-8",
    )
    session = FakeSession(FakeResponse(payload={}))

    def no_engines(*_args: object, **_kwargs: object) -> EnginePhaseResult:
        return EnginePhaseResult()

    monkeypatch.setattr(download_module, "run_engine_phase", no_engines)
    acquirer = DownloadAcquirer(
        options=EngineOptions(),
        services=EngineServices(
            session=session.as_session(), timeout=1.0, cache_root=cache_root
        ),
        working_directory=tmp_path,
    )

    acquired = acquirer.acquire("1ABC")

    assert len(session.calls) == 1
    receipt = acquired.input_receipts[0]
    assert receipt["role"] == "rcsb_entry_metadata"
    assert receipt["cache_present_before_acquisition"] is False


def test_one_entry_download_failure_is_persisted_without_parsing(
    tmp_path: Path,
) -> None:
    stages: list[ProgressStage] = []

    def record(stage: ProgressStage, *_args: object) -> None:
        stages.append(stage)

    build = _build(tmp_path, progress_callback=record)
    result = PdbProcessResult(
        pdb_id="1ABC",
        source="download",
        download_error="RCSB unavailable",
    )

    process_entry(build, _acquired(result))

    manifest_entry = build.run_state.manifest()["entries"]["1ABC"]
    assert manifest_entry["state"] == "failed"
    assert manifest_entry["download_ok"] is False
    assert manifest_entry["download_error"] == "RCSB unavailable"
    assert build.results == [result]
    assert build.warnings == ["RCSB unavailable"]
    assert "parse" not in stages


def test_missing_structure_file_is_a_download_failure(tmp_path: Path) -> None:
    stages: list[ProgressStage] = []

    def record(stage: ProgressStage, *_args: object) -> None:
        stages.append(stage)

    build = _build(tmp_path, progress_callback=record)
    result = PdbProcessResult(pdb_id="1ABC")

    process_entry(build, _acquired(result))

    assert result.state == "failed"
    assert result.download_error == "No structure file was acquired for 1ABC."
    assert result.parse_error is None
    assert build.warnings == ["No structure file was acquired for 1ABC."]
    assert "parse" not in stages
    logged = build.run_state.directory.entries_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(line)["pdb_id"] for line in logged] == ["1ABC"]


def test_parser_error_is_a_parse_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages: list[ProgressStage] = []

    def record(stage: ProgressStage, *_args: object) -> None:
        stages.append(stage)

    def broken_parse(*_args: object, **_kwargs: object) -> AtomSiteParseResult:
        raise RuntimeError("Error parsing 1ABC: bad cif")

    monkeypatch.setattr(processing_module, "parse_mmcif_atom_rows", broken_parse)
    build = _build(tmp_path, progress_callback=record)
    result = PdbProcessResult(pdb_id="1ABC", download_ok=True)

    process_entry(build, _acquired(result, cif_path=tmp_path / "1abc.cif"))

    assert result.state == "failed"
    assert result.parse_ok is False
    assert result.parse_error == "Error parsing 1ABC: bad cif"
    assert build.warnings == ["Error parsing 1ABC: bad cif"]
    assert stages == ["parse"]
    entry = build.run_state.manifest()["entries"]["1ABC"]
    assert entry["parse_error"] == "Error parsing 1ABC: bad cif"


def test_dropped_entry_is_written_once_with_its_inputs(tmp_path: Path) -> None:
    build = _build(tmp_path)
    receipt: dict[str, object] = {"role": "rcsb_entry_metadata", "path": "1abc.json"}
    result = PdbProcessResult(pdb_id="1ABC", download_error="RCSB unavailable")

    process_entry(build, _acquired(result, input_receipts=(receipt,)))

    run_state = build.run_state
    assert run_state.manifest()["entries"]["1ABC"]["inputs"] == [receipt]
    logged = run_state.directory.entries_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["pdb_id"] for line in logged] == ["1ABC"]


def test_resumed_build_numbers_progress_after_the_done_entries(
    tmp_path: Path,
) -> None:
    reported: list[tuple[ProgressStage, int, int, str]] = []

    def record(stage: ProgressStage, current: int, total: int, pdb_id: str) -> None:
        reported.append((stage, current, total, pdb_id))

    build = _resumed_build(
        tmp_path,
        "numbered",
        {"1ABC": "done", "2XYZ": "pending", "3DEF": "pending"},
        progress_callback=record,
    )

    build.report_progress("download", "2XYZ")
    build.record_failed_or_refused(
        PdbProcessResult(
            pdb_id="2XYZ", state="failed", download_error="network unavailable"
        ),
        input_receipts=(),
    )
    build.report_progress("download", "3DEF")

    assert build.ids_to_process == ["2XYZ", "3DEF"]
    assert reported == [("download", 2, 3, "2XYZ"), ("download", 3, 3, "3DEF")]


def test_cancel_during_the_last_entry_leaves_the_run_completed(
    tmp_path: Path,
) -> None:
    cancel_token = threading.Event()
    build = _resumed_build(
        tmp_path,
        "cancelled-late",
        {"1ABC": "done", "2XYZ": "pending"},
        cancel_token=cancel_token,
    )

    cancel_token.set()
    build.record_failed_or_refused(
        PdbProcessResult(
            pdb_id="2XYZ", state="failed", download_error="network unavailable"
        ),
        input_receipts=(),
    )
    result = build.finish()

    assert build.run_state.manifest()["run_status"] == "completed"
    assert not any("Run paused" in warning for warning in result.warnings)


def test_cancel_before_earlier_failures_are_retried_leaves_the_run_paused(
    tmp_path: Path,
) -> None:
    cancel_token = threading.Event()
    build = _resumed_build(
        tmp_path,
        "cancelled-retry",
        {"1ABC": "done", "2XYZ": "failed"},
        cancel_token=cancel_token,
    )
    cancel_token.set()

    download_module.process_downloads(build, max_workers=1)
    result = build.finish()

    assert build.ids_to_process == ["2XYZ"]
    assert build.results == []
    assert build.run_state.manifest()["run_status"] == "paused"
    assert any("Run paused" in warning for warning in result.warnings)


def test_empty_parse_is_an_output_failure_with_a_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _build(tmp_path)
    structure_ids: list[object] = []

    def empty_parse(*_args: object, **kwargs: object) -> AtomSiteParseResult:
        structure_ids.append(kwargs.get("structure_id"))
        return AtomSiteParseResult(pd.DataFrame())

    monkeypatch.setattr(processing_module, "parse_mmcif_atom_rows", empty_parse)
    result = PdbProcessResult(pdb_id="1ABC", download_ok=True)

    process_entry(build, _acquired(result, cif_path=tmp_path / "sample.cif"))

    # The parser names the rows after the entry, not after the file.
    assert structure_ids == ["1ABC"]
    assert result.parse_ok is True
    assert result.succeeded() is False
    assert result.output_error == "Parsed atom-site DataFrame was empty for 1ABC."
    assert result.state == "failed"
    assert build.warnings == ["Parsed atom-site DataFrame was empty for 1ABC."]
    entry = build.run_state.manifest()["entries"]["1ABC"]
    assert entry["parse_ok"] is True
    assert entry["state"] == "failed"


def test_failed_metadata_fetch_is_warned_once_the_structure_is_parsed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _build(tmp_path)

    def empty_parse(*_args: object, **_kwargs: object) -> AtomSiteParseResult:
        return AtomSiteParseResult(pd.DataFrame())

    monkeypatch.setattr(processing_module, "parse_mmcif_atom_rows", empty_parse)
    result = PdbProcessResult(
        pdb_id="1ABC",
        download_ok=True,
        metadata_message="RCSB metadata unavailable for 1ABC",
    )

    process_entry(build, _acquired(result, cif_path=tmp_path / "1abc.cif"))

    assert build.warnings == [
        "RCSB metadata unavailable for 1ABC",
        "Parsed atom-site DataFrame was empty for 1ABC.",
    ]


def test_acquisition_and_annotation_reports_both_reach_the_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _build(tmp_path)
    atom_dataframe = pd.DataFrame({"pdb_id": ["1ABC"], "atom_name": ["CA"]})

    def parsed_atoms(*_args: object, **_kwargs: object) -> AtomSiteParseResult:
        return AtomSiteParseResult(atom_dataframe)

    def annotated(*_args: object, **_kwargs: object) -> AnnotatedStructure:
        return AnnotatedStructure(
            atom_dataframe, EnginePhaseResult(report={"annotation_status": "ok"})
        )

    monkeypatch.setattr(processing_module, "parse_mmcif_atom_rows", parsed_atoms)
    monkeypatch.setattr(processing_module, "annotate_structure", annotated)
    result = PdbProcessResult(pdb_id="1ABC", download_ok=True)
    acquisition_result = EnginePhaseResult(
        warnings=("mirror was slow",), report={"acquisition_status": "cached"}
    )

    process_entry(
        build,
        _acquired(
            result,
            cif_path=tmp_path / "1abc.cif",
            acquisition_result=acquisition_result,
        ),
    )

    assert result.state == "done"
    assert result.engine_report == {
        "acquisition_status": "cached",
        "annotation_status": "ok",
    }
    assert "Engine warning for 1ABC: mirror was slow" in build.warnings


def test_annotation_failure_is_an_output_failure_not_a_parse_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _build(tmp_path)

    def parsed_atoms(*_args: object, **_kwargs: object) -> AtomSiteParseResult:
        return AtomSiteParseResult(pd.DataFrame({"pdb_id": ["1ABC"]}))

    def fail_annotation(*_args: object, **_kwargs: object) -> None:
        raise ValueError("bad engineered column")

    monkeypatch.setattr(processing_module, "parse_mmcif_atom_rows", parsed_atoms)
    monkeypatch.setattr(processing_module, "annotate_structure", fail_annotation)
    result = PdbProcessResult(pdb_id="1ABC", download_ok=True)

    process_entry(build, _acquired(result, cif_path=tmp_path / "1abc.cif"))

    assert result.parse_ok is True
    assert result.parse_error is None
    assert result.succeeded() is False
    assert result.output_error == "Annotation failed for 1ABC: bad engineered column"


def test_engine_reports_are_persisted_and_restored_on_resume(tmp_path: Path) -> None:
    build = _build(tmp_path)
    report: dict[str, ReportValue] = {
        "rabdam_status": "partial",
        "rabdam_delivered_pdb_redo": False,
        "rabdam_message": "rabdam-custom Bnet unavailable for 1ABC: zero variance",
        "pdb_redo_alignment_match_rate": 0.75,
    }
    build.run_state.update_entry(
        "1ABC",
        **PdbProcessResult(
            pdb_id="1ABC", state="done", engine_report=dict(report)
        ).to_manifest(),
    )

    restored = PdbProcessResult.from_manifest(
        "1ABC",
        build.run_state.manifest()["entries"]["1ABC"],
        default_source="download",
    )

    assert restored.engine_report == report


@pytest.mark.parametrize("state", ["unknown", ["done"], {"done": True}, 1])
def test_unrecognized_manifest_state_restores_as_pending(state: object) -> None:
    restored = PdbProcessResult.from_manifest(
        "1ABC", {"state": state}, default_source="download"
    )

    assert restored.state == "pending"


@pytest.mark.parametrize(
    ("stored", "restored_status"),
    [("matched", "matched"), ("unverified", "unverified"), ("partly", None), (1, None)],
)
def test_manifest_technique_status_restores_only_known_values(
    stored: object, restored_status: str | None
) -> None:
    restored = PdbProcessResult.from_manifest(
        "1ABC", {"technique_status": stored}, default_source="download"
    )

    assert restored.technique_status == restored_status


def test_restored_entry_keeps_its_source_and_an_absent_metadata_flag_is_unknown() -> (
    None
):
    restored = PdbProcessResult.from_manifest(
        "1ABC", {"state": "done", "source": "upload"}, default_source="download"
    )

    assert restored.source == "upload"
    assert restored.metadata_ok is None


def test_paused_reports_include_done_failed_and_pending_entries(
    tmp_path: Path,
) -> None:
    options = EngineOptions(values={"include_rabdam": False})
    run_state = open_run(tmp_path, "paused-download")
    run_state.load_or_initialize(
        ["1ABC", "2XYZ", "3DEF"],
        _build_settings(options),
        schema_fingerprint=_download_fingerprint(options),
    )
    partial = run_state.directory.partials_directory / "1ABC.csv"
    partial.write_text("pdb_id,atom_name\n1ABC,CA\n", encoding="utf-8")
    run_state.update_entry(
        "1ABC",
        state="done",
        source="download",
        download_ok=True,
        parse_ok=True,
        partial_paths={"flat": "partials/1ABC.csv"},
    )
    run_state.update_entry(
        "2XYZ",
        state="failed",
        source="download",
        download_ok=False,
        download_error="network unavailable",
        error="network unavailable",
    )
    # The build is cancelled before it starts. 3DEF is never acquired and
    # stays pending.
    cancel_token = threading.Event()
    cancel_token.set()

    result = build_atom_site_dataset(
        ["1ABC", "2XYZ", "3DEF"],
        tmp_path,
        output_filename="paused.csv",
        max_workers=1,
        run_id="paused-download",
        resume=True,
        cancel_token=cancel_token,
        engine_options=options,
    )

    assert [pdb_result.state for pdb_result in result.pdb_results] == [
        "done",
        "failed",
        "pending",
    ]
    with result.run_report_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["pdb_id"] for row in rows] == ["1ABC", "2XYZ", "3DEF"]
    assert [row["state"] for row in rows] == ["done", "failed", "pending"]
    assert rows[1]["download_error"] == "network unavailable"
    assert rows[2]["download_success"] == ""
    summary = result.run_summary_path.read_text(encoding="utf-8")
    assert "| 2XYZ | download | failed |" in summary
    assert "| 3DEF | download | pending |" in summary


def test_resume_with_no_remaining_entries_assembles_without_acquisition(
    tmp_path: Path,
) -> None:
    options = EngineOptions(values={"include_rabdam": False})
    run_state = open_run(tmp_path, "completed-download")
    run_state.load_or_initialize(
        ["1ABC"],
        _build_settings(options),
        schema_fingerprint=_download_fingerprint(options),
    )
    partial = run_state.directory.partials_directory / "1ABC.csv"
    partial.write_text("pdb_id,atom_name\n1ABC,CA\n", encoding="utf-8")
    run_state.update_entry(
        "1ABC",
        state="done",
        source="download",
        download_ok=True,
        parse_ok=True,
        partial_paths={"flat": "partials/1ABC.csv"},
    )

    result = build_atom_site_dataset(
        ["1ABC"],
        tmp_path,
        output_filename="resumed.csv",
        max_workers=1,
        run_id="completed-download",
        resume=True,
        engine_options=options,
    )

    assert result.atom_count == 1
    assert result.csv_path.read_text(encoding="utf-8").splitlines() == [
        "pdb_id,atom_name",
        "1ABC,CA",
    ]


def test_resume_naming_some_entries_settles_the_whole_run(tmp_path: Path) -> None:
    options = EngineOptions(values={"include_rabdam": False})
    run_state = open_run(tmp_path, "subset-download")
    run_state.load_or_initialize(
        ["1ABC", "2XYZ"],
        _build_settings(options),
        schema_fingerprint=_download_fingerprint(options),
    )
    for pdb_id in ("1ABC", "2XYZ"):
        partial = run_state.directory.partials_directory / f"{pdb_id}.csv"
        partial.write_text(f"pdb_id,atom_name\n{pdb_id},CA\n", encoding="utf-8")
        run_state.update_entry(
            pdb_id,
            state="done",
            source="download",
            download_ok=True,
            parse_ok=True,
            partial_paths={"flat": f"partials/{pdb_id}.csv"},
        )

    result = build_atom_site_dataset(
        ["2XYZ"],
        tmp_path,
        output_filename="subset.csv",
        run_id="subset-download",
        resume=True,
        engine_options=options,
    )

    assert result.pdb_ids == ["1ABC", "2XYZ"]
    assert result.csv_path.read_text(encoding="utf-8").splitlines() == [
        "pdb_id,atom_name",
        "1ABC,CA",
        "2XYZ,CA",
    ]
    with result.run_report_path.open(encoding="utf-8", newline="") as stream:
        assert [row["pdb_id"] for row in csv.DictReader(stream)] == ["1ABC", "2XYZ"]


def test_download_api_deduplicates_ids_after_normalization(tmp_path: Path) -> None:
    options = EngineOptions(values={"include_rabdam": False})
    run_state = open_run(tmp_path, "duplicate-download")
    run_state.load_or_initialize(
        ["1ABC", "2XYZ"],
        _build_settings(options),
        schema_fingerprint=_download_fingerprint(options),
    )
    for pdb_id in ("1ABC", "2XYZ"):
        partial = run_state.directory.partials_directory / f"{pdb_id}.csv"
        partial.write_text(f"pdb_id,atom_name\n{pdb_id},CA\n", encoding="utf-8")
        run_state.update_entry(
            pdb_id,
            state="done",
            source="download",
            download_ok=True,
            parse_ok=True,
            partial_paths={"flat": f"partials/{pdb_id}.csv"},
        )

    result = build_atom_site_dataset(
        ["1abc", "1ABC", "2xyz", "2XYZ"],
        tmp_path,
        output_filename="deduplicated.csv",
        run_id="duplicate-download",
        resume=True,
        engine_options=options,
    )

    assert result.pdb_ids == ["1ABC", "2XYZ"]
    assert result.atom_count == 2
    assert result.warnings == [
        "Duplicate PDB IDs were ignored after normalization: 1ABC, 2XYZ."
    ]
    assert run_state.manifest()["pdb_ids"] == ["1ABC", "2XYZ"]


def test_finish_reports_columns_from_every_partial(tmp_path: Path) -> None:
    options = EngineOptions(values={"include_rabdam": False})
    run_state = open_run(tmp_path, "evolving-download")
    run_state.load_or_initialize(
        ["1ABC", "2XYZ"],
        _build_settings(options),
        schema_fingerprint=_download_fingerprint(options),
    )
    first = run_state.directory.partials_directory / "1ABC.csv"
    second = run_state.directory.partials_directory / "2XYZ.csv"
    first.write_text("pdb_id,first_value\n1ABC,1\n", encoding="utf-8")
    second.write_text("pdb_id,later_value\n2XYZ,2\n", encoding="utf-8")
    for pdb_id, partial in (("1ABC", first), ("2XYZ", second)):
        run_state.update_entry(
            pdb_id,
            state="done",
            source="download",
            download_ok=True,
            parse_ok=True,
            partial_paths={"flat": f"partials/{partial.name}"},
        )

    result = build_atom_site_dataset(
        ["1ABC", "2XYZ"],
        tmp_path,
        output_filename="evolving.csv",
        max_workers=1,
        run_id="evolving-download",
        resume=True,
        engine_options=options,
    )

    assert result.column_count == 3
    assert result.csv_path.read_text(encoding="utf-8").splitlines()[0] == (
        "pdb_id,first_value,later_value"
    )
    assert "| later_value |" in result.run_summary_path.read_text(encoding="utf-8")
