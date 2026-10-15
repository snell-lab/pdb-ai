from __future__ import annotations

import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from pdbai.engines.core import (
    STRUCTURE_CIF_PATH,
    EngineContext,
    EngineOptions,
    EnginePhaseResult,
    EngineServices,
)
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files
from pdbai.orchestration import download as download_module
from pdbai.orchestration import processing as processing_module
from pdbai.orchestration.build import DatasetBuild
from pdbai.orchestration.download import DownloadAcquirer
from pdbai.orchestration.processing import AcquiredEntry, process_entry
from pdbai.orchestration.technique_intake import (
    ReportedExperiment,
    decide_technique,
    intake_warnings,
)
from pdbai.reporting import write_run_report
from pdbai.results import MetadataFetchResult, PdbProcessResult
from pdbai.runs import list_run_summaries, open_run
from pdbai.structure.mmcif_parser import AtomSiteParseResult

# The default options admit X-ray diffraction from any radiation source.
_XRAY = EngineOptions()


def _reported(
    *methods: str, radiation_sources: tuple[str, ...] = ()
) -> ReportedExperiment:
    return ReportedExperiment(
        experimental_methods=methods, radiation_sources=radiation_sources
    )


def test_reported_experiment_reads_the_rcsb_record_and_the_file() -> None:
    from_record = ReportedExperiment.from_rcsb_metadata(
        {
            "experimental_methods": ["X-RAY DIFFRACTION", "NEUTRON DIFFRACTION"],
            "radiation_sources": [
                "SYNCHROTRON",
                "Synchrotron",
                "rotating-anode X-ray tube",
            ],
            "structure_determination_methodology": "integrative",
        }
    )
    from_file = ReportedExperiment.from_parsed_file(
        AtomSiteParseResult(
            pd.DataFrame(),
            experimental_methods=("X-RAY DIFFRACTION",),
            is_integrative=True,
            radiation_sources=("Rotating Copper Anode",),
        )
    )

    assert from_record == ReportedExperiment(
        experimental_methods=("X-RAY DIFFRACTION", "NEUTRON DIFFRACTION"),
        # Source names are normalized. A source repeated for each dataset is
        # kept once.
        radiation_sources=("SYNCHROTRON", "ROTATING ANODE"),
        is_integrative=True,
    )
    assert ReportedExperiment.from_rcsb_metadata({}) == ReportedExperiment()
    assert from_file == ReportedExperiment(
        experimental_methods=("X-RAY DIFFRACTION",),
        radiation_sources=("ROTATING ANODE",),
        is_integrative=True,
    )


def test_decision_matches_refuses_or_leaves_unverified() -> None:
    matched = decide_technique("1ABC", _reported("X-RAY DIFFRACTION"), _XRAY)
    joint = decide_technique(
        "10YN",
        _reported("X-RAY DIFFRACTION", "NEUTRON DIFFRACTION"),
        EngineOptions(technique="neutron_diffraction"),
    )
    refused = decide_technique("8EMX", _reported("ELECTRON MICROSCOPY"), _XRAY)
    unverified = decide_technique("1ABC", ReportedExperiment(), _XRAY)
    integrative_model = ReportedExperiment(is_integrative=True)
    integrative = decide_technique(
        "9INT", integrative_model, EngineOptions(technique="integrative")
    )
    integrative_refused = decide_technique("9INT", integrative_model, _XRAY)

    assert matched.status == "matched" and matched.message is None
    assert joint.status == "matched"
    assert joint.reported.experimental_methods == (
        "X-RAY DIFFRACTION",
        "NEUTRON DIFFRACTION",
    )
    assert refused.status == "refused"
    assert refused.message == (
        "8EMX reports ELECTRON MICROSCOPY, not X-ray diffraction; refused at "
        "intake for technique 'x_ray_diffraction'."
    )
    assert unverified.status == "unverified"
    assert unverified.message == (
        "1ABC reports no experimental method, so it cannot be verified as X-ray "
        "diffraction; admitted unverified at intake for technique "
        "'x_ray_diffraction'."
    )
    assert integrative.status == "matched"
    assert integrative_refused.status == "refused"
    assert integrative_refused.message == (
        "9INT reports an integrative model, not X-ray diffraction; refused at "
        "intake for technique 'x_ray_diffraction'."
    )


def test_source_filter_refuses_other_and_missing_sources() -> None:
    xfel_only = EngineOptions(xray_source="free_electron_laser")
    xfel = decide_technique(
        "9EZQ",
        _reported("X-RAY DIFFRACTION", radiation_sources=("FREE ELECTRON LASER",)),
        xfel_only,
    )
    synchrotron = decide_technique(
        "5MCC",
        _reported("X-RAY DIFFRACTION", radiation_sources=("SYNCHROTRON",)),
        xfel_only,
    )
    synchrotron_only = EngineOptions(xray_source="synchrotron")
    missing = decide_technique("1OLD", _reported("X-RAY DIFFRACTION"), synchrotron_only)
    nothing_reported = decide_technique("1ABC", ReportedExperiment(), synchrotron_only)
    source_without_method = decide_technique(
        "1ABC", ReportedExperiment(radiation_sources=("SYNCHROTRON",)), synchrotron_only
    )
    unfiltered = decide_technique("1OLD", _reported("X-RAY DIFFRACTION"), _XRAY)
    joint = decide_technique(
        "10YN",
        _reported(
            "X-RAY DIFFRACTION",
            "NEUTRON DIFFRACTION",
            radiation_sources=("ROTATING ANODE", "NUCLEAR REACTOR"),
        ),
        EngineOptions(xray_source="rotating_anode"),
    )
    wrong_technique = decide_technique(
        "8EMX",
        _reported("ELECTRON MICROSCOPY", radiation_sources=("SYNCHROTRON",)),
        synchrotron_only,
    )

    assert xfel.status == "matched"
    assert xfel.reported.radiation_sources == ("FREE ELECTRON LASER",)
    assert synchrotron.status == "refused"
    assert synchrotron.message == (
        "5MCC reports SYNCHROTRON, not X-ray free-electron laser; refused at "
        "intake for xray_source 'free_electron_laser'."
    )
    # An entry that reports no source cannot be verified, so a source filter
    # refuses it.
    assert missing.status == "refused"
    assert missing.message == (
        "1OLD reports no radiation source, so it cannot be verified as "
        "Synchrotron; refused at intake for xray_source 'synchrotron'."
    )
    # An entry that reports no method is still refused when it reports no
    # source. It is admitted unverified when its source matches.
    assert nothing_reported.status == "refused"
    assert nothing_reported.message == (
        "1ABC reports no radiation source, so it cannot be verified as "
        "Synchrotron; refused at intake for xray_source 'synchrotron'."
    )
    assert source_without_method.status == "unverified"
    assert source_without_method.message == (
        "1ABC reports no experimental method, so it cannot be verified as X-ray "
        "diffraction; admitted unverified at intake for technique "
        "'x_ray_diffraction'."
    )
    assert unfiltered.status == "matched"
    assert joint.status == "matched"
    # The technique is checked first, so its refusal message is the one
    # given.
    assert wrong_technique.status == "refused"
    assert wrong_technique.message == (
        "8EMX reports ELECTRON MICROSCOPY, not X-ray diffraction; refused at "
        "intake for technique 'x_ray_diffraction'."
    )


def test_intake_warnings_summarize_refused_and_unverified_entries() -> None:
    refused = [
        PdbProcessResult(pdb_id=f"{number}EMX", state="refused")
        for number in range(1, 23)
    ]
    unverified = PdbProcessResult(pdb_id="1ABC", technique_status="unverified")
    done = PdbProcessResult(pdb_id="2ABC", state="done", technique_status="matched")

    # A warning names at most 20 entries, so two of the 22 refusals are only
    # counted.
    warnings = intake_warnings(
        [*refused, unverified, done], EngineOptions(xray_source="synchrotron")
    )

    assert warnings == [
        "Technique intake refused 22 entries (technique 'x_ray_diffraction', "
        "xray_source 'synchrotron'): "
        + ", ".join(f"{number}EMX" for number in range(1, 21))
        + ", and 2 more. See the run report for each entry's reported methods "
        "and radiation sources.",
        "Technique intake admitted 1 entry unverified (technique "
        "'x_ray_diffraction', xray_source 'synchrotron'), with no experimental "
        "method reported: 1ABC.",
    ]
    assert intake_warnings([refused[0]], _XRAY) == [
        "Technique intake refused 1 entry (technique 'x_ray_diffraction'): 1EMX. "
        "See the run report for each entry's reported methods and radiation "
        "sources."
    ]
    assert intake_warnings([done], _XRAY) == []


def test_source_refusal_is_recorded_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(
        tmp_path,
        EngineOptions(
            xray_source="free_electron_laser", values={"include_rabdam": False}
        ),
    )

    def synchrotron_metadata(*_args: object, **_kwargs: object) -> MetadataFetchResult:
        return _metadata(
            experimental_methods=["X-RAY DIFFRACTION"],
            radiation_sources=["SYNCHROTRON"],
        )

    monkeypatch.setattr(download_module, "fetch_rcsb_metadata", synchrotron_metadata)

    def never_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("acquisition engines ran for a refused entry")

    monkeypatch.setattr(download_module, "run_engine_phase", never_download)

    acquired = harness.acquire("1ABC")
    harness.process(acquired)

    result = harness.build.results[0]
    assert result.refused()
    assert result.radiation_sources == ["SYNCHROTRON"]
    entry = harness.build.run_state.manifest()["entries"]["1ABC"]
    assert entry["radiation_sources"] == ["SYNCHROTRON"]
    assert "xray_source 'free_electron_laser'" in entry["error"]


def test_applying_a_refusal_marks_the_result_refused_not_failed() -> None:
    result = PdbProcessResult(pdb_id="8EMX")

    decide_technique("8EMX", _reported("ELECTRON MICROSCOPY"), _XRAY).apply_to(result)

    assert result.state == "refused"
    assert result.refused()
    assert result.succeeded() is False
    assert result.technique_status == "refused"
    assert result.experimental_methods == ["ELECTRON MICROSCOPY"]
    assert result.to_manifest()["state"] == "refused"


@dataclass
class _Harness:
    """A download build that acquires and processes one entry at a time.

    It takes each entry through the steps process_downloads runs, without
    the worker threads.
    """

    acquirer: DownloadAcquirer
    build: DatasetBuild

    def acquire(self, pdb_id: str) -> AcquiredEntry:
        return self.acquirer.acquire(pdb_id)

    def process(self, acquired: AcquiredEntry) -> None:
        process_entry(self.build, acquired)


def _harness(tmp_path: Path, options: EngineOptions | None = None) -> _Harness:
    active = options or EngineOptions(values={"include_rabdam": False})
    run_state = open_run(tmp_path, "intake-test")
    run_state.load_or_initialize(
        ["1ABC"], active.as_settings_dict(), schema_fingerprint="test-schema"
    )
    build = DatasetBuild(
        run_state=run_state,
        working_directory=tmp_path,
        pdb_ids=["1ABC"],
        options=active,
    )
    return _Harness(
        acquirer=DownloadAcquirer(
            options=active,
            services=EngineServices(
                session=requests.Session(), timeout=1.0, cache_root=tmp_path / "mmcif"
            ),
            working_directory=tmp_path,
        ),
        build=build,
    )


def _metadata(**fields: Any) -> MetadataFetchResult:
    return MetadataFetchResult(metadata=dict(fields), ok=True, level="info")


def test_mismatched_entry_is_refused_before_any_structure_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path)

    def em_metadata(*_args: object, **_kwargs: object) -> MetadataFetchResult:
        return _metadata(experimental_methods=["ELECTRON MICROSCOPY"])

    monkeypatch.setattr(download_module, "fetch_rcsb_metadata", em_metadata)

    def never_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("acquisition engines ran for a refused entry")

    monkeypatch.setattr(download_module, "run_engine_phase", never_download)

    acquired = harness.acquire("1ABC")
    harness.process(acquired)

    result = harness.build.results[0]
    assert result.refused()
    assert result.technique_status == "refused"
    assert result.experimental_methods == ["ELECTRON MICROSCOPY"]
    assert STRUCTURE_CIF_PATH not in acquired.context.artifacts
    # No structure was requested, so only the metadata receipt is recorded.
    assert [receipt["role"] for receipt in acquired.input_receipts] == [
        "rcsb_entry_metadata"
    ]
    entry = harness.build.run_state.manifest()["entries"]["1ABC"]
    assert entry["state"] == "refused"
    assert entry["technique_status"] == "refused"
    assert entry["experimental_methods"] == ["ELECTRON MICROSCOPY"]
    assert "not X-ray diffraction" in entry["error"]
    # A refusal is summarized once at the end of the run, not warned per
    # entry.
    assert harness.build.warnings == []


def test_matching_entry_proceeds_and_records_its_methods(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path)

    def joint_metadata(*_args: object, **_kwargs: object) -> MetadataFetchResult:
        return _metadata(
            experimental_methods=["X-RAY DIFFRACTION", "NEUTRON DIFFRACTION"]
        )

    monkeypatch.setattr(download_module, "fetch_rcsb_metadata", joint_metadata)
    calls: list[str] = []

    def fake_phase(context: Any, *_args: object, **_kwargs: object) -> Any:
        calls.append(context.pdb_id)
        context.artifacts.put(STRUCTURE_CIF_PATH, tmp_path / "1abc.cif")
        return EnginePhaseResult()

    monkeypatch.setattr(download_module, "run_engine_phase", fake_phase)

    acquired = harness.acquire("1ABC")

    assert calls == ["1ABC"]
    assert acquired.result.technique_status == "matched"
    assert acquired.result.experimental_methods == [
        "X-RAY DIFFRACTION",
        "NEUTRON DIFFRACTION",
    ]


def test_unfetched_record_leaves_intake_to_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(
        tmp_path,
        EngineOptions(xray_source="synchrotron", values={"include_rabdam": False}),
    )

    def failed_fetch(*_args: object, **_kwargs: object) -> MetadataFetchResult:
        return MetadataFetchResult(
            metadata={},
            ok=False,
            message="Metadata fetch failed for 1ABC: timed out",
            level="error",
        )

    monkeypatch.setattr(download_module, "fetch_rcsb_metadata", failed_fetch)
    calls: list[str] = []

    def fake_phase(context: Any, *_args: object, **_kwargs: object) -> Any:
        calls.append(context.pdb_id)
        context.artifacts.put(STRUCTURE_CIF_PATH, tmp_path / "1abc.cif")
        return EnginePhaseResult()

    monkeypatch.setattr(download_module, "run_engine_phase", fake_phase)

    acquired = harness.acquire("1ABC")

    # Without a record nothing is known yet. The structure is downloaded, and
    # its own file decides once parsed.
    assert calls == ["1ABC"]
    assert acquired.result.technique_status is None
    assert not acquired.result.refused()


def test_file_methods_decide_when_the_rcsb_record_names_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _harness(tmp_path)

    def parsed_em(
        _cif_path: Path, _metadata: dict[str, object], **_kwargs: object
    ) -> AtomSiteParseResult:
        return AtomSiteParseResult(
            pd.DataFrame({"pdb_id": ["1ABC"]}),
            experimental_methods=("ELECTRON MICROSCOPY",),
        )

    monkeypatch.setattr(processing_module, "parse_mmcif_atom_rows", parsed_em)
    result = PdbProcessResult(pdb_id="1ABC", download_ok=True)
    decide_technique("1ABC", ReportedExperiment(), harness.build.options).apply_to(
        result
    )
    context = EngineContext("1ABC", pd.DataFrame(), harness.build.options)
    context.artifacts.put(STRUCTURE_CIF_PATH, tmp_path / "1abc.cif")
    acquired = AcquiredEntry(result=result, context=context)

    harness.process(acquired)

    assert result.refused()
    assert result.parse_ok is True
    assert result.experimental_methods == ["ELECTRON MICROSCOPY"]
    entry = harness.build.run_state.manifest()["entries"]["1ABC"]
    assert entry["state"] == "refused"
    assert entry["download_ok"] is True


def test_refused_entries_are_restored_and_summarized_on_finish(tmp_path: Path) -> None:
    options = EngineOptions(values={"include_rabdam": False})
    run_state = open_run(tmp_path, "refused-resume")
    run_state.load_or_initialize(
        ["1ABC", "2EMX"], options.as_settings_dict(), schema_fingerprint="test-schema"
    )
    partial = run_state.directory.partials_directory / "1ABC.csv"
    partial.write_text("pdb_id,atom_name\n1ABC,CA\n", encoding="utf-8")
    run_state.update_entry(
        "1ABC",
        state="done",
        source="download",
        download_ok=True,
        parse_ok=True,
        technique_status="matched",
        experimental_methods=["X-RAY DIFFRACTION"],
        partial_paths={"flat": "partials/1ABC.csv"},
    )
    run_state.update_entry(
        "2EMX",
        state="refused",
        source="download",
        technique_status="refused",
        technique_message="2EMX reports ELECTRON MICROSCOPY, not X-ray diffraction.",
        experimental_methods=["ELECTRON MICROSCOPY"],
        error="2EMX reports ELECTRON MICROSCOPY, not X-ray diffraction.",
    )
    build = DatasetBuild(
        run_state=run_state,
        working_directory=tmp_path,
        pdb_ids=["1ABC", "2EMX"],
        options=options,
        output_filename="refused.csv",
        resume=True,
    )

    result = build.finish()

    # A refused entry does not count as done, so a resume checks it again.
    # That is cheap, because its metadata is cached. This build finishes
    # without recording it, which leaves the run paused.
    assert build.ids_to_process == ["2EMX"]
    states = {pdb_result.pdb_id: pdb_result.state for pdb_result in result.pdb_results}
    assert states == {"1ABC": "done", "2EMX": "refused"}
    refused = next(
        pdb_result for pdb_result in result.pdb_results if pdb_result.pdb_id == "2EMX"
    )
    assert refused.technique_status == "refused"
    assert refused.experimental_methods == ["ELECTRON MICROSCOPY"]
    assert any(
        "Technique intake refused 1 entry" in warning and "2EMX" in warning
        for warning in result.warnings
    )
    with result.run_report_path.open(encoding="utf-8", newline="") as stream:
        rows = {row["pdb_id"]: row for row in csv.DictReader(stream)}
    assert rows["2EMX"]["state"] == "refused"
    assert rows["2EMX"]["technique_status"] == "refused"
    assert rows["2EMX"]["experimental_methods"] == "ELECTRON MICROSCOPY"
    assert rows["2EMX"]["download_success"] == ""
    assert rows["1ABC"]["technique_status"] == "matched"
    summaries = list_run_summaries(tmp_path)
    assert summaries[0]["counts"] == {
        "done": 1,
        "failed": 0,
        "refused": 1,
        "pending": 0,
        "total": 2,
    }
    assert summaries[0]["status"] == "paused"
    summary = result.run_summary_path.read_text(encoding="utf-8")
    assert (
        "| 2EMX | download | refused | refused (ELECTRON MICROSCOPY) | n/a |" in summary
    )


def test_run_report_writes_technique_columns(tmp_path: Path) -> None:
    result = PdbProcessResult(pdb_id="1ABC", download_ok=True, parse_ok=True)
    decide_technique("1ABC", _reported("X-RAY DIFFRACTION"), _XRAY).apply_to(result)
    path = tmp_path / "run_report.csv"

    write_run_report(path, [result], report_fields=[], engine_table_names=[])

    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert rows[0]["technique_status"] == "matched"
    assert rows[0]["technique_message"] == ""
    assert rows[0]["experimental_methods"] == "X-RAY DIFFRACTION"
    assert rows[0]["radiation_sources"] == ""


def _with_method(minimal_cif: Path, method: str) -> Path:
    text = minimal_cif.read_text(encoding="utf-8")
    minimal_cif.write_text(
        text.replace("data_1abc\n", f"data_1abc\n_exptl.method '{method}'\n"),
        encoding="utf-8",
    )
    return minimal_cif


def test_uploaded_file_of_another_technique_is_refused(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    nmr = _with_method(minimal_cif, "SOLUTION NMR")

    with pytest.raises(RuntimeError, match="refused at intake"):
        build_atom_site_dataset_from_mmcif_files(
            [nmr],
            tmp_path / "work",
            run_id="upload-refused",
            engine_options=offline_options,
        )

    manifest = open_run(tmp_path / "work", "upload-refused").manifest()
    entry = manifest["entries"]["1ABC"]
    assert entry["state"] == "refused"
    assert entry["technique_status"] == "refused"
    assert entry["experimental_methods"] == ["SOLUTION NMR"]
    assert manifest["settings"]["technique"] == "x_ray_diffraction"


def test_uploaded_file_matching_the_selected_technique_is_processed(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    nmr = _with_method(minimal_cif, "SOLUTION NMR")

    result = build_atom_site_dataset_from_mmcif_files(
        [nmr],
        tmp_path / "work",
        run_id="upload-nmr",
        engine_options=dataclasses.replace(offline_options, technique="solution_nmr"),
    )

    dataframe = pd.read_csv(result.csv_path)
    assert result.pdb_results[0].technique_status == "matched"
    assert dataframe["experimental_methods"].unique().tolist() == ["SOLUTION NMR"]
    assert dataframe["is_multi_method"].unique().tolist() == [False]
    assert not any("refused" in warning for warning in result.warnings)


def test_uploaded_file_is_filtered_by_its_own_source_category(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    text = minimal_cif.read_text(encoding="utf-8")
    minimal_cif.write_text(
        text.replace(
            "data_1abc\n",
            "data_1abc\n_exptl.method 'X-RAY DIFFRACTION'\n"
            "_diffrn_source.source 'rotating-anode X-ray tube'\n",
        ),
        encoding="utf-8",
    )
    options = dataclasses.replace(offline_options, xray_source="rotating_anode")

    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        tmp_path / "work",
        run_id="upload-source",
        engine_options=options,
    )

    dataframe = pd.read_csv(result.csv_path)
    assert result.pdb_results[0].technique_status == "matched"
    assert result.pdb_results[0].radiation_sources == ["ROTATING ANODE"]
    assert dataframe["radiation_sources"].unique().tolist() == ["ROTATING ANODE"]

    options = dataclasses.replace(options, xray_source="synchrotron")
    with pytest.raises(RuntimeError, match="not Synchrotron"):
        build_atom_site_dataset_from_mmcif_files(
            [minimal_cif],
            tmp_path / "work",
            run_id="upload-source-refused",
            engine_options=options,
        )


def test_uploaded_file_without_a_method_is_admitted_as_unverified(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        tmp_path / "work",
        run_id="upload-unverified",
        engine_options=offline_options,
    )

    dataframe = pd.read_csv(result.csv_path)
    assert result.pdb_results[0].technique_status == "unverified"
    assert bool(dataframe["experimental_methods"].isna().all())
    assert bool(dataframe["is_multi_method"].isna().all())
    assert [w for w in result.warnings if "unverified" in w] == [
        "Technique intake admitted 1 entry unverified (technique "
        "'x_ray_diffraction'), with no experimental method reported: 1ABC."
    ]


def test_incompatible_engine_is_rejected_before_any_upload_work(
    minimal_cif: Path, tmp_path: Path
) -> None:
    with pytest.raises(
        ValueError, match="RABDAM \\(include_rabdam\\) is not available"
    ):
        build_atom_site_dataset_from_mmcif_files(
            [minimal_cif],
            tmp_path / "work",
            run_id="incompatible",
            engine_options=EngineOptions(
                technique="electron_microscopy", values={"include_rabdam": True}
            ),
        )

    assert not (tmp_path / "work" / "runs" / "incompatible").exists()
