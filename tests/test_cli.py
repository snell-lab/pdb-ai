from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any

import pytest

from pdbai import cli
from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import EngineOptions
from pdbai.results import DatasetBuildResult, NoDatasetError, PdbProcessResult
from pdbai.runs import RunLockedError, RunNotResumableError


@pytest.mark.parametrize(
    "arguments",
    [
        ["1ABC", "--resume"],
        ["1ABC", "--workers", "0"],
        ["1ABC", "--column-filter-mode", "include"],
        ["1ABC", "--column-filter-list", "pdb_id"],
        ["1ABC", "--column-filter-mode", "keep", "--column-filter-list", "pdb_id"],
        ["1ABC", "--sasa-points", "0"],
    ],
)
def test_cli_rejects_incomplete_or_invalid_arguments(
    arguments: list[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(arguments)

    assert caught.value.code == 2


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--sasa-points", "500"], "--sasa-points requires --include-sasa"),
        (
            ["--sasa-include-experimental-additives"],
            "--sasa-include-experimental-additives requires --include-sasa",
        ),
        (["--rabdam-source", "rcsb"], "--rabdam-source requires --include-rabdam"),
    ],
)
def test_cli_refuses_a_setting_of_an_engine_left_off(
    arguments: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(["1ABC", *arguments])

    assert caught.value.code == 2
    assert capsys.readouterr().err.endswith(f"error: {message}\n")


def test_cli_accepts_a_setting_of_an_engine_a_profile_turns_on() -> None:
    _, settings = cli.parse_args(
        ["1ABC", "--profile", "ml_minimal", "--sasa-accuracy", "fast"]
    )

    assert settings.options.choice("sasa_accuracy_mode") == "fast"


def test_cli_accepts_complete_resume_and_filter_arguments() -> None:
    args, _ = cli.parse_args(
        [
            "1ABC",
            "--resume",
            "--run-id",
            "existing-run",
            "--workers",
            "2",
            "--column-filter-mode",
            "include",
            "--column-filter-list",
            "pdb_id",
        ]
    )

    assert args.resume is True
    assert args.run_id == "existing-run"
    assert args.workers == 2


def test_cli_reports_a_malformed_pdb_id_as_a_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(["1ABC", "NONE"])

    assert caught.value.code == 2
    assert "Invalid PDB ID 'NONE'" in capsys.readouterr().err


def test_cli_normalizes_the_pdb_ids_it_is_given() -> None:
    args, _ = cli.parse_args([" 1abc", "4hhb"])

    assert args.pdb_ids == ["1ABC", "4HHB"]


def test_cli_column_filter_mode_ignores_case() -> None:
    _, settings = cli.parse_args(
        ["1ABC", "--column-filter-mode", "Include", "--column-filter-list", "pdb_id"]
    )

    assert settings.column_filter_mode == "include"
    assert settings.column_filter_list == ("pdb_id",)


def test_cli_defaults_to_x_ray_with_every_engine_off() -> None:
    args, settings = cli.parse_args(["1ABC"])
    options = settings.options

    assert args.technique == "x_ray_diffraction"
    for option in get_engine_registry().option_specs():
        if option.kind == "flag" and option.refines is None:
            assert options.flag(option.name) is False, option.name


def test_cli_rabdam_is_opt_in_and_needs_x_ray(
    capsys: pytest.CaptureFixture[str],
) -> None:
    _, enabled = cli.parse_args(["1ABC", "--include-rabdam"])
    _, default = cli.parse_args(["1ABC", "--technique", "solution_nmr"])

    assert enabled.options.flag("include_rabdam") is True
    assert default.options.flag("include_rabdam") is False
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(["1ABC", "--no-rabdam"])
    assert caught.value.code == 2
    for technique in ("electron_microscopy", "solution_nmr"):
        with pytest.raises(SystemExit) as caught:
            cli.parse_args(["1ABC", "--technique", technique, "--include-rabdam"])
        assert caught.value.code == 2
        assert "RABDAM (include_rabdam) is not available" in capsys.readouterr().err


def test_cli_xray_source_defaults_to_all_and_needs_the_x_ray_technique() -> None:
    args, _ = cli.parse_args(["1ABC", "--xray-source", "free_electron_laser"])

    assert args.xray_source == "free_electron_laser"
    assert cli.parse_args(["1ABC"])[0].xray_source == "all"
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(
            [
                "1ABC",
                "--technique",
                "electron_microscopy",
                "--xray-source",
                "synchrotron",
            ]
        )
    assert caught.value.code == 2
    with pytest.raises(SystemExit):
        cli.parse_args(["1ABC", "--xray-source", "xfel"])


def test_cli_refuses_an_engine_the_technique_rules_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    called = False

    def fake_build(**_kwargs: object) -> DatasetBuildResult:
        nonlocal called
        called = True
        raise AssertionError("build must not start")

    monkeypatch.setattr(cli, "build_atom_site_dataset", fake_build)

    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "1ABC",
                "--technique",
                "solution_nmr",
                "--include-pdb-redo",
                "--workdir",
                str(tmp_path),
            ]
        )

    captured = capsys.readouterr()
    assert caught.value.code == 2
    assert called is False
    assert captured.out == ""
    assert "usage:" in captured.err
    assert (
        "PDB-Redo comparison (include_pdb_redo) is not available for Solution "
        "nuclear magnetic resonance"
    ) in captured.err


def test_cli_lists_refused_entries_without_failing_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    refused = PdbProcessResult(
        pdb_id="8EMX",
        state="refused",
        technique_status="refused",
        technique_message="8EMX reports ELECTRON MICROSCOPY, not X-ray diffraction.",
        experimental_methods=["ELECTRON MICROSCOPY"],
    )
    done = PdbProcessResult(
        pdb_id="1ABC", download_ok=True, parse_ok=True, state="done"
    )
    seen: dict[str, object] = {}

    def fake_build(**kwargs: object) -> DatasetBuildResult:
        seen.update(kwargs)
        return DatasetBuildResult(
            csv_path=tmp_path / "dataset.csv",
            run_report_path=tmp_path / "run_report.csv",
            run_summary_path=tmp_path / "run_summary.md",
            atom_count=1,
            column_count=1,
            pdb_ids=["1ABC", "8EMX"],
            preview_rows=[],
            warnings=[],
            pdb_results=[done, refused],
        )

    monkeypatch.setattr(cli, "build_atom_site_dataset", fake_build)

    code = cli.main(["1ABC", "8EMX", "--workdir", str(tmp_path)])

    out = capsys.readouterr().out
    assert code == 0
    assert "Technique: X-ray diffraction ('x_ray_diffraction')" in out
    assert "1 refused at technique intake" in out
    assert "  - 8EMX: ELECTRON MICROSCOPY" in out
    assert cast_options(seen["engine_options"]).technique == "x_ray_diffraction"
    assert cast_options(seen["engine_options"]).xray_source == "all"


def cast_options(value: object) -> EngineOptions:
    assert isinstance(value, EngineOptions)
    return value


def test_cli_returns_partial_failure_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    failed = PdbProcessResult(
        pdb_id="1ABC",
        state="failed",
        download_error="network unavailable",
    )

    def fake_build(**_kwargs: object) -> DatasetBuildResult:
        return DatasetBuildResult(
            csv_path=tmp_path / "dataset.csv",
            run_report_path=tmp_path / "run_report.csv",
            run_summary_path=tmp_path / "run_summary.md",
            atom_count=0,
            column_count=0,
            pdb_ids=["1ABC"],
            preview_rows=[],
            warnings=[],
            pdb_results=[failed],
        )

    monkeypatch.setattr(cli, "build_atom_site_dataset", fake_build)

    assert cli.main(["1ABC"]) == 2


def test_cli_reports_a_run_with_no_done_entry_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    error = NoDatasetError("No atom-site data could be produced for any entry.")

    def fake_build(**_kwargs: object) -> DatasetBuildResult:
        raise error

    monkeypatch.setattr(cli, "build_atom_site_dataset", fake_build)
    prog = cli._build_parser().prog  # pyright: ignore[reportPrivateUsage]

    assert cli.main(["1ABC"]) == 2
    assert capsys.readouterr().err.endswith(f"{prog}: error: {error}\n")


def _selected(*pdb_ids: str, total: int | None = None, limit: int | None = None) -> Any:
    from pdbai.selection import Selection, SelectionResult

    return SelectionResult(
        filters=Selection("x_ray_diffraction", "free_electron_laser", limit=limit),
        pdb_ids=tuple(pdb_ids),
        total_count=total if total is not None else len(pdb_ids),
        service="rcsb_search",
        service_url="https://search.example/query",
        query={"query": {}},
        retrieved_at=1_700_000_000.0,
        # Here a total above the identifiers given stands for matches that the
        # limit dropped.
        truncated=total is not None and total > len(pdb_ids),
    )


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        ["1ABC", "--select"],
        ["--list-only"],
        ["1ABC", "--list-only"],
        ["1ABC", "--limit", "5"],
        ["--select", "--limit", "0"],
    ],
)
def test_cli_selection_flags_need_each_other(arguments: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(arguments)

    assert caught.value.code == 2


def test_cli_select_parses_without_positional_ids() -> None:
    args, _ = cli.parse_args(
        ["--select", "--xray-source", "free_electron_laser", "--limit", "10"]
    )

    assert args.pdb_ids == []
    assert args.select is True
    assert args.limit == 10
    assert args.list_only is False
    assert (args.min_temperature_K, args.max_temperature_K, args.max_resolution_A) == (
        None,
        None,
        None,
    )


def test_cli_has_a_flag_for_every_declared_search_filter() -> None:
    from pdbai.selection_filters import SEARCH_FILTERS

    help_text = cli._build_parser().format_help()  # pyright: ignore[reportPrivateUsage]

    for search_filter in SEARCH_FILTERS:
        args, _ = cli.parse_args(["--select", search_filter.cli_flag, "2.5"])
        assert getattr(args, search_filter.id) == 2.5
        assert f"{search_filter.cli_flag} " in help_text


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        (["1ABC", "--min-temperature", "273"], "--min-temperature requires --select"),
        (["1ABC", "--max-resolution", "2"], "--max-resolution requires --select"),
        (
            ["--select", "--max-temperature", "0"],
            "argument --max-temperature: --max-temperature must be a positive "
            "number; got '0'.",
        ),
        (
            ["--select", "--max-resolution", "2e0"],
            "argument --max-resolution: --max-resolution must be a positive "
            "number; got '2e0'.",
        ),
        (
            ["--select", "--min-temperature", "300", "--max-temperature", "100"],
            "The minimum temperature (300 K) is above the maximum (100 K), so no "
            "entry can match.",
        ),
    ],
)
def test_cli_refuses_a_range_filter_it_cannot_search_with(
    arguments: list[str], error: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as caught:
        cli.parse_args(arguments)

    assert caught.value.code == 2
    assert capsys.readouterr().err.endswith(f"error: {error}\n")


def test_cli_reads_the_uniprot_accessions_from_a_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pdbai.selection import Selection

    listed = tmp_path / "accessions.txt"
    listed.write_text("P00698\np0dtd1, P00698-2;P69905\n", encoding="utf-8")
    asked: list[Selection] = []

    def fake_search(selection: Selection) -> Any:
        asked.append(selection)
        return _selected("11YO")

    monkeypatch.setattr(cli, "search_archive", fake_search)

    code = cli.main(["--select", "--uniprot-accessions", str(listed), "--list-only"])

    assert code == 0
    assert asked == [
        Selection(
            "x_ray_diffraction", uniprot_accessions=("P00698", "P0DTD1", "P69905")
        )
    ]
    # Without --select the flag has nothing to narrow.
    with pytest.raises(SystemExit):
        cli.parse_args(["1ABC", "--uniprot-accessions", str(listed)])


@pytest.mark.parametrize(
    ("content", "error"),
    [
        (None, "--uniprot-accessions cannot read"),
        ("P00698 lysozyme\n", "must hold UniProt accessions; got 'lysozyme'."),
        ("\n", "names no UniProt accession"),
    ],
)
def test_cli_refuses_an_accession_file_it_cannot_search_with(
    tmp_path: Path, content: str | None, error: str, capsys: pytest.CaptureFixture[str]
) -> None:
    listed = tmp_path / "accessions.txt"
    if content is not None:
        listed.write_text(content, encoding="utf-8")

    with pytest.raises(SystemExit) as caught:
        cli.parse_args(["--select", "--uniprot-accessions", str(listed)])

    assert caught.value.code == 2
    assert error in capsys.readouterr().err


def test_cli_searches_with_the_range_filters_it_was_given(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pdbai.selection import Selection

    asked: list[Selection] = []

    def fake_search(selection: Selection) -> Any:
        asked.append(selection)
        return _selected("11YO", "13CH")

    def fake_build(**_kwargs: object) -> DatasetBuildResult:
        raise AssertionError("list-only must not build")

    monkeypatch.setattr(cli, "search_archive", fake_search)
    monkeypatch.setattr(cli, "build_atom_site_dataset", fake_build)

    code = cli.main(
        [
            "--select",
            "--xray-source",
            "synchrotron",
            "--min-temperature",
            "273",
            "--max-resolution",
            "2.5",
            "--list-only",
        ]
    )

    assert code == 0
    assert asked == [
        Selection(
            "x_ray_diffraction",
            "synchrotron",
            bounds={"min_temperature_K": 273.0, "max_resolution_A": 2.5},
        )
    ]
    assert (
        "Searching the PDB for: X-ray diffraction (XRD), Synchrotron, temperature "
        "at least 273 K, resolution at most 2.5 Å" in capsys.readouterr().err
    )


def test_cli_list_only_prints_ids_on_stdout_and_the_rest_on_stderr(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pdbai.selection import Selection

    asked: list[Selection] = []

    def fake_search(selection: Selection) -> Any:
        asked.append(selection)
        return _selected("11YO", "13CH", total=1041, limit=2)

    def fake_build(**_kwargs: object) -> DatasetBuildResult:
        raise AssertionError("list-only must not build")

    monkeypatch.setattr(cli, "search_archive", fake_search)
    monkeypatch.setattr(cli, "build_atom_site_dataset", fake_build)

    code = cli.main(
        [
            "--select",
            "--xray-source",
            "free_electron_laser",
            "--limit",
            "2",
            "--list-only",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert asked == [Selection("x_ray_diffraction", "free_electron_laser", limit=2)]
    assert captured.out == "11YO\n13CH\n"
    assert "Technique: X-ray diffraction" in captured.err
    assert "PDB search matched 1041 entries" in captured.err
    assert "keeping the first 2 (--limit 2)" in captured.err


def test_cli_select_feeds_the_cohort_and_its_record_into_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pdbai.selection import SelectionResult

    received: dict[str, object] = {}

    def fake_build(**kwargs: object) -> DatasetBuildResult:
        received.update(kwargs)
        return DatasetBuildResult(
            csv_path=tmp_path / "dataset.csv",
            run_report_path=tmp_path / "run_report.csv",
            run_summary_path=tmp_path / "run_summary.md",
            atom_count=0,
            column_count=0,
            pdb_ids=list(kwargs["pdb_ids"]),  # type: ignore[call-overload]
            preview_rows=[],
            warnings=[],
        )

    def two_entries(_selection: object) -> Any:
        return _selected("11YO", "13CH")

    monkeypatch.setattr(cli, "search_archive", two_entries)
    monkeypatch.setattr(cli, "build_atom_site_dataset", fake_build)

    code = cli.main(
        [
            "--select",
            "--xray-source",
            "free_electron_laser",
            "--workdir",
            str(tmp_path),
        ]
    )

    out = capsys.readouterr().out
    assert code == 0
    assert received["pdb_ids"] == ["11YO", "13CH"]
    search_result = received["search_result"]
    assert isinstance(search_result, SelectionResult)
    record = search_result.to_dict()
    assert record["pdb_ids"] == ["11YO", "13CH"]
    assert record["service"] == "rcsb_search"
    assert record["filters"] == {
        "technique": "x_ray_diffraction",
        "xray_source": "free_electron_laser",
        "limit": None,
        "min_temperature_K": None,
        "max_temperature_K": None,
        "max_resolution_A": None,
        "uniprot_accessions": None,
    }
    assert (
        "Searching the PDB for: X-ray diffraction (XRD), X-ray free-electron laser (XFEL)"
        in out
    )
    assert "PDB search matched 2 entries (rcsb_search at 2023-11-14T22:13:20Z)." in out


def test_cli_select_stops_when_the_search_fails_or_matches_nothing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from pdbai.selection import ArchiveSearchError

    def fake_build(**_kwargs: object) -> DatasetBuildResult:
        raise AssertionError("build must not start")

    monkeypatch.setattr(cli, "build_atom_site_dataset", fake_build)

    def failing(_selection: object) -> Any:
        raise ArchiveSearchError("PDB search failed: down")

    monkeypatch.setattr(cli, "search_archive", failing)
    prog = cli._build_parser().prog  # pyright: ignore[reportPrivateUsage]
    assert cli.main(["--select"]) == 2
    assert capsys.readouterr().err.endswith(f"{prog}: error: PDB search failed: down\n")

    def no_entries(_selection: object) -> Any:
        return _selected()

    monkeypatch.setattr(cli, "search_archive", no_entries)
    assert cli.main(["--select"]) == 2
    assert capsys.readouterr().err.endswith(
        f"{prog}: error: no PDB entries matched the selection\n"
    )


def test_python_dash_m_pdbai_runs_the_cli(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["pdbai", "--help"])

    with pytest.raises(SystemExit) as caught:
        runpy.run_module("pdbai", run_name="__main__")

    assert caught.value.code == 0
    assert "usage:" in capsys.readouterr().out


def test_cli_refuses_a_run_id_that_does_not_fit_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fake_build(**_kwargs: object) -> DatasetBuildResult:
        raise AssertionError("build must not start")

    monkeypatch.setattr(cli, "build_atom_site_dataset", fake_build)
    workdir = ["--workdir", str(tmp_path)]

    # A mistyped ID would otherwise start a new run from scratch.
    assert cli.main(["1ABC", *workdir, "--run-id", "missing", "--resume"]) == 2
    assert "run 'missing' does not exist" in capsys.readouterr().err
    assert not (tmp_path / "runs" / "missing").exists()

    (tmp_path / "runs" / "finished").mkdir(parents=True)
    assert cli.main(["1ABC", *workdir, "--run-id", "finished"]) == 2
    assert "run 'finished' already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    "error",
    [
        RunNotResumableError("Run 'old' was created with different settings."),
        RunLockedError("Run 'old' is held by another worker."),
    ],
)
def test_cli_reports_a_run_it_cannot_build_as_a_usage_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    def fake_build(**_kwargs: object) -> DatasetBuildResult:
        raise error

    monkeypatch.setattr(cli, "build_atom_site_dataset", fake_build)
    (tmp_path / "runs" / "old").mkdir(parents=True)
    prog = cli._build_parser().prog  # pyright: ignore[reportPrivateUsage]

    arguments = ["1ABC", "--workdir", str(tmp_path), "--run-id", "old", "--resume"]
    assert cli.main(arguments) == 2
    assert capsys.readouterr().err.endswith(f"{prog}: error: {error}\n")
