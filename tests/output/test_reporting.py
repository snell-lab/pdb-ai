from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import pandas as pd
import pytest

from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import (
    HOST_REPORT_COLUMNS,
    EngineOptions,
    OptionSpec,
    ReportField,
    TableColumnSpec,
    TableLink,
    TableSpec,
)
from pdbai.reporting import run_report, write_run_report, write_run_summary
from pdbai.reporting.column_statistics import MAX_DISTINCT_COUNT
from pdbai.results import ColumnFilterSummary, PdbProcessResult
from pdbai.schema import describe_column


def test_run_report_distinguishes_disabled_engines_from_failures(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "run_report.csv"
    disabled = PdbProcessResult(
        pdb_id="1ABC",
        state="done",
        download_ok=True,
        parse_ok=True,
    )
    failed = PdbProcessResult(
        pdb_id="2XYZ",
        download_ok=True,
        parse_ok=True,
        state="failed",
        output_error="Annotation failed for 2XYZ: engine exploded",
        metadata_message="metadata incomplete",
        engine_report={
            "validation_status": "parse_failed",
            "validation_message": "validation XML malformed",
            "sasa_status": "degraded",
            "sasa_message": "unsupported atom",
            "pdb_redo_alignment_match_rate": 0.75,
            "pdb_redo_alignment_rcsb_ambiguous_atom_count": 1,
            "pdb_redo_alignment_pdb_redo_ambiguous_atom_count": 2,
            "rabdam_status": "partial",
            "rabdam_delivered_pdb_redo": False,
            "rabdam_delivered_rcsb": True,
            "rabdam_alignment_match_rate": 0.5,
            "rabdam_alignment_rcsb_ambiguous_atom_count": 3,
            "rabdam_alignment_pdb_redo_ambiguous_atom_count": 4,
            "rabdam_message": "rabdam-custom Bnet unavailable for 2XYZ: zero variance",
        },
    )

    options = EngineOptions(
        values={
            "include_pdb_redo": True,
            "include_rabdam": True,
            "rabdam_source": "both",
            "include_sasa": True,
            "include_validation": True,
        }
    )
    write_run_report(
        report_path,
        [disabled, failed],
        report_fields=get_engine_registry().report_fields(options),
        engine_table_names=[],
    )

    with report_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["pdb_id"] == "1ABC"
    assert rows[0]["state"] == "done"
    assert rows[0]["success"] == "True"
    assert rows[0]["download_success"] == "True"
    assert rows[0]["parse_success"] == "True"
    # An engine that did not report for a structure leaves its fields empty.
    assert rows[0]["rabdam_status"] == ""
    assert rows[1]["pdb_id"] == "2XYZ"
    assert rows[1]["state"] == "failed"
    assert rows[1]["success"] == "False"
    assert rows[1]["parse_success"] == "True"
    assert rows[1]["parse_error"] == ""
    assert rows[1]["output_error"] == "Annotation failed for 2XYZ: engine exploded"
    assert rows[1]["metadata_message"] == "metadata incomplete"
    assert rows[1]["validation_status"] == "parse_failed"
    assert rows[1]["validation_message"] == "validation XML malformed"
    assert rows[1]["sasa_message"] == "unsupported atom"
    assert rows[1]["pdb_redo_alignment_match_rate"] == "0.75"
    assert rows[1]["pdb_redo_alignment_rcsb_ambiguous_atom_count"] == "1"
    assert rows[1]["pdb_redo_alignment_pdb_redo_ambiguous_atom_count"] == "2"
    assert rows[1]["rabdam_status"] == "partial"
    assert rows[1]["rabdam_delivered_pdb_redo"] == "False"
    assert rows[1]["rabdam_delivered_rcsb"] == "True"
    assert rows[1]["rabdam_alignment_match_rate"] == "0.5"
    assert rows[1]["rabdam_alignment_rcsb_ambiguous_atom_count"] == "3"
    assert rows[1]["rabdam_alignment_pdb_redo_ambiguous_atom_count"] == "4"
    assert rows[1]["rabdam_message"] == (
        "rabdam-custom Bnet unavailable for 2XYZ: zero variance"
    )


def _read_report(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def test_run_report_orders_host_columns_then_fields_then_row_counts(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "run_report.csv"
    done = PdbProcessResult(
        pdb_id="1ABC",
        state="done",
        download_ok=True,
        parse_ok=True,
        engine_report={"stub_status": "ok"},
        engine_table_row_counts={"stub_sites": 2, "stub_bonds": 0},
    )
    pending = PdbProcessResult(pdb_id="2XYZ")

    write_run_report(
        report_path,
        [done, pending],
        report_fields=[ReportField("stub_status", "Outcome.")],
        engine_table_names=["stub_sites", "stub_bonds"],
    )

    columns, rows = _read_report(report_path)
    assert columns == [
        *HOST_REPORT_COLUMNS,
        "stub_status",
        "stub_sites_row_count",
        "stub_bonds_row_count",
    ]
    assert [row["pdb_id"] for row in rows] == ["1ABC", "2XYZ"]
    assert rows[0]["stub_status"] == "ok"
    assert rows[0]["stub_sites_row_count"] == "2"
    assert rows[0]["stub_bonds_row_count"] == "0"
    # An unfinished entry has no row counts and no engine report.
    assert rows[1]["stub_status"] == ""
    assert rows[1]["stub_sites_row_count"] == ""
    assert rows[1]["stub_bonds_row_count"] == ""


def test_run_report_fills_every_host_column_of_a_fully_recorded_entry(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "run_report.csv"
    result = PdbProcessResult(
        pdb_id="1ABC",
        state="failed",
        technique_status="matched",
        technique_message="Reports X-ray diffraction.",
        experimental_methods=["X-RAY DIFFRACTION"],
        radiation_sources=["SYNCHROTRON"],
        download_ok=True,
        download_error="Retried once.",
        parse_ok=True,
        parse_error="Ignored an unknown category.",
        output_error="Disk full.",
        metadata_ok=True,
        metadata_message="Fetched.",
    )

    write_run_report(report_path, [result], report_fields=[], engine_table_names=[])

    _, rows = _read_report(report_path)
    assert [column for column, cell in rows[0].items() if cell == ""] == []


def test_run_report_leaves_the_steps_an_entry_never_reached_empty(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "run_report.csv"
    refused_on_record = PdbProcessResult(
        pdb_id="1ABC",
        state="refused",
        technique_status="refused",
        metadata_ok=True,
    )
    # With no RCSB record, the parsed file decides. So this entry was
    # downloaded and parsed before intake refused it.
    refused_on_file = PdbProcessResult(
        pdb_id="2XYZ",
        state="refused",
        technique_status="refused",
        download_ok=True,
        parse_ok=True,
        metadata_ok=False,
        metadata_message="RCSB record unavailable",
    )
    upload = PdbProcessResult(
        pdb_id="MODEL", source="upload", state="done", download_ok=True, parse_ok=True
    )
    pending = PdbProcessResult.from_manifest(
        "3DEF", {"state": "pending"}, default_source="download"
    )
    download_failed = PdbProcessResult(
        pdb_id="4GHI", state="failed", download_error="network unavailable"
    )

    write_run_report(
        report_path,
        [refused_on_record, refused_on_file, upload, pending, download_failed],
        report_fields=[],
        engine_table_names=[],
    )

    _, rows = _read_report(report_path)
    steps = [
        (
            row["metadata_success"],
            row["download_success"],
            row["parse_success"],
        )
        for row in rows
    ]
    assert steps == [
        ("True", "", ""),
        ("False", "True", "True"),
        ("", "True", "True"),
        ("", "", ""),
        # An acquisition that raised leaves the metadata step unknown, since
        # the error may have come before the record was sought.
        ("", "False", ""),
    ]


def test_run_report_of_a_run_without_results_has_only_its_header(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "run_report.csv"

    write_run_report(
        report_path, [], report_fields=[], engine_table_names=["stub_sites"]
    )

    columns, rows = _read_report(report_path)
    assert columns == [*HOST_REPORT_COLUMNS, "stub_sites_row_count"]
    assert rows == []


def test_run_report_replaces_the_previous_report_only_once_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path = tmp_path / "run_report.csv"
    report_path.write_text("previous report\n", encoding="utf-8")

    def fail(result: PdbProcessResult) -> dict[str, object]:
        raise RuntimeError(f"cannot report {result.pdb_id}")

    with monkeypatch.context() as patch:
        patch.setattr(run_report, "_host_cells", fail)
        with pytest.raises(RuntimeError, match="cannot report 1ABC"):
            write_run_report(
                report_path,
                [PdbProcessResult(pdb_id="1ABC")],
                report_fields=[],
                engine_table_names=[],
            )

    assert report_path.read_text(encoding="utf-8") == "previous report\n"
    assert list(tmp_path.iterdir()) == [report_path]

    write_run_report(
        report_path,
        [PdbProcessResult(pdb_id="1ABC")],
        report_fields=[],
        engine_table_names=[],
    )

    _, rows = _read_report(report_path)
    assert [row["pdb_id"] for row in rows] == ["1ABC"]
    assert list(tmp_path.iterdir()) == [report_path]


def _write_summary(
    tmp_path: Path,
    *,
    options: EngineOptions | None = None,
    option_specs: Sequence[OptionSpec] = (),
    pdb_results: Sequence[PdbProcessResult] = (),
    report_fields: Sequence[ReportField] = (),
    engine_table_specs: Sequence[TableSpec] = (),
    output_paths_by_table: Mapping[str, Path] | None = None,
    columns_by_table: Mapping[str, Sequence[str]] | None = None,
    row_counts_by_table: Mapping[str, int] | None = None,
    run_manifest: Mapping[str, object] | None = None,
    column_filter: ColumnFilterSummary | None = None,
    warnings: Sequence[str] = (),
) -> str:
    summary_path = tmp_path / "run_summary.md"
    write_run_summary(
        summary_path,
        options=options or EngineOptions(),
        option_specs=option_specs,
        pdb_results=pdb_results,
        report_fields=report_fields,
        engine_table_specs=engine_table_specs,
        output_paths_by_table=output_paths_by_table or {},
        columns_by_table=columns_by_table or {},
        row_counts_by_table=row_counts_by_table or {},
        run_manifest=run_manifest or {},
        column_filter=column_filter,
        warnings=warnings,
    )
    return summary_path.read_text(encoding="utf-8")


def _cells(line: str) -> list[str]:
    """Return the cells of one table row, split at each unescaped "|"."""
    return [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]


def _table_rows(summary: str, header: str) -> list[list[str]]:
    """Return the rows of each table whose header starts with ``header``."""
    lines = summary.splitlines()
    rows: list[list[str]] = []
    for index, line in enumerate(lines):
        if not line.startswith(header):
            continue
        # The line after the header is the separator row, so the rows start
        # two lines down.
        for row_line in lines[index + 2 :]:
            if not row_line.startswith("|"):
                break
            rows.append(_cells(row_line))
    return rows


def test_markdown_summary_contains_configuration_status_and_data_dictionary(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame({"pdb_id": ["1ABC"], "x": [1.25]}).to_csv(dataset_path, index=False)
    result = PdbProcessResult(
        pdb_id="1ABC",
        source="upload",
        state="done",
        download_ok=True,
        parse_ok=True,
        engine_report={
            "validation_status": "cached",
            "pdb_redo_alignment_match_rate": 0.75,
            "rabdam_status": "partial",
            "rabdam_delivered_pdb_redo": False,
            "rabdam_delivered_rcsb": True,
            "rabdam_alignment_match_rate": 0.5,
        },
    )
    registry = get_engine_registry()
    summary_fields = registry.report_fields(
        EngineOptions(
            values={
                "include_pdb_redo": True,
                "include_rabdam": True,
                "rabdam_source": "both",
                "include_validation": True,
            }
        )
    )

    summary = _write_summary(
        tmp_path,
        options=EngineOptions(values={"include_rabdam": False}),
        option_specs=registry.option_specs(),
        pdb_results=[result],
        output_paths_by_table={"flat": dataset_path},
        columns_by_table={"flat": ["pdb_id", "x"]},
        report_fields=summary_fields,
        run_manifest={
            "provenance": {
                "execution_sessions": [
                    {
                        "captured_at": "2026-08-14T12:00:00Z",
                        "code": {
                            "repository": "https://example.test/pdb-ai",
                            "git_commit": "abc123",
                            "git_branch": "main",
                            "git_dirty": False,
                        },
                        "runtime": {
                            "python_implementation": "CPython",
                            "python_version": "3.14.0",
                            "operating_system": "Linux",
                            "os_release": "test",
                            "machine": "x86_64",
                        },
                        "dependencies": [
                            {
                                "name": "pandas",
                                "version": "3.0.0",
                                "direct": True,
                                "status": "installed",
                            }
                        ],
                    }
                ],
                "outputs": [
                    {
                        "role": "dataset_flat",
                        "path": "atoms.csv",
                        "sha256": "def456",
                        "size_bytes": 42,
                        "observed_at": "2026-08-14T12:01:00Z",
                    }
                ],
            },
            "entries": {
                "1ABC": {
                    "inputs": [
                        {
                            "role": "uploaded_mmcif",
                            "path": "1abc.cif",
                            "sha256": "abc456",
                            "size_bytes": 100,
                            "observed_at": "2026-08-14T12:00:30Z",
                        }
                    ]
                }
            },
        },
    )

    # The per-structure columns follow the report fields given. The options
    # leave RABDAM off, and its columns still appear.
    assert "| Validation |" in summary
    assert "| cached |" in summary
    assert "| RABDAM | RABDAM PDB-Redo | RABDAM RCSB | RABDAM match rate |" in summary
    assert "| partial | False | True | 50.0% |" in summary
    assert "| PDB-Redo match rate |" in summary
    assert "| 75.0% |" in summary
    assert "# pdbai Run Summary" in summary
    assert "## Run Configuration" in summary
    assert "- **include_rabdam** (`False`)" in summary
    assert "## Reproducibility Log" in summary
    assert "| Git commit | `abc123` |" in summary
    assert "| Python | CPython 3.14.0 |" in summary
    assert "| pandas | 3.0.0 | direct | installed |" in summary
    assert (
        "| 1ABC | uploaded_mmcif | n/a | 1abc.cif | `abc456` | 100 | n/a | n/a "
        "| 2026-08-14T12:00:30Z | n/a |"
    ) in summary
    assert "| dataset_flat | atoms.csv | `def456` | 42 | 2026-08-14T12:01:00Z |" in (
        summary
    )
    assert "| 1ABC | upload | done | n/a | True | True |" in summary
    assert "- **technique** (`x_ray_diffraction`)" in summary
    assert (
        f'Distinct values reads "high-cardinality" past {MAX_DISTINCT_COUNT} '
        "distinct values"
    ) in summary
    assert "### Table: `flat`" in summary
    assert (
        "| Column | Description | Type | % Null | Minimum | Maximum | Mean "
        "| Standard deviation | Distinct values |"
    ) in summary
    assert (
        f"| pdb_id | {describe_column('pdb_id')} | text | 0 | 4 | 4 | n/a | n/a | 1 |"
    ) in summary
    # A lone value has no spread. The population standard deviation says 0,
    # where the sample one would be undefined.
    assert (
        "| x | Atom Cartesian x coordinate in Å from _atom_site.Cartn_x. "
        "| numeric | 0 | 1.25 | 1.25 | 1.25 | 0 | 1 |"
    ) in summary


def test_run_summary_says_how_technique_intake_decided(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path)

    technique_line = next(
        line for line in summary.splitlines() if line.startswith("- **technique**")
    )
    assert "reports no method was admitted as unverified" in technique_line
    assert "otherwise the structure file decided once parsed" in technique_line
    assert "before download" not in technique_line
    # Without a source filter the summary does not mention one.
    assert "**xray_source**" not in summary


def test_run_summary_describes_a_chosen_xray_source(tmp_path: Path) -> None:
    summary = _write_summary(tmp_path, options=EngineOptions(xray_source="synchrotron"))

    assert (
        "- **xray_source** (`synchrotron`): within X-ray diffraction, only entries "
        "reporting a Synchrotron source (`SYNCHROTRON`) were processed"
    ) in summary


def test_run_summary_nests_each_refinement_beneath_its_flag_while_it_is_on(
    tmp_path: Path,
) -> None:
    option_specs = [
        OptionSpec("include_stub", label="Stub", help="Run the stub."),
        OptionSpec(
            "stub_detail", label="Detail", help="Add detail.", refines="include_stub"
        ),
        OptionSpec(
            "stub_depth",
            label="Depth",
            help="How deep.",
            kind="count",
            default=1,
            refines="stub_detail",
        ),
    ]

    enabled = _write_summary(
        tmp_path,
        option_specs=option_specs,
        options=EngineOptions(
            values={"include_stub": True, "stub_detail": True, "stub_depth": 3}
        ),
    )
    disabled = _write_summary(
        tmp_path,
        option_specs=option_specs,
        options=EngineOptions(
            values={"include_stub": False, "stub_detail": True, "stub_depth": 3}
        ),
    )

    assert "\n- **include_stub** (`True`)\n" in enabled
    assert "\n  - **stub_detail** (`True`)\n" in enabled
    assert "\n    - **stub_depth** (`3`)\n" in enabled
    assert "- **include_stub** (`False`)" in disabled
    assert "stub_detail" not in disabled
    assert "stub_depth" not in disabled


def test_run_summary_says_why_a_session_has_no_dependency_inventory(
    tmp_path: Path,
) -> None:
    summary = _write_summary(
        tmp_path,
        run_manifest={
            "provenance": {
                "execution_sessions": [
                    {
                        "dependencies": None,
                        "dependencies_error": "Invalid requirement in metadata",
                    }
                ]
            }
        },
    )

    assert "Not recorded: Invalid requirement in metadata" in summary
    assert "| Package | Version | Relationship | Status |" not in summary


def test_run_summary_fills_tables_with_nothing_recorded_with_one_missing_row(
    tmp_path: Path,
) -> None:
    summary = _write_summary(
        tmp_path,
        run_manifest={
            "provenance": {"execution_sessions": [{"dependencies": list[object]()}]}
        },
    )

    assert _table_rows(summary, "| Package |") == [["n/a"] * 4]
    assert _table_rows(summary, "| Structure | Role |") == [["n/a"] * 10]
    assert _table_rows(summary, "| Role | File |") == [["n/a"] * 5]
    assert _table_rows(summary, "| PDB ID |") == [["n/a"] * 6]
    assert "- Run ID: n/a" in summary
    assert "| Python | n/a |" in summary


def test_run_summary_keeps_every_table_row_on_one_line(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    pd.DataFrame({"pdb_id": ["1ABC"], "raw|field": ["a|b"]}).to_csv(
        dataset_path, index=False
    )

    summary = _write_summary(
        tmp_path,
        pdb_results=[
            PdbProcessResult(
                pdb_id="1ABC",
                state="done",
                download_ok=True,
                parse_ok=True,
                engine_report={"stub_note": "first line\nsecond | line"},
            )
        ],
        report_fields=[
            ReportField("stub_note", "A note.", in_summary=True, label="Note")
        ],
        output_paths_by_table={"flat": dataset_path},
        columns_by_table={"flat": ["pdb_id", "raw|field"]},
        run_manifest={
            "provenance": {
                "execution_sessions": [
                    {
                        "dependencies": list[object](),
                        "reference_data": {"stub": {"odd|key": "odd\nvalue"}},
                    }
                ]
            }
        },
    )

    assert _table_rows(summary, "| PDB ID |")[0][-1] == "first line second \\| line"
    assert _table_rows(summary, "| Field | Value |") == [["odd\\|key", "odd value"]]
    raw_row = _table_rows(summary, "| Column |")[-1]
    assert raw_row[0] == "raw\\|field"
    assert len(raw_row) == 9


def test_run_summary_shows_fractions_as_percentages_and_numbers_as_they_are(
    tmp_path: Path,
) -> None:
    summary = _write_summary(
        tmp_path,
        pdb_results=[
            PdbProcessResult(
                pdb_id="1ABC",
                state="done",
                download_ok=True,
                parse_ok=True,
                engine_report={
                    "stub_rate": 0.5,
                    "stub_shift": 0.3,
                    "stub_count": 3.0,
                },
            )
        ],
        report_fields=[
            ReportField("stub_rate", "A share.", kind="fraction", in_summary=True),
            ReportField("stub_shift", "In Å.", kind="number", in_summary=True),
            ReportField("stub_count", "Atoms.", kind="number", in_summary=True),
        ],
    )

    assert _table_rows(summary, "| PDB ID |")[0][-3:] == ["50.0%", "0.3", "3"]


def test_run_summary_marks_the_steps_an_entry_never_reached(tmp_path: Path) -> None:
    refused_on_record = PdbProcessResult(
        pdb_id="1ABC", state="refused", technique_status="refused"
    )
    # With no RCSB record, the parsed file decides. So this entry was
    # downloaded and parsed before intake refused it.
    refused_on_file = PdbProcessResult(
        pdb_id="2XYZ",
        state="refused",
        technique_status="refused",
        experimental_methods=["ELECTRON MICROSCOPY"],
        download_ok=True,
        parse_ok=True,
    )
    pending = PdbProcessResult(pdb_id="3DEF")

    summary = _write_summary(
        tmp_path, pdb_results=[refused_on_record, refused_on_file, pending]
    )

    assert _table_rows(summary, "| PDB ID |") == [
        ["1ABC", "download", "refused", "refused", "n/a", "n/a"],
        [
            "2XYZ",
            "download",
            "refused",
            "refused (ELECTRON MICROSCOPY)",
            "True",
            "True",
        ],
        ["3DEF", "download", "pending", "n/a", "n/a", "n/a"],
    ]


def test_run_summary_records_the_entry_selection(tmp_path: Path) -> None:
    summary = _write_summary(
        tmp_path,
        run_manifest={
            "selection": {
                "description": "resolution ≤ 2.0 Å",
                "service": "rcsb_search",
                "service_url": "https://search.rcsb.org/rcsbsearch/v2/query",
                "retrieved_at_iso": "2026-08-14T12:00:00Z",
                "total_count": 250,
                "returned_count": 100,
                "truncated": True,
            }
        },
    )

    assert "### Entry selection" in summary
    assert _table_rows(summary, "| Field | Recorded value |") == [
        ["Criteria", "resolution ≤ 2.0 Å"],
        ["Service", "rcsb_search"],
        ["Service URL", "`https://search.rcsb.org/rcsbsearch/v2/query`"],
        ["Retrieved at (UTC)", "2026-08-14T12:00:00Z"],
        ["Entries matched", "250"],
        ["Entries kept", "100"],
        ["Truncated by a limit", "True"],
    ]
    assert "`selection.query` in `manifest.json`" in summary
    # No range filter was set, so the summary has nothing to say about one.
    assert "was left out" not in summary


def test_run_summary_says_what_a_range_filter_left_out(tmp_path: Path) -> None:
    selection = {
        "description": "Neutron diffraction, temperature at least 273 K",
        "filters": {
            "technique": "neutron_diffraction",
            "min_temperature_K": 273.0,
            "max_temperature_K": None,
        },
    }

    summary = _write_summary(tmp_path, run_manifest={"selection": selection})

    assert ["Criteria", "Neutron diffraction, temperature at least 273 K"] in (
        _table_rows(summary, "| Field | Recorded value |")
    )
    assert (
        "A range in the criteria is strict. An entry was kept only when every "
        "value it records is in range, so an entry that records none was left out."
        in summary
    )
    unset = {**selection, "filters": {"max_temperature_K": None}}
    assert "was left out" not in _write_summary(
        tmp_path, run_manifest={"selection": unset}
    )


def test_run_summary_records_the_column_filter_and_the_warnings(
    tmp_path: Path,
) -> None:
    summary = _write_summary(
        tmp_path,
        column_filter=ColumnFilterSummary(
            mode="include",
            matched_columns=["pdb_id", "bdamage_rcsb"],
            missing_columns=["no_such"],
            requested_columns=["pdb_id", "bdamage*", "no_such"],
            available_columns=["pdb_id", "bdamage_rcsb", "x"],
        ),
        warnings=["Column 'no_such' matched nothing.", "Second\nwarning"],
    )

    assert (
        "Mode `include`: the tables kept only the columns matching these 3 names "
        "and patterns, in this order: `pdb_id`, `bdamage*`, `no_such`." in summary
    )
    assert "These matched no column and were ignored: `no_such`." in summary
    assert "- Column 'no_such' matched nothing.\n- Second warning\n" in summary


def test_run_summary_says_when_there_is_no_filter_and_no_warning(
    tmp_path: Path,
) -> None:
    summary = _write_summary(tmp_path)

    assert "## Column Filter\n\nNo column filter was applied.\n" in summary
    assert "## Warnings\n\nNone.\n" in summary


def test_run_summary_without_a_selection_leaves_the_section_out(
    tmp_path: Path,
) -> None:
    assert "### Entry selection" not in _write_summary(tmp_path)


def test_run_summary_lists_reference_data_and_leaves_nested_fields_to_the_manifest(
    tmp_path: Path,
) -> None:
    summary = _write_summary(
        tmp_path,
        run_manifest={
            "provenance": {
                "execution_sessions": [
                    {
                        "dependencies": list[object](),
                        "reference_data": {
                            "stub_reference": {
                                "label": "Stub reference database",
                                "sha256": "abc",
                                "entry_count": 12,
                                "source": {"ref": "v1"},
                            },
                            "flat_reference": {"entry_count": 3},
                            "empty_reference": {},
                        },
                    }
                ]
            }
        },
    )

    assert "**Stub reference database**" in summary
    assert "| sha256 | abc |" in summary
    assert "| entry_count | 12 |" in summary
    assert "| source |" not in summary
    assert "**flat_reference**" in summary
    assert "**empty_reference**" not in summary
    assert summary.count("Fields holding lists or objects are recorded in") == 1


def test_run_summary_describes_engine_tables_and_their_links(tmp_path: Path) -> None:
    bonds_path = tmp_path / "dataset_stub_bonds.csv"
    bonds_path.write_text("bond_id,chain_id,ligand_chain_id\n1,A,B\n", encoding="utf-8")
    bonds = TableSpec(
        name="stub_bonds",
        description="One row per bond.",
        key=("bond_id",),
        columns=(
            TableColumnSpec("bond_id", description="Bond number."),
            TableColumnSpec("chain_id", description="Protein chain."),
            TableColumnSpec("ligand_chain_id", description="Ligand chain."),
        ),
        links=(
            TableLink("chain", ("chain_id",), role="protein"),
            TableLink(
                "chain",
                ("ligand_chain_id",),
                target_columns=("chain_id",),
                role="ligand",
            ),
        ),
    )

    summary = _write_summary(
        tmp_path,
        engine_table_specs=[bonds],
        output_paths_by_table={"stub_bonds": bonds_path},
        columns_by_table={"stub_bonds": ["bond_id", "chain_id", "ligand_chain_id"]},
        row_counts_by_table={"stub_bonds": 1},
        run_manifest={"engine_tables": [{"name": "stub_bonds", "engine": "stub"}]},
    )

    assert _table_rows(summary, "| Table | Engine |") == [
        [
            "`stub_bonds`",
            "stub",
            "dataset_stub_bonds.csv",
            "1",
            "bond_id",
            "protein: chain_id → chain; ligand: ligand_chain_id → chain (chain_id)",
            "One row per bond.",
        ]
    ]
    # An engine table's columns are one group, described by its table spec.
    assert "**Columns**" in summary
    assert "| ligand_chain_id | Ligand chain. | text |" in summary


def test_data_dictionary_describes_each_kind_of_column(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.csv"
    serials = list(range(1, MAX_DISTINCT_COUNT + 2))
    pd.DataFrame(
        {
            "pdb_id": ["1ABC"] * len(serials),
            "raw_field": ["a"] * len(serials),
            "serial": serials,
            "flagged": ["true", "false"] * (len(serials) // 2) + ["true"],
            "blank": [None] * len(serials),
        }
    ).to_csv(dataset_path, index=False)
    columns = ["pdb_id", "raw_field", "serial", "flagged", "blank"]

    summary = _write_summary(
        tmp_path,
        output_paths_by_table={"flat": dataset_path},
        columns_by_table={"flat": columns},
    )

    rows = {row[0]: row for row in _table_rows(summary, "| Column |")}
    assert "**Other / Raw mmCIF Fields**" in summary
    assert rows["raw_field"] == [
        "raw_field",
        "n/a",
        "text",
        "0",
        "1",
        "1",
        "n/a",
        "n/a",
        "1",
    ]
    # Whole numbers print as integers. Past the cap, the distinct values are
    # not counted.
    assert rows["serial"][2:] == [
        "numeric",
        "0",
        "1",
        str(len(serials)),
        str((len(serials) + 1) // 2),
        "29.15476",
        "high-cardinality",
    ]
    assert rows["flagged"] == [
        "flagged",
        "n/a",
        "boolean",
        "0",
        "n/a",
        "n/a",
        "n/a",
        "n/a",
        "2",
    ]
    assert rows["blank"] == [
        "blank",
        "n/a",
        "empty",
        "100",
        "n/a",
        "n/a",
        "n/a",
        "n/a",
        "0",
    ]


def test_run_summary_keeps_the_previous_summary_when_writing_fails(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "run_summary.md"
    summary_path.write_text("previous summary\n", encoding="utf-8")
    dataset_path = tmp_path / "dataset.csv"
    dataset_path.write_text("pdb_id\n1ABC\n", encoding="utf-8")

    with pytest.raises(ValueError):
        _write_summary(
            tmp_path,
            output_paths_by_table={"flat": dataset_path},
            columns_by_table={"flat": ["pdb_id", "missing"]},
        )

    assert summary_path.read_text(encoding="utf-8") == "previous summary\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "dataset.csv",
        "run_summary.md",
    ]
