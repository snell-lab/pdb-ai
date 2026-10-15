from __future__ import annotations

from pathlib import Path

import pytest

from pdbai import cli
from pdbai.results import DatasetBuildResult, PdbProcessResult


def test_cli_counts_an_entry_refused_on_its_parsed_file_only_as_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # With no RCSB record, intake judges the entry by its parsed file. A
    # refusal at that late stage still does not count as an output failure.
    refused = PdbProcessResult(
        pdb_id="8EMX",
        state="refused",
        technique_status="refused",
        experimental_methods=["ELECTRON MICROSCOPY"],
        download_ok=True,
        parse_ok=True,
    )
    done = PdbProcessResult(
        pdb_id="1ABC", state="done", metadata_ok=True, download_ok=True, parse_ok=True
    )

    def fake_build(**_arguments: object) -> DatasetBuildResult:
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
    assert "PDB entries in the dataset: 1 of 2" in out
    assert "1 refused at technique intake" in out
    assert "0 output failed" in out
    assert "Output failures" not in out


def test_cli_counts_an_entry_that_failed_after_parsing_as_an_output_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    failed = PdbProcessResult(
        pdb_id="1ABC",
        state="failed",
        metadata_ok=True,
        download_ok=True,
        parse_ok=True,
        output_error="Annotation failed for 1ABC: engine exploded",
    )

    def fake_build(**_arguments: object) -> DatasetBuildResult:
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

    code = cli.main(["1ABC", "--workdir", str(tmp_path)])

    out = capsys.readouterr().out
    assert code == 2
    assert "0 download failed, 0 parse failed, 1 output failed" in out
    assert "  - 1ABC: Annotation failed for 1ABC: engine exploded" in out
