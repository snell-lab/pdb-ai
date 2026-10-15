from __future__ import annotations

import dataclasses
import gzip
import json
import shutil
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from pdbai.engines.core import EngineOptions
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files
from pdbai.runs import RunState, open_run


def test_uploaded_mmcif_build_runs_end_to_end_without_network(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    working_directory = tmp_path / "work"

    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        working_directory,
        output_filename="atoms.csv",
        column_filter_mode="include",
        column_filter_list=["pdb_id", "atom_name", "x"],
        run_id="offline-flat",
        engine_options=offline_options,
    )

    dataframe = pd.read_csv(result.csv_path)
    assert result.atom_count == 4
    assert result.column_count == 3
    assert result.pdb_ids == ["1ABC"]
    assert dataframe.columns.tolist() == ["pdb_id", "atom_name", "x"]
    assert dataframe["atom_name"].tolist() == ["N", "CA", "C", "ZN"]
    assert result.pdb_results[0].parse_ok
    assert result.pdb_results[0].source == "upload"
    assert result.run_report_path.exists()
    assert result.run_summary_path.exists()
    manifest = json.loads(
        (working_directory / "runs" / "offline-flat" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    provenance = manifest["provenance"]
    session = provenance["execution_sessions"][0]
    assert session["code"]["application"] == "PDB-AI"
    assert len(session["code"]["git_commit"]) == 40
    assert session["runtime"]["python_version"]
    assert session["dependencies"]
    input_receipt = manifest["entries"]["1ABC"]["inputs"][0]
    assert input_receipt["role"] == "uploaded_mmcif"
    upload_input = manifest["settings"]["upload_inputs"][0]
    assert input_receipt["sha256"] == upload_input["content_sha256"]
    assert input_receipt["source_url"] is None
    output_receipt = provenance["outputs"][0]
    assert output_receipt["role"] == "dataset_flat"
    assert output_receipt["path"] == "dataset.csv"
    assert len(output_receipt["sha256"]) == 64
    published_receipt = provenance["outputs"][1]
    assert published_receipt["role"] == "published_dataset_flat"
    assert published_receipt["path"] == "atoms.csv"
    assert published_receipt["sha256"] == output_receipt["sha256"]
    summary = result.run_summary_path.read_text(encoding="utf-8")
    assert "## Reproducibility Log" in summary
    assert "Installed runtime dependency closure" in summary
    assert "uploaded_mmcif" in summary
    assert "dataset_flat" in summary


def test_star_schema_build_restores_join_keys_after_column_filtering(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        tmp_path / "work",
        column_filter_mode="include",
        column_filter_list=["x", "resolution", "citation_doi"],
        run_id="offline-star",
        engine_options=dataclasses.replace(
            offline_options, dataset_shape="star_schema"
        ),
    )

    atoms = pd.read_csv(result.csv_path)
    assert "residue_key" in atoms.columns
    assert {"pdb_id", "chain_id", "res_num", "res_name", "x"}.issubset(atoms.columns)
    assert set(result.additional_csv_paths) == {
        "residues",
        "chains",
        "structures",
        "citations",
    }
    structures = pd.read_csv(result.additional_csv_paths["structures"])
    citations = pd.read_csv(result.additional_csv_paths["citations"])
    assert structures["resolution"].tolist() == [1.5]
    assert "citation_doi" in citations.columns


def test_exclude_filter_removes_matched_columns_and_keeps_the_rest_in_order(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    unfiltered = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        tmp_path / "unfiltered",
        run_id="offline-unfiltered",
        engine_options=offline_options,
    )
    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        tmp_path / "work",
        column_filter_mode="exclude",
        column_filter_list=["x", "_cell.*", "not_a_column"],
        run_id="offline-exclude",
        engine_options=offline_options,
    )

    every_column = pd.read_csv(unfiltered.csv_path).columns.tolist()
    excluded = ["x", *(name for name in every_column if name.startswith("_cell."))]
    assert "_cell.length_a" in excluded
    assert pd.read_csv(result.csv_path).columns.tolist() == [
        name for name in every_column if name not in excluded
    ]
    assert result.column_filter is not None
    assert result.column_filter.missing_columns == ["not_a_column"]
    # The run resolves its filter against the columns as built, so it
    # records them before the filter removes any.
    run_state = open_run(tmp_path / "work", "offline-exclude")
    assert set(excluded) <= set(run_state.available_columns())


def test_star_schema_exclude_filter_keeps_the_join_keys_it_names(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        tmp_path / "work",
        column_filter_mode="exclude",
        column_filter_list=["chain_id", "x"],
        run_id="offline-star-exclude",
        engine_options=dataclasses.replace(
            offline_options, dataset_shape="star_schema"
        ),
    )

    atoms = pd.read_csv(result.csv_path)
    residues = pd.read_csv(result.additional_csv_paths["residues"])
    chains = pd.read_csv(result.additional_csv_paths["chains"])
    # chain_id is part of the residue and chain keys, so the filter cannot
    # drop it.
    for table in (atoms, residues, chains):
        assert "chain_id" in table.columns
    assert "x" not in atoms.columns
    assert "y" in atoms.columns
    assert result.column_filter is not None
    assert result.column_filter.matched_columns == ["chain_id", "x"]


def test_upload_collisions_use_assigned_ids_in_rows_and_manifest(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    first = tmp_path / "first" / "sample.cif"
    second = tmp_path / "second" / "sample.cif"
    compressed = tmp_path / "third" / "sample.cif.gz"
    # This file is named like the ID the second file is given. It still gets
    # an ID of its own.
    named_like_an_id = tmp_path / "fourth" / "sample_2.cif"
    for path in (first, second, compressed, named_like_an_id):
        path.parent.mkdir()
    shutil.copyfile(minimal_cif, first)
    shutil.copyfile(minimal_cif, second)
    shutil.copyfile(minimal_cif, named_like_an_id)
    with minimal_cif.open("rb") as source, gzip.open(compressed, "wb") as destination:
        shutil.copyfileobj(source, destination)

    working_directory = tmp_path / "work"
    result = build_atom_site_dataset_from_mmcif_files(
        [first, second, compressed, named_like_an_id],
        working_directory,
        output_filename="collisions.csv",
        run_id="upload-collisions",
        engine_options=offline_options,
    )

    assert result.pdb_ids == ["SAMPLE", "SAMPLE_2", "SAMPLE_3", "SAMPLE_2_2"]
    assert pd.read_csv(result.csv_path)["pdb_id"].value_counts().to_dict() == {
        "SAMPLE": 4,
        "SAMPLE_2": 4,
        "SAMPLE_3": 4,
        "SAMPLE_2_2": 4,
    }
    manifest = open_run(working_directory, "upload-collisions").manifest()
    upload_inputs = manifest["settings"]["upload_inputs"]
    assert [item["structure_id"] for item in upload_inputs] == result.pdb_ids
    assert [item["source_filename"] for item in upload_inputs] == [
        "sample.cif",
        "sample.cif",
        "sample.cif.gz",
        "sample_2.cif",
    ]
    for pdb_id, upload_input in zip(result.pdb_ids, upload_inputs, strict=True):
        receipt = manifest["entries"][pdb_id]["inputs"][0]
        assert len(receipt["sha256"]) == 64
        assert receipt["size_bytes"] > 0
        assert receipt["sha256"] == upload_input["content_sha256"]


def test_resumed_upload_build_numbers_progress_after_the_done_entries(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    paths = [tmp_path / f"{name}.cif" for name in ("a", "b", "c")]
    for path in paths:
        shutil.copyfile(minimal_cif, path)
    working_directory = tmp_path / "work"
    cancel_token = threading.Event()
    parsed: list[tuple[str, int, int]] = []

    def record(stage: str, current: int, total: int, pdb_id: str) -> None:
        if stage != "parse":
            return
        parsed.append((pdb_id, current, total))
        # A cancel while the second entry is in hand still lets it be
        # recorded. The third entry is left for the resumed build.
        if current == 2:
            cancel_token.set()

    build_atom_site_dataset_from_mmcif_files(
        paths,
        working_directory,
        run_id="upload-numbered",
        progress_callback=record,
        cancel_token=cancel_token,
        engine_options=offline_options,
    )
    assert (
        open_run(working_directory, "upload-numbered").manifest()["run_status"]
        == "paused"
    )
    build_atom_site_dataset_from_mmcif_files(
        paths,
        working_directory,
        run_id="upload-numbered",
        resume=True,
        progress_callback=record,
        engine_options=offline_options,
    )

    assert parsed == [("A", 1, 3), ("B", 2, 3), ("C", 3, 3)]


def test_done_entry_is_written_to_the_manifest_once(
    minimal_cif: Path,
    tmp_path: Path,
    offline_options: EngineOptions,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[str] = []
    update_entry = RunState.update_entry

    def counting_update_entry(self: RunState, pdb_id: str, **fields: Any) -> None:
        written.append(pdb_id)
        update_entry(self, pdb_id, **fields)

    monkeypatch.setattr(RunState, "update_entry", counting_update_entry)

    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        tmp_path / "work",
        run_id="upload-written-once",
        engine_options=offline_options,
    )

    assert result.pdb_results[0].state == "done"
    assert written == ["1ABC"]


def test_upload_resume_rejects_changed_input_content(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    working_directory = tmp_path / "work"
    build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        working_directory,
        run_id="upload-fingerprint",
        engine_options=offline_options,
    )
    minimal_cif.write_text(
        minimal_cif.read_text(encoding="utf-8") + "\n# content changed\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different settings"):
        build_atom_site_dataset_from_mmcif_files(
            [minimal_cif],
            working_directory,
            run_id="upload-fingerprint",
            resume=True,
            engine_options=offline_options,
        )


def test_column_filter_generator_is_materialized_once(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    requested_columns = (column for column in ["pdb_id", "atom_name"])

    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        tmp_path / "work",
        column_filter_mode="include",
        column_filter_list=requested_columns,
        run_id="generator-filter",
        engine_options=offline_options,
    )

    assert pd.read_csv(result.csv_path).columns.tolist() == ["pdb_id", "atom_name"]
    assert result.column_filter is not None
    assert result.column_filter.matched_columns == ["pdb_id", "atom_name"]


def test_unmatched_wildcard_filter_is_reported(
    minimal_cif: Path, tmp_path: Path, offline_options: EngineOptions
) -> None:
    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif],
        tmp_path / "work",
        column_filter_mode="include",
        column_filter_list=["*_definitely_missing"],
        run_id="missing-filter",
        engine_options=offline_options,
    )

    assert result.column_filter is not None
    assert result.column_filter.matched_columns == []
    assert result.column_filter.missing_columns == ["*_definitely_missing"]
    assert pd.read_csv(result.csv_path).columns.tolist() == ["pdb_id"]
    assert any(
        "No columns matched the requested include filter" in warning
        for warning in result.warnings
    )
