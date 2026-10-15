from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from pdbai.column_filtering import ColumnFilterMode
from pdbai.engines.core import EngineOptions
from pdbai.orchestration import (
    build_atom_site_dataset,
    build_atom_site_dataset_from_mmcif_files,
)


@pytest.mark.parametrize(
    ("mode", "values", "message"),
    [
        ("include", None, "column_filter_list"),
        (None, ["pdb_id"], "column_filter_mode"),
        ("invalid", ["pdb_id"], "column_filter_mode"),
    ],
)
def test_download_api_rejects_incomplete_column_filters(
    tmp_path: Path,
    mode: str | None,
    values: list[str] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        build_atom_site_dataset(
            ["1ABC"],
            tmp_path,
            # An untyped caller can pass any string, so the builder checks the
            # mode at run time as well.
            column_filter_mode=cast(ColumnFilterMode | None, mode),
            column_filter_list=values,
            engine_options=EngineOptions(values={"include_rabdam": False}),
        )


def test_download_api_rejects_nonpositive_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_workers must be a positive integer"):
        build_atom_site_dataset(
            ["1ABC"],
            tmp_path,
            max_workers=0,
            engine_options=EngineOptions(values={"include_rabdam": False}),
        )


def test_download_api_requires_run_id_when_resuming(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id must be provided"):
        build_atom_site_dataset(
            ["1ABC"],
            tmp_path,
            resume=True,
            engine_options=EngineOptions(values={"include_rabdam": False}),
        )


def test_upload_api_requires_run_id_when_resuming(tmp_path: Path) -> None:
    # The file does not exist. The error shows the arguments are checked
    # before any file is read.
    with pytest.raises(ValueError, match="run_id must be provided"):
        build_atom_site_dataset_from_mmcif_files(
            [tmp_path / "1abc.cif"],
            tmp_path,
            resume=True,
            engine_options=EngineOptions(values={"include_rabdam": False}),
        )


def test_invalid_output_filename_is_rejected_before_creating_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"

    with pytest.raises(ValueError, match="Output filename"):
        build_atom_site_dataset(
            ["1ABC"],
            workspace,
            output_filename="nested/dataset.csv",
            engine_options=EngineOptions(values={"include_rabdam": False}),
        )

    assert not workspace.exists()


@pytest.mark.parametrize(
    ("output_filename", "message"),
    [
        ("runs", "ends in .csv"),
        ("taken.csv", "not a file"),
        # This name is valid alone. The star schema adds a table name to it
        # for each other table, which takes those names past the limit.
        ("a" * 209 + ".csv", "at most 217 UTF-8 bytes"),
    ],
)
def test_an_output_filename_that_cannot_be_published_is_rejected_before_the_build(
    tmp_path: Path, output_filename: str, message: str
) -> None:
    (tmp_path / "taken.csv").mkdir()

    with pytest.raises(ValueError, match=message):
        build_atom_site_dataset(
            ["1ABC"],
            tmp_path,
            output_filename=output_filename,
            engine_options=EngineOptions(
                dataset_shape="star_schema", values={"include_rabdam": False}
            ),
        )

    assert not (tmp_path / "runs").exists()
