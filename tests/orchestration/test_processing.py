from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pdbai.engines.catalog import get_engine_registry
from pdbai.engines.core import (
    STRUCTURE_CIF_PATH,
    EngineContext,
    EngineOptions,
    EnginePhaseResult,
)
from pdbai.orchestration import build as build_module
from pdbai.orchestration import processing as processing_module
from pdbai.orchestration.annotation import AnnotatedStructure
from pdbai.orchestration.build import DatasetBuild
from pdbai.orchestration.processing import AcquiredEntry, process_entry
from pdbai.results import PdbProcessResult
from pdbai.runs import open_run
from pdbai.structure.mmcif_parser import AtomSiteParseResult

# These atom rows lack the residue author identity columns that a
# star-schema split needs.
_UNSPLITTABLE_ATOMS = pd.DataFrame({"pdb_id": ["1ABC"], "atom_name": ["CA"]})


def _build(tmp_path: Path, options: EngineOptions) -> DatasetBuild:
    active = get_engine_registry().with_defaults(options)
    run_state = open_run(tmp_path, "processing-test")
    run_state.load_or_initialize(
        ["1ABC"], active.as_settings_dict(), schema_fingerprint="test-schema"
    )
    return DatasetBuild(
        run_state=run_state,
        working_directory=tmp_path,
        pdb_ids=["1ABC"],
        options=active,
    )


def _annotated_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, atoms: pd.DataFrame
) -> AcquiredEntry:
    """Return an acquired 1ABC whose parse and annotation give ``atoms``."""

    def parsed(*_args: object, **_kwargs: object) -> AtomSiteParseResult:
        return AtomSiteParseResult(atoms)

    def annotated(*_args: object, **_kwargs: object) -> AnnotatedStructure:
        return AnnotatedStructure(atoms, EnginePhaseResult())

    monkeypatch.setattr(processing_module, "parse_mmcif_atom_rows", parsed)
    monkeypatch.setattr(processing_module, "annotate_structure", annotated)
    result = PdbProcessResult(
        pdb_id="1ABC", download_ok=True, technique_status="matched"
    )
    context = EngineContext("1ABC", pd.DataFrame(), EngineOptions())
    context.artifacts.put(STRUCTURE_CIF_PATH, tmp_path / "1abc.cif")
    return AcquiredEntry(result=result, context=context)


def test_a_table_split_error_fails_the_entry_not_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _build(tmp_path, EngineOptions(dataset_shape="star_schema"))
    acquired = _annotated_entry(tmp_path, monkeypatch, _UNSPLITTABLE_ATOMS)

    process_entry(build, acquired)

    result = acquired.result
    assert result.state == "failed"
    assert result.output_error is not None
    assert result.output_error.startswith(
        "Writing the tables failed for 1ABC: Star-schema normalization requires "
        "the residue author identity columns"
    )
    assert build.warnings == [result.output_error]
    assert build.results == [result]
    entry = build.run_state.manifest()["entries"]["1ABC"]
    assert (entry["state"], entry["error"]) == ("failed", result.output_error)
    assert not any(build.run_state.directory.partials_directory.iterdir())


def test_an_error_writing_partials_ends_the_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = _build(tmp_path, EngineOptions())
    acquired = _annotated_entry(tmp_path, monkeypatch, _UNSPLITTABLE_ATOMS)

    def full_disk(*_args: object, **_kwargs: object) -> dict[str, Path]:
        raise OSError("No space left on device")

    monkeypatch.setattr(build_module, "write_partial_csvs", full_disk)

    with pytest.raises(OSError, match="No space left on device"):
        process_entry(build, acquired)
    assert build.results == []
