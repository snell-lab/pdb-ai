from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pytest

from pdbai import cli
from pdbai.engines.core import (
    STRUCTURE_METADATA,
    ArtifactKey,
    ColumnSpec,
    EngineContext,
    EngineOptions,
    EnginePhase,
    EngineResult,
    EngineSpec,
    OptionSpec,
    ReportField,
)
from pdbai.orchestration import build_atom_site_dataset_from_mmcif_files
from pdbai.runs import open_run
from pdbai.webapp import create_app

# A third-party engine pair. The counter counts atoms and publishes the
# count, and the reader reads it. Nothing outside this file knows about
# either engine.
ATOM_COUNT = ArtifactKey("stub_atom_count", int)


class CountingEngine:
    spec = EngineSpec(
        name="stub_counter",
        display_name="Stub counter",
        phase=EnginePhase.ANNOTATION,
        toggle="include_stub",
        consumes=(STRUCTURE_METADATA,),
        provides=(ATOM_COUNT,),
        options=(
            OptionSpec("include_stub", label="Stub analysis", help="Run the stub."),
            OptionSpec(
                "stub_scale",
                label="Stub scale",
                help="Multiply the count by this.",
                kind="count",
                default=None,
                refines="include_stub",
            ),
        ),
        report_fields=(
            ReportField(
                "stub_status",
                "Stub outcome.",
                on_failure="failed",
                in_summary=True,
                label="Stub",
            ),
            ReportField("stub_message", "Why it failed.", carries_message=True),
        ),
        columns=(ColumnSpec("stub_scaled_count", group="Stub Fields"),),
    )

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail

    def run(self, context: EngineContext) -> EngineResult:
        if self.fail:
            raise RuntimeError("stub exploded")
        scale = context.options.value("stub_scale") or 1
        count = len(context.atom_dataframe) * int(scale)
        return EngineResult(
            columns=pd.DataFrame(
                {"stub_scaled_count": count}, index=context.atom_dataframe.index
            ),
            artifacts={ATOM_COUNT: count},
            report={"stub_status": "ok", "stub_message": None},
        )


class ReadingEngine:
    spec = EngineSpec(
        name="stub_reader",
        display_name="Stub reader",
        phase=EnginePhase.ANNOTATION,
        toggle="include_stub",
        dependencies=("stub_counter",),
        consumes=(ATOM_COUNT,),
        columns=(ColumnSpec("stub_seen_count", group="Stub Fields"),),
    )

    def run(self, context: EngineContext) -> EngineResult:
        count = context.artifacts.get(ATOM_COUNT)
        return EngineResult(
            columns=pd.DataFrame(
                {"stub_seen_count": count}, index=context.atom_dataframe.index
            )
        )


@pytest.fixture
def stub_engines(install_engines: Callable[..., None]) -> None:
    install_engines(CountingEngine(), ReadingEngine())


def _options() -> EngineOptions:
    return EngineOptions(
        values={"include_rabdam": False, "include_stub": True, "stub_scale": 2}
    )


@pytest.mark.usefixtures("stub_engines")
def test_a_registered_engine_runs_end_to_end_without_other_changes(
    minimal_cif: Path, tmp_path: Path
) -> None:
    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif], tmp_path / "work", run_id="stub", engine_options=_options()
    )

    dataframe = pd.read_csv(result.csv_path)
    atoms = len(dataframe)
    assert dataframe["stub_scaled_count"].unique().tolist() == [atoms * 2]
    # The reader computes nothing itself, so this count came from the
    # counter's artifact.
    assert dataframe["stub_seen_count"].unique().tolist() == [atoms * 2]

    with result.run_report_path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    assert row["stub_status"] == "ok"
    manifest = open_run(tmp_path / "work", "stub").manifest()
    assert manifest["entries"]["1ABC"]["engine_report"]["stub_status"] == "ok"
    engine_settings = manifest["settings"]["engine_settings"]
    assert engine_settings["include_stub"] is True
    assert engine_settings["stub_scale"] == 2
    summary = result.run_summary_path.read_text(encoding="utf-8")
    assert "**include_stub** (`True`)" in summary
    assert "| Stub |" in summary


def test_a_failing_optional_engine_does_not_lose_the_structure(
    minimal_cif: Path, tmp_path: Path, install_engines: Callable[..., None]
) -> None:
    install_engines(CountingEngine(fail=True), ReadingEngine())

    result = build_atom_site_dataset_from_mmcif_files(
        [minimal_cif], tmp_path / "work", run_id="stub", engine_options=_options()
    )

    dataframe = pd.read_csv(result.csv_path)
    assert result.pdb_results[0].state == "done"
    assert dataframe["stub_scaled_count"].isna().all()
    assert result.pdb_results[0].engine_report["stub_status"] == "failed"
    assert result.pdb_results[0].engine_report["stub_message"] == "stub exploded"
    assert any("Stub counter failed: stub exploded" in w for w in result.warnings)


@pytest.mark.usefixtures("stub_engines")
def test_the_command_line_and_web_form_offer_the_engine_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _ = cli.parse_args(["1ABC", "--include-stub", "--stub-scale", "3"])
    assert (args.include_stub, args.stub_scale) == (True, 3)

    monkeypatch.chdir(tmp_path)
    html = create_app().test_client().get("/").data.decode()
    assert 'data-option="include_stub"' in html
    assert 'data-refines="include_stub"' in html
