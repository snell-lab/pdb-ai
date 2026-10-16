"""The record of one build of a claimed run, and the run's settlement.

A DatasetBuild holds what download and upload builds share once the run is
claimed. It fixes the entries the build will process and reports progress.
It records each entry in the run's manifest as soon as the entry ends. A done
entry also leaves one partial CSV for each table the run writes. When the
build ends, finish settles the run.

Progress is counted against the run's whole list of identifiers. A resumed
build therefore starts counting after the entries already done.

The run's column filter is applied to each entry before its rows are split
into tables. A partial therefore never holds a column the run drops.

Settlement covers every identifier in the run, not only those this build
recorded. The run report therefore describes the whole run, however many
builds it took.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import InitVar, dataclass, field
from pathlib import Path

import pandas as pd

from .. import dataset_shapes
from ..column_filtering import ColumnFilterMode, filter_dataframe_columns
from ..engines.catalog import get_engine_registry
from ..engines.core import DatasetSource, EngineOptions
from ..json_values import object_dict
from ..results import DatasetBuildResult, PdbProcessResult
from ..runs import (
    ProgressCallback,
    ProgressStage,
    RunState,
    settle_run,
    write_partial_csvs,
)
from .annotation import AnnotatedStructure
from .technique_intake import intake_warnings


@dataclass
class DatasetBuild:
    """One build of a claimed run, with its settings and its entry outcomes."""

    run_state: RunState
    working_directory: Path
    pdb_ids: list[str]
    options: EngineOptions
    # The primary table's published name. None lets settlement choose it.
    output_filename: str | None = None
    progress_callback: ProgressCallback | None = None
    # When set, the entries an earlier build left done are not processed.
    resume: InitVar[bool] = False
    # Set by the caller to cancel the build once the entry in hand is recorded.
    cancel_token: threading.Event | None = None
    # When set, settlement copies the assembled tables into the working
    # directory. Otherwise they stay in the run directory only.
    publish: bool = True
    source: DatasetSource = "download"
    column_filter_mode: ColumnFilterMode | None = None
    column_filter_list: list[str] = field(default_factory=list[str])
    # Warnings for the build's result. Processing adds to them, and finish
    # adds the intake summaries.
    warnings: list[str] = field(default_factory=list[str])
    # One per entry this build recorded, in the order they were recorded.
    results: list[PdbProcessResult] = field(default_factory=list[PdbProcessResult])
    # This build's entries to process, in run order. The list is fixed when
    # the build starts and never shrinks, so progress and pausing can count
    # against it.
    ids_to_process: list[str] = field(init=False)

    def __post_init__(self, resume: bool) -> None:
        """Fix the entries this build will process.

        A new build takes every entry. A resumed build takes those not done. An
        entry counts as done only when a partial exists for every table the run
        writes. A refused or failed entry is therefore processed again.
        """
        if not resume:
            self.ids_to_process = list(self.pdb_ids)
            return
        done = self.run_state.done_pdb_ids(
            required_tables=dataset_shapes.output_table_names(self.options, self.source)
        )
        self.ids_to_process = [pdb_id for pdb_id in self.pdb_ids if pdb_id not in done]

    def cancel_requested(self) -> bool:
        """Return whether the caller asked to cancel the build."""
        return self.cancel_token is not None and self.cancel_token.is_set()

    def report_progress(self, stage: ProgressStage, pdb_id: str) -> None:
        """Report that the entry in hand reached a stage.

        The entry's position counts the entries done before this build and
        those this build has recorded.
        """
        position = len(self.pdb_ids) - len(self.ids_to_process) + len(self.results) + 1
        self.run_state.report_progress(
            self.progress_callback, stage, position, len(self.pdb_ids), pdb_id
        )

    def record_done(
        self,
        result: PdbProcessResult,
        annotated: AnnotatedStructure,
        *,
        input_receipts: Sequence[Mapping[str, object]],
    ) -> None:
        """Write an entry's partial CSVs and record it as done.

        The run's column filter is applied to the annotated rows first. A
        column the star schema needs to link its tables is kept even when the
        filter removes it, and a parser helper column the split needs is used
        and then dropped. The rows are then split into the tables of the
        run's dataset shape. Each base table and each enabled engine table is
        written as one partial CSV. ``result`` is marked done and given each
        engine table's row count. An error writing a partial or the manifest
        is raised.
        """
        # The run keeps one list of every column any entry offered. It is
        # taken before filtering, so the finished run can report which
        # requested columns no entry had. The list is kept per run, not per
        # entry, because entries offer largely the same few hundred columns.
        self.run_state.add_available_columns(map(str, annotated.dataframe.columns))
        tables = dataset_shapes.filtered_base_tables(
            self._column_filtered(annotated.dataframe),
            annotated.dataframe,
            self.options.dataset_shape,
        )
        # Every enabled engine table gets a partial, even one with only a
        # header. Resuming counts an entry as done only when all its partials
        # exist. Assembly also joins a table's partials under one header.
        engine_tables = annotated.phase_result.tables
        engine_specs = get_engine_registry().enabled_table_specs(
            self.options, self.source
        )
        for spec in engine_specs:
            table_dataframe = engine_tables.get(spec.name)
            tables[spec.name] = (
                spec.empty_dataframe() if table_dataframe is None else table_dataframe
            )
        partial_paths = write_partial_csvs(
            tables, self.run_state.directory, result.pdb_id
        )

        result.state = "done"
        result.engine_table_row_counts = {
            spec.name: len(tables[spec.name]) for spec in engine_specs
        }
        root = self.run_state.directory.root
        self.run_state.update_entry(
            result.pdb_id,
            **result.to_manifest(),
            inputs=list(input_receipts),
            partial_paths={
                table: path.relative_to(root).as_posix()
                for table, path in partial_paths.items()
            },
            row_count=len(annotated.dataframe),
            # An earlier failed attempt may have left an error in the manifest
            # entry. The result has no field that would replace it, so it is
            # cleared here.
            error=None,
        )
        self.results.append(result)

    def record_failed_or_refused(
        self,
        result: PdbProcessResult,
        *,
        input_receipts: Sequence[Mapping[str, object]],
    ) -> None:
        """Record an entry that failed or was refused, and why.

        The entry's recorded error is its refusal message, or else the error of
        the stage it failed at.
        """
        error: str | None = (
            result.technique_message
            if result.refused()
            else result.download_error or result.parse_error or result.output_error
        )
        self.run_state.update_entry(
            result.pdb_id,
            **result.to_manifest(),
            inputs=list(input_receipts),
            error=error,
        )
        self.results.append(result)

    def finish(self) -> DatasetBuildResult:
        """Settle the run from every entry's outcome and return the result.

        The run is left paused when the build was cancelled before it recorded
        every entry it set out to process. Otherwise it is left completed.
        When the run has no done entry, finish raises RunPausedError if the
        build was cancelled and NoDatasetError if not. RuntimeError is raised
        when assembly runs out of memory. An error is raised without being
        recorded. The caller that claimed the run leaves it in error.
        """
        run_results = self._run_results()
        return settle_run(
            run_state=self.run_state,
            working_directory=self.working_directory,
            output_filename=self.output_filename,
            pdb_ids=self.pdb_ids,
            pdb_results=run_results,
            warnings=[*self.warnings, *intake_warnings(run_results, self.options)],
            options=self.options,
            column_filter_mode=self.column_filter_mode,
            column_filter_list=self.column_filter_list,
            # Each recorded entry adds one result. Any stop other than a
            # cancel is an error raised before finish is called. So fewer
            # results than entries to process means the build was cancelled.
            paused=len(self.results) < len(self.ids_to_process),
            progress_callback=self.progress_callback,
            publish=self.publish,
            source=self.source,
        )

    def _column_filtered(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if self.column_filter_mode is None:
            return dataframe
        return filter_dataframe_columns(
            dataframe, self.column_filter_mode, self.column_filter_list
        )

    def _run_results(self) -> list[PdbProcessResult]:
        # One result per identifier, in order. An entry this build did not
        # record was either done by an earlier build or not reached before a
        # cancel. Its result is restored from its manifest entry.
        recorded = {result.pdb_id: result for result in self.results}
        entries = object_dict(self.run_state.manifest().get("entries"))
        return [
            recorded[pdb_id]
            if pdb_id in recorded
            else PdbProcessResult.from_manifest(
                pdb_id, object_dict(entries.get(pdb_id)), default_source=self.source
            )
            for pdb_id in self.pdb_ids
        ]


__all__ = ["DatasetBuild"]
