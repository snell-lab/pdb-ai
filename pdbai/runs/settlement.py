"""Settling a run, the step that ends every build.

A build settles its run when it finishes or is paused. A run that settles is
left completed or paused. A run that cannot settle is left in error, with
the reason in its manifest, so the Runs panel can show why it stopped.

A paused run is settled like a finished one. The entries done before the
pause then already form a dataset the user can open. Resuming the run
assembles every table again from all its partials, so nothing assembled
before the pause is kept or patched.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path

from .. import dataset_shapes
from ..column_filtering import (
    ColumnFilterMode,
    missing_columns_warning,
    resolve_requested_columns,
)
from ..engines.catalog import get_engine_registry
from ..engines.core import DatasetSource, EngineOptions, TableSpec
from ..json_values import object_dict
from ..provenance import file_receipt
from ..reporting import write_run_report, write_run_summary
from ..results import (
    ColumnFilterSummary,
    DatasetBuildResult,
    NoDatasetError,
    PdbProcessResult,
    RunPausedError,
)
from .assembly import (
    AssemblyProgressCallback,
    assemble_final_csvs,
    read_csv_columns,
    read_preview_rows,
)
from .publish import publish_tables
from .state import ProgressCallback, RunState, entries_of
from .storage import RunDirectory

# How many rows of the primary table the build result carries as a preview.
# The web app's filter tool previews the same number, and the page titles
# either preview by it.
PREVIEW_ROW_COUNT = 25

# How many entry errors the message of a run without data quotes. The rest
# are only counted.
_MAX_ENTRY_ERRORS_IN_MESSAGE = 20


def _done_partials(
    run_directory: RunDirectory,
    entries: Mapping[str, Mapping[str, object]],
    pdb_ids: Sequence[str],
    table_names: Sequence[str],
) -> dict[str, list[Path]]:
    # Each table's partials of the done entries, in the order of pdb_ids.
    partials_by_table: dict[str, list[Path]] = {table: [] for table in table_names}
    for pdb_id in pdb_ids:
        entry = entries.get(pdb_id, {})
        if entry.get("state") != "done":
            continue
        recorded_paths = object_dict(entry.get("partial_paths"))
        for table in table_names:
            path = run_directory.contained_partial_path(recorded_paths.get(table))
            if path is not None:
                partials_by_table[table].append(path)
    return partials_by_table


def _assembly_progress_reporter(
    run_state: RunState, progress_callback: ProgressCallback | None
) -> AssemblyProgressCallback:
    def report(current: int, total: int, name: str) -> None:
        # Every step reaches the callback, but only about a hundred are
        # logged, so the event log does not grow with the number of partials.
        step = max(1, total // 100)
        if current == total or current == 1 or current % step == 0:
            run_state.report_progress(
                progress_callback, "assemble", current, total, name
            )
        elif progress_callback:
            progress_callback("assemble", current, total, name)

    return report


def _write_reports(
    run_state: RunState,
    *,
    options: EngineOptions,
    source: DatasetSource,
    engine_tables: Sequence[TableSpec],
    pdb_results: list[PdbProcessResult],
    output_paths_by_table: Mapping[str, Path],
    columns_by_table: Mapping[str, list[str]],
    row_counts_by_table: Mapping[str, int],
    column_filter: ColumnFilterSummary | None,
    warnings: Sequence[str],
) -> None:
    # The run report and summary are written whether or not any table was
    # assembled.
    registry = get_engine_registry()
    engine_report_fields = registry.report_fields(options, source)
    write_run_report(
        run_state.directory.report_path,
        pdb_results,
        report_fields=engine_report_fields,
        engine_table_names=[table.name for table in engine_tables],
    )
    write_run_summary(
        run_state.directory.summary_path,
        options=options,
        option_specs=registry.option_specs(),
        pdb_results=pdb_results,
        report_fields=engine_report_fields,
        engine_table_specs=engine_tables,
        output_paths_by_table=output_paths_by_table,
        columns_by_table=columns_by_table,
        row_counts_by_table=row_counts_by_table,
        run_manifest=run_state.manifest(),
        column_filter=column_filter,
        warnings=warnings,
    )


def _no_data_message(
    pdb_ids: Sequence[str],
    entries: Mapping[str, Mapping[str, object]],
    warnings: Sequence[str],
) -> str:
    entry_errors = [
        f"{pdb_id}: {error}"
        for pdb_id in pdb_ids
        if (error := entries.get(pdb_id, {}).get("error"))
    ]
    shown = entry_errors[:_MAX_ENTRY_ERRORS_IN_MESSAGE]
    if len(entry_errors) > len(shown):
        shown.append(f"and {len(entry_errors) - len(shown)} more")
    detail = "; ".join(shown) or "; ".join(warnings) or "no details recorded"
    return f"No atom-site data could be produced for any entry of the run. Reason(s): {detail}"


def _final_warnings(
    warnings: Sequence[str],
    filter_mode: ColumnFilterMode | None,
    matched_columns: list[str],
    missing_columns: list[str],
    available_columns: Sequence[str],
) -> list[str]:
    final = list(warnings)
    if filter_mode == "include" and not matched_columns and missing_columns:
        final.append(
            "No columns matched the requested include filter; only available "
            "structure identity columns were retained."
        )
    if missing_columns:
        final.append(missing_columns_warning(missing_columns, available_columns))
    return final


@contextmanager
def _failure_recorded(run_state: RunState) -> Generator[None]:
    """Leave the run in error when the block raises, and re-raise.

    RunPausedError passes through without a record, since the run is then
    left paused. A MemoryError, whose own message is usually empty, is raised
    as a RuntimeError that says the partials are kept for a resume. A record
    that cannot be written, as on a full disk, does not replace the error
    being recorded.
    """
    try:
        yield
    except RunPausedError:
        raise
    except MemoryError as exception:
        message = (
            f"Final assembly ran out of memory: {exception}. The per-structure "
            f"partials are still in runs/{run_state.directory.root.name}/partials/; "
            "resume the run to retry."
        )
        with suppress(OSError):
            run_state.set_run_status("error", error=message)
        raise RuntimeError(message) from exception
    except Exception as exception:
        with suppress(OSError):
            run_state.set_run_status(
                "error", error=str(exception) or type(exception).__name__
            )
        raise


def settle_run(
    *,
    run_state: RunState,
    working_directory: Path,
    output_filename: str | None,
    pdb_ids: Sequence[str],
    pdb_results: list[PdbProcessResult],
    warnings: Sequence[str],
    options: EngineOptions,
    column_filter_mode: ColumnFilterMode | None,
    column_filter_list: Sequence[str],
    paused: bool,
    progress_callback: ProgressCallback | None,
    publish: bool,
    source: DatasetSource,
) -> DatasetBuildResult:
    """Assemble the run's tables, write its reports, and build its result.

    ``run_state`` is the claimed and loaded run. Each table is assembled from
    the partials of the done entries among ``pdb_ids``, in that order. The
    run report and summary are written even when no entry is done.

    ``paused`` says the build stopped before it recorded every entry. The run
    is then left paused, and otherwise completed. The result's ``paused``
    says which.

    With ``publish``, the tables are also copied into the working directory,
    the primary table as ``output_filename``. When that is None, the run
    keeps the name it was last published under, or else takes the first free
    automatic name. Without ``publish``, the result points at the tables in
    the run directory and ``output_filename`` is not used.

    The result's warnings are ``warnings`` followed by those about the column
    filter.

    A paused run with no done entry raises RunPausedError and is left paused.
    Any other run with no done entry raises NoDatasetError. Assembly that
    runs out of memory raises RuntimeError. Any other error that stops
    assembly, publishing, or the reports is raised as it is. Every error but RunPausedError leaves
    the run in error with its message recorded, unless the record itself
    cannot be written.
    """
    with _failure_recorded(run_state):
        directory = run_state.directory
        entries = entries_of(run_state.manifest())
        registry = get_engine_registry()
        engine_tables = registry.enabled_table_specs(options, source)
        table_names = dataset_shapes.output_table_names(options, source)
        primary_table = table_names[0]
        run_state.set_engine_tables(
            [
                {**table.to_record(), "engine": registry.table_owner(table.name)}
                for table in engine_tables
            ]
        )

        partials_by_table = _done_partials(directory, entries, pdb_ids, table_names)
        done_count = len(partials_by_table[primary_table])

        available_columns = run_state.available_columns()
        matched_columns, missing_columns = resolve_requested_columns(
            available_columns, column_filter_list
        )
        column_filter = (
            None
            if column_filter_mode is None
            else ColumnFilterSummary(
                mode=column_filter_mode,
                matched_columns=matched_columns,
                missing_columns=missing_columns,
                requested_columns=list(column_filter_list),
                available_columns=available_columns,
            )
        )
        final_warnings = _final_warnings(
            warnings,
            column_filter_mode,
            matched_columns,
            missing_columns,
            available_columns,
        )
        if paused and done_count:
            final_warnings.append(
                "Run paused with a partial dataset assembled from "
                f"{done_count} completed structure(s). Resume to continue."
            )

        if not done_count:
            _write_reports(
                run_state,
                options=options,
                source=source,
                engine_tables=engine_tables,
                pdb_results=pdb_results,
                output_paths_by_table={},
                columns_by_table={},
                row_counts_by_table={},
                column_filter=column_filter,
                warnings=final_warnings,
            )
            if paused:
                run_state.set_run_status("paused")
                raise RunPausedError(
                    "Run paused before any structure completed; no dataset assembled."
                )
            raise NoDatasetError(_no_data_message(pdb_ids, entries, warnings))

        recorded_columns = run_state.canonical_columns_by_table()
        column_order_by_table: dict[str, list[str]] = {
            # A table's columns come in the order its partials give them, and
            # the partials in the order of pdb_ids, so the same entries give
            # the same file whichever download finished first. The columns
            # recorded as entries were built serve only a table with no
            # partial.
            table: [] if partials_by_table[table] else recorded_columns.get(table, [])
            for table in table_names
        }
        for engine_table in engine_tables:
            # An engine table's columns keep the order the engine declares, not
            # the order the partials happened to record them in.
            column_order_by_table[engine_table.name] = list(engine_table.column_names)
        output_paths_by_table = {
            table: directory.dataset_path_for(table) for table in table_names
        }
        partial_count = sum(len(partials) for partials in partials_by_table.values())
        run_state.report_progress(
            progress_callback, "assemble", 0, partial_count, "dataset"
        )
        row_counts = assemble_final_csvs(
            partials_by_table,
            output_paths_by_table,
            column_order_by_table,
            _assembly_progress_reporter(run_state, progress_callback),
        )
        columns_by_table = {
            table: read_csv_columns(path)
            for table, path in output_paths_by_table.items()
        }
        for table, columns in columns_by_table.items():
            run_state.set_canonical_columns(table, columns)

        output_receipts = [
            file_receipt(path, role=f"dataset_{table}")
            for table, path in output_paths_by_table.items()
        ]
        primary_path = output_paths_by_table[primary_table]
        additional_paths = {
            table: output_paths_by_table[table] for table in table_names[1:]
        }
        if publish:
            run_state.report_progress(progress_callback, "finalize", 0, 1, "dataset")
            # A run settled again is published over its earlier copy.
            published = publish_tables(
                working_directory,
                output_filename or run_state.published_filename(),
                output_paths_by_table,
            )
            run_state.set_published_filename(published.primary.name)
            run_state.report_progress(
                progress_callback, "finalize", 1, 1, published.primary.name
            )
            primary_path = published.primary
            additional_paths = published.additional
            output_receipts.extend(published.receipts)
        run_state.set_output_receipts(output_receipts)

        _write_reports(
            run_state,
            options=options,
            source=source,
            engine_tables=engine_tables,
            pdb_results=pdb_results,
            output_paths_by_table=output_paths_by_table,
            columns_by_table=columns_by_table,
            row_counts_by_table=row_counts,
            column_filter=column_filter,
            warnings=final_warnings,
        )
        run_state.set_run_status("paused" if paused else "completed")

        return DatasetBuildResult(
            csv_path=primary_path,
            run_report_path=directory.report_path,
            run_summary_path=directory.summary_path,
            atom_count=row_counts[primary_table],
            column_count=len(columns_by_table[primary_table]),
            pdb_ids=list(pdb_ids),
            preview_rows=read_preview_rows(
                output_paths_by_table[primary_table], limit=PREVIEW_ROW_COUNT
            ),
            warnings=final_warnings,
            pdb_results=pdb_results,
            column_filter=column_filter,
            additional_csv_paths=additional_paths,
            paused=paused,
        )


__all__ = ["PREVIEW_ROW_COUNT", "settle_run"]
