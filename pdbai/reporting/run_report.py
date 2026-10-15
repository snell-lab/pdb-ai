"""The run report, a CSV file with one row for each entry of a run.

The report lets a reader see what became of every requested entry without
opening the manifest. Each row starts with what the host did with the entry.
The report fields of the engines the run enables follow, in engine order.
Last comes each engine table's row count, in a column named after the table
with the suffix _row_count. Flags read True or False, and the experimental
methods and radiation sources are joined with "; ".

The columns follow from the run's settings rather than from what each entry
reported, so every row has the same columns. A cell is empty when its step was
not tried, its engine did not report, or the entry is unfinished and its
tables have no count. An unknown count is left empty rather than written as
0, which would claim the table holds no rows for the entry. ReportField
refuses a name the host uses or one ending in _row_count, so the three parts
never share a column. The file is replaced atomically, so a reader never sees
it half-written.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

from ..atomic import write_atomically
from ..engines.core import HOST_REPORT_COLUMNS, ROW_COUNT_COLUMN_SUFFIX, ReportField
from ..results import PdbProcessResult


def _host_cells(result: PdbProcessResult) -> dict[str, object]:
    # One cell for each of HOST_REPORT_COLUMNS, in the same order.
    return {
        "pdb_id": result.pdb_id,
        "source": result.source,
        "state": result.state,
        "success": result.succeeded(),
        "technique_status": result.technique_status,
        "technique_message": result.technique_message,
        "experimental_methods": "; ".join(result.experimental_methods),
        "radiation_sources": "; ".join(result.radiation_sources),
        "download_success": result.download_succeeded(),
        "download_error": result.download_error,
        "parse_success": result.parse_succeeded(),
        "parse_error": result.parse_error,
        "output_error": result.output_error,
        "metadata_success": result.metadata_succeeded(),
        "metadata_message": result.metadata_message,
        "metadata_level": result.metadata_level,
    }


def write_run_report(
    report_path: Path,
    pdb_results: Sequence[PdbProcessResult],
    *,
    report_fields: Sequence[ReportField],
    engine_table_names: Sequence[str],
) -> None:
    """Write the run report, one row for each result in the order given.

    The report fields are those of the engines the run enables, and the table
    names those of the tables they write. A run with no results gets a report
    with only its header.
    """
    row_count_columns = {
        table_name: f"{table_name}{ROW_COUNT_COLUMN_SUFFIX}"
        for table_name in engine_table_names
    }
    columns = [
        *HOST_REPORT_COLUMNS,
        *(report_field.name for report_field in report_fields),
        *row_count_columns.values(),
    ]
    with (
        write_atomically(report_path) as scratch,
        scratch.open("w", newline="", encoding="utf-8") as report_file,
    ):
        writer = csv.DictWriter(report_file, fieldnames=columns)
        writer.writeheader()
        for result in pdb_results:
            writer.writerow(
                {
                    **_host_cells(result),
                    **{
                        report_field.name: result.engine_report.get(report_field.name)
                        for report_field in report_fields
                    },
                    **{
                        column: result.engine_table_row_counts.get(table_name)
                        for table_name, column in row_count_columns.items()
                    },
                }
            )


__all__ = ["write_run_report"]
