"""The run summary, a Markdown file written beside a run's tables.

The summary is the account of a run written for a person to read. It gives
what each setting did, the column filter, the run's warnings, the manifest's
reproducibility record, each entry's outcome, the engine tables, and a data
dictionary of every assembled table.

The summary names no engine. Each engine setting writes its own line. Each
report field marked for the summary becomes a column of the status table.
Each engine table and its columns are described by the engine's table spec.
A new engine is therefore summarized without a change here.

The reproducibility log is rendered from the run manifest, not from the live
run, so it shows the record that manifest.json keeps. A field the manifest
lacks reads as missing and does not stop the summary. Every value is made to
fit one table cell, so text from a structure file or a server cannot break a
table. The data dictionary reads each table back from disk, so it describes
the files the run wrote. It groups a dataset table's columns as the schema
catalog groups them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from ..atomic import write_text_atomically
from ..engines.core import (
    ALL_XRAY_SOURCES,
    DATASET_SHAPES,
    EngineOptions,
    OptionSpec,
    ReportField,
    ReportKind,
    ReportValue,
    TableLink,
    TableSpec,
    get_technique,
    get_xray_source,
)
from ..json_values import object_dict, object_list
from ..results import ColumnFilterSummary, PdbProcessResult
from ..schema import columns_by_group, describe_column
from ..selection_filters import SEARCH_FILTERS
from .column_statistics import (
    MAX_DISTINCT_COUNT,
    ColumnStatistics,
    read_column_statistics,
)

# What a table cell shows when its value is missing, empty, or not applicable.
_MISSING = "n/a"

# A whole float smaller than this is written as an integer. A larger one would
# print as a long run of digits, so it is written in exponent form instead.
_LARGEST_WRITTEN_WHOLE = 1e15

# The significant figures kept for a number that is not whole. Seven keep the
# three decimals of an mmCIF coordinate up to 9999.999 Å.
_SIGNIFICANT_FIGURES = 7

# The header of each data dictionary table. _data_dictionary_row gives one
# cell for each, in this order. The section's introduction explains them.
_DATA_DICTIONARY_HEADERS = (
    "Column",
    "Description",
    "Type",
    "% Null",
    "Minimum",
    "Maximum",
    "Mean",
    "Standard deviation",
    "Distinct values",
)


def _format_number(value: float) -> str:
    # A whole float, such as a serial number read back from a CSV file, is
    # written as an integer, 12 rather than 12.0.
    if value.is_integer() and abs(value) < _LARGEST_WRITTEN_WHOLE:
        return str(int(value))
    return f"{value:.{_SIGNIFICANT_FIGURES}g}"


def _cell(value: object) -> str:
    # A table row must stay on one line, so each run of whitespace becomes
    # one space. A bare "|" would start a new cell, so each one is escaped.
    if value is None:
        return _MISSING
    text = _format_number(value) if isinstance(value, float) else str(value)
    text = " ".join(text.split())
    return text.replace("|", "\\|") if text else _MISSING


def _code(value: object) -> str | None:
    # A hash or identifier is set as code so it reads exactly. A missing one
    # stays None, so its cell reads as missing rather than as empty code.
    if value is None or value == "":
        return None
    return f"`{value}`"


def _joined(*values: object) -> str | None:
    # A value recorded in parts, such as a Python implementation and its
    # version, is shown as the parts present, separated by spaces.
    present = [str(value) for value in values if value is not None and value != ""]
    return " ".join(present) or None


def _row(cells: Iterable[object]) -> str:
    return "| " + " | ".join(_cell(cell) for cell in cells) + " |"


def _table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    # A table with no rows gets one row of missing values, so its header does
    # not stand alone. The blank line after the rows ends the table.
    body = [_row(row) for row in rows]
    return [
        _row(headers),
        "|" + "---|" * len(headers),
        *(body or [_row([None] * len(headers))]),
        "",
    ]


def _append_setting(
    lines: list[str],
    option: OptionSpec,
    option_specs: Sequence[OptionSpec],
    options: EngineOptions,
    indent: str,
) -> None:
    # A setting that refines a flag is listed beneath that flag, and only while
    # the flag is on. Refinements of a refinement nest one level further.
    lines.append(f"{indent}- {option.summary_line(options)}")
    if option.kind != "flag" or not options.flag(option.name):
        return
    for refinement in option_specs:
        if refinement.refines == option.name:
            _append_setting(lines, refinement, option_specs, options, indent + "  ")


def _append_run_configuration(
    lines: list[str], options: EngineOptions, option_specs: Sequence[OptionSpec]
) -> None:
    technique = get_technique(options.technique)
    lines.extend(
        [
            "## Run Configuration",
            "",
            f"- **technique** (`{technique.id}`): an entry was admitted when its "
            f"reported experimental methods include {technique.label} "
            f"(`{technique.method}`) and refused when they name only other "
            "techniques. An entry that reports no method was admitted as "
            "unverified. The entry's RCSB record decided when it named a method; "
            "otherwise the structure file decided once parsed, so such an entry's "
            "file was acquired and parsed before it could be refused. The "
            "Technique column of the per-structure status gives each decision.",
        ]
    )
    if options.xray_source != ALL_XRAY_SOURCES:
        source = get_xray_source(options.xray_source)
        lines.append(
            f"- **xray_source** (`{source.id}`): within X-ray diffraction, only "
            f"entries reporting a {source.label} source (`{source.source}`) were "
            "processed; entries reporting other sources, or none, were refused at "
            "intake."
        )
    # Each engine setting writes its own line, so no engine is named here.
    for option in option_specs:
        if option.refines is None:
            _append_setting(lines, option, option_specs, options, indent="")
    shape = DATASET_SHAPES[options.dataset_shape]
    lines.extend(
        [
            f"- **dataset_shape** (`{options.dataset_shape}`): output as "
            f"{shape.description}",
            "",
        ]
    )


def _names(values: Iterable[str]) -> str:
    return ", ".join(f"`{value}`" for value in values)


def _append_column_filter(
    lines: list[str], column_filter: ColumnFilterSummary | None
) -> None:
    # A profile is applied as its column filter, so this records what a
    # profile kept as well as a filter given by hand.
    lines.extend(["## Column Filter", ""])
    if column_filter is None:
        lines.extend(["No column filter was applied.", ""])
        return
    kept = "kept only" if column_filter.mode == "include" else "removed"
    lines.append(
        f"Mode `{column_filter.mode}`: the tables {kept} the columns matching "
        f"these {len(column_filter.requested_columns)} names and patterns, in "
        f"this order: {_names(column_filter.requested_columns)}."
    )
    if column_filter.missing_columns:
        lines.append(
            "These matched no column and were ignored: "
            f"{_names(column_filter.missing_columns)}."
        )
    lines.append("")


def _append_warnings(lines: list[str], warnings: Sequence[str]) -> None:
    lines.extend(["## Warnings", ""])
    lines.extend([f"- {_cell(warning)}" for warning in warnings] or ["None."])
    lines.append("")


def _append_selection(lines: list[str], selection: Mapping[str, object]) -> None:
    # The cohort came from an archive search. Recording what was asked, of
    # which service, and when lets a reader run the query again and compare
    # its answer with this list.
    if not selection:
        return
    lines.extend(
        [
            "### Entry selection",
            "",
            "The PDB identifiers for this run were selected by searching the "
            "archive rather than typed in.",
            "",
            *_table(
                ["Field", "Recorded value"],
                [
                    ["Criteria", selection.get("description")],
                    ["Service", selection.get("service")],
                    ["Service URL", _code(selection.get("service_url"))],
                    ["Retrieved at (UTC)", selection.get("retrieved_at_iso")],
                    ["Entries matched", selection.get("total_count")],
                    ["Entries kept", selection.get("returned_count")],
                    ["Truncated by a limit", selection.get("truncated")],
                ],
            ),
            "The exact query sent is recorded under `selection.query` in "
            "`manifest.json`.",
            "",
        ]
    )
    # A range filter leaves entries out for reasons the criteria do not show,
    # so a selection that used one says what a range meant.
    filters = object_dict(selection.get("filters"))
    if any(
        filters.get(search_filter.id) is not None for search_filter in SEARCH_FILTERS
    ):
        lines.extend(
            [
                "A range in the criteria is strict. An entry was kept only when "
                "every value it records is in range, so an entry that records "
                "none was left out. The range narrowed the search only, and "
                "the dataset's own columns record what each entry holds.",
                "",
            ]
        )


def _append_dependencies(lines: list[str], session: Mapping[str, object]) -> None:
    lines.extend(["**Installed runtime dependency closure**", ""])
    dependencies_error = session.get("dependencies_error")
    if dependencies_error is not None:
        lines.extend([f"Not recorded: {_cell(dependencies_error)}", ""])
        return
    rows: list[list[object]] = []
    for raw_dependency in object_list(session.get("dependencies")):
        dependency = object_dict(raw_dependency)
        direct = dependency.get("direct")
        relationship = (
            "direct" if direct is True else "transitive" if direct is False else None
        )
        rows.append(
            [
                dependency.get("name"),
                dependency.get("version"),
                relationship,
                dependency.get("status"),
            ]
        )
    lines.extend(_table(["Package", "Version", "Relationship", "Status"], rows))


def _append_reference_data(lines: list[str], session: Mapping[str, object]) -> None:
    # Each engine describes the data files it ships with. The summary lists
    # what was recorded without knowing which engine wrote it. A field holding
    # a list or an object does not fit a cell, so it is left to the manifest.
    for name, raw_receipt in object_dict(session.get("reference_data")).items():
        receipt = object_dict(raw_receipt)
        if not receipt:
            continue
        fields = [[key, value] for key, value in receipt.items() if key != "label"]
        flat_fields = [
            [key, value] for key, value in fields if not isinstance(value, dict | list)
        ]
        lines.extend(
            [
                f"**{_cell(receipt.get('label') or name)}**",
                "",
                *_table(["Field", "Value"], flat_fields),
            ]
        )
        if len(flat_fields) < len(fields):
            lines.extend(
                [
                    "Fields holding lists or objects are recorded in `manifest.json`.",
                    "",
                ]
            )


def _append_execution_session(
    lines: list[str], number: int, session: Mapping[str, object]
) -> None:
    code = object_dict(session.get("code"))
    runtime = object_dict(session.get("runtime"))
    lines.extend(
        [
            f"### Execution session {number}",
            "",
            *_table(
                ["Field", "Recorded value"],
                [
                    ["Captured at (UTC)", session.get("captured_at")],
                    ["Repository", code.get("repository")],
                    ["Git commit", _code(code.get("git_commit"))],
                    ["Git branch", code.get("git_branch")],
                    ["Uncommitted changes present", code.get("git_dirty")],
                    [
                        "Python",
                        _joined(
                            runtime.get("python_implementation"),
                            runtime.get("python_version"),
                        ),
                    ],
                    [
                        "Operating system",
                        _joined(
                            runtime.get("operating_system"), runtime.get("os_release")
                        ),
                    ],
                    ["Machine", runtime.get("machine")],
                ],
            ),
        ]
    )
    _append_dependencies(lines, session)
    _append_reference_data(lines, session)


def _append_input_files(lines: list[str], entries: Mapping[str, object]) -> None:
    rows: list[list[object]] = []
    for pdb_id, raw_entry in entries.items():
        for raw_input in object_list(object_dict(raw_entry).get("inputs")):
            receipt = object_dict(raw_input)
            rows.append(
                [
                    pdb_id,
                    receipt.get("role"),
                    receipt.get("available"),
                    receipt.get("path"),
                    _code(receipt.get("sha256")),
                    receipt.get("size_bytes"),
                    receipt.get("source_url"),
                    receipt.get("file_modified_at"),
                    receipt.get("observed_at"),
                    receipt.get("cache_present_before_acquisition"),
                ]
            )
    headers = [
        "Structure",
        "Role",
        "Available",
        "File",
        "SHA-256",
        "Bytes",
        "Source",
        "File modified (UTC)",
        "Observed (UTC)",
        "Cache existed before acquisition",
    ]
    lines.extend(["### Input sources and files", "", *_table(headers, rows)])


def _append_output_files(lines: list[str], outputs: Sequence[object]) -> None:
    rows: list[list[object]] = []
    for raw_output in outputs:
        receipt = object_dict(raw_output)
        rows.append(
            [
                receipt.get("role"),
                receipt.get("path"),
                _code(receipt.get("sha256")),
                receipt.get("size_bytes"),
                receipt.get("observed_at"),
            ]
        )
    headers = ["Role", "File", "SHA-256", "Bytes", "Created/observed (UTC)"]
    lines.extend(["### Output files", "", *_table(headers, rows)])


def _append_reproducibility_log(
    lines: list[str], manifest: Mapping[str, object]
) -> None:
    provenance = object_dict(manifest.get("provenance"))
    lines.extend(
        [
            "## Reproducibility Log",
            "",
            f"- Run ID: {_cell(_code(manifest.get('run_id')))}",
            f"- Settings hash: {_cell(_code(manifest.get('settings_hash')))}",
            "",
        ]
    )
    _append_selection(lines, object_dict(manifest.get("selection")))
    sessions = object_list(provenance.get("execution_sessions"))
    for number, raw_session in enumerate(sessions, start=1):
        _append_execution_session(lines, number, object_dict(raw_session))
    _append_input_files(lines, object_dict(manifest.get("entries")))
    _append_output_files(lines, object_list(provenance.get("outputs")))


def _technique_cell(result: PdbProcessResult) -> str | None:
    # The cell gives how intake decided the entry, then the methods the entry
    # reported.
    status = result.technique_status
    if status is None:
        return None
    methods = "; ".join(result.experimental_methods)
    return f"{status} ({methods})" if methods else status


def _format_report_value(value: ReportValue, kind: ReportKind) -> ReportValue:
    # A fraction is a share from 0 to 1, shown as a percentage.
    if (
        kind == "fraction"
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    ):
        return f"{value:.1%}"
    return value


def _append_structure_status(
    lines: list[str],
    pdb_results: Sequence[PdbProcessResult],
    report_fields: Sequence[ReportField],
) -> None:
    summary_fields = [
        report_field for report_field in report_fields if report_field.in_summary
    ]
    headers = [
        "PDB ID",
        "Source",
        "State",
        "Technique",
        "Downloaded",
        "Parsed",
        *(report_field.label or report_field.name for report_field in summary_fields),
    ]
    rows = [
        [
            result.pdb_id,
            result.source,
            result.state,
            _technique_cell(result),
            result.download_succeeded(),
            result.parse_succeeded(),
            *(
                _format_report_value(
                    result.engine_report.get(report_field.name), report_field.kind
                )
                for report_field in summary_fields
            ),
        ]
        for result in pdb_results
    ]
    lines.extend(
        [
            "## Per-Structure Status",
            "",
            "A step an entry never reached, such as the download of one refused "
            f"on its RCSB record, reads {_MISSING}.",
            "",
            *_table(headers, rows),
        ]
    )


def _describe_link(link: TableLink) -> str:
    # A link reads like "ligand: ligand_chain_id → chain (chain_id)". The
    # target's columns are named only where they differ from the link's own.
    text = f"{', '.join(link.columns)} → {link.target}"
    if link.resolved_target_columns != link.columns:
        text += f" ({', '.join(link.resolved_target_columns)})"
    return f"{link.role}: {text}" if link.role else text


def _engine_table_owners(manifest: Mapping[str, object]) -> dict[str, object]:
    # Settlement records each engine table in the manifest with the engine
    # that declares it.
    records = [
        object_dict(raw_record)
        for raw_record in object_list(manifest.get("engine_tables"))
    ]
    return {str(record.get("name")): record.get("engine") for record in records}


def _append_engine_tables(
    lines: list[str],
    engine_table_specs: Sequence[TableSpec],
    manifest: Mapping[str, object],
    output_paths_by_table: Mapping[str, Path],
    row_counts_by_table: Mapping[str, int],
) -> None:
    # An engine table's rows are sites or bonds rather than atoms, so the
    # summary says what each holds and how it joins back to the dataset.
    if not engine_table_specs:
        return
    owners = _engine_table_owners(manifest)
    headers = [
        "Table",
        "Engine",
        "File",
        "Rows",
        "Key",
        "Links",
        "Description",
    ]
    rows: list[list[object]] = []
    for table_spec in engine_table_specs:
        path = output_paths_by_table.get(table_spec.name)
        rows.append(
            [
                _code(table_spec.name),
                owners.get(table_spec.name),
                None if path is None else path.name,
                row_counts_by_table.get(table_spec.name),
                ", ".join(table_spec.key),
                "; ".join(_describe_link(link) for link in table_spec.links),
                table_spec.description,
            ]
        )
    lines.extend(
        [
            "## Engine Tables",
            "",
            "Engines that record relationships between atoms write them as "
            "separate tables. Link columns join each row back to the dataset "
            "or to another table of the same engine.",
            "",
            *_table(headers, rows),
        ]
    )


def _column_groups(
    columns: Sequence[str], engine_table_spec: TableSpec | None
) -> list[tuple[str, list[str]]]:
    # An engine table is one group, in the order of its file. Settlement
    # writes that file in the order the table spec declares. A dataset table's
    # columns follow the schema catalog's groups. Columns the catalog does not
    # declare, such as raw mmCIF fields, form a last group of their own.
    if engine_table_spec is not None:
        return [("Columns", list(columns))]
    present = set(columns)
    groups: list[tuple[str, list[str]]] = []
    for group_name, catalog_columns in columns_by_group():
        group_columns = [column for column in catalog_columns if column in present]
        if group_columns:
            groups.append((group_name, group_columns))
    grouped = {
        column for _group_name, group_columns in groups for column in group_columns
    }
    other = [column for column in columns if column not in grouped]
    if other:
        groups.append(("Other / Raw mmCIF Fields", other))
    return groups


def _column_descriptions(
    columns: Sequence[str], engine_table_spec: TableSpec | None
) -> dict[str, str]:
    # Engine-table columns are not in the dataset schema catalog, so their
    # table spec describes them.
    if engine_table_spec is not None:
        return {column.name: column.description for column in engine_table_spec.columns}
    return {column: describe_column(column) for column in columns}


def _data_dictionary_row(
    column: str, description: str, statistics: ColumnStatistics
) -> list[object]:
    value_cells: list[object]
    if statistics.kind == "numeric":
        value_cells = [
            statistics.minimum,
            statistics.maximum,
            statistics.mean,
            statistics.standard_deviation,
        ]
    elif statistics.kind == "text":
        value_cells = [statistics.minimum_length, statistics.maximum_length, None, None]
    else:
        value_cells = [None, None, None, None]
    return [
        column,
        description,
        statistics.kind,
        round(statistics.percent_null, 2),
        *value_cells,
        (
            "high-cardinality"
            if statistics.distinct_count is None
            else statistics.distinct_count
        ),
    ]


def _append_data_dictionary(
    lines: list[str],
    output_paths_by_table: Mapping[str, Path],
    columns_by_table: Mapping[str, Sequence[str]],
    engine_table_specs: Sequence[TableSpec],
) -> None:
    lines.extend(
        [
            "## Data Dictionary",
            "",
            "Summary statistics for each column of every output table in this "
            "run, computed over every row of the table. A text column's Minimum "
            "and Maximum are the lengths of its shortest and longest values. "
            "Standard deviation is the population standard deviation, so it is "
            "0 for a column holding one value, or one value throughout. "
            "Distinct values reads "
            f'"high-cardinality" past {MAX_DISTINCT_COUNT} distinct values, '
            "since counting further would hold every value in memory.",
            "",
        ]
    )
    specs_by_name = {table_spec.name: table_spec for table_spec in engine_table_specs}
    for table, csv_path in output_paths_by_table.items():
        columns = columns_by_table[table]
        engine_table_spec = specs_by_name.get(table)
        statistics = read_column_statistics(csv_path, columns)
        descriptions = _column_descriptions(columns, engine_table_spec)
        lines.extend([f"### Table: `{table}`", ""])
        for group_name, group_columns in _column_groups(columns, engine_table_spec):
            rows = [
                _data_dictionary_row(
                    column, descriptions.get(column, ""), statistics[column]
                )
                for column in group_columns
            ]
            lines.extend(
                [f"**{group_name}**", "", *_table(_DATA_DICTIONARY_HEADERS, rows)]
            )


def write_run_summary(
    summary_path: Path,
    *,
    options: EngineOptions,
    option_specs: Sequence[OptionSpec],
    pdb_results: Sequence[PdbProcessResult],
    report_fields: Sequence[ReportField],
    engine_table_specs: Sequence[TableSpec],
    output_paths_by_table: Mapping[str, Path],
    columns_by_table: Mapping[str, Sequence[str]],
    row_counts_by_table: Mapping[str, int],
    run_manifest: Mapping[str, object],
    column_filter: ColumnFilterSummary | None = None,
    warnings: Sequence[str] = (),
) -> None:
    """Write the run summary to a file, replacing any earlier one.

    The option specs are every declared engine setting. The report fields
    belong to the engines the run enables, and the table specs describe the
    tables those engines write. The manifest supplies the reproducibility log
    and the engine that owns each engine table. An engine table without a row
    count shows its rows as missing. Each table among the output paths is read
    in full for its statistics, and the columns by table must name its
    columns. Raises ValueError when a named column is not in its file.
    ``column_filter`` is the filter the tables were cut with, None for none,
    and ``warnings`` the run's warnings.
    """
    lines = ["# pdbai Run Summary", ""]
    _append_run_configuration(lines, options, option_specs)
    _append_column_filter(lines, column_filter)
    _append_warnings(lines, warnings)
    _append_reproducibility_log(lines, run_manifest)
    _append_structure_status(lines, pdb_results, report_fields)
    _append_engine_tables(
        lines,
        engine_table_specs,
        run_manifest,
        output_paths_by_table,
        row_counts_by_table,
    )
    _append_data_dictionary(
        lines, output_paths_by_table, columns_by_table, engine_table_specs
    )
    write_text_atomically(summary_path, "\n".join(lines))


__all__ = ["write_run_summary"]
