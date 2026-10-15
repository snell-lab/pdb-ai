"""Running one phase of engines on a structure, and merging their results.

Engines run one after another on a shared EngineContext, in the order the
registry gives. Each result is checked against its engine's declarations
before its columns are merged or its artifacts published. An undeclared
column, table, report field, or artifact is refused, and so is a column
named twice or an artifact of a type its declaration does not give. So is a
table that breaks its own declaration, and so are atom columns that would
change which rows the atom DataFrame holds. A refused result counts as a
failure of its engine and changes nothing.

A failed engine that is not required still leaves outputs of a fixed form,
as run_engine_phase describes. So every structure of a run writes the same
declared columns and tables, whatever failed. The partial files of one table
then share one header, and a failure reads as missing values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import pandas as pd

from .contracts import (
    ColumnDtype,
    Engine,
    EngineContext,
    EnginePhase,
    EngineResult,
    EngineSpec,
    ReportValue,
    SourceFile,
)
from .registry import EngineRegistry
from .tables import validate_tables


def _failure_message(exception: Exception) -> str:
    # Some exceptions, such as a bare RuntimeError, carry no text. The type's
    # name then says at least what went wrong.
    return str(exception) or type(exception).__name__


class EngineExecutionError(RuntimeError):
    """The failure of a required engine, which fails the entry.

    ``engine_name`` names the engine, and the error it raised is the cause.
    """

    def __init__(self, engine_name: str, cause: Exception) -> None:
        """Record the engine's name, with ``cause`` as the underlying error."""
        super().__init__(f"Engine {engine_name!r} failed: {_failure_message(cause)}")
        self.engine_name = engine_name
        self.__cause__ = cause


@dataclass(frozen=True)
class EnginePhaseResult:
    """What the engines of one phase produced besides atom columns."""

    warnings: tuple[str, ...] = ()
    # Every table of every engine that ran, by table name, checked and in
    # fixed form. A failed engine's tables have no rows.
    tables: dict[str, pd.DataFrame] = field(default_factory=dict[str, pd.DataFrame])
    # Every declared report field of every engine that ran, by field name.
    report: dict[str, ReportValue] = field(default_factory=dict[str, ReportValue])
    source_files: tuple[SourceFile, ...] = ()

    def merged(self, other: EnginePhaseResult) -> EnginePhaseResult:
        """Return this result followed by another phase's result.

        Where both results hold a key, the other result's value wins.
        """
        return EnginePhaseResult(
            warnings=(*self.warnings, *other.warnings),
            tables={**self.tables, **other.tables},
            report={**self.report, **other.report},
            source_files=(*self.source_files, *other.source_files),
        )


def _merge_result_columns(
    context: EngineContext, spec: EngineSpec, result: EngineResult
) -> None:
    # Every check comes before the atom DataFrame is replaced, so a refused
    # result leaves it as it was.
    columns = result.columns
    if columns is None:
        return
    if columns.columns.empty:
        raise ValueError(
            f"Engine {spec.name!r} returned a DataFrame with no columns; "
            "return columns=None to contribute nothing"
        )
    # The atom DataFrame names its columns by text, so the result's names are
    # read as text too.
    names = [str(name) for name in columns.columns]
    columns = columns.set_axis(names, axis="columns")
    # A name given twice would reach the atom DataFrame twice, and no later
    # step could tell which of the two to keep.
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(
            f"Engine {spec.name!r} returned columns more than once: {repeated}"
        )
    declared = {column.name for column in spec.columns}
    undeclared = [name for name in names if name not in declared]
    if undeclared:
        raise ValueError(
            f"Engine {spec.name!r} returned undeclared columns: {undeclared}"
        )
    # Later engines and the partial files rely on the atom rows staying as
    # they were, so the result must not add, drop, or reorder any.
    if not columns.index.equals(context.atom_dataframe.index):
        raise ValueError(
            f"Engine {spec.name!r} returned columns that do not use the atom "
            "DataFrame's index"
        )
    # A replaced column keeps its position, and new columns follow the
    # existing ones.
    context.atom_dataframe = context.atom_dataframe.assign(
        **{name: columns[name] for name in names}
    )


def _validated_report(spec: EngineSpec, result: EngineResult) -> dict[str, ReportValue]:
    declared = set(spec.report_field_names)
    undeclared = sorted(set(result.report) - declared)
    if undeclared:
        raise ValueError(
            f"Engine {spec.name!r} reported undeclared fields: {undeclared}"
        )
    for name, value in result.report.items():
        # Checked at run time, since an engine from outside PDB-AI may not be
        # type-checked.
        raw = cast(object, value)
        if raw is not None and not isinstance(raw, (bool, int, float, str)):
            raise ValueError(
                f"Engine {spec.name!r} reported {raw!r} of type "
                f"{type(raw).__name__} for {name!r}; a report value must be a "
                "Boolean, a number, text, or None"
            )
    return {name: result.report.get(name) for name in spec.report_field_names}


def _check_result_artifacts(spec: EngineSpec, result: EngineResult) -> None:
    declared = {key.name: key for key in spec.provides}
    undeclared = sorted(
        key.name for key in result.artifacts if key.name not in declared
    )
    if undeclared:
        raise ValueError(
            f"Engine {spec.name!r} published undeclared artifacts: {undeclared}"
        )
    for key in result.artifacts:
        # The store finds a value by its key's name. Under a key of another
        # type, a value of the wrong type could be published, and a consumer
        # reading it with the declared key would get a TypeError, not a value.
        if key.type is not declared[key.name].type:
            raise ValueError(
                f"Engine {spec.name!r} published artifact {key.name!r} under a "
                f"key of type {key.type.__name__}; it declares type "
                f"{declared[key.name].type.__name__}"
            )
    # The store checks each type as it publishes, but by then the columns
    # are merged and earlier artifacts are published.
    for key, value in result.artifacts.items():
        if not isinstance(value, key.type):
            raise ValueError(
                f"Engine {spec.name!r} published artifact {key.name!r} of type "
                f"{type(value).__name__}; its key requires type {key.type.__name__}"
            )


def _failure_report(spec: EngineSpec, message: str) -> dict[str, ReportValue]:
    return {
        report_field.name: (
            message if report_field.carries_message else report_field.on_failure
        )
        for report_field in spec.report_fields
    }


def _missing_column(dtype: ColumnDtype, index: pd.Index) -> pd.Series:
    # The column holds the type it would have had, so a later engine that
    # reads it need not allow for a failed one. A Boolean column stays
    # nullable Boolean and a numeric one stays a number.
    if dtype == "boolean":
        return pd.Series(pd.NA, index=index, dtype="boolean")
    if dtype == "numeric":
        return pd.Series(float("nan"), index=index, dtype="float64")
    return pd.Series(pd.NA, index=index, dtype=object)


def _fill_failed_columns(context: EngineContext, spec: EngineSpec) -> None:
    # A refused result changes nothing, so a column the atoms already hold
    # came from the parser or an earlier engine and is kept. Only the
    # enabled columns are filled, as the others are dropped before output.
    filler = {
        column.name: _missing_column(column.dtype, context.atom_dataframe.index)
        for column in spec.columns
        if column.name not in context.atom_dataframe.columns
        and column.is_enabled(context.options, source=context.source)
    }
    if filler:
        context.atom_dataframe = pd.concat(
            [
                context.atom_dataframe,
                pd.DataFrame(filler, index=context.atom_dataframe.index),
            ],
            axis=1,
        )


def _run_one(
    engine: Engine, context: EngineContext
) -> tuple[EngineResult, dict[str, pd.DataFrame], dict[str, ReportValue]]:
    spec = engine.spec
    result = engine.run(context)
    # Everything that can be checked alone is checked before the columns
    # merge. _merge_result_columns checks the columns before it replaces the atom
    # DataFrame. So a refused result leaves the context as it was.
    tables = validate_tables(spec.name, spec.tables, result.tables)
    report = _validated_report(spec, result)
    _check_result_artifacts(spec, result)
    _merge_result_columns(context, spec, result)
    context.artifacts.update(cast(Any, result.artifacts))
    return result, tables, report


def run_engine_phase(
    context: EngineContext,
    phase: EnginePhase,
    *,
    registry: EngineRegistry,
) -> EnginePhaseResult:
    """Run one phase's engines on a structure and report what they produced.

    The engines are those the registry orders for ``phase``, the run's
    options, and the dataset source, with their dependencies. Each runs
    after the engines it depends on. Each one's columns are merged into
    context.atom_dataframe, and its artifacts are published to
    context.artifacts, before the next one runs.

    An engine fails when it raises, or when its result is refused. A required
    engine's failure raises EngineExecutionError, with the engine's error as
    its cause, and the engines after it do not run. Any other engine's
    failure message is recorded in context.failures under the engine's name,
    and a warning names it. The engines after it still run. Its enabled
    columns that the atoms do not already hold are added with only missing
    values, its tables have no rows, and its report fields take their
    failure values. A failed engine's source files are not reported.
    """
    warnings: list[str] = []
    tables: dict[str, pd.DataFrame] = {}
    report: dict[str, ReportValue] = {}
    source_files: list[SourceFile] = []
    for engine in registry.engines_to_run(
        phase=phase, options=context.options, source=context.source
    ):
        spec = engine.spec
        try:
            result, engine_tables, engine_report = _run_one(engine, context)
        except Exception as exception:
            if spec.required:
                raise EngineExecutionError(spec.name, exception) from exception
            message = _failure_message(exception)
            context.failures[spec.name] = message
            warnings.append(f"{spec.display_name} failed: {message}")
            report.update(_failure_report(spec, message))
            tables.update(validate_tables(spec.name, spec.tables, {}))
            _fill_failed_columns(context, spec)
            continue
        warnings.extend(result.warnings)
        tables.update(engine_tables)
        report.update(engine_report)
        source_files.extend(result.source_files)
    return EnginePhaseResult(
        warnings=tuple(warnings),
        tables=tables,
        report=report,
        source_files=tuple(source_files),
    )


__all__ = [
    "EngineExecutionError",
    "EnginePhaseResult",
    "run_engine_phase",
]
