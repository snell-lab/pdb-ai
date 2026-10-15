"""Annotation, the step that adds the engines' columns to a parsed structure.

Annotation comes after parsing and the technique check. Each enabled
annotation engine, such as SASA, RABDAM, or the PDB-Redo comparison, reads
the structure's atom rows and what the acquisition engines fetched. It then
adds its own columns, and it may also fill tables of its own.

The rows are conformed to the run's schema only after the last engine has
run, since any engine may add or change a column. The run's column filter is
not applied here. The build records every column an entry offers before it
filters them, so annotation hands on the full set.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..engines.catalog import get_engine_registry
from ..engines.core import (
    EngineContext,
    EnginePhase,
    EnginePhaseResult,
    run_engine_phase,
)
from ..schema import disabled_columns, normalize_boolean_columns, order_columns


@dataclass(frozen=True)
class AnnotatedStructure:
    """One structure's annotated atom rows and what the engines reported."""

    # Every column the run's options do not disable, in display order. The
    # run's column filter has not been applied yet.
    dataframe: pd.DataFrame
    # Only annotation engines write tables, so this result holds every engine
    # table of the structure.
    phase_result: EnginePhaseResult


def annotate_structure(
    atom_dataframe: pd.DataFrame, context: EngineContext
) -> AnnotatedStructure:
    """Run the annotation engines on one parsed structure and conform its rows.

    ``context`` carries what the host and the acquisition engines published.
    Columns the run's options disable are dropped, and the rest are put in
    display order. A required engine's failure raises EngineExecutionError. A
    declared Boolean column that holds anything but True, False, 1, 0, or a
    missing value raises ValueError, and so does a repeated column name.
    """
    context.atom_dataframe = atom_dataframe
    phase_result = run_engine_phase(
        context, EnginePhase.ANNOTATION, registry=get_engine_registry()
    )
    dataframe = context.atom_dataframe.drop(
        columns=list(disabled_columns(context.options, source=context.source)),
        errors="ignore",
    )
    # No engine runs after this point, so none can leave 1.0, 0.0, or NaN in
    # a column declared nullable Boolean.
    dataframe = normalize_boolean_columns(dataframe)
    return AnnotatedStructure(
        dataframe[order_columns(list(map(str, dataframe.columns)))], phase_result
    )


__all__ = ["AnnotatedStructure", "annotate_structure"]
