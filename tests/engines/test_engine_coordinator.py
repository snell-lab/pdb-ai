from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pdbai.engines.core import (
    ArtifactKey,
    ColumnSpec,
    EngineContext,
    EngineExecutionError,
    EngineOptions,
    EnginePhase,
    EngineRegistry,
    EngineResult,
    EngineSpec,
    ReportField,
    run_engine_phase,
)


@dataclass
class FunctionEngine:
    """An engine whose result is computed by a function of the context."""

    spec: EngineSpec
    produce: Callable[[EngineContext], EngineResult]
    calls: list[str] = field(default_factory=list[str])

    def run(self, context: EngineContext) -> EngineResult:
        self.calls.append(self.spec.name)
        return self.produce(context)


def _raising(exception: Exception) -> Callable[[EngineContext], EngineResult]:
    def produce(_context: EngineContext) -> EngineResult:
        raise exception

    return produce


def _spec(
    name: str,
    *,
    columns: tuple[ColumnSpec, ...] = (),
    phase: EnginePhase = EnginePhase.ANNOTATION,
    dependencies: tuple[str, ...] = (),
    required: bool = False,
    default_enabled: bool = True,
    provides: tuple[ArtifactKey[Any], ...] = (),
    report_fields: tuple[ReportField, ...] = (),
) -> EngineSpec:
    return EngineSpec(
        name=name,
        display_name=name.title(),
        phase=phase,
        columns=columns,
        dependencies=dependencies,
        required=required,
        default_enabled=default_enabled,
        provides=provides,
        report_fields=report_fields,
    )


def _context() -> EngineContext:
    return EngineContext(
        pdb_id="1ABC",
        atom_dataframe=pd.DataFrame(
            {"chain_id": ["A", "A", "B"], "score": [0.0, 0.0, 0.0], "tail": [1, 2, 3]},
            index=[10, 20, 30],
        ),
        options=EngineOptions(),
    )


def _run(context: EngineContext, *engines: FunctionEngine) -> Any:
    return run_engine_phase(
        context, EnginePhase.ANNOTATION, registry=EngineRegistry(engines)
    )


def _score_column(name: str = "score") -> ColumnSpec:
    return ColumnSpec(name, group="Test Fields", dtype="numeric")


def test_a_result_naming_a_column_twice_is_refused() -> None:
    context = _context()
    engine = FunctionEngine(
        _spec("twice", columns=(_score_column("extra"),), required=True),
        lambda context: EngineResult(
            columns=pd.DataFrame(
                [[1.0, 2.0]] * 3, columns=["extra", "extra"], index=[10, 20, 30]
            )
        ),
    )

    with pytest.raises(EngineExecutionError, match=r"more than once: \['extra'\]"):
        _run(context, engine)
    assert "extra" not in context.atom_dataframe.columns


def test_a_replaced_column_keeps_its_position() -> None:
    context = _context()
    engine = FunctionEngine(
        _spec("scorer", columns=(_score_column(), _score_column("new"))),
        lambda context: EngineResult(
            columns=pd.DataFrame(
                {"new": [1.0, 1.0, 2.0], "score": [0.5, 0.5, 0.25]},
                index=context.atom_dataframe.index,
            )
        ),
    )

    _run(context, engine)

    atoms = context.atom_dataframe
    assert list(atoms.columns) == ["chain_id", "score", "tail", "new"]
    assert atoms["score"].tolist() == [0.5, 0.5, 0.25]
    assert atoms["new"].tolist() == [1.0, 1.0, 2.0]
    assert atoms.index.tolist() == [10, 20, 30]


def test_a_required_failure_without_text_is_named_by_its_type() -> None:
    engine = FunctionEngine(
        _spec("silent", required=True),
        _raising(RuntimeError()),
    )

    with pytest.raises(EngineExecutionError) as caught:
        _run(_context(), engine)

    assert str(caught.value) == "Engine 'silent' failed: RuntimeError"
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_a_required_failure_stops_the_engines_after_it() -> None:
    calls: list[str] = []
    first = FunctionEngine(
        _spec("first", required=True),
        _raising(RuntimeError("offline")),
        calls=calls,
    )
    second = FunctionEngine(
        _spec("second", dependencies=("first",)),
        lambda _context: EngineResult(),
        calls=calls,
    )

    with pytest.raises(EngineExecutionError, match="'first' failed: offline"):
        _run(_context(), second, first)
    assert calls == ["first"]


def test_a_phase_runs_its_engines_after_their_dependencies() -> None:
    # The consumer is registered first and reads the producer's column, so it
    # sees the column only when the producer runs first.
    calls: list[str] = []
    fetcher = FunctionEngine(
        _spec("fetcher", phase=EnginePhase.ACQUISITION, default_enabled=False),
        lambda _context: EngineResult(),
        calls=calls,
    )
    producer = FunctionEngine(
        _spec("producer", columns=(_score_column("base"),), default_enabled=False),
        lambda context: EngineResult(
            columns=pd.DataFrame(
                {"base": [1.0, 2.0, 3.0]}, index=context.atom_dataframe.index
            )
        ),
        calls=calls,
    )
    consumer = FunctionEngine(
        _spec(
            "consumer",
            columns=(_score_column("doubled"),),
            dependencies=("producer", "fetcher"),
        ),
        lambda context: EngineResult(
            columns=pd.DataFrame({"doubled": context.atom_dataframe["base"] * 2})
        ),
        calls=calls,
    )
    registry = EngineRegistry([consumer, producer, fetcher])
    context = _context()

    run_engine_phase(context, EnginePhase.ANNOTATION, registry=registry)

    # The acquisition dependency belongs to the other phase, so it does not
    # run here.
    assert calls == ["producer", "consumer"]
    assert context.atom_dataframe["doubled"].tolist() == [2.0, 4.0, 6.0]


def test_a_failed_engine_fills_its_columns_with_their_own_types() -> None:
    columns = (
        ColumnSpec("failed_number", group="Test Fields", dtype="numeric"),
        ColumnSpec("failed_flag", group="Test Fields", dtype="boolean"),
        ColumnSpec("failed_text", group="Test Fields"),
        # A column the atoms already hold is kept as it is.
        ColumnSpec("tail", group="Test Fields", dtype="numeric"),
    )
    engine = FunctionEngine(
        _spec("failing", columns=columns),
        _raising(RuntimeError("no data")),
    )
    context = _context()

    _run(context, engine)

    atoms = context.atom_dataframe
    assert context.failures == {"failing": "no data"}
    assert atoms["failed_number"].dtype == np.float64
    assert isinstance(atoms["failed_flag"].dtype, pd.BooleanDtype)
    assert atoms["failed_text"].dtype == object
    assert atoms[["failed_number", "failed_flag", "failed_text"]].isna().all().all()
    assert atoms["tail"].tolist() == [1, 2, 3]


def test_a_report_value_of_another_type_is_refused_by_its_type() -> None:
    engine = FunctionEngine(
        _spec("counter", report_fields=(ReportField("counter_total", "Total."),)),
        lambda _context: EngineResult(report={"counter_total": np.int64(3)}),  # type: ignore[dict-item]
    )

    context = _context()

    report = _run(context, engine)

    assert context.failures["counter"] == (
        "Engine 'counter' reported np.int64(3) of type int64 for 'counter_total'; "
        "a report value must be a Boolean, a number, text, or None"
    )
    assert report.report == {"counter_total": None}
