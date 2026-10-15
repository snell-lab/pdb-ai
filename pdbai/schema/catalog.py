"""The one catalog of every declared dataset column, in display order.

The catalog starts from declaration order. The base columns come first, as
base.py lists them. Each engine's columns follow, in the order the engines
were registered. The columns are then gathered by display group, and each
group sits where its first column falls. An engine column in a base group
therefore sits among the base columns.

Each column name has one owner. When two engines declare one name, the second
is refused as it registers. When an engine column takes a base column's name,
every function here raises ValueError. A name the catalog does not declare,
such as most raw mmCIF items, has no specification, granularity, or
description here.

The catalog is built on first use. It is rebuilt whenever the registered
engines change, as when code outside PDB-AI registers an engine or a test
installs another registry. The check compares each engine declaration by
identity, not by name. A new registry whose engines reuse the old names is
therefore still noticed.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from ..engines.catalog import get_engine_registry
from ..engines.core import ColumnGranularity, DatasetSource, EngineOptions, EngineSpec
from .base import BASE_COLUMN_SPECS
from .contracts import ColumnRole, DatasetColumnSpec


@dataclass(frozen=True)
class _Catalog:
    # The engine declarations the catalog was built from, kept alive so that
    # their identities stay a valid cache key.
    engines: tuple[EngineSpec, ...]
    columns: tuple[DatasetColumnSpec, ...]
    by_name: dict[str, DatasetColumnSpec]
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    boolean_columns: frozenset[str]


_cached: _Catalog | None = None


def _build_catalog(engines: tuple[EngineSpec, ...]) -> _Catalog:
    # A dict keeps its keys in insertion order. Each group therefore sits
    # where its first column falls, and its columns keep declaration order.
    declared: dict[str, DatasetColumnSpec] = {}
    grouped: dict[str, list[DatasetColumnSpec]] = {}
    engine_columns = (
        DatasetColumnSpec(column, engine)
        for engine in engines
        for column in engine.columns
    )
    for spec in (*BASE_COLUMN_SPECS, *engine_columns):
        existing = declared.get(spec.name)
        if existing is not None:
            owners = (
                f"twice by {spec.owner!r}"
                if existing.owner == spec.owner
                else f"by both {existing.owner!r} and {spec.owner!r}"
            )
            raise ValueError(f"Column {spec.name!r} is declared {owners}")
        declared[spec.name] = spec
        grouped.setdefault(spec.column.group, []).append(spec)
    columns = tuple(spec for specs in grouped.values() for spec in specs)
    return _Catalog(
        engines=engines,
        columns=columns,
        by_name=declared,
        groups=tuple(
            (group, tuple(spec.name for spec in specs))
            for group, specs in grouped.items()
        ),
        boolean_columns=frozenset(
            spec.name for spec in columns if spec.column.dtype == "boolean"
        ),
    )


def _same_engines(
    built_from: tuple[EngineSpec, ...], registered: tuple[EngineSpec, ...]
) -> bool:
    return len(built_from) == len(registered) and all(
        mine is theirs for mine, theirs in zip(built_from, registered, strict=True)
    )


def _catalog() -> _Catalog:
    # Every column lookup goes through here, so the catalog is cached. It is
    # rebuilt only when the registered engines change.
    global _cached
    engines = get_engine_registry().registered_specs
    if _cached is None or not _same_engines(_cached.engines, engines):
        _cached = _build_catalog(engines)
    return _cached


def _enabled_column_specs(
    options: EngineOptions, *, source: DatasetSource
) -> tuple[DatasetColumnSpec, ...]:
    return tuple(
        spec for spec in _catalog().columns if spec.is_enabled(options, source=source)
    )


def declared_column_specs() -> tuple[DatasetColumnSpec, ...]:
    """Return every declared column in display order."""
    return _catalog().columns


def columns_by_group() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return each display group's name with the names of its columns.

    Both the groups and the names within each are in display order.
    """
    return _catalog().groups


def order_columns(columns: Sequence[str]) -> list[str]:
    """Return the columns in display order.

    Columns the catalog does not declare follow the declared ones, in the order
    they were given. Raises ValueError when a name repeats, since repeated
    columns cannot be ordered by name.
    """
    counts = Counter(columns)
    repeated = sorted(column for column, count in counts.items() if count > 1)
    if repeated:
        raise ValueError(
            f"Cannot order columns with repeated name(s): {', '.join(repeated)}"
        )
    catalog = _catalog()
    return [
        *(spec.name for spec in catalog.columns if spec.name in counts),
        *(column for column in columns if column not in catalog.by_name),
    ]


def get_column_spec(name: str) -> DatasetColumnSpec | None:
    """Return the specification of a column, or None if it is undeclared."""
    return _catalog().by_name.get(name)


def column_granularity(name: str) -> ColumnGranularity | None:
    """Return the granularity of a column, or None if it is undeclared."""
    spec = get_column_spec(name)
    return spec.column.granularity if spec is not None else None


def describe_column(name: str) -> str:
    """Return the description of a named column.

    A column the catalog does not declare has an empty description.
    """
    spec = get_column_spec(name)
    return spec.column.description if spec is not None else ""


def declared_boolean_columns() -> frozenset[str]:
    """Return the names of the columns declared to hold Boolean values."""
    return _catalog().boolean_columns


def columns_with_role(role: ColumnRole) -> tuple[str, ...]:
    """Return the names of the columns that play a star-schema role.

    They are in display order, which is the order the star schema lists its key
    columns in.
    """
    return tuple(spec.name for spec in _catalog().columns if role in spec.roles)


def disabled_columns(options: EngineOptions, *, source: DatasetSource) -> set[str]:
    """Return the names of the declared columns a run does not write.

    A column is written when the run's options enable it and it applies to the
    run's source.
    """
    enabled = {spec.name for spec in _enabled_column_specs(options, source=source)}
    return {spec.name for spec in _catalog().columns if spec.name not in enabled}


def schema_fingerprint(options: EngineOptions, *, source: DatasetSource) -> str:
    """Return a short hash of the columns and engine tables a run writes.

    It covers the name, type, granularity, and star-schema roles of each
    enabled declared column, in display order. It also covers the declaration
    of each enabled engine table. Partials written under two fingerprints
    cannot be safely assembled together. This holds even when the run's
    settings are equal, since another version of PDB-AI may have renamed or
    retyped a column or moved it to another table.
    """
    columns = [
        [
            spec.name,
            spec.column.dtype,
            spec.column.granularity.value,
            sorted(spec.roles),
        ]
        for spec in _enabled_column_specs(options, source=source)
    ]
    tables = [
        table.to_record()
        for table in get_engine_registry().enabled_table_specs(options, source)
    ]
    blob = json.dumps({"columns": columns, "tables": tables}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


__all__ = [
    "column_granularity",
    "columns_by_group",
    "columns_with_role",
    "declared_boolean_columns",
    "declared_column_specs",
    "describe_column",
    "disabled_columns",
    "get_column_spec",
    "order_columns",
    "schema_fingerprint",
]
