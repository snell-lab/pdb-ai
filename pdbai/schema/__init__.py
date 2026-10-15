"""Every column a dataset can hold, declared once and listed in display order.

contracts.py defines DatasetColumnSpec, which pairs a column declaration with
its owner, either the parser or one engine. A pairing the schema cannot use,
such as one with a misspelt role, raises ValueError.

base.py declares the columns the parser writes. Each engine declares its own
columns.

catalog.py combines the declarations into one catalog. It starts from the
base columns, then adds each engine's columns in the order the engines were
registered. It then gathers each column with the rest of its display group.
The catalog decides which columns a run writes, their order, and their
descriptions. It also gives the fingerprint that stops a run from resuming on
partials written under another schema. The catalog is rebuilt whenever the
registered engines change.

boolean_columns.py checks every column declared boolean, once the last engine
has run. Such a column must end up holding True, False, or a missing value.
normalize_boolean_columns converts 1 and 0 and raises ValueError for any other
value.

A column name has one owner. When two engines declare the same name, the
second is refused as it registers. When an engine takes the name of a base
column, the catalog raises ValueError the next time it is read.

This package imports the engine registry. An engine must therefore not import
this package, or the two would import each other.
"""

from .boolean_columns import normalize_boolean_columns
from .catalog import (
    column_granularity,
    columns_by_group,
    columns_with_role,
    declared_boolean_columns,
    declared_column_specs,
    describe_column,
    disabled_columns,
    get_column_spec,
    order_columns,
    schema_fingerprint,
)
from .contracts import (
    CHAIN_JOIN_COLUMN,
    RESIDUE_JOIN_COLUMN,
    ColumnRole,
    DatasetColumnSpec,
)

__all__ = [
    "CHAIN_JOIN_COLUMN",
    "RESIDUE_JOIN_COLUMN",
    "ColumnRole",
    "DatasetColumnSpec",
    "column_granularity",
    "columns_by_group",
    "columns_with_role",
    "declared_boolean_columns",
    "declared_column_specs",
    "describe_column",
    "disabled_columns",
    "get_column_spec",
    "normalize_boolean_columns",
    "order_columns",
    "schema_fingerprint",
]
