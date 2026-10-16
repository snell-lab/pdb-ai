"""A declared column as the application schema sees it.

DatasetColumnSpec pairs a column declaration with the engine that owns it, or
with none for a base column the parser writes. It answers which switch the
column is attributed to and whether a run with given options and source writes
it.

Only a base column plays a role in building the star schema's keys. The keys
are built in every run, and an engine's columns exist only when the engine
runs. Two kinds of name are reserved. One is the name of each join column the
split assigns, and the other is the owner name of the base columns. A clash
with either is refused where it is declared, before it can reach a written
dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..engines.core import ColumnSpec, DatasetSource, EngineOptions, EngineSpec

# The parts a base column can play in the star-schema split, which
# dataset_shapes reads through columns_with_role:
#   residue_author_identity  Required. Identifies a residue by its named
#                            columns and leads the atoms and residues tables.
#                            These hold author fields, or label fields for a
#                            structure whose residue_numbering is label.
#   optional_residue_key     The model and the mmCIF label items that follow
#                            the author identity where the DataFrame has them.
#   chain_author_identity    Leads the chains table.
#   chain_identity           The chain's label identity and model, repeated in
#                            the chains table where the DataFrame has them.
#   structure_key            Keys the structures and citations tables.
#   atom_identity            The atoms table's first columns after its keys.
# The join keys are built from the residue_author_identity and
# optional_residue_key columns, so a column filter never removes them from a
# star-schema run. The chain_author_identity, chain_identity and structure_key
# columns must be a subset of those, so every key column is kept too.
type ColumnRole = Literal[
    "atom_identity",
    "chain_author_identity",
    "chain_identity",
    "optional_residue_key",
    "residue_author_identity",
    "structure_key",
]

# The owner reported for a base column, so no engine may take it as its name.
PARSER_OWNER = "parser"

# The star schema builds these join columns from each row's canonical
# identity. They link atoms to residues and residues to chains. The split
# assigns them, so no column may be declared under either name.
RESIDUE_JOIN_COLUMN = "residue_key"
CHAIN_JOIN_COLUMN = "chain_key"
JOIN_COLUMNS: tuple[str, ...] = (RESIDUE_JOIN_COLUMN, CHAIN_JOIN_COLUMN)


@dataclass(frozen=True)
class DatasetColumnSpec:
    """One declared column, placed in the application schema with its owner.

    A base column belongs to the parser and has no engine. An engine column
    keeps its owning engine, which gates it. A run writes the column only when
    the engine is enabled and applies to the run's source, and the column's own
    declaration can only narrow that.

    Raises ValueError for a column named after a star-schema join column and
    for an engine named "parser".
    """

    column: ColumnSpec
    # None for a base column, which the parser writes.
    engine: EngineSpec | None
    # Empty for a base column that plays no part, and always for an engine
    # column.
    roles: frozenset[ColumnRole] = frozenset()

    def __post_init__(self) -> None:
        """Refuse a name the star schema or the base columns reserve."""
        name = self.column.name
        if name in JOIN_COLUMNS:
            raise ValueError(
                f"Column name {name!r} is reserved for a star-schema join column"
            )
        if self.engine is not None and self.engine.name == PARSER_OWNER:
            raise ValueError(
                f"Engine name {PARSER_OWNER!r} is reserved for the owner of the "
                "base columns"
            )

    @property
    def name(self) -> str:
        """Return the column name."""
        return self.column.name

    @property
    def owner(self) -> str:
        """Return the name of the engine that declares the column.

        A base column returns "parser".
        """
        return PARSER_OWNER if self.engine is None else self.engine.name

    @property
    def toggle(self) -> str | None:
        """Return the switch the column is attributed to.

        The column's own toggle wins over its engine's. None means no switch is
        attributed, as for a base column that names none.
        """
        if self.column.toggle is not None or self.engine is None:
            return self.column.toggle
        return self.engine.toggle

    def is_enabled(self, options: EngineOptions, *, source: DatasetSource) -> bool:
        """Return whether a run writes the column under these options.

        The column's sources and its engine's sources both have to include the
        run's source.
        """
        if self.engine is not None and (
            source not in self.engine.sources or not self.engine.is_enabled(options)
        ):
            return False
        return self.column.is_enabled(options, source=source)


__all__ = [
    "CHAIN_JOIN_COLUMN",
    "JOIN_COLUMNS",
    "PARSER_OWNER",
    "RESIDUE_JOIN_COLUMN",
    "ColumnRole",
    "DatasetColumnSpec",
]
