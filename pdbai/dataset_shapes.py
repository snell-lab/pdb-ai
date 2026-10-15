"""Splitting a structure's atom rows into the tables of its dataset shape.

Every engine annotates the same atom DataFrame. The dataset shape decides only
how that DataFrame is written. The flat shape writes it as it is, with one
row per atom. Each residue, chain, and entry value repeats on every atom it
describes. The star schema writes each value once, in five tables: atoms,
residues, chains, structures, and citations. Keys built from each row's
canonical residue and chain identity link those tables. In either shape the
engine tables follow the base tables unchanged.

Citation columns get a table of their own. That keeps an entry's scientific
values apart from its bibliography. A row with no canonical residue or chain
identity gets a surrogate key of its own. The key is built from the row's
entry, model, and position, so the same rows always get the same keys. One
key shared by all such rows would make one residue out of unrelated atoms.
Dropping the rows would lose atoms that the flat shape keeps.

The keys come from residue_identity.py, which builds the same key for a
residue whether its structure's chain_id and res_num hold author or label
values. So the split needs no rule of its own for a structure whose
residue_numbering is label. Its named identity columns are copied to the
residues and chains tables as they are, and residue_numbering goes to the
structures table to say how to read them.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence

import pandas as pd

from .engines.catalog import get_engine_registry
from .engines.core import (
    DATASET_SHAPES,
    ColumnGranularity,
    DatasetShapeName,
    DatasetSource,
    EngineOptions,
)
from .schema import (
    CHAIN_JOIN_COLUMN,
    RESIDUE_JOIN_COLUMN,
    column_granularity,
    columns_with_role,
)
from .structure.residue_identity import (
    ResidueIdentities,
    build_residue_identities,
)

logger = logging.getLogger(__name__)

# The columns of each role, which ColumnRole describes.
_RESIDUE_AUTHOR_IDENTITY_COLUMNS = columns_with_role("residue_author_identity")
_OPTIONAL_RESIDUE_KEY_COLUMNS = columns_with_role("optional_residue_key")
_CHAIN_AUTHOR_IDENTITY_COLUMNS = columns_with_role("chain_author_identity")
_CHAIN_IDENTITY_COLUMNS = columns_with_role("chain_identity")
_STRUCTURE_KEY_COLUMNS = columns_with_role("structure_key")
_ATOM_IDENTITY_COLUMNS = columns_with_role("atom_identity")

# The columns the star schema builds its keys from. The chain and structure
# key roles name a subset of them, so this covers every identity key column.
_KEY_SOURCE_COLUMNS = (
    *_RESIDUE_AUTHOR_IDENTITY_COLUMNS,
    *_OPTIONAL_RESIDUE_KEY_COLUMNS,
)

# Every key column. These go with the tables they key, whatever their
# granularity.
_KEY_COLUMNS = frozenset({RESIDUE_JOIN_COLUMN, CHAIN_JOIN_COLUMN, *_KEY_SOURCE_COLUMNS})

# Name prefixes of the mmCIF bibliography categories. "_citation" covers
# _citation, _citation_author, and _citation_editor. _audit_author lists the
# entry's depositors.
_CITATION_CATEGORY_PREFIXES = ("_citation", "_audit_author")

# Name prefix of the parser's helper columns, such as _pdbai_entity_type.
# They say whether a row belongs to a polymer, which decides its residue key.
# They are not declared columns and carry no role, so they are found by name.
_PARSER_HELPER_PREFIX = "_pdbai_"


def output_table_names(options: EngineOptions, source: DatasetSource) -> list[str]:
    """Return every table a run writes, the base tables before the engine ones.

    The first name is the primary table. The names differ between the dataset
    shapes only in the base tables.
    """
    base = DATASET_SHAPES[options.dataset_shape].tables
    tables = get_engine_registry().enabled_table_specs(options, source)
    return [*base, *(table.name for table in tables)]


def _with_row_surrogates(
    keys: pd.Series, identities: ResidueIdentities, label: str
) -> pd.Series:
    # The surrogate names the entry, the model, and the row's position. So it
    # is unique within the DataFrame, and the same on every split.
    invalid = keys.isna()
    if not invalid.any():
        return keys
    positions = pd.Series(range(len(keys)), index=keys.index).astype(str)
    prefix = identities.pdb_id.astype(str) + "|" + identities.model.astype(str) + "|"
    return keys.where(~invalid, label + "|" + prefix + positions)


def _with_join_columns(atom_dataframe: pd.DataFrame) -> pd.DataFrame:
    identities = build_residue_identities(atom_dataframe)
    invalid_residues = int(identities.residue_join_keys.isna().sum())
    if invalid_residues:
        logger.warning(
            "Star-schema normalization found %d atom row(s) without canonical "
            "residue identity; preserving them as deterministic per-row "
            "surrogates rather than merging them.",
            invalid_residues,
        )
    dataframe = atom_dataframe.copy()
    dataframe[RESIDUE_JOIN_COLUMN] = _with_row_surrogates(
        identities.residue_join_keys, identities, "invalid-residue-row"
    )
    dataframe[CHAIN_JOIN_COLUMN] = _with_row_surrogates(
        identities.chain_keys, identities, "invalid-chain-row"
    )
    return dataframe


def _granularity(column: str) -> ColumnGranularity:
    # A column the schema does not declare is taken to be per atom. The
    # exception is an mmCIF item (_category.item) outside _atom_site. The
    # parser repeats such an item on every atom, since it describes the whole
    # entry. An item of a bibliography category goes to the citations table,
    # and any other goes to structures.
    declared = column_granularity(column)
    if declared is not None:
        return declared
    is_mmcif_item = column.startswith("_") and "." in column
    if is_mmcif_item and column.startswith(_CITATION_CATEGORY_PREFIXES):
        return ColumnGranularity.CITATION
    if is_mmcif_item and not column.startswith("_atom_site."):
        return ColumnGranularity.STRUCTURE
    return ColumnGranularity.ATOM


def _columns_by_granularity(
    columns: Iterable[str],
) -> dict[ColumnGranularity, list[str]]:
    by_granularity: dict[ColumnGranularity, list[str]] = {
        granularity: [] for granularity in ColumnGranularity
    }
    for column in columns:
        if column not in _KEY_COLUMNS:
            by_granularity[_granularity(column)].append(column)
    return by_granularity


def _present_columns(dataframe: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [column for column in columns if column in dataframe.columns]


def _one_row_per(
    dataframe: pd.DataFrame,
    level: ColumnGranularity,
    *,
    key: Sequence[str],
    identity: Sequence[str] = (),
    values: Sequence[str],
) -> pd.DataFrame:
    """Return one row per distinct ``key``, with key, identity, and values.

    Each column holds the first value in its group that is not missing. So a
    value that only some rows carry is kept, even when the group's first row
    lacks it. Every value column should be constant within one ``level``. A
    column that varies within one gets a warning that suggests a finer
    granularity. Its first value is still kept.
    """
    # The groups are not sorted, and a group whose key is missing is kept.
    # That way they match the rows of drop_duplicates below, in the same
    # order. nunique skips missing values, so a value missing from only some
    # rows of a group does not count as varying.
    groups = dataframe.groupby(list(key), sort=False, dropna=False)
    counts = groups[list(values)].nunique()
    for column in counts.columns[counts.gt(1).any()]:
        logger.warning(
            "Star-schema normalization found column %r varying within one %s; "
            "only the first value present is kept. Declare a finer granularity "
            "for the column.",
            column,
            level.value,
        )
    rows = (
        dataframe[[*key, *identity, *values]]
        .drop_duplicates(subset=list(key))
        .reset_index(drop=True)
    )
    # A group's first row rarely lacks a value. On a large entry, first()
    # over every column costs about as much again as the nunique above. So
    # first() is used only for the columns that have a gap.
    gaps = [column for column in (*identity, *values) if rows[column].isna().any()]
    if gaps:
        firsts = groups[gaps].first().reset_index(drop=True)
        for column in gaps:
            rows[column] = firsts[column]
    return rows


def _normalize_to_star_schema(atom_dataframe: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Split atom rows into atoms, residues, chains, structures, and citations.

    Every column goes to the table of its granularity. Atoms link to residues
    by residue_key, and residues to chains by chain_key. Structures and
    citations are keyed by the structure key. The atoms and residues tables
    start with the residue's author identity and its mmCIF identity items.
    The chains table starts with the chain's author and label identity. A
    DataFrame without every residue author identity column raises KeyError.
    """
    missing = [
        column
        for column in _RESIDUE_AUTHOR_IDENTITY_COLUMNS
        if column not in atom_dataframe.columns
    ]
    if missing:
        raise KeyError(
            "Star-schema normalization requires the residue author identity columns "
            f"{', '.join(_RESIDUE_AUTHOR_IDENTITY_COLUMNS)}; missing: {', '.join(missing)}"
        )
    dataframe = _with_join_columns(atom_dataframe)
    by_granularity = _columns_by_granularity(map(str, dataframe.columns))
    residue_identity = [
        *_RESIDUE_AUTHOR_IDENTITY_COLUMNS,
        *_present_columns(dataframe, _OPTIONAL_RESIDUE_KEY_COLUMNS),
    ]
    atom_identity = _present_columns(dataframe, _ATOM_IDENTITY_COLUMNS)
    atom_values = [
        column
        for column in by_granularity[ColumnGranularity.ATOM]
        if column not in atom_identity
    ]
    atoms = dataframe[
        [RESIDUE_JOIN_COLUMN, *residue_identity, *atom_identity, *atom_values]
    ]
    return {
        "atoms": atoms.reset_index(drop=True),
        "residues": _one_row_per(
            dataframe,
            ColumnGranularity.RESIDUE,
            key=[RESIDUE_JOIN_COLUMN],
            identity=[CHAIN_JOIN_COLUMN, *residue_identity],
            values=by_granularity[ColumnGranularity.RESIDUE],
        ),
        "chains": _one_row_per(
            dataframe,
            ColumnGranularity.CHAIN,
            key=[CHAIN_JOIN_COLUMN],
            identity=[
                *_CHAIN_AUTHOR_IDENTITY_COLUMNS,
                *_present_columns(dataframe, _CHAIN_IDENTITY_COLUMNS),
            ],
            values=by_granularity[ColumnGranularity.CHAIN],
        ),
        "structures": _one_row_per(
            dataframe,
            ColumnGranularity.STRUCTURE,
            key=_STRUCTURE_KEY_COLUMNS,
            values=by_granularity[ColumnGranularity.STRUCTURE],
        ),
        "citations": _one_row_per(
            dataframe,
            ColumnGranularity.STRUCTURE,
            key=_STRUCTURE_KEY_COLUMNS,
            values=by_granularity[ColumnGranularity.CITATION],
        ),
    }


def with_key_source_columns(
    filtered: pd.DataFrame, unfiltered: pd.DataFrame, dataset_shape: DatasetShapeName
) -> pd.DataFrame:
    """Return a column-filtered DataFrame with the columns its shape needs.

    A star-schema split builds its keys from the key source columns, the
    residue author identity and optional residue key columns, and from the
    parser's helper columns. Any of these that the filter removed are
    taken back from ``unfiltered`` and added at the end. A flat DataFrame
    needs none of them and is returned as it is.
    """
    if dataset_shape != "star_schema":
        return filtered
    needed = [
        *_KEY_SOURCE_COLUMNS,
        *(
            column
            for column in unfiltered.columns
            if column.startswith(_PARSER_HELPER_PREFIX)
        ),
    ]
    removed = [
        column
        for column in needed
        if column in unfiltered.columns and column not in filtered.columns
    ]
    if not removed:
        return filtered
    return pd.concat([filtered, unfiltered[removed]], axis=1)


def base_tables(
    atom_dataframe: pd.DataFrame, dataset_shape: DatasetShapeName
) -> dict[str, pd.DataFrame]:
    """Split one structure's atom rows into the base tables of a dataset shape.

    The flat shape's one table is the DataFrame itself. A star-schema split
    raises KeyError when the DataFrame lacks a residue author identity column.
    """
    if dataset_shape == "star_schema":
        return _normalize_to_star_schema(atom_dataframe)
    return {"flat": atom_dataframe}


def filtered_base_tables(
    filtered: pd.DataFrame, unfiltered: pd.DataFrame, dataset_shape: DatasetShapeName
) -> dict[str, pd.DataFrame]:
    """Split column-filtered atom rows into the base tables of a dataset shape.

    The star schema takes back the columns its keys are built from, as
    with_key_source_columns does. The key source columns stay in the tables,
    since they identify each row. The parser's helper columns only decide
    which residues are polymer, so those the filter removed are dropped again
    once the keys are built.
    """
    restored = with_key_source_columns(filtered, unfiltered, dataset_shape)
    helpers = [
        column
        for column in restored.columns
        if column.startswith(_PARSER_HELPER_PREFIX) and column not in filtered.columns
    ]
    tables = base_tables(restored, dataset_shape)
    if not helpers:
        return tables
    return {
        name: table.drop(columns=[column for column in helpers if column in table])
        for name, table in tables.items()
    }


__all__ = [
    "base_tables",
    "filtered_base_tables",
    "output_table_names",
    "with_key_source_columns",
]
