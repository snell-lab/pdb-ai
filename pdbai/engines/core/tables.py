"""Checking engine tables, and one shared identifier for an atom site.

An engine returns each of its tables as a DataFrame in whatever form it
built it. validate_tables brings each one into the fixed form its TableSpec
declares, or refuses it. So every structure writes every table of the
engines it runs, with the same columns in the same order and types.

Rows in different engine tables often describe the same atom, so they share
one way to name it. site_id joins the entry, model, chain, residue number,
insertion code, atom name, and alternate conformation with colons, as in
1ABC:1:A:301::FE: for the iron of a heme at residue 301. A rerun gives the
same identifier. Key columns beside site_id keep the parts apart, so that a
row can join back to the atom level of the dataset.

The chain, residue number, and residue name are the atom table's chain_id,
res_num, and res_name, whichever numbering the structure uses. They are
author values for most structures and label values for a structure whose
residue_numbering is label. Nothing here reads an author item directly, so
an identifier needs no rule of its own for such a structure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from ...missing_text import MISSING_TEXTS
from ...nullable_booleans import to_nullable_boolean
from .contracts import TableColumnSpec, TableLink, TableSpec

SITE_ID_COLUMN = "site_id"

# Each tuple holds a key column's name, the atom DataFrame column it comes
# from, its dtype, and its description. The order is that of site_id, with
# the residue name added. The insertion code and the alternate location keep
# their raw mmCIF names in the dataset. The table columns get short names and
# link back through the raw ones.
_SITE_KEY_SOURCES: tuple[tuple[str, str, str, str], ...] = (
    ("pdb_id", "pdb_id", "string", "Four-character PDB entry accession."),
    ("model_num", "model_num", "Int64", "Coordinate-model number."),
    (
        "chain_id",
        "chain_id",
        "string",
        "Chain identifier, as the atom-level chain_id column holds it.",
    ),
    (
        "res_num",
        "res_num",
        "string",
        "Residue number, as the atom-level res_num column holds it.",
    ),
    (
        "ins_code",
        "_atom_site.pdbx_PDB_ins_code",
        "string",
        "Author-assigned insertion code; empty when absent.",
    ),
    (
        "res_name",
        "res_name",
        "string",
        "Residue or component name, as the atom-level res_name column holds it.",
    ),
    ("atom_name", "atom_name", "string", "Atom name."),
    (
        "altloc",
        "_atom_site.label_alt_id",
        "string",
        "Alternate-conformation identifier; empty when the atom has none.",
    ),
)

# The residue name is a key column but not part of site_id. The other parts
# already identify the atom, and leaving it out keeps site_id short.
_SITE_ID_PARTS = (
    "pdb_id",
    "model_num",
    "chain_id",
    "res_num",
    "ins_code",
    "atom_name",
    "altloc",
)


def site_key_columns(prefix: str = "") -> tuple[TableColumnSpec, ...]:
    """Return the column specs that identify one atom in an engine table.

    The columns are site_id, then the parts it is built from with the residue
    name added. Each name starts with ``prefix``, such as "ligand_" for the
    second atom of a bond. The columns match those site_key_dataframe
    returns.
    """
    site_id = TableColumnSpec(
        name=f"{prefix}{SITE_ID_COLUMN}",
        description=(
            "Stable atom-site identifier, "
            "entry:model:chain:residue:insertion:atom:altloc."
        ),
    )
    return (
        site_id,
        *(
            TableColumnSpec(
                name=f"{prefix}{name}",
                dtype=dtype,
                description=description,
            )
            for name, _source, dtype, description in _SITE_KEY_SOURCES
        ),
    )


def site_atom_link(prefix: str = "", *, role: str = "") -> TableLink:
    """Return the link from the site key columns with ``prefix`` to the atoms.

    ``role`` tells this link apart from another link to atoms in one table.
    """
    return TableLink(
        target="atom",
        columns=tuple(f"{prefix}{name}" for name, *_ in _SITE_KEY_SOURCES),
        target_columns=tuple(source for _name, source, *_ in _SITE_KEY_SOURCES),
        role=role,
    )


def _identifier_text(values: pd.Series) -> pd.Series:
    # A number read from mmCIF can arrive as a float such as 1.0, and an
    # identifier needs "1". Every text that stands for a missing value
    # becomes empty.

    def convert(value: object) -> str:
        if value is None or value is pd.NA:
            return ""
        if isinstance(value, float):
            if value != value:
                return ""
            if value.is_integer():
                return str(int(value))
        text = str(value).strip()
        return "" if text.lower() in MISSING_TEXTS else text

    return values.map(convert).astype("string")


def site_key_dataframe(atoms: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Return site_id and the site key columns for the given atom rows.

    The result keeps the rows' index, and its column names start with
    ``prefix``. A part the atom DataFrame has no column for is empty, and so
    is a missing value. A model number that is not a whole number is missing
    from the model_num column, while site_id keeps its text.
    """
    parts: dict[str, pd.Series] = {}
    for name, source, _dtype, _description in _SITE_KEY_SOURCES:
        if source in atoms.columns:
            text = _identifier_text(atoms[source])
        else:
            text = pd.Series("", index=atoms.index, dtype="string")
        parts[name] = text
    site_id = parts[_SITE_ID_PARTS[0]]
    for name in _SITE_ID_PARTS[1:]:
        site_id = site_id + ":" + parts[name]

    dataframe = pd.DataFrame(index=atoms.index)
    dataframe[f"{prefix}{SITE_ID_COLUMN}"] = site_id
    for name, _source, dtype, _description in _SITE_KEY_SOURCES:
        values = parts[name]
        if dtype == "Int64":
            numbers = pd.to_numeric(values.mask(values == ""), errors="coerce")
            # The cast to Int64 would cut 1.5 down to 1 without a word.
            dataframe[f"{prefix}{name}"] = numbers.where(numbers % 1 == 0).astype(
                "Int64"
            )
        else:
            dataframe[f"{prefix}{name}"] = values
    return dataframe


def _numbers(values: pd.Series, *, whole: bool) -> pd.Series:
    # A Boolean is refused although pandas reads it as 1 or 0, since True is
    # not a count or a measurement.
    if pd.api.types.is_bool_dtype(values.dtype) or any(
        isinstance(value, (bool, np.bool_)) for value in values.tolist()
    ):
        raise ValueError("it holds a Boolean")
    numbers = pd.to_numeric(values, errors="raise")
    if whole:
        # The cast to Int64 would cut 1.5 down to 1 without a word.
        present = numbers.dropna()
        fractional = present[present % 1 != 0]
        if not fractional.empty:
            raise ValueError(f"{fractional.tolist()[0]!r} is not a whole number")
        return numbers.astype("Int64")
    return numbers.astype("float64")


def _cast(dataframe: pd.DataFrame, column: TableColumnSpec, owner: str) -> pd.Series:
    values = dataframe[column.name]
    try:
        if column.dtype == "string":
            return values.astype("string")
        if column.dtype == "boolean":
            return to_nullable_boolean(values, column=column.name)
        return _numbers(values, whole=column.dtype == "Int64")
    except (TypeError, ValueError) as exception:
        raise ValueError(
            f"{owner} column {column.name!r} cannot be read as {column.dtype}: {exception}"
        ) from exception


def _validate_table(
    engine_name: str, spec: TableSpec, dataframe: pd.DataFrame
) -> pd.DataFrame:
    """Check one returned table against its spec and return it in fixed form.

    The result has every declared column, in declared order and type, and an
    index that counts from 0. A declared column the engine left out holds
    only missing values. An undeclared column raises ValueError. So do a
    column given twice, a missing key or link column, a missing or duplicate
    key, and a value outside its column's categories or type. A Boolean in a
    number column and a fraction in an Int64 column are outside its type.
    """
    owner = f"Engine {engine_name!r} table {spec.name!r}"
    names = [str(column) for column in dataframe.columns]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"{owner} returned columns more than once: {repeated}")
    declared = set(spec.column_names)
    undeclared = [str(column) for column in dataframe.columns if column not in declared]
    if undeclared:
        raise ValueError(f"{owner} returned undeclared columns: {undeclared}")
    required = [*spec.key, *spec.link_columns]
    missing = [column for column in dict.fromkeys(required) if column not in dataframe]
    if missing:
        raise ValueError(f"{owner} is missing key or link columns: {missing}")

    dataframe = dataframe.reset_index(drop=True)
    fixed = spec.empty_dataframe().reindex(range(len(dataframe)))
    for column in spec.columns:
        if column.name in dataframe.columns:
            fixed[column.name] = _cast(dataframe, column, owner)

    key = list(spec.key)
    if fixed[key].isna().any().any():
        raise ValueError(f"{owner} has rows with missing key values")
    if fixed.duplicated(subset=key).any():
        raise ValueError(f"{owner} returned duplicate rows for key {key}")
    for column in spec.columns:
        if column.categories is None:
            continue
        values = fixed[column.name].dropna()
        unknown = sorted(set(values) - set(column.categories))
        if unknown:
            raise ValueError(
                f"{owner} column {column.name!r} holds undeclared categories: {unknown}"
            )
    return fixed


def validate_tables(
    engine_name: str,
    specs: Sequence[TableSpec],
    returned: Mapping[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Return every declared table of an engine, each in fixed form.

    ``specs`` are the engine's declared tables and ``returned`` the tables it
    gave, by name. The result holds every declared table in declared order.
    A declared table the engine did not return comes back with no rows. A
    table the engine does not declare raises ValueError, and so does a
    returned table that does not fit its declaration.
    """
    by_name = {spec.name: spec for spec in specs}
    unknown = sorted(set(returned) - set(by_name))
    if unknown:
        raise ValueError(
            f"Engine {engine_name!r} returned undeclared tables: {unknown}"
        )
    return {
        spec.name: (
            _validate_table(engine_name, spec, returned[spec.name])
            if spec.name in returned
            else spec.empty_dataframe()
        )
        for spec in specs
    }


__all__ = [
    "SITE_ID_COLUMN",
    "site_atom_link",
    "site_key_columns",
    "site_key_dataframe",
    "validate_tables",
]
