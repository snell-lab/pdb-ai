from __future__ import annotations

from pathlib import Path

import pandas as pd

from pdbai.engines.core import ColumnGranularity
from pdbai.schema.base import BASE_COLUMN_SPECS
from pdbai.structure.mmcif_parser import parse_mmcif_atom_rows


def _parsed(minimal_cif: Path) -> pd.DataFrame:
    return parse_mmcif_atom_rows(minimal_cif, {}, structure_id="1ABC").dataframe


def test_the_parser_writes_every_base_column(minimal_cif: Path) -> None:
    written = set(map(str, _parsed(minimal_cif).columns))

    assert {spec.name for spec in BASE_COLUMN_SPECS} <= written


def test_every_named_column_the_parser_writes_is_declared(minimal_cif: Path) -> None:
    declared = {spec.name for spec in BASE_COLUMN_SPECS}
    named = {
        column
        for column in map(str, _parsed(minimal_cif).columns)
        if not column.startswith("_")
    }

    assert named <= declared


def test_the_parser_writes_declared_boolean_columns_as_booleans(
    minimal_cif: Path,
) -> None:
    dataframe = _parsed(minimal_cif)

    for spec in BASE_COLUMN_SPECS:
        if spec.column.dtype == "boolean":
            assert pd.api.types.is_bool_dtype(dataframe[spec.name].dtype), spec.name


def test_base_columns_are_the_parsers_and_have_no_engine() -> None:
    assert all(spec.engine is None for spec in BASE_COLUMN_SPECS)
    assert {spec.owner for spec in BASE_COLUMN_SPECS} == {"parser"}


def test_every_key_role_is_covered_by_the_star_schema_identity_columns() -> None:
    def columns(role: str) -> set[str]:
        return {spec.name for spec in BASE_COLUMN_SPECS if role in spec.roles}

    identity = columns("residue_author_identity") | columns("optional_residue_key")

    for role in ("chain_author_identity", "chain_identity", "structure_key"):
        assert columns(role) <= identity, role


def test_the_numbering_column_is_per_structure_wherever_chain_id_is_kept() -> None:
    columns = {spec.name: spec.column for spec in BASE_COLUMN_SPECS}

    numbering = columns["residue_numbering"]
    assert numbering.granularity is ColumnGranularity.STRUCTURE
    assert numbering.dtype == "text"
    assert numbering.profiles == columns["chain_id"].profiles
