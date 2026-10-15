"""The deposited metal coordination records of a structure file.

read_metal_records reads the _struct_conn category of the flat mmCIF
dictionary and keeps the rows whose connection type is metalc, in any case.
Each text is read as clean_string_series reads atom identity: stripped, and
empty when the file writes a placeholder.

A kept record holds both partners' addresses and whether the two take the
same symmetry operator. A partner without an operator takes the identity.
Partners under one operator are atoms of one molecule, so the distance
between them is that of the file. Partners under different operators join
the molecule to a symmetry mate, and compute.py then measures the nearest
image.

Which partner is the metal is not decided here, since a record does not say,
and the atom table knows each atom's element.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from ....structure.residue_identity import clean_string_series
from .constants import IDENTITY_SYMMETRY_OPERATOR, METAL_CONNECTION_TYPE

_STRUCT_CONN_PREFIX = "_struct_conn."


@dataclass(frozen=True)
class PartnerAddress:
    """Where one partner of a record sits, by its label and author fields.

    Each field is stripped text, empty when the file gives no value.
    """

    label_asym_id: str
    label_seq_id: str
    auth_asym_id: str
    auth_seq_id: str
    ins_code: str
    atom_name: str


@dataclass(frozen=True)
class MetalRecord:
    """One deposited metalc record."""

    record_id: str
    first: PartnerAddress
    second: PartnerAddress
    # Whether both partners take one symmetry operator.
    same_operator: bool


def _partners(category: pd.DataFrame, number: int) -> list[tuple[PartnerAddress, str]]:
    """Return partner ``number`` (1 or 2) of every row, with its operator."""

    def text(item: str) -> list[str]:
        values: list[str] = clean_string_series(
            category, f"{_STRUCT_CONN_PREFIX}{item}"
        ).tolist()
        return values

    prefix = f"ptnr{number}"
    atoms = [
        label or author
        for label, author in zip(
            text(f"{prefix}_label_atom_id"), text(f"{prefix}_auth_atom_id"), strict=True
        )
    ]
    return [
        (
            PartnerAddress(label_asym, label_seq, auth_asym, auth_seq, ins_code, atom),
            (operator or IDENTITY_SYMMETRY_OPERATOR).replace("_", ""),
        )
        for label_asym, label_seq, auth_asym, auth_seq, ins_code, atom, operator in zip(
            text(f"{prefix}_label_asym_id"),
            text(f"{prefix}_label_seq_id"),
            text(f"{prefix}_auth_asym_id"),
            text(f"{prefix}_auth_seq_id"),
            text(f"pdbx_{prefix}_PDB_ins_code"),
            atoms,
            text(f"{prefix}_symmetry"),
            strict=True,
        )
    ]


def read_metal_records(
    mmcif_dict: Mapping[str, Sequence[str | None]],
) -> tuple[MetalRecord, ...]:
    """Return the metalc records of a flat mmCIF dictionary, in file order.

    A file without _struct_conn gives no record rather than an error.
    """
    items = {
        name: list(values)
        for name, values in mmcif_dict.items()
        if name.startswith(_STRUCT_CONN_PREFIX)
    }
    if not items:
        return ()
    category = pd.DataFrame(items)
    record_ids: list[str] = clean_string_series(
        category, f"{_STRUCT_CONN_PREFIX}id"
    ).tolist()
    connection_types: list[str] = (
        clean_string_series(category, f"{_STRUCT_CONN_PREFIX}conn_type_id")
        .str.lower()
        .tolist()
    )
    records: list[MetalRecord] = []
    for record_id, connection_type, (first, first_operator), (
        second,
        second_operator,
    ) in zip(
        record_ids,
        connection_types,
        _partners(category, 1),
        _partners(category, 2),
        strict=True,
    ):
        if connection_type != METAL_CONNECTION_TYPE:
            continue
        records.append(
            MetalRecord(record_id, first, second, first_operator == second_operator)
        )
    return tuple(records)


__all__ = ["MetalRecord", "PartnerAddress", "read_metal_records"]
