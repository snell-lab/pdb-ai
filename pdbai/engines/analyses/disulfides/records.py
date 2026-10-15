"""The deposited disulfide records of a structure file, read from _struct_conn.

read_disulfide_records reads the _struct_conn category of the flat mmCIF
dictionary, never the atom table. The parser repeats a one-row category on
every atom, but a record has no positional link to the atoms, so a structure
with one record would otherwise look like a structure with one per atom.

Each text is read as clean_string_series reads atom identity: stripped, and
empty when the file writes a placeholder. A record is kept when its
connection type is disulf, in any case, and it joins two SG atoms of
cysteine. A partner names its residue by the author component, or by the
label component when that is missing, and its atom by the label atom name,
or by the author one. A record that names another residue or atom, such as
a cysteine joined to mercaptoethanol or glutathione, or an oxidized
cysteine, is outside what the table holds and is skipped without a count.

A record joins two atoms of one molecule when both partners take the same
symmetry operator, since one operator applied to both leaves their distance
unchanged. A partner without an operator takes the identity. A record whose
partners take different operators joins the molecule to a symmetry mate. It
is counted and otherwise set aside, since the atoms of the file alone cannot
show its geometry. A disulfide across a two-fold axis names one residue
twice under two operators, so this check comes before any check that the
two residues differ.

Each kept record holds both partners' author and label addresses, and
compute.py decides which one finds the residue. The alternate location and
the distance a record states are not read. A row summarizes every conformer
of its two residues, and its distances are measured from the coordinates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from ....structure.residue_identity import clean_string_series
from .constants import (
    CYSTEINE_RESIDUE_NAME,
    DISULFIDE_CONNECTION_TYPE,
    IDENTITY_SYMMETRY_OPERATOR,
    SULFUR_ATOM_NAME,
)

_STRUCT_CONN_PREFIX = "_struct_conn."


@dataclass(frozen=True)
class PartnerAddress:
    """Where one partner of a record sits, by its author and its label fields.

    Each field is stripped text, empty when the file gives no value.
    """

    chain_id: str
    res_num: str
    ins_code: str
    label_asym_id: str
    label_seq_id: str

    @property
    def has_author_fields(self) -> bool:
        """Return whether the author chain and residue number are both given."""
        return bool(self.chain_id and self.res_num)


@dataclass(frozen=True)
class DisulfideRecord:
    """A deposited disulfide that joins two cysteine SG atoms of one molecule."""

    record_id: str
    first: PartnerAddress
    second: PartnerAddress


@dataclass(frozen=True)
class DisulfideRecords:
    """The disulfide records of one structure file.

    ``records`` holds the records within one molecule, in file order.
    ``symmetry_record_count`` counts the cysteine disulfides that join the
    molecule to a symmetry mate.
    """

    records: tuple[DisulfideRecord, ...]
    symmetry_record_count: int


@dataclass(frozen=True)
class _Partner:
    """One partner of a _struct_conn row, as the rules of this module read it."""

    address: PartnerAddress
    # Whether the partner may be a cysteine SG. It may when each of its
    # component and atom names is either missing or the one the table is about.
    in_scope: bool
    # The symmetry operator without its underscore, so 1_555 and 1555 agree.
    operator: str


def _normalized_operator(text: str) -> str:
    """Return a symmetry operator without underscores, the identity if empty."""
    return (text or IDENTITY_SYMMETRY_OPERATOR).replace("_", "")


def _partners(category: pd.DataFrame, number: int) -> list[_Partner]:
    """Return partner ``number`` (1 or 2) of every row of ``category``."""

    def text(item: str) -> list[str]:
        values: list[str] = clean_string_series(
            category, f"{_STRUCT_CONN_PREFIX}{item}"
        ).tolist()
        return values

    prefix = f"ptnr{number}"
    components = [
        (author or label).upper()
        for author, label in zip(
            text(f"{prefix}_auth_comp_id"), text(f"{prefix}_label_comp_id"), strict=True
        )
    ]
    atoms = [
        label or author
        for label, author in zip(
            text(f"{prefix}_label_atom_id"), text(f"{prefix}_auth_atom_id"), strict=True
        )
    ]
    addresses = [
        PartnerAddress(*fields)
        for fields in zip(
            text(f"{prefix}_auth_asym_id"),
            text(f"{prefix}_auth_seq_id"),
            text(f"pdbx_{prefix}_PDB_ins_code"),
            text(f"{prefix}_label_asym_id"),
            text(f"{prefix}_label_seq_id"),
            strict=True,
        )
    ]
    return [
        _Partner(
            address=address,
            in_scope=component in ("", CYSTEINE_RESIDUE_NAME)
            and atom in ("", SULFUR_ATOM_NAME),
            operator=_normalized_operator(operator),
        )
        for address, component, atom, operator in zip(
            addresses, components, atoms, text(f"{prefix}_symmetry"), strict=True
        )
    ]


def read_disulfide_records(
    mmcif_dict: Mapping[str, Sequence[str | None]],
) -> DisulfideRecords:
    """Return the cysteine disulfide records of a flat mmCIF dictionary.

    A file without _struct_conn, or a row without a connection type, gives no
    record rather than an error. Records outside the table's scope are left
    out, and records to a symmetry mate are only counted, as the module
    describes.
    """
    items = {
        name: list(values)
        for name, values in mmcif_dict.items()
        if name.startswith(_STRUCT_CONN_PREFIX)
    }
    if not items:
        return DisulfideRecords((), 0)
    category = pd.DataFrame(items)
    record_ids: list[str] = clean_string_series(
        category, f"{_STRUCT_CONN_PREFIX}id"
    ).tolist()
    connection_types: list[str] = (
        clean_string_series(category, f"{_STRUCT_CONN_PREFIX}conn_type_id")
        .str.lower()
        .tolist()
    )
    records: list[DisulfideRecord] = []
    symmetry_record_count = 0
    for record_id, connection_type, first, second in zip(
        record_ids,
        connection_types,
        _partners(category, 1),
        _partners(category, 2),
        strict=True,
    ):
        if connection_type != DISULFIDE_CONNECTION_TYPE:
            continue
        if not (first.in_scope and second.in_scope):
            continue
        if first.operator != second.operator:
            symmetry_record_count += 1
            continue
        records.append(DisulfideRecord(record_id, first.address, second.address))
    return DisulfideRecords(tuple(records), symmetry_record_count)


__all__ = [
    "DisulfideRecord",
    "DisulfideRecords",
    "PartnerAddress",
    "read_disulfide_records",
]
