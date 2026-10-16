"""The deposited _struct_conn records of a structure file, as site engines read them.

read_struct_conn reads the _struct_conn category of the flat mmCIF
dictionary, never the atom table. The parser repeats a one-row category on
every atom, but a record has no positional link to the atoms, so a structure
with one record would otherwise look like a structure with one per atom.

Each text is read as clean_string_series reads atom identity, so it is
stripped, and empty when the file writes a placeholder. The connection type is
lower-cased, so an engine matches it in any case. A partner names its residue
by the author component, upper-cased, or by the label component when that is
missing, and its atom by the label atom name, or by the author one. A partner
without a symmetry operator takes the identity, and an operator is compared
without its underscore, so 1_555 and 1555 agree. Partners under one operator
are atoms of one molecule, since one operator applied to both leaves their
distance unchanged. Partners under different operators join the molecule to a
symmetry mate.

Which records an engine keeps, and what it does with a record to a symmetry
mate, is the engine's own. The metal-site engine's records.py keeps metalc
records, and the disulfide engine's keeps disulf records between cysteine
SG atoms. Both name how a site was found by the same three methods, which
detection_method gives.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from ...structure.residue_identity import clean_string_series

_STRUCT_CONN_PREFIX = "_struct_conn."

# The symmetry operator that leaves an atom where the file puts it. The
# _struct_conn record of a partner without an operator uses this one.
IDENTITY_SYMMETRY_OPERATOR = "1_555"

# Whether a bond or a pair was found from a deposited record, from distance,
# or from both. The metal-site and disulfide tables use the same names.
DETECTED_BY_RECORD = "struct_conn"
DETECTED_BY_DISTANCE = "distance"
DETECTED_BY_RECORD_AND_DISTANCE = "struct_conn_and_distance"
DETECTION_METHODS: tuple[str, ...] = (
    DETECTED_BY_RECORD,
    DETECTED_BY_DISTANCE,
    DETECTED_BY_RECORD_AND_DISTANCE,
)


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
    # The author component, or the label one where that is missing, in upper
    # case.
    component: str
    # The label atom name, or the author one where that is missing.
    atom_name: str

    @property
    def has_author_fields(self) -> bool:
        """Return whether the author chain and residue number are both given."""
        return bool(self.auth_asym_id and self.auth_seq_id)


@dataclass(frozen=True)
class StructConnRecord:
    """One row of _struct_conn."""

    record_id: str
    # The connection type in lower case, empty when the file gives none.
    connection_type: str
    first: PartnerAddress
    second: PartnerAddress
    # Whether both partners take one symmetry operator.
    same_operator: bool


def _normalized_operator(text: str) -> str:
    """Return a symmetry operator without underscores, the identity if empty."""
    return (text or IDENTITY_SYMMETRY_OPERATOR).replace("_", "")


def _partners(category: pd.DataFrame, number: int) -> list[tuple[PartnerAddress, str]]:
    """Return partner ``number`` (1 or 2) of every row, with its operator."""

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
    return [
        (
            PartnerAddress(
                label_asym, label_seq, auth_asym, auth_seq, ins_code, component, atom
            ),
            _normalized_operator(operator),
        )
        for label_asym, label_seq, auth_asym, auth_seq, ins_code, component, atom, (
            operator
        ) in zip(
            text(f"{prefix}_label_asym_id"),
            text(f"{prefix}_label_seq_id"),
            text(f"{prefix}_auth_asym_id"),
            text(f"{prefix}_auth_seq_id"),
            text(f"pdbx_{prefix}_PDB_ins_code"),
            components,
            atoms,
            text(f"{prefix}_symmetry"),
            strict=True,
        )
    ]


def read_struct_conn(
    mmcif_dict: Mapping[str, Sequence[str | None]],
) -> tuple[StructConnRecord, ...]:
    """Return every _struct_conn row of a flat mmCIF dictionary, in file order.

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
    return tuple(
        StructConnRecord(
            record_id,
            connection_type,
            first,
            second,
            same_operator=first_operator == second_operator,
        )
        for record_id, connection_type, (first, first_operator), (
            second,
            second_operator,
        ) in zip(
            record_ids,
            connection_types,
            _partners(category, 1),
            _partners(category, 2),
            strict=True,
        )
    )


def detection_method(*, by_record: bool, by_distance: bool) -> str:
    """Return the detection method of a bond or pair found by the sources given."""
    if by_record and by_distance:
        return DETECTED_BY_RECORD_AND_DISTANCE
    return DETECTED_BY_RECORD if by_record else DETECTED_BY_DISTANCE


__all__ = [
    "DETECTED_BY_DISTANCE",
    "DETECTED_BY_RECORD",
    "DETECTED_BY_RECORD_AND_DISTANCE",
    "DETECTION_METHODS",
    "IDENTITY_SYMMETRY_OPERATOR",
    "PartnerAddress",
    "StructConnRecord",
    "detection_method",
    "read_struct_conn",
]
