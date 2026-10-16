"""The deposited disulfide records of a structure file, read from _struct_conn.

read_disulfide_records reads the _struct_conn category of the flat mmCIF
dictionary through struct_conn.py, never the atom table. The parser repeats
a one-row category on every atom, but a record has no positional link to the
atoms, so a structure with one record would otherwise look like a structure
with one per atom.

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

from ..struct_conn import PartnerAddress, StructConnRecord, read_struct_conn
from .constants import (
    CYSTEINE_RESIDUE_NAME,
    DISULFIDE_CONNECTION_TYPE,
    SULFUR_ATOM_NAME,
)


@dataclass(frozen=True)
class DisulfideRecords:
    """The disulfide records of one structure file.

    ``records`` holds the records within one molecule, in file order.
    ``symmetry_record_count`` counts the cysteine disulfides that join the
    molecule to a symmetry mate.
    """

    records: tuple[StructConnRecord, ...]
    symmetry_record_count: int


def _in_scope(partner: PartnerAddress) -> bool:
    """Return whether a partner may be a cysteine SG.

    It may when each of its component and atom names is either missing or
    the one the table is about.
    """
    return partner.component in ("", CYSTEINE_RESIDUE_NAME) and partner.atom_name in (
        "",
        SULFUR_ATOM_NAME,
    )


def read_disulfide_records(
    mmcif_dict: Mapping[str, Sequence[str | None]],
) -> DisulfideRecords:
    """Return the cysteine disulfide records of a flat mmCIF dictionary.

    A file without _struct_conn, or a row without a connection type, gives no
    record rather than an error. Records outside the table's scope are left
    out, and records to a symmetry mate are only counted, as the module
    describes.
    """
    records: list[StructConnRecord] = []
    symmetry_record_count = 0
    for record in read_struct_conn(mmcif_dict):
        if record.connection_type != DISULFIDE_CONNECTION_TYPE:
            continue
        if not (_in_scope(record.first) and _in_scope(record.second)):
            continue
        if not record.same_operator:
            symmetry_record_count += 1
            continue
        records.append(record)
    return DisulfideRecords(tuple(records), symmetry_record_count)


__all__ = ["DisulfideRecords", "read_disulfide_records"]
