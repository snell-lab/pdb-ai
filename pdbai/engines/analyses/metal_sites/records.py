"""The deposited metal coordination records of a structure file.

read_metal_records reads the _struct_conn category of the flat mmCIF
dictionary through struct_conn.py and keeps the rows whose connection type
is metalc, in any case. Each text is read as clean_string_series reads atom
identity: stripped, and empty when the file writes a placeholder.

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

from ..struct_conn import StructConnRecord, read_struct_conn
from .constants import METAL_CONNECTION_TYPE


def read_metal_records(
    mmcif_dict: Mapping[str, Sequence[str | None]],
) -> tuple[StructConnRecord, ...]:
    """Return the metalc records of a flat mmCIF dictionary, in file order.

    A file without _struct_conn gives no record rather than an error.
    """
    return tuple(
        record
        for record in read_struct_conn(mmcif_dict)
        if record.connection_type == METAL_CONNECTION_TYPE
    )


__all__ = ["read_metal_records"]
