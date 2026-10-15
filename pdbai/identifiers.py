"""Normalizing a PDB identifier to the one spelling every caller uses.

An identifier is normalized wherever it enters. The command line, the build
API, and the web app normalize the identifiers they are given. A saved
selection and the archive search normalize theirs. Each engine normalizes
again before it builds a URL or a cache path from the lower-cased form. So one
entry never appears under two spellings in a run's manifest or in the cache. A
path or a URL fragment also never reaches the file system or the network as
an identifier.
"""

from __future__ import annotations

import re

# The pattern is matched before upper-casing. Some non-ASCII letters, such as
# the dotless "ı", upper-case to ASCII ones.
_PDB_ID = re.compile(r"[1-9][A-Za-z0-9]{3}")


def normalize_pdb_id(value: str) -> str:
    """Return ``value`` without surrounding whitespace and upper-cased.

    Only the four-character form is accepted. That is a digit from 1 to 9
    followed by three ASCII letters or digits. Anything else raises
    ValueError, the extended form pdb_00001abc included. The message begins
    "Invalid PDB ID" and quotes the value as given. So a path, a URL fragment,
    or a stray word such as NONE never passes as an identifier.
    """
    stripped = value.strip()
    if not _PDB_ID.fullmatch(stripped):
        raise ValueError(
            f"Invalid PDB ID {value!r}; expected a digit from 1 to 9 followed by "
            "three ASCII letters or digits."
        )
    return stripped.upper()


__all__ = ["normalize_pdb_id"]
